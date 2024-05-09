import json
import os
import os.path as osp
from collections import defaultdict
from typing import Callable, List, Optional

import torch
from torch_sparse import SparseTensor

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import index_to_mask
from sklearn.model_selection import train_test_split

from deeprobust.graph.data import Dataset
import os.path as osp
import numpy as np

class CustomDataset(Dataset):
    def __init__(self, root, name, setting='gcn', seed=None, require_mask=False):
        '''
        Adopted from https://github.com/DSE-MSU/DeepRobust/blob/master/deeprobust/graph/data/dataset.py
        '''
        self.name = name.lower()
        self.setting = setting.lower()

        self.seed = seed
        self.url = None
        self.root = osp.expanduser(osp.normpath(root))
        self.data_folder = osp.join(root, self.name)
        self.data_filename = self.data_folder + '.npz'
        # Make sure dataset file exists
        assert osp.exists(self.data_filename), f"{self.data_filename} does not exist!" 
        self.require_mask = require_mask

        self.require_lcc = True if setting == 'nettack' else False
        self.adj, self.features, self.labels = self.load_data()
        self.idx_train, self.idx_val, self.idx_test = self.get_train_val_test()
        if self.require_mask:
            self.get_mask()
    
    def get_adj(self):
        adj, features, labels = self.load_npz(self.data_filename)
        adj = adj + adj.T
        adj = adj.tolil()
        adj[adj > 1] = 1

        if self.require_lcc:
            lcc = self.largest_connected_components(adj)

            # adj = adj[lcc][:, lcc]
            adj_row = adj[lcc]
            adj_csc = adj_row.tocsc()
            adj_col = adj_csc[:, lcc]
            adj = adj_col.tolil()

            features = features[lcc]
            labels = labels[lcc]
            assert adj.sum(0).A1.min() > 0, "Graph contains singleton nodes"

        # whether to set diag=0?
        adj.setdiag(0)
        adj = adj.astype("float32").tocsr()
        adj.eliminate_zeros()

        assert np.abs(adj - adj.T).sum() == 0, "Input graph is not symmetric"
        assert adj.max() == 1 and len(np.unique(adj[adj.nonzero()].A1)) == 1, "Graph must be unweighted"

        return adj, features, labels
    
    def get_train_val_test(self):
        if self.setting == "exist":
            with np.load(self.data_filename) as loader:
                idx_train = loader["idx_train"]
                idx_val = loader["idx_val"]
                idx_test = loader["idx_test"]
            return idx_train, idx_val, idx_test
        else:
            return super().get_train_val_test()
        
        
class SynHeterophilicDataset(InMemoryDataset):
    r'''A synthetic heterophilc dataset provided in H2GNN paper.
    
    '''

    names = {
        'cora': 'Cora',
        'products': 'Products',
    }

    def __init__(self, root: str, name: str, homophily: str='h1.00-r1',
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = f"{name.lower()}+{homophily}"
        self.type = name
        self.homophily = homophily
        assert self.type in set(self.names.keys())
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.type, self.homophily, 'processed')

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    # def download(self):
    #     url = self.urls[self.name]
    #     path = download_url(url, self.raw_dir)
    #     extract_zip(path, self.raw_dir)
    #     os.unlink(path)

    def process(self):
        # Load the dataset in file `syn-cora/h0.00-r1.npz`
        # `seed` controls the generation of training, validation and test splits
        dataset = CustomDataset(root=osp.join(self.root, f"syn-{self.type}"), name=self.homophily, \
                                setting="gcn", seed=15)

        data = HeteroData()

        if self.type == 'cora':
            node_name = 'paper'
            edge_name = ('paper', 'cites', 'paper')
        elif self.type == 'products':
            node_name = 'product'
            edge_name = ('product', 'buy_with', 'product')
        data[node_name].x = torch.from_numpy(dataset.features.toarray())
        data[node_name].y = torch.from_numpy(dataset.labels).type(torch.long)
        rows, cols = dataset.adj.nonzero()
        rows = torch.from_numpy(rows).type(torch.long)
        cols = torch.from_numpy(cols).type(torch.long)
        num_nodes = dataset.features.shape[0]
        adj_t = SparseTensor(row=rows, col=cols, value=None, sparse_sizes=(num_nodes, num_nodes)).t()
        adj_t = adj_t.to_symmetric()
        data[edge_name].adj_t = adj_t

        # splits = dataset.get_train_val_test()

        # train_mask = json.load(open('~/train_mask.json'))
        # split_names = ['train_mask', 'val_mask', 'test_mask']
        # for i in range(len(split_names)):
        #     mask = torch.tensor(train_mask[split_names[i]])
        #     data[node_name][split_names[i]] = mask

        # Try to reproduce the split ratio in the H2GCN paper (train 25%, val 25%, test 50%)
        train_indices, temp_indices = train_test_split(np.arange(num_nodes), train_size=0.25)
        val_indices, test_indices = train_test_split(temp_indices, test_size=0.667)
        splits = [train_indices, val_indices, test_indices]

        split_names = ['train_mask', 'val_mask', 'test_mask']
        for i in range(len(splits)):
            mask = index_to_mask(torch.from_numpy(splits[i]), size=data[node_name].y.shape[0])
            data[node_name][split_names[i]] = mask

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.name}()'