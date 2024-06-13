import json
import os
import os.path as osp
from collections import defaultdict
from typing import Callable, List, Optional

import numpy as np
import scipy.sparse as sp
from sklearn.model_selection import train_test_split
import torch
from torch_sparse import SparseTensor
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset
)

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import index_to_mask, to_undirected
from .utils import (even_quantile_labels, get_sparse_tensor)


class MAGDataset(InMemoryDataset):
    r"""The :obj:`ogbn-mag` and modified :obj:`mag-year` dataset from the
    `"Open Graph Benchmark: Datasets for Machine Learning on Graphs"
    <https://arxiv.org/abs/2005.00687>`_ paper.

    :obj:`ogbn-mag` and :obj:`mag-year` are heterogeneous graphs composed
    of a subset of the Microsoft Academic Graph (MAG).
    It contains four types of entities — papers (736,389 nodes), authors
    (1,134,649 nodes), institutions (8,740 nodes), and fields of study
    (59,965 nodes) — as well as four types of directed relations connecting two
    types of entities.
    Each paper is associated with a 128-dimensional :obj:`word2vec` feature
    vector, and all the other types of entities are originally not associated
    with input node features. We average the node features of all the
    published paper of an author to obtain the author feature.

    The task of :obj:`ogbn-mag` is to predict the venue (conference or journal)
    of each paper. In total, there are 349 different venues.
    The task of :obj:`mag-year` is to predict year that the paper is published.
    The five classes are chosen by partitioning the published year so that class
    ratios are approximately balanced.

    Args:
        root (str): Root directory where the dataset should be saved.
        name (str): The name of the dataset (one of :obj:`"ogbn-mag"`,
            :obj:`"mag-year"`)
        rand_split (bool, optional): Whether to randomly re-split the dataset.
            This option is only applicable to :obj:`mag-year`.
            (default: :obj:`False`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
    
    """

    names = ['ogbn-mag', 'mag-year']

    def __init__(self, root: str, name: str, rand_split: bool = False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        self.rand_split = rand_split
        assert self.name in self.names
        super().__init__(root, transform, pre_transform, force_reload=rand_split)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        x = ['info.dat', 'node.dat', 'link.dat', 'label.dat', 'label.dat.test']
        return [osp.join(self.names[self.name], f) for f in x]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def process(self):
        dataset = PygNodePropPredDataset(name='ogbn-mag', root=self.root)
        graph = dataset[0]
        data = HeteroData()

        # Add edges (sparse adj_t)
        for edge_type in graph.edge_reltype:
            src_type, rel, dst_type = edge_type
            data[(src_type, rel, dst_type)].edge_index = graph.edge_index_dict[edge_type]
            data[(src_type, rel, dst_type)].adj_t = \
                get_sparse_tensor(graph.edge_index_dict[edge_type], 
                                num_src_nodes=graph.num_nodes_dict[src_type],
                                num_dst_nodes=graph.num_nodes_dict[dst_type])
            if src_type == dst_type:
                data[(src_type, rel, dst_type)].edge_index = \
                    to_undirected(data[(src_type, rel, dst_type)].edge_index)
                data[(src_type, rel, dst_type)].adj_t = \
                    data[(src_type, rel, dst_type)].adj_t.to_symmetric()
            else:
                row, col = graph.edge_index_dict[edge_type]
                rev_edge_index = torch.stack([col, row], dim=0)
                data[(dst_type, 'rev_' + rel, src_type)].edge_index = rev_edge_index
                data[(dst_type, 'rev_' + rel, src_type)].adj_t = \
                    get_sparse_tensor(rev_edge_index, 
                                    num_src_nodes=graph.num_nodes_dict[dst_type],
                                    num_dst_nodes=graph.num_nodes_dict[src_type])
        
        # Add node features
        def normalize(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            mx = r_mat_inv.dot(mx)
            return mx

        data['paper'].x = graph.x_dict['paper'] # torch.cat((graph.x_dict['paper'], torch.log10(deg['paper'].reshape(-1, 1))), axis=-1)
        data['paper'].y = graph.y_dict['paper'].squeeze()

        if not self.rand_split:
            split_idx = dataset.get_idx_split()
            train_paper = split_idx['train']['paper']
            valid_paper = split_idx['valid']['paper']
            test_paper  = split_idx['test']['paper']
        else:
            train_size=.85 # 85%
            valid_size=.6 # 9%
            train_paper, temp_paper = train_test_split(torch.where(data['paper'].y != -1)[0], train_size=train_size)
            valid_paper, test_paper = train_test_split(temp_paper, train_size=valid_size)
        data['paper'].train_mask = index_to_mask(train_paper, size=graph.y_dict['paper'].shape[0])
        data['paper'].val_mask = index_to_mask(valid_paper, size=graph.y_dict['paper'].shape[0])
        data['paper'].test_mask = index_to_mask(test_paper, size=graph.y_dict['paper'].shape[0])

        # Average paper's embedding to corresponding author and field of study
        for edge_type in [('author', 'writes', 'paper'), ('field_of_study', 'rev_has_topic', 'paper')]:
            src_type, rel, dst_type = edge_type
            edge_index = data.edge_index_dict[edge_type]
            v = torch.ones(edge_index.shape[1])
            m = normalize(sp.coo_matrix((v, edge_index), \
                shape=(graph.num_nodes_dict[src_type], graph.num_nodes_dict[dst_type])))
            out = m.dot(data[dst_type].x)
            data[src_type].x = torch.from_numpy(out) # torch.cat((torch.from_numpy(out), torch.log10(deg[node_type].reshape(-1, 1))), axis=-1)

        # Average author's embedding to corresponding institution
        edge_type = ('institution', 'rev_affiliated_with', 'author')
        edge_index = data.edge_index_dict[edge_type]
        v = torch.ones(edge_index.shape[1])
        m = normalize(sp.coo_matrix((v, edge_index), \
            shape=(graph.num_nodes_dict['institution'], graph.num_nodes_dict['author'])))
        out = m.dot(data['author'].x)
        data['institution'].x = torch.from_numpy(out) # torch.cat((torch.from_numpy(out), torch.log10(deg['institution'].reshape(-1, 1))), axis=-1)  

        if self.name == 'mag-year':
            label = even_quantile_labels(graph['node_year']['paper'].squeeze().numpy(), nclasses=5, verbose=False)
            data['paper'].y = torch.from_numpy(label).squeeze() 
            train_size=.5 # 50%
            valid_size=.5 # 25%
            train_idx, temp_idx = train_test_split(torch.where(data['paper'].y != -1)[0], train_size=train_size)
            val_idx, test_idx = train_test_split(temp_idx, train_size=valid_size)
            data['paper'].train_mask = index_to_mask(train_idx, size=data['paper'].y.shape[0])
            data['paper'].val_mask = index_to_mask(val_idx, size=data['paper'].y.shape[0])
            data['paper'].test_mask = index_to_mask(test_idx, size=data['paper'].y.shape[0])

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return 'MAGDataset()'
