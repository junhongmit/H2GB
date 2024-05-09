import json
import os
import os.path as osp
import numpy as np
from collections import defaultdict
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_sparse import SparseTensor

def get_sparse_tensor(edge_index, num_nodes=None, num_src_nodes=None, num_dst_nodes=None, return_e_id=False):
    if isinstance(edge_index, SparseTensor):
        adj_t = edge_index
        if return_e_id:
            value = torch.arange(adj_t.nnz())
            adj_t = adj_t.set_value(value, layout='coo')
        return adj_t


    if (num_nodes is None) and (num_src_nodes is None) and (num_dst_nodes is None):
        num_src_nodes = int(edge_index.max()) + 1
        num_dst_nodes = num_src_nodes
    elif (num_src_nodes is None) and (num_dst_nodes is None):
        num_src_nodes, num_dst_nodes = num_nodes


    value = torch.arange(edge_index.size(1)) if return_e_id else None
    return SparseTensor(row=edge_index[0], col=edge_index[1],
                        value=value,
                        sparse_sizes=(num_src_nodes, num_dst_nodes)).t()

# To prepare the dataset, this piece of code need to be run under GPT-GNN's repo
class IGBDataset(InMemoryDataset):
    def __init__(self, root: str, name: str, classes=19, in_memory=True,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.root = root
        self.name = name
        self.in_memory = in_memory
        self.classes = classes
        super().__init__(root, transform, pre_transform)
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
        return f'data-{self.classes}.pt'

    # def download(self):
    #     url = self.urls[self.name]
    #     path = download_url(url, self.raw_dir)
    #     extract_zip(path, self.raw_dir)
    #     os.unlink(path)

    def process(self):
        data = HeteroData()

        if self.in_memory:
            paper_node_features = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed', 
            'paper', 'node_feat.npy')))
            if self.classes == 19:
                paper_node_labels = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed', 
            'paper', 'node_label_19.npy'))).to(torch.long)  
            else:
                paper_node_labels = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed', 
            'paper', 'node_label_2K.npy'))).to(torch.long)
        else:
            paper_node_features = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed', 
            'paper', 'node_feat.npy'), mmap_mode='r'))
            if self.classes == 19:
                paper_node_labels = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed', 
            'paper', 'node_label_19.npy'), mmap_mode='r')).to(torch.long)  
            else:
                paper_node_labels = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed', 
            'paper', 'node_label_2K.npy'), mmap_mode='r')).to(torch.long)                

        data['paper'].x = paper_node_features
        data['paper'].y = paper_node_labels
        if self.in_memory:
            author_node_features = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed', 
            'author', 'node_feat.npy')))
        else:
            author_node_features = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed', 
            'author', 'node_feat.npy'), mmap_mode='r'))
        data['author'].x = author_node_features

        if self.in_memory:
            institute_node_features = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed', 
            'institute', 'node_feat.npy')))       
        else:
            institute_node_features = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed', 
            'institute', 'node_feat.npy'), mmap_mode='r'))
        data['institute'].x = institute_node_features

        if self.in_memory:
            fos_node_features = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed', 
            'fos', 'node_feat.npy')))       
        else:
            fos_node_features = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed', 
            'fos', 'node_feat.npy'), mmap_mode='r'))
        data['fos'].x = fos_node_features

        if self.in_memory:
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed',
            'paper__cites__paper', 'edge_index.npy')))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed', 
            'paper__written_by__author', 'edge_index.npy')))
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed', 
            'author__affiliated_to__institute', 'edge_index.npy')))
            paper_fos_edges = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed', 
            'paper__topic__fos', 'edge_index.npy')))
        else:
            paper_paper_edges = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed',
            'paper__cites__paper', 'edge_index.npy'), mmap_mode='r'))
            author_paper_edges = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed', 
            'paper__written_by__author', 'edge_index.npy'), mmap_mode='r'))
            affiliation_author_edges = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed', 
            'author__affiliated_to__institute', 'edge_index.npy'), mmap_mode='r'))
            paper_fos_edges = torch.from_numpy(np.load(osp.join(self.root, self.name, 'processed', 
            'paper__topic__fos', 'edge_index.npy'), mmap_mode='r'))

        data[('paper', 'cites', 'paper')].edge_index = paper_paper_edges.T
        data[('paper', 'written_by', 'author')].edge_index = author_paper_edges.T
        data[('author', 'affiliated_to', 'institute')].edge_index = affiliation_author_edges.T
        data[('paper', 'topic', 'fos')].edge_index = paper_fos_edges.T

        # if self.name == 'medium':
        #     import time
        #     for edge_type in data.edge_types:
        #         src, _, dst = edge_type
        #         start = time.time()
        #         adj_t = get_sparse_tensor(data[edge_type].edge_index, 
        #                                 num_src_nodes=data.num_nodes_dict[src],
        #                                 num_dst_nodes=data.num_nodes_dict[dst])
        #         end = time.time()
        #         print("Get sparse tensor cost:", round(end - start, 3), "seconds")
        #         start = end
        #         adj_t = adj_t.to_symmetric()
        #         end = time.time()
        #         print("To symmetric cost:", round(end - start, 3), "seconds")
        #         data[edge_type].adj_t = adj_t
        
        # self.graph = dgl.remove_self_loop(self.graph, etype='cites')
        # self.graph = dgl.add_self_loop(self.graph, etype='cites')
        
        n_nodes = data.num_nodes_dict['paper']

        n_train = int(n_nodes * 0.6)
        n_val = int(n_nodes * 0.2)

        train_mask = torch.zeros(n_nodes, dtype=torch.bool)
        val_mask = torch.zeros(n_nodes, dtype=torch.bool)
        test_mask = torch.zeros(n_nodes, dtype=torch.bool)
        
        train_mask[:n_train] = True
        val_mask[n_train:n_train + n_val] = True
        test_mask[n_train + n_val:] = True
        
        data['paper'].train_mask = train_mask
        data['paper'].val_mask = val_mask
        data['paper'].test_mask = test_mask

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.names[self.name]}()'

class IGB260M(object):
    def __init__(self, root: str, size: str, in_memory: int, \
        classes: int, synthetic: int):
        self.root = root
        self.name = size
        self.synthetic = synthetic
        self.in_memory = in_memory
        self.num_classes = classes

    def num_nodes(self):
        if self.name == 'experimental':
            return 100000
        elif self.name == 'small':
            return 1000000
        elif self.name == 'medium':
            return 10000000
        elif self.name == 'large':
            return 100000000
        elif self.name == 'full':
            return 269346174

    @property
    def paper_feat(self) -> np.ndarray:
        num_nodes = self.num_nodes()
        # TODO: temp for bafs. large and full special case
        if self.name == 'large' or self.name == 'full':
            path = osp.join(self.root, 'full', 'processed', 'paper', 'node_feat.npy')
            emb = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes,1024))
        else:
            path = osp.join(self.root, self.name, 'processed', 'paper', 'node_feat.npy')
            if self.synthetic:
                emb = np.random.rand(num_nodes, 1024).astype('f')
            else:
                if self.in_memory:
                    emb = np.load(path)
                else:
                    emb = np.load(path, mmap_mode='r')

        return emb

    @property
    def paper_label(self) -> np.ndarray:

        if self.name == 'large' or self.name == 'full':
            num_nodes = self.num_nodes()
            if self.num_classes == 19:
                path = osp.join(self.root, 'full', 'processed', 'paper', 'node_label_19.npy')
                node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes))
                # Actual number 227130858
            else:
                path = osp.join(self.root, 'full', 'processed', 'paper', 'node_label_2K.npy')
                node_labels = np.memmap(path, dtype='float32', mode='r',  shape=(num_nodes))
                # Actual number 157675969

        else:
            if self.num_classes == 19:
                path = osp.join(self.root, self.name, 'processed', 'paper', 'node_label_19.npy')
            else:
                path = osp.join(self.root, self.name, 'processed', 'paper', 'node_label_2K.npy')
            if self.in_memory:
                node_labels = np.load(path)
            else:
                node_labels = np.load(path, mmap_mode='r')
        return node_labels

    @property
    def paper_edge(self) -> np.ndarray:
        path = osp.join(self.root, self.name, 'processed', 'paper__cites__paper', 'edge_index.npy')
        # if self.name == 'full':
        #     path = '/mnt/nvme15/IGB260M_part_2/full/processed/paper__cites__paper/edge_index.npy'
        if self.in_memory:
            return np.load(path)
        else:
            return np.load(path, mmap_mode='r')