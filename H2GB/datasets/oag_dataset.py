import json
import os
import os.path as osp
from collections import defaultdict
from typing import Callable, List, Optional

import numpy as np
import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from .utils import download_dataset

# To prepare the dataset, this piece of code need to be run under GPT-GNN's repo
class OAGDataset(InMemoryDataset):
    r"""A variety of new heterogeneous graph benchmark datasets composed
    of subsets of Open Academic Graph (OAG) from `"OAG: Toward Linking
    Large-scale Heterogeneous Entity Graphs" <https://dl.acm.org/doi/10.1145/3292500.3330785>`_
    paper. Each of the datasets contains papers from three different subject
    domains -- computer science (:obj:`oag-cs`), engineering (:obj:`oag-eng`),
    and chemistry (:obj:`oag-chem`). These datasets contain four types of
    entities -- papers, authors, institutions, and field of study. 
    Each paper is associated with a 768-dimensional feature vector generated
    from a pre-trained XLNet applying on the paper titles. The representation
    of each word in the title are weighted by each word's attention to get
    the title representation for each paper. Each paper node is labeled with
    its published venue (paper or conference).
    
    We split the papers published up to 2016 as the training set, papers
    published in 2017 as the validation set, and papers published in 2018 and
    2019 as the test set. The publication year of each paper is also included
    in these datasets.

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

    urls = {
        'cs': 'https://drive.google.com/file/d/115WygJhRo1DxVLpLzJF-hFCamGc7JY5w/view?usp=drive_link',
        'engineering': 'https://drive.google.com/file/d/1_n605385TzqqaVIiMQcKziSv5BUG4f4Y/view?usp=drive_link',
        'chemistry': 'https://drive.google.com/file/d/1S13pnOk2-bPevWQafl6lQj8QOy6BK7Ca/view?usp=drive_link'
    }

    names = {
        'cs': 'CS_20190919',
        'engineering': 'Engineering',
        'chemistry': 'Chemistry'
    }

    def __init__(self, root: str, name: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name.lower()
        assert self.name in set(self.names.keys())
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
        x = [f'graph_{self.names[self.name]}.pt']
        return x

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    def download(self):
        url = self.urls[self.name]
        download_dataset(url, self.raw_dir)

    def process(self):
        graph = torch.load(os.path.join(self.raw_dir, f'graph_{self.names[self.name]}.pt'))
        
        data = HeteroData()

        # Add nodes
        for idx, node_type in enumerate(graph['node_forward']):
            if node_type == 'venue':
                continue
            for i in range(len(graph['node_forward'][node_type])):
                assert graph['node_forward'][node_type][graph['node_feature'][node_type].iloc[i]['id']] == i
            data[node_type].x = torch.tensor(np.stack(graph['node_feature'][node_type]['emb'].values), dtype=torch.float)
        data['paper'].year = torch.tensor(np.stack(graph['node_feature']['paper']['time'].values))

        # Add edges
        for tar in graph['edge_list'].keys():
            for src in graph['edge_list'][tar].keys():
                for rel in graph['edge_list'][tar][src].keys():
                    rel_name = rel
                    if rel.startswith('rev_'):
                        continue
                    if src == 'paper' and tar == 'venue':
                        continue
                    # if src == 'author' and tar == 'paper':
                    #     rel_name = 'AP_write'
                    # if src == 'paper' and tar == 'field':
                    #     rel_name = 'PF_in'
                    print(f'Working on edge type {(src, rel_name, tar)}...')
                    total_edges, cur_edges = 0, 0
                    for tar_id, items in graph['edge_list'][tar][src][rel].items():
                        total_edges += len(items)
                    edge_index = torch.empty(2, total_edges, dtype=torch.long)
                    for tar_id, src_ids in graph['edge_list'][tar][src][rel].items():
                        for idx, src_id in enumerate(src_ids.keys()):
                            edge_index[0, idx + cur_edges] = src_id
                            edge_index[1, idx + cur_edges] = tar_id
                        cur_edges += len(src_ids)
                    
                    try:
                        num_edges_dict = data.num_edges_dict
                    except:
                        num_edges_dict = []
                    if (src, rel_name, tar) in num_edges_dict:
                        data[(src, rel_name, tar)].edge_index = torch.cat((data[(src, rel_name, tar)].edge_index, edge_index), dim=1)
                    else:
                        data[(src, rel_name, tar)].edge_index = edge_index
        
        # Add label
        src = 'paper'
        tar = 'venue'
        rel = 'PV_Journal' # The classification domain is Jounrnal paper
        y = torch.empty(len(graph['node_forward']['paper']), dtype=torch.long).fill_(-1)
        count = -1
        count_dict = {}
        for tar_id, src_ids in graph['edge_list'][tar][src][rel].items():
            for idx, src_id in enumerate(src_ids.keys()):
                idx = count_dict.get(tar_id, count + 1)
                count_dict[tar_id] = idx
                y[src_id] = idx
                count = max(count, idx)
        data['paper'].y = y
        data['paper'].train_mask = torch.logical_and(data['paper'].y != -1, data['paper'].year <= 2016)
        data['paper'].val_mask = torch.logical_and(data['paper'].y != -1, torch.logical_and(data['paper'].year >= 2017, data['paper'].year <= 2017))
        data['paper'].test_mask = torch.logical_and(data['paper'].y != -1, data['paper'].year >= 2018)
        data.validate()

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.names[self.name]}()'