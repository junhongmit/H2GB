import json
import os
import os.path as osp
from collections import defaultdict
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)

import dill
class RenameUnpickler(dill.Unpickler):
    def find_class(self, module, name):
        renamed_module = module
        if module == "pyHGT.data" or module == 'data':
            renamed_module = r"GPT_GNN.data"
        return super(RenameUnpickler, self).find_class(renamed_module, name)
def renamed_load(file_obj):
    return RenameUnpickler(file_obj).load()

# To prepare the dataset, this piece of code need to be run under GPT-GNN's repo
class OAGDataset(InMemoryDataset):

    names = {
        'art': 'Art',
        'business': 'Business',
        'chemistry': 'Chemistry',
        'cs': 'CS_20190919',
        'engineering': 'Engineering',
        'material': 'Materials_science',
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

    # def download(self):
    #     url = self.urls[self.name]
    #     path = download_url(url, self.raw_dir)
    #     extract_zip(path, self.raw_dir)
    #     os.unlink(path)

    def process(self):
        graph = renamed_load(open(f'/nobackup/users/junhong/Data/preprocessed_OAG/graph_{self.names[self.name]}.pk', 'rb'))
        
        data = HeteroData()

        # Add nodes
        for idx, node_type in enumerate(graph.node_forward):
            if node_type == 'venue':
                continue
            data[node_type].x = torch.tensor(graph.node_feature[node_type]['emb'].to_list()).to(torch.float)
        data['paper'].year = torch.tensor(graph.node_feature['paper']['time'].to_list())

        # Add edges
        for tar in graph.edge_list.keys():
            for src in graph.edge_list[tar].keys():
                for rel in graph.edge_list[tar][src].keys():
                    rel_name = rel
                    if rel.startswith('rev_'):
                        continue
                    if src == 'paper' and tar == 'venue':
                        continue
                    # if src == 'author' and tar == 'paper':
                    #     rel_name = 'AP_write'
                    # Only keep L2 field connections
                    # if src == 'paper' and tar == 'field' and rel != 'PF_in_L2':
                    #     continue
                        # rel_name = 'PF_in'
                    print((src, rel_name, tar))
                    total_edges, cur_edges = 0, 0
                    for tar_id, items in graph.edge_list[tar][src][rel].items():
                        total_edges += len(items)
                    edge_index = torch.empty(2, total_edges, dtype=torch.long)
                    for tar_id, src_ids in graph.edge_list[tar][src][rel].items():
                        for idx, src_id in enumerate(src_ids.keys()):
                            edge_index[0, idx + cur_edges] = src_id
                            edge_index[1, idx + cur_edges] = tar_id
                        cur_edges += len(src_ids)
                    if (src, rel_name, tar) in data.edge_types:
                        data[(src, rel_name, tar)].edge_index = torch.cat((data[(src, rel_name, tar)].edge_index, edge_index), dim=1)
                    else:
                        data[(src, rel_name, tar)].edge_index = edge_index
                        
        # Add label
        src = 'paper'
        tar = 'venue'
        rel = 'PV_Journal' # The classification domain is Jounrnal paper
        y = torch.empty(len(graph.node_forward['paper']), dtype=torch.long).fill_(-1)
        # for rel in graph.edge_list[tar][src].keys():
        graph.edge_list[tar][src][rel]
        count = -1
        count_dict = {}
        for tar_id, src_ids in graph.edge_list[tar][src][rel].items():
            for idx, src_id in enumerate(src_ids.keys()):
                idx = count_dict.get(tar_id, count + 1)
                count_dict[tar_id] = idx
                y[src_id] = idx
                count = max(count, idx)
        data['paper'].y = y
        data['paper'].train_mask = torch.logical_and(data['paper'].y != -1, data['paper'].year <= 2016)
        data['paper'].val_mask = torch.logical_and(data['paper'].y != -1, torch.logical_and(data['paper'].year >= 2017, data['paper'].year <= 2017))
        data['paper'].test_mask = torch.logical_and(data['paper'].y != -1, data['paper'].year >= 2018)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.names[self.name]}()'