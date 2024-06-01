import json
import os
import os.path as osp
from collections import defaultdict
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset
)

from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.utils import index_to_mask, to_undirected


class MAGDataset(InMemoryDataset):

    def __init__(self, root: str,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'ogbn-mag', 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, 'ogbn-mag', 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        x = ['info.dat', 'node.dat', 'link.dat', 'label.dat', 'label.dat.test']
        return [osp.join(self.names[self.name], f) for f in x]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    # def download(self):
    #     url = self.urls[self.name]
    #     path = download_url(url, self.raw_dir)
    #     extract_zip(path, self.raw_dir)
    #     os.unlink(path)

    def process(self):
        dataset = PygNodePropPredDataset(name='ogbn-mag', root=self.root)
        data = dataset[0]
        
        hetero_data = HeteroData()

        # Add edges
        deg = {key : torch.zeros(data.num_nodes_dict[key]) for key in data.num_nodes_dict}
        for edge_type in data.edge_reltype:
            src_type, rel, dst_type = edge_type
            hetero_data[(src_type, rel, dst_type)].edge_index = data.edge_index_dict[edge_type]
            if src_type == dst_type:
                hetero_data[(dst_type, rel, src_type)].edge_index = to_undirected(data.edge_index_dict[edge_type])
            else:
                row, col = data.edge_index_dict[edge_type]
                rev_edge_index = torch.stack([col, row], dim=0)
                hetero_data[(dst_type, 'rev_' + rel, src_type)].edge_index = rev_edge_index
            deg[src_type] = deg[src_type].add(torch.bincount(data.edge_index_dict[edge_type][0, :]))
            deg[dst_type] = deg[dst_type].add(torch.bincount(data.edge_index_dict[edge_type][1, :]))
            
        # Add node features
        def normalize(mx):
            """Row-normalize sparse matrix"""
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.
            r_mat_inv = sp.diags(r_inv)
            mx = r_mat_inv.dot(mx)
            return mx

        hetero_data['paper'].x = torch.cat((data.x_dict['paper'], torch.log10(deg['paper'].reshape(-1, 1))), axis=-1)
        hetero_data['paper'].y = data.y_dict['paper']

        split_idx = dataset.get_idx_split()
        train_paper = split_idx['train']['paper']
        valid_paper = split_idx['valid']['paper']
        test_paper  = split_idx['test']['paper']
        hetero_data['paper'].train_mask = index_to_mask(train_paper, size=data.y_dict['paper'].shape[0])
        hetero_data['paper'].val_mask = index_to_mask(valid_paper, size=data.y_dict['paper'].shape[0])
        hetero_data['paper'].test_mask = index_to_mask(test_paper, size=data.y_dict['paper'].shape[0])
        for node_type in data.num_nodes_dict:
            if node_type in ['paper', 'institution']:
                continue
            for edge_type in hetero_data.edge_index_dict:
                src_type, rel, dst_type = edge_type
                if src_type == node_type and dst_type == 'paper':
                    edge_index = hetero_data.edge_index_dict[edge_type]
                    v = torch.ones(edge_index.shape[1])
                    m = normalize(sp.coo_matrix((v, edge_index), \
                        shape=(data.num_nodes_dict[node_type], data.num_nodes_dict['paper'])))

                    out = m.dot(data.x_dict['paper'])
                    hetero_data[node_type].x = torch.cat((torch.from_numpy(out), torch.log10(deg[node_type].reshape(-1, 1))), axis=-1)

        for edge_type in hetero_data.edge_index_dict:
            src_type, rel, dst_type = edge_type
            if src_type == 'institution' and dst_type == 'author':
                edge_index = hetero_data.edge_index_dict[edge_type]
                v = torch.ones(edge_index.shape[1])
                m = normalize(sp.coo_matrix((v, edge_index), \
                    shape=(data.num_nodes_dict['institution'], data.num_nodes_dict['author'])))

                out = m.dot(hetero_data['author'].x[:, :-1])
                hetero_data['institution'].x = torch.cat((torch.from_numpy(out), torch.log10(deg['institution'].reshape(-1, 1))), axis=-1)  

        if self.pre_transform is not None:
            hetero_data = self.pre_transform(hetero_data)

        torch.save(self.collate([hetero_data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return 'MAGDataset()'
