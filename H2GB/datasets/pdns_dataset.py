import os, shutil
import os.path as osp
import numpy as np
import pandas as pd
from typing import Optional, List, Callable

import torch
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from .utils import download_dataset

class PDNSDataset(InMemoryDataset):
    r"""
    PDNS is a heterogeneous cybersecurity graph of passive DNS data from the
    `"PDNS-Net: A Large Heterogeneous Graph Benchmark Dataset of Network
    Resolutions for Graph Learning" <https://arxiv.org/abs/2203.07969>`_ paper.
     
    The dataset is constructed from a seed set of malicious domains collected
    from VirusTotal and the hosting infrastructure behind these seed domains
    are extracted from a popular passive DNS repository that passively records
    most of the domain resolution occur around the world. It consists of two
    kinds of entities, domain node and IP node, and four types of relation, such
    as domain is similar to domain and domain resolve to an IP. Each domain node
    is associated with a 10-dimensional node feature vector extracted from
    pre-processed domain name, such as the number of subdomains, impersonation
    to a popular top brand, etc. The domain node is labeled with a binary label
    tagging if it is a malicious domain. We follow the official dataset splitting,
    where the test set is obtained over time.

    Args:
        root (str): Root directory where the dataset should be saved.
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    
    """

    url = 'https://drive.google.com/file/d/1xCU2-kOyAEj3bIe4iBQvflLJkU0NTJVc/view?usp=drive_link'

    def __init__(self, root: str, transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None, 
                 start: int = 0, end: int = 5, test_list: List[int] = [6],
                 balance_gt: bool = False, domain_file: str = 'domains.csv'):
        super().__init__(root, transform, pre_transform)
        # processed = self._load_dataset(root)
        self.name = 'PDNS'
        processed = self._load_train_test(root, start=start, end=end, test_list=test_list, balance_gt=balance_gt, domain_file=domain_file)
        self.data, self.slices = self.collate([processed])
        
    @property
    def num_classes(self) -> int:
        return 2
    
    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def raw_file_names(self) -> List[str]:
        file_names = ['domain_features.csv', 'domain_features_map.txt', 'domains.csv',
            'domains2.csv', 'ip_features.csv', 'ip_features_map.txt', 'ips.csv',
            'timestamp_0_edges.csv', 'timestamp_10_edges.csv', 'timestamp_11_edges.csv',
            'timestamp_12_edges.csv', 'timestamp_13_edges.csv', 'timestamp_14_edges.csv',
            'timestamp_15_edges.csv', 'timestamp_16_edges.csv', 'timestamp_17_edges.csv',
            'timestamp_18_edges.csv', 'timestamp_19_edges.csv', 'timestamp_1_edges.csv',
            'timestamp_20_edges.csv', 'timestamp_21_edges.csv', 'timestamp_22_edges.csv', 
            'timestamp_23_edges.csv', 'timestamp_24_edges.csv', 'timestamp_25_edges.csv',
            'timestamp_26_edges.csv', 'timestamp_27_edges.csv', 'timestamp_28_edges.csv',
            'timestamp_29_edges.csv', 'timestamp_2_edges.csv', 'timestamp_30_edges.csv',
            'timestamp_31_edges.csv', 'timestamp_32_edges.csv', 'timestamp_33_edges.csv',
            'timestamp_34_edges.csv', 'timestamp_35_edges.csv', 'timestamp_36_edges.csv',
            'timestamp_37_edges.csv', 'timestamp_38_edges.csv', 'timestamp_39_edges.csv',
            'timestamp_3_edges.csv', 'timestamp_40_edges.csv', 'timestamp_41_edges.csv',
            'timestamp_42_edges.csv', 'timestamp_43_edges.csv', 'timestamp_44_edges.csv',
            'timestamp_45_edges.csv', 'timestamp_46_edges.csv', 'timestamp_47_edges.csv',
            'timestamp_48_edges.csv', 'timestamp_49_edges.csv', 'timestamp_4_edges.csv',
            'timestamp_50_edges.csv', 'timestamp_51_edges.csv', 'timestamp_52_edges.csv',
            'timestamp_53_edges.csv', 'timestamp_54_edges.csv', 'timestamp_55_edges.csv',
            'timestamp_56_edges.csv', 'timestamp_57_edges.csv', 'timestamp_58_edges.csv',
            'timestamp_59_edges.csv', 'timestamp_5_edges.csv', 'timestamp_60_edges.csv',
            'timestamp_6_edges.csv', 'timestamp_7_edges.csv', 'timestamp_8_edges.csv',
            'timestamp_9_edges.csv']

        return file_names
    
    def download(self) -> None:
        if not all([osp.exists(f) for f in self.raw_paths]):
            download_dataset(self.url, self.root)
    
    def _load_dataset(self, root):
        edge_type_nodes = {
            'apex': ['domain_node', 'domain_node'],
            'similar': ['domain_node', 'domain_node'],
            'resolves': ['domain_node', 'ip_node']
        }

        directory = [osp.join(root, 'raw', f) for f in os.listdir(osp.join(root, 'raw'))]
        edge_files = [f for f in directory if osp.isfile(f) and 'timestamp' in f]
        edge_files = sorted(edge_files, key=lambda x: int(x.split("/")[-1].split("_")[1]))

        domains = pd.read_csv(osp.join(root, 'raw', 'domains.csv')).sort_values('domain_node')
        label_dict = {
            'domain_node': domains['type'].apply(lambda x: 1 if x == 'malicious' else (0 if x == 'benign' else -1)).values
        }

        domains['domain_node']

        edge_indices = []
        edge_weights = []
        targets = []
        for edge_file in edge_files[:3]:
            all_edges = pd.read_csv(edge_file)
            domain_set = set()
            timestamp_edges = {}

            for edge_type, edges in all_edges.groupby('type'):
                src_type, dst_type = edge_type_nodes[edge_type]
                if src_type == 'domain_node':
                    domain_set.update(edges['source'].astype(int).values.T)
                if dst_type == 'domain_node':
                    domain_set.update(edges['target'].astype(int).values.T)
                # print(edge_file, src_type, dst_type, len(domain_set))                    
                key = (src_type, edge_type, dst_type)
                timestamp_edges[key] = edges.loc[:, ['source', 'target']].astype(int).values.T

            edge_indices.append(timestamp_edges)
            edge_weights.append(None)
            temporal_labels = {'domain_node':np.asarray([-1 if index not in domain_set else value for index, value in enumerate(label_dict['domain_node'])])}
            targets.append(temporal_labels)

        domain_features = pd.read_csv(osp.join(root, 'raw', 'domain_features.csv')).sort_values('domain_node')
        ip_features = pd.read_csv(osp.join(root, 'raw', 'ip_features.csv')).sort_values('ip_node')

        feature_dict = {
            'domain_node': domain_features.set_index('domain_node').values,
            'ip_node': ip_features.set_index('ip_node').values
        }

        graph = DynamicHeteroGraphStaticSignal(edge_indices, edge_weights, feature_dict, targets)
        return graph

    def _load_train_test(self, root, domain_file, start=None, end=None, test_list=None, num_val=0.2, balance_gt=False, ):
        
        directory = [osp.join(root, 'raw', f) for f in os.listdir(osp.join(root, 'raw'))]
        edge_files = [f for f in directory if osp.isfile(f) and 'timestamp' in f]
        edge_files = sorted(edge_files, key=lambda x: int(x.split("/")[-1].split("_")[1]))

        domains = pd.read_csv(osp.join(root, 'raw', domain_file)).sort_values('domain_node')
        label_dict = {
            'domain_node': domains['type'].apply(lambda x: 1 if x == 'malicious' else (0 if x == 'benign' else -1)).values
        }
        print('Total labeled', len(label_dict['domain_node']))

        domains['domain_node']

        edge_indices = []
        edge_weights = []
        targets = []
        # data = HeteroData()

        if start is not None and end is not None:

            domain_features = pd.read_csv(osp.join(root, 'raw', 'domain_features.csv')).sort_values('domain_node')
            ip_features = pd.read_csv(osp.join(root, 'raw', 'ip_features.csv')).sort_values('ip_node')

            self.train_data, _ = self._load_data(domain_features, ip_features, label_dict, edge_files, start, end, num_val=num_val, balance_gt=balance_gt)
            
            self.test_data = []
            for test in test_list:
                # remove previous labeled domains from testing
                prev_edges = pd.DataFrame()
                for edge_file in edge_files[start:test-1]:
                    prev_edges = pd.concat([prev_edges, pd.read_csv(edge_file)], axis=0)     
                _, _, domain_set = self._load_graph(prev_edges, label_dict)

                test_label_dict = dict()
                test_label_dict['domain_node'] = [label if index not in domain_set else -1 for index,label in enumerate(label_dict['domain_node'])]

                test_data, _ = self._load_data(domain_features, ip_features, test_label_dict, edge_files, start, test, num_val=1.0, balance_gt=False)
                self.test_data.append(test_data)
            """data['domain_node'].x = torch.from_numpy(domain_features.set_index('domain_node').values).float()
            data['ip_node'].x = torch.from_numpy(ip_features.set_index('ip_node').values).float()

            train_edges = pd.DataFrame()
            for edge_file in edge_files[start:end]:
                train_edges = pd.concat([train_edges, pd.read_csv(edge_file)], axis=0)     

            timestamp_edges, temporal_labels = self._load_graph(train_edges, label_dict) 
            for key, value in timestamp_edges.items(): # assign edges
                data[key].edge_index = torch.from_numpy(value)
            data['domain_node'].y = torch.from_numpy(temporal_labels['domain_node'])

            labeled = temporal_labels['domain_node'] < 2
            labeled_indices = labeled.nonzero()[0]
            print('Labeled node count:', len(labeled_indices))
            
            # balance benign and mal nodes
            if balance_gt:
                mal_nodes = (temporal_labels['domain_node'] == 1).nonzero()[0]
                ben_nodes = (temporal_labels['domain_node'] == 0).nonzero()[0]
                min_count = min(len(mal_nodes), len(ben_nodes))
                # perm = torch.randperm(min_count)
                mal_nodes = mal_nodes[torch.randperm(len(mal_nodes))[:min_count]]
                ben_nodes = ben_nodes[torch.randperm(len(ben_nodes))[:min_count]]
                labeled_indices = np.concatenate((mal_nodes, ben_nodes))
                print('After balancing labeled count:', len(labeled_indices))
            

            n_nodes = len(labeled_indices)
            perm = torch.randperm(n_nodes)

            val_idx = labeled_indices[perm[:int(n_nodes * num_val)]]
            train_idx = labeled_indices[perm[int(n_nodes * num_val):]]

            for v, idx in [('train', train_idx),('val', val_idx)]:
                mask = torch.zeros(len(temporal_labels['domain_node']), dtype=torch.bool)
                mask[idx] = True
                data['domain_node'][f'{v}_mask'] = mask
"""
            # edge_indices.append(timestamp_edges)
            # edge_weights.append(None)
            # targets.append(temporal_labels)          
        else:
            for edge_file in edge_files:
                all_edges = pd.read_csv(edge_file)
                timestamp_edges, temporal_labels, _ = self._load_graph(all_edges, label_dict)

                edge_indices.append(timestamp_edges)
                edge_weights.append(None)
                targets.append(temporal_labels)

        domain_features = pd.read_csv(osp.join(root, 'raw', 'domain_features.csv')).sort_values('domain_node')
        ip_features = pd.read_csv(osp.join(root, 'raw', 'ip_features.csv')).sort_values('ip_node')

        feature_dict = {
            'domain_node': domain_features.set_index('domain_node').values,
            'ip_node': ip_features.set_index('ip_node').values
        }

        graph = self.train_data
        # graph = DynamicHeteroGraphStaticSignal(edge_indices, edge_weights, feature_dict, targets)
        return graph

    def _load_graph(self, all_edges, label_dict):
        edge_type_nodes = {
            'apex': ['domain_node', 'domain_node'],
            'similar': ['domain_node', 'domain_node'],
            'resolves': ['domain_node', 'ip_node']
        }
        domain_set = set()
        timestamp_edges = {}

        for edge_type, edges in all_edges.groupby('type'):
            src_type, dst_type = edge_type_nodes[edge_type]
            if src_type == 'domain_node':
                domain_set.update(edges['source'].astype(int).values.T)
            if dst_type == 'domain_node':
                domain_set.update(edges['target'].astype(int).values.T)
            # print(edge_file, src_type, dst_type, len(domain_set))                    
            key = (src_type, edge_type, dst_type)
            timestamp_edges[key] = edges.loc[:, ['source', 'target']].astype(int).values.T
        temporal_labels = {'domain_node':np.asarray([-1 if index not in domain_set else value for index, value in enumerate(label_dict['domain_node'])])}
        return timestamp_edges, temporal_labels, domain_set

    def _load_data(self, domain_features, ip_features, label_dict, edge_files, start, end, num_val=0.2, num_test=0.1, balance_gt=False):
        data = HeteroData()
        data['domain_node'].x = torch.from_numpy(domain_features.set_index('domain_node').values).float()
        data['ip_node'].x = torch.from_numpy(ip_features.set_index('ip_node').values).float()

        train_edges = pd.DataFrame()
        for edge_file in edge_files[start:end]:
            train_edges = pd.concat([train_edges, pd.read_csv(edge_file)], axis=0)     

        timestamp_edges, temporal_labels, domain_set = self._load_graph(train_edges, label_dict) 
        for key, value in timestamp_edges.items(): # assign edges
            data[key].edge_index = torch.from_numpy(value)
        data['domain_node'].y = torch.from_numpy(temporal_labels['domain_node'])

        labeled = temporal_labels['domain_node'] != -1
        labeled_indices = labeled.nonzero()[0]
        print('Labeled node count for {}, {}:'.format(start,end), len(labeled_indices))
        
        # balance benign and mal nodes
        if balance_gt:
            mal_nodes = (temporal_labels['domain_node'] == 1).nonzero()[0]
            ben_nodes = (temporal_labels['domain_node'] == 0).nonzero()[0]
            min_count = min(len(mal_nodes), len(ben_nodes))
            # perm = torch.randperm(min_count)
            mal_nodes = mal_nodes[torch.randperm(len(mal_nodes))[:min_count]]
            ben_nodes = ben_nodes[torch.randperm(len(ben_nodes))[:min_count]]
            labeled_indices = np.concatenate((mal_nodes, ben_nodes))
            print('After balancing labeled count:', len(labeled_indices))

        n_nodes = len(labeled_indices)
        perm = torch.randperm(n_nodes)

        test_idx = labeled_indices[perm[:int(n_nodes * num_test)]]
        val_idx = labeled_indices[perm[int(n_nodes * num_test) : int(n_nodes * (num_val + num_test))]]
        train_idx = labeled_indices[perm[int(n_nodes * (num_val + num_test)):]]

        for v, idx in [('train', train_idx),('val', val_idx),('test', test_idx)]:
            mask = torch.zeros(len(temporal_labels['domain_node']), dtype=torch.bool)
            mask[idx] = True
            data['domain_node'][f'{v}_mask'] = mask

        return data, domain_set