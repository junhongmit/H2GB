import time, copy, random
import os.path as osp
from tqdm import tqdm
from typing import Union
from networkx import number_of_nodes
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from torch_geometric.data import (Data, HeteroData, InMemoryDataset)
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (to_torch_sparse_tensor, mask_to_index)
from H2GB.graphgym.config import cfg
from H2GB.graphgym.register import register_sampler
from torch_geometric.loader import (NeighborLoader, GraphSAINTRandomWalkSampler, HGTLoader, 
                                    RandomNodeLoader, LinkNeighborLoader)
from multiprocessing import Pool
# from H2GB.sampler.hgt_loader import HGTLoader

class LoaderWrapper:
    def __init__(self, dataloader, n_step=-1, split='train'):
        self.step = n_step if n_step > 0 else len(dataloader)
        self.idx = 0
        self.loader = dataloader
        self.split = split
        self.iter_loader = iter(dataloader)
    
    def __iter__(self):
        return self

    def __len__(self):
        if self.step > 0:
            return self.step
        else:
            return len(self.loader)

    def __next__(self):
        # if reached number of steps desired, stop
        if self.idx == self.step or self.idx == len(self.loader):
            self.idx = 0
            if self.split in ['val', 'test']:
                # Make sure we are always using the same set of data for evaluation
                self.iter_loader = iter(self.loader)
            raise StopIteration
        else:
            self.idx += 1
        # while True
        try:
            return next(self.iter_loader)
        except StopIteration:
            # reinstate iter_loader, then continue
            self.iter_loader = iter(self.loader)
            return next(self.iter_loader)
    
    def set_step(self, n_step):
        self.step = n_step

def convert_batch(batch):
    for node_type, x in batch.x_dict.items():
        batch[node_type].x = x.to(torch.float) 
    for node_type, y in batch.y_dict.items():
        batch[node_type].y = y.to(torch.long) 
    return batch
@register_sampler('hetero_neighbor')
def get_NeighborLoader(dataset, batch_size, shuffle=True, split='train'):
    data = dataset[0]
    sample_sizes = {key: cfg.train.neighbor_sizes for key in data.edge_types}
    
    start = time.time()
    loader_train = \
        LoaderWrapper( \
            NeighborLoader(
                data,
                num_neighbors=sample_sizes,
                input_nodes=(cfg.dataset.task_entity, dataset.data[cfg.dataset.task_entity][split + '_mask']),
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=cfg.num_workers,
                persistent_workers=cfg.train.persistent_workers,
                pin_memory=cfg.train.pin_memory,
                transform=convert_batch),
            getattr(cfg, 'val' if split == 'test' else split).iter_per_epoch,
            split
        )
    end = time.time()
    print(f'Data {split} loader initialization took:', round(end - start, 3), 'seconds.')
    
    return loader_train

# def convert_batch_to_hetero(data):
#     hetero_data = HeteroData.from_dict({
#         "node_type": {
#             "x": data.x, 
#             "y": data.y, 
#             "train_mask": data.train_mask,
#             "val_mask": data.val_mask,
#             "test_mask": data.test_mask,
#         }, 
#         ("node_type", "edge_type", "node_type"): data.edge_index})
#     return hetero_data
@register_sampler('graphsaint_rw')
def get_GrashSAINTRandomWalkLoader(dataset, batch_size, shuffle=True, split='train'):
    data = dataset[0]
    if isinstance(data, HeteroData):
        metadata = data.metadata()
        assert len(metadata[0]) + len(metadata[1]) == 2, "GraphSAINT sampler needs to be used on homogeneous graph!"

    start = time.time()
    loader_train = \
        GraphSAINTRandomWalkSampler(data.to_homogeneous(),
                                    batch_size=batch_size,
                                    walk_length=cfg.train.walk_length,
                                    num_steps=getattr(cfg, 'val' if split == 'test' else split).iter_per_epoch,
                                    sample_coverage=0,
                                    shuffle=shuffle,
                                    num_workers=cfg.num_workers,
                                    pin_memory=True)
    end = time.time()
    print(f'Data {split} loader initialization took:', round(end - start, 3), 'seconds.')
    
    return loader_train

@register_sampler('hetero_random_node')
def get_RandomNodeLoader(dataset, batch_size, shuffle=True, split='train'):
    data = dataset[0]

    start = time.time()
    loader_train = \
        RandomNodeLoader(data,
                        num_parts=cfg.train.num_parts,
                        shuffle=shuffle,
                        num_workers=cfg.num_workers,
                        pin_memory=True)
    end = time.time()
    print(f'Data {split} loader initialization took:', round(end - start, 3), 'seconds.')
    
    return loader_train

@register_sampler('hgt')
def get_HGTloader(dataset, batch_size, shuffle=True, split='train'):
    # Note: sending the whole dataset to GPU seems has no performance difference. But they would occupy 24GB VRAM.
    # data = dataset[0].to(cfg.device, 'x', 'y')
    sample_sizes = {key: cfg.train.neighbor_sizes for key in dataset.data.node_types}
    
    loader_train = \
        LoaderWrapper( \
            HGTLoader(dataset[0],
                    num_samples=sample_sizes,
                    input_nodes=(cfg.dataset.task_entity, dataset.data[cfg.dataset.task_entity][split + '_mask']),
                    # Use a batch size for sampling training nodes of input type
                    batch_size=batch_size, shuffle=shuffle,
                    num_workers=cfg.num_workers,
                    persistent_workers=cfg.train.persistent_workers,
                    pin_memory=cfg.train.pin_memory,
                    transform=convert_batch),
            getattr(cfg, 'val' if split == 'test' else split).iter_per_epoch,
            split
        )

    # train_input_nodes = ('paper', dataset.data[cfg.dataset.task_entity].train_mask)
    # kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}
    # loader_train = HGTLoader(dataset[0], num_samples={key: [128] * 4 for key in dataset[0].node_types}, shuffle=True,
    #                             input_nodes=train_input_nodes, **kwargs)

    return loader_train

def hop2seq(adj, features, K):
    # size= (N, 1, K+1, d )
    device = 'cpu' # If doesn't fit in the GPU, change device to cpu
    # device = cfg.device

    adj = adj.to(torch.device(device))
    features = features.to(torch.device(device))
    nodes_features = torch.empty(features.shape[0], K+1, features.shape[1], device=device)

    index = torch.arange(features.shape[0])
    nodes_features[index, 0, :] = features[index]
    # for i in range(features.shape[0]):
    #     nodes_features[i, 0, 0, :] = features[i]

    x = features #+ torch.zeros_like(features)

    for i in range(K):
        x = torch.matmul(adj, x)

        nodes_features[index, i + 1, :] = x[index]
        # for index in range(features.shape[0]):
        #     nodes_features[index, 0, i + 1, :] = x[index]        

    return nodes_features.to('cpu')

@register_sampler('nag')
def get_NAGloader(dataset, batch_size, shuffle=True, split='train'):
    r'''
    Full batch loader for NAGphormer.
    '''
    data = dataset[0]
    if isinstance(data, HeteroData):
        data = data.to_homogeneous()

    # Pay attention: the adj_t matters! Originally, the NAGphormer
    # author applied matmul on adj and x, this is not an issue for
    # undirected graph. But our experiments were done on directed graph!
    adj_t = to_torch_sparse_tensor(data.edge_index, size=(data.num_nodes, data.num_nodes)).t()
    features = hop2seq(adj_t, data.x, cfg.gnn.hops)  # return (N, hops+1, d)

    if isinstance(dataset[0], HeteroData):
        # Seems like the to_homogeneous() not return the correct mask.
        # It has been reported to Github: https://github.com/pyg-team/pytorch_geometric/issues/8856
        node_idx = 0
        for idx, node_type in enumerate(dataset[0].node_types):
            if node_type == cfg.dataset.task_entity:
                node_idx = idx
                break
        features = features[data.node_type == node_idx][dataset[0][cfg.dataset.task_entity][f"{split}_mask"]]
        label = data.y[data.node_type == node_idx][dataset[0][cfg.dataset.task_entity][f"{split}_mask"]]
        batch_data = TensorDataset(features, label)
        # print(features, label)
    else:
        batch_data = TensorDataset(features[data[f"{split}_mask"]], data.y[data[f"{split}_mask"]])

    loader_train = DataLoader(batch_data, batch_size=batch_size, shuffle=shuffle,
                              num_workers=cfg.num_workers,
                              pin_memory=cfg.train.pin_memory,
                              persistent_workers=cfg.train.persistent_workers)
    
    return loader_train


def convert_hop2seq(batch):
    def hop2seq(adj, features, K):
        # size= (N, 1, K+1, d )
        # adj = adj.to(torch.device(cfg.device))
        # features = features.to(torch.device(cfg.device))
        nodes_features = torch.empty(features.shape[0], K+1, features.shape[1])

        index = torch.arange(features.shape[0])
        nodes_features[index, 0, :] = features[index]
        # for i in range(features.shape[0]):
        #     nodes_features[i, 0, 0, :] = features[i]

        x = features #+ torch.zeros_like(features)

        for i in range(K):
            x = torch.matmul(adj, x)
            nodes_features[index, i + 1, :] = x[index]       

        return nodes_features#.to('cpu')
    if isinstance(batch, HeteroData):
        homo = batch.to_homogeneous()
    
    # Pay attention: the adj_t matters! Originally, the NAGphormer
    # author applied matmul on adj and x, this is not an issue for
    # undirected graph. But our experiments were done on directed graph!
    adj_t = to_torch_sparse_tensor(homo.edge_index, size=(homo.num_nodes, homo.num_nodes)).t()
    features = hop2seq(adj_t, homo.x, cfg.gnn.hops)  # return (N, hops+1, d)
    return [features[:batch[cfg.dataset.task_entity].batch_size],\
             homo.y[:batch[cfg.dataset.task_entity].batch_size]]
@register_sampler('nag+hgt')
def get_NAGHGTloader(dataset, batch_size, shuffle=True, split='train'):
    r'''
    Minibatch loader for NAGphormer
    '''
    sample_sizes = {key: cfg.train.neighbor_sizes for key in dataset.data.node_types}
    
    loader_train = \
        LoaderWrapper( \
            HGTLoader(dataset[0],
                    num_samples=sample_sizes,
                    input_nodes=(cfg.dataset.task_entity, dataset.data[cfg.dataset.task_entity][f"{split}_mask"]),
                    # Use a batch size for sampling training nodes of input type
                    batch_size=batch_size, shuffle=shuffle,
                    num_workers=cfg.num_workers,
                    persistent_workers=cfg.train.persistent_workers,
                    pin_memory=cfg.train.pin_memory,
                    transform=convert_hop2seq),
            getattr(cfg, 'val' if split == 'test' else split).iter_per_epoch,
            split
        )

    # train_input_nodes = ('paper', dataset.data[cfg.dataset.task_entity].train_mask)
    # kwargs = {'batch_size': 1024, 'num_workers': 6, 'persistent_workers': True}
    # loader_train = HGTLoader(dataset[0], num_samples={key: [128] * 4 for key in dataset[0].node_types}, shuffle=True,
    #                             input_nodes=train_input_nodes, **kwargs)

    return loader_train

def sample_neighbors_for_nodes(node_indices, neighbors, len_seq):
    results = []

    for idx, node in enumerate(node_indices):
        node_seq = [0 for _ in range(len_seq)]
        node_seq[0] = node
        cnt = 1
        scnt = 0
        current_node = node
        while (cnt < len_seq):
            sample_list = neighbors[current_node]
            nsampled = max(len(sample_list), 1)
            sampled_list = random.sample(sample_list, nsampled)
            for i in range(nsampled):
                node_seq[cnt] = sampled_list[i]
                cnt += 1
                if cnt == len_seq:
                    break
            scnt += 1
            current_node = node_seq[scnt]
        results.append(node_seq)

    return results

@register_sampler('hin')
def get_HINormerLoader(dataset, batch_size, shuffle=True, split='train'):
    data = dataset[0]
    if isinstance(data, HeteroData):
        data = data.to_homogeneous()
        node_idx = 0
        for idx, node_type in enumerate(dataset[0].node_types):
            if node_type == cfg.dataset.task_entity:
                node_idx = idx
                break

    task_nodes = mask_to_index(data.node_type == node_idx)
    # The neighbor_sizes is a list, so we extract the first element as the neighbor size here
    len_seq = cfg.train.neighbor_sizes[0]
    print('len_seq:', len_seq)
    if not osp.exists(osp.join('/nobackup/users/junhong/Data', f'{dataset.name}.pt')):
        # Initialize the node sequence tensor
        node_seq = torch.zeros((task_nodes.numel(), len_seq), dtype=torch.long)

        # Convert edge_index to a list of neighbors for each node
        neighbors = [set() for _ in range(data.num_nodes)]
        for source, target in data.edge_index.t().numpy().tolist():
            neighbors[source].add(target)
            # If the graph is undirected, uncomment the next line
            neighbors[target].add(source)

        st = time.time()
        num_workers = 48
        avg = len(task_nodes) / float(num_workers)
        node_chunks = []
        last = 0.0
        while last < len(task_nodes):
            node_chunks.append(task_nodes[int(last):int(last + avg)].numpy())
            last += avg

        # Initialize the pool of workers
        with Pool(processes=num_workers) as pool:
            # Map each chunk of nodes to the worker function
            results = pool.starmap(sample_neighbors_for_nodes, [(chunk, neighbors, len_seq) for chunk in node_chunks])
        node_seq = torch.cat([torch.tensor(i) for i in results], dim=0)
        torch.save(node_seq, osp.join('/nobackup/users/junhong/Data', f'{dataset.name}.pt'))
        print(time.time() - st)
    node_seq = torch.load(osp.join('/nobackup/users/junhong/Data', f'{dataset.name}.pt'))

    # with Pool(processes=8) as pool:
    #     node_info_list = [(node, len_seq, neighbors) for node in task_nodes]
    #     result = list(tqdm(pool.imap(sample_neighbors_for_node, node_info_list), total=len(task_nodes)))
    print(node_seq, node_seq.shape)
    
    # # Sample neighbors for each node
    # for idx, node in tqdm(enumerate(task_nodes), total=len(task_nodes)):
    #     node_seq[idx, 0] = node  # Start with the node itself
    #     cnt = 1
    #     scnt = 0
    #     current_node = node
    #     while (cnt < len_seq):
    #         sample_list = neighbors[current_node]
    #         nsampled = max(len(sample_list), 1)
    #         sampled_list = random.sample(sample_list, nsampled)
    #         for i in range(nsampled):
    #             node_seq[idx, cnt] = sampled_list[i]
    #             cnt += 1
    #             if cnt == len_seq:
    #                 break
    #         scnt += 1
    #         current_node = node_seq[idx, scnt].item()

    if isinstance(dataset[0], HeteroData):
        # Seems like the to_homogeneous() not return the correct mask.
        # It has been reported to Github: https://github.com/pyg-team/pytorch_geometric/issues/8856
        node_idx = 0
        for idx, node_type in enumerate(dataset[0].node_types):
            if node_type == cfg.dataset.task_entity:
                node_idx = idx
                break
        features = data.x
        seq = node_seq[dataset[0][cfg.dataset.task_entity][f"{split}_mask"]]
        label = data.y[data.node_type == node_idx]
        batch_data = TensorDataset(features.unsqueeze(0), label.unsqueeze(0), seq.unsqueeze(0))
        # batch_data = TensorDataset(features, label, seq)
    else:
        batch_data = TensorDataset(data.x[node_seq][data[f"{split}_mask"]], data.y[data[f"{split}_mask"]])
    loader_train = DataLoader(batch_data, batch_size=None, #cfg.train.batch_size,#None, 
                              shuffle=shuffle,
                              num_workers=cfg.num_workers,
                              pin_memory=cfg.train.pin_memory,
                              persistent_workers=cfg.train.persistent_workers)
    # loader_train = batch_data
    
    return loader_train


# def convert_multigraph(batch):
#     if isinstance(batch, HeteroData):
#         num_original_nodes = batch.num_nodes
#         for edge_type in batch.edge_types:
#             num_edges = batch[edge_type].edge_index.size(1)
#             dummy_nodes_start = num_original_nodes
            
#             _, inverse_indices, counts = torch.unique(batch[edge_type].edge_index, return_inverse=True, return_counts=True, dim=1)
#             is_duplicate = counts > 1

#             # Placeholder for new edges
#             new_edges = []
#             new_edge_attr = [] if batch[edge_type].edge_attr is not None else None
#             new_edge_label = [] if batch[edge_type].edge_label is not None else None
#             dummy_node_counter = 0

#             for i in range(num_edges):
#                 src, dst = data.edge_index[:, i]
#                 if is_duplicate[inverse_indices[i]]:
#                     # Add dummy node for duplicate edges
#                     dummy_node = dummy_nodes_start + dummy_node_counter
#                     dummy_node_counter += 1
#                     new_edges.extend([[src, dummy_node], [dummy_node, dst]])
#                 else:
#                     # Keep unique edges as they are
#                     new_edges.append([src, dst])
                    
#                 if new_edge_attr is not None and is_duplicate[inverse_indices[i]]:
#                     attr = data.edge_attr[i].tolist()
#                     new_edge_attr.extend([attr, attr])  # Duplicate the attribute for both new edges
#                 elif new_edge_attr is not None:
#                     new_edge_attr.append(data.edge_attr[i].tolist())
                    
#                 if new_edge_label is not None and is_duplicate[inverse_indices[i]]:
#                     label = data.edge_label[i].tolist()
#                     new_edge_label.extend([label, label])  # Duplicate the label for both new edges
#                 elif new_edge_label is not None:
#                     new_edge_label.append(data.edge_label[i].tolist())

#             new_edge_index = torch.tensor(new_edges, dtype=torch.long).t()
#             if new_edge_attr is not None:
#                 new_edge_attr = torch.tensor(new_edge_attr, dtype=data.edge_attr.dtype)
        
#         new_data = HeteroData(edge_index=new_edge_index, edge_attr=new_edge_attr, num_nodes=num_original_nodes+dummy_node_counter)
#     else:
#         pass
#     return data

class AddEgoIds(BaseTransform):
    r"""Add IDs to the centre nodes of the batch.
    """
    def __init__(self):
        pass

    def __call__(self, data: Union[Data, HeteroData]):
        x = data.x if not isinstance(data, HeteroData) else data['node'].x
        device = x.device
        ids = torch.zeros((x.shape[0], 1), device=device)
        if not isinstance(data, HeteroData):
            nodes = torch.unique(data.edge_label_index.view(-1)).to(device)
        else:
            nodes = torch.unique(data['node', 'to', 'node'].edge_label_index.view(-1)).to(device)
        ids[nodes] = 1
        if not isinstance(data, HeteroData):
            data.x = torch.cat([x, ids], dim=1)
        else: 
            data['node'].x = torch.cat([x, ids], dim=1)
        
        return data

@register_sampler('link_neighbor')
def get_LinkNeighborLoader(dataset, batch_size, shuffle=True, split='train'):
    task = cfg.dataset.task_entity
    # data = dataset[0]

    # def z_norm(data):
    #     std = data.std(0).unsqueeze(0)
    #     std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
    #     return (data - data.mean(0).unsqueeze(0)) / std

    # Slice the training/validation/test set based on the edge mask
    # data_copy = copy.deepcopy(data)
    # for edge_type in data.edge_types:
    #     mask = data_copy[edge_type][f'{split}_edge_mask']
    #     data_copy[edge_type].edge_index = data_copy[edge_type].edge_index[:, mask]
    #     # Independent normalization on each dataset is important for performance
    #     data_copy[edge_type].edge_attr = z_norm(data_copy[edge_type].edge_attr[mask])
    #     if hasattr(data_copy[edge_type], 'y'):
    #         data_copy[edge_type].y = data_copy[edge_type].y[mask]

    dataset_idx = {'train': 0, 'val': 1, 'test': 2}
    data = dataset[dataset_idx[split]]
    mask = data[task].split_mask
    edge_label_index = data[task].edge_index[:, mask]
    edge_label = data[task].y[mask]
    loader_train = \
        LoaderWrapper( \
            LinkNeighborLoader(
                data=data,
                num_neighbors=cfg.train.neighbor_sizes,
                # neg_sampling_ratio=0.0,
                edge_label_index=(task, edge_label_index),
                edge_label=edge_label,
                batch_size=batch_size,
                num_workers=cfg.num_workers,
                shuffle=shuffle,
                transform=AddEgoIds()
            ),
            getattr(cfg, 'val' if split == 'test' else split).iter_per_epoch,
            split
        )

    return loader_train