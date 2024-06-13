from typing import Callable, List, Optional, Union

import time, copy, random
import os.path as osp
import numpy as np
import torch
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset
from torch_sparse import SparseTensor

from torch_geometric.data import (Data, HeteroData)
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (to_torch_sparse_tensor, mask_to_index, sort_edge_index)
from H2GB.graphgym.config import cfg
from H2GB.graphgym.register import register_sampler
from torch_geometric.loader import (NeighborLoader, GraphSAINTRandomWalkSampler, HGTLoader, 
                                    RandomNodeLoader, LinkNeighborLoader)
from multiprocessing import Pool

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
    r"""
    A heterogeneous graph sampler that performs neighbor sampling as introduced
    in the `"Inductive Representation Learning on Large Graphs"
    <https://arxiv.org/abs/1706.02216>`_ paper.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    More specifically, :obj:`neighbor_sizes` in the configuration denotes
    how much neighbors are sampled for each node in each iteration.
    :class:`~torch_geometric.loader.NeighborLoader` takes in this list of
    :obj:`num_neighbors` and iteratively samples :obj:`num_neighbors[i]` for
    each node involved in iteration :obj:`i - 1`.

    Sampled nodes are sorted based on the order in which they were sampled.
    In particular, the first :obj:`batch_size` nodes represent the set of
    original mini-batch nodes.

    Args:
        dataset (Any): A :class:`~torch_geometric.data.InMemoryDataset` dataset object.
        batch_size (int): The number of seed nodes (first nodes in the batch).
        shuffle (bool): Whether to shuffle the data or not (default: :obj:`True`).
        split (str): Specify which data split (:obj:`train`, :obj:`val`, :obj:`test`) is
            for this sampler. This determines some sampling parameter loaded from the
            configuration file, such as :obj:`iter_per_epoch`.
    """

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

class AdjRowLoader():
    def __init__(self, data, idx, num_parts=100, full_epoch=False):
        """
        if not full_epoch, then just return one chunk of nodes
        """
        if isinstance(data, HeteroData):
            homo = data.to_homogeneous()
            node_idx = 0
            for id, node_type in enumerate(data.node_types):
                if node_type == cfg.dataset.task_entity:
                    node_idx = id
                    break
            mask = torch.isin(homo.edge_index[0, :], mask_to_index(homo.node_type == node_idx))
            homo.edge_index = homo.edge_index[:, mask]
            data = homo
        self.data = data
        self.full_epoch = full_epoch
        N = data.num_nodes
        self.node_feat = data.x
        self.label = data.y
        self.edge_index = data.edge_index
        self.edge_index = sort_edge_index(self.edge_index)
        first, last = self.edge_index[0, 0].item(), self.edge_index[0, -1].item() + 1
        self.part_spots = [first]
        self.part_nodes = [first]
        self.idx = idx
        self.mask = torch.zeros(N, dtype=torch.bool)#, device=device)
        self.mask[idx] = True
        num_edges = self.edge_index.shape[1]
        approx_size = num_edges // num_parts
        approx_part_spots = list(range(approx_size, num_edges, approx_size))[:-1]
        for idx in approx_part_spots:
            last_node = -1 if idx == 0 else self.edge_index[0,self.part_spots[-1] - 1]
            curr_idx = idx
            while curr_idx < self.edge_index.shape[1] and self.edge_index[0,curr_idx] == last_node:
                curr_idx += 1
            curr_node = self.edge_index[0,curr_idx].item()
            while curr_idx < self.edge_index.shape[1] and self.edge_index[0,curr_idx] == curr_node:
                curr_idx += 1
            self.part_nodes.append(self.edge_index[0, curr_idx].item())
            self.part_spots.append(curr_idx)

            # Ensure we have at least a node not masked
            # node_ids = list(range(self.part_nodes[-2], self.part_nodes[-1]))
            # if self.mask[node_ids].sum() == 0:
            #     self.part_nodes.pop(-1)
            #     self.part_spots.pop(-1)
        self.part_nodes.append(last)
        self.part_spots.append(self.edge_index.shape[1])
        print(self.part_nodes, self.part_spots)

    def __len__(self):
        return len(self.part_nodes) - 1
    
    def __iter__(self):
        self.k = 0
        return self
    
    def __next__(self):
        if self.k >= len(self.part_spots)-1:
            raise StopIteration
            
        if not self.full_epoch:
            self.k = np.random.randint(len(self.part_spots)-1)
            
        tg_data = Data()
        batch_edge_index = self.edge_index[:, self.part_spots[self.k]:self.part_spots[self.k+1]]
        node_ids = list(range(self.part_nodes[self.k], self.part_nodes[self.k+1]))
        tg_data.node_ids = node_ids
        tg_data.edge_index = batch_edge_index
        batch_node_feat = self.node_feat[node_ids]
        tg_data.x = batch_node_feat
        tg_data.edge_attr = None
        tg_data.y = self.label[node_ids]
        tg_data.num_nodes = len(node_ids)
        mask = self.mask[node_ids]
        tg_data.mask = mask
        tg_data.train_mask = mask
        tg_data.val_mask = mask
        tg_data.test_mask = mask
        self.k += 1
        
        if not self.full_epoch:
            self.k = float('inf')
        return tg_data
@register_sampler('linkx')
def get_LINKXLoader(dataset, batch_size, shuffle=True, split='train'):
    data = dataset[0]
    if isinstance(data, HeteroData):
        homo = data.to_homogeneous()
        node_idx = 0
        for idx, node_type in enumerate(data.node_types):
            if node_type == cfg.dataset.task_entity:
                node_idx = idx
                break
        mask = data[cfg.dataset.task_entity][f'{split}_mask']
        mask = mask_to_index(homo.node_type == node_idx)[mask]
    else:
        mask = mask_to_index(data[f'{split}_mask'])
    
    
    start = time.time()
    loader_train = \
        AdjRowLoader(
            data,
            mask,
            num_parts=100,
            full_epoch=True
        )
    end = time.time()
    print(f'Data {split} loader initialization took:', round(end - start, 3), 'seconds.')
    
    return loader_train


@register_sampler('graphsaint_rw')
def get_GrashSAINTRandomWalkLoader(dataset, batch_size, shuffle=True, split='train'):
    r"""
    A homogeneous random-walk based graph sampler from the `"GraphSAINT: Graph
    Sampling Based Inductive Learning Method"
    <https://arxiv.org/abs/1907.04931>`_ paper.
    Given a graph in a :obj:`data` object, this class samples nodes and
    constructs subgraphs that can be processed in a mini-batch fashion.
    Normalization coefficients for each mini-batch are given via
    :obj:`node_norm` and :obj:`edge_norm` data attributes.

    Args:
        dataset (Any): A :class:`~torch_geometric.data.InMemoryDataset` dataset object.
        batch_size (int): The number of seed nodes (first nodes in the batch).
        shuffle (bool): Whether to shuffle the data or not (default: :obj:`True`).
        split (str): Specify which data split (:obj:`train`, :obj:`val`, :obj:`test`) is
            for this sampler. This determines some sampling parameter loaded from the
            configuration file, such as :obj:`iter_per_epoch`.
    """

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
    r"""
    A heterogeneous graph sampler that from the `"Heterogeneous Graph
    Transformer" <https://arxiv.org/abs/2003.01332>`_ paper.
    This loader allows for mini-batch training of GNNs on large-scale graphs
    where full-batch training is not feasible.

    The sampler tries to (1) keep a similar number of nodes and edges for
    each type and (2) keep the sampled sub-graph dense to minimize the
    information loss and reduce the sample variance.

    Methodically, HGSampler keeps track of a node budget for each node type,
    which is then used to determine the sampling probability of a node.
    In particular, the probability of sampling a node is determined by the
    number of connections to already sampled nodes and their node degrees.
    With this, HGSampler will sample a fixed amount of neighbors for each
    node type in each iteration, as given by the :obj:`neighbor_sizes`
    argument from the configuration.

    Sampled nodes are sorted based on the order in which they were sampled.
    In particular, the first :obj:`batch_size` nodes represent the set of
    original mini-batch nodes.

    Args:
        dataset (Any): A :class:`~torch_geometric.data.InMemoryDataset` dataset object.
        batch_size (int): The number of seed nodes (first nodes in the batch).
        shuffle (bool): Whether to shuffle the data or not (default: :obj:`True`).
        split (str): Specify which data split (:obj:`train`, :obj:`val`, :obj:`test`) is
            for this sampler. This determines some sampling parameter loaded from the
            configuration file, such as :obj:`iter_per_epoch`.
    """


    # Note: sending the whole dataset to GPU seems has no performance difference. But they would occupy 24GB VRAM.
    # data = dataset[0].to(cfg.device, 'x', 'y')
    if len(cfg.train.neighbor_sizes_dict) > 0:
        sample_sizes = eval(cfg.train.neighbor_sizes_dict)
    else:
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
        data.x = data.x.nan_to_num()

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


class LocalSampler(torch.utils.data.DataLoader):
    def __init__(self, data, #edge_index: Union[Tensor, SparseTensor],
                 sizes: List[int], node_idx: Optional[Tensor] = None,
                 num_nodes: Optional[int] = None, return_e_id: bool = True,
                 transform: Callable = None, **kwargs):

        self.data = data
        edge_index = data.edge_index.to('cpu')

        if 'collate_fn' in kwargs:
            del kwargs['collate_fn']
        if 'dataset' in kwargs:
            del kwargs['dataset']

        # Save for Pytorch Lightning...
        self.edge_index = edge_index
        self.node_idx = node_idx
        self.num_nodes = num_nodes

        self.sizes = sizes
        self.return_e_id = return_e_id
        self.transform = transform
        self.is_sparse_tensor = isinstance(edge_index, SparseTensor)
        self.__val__ = None

        # Obtain a *transposed* `SparseTensor` instance.
        if not self.is_sparse_tensor:
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.bool):
                num_nodes = node_idx.size(0)
            if (num_nodes is None and node_idx is not None
                    and node_idx.dtype == torch.long):
                num_nodes = max(int(edge_index.max()), int(node_idx.max())) + 1
            if num_nodes is None:
                num_nodes = int(edge_index.max()) + 1

            value = torch.arange(edge_index.size(1)) if return_e_id else None
            self.adj_t = SparseTensor(row=edge_index[0], col=edge_index[1],
                                      value=value,
                                      sparse_sizes=(num_nodes, num_nodes)).t()
        else:
            adj_t = edge_index
            if return_e_id:
                self.__val__ = adj_t.storage.value()
                value = torch.arange(adj_t.nnz())
                adj_t = adj_t.set_value(value, layout='coo')
            self.adj_t = adj_t

        self.adj_t.storage.rowptr()

        if node_idx is None:
            node_idx = torch.arange(self.adj_t.sparse_size(0))
        elif node_idx.dtype == torch.bool:
            node_idx = node_idx.nonzero(as_tuple=False).view(-1)

        super().__init__(
            node_idx.view(-1).tolist(), collate_fn=self.sample, **kwargs)

    def sample(self, batch):
        if not isinstance(batch, Tensor):
            batch = torch.tensor(batch)
        batch_size: int = len(batch)

        edge_index, edge_dist = [], []
        for i in range(len(batch)) :
            out = self.sample_one(batch[i:i+1])
            edge_index.append(out[0])
            edge_dist.append(out[1])
        edge_index = torch.cat(edge_index, dim=1)
        edge_dist = torch.cat(edge_dist, dim=1)

        node_idx = torch.unique(edge_index[0]) # source nodes, will include target
        node_idx_flag = torch.tensor([i not in batch for i in node_idx])
        node_idx = node_idx[node_idx_flag]
        node_idx = torch.cat([batch, node_idx])

        # relabel
        node_idx_all = torch.zeros(self.num_nodes, dtype=torch.long)
        node_idx_all[node_idx] = torch.arange(node_idx.size(0))
        edge_index = node_idx_all[edge_index]

        data = Data()
        data.x = self.data.x[node_idx]
        data.y = self.data.y[node_idx]
        data.edge_index = edge_index
        data.edge_attr = edge_dist
        data.n_id = node_idx
        data.batch_size = batch_size
        data.pos_enc = self.data.pestat_Hetero_Node2Vec[node_idx[:batch_size]] #torch.zeros((batch_size, cfg.gnn.dim_inner))
        return data
        # return torch.cat([edge_index, edge_dist], dim=0), node_idx, batch_size

    def sample_one(self, idx):
        assert idx.dim() == 1 and len(idx) == 1

        n_id = idx
        ptrs = []
        for size in self.sizes:
            adj_t, n_id = self.adj_t.sample_adj(n_id, size, replace=False)
            # e_id = adj_t.storage.value()
            total_target = adj_t.sparse_sizes()[::-1] # (total, target)
            total_size = total_target[0]
            ptrs.append(total_size)

        target = torch.tensor([idx.item()] * len(n_id))
        dist = torch.ones(len(n_id))

        for i, ptr in enumerate(reversed(ptrs)) :
            dist[:ptr] = len(self.sizes) - i
        dist[0] = 0
        edge_dist = dist.long()
        # edge_index = torch.stack([target, n_id]) #BUG
        edge_index = torch.stack([n_id, target]) #edge_index[0]:source, edge_index[1]:target
        # https://pytorch-geometric.readthedocs.io/en/latest/notes/sparse_tensor.html

        edge_dist_count = [ptrs[i+1]-ptrs[i] for i in range(len(ptrs)-1)]
        edge_dist_count = [1, ptrs[0]-1] + edge_dist_count
        edge_dist_count = torch.tensor(edge_dist_count)
        edge_dist = torch.stack([edge_dist, edge_dist_count[edge_dist]])

        return edge_index, edge_dist

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(sizes={self.sizes})'

@register_sampler('goat')
def get_GOATLoader(dataset, batch_size, shuffle=True, split='train'):
    data = dataset[0]
    if isinstance(data, HeteroData):
        homo = data.to_homogeneous()
        node_idx = 0
        for idx, node_type in enumerate(dataset[0].node_types):
            if node_type == cfg.dataset.task_entity:
                node_idx = idx
                break
        node_idx = mask_to_index(homo.node_type == node_idx)[data[cfg.dataset.task_entity][f'{split}_mask']]
    else:
        homo = data
        node_idx = mask_to_index(data[f'{split}_mask'])
    sample_sizes = cfg.train.neighbor_sizes

    start = time.time()
    loader_train = \
        LoaderWrapper( \
            LocalSampler(
                homo,
                sizes=sample_sizes,
                node_idx=node_idx,
                num_nodes=homo.num_nodes,
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
    is_hetero = False
    if isinstance(data, HeteroData):
        is_hetero = True
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
        print(node_chunks)

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

    if is_hetero:
        # Seems like the to_homogeneous() not return the correct mask.
        # It has been reported to Github: https://github.com/pyg-team/pytorch_geometric/issues/8856
        mask = dataset[0][cfg.dataset.task_entity][f"{split}_mask"]
        features = data.x # Provide all node features
        label = data.y # Provide all node labels # [data.node_type == node_idx]
        seq = node_seq[mask] # Provide split indices
        batch_data = TensorDataset(features.unsqueeze(0), label.unsqueeze(0), seq.unsqueeze(0))
        # batch_data = TensorDataset(features, label, seq)
    else:
        mask = data[f"{split}_mask"]
        # features = data.x[node_seq][mask]
        # label = data.y[mask]
        # batch_data = TensorDataset(features.unsqueeze(0), label.unsqueeze(0))
    loader_train = DataLoader(batch_data, batch_size=None, #cfg.train.batch_size,#None, 
                              shuffle=shuffle,
                              num_workers=cfg.num_workers,
                              pin_memory=cfg.train.pin_memory,
                              persistent_workers=cfg.train.persistent_workers)
    # loader_train = batch_data
    
    return loader_train
