import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_sparse import SparseTensor, matmul
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv.gcn_conv import gcn_norm

from H2GB.graphgym.models import head  # noqa, register module
from H2GB.graphgym import register as register
from H2GB.graphgym.config import cfg
from H2GB.graphgym.models.gnn import GNNPreMP
from H2GB.graphgym.register import register_network
from H2GB.graphgym.models.layer import BatchNorm1dNode

class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in, dataset):
        super(FeatureEncoder, self).__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner, dataset)
            if cfg.dataset.node_encoder_bn:
                # self.node_encoder_bn = BatchNorm1dNode(
                #     new_layer_config(cfg.gnn.dim_inner, -1, -1, has_act=False,
                #                      has_bias=False, cfg=cfg))
                self.node_encoder_bn = BatchNorm1dNode(cfg.gnn.dim_inner)
            # Update dim_in to reflect the new dimension of the node features
            if self.is_hetero:
                self.dim_in = {node_type: cfg.gnn.dim_inner for node_type in dim_in}
            else:
                self.dim_in = cfg.gnn.dim_inner
        if cfg.dataset.edge_encoder:
            # Hard-limit max edge dim for PNA.
            if 'PNA' in cfg.gt.layer_type:
                cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
            else:
                cfg.gnn.dim_edge = cfg.gnn.dim_inner
            # Encode integer edge features via nn.Embeddings
            EdgeEncoder = register.edge_encoder_dict[
                cfg.dataset.edge_encoder_name]
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge, dataset)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(cfg.gnn.dim_edge)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch

@register_network('LPModel')
class MultiLP(torch.nn.Module):
    """ label propagation, with possibly multiple hops of the adjacency """
    
    def __init__(self, dim_in, dim_out, dataset, 
                 num_iters=50, mult_bin=False):
        super(MultiLP, self).__init__()
        self.out_channels = dim_out
        self.alpha = cfg.gnn.alpha
        self.hops = cfg.gnn.hops
        self.num_iters = num_iters
        self.mult_bin = mult_bin # handle multiple binary tasks
        self.placeholder = torch.nn.Parameter(torch.empty(1))
        
    def forward(self, batch):
        if isinstance(batch, HeteroData):
            homo = batch.to_homogeneous()
            label, edge_index = homo.y, homo.edge_index
            node_type_tensor = homo.node_type
            for idx, node_type in enumerate(batch.node_types):
                if node_type == cfg.dataset.task_entity:
                    node_idx = idx
                    break
            train_idx = torch.arange(batch.num_nodes, device=node_type_tensor.device)[node_type_tensor == node_idx]
            train_idx = train_idx[batch[cfg.dataset.task_entity].train_mask]
        else:
            label, edge_index = batch.y, batch.edge_index
        n = batch.num_nodes
        edge_weight=None

        if isinstance(edge_index, torch.Tensor):
            edge_index, edge_weight = gcn_norm( 
                edge_index, edge_weight, n, False)
            row, col = edge_index
            # transposed if directed
            adj_t = SparseTensor(row=col, col=row, value=edge_weight, sparse_sizes=(n, n))
        elif isinstance(edge_index, SparseTensor):
            edge_index = gcn_norm(
                edge_index, edge_weight, n, False)
            edge_weight=None
            adj_t = edge_index

        binary = False
        out_channels = self.out_channels
        if self.out_channels == 1:
            binary = True
            # binary
            out_channels = 2
        y = torch.zeros((n, out_channels)).to(adj_t.device())
        if label.dim() == 1 or label.shape[1] == 1:
            # binary or multi-class
            y[train_idx] = F.one_hot(label[train_idx], out_channels).to(y) # .squeeze(1)
        elif self.mult_bin:
            y = torch.zeros((n, 2*out_channels)).to(adj_t.device())
            for task in range(label.shape[1]):
                y[train_idx, 2*task:2*task+2] = F.one_hot(label[train_idx, task], 2).to(y)
        else:
            y[train_idx] = label[train_idx].to(y)
        result = y.clone()
        for _ in range(self.num_iters):
            for _ in range(self.hops):
                result = matmul(adj_t, result)
            result *= self.alpha
            result += (1-self.alpha)*y

        if self.mult_bin:
            output = torch.zeros((n, out_channels)).to(result)
            for task in range(label.shape[1]):
                output[:, task] = result[:, 2*task+1]
            result = output

        if binary:
            result = result.max(dim=1)[1].float()
        mask = f'{batch.split}_mask'
        if isinstance(batch, HeteroData):
            result = result[node_type_tensor == node_idx]
            label = label[node_type_tensor == node_idx]
            task = cfg.dataset.task_entity
            if hasattr(batch[task], 'batch_size'):
                batch_size = batch[task].batch_size
                return result[:batch_size], \
                    label[:batch_size]
            else:
                mask = f'{batch.split}_mask'
                return result[batch[task][mask]], \
                    label[batch[task][mask]]
        else:
            return result[batch[mask]], label[batch[mask]]