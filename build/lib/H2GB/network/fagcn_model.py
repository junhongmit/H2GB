import torch
import torch_sparse
from torch_sparse import SparseTensor, matmul
from torch_geometric.utils import degree
from torch_geometric.nn import MessagePassing
from torch_geometric.data import HeteroData
import scipy.sparse
import torch.nn as nn
import torch.nn.functional as F

from H2GB.graphgym import register as register
from H2GB.graphgym.config import cfg
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
            if 'Hetero_SDPE' in cfg.dataset.node_encoder_name:
                # These encoders don't affect the input dimension
                self.dim_in = dim_in
            else:
                self.dim_in = {node_type: cfg.gnn.dim_inner for node_type in dim_in}
        # if cfg.dataset.edge_encoder:
        #     # Hard-limit max edge dim for PNA.
        #     if 'PNA' in cfg.gt.layer_type:
        #         cfg.gnn.dim_edge = min(128, cfg.gnn.dim_inner)
        #     else:
        #         cfg.gnn.dim_edge = cfg.gnn.dim_inner
        #     # Encode integer edge features via nn.Embeddings
        #     EdgeEncoder = register.edge_encoder_dict[
        #         cfg.dataset.edge_encoder_name]
        #     self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge)
        #     if cfg.dataset.edge_encoder_bn:
        #         self.edge_encoder_bn = BatchNorm1dNode(cfg.gnn.dim_edge)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch

class FALayer(MessagePassing):
    def __init__(self, num_hidden, dropout):
        super(FALayer, self).__init__(aggr='add')
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(2 * num_hidden, 1)
        nn.init.xavier_normal_(self.gate.weight, gain=1.414)

    def forward(self, h, edge_index):
        row, col = edge_index
        norm_degree = degree(row, num_nodes=h.shape[0]).clamp(min=1)
        norm_degree = torch.pow(norm_degree, -0.5)
        h2 = torch.cat([h[row], h[col]], dim=1)
        g = torch.tanh(self.gate(h2)).squeeze()
        norm = g * norm_degree[row] * norm_degree[col]
        norm = self.dropout(norm)
        return self.propagate(edge_index, size=(h.size(0), h.size(0)), x=h, norm=norm)

    def message(self, x_j, norm):
        return norm.view(-1,1) * x_j

    def update(self, aggr_out):
        return aggr_out



@register_network('FAGCNModel')
class FAGCN(nn.Module):
    def __init__(self, dim_in, dim_out, dataset,
                 eps = 0.3):
        super(FAGCN, self).__init__()
        self.encoder = FeatureEncoder(dim_in, dataset)

        dim_h = cfg.gnn.dim_inner
        self.eps = eps
        self.layer_num = cfg.gnn.layers_mp
        self.dropout = cfg.gnn.dropout
        self.layers = nn.ModuleList()
        for i in range(self.layer_num):
            self.layers.append(FALayer(dim_h, self.dropout))

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_h, dim_out, dataset)

    def forward(self, batch):
        batch = self.encoder(batch)

        if isinstance(batch, HeteroData):
            homo = batch.to_homogeneous()
            x, edge_index = homo.x, homo.edge_index
            node_type_tensor = homo.node_type
        else:
            x, edge_index = batch.x, batch.edge_index
        h = x
       
        h = F.dropout(h, p=self.dropout, training=self.training)
        raw = h
        for i in range(self.layer_num):
            h = self.layers[i](h, edge_index)
            h = self.eps * raw + h
        
        if isinstance(batch, HeteroData):
            for idx, node_type in enumerate(batch.node_types):
                mask = node_type_tensor==idx
                batch[node_type].x = h[mask]
        else:
            batch.x = h
        return self.post_mp(batch)