import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

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

@register_network('MLPModel')
class MLPModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)
        self.input_drop = nn.Dropout(cfg.gnn.input_dropout)

        self.encoder = FeatureEncoder(dim_in, dataset)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            if self.is_hetero:
                self.pre_mp_dict = torch.nn.ModuleDict()
                for node_type in dim_in:
                    self.pre_mp_dict[node_type] = GNNPreMP(
                        dim_in[node_type], cfg.gnn.dim_inner)
            else:
                self.pre_mp = GNNPreMP(dim_in, cfg.gnn.dim_inner)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(cfg.gnn.dim_inner, dim_out, dataset)

    def forward(self, batch):
        batch = self.encoder(batch)
        if isinstance(batch, HeteroData):
            x_dict = {
                node_type: self.input_drop(batch[node_type].x) for node_type in batch.node_types
            }

            if cfg.gnn.layers_pre_mp > 0:
                x_dict = {
                    node_type: self.pre_mp_dict[node_type](x)
                    for node_type, x in x_dict.items()
                }
        
            # There are Dropout (gnn.dropout) and activation layer (gnn.act) in the pre_mp

            for node_type in batch.node_types:
                batch[node_type].x = x_dict[node_type]
        else:
            x = batch.x
            x = self.input_drop(x)
            if cfg.gnn.layers_pre_mp > 0:
                x = self.pre_mp(x)
            batch.x = x
        return self.post_mp(batch)
