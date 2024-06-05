import torch
import torch.nn as nn
from torch_geometric.data import HeteroData

from H2GB.graphgym.config import cfg
from H2GB.graphgym.register import register_node_encoder, register_edge_encoder

class GeneralLayer(nn.Module):
    '''General wrapper for layers'''
    def __init__(self,
                 name,
                 dim_in,
                 dim_out,
                 has_act=True,
                 has_bn=False,
                 has_ln=True,
                 has_l2norm=False,
                 dropout=0.0,
                 act='relu',
                 **kwargs):
        super(GeneralLayer, self).__init__()
        self.has_l2norm = has_l2norm
        self.layer = layer_dict[name](dim_in,
                                      dim_out,
                                      bias=not has_bn,
                                      **kwargs)
        layer_wrapper = []
        if has_bn:
            layer_wrapper.append(
                nn.BatchNorm1d(dim_out, eps=cfg.bn.eps, momentum=cfg.bn.mom))
        if has_ln:
            layer_wrapper.append(nn.LayerNorm(dim_out))
        if dropout > 0:
            layer_wrapper.append(
                nn.Dropout(p=dropout, inplace=cfg.mem.inplace))
        if has_act:
            layer_wrapper.append(act_dict[act])
        self.post_layer = nn.Sequential(*layer_wrapper)

    def forward(self, batch):
        batch = self.layer(batch)
        if isinstance(batch, torch.Tensor):
            batch = self.post_layer(batch)
            if self.has_l2norm:
                batch = F.normalize(batch, p=2, dim=1)
        else:
            batch.x = self.post_layer(batch.x)
            if self.has_l2norm:
                batch.x = F.normalize(batch.x, p=2, dim=1)
        return batch


class GeneralMultiLayer(nn.Module):
    '''General wrapper for stack of layers'''
    def __init__(self,
                 name,
                 num_layers,
                 dim_in,
                 dim_out,
                 dim_inner=None,
                 final_act=True,
                 **kwargs):
        super(GeneralMultiLayer, self).__init__()
        dim_inner = dim_in if dim_inner is None else dim_inner
        for i in range(num_layers):
            d_in = dim_in if i == 0 else dim_inner
            d_out = dim_out if i == num_layers - 1 else dim_inner
            has_act = final_act if i == num_layers - 1 else True
            layer = GeneralLayer(name, d_in, d_out, has_act, **kwargs)
            self.add_module('Layer_{}'.format(i), layer)

    def forward(self, batch):
        for layer in self.children():
            batch = layer(batch)
        return batch

@register_node_encoder('Hetero_Raw')
class HeteroRawNodeEncoder(torch.nn.Module):
    """
    The raw feature node encoder

    Args:
        emb_dim (int): Output embedding dimension
    """
    def __init__(self, dim_emb, dataset):
        super().__init__()
        self.dim_in = cfg.share.dim_in
        self.metadata = dataset[0].metadata()
        data = dataset[0]
        # pecfg           = cfg.posenc_Hetero_Raw
        # self.layers     = pecfg.layers
        # self.dropout    = pecfg.dropout
        # self.act        = pecfg.act
        
        if not isinstance(dim_emb, dict):
            dim_emb = {node_type: dim_emb for node_type in self.dim_in}
        
        self.linear = nn.ModuleDict()
        for node_type in self.metadata[0]:
            if hasattr(data[node_type], 'x'):
                self.linear[node_type] = nn.Linear(
                    self.dim_in[node_type], dim_emb[node_type])
            # self.linear[node_type] = GeneralMultiLayer('linear', 
            #                  self.pre_layers, self.dim_in[node_type], dim_emb[node_type],
            #                  dim_inner=dim_emb[node_type], final_act=True,
            #                  has_bn=self.batch_norm, has_ln=self.layer_norm,
            #                  dropout=self.dropout, act=self.act)

        self.encoder = nn.ModuleDict(
            {
                node_type: nn.Embedding(data[node_type].num_nodes, dim_emb[node_type])
                for node_type in data.node_types
                if not hasattr(data[node_type], 'x')
            }
        )
        
    def forward(self, batch):
        if isinstance(batch, HeteroData):
            # Only changing the x itself can make sure the to_homogeneous() function works well later
            for node_type in batch.node_types:
                if hasattr(batch[node_type], 'x'):
                    batch[node_type].x = self.linear[node_type](batch[node_type].x)
                else:
                    if cfg.train.sampler == 'full_batch':
                        batch[node_type].x = self.encoder[node_type].weight.data #(torch.arange(batch[node_type].num_nodes))
                    else:
                        batch[node_type].x = self.encoder[node_type](batch[node_type].n_id)
        else:
            x = batch.x
            batch.x = list(self.linear.values())[0](x)

        return batch


@register_edge_encoder('Hetero_Raw')
class HeteroRawEdgeEncoder(torch.nn.Module):
    """
    The raw feature edge encoder

    Args:
        emb_dim (int): Output embedding dimension
    """
    def __init__(self, dim_emb, dataset):
        super().__init__()
        self.dim_in = cfg.share.dim_in
        self.metadata = dataset[0].metadata()
        
        if not isinstance(dim_emb, dict):
            dim_emb = {edge_type: dim_emb for edge_type in self.dim_in}
        
        self.linear = nn.ModuleDict()
        for edge_type in self.metadata[1]:
            edge_type = '__'.join(edge_type)
            self.linear[edge_type] = nn.Linear(
                self.dim_in[edge_type], dim_emb[edge_type])
    
    def forward(self, batch):
        # print(batch)
        # print(batch[('node', 'to', 'node')].e_id)
        # print(batch[('node', 'to', 'node')].input_id)
        # print(torch.isin(batch[('node', 'to', 'node')].e_id, batch[('node', 'to', 'node')].input_id).sum())
        # print(batch[('node', 'to', 'node')].edge_index)
        # print(batch[('node', 'to', 'node')].edge_label_index)

        if isinstance(batch, HeteroData):
            # Only changing the x itself can make sure the to_homogeneous() function works well later
            for edge_type, edge_attr in batch.collect("edge_attr").items():
                batch[edge_type].edge_attr = self.linear['__'.join(edge_type)](edge_attr) 
        else:
            edge_attr = batch.edge_attr
            batch.edge_attr = list(self.linear.values())[0](edge_attr)

        return batch