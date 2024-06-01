import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from H2GB.graphgym import register as register
from H2GB.graphgym.config import cfg
from H2GB.graphgym.models.layer import layer_dict, act_dict
from H2GB.graphgym.register import register_node_encoder
from torch_geometric.nn import (MLP, HeteroConv, GraphConv, SAGEConv, \
                                GINConv, GATConv)

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


@register_node_encoder('Homo_GNN')
class HomoGNNEncoder(torch.nn.Module):
    """
    The GNN based structure extractor, works for homogeneous graph.

    Args:
        dim_in: the raw feature dimension
        dim_emb: the expected embedding dimension (only effective when `reshape_x`=)
    """
    def __init__(self, dim_emb, dataset, reshape_x=True):
        super().__init__()
        pecfg           = cfg.posenc_Homo_GNN
        self.dim_in = dim_emb - pecfg.dim_pe
        
        self.reshape_x  = reshape_x

        self.drop       = nn.Dropout(pecfg.dropout)
        self.input_drop = nn.Dropout(pecfg.input_dropout)
        self.activation = register.act_dict[pecfg.act]
        self.layer_norm = pecfg.layer_norm
        self.batch_norm = pecfg.batch_norm

        self.dim_hidden      = pecfg.dim_pe
        self.layer_type      = pecfg.model
        self.pre_layers      = pecfg.pre_layers
        self.layers          = pecfg.layers
        self.heads           = pecfg.n_heads
        self.agg             = pecfg.agg
        self.attn_dropout    = pecfg.attn_dropout
        self.dropout         = pecfg.dropout
        self.act             = pecfg.act
                    
        if self.pre_layers > 0:    
            self.pre_mp = GeneralMultiLayer('linear', 
                            self.pre_layers, self.dim_in, self.dim_hidden,
                            dim_inner=self.dim_hidden, final_act=True,
                            has_bn=self.batch_norm, has_ln=self.layer_norm,
                            dropout=self.dropout, act=self.act)

        self.convs = nn.ModuleList()
        self.linears = nn.ModuleList()
        if self.layer_norm or self.batch_norm:
            self.norms = nn.ModuleList()
        norm = None
        if self.layer_norm or self.batch_norm:
            if self.layer_norm:
                norm = nn.LayerNorm(self.dim_hidden)
            elif self.batch_norm:
                norm = nn.BatchNorm1d(self.dim_hidden)
        for i in range(self.layers):
            norm_dim = self.dim_hidden
            if self.layer_type == 'GCN':
                # Changed to GraphConv according to https://github.com/pyg-team/pytorch_geometric/discussions/3479
                conv = GraphConv(self.dim_hidden, self.dim_hidden, aggr=self.agg)
            elif self.layer_type == 'GraphSAGE':
                conv = SAGEConv(self.dim_hidden, self.dim_hidden, aggr=self.agg)
            elif self.layer_type == 'GIN':
                mlp = MLP(
                    [self.dim_hidden, self.dim_hidden, self.dim_hidden],
                    act=self.activation,
                    norm=norm,
                )
                conv = GINConv(mlp)
            elif self.layer_type == 'GAT':
                norm_dim = self.heads * self.dim_hidden
                if i == 0:
                    conv = GATConv(self.dim_hidden, self.dim_hidden, heads=self.heads,
                                           concat=True, add_self_loops=False)
                    norm_dim = cfg.gnn.attn_heads * cfg.gnn.dim_inner
                elif i < cfg.gnn.layers_mp - 1:
                    conv = GATConv(self.heads * self.dim_hidden, self.dim_hidden, heads=self.heads,
                                           concat=True, add_self_loops=False)
                    norm_dim = cfg.gnn.attn_heads * cfg.gnn.dim_inner
                else:
                    conv = GATConv(self.heads * self.dim_hidden, self.dim_hidden, heads=self.heads,
                                           concat=False, add_self_loops=True)
                    norm_dim = cfg.gnn.dim_inner
            self.convs.append(conv)
            if self.layer_norm or self.batch_norm:
                if self.layer_norm:
                    self.norms.append(nn.LayerNorm(norm_dim))
                elif self.batch_norm:
                    self.norms.append(nn.BatchNorm1d(norm_dim))

        if not isinstance(dim_emb, dict):
            dim_emb = {node_type: dim_emb for node_type in cfg.share.dim_in}
        
        if self.reshape_x:
            self.linear = nn.Linear(self.dim_in, dim_emb)

        
    def forward(self, batch):
        if isinstance(batch, HeteroData):
            homo = batch.to_homogeneous()
            x, edge_index = homo.x, homo.edge_index
            x = x.nan_to_num()
            node_type_tensor = homo.node_type
        else:
            x, edge_index = batch.x, batch.edge_index

        if self.pre_layers > 0:
            x = self.pre_mp(x)

        for i in range(self.layers): #[:-1]
            x = self.convs[i](x, edge_index)
            if self.layer_norm or self.batch_norm:
                x = self.norms[i](self.drop(self.activation(x)))

        for idx, node_type in enumerate(batch.node_types):
            mask = node_type_tensor==idx
            if self.reshape_x:
                h = self.linear(x[mask])
            else:
                h = x[mask]
            # Concatenate final PEs to input embedding
            if isinstance(batch, HeteroData):
                batch[node_type].x = torch.cat((batch[node_type].x, h), dim=-1)
            else:
                batch.x = torch.cat((batch.x, h), dim=-1)
        return batch
