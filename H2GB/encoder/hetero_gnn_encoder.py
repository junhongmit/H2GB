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


@register_node_encoder('Hetero_GNN')
class HeteroGNNEncoder(torch.nn.Module):
    """
    The GNN based structure extractor, works for both homogeneous / heterogeneous graph.
    Thus, termed hetero GNN encoder.

    Args:
        dim_in: the raw feature dimension
        dim_emb: the expected embedding dimension (only effective when `reshape_x`=)
    """
    def __init__(self, dim_emb, dataset, reshape_x=True):
        super().__init__()
        pecfg           = cfg.posenc_Hetero_GNN
        # self.dim_in = cfg.share.dim_in
        if isinstance(dataset[0], HeteroData):
            self.metadata   = dataset[0].metadata()
            self.dim_in = {node_type: dim_emb - pecfg.dim_pe for node_type in cfg.share.dim_in}
        else:
            self.metadata   = [['node_type'], [('node_type', 'edge_type', 'node_type')]]
            self.dim_in = {'node_type': dim_emb - pecfg.dim_pe}
        
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
            self.pre_mp_dict = torch.nn.ModuleDict()
            for node_type in self.dim_in:
                self.pre_mp_dict[node_type] = GeneralMultiLayer('linear', 
                             self.pre_layers, self.dim_in[node_type], self.dim_hidden,
                             dim_inner=self.dim_hidden, final_act=True,
                             has_bn=self.batch_norm, has_ln=self.layer_norm,
                             dropout=self.dropout, act=self.act)

        # self.conv = Sequential('x, edge_index', [(SAGEConv(cfg.gnn.dim_inner, cfg.gnn.dim_inner), 'x, edge_index -> x')])
        # self.conv = to_hetero(self.conv, data.metadata(), aggr='sum')

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
                conv = HeteroConv({
                    edge_type: GraphConv(self.dim_hidden, self.dim_hidden, aggr=self.agg) 
                    for edge_type in self.metadata[1]
                })
            elif self.layer_type == 'GraphSAGE':
                conv = HeteroConv({
                        edge_type: SAGEConv(self.dim_hidden, self.dim_hidden, aggr=self.agg) for edge_type in self.metadata[1]
                })
            elif self.layer_type == 'GIN':
                mlp = MLP(
                    [self.dim_hidden, self.dim_hidden, self.dim_hidden],
                    act=self.activation,
                    norm=norm,
                )
                conv = HeteroConv({
                        edge_type: GINConv(mlp) for edge_type in self.metadata[1]
                })
            elif self.layer_type == 'GAT':
                norm_dim = self.heads * self.dim_hidden
                if i == 0:
                    conv = HeteroConv({
                        edge_type: GATConv(self.dim_hidden, self.dim_hidden, heads=self.heads,
                                           concat=True, add_self_loops=False)
                        for edge_type in self.metadata[1]
                    })
                    norm_dim = cfg.gnn.attn_heads * cfg.gnn.dim_inner
                elif i < cfg.gnn.layers_mp - 1:
                    conv = HeteroConv({
                        edge_type: GATConv(self.heads * self.dim_hidden, self.dim_hidden, heads=self.heads,
                                           concat=True, add_self_loops=False)
                        for edge_type in self.metadata[1]
                    })
                    norm_dim = cfg.gnn.attn_heads * cfg.gnn.dim_inner
                else:
                    conv = HeteroConv({
                        edge_type: GATConv(self.heads * self.dim_hidden, self.dim_hidden, heads=self.heads,
                                           concat=False, add_self_loops=True)
                        for edge_type in self.metadata[1]
                    })
                    norm_dim = cfg.gnn.dim_inner
            self.convs.append(conv)
            # if cfg.gnn.use_linear:
            #     if i < cfg.gnn.layers_mp - 1:
            #         self.linears.append(nn.ModuleDict())
            #         for node_type in self.metadata[0]:
            #             self.linears[-1][node_type] = nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner, bias=False)
            if self.layer_norm or self.batch_norm:
                self.norms.append(nn.ModuleDict())
                for node_type in self.metadata[0]:
                    if self.layer_norm:
                        self.norms[-1][node_type] = nn.LayerNorm(norm_dim)
                    elif self.batch_norm:
                        self.norms[-1][node_type] = nn.BatchNorm1d(norm_dim)

        if not isinstance(dim_emb, dict):
            dim_emb = {node_type: dim_emb for node_type in self.dim_in}
        
        if self.reshape_x:
            self.linear = nn.ModuleDict()
            for node_type in self.metadata[0]:
                    self.linear[node_type] = nn.Linear(
                        self.dim_in[node_type], dim_emb[node_type])

        
    def forward(self, batch):
        if isinstance(batch, HeteroData):
            x_dict, edge_index_dict = batch.collect('x'), batch.collect('edge_index')
        else:
            x_dict = {self.metadata[0][0]: batch.x}
            edge_index_dict = {self.metadata[1][0]: batch.edge_index}
        x_dict = {
            node_type: self.input_drop(x)
            for node_type, x in x_dict.items()
        }

        if self.pre_layers > 0:
            x_dict = {
                node_type: self.pre_mp_dict[node_type](x) for node_type, x in x_dict.items()
            }

        # x_dict = {
        #     node_type: x + self.embedding(batch[node_type].node_id) for node_type, x in x_dict.items()
        # } 

        for i in range(self.layers): #[:-1]
            x_dict = self.convs[i](x_dict, edge_index_dict)
            # if i < cfg.gnn.layers_mp - 1:
            #     if cfg.gnn.use_linear:
            #         x_dict = {
            #             node_type: x + self.linears[i][node_type](x) for node_type, x in x_dict.items()
            #         }
            if self.layer_norm or self.batch_norm:
                x_dict = {
                    node_type: self.norms[i][node_type](x) for node_type, x in x_dict.items()
                }
            x_dict = {
                node_type: self.drop(self.activation(x))
                for node_type, x in x_dict.items()
            }

        for node_type, x in x_dict.items():
            if self.reshape_x:
                h = self.linear[node_type](x_dict[node_type])
            else:
                h = x_dict[node_type]
            # Concatenate final PEs to input embedding
            if isinstance(batch, HeteroData):
                batch[node_type].x = torch.cat((batch[node_type].x, h), dim=-1)
            else:
                batch.x = torch.cat((batch.x, h), dim=-1)
        return batch
