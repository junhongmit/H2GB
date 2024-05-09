import torch
import torch.nn as nn
import torch.nn.functional as F

from H2GB.graphgym.models import head  # noqa, register module
from H2GB.graphgym import register as register
from H2GB.graphgym.config import cfg
from H2GB.graphgym.models.gnn import GNNPreMP
from H2GB.graphgym.register import register_network
from torch_geometric.nn import (Sequential, Linear, HeteroConv, SAGEConv, GATConv, HANConv, \
                                HGTConv, to_hetero, to_hetero_with_bases)
from H2GB.graphgym.models.layer import BatchNorm1dNode
from H2GB.layer.gt_layer import GTLayer


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in, data):
        super(FeatureEncoder, self).__init__()
        self.dim_in = dim_in
        if cfg.dataset.node_encoder:
            # Encode integer node features via nn.Embeddings
            NodeEncoder = register.node_encoder_dict[
                cfg.dataset.node_encoder_name]
            self.node_encoder = NodeEncoder(cfg.gnn.dim_inner, data)
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

@register_network('HeteroGNNModel')
class HeteroGNNModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.drop = nn.Dropout(cfg.gnn.dropout)
        self.input_drop = nn.Dropout(cfg.gnn.input_dropout)
        self.metadata = dataset[0].metadata()
        self.activation = register.act_dict[cfg.gnn.act]
        self.layer_norm = cfg.gnn.layer_norm
        self.batch_norm = cfg.gnn.batch_norm
        # task_entity = cfg.dataset.task_entity
        # self.embedding = torch.nn.Embedding(data[task_entity[0]].num_nodes, cfg.gnn.dim_inner)

        self.encoder = FeatureEncoder(dim_in, dataset)
        dim_in = self.encoder.dim_in

        if cfg.gnn.layers_pre_mp > 0:
            self.pre_mp_dict = torch.nn.ModuleDict()
            for node_type in dim_in:
                self.pre_mp_dict[node_type] = GNNPreMP(
                    dim_in[node_type], cfg.gnn.dim_inner,
                    has_bn=cfg.gnn.batch_norm, has_ln=cfg.gnn.layer_norm
                )

        # self.conv = Sequential('x, edge_index', [(SAGEConv(cfg.gnn.dim_inner, cfg.gnn.dim_inner), 'x, edge_index -> x')])
        # self.conv = to_hetero(self.conv, data.metadata(), aggr='sum')

        self.convs = nn.ModuleList()
        self.linears = nn.ModuleList()
        if self.layer_norm or self.batch_norm:
            self.norms = nn.ModuleList()
        for i in range(cfg.gnn.layers_mp):
            norm_dim = cfg.gnn.dim_inner
            # if cfg.gnn.layer_type == 'GCN':
            #     conv = HeteroConv({
            #         edge_type: GraphConv(cfg.gnn.dim_inner, cfg.gnn.dim_inner, agg=cfg.gnn.agg, add_self_loops=False) 
            #         for edge_type in self.metadata[1]
            #     })
            if cfg.gnn.layer_type == 'RGraphSAGE':
                conv = HeteroConv({
                        edge_type: SAGEConv(cfg.gnn.dim_inner, cfg.gnn.dim_inner, aggr=cfg.gnn.agg) for edge_type in self.metadata[1]
                    })
            elif cfg.gnn.layer_type == 'RGAT':
                if i == 0:
                    conv = HeteroConv({
                        edge_type: GATConv(cfg.gnn.dim_inner, cfg.gnn.dim_inner, heads=cfg.gnn.attn_heads, 
                                           dropout=cfg.gnn.attn_dropout, concat=True,
                                           add_self_loops=True if edge_type[0] == edge_type[2] else False)
                        for edge_type in self.metadata[1]
                    })
                    norm_dim = cfg.gnn.attn_heads * cfg.gnn.dim_inner
                elif i < cfg.gnn.layers_mp - 1:
                    conv = HeteroConv({
                        edge_type: GATConv(cfg.gnn.attn_heads * cfg.gnn.dim_inner, cfg.gnn.dim_inner, heads=cfg.gnn.attn_heads,
                                           dropout=cfg.gnn.attn_dropout, concat=True,
                                           add_self_loops=True if edge_type[0] == edge_type[2] else False)
                        for edge_type in self.metadata[1]
                    })
                    norm_dim = cfg.gnn.attn_heads * cfg.gnn.dim_inner
                else:
                    conv = HeteroConv({
                        edge_type: GATConv(cfg.gnn.attn_heads * cfg.gnn.dim_inner, cfg.gnn.dim_inner, heads=cfg.gnn.attn_heads,
                                           dropout=cfg.gnn.attn_dropout, concat=False,
                                           add_self_loops=True if edge_type[0] == edge_type[2] else False)
                        for edge_type in self.metadata[1]
                    })
                    norm_dim = cfg.gnn.dim_inner
            elif cfg.gnn.layer_type == 'HAN':
                conv = HANConv(in_channels=cfg.gnn.dim_inner, out_channels=cfg.gnn.dim_inner, metadata=self.metadata,
                           heads=cfg.gnn.attn_heads, dropout=cfg.gnn.dropout) # group=cfg.gnn.agg # Removed since PyG 2.4.0
            elif cfg.gnn.layer_type == 'HGT':
                conv = HGTConv(in_channels=cfg.gnn.dim_inner, out_channels=cfg.gnn.dim_inner, metadata=self.metadata,
                           heads=cfg.gnn.attn_heads) # group=cfg.gnn.agg # Removed since PyG 2.4.0
            self.convs.append(conv)
            if cfg.gnn.use_linear:
                if i < cfg.gnn.layers_mp - 1:
                    self.linears.append(nn.ModuleDict())
                    for node_type in self.metadata[0]:
                        self.linears[-1][node_type] = nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner, bias=False)
            if self.layer_norm or self.batch_norm:
                self.norms.append(nn.ModuleDict())
                for node_type in self.metadata[0]:
                    if self.layer_norm:
                        self.norms[-1][node_type] = nn.LayerNorm(norm_dim)
                    elif self.batch_norm:
                        self.norms[-1][node_type] = nn.BatchNorm1d(norm_dim)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(cfg.gnn.dim_inner, dim_out, dataset)

    # def forward(self, batch):
    #     x, edge_index = batch.x, batch.edge_index

    #     for i in range(len(self.convs)):
    #         x = self.convs[i](x, edge_index)
    #         x = F.relu(x)
    #         x = F.dropout(x, p=0.1, training=self.training)

    #     batch.x = x
    #     batch = self.post_mp(batch)
    #     return batch

    def forward(self, batch):
        batch = self.encoder(batch)

        x_dict, edge_index_dict = batch.collect('x'), batch.collect('edge_index')

        x_dict = {
            node_type: self.input_drop(x) for node_type, x in x_dict.items()
        }

        if cfg.gnn.layers_pre_mp > 0:
            x_dict = {
                node_type: self.pre_mp_dict[node_type](x) for node_type, x in x_dict.items()
            }

        # x_dict = {
        #     node_type: x + self.embedding(batch[node_type].node_id) for node_type, x in x_dict.items()
        # } 

        for i in range(cfg.gnn.layers_mp): #[:-1]
            x_dict = self.convs[i](x_dict, edge_index_dict)
            if i < cfg.gnn.layers_mp - 1:
                if cfg.gnn.use_linear:
                    x_dict = {
                        node_type: x + self.linears[i][node_type](x) for node_type, x in x_dict.items()
                    }
            if self.layer_norm or self.batch_norm:
                x_dict = {
                    node_type: self.norms[i][node_type](x) for node_type, x in x_dict.items()
                }
            # x_dict = {
            #     node_type: self.activation(x)
            #     for node_type, x in x_dict.items()
            # }
            x_dict = {
                node_type: self.drop(self.activation(x))
                for node_type, x in x_dict.items()
            }
        # x_dict = self.convs[-1](x_dict, edge_index_dict)

        for node_type in batch.num_nodes_dict:
            batch[node_type].x = x_dict[node_type]
        return self.post_mp(batch)
    




# import math
# from typing import Dict, List, Optional, Union

# import torch
# import torch.nn.functional as F
# from torch import Tensor
# from torch.nn import Parameter
# from torch_sparse import SparseTensor

# from torch_geometric.nn.conv import MessagePassing
# from torch_geometric.nn.dense import Linear
# from torch_geometric.nn.inits import glorot, ones, reset
# from torch_geometric.typing import EdgeType, Metadata, NodeType
# from torch_geometric.utils import softmax


# def group(xs: List[Tensor], aggr: Optional[str]) -> Optional[Tensor]:
#     if len(xs) == 0:
#         return None
#     elif aggr is None:
#         return torch.stack(xs, dim=1)
#     elif len(xs) == 1:
#         return xs[0]
#     else:
#         out = torch.stack(xs, dim=0)
#         out = getattr(torch, aggr)(out, dim=0)
#         out = out[0] if isinstance(out, tuple) else out
#         return out


# class HGTConv(MessagePassing):
#     r"""The Heterogeneous Graph Transformer (HGT) operator from the
#     `"Heterogeneous Graph Transformer" <https://arxiv.org/abs/2003.01332>`_
#     paper.

#     .. note::

#         For an example of using HGT, see `examples/hetero/hgt_dblp.py
#         <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
#         hetero/hgt_dblp.py>`_.

#     Args:
#         in_channels (int or Dict[str, int]): Size of each input sample of every
#             node type, or :obj:`-1` to derive the size from the first input(s)
#             to the forward method.
#         out_channels (int): Size of each output sample.
#         metadata (Tuple[List[str], List[Tuple[str, str, str]]]): The metadata
#             of the heterogeneous graph, *i.e.* its node and edge types given
#             by a list of strings and a list of string triplets, respectively.
#             See :meth:`torch_geometric.data.HeteroData.metadata` for more
#             information.
#         heads (int, optional): Number of multi-head-attentions.
#             (default: :obj:`1`)
#         group (string, optional): The aggregation scheme to use for grouping
#             node embeddings generated by different relations.
#             (:obj:`"sum"`, :obj:`"mean"`, :obj:`"min"`, :obj:`"max"`).
#             (default: :obj:`"sum"`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch_geometric.nn.conv.MessagePassing`.
#     """
#     def __init__(
#         self,
#         in_channels: Union[int, Dict[str, int]],
#         out_channels: int,
#         metadata: Metadata,
#         heads: int = 1,
#         group: str = "sum",
#         **kwargs,
#     ):
#         super().__init__(aggr='add', node_dim=0, **kwargs)

#         if not isinstance(in_channels, dict):
#             in_channels = {node_type: in_channels for node_type in metadata[0]}

#         self.in_channels = in_channels
#         self.out_channels = out_channels
#         self.heads = heads
#         self.group = group

#         self.attn = torch.nn.ModuleDict()
#         for edge_type in metadata[1]:
#             edge_type = '__'.join(edge_type)
#             self.attn[edge_type] = torch.nn.MultiheadAttention(
#                     out_channels, heads, batch_first=True)

#         self.k_lin = torch.nn.ModuleDict()
#         self.q_lin = torch.nn.ModuleDict()
#         self.v_lin = torch.nn.ModuleDict()
#         self.a_lin = torch.nn.ModuleDict()
#         self.skip = torch.nn.ParameterDict()
#         for node_type, in_channels in self.in_channels.items():
#             self.k_lin[node_type] = Linear(in_channels, out_channels)
#             self.q_lin[node_type] = Linear(in_channels, out_channels)
#             self.v_lin[node_type] = Linear(in_channels, out_channels)
#             self.a_lin[node_type] = Linear(out_channels, out_channels)
#             self.skip[node_type] = Parameter(torch.Tensor(1))

#         self.a_rel = torch.nn.ParameterDict()
#         self.m_rel = torch.nn.ParameterDict()
#         self.p_rel = torch.nn.ParameterDict()
#         dim = out_channels // heads
#         for edge_type in metadata[1]:
#             edge_type = '__'.join(edge_type)
#             self.a_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
#             self.m_rel[edge_type] = Parameter(torch.Tensor(heads, dim, dim))
#             self.p_rel[edge_type] = Parameter(torch.Tensor(heads))

#         self.reset_parameters()

#     def reset_parameters(self):
#         reset(self.k_lin)
#         reset(self.q_lin)
#         reset(self.v_lin)
#         reset(self.a_lin)
#         ones(self.skip)
#         ones(self.p_rel)
#         glorot(self.a_rel)
#         glorot(self.m_rel)

#     def forward(
#         self,
#         x_dict: Dict[NodeType, Tensor],
#         edge_index_dict: Union[Dict[EdgeType, Tensor],
#                                Dict[EdgeType, SparseTensor]]  # Support both.
#     ) -> Dict[NodeType, Optional[Tensor]]:
#         r"""
#         Args:
#             x_dict (Dict[str, Tensor]): A dictionary holding input node
#                 features  for each individual node type.
#             edge_index_dict (Dict[str, Union[Tensor, SparseTensor]]): A
#                 dictionary holding graph connectivity information for each
#                 individual edge type, either as a :obj:`torch.LongTensor` of
#                 shape :obj:`[2, num_edges]` or a
#                 :obj:`torch_sparse.SparseTensor`.+

#         :rtype: :obj:`Dict[str, Optional[Tensor]]` - The output node embeddings
#             for each node type.
#             In case a node type does not receive any message, its output will
#             be set to :obj:`None`.
#         """

#         H, D = self.heads, self.out_channels // self.heads

#         k_dict, q_dict, v_dict, out_dict = {}, {}, {}, {}

#         # Iterate over node-types:
#         for node_type, x in x_dict.items():
#             k_dict[node_type] = self.k_lin[node_type](x).view(-1, H, D)
#             q_dict[node_type] = self.q_lin[node_type](x).view(-1, H, D)
#             v_dict[node_type] = self.v_lin[node_type](x).view(-1, H, D)
#             out_dict[node_type] = []

#         # Iterate over edge-types:
#         for edge_type, edge_index in edge_index_dict.items():
#             src_type, _, dst_type = edge_type
#             edge_type = '__'.join(edge_type)

#             a_rel = self.a_rel[edge_type]
#             k = (k_dict[src_type].transpose(0, 1) ).transpose(1, 0) # @ a_rel

#             m_rel = self.m_rel[edge_type]
#             v = (v_dict[src_type].transpose(0, 1) ).transpose(1, 0) # @ m_rel

#             # propagate_type: (k: Tensor, q: Tensor, v: Tensor, rel: Tensor)
#             out = self.propagate(edge_index, k=k, q=q_dict[dst_type], v=v,
#                                  rel=self.p_rel[edge_type], size=None)
#             out_dict[dst_type].append(out)

#         # h_dict = x_dict
#         # for node_type, x in x_dict.items():
#         #     out_dict[node_type] = []

#         # # D = self.out_channels
#         # for edge_type, edge_index in edge_index_dict.items():
#         #         src, _, dst = edge_type
#         #         edge_type = '__'.join(edge_type)
#         #         L = h_dict[dst].shape[0]
#         #         S = h_dict[src].shape[0]
#         #         q = h_dict[dst]#.view(1, -1, D)
#         #         k = h_dict[src]#.view(1, -1, D)
#         #         v = h_dict[src]#.view(1, -1, D)

#         #         # Avoid the nan from attention mask
#         #         # attn_mask = torch.ones(L, S, dtype=torch.float32, device=q.device) * (-1e9)
#         #         attn_mask = torch.zeros(L, S, dtype=torch.int8, device=q.device)
#         #         attn_mask[edge_index[1, :], edge_index[0, :]] = 1
#         #         # attn_mask[edge_index[0, :], edge_index[1, :]] = 1 # reversed

#         #         q = self.q_lin[dst](q).view(1, -1, H, D)
#         #         k = self.k_lin[src](k).view(1, -1, H, D)
#         #         v = self.v_lin[src](v).view(1, -1, H, D)

#         #         # transpose to get dimensions bs * h * sl * d_model
#         #         q = q.transpose(1,2)
#         #         k = k.transpose(1,2)
#         #         v = v.transpose(1,2)

#         #         scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(D)
#         #         attn_mask = attn_mask.view(1, L, S).unsqueeze(1)
#         #         scores = scores.masked_fill(attn_mask == 0, -1e9)
#         #         scores = F.softmax(scores, dim=-1)
                    
#         #         # if dropout is not None:
#         #         #     scores = dropout(scores)
                    
#         #         h = torch.matmul(scores, v)
#         #         h = h.transpose(1,2).contiguous().view(1, -1, H * D)
#         #         h = self.a_lin[dst](h)

#         #         # h, A = self.attn[edge_type](q, k, v,
#         #         #             attn_mask=attn_mask,
#         #         #             need_weights=True)
#         #         #             # average_attn_weights=False)

#         #         # attn_weights = A.detach().cpu()
#         #         out_dict[dst].append(h.squeeze())

#         # Iterate over node-types:
#         for node_type, outs in out_dict.items():
#             out = group(outs, self.group)

#             if out is None:
#                 out_dict[node_type] = None
#                 continue

#             # out = self.a_lin[node_type](F.gelu(out))
#             if out.size(-1) == x_dict[node_type].size(-1):
#                 alpha = self.skip[node_type].sigmoid()
#                 out = alpha * out + (1 - alpha) * x_dict[node_type]
#             # out = out + x_dict[node_type]
#             out_dict[node_type] = out

#         return out_dict

#     def message(self, k_j: Tensor, q_i: Tensor, v_j: Tensor, rel: Tensor,
#                 index: Tensor, ptr: Optional[Tensor],
#                 size_i: Optional[int]) -> Tensor:

#         alpha = (q_i * k_j).sum(dim=-1) #* rel
#         # print(q_i.shape, k_j.shape, alpha.shape)
#         alpha = alpha / math.sqrt(q_i.size(-1))
#         alpha = softmax(alpha, index, ptr, size_i)
#         out = v_j * alpha.view(-1, self.heads, 1)
#         # print(v_j.shape, alpha.shape, out.shape)
#         return out.view(-1, self.out_channels)

#     def __repr__(self) -> str:
#         return (f'{self.__class__.__name__}(-1, {self.out_channels}, '
#                 f'heads={self.heads})')
