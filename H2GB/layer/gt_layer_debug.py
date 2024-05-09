import math, time
import torch
import torch_sparse
import numpy as np
from torch_scatter import scatter_max
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import H2GB.graphgym.register as register
from H2GB.graphgym.config import cfg
from torch_geometric.data import HeteroData
import torch_geometric.nn as pygnn
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.data import Batch
from torch_geometric.nn.inits import glorot, ones, reset
from torch_geometric.nn import (Linear, Sequential, HeteroConv, GraphConv, SAGEConv, GATConv, \
                                to_hetero, to_hetero_with_bases)
from torch_geometric.utils import to_dense_batch, to_undirected
from torch_geometric.nn import GraphSAGE
from H2GB.timer import runtime_stats_cuda, is_performance_stats_enabled, enable_runtime_stats, disable_runtime_stats

from H2GB.layer.gatedgcn_layer import GatedGCNLayer
from H2GB.layer.bigbird_layer import SingleBigBirdLayer

# # Single attention block for different node type
# class GTLayer(nn.Module):
#     r"""Graph Transformer layer

#     """
#     def __init__(self, in_channels, out_channels, dim_h, metadata, num_heads=1,
#                  dropout=0.0, attn_dropout=0.0, layer_norm=False, batch_norm=True, **kwargs):
#         super(GTLayer, self).__init__()

#         self.dim_h = dim_h
#         self.num_heads = num_heads
#         self.attn_dropout = attn_dropout
#         self.layer_norm = layer_norm
#         self.batch_norm = batch_norm
#         self.metadata = metadata

#         self.lin_dict = torch.nn.ModuleDict()
#         self.skip = torch.nn.ParameterDict()
#         for node_type in metadata[0]:
#             # Different node type have a different projection matrix
#             self.lin_dict[node_type] = Linear(in_channels, in_channels)
#             self.skip[node_type] = Parameter(torch.Tensor(1))

#         self.self_attn = torch.nn.MultiheadAttention(
#                 dim_h, num_heads, dropout=self.attn_dropout, batch_first=True)
        
#         # Feed Forward block.
#         self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
#         self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
#         self.act_fn_ff = nn.ReLU()
#         if self.layer_norm:
#             self.norm2 = pygnn.norm.LayerNorm(dim_h)
#             # self.norm2 = pygnn.norm.GraphNorm(dim_h)
#             # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
#         if self.batch_norm:
#             self.norm2 = nn.BatchNorm1d(dim_h)
#         self.ff_dropout1 = nn.Dropout(dropout)
#         self.ff_dropout2 = nn.Dropout(dropout)

#         if self.layer_norm:
#             self.norm1_attn = torch.nn.ModuleDict()
#             for node_type in metadata[0]:
#                 # Different node type have a different projection matrix
#                 self.norm1_attn[node_type] = pygnn.norm.LayerNorm(dim_h)
#             self.norm2 = pygnn.norm.LayerNorm(dim_h)
#         self.in_channels = in_channels
#         self.out_channels = out_channels


#     def reset_parameters(self):
#         pass

#     def forward(self, x_dict, edge_index_dict):
#         # x_dict, edge_index_dict = batch.x_dict, batch.edge_index_dict
#         h_dict = x_dict

#         h_list = []
#         pos, size = {}, {}
#         cur = 0
#         node_types = sorted(list(h_dict.keys()))
#         for node_type in node_types:
#             # Different node type have a different projection matrix
#             # h_list.append(self.lin_dict[node_type](h_dict[node_type]))
#             h_list.append(h_dict[node_type])
#             pos[node_type] = cur
#             size[node_type] = h_list[-1].shape[0]
#             cur += size[node_type]
            
#         h = torch.vstack(h_list)

#         L = h.shape[0]
#         attn_mask = torch.ones(L, L, dtype=torch.float32, device=h.device) * (-1e9)
#         for rel, edge_index in edge_index_dict.items():
#             src, _, dst = rel
#             attn_mask[pos[dst] + edge_index[1, :], pos[src] + edge_index[0, :]] = 1
#             # attn_mask[pos[src] + edge_index[0, :], pos[dst] + edge_index[1, :]] = False

#         h = h.reshape([1, -1, self.in_channels])
#         # Multi-head attention.
#         h, A = self.self_attn(h, h, h,
#                             # attn_mask=attn_mask,
#                             need_weights=True)
#                             # average_attn_weights=False)
#         attn_weights = A.detach().cpu()

#         # h = h + self._ff_block(h)
#         # if self.layer_norm:
#         #     h = self.norm2(h)

#         h = h.reshape([-1, self.in_channels])
#         for node_type in node_types:
#             # Different node type have a different projection matrix
#             out = h[pos[node_type] : pos[node_type] + size[node_type], :]
#             # mask = torch.any(out.isnan(),dim=1)
#             # with torch.no_grad():
#             #     out[mask, :] = 0
#             # out = out.nan_to_num(0)
#             h_dict[node_type] = h_dict[node_type] + out

#             # alpha = self.skip[node_type].sigmoid()
#             # h_dict[node_type] = alpha * h_dict[node_type] + (1 - alpha) * x_dict[node_type]
#             # print(out)
#             # print(torch.any(h_dict[node_type].isnan(),dim=1))
#             # h_dict[node_type][mask, :] = h_dict[node_type][mask, :] + out[mask, :]

#             if self.layer_norm:
#                 h_dict[node_type] = self.norm1_attn[node_type](h_dict[node_type])

#         # h_attn = self.dropout_attn(h_attn)
#         # h_attn = h_in1 + h_attn  # Residual connection.
#         # if self.layer_norm:
#         #     h_attn = self.norm1_attn(h_attn, batch.batch)
#         # if self.batch_norm:
#         #     h_attn = self.norm1_attn(h_attn)
#         # h_out_list.append(h_attn)

#         # batch.x_dict = h_dict
#         x_dict = h_dict
#         return x_dict
    
#     def _ff_block(self, x):
#         """Feed Forward block.
#         """
#         x = self.ff_dropout1(self.act_fn_ff(self.ff_linear1(x)))
#         return self.ff_dropout2(self.ff_linear2(x))

#     def __repr__(self):
#         return '{}({}, {})'.format(self.__class__.__name__, self.in_channels,
#                                    self.out_channels)

# Different attention block for different edge type
class GTLayer(nn.Module):
    r"""Graph Transformer layer

    """
    def __init__(self, dim_in, dim_h, dim_out, metadata, local_gnn_type, global_model_type, index, num_heads=1,
                 layer_norm=False, batch_norm=False, return_attention=False, **kwargs):
        super(GTLayer, self).__init__()

        self.dim_in = dim_in
        self.dim_h = dim_h
        self.dim_out = dim_out
        self.index = index
        self.num_heads = num_heads
        self.layer_norm = layer_norm
        self.batch_norm = batch_norm
        self.activation = register.act_dict[cfg.gt.act]
        self.metadata = metadata
        self.return_attention = return_attention
        self.local_gnn_type = local_gnn_type
        self.global_model_type = global_model_type
        self.kHop = 2
        self.bias = Parameter(torch.Tensor(self.kHop))

        # Global Attention
        if global_model_type == 'None':
            self.attn = None
        elif global_model_type == 'TorchTransformer':
            self.attn = torch.nn.ModuleDict()
            self.attn = torch.nn.MultiheadAttention(
                    dim_h, num_heads, dropout=cfg.gt.attn_dropout, batch_first=True)
        # elif global_model_type == 'NodeTransformer':
        #     self.k_lin = torch.nn.ModuleDict()
        #     self.q_lin = torch.nn.ModuleDict()
        #     self.v_lin = torch.nn.ModuleDict()
        #     self.o_lin = torch.nn.ModuleDict()
        #     self.skip = torch.nn.ParameterDict()
        #     for node_type in metadata[0]:
        #         # Different node type have a different projection matrix
        #         self.k_lin[node_type] = Linear(dim_in, dim_h)
        #         self.q_lin[node_type] = Linear(dim_in, dim_h)
        #         self.v_lin[node_type] = Linear(dim_in, dim_h)
        #         self.o_lin[node_type] = Linear(dim_h, dim_out)
        #         self.skip[node_type] = Parameter(torch.Tensor(1))
        # elif global_model_type == 'EdgeTransformer':
        #     self.k_lin = torch.nn.ModuleDict()
        #     self.q_lin = torch.nn.ModuleDict()
        #     self.v_lin = torch.nn.ModuleDict()
        #     self.o_lin = torch.nn.ModuleDict()
        #     self.skip = torch.nn.ParameterDict()
        #     for node_type in metadata[0]:
        #         self.skip[node_type] = Parameter(torch.Tensor(1))
        #     for edge_type in metadata[1]:
        #         edge_type = '__'.join(edge_type)
        #         # Different edge type have a different projection matrix
        #         self.k_lin[edge_type] = Linear(dim_in, dim_h)
        #         self.q_lin[edge_type] = Linear(dim_in, dim_h)
        #         self.v_lin[edge_type] = Linear(dim_in, dim_h)
        #         self.o_lin[edge_type] = Linear(dim_h, dim_out)
        # elif global_model_type == 'SparseNodeTransformer':
        #     self.k_lin = torch.nn.ModuleDict()
        #     self.q_lin = torch.nn.ModuleDict()
        #     self.v_lin = torch.nn.ModuleDict()
        #     self.o_lin = torch.nn.ModuleDict()
        #     self.skip = torch.nn.ParameterDict()
        #     for node_type in metadata[0]:
        #         # Different node type have a different projection matrix
        #         self.k_lin[node_type] = Linear(dim_in, dim_h)
        #         self.q_lin[node_type] = Linear(dim_in, dim_h)
        #         self.v_lin[node_type] = Linear(dim_in, dim_h)
        #         self.o_lin[node_type] = Linear(dim_h, dim_out)
        #         self.skip[node_type] = Parameter(torch.Tensor(1))
        # elif global_model_type == 'SparseEdgeTransformer':
        #     self.k_lin = torch.nn.ModuleDict()
        #     self.q_lin = torch.nn.ModuleDict()
        #     self.v_lin = torch.nn.ModuleDict()
        #     self.o_lin = torch.nn.ModuleDict()
        #     self.skip = torch.nn.ParameterDict()
        #     # for node_type in metadata[0]:
        #     #     # Different node type have a different projection matrix
        #     #     self.k_lin[node_type] = Linear(dim_h, dim_h)
        #     #     self.q_lin[node_type] = Linear(dim_h, dim_h)
        #     #     self.v_lin[node_type] = Linear(dim_h, dim_h)
        #     #     self.skip[node_type] = Parameter(torch.Tensor(1))
        #     for node_type in metadata[0]:
        #         self.skip[node_type] = Parameter(torch.Tensor(1))
        #     for edge_type in metadata[1]:
        #         edge_type = '__'.join(edge_type)
        #         # Different edge type have a different projection matrix
        #         self.k_lin[edge_type] = Linear(dim_in, dim_h)
        #         self.q_lin[edge_type] = Linear(dim_in, dim_h)
        #         self.v_lin[edge_type] = Linear(dim_in, dim_h)
        #         self.o_lin[edge_type] = Linear(dim_h, dim_out)
        # elif global_model_type == 'BigBirdEdgeTransformer':
        #     bigbird_cfg = cfg.gt.bigbird
        #     bigbird_cfg.dim_hidden = dim_h
        #     bigbird_cfg.n_heads = num_heads
        #     bigbird_cfg.dropout = cfg.gt.attn_dropout
        #     self.attn = torch.nn.ModuleDict()
        #     for edge_type in metadata[1]:
        #         edge_type = '__'.join(edge_type)
        #         self.attn[edge_type] = SingleBigBirdLayer(bigbird_cfg)
        self.attn_bias = nn.Embedding(2, self.num_heads)
            
        # Local MPNNs
        self.local_gnn_with_edge_attr = True
        if local_gnn_type == 'None':
            self.local_model = None
        
        # MPNNs without edge attributes support.
        elif local_gnn_type == 'GCN':
            self.local_model = HeteroConv({
                edge_type: GraphConv(self.dim_in, self.dim_out, agg=cfg.gnn.agg, add_self_loops=False) 
                for edge_type in self.metadata[1]
            })
        elif local_gnn_type == 'SAGE':
            self.local_model = HeteroConv({
                edge_type: SAGEConv(self.dim_in, self.dim_out, agg=cfg.gnn.agg) for edge_type in self.metadata[1]
            })
        elif local_gnn_type == 'GAT':
            self.local_model = HeteroConv({
                edge_type: GATConv(self.dim_in, self.dim_out, heads=cfg.gnn.attn_heads, 
                                    agg=cfg.gnn.agg) for edge_type in self.metadata[1]
            })
        elif local_gnn_type == 'GatedGCN':
            self.local_model = GatedGCNLayer(
                in_dim=self.dim_in, out_dim=self.dim_out,
                dropout=cfg.gnn.dropout, residual=True, act=cfg.gnn.act
            )
        
        # elif local_gnn_type == "SAGE":
        #     self.local_gnn_with_edge_attr = False
        #     # self.local_model = pygnn.GCNConv(dim_h, dim_h)
        #     self.local_model = Sequential('x, edge_index', [(pygnn.SAGEConv(dim_h, dim_h), 'x, edge_index -> x')])
        # elif local_gnn_type == 'GIN':
        #     self.local_gnn_with_edge_attr = False
        #     gin_nn = nn.Sequential(Linear(dim_h, dim_h),
        #                            self.activation(),
        #                            Linear(dim_h, dim_h))
        #     self.local_model = pygnn.GINConv(gin_nn)
        # # MPNNs supporting also edge attributes.
        # elif local_gnn_type == 'GENConv':
        #     self.local_model = pygnn.GENConv(dim_h, dim_h)
        # elif local_gnn_type == 'GINE':
        #     gin_nn = nn.Sequential(Linear(dim_h, dim_h),
        #                            self.activation(),
        #                            Linear(dim_h, dim_h))
        #     # if self.equivstable_pe:  # Use specialised GINE layer for EquivStableLapPE.
        #     #     self.local_model = GINEConvESLapPE(gin_nn)
        #     # else:
        #     self.local_model = pygnn.GINEConv(gin_nn)
        # elif local_gnn_type == 'GAT':
        #     self.local_model = pygnn.GATConv(in_channels=dim_h,
        #                                      out_channels=dim_h // num_heads,
        #                                      heads=num_heads,
        #                                      edge_dim=dim_h)
        # if self.local_model != None:
        #     self.local_model = to_hetero(self.local_model, metadata, aggr='sum')



            # self.norm2 = pygnn.norm.GraphNorm(dim_h)
            # self.norm2 = pygnn.norm.InstanceNorm(dim_h)
        self.norm1_local = torch.nn.ModuleDict()
        self.norm1_global = torch.nn.ModuleDict()
        self.norm2_ffn = torch.nn.ModuleDict()
        if self.layer_norm:
            self.norm1_local = nn.LayerNorm(dim_h)
            self.norm1_global = nn.LayerNorm(dim_h)
        if self.batch_norm:
            self.norm1_local = nn.BatchNorm1d(dim_h)
            self.norm1_global = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(cfg.gnn.dropout)
        self.dropout_global = nn.Dropout(cfg.gt.dropout)
        self.dropout_attn = nn.Dropout(cfg.gt.attn_dropout)

        # if cfg.gt.residual == 'Concat':
        #     dim_h *= 2
            # Different node type have a different projection matrix
        if self.layer_norm:
            self.norm2_ffn = nn.LayerNorm(dim_h)
        if self.batch_norm:
            self.norm2_ffn = nn.BatchNorm1d(dim_h)
        
        # Feed Forward block.
        if cfg.gt.ffn in ['Single', 'Type']:
            self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
            self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        elif cfg.gt.ffn == 'Type':
            self.ff_linear1_type = torch.nn.ModuleDict()
            self.ff_linear2_type = torch.nn.ModuleDict()
            for node_type in metadata[0]:
                self.ff_linear1_type[node_type] = nn.Linear(dim_h, dim_h * 2)
                self.ff_linear2_type[node_type] = nn.Linear(dim_h * 2, dim_h)
        
        self.ff_dropout1 = nn.Dropout(cfg.gt.dropout)
        self.ff_dropout2 = nn.Dropout(cfg.gt.dropout)
        self.reset_parameters()


    def reset_parameters(self):
        pass
        # ones(self.skip)


    def forward(self, batch):
        # x_dict, edge_index_dict = batch.x_dict, batch.edge_index_dict
        h = batch.x #x_dict['node_type']
        h_in = h

        h_out_list = []
        # Local MPNN with edge attributes.
        if self.local_gnn_type != 'None':
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.

            # if self.local_gnn_with_edge_attr:
            #     # if self.equivstable_pe:
            #     #     h_local = self.local_model(h,
            #     #                                 batch.edge_index,
            #     #                                 batch.edge_attr,
            #     #                                 batch.pe_EquivStableLapPE)
            #     # else:
            #     h_local = self.local_model(h,
            #                                 batch.edge_index,
            #                                 batch.edge_attr)
            # else:
            #     h_local = self.local_model(h, batch.edge_index)
            if self.local_gnn_type == 'GatedGCN':
                local_out = self.local_model(Batch(batch=batch,
                                            x=h,
                                            edge_index=batch.edge_index,
                                            edge_attr=batch.edge_attr))
                h_local = local_out.x
                batch.edge_attr = local_out.edge_attr
            # else:
            #     h_local_dict = self.local_model(h_dict, edge_index_dict)

            # h_local = self.dropout_local(h_local)
            # h_local = h_local + h_in
            
            # # Residual connection.
            # h_local_dict = { 
            #     node_type: h_local_dict[node_type] + h_in_dict[node_type] for node_type in h_local_dict.keys()
            # }

            if self.layer_norm or self.batch_norm:
                h_local = self.norm1_local(h_local)
            h_out_list = h_out_list + [h_local]


        if self.global_model_type != 'None':
            # Pre-normalization
            # if self.layer_norm or self.batch_norm:
            #     h_dict = {
            #         node_type: self.norm1_global[node_type](h_dict[node_type]) for node_type in h_dict.keys()
            #     }
            

            if self.global_model_type == 'TorchTransformer':
                D = self.dim_h
                if hasattr(batch, 'batch'): # With batch dimension
                    h_dense, key_padding_mask = to_dense_batch(h, batch.batch)
                    q = h_dense
                    k = h_dense
                    v = h_dense
                    h_attn, A = self.attn(q, k, v,
                                attn_mask=None,
                                key_padding_mask=~key_padding_mask,
                                need_weights=True)
                                # average_attn_weights=False)
                    h_attn = h_attn[key_padding_mask]

                    # attn_weights = A.detach().cpu()
                # else:
                    # L = h_dict[dst].shape[0]
                    # S = h_dict[src].shape[0]
                    # q = h_dict[dst].view(1, -1, D)
                    # k = h_dict[src].view(1, -1, D)
                    # v = h_dict[src].view(1, -1, D)

                    # if cfg.gt.attn_mask == 'Edge':
                    #     # Avoid the nan from attention mask
                    #     attn_mask = torch.ones(L, S, dtype=torch.float32, device=q.device) * (-1e9)
                    #     attn_mask[edge_index[1, :], edge_index[0, :]] = 1
                    #     # attn_mask[edge_index[0, :], edge_index[1, :]] = 1 # reversed
                    # elif cfg.gt.attn_mask == 'Bias':
                    #     attn_mask = batch.attn_bias[self.index, :, :, :]
                    # else:
                    #     attn_mask = None
                    # # print(attn_mask, torch.max(attn_mask))
                    # h, A = self.attn[edge_type](q, k, v,
                    #             attn_mask=attn_mask,
                    #             need_weights=True)
                    #             # average_attn_weights=False)

                    # # attn_weights = A.detach().cpu()
                    # h_attn_dict_list[dst].append(h.view(-1, D))

            # elif self.global_model_type == 'NodeTransformer':
            #     # st = time.time()
            #     H, D = self.num_heads, self.dim_h // self.num_heads
            #     homo_data = batch.to_homogeneous()
            #     edge_index = homo_data.edge_index
            #     node_type_tensor = homo_data.node_type
            #     edge_type_tensor = homo_data.edge_type
            #     q = torch.empty((homo_data.x.shape[0], self.dim_h), device=homo_data.x.device)
            #     k = torch.empty((homo_data.x.shape[0], self.dim_h), device=homo_data.x.device)
            #     v = torch.empty((homo_data.x.shape[0], self.dim_h), device=homo_data.x.device)
            #     for idx, node_type in enumerate(batch.num_nodes_dict.keys()):
            #         mask = node_type_tensor==idx
            #         q[mask] = self.q_lin[node_type](h_dict[node_type])
            #         k[mask] = self.k_lin[node_type](h_dict[node_type])
            #         v[mask] = self.v_lin[node_type](h_dict[node_type])
            #     L = homo_data.x.shape[0]

            #     q = q.view(1, -1, H, D)
            #     k = k.view(1, -1, H, D)
            #     v = v.view(1, -1, H, D)

            #     # transpose to get dimensions bs * h * sl * d_model
            #     q = q.transpose(1,2)
            #     k = k.transpose(1,2)
            #     v = v.transpose(1,2)

            #     scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(D)


            #     if cfg.gt.attn_mask == 'Edge':
            #         with torch.no_grad():
            #             attn_mask = torch.zeros(L, L, dtype=torch.uint8, device=edge_index.device)
            #             attn_mask[edge_index[1, :], edge_index[0, :]] = 1
            #             output_mask = torch.zeros(L, 1, dtype=torch.uint8, device=q.device)
            #             output_mask[torch.unique(edge_index[1, :]), :] = 1
            #     elif cfg.gt.attn_mask == 'kHop':
            #         with torch.no_grad():
            #             ones = torch.ones(edge_index.shape[1], device=edge_index.device)

            #             edge_index_list = [edge_index]
            #             edge_index_k = edge_index
            #             for i in range(1, self.kHop):
            #                 # print(edge_index_k.shape, int(edge_index_k.max()), L)
            #                 edge_index_k, _ = torch_sparse.spspmm(edge_index_k, torch.ones(edge_index_k.shape[1], device=edge_index.device), 
            #                                                     edge_index, ones, 
            #                                                     L, L, L, True)
            #                 edge_index_list.append(edge_index_k)
                    
            #             attn_mask = torch.full((L, L), -1e9, dtype=torch.float32, device=edge_index.device)
            #             for idx, edge_index in enumerate(reversed(edge_index_list)):
            #                 attn_mask[edge_index[1, :], edge_index[0, :]] = self.bias[idx]
            #             output_mask = torch.zeros(L, 1, dtype=torch.uint8, device=q.device)
            #             output_mask[torch.unique(edge_index_k[1, :]), :] = 1
            #     # elif cfg.gt.attn_mask == 'Hierarchy':
            #     #     with torch.no_grad():
            #     #         ones = torch.ones(edge_index.shape[1], device=edge_index.device)

            #     #         edge_index_k = edge_index
            #     #         for i in range(self.index):
            #     #             # print(edge_index_k.shape, int(edge_index_k.max()), L)
            #     #             edge_index_k, _ = torch_sparse.spspmm(edge_index_k, torch.ones(edge_index_k.shape[1], device=edge_index.device), 
            #     #                                                 edge_index, ones, 
            #     #                                                 L, L, L, True)
                    
            #     #         attn_mask = torch.zeros(L, L, dtype=torch.uint8, device=q.x.device)
            #     #         attn_mask[edge_index_k[1, :], edge_index_k[0, :]] = 1
            #     # elif cfg.gt.attn_mask == 'Bias':
            #     #     attn_mask = batch.attn_bias[self.index, :, :, :]
            #     else:
            #         attn_mask = None
            #         output_mask = None
                
            #     if attn_mask is not None:
            #         attn_mask = attn_mask.view(1, -1, L, L)
            #         if attn_mask.dtype == torch.uint8:
            #             masked_scores = scores.masked_fill(attn_mask == 0, -1e9)
            #         elif attn_mask.is_floating_point():
            #             masked_scores = scores + attn_mask
            #     else:
            #         masked_scores = scores
            #     masked_scores = F.softmax(masked_scores, dim=-1)
            #     masked_scores = self.dropout_attn(masked_scores)
                
            #     # out = torch.matmul(masked_scores, v)
            #     # out = out.masked_fill(output_mask == 0, 0)
            #     out = torch.matmul(masked_scores, v)
            #     # out = torch.matmul(masked_scores, v)
            #     if output_mask is not None:
            #         out = out.masked_fill(output_mask == 0, 0)
            #     out = out.transpose(1,2).contiguous().view(1, -1, H * D)

            #     for idx, node_type in enumerate(batch.num_nodes_dict.keys()):
            #         out_type = self.o_lin[node_type](out[:, node_type_tensor == idx, :])
            #         h_attn_dict_list[node_type].append(out_type.squeeze())

            #     # edge_type_dict = {}
            #     # for edge_id, edge_type in enumerate(self.metadata[1]):
            #     #     if edge_type not in edge_index_dict:
            #     #         continue
            #     #     src, _, dst = edge_type
            #     #     edge_type_dict[(src, dst)] = edge_type_dict.get((src, dst), []) + [edge_id]

            #     # for (src, dst), edge_ids in edge_type_dict.items():
            #     #     src_id, dst_id = -1, -1
            #     #     for idx, node_type in enumerate(batch.num_nodes_dict.keys()):
            #     #         src_id = idx if node_type == src else src_id
            #     #         dst_id = idx if node_type == dst else dst_id
            #     #     with torch.no_grad():
            #     #         # edge_mask = torch.logical_and(node_type_tensor[edge_index[0, :]] == src_id, node_type_tensor[edge_index[1, :]] == dst_id)
            #     #         edge_mask = edge_type_tensor == edge_ids[0]
            #     #         for edge_id in edge_ids[1:]:
            #     #             edge_mask = torch.logical_or(edge_mask, edge_type_tensor == edge_id)
            #     #         masked_edge_index = edge_index[:, edge_mask]
            #     #         # print(src, src_id, dst, dst_id, masked_edge_index.shape)

            #     #     if cfg.gt.attn_mask == 'Edge':
            #     #         with torch.no_grad():
            #     #             src_mask = node_type_tensor == src_id
            #     #             dst_mask = node_type_tensor == dst_id
            #     #             attn_mask = torch.zeros(L, L, dtype=torch.uint8, device=edge_index.device)
            #     #             attn_mask[masked_edge_index[1, :], masked_edge_index[0, :]] = 1
            #     #             output_mask = torch.zeros(L, 1, dtype=torch.uint8, device=q.device)
            #     #             output_mask[torch.unique(masked_edge_index[1, :]), :] = 1
            #     #     # elif cfg.gt.attn_mask == 'Hierarchy':
            #     #     #     with torch.no_grad():
            #     #     #         ones = torch.ones(edge_index.shape[1], device=edge_index.device)

            #     #     #         edge_index_k = edge_index
            #     #     #         for i in range(self.index):
            #     #     #             # print(edge_index_k.shape, int(edge_index_k.max()), L)
            #     #     #             edge_index_k, _ = torch_sparse.spspmm(edge_index_k, torch.ones(edge_index_k.shape[1], device=edge_index.device), 
            #     #     #                                                 edge_index, ones, 
            #     #     #                                                 L, L, L, True)
                        
            #     #     #         attn_mask = torch.zeros(L, L, dtype=torch.uint8, device=q.x.device)
            #     #     #         attn_mask[edge_index_k[1, :], edge_index_k[0, :]] = 1
            #     #     # elif cfg.gt.attn_mask == 'Bias':
            #     #     #     attn_mask = batch.attn_bias[self.index, :, :, :]
            #     #     else:
            #     #         attn_mask = None
            #     #         output_mask = None
                    
            #     #     if attn_mask is not None:
            #     #         attn_mask = attn_mask.view(1, -1, L, L)
            #     #         attn_mask = attn_mask[:, :, dst_mask, :]
            #     #         output_mask = output_mask[dst_mask, :]
            #     #         if attn_mask.dtype == torch.uint8:
            #     #             masked_scores = scores[:, :, dst_mask, :].masked_fill(attn_mask == 0, -1e9)
            #     #         elif attn_mask.is_floating_point():
            #     #             masked_scores = scores + attn_mask
            #     #     masked_scores = F.softmax(masked_scores, dim=-1)
            #     #     # masked_scores = self.dropout_attn(masked_scores)
                    
            #     #     # out = torch.matmul(masked_scores, v)
            #     #     # out = out.masked_fill(output_mask == 0, 0)
            #     #     out = torch.matmul(masked_scores[:, :, :, src_mask], v[:, :, src_mask, :])
            #     #     # out = torch.matmul(masked_scores, v)
            #     #     if output_mask is not None:
            #     #         out = out.masked_fill(output_mask == 0, 0)
            #     #     out = out.transpose(1,2).contiguous().view(1, -1, H * D)
            #     #     out = self.o_lin[dst](out)
            #     #     # for idx, node_type in enumerate(h_dict.keys()):
            #     #     #     h_attn_dict_list[node_type].append(self.o_lin[node_type](out[node_type_tensor == idx]))
            #     #     h_attn_dict_list[dst].append(out.squeeze())




            #     # for edge_id, edge_type in enumerate(self.metadata[1]):
            #     #     if edge_type not in edge_index_dict:
            #     #         continue
            #     #     src, _, dst = edge_type
            #     #     edge_type = '__'.join(edge_type)
            #     #     src_id, dst_id = -1, -1
            #     #     for idx, node_type in enumerate(batch.num_nodes_dict.keys()):
            #     #         src_id = idx if node_type == src else src_id
            #     #         dst_id = idx if node_type == dst else dst_id
            #     #     with torch.no_grad():
            #     #         # edge_mask = torch.logical_and(node_type_tensor[edge_index[0, :]] == src_id, node_type_tensor[edge_index[1, :]] == dst_id)
            #     #         edge_mask = edge_type_tensor == edge_id
            #     #         masked_edge_index = edge_index[:, edge_mask]
            #     #         # print(src, src_id, dst, dst_id, masked_edge_index.shape)

            #     #     if cfg.gt.attn_mask == 'Edge':
            #     #         with torch.no_grad():
            #     #             src_mask = node_type_tensor == src_id
            #     #             dst_mask = node_type_tensor == dst_id
            #     #             attn_mask = torch.zeros(L, L, dtype=torch.uint8, device=edge_index.device)
            #     #             attn_mask[masked_edge_index[1, :], masked_edge_index[0, :]] = 1
            #     #             output_mask = torch.zeros(L, 1, dtype=torch.uint8, device=q.device)
            #     #             output_mask[torch.unique(masked_edge_index[1, :]), :] = 1
            #     #     # elif cfg.gt.attn_mask == 'Hierarchy':
            #     #     #     with torch.no_grad():
            #     #     #         ones = torch.ones(edge_index.shape[1], device=edge_index.device)

            #     #     #         edge_index_k = edge_index
            #     #     #         for i in range(self.index):
            #     #     #             # print(edge_index_k.shape, int(edge_index_k.max()), L)
            #     #     #             edge_index_k, _ = torch_sparse.spspmm(edge_index_k, torch.ones(edge_index_k.shape[1], device=edge_index.device), 
            #     #     #                                                 edge_index, ones, 
            #     #     #                                                 L, L, L, True)
                        
            #     #     #         attn_mask = torch.zeros(L, L, dtype=torch.uint8, device=q.x.device)
            #     #     #         attn_mask[edge_index_k[1, :], edge_index_k[0, :]] = 1
            #     #     # elif cfg.gt.attn_mask == 'Bias':
            #     #     #     attn_mask = batch.attn_bias[self.index, :, :, :]
            #     #     else:
            #     #         attn_mask = None
            #     #         output_mask = None
                    
            #     #     if attn_mask is not None:
            #     #         attn_mask = attn_mask.view(1, -1, L, L)
            #     #         attn_mask = attn_mask[:, :, dst_mask, :]
            #     #         output_mask = output_mask[dst_mask, :]
            #     #         if attn_mask.dtype == torch.uint8:
            #     #             masked_scores = scores[:, :, dst_mask, :].masked_fill(attn_mask == 0, -1e9)
            #     #         elif attn_mask.is_floating_point():
            #     #             masked_scores = scores + attn_mask
            #     #     masked_scores = F.softmax(masked_scores, dim=-1)
            #     #     # masked_scores = self.dropout_attn(masked_scores)
                    
            #     #     # out = torch.matmul(masked_scores, v)
            #     #     # out = out.masked_fill(output_mask == 0, 0)
            #     #     out = torch.matmul(masked_scores[:, :, :, src_mask], v[:, :, src_mask, :])
            #     #     # out = torch.matmul(masked_scores, v)
            #     #     if output_mask is not None:
            #     #         out = out.masked_fill(output_mask == 0, 0)
            #     #     out = out.transpose(1,2).contiguous().view(1, -1, H * D)
            #     #     out = self.o_lin[dst](out)
            #     #     # for idx, node_type in enumerate(h_dict.keys()):
            #     #     #     h_attn_dict_list[node_type].append(self.o_lin[node_type](out[node_type_tensor == idx]))
            #     #     h_attn_dict_list[dst].append(out.squeeze())
            #     # print('NodeGT:', time.time() - st)

            #     # for node_type in h_attn_dict_list:
            #     #     for idx in range(len(h_attn_dict_list[node_type])):
            #     #         if not torch.all(torch.eq(h_attn_dict_list_ori[node_type][idx], h_attn_dict_list[node_type][idx])):
            #     #             mask = ~torch.eq(h_attn_dict_list_ori[node_type][idx], h_attn_dict_list[node_type][idx])
            #     #             print(torch.mean(torch.abs(h_attn_dict_list_ori[node_type][idx][mask] - h_attn_dict_list[node_type][idx][mask])))
            #     #             print(node_type, idx)

            #     # if cfg.gt.attn_mask == 'Edge':
            #     #     with torch.no_grad():
            #     #         attn_mask = torch.zeros(L, L, dtype=torch.uint8, device=edge_index.device)
            #     #         attn_mask[edge_index[1, :], edge_index[0, :]] = 1
            #     # elif cfg.gt.attn_mask == 'Hierarchy':
            #     #     with torch.no_grad():
            #     #         ones = torch.ones(edge_index.shape[1], device=edge_index.device)

            #     #         edge_index_k = edge_index
            #     #         for i in range(self.index):
            #     #             # print(edge_index_k.shape, int(edge_index_k.max()), L)
            #     #             edge_index_k, _ = torch_sparse.spspmm(edge_index_k, torch.ones(edge_index_k.shape[1], device=edge_index.device), 
            #     #                                                 edge_index, ones, 
            #     #                                                 L, L, L, True)
                    
            #     #         attn_mask = torch.zeros(L, L, dtype=torch.uint8, device=q.x.device)
            #     #         attn_mask[edge_index_k[1, :], edge_index_k[0, :]] = 1
            #     # elif cfg.gt.attn_mask == 'Bias':
            #     #     attn_mask = batch.attn_bias[self.index, :, :, :]
            #     # else:
            #     #     attn_mask = None
                
            #     # if attn_mask is not None:
            #     #     attn_mask = attn_mask.view(1, -1, L, L)
            #     #     if attn_mask.dtype == torch.uint8:
            #     #         masked_scores = scores.masked_fill(attn_mask == 0, -1e9)
            #     #     elif attn_mask.is_floating_point():
            #     #         masked_scores = scores + attn_mask
            #     # masked_scores = F.softmax(masked_scores, dim=-1)
            #     # masked_scores = self.dropout_attn(masked_scores)
                    
            #     # out = torch.matmul(masked_scores, v)
            #     # out = out.transpose(1,2).contiguous().view(1, -1, H * D).squeeze()
            #     # for idx, node_type in enumerate(h_dict.keys()):
            #     #     h_attn_dict_list[node_type].append(self.o_lin[node_type](out[node_type_tensor == idx]))

            # elif self.global_model_type == 'EdgeTransformer':
            #     # print(batch)
            #     H, D = self.num_heads, self.dim_h // self.num_heads
            #     for edge_type, edge_index in edge_index_dict.items():
            #         src, _, dst = edge_type
            #         edge_type = '__'.join(edge_type)
            #         L = h_dict[dst].shape[0]
            #         S = h_dict[src].shape[0]
            #         q = h_dict[dst]
            #         k = h_dict[src]
            #         v = h_dict[src]

            #         if cfg.gt.attn_mask == 'Edge':
            #             attn_mask = torch.zeros(L, S, dtype=torch.uint8, device=q.device)
            #             attn_mask[edge_index[1, :], edge_index[0, :]] = 1
            #             output_mask = torch.zeros(L, 1, dtype=torch.uint8, device=q.device)
            #             output_mask[torch.unique(edge_index[1, :]), :] = 1
            #         elif cfg.gt.attn_mask == 'Two':
            #             # edge_index = to_undirected(edge_index)

            #             # CPU version
            #             row, col = edge_index
            #             edge_index2, _ = torch_sparse.spspmm(edge_index.cpu(), torch.ones(col.shape, device=edge_index.cpu().device), 
            #                                                 edge_index.cpu(), torch.ones(col.shape, device=edge_index.cpu().device), 
            #                                                 L, L, L, True)
            #             edge_index2 = edge_index2.to(q.device)
            #             attn_mask = torch.ones(H, L, S, device=q.device) * (-1e9)
            #             attn_mask[:, edge_index2[1, :], edge_index2[0, :]] = self.attn_bias(torch.from_numpy(np.array([1], dtype=int)).to(edge_index.device)).view(-1, 1)
            #             attn_mask[:, edge_index[1, :], edge_index[0, :]] = self.attn_bias(torch.from_numpy(np.array([0], dtype=int)).to(edge_index.device)).view(-1, 1)
            #         elif cfg.gt.attn_mask == 'Bias':
            #             attn_mask = batch.attn_bias[self.index, :, :, :]
            #         else:
            #             attn_mask = None
            #             output_mask = None

            #         # q = self.q_lin[dst](q).view(1, -1, H, D)
            #         # k = self.k_lin[src](k).view(1, -1, H, D)
            #         # v = self.v_lin[src](v).view(1, -1, H, D)
            #         q = self.q_lin[edge_type](q).view(1, -1, H, D)
            #         k = self.k_lin[edge_type](k).view(1, -1, H, D)
            #         v = self.v_lin[edge_type](v).view(1, -1, H, D)

            #         # transpose to get dimensions bs * h * sl * d_model
            #         q = q.transpose(1,2)
            #         k = k.transpose(1,2)
            #         v = v.transpose(1,2)

            #         scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(D)
            #         if attn_mask is not None:
            #             attn_mask = attn_mask.view(1, -1, L, S)
            #             if attn_mask.dtype == torch.uint8:
            #                 scores = scores.masked_fill(attn_mask == 0, -1e9)
            #             elif attn_mask.is_floating_point():
            #                 scores = scores + attn_mask
            #         scores = F.softmax(scores, dim=-1)
            #         scores = self.dropout_attn(scores)
                        
            #         out = torch.matmul(scores, v)
            #         if output_mask is not None:
            #             out = out.masked_fill(output_mask == 0, 0)
            #         out = out.transpose(1,2).contiguous().view(1, -1, H * D)
            #         out = self.o_lin[edge_type](out)

            #         h_attn_dict_list[dst].append(out.squeeze())

            # elif self.global_model_type == 'SparseNodeTransformer':
            #     # st = time.time()
            #     H, D = self.num_heads, self.dim_h // self.num_heads
            #     homo_data = batch.to_homogeneous()
            #     edge_index = homo_data.edge_index
            #     node_type_tensor = homo_data.node_type
            #     # edge_type_tensor = homo_data.edge_type
            #     q = torch.empty((homo_data.x.shape[0], self.dim_h), device=homo_data.x.device)
            #     k = torch.empty((homo_data.x.shape[0], self.dim_h), device=homo_data.x.device)
            #     v = torch.empty((homo_data.x.shape[0], self.dim_h), device=homo_data.x.device)
            #     for idx, node_type in enumerate(batch.num_nodes_dict.keys()):
            #         mask = node_type_tensor==idx
            #         q[mask] = self.q_lin[node_type](h_dict[node_type])
            #         k[mask] = self.k_lin[node_type](h_dict[node_type])
            #         v[mask] = self.v_lin[node_type](h_dict[node_type])
            #     src_nodes, dst_nodes = edge_index
            #     num_edges = edge_index.shape[1]
            #     L = homo_data.x.shape[0]
            #     S = homo_data.x.shape[0]

            #     q = q.view(-1, H, D)
            #     k = k.view(-1, H, D)
            #     v = v.view(-1, H, D)

            #     # transpose to get dimensions h * sl * d_model
            #     q = q.transpose(0,1)
            #     k = k.transpose(0,1)
            #     v = v.transpose(0,1)

            #     if cfg.gt.attn_mask == 'Edge':
            #         # Compute query and key for each edge
            #         edge_q = q[:, dst_nodes, :]  # Queries for destination nodes # h * edges * d_model
            #         edge_k = k[:, src_nodes, :]  # Keys for source nodes
            #         edge_v = v[:, src_nodes, :]

            #         # Compute attention scores
            #         edge_scores = torch.sum(edge_q * edge_k, dim=-1) / math.sqrt(D) # h * edges
            #         edge_scores_copy = edge_scores
                    
            #         expanded_dst_nodes = dst_nodes.repeat(H, 1)  # Repeat dst_nodes for each head
                    
            #         # Step 2: Calculate max for each destination node per head using scatter_max
            #         max_scores, _ = scatter_max(edge_scores, expanded_dst_nodes, dim=1, dim_size=L)
            #         max_scores = max_scores.gather(1, expanded_dst_nodes)

            #         # Step 3: Exponentiate scores and sum
            #         expanded_dst_nodes = dst_nodes.repeat(H, 1)
            #         exp_scores = torch.exp(edge_scores - max_scores)
            #         sum_exp_scores = torch.zeros((H, L), device=edge_scores.device)
            #         sum_exp_scores.scatter_add_(1, expanded_dst_nodes, exp_scores)
            #         # sum_exp_scores.clamp_(min=1e-9)

            #         # Step 4: Apply softmax
            #         edge_scores = exp_scores / sum_exp_scores.gather(1, expanded_dst_nodes)
            #         edge_scores = edge_scores.unsqueeze(-1)
            #         edge_scores = self.dropout_attn(edge_scores)

            #         out = torch.zeros((H, L, D), device=q.device)
            #         out.scatter_add_(1, dst_nodes.unsqueeze(-1).expand((H, num_edges, D)), edge_scores * edge_v)

            #     else:
            #         scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(D)
            #         scores = F.softmax(scores, dim=-1)
            #         scores = self.dropout_attn(scores)
                    
            #         out = torch.matmul(scores, v)

            #     out = out.transpose(0,1).contiguous().view(-1, H * D)

            #     for idx, node_type in enumerate(batch.num_nodes_dict.keys()):
            #         out_type = self.o_lin[node_type](out[:, node_type_tensor == idx, :])
            #         h_attn_dict_list[node_type].append(out_type.squeeze())

            # elif self.global_model_type == 'SparseEdgeTransformer':
            #     if self.return_attention:
            #         saved_scores = {}
            #     runtime_stats_cuda.start_region("attention")
            #     H, D = self.num_heads, self.dim_h // self.num_heads
            #     for edge_type, edge_index in edge_index_dict.items():
            #         src, _, dst = edge_type
            #         edge_type = '__'.join(edge_type)
            #         src_nodes, dst_nodes = edge_index
            #         num_edges = edge_index.shape[1]
            #         L = h_dict[dst].shape[0]
            #         S = h_dict[src].shape[0]
            #         q = h_dict[dst]
            #         k = h_dict[src]
            #         v = h_dict[src]

            #         q = self.q_lin[edge_type](q).view(-1, H, D)
            #         k = self.k_lin[edge_type](k).view(-1, H, D)
            #         v = self.v_lin[edge_type](v).view(-1, H, D)

            #         # transpose to get dimensions bs * h * sl * d_model
            #         q = q.transpose(0,1)
            #         k = k.transpose(0,1)
            #         v = v.transpose(0,1)

            #         if cfg.gt.attn_mask == 'Edge':
            #             # Compute query and key for each edge
            #             edge_q = q[:, dst_nodes, :]  # Queries for destination nodes # bs * h * edges * d_model
            #             edge_k = k[:, src_nodes, :]  # Keys for source nodes
            #             edge_v = v[:, src_nodes, :]

            #             # # Compute attention scores
            #             # edge_scores = torch.sum(edge_q * edge_k, dim=-1) / math.sqrt(D) # bs * h * edges
            #             # # edge_scores.clamp_(min=1e-9)
            #             # edge_scores_copy = edge_scores
            #             # if self.return_attention:
            #             #     if 'saved_scores' in batch:
            #             #         edge_scores = edge_scores + batch.saved_scores[edge_type]
            #             #     saved_scores[edge_type] = edge_scores

            #             # expanded_dst_nodes = dst_nodes.repeat(1, H, 1)
            #             # # Step 2 & 3: Calculate max for numerical stability
            #             # max_scores = torch.zeros((1, H, L), device=edge_scores.device)
            #             # max_scores.scatter_add_(2, expanded_dst_nodes, edge_scores)
            #             # max_scores, _ = max_scores.max(dim=2, keepdim=True)

            #             # # Step 4: Exponentiate scores and sum
            #             # exp_scores = torch.exp(edge_scores - max_scores) # - max_scores
            #             # sum_exp_scores = torch.zeros((1, H, L), device=edge_scores.device)
            #             # sum_exp_scores.scatter_add_(2, expanded_dst_nodes, exp_scores)
            #             # sum_exp_scores = sum_exp_scores.gather(2, expanded_dst_nodes)

            #             # # Step 5: Apply softmax
            #             # edge_scores = exp_scores / sum_exp_scores
            #             # edge_scores = edge_scores.unsqueeze(-1)
            #             # edge_scores = self.dropout_attn(edge_scores)

            #             # Compute attention scores
            #             edge_scores = torch.sum(edge_q * edge_k, dim=-1) / math.sqrt(D) # h * edges
                        
            #             expanded_dst_nodes = dst_nodes.repeat(H, 1)  # Repeat dst_nodes for each head
                        
            #             # Step 2: Calculate max for each destination node per head using scatter_max
            #             max_scores, _ = scatter_max(edge_scores, expanded_dst_nodes, dim=1, dim_size=L)
            #             max_scores = max_scores.gather(1, expanded_dst_nodes)

            #             # Step 3: Exponentiate scores and sum
            #             exp_scores = torch.exp(edge_scores - max_scores) # - max_scores
            #             sum_exp_scores = torch.zeros((H, L), device=edge_scores.device)
            #             sum_exp_scores.scatter_add_(1, expanded_dst_nodes, exp_scores)

            #             # Step 4: Apply softmax
            #             edge_scores = exp_scores / sum_exp_scores.gather(1, expanded_dst_nodes)
            #             edge_scores = edge_scores.unsqueeze(-1)
            #             edge_scores = self.dropout_attn(edge_scores)

            #             out = torch.zeros((H, L, D), device=q.device)
            #             out.scatter_add_(1, dst_nodes.unsqueeze(-1).expand((H, num_edges, D)), edge_scores * edge_v)

            #             # if edge_scores.isnan().any():
            #             #     print('Edge scores has a nan!')
            #             #     print('Dot product:', edge_scores_copy)
            #             #     print('Max scores:', max_scores)
            #             #     print('Sum exp scores:', sum_exp_scores)
            #             #     print('Edge scores:', edge_scores)

            #             out = torch.zeros((H, L, D), device=q.device)
            #             out.scatter_add_(1, dst_nodes.unsqueeze(-1).expand((H, num_edges, D)), edge_scores * edge_v)
            #         elif cfg.gt.attn_mask == 'Two':
            #             raise NotImplementedError(f"Two hop attention masking for sparse edge transformer is not implemented!")
            #             # edge_index = to_undirected(edge_index)

            #             # CPU version
            #             # row, col = edge_index
            #             # edge_index2, _ = torch_sparse.spspmm(edge_index.cpu(), torch.ones(col.shape, device=edge_index.cpu().device), 
            #             #                                     edge_index.cpu(), torch.ones(col.shape, device=edge_index.cpu().device), 
            #             #                                     L, L, L, True)
            #             # edge_index2 = edge_index2.to(q.device)
            #             # attn_mask = torch.ones(H, L, S, device=q.device) * (-1e9)
            #             # attn_mask[:, edge_index2[1, :], edge_index2[0, :]] = self.attn_bias(torch.from_numpy(np.array([1], dtype=int)).to(edge_index.device)).view(-1, 1)
            #             # attn_mask[:, edge_index[1, :], edge_index[0, :]] = self.attn_bias(torch.from_numpy(np.array([0], dtype=int)).to(edge_index.device)).view(-1, 1)
            #         elif cfg.gt.attn_mask == 'Bias':
            #             raise NotImplementedError(f"Attention bias for sparse edge transformer is not implemented!")
            #             # attn_mask = batch.attn_bias[self.index, :, :, :]
            #         else:
            #             scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(D)
            #             scores = F.softmax(scores, dim=-1)
            #             scores = self.dropout_attn(scores)
                        
            #             out = torch.matmul(scores, v)

            #         # indices = torch.empty((H, num_edges, 3), dtype=int, device=edge_index.device)
            #         # for i in range(H):
            #         #     indices[i, :, 0] = i
            #         #     indices[i, :, 1] = edge_index[1, :]
            #         #     indices[i, :, 2] = edge_index[0, :]
            #         # # indices[:, :, 1:] = edge_index.T.repeat((8, 1, 1))
            #         # indices = indices.reshape((H * num_edges, 3))
            #         # values = edge_scores.reshape(H * num_edges)
            #         # size = [H, L, S]
            #         # sparse_scores = torch.sparse_coo_tensor(indices.T, values, size)
            #         # sparse_scores = torch.sparse.softmax(sparse_scores, dim=2)
            #         # sparse_scores = self.dropout_attn(sparse_scores)
            #         # edge_scores = sparse_scores._values().reshape((1, H, num_edges, 1))
            #         # out = torch.zeros((1, H, S, D), device=q.device)
            #         # out.scatter_add_(2, dst_nodes.unsqueeze(-1).expand((1, H, num_edges, D)), edge_scores * edge_v)
            #         # # dense_scores = sparse_scores.to_dense().unsqueeze(0)
            #         # outputs = []
            #         # for h in range(H):
            #         #     # Extract the scores for the current head
            #         #     sparse_scores_h = sparse_scores[h]

            #         #     # Perform sparse-dense matrix multiplication for each head
            #         #     out_h = torch.sparse.mm(sparse_scores_h, v[0, h, :, :])

            #         #     # Store the result
            #         #     outputs.append(out_h)

            #         # # Concatenate the outputs from all heads
            #         # out = torch.stack(outputs, dim=0).unsqueeze(0)

            #         # if output_mask is not None:
            #         #     out = out.masked_fill(output_mask == 0, 0)
            #         out = out.transpose(0,1).contiguous().view(-1, H * D)
            #         out = self.o_lin[edge_type](out)

            #         h_attn_dict_list[dst].append(out.squeeze())
            #     runtime_stats_cuda.end_region("attention")

            h_attn = self.dropout_global(h_attn)
            # h_attn = {}
            # for node_type in h_dict.keys():
            #     h_attn_dict[node_type] = torch.sum(torch.stack(h_attn_dict_list[node_type], dim=0), dim=0)
            #     h_attn_dict[node_type] = self.dropout_global(h_attn_dict[node_type])
            
            h_attn = h_attn + h_in
            
            # Post-normalization
            if self.layer_norm or self.batch_norm:
                h_attn = self.norm1_global(h_attn)
            
            # Concat output
            h_out_list = h_out_list + [h_attn]

        h = sum(h_out_list)
        # h_dict = {
        #     node_type: sum(h_out_dict_list[node_type]) for node_type in h_dict
        # }

        # # Residual connection
        # if cfg.gt.residual == 'Fixed':
        #     h_dict = {
        #         node_type: h_dict[node_type] + h_in_dict[node_type]
        #         for node_type in h_dict
        #     }
        # elif cfg.gt.residual == 'Learn':
        #     alpha = self.skip[node_type].sigmoid()
        #     h_dict = {
        #         node_type: alpha * h_dict[node_type] + (1 - alpha) * h_in_dict[node_type]
        #         for node_type in h_dict
        #     }

        if cfg.gt.ffn != 'none':
            # Pre-normalization
            # if self.layer_norm or self.batch_norm:
            #     h_dict = {
            #         node_type: self.norm2_ffn[node_type](h_dict[node_type]) for node_type in h_dict
            #     }
            if cfg.gt.ffn == 'Type':
                h = h + self._ff_block(h)
            elif cfg.gt.ffn == 'Single':
                h = h + self._ff_block(h)
            else:
                raise ValueError(
                    f"Invalid GT FFN option {cfg.gt.ffn}"
                )
                
            # Post-normalization
            if self.layer_norm or self.batch_norm:
                h = self.norm2_ffn(h)
        
        # if cfg.gt.residual == 'Concat':
        #     h_dict = {
        #         node_type: torch.cat((h_in_dict[node_type], h_dict[node_type]), dim=1)
        #         for node_type in h_dict.keys()
        #     }

        runtime_stats_cuda.end_region("gt-layer")
        # x_dict = {'node_type': h}
        batch.x = h

        if self.return_attention:
            return x_dict, saved_scores
        return batch
    
    def _ff_block_type(self, x, node_type):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1_type[node_type](x)))
        return self.ff_dropout2(self.ff_linear2_type[node_type](x))
    
    def _ff_block(self, x):
        """Feed Forward block.
        """
        x = self.ff_dropout1(self.activation(self.ff_linear1(x)))
        return self.ff_dropout2(self.ff_linear2(x))

    # def __repr__(self):
    #     return '{}({}, {})'.format(self.__class__.__name__, self.dim_h,
    #                                self.dim_h)

