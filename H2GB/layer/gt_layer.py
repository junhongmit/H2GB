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
from torch_geometric.nn.inits import glorot, zeros, ones, reset
from torch_geometric.nn import (Linear, MLP, HeteroConv, GraphConv, SAGEConv, GINConv, GINEConv, \
                                GATConv, to_hetero, to_hetero_with_bases)
from torch_geometric.utils import to_dense_batch, to_undirected
from torch_geometric.nn import GraphSAGE
from H2GB.timer import runtime_stats_cuda, is_performance_stats_enabled, enable_runtime_stats, disable_runtime_stats

from H2GB.layer.gatedgcn_layer import GatedGCNLayer
from H2GB.layer.bigbird_layer import SingleBigBirdLayer


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
        self.kHop = cfg.gt.hops
        self.bias = Parameter(torch.Tensor(self.kHop))
        self.attn_bi = Parameter(torch.empty(self.num_heads, self.kHop))

        # Residual connection
        self.skip_local = torch.nn.ParameterDict()
        self.skip_global = torch.nn.ParameterDict()
        for node_type in metadata[0]:
            self.skip_local[node_type] = Parameter(torch.Tensor(1))
            self.skip_global[node_type] = Parameter(torch.Tensor(1))


        # Global Attention
        if global_model_type == 'None':
            self.attn = None
        elif global_model_type == 'TorchTransformer':
            self.attn = torch.nn.MultiheadAttention(
                        dim_h, num_heads, dropout=cfg.gt.attn_dropout, batch_first=True)
            # self.attn = torch.nn.ModuleDict()
            # for edge_type in metadata[1]:
            #     edge_type = '__'.join(edge_type)
            #     self.attn[edge_type] = torch.nn.MultiheadAttention(
            #             dim_h, num_heads, dropout=cfg.gt.attn_dropout, batch_first=True)
        elif global_model_type == 'NodeTransformer':
            self.k_lin = torch.nn.ModuleDict()
            self.q_lin = torch.nn.ModuleDict()
            self.v_lin = torch.nn.ModuleDict()
            self.o_lin = torch.nn.ModuleDict()
            for node_type in metadata[0]:
                # Different node type have a different projection matrix
                self.k_lin[node_type] = Linear(dim_in, dim_h)
                self.q_lin[node_type] = Linear(dim_in, dim_h)
                self.v_lin[node_type] = Linear(dim_in, dim_h)
                self.o_lin[node_type] = Linear(dim_h, dim_out)
        elif global_model_type == 'EdgeTransformer':
            self.k_lin = torch.nn.ModuleDict()
            self.q_lin = torch.nn.ModuleDict()
            self.v_lin = torch.nn.ModuleDict()
            self.o_lin = torch.nn.ModuleDict()
            for edge_type in metadata[1]:
                edge_type = '__'.join(edge_type)
                # Different edge type have a different projection matrix
                self.k_lin[edge_type] = Linear(dim_in, dim_h)
                self.q_lin[edge_type] = Linear(dim_in, dim_h)
                self.v_lin[edge_type] = Linear(dim_in, dim_h)
                self.o_lin[edge_type] = Linear(dim_h, dim_out)
        elif global_model_type == 'SparseNodeTransformer' or global_model_type == 'SparseNodeTransformer_Test':
            self.k_lin = torch.nn.ModuleDict()
            self.q_lin = torch.nn.ModuleDict()
            self.v_lin = torch.nn.ModuleDict()
            self.o_lin = torch.nn.ModuleDict()
            for node_type in metadata[0]:
                # Different node type have a different projection matrix
                self.k_lin[node_type] = Linear(dim_in, dim_h)
                self.q_lin[node_type] = Linear(dim_in, dim_h)
                self.v_lin[node_type] = Linear(dim_in, dim_h)
                self.o_lin[node_type] = Linear(dim_h, dim_out)
            H, D = self.num_heads, self.dim_h // self.num_heads
            if cfg.gt.edge_weight:
                self.edge_weights = nn.Parameter(torch.Tensor(len(metadata[1]), H, D, D))
                self.msg_weights = nn.Parameter(torch.Tensor(len(metadata[1]), H, D, D))
                nn.init.xavier_uniform_(self.edge_weights)
                nn.init.xavier_uniform_(self.msg_weights)
        elif global_model_type == 'SparseEdgeTransformer' or global_model_type == 'SparseEdgeTransformer_Test':
            self.k_lin = torch.nn.ModuleDict()
            self.q_lin = torch.nn.ModuleDict()
            self.v_lin = torch.nn.ModuleDict()
            self.e_lin = torch.nn.ModuleDict()
            self.g_lin = torch.nn.ModuleDict()
            self.oe_lin = torch.nn.ModuleDict()
            self.o_lin = torch.nn.ModuleDict()
            for edge_type in metadata[1]:
                edge_type = '__'.join(edge_type)
                # Different edge type have a different projection matrix
                self.k_lin[edge_type] = Linear(dim_in, dim_h)
                self.q_lin[edge_type] = Linear(dim_in, dim_h)
                self.v_lin[edge_type] = Linear(dim_in, dim_h)
                self.e_lin[edge_type] = Linear(dim_in, dim_h)
                self.g_lin[edge_type] = Linear(dim_h, dim_out)
                self.oe_lin[edge_type] = Linear(dim_h, dim_out)
                self.o_lin[edge_type] = Linear(dim_h, dim_out)
            
        # Local MPNNs
        norm = None
        if self.layer_norm or self.batch_norm:
            if self.layer_norm:
                norm = nn.LayerNorm(cfg.gt.dim_hidden)
            elif self.batch_norm:
                norm = nn.BatchNorm1d(cfg.gt.dim_hidden)
        self.local_gnn_with_edge_attr = True
        if local_gnn_type == 'None':
            self.local_model = None
        
        # MPNNs without edge attributes support.
        elif local_gnn_type == 'GCN':
            self.local_model = HeteroConv({
                edge_type: GraphConv(self.dim_in, self.dim_out, aggr=cfg.gnn.agg,) # add_self_loops=False) 
                for edge_type in self.metadata[1]
            })
        elif local_gnn_type == 'GraphSAGE':
            self.local_model = HeteroConv({
                edge_type: SAGEConv(self.dim_in, self.dim_out, aggr=cfg.gnn.agg, project=True) for edge_type in self.metadata[1]
            })
        elif local_gnn_type == 'GIN':
            mlp = MLP(
                [cfg.gt.dim_hidden, cfg.gt.dim_hidden, cfg.gt.dim_hidden],
                act=self.activation,
                norm=norm,
            )
            self.local_model = HeteroConv({
                    edge_type: GINConv(mlp) for edge_type in self.metadata[1]
            })
        elif local_gnn_type == 'GINE':
            mlp = nn.Sequential(
                    nn.Linear(cfg.gt.dim_hidden, cfg.gt.dim_hidden), 
                    nn.ReLU(), 
                    nn.Linear(cfg.gt.dim_hidden, cfg.gt.dim_hidden)
                )
            self.local_model = HeteroConv({
                    edge_type: GINEConv(mlp, edge_dim=cfg.gt.dim_hidden) for edge_type in self.metadata[1]
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
        self.project = torch.nn.ModuleDict()
        for node_type in metadata[0]:
            self.project[node_type] = Linear(dim_h * 2, dim_h)
            if self.layer_norm:
                self.norm1_local[node_type] = nn.LayerNorm(dim_h)
                self.norm1_global[node_type] = nn.LayerNorm(dim_h)
            if self.batch_norm:
                self.norm1_local[node_type] = nn.BatchNorm1d(dim_h)
                self.norm1_global[node_type] = nn.BatchNorm1d(dim_h)
        self.norm1_edge_local = torch.nn.ModuleDict()
        self.norm1_edge_global = torch.nn.ModuleDict()
        self.norm2_edge_ffn = torch.nn.ModuleDict()
        for edge_type in metadata[1]:
            edge_type = "__".join(edge_type)
            if self.layer_norm:
                self.norm1_edge_local[edge_type] = nn.LayerNorm(dim_h)
                self.norm1_edge_global[edge_type] = nn.LayerNorm(dim_h)
            if self.batch_norm:
                self.norm1_edge_local[edge_type] = nn.BatchNorm1d(dim_h)
                self.norm1_edge_global[edge_type] = nn.BatchNorm1d(dim_h)
        self.dropout_local = nn.Dropout(cfg.gnn.dropout)
        self.dropout_global = nn.Dropout(cfg.gt.dropout)
        self.dropout_attn = nn.Dropout(cfg.gt.attn_dropout)

        # if cfg.gt.residual == 'Concat':
        #     dim_h *= 2
        for node_type in metadata[0]:
            # Different node type have a different projection matrix
            if self.layer_norm:
                self.norm2_ffn[node_type] = nn.LayerNorm(dim_h)
            if self.batch_norm:
                self.norm2_ffn[node_type] = nn.BatchNorm1d(dim_h)
        
        # Feed Forward block.
        if cfg.gt.ffn == 'Single':
            self.ff_linear1 = nn.Linear(dim_h, dim_h * 2)
            self.ff_linear2 = nn.Linear(dim_h * 2, dim_h)
        elif cfg.gt.ffn == 'Type':
            self.ff_linear1_type = torch.nn.ModuleDict()
            self.ff_linear2_type = torch.nn.ModuleDict()
            for node_type in metadata[0]:
                self.ff_linear1_type[node_type] = nn.Linear(dim_h, dim_h * 2)
                self.ff_linear2_type[node_type] = nn.Linear(dim_h * 2, dim_h)
            self.ff_linear1_edge_type = torch.nn.ModuleDict()
            self.ff_linear2_edge_type = torch.nn.ModuleDict()
            for edge_type in metadata[1]:
                edge_type = "__".join(edge_type)
                self.ff_linear1_edge_type[edge_type] = nn.Linear(dim_h, dim_h * 2)
                self.ff_linear2_edge_type[edge_type] = nn.Linear(dim_h * 2, dim_h)
        
        self.ff_dropout1 = nn.Dropout(cfg.gt.dropout)
        self.ff_dropout2 = nn.Dropout(cfg.gt.dropout)
        self.reset_parameters()


    def reset_parameters(self):
        pass
        zeros(self.attn_bi)
        # ones(self.skip)


    def forward(self, batch):
        # x_dict, edge_index_dict = batch.x_dict, batch.edge_index_dict
        has_edge_attr = False
        if isinstance(batch, HeteroData):
            h_dict, edge_index_dict = batch.collect('x'), batch.collect('edge_index')
            if sum(batch.num_edge_features.values()):
                edge_attr_dict = batch.collect('edge_attr')
                has_edge_attr = True
        else:
            h_dict = {'node_type': batch.x}
            edge_index_dict = {('node_type', 'edge_type', 'node_type'): batch.edge_index}
            if sum(batch.num_edge_features.values()):
                edge_attr_dict = {('node_type', 'edge_type', 'node_type'): batch.edge_attr}
                has_edge_attr = True
        h_in_dict = h_dict#.copy()
        if has_edge_attr:
            edge_attr_in_dict = edge_attr_dict.copy()

        h_out_dict_list = {node_type: [] for node_type in h_dict}
        runtime_stats_cuda.start_region("gt-layer")

        # Local MPNN with edge attributes.
        if self.local_gnn_type != 'None':
            self.local_model: pygnn.conv.MessagePassing  # Typing hint.

            # Pre-normalization
            if self.layer_norm or self.batch_norm:
                h_dict = {
                    node_type: self.norm1_local[node_type](h_dict[node_type]) 
                    for node_type in batch.node_types
                }

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
                if isinstance(batch, HeteroData):
                    homo = batch.to_homogeneous()
                else:
                    homo = batch
                local_out = self.local_model(Batch(batch=homo,
                                            x=homo.x,
                                            edge_index=homo.edge_index,
                                            edge_attr=homo.edge_attr))
                h_local_dict = {
                    node_type: local_out.x for node_type in h_dict
                }
                batch.edge_attr = local_out.edge_attr
            else:
                if has_edge_attr:
                    h_local_dict = self.local_model(h_dict, edge_index_dict, edge_attr_dict)
                else:
                    h_local_dict = self.local_model(h_dict, edge_index_dict)

                h_local_dict = {
                    node_type: self.dropout_local(h_local_dict[node_type]) for node_type in h_local_dict
                }
                
                # Residual connection.
                if cfg.gt.residual == 'Fixed':
                    h_local_dict = {
                        node_type: h_local_dict[node_type] + h_in_dict[node_type]
                        for node_type in h_dict
                    }
                elif cfg.gt.residual == 'Learn':
                    alpha_dict = {
                        node_type: self.skip_local[node_type].sigmoid() for node_type in h_dict
                    }
                    h_local_dict = {
                        node_type: alpha_dict[node_type] * h_local_dict[node_type] + \
                            (1 - alpha_dict[node_type]) * h_in_dict[node_type]
                        for node_type in h_dict
                    }
                elif cfg.gt.residual != 'none':
                    raise ValueError(
                        f"Invalid local residual option {cfg.gt.residual}"
                    )

            # Post-normalization
            # if self.layer_norm or self.batch_norm:
            #     h_local_dict = {
            #         node_type: self.norm1_local[node_type](h_local_dict[node_type]) for node_type in h_local_dict
            #     }
            
            h_out_dict_list = {
                node_type: h_out_dict_list[node_type] + [h_local_dict[node_type]] for node_type in h_dict
            }

        if self.global_model_type != 'None':
            # Pre-normalization
            if self.layer_norm or self.batch_norm:
                h_dict = {
                    node_type: self.norm1_global[node_type](h_dict[node_type])
                    for node_type in batch.node_types
                }
                if has_edge_attr:
                    edge_attr_dict = {
                        edge_type: self.norm1_edge_global["__".join(edge_type)](edge_attr_dict[edge_type])
                        for edge_type in batch.edge_types
                    }
            

            h_attn_dict_list = {node_type: [] for node_type in h_dict}
            if self.global_model_type == 'TorchTransformer':
                D = self.dim_h

                homo_data = batch.to_homogeneous()
                h = homo_data.x
                edge_index = homo_data.edge_index
                node_type_tensor = homo_data.node_type
                edge_type_tensor = homo_data.edge_type
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
                    h_attn_dict_list[dst].append(h_attn)
                else:
                    L = h.shape[0]
                    S = h.shape[0]
                    q = h.view(1, -1, D)
                    k = h.view(1, -1, D)
                    v = h.view(1, -1, D)

                    if cfg.gt.attn_mask in ['Edge', 'kHop']:
                        attn_mask = torch.full((L, S), -1e9, dtype=torch.float32, device=edge_index.device)
                        if cfg.gt.attn_mask == 'kHop':
                            with torch.no_grad():
                                ones = torch.ones(edge_index.shape[1], device=edge_index.device)

                                edge_index_list = [edge_index]
                                edge_index_k = edge_index
                                for i in range(1, self.kHop):
                                    # print(edge_index_k.shape, int(edge_index_k.max()), L)
                                    edge_index_k, _ = torch_sparse.spspmm(edge_index_k, torch.ones(edge_index_k.shape[1], device=edge_index.device), 
                                                                        edge_index, ones, 
                                                                        L, L, L, True)
                                    edge_index_list.append(edge_index_k)
                            
                            for idx, edge_index in enumerate(reversed(edge_index_list)):
                                attn_mask[edge_index[1, :], edge_index[0, :]] = self.bias[idx]
                        else:
                            # Avoid the nan from attention mask
                            attn_mask[edge_index[1, :], edge_index[0, :]] = 1
                    
                    elif cfg.gt.attn_mask == 'Bias':
                        attn_mask = batch.attn_bi[self.index, :, :, :]
                    else:
                        attn_mask = None

                    h, A = self.attn(q, k, v,
                                attn_mask=attn_mask,
                                need_weights=True)
                                # average_attn_weights=False)

                    # attn_weights = A.detach().cpu()
                    h = h.view(1, -1, D)
                    for idx, node_type in enumerate(batch.node_types):
                        out_type = h[:, node_type_tensor == idx, :]
                        h_attn_dict_list[node_type].append(out_type.squeeze())

                # for edge_type, edge_index in edge_index_dict.items():
                #     src, _, dst = edge_type
                #     edge_type = '__'.join(edge_type)
                #     if hasattr(batch, 'batch'): # With batch dimension
                #         h_dense, key_padding_mask = to_dense_batch(h_dict[src], batch.batch)
                #         q = h_dense
                #         k = h_dense
                #         v = h_dense
                #         h_attn, A = self.attn[edge_type](q, k, v,
                #                     attn_mask=None,
                #                     key_padding_mask=~key_padding_mask,
                #                     need_weights=True)
                #                     # average_attn_weights=False)
                #         h_attn = h_attn[key_padding_mask]

                #         # attn_weights = A.detach().cpu()
                #         h_attn_dict_list[dst].append(h_attn)
                #     else:
                #         L = h_dict[dst].shape[0]
                #         S = h_dict[src].shape[0]
                #         q = h_dict[dst].view(1, -1, D)
                #         k = h_dict[src].view(1, -1, D)
                #         v = h_dict[src].view(1, -1, D)

                #         if cfg.gt.attn_mask in ['Edge', 'kHop']:
                #             if cfg.gt.attn_mask == 'kHop':
                #                 with torch.no_grad():
                #                     ones = torch.ones(edge_index.shape[1], device=edge_index.device)

                #                     edge_index_list = [edge_index]
                #                     edge_index_k = edge_index
                #                     for i in range(1, self.kHop):
                #                         # print(edge_index_k.shape, int(edge_index_k.max()), L)
                #                         edge_index_k, _ = torch_sparse.spspmm(edge_index_k, torch.ones(edge_index_k.shape[1], device=edge_index.device), 
                #                                                             edge_index, ones, 
                #                                                             L, L, L, True)
                #                         edge_index_list.append(edge_index_k)
                                
                #                 attn_mask = torch.full((L, S), -1e9, dtype=torch.float32, device=edge_index.device)
                #                 for idx, edge_index in enumerate(reversed(edge_index_list)):
                #                     attn_mask[edge_index[1, :], edge_index[0, :]] = self.bias[idx]
                #                 src_nodes, dst_nodes = edge_index_k
                #                 num_edges = edge_index_k.shape[1]
                #             else:
                #                 # Avoid the nan from attention mask
                #                 attn_mask = torch.full((L, S), -1e9, dtype=torch.float32, device=edge_index.device)
                #                 attn_mask[edge_index[1, :], edge_index[0, :]] = 1
                        
                #         elif cfg.gt.attn_mask == 'Bias':
                #             attn_mask = batch.attn_bi[self.index, :, :, :]
                #         else:
                #             attn_mask = None
                #         # print(attn_mask, torch.max(attn_mask))
                #         h, A = self.attn[edge_type](q, k, v,
                #                     attn_mask=attn_mask,
                #                     need_weights=True)
                #                     # average_attn_weights=False)

                #         # attn_weights = A.detach().cpu()
                #         h_attn_dict_list[dst].append(h.view(-1, D))

            elif self.global_model_type == 'NodeTransformer':
                # st = time.time()
                H, D = self.num_heads, self.dim_h // self.num_heads
                homo_data = batch.to_homogeneous()
                edge_index = homo_data.edge_index
                node_type_tensor = homo_data.node_type
                edge_type_tensor = homo_data.edge_type
                q = torch.empty((homo_data.x.shape[0], self.dim_h), device=homo_data.x.device)
                k = torch.empty((homo_data.x.shape[0], self.dim_h), device=homo_data.x.device)
                v = torch.empty((homo_data.x.shape[0], self.dim_h), device=homo_data.x.device)
                for idx, node_type in enumerate(batch.num_nodes_dict.keys()):
                    mask = node_type_tensor==idx
                    q[mask] = self.q_lin[node_type](h_dict[node_type])
                    k[mask] = self.k_lin[node_type](h_dict[node_type])
                    v[mask] = self.v_lin[node_type](h_dict[node_type])
                L = homo_data.x.shape[0]

                q = q.view(1, -1, H, D)
                k = k.view(1, -1, H, D)
                v = v.view(1, -1, H, D)

                # transpose to get dimensions bs * h * sl * d_model
                q = q.transpose(1,2)
                k = k.transpose(1,2)
                v = v.transpose(1,2)

                scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(D)


                if cfg.gt.attn_mask == 'Edge':
                    with torch.no_grad():
                        attn_mask = torch.zeros(L, L, dtype=torch.uint8, device=edge_index.device)
                        attn_mask[edge_index[1, :], edge_index[0, :]] = 1
                        output_mask = torch.zeros(L, 1, dtype=torch.uint8, device=q.device)
                        output_mask[torch.unique(edge_index[1, :]), :] = 1
                elif cfg.gt.attn_mask == 'kHop':
                    with torch.no_grad():
                        ones = torch.ones(edge_index.shape[1], device=edge_index.device)

                        edge_index_list = [edge_index]
                        edge_index_k = edge_index
                        for i in range(1, self.kHop):
                            # print(edge_index_k.shape, int(edge_index_k.max()), L)
                            edge_index_k, _ = torch_sparse.spspmm(edge_index_k, torch.ones(edge_index_k.shape[1], device=edge_index.device), 
                                                                edge_index, ones, 
                                                                L, L, L, True)
                            edge_index_list.append(edge_index_k)
                    
                        attn_mask = torch.full((L, L), -1e9, dtype=torch.float32, device=edge_index.device)
                        for idx, edge_index in enumerate(reversed(edge_index_list)):
                            attn_mask[edge_index[1, :], edge_index[0, :]] = self.bias[idx]
                        output_mask = torch.zeros(L, 1, dtype=torch.uint8, device=q.device)
                        output_mask[torch.unique(edge_index_k[1, :]), :] = 1
                # elif cfg.gt.attn_mask == 'Hierarchy':
                #     with torch.no_grad():
                #         ones = torch.ones(edge_index.shape[1], device=edge_index.device)

                #         edge_index_k = edge_index
                #         for i in range(self.index):
                #             # print(edge_index_k.shape, int(edge_index_k.max()), L)
                #             edge_index_k, _ = torch_sparse.spspmm(edge_index_k, torch.ones(edge_index_k.shape[1], device=edge_index.device), 
                #                                                 edge_index, ones, 
                #                                                 L, L, L, True)
                    
                #         attn_mask = torch.zeros(L, L, dtype=torch.uint8, device=q.x.device)
                #         attn_mask[edge_index_k[1, :], edge_index_k[0, :]] = 1
                # elif cfg.gt.attn_mask == 'Bias':
                #     attn_mask = batch.attn_bias[self.index, :, :, :]
                else:
                    attn_mask = None
                    output_mask = None
                
                if attn_mask is not None:
                    attn_mask = attn_mask.view(1, -1, L, L)
                    if attn_mask.dtype == torch.uint8:
                        masked_scores = scores.masked_fill(attn_mask == 0, -1e9)
                    elif attn_mask.is_floating_point():
                        masked_scores = scores + attn_mask
                else:
                    masked_scores = scores
                masked_scores = F.softmax(masked_scores, dim=-1)
                masked_scores = self.dropout_attn(masked_scores)
                
                out = torch.matmul(masked_scores, v)
                if output_mask is not None:
                    out = out.masked_fill(output_mask == 0, 0)
                out = out.transpose(1,2).contiguous().view(1, -1, H * D)

                for idx, node_type in enumerate(batch.num_nodes_dict.keys()):
                    out_type = self.o_lin[node_type](out[:, node_type_tensor == idx, :])
                    h_attn_dict_list[node_type].append(out_type.squeeze())

            elif self.global_model_type == 'EdgeTransformer':
                H, D = self.num_heads, self.dim_h // self.num_heads
                for edge_type, edge_index in edge_index_dict.items():
                    src, _, dst = edge_type
                    edge_type = '__'.join(edge_type)
                    L = h_dict[dst].shape[0]
                    S = h_dict[src].shape[0]
                    q = h_dict[dst]
                    k = h_dict[src]
                    v = h_dict[src]

                    if cfg.gt.attn_mask == 'Edge':
                        attn_mask = torch.zeros(L, S, dtype=torch.uint8, device=q.device)
                        attn_mask[edge_index[1, :], edge_index[0, :]] = 1
                        output_mask = torch.zeros(L, 1, dtype=torch.uint8, device=q.device)
                        output_mask[torch.unique(edge_index[1, :]), :] = 1
                    elif cfg.gt.attn_mask == 'kHop':
                        with torch.no_grad():
                            ones = torch.ones(edge_index.shape[1], device=edge_index.device)

                            edge_index_list = [edge_index]
                            edge_index_k = edge_index
                            for i in range(1, self.kHop):
                                # print(edge_index_k.shape, int(edge_index_k.max()), L)
                                edge_index_k, _ = torch_sparse.spspmm(edge_index_k, torch.ones(edge_index_k.shape[1], device=edge_index.device), 
                                                                    edge_index, ones, 
                                                                    L, L, L, True)
                                edge_index_list.append(edge_index_k)
                        
                        attn_mask = torch.full((L, S), -1e9, dtype=torch.float32, device=edge_index.device)
                        for idx, edge_index in enumerate(reversed(edge_index_list)):
                            attn_mask[edge_index[1, :], edge_index[0, :]] = self.bias[idx]
                    
                        output_mask = torch.zeros(L, 1, dtype=torch.uint8, device=q.device)
                        output_mask[torch.unique(edge_index_k[1, :]), :] = 1

                        # row, col = edge_index
                        # edge_index2, _ = torch_sparse.spspmm(edge_index.cpu(), torch.ones(col.shape, device=edge_index.cpu().device), 
                        #                                     edge_index.cpu(), torch.ones(col.shape, device=edge_index.cpu().device), 
                        #                                     L, L, L, True)
                        # edge_index2 = edge_index2.to(q.device)
                        # attn_mask = torch.ones(H, L, S, device=q.device) * (-1e9)
                        # attn_mask[:, edge_index2[1, :], edge_index2[0, :]] = self.attn_bias(torch.from_numpy(np.array([1], dtype=int)).to(edge_index.device)).view(-1, 1)
                        # attn_mask[:, edge_index[1, :], edge_index[0, :]] = self.attn_bias(torch.from_numpy(np.array([0], dtype=int)).to(edge_index.device)).view(-1, 1)
                    elif cfg.gt.attn_mask == 'Bias':
                        attn_mask = batch.attn_bi[self.index, :, :, :]
                    else:
                        attn_mask = None
                        output_mask = None

                    # q = self.q_lin[dst](q).view(1, -1, H, D)
                    # k = self.k_lin[src](k).view(1, -1, H, D)
                    # v = self.v_lin[src](v).view(1, -1, H, D)
                    q = self.q_lin[edge_type](q).view(1, -1, H, D)
                    k = self.k_lin[edge_type](k).view(1, -1, H, D)
                    v = self.v_lin[edge_type](v).view(1, -1, H, D)

                    # transpose to get dimensions bs * h * sl * d_model
                    q = q.transpose(1,2)
                    k = k.transpose(1,2)
                    v = v.transpose(1,2)

                    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(D)
                    scores = torch.clamp(scores, min=-5, max=5)
                    if attn_mask is not None:
                        attn_mask = attn_mask.view(1, -1, L, S)
                        if attn_mask.dtype == torch.uint8:
                            scores = scores.masked_fill(attn_mask == 0, -1e9)
                        elif attn_mask.is_floating_point():
                            scores = scores + attn_mask
                    scores = F.softmax(scores, dim=-1)
                    scores = self.dropout_attn(scores)
                        
                    out = torch.matmul(scores, v)
                    if output_mask is not None:
                        out = out.masked_fill(output_mask == 0, 0)
                    out = out.transpose(1,2).contiguous().view(1, -1, H * D)
                    out = self.o_lin[edge_type](out)

                    h_attn_dict_list[dst].append(out.squeeze())

            elif self.global_model_type == 'SparseNodeTransformer':
                # Test if Signed attention is beneficial
                # st = time.time()
                H, D = self.num_heads, self.dim_h // self.num_heads
                homo_data = batch.to_homogeneous()
                edge_index = homo_data.edge_index
                node_type_tensor = homo_data.node_type
                edge_type_tensor = homo_data.edge_type
                q = torch.empty((homo_data.x.shape[0], self.dim_h), device=homo_data.x.device)
                k = torch.empty((homo_data.x.shape[0], self.dim_h), device=homo_data.x.device)
                v = torch.empty((homo_data.x.shape[0], self.dim_h), device=homo_data.x.device)
                for idx, node_type in enumerate(batch.num_nodes_dict.keys()):
                    mask = node_type_tensor==idx
                    q[mask] = self.q_lin[node_type](h_dict[node_type])
                    k[mask] = self.k_lin[node_type](h_dict[node_type])
                    v[mask] = self.v_lin[node_type](h_dict[node_type])
                src_nodes, dst_nodes = edge_index
                num_edges = edge_index.shape[1]
                L = homo_data.x.shape[0]
                S = homo_data.x.shape[0]

                q = q.view(-1, H, D)
                k = k.view(-1, H, D)
                v = v.view(-1, H, D)

                # transpose to get dimensions h * sl * d_model
                q = q.transpose(0,1)
                k = k.transpose(0,1)
                v = v.transpose(0,1)

                if cfg.gt.attn_mask in ['Edge', 'kHop', 'kHop_diffusion']:
                    if cfg.gt.attn_mask in ['kHop', 'kHop_diffusion']:
                        with torch.no_grad():
                            edge_index_list = [edge_index]

                            # m = batch.num_nodes_dict['paper']
                            # n = batch.num_nodes_dict['author']
                            # o = batch.num_nodes_dict['paper']
                            # edge_index_1 = batch[('paper', 'rev_writes', 'author')].edge_index
                            # edge_index_2 = batch[('author', 'writes', 'paper')].edge_index
                            # adj_1 = torch.sparse_coo_tensor(edge_index_1, torch.ones(edge_index_1.size(1)), (m, n), device=edge_index_1.device)
                            # adj_2 = torch.sparse_coo_tensor(edge_index_2, torch.ones(edge_index_2.size(1)), (n, o), device=edge_index_2.device)
                            # adj = torch.sparse.mm(adj_1, adj_2)
                            # edge_index_k = adj.indices()
                            # edge_index_list.append(edge_index_k)

                            # m = batch.num_nodes_dict['paper']
                            # n = batch.num_nodes_dict['author']
                            # o = batch.num_nodes_dict['paper']
                            # edge_index_1 = batch[('paper', 'rev_AP_write_first', 'author')].edge_index
                            # edge_index_2 = batch[('author', 'AP_write_first', 'paper')].edge_index
                            # adj_1 = torch.sparse_coo_tensor(edge_index_1, torch.ones(edge_index_1.size(1)), (m, n), device=edge_index_1.device)
                            # adj_2 = torch.sparse_coo_tensor(edge_index_2, torch.ones(edge_index_2.size(1)), (n, o), device=edge_index_2.device)
                            # adj = torch.sparse.mm(adj_1, adj_2)
                            # edge_index_k = adj.indices()
                            # edge_index_list.append(edge_index_k)

                            # m = batch.num_nodes_dict['paper']
                            # n = batch.num_nodes_dict['paper']
                            # o = batch.num_nodes_dict['paper']
                            # edge_index_1 = batch[('paper', 'PP_cite', 'paper')].edge_index
                            # edge_index_2 = batch[('paper', 'PP_cite', 'paper')].edge_index
                            # adj_1 = torch.sparse_coo_tensor(edge_index_1, torch.ones(edge_index_1.size(1)), (m, n), device=edge_index_1.device)
                            # adj_2 = torch.sparse_coo_tensor(edge_index_2, torch.ones(edge_index_2.size(1)), (n, o), device=edge_index_2.device)
                            # adj = torch.sparse.mm(adj_1, adj_2)
                            # edge_index_k = adj.indices()
                            # edge_index_list.append(edge_index_k)

                            # m = batch.num_nodes_dict['paper']
                            # n = batch.num_nodes_dict['author']
                            # o = batch.num_nodes_dict['paper']
                            # edge_index_1 = batch[('paper', 'rev_AP_write_other', 'author')].edge_index
                            # edge_index_2 = batch[('author', 'AP_write_other', 'paper')].edge_index
                            # adj_1 = torch.sparse_coo_tensor(edge_index_1, torch.ones(edge_index_1.size(1)), (m, n), device=edge_index_1.device)
                            # adj_2 = torch.sparse_coo_tensor(edge_index_2, torch.ones(edge_index_2.size(1)), (n, o), device=edge_index_2.device)
                            # adj = torch.sparse.mm(adj_1, adj_2)
                            # edge_index_k = adj.indices()
                            # edge_index_list.append(edge_index_k)

                            edge_index_k = torch.cat(edge_index_list, dim=1)

                            # ones = torch.ones(edge_index.shape[1], device=edge_index.device)
                            # edge_index_list = [edge_index]
                            # edge_index_k = edge_index
                            # for i in range(1, self.kHop):
                            #     # print(edge_index_k.shape, int(edge_index_k.max()), L)
                            #     edge_index_k, _ = torch_sparse.spspmm(edge_index_k, torch.ones(edge_index_k.shape[1], device=edge_index.device), 
                            #                                         edge_index, ones, 
                            #                                         L, L, L, True)
                            #     edge_index_list.append(edge_index_k)
                        
                        attn_mask = torch.full((L, L), -1e9, dtype=torch.float32, device=edge_index.device)
                        for idx, edge_index in enumerate(reversed(edge_index_list)):
                            attn_mask[edge_index[1, :], edge_index[0, :]] = self.bias[idx]
                        src_nodes, dst_nodes = edge_index_k
                        num_edges = edge_index_k.shape[1]
                    else:
                        src_nodes, dst_nodes = edge_index
                        num_edges = edge_index.shape[1]
                    # Compute query and key for each edge
                    edge_q = q[:, dst_nodes, :]  # Queries for destination nodes # num_heads * num_edges * d_k
                    edge_k = k[:, src_nodes, :]  # Keys for source nodes
                    edge_v = v[:, src_nodes, :]

                    if hasattr(self, 'edge_weights'):
                        edge_weight = self.edge_weights[edge_type_tensor]  # (num_edges, num_heads, d_k, d_k)

                        edge_weight = edge_weight.transpose(0, 1)  # Transpose for batch matrix multiplication: (num_heads, num_edges, d_k, d_k)
                        # edge_k = edge_k.transpose(0, 1)  # Transpose to (num_edges, num_heads, d_k)
                        edge_k = edge_k.unsqueeze(-1) # Add dimension for matrix multiplication (num_heads, num_edges, d_k, 1)

                        # print(edge_weight.shape, edge_k.shape)
                        edge_k = torch.matmul(edge_weight, edge_k)  # (num_heads, num_edges, d_k, 1)
                        edge_k = edge_k.squeeze(-1)  # Remove the extra dimension (num_heads, num_edges, d_k)
                    # edge_k = edge_k.transpose(0, 1)  # Transpose back (num_edges, num_heads, d_k)

                    # Apply weight matrix to keys
                    # edge_k = torch.einsum('ehij,hej->hei', edge_weight, edge_k)
                    # msg_weight = self.msg_weights[edge_type_tensor]
                    # edge_v = torch.einsum('ehij,hej->hei', msg_weight, edge_v)

                    # Compute attention scores
                    edge_scores = torch.sum(edge_q * edge_k, dim=-1) / math.sqrt(D) # num_heads * num_edges
                    edge_scores = torch.clamp(edge_scores, min=-5, max=5)
                    if cfg.gt.attn_mask in ['kHop', 'kHop_diffusion']:
                        edge_scores = edge_scores + attn_mask[dst_nodes, src_nodes]

                    expanded_dst_nodes = dst_nodes.repeat(H, 1)  # Repeat dst_nodes for each head
                    
                    # Step 2: Calculate max for each destination node per head using scatter_max
                    max_scores, _ = scatter_max(edge_scores, expanded_dst_nodes, dim=1, dim_size=L)
                    max_scores = max_scores.gather(1, expanded_dst_nodes)

                    # Step 3: Exponentiate scores and sum
                    exp_scores = torch.exp(edge_scores - max_scores)
                    sum_exp_scores = torch.zeros((H, L), device=edge_scores.device)
                    sum_exp_scores.scatter_add_(1, expanded_dst_nodes, exp_scores)
                    # sum_exp_scores.clamp_(min=1e-9)

                    # Step 4: Apply softmax
                    edge_scores = exp_scores / sum_exp_scores.gather(1, expanded_dst_nodes)
                    edge_scores = edge_scores.unsqueeze(-1)
                    edge_scores = self.dropout_attn(edge_scores)

                    out = torch.zeros((H, L, D), device=q.device)
                    out.scatter_add_(1, dst_nodes.unsqueeze(-1).expand((H, num_edges, D)), edge_scores * edge_v)

                else:
                    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(D)
                    scores = F.softmax(scores, dim=-1)
                    scores = self.dropout_attn(scores)
                    
                    out = torch.matmul(scores, v)

                out = out.transpose(0,1).contiguous().view(-1, H * D)

                for idx, node_type in enumerate(batch.num_nodes_dict.keys()):
                    out_type = self.o_lin[node_type](out[node_type_tensor == idx, :])
                    h_attn_dict_list[node_type].append(out_type.squeeze())

            elif self.global_model_type == 'SparseNodeTransformer_Test':
                # st = time.time()
                H, D = self.num_heads, self.dim_h // self.num_heads
                homo_data = batch.to_homogeneous()
                edge_index = homo_data.edge_index
                node_type_tensor = homo_data.node_type
                # edge_type_tensor = homo_data.edge_type
                q = torch.empty((homo_data.x.shape[0], self.dim_h), device=homo_data.x.device)
                k = torch.empty((homo_data.x.shape[0], self.dim_h), device=homo_data.x.device)
                v = torch.empty((homo_data.x.shape[0], self.dim_h), device=homo_data.x.device)
                for idx, node_type in enumerate(batch.num_nodes_dict.keys()):
                    mask = node_type_tensor==idx
                    q[mask] = self.q_lin[node_type](h_dict[node_type])
                    k[mask] = self.k_lin[node_type](h_dict[node_type])
                    v[mask] = self.v_lin[node_type](h_dict[node_type])
                src_nodes, dst_nodes = edge_index
                num_edges = edge_index.shape[1]
                L = homo_data.x.shape[0]
                S = homo_data.x.shape[0]

                q = q.view(-1, H, D)
                k = k.view(-1, H, D)
                v = v.view(-1, H, D)

                # transpose to get dimensions h * sl * d_model
                q = q.transpose(0,1)
                k = k.transpose(0,1)
                v = v.transpose(0,1)

                if cfg.gt.attn_mask in ['Edge', 'kHop']:
                    if cfg.gt.attn_mask == 'kHop':
                        with torch.no_grad():
                            m = batch.num_nodes_dict['paper']
                            n = batch.num_nodes_dict['author']
                            o = batch.num_nodes_dict['paper']
                            edge_index_1 = batch[('paper', 'rev_writes', 'author')].edge_index
                            edge_index_2 = batch[('author', 'writes', 'paper')].edge_index
                            adj_1 = torch.sparse_coo_tensor(edge_index_1, torch.ones(edge_index_1.size(1)), (m, n), device=edge_index_1.device)
                            adj_2 = torch.sparse_coo_tensor(edge_index_2, torch.ones(edge_index_2.size(1)), (n, o), device=edge_index_2.device)
                            adj = torch.sparse.mm(adj_1, adj_2)

                            edge_index_list = [edge_index]
                            edge_index_k = adj.indices()
                            edge_index_list.append(edge_index_k)
                            edge_index_k = torch.cat(edge_index_list, dim=1)

                            # ones = torch.ones(edge_index.shape[1], device=edge_index.device)
                            # edge_index_list = [edge_index]
                            # edge_index_k = edge_index
                            # for i in range(1, self.kHop):
                            #     # print(edge_index_k.shape, int(edge_index_k.max()), L)
                            #     edge_index_k, _ = torch_sparse.spspmm(edge_index_k, torch.ones(edge_index_k.shape[1], device=edge_index.device), 
                            #                                         edge_index, ones, 
                            #                                         L, L, L, True)
                            #     edge_index_list.append(edge_index_k)
                        
                        attn_mask = torch.full((L, L), -1e9, dtype=torch.float32, device=edge_index.device)
                        for idx, edge_index in enumerate(reversed(edge_index_list)):
                            attn_mask[edge_index[1, :], edge_index[0, :]] = self.bias[idx]
                        src_nodes, dst_nodes = edge_index_k
                        num_edges = edge_index_k.shape[1]
                    else:
                        src_nodes, dst_nodes = edge_index
                        num_edges = edge_index.shape[1]
                    # Compute query and key for each edge
                    edge_q = q[:, dst_nodes, :]  # Queries for destination nodes # h * edges * d_model
                    edge_k = k[:, src_nodes, :]  # Keys for source nodes
                    edge_v = v[:, src_nodes, :]

                    # Compute attention scores
                    edge_scores = torch.sum(edge_q * edge_k, dim=-1) / math.sqrt(D) # h * edges
                    edge_scores = torch.clamp(edge_scores, min=-5, max=5)

                    edge_scores_sign = torch.sgn(edge_scores)
                    edge_scores = torch.abs(edge_scores)
                    if cfg.gt.attn_mask == 'kHop':
                        edge_scores = edge_scores + attn_mask[dst_nodes, src_nodes]

                    expanded_dst_nodes = dst_nodes.repeat(H, 1)  # Repeat dst_nodes for each head
                    
                    # Step 2: Calculate max for each destination node per head using scatter_max
                    max_scores, _ = scatter_max(edge_scores, expanded_dst_nodes, dim=1, dim_size=L)
                    max_scores = max_scores.gather(1, expanded_dst_nodes)

                    # Step 3: Exponentiate scores and sum
                    exp_scores = torch.exp(edge_scores - max_scores)
                    sum_exp_scores = torch.zeros((H, L), device=edge_scores.device)
                    sum_exp_scores.scatter_add_(1, expanded_dst_nodes, exp_scores)
                    # sum_exp_scores.clamp_(min=1e-9)

                    # Step 4: Apply softmax
                    edge_scores = exp_scores / sum_exp_scores.gather(1, expanded_dst_nodes)
                    edge_scores = edge_scores * edge_scores_sign
                    edge_scores = edge_scores.unsqueeze(-1)
                    edge_scores = self.dropout_attn(edge_scores)

                    out = torch.zeros((H, L, D), device=q.device)
                    out.scatter_add_(1, dst_nodes.unsqueeze(-1).expand((H, num_edges, D)), edge_scores * edge_v)

                else:
                    scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(D)
                    scores = F.softmax(scores, dim=-1)
                    scores = self.dropout_attn(scores)
                    
                    out = torch.matmul(scores, v)

                out = out.transpose(0,1).contiguous().view(-1, H * D)

                for idx, node_type in enumerate(batch.num_nodes_dict.keys()):
                    out_type = self.o_lin[node_type](out[node_type_tensor == idx, :])
                    h_attn_dict_list[node_type].append(out_type.squeeze())

            elif self.global_model_type == 'SparseEdgeTransformer':
                if self.return_attention:
                    saved_scores = {}
                runtime_stats_cuda.start_region("attention")
                H, D = self.num_heads, self.dim_h // self.num_heads
                for edge_type_tuple, edge_index in edge_index_dict.items():
                    src, _, dst = edge_type_tuple
                    edge_type = '__'.join(edge_type_tuple)
                    L = h_dict[dst].shape[0]
                    S = h_dict[src].shape[0]
                    q = h_dict[dst]
                    k = h_dict[src]
                    v = h_dict[src]
                    if has_edge_attr:
                        # src_nodes, dst_nodes = edge_index
                        edge_attr = edge_attr_dict[edge_type_tuple]
                        edge_attr = self.e_lin[edge_type](edge_attr).view(-1, H, D)
                        # edge_attr = self.e_lin[edge_type](torch.cat((h_dict[src][src_nodes], h_dict[dst][dst_nodes], edge_attr), dim=-1)).view(-1, H, D)
                        edge_attr = edge_attr.transpose(0,1) # (h, sl, d_model)

                        edge_gate = edge_attr_dict[edge_type_tuple]
                        edge_gate = self.g_lin[edge_type](edge_gate).view(-1, H, D)
                        # edge_gate = self.g_lin[edge_type](torch.cat((h_dict[src][src_nodes], h_dict[dst][dst_nodes], edge_gate), dim=-1)).view(-1, H, D)
                        edge_gate = edge_gate.transpose(0,1) # (h, sl, d_model)

                    q = self.q_lin[edge_type](q).view(-1, H, D)
                    k = self.k_lin[edge_type](k).view(-1, H, D)
                    v = self.v_lin[edge_type](v).view(-1, H, D)

                    # transpose to get dimensions (h, sl, d_model)
                    q = q.transpose(0,1)
                    k = k.transpose(0,1)
                    v = v.transpose(0,1)

                    if cfg.gt.attn_mask in ['Edge', 'kHop']:
                        if cfg.gt.attn_mask == 'kHop':
                            with torch.no_grad():
                                ones = torch.ones(edge_index.shape[1], device=edge_index.device)

                                edge_index_list = [edge_index]
                                edge_index_k = edge_index
                                for i in range(1, self.kHop):
                                    # print(edge_index_k.shape, int(edge_index_k.max()), L)
                                    print(edge_index_k.max(), L)
                                    edge_index_k, _ = torch_sparse.spspmm(edge_index_k, torch.ones(edge_index_k.shape[1], device=edge_index.device), 
                                                                        edge_index, ones, 
                                                                        L, L, L, True)
                                    edge_index_list.append(edge_index_k)
                            
                            attn_mask = torch.full((L, L), -1e9, dtype=torch.float32, device=edge_index.device)
                            for idx, edge_index in enumerate(reversed(edge_index_list)):
                                attn_mask[edge_index[1, :], edge_index[0, :]] = self.bias[idx]
                            src_nodes, dst_nodes = edge_index_k
                            num_edges = edge_index_k.shape[1]
                        else:
                            src_nodes, dst_nodes = edge_index
                            num_edges = edge_index.shape[1]
                        # Compute query and key for each edge
                        edge_q = q[:, dst_nodes, :]  # Queries for destination nodes # (h, edges, d_model)
                        edge_k = k[:, src_nodes, :]  # Keys for source nodes            4, 1000,  16  4 * 16 = 64 
                        edge_v = v[:, src_nodes, :]

                        # Compute attention scores
                        edge_scores = edge_q * edge_k # (h, edges, d_model)
                        # print(edge_q.shape, edge_k.shape, edge_scores.shape, edge_attr.shape)
                        if has_edge_attr:
                            edge_scores = edge_scores + edge_attr
                            edge_v = edge_v * F.sigmoid(edge_gate)
                            edge_attr = edge_scores
                            # edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2
                        # edge_scores = edge_attr

                        edge_scores = torch.sum(edge_scores, dim=-1) / math.sqrt(D) # (h, edges)
                        edge_scores = torch.clamp(edge_scores, min=-5, max=5)
                        if cfg.gt.attn_mask == 'kHop':
                            edge_scores = edge_scores + attn_mask[dst_nodes, src_nodes]
                        
                        expanded_dst_nodes = dst_nodes.repeat(H, 1)  # Repeat dst_nodes for each head
                        
                        # Step 2: Calculate max for each destination node per head using scatter_max
                        max_scores, _ = scatter_max(edge_scores, expanded_dst_nodes, dim=1, dim_size=L)
                        max_scores = max_scores.gather(1, expanded_dst_nodes)

                        # Step 3: Exponentiate scores and sum
                        exp_scores = torch.exp(edge_scores - max_scores) # - max_scores
                        sum_exp_scores = torch.zeros((H, L), device=edge_scores.device)
                        sum_exp_scores.scatter_add_(1, expanded_dst_nodes, exp_scores)

                        # Step 4: Apply softmax
                        edge_scores = exp_scores / sum_exp_scores.gather(1, expanded_dst_nodes)
                        edge_scores = edge_scores.unsqueeze(-1)
                        edge_scores = self.dropout_attn(edge_scores)

                        out = torch.zeros((H, L, D), device=q.device)
                        out.scatter_add_(1, dst_nodes.unsqueeze(-1).expand((H, num_edges, D)), edge_scores * edge_v)

                        # if edge_scores.isnan().any():
                        #     print('Edge scores has a nan!')
                        #     print('Dot product:', edge_scores_copy)
                        #     print('Max scores:', max_scores)
                        #     print('Sum exp scores:', sum_exp_scores)
                        #     print('Edge scores:', edge_scores)
                    elif cfg.gt.attn_mask == 'kHop':
                        raise NotImplementedError(f"Two hop attention masking for sparse edge transformer is not implemented!")
                    elif cfg.gt.attn_mask == 'Bias':
                        raise NotImplementedError(f"Attention bias for sparse edge transformer is not implemented!")
                        # attn_mask = batch.attn_bias[self.index, :, :, :]
                    else:
                        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(D)
                        scores = F.softmax(scores, dim=-1)
                        scores = self.dropout_attn(scores)
                        
                        out = torch.matmul(scores, v)

                    out = out.transpose(0,1).contiguous().view(-1, H * D)
                    out = self.o_lin[edge_type](out)
                    h_attn_dict_list[dst].append(out.squeeze())
                    
                    if has_edge_attr:
                        edge_attr = edge_attr.transpose(0,1).contiguous().view(-1, H * D)
                        edge_attr = self.oe_lin[edge_type](edge_attr)
                        edge_attr_dict[edge_type_tuple] = edge_attr
                runtime_stats_cuda.end_region("attention")
            elif self.global_model_type == 'SparseEdgeTransformer_Test':
                # Test if Signed attention is beneficial
                if self.return_attention:
                    saved_scores = {}
                runtime_stats_cuda.start_region("attention")
                H, D = self.num_heads, self.dim_h // self.num_heads
                for edge_type_tuple, edge_index in edge_index_dict.items():
                    src, _, dst = edge_type_tuple
                    edge_type = '__'.join(edge_type_tuple)
                    L = h_dict[dst].shape[0]
                    S = h_dict[src].shape[0]
                    q = h_dict[dst]
                    k = h_dict[src]
                    v = h_dict[src]
                    if has_edge_attr:
                        edge_attr = edge_attr_dict[edge_type_tuple]
                        edge_attr = self.e_lin[edge_type](edge_attr).view(-1, H, D)
                        edge_attr = edge_attr.transpose(0,1) # (h, sl, d_model)

                        edge_gate = edge_attr_dict[edge_type_tuple]
                        edge_gate = self.g_lin[edge_type](edge_gate).view(-1, H, D)
                        edge_gate = edge_gate.transpose(0,1) # (h, sl, d_model)

                    q = self.q_lin[edge_type](q).view(-1, H, D)
                    k = self.k_lin[edge_type](k).view(-1, H, D)
                    v = self.v_lin[edge_type](v).view(-1, H, D)

                    # transpose to get dimensions (h, sl, d_model)
                    q = q.transpose(0,1)
                    k = k.transpose(0,1)
                    v = v.transpose(0,1)

                    if cfg.gt.attn_mask in ['Edge', 'kHop']:
                        if cfg.gt.attn_mask == 'kHop':
                            with torch.no_grad():
                                ones = torch.ones(edge_index.shape[1], device=edge_index.device)

                                edge_index_list = [edge_index]
                                edge_index_k = edge_index
                                for i in range(1, self.kHop):
                                    # print(edge_index_k.shape, int(edge_index_k.max()), L)
                                    edge_index_k, _ = torch_sparse.spspmm(edge_index_k, torch.ones(edge_index_k.shape[1], device=edge_index.device), 
                                                                        edge_index, ones, 
                                                                        L, L, L, True)
                                    edge_index_list.append(edge_index_k)
                            
                            attn_mask = torch.full((L, L), -1e9, dtype=torch.float32, device=edge_index.device)
                            for idx, edge_index in enumerate(reversed(edge_index_list)):
                                attn_mask[edge_index[1, :], edge_index[0, :]] = self.bias[idx]
                            src_nodes, dst_nodes = edge_index_k
                            num_edges = edge_index_k.shape[1]
                        else:
                            src_nodes, dst_nodes = edge_index
                            num_edges = edge_index.shape[1]
                        # Compute query and key for each edge
                        edge_q = q[:, dst_nodes, :]  # Queries for destination nodes # (h, edges, d_model)
                        edge_k = k[:, src_nodes, :]  # Keys for source nodes            4, 1000,  16  4 * 16 = 64 
                        edge_v = v[:, src_nodes, :]

                        # Compute attention scores
                        edge_scores = edge_q * edge_k # (h, edges, d_model)
                        # print(edge_q.shape, edge_k.shape, edge_scores.shape, edge_attr.shape)
                        if has_edge_attr:
                            edge_scores = edge_scores + edge_attr
                            edge_v = edge_v * F.sigmoid(edge_gate)
                            edge_attr = edge_scores
                        # edge_scores = edge_attr

                        edge_scores = torch.sum(edge_scores, dim=-1) / math.sqrt(D) # (h, edges)
                        edge_scores = torch.clamp(edge_scores, min=-5, max=5)

                        edge_scores_sign = torch.sgn(edge_scores)
                        edge_scores = torch.abs(edge_scores)
                        if cfg.gt.attn_mask == 'kHop':
                            edge_scores = edge_scores + attn_mask[dst_nodes, src_nodes]
                        
                        expanded_dst_nodes = dst_nodes.repeat(H, 1)  # Repeat dst_nodes for each head
                        
                        # Step 2: Calculate max for each destination node per head using scatter_max
                        max_scores, _ = scatter_max(edge_scores, expanded_dst_nodes, dim=1, dim_size=L)
                        max_scores = max_scores.gather(1, expanded_dst_nodes)

                        # Step 3: Exponentiate scores and sum
                        exp_scores = torch.exp(edge_scores - max_scores) # - max_scores
                        sum_exp_scores = torch.zeros((H, L), device=edge_scores.device)
                        sum_exp_scores.scatter_add_(1, expanded_dst_nodes, exp_scores)

                        # Step 4: Apply softmax
                        edge_scores = exp_scores / sum_exp_scores.gather(1, expanded_dst_nodes)
                        edge_scores = edge_scores * edge_scores_sign
                        edge_scores = edge_scores.unsqueeze(-1)
                        edge_scores = self.dropout_attn(edge_scores)

                        out = torch.zeros((H, L, D), device=q.device)
                        out.scatter_add_(1, dst_nodes.unsqueeze(-1).expand((H, num_edges, D)), edge_scores * edge_v)

                        # if edge_scores.isnan().any():
                        #     print('Edge scores has a nan!')
                        #     print('Dot product:', edge_scores_copy)
                        #     print('Max scores:', max_scores)
                        #     print('Sum exp scores:', sum_exp_scores)
                        #     print('Edge scores:', edge_scores)
                    elif cfg.gt.attn_mask == 'Bias':
                        raise NotImplementedError(f"Attention bias for sparse edge transformer is not implemented!")
                        # attn_mask = batch.attn_bias[self.index, :, :, :]
                    else:
                        scores = torch.matmul(q, k.transpose(-2, -1)) /  math.sqrt(D)
                        scores = F.softmax(scores, dim=-1)
                        scores = self.dropout_attn(scores)
                        
                        out = torch.matmul(scores, v)

                    out = out.transpose(0,1).contiguous().view(-1, H * D)
                    out = self.o_lin[edge_type](out)
                    h_attn_dict_list[dst].append(out.squeeze())
                    
                    if has_edge_attr:
                        edge_attr = edge_attr.transpose(0,1).contiguous().view(-1, H * D)
                        edge_attr = self.oe_lin[edge_type](edge_attr)
                        edge_attr_dict[edge_type_tuple] = edge_attr
                runtime_stats_cuda.end_region("attention")

            h_attn_dict = {}
            for node_type in h_attn_dict_list:
                # h_attn_dict[node_type] = torch.zeros_like(h_in_dict[node_type])
                h_attn_dict[node_type] = torch.sum(torch.stack(h_attn_dict_list[node_type], dim=0), dim=0)
                h_attn_dict[node_type] = self.dropout_global(h_attn_dict[node_type])

            if cfg.gt.residual == 'Fixed':
                h_attn_dict = {
                    node_type: h_attn_dict[node_type] + h_in_dict[node_type]
                    for node_type in batch.node_types
                }

                if has_edge_attr:
                    edge_attr_dict = {
                        edge_type: edge_attr_dict[edge_type] + edge_attr_in_dict[edge_type]
                        for edge_type in batch.edge_types
                    }
            elif cfg.gt.residual == 'Learn':
                alpha_dict = {
                    node_type: self.skip_global[node_type].sigmoid() for node_type in batch.node_types
                }
                h_attn_dict = {
                    node_type: alpha_dict[node_type] * h_attn_dict[node_type] + \
                        (1 - alpha_dict[node_type]) * h_in_dict[node_type]
                    for node_type in batch.node_types
                }
            elif cfg.gt.residual != 'none':
                raise ValueError(
                    f"Invalid attention residual option {cfg.gt.residual}"
                )
            
            # Post-normalization
            # if self.layer_norm or self.batch_norm:
            #     h_attn_dict = {
            #         node_type: self.norm1_global[node_type](h_attn_dict[node_type])
            #         for node_type in batch.node_types
            #     }
            #     if has_edge_attr:
            #         edge_attr_dict = {
            #             edge_type: self.norm1_edge_global["__".join(edge_type)](edge_attr_dict[edge_type])
            #             for edge_type in batch.edge_types
            #         }

            
            # Concat output
            h_out_dict_list = {
                node_type: h_out_dict_list[node_type] + [h_attn_dict[node_type]] for node_type in batch.node_types
            }

        # Combine local / global information
        h_dict = {
            node_type: sum(h_out_dict_list[node_type]) for node_type in batch.node_types
        }
        # h_dict = {
        #     node_type: self.project[node_type](torch.cat(h_out_dict_list[node_type], dim=-1)) for node_type in batch.node_types
        # }

        if cfg.gt.ffn != 'none':
            # Pre-normalization
            if self.layer_norm or self.batch_norm:
                h_dict = {
                    node_type: self.norm2_ffn[node_type](h_dict[node_type])
                    for node_type in batch.node_types
                }
            
            if cfg.gt.ffn == 'Type':
                h_dict = {
                    node_type: h_dict[node_type] + self._ff_block_type(h_dict[node_type], node_type)
                    for node_type in batch.node_types
                }
                if has_edge_attr:
                    edge_attr_dict = {
                        edge_type: edge_attr_dict[edge_type] + self._ff_block_edge_type(edge_attr_dict[edge_type], edge_type)
                        for edge_type in batch.edge_types
                    }
            elif cfg.gt.ffn == 'Single':
                h_dict = {
                    node_type: h_dict[node_type] + self._ff_block(h_dict[node_type])
                    for node_type in batch.node_types
                }
            else:
                raise ValueError(
                    f"Invalid GT FFN option {cfg.gt.ffn}"
                )
                
            # Post-normalization
            # if self.layer_norm or self.batch_norm:
            #     h_dict = {
            #         node_type: self.norm2_ffn[node_type](h_dict[node_type])
            #         for node_type in batch.node_types
            #     }
        
        if cfg.gt.residual == 'Concat':
            h_dict = {
                node_type: torch.cat((h_in_dict[node_type], h_dict[node_type]), dim=1)
                for node_type in batch.node_types
            }

        runtime_stats_cuda.end_region("gt-layer")

        if isinstance(batch, HeteroData):
            for node_type in batch.node_types:
                batch[node_type].x = h_dict[node_type]
            if has_edge_attr:
                for edge_type in batch.edge_types:
                    batch[edge_type].edge_attr = edge_attr_dict[edge_type]
        else:
            batch.x = h_dict['node_type']

        if self.return_attention:
            return batch, saved_scores
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
    
    def _ff_block_edge_type(self, x, edge_type):
        """Feed Forward block.
        """
        edge_type = "__".join(edge_type)
        x = self.ff_dropout1(self.activation(self.ff_linear1_edge_type[edge_type](x)))
        return self.ff_dropout2(self.ff_linear2_edge_type[edge_type](x))

    # def __repr__(self):
    #     return '{}({}, {})'.format(self.__class__.__name__, self.dim_h,
    #                                self.dim_h)

