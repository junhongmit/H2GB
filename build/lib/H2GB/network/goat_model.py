import math
from turtle import xcor
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.parameter import Parameter
from torch_sparse import SparseTensor
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.dense.linear import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor
from torch_geometric.utils import softmax

from H2GB.graphgym.models import head  # noqa, register module
from H2GB.graphgym import register as register
from H2GB.graphgym.config import cfg
from H2GB.graphgym.models.gnn import GNNPreMP
from H2GB.graphgym.register import register_network
from H2GB.graphgym.models.layer import BatchNorm1dNode

from einops import rearrange, repeat, reduce
from einops.layers.torch import Rearrange

class VectorQuantizerEMA(nn.Module):
    def __init__(
            self, 
            num_embeddings, 
            embedding_dim, 
            decay=0.99
        ):
        super(VectorQuantizerEMA, self).__init__()

        self._embedding_dim = embedding_dim
        self._num_embeddings = num_embeddings

        self.register_buffer('_embedding', torch.randn(self._num_embeddings, self._embedding_dim*2))
        self.register_buffer('_embedding_output', torch.randn(self._num_embeddings, self._embedding_dim*2))
        self.register_buffer('_ema_cluster_size', torch.zeros(num_embeddings))
        self.register_buffer('_ema_w', torch.randn(self._num_embeddings, self._embedding_dim*2))

        self._decay = decay
        self.bn = torch.nn.BatchNorm1d(self._embedding_dim*2, affine=False)


    def get_k(self) :
        return self._embedding_output

    def get_v(self) :
        return self._embedding_output[:, :self._embedding_dim]

    def update(self, x):
        inputs_normalized = self.bn(x) 
        embedding_normalized = self._embedding

        # Calculate distances
        distances = (torch.sum(inputs_normalized ** 2, dim=1, keepdim=True)
                     + torch.sum(embedding_normalized ** 2, dim=1)
                     - 2 * torch.matmul(inputs_normalized, embedding_normalized.t()))

        # Encoding
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.shape[0], self._num_embeddings, device=x.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Use EMA to update the embedding vectors
        if self.training:
            self._ema_cluster_size.data = self._ema_cluster_size * self._decay + (1 - self._decay) * torch.sum(encodings, 0)

            # Laplace smoothing of the cluster size
            n = torch.sum(self._ema_cluster_size.data)
            self._ema_cluster_size.data = (self._ema_cluster_size + 1e-5) / (n + self._num_embeddings * 1e-5) * n
                    
            # if torch.count_nonzero(self._ema_cluster_size) != self._ema_cluster_size.shape[0] :
            #     raise ValueError('Bad Init!')

            dw = torch.matmul(encodings.t(), inputs_normalized)
            self._ema_w.data = self._ema_w * self._decay + (1 - self._decay) * dw
            self._embedding.data = self._ema_w / self._ema_cluster_size.unsqueeze(1)

            running_std = torch.sqrt(self.bn.running_var + 1e-5).unsqueeze(dim=0)
            running_mean = self.bn.running_mean.unsqueeze(dim=0)
            self._embedding_output.data = self._embedding*running_std + running_mean

        return encoding_indices

class TransformerConv(MessagePassing):

    _alpha: OptTensor

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        global_dim: int,
        num_nodes: int,
        spatial_size: int,
        heads: int = 1,
        concat: bool = True,
        beta: bool = False,
        dropout: float = 0.,
        edge_dim: Optional[int] = None,
        bias: bool = True,
        skip: bool = True,
        dist_count_norm: bool = True,
        conv_type: str = 'local',
        num_centroids: Optional[int] = None,
        # centroid_dim: int = 64,
        **kwargs,
    ):
        kwargs.setdefault('aggr', 'add')
        super(TransformerConv, self).__init__(node_dim=0, **kwargs)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.heads = heads
        self.beta = beta and skip
        self.skip = skip
        self.concat = concat
        self.dropout = dropout
        self.edge_dim = edge_dim
        self.spatial_size = spatial_size
        self.dist_count_norm = dist_count_norm
        self.conv_type = conv_type
        self.num_centroids = num_centroids
        self._alpha = None

        self.lin_key = Linear(in_channels, heads * out_channels)
        self.lin_query = Linear(in_channels, heads * out_channels)
        self.lin_value = Linear(in_channels, heads * out_channels)
        # if edge_dim is not None:
        #     self.lin_edge = Linear(edge_dim, heads * out_channels, bias=False)
        # else:
        #     self.lin_edge = self.register_parameter('lin_edge', None)

        if concat:
            self.lin_skip = Linear(in_channels, heads * out_channels,
                                   bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * heads * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)
        else:
            self.lin_skip = Linear(in_channels, out_channels, bias=bias)
            if self.beta:
                self.lin_beta = Linear(3 * out_channels, 1, bias=False)
            else:
                self.lin_beta = self.register_parameter('lin_beta', None)

        spatial_add_pad = 1
        self.spatial_encoder = torch.nn.Embedding(spatial_size+spatial_add_pad, heads)
        
        if self.conv_type != 'local' :
            self.vq = VectorQuantizerEMA(
                num_centroids, 
                global_dim, 
                decay=0.99
            )
            c = torch.randint(0, num_centroids, (num_nodes,), dtype=torch.short)
            self.register_buffer('c_idx', c)
            self.attn_fn = F.softmax

            self.lin_proj_g = Linear(in_channels, global_dim)
            self.lin_key_g = Linear(global_dim*2, heads * out_channels)
            self.lin_query_g = Linear(global_dim*2, heads * out_channels)
            self.lin_value_g = Linear(global_dim, heads * out_channels)

        self.reset_parameters()

    def reset_parameters(self):
        self.lin_key.reset_parameters()
        self.lin_query.reset_parameters()
        self.lin_value.reset_parameters()
        # if self.edge_dim:
        #     self.lin_edge.reset_parameters()
        self.lin_skip.reset_parameters()
        if self.beta:
            self.lin_beta.reset_parameters()

        torch.nn.init.zeros_(self.spatial_encoder.weight)


    def forward(self, x: Tensor, edge_index: Adj, edge_attr: OptTensor = None, 
                    pos_enc=None, batch_idx=None):

        if self.conv_type == 'local' :
            out = self.local_forward(x, edge_index, edge_attr)[:len(batch_idx)]

        elif self.conv_type == 'global' :
            out = self.global_forward(x[:len(batch_idx)], pos_enc, batch_idx)

        elif self.conv_type == 'full' :
            out_local = self.local_forward(x, edge_index, edge_attr)[:len(batch_idx)]
            out_global = self.global_forward(x[:len(batch_idx)], pos_enc, batch_idx)
            out = torch.cat([out_local, out_global], dim=1)
            
        else :
            raise NotImplementedError

        return out


    def global_forward(self, x, pos_enc, batch_idx):

        d, h = self.out_channels, self.heads
        scale = 1.0 / math.sqrt(d)

        q_x = torch.cat([self.lin_proj_g(x), pos_enc], dim=1)

        k_x = self.vq.get_k()
        v_x = self.vq.get_v()

        q = self.lin_query_g(q_x)
        k = self.lin_key_g(k_x)
        v = self.lin_value_g(v_x)

        q, k, v = map(lambda t: rearrange(t, 'n (h d) -> h n d', h=h), (q, k, v))
        dots = torch.einsum('h i d, h j d -> h i j', q, k) * scale

        c, c_count = self.c_idx.unique(return_counts=True)
        # print(f'c count mean:{c_count.float().mean().item()}, min:{c_count.min().item()}, max:{c_count.max().item()}')

        centroid_count = torch.zeros(self.num_centroids, dtype=torch.long).to(x.device)
        centroid_count[c.to(torch.long)] = c_count
        dots += torch.log(centroid_count.view(1,1,-1))

        attn = self.attn_fn(dots, dim = -1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)

        out = torch.einsum('h i j, h j d -> h i d', attn, v)
        out = rearrange(out, 'h n d -> n (h d)')

        # Update the centroids
        if self.training :
            x_idx = self.vq.update(q_x)
            self.c_idx[batch_idx] = x_idx.squeeze().to(torch.short)

        return out

    def local_forward(self, x: Tensor, edge_index: Adj,
                    edge_attr: OptTensor = None):
            
        H, C = self.heads, self.out_channels

        query = self.lin_query(x).view(-1, H, C)
        key = self.lin_key(x).view(-1, H, C)
        value = self.lin_value(x).view(-1, H, C)

        # propagate_type: (query: Tensor, key:Tensor, value: Tensor, edge_attr: OptTensor) # noqa
        out = self.propagate(edge_index, query=query, key=key, value=value,
                            edge_attr=edge_attr, size=None)
        alpha = self._alpha
        self._alpha = None

        if self.concat:
            out = out.view(-1, H * C)
        else:
            out = out.mean(dim=1)

        if self.skip:
            x_r = self.lin_skip(x[1])
            if self.lin_beta is not None:
                beta = self.lin_beta(torch.cat([out, x_r, out - x_r], dim=-1))
                beta = beta.sigmoid()
                out = beta * x_r + (1 - beta) * out
            else:
                out += x_r

        return out


    def message(self, query_i: Tensor, key_j: Tensor, value_j: Tensor,
                    edge_attr: OptTensor, index: Tensor, ptr: OptTensor,
                    size_i: Optional[int]) -> Tensor:

        # if self.lin_edge is not None:
        #     assert edge_attr is not None
        #     edge_attr = self.lin_edge(edge_attr).view(-1, self.heads, self.out_channels)
        #     key_j += edge_attr

        alpha = (query_i * key_j).sum(dim=-1) / math.sqrt(self.out_channels)
        edge_dist, edge_dist_count = edge_attr[0], edge_attr[1]

        alpha += self.spatial_encoder(edge_dist)

        if self.dist_count_norm :
            alpha -= torch.log(edge_dist_count).unsqueeze_(1)

        alpha = softmax(alpha, index, ptr, size_i)
        self._alpha = alpha
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        out = value_j
        out *= alpha.view(-1, self.heads, 1)
        return out

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, heads={self.heads})')


# GOAT implementation from https://github.com/devnkong/GOAT/blob/main/large_model.py
@register_network('GOATModel')
class Transformer(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dataset,
                 skip=False, dist_count_norm=True, 
                 num_layers=1, num_centroids=4096):
        super(Transformer, self).__init__()
        data = dataset[0]

        # Add embedding to featureless nodes
        # GOAT only support homogeenous graph
        if isinstance(data, HeteroData):
            homo = data.to_homogeneous()
        else:
            homo = data
        self.emb_mask = torch.isnan(homo.x[:, 0]).to(cfg.device)
        if self.emb_mask.sum() > 0:
            self.emb = torch.nn.Embedding(self.emb_mask.sum(), homo.x.shape[1])
            self.emb_idx = torch.nonzero(self.emb_mask).squeeze()
            self.mapping_mask = torch.zeros(len(self.emb_mask), dtype=torch.long) - 1
            self.mapping_mask[self.emb_idx] = torch.arange(len(self.emb_idx))
            self.mapping_mask = self.mapping_mask.to(cfg.device)

        # self.fc_in = nn.Linear(in_channels, hidden_channels) ###################
        no_bn = not cfg.gnn.batch_norm
        if cfg.gnn.batch_norm:
            norm_func = nn.BatchNorm1d
        elif cfg.gnn.layer_norm:
            norm_func = nn.LayerNorm

        num_nodes = data.num_nodes
        conv_type = cfg.gnn.layer_type
        in_channels = list(dim_in.values())[0]
        hidden_channels = cfg.gnn.dim_inner
        global_dim = cfg.gnn.dim_inner
        out_channels = dim_out
        heads = cfg.gnn.attn_heads
        ff_dropout = cfg.gnn.dropout
        attn_dropout = cfg.gnn.attn_dropout
        spatial_size = len(cfg.train.neighbor_sizes)

        if no_bn :
            self.fc_in = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Linear(hidden_channels, hidden_channels)
            )            
        else :
            self.fc_in = nn.Sequential(
                nn.Linear(in_channels, hidden_channels),
                norm_func(hidden_channels),
                nn.ReLU(),
                nn.Dropout(ff_dropout),
                nn.Linear(hidden_channels, hidden_channels)
            )

        self.convs = torch.nn.ModuleList()
        self.ffs = torch.nn.ModuleList()

        assert num_layers == 1
        for _ in range(num_layers):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    global_dim=global_dim,
                    num_nodes=num_nodes,
                    spatial_size=spatial_size,
                    heads=heads,
                    dropout=attn_dropout, 
                    skip=skip, 
                    dist_count_norm=dist_count_norm,
                    conv_type=conv_type,
                    num_centroids=num_centroids
                )
            )
            h_times = 2 if conv_type == 'full' else 1

            if no_bn :
                self.ffs.append(
                    nn.Sequential(
                        nn.Linear(h_times*hidden_channels*heads, hidden_channels*heads),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                        nn.Linear(hidden_channels*heads, hidden_channels),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                    )
                )
            else :
                self.ffs.append(
                    nn.Sequential(
                        nn.Linear(h_times*hidden_channels*heads, hidden_channels*heads),
                        norm_func(hidden_channels*heads),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                        nn.Linear(hidden_channels*heads, hidden_channels),
                        norm_func(hidden_channels),
                        nn.ReLU(),
                        nn.Dropout(ff_dropout),
                    )
                )

        self.fc_out = torch.nn.Linear(hidden_channels, out_channels)

    def reset_parameters(self):
        self.fc_in.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        for ff in self.ffs:
            ff.reset_parameters()
        self.fc_out.reset_parameters()

    def forward(self, batch):
        if isinstance(batch, HeteroData):
            homo = batch.to_homogeneous()
            x, pos_enc, label, edge_index = homo.x, homo.pos_enc, homo.y, homo.edge_index
            edge_attr = homo.edge_attr
            node_type_tensor = homo.node_type
            batch_idx = homo.n_id[:homo.batch_size]
            for idx, node_type in enumerate(batch.node_types):
                if node_type == cfg.dataset.task_entity:
                    node_idx = idx
                    break
        else:
            x, pos_enc, label, edge_index = batch.x, batch.pos_enc, batch.y, batch.edge_index
            edge_attr = batch.edge_attr
            batch_idx = batch.n_id[:batch.batch_size]

            # Add embedding for featureless nodes
            nan_mask_batch = self.emb_mask[batch.n_id]
            nan_indices = torch.where(nan_mask_batch)[0]
            if len(nan_indices) > 0:
                local_emb_indices = self.mapping_mask[batch.n_id[nan_indices]]
                x[nan_indices] = self.emb(local_emb_indices)

        x = self.fc_in(x)
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_attr, pos_enc, batch_idx)
            x = self.ffs[i](x)
        x = self.fc_out(x)

        mask = f'{batch.split}_mask'
        if isinstance(batch, HeteroData):
            x = x[node_type_tensor == node_idx]
            label = label[node_type_tensor == node_idx]
            task = cfg.dataset.task_entity
            if hasattr(batch[task], 'batch_size'):
                batch_size = batch[task].batch_size
                return x[:batch_size], \
                    label[:batch_size]
            else:
                mask = f'{batch.split}_mask'
                return x[batch[task][mask]], \
                    label[batch[task][mask]]
        else:
            if hasattr(batch, 'batch_size'):
                batch_size = batch.batch_size
                return x[:batch_size], \
                    label[:batch_size]
            return x[batch[mask]], label[batch[mask]]

    def global_forward(self, x, pos_enc, batch_idx):
        x = self.fc_in(x)
        for i, conv in enumerate(self.convs):
            x = conv.global_forward(x,  pos_enc, batch_idx)
            x = self.ffs[i](x)
        x = self.fc_out(x)
        return x