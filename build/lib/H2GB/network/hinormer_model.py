from pickle import FALSE
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from H2GB.graphgym.models import head  # noqa, register module
from H2GB.graphgym import register as register
from H2GB.graphgym.config import cfg
from H2GB.graphgym.models.gnn import GNNPreMP
from H2GB.graphgym.register import register_network
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import degree
from H2GB.graphgym.models.layer import BatchNorm1dNode

class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in, data):
        super(FeatureEncoder, self).__init__()
        self.is_hetero = isinstance(data, HeteroData)
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
            self.edge_encoder = EdgeEncoder(cfg.gnn.dim_edge, data)
            if cfg.dataset.edge_encoder_bn:
                self.edge_encoder_bn = BatchNorm1dNode(cfg.gnn.dim_edge)

    def forward(self, batch):
        for module in self.children():
            batch = module(batch)
        return batch


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init


class REConv(MessagePassing):
    def __init__(self, in_channels, out_channels, num_type=4, norm='both', bias=True, activation=None):
        super(REConv, self).__init__(aggr='add')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_type = num_type
        self.norm = norm
        self.activation = activation

        self.weight = nn.Parameter(torch.Tensor(in_channels, out_channels))
        self.weight_type = nn.Parameter(torch.ones(num_type))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x, edge_index, type_info):
        # Optionally apply normalization
        if self.norm in ['both', 'left']:
            row, col = edge_index
            deg = degree(row, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            # norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
            # x = x * norm.view(-1, 1)
            norm = deg_inv_sqrt
            shp = norm.shape + (1,) * (x.dim() - 1)
            norm = norm.reshape(shp)
            x = x * norm


        # Apply linear transformation and type-specific weighting
        x = torch.matmul(x, self.weight)
        x = x * self.weight_type[type_info].view(-1, 1)

        # Message passing
        out = self.propagate(edge_index, x=x)

        # Optionally apply normalization
        if self.norm in ['both', 'right']:
            row, col = edge_index
            deg = degree(col, x.size(0), dtype=x.dtype)
            deg_inv_sqrt = deg.pow(-0.5)
            deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
            # norm = deg_inv_sqrt[col]
            norm = deg_inv_sqrt
            shp = norm.shape + (1,) * (x.dim() - 1)
            norm = norm.reshape(shp)
            out = out * norm.view(-1, 1)

        if self.bias is not None:
            out += self.bias

        if self.activation:
            out = self.activation(out)

        return out

    def message(self, x_j):
        # x_j denotes the features of source nodes j, which are being aggregated at the destination nodes.
        return x_j


class AGTLayer(nn.Module):
    def __init__(self, embeddings_dimension, 
                 nheads=2, att_dropout=0.5, emb_dropout=0.5, 
                 temper=1.0, rl=False, rl_dim=4, beta=1):

        super(AGTLayer, self).__init__()

        self.nheads = nheads
        self.embeddings_dimension = embeddings_dimension

        self.head_dim = self.embeddings_dimension // self.nheads


        self.leaky = nn.LeakyReLU(0.01)

        self.temper = temper

        self.rl_dim = rl_dim

        self.beta = beta

        self.linear_l = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias=False)
        self.linear_r = nn.Linear(
            self.embeddings_dimension, self.head_dim * self.nheads, bias=False)

        self.att_l = nn.Linear(self.head_dim, 1, bias=False)
        self.att_r = nn.Linear(self.head_dim, 1, bias=False)

       
        if rl:
            self.r_source = nn.Linear(rl_dim, rl_dim * self.nheads, bias=False)
            self.r_target = nn.Linear(rl_dim, rl_dim * self.nheads, bias=False)

        self.linear_final = nn.Linear(
            self.head_dim * self.nheads, self.embeddings_dimension, bias=False)
        self.dropout1 = nn.Dropout(att_dropout)
        self.dropout2 = nn.Dropout(emb_dropout)

        self.LN = nn.LayerNorm(embeddings_dimension)

    def forward(self, h, rh=None):
        batch_size = h.size()[0]
        fl = self.linear_l(h).reshape(batch_size, -1, self.nheads, self.head_dim).transpose(1, 2)
        fr = self.linear_r(h).reshape(batch_size, -1, self.nheads, self.head_dim).transpose(1, 2)

        score = self.att_l(self.leaky(fl)) + self.att_r(self.leaky(fr)).permute(0, 1, 3, 2)

        if rh is not None:
            r_k = self.r_source(rh).reshape(batch_size, -1, self.nheads, self.rl_dim).transpose(1,2)
            r_q = self.r_target(rh).reshape(batch_size, -1, self.nheads, self.rl_dim).permute(0, 2, 3, 1)
            score_r = r_k @ r_q
            score = score + self.beta * score_r

        score = score / self.temper

        score = F.softmax(score, dim=-1)
        score = self.dropout1(score)

        context = score @ fr

        h_sa = context.transpose(1,2).reshape(batch_size, -1, self.head_dim * self.nheads)
        fh = self.linear_final(h_sa)
        fh = self.dropout2(fh)

        h = self.LN(h + fh)

        return h

@register_network('HINormerModel')
class HINormer(nn.Module):
    def __init__(self, dim_in, dim_out, dataset,
                 temper=1.0, beta=1):
        super(HINormer, self).__init__()
        homo = dataset[0].to_homogeneous()
        self.edge_index = homo.edge_index.to(cfg.device)
        self.node_type = homo.node_type.to(cfg.device)

        self.embeddings_dimension = cfg.gnn.dim_inner
        self.num_layers = 1
        self.num_classes = dim_out
        self.num_gnns = cfg.gnn.layers_mp
        self.nheads = cfg.gnn.attn_heads
        self.num_types = len(dataset[0].node_types)
        self.type_emb = torch.eye(self.num_types).to(cfg.device)
        # self.encoder = FeatureEncoder(dim_in, data)
        # dim_in = self.encoder.dim_in

        self.fc_list = nn.ModuleList([
            nn.Linear(in_dim, self.embeddings_dimension)
            for in_dim in dim_in.values()
        ])

        self.dropout = cfg.gnn.dropout
        self.GCNLayers = nn.ModuleList([
            GCNConv(self.embeddings_dimension, self.embeddings_dimension) for _ in range(self.num_gnns)
        ])
        # Replace RELayers with an equivalent PyG layer or adapt REConv to PyG
        self.RELayers = nn.ModuleList([
            REConv(self.num_types, self.num_types, activation=F.relu, num_type=self.num_types) for _ in range(self.num_gnns)
        ])
        self.GTLayers = nn.ModuleList([
            AGTLayer(self.embeddings_dimension, self.nheads, self.dropout, self.dropout, temper=temper, 
                     rl=True, rl_dim=self.num_types, beta=beta) 
            for _ in range(self.num_layers)
        ])
        self.drop = nn.Dropout(self.dropout)

        self.prediction = nn.Linear(self.embeddings_dimension, dim_out)
        # self.prediction = nn.Linear(dim_in['paper'], dim_out)
        # self.lin1 = nn.Linear(dim_in['paper'], 128)
        # self.lin2 = nn.Linear(128, dim_out)

    def forward(self, batch, norm=False):
        # batch = self.encoder(batch)
        x, label, seqs = batch
        ego_nodes = seqs[:, 0]

        # homo = batch.to_homogeneous()
        # x, edge_index = homo.x, homo.edge_index
        edge_index = self.edge_index
        node_type_tensor = self.node_type

        h = []
        for idx in range(len(self.fc_list)):
            mask = node_type_tensor==idx
            h.append(self.fc_list[idx](x[mask]))

        gh = torch.cat(h, dim=0)
        r = self.type_emb[node_type_tensor]


        for layer in range(self.num_gnns):
            gh = F.relu(self.GCNLayers[layer](gh, edge_index))
            gh = self.drop(gh)
            r = F.relu(self.RELayers[layer](r, edge_index, node_type_tensor))

        h = gh[seqs]
        # r = r[seqs]
        # for layer in range(self.num_layers):
        #     h = self.GTLayers[layer](h, rh=r)

        # Write back
        # if isinstance(batch, HeteroData):
        #     for idx, node_type in enumerate(batch.node_types):
        #         if node_type == cfg.dataset.task_entity:
        #             mask = node_type_tensor==idx
        #             batch[node_type].x = x[mask]
        #             break
        # else:
        #     batch.x = x

        output = self.prediction(h[:, 0, :])
        if norm:
            output = output / (torch.norm(output, dim=1, keepdim=True) + 1e-12)
        return output, label[ego_nodes]