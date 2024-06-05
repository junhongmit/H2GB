import math, torch
import numpy as np
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
import torch_sparse
from torch_sparse import SparseTensor, matmul
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import GCNConv, SGConv, GATConv, JumpingKnowledge, APPNP, GCN2Conv, MessagePassing
from torch_geometric.utils.convert import to_scipy_sparse_matrix
from torch_geometric.utils import to_torch_coo_tensor
import scipy.sparse as sp
from sklearn.preprocessing import normalize as sk_normalize

from H2GB.graphgym.models import head  # noqa, register module
from H2GB.graphgym import register as register
from H2GB.graphgym.config import cfg
from H2GB.graphgym.models.gnn import GNNPreMP
from H2GB.graphgym.register import register_network
from torch_geometric.nn import Linear, HGTConv
from torch_geometric.loader import NeighborLoader
from H2GB.graphgym.models.layer import BatchNorm1dNode

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


class GraphConvolution(nn.Module):
    """
    Simple GCN layer, similar to https://arxiv.org/abs/1609.02907
    """

    def __init__(self, in_features, out_features, model_type, output_layer=0, variant=False):
        super(GraphConvolution, self).__init__()
        self.in_features, self.out_features, self.output_layer, self.model_type, self.variant = in_features, out_features, output_layer, model_type, variant
        self.att_low, self.att_high, self.att_mlp = 0, 0, 0

        self.weight_low, self.weight_high, self.weight_mlp = Parameter(torch.FloatTensor(in_features, out_features)), Parameter(
            torch.FloatTensor(in_features, out_features)), Parameter(torch.FloatTensor(in_features, out_features))
        self.att_vec_low, self.att_vec_high, self.att_vec_mlp = Parameter(torch.FloatTensor(out_features, 1)), Parameter(
            torch.FloatTensor(out_features, 1)), Parameter(torch.FloatTensor(out_features, 1))
        self.low_param, self.high_param, self.mlp_param = Parameter(torch.FloatTensor(
            1, 1)), Parameter(torch.FloatTensor(1, 1)), Parameter(torch.FloatTensor(1, 1))

        self.att_vec = Parameter(torch.FloatTensor(3, 3))

        self.reset_parameters()

    def reset_parameters(self):

        stdv = 1. / math.sqrt(self.weight_mlp.size(1))
        std_att = 1. / math.sqrt(self.att_vec_mlp.size(1))

        std_att_vec = 1. / math.sqrt(self.att_vec.size(1))
        self.weight_low.data.uniform_(-stdv, stdv)
        self.weight_high.data.uniform_(-stdv, stdv)
        self.weight_mlp.data.uniform_(-stdv, stdv)
        self.att_vec_high.data.uniform_(-std_att, std_att)
        self.att_vec_low.data.uniform_(-std_att, std_att)
        self.att_vec_mlp.data.uniform_(-std_att, std_att)

        self.att_vec.data.uniform_(-std_att_vec, std_att_vec)

    def attention(self, output_low, output_high, output_mlp):
        T = 3
        att = torch.softmax(torch.mm(torch.sigmoid(torch.cat([torch.mm((output_low), self.att_vec_low), torch.mm(
            (output_high), self.att_vec_high), torch.mm((output_mlp), self.att_vec_mlp)], 1)), self.att_vec)/T, 1)
        return att[:, 0][:, None], att[:, 1][:, None], att[:, 2][:, None]

    def forward(self, input, adj_low, adj_high):
        output = 0
        if self.model_type == 'mlp':
            output_mlp = (torch.mm(input, self.weight_mlp))
            return output_mlp
        elif self.model_type == 'sgc' or self.model_type == 'gcn':
            output_low = torch.mm(adj_low, torch.mm(input, self.weight_low))
            return output_low
        elif self.model_type == 'acmgcn' or self.model_type == 'acmsnowball':
            if self.variant:
                output_low = (torch.spmm(adj_low, F.relu(
                    torch.mm(input, self.weight_low))))
                output_high = (torch.spmm(adj_high, F.relu(
                    torch.mm(input, self.weight_high))))
                output_mlp = F.relu(torch.mm(input, self.weight_mlp))
            else:
                output_low = F.relu(torch.spmm(
                    adj_low, (torch.mm(input, self.weight_low))))
                output_high = F.relu(torch.spmm(
                    adj_high, (torch.mm(input, self.weight_high))))
                output_mlp = F.relu(torch.mm(input, self.weight_mlp))

            self.att_low, self.att_high, self.att_mlp = self.attention(
                (output_low), (output_high), (output_mlp))
            # 3*(output_low + output_high + output_mlp) #
            return 3*(self.att_low*output_low + self.att_high*output_high + self.att_mlp*output_mlp)
        elif self.model_type == 'acmsgc':
            output_low = torch.spmm(adj_low, torch.mm(input, self.weight_low))
            # torch.mm(input, self.weight_high) - torch.spmm(self.A_EXP,  torch.mm(input, self.weight_high))
            output_high = torch.spmm(
                adj_high,  torch.mm(input, self.weight_high))
            output_mlp = torch.mm(input, self.weight_mlp)

            # self.attention(F.relu(output_low), F.relu(output_high), F.relu(output_mlp))
            self.att_low, self.att_high, self.att_mlp = self.attention(
                (output_low), (output_high), (output_mlp))
            # 3*(output_low + output_high + output_mlp) #
            return 3*(self.att_low*output_low + self.att_high*output_high + self.att_mlp*output_mlp)

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
            + str(self.in_features) + ' -> ' \
            + str(self.out_features) + ')'
    

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor."""
    sparse_mx = sparse_mx.tocoo().astype(np.float32)
    indices = torch.from_numpy(
        np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)


def row_normalized_adjacency(adj):
    adj = sp.coo_matrix(adj)
    adj = adj + sp.eye(adj.shape[0])
    adj_normalized = sk_normalize(adj, norm='l1', axis=1)
    # row_sum = np.array(adj.sum(1))
    # row_sum = (row_sum == 0)*1+row_sum
    # adj_normalized = adj/row_sum
    return sp.coo_matrix(adj_normalized)

def get_adj_high(adj_low):
    adj_high = -adj_low + sp.eye(adj_low.shape[0])
    return adj_high

# def normalize_sparse_adjacency(adj):
#     """Normalize sparse adjacency matrix with the L1 norm."""
#     row_sum = torch.sparse.sum(adj, dim=1).to_dense()
#     row_sum = row_sum + (row_sum == 0).float()  # To prevent division by zero
#     inv_row_sum = torch.pow(row_sum, -1)
#     inv_row_sum[torch.isinf(inv_row_sum)] = 0.0
#     inv_diag_matrix = torch.diag(inv_row_sum)
#     adj = torch.sparse.mm(inv_diag_matrix, adj)
#     return adj

def normalize_sparse_adjacency(adj):
    """Normalize sparse adjacency matrix with the L1 norm using row-wise multiplication."""
    row_sum = torch.sparse.sum(adj, dim=1).to_dense()
    row_sum = row_sum + (row_sum == 0).float()  # Prevent division by zero
    inv_row_sum = torch.pow(row_sum, -1)
    inv_row_sum[torch.isinf(inv_row_sum)] = 0.0
    inv_row_sum = inv_row_sum.view(-1, 1)  # Convert to column vector for broadcasting

    # Sparse multiplication
    row_indices = adj._indices()[0]
    values = adj._values() * inv_row_sum[row_indices].squeeze()

    return torch.sparse_coo_tensor(adj._indices(), values, adj.size(), device=adj.device)


def sparse_eye(n, device):
    """Create an identity matrix as a sparse tensor."""
    indices = torch.arange(n, device=device).repeat(2, 1)
    values = torch.ones(n, device=device)
    return torch.sparse_coo_tensor(indices, values, (n, n), device=device)


def get_normalized_adjacency_matrices(edge_index, num_nodes, device):
    """Get the low-pass (adj_low) and high-pass (adj_high) adjacency matrices."""
    adj_low = to_torch_coo_tensor(edge_index, size=(num_nodes, num_nodes))
    adj_low = adj_low.to(device)
    adj_low = normalize_sparse_adjacency(adj_low)
    identity = sparse_eye(num_nodes, device)
    adj_high = identity - adj_low
    return adj_low, adj_high

@register_network('ACMGCNModel')
class ACMGCN(nn.Module):
    def __init__(self, dim_in, dim_out, dataset,
                 nlayers=1, variant=False):
        super(ACMGCN, self).__init__()
        self.encoder = FeatureEncoder(dim_in, dataset)

        nfeat = cfg.gnn.dim_inner
        nhid = cfg.gnn.dim_inner
        nclass = dim_out
        self.gcns, self.mlps = nn.ModuleList(), nn.ModuleList()
        self.model_type, self.nlayers = cfg.gnn.layer_type, nlayers
        if self.model_type == 'mlp':
            self.gcns.append(GraphConvolution(
                nfeat, nhid, model_type=self.model_type))
            self.gcns.append(GraphConvolution(
                nhid, nclass, model_type=self.model_type, output_layer=1))
        elif self.model_type == 'gcn' or self.model_type == 'acmgcn':
            self.gcns.append(GraphConvolution(
                nfeat, nhid,  model_type=self.model_type, variant=variant))
            for _ in range(1, cfg.gnn.layers_mp - 1):
                self.gcns.append(GraphConvolution(
                    nhid, nhid,  model_type=self.model_type, variant=variant))
            self.gcns.append(GraphConvolution(
                nhid, nclass,  model_type=self.model_type, output_layer=1, variant=variant))
        elif self.model_type == 'sgc' or self.model_type == 'acmsgc':
            self.gcns.append(GraphConvolution(
                nfeat, nclass, model_type=self.model_type))
        elif self.model_type == 'acmsnowball':
            for k in range(nlayers):
                self.gcns.append(GraphConvolution(
                    k * nhid + nfeat, nhid, model_type=self.model_type, variant=variant))
            self.gcns.append(GraphConvolution(
                nlayers * nhid + nfeat, nclass, model_type=self.model_type, variant=variant))
        
        self.dropout = cfg.gnn.dropout

    def reset_parameters(self):
        for gcn in self.gcns:
            gcn.reset_parameters()

    def forward(self, batch):
        batch = self.encoder(batch)
        if isinstance(batch, HeteroData):
            homo = batch.to_homogeneous()
            x, label, edge_index = homo.x, homo.y, homo.edge_index
            node_type_tensor = homo.node_type
            num_nodes = homo.num_nodes
            for idx, node_type in enumerate(batch.node_types):
                if node_type == cfg.dataset.task_entity:
                    node_idx = idx
                    break
        else:
            x, label, edge_index = batch.x, batch.y, batch.edge_index
        
        adj_low, adj_high = get_normalized_adjacency_matrices(edge_index, num_nodes, cfg.device)

        if self.model_type == 'acmgcn' or self.model_type == 'acmsgc' or self.model_type == 'acmsnowball':
            x = F.dropout(x, self.dropout, training=self.training)

        if self.model_type == 'acmsnowball':
            list_output_blocks = []
            for layer, layer_num in zip(self.gcns, np.arange(self.nlayers)):
                if layer_num == 0:
                    list_output_blocks.append(F.dropout(
                        F.relu(layer(x, adj_low, adj_high)), self.dropout, training=self.training))
                else:
                    list_output_blocks.append(F.dropout(F.relu(layer(torch.cat(
                        [x] + list_output_blocks[0: layer_num], 1), adj_low, adj_high)), self.dropout, training=self.training))
            return self.gcns[-1](torch.cat([x] + list_output_blocks, 1), adj_low, adj_high)

        fea = (self.gcns[0](x, adj_low, adj_high))

        if self.model_type == 'gcn' or self.model_type == 'mlp' or self.model_type == 'acmgcn':
            fea = F.dropout(F.relu(fea), self.dropout, training=self.training)
            fea = self.gcns[-1](fea, adj_low, adj_high)


        mask = f'{batch.split}_mask'
        if isinstance(batch, HeteroData):
            fea = fea[node_type_tensor == node_idx]
            label = label[node_type_tensor == node_idx]
            task = cfg.dataset.task_entity
            if hasattr(batch[task], 'batch_size'):
                batch_size = batch[task].batch_size
                return fea[:batch_size], \
                    label[:batch_size]
            else:
                return fea[batch[task][mask]], \
                    label[batch[task][mask]]
        else:
            return fea[batch[mask]], label[batch[mask]]