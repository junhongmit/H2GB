from tqdm import tqdm
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_mean
from torch_sparse import SparseTensor, matmul
from torch_geometric.data import HeteroData
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import (add_remaining_self_loops, remove_self_loops)

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


def scipy_coo_matrix_to_torch_sparse_tensor(sparse_mx):
    indices1 = torch.from_numpy(np.stack([sparse_mx.row, sparse_mx.col]).astype(np.int64))
    values1 = torch.from_numpy(sparse_mx.data)
    shape1 = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices=indices1, values=values1, size=shape1)

def cal_filter(edge_index, num_nodes):
    edge_index = edge_index.cpu()
    N = num_nodes

    # A
    edge_index, _ = remove_self_loops(edge_index=edge_index)
    edge_index_sl, _ = add_remaining_self_loops(edge_index=edge_index)
    
    # D
    adj_data = np.ones([edge_index.shape[1]], dtype=np.float32)
    adj_sp = sp.csr_matrix((adj_data, (edge_index[0], edge_index[1])), shape=[N, N])

    adj_sl_data = np.ones([edge_index_sl.shape[1]], dtype=np.float32)
    adj_sl_sp = sp.csr_matrix((adj_sl_data, (edge_index_sl[0], edge_index_sl[1])), shape=[N, N])

    # D-1/2
    deg = np.array(adj_sl_sp.sum(axis=1)).flatten()
    deg_sqrt_inv = np.power(deg, -0.5)
    deg_sqrt_inv[deg_sqrt_inv == float('inf')] = 0.0
    deg_sqrt_inv = sp.diags(deg_sqrt_inv)

    # filters
    DAD = sp.coo_matrix(deg_sqrt_inv * adj_sp * deg_sqrt_inv)
    DAD = scipy_coo_matrix_to_torch_sparse_tensor(DAD)

    return DAD

@register_network('LSGNNModel')
class LSGNN(MessagePassing):
    """local similarity graph neural network"""
    def __init__(
        self, dim_in, dim_out, dataset,
        k=5, beta=1, gamma=0.5, method='norm2'
        ):
        super(LSGNN, self).__init__(aggr='add')

        in_channels = list(dim_in.values())[0]
        out_channels = dim_out
        hidden_channels = cfg.gnn.dim_inner
        num_reduce_layers = cfg.gnn.layers_mp
        self.num_nodes = dataset[0].num_nodes
        self.K = k
        self.beta = beta
        self.gamma = gamma
        self.method = method
        self.dp = cfg.gnn.dropout

        # Add embedding to featureless nodes
        # LSGNN only support homogeenous graph
        data = dataset.data
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
            
            max_node_type = len(data.node_types)
            self.start_indices = torch.zeros(max_node_type + 1, dtype=torch.long, device=cfg.device)
            for idx, node_type in enumerate(data.node_types):
                self.start_indices[idx + 1] = self.start_indices[idx] + data.num_nodes_dict[node_type]
            

        self.dist_mlp = nn.Sequential(
            nn.Linear(2, hidden_channels), 
            nn.SiLU(), 
            nn.Linear(hidden_channels, 1)
        )

        self.alpha_mlp = nn.Sequential(
            nn.Linear(2, hidden_channels), 
            nn.SiLU(), 
            nn.Linear(hidden_channels, 3*self.K)
        )

        if num_reduce_layers == 1:
            self.reduce = [nn.Parameter(torch.zeros([2*self.K+1, in_channels, hidden_channels]))]
        elif num_reduce_layers > 1:
            self.reduce = [nn.Parameter(torch.zeros([2*self.K+1, in_channels, 2*hidden_channels]))]
            for _ in range(num_reduce_layers-2):
                self.reduce.append(nn.Parameter(torch.zeros([2*self.K+1, in_channels, 2*hidden_channels])))
            self.reduce.append(nn.Parameter(torch.zeros([2*self.K+1, in_channels, hidden_channels])))
        else:
            raise NotImplementedError
        self.reset_parameters()

        if cfg.gnn.A_embed:
            self.A_mlp = nn.Sequential(
                nn.Linear(self.num_nodes, hidden_channels), 
                nn.BatchNorm1d(hidden_channels), 
                nn.ReLU(), 
            )
            final_nz = self.K + 2
        else:
            self.A_mlp = None
            final_nz = self.K + 1

        # if config['out_mlp']:
        if False:
            self.out_linear = nn.Sequential(
                nn.Linear(final_nz*hidden_channels, 2*hidden_channels), 
                nn.BatchNorm1d(2*hidden_channels), 
                nn.ReLU(), 
                nn.Linear(2*hidden_channels, out_channels)
            )
        else:
            self.out_linear = nn.Linear(final_nz*hidden_channels, out_channels)

        self.cache = None

        # data = dataset[0]
        # data = data.to_homogeneous()
        # x, edge_index = data.x, data.edge_index
        # print('pre-computing distances...')
        # self.dist = self.dist(x, edge_index).to(cfg.device)

        # print('pre-computing DAD...')
        # N = x.shape[0]
        # DAD = cal_filter(edge_index, N)
        # #DAD = DAD.to(cfg.device)

        # print('pre-computing intermediate representations...')
        # self.x_out_L_out_H = self.prop(x, edge_index, DAD).to(cfg.device)

    @torch.no_grad()
    def reset_parameters(self):
        for i, param in enumerate(self.reduce):
            self.register_parameter(f'reduce_{i}', param)
            nn.init.xavier_uniform_(param.data)

    def dist(self, x, edge_index:torch.Tensor, verbose=False):
        src, tgt = edge_index
        
        # if self.ds not in c.large_graph:
        if self.method == 'cos':
            dist = (x[src] * x[tgt]).sum(dim=-1)
        elif self.method == 'norm2':
            dist = torch.norm(x[src] - x[tgt], p=2, dim=-1)
        # else:
        # split_size = 10000
        # dist = []
        # for ei_i in tqdm(edge_index.split(split_size, dim=-1), ncols=70, disable=not verbose):
        #     src_i, tgt_i = ei_i
        #     if self.method == 'cos':                    
        #         dist.append((x[src_i] * x[tgt_i]).sum(dim=-1))
        #     elif self.method == 'norm2':
        #         dist.append(torch.norm(x[src_i] - x[tgt_i], p=2, dim=-1))
        # dist = torch.cat(dist, dim=0)

        dist = dist.view(-1, 1)
        dist = torch.cat([dist, dist.square()], dim=-1)
        return dist

    def local_sim(self, x, edge_index, dist=None):
        if dist is None:
            dist = self.dist(x, edge_index)

        _, tgt = edge_index
        dist = self.dist_mlp(dist).view(-1)
        return scatter_mean(dist, tgt, out=torch.zeros([x.shape[0]], device=x.device))

    def prop(self, x, edge_index, DAD=None):
        N, _ = x.shape
        dev = x.device

        # cal filters
        if DAD == None:
            DAD = cal_filter(edge_index=edge_index, num_nodes=N).to(dev)

        # beta
        # if self.ds not in c.large_graph:
        #     I = torch.eye(N, device=dev)
        #     filter_l = SparseTensor.from_dense(self.beta * I + DAD)
        #     filter_h = SparseTensor.from_dense((1 - self.beta) * I - DAD)
        # else:
        I_indices = torch.LongTensor([[i,i] for i in range(N)]).t()
        I_values = torch.tensor([1. for _ in range(N)])
        I_sparse = torch.sparse.FloatTensor(indices=I_indices, values=I_values, size=torch.Size([N,N])).to(dev)
        filter_l = SparseTensor.from_torch_sparse_coo_tensor(self.beta * I_sparse + DAD)
        filter_h = SparseTensor.from_torch_sparse_coo_tensor((1. - self.beta) * I_sparse - DAD)

        # propagate first
        x = x.type(torch.float32)
        # first propagation
        x_L = x.clone()
        x_H = x.clone()
        x_L = self.propagate(edge_index=filter_l, x=x_L)
        x_H = self.propagate(edge_index=filter_h, x=x_H)
        out_L = [x_L]
        out_H = [x_H]
        x_L_sum = 0
        x_H_sum = 0
        # continue propagation
        for _ in range(1, self.K):
            x_L_sum = x_L_sum + out_L[-1]
            x_H_sum = x_H_sum + out_H[-1]
            x_L = self.propagate(
                edge_index=filter_l, x=(1-self.gamma)*x-self.gamma*x_L_sum)
            x_H = self.propagate(
                edge_index=filter_h, x=(1-self.gamma)*x-self.gamma*x_H_sum)

            out_L.append(x_L)
            out_H.append(x_H)

        x_out_L_out_H = torch.stack([x] + out_L + out_H, dim=0)
        return x_out_L_out_H

    def forward(self, batch):
        if isinstance(batch, HeteroData):
            homo = batch.to_homogeneous()
            data = homo
            x, label, edge_index = homo.x, homo.y, homo.edge_index
            node_type_tensor = homo.node_type
            for idx, node_type in enumerate(batch.node_types):
                if node_type == cfg.dataset.task_entity:
                    node_idx = idx
                    break
            
            # Add embedding for featureless nodes
            homo.n_id = self.start_indices[homo.node_type] + homo.n_id
            nan_mask_batch = self.emb_mask[homo.n_id]
            nan_indices = torch.where(nan_mask_batch)[0]
            if len(nan_indices) > 0:
                local_emb_indices = self.mapping_mask[homo.n_id[nan_indices]]
                x[nan_indices] = self.emb(local_emb_indices)
        else:
            label, edge_index = batch.y, batch.edge_index
            data = batch
        N, _ = x.shape
        dev = x.device

        # cal node sim
        local_sim = self.local_sim(x, edge_index) # self.dist
        ls_ls2 = torch.cat([local_sim.view(-1, 1), local_sim.view(-1, 1).square()], dim=-1)

        self.x_out_L_out_H = self.prop(x, edge_index).to(dev)

        # cal alpha
        alpha = self.alpha_mlp(ls_ls2)  # (N, 3K)
        stack_alpha = alpha.reshape([N, self.K, 3])
        alpha_I = stack_alpha[:,:,0].t().unsqueeze(-1)
        alpha_L = stack_alpha[:,:,1].t().unsqueeze(-1)
        alpha_H = stack_alpha[:,:,2].t().unsqueeze(-1)

        # reduce dimensional
        for reduce_layer in self.reduce:
            x_out_L_out_H = torch.bmm(self.x_out_L_out_H, reduce_layer)
            x_out_L_out_H = F.normalize(x_out_L_out_H, p=2, dim=-1)
            x_out_L_out_H = F.relu(x_out_L_out_H)

        x = x_out_L_out_H[0,:,:]               # (N, hdim)
        out_I = x.expand(self.K, -1, -1)       # (K, N, hdim)
        out_L = x_out_L_out_H[1:self.K+1,:,:]  # (K, N, hdim)
        out_H = x_out_L_out_H[self.K+1:,:,:]   # (K, N, hdim)

        # fusion: (K, N, hdim)
        out = alpha_I * out_I + alpha_L * out_L + alpha_H * out_H

        # embedding A and concat representations
        if self.A_mlp is not None:
            if hasattr(data, "n_id"):
                row, col = data.edge_index[0, :], data.n_id[data.edge_index[1, :]]
            else:
                row, col = data.edge_index
            A = SparseTensor(
                row=row, col=col, 
                value=torch.ones([edge_index.shape[1]]).to(dev), 
                sparse_sizes=[N, self.num_nodes]).to_torch_sparse_coo_tensor()
            A = self.A_mlp(A)
            out = torch.cat([x.unsqueeze(0), out, A.unsqueeze(0)], dim=0)
        else:
            out = torch.cat([x.unsqueeze(0), out], dim=0)

        # norm (K+1, N, hdim)
        # if self.config['out_norm']
        if True:
            out = F.normalize(out, p=2, dim=-1)
        out = F.dropout(out, self.dp, self.training)
        out = out.permute(1, 0, 2).reshape(N, -1)

        # prediction
        out = self.out_linear(out)
        
        mask = f'{batch.split}_mask'
        if isinstance(batch, HeteroData):
            out = out[node_type_tensor == node_idx]
            label = label[node_type_tensor == node_idx]
            task = cfg.dataset.task_entity
            if hasattr(batch[task], 'batch_size'):
                batch_size = batch[task].batch_size
                return out[:batch_size], \
                    label[:batch_size]
            else:
                mask = f'{batch.split}_mask'
                return out[batch[task][mask]], \
                    label[batch[task][mask]]
        else:
            return out[batch[mask]], label[batch[mask]]

    def message(self, x_j, norm):
        # x_j: (E, out_channels)
        # norm: (E)
        return norm.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t, x):
        return matmul(adj_t, x, reduce=self.aggr)