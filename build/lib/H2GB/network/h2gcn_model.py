import torch
import torch_sparse
from torch_sparse import SparseTensor, matmul
from torch_geometric.data import HeteroData
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.nn import GCNConv, SGConv, GATConv, JumpingKnowledge, APPNP, GCN2Conv, MessagePassing
import scipy.sparse
import torch.nn as nn
import torch.nn.functional as F

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
    

def eidx_to_sp(n: int, edge_index: torch.Tensor, device=None) -> torch.sparse.Tensor:
    indices = edge_index
    values = torch.FloatTensor([1.0] * len(edge_index[0])).to(edge_index.device)
    coo = torch.sparse_coo_tensor(indices=indices, values=values, size=[n, n])
    if device is None:
        device = edge_index.device
    return coo.to(device)

# @register_network('H2GCNModel')
# class H2GCN(nn.Module):
#     def __init__(
#             self,
#             dim_in: int,
#             dim_out: int,
#             data,
#             k: int = 2,
#             dropout: float = 0.5,
#             use_relu: bool = True
#     ):
#         super(H2GCN, self).__init__()
#         self.encoder = FeatureEncoder(dim_in, data)

#         self.dropout = cfg.gnn.dropout
#         self.k = k
#         self.act = F.relu if use_relu else lambda x: x
#         self.use_relu = use_relu
#         self.w_embed = nn.Parameter(
#             torch.zeros(size=(cfg.gnn.dim_inner, cfg.gnn.dim_inner)),
#             requires_grad=True
#         )
#         # self.w_classify = nn.Parameter(
#         #     torch.zeros(size=((2 ** (self.k + 1) - 1) * cfg.gnn.dim_inner, cfg.gnn.dim_inner)),
#         #     requires_grad=True
#         # )
#         self.params = [self.w_embed]
#         self.initialized = False
#         self.num_nodes = list(data.x_dict.values())[0].shape[0]
#         self.a1 = None
#         self.a2 = None

#         GNNHead = register.head_dict[cfg.gnn.head]
#         self.post_mp = GNNHead((2 ** (self.k + 1) - 1) * cfg.gnn.dim_inner, dim_out, data)

#         self.reset_params()

#     def reset_params(self):
#         nn.init.xavier_uniform_(self.w_embed)
# #         nn.init.xavier_uniform_(self.w_classify)

#     @staticmethod
#     def _indicator(sp_tensor: torch.sparse.Tensor) -> torch.sparse.Tensor:
#         csp = sp_tensor.coalesce()
#         return torch.sparse_coo_tensor(
#             indices=csp.indices(),
#             values=torch.where(csp.values() > 0, 1, 0),
#             size=csp.size(),
#             dtype=torch.float
#         )

#     @staticmethod
#     def _spspmm(sp1: torch.sparse.Tensor, sp2: torch.sparse.Tensor) -> torch.sparse.Tensor:
#         assert sp1.shape[1] == sp2.shape[0], 'Cannot multiply size %s with %s' % (sp1.shape, sp2.shape)
#         sp1, sp2 = sp1.coalesce(), sp2.coalesce()
#         index1, value1 = sp1.indices(), sp1.values()
#         index2, value2 = sp2.indices(), sp2.values()
#         m, n, k = sp1.shape[0], sp1.shape[1], sp2.shape[1]
#         indices, values = torch_sparse.spspmm(index1, value1, index2, value2, m, n, k)
#         return torch.sparse_coo_tensor(
#             indices=indices,
#             values=values,
#             size=(m, k),
#             dtype=torch.float
#         )

#     @classmethod
#     def _adj_norm(cls, adj: torch.sparse.Tensor) -> torch.sparse.Tensor:
#         n = adj.size(0)

#         d_diag = torch.pow(torch.sparse.sum(adj, dim=1).values(), -0.5)
#         d_diag = torch.where(torch.isinf(d_diag), torch.full_like(d_diag, 0), d_diag)
#         d_tiled = torch.sparse_coo_tensor(
#             indices=[list(range(n)), list(range(n))],
#             values=d_diag,
#             size=(n, n)
#         )
#         return cls._spspmm(cls._spspmm(d_tiled, adj), d_tiled)

#     def _prepare_prop(self, adj):
#         n = adj.size(0)
#         device = adj.device
#         self.initialized = True
#         sp_eye = torch.sparse_coo_tensor(
#             indices=[list(range(n)), list(range(n))],
#             values=[1.0] * n,
#             size=(n, n),
#             dtype=torch.float
#         ).to(device)
#         # initialize A1, A2
#         a1 = self._indicator(adj - sp_eye)
#         a2 = self._indicator(self._spspmm(adj, adj) - adj - sp_eye)
#         # norm A1 A2
#         self.a1 = self._adj_norm(a1)#.to(cfg.device)
#         self.a2 = self._adj_norm(a2)#.to(cfg.device)
#         # n = self.num_nodes
        
#         # if isinstance(edge_index, SparseTensor):
#         #     dev = edge_index.device
#         #     adj_t = edge_index
#         #     adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
#         #     adj_t[adj_t > 0] = 1
#         #     adj_t[adj_t < 0] = 0
#         #     adj_t = SparseTensor.from_scipy(adj_t).to(dev)
#         # elif isinstance(edge_index, torch.Tensor):
#         #     row, col = edge_index
#         #     adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

#         # adj_t.remove_diag(0)
#         # adj_t2 = matmul(adj_t, adj_t)
#         # adj_t2.remove_diag(0)
#         # adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
#         # adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
#         # adj_t2 = adj_t2 - adj_t
#         # adj_t2[adj_t2 > 0] = 1
#         # adj_t2[adj_t2 < 0] = 0

#         # adj_t = SparseTensor.from_scipy(adj_t)
#         # adj_t2 = SparseTensor.from_scipy(adj_t2)
        
#         # adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
#         # adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

#         # self.a1 = adj_t.to(cfg.device)
#         # self.a2 = adj_t2.to(cfg.device)

#     def forward(self, batch):
#         batch = self.encoder(batch)
#         if isinstance(batch, HeteroData):
#             data = batch.to_homogeneous()
#             x, edge_index = data.x, data.edge_index
#         else:
#             x, edge_index = batch.x, batch.edge_index
#         # x, edge_index = list(batch.x_dict.values())[0], list(batch.edge_index_dict.values())[0]
#         n = x.size()[0]
#         # if not self.initialized:
#         adj = eidx_to_sp(n, edge_index)
#         self._prepare_prop(adj)
#         # H2GCN propagation
#         rs = [self.act(torch.mm(x, self.w_embed))]
#         for i in range(self.k):
#             r_last = rs[-1]
#             r1 = torch.spmm(self.a1, r_last)
#             r2 = torch.spmm(self.a2, r_last)
#             rs.append(self.act(torch.cat([r1, r2], dim=1)))
#         r_final = torch.cat(rs, dim=1)
#         r_final = F.dropout(r_final, self.dropout, training=self.training)

#         if isinstance(batch, HeteroData):
#             batch[cfg.dataset.task_entity].x = r_final # torch.mm(r_final, self.w_classify)
#         else:
#             batch.x = x
#         return self.post_mp(batch)

class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.bns = nn.ModuleList()
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.bns.append(nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hidden_channels, hidden_channels))
                self.bns.append(nn.BatchNorm1d(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))

        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, data, input_tensor=False):
        if not input_tensor:
            x = data.graph['node_feat']
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class H2GCNConv(nn.Module):
    """ Neighborhood aggregation step """
    def __init__(self):
        super(H2GCNConv, self).__init__()

    def reset_parameters(self):
        pass

    def forward(self, x, adj_t, adj_t2):
        x1 = matmul(adj_t, x)
        x2 = matmul(adj_t2, x)
        return torch.cat([x1, x2], dim=1)

@register_network('H2GCNModel')
class H2GCN(nn.Module):
    """ our implementation """
    def __init__(self, dim_in, dim_out, data,
                    num_layers=3, dropout=0.5, save_mem=False, num_mlp_layers=1,
                    use_bn=True, conv_dropout=True):
        super(H2GCN, self).__init__()
        self.encoder = FeatureEncoder(dim_in, data)

        in_channels = cfg.gnn.dim_inner
        hidden_channels = cfg.gnn.dim_inner
        edge_index = list(data.edge_index_dict.values())[0]
        num_nodes = list(data.x_dict.values())[0].shape[0]
        self.feature_embed = MLP(in_channels, hidden_channels,
                hidden_channels, num_layers=num_mlp_layers, dropout=dropout)


        self.convs = nn.ModuleList()
        self.convs.append(H2GCNConv())

        self.bns = nn.ModuleList()
        self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        for l in range(num_layers - 1):
            self.convs.append(H2GCNConv())
            if l != num_layers-2:
                self.bns.append(nn.BatchNorm1d(hidden_channels*2*len(self.convs) ) )

        self.dropout = dropout
        self.activation = F.relu
        self.use_bn = use_bn
        self.conv_dropout = conv_dropout # dropout neighborhood aggregation steps

        self.jump = JumpingKnowledge('cat')
        last_dim = hidden_channels*(2**(num_layers+1)-1)

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(last_dim, dim_out, data)

        self.num_nodes = num_nodes
        self.init_adj(edge_index)

    def reset_parameters(self):
        self.feature_embed.reset_parameters()
        self.final_project.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def init_adj(self, edge_index):
        """ cache normalized adjacency and normalized strict two-hop adjacency,
        neither has self loops
        """
        n = self.num_nodes
        
        if isinstance(edge_index, SparseTensor):
            dev = edge_index.device
            adj_t = edge_index
            adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
            adj_t[adj_t > 0] = 1
            adj_t[adj_t < 0] = 0
            adj_t = SparseTensor.from_scipy(adj_t).to(dev)
        elif isinstance(edge_index, torch.Tensor):
            row, col = edge_index
            adj_t = SparseTensor(row=col, col=row, value=None, sparse_sizes=(n, n))

        adj_t.remove_diag(0)
        adj_t2 = matmul(adj_t, adj_t)
        adj_t2.remove_diag(0)
        adj_t = scipy.sparse.csr_matrix(adj_t.to_scipy())
        adj_t2 = scipy.sparse.csr_matrix(adj_t2.to_scipy())
        adj_t2 = adj_t2 - adj_t
        adj_t2[adj_t2 > 0] = 1
        adj_t2[adj_t2 < 0] = 0

        adj_t = SparseTensor.from_scipy(adj_t)
        adj_t2 = SparseTensor.from_scipy(adj_t2)
        
        adj_t = gcn_norm(adj_t, None, n, add_self_loops=False)
        adj_t2 = gcn_norm(adj_t2, None, n, add_self_loops=False)

        self.adj_t = adj_t.to(cfg.device)
        self.adj_t2 = adj_t2.to(cfg.device)



    def forward(self, batch):
        batch = self.encoder(batch)
        if isinstance(batch, HeteroData):
            data = batch.to_homogeneous()
            x, edge_index = data.x, data.edge_index
            node_type_tensor = homo.node_type
        else:
            x, edge_index = batch.x, batch.edge_index
        self.num_nodes = x.size()[0]
        self.init_adj(edge_index)

        adj_t = self.adj_t
        adj_t2 = self.adj_t2
        
        x = self.feature_embed(x, input_tensor=True)
        x = self.activation(x)
        xs = [x]
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, adj_t, adj_t2) 
            if self.use_bn:
                x = self.bns[i](x)
            xs.append(x)
            if self.conv_dropout:
                x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, adj_t, adj_t2)
        if self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)
        xs.append(x)

        x = self.jump(xs)
        if not self.conv_dropout:
            x = F.dropout(x, p=self.dropout, training=self.training)

        if isinstance(batch, HeteroData):
            for idx, node_type in enumerate(batch.node_types):
                mask = node_type_tensor==idx
                batch[node_type].x = x[mask]
        else:
            batch.x = x
        return self.post_mp(batch)