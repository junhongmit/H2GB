import torch
from torch_sparse import SparseTensor
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from H2GB.graphgym import register as register
from H2GB.graphgym.config import cfg
from H2GB.graphgym.register import register_network
from torch_geometric.nn import MLP
from H2GB.graphgym.models.layer import BatchNorm1dNode


class FeatureEncoder(torch.nn.Module):
    """
    Encoding node and edge features

    Args:
        dim_in (int): Input feature dimension
    """
    def __init__(self, dim_in, dataset):
        super(FeatureEncoder, self).__init__()
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
            x = data.to_homogeneous().x
        else:
            x = data
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.bns[i](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

@register_network('LINKXModel')
class LINKX(nn.Module):	
    """ our LINKX method with skip connections 
        a = MLP_1(A), x = MLP_2(X), MLP_3(sigma(W_1[a, x] + a + x))
    """

    def __init__(self, dim_in, dim_out, dataset, cache=False, inner_activation=False, inner_dropout=False, init_layers_A=1, init_layers_X=1):
        super(LINKX, self).__init__()
        self.encoder = FeatureEncoder(dim_in, dataset)
        data = dataset[0]
        data = data.to_homogeneous()
    
        self.mlpA = MLP(data.num_nodes, cfg.gnn.dim_inner, 
                        cfg.gnn.dim_inner, init_layers_A, dropout=0)
        self.mlpX = MLP(cfg.gnn.dim_inner, cfg.gnn.dim_inner,
                        cfg.gnn.dim_inner, init_layers_X, dropout=0)
        self.W = nn.Linear(2*cfg.gnn.dim_inner, cfg.gnn.dim_inner)
        self.mlp_final = MLP(cfg.gnn.dim_inner, cfg.gnn.dim_inner,
                             dim_out, cfg.gnn.layers_mp, dropout=cfg.gnn.dropout)
        self.in_channels = cfg.gnn.dim_inner
        self.num_nodes = data.num_nodes
        self.A = None
        self.inner_activation = inner_activation
        self.inner_dropout = inner_dropout

    def reset_parameters(self):	
        self.mlpA.reset_parameters()	
        self.mlpX.reset_parameters()
        self.W.reset_parameters()
        self.mlp_final.reset_parameters()	

    def forward(self, batch):
        batch = self.encoder(batch)
        if isinstance(batch, HeteroData):
            data = batch.to_homogeneous()
            node_type_tensor = data.node_type
        else:
            data = batch
        m = data.num_nodes
        if hasattr(data, "n_id"):
            row, col = data.edge_index[0, :], data.n_id[data.edge_index[1, :]]
        else:
            row, col = data.edge_index
        row = row-row.min()
        A = SparseTensor(row=row, col=col,	
                 sparse_sizes=(m, self.num_nodes)
                        ).to_torch_sparse_coo_tensor()

        xA = self.mlpA(A, input_tensor=True)
        xX = self.mlpX(data.x, input_tensor=True)
        x = torch.cat((xA, xX), axis=-1)
        x = self.W(x)
        if self.inner_dropout:
            x = F.dropout(x)
        if self.inner_activation:
            x = F.relu(x)
        x = F.relu(x + xA + xX)
        x = self.mlp_final(x, input_tensor=True)

        if isinstance(batch, HeteroData):
            for idx, node_type in enumerate(batch.node_types):
                mask = node_type_tensor==idx
                batch[node_type].x = x[mask]
            task = cfg.dataset.task_entity
            if hasattr(batch[task], 'batch_size'):
                batch_size = batch[task].batch_size
                return batch[task].x[:batch_size], \
                    batch[task].y[:batch_size]
            else:
                mask = f'{batch.split}_mask'
                return batch[task].x[batch[task][mask]], \
                    batch[task].y[batch[task][mask]]
        else:
            mask = f'{batch.split}_mask'
            # print(batch[mask], batch[mask].sum())
            # print(x[batch[mask]], batch.y[batch[mask]], batch.y[batch[mask]].numel())
            return x[batch[mask]], batch.y[batch[mask]]