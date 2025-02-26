import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import (HeteroData, Batch)

from H2GB.graphgym.models import head  # noqa, register module
from H2GB.graphgym import register as register
from H2GB.graphgym.config import cfg
from H2GB.graphgym.models.gnn import GNNPreMP
from H2GB.graphgym.register import register_network
from torch_geometric.nn import (Sequential, Linear, HeteroConv, GATConv, RGCNConv)
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn import (GCN, GraphSAGE, GIN, GAT, PNA)
from H2GB.graphgym.models.layer import BatchNorm1dNode
from H2GB.layer.gatedgcn_layer import GatedGCNLayer
from H2GB.layer.polyformer_layer import PolyFormerBlock

def init_params(module, n_layers):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()
    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)



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
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
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


@register_network('PolyFormerModel')
class PolyFormer(nn.Module): 
    def __init__(self, dim_in, dim_out, dataset, base="mono", q=2.0):
        super(PolyFormer,self).__init__()
        self.dropout = cfg.gnn.dropout
        self.nlayers = cfg.gnn.layers_mp
        self.dataset = dataset
        self.input_dim = list(dim_in.values())[0]
        self.hidden_dim = cfg.gnn.dim_inner
        self.num_heads = cfg.gnn.attn_heads
        
        self.attn = nn.ModuleList([PolyFormerBlock(self.hidden_dim, self.num_heads, base, q) for _ in range(self.nlayers)])
        self.K = cfg.gnn.hops + 1
        self.base = base

        self.lin1 = Linear(self.input_dim, cfg.gnn.dim_inner)
        self.lin2 = Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner)
        self.lin3 = Linear(cfg.gnn.dim_inner, dim_out)
        self.reset_parameters()

    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, batch):
        x, label = batch
        input_mat = x[:, :self.K, :]
        # input_mat = torch.stack(input_mat, dim = 1) # [N,k,d]
        input_mat = self.lin1(input_mat) # just for common dataset
       
        for block in self.attn:
            input_mat = block(input_mat)
            
        x = torch.sum(input_mat, dim = 1) # [N,d]
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        output = self.lin3(x)
        
        return output, label
        # return F.log_softmax(x, dim=1)
    

class FeedForwardModule(nn.Module):
    def __init__(self, hidden_dim, dropout=0.1):
        super().__init__()
        self.linear_1 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.dropout_1 = nn.Dropout(p=dropout)
        self.act = nn.GELU()
        self.linear_2 = nn.Linear(in_features=hidden_dim, out_features=hidden_dim)
        self.dropout_2 = nn.Dropout(p=dropout)
        self.reset_parameters()
    
    def reset_parameters(self):
        self.linear_1.reset_parameters()
        self.linear_2.reset_parameters()

    def forward(self, x):
        x = self.linear_1(x)
        x = self.dropout_1(x)
        x = self.act(x)
        x = self.linear_2(x)
        x = self.dropout_2(x)
        return x
    
    