import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, GATConv, Linear
from torch_geometric.utils import softmax
from torch_geometric.data import HeteroData

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

class SHGNConv(MessagePassing):
    def __init__(self, num_etypes, in_feats, out_feats,
                 negative_slope=0.2, residual=False,
                 allow_zero_in_degree=False, bias=False, alpha=0.):
        super(SHGNConv, self).__init__(node_dim=0)  # Aggregation over nodes.
        self.edge_feats = cfg.gnn.dim_inner
        self.num_heads = cfg.gnn.attn_heads
        self.in_feats = in_feats
        self.out_feats = out_feats
        self.allow_zero_in_degree = allow_zero_in_degree

        self.edge_emb = nn.Embedding(num_etypes, self.edge_feats)
        self.fc = nn.Linear(self.in_feats, self.out_feats * self.num_heads, bias=False)
        self.fc_e = nn.Linear(self.edge_feats, self.edge_feats * self.num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.Tensor(1, self.num_heads, self.out_feats))
        self.attn_r = nn.Parameter(torch.Tensor(1, self.num_heads, self.out_feats))
        self.attn_e = nn.Parameter(torch.Tensor(1, self.num_heads, self.edge_feats))
        self.feat_drop = nn.Dropout(cfg.gnn.dropout)
        self.attn_drop = nn.Dropout(cfg.gnn.attn_dropout)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            self.res_fc = nn.Linear(self.in_feats, self.num_heads * self.out_feats, bias=False) \
            if self.in_feats != self.out_feats else nn.Identity()
        else:
            self.res_fc = None
        self.activation = register.act_dict[cfg.gnn.act]
        self.bias = nn.Parameter(torch.zeros((1, self.num_heads, self.out_feats))) if bias else None
        self.alpha = alpha

        self.reset_parameters()

    def reset_parameters(self):
        gain = torch.nn.init.calculate_gain('relu')
        torch.nn.init.xavier_normal_(self.fc.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.attn_l, gain=gain)
        torch.nn.init.xavier_normal_(self.attn_r, gain=gain)
        torch.nn.init.xavier_normal_(self.attn_e, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            torch.nn.init.xavier_normal_(self.res_fc.weight, gain=gain)
        torch.nn.init.xavier_normal_(self.fc_e.weight, gain=gain)

    def forward(self, x, edge_index, edge_type, res_attn=None):
        x = self.feat_drop(x)
        x = self.fc(x).view(-1, self.num_heads, self.out_feats)

        edge_emb = self.edge_emb(edge_type)
        e_feat = self.fc_e(edge_emb).view(-1, self.num_heads, self.edge_feats)
        ee = (e_feat * self.attn_e).sum(dim=-1, keepdim=True)

        row, col = edge_index
        alpha_l = (x[row] * self.attn_l).sum(dim=-1, keepdim=True)
        alpha_r = (x[col] * self.attn_r).sum(dim=-1, keepdim=True)
        alpha = alpha_l + alpha_r + ee
        alpha = self.leaky_relu(alpha)
        alpha = softmax(alpha, col, num_nodes=x.size(0))
        alpha = self.attn_drop(alpha)

        if res_attn is not None:
            alpha = alpha * (1 - self.alpha) + res_attn * self.alpha

        out = self.propagate(edge_index, x=x, alpha=alpha)
        if self.res_fc is not None:
            out += self.res_fc(x[col]).view(-1, self.num_heads, self.out_feats)

        if self.bias is not None:
            out += self.bias

        if self.activation:
            out = self.activation(out)

        return out, alpha
    

@register_network('SHGNModel')
class SHGN(nn.Module):
    def __init__(self, dim_in, dim_out, dataset,
                 negative_slope=0.2, residual=False, alpha=0.):
        super(SHGN, self).__init__()
        self.num_layers = cfg.gnn.layers_mp
        self.gat_layers = nn.ModuleList()
        self.num_heads = cfg.gnn.attn_heads

        self.encoder = FeatureEncoder(dim_in, dataset)
        
        # Create an initial linear transformation for each feature dimension
        # self.fc_list = nn.ModuleList([Linear(in_features, num_hidden, bias=True) for in_features in num_features])
        # for fc in self.fc_list:
        #     torch.nn.init.xavier_normal_(fc.weight, gain=1.414)
        
        # Create GAT layers
        num_etypes = len(dataset[0].edge_types)
        num_hidden = cfg.gnn.dim_inner
        num_classes = dim_out
        self.gat_layers.append(SHGNConv(num_etypes, num_hidden, 
                                        num_hidden,
                                        negative_slope, False, alpha=alpha))
        for l in range(1, self.num_layers):
            self.gat_layers.append(SHGNConv(num_etypes, num_hidden * self.num_heads, 
                                            num_hidden,
                                            negative_slope, residual, alpha=alpha))
        self.gat_layers.append(SHGNConv(num_etypes, num_hidden * self.num_heads,
                                        num_classes,
                                        negative_slope, residual, None, alpha=alpha))
        self.epsilon = torch.tensor(1e-12, device=cfg.device)

    def forward(self, batch):
        batch = self.encoder(batch)

        if isinstance(batch, HeteroData):
            homo = batch.to_homogeneous()
            x, label, edge_index = homo.x, homo.y, homo.edge_index
            node_type_tensor = homo.node_type
            edge_type_tensor = homo.edge_type
            for idx, node_type in enumerate(batch.node_types):
                if node_type == cfg.dataset.task_entity:
                    node_idx = idx
                    break
        else:
            x, label, edge_index = batch.x, batch.y, batch.edge_index

        # h_list = [fc(x) for fc, x in zip(self.fc_list, x_list)]
        # h = torch.cat(h_list, 0)
        h = x
        res_attn = None
        for l in range(self.num_layers):
            h, res_attn = self.gat_layers[l](h, edge_index, edge_type_tensor, res_attn=res_attn)
            h = h.view(-1, self.gat_layers[l].num_heads * self.gat_layers[l].out_feats)
        logits, _ = self.gat_layers[-1](h, edge_index, edge_type_tensor, res_attn=None)
        logits = logits.mean(1)
        logits = logits / torch.max(torch.norm(logits, p=2, dim=1, keepdim=True), self.epsilon)

        mask = f'{batch.split}_mask'
        if isinstance(batch, HeteroData):
            logits = logits[node_type_tensor == node_idx]
            label = label[node_type_tensor == node_idx]
            task = cfg.dataset.task_entity
            return logits[batch[task][mask]], label[batch[task][mask]]
        else:
            return logits[batch[mask]], label[batch[mask]]