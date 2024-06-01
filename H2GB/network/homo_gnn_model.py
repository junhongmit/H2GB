from matplotlib.cbook import is_math_text
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import (HeteroData, Batch)


from H2GB.graphgym import register as register
from H2GB.graphgym.config import cfg
from H2GB.graphgym.models.gnn import GNNPreMP
from H2GB.graphgym.register import register_network
from torch_geometric.nn import (GATConv, RGCNConv, APPNP)
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn import (GCN, GraphSAGE, GIN, PNA)
from H2GB.graphgym.models.layer import BatchNorm1dNode
from H2GB.layer.gatedgcn_layer import GatedGCNLayer


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

@register_network('HomoGNNModel')
class HomoGNNModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)
        if self.is_hetero:
            self.metadata = dataset[0].metadata()
        self.drop = nn.Dropout(cfg.gnn.dropout)
        self.input_drop = nn.Dropout(cfg.gnn.input_dropout)
        self.activation = register.act_dict[cfg.gnn.act]
        self.layer_norm = cfg.gnn.layer_norm
        self.batch_norm = cfg.gnn.batch_norm
        # task_entity = cfg.dataset.task_entity
        # self.embedding = torch.nn.Embedding(dataset[0][task_entity[0]].num_nodes, cfg.gnn.dim_inner)

        self.encoder = FeatureEncoder(dim_in, dataset)
        dim_in = self.encoder.dim_in
        dim_h_total = cfg.gnn.dim_inner

        if cfg.gnn.layers_pre_mp > 0:
            # if self.is_hetero:
            #     self.pre_mp_dict = torch.nn.ModuleDict()
            #     for node_type in dim_in:
            #         self.pre_mp_dict[node_type] = GNNPreMP(
            #             dim_in[node_type], cfg.gnn.dim_inner,
            #             has_bn=cfg.gnn.batch_norm, has_ln=cfg.gnn.layer_norm
            #         )
            # else:

            # For homogeneous graph model, we assume the input is already embedded
            # to the same dimension as the dim_inner
            self.pre_mp = GNNPreMP(
                cfg.gnn.dim_inner, cfg.gnn.dim_inner,
                has_bn=cfg.gnn.batch_norm, has_ln=cfg.gnn.layer_norm
            )

        self.model = None
        # Following the PyG implementation whenver possible
        norm = None
        if self.layer_norm or self.batch_norm:
            if self.layer_norm:
                norm = nn.LayerNorm(cfg.gnn.dim_inner)
            elif self.batch_norm:
                norm = nn.BatchNorm1d(cfg.gnn.dim_inner)
        if cfg.gnn.layer_type == 'GCN':
            self.model = GCN(
                in_channels=cfg.gnn.dim_inner, hidden_channels=cfg.gnn.dim_inner, num_layers=cfg.gnn.layers_mp,
                out_channels=cfg.gnn.dim_inner, dropout=cfg.gnn.dropout, act=self.activation, norm=norm,
                jk='cat' if cfg.gnn.jumping_knowledge else None
            )
        elif cfg.gnn.layer_type == 'GraphSAGE':
            self.model = GraphSAGE(
                in_channels=cfg.gnn.dim_inner, hidden_channels=cfg.gnn.dim_inner, num_layers=cfg.gnn.layers_mp,
                out_channels=cfg.gnn.dim_inner, dropout=cfg.gnn.dropout, act=self.activation, norm=norm,
                jk='cat' if cfg.gnn.jumping_knowledge else None
            )
        elif cfg.gnn.layer_type == 'GIN':
            self.model = GIN(
                in_channels=cfg.gnn.dim_inner, hidden_channels=cfg.gnn.dim_inner, num_layers=cfg.gnn.layers_mp,
                out_channels=cfg.gnn.dim_inner, dropout=cfg.gnn.dropout, act=self.activation, norm=norm,
                jk='cat' if cfg.gnn.jumping_knowledge else None
            )
        # Official PyG GAT implementation doesn't reproduce correct results
        # elif cfg.gnn.layer_type == 'GAT':
        #     self.model = GAT(
        #         in_channels=cfg.gnn.dim_inner, hidden_channels=cfg.gnn.dim_inner, num_layers=cfg.gnn.layers_mp,
        #         out_channels=cfg.gnn.dim_inner, heads=cfg.gnn.attn_heads, dropout=cfg.gnn.dropout, act=self.activation, norm=norm,
        #         jk='cat' if cfg.gnn.jumping_knowledge else None
        #     )
        elif cfg.gnn.layer_type == 'APPNP':
            self.model = APPNP(
                K=cfg.gnn.K, alpha=cfg.gnn.alpha, dropout=cfg.gnn.dropout
            )
        elif cfg.gnn.layer_type == 'PNA':
            aggregators = ['mean', 'min', 'max', 'std']
            scalers = ['identity', 'amplification', 'attenuation']
            self.model = PNA(
                in_channels=cfg.gnn.dim_inner, hidden_channels=cfg.gnn.dim_inner, num_layers=cfg.gnn.layers_mp,
                out_channels=cfg.gnn.dim_inner, dropout=cfg.gnn.dropout, act=self.activation, norm=norm,
                aggregators=aggregators, scaler=scalers, 
            )
        else:
            self.convs = nn.ModuleList()
            self.linears = nn.ModuleList()
            if self.layer_norm or self.batch_norm:
                self.norms = nn.ModuleList()
            for i in range(cfg.gnn.layers_mp):
                norm_dim = cfg.gnn.dim_inner
                # if cfg.gnn.layer_type == 'GCN':
                #     # conv = GraphConv(cfg.gnn.dim_inner, cfg.gnn.dim_inner, agg=cfg.gnn.agg, add_self_loops=False) 
                #     conv = GCNConv(cfg.gnn.dim_inner, cfg.gnn.dim_inner, agg=cfg.gnn.agg) 
                # elif cfg.gnn.layer_type == 'SAGE':
                #     conv = SAGEConv(cfg.gnn.dim_inner, cfg.gnn.dim_inner, agg=cfg.gnn.agg)
                # elif cfg.gnn.layer_type == 'GAT':
                #     num_heads = cfg.gnn.attn_heads if i < cfg.gnn.layers_mp - 1 else 1
                #     if i > 0:
                #         conv = GATConv(cfg.gnn.attn_heads * cfg.gnn.dim_inner, cfg.gnn.dim_inner,\
                #                         heads=num_heads, dropout=cfg.gnn.attn_dropout, add_self_loops=False)
                #         norm_dim = num_heads * cfg.gnn.dim_inner
                #     else:
                #         conv = GATConv(cfg.gnn.dim_inner, cfg.gnn.dim_inner, heads=num_heads, add_self_loops=False)
                #         norm_dim = cfg.gnn.attn_heads * cfg.gnn.dim_inner
                # elif cfg.gnn.layer_type == 'HGT':
                #     conv = HGTConv(cfg.gnn.dim_inner, cfg.gnn.dim_inner, self.metadata,
                #             cfg.gnn.attn_heads, group=cfg.gnn.agg)
                # elif cfg.gnn.layer_type == 'GT':
                #     conv = GTLayer(cfg.gt.dim_hidden, self.metadata,
                #         'None', 'Transformer', i,
                #         cfg.gnn.attn_heads, 
                #         layer_norm=False,
                #         batch_norm=False)
                if cfg.gnn.layer_type == 'RGCN':
                    conv = RGCNConv(
                        in_channels=cfg.gnn.dim_inner, out_channels=cfg.gnn.dim_inner, 
                        num_relations=len(self.metadata[1]),
                    )
                elif cfg.gnn.layer_type == 'GatedGCN':
                    conv = GatedGCNLayer(
                        in_dim=cfg.gnn.dim_inner, out_dim=cfg.gnn.dim_inner,
                        dropout=cfg.gnn.dropout, residual=True, act=cfg.gnn.act
                    )
                elif cfg.gnn.layer_type == 'GAT':
                    if i == 0:
                        conv = GATConv(cfg.gnn.dim_inner, cfg.gnn.dim_inner, heads=cfg.gnn.attn_heads, 
                                       concat=True, add_self_loops=True)
                        norm_dim = cfg.gnn.attn_heads * cfg.gnn.dim_inner
                    elif i < cfg.gnn.layers_mp - 1:
                        conv = GATConv(cfg.gnn.attn_heads * cfg.gnn.dim_inner, cfg.gnn.dim_inner,heads=cfg.gnn.attn_heads,
                                       concat=True, add_self_loops=True)
                        norm_dim = cfg.gnn.attn_heads * cfg.gnn.dim_inner
                    else:
                        conv = GATConv(cfg.gnn.attn_heads * cfg.gnn.dim_inner, cfg.gnn.dim_inner,heads=cfg.gnn.attn_heads,
                                       concat=False, add_self_loops=True)
                        norm_dim = cfg.gnn.dim_inner
                else:
                    raise NotImplementedError(f"{cfg.gnn.layer_type} is not implemented!")
                    
                self.convs.append(conv)
                if cfg.gnn.use_linear:
                    if i < cfg.gnn.layers_mp - 1:
                        self.linears.append(nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner, bias=False))
                if self.layer_norm or self.batch_norm:
                    if self.layer_norm:
                        self.norms.append(nn.LayerNorm(norm_dim))
                    elif self.batch_norm:
                        self.norms.append(nn.BatchNorm1d(norm_dim))

            if cfg.gnn.jumping_knowledge:
                self.jk = JumpingKnowledge('cat', cfg.gnn.dim_inner, cfg.gnn.layers_mp)
                if cfg.gnn.layer_type == 'GAT':
                    dim_h_total = cfg.gnn.dim_inner * (cfg.gnn.attn_heads * (cfg.gnn.layers_mp - 1) + 1)
                else:
                    dim_h_total = cfg.gnn.layers_mp * cfg.gnn.dim_inner

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_h_total, dim_out, dataset)


    def forward(self, batch):
        batch = self.encoder(batch)

        if isinstance(batch, HeteroData):
            homo = batch.to_homogeneous()
            x, edge_index = homo.x, homo.edge_index
            x = x.nan_to_num()
            node_type_tensor = homo.node_type
        else:
            x, edge_index = batch.x, batch.edge_index
        
        x = self.input_drop(x)
        if cfg.gnn.layers_pre_mp > 0:
            x = self.pre_mp(x)

        # x_dict = {
        #     node_type: x + self.embedding(batch[node_type].node_id) for node_type, x in x_dict.items()
        # } 

        if cfg.gnn.layer_type in ['GCN', 'GraphSAGE', 'GIN', 'APPNP', 'PNA']:
            x = self.model(x, edge_index)
        else:
            xs = []
            for i in range(cfg.gnn.layers_mp): #[:-1]
                # if cfg.gnn.layer_type == 'GT':
                #     x = self.convs[i](x, edge_index, batch)
                if cfg.gnn.layer_type == 'RGCN':
                    x = self.convs[i](x, edge_index, homo.edge_type)
                elif cfg.gnn.layer_type == 'GatedGCN':
                    out = self.convs[i](Batch(batch=batch,
                                              x=x,
                                              edge_index=batch.edge_index,
                                              edge_attr=batch.edge_attr))
                    x = out.x
                    batch.edge_attr = out.edge_attr
                else:
                    x = self.convs[i](x, edge_index)
                if i < cfg.gnn.layers_mp - 1:
                    if cfg.gnn.use_linear:
                        x = x + self.linears[i](x)
                if self.layer_norm or self.batch_norm:
                    x = self.norms[i](x)
                x = self.drop(self.activation(x))
                if hasattr(self, 'jk'):
                    xs.append(x)

            x = self.jk(xs) if hasattr(self, 'jk') else x

        # Write back
        if isinstance(batch, HeteroData):
            for idx, node_type in enumerate(batch.num_nodes_dict.keys()):
                mask = node_type_tensor==idx
                batch[node_type].x = x[mask]
        else:
            batch.x = x
        return self.post_mp(batch)
