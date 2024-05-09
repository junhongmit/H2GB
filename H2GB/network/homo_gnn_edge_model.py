from matplotlib.cbook import is_math_text
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import (HeteroData, Batch)

from H2GB.graphgym.models import head  # noqa, register module
from H2GB.graphgym import register as register
from H2GB.graphgym.config import cfg
from H2GB.graphgym.models.gnn import GNNPreMP
from H2GB.graphgym.register import register_network
from torch_geometric.nn import (BatchNorm, Linear, GINEConv, GATConv, RGCNConv)
from torch_geometric.nn.models.jumping_knowledge import JumpingKnowledge
from torch_geometric.nn import (GCN, GraphSAGE, GIN, GAT, PNA)
from torch_geometric.utils import mask_to_index
from H2GB.graphgym.models.layer import BatchNorm1dNode
from H2GB.layer.gatedgcn_layer import GatedGCNLayer
from H2GB.layer.gt_layer import GTLayer


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
            NodeEncoder = register.node_encoder_dict[cfg.dataset.node_encoder_name]
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

@register_network('HomoGNNEdgeModel')
class HomoGNNEdgeModel(torch.nn.Module):
    def __init__(self, dim_in, dim_out, data):
        super().__init__()
        self.is_hetero = isinstance(data, HeteroData)
        if self.is_hetero:
            self.metadata = data.metadata()
        self.drop = nn.Dropout(cfg.gnn.dropout)
        self.input_drop = nn.Dropout(cfg.gnn.input_dropout)
        self.activation = register.act_dict[cfg.gnn.act]
        self.layer_norm = cfg.gnn.layer_norm
        self.batch_norm = cfg.gnn.batch_norm
        task_entity = cfg.dataset.task_entity
        if ', ' in task_entity:
            task_entity = tuple(task_entity.split(', '))
            self.embedding = torch.nn.Embedding(data[task_entity[0]].num_nodes, cfg.gnn.dim_inner)

        self.encoder = FeatureEncoder(dim_in, data)
        dim_in = self.encoder.dim_in
        dim_h_total = cfg.gnn.dim_inner

        if cfg.gnn.layers_pre_mp > 0:
            if self.is_hetero:
                self.pre_mp_dict = torch.nn.ModuleDict()
                for node_type in dim_in:
                    self.pre_mp_dict[node_type] = GNNPreMP(
                        dim_in[node_type], cfg.gnn.dim_inner,
                        has_bn=cfg.gnn.batch_norm, has_ln=cfg.gnn.layer_norm
                    )
            else:
                self.pre_mp = GNNPreMP(
                        dim_in, cfg.gnn.dim_inner,
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
                if cfg.gnn.layer_type == 'GINE':
                    mlp = nn.Sequential(
                            nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner), 
                            nn.ReLU(), 
                            nn.Linear(cfg.gnn.dim_inner, cfg.gnn.dim_inner)
                        )
                    conv = GINEConv(mlp, edge_dim=cfg.gnn.dim_inner)
                elif cfg.gnn.layer_type == 'GATE':
                    tmp_out = cfg.gnn.dim_inner // cfg.gnn.attn_heads
                    conv = GATConv(cfg.gnn.dim_inner, tmp_out, cfg.gnn.attn_heads,
                                   concat = True, dropout = self.dropout, add_self_loops = True, edge_dim=cfg.gnn.dim_inner)
                elif cfg.gnn.layer_type == 'RGCN':
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
        self.post_mp = GNNHead(dim_h_total, dim_out, data)


    def forward(self, batch):
        batch = self.encoder(batch)

        if isinstance(batch, HeteroData):
            if cfg.gnn.layers_pre_mp > 0:
                for node_type in batch.node_types:
                    batch[node_type].x = self.pre_mp_dict[node_type](batch[node_type].x) 

            homo = batch.to_homogeneous()
            x, edge_index, edge_attr = homo.x, homo.edge_index, homo.edge_attr
            node_type_tensor = homo.node_type
            edge_type_tensor = homo.edge_type
        else:
            if cfg.gnn.layers_pre_mp > 0:
                batch.x = self.pre_mp(batch.x)
            x, edge_index, edge_attr = batch.x, batch.edge_index, batch.edge_attr
        x = self.input_drop(x)

        # x_dict = {
        #     node_type: x + self.embedding(batch[node_type].node_id) for node_type, x in x_dict.items()
        # } 

        if cfg.gnn.layer_type in ['GCN', 'GraphSAGE', 'GIN', 'PNA']:
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
                    # x = (x + F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr)))) / 2
                    x = self.convs[i](x, edge_index, edge_attr)
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
            for idx, node_type in enumerate(batch.node_types):
                node_mask = node_type_tensor == idx
                batch[node_type].x = x[node_mask]
            for idx, edge_type in enumerate(batch.edge_types):
                edge_mask = edge_type_tensor == idx
                batch[edge_type].edge_attr = edge_attr[edge_mask]
        else:
            batch.x = x
            batch.edge_attr = edge_attr
        return self.post_mp(batch)

@register_network('GINEModel')
class GINe(torch.nn.Module):
    def __init__(self, dim_in, dim_out, data,
                 edge_updates=False, residual=True, 
                edge_dim=None, dropout=0.0, final_dropout=0.5):
        super().__init__()
        # self.train_edge_inds = mask_to_index(data[cfg.dataset.task_entity].train_edge_mask).to(cfg.device)
        # self.val_edge_inds = mask_to_index(data[cfg.dataset.task_entity].val_edge_mask).to(cfg.device)
        # self.test_edge_inds = mask_to_index(data[cfg.dataset.task_entity].test_edge_mask).to(cfg.device)
        # self.train_inds = mask_to_index(data[cfg.dataset.task_entity].train_mask).to(cfg.device)
        # self.val_inds = mask_to_index(data[cfg.dataset.task_entity].val_mask).to(cfg.device)
        # self.test_inds = mask_to_index(data[cfg.dataset.task_entity].test_mask).to(cfg.device)
        self.data = data

        self.n_hidden = cfg.gnn.dim_inner
        self.num_gnn_layers = cfg.gnn.layers_mp
        self.edge_updates = edge_updates
        self.final_dropout = final_dropout

        self.encoder = FeatureEncoder(dim_in, data)
        dim_in = self.encoder.dim_in
        dim_h = cfg.gnn.dim_inner

        self.node_emb = nn.Linear(1, self.n_hidden)
        self.edge_emb = nn.Linear(4, self.n_hidden)

        self.convs = nn.ModuleList()
        self.emlps = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        for _ in range(self.num_gnn_layers):
            conv = GINEConv(nn.Sequential(
                nn.Linear(self.n_hidden, self.n_hidden), 
                nn.ReLU(), 
                nn.Linear(self.n_hidden, self.n_hidden)
                ), edge_dim=self.n_hidden)
            if self.edge_updates: self.emlps.append(nn.Sequential(
                nn.Linear(3 * self.n_hidden, self.n_hidden),
                nn.ReLU(),
                nn.Linear(self.n_hidden, self.n_hidden),
            ))
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.n_hidden))

        GNNHead = register.head_dict[cfg.gnn.head]
        self.post_mp = GNNHead(dim_h, dim_out, data)

        # self.mlp = nn.Sequential(Linear(self.n_hidden*3, 50), nn.ReLU(), nn.Dropout(self.final_dropout),Linear(50, 25), nn.ReLU(), nn.Dropout(self.final_dropout),
        #                       Linear(25, dim_out))

    def forward(self, batch):
        batch = self.encoder(batch)
        
        homo = batch.to_homogeneous()
        x, edge_index, edge_attr = homo.x, homo.edge_index, homo.edge_attr
        node_type_tensor = homo.node_type
        edge_type_tensor = homo.edge_type
        src, dst = edge_index

        # x = self.node_emb(x)
        # edge_attr = self.edge_emb(edge_attr)

        for i in range(self.num_gnn_layers):
            x = (F.relu(self.batch_norms[i](self.convs[i](x, edge_index, edge_attr))))
            if self.edge_updates: 
                edge_attr = edge_attr + self.emlps[i](torch.cat([x[src], x[dst], edge_attr], dim=-1)) / 2

        for idx, node_type in enumerate(batch.node_types):
            node_mask = node_type_tensor == idx
            batch[node_type].x = x[node_mask]
        for idx, edge_type in enumerate(batch.edge_types):
            edge_mask = edge_type_tensor == idx
            batch[edge_type].edge_attr = edge_attr[edge_mask]
        
        # task = cfg.dataset.task_entity
        # x, edge_index, edge_attr = batch[task[0]].x, batch[task].edge_index, batch[task].edge_attr
        # x = x[edge_index.T].reshape(-1, 2 * self.n_hidden).relu()
        # x = torch.cat((x, edge_attr.view(-1, edge_attr.shape[1])), 1)
        # out = x

        # mask = torch.isin(getattr(self, f'{batch.split}_edge_inds')[batch[task].e_id], 
        #                   getattr(self, f'{batch.split}_inds')[batch[task].input_id])

        # return self.mlp(out)[mask], \
        #        batch[task].y[mask]
        return self.post_mp(batch)