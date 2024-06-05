import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from H2GB.graphgym.config import cfg
from H2GB.graphgym.register import register_node_encoder, register_edge_encoder


@register_node_encoder('Hetero_Label')
class HeteroLabelNodeEncoder(torch.nn.Module):
    """
    The label node encoder for label propagation

    Args:
        emb_dim (int): Output embedding dimension
    """
    def __init__(self, dim_emb, dataset, reshape_x=True):
        super().__init__()
        self.dim_in = cfg.share.dim_in
        self.dim_out = cfg.share.dim_out
        self.metadata = dataset[0].metadata()
        pecfg         = cfg.posenc_Hetero_Label
        dim_hidden    = pecfg.dim_pe

        if not isinstance(dim_emb, dict):
            dim_emb = {node_type: dim_emb for node_type in self.dim_in}
        
        self.linear = nn.ModuleDict()
        node_type = cfg.dataset.task_entity
        self.linear = nn.Linear(
            self.dim_out + 1, dim_emb[node_type])
        
    def forward(self, batch, p=0.7):
        if isinstance(batch, HeteroData):
            node_type = cfg.dataset.task_entity
            if batch.split == 'test':
                # Make use of both training and validation label during testing
                mask = batch[node_type].train_mask | batch[node_type].val_mask
            else:
                # Make use of only training label during training/validation
                mask = batch[node_type].train_mask
            # label = batch[node_type].y[mask].squeeze().clone()
            label = batch[node_type].y.squeeze().clone()
            label[~mask] = self.dim_out
            if batch.split == 'train':
                ratio = p
                n = label.numel() 
                if ratio < 1:       
                    index = torch.arange(n)[torch.rand(n) < ratio]            
                    label[index] = self.dim_out
                else:
                    label[:] = self.dim_out
            label = F.one_hot(label, self.dim_out + 1).type(torch.float32)
            
            # Only changing the x itself can make sure the to_homogeneous() function works well later
            # batch[node_type].x = torch.cat((batch[node_type].x, self.linear(label)), dim=-1)
            batch[node_type].x = batch[node_type].x + self.linear(label)
        else:
            x = batch.x
            batch.x = list(self.linear.values())[0](x)

            

        return batch