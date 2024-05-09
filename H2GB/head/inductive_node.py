import torch.nn as nn
from H2GB.graphgym.config import cfg
from H2GB.graphgym.models.layer import MLP
from H2GB.graphgym.register import register_head


@register_head('inductive_node')
class GNNInductiveNodeHead(nn.Module):
    """
    GNN prediction head for inductive node prediction tasks.

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension. For binary prediction, dim_out=1.
    """

    def __init__(self, dim_in, dim_out, dataset):
        super(GNNInductiveNodeHead, self).__init__()
        self.layer_post_mp = MLP(dim_in, dim_out, num_layers=max(cfg.gnn.layers_post_mp, cfg.gt.layers_post_gt))
                             #has_act=False, has_bias=True)

    def _apply_index(self, batch):
        return batch.x, batch.y

    def forward(self, batch):
        batch = self.layer_post_mp(batch)
        pred, label = self._apply_index(batch)
        return pred, label
