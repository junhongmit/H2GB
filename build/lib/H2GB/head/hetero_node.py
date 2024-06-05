import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from H2GB.graphgym.register import register_head
from H2GB.graphgym.config import cfg
from H2GB.graphgym.models.layer import MLP


@register_head('hetero_node')
class HeteroGNNNodeHead(nn.Module):
    r'''Head of Hetero GNN, node prediction
    Auto-adaptive to both homogeneous and heterogeneous data.
    '''
    def __init__(self, dim_in, dim_out, dataset):
        super().__init__()
        self.is_hetero = isinstance(dataset[0], HeteroData)

        self.layer_post_mp = MLP(dim_in,
                                 dim_out,
                                 num_layers=max(cfg.gnn.layers_post_mp, cfg.gt.layers_post_gt),
                                 bias=True)

    def _apply_index(self, batch):
        # mask = '{}_mask'.format(batch.split)
        # return batch.x_dict[cfg.dataset.task_entity][batch[cfg.dataset.task_entity][mask]], \
        #        batch.y_dict[cfg.dataset.task_entity][batch[cfg.dataset.task_entity][mask]]
        task = cfg.dataset.task_entity
        # The front [:batch_size] nodes are the original input nodes in HGTLoader
        if isinstance(batch, HeteroData):
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
            return batch.x[batch[mask]], batch.y[batch[mask]]

    def forward(self, batch):
        if isinstance(batch, HeteroData):
            x = batch[cfg.dataset.task_entity].x
            x = self.layer_post_mp(x)
            batch[cfg.dataset.task_entity].x = x
        else:
            batch.x = self.layer_post_mp(batch.x)

        pred, label = self._apply_index(batch)
        return pred, label
