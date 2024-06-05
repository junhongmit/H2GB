import torch.nn as nn

from H2GB.graphgym.config import cfg
from H2GB.graphgym.register import register_loss


# def loss_example(pred, true):
#     if cfg.model.loss_fun == 'smoothl1':
#         l1_loss = nn.SmoothL1Loss()
#         loss = l1_loss(pred, true)
#         return loss, pred


# register_loss('smoothl1', loss_example)
