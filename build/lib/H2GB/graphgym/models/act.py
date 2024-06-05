import torch.nn as nn

import H2GB.graphgym.register as register
from H2GB.graphgym.config import cfg
from H2GB.graphgym.register import register_act

act_dict = {
    'relu': nn.ReLU(inplace=cfg.mem.inplace),
    'selu': nn.SELU(inplace=cfg.mem.inplace),
    'prelu': nn.PReLU(),
    'elu': nn.ELU(inplace=cfg.mem.inplace),
    'lrelu_01': nn.LeakyReLU(negative_slope=0.1, inplace=cfg.mem.inplace),
    'lrelu_025': nn.LeakyReLU(negative_slope=0.25, inplace=cfg.mem.inplace),
    'lrelu_05': nn.LeakyReLU(negative_slope=0.5, inplace=cfg.mem.inplace),
}

act_dict = {**register.act_dict, **act_dict} 