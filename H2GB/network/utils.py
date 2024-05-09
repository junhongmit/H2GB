import torch
import torch.nn as nn
import torch.nn.functional as F

import H2GB.graphgym.register as register
from H2GB.graphgym.config import cfg
from H2GB.graphgym.init import init_weights
from H2GB.graphgym.models.feature_encoder import (edge_encoder_dict,
                                                 node_encoder_dict)
from H2GB.graphgym.models.head import head_dict
from H2GB.graphgym.models.layer import (BatchNorm1dEdge, BatchNorm1dNode,
                                       GeneralLayer, GeneralMultiLayer)

def GTPreNN(dim_in, dim_out, **kwargs):
    """
    Wrapper for NN layer before Graph Transformer

    Args:
        dim_in (int): Input dimension
        dim_out (int): Output dimension
        num_layers (int): Number of layers

    """
    return GeneralMultiLayer('linear',
                             cfg.gt.layers_pre_gt,
                             dim_in,
                             dim_out,
                             dim_inner=dim_out,
                             final_act=True,
                             **kwargs)