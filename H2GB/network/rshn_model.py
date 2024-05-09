import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import HeteroData

from H2GB.graphgym.models import head  # noqa, register module
from H2GB.graphgym import register as register
from H2GB.graphgym.config import cfg
from H2GB.graphgym.models.gnn import GNNPreMP
from H2GB.graphgym.register import register_network
from torch_geometric.nn import GCNConv, MessagePassing
from torch_geometric.utils import degree
from H2GB.graphgym.models.layer import BatchNorm1dNode

