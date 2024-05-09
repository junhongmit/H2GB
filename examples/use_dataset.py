import os, argparse
import torch
import logging

import H2GB
from H2GB.graphgym.config import (cfg, set_cfg, load_cfg)
from H2GB.graphgym.loader import create_dataset

# Load cmd line args
parser = argparse.ArgumentParser(description='H2GB')
parser.add_argument('--cfg', dest='cfg_file', type=str, required=True,
                    help='The configuration file path.')
parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                    help='See graphgym/config.py for remaining options.')

# args = parser.parse_args(["--cfg", "configs/ogbn-mag/ogbn-mag-MLP.yaml", "wandb.use", "False"])
args = parser.parse_args(["--cfg", "configs/mag-year/mag-year-MLP.yaml", "wandb.use", "False"])
# args = parser.parse_args(["--cfg", "configs/oag-cs/oag-cs-MLP.yaml", "wandb.use", "False"])
# args = parser.parse_args(["--cfg", "configs/IEEE-CIS/IEEE-CIS-MLP.yaml", "wandb.use", "False"])
# args = parser.parse_args(["--cfg", "configs/Heterophilic_snap-patents/Heterophilic_snap-patents-GCN+SparseEdgeGT+2Hop+Metapath+LP+MS.yaml", "wandb.use", "False"])

# Load config file
set_cfg(cfg)
load_cfg(cfg, args)

# Set Pytorch environment
torch.set_num_threads(cfg.num_threads)

dataset = create_dataset()
print(dataset)

data = dataset[0]
print(data)