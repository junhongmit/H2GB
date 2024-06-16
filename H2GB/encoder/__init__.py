from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules
    if isfile(f) and not f.endswith('__init__.py')
]

from .raw_encoder import RawNodeEncoder, RawEdgeEncoder
from .hetero_raw_encoder import HeteroRawNodeEncoder, HeteroRawEdgeEncoder
from .hetero_pos_encoder import (HeteroPENodeEncoder, MetapathNodeEncoder,
                                 Node2VecNodeEncoder, TransENodeEncoder,
                                 ComplExNodeEncoder, DistMultNodeEncoder)
from .hetero_label_encoder import HeteroLabelNodeEncoder
encoders = [
    'RawNodeEncoder',
    'RawEdgeEncoder',
    'HeteroRawNodeEncoder',
    'HeteroRawEdgeEncoder',
    'HeteroPENodeEncoder',
    'MetapathNodeEncoder',
    'Node2VecNodeEncoder',
    'TransENodeEncoder',
    'ComplExNodeEncoder',
    'DistMultNodeEncoder',
    'HeteroLabelNodeEncoder',
]