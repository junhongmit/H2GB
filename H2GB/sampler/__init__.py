from os.path import dirname, basename, isfile, join
import glob

modules = glob.glob(join(dirname(__file__), "*.py"))
__all__ = [
    basename(f)[:-3] for f in modules
    if isfile(f) and not f.endswith('__init__.py')
]

from .custom_sampler import (get_NeighborLoader, get_GrashSAINTRandomWalkLoader, 
                             get_HGTloader, get_LINKXLoader, get_NAGloader, get_GOATLoader,
                             get_HINormerLoader)


samplers = [
    'get_NeighborLoader',
    'get_GrashSAINTRandomWalkLoader',
    'get_HGTloader',
    'get_LINKXLoader',
    'get_NAGloader',
    'get_GOATLoader',
    'get_HINormerLoader'
]