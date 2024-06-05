import torch_geometric
from torch_geometric.data import NeighborSampler
#from torch_geometric.loader import NeighborSampler
from torch_sparse.tensor import SparseTensor

if torch_geometric.__version__ < '2.0.0':
    from torch_geometric.data.sampler import Adj, EdgeIndex
else:
    from torch_geometric.loader.neighbor_sampler import Adj, EdgeIndex

# import faulthandler
# faulthandler.enable()


# NOTE: line_profiler makes __builtins__ a dict, not a module...
builtins = __builtins__ if isinstance(__builtins__, dict) \
    else vars(__builtins__)
if 'profile' not in builtins:
    def profile(func):
        return func
profile = profile


# Monkey-patch torch_geometric for this
def Adj__pin_memory(self, *args, **kwargs):
    adj = self.adj_t.pin_memory(*args, **kwargs)
    e_id = self.e_id.pin_memory(*args, **kwargs) if self.e_id is not None \
        else None
    return type(self)(adj, e_id, self.size)


Adj.pin_memory = Adj__pin_memory


def sparse_record_stream(self: SparseTensor, stream):
    row = self._row
    if row is not None:
        row = row.record_stream(stream)
    rowptr = self._rowptr
    if rowptr is not None:
        rowptr = rowptr.record_stream(stream)
    col = self._col.record_stream(stream)
    value = self._value
    if value is not None:
        value = value.record_stream(stream)
    rowcount = self._rowcount
    if rowcount is not None:
        rowcount = rowcount.record_stream(stream)
    colptr = self._colptr
    if colptr is not None:
        colptr = colptr.record_stream(stream)
    colcount = self._colcount
    if colcount is not None:
        colcount = colcount.record_stream(stream)
    csr2csc = self._csr2csc
    if csr2csc is not None:
        csr2csc = csr2csc.record_stream(stream)
    csc2csr = self._csc2csr
    if csc2csr is not None:
        csc2csr = csc2csr.record_stream(stream)


def Adj__record_stream(self, stream):
    sparse_record_stream(self.adj_t.storage, stream)
    if self.e_id is not None:
        self.e_id.record_stream(stream)


Adj.record_stream = Adj__record_stream


# Monkey-patch torch_geometric for this
def EdgeIndex__pin_memory(self, *args, **kwargs):
    edge_index = self.edge_index.pin_memory(*args, **kwargs)
    e_id = self.e_id.pin_memory(*args, **kwargs) if self.e_id is not None \
        else None
    return type(self)(edge_index, e_id, self.size)


EdgeIndex.pin_memory = EdgeIndex__pin_memory


# Monkey-patch NeighborSampler for profiling
NeighborSampler.__init__ = profile(NeighborSampler.__init__)
NeighborSampler.sample = profile(NeighborSampler.sample)
SparseTensor.sample = profile(SparseTensor.sample)
SparseTensor.sample_adj = profile(SparseTensor.sample_adj)
