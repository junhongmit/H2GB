# from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

# import copy
# import torch
# from torch import Tensor

# from torch_geometric.data import HeteroData
# from torch_geometric.loader.base import DataLoaderIterator
# from torch_geometric.loader.utils import filter_node_store_, filter_edge_store_, edge_type_to_str, to_hetero_csc
# from torch_geometric.typing import NodeType
# from torch_geometric.typing import EdgeType, OptTensor


# def filter_hetero_data(
#     data: HeteroData,
#     node_dict: Dict[str, Tensor],
#     row_dict: Dict[str, Tensor],
#     col_dict: Dict[str, Tensor],
#     edge_dict: Dict[str, Tensor],
#     perm_dict: Dict[str, OptTensor],
# ) -> HeteroData:
#     # Filters a heterogeneous data object to only hold nodes in `node` and
#     # edges in `edge` for each node and edge type, respectively:
#     out = copy.copy(data)

#     for node_type in data.node_types:
#         if node_type in node_dict:
#             filter_node_store_(data[node_type], out[node_type],
#                             node_dict[node_type])
#         else:
#             del out[node_type]

#     for edge_type in data.edge_types:
#         edge_type_str = edge_type_to_str(edge_type)
#         if edge_type_str in row_dict:
#             filter_edge_store_(data[edge_type], out[edge_type],
#                             row_dict[edge_type_str], col_dict[edge_type_str],
#                             edge_dict[edge_type_str], perm_dict[edge_type_str])
#         else:
#             del out[edge_type_str]

#     return out

# class HGTLoader(torch.utils.data.DataLoader):
#     r"""The Heterogeneous Graph Sampler from the `"Heterogeneous Graph
#     Transformer" <https://arxiv.org/abs/2003.01332>`_ paper.
#     This loader allows for mini-batch training of GNNs on large-scale graphs
#     where full-batch training is not feasible.

#     :class:`~torch_geometric.data.HGTLoader` tries to (1) keep a similar
#     number of nodes and edges for each type and (2) keep the sampled sub-graph
#     dense to minimize the information loss and reduce the sample variance.

#     Methodically, :class:`~torch_geometric.data.HGTLoader` keeps track of a
#     node budget for each node type, which is then used to determine the
#     sampling probability of a node.
#     In particular, the probability of sampling a node is determined by the
#     number of connections to already sampled nodes and their node degrees.
#     With this, :class:`~torch_geometric.data.HGTLoader` will sample a fixed
#     amount of neighbors for each node type in each iteration, as given by the
#     :obj:`num_samples` argument.

#     Sampled nodes are sorted based on the order in which they were sampled.
#     In particular, the first :obj:`batch_size` nodes represent the set of
#     original mini-batch nodes.

#     .. note::

#         For an example of using :class:`~torch_geometric.data.HGTLoader`, see
#         `examples/hetero/to_hetero_mag.py
#         <https://github.com/pyg-team/pytorch_geometric/blob/master/examples/
#         hetero/to_hetero_mag.py>`_.

#     .. code-block:: python

#         from torch_geometric.loader import HGTLoader
#         from torch_geometric.datasets import OGB_MAG

#         hetero_data = OGB_MAG(path)[0]

#         loader = HGTLoader(
#             hetero_data,
#             # Sample 512 nodes per type and per iteration for 4 iterations
#             num_samples={key: [512] * 4 for key in hetero_data.node_types},
#             # Use a batch size of 128 for sampling training nodes of type paper
#             batch_size=128,
#             input_nodes=('paper', hetero_data['paper'].train_mask),
#         )

#         sampled_hetero_data = next(iter(loader))
#         print(sampled_data.batch_size)
#         >>> 128

#     Args:
#         data (torch_geometric.data.HeteroData): The
#             :class:`~torch_geometric.data.HeteroData` graph data object.
#         num_samples (List[int] or Dict[str, List[int]]): The number of nodes to
#             sample in each iteration and for each node type.
#             If given as a list, will sample the same amount of nodes for each
#             node type.
#         input_nodes (str or Tuple[str, torch.Tensor]): The indices of nodes for
#             which neighbors are sampled to create mini-batches.
#             Needs to be passed as a tuple that holds the node type and
#             corresponding node indices.
#             Node indices need to be either given as a :obj:`torch.LongTensor`
#             or :obj:`torch.BoolTensor`.
#             If node indices are set to :obj:`None`, all nodes of this specific
#             type will be considered.
#         transform (Callable, optional): A function/transform that takes in
#             an a sampled mini-batch and returns a transformed version.
#             (default: :obj:`None`)
#         filter_per_worker (bool, optional): If set to :obj:`True`, will filter
#             the returning data in each worker's subprocess rather than in the
#             main process.
#             Setting this to :obj:`True` is generally not recommended:
#             (1) it may result in too many open file handles,
#             (2) it may slown down data loading,
#             (3) it requires operating on CPU tensors.
#             (default: :obj:`False`)
#         **kwargs (optional): Additional arguments of
#             :class:`torch.utils.data.DataLoader`, such as :obj:`batch_size`,
#             :obj:`shuffle`, :obj:`drop_last` or :obj:`num_workers`.
#     """
#     def __init__(
#         self,
#         data: HeteroData,
#         num_samples: Union[List[int], Dict[NodeType, List[int]]],
#         input_nodes: Union[NodeType, Tuple[NodeType, Optional[Tensor]]],
#         transform: Callable = None,
#         filter_per_worker: bool = False,
#         **kwargs,
#     ):
#         if 'collate_fn' in kwargs:
#             del kwargs['collate_fn']

#         if isinstance(num_samples, (list, tuple)):
#             num_samples = {key: num_samples for key in data.node_types}

#         if isinstance(input_nodes, str):
#             input_nodes = (input_nodes, None)
#         assert isinstance(input_nodes, (list, tuple))
#         assert len(input_nodes) == 2
#         assert isinstance(input_nodes[0], str)
#         if input_nodes[1] is None:
#             index = torch.arange(data[input_nodes[0]].num_nodes)
#             input_nodes = (input_nodes[0], index)
#         elif input_nodes[1].dtype == torch.bool:
#             index = input_nodes[1].nonzero(as_tuple=False).view(-1)
#             input_nodes = (input_nodes[0], index)

#         self.data = data
#         self.num_samples = num_samples
#         self.input_nodes = input_nodes
#         self.num_hops = max([len(v) for v in num_samples.values()])
#         self.transform = transform
#         self.filter_per_worker = filter_per_worker
#         self.sample_fn = torch.ops.torch_sparse.hgt_sample

#         # Convert the graph data into a suitable format for sampling.
#         # NOTE: Since C++ cannot take dictionaries with tuples as key as
#         # input, edge type triplets are converted into single strings.
#         self.colptr_dict, self.row_dict, self.perm_dict = to_hetero_csc(
#             data, device='cpu', share_memory=kwargs.get('num_workers', 0) > 0)

#         super().__init__(input_nodes[1].tolist(), collate_fn=self.collate_fn,
#                          **kwargs)

#     def sample(self, indices: List[int]) -> HeteroData:
#         input_node_dict = {self.input_nodes[0]: torch.tensor(indices)}
#         node_dict, row_dict, col_dict, edge_dict = self.sample_fn(
#             self.colptr_dict,
#             self.row_dict,
#             input_node_dict,
#             self.num_samples,
#             self.num_hops,
#         )
#         return node_dict, row_dict, col_dict, edge_dict, len(indices)

#     def filter_fn(self, out: Any) -> HeteroData:
#         node_dict, row_dict, col_dict, edge_dict, batch_size = out

#         data = filter_hetero_data(self.data, node_dict, row_dict, col_dict,
#                                   edge_dict, self.perm_dict)
#         data[self.input_nodes[0]].batch_size = batch_size

#         return data if self.transform is None else self.transform(data)

#     def collate_fn(self, indices: List[int]) -> Any:
#         out = self.sample(indices)
#         if self.filter_per_worker:
#             # We execute `filter_fn` in the worker process.
#             out = self.filter_fn(out)
#         return out

#     def _get_iterator(self) -> Iterator:
#         if self.filter_per_worker:
#             return super()._get_iterator()
#         # We execute `filter_fn` in the main process.
#         return DataLoaderIterator(super()._get_iterator(), self.filter_fn)

#     def __repr__(self) -> str:
#         return f'{self.__class__.__name__}()'
