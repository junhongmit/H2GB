import torch
import torch.nn as nn
import torch_sparse
import numpy as np
import networkx as nx
from torch_geometric.utils import to_dense_adj, to_networkx
import H2GB.graphgym.register as register
from H2GB.graphgym.config import cfg
from H2GB.graphgym.register import register_node_encoder
from H2GB.transform.posenc_stats import get_rw_landing_probs
import multiprocessing
from itertools import zip_longest

import time

# Permutes from (node, node, head) to (head, node, node)
HEAD_NODE_NODE = (2, 0, 1)
# Permutes from (layer, node, node, head) to (layer, head, node, node)
LAYER_HEAD_NODE_NODE = (0, 3, 1, 2)

def shortest_distance_ab_worker(graph, num_spatial_types, num_edge_types, nodes_range):
    if cfg.posenc_Hetero_SDAB.enable_path:
        N = len(graph.nodes)
        L = cfg.gt.layers
        spatial_types = np.empty((L, len(nodes_range), N), dtype=int)
        for i in range(L):
            spatial_types[i, :, :].fill(i * (num_spatial_types + 1) + num_spatial_types)
        shortest_path_types = np.empty((L, len(nodes_range), N, num_spatial_types), dtype=int)
        shortest_path_types.fill(-1)
        for node in nodes_range:
            i = node - nodes_range[0]
            for j, path in nx.single_source_shortest_path(graph, node, cutoff=num_spatial_types).items():
                length = len(path) - 1
                spatial_types[:, i, j] = np.array([length + l * (num_spatial_types + 1) for l in range(L)])
                path_attr = [[(graph.edges[path[k], path[k + 1]]['edge_type'] + l * num_edge_types) for k in range(length)] for l in range(L)] # len(path) * (num_edge_types)
                shortest_path_types[:, i, j, :length] = np.array(path_attr, dtype=int)
        return (spatial_types, shortest_path_types)
    else:
        N = len(graph.nodes)
        L = cfg.gt.layers
        spatial_types = np.empty((L, len(nodes_range), N), dtype=int)
        for i in range(L):
            spatial_types[i, :, :].fill(i * (num_spatial_types + 1) + num_spatial_types)
        for node in nodes_range:
            i = node - nodes_range[0]
            for j, path in nx.single_source_shortest_path(graph, node, cutoff=num_spatial_types).items():
                length = len(path) - 1
                spatial_types[:, i, j] = np.array([length + l * (num_spatial_types + 1) for l in range(L)])
        return spatial_types

# Calculate all-pair shortest distance and assign an learnable attention bias between node i and j
@register_node_encoder('Hetero_SDAB')
class HeteroDistanceAttentionBias(torch.nn.Module):
    def __init__(self, dim_in, dim_emb, data, expand_x=False):
        super().__init__()
        self.metadata = data.metadata()
        self.num_layers = cfg.gt.layers
        self.num_heads = cfg.gt.n_heads
        self.num_spatial_types = cfg.posenc_Hetero_SDAB.num_spatial_types
        self.num_edge_types = len(self.metadata[1])

        self.spatial_encoder = torch.nn.Embedding(self.num_layers * (self.num_spatial_types + 1), self.num_heads)
        self.edge_dis_encoder = torch.nn.Embedding(self.num_layers * self.num_edge_types, self.num_heads)

        self.num_workers = 16
        # self.manager = multiprocessing.Manager()
        # self.shared_graph = self.manager.Namespace()
        # self.shared_graph.graph = nx.DiGraph()
        self.pool = multiprocessing.Pool(processes=self.num_workers)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.spatial_encoder.weight.data.normal_(std=0.02)
        # For debug purpose only
        #self.spatial_encoder.weight.data.copy_(torch.arange(0, cfg.posenc_Hetero_SDPE.num_spatial_types + 1).view(-1, 1).expand(-1, 8))
        self.edge_dis_encoder.weight.data.normal_(std=0.02)

    def forward(self, batch):
        # start = time.time()
        data = batch.to_homogeneous()
        # end = time.time()
        # print(end - start)
        # start = end
        graph: nx.DiGraph = to_networkx(data, node_attrs=['node_type'], edge_attrs=['edge_type'])
        # self.shared_graph.graph = graph

        # end = time.time()
        # print(end - start)
        # start = end
        N = len(graph.nodes)
        L = cfg.gt.layers
        # shortest_paths = nx.shortest_path(graph)
        # shortest_path_lengths = nx.shortest_path_length(graph)
        # end = time.time()
        # print(end - start)
        # start = end

        if cfg.posenc_Hetero_SDAB.enable_path:
            # spatial_types = np.empty((L, N, N), dtype=np.int)
            # spatial_types.fill(self.num_spatial_types)
            # shortest_path_types = np.empty((L, N, N, self.num_spatial_types), dtype=int)
            # shortest_path_types.fill(-1)

            # if hasattr(data, "edge_attr") and data.edge_attr is not None:
            #     shortest_path_types = torch.zeros(N ** 2, distance, dtype=torch.long)
            #     edge_attr = torch.zeros(N, N, dtype=torch.long)
            #     edge_attr[data.edge_index[0], data.edge_index[1]] = data.edge_attr

            # for i, lengths in shortest_path_lengths:
            #     for j, length in lengths.items():
            #         spatial_types[i, j] = length if length <= self.num_spatial_types else self.num_spatial_types

            # for i in graph:
            #     for j, path in nx.single_source_shortest_path(graph, i, cutoff=self.num_spatial_types).items():
            #         length = len(path) - 1
            #         spatial_types[i, j] = length
            #         path_attr = [graph.edges[path[k], path[k + 1]]['edge_type'] for k in range(length)] # len(path) * (num_edge_types)
            #         shortest_path_types[i, j, :len(path) - 1] = np.array(path_attr, dtype=int)

            chunks = []
            pos = 0
            for i in range(self.num_workers - 1):
                chunks.append(list(range(pos, pos + N // self.num_workers)))
                pos += N // self.num_workers
            chunks.append(list(range(pos, N)))
            output = self.pool.starmap(shortest_distance_ab_worker, [(graph, self.num_spatial_types, self.num_edge_types, chunks[i]) for i in range(self.num_workers)])
            spatial_types = np.concatenate([i[0] for i in output], axis=1)
            shortest_path_types = np.concatenate([i[1] for i in output], axis=1)

            # chunks = [iter(shortest_paths.items())] * (len(shortest_paths) // 3)
            # g = [dict(filter(None, v)) for v in zip_longest(*chunks)]
            # pool = multiprocessing.Pool(processes=len(g))
            # print(len(g))
            # output = pool.starmap(shortest_distance_ab_worker, [(N, self.num_spatial_types, g[i]) for i in range(len(g))])
            # spatial_types_2 = torch.vstack([torch.from_numpy(i) for i in output])
            # print(spatial_types_2.device)
            # end = time.time()
            # print(end - start)
            # start = end

            # for i, paths in shortest_paths.items():
            #     for j, path in paths.items():
            #         spatial_types[i, j] = len(path) - 1 if len(path) <= self.num_spatial_types else self.num_spatial_types

            shortest_path_types = torch.from_numpy(shortest_path_types).to(data.x.device)
            mask = shortest_path_types != -1
            path_encoding = torch.zeros((L, N, N, self.num_spatial_types, self.num_heads), device=data.x.device)
            path_encoding[mask, :] = self.edge_dis_encoder(shortest_path_types[mask])
            path_encoding = torch.sum(path_encoding, dim=3) / (torch.sum(mask, dim=-1).unsqueeze(-1) + 1e-6)
            path_encoding = path_encoding.permute(LAYER_HEAD_NODE_NODE)

            # print(spatial_types.device)
            # end = time.time()
            # print(end - start)
            # start = end

            # print(torch.all(torch.eq(spatial_types_2, spatial_types)))

            # for i, paths in shortest_paths.items():
            #     for j, path in paths.items():
            #         if len(path) > self.num_spatial_types:
            #             path = path[:self.num_spatial_types]

            #         assert len(path) >= 1
            #         spatial_types[i, j] = len(path) - 1

            #         # if len(path) > 1 and hasattr(data, "edge_attr") and data.edge_attr is not None:
            #         #     path_attr = [
            #         #         edge_attr[path[k], path[k + 1]] for k in
            #         #         range(len(path) - 1)  # len(path) * (num_edge_types)
            #         #     ]

            #         #     # We map each edge-encoding-distance pair to a distinct value
            #         #     # and so obtain dist * num_edge_features many encodings
            #         #     shortest_path_types[i * N + j, :len(path) - 1] = torch.tensor(
            #         #         path_attr, dtype=torch.long)
                    
            spatial_types = torch.from_numpy(spatial_types).to(data.x.device)
            spatial_encodings = self.spatial_encoder(spatial_types).permute(LAYER_HEAD_NODE_NODE)
            batch.attn_bias = spatial_encodings + path_encoding
        else:
            # spatial_types = np.empty((L, N, N), dtype=np.int)
            # spatial_types.fill(self.num_spatial_types)

            chunks = []
            pos = 0
            for i in range(self.num_workers - 1):
                chunks.append(list(range(pos, pos + N // self.num_workers)))
                pos += N // self.num_workers
            chunks.append(list(range(pos, N)))
            output = self.pool.starmap(shortest_distance_ab_worker, [(graph, self.num_spatial_types, chunks[i]) for i in range(self.num_workers)])
            spatial_types = np.concatenate([i for i in output], axis=1)

            spatial_types = torch.from_numpy(spatial_types).to(data.x.device)
            spatial_encodings = self.spatial_encoder(spatial_types).permute(LAYER_HEAD_NODE_NODE)
            batch.attn_bias = spatial_encodings

        # end = time.time()
        # print('SDAB:', end - start)
        # start = end
        return batch
    




@register_node_encoder('Hetero_kHopAB')
class kHopAttentionBias(torch.nn.Module):
    def __init__(self, dim_in, dim_emb, data, expand_x=False):
        super().__init__()
        self.num_heads = cfg.gt.n_heads
        self.num_layers = cfg.gt.layers
        self.num_spatial_types = cfg.posenc_Hetero_kHopAB.num_spatial_types

        self.spatial_encoder = torch.nn.Embedding(self.num_layers * self.num_spatial_types, self.num_heads)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.spatial_encoder.weight.data.normal_(std=0.02)
        # For debug purpose only
        #self.spatial_encoder.weight.data.copy_(torch.arange(0, cfg.posenc_Hetero_SDPE.num_spatial_types + 1).view(-1, 1).expand(-1, 8))
        # self.edge_dis_encoder.weight.data.normal_(std=0.02)

    def forward(self, batch):
        data = batch.to_homogeneous()
        edge_index = data.edge_index
        N = batch.num_nodes
        H = self.num_heads
        L = self.num_layers

        with torch.no_grad():
            edge_index_list = [edge_index]
            edge_index_k = edge_index
            for i in range(self.num_spatial_types):
                edge_index_k, _ = torch_sparse.spspmm(edge_index_k.cpu(), torch.ones(edge_index_k.shape[1], device='cpu'), 
                                                    edge_index.cpu(), torch.ones(edge_index.shape[1], device='cpu'), 
                                                    N, N, N, True)
                edge_index_k = edge_index_k.to(edge_index.device)
                edge_index_list.append(edge_index_k)

        attn_mask = torch.empty(L, H, N, N, dtype=torch.float32, device=edge_index.device).fill_(-1e9)
        for i in range(self.num_layers):
            for j in range(self.num_spatial_types - 1, -1, -1):
                attn_mask[i, :, edge_index_list[j][1, :], edge_index_list[j][0, :]] = \
                    self.spatial_encoder(torch.tensor([i * self.num_spatial_types + j], dtype=int, device=edge_index.device)).view(-1, 1)

        batch.attn_bias = attn_mask
        return batch
    

def khop_worker(batch, chunks):
    edge_dict = {}
    for edge_types in chunks:
        src_type, rel, next_type = edge_types[0]
        edge_index_k = batch.edge_index_dict[edge_types[0]]
        # print(edge_types[0], edge_index_k.shape, edge_index_k)
        for edge_type in edge_types[1:]:
            next_type = edge_type[0]
            rel = rel + '+' + edge_type[1]
            edge_index = batch.edge_index_dict[edge_type]
            print('Before spspmm')

            # print(edge_type, edge_index.shape, edge_index)
            edge_index_k, _ = torch_sparse.spspmm(edge_index_k.cpu(), torch.ones(edge_index_k.shape[1], device='cpu'), 
                                                edge_index.cpu(), torch.ones(edge_index.shape[1], device='cpu'), 
                                                batch.num_nodes_dict[src_type], batch.num_nodes_dict[next_type], batch.num_nodes_dict[edge_type[2]], True)
            print('After spspmm')
        # print(edge_index_k.shape, edge_index_k)
        edge_dict[(edge_types[0][0], rel, edge_types[-1][2])] = edge_index_k
    return edge_dict

from collections import deque
@register_node_encoder('Hetero_kHopAug')
class kHopAugmentation(torch.nn.Module):
    def __init__(self, dim_in, dim_emb, data, expand_x=False):
        super().__init__()
        self.num_heads = cfg.gt.n_heads
        self.num_layers = cfg.gt.layers
        self.num_spatial_types = cfg.posenc_Hetero_kHopAug.num_spatial_types

        self.spatial_encoder = torch.nn.Embedding(self.num_layers * self.num_spatial_types, self.num_heads)
        self.num_workers = 8
        self.pool = multiprocessing.Pool(processes=self.num_workers)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.spatial_encoder.weight.data.normal_(std=0.02)
        # For debug purpose only
        #self.spatial_encoder.weight.data.copy_(torch.arange(0, cfg.posenc_Hetero_SDPE.num_spatial_types + 1).view(-1, 1).expand(-1, 8))
        # self.edge_dis_encoder.weight.data.normal_(std=0.02)

    def forward(self, batch):
        # src_dict = {}
        # for edge_type in batch.edge_index_dict.keys():
        #     src = edge_type[0]
        #     src_dict[src] = src_dict.get(src, []) + [edge_type]
        src_dict = {'paper': [('paper', 'PP_cite', 'paper'),
                ('paper', 'PF_in_L0', 'field'),
                ('paper', 'PF_in_L1', 'field'),
                ('paper', 'PF_in_L2', 'field'),
                ('paper', 'rev_AP_write_first', 'author')],
                'author': [
                ('author', 'AP_write_first', 'paper'),
                ('author', 'in', 'affiliation')],
                'field': [('field', 'FF_in', 'field'),
                ('field', 'rev_PF_in_L0', 'paper'),
                ('field', 'rev_PF_in_L1', 'paper'),
                ('field', 'rev_PF_in_L2', 'paper')],
                'affiliation': [('affiliation', 'rev_in', 'author')]}

        node_types = list(batch.x_dict.keys())
        queue = deque([(node_type, []) for node_type in node_types])

        result = []
        while queue:
            node_type, hist = queue.popleft()
            if len(hist) == self.num_spatial_types:
                result.append(hist)
                continue
            for edge_type in src_dict[node_type]:
                dst_node_type = edge_type[2]
                queue.append((dst_node_type, hist + [edge_type]))

        # chunks = []
        # pos, size = 0, len(result) // self.num_workers
        # for i in range(self.num_workers):
        #     if i == self.num_workers - 1:
        #         chunks.append(result[pos:])
        #     else:
        #         chunks.append(result[pos : pos+size])
        #     pos += size

        # output = self.pool.starmap(khop_worker, [(batch.detach().cpu(), chunks[i]) for i in range(self.num_workers)])
        # for edge_dict in output:
        #     for key, value in edge_dict.items():
        #         batch[key].edge_index = value.to(torch.device(cfg.device))

        edge_dict = {}
        for edge_types in result:
            src_type, rel, next_type = edge_types[0]
            edge_index_k = batch.edge_index_dict[edge_types[0]]
            # print(edge_types[0], edge_index_k.shape, edge_index_k)
            for edge_type in edge_types[1:]:
                next_type = edge_type[0]
                rel = rel + '+' + edge_type[1]
                edge_index = batch.edge_index_dict[edge_type]
                
                # print(edge_type, edge_index.shape, edge_index)
                edge_index_k, _ = torch_sparse.spspmm(edge_index_k.cpu(), torch.ones(edge_index_k.shape[1], device='cpu'), 
                                                    edge_index.cpu(), torch.ones(edge_index.shape[1], device='cpu'), 
                                                    batch.num_nodes_dict[src_type], batch.num_nodes_dict[next_type], batch.num_nodes_dict[edge_type[2]], True)
            # print(edge_index_k.shape, edge_index_k)
            batch[(edge_types[0][0], rel, edge_types[-1][2])].edge_index = edge_index_k
        
        return batch
    

# def shortest_distance_pe_worker(graph, num_spatial_types, nodes_range):
#     if cfg.posenc_Hetero_SDAB.enable_path:
#         N = len(graph.nodes)
#         spatial_types = np.empty((len(nodes_range), N), dtype=int)
#         spatial_types.fill(num_spatial_types)
#         shortest_path_types = np.empty((len(nodes_range), N, num_spatial_types), dtype=int)
#         shortest_path_types.fill(-1)
#         for node in nodes_range:
#             i = node - nodes_range[0]
#             for j, path in nx.single_source_shortest_path(graph, node, cutoff=num_spatial_types).items():
#                 length = len(path) - 1
#                 spatial_types[i, j] = length
#                 path_attr = [graph.edges[path[k], path[k + 1]]['edge_type'] for k in range(length)] # len(path) * (num_edge_types)
#                 shortest_path_types[i, j, :length] = np.array(path_attr, dtype=int)
#         return (spatial_types, shortest_path_types)
#     else:
#         N = len(graph.nodes)
#         L = cfg.gt.layers
#         spatial_types = np.empty((L, len(nodes_range), N), dtype=int)
#         for i in range(L):
#             spatial_types[i, :, :].fill((i + 1) * num_spatial_types)
#         for node in nodes_range:
#             i = node - nodes_range[0]
#             for j, path in nx.single_source_shortest_path(graph, node, cutoff=num_spatial_types).items():
#                 length = len(path) - 1
#                 spatial_types[:, i, j] = np.array([length + l * num_spatial_types for l in range(L)])
#         return spatial_types
# Calculate all-pair shortest distance and assign an learnable attention bias between node i and j
@register_node_encoder('Hetero_SDPE')
class HeteroDistancePositionEncoding(torch.nn.Module):
    def __init__(self, dim_in, dim_emb, data, expand_x=False):
        super().__init__()
        self.metadata = data.metadata()
        self.expand_x = expand_x
        self.num_spatial_types = cfg.posenc_Hetero_SDPE.num_spatial_types
        dim_pe = cfg.posenc_Hetero_SDPE.dim_pe  # Size of the kernel-based PE embedding

        self.spatial_encoder = torch.nn.Embedding(self.num_spatial_types + 1, dim_pe)

        self.num_workers = 8
        self.pool = multiprocessing.Pool(processes=self.num_workers)
        self.reset_parameters()
        
    def reset_parameters(self):
        self.spatial_encoder.weight.data.normal_(std=0.02)
        # For debug purpose only
        #self.spatial_encoder.weight.data.copy_(torch.arange(0, cfg.posenc_Hetero_SDPE.num_spatial_types + 1).view(-1, 1).expand(-1, 8))

    def forward(self, batch):
        assert 'batch_size' in batch[cfg.dataset.task_entity]

        # start = time.time()
        data = batch.to_homogeneous()
        # end = time.time()
        # print(end - start)
        # start = end
        graph: nx.DiGraph = to_networkx(data, node_attrs=['node_type'], edge_attrs=['edge_type'])
        # end = time.time()
        # print(end - start)
        # start = end
        N = len(graph.nodes)
        # shortest_paths = nx.shortest_path(graph)
        # shortest_path_lengths = nx.shortest_path_length(graph)
        # end = time.time()
        # print(end - start)
        # start = end

        batch_size = batch[cfg.dataset.task_entity].batch_size
        batch.x_dict[cfg.dataset.task_entity][:batch_size]

        spatial_types = np.empty((batch_size, N), dtype=np.int)
        spatial_types.fill(self.num_spatial_types)

        for i in list(graph.nodes)[:batch_size]:
            for j, path in nx.single_source_shortest_path(graph, i, cutoff=self.num_spatial_types).items():
                length = len(path) - 1
                spatial_types[i, j] = length

        # chunks = []
        # pos = 0
        # for i in range(self.num_workers - 1):
        #     chunks.append(list(range(pos, pos + N // self.num_workers)))
        #     pos += N // self.num_workers
        # chunks.append(list(range(pos, N)))
        # output = self.pool.starmap(shortest_distance_pe_worker, [(graph, self.num_spatial_types, chunks[i]) for i in range(self.num_workers)])
        # spatial_types = np.concatenate([i for i in output], axis=1)

        spatial_types = torch.from_numpy(spatial_types).to(data.x.device)
        spatial_encodings = torch.sum(self.spatial_encoder(spatial_types), dim=0)

        # Expand node features if needed
        pos_enc = {}
        for idx, (k, v) in enumerate(batch.num_nodes_dict.items()):
            pos_enc[k] = spatial_encodings[data.node_type == idx]

        for node_type in batch.x_dict:
            if self.expand_x:
                h = self.linear_x[node_type](batch.x_dict[node_type])
            else:
                h = batch.x_dict[node_type]
            # Concatenate final PEs to input embedding
            out[node_type] = torch.cat((h, pos_enc[node_type]), 1)
            batch[node_type].x = out

        return batch