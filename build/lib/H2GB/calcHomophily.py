import datetime
import os, argparse, logging
from tqdm import tqdm
import torch
from torch_sparse import SparseTensor
from torch_geometric.utils import degree
from torch_geometric.data import HeteroData


def calcHomophily(data: HeteroData, task_entity: str = None, chunk_size: int = 200):
    r"""
    Calculate homophily index on a given heterogeneous graph.
    
    All the 2-hop metapaths in the heterogeneous graph will be extracted,
    then homogeneous homophily index will be calculated over all the
    metapath-induced subgraphs. The calculation is designed to work on
    large-scale graph by using a chunked metapath-induced subgraph
    generation. The adjacency matrix will be chunked into small pieces,
    and multiply with the second adjacency matrix. The homophily index
    is also calculated in a batch manner.

    The edge homophily, class-insenstive homophily and adjusted homophily
    will be returned.

    - In the `"Beyond Homophily in Graph Neural Networks: Current Limitations
      and Effective Designs" <https://arxiv.org/abs/2006.11468>`_ paper, the
      homophily is the fraction of edges in a graph which connects nodes
      that have the same class label:

      .. math::
        \frac{| \{ (v,w) : (v,w) \in \mathcal{E} \wedge y_v = y_w \} | }
        {|\mathcal{E}|}

      That measure is called the *edge homophily ratio*.

    - In the `"Large-Scale Learning on Non-Homophilous Graphs: New Benchmarks
      and Strong Simple Methods" <https://arxiv.org/abs/2110.14446>`_ paper,
      edge homophily is modified to be insensitive to the number of classes
      and size of each class:

      .. math::
        \frac{1}{C-1} \sum_{k=1}^{C} \max \left(0, h_k - \frac{|\mathcal{C}_k|}
        {|\mathcal{V}|} \right),

      where :math:`C` denotes the number of classes, :math:`|\mathcal{C}_k|`
      denotes the number of nodes of class :math:`k`, and :math:`h_k` denotes
      the edge homophily ratio of nodes of class :math:`k`.

      Thus, that measure is called the *class insensitive edge homophily
      ratio*.

    - In the `"Characterizing Graph Datasets for Node Classification: 
      Homophily-Heterophily Dichotomy and Beyond" <https://arxiv.org/abs/2110.14446>`_ 
      paper, adjusted homophily edge homophily adjusted for the expected number of
      edges connecting nodes with the same class label (taking into account the
      number of classes, their sizes, and the distribution of node degrees among
      them).

      .. math::
        \frac{h_{edge} - \sum_{k=1}^C \bar{p}(k)^2}
        {1 - \sum_{k=1}^C \bar{p}(k)^2},

      where :math:`h_{edge}` denotes edge homophily, :math:`C` denotes the
      number of classes, and :math:`\bar{p}(\cdot)` is the empirical
      degree-weighted distribution of classes:
      :math:`\bar{p}(k) = \frac{\sum_{v\,:\,y_v = k} d(v)}{2|E|}`,
      where :math:`d(v)` is the degree of node :math:`v`.

      It has been shown that adjusted homophily satisifes more desirable
      properties than other homophily measures, which makes it appropriate for
      comparing the levels of homophily across datasets with different number
      of classes, different class sizes, andd different degree distributions
      among classes.

      Adjusted homophily can be negative. If adjusted homophily is zero, then
      the edge pattern in the graph is independent of node class labels. If it
      is positive, then the nodes in the graph tend to connect to nodes of the
      same class more often, and if it is negative, than the nodes in the graph
      tend to connect to nodes of different classes more often (compared to the
      null model where edges are independent of node class labels).

    Args:
        data (HeteroData): The input heterogeneous graph.
        task_entity (str): The task entity (target node type) that include the
            class labels, for example :obj:`"paper"` for an academic network.
            (default: :obj:`None`)
        chunk_size (int, optional): The chunk size of the metapath-induced
            subgraph calculation. (default: 200)
    """
    results = []
    def extract_metapath(edge_types, cur, metapath, hop, task_entity=None):
        if hop < 1:
            if task_entity is None:
                results.append(metapath)
            elif cur == task_entity:
                results.append(metapath)
            return
        for edge_type in edge_types:
            src, _, dst = edge_type    
            # if src != dst and src == cur:
            if src == cur:
                extract_metapath(edge_types, dst, metapath + [edge_type], hop - 1, task_entity)
        return results

    metapaths = extract_metapath(data.edge_types, task_entity, [], 2, task_entity)
    print(metapaths)

    label = data[task_entity].y
    device = label.device
    # Some dataset are not fully labeled. We fill in a random class label
    num_classes = label.max() + 1
    mask = label == -1
    label[mask] = torch.randint(0, num_classes, size=(mask.sum(),), device=device)

    h_edges = []
    h_insensitives = []
    h_adjs = []
    for metapath in metapaths:
        src, rel, dst = metapath[0]
        m = data.num_nodes_dict[src]
        n = data.num_nodes_dict[dst]
        k = data.num_nodes_dict[src]
        edge_index_1 = data[metapath[0]].edge_index
        edge_index_2 = data[metapath[1]].edge_index
        adj_1 = SparseTensor(row=edge_index_1[0], col=edge_index_1[1], value=None,
                            sparse_sizes=(m, n)).to(device)
        adj_2 = SparseTensor(row=edge_index_2[0], col=edge_index_2[1], value=None,
                            sparse_sizes=(n, k)).to(device)

        # Chunked multiplication
        num_nodes = label.shape[0]
        num_edges = 0
        num_classes = int(label.max()) + 1
        counts = label.bincount(minlength=num_classes)
        counts = counts.view(1, num_classes)
        proportions = counts / num_nodes

        n_rows = adj_1.size(0)
        nomin, denomin = 0, 0
        nomin = torch.zeros(num_classes, device=device, dtype=torch.float64)
        denomin = torch.zeros(num_classes, device=device, dtype=torch.float64)
        deg = torch.zeros(num_nodes).to(device)
        pbar = tqdm(range(0, n_rows, chunk_size))
        for i in pbar:
            end = min(i + chunk_size, n_rows)
            chunk = adj_1[i:end]  # Get the chunk of rows
            result_chunk = chunk @ adj_2
            
            row, col, _ = result_chunk.coo()
            edge_index = torch.stack([row, col], dim=0)

            num_edges += edge_index.shape[1]
            deg += degree(edge_index[1], num_nodes=num_nodes)

            out = torch.zeros(row.size(0), device=device, dtype=torch.float64)
            out[label[row + i] == label[col]] = 1.
            nomin.scatter_add_(0, label[col], out)
            denomin.scatter_add_(0, label[col], out.new_ones(row.size(0)))

            pbar.set_description(f"Running edge homophily:"
                                f"{round(float(nomin.sum() / denomin.sum()), 3)}")
            

        if float(denomin.sum()) > 0:
            h_edge = float(nomin.sum() / denomin.sum())
            h_insensitive = torch.nan_to_num(nomin / denomin)
            h_insensitive = float((h_insensitive - proportions).clamp_(min=0).sum(dim=-1))
            h_insensitive /= num_classes - 1

            degree_sums = torch.zeros(num_classes).to(device)
            degree_sums.index_add_(dim=0, index=label, source=deg)
            adjust = (degree_sums**2).sum() / float(num_edges ** 2)
            h_adj = (h_edge - adjust) / (1 - adjust)
            print(f"Results for metapath {metapath} is:")
            print(f"Edge Homophily: {h_edge}" )
            print(f"Class insensitive edge homophily: {h_insensitive}")
            print(f"Adjusted homophily: {h_adj}")
            h_edges.append(h_edge)
            h_insensitives.append(h_insensitive)
            h_adjs.append(h_adj)
        else:
            print(f"No existing edge within this metapath subgraph, skip.")

    print("Overall homophily:")
    print(f"Edge Homophily: {sum(h_edges) / len(h_edges)}" )
    print(f"Class insensitive edge homophily: {sum(h_insensitives) / len(h_insensitives)}")
    print(f"Adjusted edge homophily: {sum(h_adjs) / len(h_adjs)}")

    return h_edges, h_insensitives, h_adjs


if __name__ == '__main__':
    import H2GB  # noqa, register custom modules
    from H2GB.graphgym.cmd_args import parse_args
    from H2GB.graphgym.config import (cfg, set_cfg, load_cfg)
    from H2GB.graphgym.loader import create_dataset
    from H2GB.graphgym.utils.device import auto_select_device

    # Load cmd line args
    parser = argparse.ArgumentParser(description='H2GB')
    parser.add_argument('--cfg', dest='cfg_file', type=str, required=True,
                        help='The configuration file path.')
    parser.add_argument('opts', default=None, nargs=argparse.REMAINDER,
                        help='See graphgym/config.py for remaining options.')
    args = parse_args()
    # args = parser.parse_args(["--cfg", "configs/ogbn-mag/ogbn-mag-MLP.yaml", "wandb.use", "False"])
    # args = parser.parse_args(["--cfg", "configs/mag-year/mag-year-MLP.yaml", "wandb.use", "False"])
    # args = parser.parse_args(["--cfg", "configs/oag-cs/oag-cs-MLP.yaml", "wandb.use", "False"])
    # args = parser.parse_args(["--cfg", "configs/IEEE-CIS/IEEE-CIS-MLP.yaml", "wandb.use", "False"])
    # args = parser.parse_args(["--cfg", "configs/Heterophilic_snap-patents/Heterophilic_snap-patents-GCN+SparseEdgeGT+2Hop+Metapath+LP+MS.yaml", "wandb.use", "False"])

    # Load config file
    set_cfg(cfg)
    load_cfg(cfg, args)

    # Set Pytorch environment
    torch.set_num_threads(cfg.num_threads)

    auto_select_device(strategy='greedy')

    dataset = create_dataset()
    data = dataset[0]
    data = data.to(cfg.device)

    calcHomophily(data, cfg.dataset.task_entity)