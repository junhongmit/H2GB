from collections import namedtuple
import os
import os.path as osp
from tqdm import tqdm
from typing import Tuple
import torch
from torch_geometric.nn import Node2Vec
from torch_geometric.nn.kge import (ComplEx, DistMult, TransE)
from torch_geometric.data import HeteroData

from H2GB.graphgym.config import cfg
from H2GB.loader.embedding.metapath2vec import MetaPath2Vec

NODE2VEC_PT_NAME = 'node2vec'
METAPATH_PT_NAME = 'metapath'

def preprocess_Node2Vec(pe_dir, dataset):
    embedding_dir = osp.join(pe_dir, NODE2VEC_PT_NAME + '_' + str(cfg.posenc_Hetero_Node2Vec.dim_pe) + '.pt')
    temp_path = osp.join(pe_dir, 'temp_' + str(cfg.posenc_Hetero_Node2Vec.dim_pe) + '.pt')
    device = cfg.device

    walk_length = 20
    context_size = 10
    walks_per_node = 30
    num_negative_samples = 8
    batch_size = 128
    lr = 0.1
    if dataset.name in ['ogbn-arxiv', 'arxiv-year']:
        # walk_length = 80
        # context_size = 20
        # walks_per_node = 10
        # num_negative_samples = 1
        # batch_size = 128
        walk_length = 64
        context_size = 7
        walks_per_node = 30
        num_negative_samples = 8
        batch_size = 128
        lr = 0.01
    elif dataset.name in ['cs', 'engineering', 'chemistry']:
        walk_length = 64
        context_size = 7
        walks_per_node = 30
        num_negative_samples = 8
        batch_size = 128
        lr = 0.01
    elif dataset.name == 'ogbn-papers100M':
        walk_length = 20
        context_size = 10
        walks_per_node = 10
        num_negative_samples = 1
        batch_size = 128
        lr = 0.01
        device = 'cpu'
    elif dataset.name == 'ogbn-mag':
        walk_length = 64
        context_size = 7
        walks_per_node = 30
        num_negative_samples = 1
        batch_size = 128
        lr = 0.01
    elif dataset.name == 'pokec':
        walk_length = 80
        context_size = 20
        walks_per_node = 10
        num_negative_samples = 1
        batch_size = 128
        lr = 0.01
    elif dataset.name == 'RCDD':
        walk_length = 80
        context_size = 20
        walks_per_node = 10
        num_negative_samples = 1
        batch_size = 128
        lr = 0.01

    print(f'Preparing Node2Vec encoding for dataset {dataset.name}.')
    models = []
    for idx, data in tqdm(enumerate(dataset)):
        print(data)
        if isinstance(data, HeteroData):
            data = data.to_homogeneous()
        model = Node2Vec(data.edge_index, embedding_dim=cfg.posenc_Hetero_Node2Vec.dim_pe, 
                        walk_length=walk_length, context_size=context_size, walks_per_node=walks_per_node,
                        num_negative_samples=num_negative_samples, num_nodes=data.num_nodes, sparse=True).to(device)
        
        loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=4)
        optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)

        def train(epoch, log_steps=100, eval_steps=100):
            model.train()

            total_loss = 0
            for i, (pos_rw, neg_rw) in enumerate(loader):
                optimizer.zero_grad()
                loss = model.loss(pos_rw.to(device), neg_rw.to(device))
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                if (i + 1) % log_steps == 0:
                    print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                        f'Loss: {total_loss / log_steps:.4f}'))
                    total_loss = 0

                if (i + 1) % eval_steps == 0:
                    acc = test()
                    print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                        f'Acc: {acc:.4f}'))

        @torch.no_grad()
        def test(train_ratio=0.1):
            model.eval()

            mask = (data.y >= 0).squeeze()
            z = model()
            y = data.y.squeeze()
            z = z[mask]
            y = y[mask]
            z = z[:8192]
            y = y[:8192]

            perm = torch.randperm(z.size(0))
            train_perm = perm[:int(z.size(0) * train_ratio)]
            test_perm = perm[int(z.size(0) * train_ratio):]
            if y[train_perm].unique().numel() == 1:
                print('Skip since only contains one class')
                return float('nan')

            return model.test(z[train_perm], y[train_perm], z[test_perm], y[test_perm],
                            max_iter=150)

        for epoch in range(1, 6):
            train(epoch)
            acc = test()
            print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')
            emb = {
                'dim_pe': model.embedding_dim,
                'walk_length': model.walk_length,
                'context_size': model.context_size,
                'walks_per_node': model.walks_per_node,
                'model': model.embedding.weight.data.cpu(),
                'epoch': epoch
            }

            if not osp.exists(pe_dir):
                os.makedirs(pe_dir)
            torch.save(emb, temp_path)
        models.append(model.embedding.weight.data.cpu())
        # if idx >= 10:
        #     break
            
    emb = {
        'dim_pe': model.embedding_dim,
        'walk_length': model.walk_length,
        'context_size': model.context_size,
        'walks_per_node': model.walks_per_node,
        'model': models[0] if len(models) == 1 else models #model.embedding.weight.data.cpu()
    }

    if not osp.exists(pe_dir):
        os.makedirs(pe_dir)
    torch.save(emb, embedding_dir)

def check_Node2Vec(pe_dir):
    return osp.exists(osp.join(pe_dir, NODE2VEC_PT_NAME + '_' + str(cfg.posenc_Hetero_Node2Vec.dim_pe) + '.pt'))

def load_Node2Vec(pe_dir):
    embedding_path = osp.join(pe_dir, NODE2VEC_PT_NAME + '_' + str(cfg.posenc_Hetero_Node2Vec.dim_pe) + '.pt')
    emb = torch.load(embedding_path)
    return emb['model']


def preprocess_Metapath(pe_dir, dataset):
    embedding_path = osp.join(pe_dir, METAPATH_PT_NAME + '_' + str(cfg.posenc_Hetero_Metapath.dim_pe) + '.pt')
    temp_path = osp.join(pe_dir, 'temp_' + str(cfg.posenc_Hetero_Metapath.dim_pe) + '.pt')
    data = dataset[0]
    device = cfg.device
    
    # Each metapath must begin and end on the same node type, which must be the node 
    # type that you intend to obtain embeddings for.
    # When walk_length is greater than the length of a metapath, the metapath is automatically 
    # repeated to fill the length of the walk.
    # The graph should have edge types that connect two adjacent nodes in a metapath. 
    # For example, for Blog Catalog 3, there aren’t any edge types connecting two group nodes,
    # so ["group, "group"] will not be a useful metapath.
    if dataset.name in ['cs', 'engineering', 'chemistry']:
        # metapath = [
        #     ('paper', 'PF_in_L0', 'field'),
        #     ('field', 'rev_PF_in_L0', 'paper'),
        #     ('paper', 'rev_AP_write_first', 'author'), 
        #     ('author', 'in', 'affiliation'), 
        #     ('affiliation', 'rev_in', 'author'), 
        #     ('author', 'AP_write_first', 'paper')
        # ]
        data.edge_index_dict = {
            ('paper', 'PP_cite', 'paper'): data.edge_index_dict[('paper', 'PP_cite', 'paper')],
            ('author', 'AP_write', 'paper'): torch.cat(
                (data.edge_index_dict[('author', 'AP_write_first', 'paper')],
                 data.edge_index_dict[('author', 'AP_write_other', 'paper')],
                 data.edge_index_dict[('author', 'AP_write_last', 'paper')]), dim=-1
            ),
            ('paper', 'rev_AP_write', 'author'): torch.cat(
                (data.edge_index_dict[('paper', 'rev_AP_write_first', 'author')],
                 data.edge_index_dict[('paper', 'rev_AP_write_other', 'author')],
                 data.edge_index_dict[('paper', 'rev_AP_write_last', 'author')]), dim=-1
            ),
            ('field', 'FF_in', 'field'): data.edge_index_dict[('field', 'FF_in', 'field')],
            ('paper', 'PF_in', 'field'): torch.cat(
                (data.edge_index_dict[('paper', 'PF_in_L0', 'field')],
                 data.edge_index_dict[('paper', 'PF_in_L1', 'field')],
                 data.edge_index_dict[('paper', 'PF_in_L2', 'field')],
                 data.edge_index_dict[('paper', 'PF_in_L3', 'field')],
                 data.edge_index_dict[('paper', 'PF_in_L4', 'field')],
                 data.edge_index_dict[('paper', 'PF_in_L5', 'field')],), dim=-1
            ),
            ('field', 'rev_PF_in', 'paper'): torch.cat(
                (data.edge_index_dict[('field', 'rev_PF_in_L0', 'paper')],
                 data.edge_index_dict[('field', 'rev_PF_in_L1', 'paper')],
                 data.edge_index_dict[('field', 'rev_PF_in_L2', 'paper')],
                 data.edge_index_dict[('field', 'rev_PF_in_L3', 'paper')],
                 data.edge_index_dict[('field', 'rev_PF_in_L4', 'paper')],
                 data.edge_index_dict[('field', 'rev_PF_in_L5', 'paper')],), dim=-1
            ),
            ('author', 'in', 'affiliation'): data.edge_index_dict[('author', 'in', 'affiliation')],
            ('affiliation', 'rev_in', 'author'): data.edge_index_dict[('affiliation', 'rev_in', 'author')]
        }
        metapath = [
            ('author', 'AP_write', 'paper'),
            ('paper', 'PF_in', 'field'),
            ('field', 'rev_PF_in', 'paper'),
            ('paper', 'rev_AP_write', 'author'),
            ('author', 'in', 'affiliation'),
            ('affiliation', 'rev_in', 'author'),
            ('author', 'AP_write', 'paper'),
            ('paper', 'PP_cite', 'paper'),
            ('paper', 'rev_AP_write', 'author'),
        ]
        walk_length = 64
        context_size = 7
        walks_per_node = 30
        num_negative_samples = 8
        batch_size = 128
        lr = 0.01
    elif dataset.name in ['tiny', 'small', 'medium']:
        metapath = [
            ('author', 'rev_written_by', 'paper'),
            ('paper', 'topic', 'fos'),
            ('fos', 'rev_topic', 'paper'),
            ('paper', 'written_by', 'author'),
            ('author', 'affiliated_to', 'institute'),
            ('institute', 'rev_affiliated_to', 'author'),
            ('author', 'rev_written_by', 'paper'),
            ('paper', 'cites', 'paper'),
            ('paper', 'written_by', 'author'),
        ]
        walk_length = 64
        context_size = 7
        walks_per_node = 30
        num_negative_samples = 8
        batch_size = 128
        lr = 0.01
    elif dataset.name == 'ogbn-mag':
        metapath = [
            ('author', 'writes', 'paper'),
            ('paper', 'has_topic', 'field_of_study'),
            ('field_of_study', 'rev_has_topic', 'paper'),
            ('paper', 'rev_writes', 'author'),
            ('author', 'affiliated_with', 'institution'),
            ('institution', 'rev_affiliated_with', 'author'),
            ('author', 'writes', 'paper'),
            ('paper', 'cites', 'paper'),
            ('paper', 'rev_writes', 'author'),
        ]
        walk_length = 64
        context_size = 7
        walks_per_node = 30
        num_negative_samples = 8
        batch_size = 128
        lr = 0.01
    elif dataset.name in ['ogbn-arxiv', 'arxiv-year']:
        metapath = [
            ('paper', 'cites', 'paper'),
        ]
        walk_length = 64
        context_size = 7
        walks_per_node = 30
        num_negative_samples = 8
        batch_size = 128
        lr = 0.01
    elif dataset.name == 'ogbn-papers100M':
        metapath = [
            ('paper', 'cites', 'paper'),
        ]
        walk_length = 20
        context_size = 10
        walks_per_node = 10
        num_negative_samples = 1
        batch_size = 128
        lr = 0.01
        # device = 'cpu'
    elif dataset.name == 'Pokec':
        metapath = [
            ('user', 'has_friend', 'user'),
            ('user', 'lists', 'life_style'),
            ('life_style', 'rev_lists', 'user'),
            ('user', 'lists', 'music'),
            ('music', 'rev_lists', 'user'),
            ('user', 'lists', 'cars'),
            ('cars', 'rev_lists', 'user'),
            ('user', 'lists', 'politics'),
            ('politics', 'rev_lists', 'user'),
            ('user', 'lists', 'relationships'),
            ('relationships', 'rev_lists', 'user'),
            ('user', 'lists', 'art_culture'),
            ('art_culture', 'rev_lists', 'user'),
            ('user', 'lists', 'hobbies_interests'),
            ('hobbies_interests', 'rev_lists', 'user'),
            ('user', 'lists', 'science_technologies'),
            ('science_technologies', 'rev_lists', 'user'),
            ('user', 'lists', 'computers_internet'),
            ('computers_internet', 'rev_lists', 'user'),
            ('user', 'lists', 'education'),
            ('education', 'rev_lists', 'user'),
            ('user', 'lists', 'sport'),
            ('sport', 'rev_lists', 'user'),
            ('user', 'lists', 'movies'),
            ('movies', 'rev_lists', 'user'),
            ('user', 'lists', 'travelling'),
            ('travelling', 'rev_lists', 'user'),
            ('user', 'lists', 'health'),
            ('health', 'rev_lists', 'user'),
            ('user', 'lists', 'companies_brands'),
            ('companies_brands', 'rev_lists', 'user')
        ]
        walk_length = 64
        context_size = 7
        walks_per_node = 30
        num_negative_samples = 8
        batch_size = 128
        lr = 0.01
    elif dataset.name == 'pokec':
        metapath = [
            ('user', 'is_friend_with', 'user'),
        ]
        walk_length = 64
        context_size = 7
        walks_per_node = 30
        num_negative_samples = 8
        batch_size = 128
        lr = 0.01
    elif dataset.name == 'snap-patents':
        metapath = [
            ('patent', 'cites', 'patent'),
        ]
        walk_length = 64
        context_size = 7
        walks_per_node = 30
        num_negative_samples = 8
        batch_size = 128
        lr = 0.01
    elif dataset.name == 'ogbn-products' or ('products+h' in dataset.name):
        metapath = [
            ('product', 'buy_with', 'product'),
        ]
        walk_length = 64
        context_size = 7
        walks_per_node = 30
        num_negative_samples = 8
        batch_size = 128
        lr = 0.01
    elif dataset.name == 'RCDD':
        metapath = [
            ('item', 'B_1', 'f'),
            ('f', 'F', 'e'),
            ('e', 'H', 'a'),
            ('a', 'G_1', 'f'),
            ('f', 'C', 'd'),
            ('d', 'C_1', 'f'),
            ('f', 'D', 'c'),
            ('c', 'D_1', 'f'),
            ('f', 'B', 'item'),
            ('item', 'A', 'b'),
            ('b', 'A_1', 'item')
        ]
        walk_length = 64
        context_size = 7
        walks_per_node = 18
        num_negative_samples = 6
        batch_size = 128
        lr = 0.01
    elif dataset.name == 'IEEE-CIS':
        metapath = [
            ('transaction', 'to', 'card1'),
            ('card1', 'rev_to', 'transaction'),
            ('transaction', 'to', 'card2'),
            ('card2', 'rev_to', 'transaction'),
            ('transaction', 'to', 'card3'),
            ('card3', 'rev_to', 'transaction'),
            ('transaction', 'to', 'card4'),
            ('card4', 'rev_to', 'transaction'),
            ('transaction', 'to', 'card5'),
            ('card5', 'rev_to', 'transaction'),
            ('transaction', 'to', 'card6'),
            ('card6', 'rev_to', 'transaction'),
            ('transaction', 'to', 'ProductCD'),
            ('ProductCD', 'rev_to', 'transaction'),
            ('transaction', 'to', 'addr1'),
            ('addr1', 'rev_to', 'transaction'),
            ('transaction', 'to', 'addr2'),
            ('addr2', 'rev_to', 'transaction'),
            ('transaction', 'to', 'P_emaildomain'),
            ('P_emaildomain', 'rev_to', 'transaction'),
            ('transaction', 'to', 'R_emaildomain'),
            ('R_emaildomain', 'rev_to', 'transaction'),
        ]
        walk_length = 64
        context_size = 7
        walks_per_node = 30
        num_negative_samples = 8
        batch_size = 128
        lr = 0.01
    elif dataset.name == 'PDNS':
        metapath = [
            ('domain_node', 'resolves', 'ip_node'),
            ('ip_node', 'rev_resolves', 'domain_node'),
            ('domain_node', 'similar', 'domain_node'),
            ('domain_node', 'apex', 'domain_node')
        ]
        walk_length = 64
        context_size = 7
        walks_per_node = 30
        num_negative_samples = 8
        batch_size = 128
        lr = 0.01
    elif dataset.name in ['Small-HI']:
        data = dataset[2]
        metapath = [
            ('node', 'to', 'node'),
        ]
        walk_length = 64
        context_size = 7
        walks_per_node = 30
        num_negative_samples = 8
        batch_size = 128
        lr = 0.01

    print(f'Preparing MetaPath2Vec encoding for dataset {dataset.name} using pre-defined metapath {metapath}.')
    model = MetaPath2Vec(data.edge_index_dict, embedding_dim=cfg.posenc_Hetero_Metapath.dim_pe,
                     metapath=metapath, walk_length=walk_length, context_size=context_size,
                     walks_per_node=walks_per_node, num_negative_samples=num_negative_samples,
                     num_nodes_dict=data.num_nodes_dict, sparse=True).to(device)
    if osp.exists(temp_path):
        print('Load last checkpoint...')
        emb = torch.load(temp_path)
        print(emb)
        model.embedding = emb['model'].to(device)

    loader = model.loader(batch_size=batch_size, shuffle=True, num_workers=cfg.num_workers)
    optimizer = torch.optim.SparseAdam(list(model.parameters()), lr=lr)

    def train(epoch, log_steps=100, eval_steps=100):
        model.train()

        total_loss = 0
        import time
        last = time.time()
        for i, (pos_rw, neg_rw) in enumerate(loader):
            optimizer.zero_grad()
            # print('Sampling:', time.time() - last)
            # last = time.time()
            loss = model.loss(pos_rw.to(device), neg_rw.to(device))
            # print('Loss:', time.time() - last)
            # last = time.time()
            loss.backward()
            optimizer.step()
            # print('Backward:', time.time() - last)
            # last = time.time()

            total_loss += loss.item()
            if (i + 1) % log_steps == 0:
                print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                    f'Loss: {total_loss / log_steps:.4f}'))
                print(time.time() - last)
                last = time.time()
                total_loss = 0

            if (i + 1) % eval_steps == 0:
                acc = test()
                print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                    f'Acc: {acc:.4f}'))
                
            if dataset.name == 'ogbn-papers100M' and (i + 1) % 10000 == 0:  # Save model every 10000 steps.
                print('Saving temporary...')
                emb = {
                    'dim_pe': model.embedding_dim,
                    'walk_length': model.walk_length,
                    'context_size': model.context_size,
                    'walks_per_node': model.walks_per_node,
                    'model': model.embedding,
                    'start': model.start,
                    'end': model.end
                }
                if not osp.exists(pe_dir):
                    os.makedirs(pe_dir)
                torch.save(emb, temp_path)

    @torch.no_grad()
    def test(train_ratio=0.1):
        node_type = cfg.dataset.task_entity
        if isinstance(node_type, Tuple):
            node_type = node_type[0]
            # Link prediction test is not supported yet
            return 0
        model.eval()

        mask = (data[node_type].y >= 0).squeeze()
        z = model(node_type)
        y = data[node_type].y.squeeze()
        z = z[mask]
        y = y[mask]
        shuffle = torch.randperm(z.size(0))[:8192]
        z = z[shuffle]
        y = y[shuffle]

        perm = torch.randperm(z.size(0))
        train_perm = perm[:int(z.size(0) * train_ratio)]
        test_perm = perm[int(z.size(0) * train_ratio):]

        return model.test(z[train_perm], y[train_perm], z[test_perm], y[test_perm],
                        max_iter=150)

    for epoch in range(1, 6):
        train(epoch)
        acc = test()
        print(f'Epoch: {epoch}, Accuracy: {acc:.4f}')
        # Save temporary checkpoint
        emb = {
            'dim_pe': model.embedding_dim,
            'walk_length': model.walk_length,
            'context_size': model.context_size,
            'walks_per_node': model.walks_per_node,
            'model': model.embedding,
            'start': model.start,
            'end': model.end,
            'epoch': epoch
        }
        if not osp.exists(pe_dir):
            os.makedirs(pe_dir)
        torch.save(emb, temp_path)
            
    emb = {
        'dim_pe': model.embedding_dim,
        'walk_length': model.walk_length,
        'context_size': model.context_size,
        'walks_per_node': model.walks_per_node,
        'model': model.embedding,
        'start': model.start,
        'end': model.end
    }

    if not osp.exists(pe_dir):
        os.makedirs(pe_dir)
    torch.save(emb, embedding_path)

def check_Metapath(pe_dir):
    return osp.exists(osp.join(pe_dir, METAPATH_PT_NAME + '_' + str(cfg.posenc_Hetero_Metapath.dim_pe) + '.pt'))

def load_Metapath(pe_dir):
    embedding_path = osp.join(pe_dir, METAPATH_PT_NAME + '_' + str(cfg.posenc_Hetero_Metapath.dim_pe) + '.pt')
    emb = torch.load(embedding_path, map_location='cpu')
    return emb

def preprocess_KGE(pe_dir, dataset, name):
    model_map = {
        'TransE': TransE,
        'ComplEx': ComplEx,
        'DistMult': DistMult,
    }
    embedding_dir = osp.join(pe_dir, f'{name}_' + str(eval(f'cfg.posenc_Hetero_{name}.dim_pe')) + '.pt')
    temp_path = osp.join(pe_dir, 'temp_' + str(eval(f'cfg.posenc_Hetero_{name}.dim_pe')) + '.pt')

    count = 0
    start, end = {}, {}
    for node_type in dataset[0].node_types:
        start[node_type] = count
        count += dataset[0].num_nodes_dict[node_type]
        end[node_type] = count
    data = dataset[0].to_homogeneous()


    print(f'Preparing {name} encoding for dataset {dataset}.')
    model = model_map[name](
        num_nodes=data.num_nodes,
        num_relations=data.num_edge_types,
        hidden_channels=eval(f'cfg.posenc_Hetero_{name}.dim_pe'),
    ).to(cfg.device)

    batch_size = 60000

    loader = model.loader(
        head_index=data.edge_index[0],
        rel_type=data.edge_type,
        tail_index=data.edge_index[1],
        batch_size=batch_size,
        shuffle=True,
    )

    optimizer_map = {
        'TransE': torch.optim.Adam(model.parameters(), lr=0.01),
        'ComplEx': torch.optim.Adagrad(model.parameters(), lr=0.001, weight_decay=1e-6),
        'DistMult': torch.optim.Adam(model.parameters(), lr=0.0001, weight_decay=1e-6),
    }
    optimizer = optimizer_map[name]


    def train(epoch, log_steps=100):
        model.train()
        total_loss = total_examples = 0
        for i, (head_index, rel_type, tail_index) in enumerate(loader):
            head_index = head_index.to(cfg.device)
            rel_type = rel_type.to(cfg.device)
            tail_index = tail_index.to(cfg.device)
            optimizer.zero_grad()
            loss = model.loss(head_index, rel_type, tail_index)
            loss.backward()
            optimizer.step()
            total_loss += float(loss) * head_index.numel()
            total_examples += head_index.numel()

            if (i + 1) % log_steps == 0:
                print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
                    f'Loss: {total_loss / total_examples:.4f}'))
                total_loss = 0
                total_examples = 0

            # if (i + 1) % eval_steps == 0:
            #     rank, hits = test()
            #     print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
            #         f'Val Mean Rank: {rank:.2f}, Val Hits@10: {hits:.4f}'))
        # return total_loss / total_examples


    @torch.no_grad()
    def test():
        model.eval()
        return model.test(
            head_index=data.edge_index[0][:2000].to(cfg.device),
            rel_type=data.edge_type[:2000].to(cfg.device),
            tail_index=data.edge_index[1][:2000].to(cfg.device),
            batch_size=20000,
            k=10,
        )

    epoch = 5
    for epoch in range(1, epoch+1):
        train(epoch)
        # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
        if epoch % 25 == 0:
            rank, hits = test()
            print(f'Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, Val Hits@10: {hits:.4f}')
        emb = {
            'start': start,
            'end': end,
            'model': model,
            'epoch': epoch
        }
        if not osp.exists(pe_dir):
            os.makedirs(pe_dir)
        torch.save(emb, temp_path)

    rank, hits_at_10 = test()
    print(f'Test Mean Rank: {rank:.2f}, Test Hits@10: {hits_at_10:.4f}')
    
    emb = {
        'start': start,
        'end': end,
        'model': model
    }

    if not osp.exists(pe_dir):
        os.makedirs(pe_dir)
    torch.save(emb, embedding_dir)

def check_KGE(pe_dir, name):
    return osp.exists(osp.join(pe_dir, f'{name}.pt'))

def load_KGE(pe_dir, name, dataset):
    embedding_dir = osp.join(pe_dir, f'{name}.pt')
    emb = torch.load(embedding_dir)

    count = 0
    start, end = {}, {}
    for node_type in dataset[0].num_nodes_dict:
        start[node_type] = count
        count += dataset[0].num_nodes_dict[node_type]
        end[node_type] = count

    result = {}
    for node_type in start:
        result[node_type] = emb['model'].node_emb.weight[start[node_type]:end[node_type]]

    return result

# def preprocess_HTNE(pe_dir, dataset):
#     embedding_path = osp.join(pe_dir, METAPATH_PT_NAME)
#     temp_path = osp.join(pe_dir, 'temp.pt')
#     device = cfg.device

#     epochs = 25
#     lr = 0.001
#     batch_size = 128

#     print(f'Preparing HTNE encoding for dataset {dataset.name}.')
#     if hasattr(dataset, 'dynamicTemporal'):
#         for split in range(len(dataset)):
#             print(f'Working on split {split}.')
#             data = dataset[split]
#             data = HTNE_Dataset(data, neg_size=10, hist_len=5)
#             # sampler = RandomSampler(data)
#             # loader = DataLoader(data, sampler=sampler, batch_size=batch_size)
#             loader = DataLoader(data, batch_size=batch_size, shuffle=True, num_workers=cfg.num_workers)

#             model = HTNE(emb_size=cfg.posenc_Hetero_HTNE.dim_pe, node_dim=data.num_nodes).to(device)
#             optimizer = torch.optim.SparseAdam(model.parameters(), lr=lr, eps=1e-8)
            
#             def train(epoch, log_steps=100):
#                 model.train()
#                 total_loss = total_examples = 0
#                 for i, batch in enumerate(loader):
#                     src = batch['source'].to(device)
#                     tar = batch['target'].to(device)
#                     dat = batch['date'].to(device)
#                     hist_nodes = batch['hist_nodes'].to(device)
#                     hist_times = batch['hist_times'].to(device)
#                     hist_masks = batch['hist_masks'].to(device)
#                     negs = batch['negs'].to(device)
#                     optimizer.zero_grad()
#                     loss = model(src, tar, dat, hist_nodes, hist_times, hist_masks, negs)
#                     loss = loss.sum()
#                     loss.backward()
#                     optimizer.step()
#                     total_loss += loss.item()
#                     total_examples += src.numel()
#                     if (i + 1) % log_steps == 0:
#                         print((f'Epoch: {epoch}, Step: {i + 1:05d}/{len(loader)}, '
#                             f'Loss: {total_loss / total_examples:.4f}'))
#                         total_loss = 0
#                         total_examples = 0

#             for epoch in range(epochs):
#                 train(epoch)
#                 # print(f'Epoch: {epoch:03d}, Loss: {loss:.4f}')
#                 # if epoch % 25 == 0:
#                 #     rank, hits = test()
#                 #     print(f'Epoch: {epoch:03d}, Val Mean Rank: {rank:.2f}, Val Hits@10: {hits:.4f}')
#             model.eval()

#     emb = {
#         'start': start,
#         'end': end,
#         'model': model
#     }

#     if not osp.exists(pe_dir):
#         os.makedirs(pe_dir)
#     torch.save(emb, embedding_dir)