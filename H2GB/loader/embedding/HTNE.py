import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset

class HTNE_Dataset(Dataset):
    def __init__(self, data, neg_size=10, hist_len=2, transform=None):
        self.neg_size = neg_size
        self.hist_len = hist_len
        self.transform = transform
        self.NEG_SAMPLING_POWER = 0.75
        self.neg_table_size = int(1e8)

        self.node2hist = dict()
        self.node_set = set()
        self.degrees = dict()

        edge_index = data['node', 'to', 'node'].edge_index
        timestamps = data['node', 'to', 'node'].timestamps

        # If the graph is undirected, you should update the history for both nodes in an edge
        for i in range(edge_index.shape[1]):
            s_node = edge_index[0, i].item()  # source node
            t_node = edge_index[1, i].item()  # target node
            d_time = timestamps[i].item()  # associated timestamp

            self.node_set.update([s_node, t_node])

            if s_node not in self.node2hist:
                self.node2hist[s_node] = list()
            self.node2hist[s_node].append((t_node, d_time))

            # If the graph is undirected, uncomment the following lines
            # if t_node not in node2hist:
            #     node2hist[t_node] = list()
            # node2hist[t_node].append((s_node, d_time))

            if s_node not in self.degrees:
                self.degrees[s_node] = 0
            if t_node not in self.degrees:
                self.degrees[t_node] = 0
            self.degrees[s_node] += 1
            self.degrees[t_node] += 1

        # Sorting historical records by time
        for s, hist in self.node2hist.items():
            self.node2hist[s] = sorted(hist, key=lambda x: x[1])

        # Computing additional stats
        self.num_nodes = len(self.node_set)
        self.data_size = sum(len(hist) for hist in self.node2hist.values())
        
        self.idx2src_id = [src for src, targets in self.node2hist.items() for _ in targets]
        self.idx2tar_id = [tar_id for targets in self.node2hist.values() for tar_id in range(len(targets))]

        self.neg_table = np.zeros((self.neg_table_size,))
        self.init_neg_table()
        print("done initializing")

    def init_neg_table(self):
        tot_sum, cur_sum, por = 0., 0., 0.
        n_id = 0
        for k in range(self.num_nodes):
            if k in self.degrees:
                tot_sum += np.power(
                    self.degrees[k],
                    self.NEG_SAMPLING_POWER
                )
        for k in range(self.neg_table_size):
            if (k + 1.) / self.neg_table_size > por:
                if n_id in self.degrees:
                    cur_sum += np.power(
                        self.degrees[n_id],
                        self.NEG_SAMPLING_POWER
                    )
                por = cur_sum / tot_sum
                n_id += 1
            self.neg_table[k] = n_id - 1

    def negative_sampling(self):
        rand_idx = np.random.randint(
            0,
            self.neg_table_size,
            (self.neg_size,)
        )
        sampled_nodes = self.neg_table[rand_idx]
        return sampled_nodes
    
    def __len__(self):
        return self.data_size

    def __getitem__(self, idx):
        # Edge-based sampling
        src = self.idx2src_id[idx]
        tar_idx = self.idx2tar_id[idx]
        tar, dat, lab = self.node2hist[src][tar_idx]
        if tar_idx - self.hist_len < 0:
            hist = self.node2hist[src][0: tar_idx]
        else:
            # take the most recent samples
            hist = self.node2hist[src][(tar_idx - self.hist_len): tar_idx]
        hist_nodes = [h[0] for h in hist]
        hist_times = [h[1] for h in hist]
        np_h_nodes = np.zeros((self.hist_len,))
        np_h_nodes[:len(hist_nodes)] = hist_nodes
        np_h_times = np.zeros((self.hist_len,))
        np_h_times[:len(hist_times)] = hist_times
        np_h_masks = np.zeros((self.hist_len,))
        np_h_masks[:len(hist_nodes)] = 1.

        neg_nodes = self.negative_sampling()

        return {
            'source': torch.tensor([src], dtype=torch.long),
            'target': torch.tensor([tar], dtype=torch.long),
            'date': torch.tensor([dat], dtype=torch.float),
            'label': torch.tensor([lab], dtype=torch.long),
            'hist_nodes': torch.tensor(np_h_nodes, dtype=torch.long),
            'hist_times': torch.tensor(np_h_times, dtype=torch.float),
            'hist_masks': torch.tensor(np_h_masks, dtype=torch.long),
            'negs': torch.tensor(neg_nodes, dtype=torch.long)
        }

# HTNE from TGEditor (in use)
class HTNE(nn.Module):
	''' For training the stand alone encoder
	'''
	def __init__(
		self,
		emb_size,
		node_dim, # the size of the dataset
	):
		super(HTNE, self).__init__()
		self.node_dim = node_dim
		self.emb_size = emb_size
		self.node_emb = nn.Embedding(self.node_dim, self.emb_size, sparse=True)
		nn.init.uniform_(
			self.node_emb.weight.data,
			- 1.0 / self.emb_size,
			1.0 / self.emb_size
		)
		self.delta = nn.Embedding(self.node_dim, 1, sparse=True)
		nn.init.constant_(self.delta.weight.data, 1)

	def HTNE_loss(self, p_lambda, n_lambda):
		pos_loss = torch.log(p_lambda.sigmoid() + 1e-6).neg()
		neg_loss = torch.log(n_lambda.neg().sigmoid() + 1e-6).sum(dim=1)
		loss =  pos_loss - neg_loss
		return loss

	def forward(
		self,
		s_nodes, # source nodes
		t_nodes, # target ndoes
		t_times, # edge times
		h_nodes, # history nodes
		h_times, # history times
		h_time_mask, # only a small size of his are considered
		n_nodes, # negative sampling nodes
	):
		src_emb = self.node_emb(s_nodes)
		tar_emb = self.node_emb(t_nodes)
		n_emb = self.node_emb(n_nodes)
		h_emb = self.node_emb(h_nodes)
        

		att = F.softmax(((src_emb - h_emb)**2).sum(dim=2).neg(), dim=1) # [batch_size, hist_len]
		p_mu = ((src_emb - tar_emb)**2).sum(dim=2).neg().squeeze(dim=1) # [batch_size, 1]
		p_alpha = ((h_emb - tar_emb)**2).sum(dim=2).neg()  # [batch_size, hist_len]
		delta = self.delta(s_nodes).squeeze(2) # [batch_size, 1]
		d_time = torch.abs(t_times - h_times) # [batch_size, hist_len]
		p_lambda = p_mu + (att * p_alpha * torch.exp(delta * d_time) * h_time_mask).sum(dim=1)
		n_mu = ((src_emb - n_emb)**2).sum(dim=2).neg() # [batch_size, neg_size]
		n_alpha = ((h_emb.unsqueeze(2) - n_emb.unsqueeze(1))**2).sum(dim=3).neg() # [batch_size, hist_len, neg_size]
		n_lambda = n_mu + (att.unsqueeze(2) * n_alpha * (torch.exp(delta * d_time).unsqueeze(2)) * h_time_mask.unsqueeze(2)).sum(dim=1)
		loss = self.HTNE_loss(p_lambda, n_lambda)
		return loss



# Original code from HTNE paper
class HTNE_a:
    def __init__(self, file_path, emb_size=128, neg_size=10, hist_len=2, directed=False,
                 learning_rate=0.01, batch_size=1000, save_step=50, epoch_num=1):
        self.emb_size = emb_size
        self.neg_size = neg_size
        self.hist_len = hist_len

        self.lr = learning_rate
        self.batch = batch_size
        self.save_step = save_step
        self.epochs = epoch_num

        self.data = HTNEDataSet(file_path, neg_size, hist_len, directed)
        self.node_dim = self.data.get_node_dim()

        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.node_emb = Variable(torch.from_numpy(np.random.uniform(
                    -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim, emb_size))).type(
                    FType).cuda(), requires_grad=True)

                self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType).cuda(), requires_grad=True)

                self.att_param = Variable(torch.diag(torch.from_numpy(np.random.uniform(
                    -1. / np.sqrt(emb_size), 1. / np.sqrt(emb_size), (emb_size,))).type(
                    FType).cuda()), requires_grad=True)
        else:
            self.node_emb = Variable(torch.from_numpy(np.random.uniform(
                -1. / np.sqrt(self.node_dim), 1. / np.sqrt(self.node_dim), (self.node_dim, emb_size))).type(
                FType), requires_grad=True)

            self.delta = Variable((torch.zeros(self.node_dim) + 1.).type(FType), requires_grad=True)

            self.att_param = Variable(torch.diag(torch.from_numpy(np.random.uniform(
                -1. / np.sqrt(emb_size), 1. / np.sqrt(emb_size), (emb_size,))).type(
                FType)), requires_grad=True)

        self.opt = SGD(lr=learning_rate, params=[self.node_emb, self.att_param, self.delta])
        self.loss = torch.FloatTensor()

    def forward(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        batch = s_nodes.size()[0]
        s_node_emb = self.node_emb.index_select(0, Variable(s_nodes.view(-1))).view(batch, -1)
        t_node_emb = self.node_emb.index_select(0, Variable(t_nodes.view(-1))).view(batch, -1)
        h_node_emb = self.node_emb.index_select(0, Variable(h_nodes.view(-1))).view(batch, self.hist_len, -1)

        att = softmax(((s_node_emb.unsqueeze(1) - h_node_emb)**2).sum(dim=2).neg(), dim=1)
        p_mu = ((s_node_emb - t_node_emb)**2).sum(dim=1).neg()
        p_alpha = ((h_node_emb - t_node_emb.unsqueeze(1))**2).sum(dim=2).neg()

        delta = self.delta.index_select(0, Variable(s_nodes.view(-1))).unsqueeze(1)
        d_time = torch.abs(t_times.unsqueeze(1) - h_times)  # (batch, hist_len)
        p_lambda = p_mu + (att * p_alpha * torch.exp(delta * Variable(d_time)) * Variable(h_time_mask)).sum(dim=1)
        
        n_node_emb = self.node_emb.index_select(0, Variable(n_nodes.view(-1))).view(batch, self.neg_size, -1)

        n_mu = ((s_node_emb.unsqueeze(1) - n_node_emb)**2).sum(dim=2).neg()
        n_alpha = ((h_node_emb.unsqueeze(2) - n_node_emb.unsqueeze(1))**2).sum(dim=3).neg()


        n_lambda = n_mu + (att.unsqueeze(2) * n_alpha * (torch.exp(delta * Variable(d_time)).unsqueeze(2)) * (Variable(h_time_mask).unsqueeze(2))).sum(dim=1)
        return p_lambda, n_lambda

    def loss_func(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                p_lambdas, n_lambdas = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
                loss = -torch.log(p_lambdas.sigmoid() + 1e-6) - torch.log(
                    n_lambdas.neg().sigmoid() + 1e-6).sum(dim=1)

        else:
            p_lambdas, n_lambdas = self.forward(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times,
                                                h_time_mask)
            loss = -torch.log(torch.sigmoid(p_lambdas) + 1e-6) - torch.log(
                torch.sigmoid(torch.neg(n_lambdas)) + 1e-6).sum(dim=1)
        return loss

    def update(self, s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask):
        if torch.cuda.is_available():
            with torch.cuda.device(DID):
                self.opt.zero_grad()
                loss = self.loss_func(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
                loss = loss.sum()
                self.loss += loss.data
                loss.backward()
                self.opt.step()
        else:
            self.opt.zero_grad()
            loss = self.loss_func(s_nodes, t_nodes, t_times, n_nodes, h_nodes, h_times, h_time_mask)
            loss = loss.sum()
            self.loss += loss.data
            loss.backward()
            self.opt.step()

    def train(self):
        for epoch in xrange(self.epochs):
            self.loss = 0.0
            loader = DataLoader(self.data, batch_size=self.batch,
                                shuffle=True, num_workers=5)
            if epoch % self.save_step == 0 and epoch != 0:
                #torch.save(self, './model/dnrl-dblp-%d.bin' % epoch)
                self.save_node_embeddings('./emb/dblp_htne_attn_%d.emb' % (epoch))

            for i_batch, sample_batched in enumerate(loader):
                if i_batch % 100 == 0 and i_batch != 0:
                    sys.stdout.write('\r' + str(i_batch * self.batch) + '\tloss: ' + str(
                        self.loss.cpu().numpy() / (self.batch * i_batch)) + '\tdelta:' + str(
                        self.delta.mean().cpu().data.numpy()))
                    sys.stdout.flush()

                if torch.cuda.is_available():
                    with torch.cuda.device(DID):
                        self.update(sample_batched['source_node'].type(LType).cuda(),
                                    sample_batched['target_node'].type(LType).cuda(),
                                    sample_batched['target_time'].type(FType).cuda(),
                                    sample_batched['neg_nodes'].type(LType).cuda(),
                                    sample_batched['history_nodes'].type(LType).cuda(),
                                    sample_batched['history_times'].type(FType).cuda(),
                                    sample_batched['history_masks'].type(FType).cuda())
                else:
                    self.update(sample_batched['source_node'].type(LType),
                                sample_batched['target_node'].type(LType),
                                sample_batched['target_time'].type(FType),
                                sample_batched['neg_nodes'].type(LType),
                                sample_batched['history_nodes'].type(LType),
                                sample_batched['history_times'].type(FType),
                                sample_batched['history_masks'].type(FType))

            sys.stdout.write('\repoch ' + str(epoch) + ': avg loss = ' +
                             str(self.loss.cpu().numpy() / len(self.data)) + '\n')
            sys.stdout.flush()

