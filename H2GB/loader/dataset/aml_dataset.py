import json, itertools
import os
import os.path as osp
import pandas as pd
import numpy as np
import datatable as dt
from datetime import datetime
from datatable import f,join,sort
from collections import defaultdict
from typing import Callable, List, Optional

import torch

from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import index_to_mask

def z_norm(data):
    std = data.std(0).unsqueeze(0)
    std = torch.where(std == 0, torch.tensor(1, dtype=torch.float32).cpu(), std)
    return (data - data.mean(0).unsqueeze(0)) / std

def format_dataset(inPath):
    r'''
    Turn text attributed dataset into a dataset only contains numbers.
    '''
    outPath = os.path.dirname(inPath) + "/formatted_transactions.csv"

    raw = dt.fread(inPath, columns = dt.str32)

    currency = dict()
    paymentFormat = dict()
    bankAcc = dict()
    account = dict()

    def get_dict_val(name, collection):
        if name in collection:
            val = collection[name]
        else:
            val = len(collection)
            collection[name] = val
        return val

    header = "EdgeID,from_id,to_id,Timestamp,\
    Amount Sent,Sent Currency,Amount Received,Received Currency,\
    Payment Format,Is Laundering\n"

    firstTs = -1

    with open(outPath, 'w') as writer:
        writer.write(header)
        for i in range(raw.nrows):
            datetime_object = datetime.strptime(raw[i,"Timestamp"], '%Y/%m/%d %H:%M')
            ts = datetime_object.timestamp()
            day = datetime_object.day
            month = datetime_object.month
            year = datetime_object.year
            hour = datetime_object.hour
            minute = datetime_object.minute

            if firstTs == -1:
                startTime = datetime(year, month, day)
                firstTs = startTime.timestamp() - 10

            ts = ts - firstTs

            cur1 = get_dict_val(raw[i,"Receiving Currency"], currency)
            cur2 = get_dict_val(raw[i,"Payment Currency"], currency)

            fmt = get_dict_val(raw[i,"Payment Format"], paymentFormat)

            fromAccIdStr = raw[i,"From Bank"] + raw[i,2]
            fromId = get_dict_val(fromAccIdStr, account)

            toAccIdStr = raw[i,"To Bank"] + raw[i,4]
            toId = get_dict_val(toAccIdStr, account)

            amountReceivedOrig = float(raw[i,"Amount Received"])
            amountPaidOrig = float(raw[i,"Amount Paid"])

            isl = int(raw[i,"Is Laundering"])

            line = '%d,%d,%d,%d,%f,%d,%f,%d,%d,%d\n' % \
                        (i,fromId,toId,ts,amountPaidOrig,cur2, amountReceivedOrig,cur1,fmt,isl)

            writer.write(line)

    formatted = dt.fread(outPath)
    formatted = formatted[:,:,sort(3)]

    formatted.to_csv(outPath)

class AMLDataset(InMemoryDataset):
    dataset_sizes = ['Small', 'Medium', 'Large']
    dataset_rates = ['LI', 'HI']
    csv_names = {
        'Small-LI': 'LI-Small_Trans.csv',
        'Small-HI': 'HI-Small_Trans.csv',
        'Medium-LI': 'LI-Medium_Trans.csv',
        'Medium-HI': 'HI-Medium_Trans.csv',
        'Large-LI': 'LI-Large_Trans.csv',
        'Large-HI': 'HI-Large_Trans.csv',
    }

    def __init__(self, root: str, name: str, reverse_mp: bool = False,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None):
        self.name = name # Small-LI
        self.dynamicTemporal = True # Setting definition attribute
        assert self.name.split('-')[0] in self.dataset_sizes
        assert self.name.split('-')[1] in self.dataset_rates
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])
        if not reverse_mp:
            del self._data['node', 'rev_to', 'node']
            del self.slices['node', 'rev_to', 'node']

    @property
    def raw_dir(self) -> str:
        return osp.join(self.root, self.name, 'raw')

    @property
    def processed_dir(self) -> str:
        return osp.join(self.root, self.name, 'processed')

    @property
    def raw_file_names(self) -> List[str]:
        x = ['info.dat', 'node.dat', 'link.dat', 'label.dat', 'label.dat.test']
        return [osp.join(self.names[self.name], f) for f in x]

    @property
    def processed_file_names(self) -> str:
        return 'data.pt'

    # def download(self):
    #     url = self.urls[self.name]
    #     path = download_url(url, self.raw_dir)
    #     extract_zip(path, self.raw_dir)
    #     os.unlink(path)

    def process(self):
        # data = HeteroData()

        # #data['user'].num_nodes = n_users  # Users do not have any features.
        # data['account'].x = model_htne_pre.node_emb.weight.data.detach().cpu()
        # data["account"].node_id = torch.arange(data.num_nodes)

        # data['account', 'transfer_to', 'account'].edge_index = torch.from_numpy(graph.df_data[["src", 'tar']].T.to_numpy())
        # data['account', 'transfer_to', 'account'].edge_label = torch.from_numpy(graph.df_data['label'].to_numpy())
        
        # train_size = 0.6
        
        # edge_indices = edge_indices = np.arange(data.num_edges)
        # edge_labels = data['account', 'transfer_to', 'account'].edge_label
        # train_indices, temp_indices = train_test_split(edge_indices, train_size=train_size, stratify=edge_labels)
        # val_indices, test_indices = train_test_split(temp_indices, test_size=0.75, stratify=edge_labels[temp_indices])
        
        # # Creating masks based on the full set of indices
        # train_mask = np.isin(range(data.num_edges), train_indices)
        # val_mask = np.isin(range(data.num_edges), val_indices)
        # test_mask = np.isin(range(data.num_edges), test_indices)

        # data['account', 'transfer_to', 'account'].train_mask = torch.from_numpy(train_mask)
        # data['account', 'transfer_to', 'account'].val_mask = torch.from_numpy(val_mask)
        # data['account', 'transfer_to', 'account'].test_mask = torch.from_numpy(test_mask)

        format_dataset(osp.join(self.root, self.csv_names[self.name]))
        transaction_file = osp.join(self.root, "formatted_transactions.csv")
        df_edges = pd.read_csv(transaction_file)

        print(f'Available Edge Features: {df_edges.columns.tolist()}')

        df_edges['Timestamp'] = df_edges['Timestamp'] - df_edges['Timestamp'].min()

        max_n_id = df_edges.loc[:, ['from_id', 'to_id']].to_numpy().max() + 1
        df_nodes = pd.DataFrame({'NodeID': np.arange(max_n_id), 'Feature': np.ones(max_n_id)})
        timestamps = torch.Tensor(df_edges['Timestamp'].to_numpy())
        y = torch.LongTensor(df_edges['Is Laundering'].to_numpy())

        print(f"Illicit ratio = {sum(y)} / {len(y)} = {sum(y) / len(y) * 100:.2f}%")
        print(f"Number of nodes (holdings doing transcations) = {df_nodes.shape[0]}")
        print(f"Number of transactions = {df_edges.shape[0]}")

        edge_features = ['Timestamp', 'Amount Received', 'Received Currency', 'Payment Format']
        node_features = ['Feature']

        print(f'Edge features being used: {edge_features}')
        print(f'Node features being used: {node_features} ("Feature" is a placeholder feature of all 1s)')

        x = torch.tensor(df_nodes.loc[:, node_features].to_numpy()).float()
        edge_index = torch.LongTensor(df_edges.loc[:, ['from_id', 'to_id']].to_numpy().T)
        edge_attr = torch.tensor(df_edges.loc[:, edge_features].to_numpy()).float()

        n_days = int(timestamps.max() / (3600 * 24) + 1)
        n_samples = y.shape[0]
        print(f'number of days and transactions in the data: {n_days} days, {n_samples} transactions')

        #data splitting
        daily_irs, weighted_daily_irs, daily_inds, daily_trans = [], [], [], [] #irs = illicit ratios, inds = indices, trans = transactions
        for day in range(n_days):
            l = day * 24 * 3600
            r = (day + 1) * 24 * 3600
            day_inds = torch.where((timestamps >= l) & (timestamps < r))[0]
            daily_irs.append(y[day_inds].float().mean())
            weighted_daily_irs.append(y[day_inds].float().mean() * day_inds.shape[0] / n_samples)
            daily_inds.append(day_inds)
            daily_trans.append(day_inds.shape[0])

        split_per = [0.6, 0.2, 0.2]
        daily_totals = np.array(daily_trans)
        d_ts = daily_totals
        I = list(range(len(d_ts)))
        split_scores = dict()
        for i,j in itertools.combinations(I, 2):
            if j >= i:
                split_totals = [d_ts[:i].sum(), d_ts[i:j].sum(), d_ts[j:].sum()]
                split_totals_sum = np.sum(split_totals)
                split_props = [v/split_totals_sum for v in split_totals]
                split_error = [abs(v-t)/t for v,t in zip(split_props, split_per)]
                score = max(split_error) #- (split_totals_sum/total) + 1
                split_scores[(i,j)] = score
            else:
                continue

        i,j = min(split_scores, key=split_scores.get)
        #split contains a list for each split (train, validation and test) and each list contains the days that are part of the respective split
        split = [list(range(i)), list(range(i, j)), list(range(j, len(daily_totals)))]
        print(f'Calculate split: {split}')

        #Now, we seperate the transactions based on their indices in the timestamp array
        split_inds = {k: [] for k in range(3)}
        for i in range(3):
            for day in split[i]:
                split_inds[i].append(daily_inds[day]) #split_inds contains a list for each split (tr,val,te) which contains the indices of each day seperately
                
        train_inds = torch.cat(split_inds[0])
        val_inds = torch.cat(split_inds[1])
        test_inds = torch.cat(split_inds[2])
        e_train = train_inds
        e_val = torch.cat([train_inds, val_inds])
        e_test = torch.cat([train_inds, val_inds, test_inds])

        
        data_list = []
        for split in ['train', 'val', 'test']:
            inds = eval(f'{split}_inds')
            e_mask = eval(f'e_{split}')

            masked_edge_index = edge_index[:, e_mask]
            masked_edge_attr = z_norm(edge_attr[e_mask])
            masked_y = y[e_mask]
            masked_timestamps = timestamps[e_mask]

            data = HeteroData()
            data['node'].x = x # z_norm(x) will render all x be 0
            data['node', 'to', 'node'].edge_index = masked_edge_index
            data['node', 'to', 'node'].edge_attr = masked_edge_attr
            # We use "y" here so LinkNeighborLoader won't mess up the edge label
            data['node', 'to', 'node'].y = masked_y
            data['node', 'to', 'node'].timestamps = masked_timestamps
            # if args.ports:
            #     #swap the in- and outgoing port numberings for the reverse edges
            #     data['node', 'rev_to', 'node'].edge_attr[:, [-1, -2]] = data['node', 'rev_to', 'node'].edge_attr[:, [-2, -1]]

            data['node', 'rev_to', 'node'].edge_index = masked_edge_index.flipud()
            data['node', 'rev_to', 'node'].edge_attr = masked_edge_attr

            # Define the labels in the training/validation/test sets
            data['node', 'to', 'node'].split_mask = index_to_mask(inds, size=masked_edge_index.shape[1])

            data_list.append(data)
        
        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(self.collate(data_list), self.processed_paths[0])
        
        # edge_attr = z_norm(edge_attr)

        # data = HeteroData()
        # data['node'].x = x # z_norm(x) will render all x be 0
        # data['node', 'to', 'node'].edge_index = edge_index
        # data['node', 'to', 'node'].edge_attr = edge_attr # z_norm(edge_attr)
        # # We use "y" here so LinkNeighborLoader won't mess up the edge label
        # data['node', 'to', 'node'].y = y
        # data['node', 'to', 'node'].timestamps = timestamps
        # # if args.ports:
        # #     #swap the in- and outgoing port numberings for the reverse edges
        # #     data['node', 'rev_to', 'node'].edge_attr[:, [-1, -2]] = data['node', 'rev_to', 'node'].edge_attr[:, [-2, -1]]

        # data['node', 'rev_to', 'node'].edge_index = edge_index.flipud()
        # data['node', 'rev_to', 'node'].edge_attr = edge_attr

        # # Define the edges in the training/validation/test sets
        # data['node', 'to', 'node'].train_edge_mask = index_to_mask(e_tr, size=edge_index.shape[1])
        # data['node', 'to', 'node'].val_edge_mask = index_to_mask(e_val, size=edge_index.shape[1])
        # data['node', 'to', 'node'].test_edge_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)
        # data['node', 'rev_to', 'node'].train_edge_mask = index_to_mask(e_tr, size=edge_index.shape[1])
        # data['node', 'rev_to', 'node'].val_edge_mask = index_to_mask(e_val, size=edge_index.shape[1])
        # data['node', 'rev_to', 'node'].test_edge_mask = torch.ones(edge_index.shape[1], dtype=torch.bool)

        # # Define the labels in the training/validation/test sets
        # data['node', 'to', 'node'].train_mask = index_to_mask(tr_inds, size=edge_index.shape[1])
        # data['node', 'to', 'node'].val_mask = index_to_mask(val_inds, size=edge_index.shape[1])
        # data['node', 'to', 'node'].test_mask = index_to_mask(te_inds, size=edge_index.shape[1])

        # if self.pre_transform is not None:
        #     data = self.pre_transform(data)

        # torch.save(self.collate([data]), self.processed_paths[0])

    def __repr__(self) -> str:
        return f'{self.names[self.name]}()'