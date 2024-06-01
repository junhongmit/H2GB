import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData, InMemoryDataset
from torch_geometric.utils import index_to_mask
from tqdm import tqdm


def get_categorical_features(feat_df, cat_cols):
    one_hot_encoded_df = pd.get_dummies(feat_df[cat_cols], columns=cat_cols)
    cat_features = torch.tensor(one_hot_encoded_df.values, dtype=torch.float32)
    
    return cat_features


def get_numerical_features(feat_df, num_cols):
    feat_df[num_cols] = feat_df[num_cols].fillna(0.0)
    num_feats = torch.tensor(feat_df[num_cols].values, dtype=torch.float32)
    
    return num_feats


def get_edge_list(df, node_type_cols):
    # Find number of unique categories for this node type
    unique_entries = df[node_type_cols].drop_duplicates().dropna()
    # Create a map of category to value
    entry_map = {val: idx for idx, val in enumerate(unique_entries)}
    # Create edge list mapping transaction to node type
    edge_list = [[], []]

    for idx, transaction in tqdm(df.iterrows()):
        node_type_val = transaction[node_type_cols]
        # Don't create nodes for NaN values
        if pd.isna(node_type_val):
            continue
        edge_list[0].append(idx)
        edge_list[1].append(entry_map[node_type_val])
    return torch.tensor(edge_list, dtype=torch.long)

def normalize(feature_matrix):
    mean = torch.mean(feature_matrix, axis=0)
    stdev = torch.sqrt(torch.sum((feature_matrix - mean)**2, axis=0)/feature_matrix.shape[0]) + 1e-9
    return mean, stdev, (feature_matrix - mean) / stdev


class CreditCardFraudDetectionDataset(InMemoryDataset):
    url = ""

    def __init__(
        self,
        root,
        transform=None,
        pre_transform=None,
        pre_filter=None,
    ):
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["credit_card_transactions-ibm_v2.csv", "sd254_cards.csv", "sd254_users.csv"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        raise RuntimeError(
            f"Dataset not found. Please download {self.raw_file_names} from "
            f"'{self.url}' and move it to '{self.raw_dir}'"
        )

    def process(self):
        transaction_df = pd.read_csv(self.raw_paths[0])
        card_df = pd.read_csv(self.raw_paths[1])
        user_df = pd.read_csv(self.raw_paths[2])

        print("Processing transactions info...")
        transaction_df['Datetime'] = pd.to_datetime(transaction_df['Year'].astype(str) + '-' +
                                        transaction_df['Month'].astype(str) + '-' +
                                        transaction_df['Day'].astype(str) + ' ' +
                                        transaction_df['Time'],
                                        format='%Y-%m-%d %H:%M').astype('int64') // 10**9
        transaction_df.sort_values(by='Datetime', inplace=True)
        transaction_df.reset_index(inplace=True)

        transaction_df['Errors?'] = transaction_df['Errors?'].fillna("Normal")
        transaction_df['Amount'] = transaction_df['Amount'].str.replace('[$,]', '', regex=True).astype(float)
        transaction_df['Log_Amount'] = np.log10(np.abs(transaction_df['Amount']) + 1e-9)
        transaction_df['Amount_Sign'] = np.sign(transaction_df['Amount'])
        
        transaction_numeric_features = ['Log_Amount', 'Datetime']
        transaction_categorical_features = ['Use Chip', 'Errors?', 'Amount_Sign']
        print("Getting transaction numerical features...")
        transaction_num_feats = get_numerical_features(transaction_df, transaction_numeric_features)
        print("Getting transaction categorical features...")
        transaction_cat_feats = get_categorical_features(transaction_df, transaction_categorical_features)
        transaction_feats = torch.cat((transaction_cat_feats, transaction_num_feats), -1)

        transaction_df['Zip'] = transaction_df['Zip'].fillna(0)
        merchant_cols = ['Merchant Name', 'Merchant City', 'Merchant State', 'Zip', 'MCC']
        unique_merchants_df = transaction_df[merchant_cols].drop_duplicates().reset_index(drop=True)
        
        merchant_tuple_to_id = {tuple(row): idx for idx, row in unique_merchants_df.iterrows()}
        transaction_indices = torch.arange(len(transaction_df), dtype=torch.long)
        merchant_indices = torch.tensor([merchant_tuple_to_id[tuple(x)] for x in transaction_df[merchant_cols].to_numpy()], dtype=torch.long)

        # Create edge index for PyG
        t_m_edge_index = torch.stack([transaction_indices, merchant_indices], dim=0)
        
        # Cards
        print("Processing cards info...")
        card_df['Expires_DateTime'] = pd.to_datetime(card_df['Expires'], format='%m/%Y').astype('int64') // 10**9
        card_df['Acct_Open_DateTime'] = pd.to_datetime(card_df['Acct Open Date'], format='%m/%Y').astype('int64') // 10**9
        card_df['Credit Limit'] = card_df['Credit Limit'].str.replace('[$,]', '', regex=True).astype(float)
        card_df['Log_Credit_Limit'] = np.log10(np.abs(card_df['Credit Limit']) + 1e-9)
        
        card_numeric_features = ['Log_Credit_Limit', 'Expires_DateTime', 'Acct_Open_DateTime', 'Year PIN last Changed']
        card_categorical_features = ['Card Brand', 'Card Type', 'Has Chip', 'Cards Issued', 'Card on Dark Web']
        print("Getting transaction numerical features...")
        card_num_feats = get_numerical_features(card_df, card_numeric_features)
        print("Getting transaction categorical features...")
        card_cat_feats = get_categorical_features(card_df, card_categorical_features)
        card_feats = torch.cat((card_cat_feats, card_num_feats), -1)
        
        card_df['Card_ID'] = card_df['User'].astype(str) + '-' + card_df['CARD INDEX'].astype(str)
        transaction_df['Card_ID'] = transaction_df['User'].astype(str) + '-' + transaction_df['Card'].astype(str)
        card_id_to_index = {cid: idx for idx, cid in enumerate(card_df['Card_ID'].unique())}
        card_df['Card_Node_Index'] = card_df['Card_ID'].map(card_id_to_index)

        source_nodes = transaction_df.index.to_numpy()
        target_nodes = transaction_df['Card_ID'].map(card_id_to_index).to_numpy()

        t_c_edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        
        # Users
        print("Processing users info...")
        user_df['Per Capita Income - Zipcode'] = user_df['Per Capita Income - Zipcode'].str.replace('[$,]', '', regex=True).astype(float)
        user_df['Log_Per Capita Income - Zipcode'] = np.log10(np.abs(user_df['Per Capita Income - Zipcode']) + 1e-9)
        user_df['Yearly Income - Person'] = user_df['Yearly Income - Person'].str.replace('[$,]', '', regex=True).astype(float)
        user_df['Log_Yearly Income - Person'] = np.log10(np.abs(user_df['Yearly Income - Person']) + 1e-9)
        user_df['Total Debt'] = user_df['Total Debt'].str.replace('[$,]', '', regex=True).astype(float)
        user_df['Log_Total Debt'] = np.log10(np.abs(user_df['Total Debt']) + 1e-9)
        
        user_numeric_features = ['Current Age', 'Retirement Age', 'Zipcode', 'Log_Per Capita Income - Zipcode', 'Log_Yearly Income - Person', 'Log_Total Debt', 'FICO Score']
        user_categorical_features = ['Gender', 'State'] # 'City'
        print("Getting transaction numerical features...")
        user_num_feats = get_numerical_features(user_df, user_numeric_features)
        print("Getting transaction categorical features...")
        user_cat_feats = get_categorical_features(user_df, user_categorical_features)
        user_feats = torch.cat((user_cat_feats, user_num_feats), -1)
        
        source_nodes = card_df['User'].to_numpy()
        target_nodes = card_df.index.to_numpy()

        u_c_edge_index = torch.tensor([source_nodes, target_nodes], dtype=torch.long)
        
        data = HeteroData()
        data["transaction"].num_nodes = len(transaction_df)
        data["transaction"].x = normalize(transaction_feats)[2] # transaction_feats
        data["transaction"].y = torch.tensor(transaction_df["Is Fraud?"].map({'No': 0, 'Yes': 1}).astype("long"))
        data["merchant"].num_nodes = len(unique_merchants_df)
        data["card"].num_nodes = len(card_df)
        data["card"].x = normalize(card_feats)[2] # card_feats
        data["user"].num_nodes = len(user_df)
        data["user"].x = normalize(user_feats)[2] # user_feats

        data["transaction", "to", "merchant"].edge_index = t_m_edge_index
        data["transaction", "to", "card"].edge_index = t_c_edge_index
        data["user", "to", "card"].edge_index = u_c_edge_index

        num_nodes = len(transaction_df)
        data['transaction'].train_mask = index_to_mask(torch.arange(0, int(0.7 * num_nodes)), num_nodes)
        data['transaction'].val_mask = index_to_mask(torch.arange(int(0.7 * num_nodes), int(0.85 * num_nodes)), num_nodes)
        data['transaction'].test_mask = index_to_mask(torch.arange(int(0.85 * num_nodes), num_nodes), num_nodes)
        data.validate()
        
        if self.pre_filter is not None:
            data = self.pre_filter(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(data, self.processed_paths[0])