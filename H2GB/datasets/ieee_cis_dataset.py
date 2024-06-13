import os.path as osp
from typing import Callable, List, Optional
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import HeteroData, InMemoryDataset
from tqdm import tqdm
from .utils import download_dataset


DEFAULT_NON_TARGET_NODE_TYPES = [
    "card1",
    "card2",
    "card3",
    "card4",
    "card5",
    "card6",
    "ProductCD",
    "addr1",
    "addr2",
    "P_emaildomain",
    "R_emaildomain",
]
DEFAULT_TARGET_CAT_FEAT_COLS = [
    "M1",
    "M2",
    "M3",
    "M4",
    "M5",
    "M6",
    "M7",
    "M8",
    "M9",
    "DeviceType",
    "DeviceInfo",
    "id_12",
    "id_13",
    "id_14",
    "id_15",
    "id_16",
    "id_17",
    "id_18",
    "id_19",
    "id_20",
    "id_21",
    "id_22",
    "id_23",
    "id_24",
    "id_25",
    "id_26",
    "id_27",
    "id_28",
    "id_29",
    "id_30",
    "id_31",
    "id_32",
    "id_33",
    "id_34",
    "id_35",
    "id_36",
    "id_37",
    "id_38",
]


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


class IeeeCisDataset(InMemoryDataset):
    r"""
    IEEE-CIS-G is a heterogeneous financial network extracted from a tabular
    transaction dataset from `IEEE-CIS Fraud Detection Kaggle Competition
    <https://kaggle.com/competitions/ieee-fraud-detection>`_.
    
    The original dataset contains credit card transactions provided by Vesta
    Corporation, a leading payment service company whose data consists of
    verified transactions. We defined a bipartite graph structure based on
    the available information linked to each credit card transaction, for
    example product code, card information, purchaser and recipient email
    domain, etc. The graph therefore contains 12 diverse entities, including
    the transaction node, and transaction information nodes. It also consists
    of 22 types of relation, connecting the transaction node to each information
    node. 
    Each transaction is associated with a 4823-dimensional feature vector extracting
    from the transaction categorical and numerical features. More description
    of the features can be found in the `Kaggle discussion <https://www.kaggle.com
    /c/ieee-fraud-detection/discussion/101203>`_. Each transaction node is labeled
    with a binary label tagging whether is a fraudulent transaction or not.
    This dataset has around 4\% of fraudulent transactions. We split the dataset
    over the transaction time.


    Args:
        root (str): Root directory where the dataset should be saved.
        non_target_node_types (List[str], optional): Define all other node
            types besides the transaction node. (default: :obj:`None`)
        target_cat_feat_cols (List[str], optional): Define the categorical
            feature columns for the transaction node. (default: :obj:`None`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            every access. (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.HeteroData` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        force_reload (bool, optional): Whether to re-process the dataset.
            (default: :obj:`False`)
    
    """

    url = "https://drive.google.com/file/d/1JBrvglTqeidTgl5ElaRjAgIPCc6udyyL/view?usp=drive_link"
    exclude_cols = ["TransactionID", "isFraud", "TransactionDT"]

    def __init__(
        self,
        root: str,
        non_target_node_types: Optional[List[str]] = None,
        target_cat_feat_cols: Optional[List[str]] = None,
        transform: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        if non_target_node_types is None:
            self.non_target_node_types = DEFAULT_NON_TARGET_NODE_TYPES
        else:
            self.non_target_node_types = non_target_node_types

        if target_cat_feat_cols is None:
            self.target_cat_feat_cols = DEFAULT_TARGET_CAT_FEAT_COLS
        else:
            self.target_cat_feat_cols = target_cat_feat_cols

        assert not set(self.non_target_node_types).intersection(set(self.exclude_cols), set(self.target_cat_feat_cols))

        super().__init__(root, transform, pre_transform)
        self.data = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ["train_transaction.csv", "train_identity.csv", "test_transaction.csv", "test_identity.csv"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        if not all([osp.exists(f) for f in self.raw_paths]):
            download_dataset(self.url, self.root)

    def process(self):
        train_transaction_df = pd.read_csv(self.raw_paths[0])
        train_identity_df = pd.read_csv(self.raw_paths[1])
        test_transaction_df = pd.read_csv(self.raw_paths[2])
        test_identity_df = pd.read_csv(self.raw_paths[3])

        transaction_df = pd.concat([train_transaction_df, test_transaction_df], axis=0)
        identity_df = pd.concat([train_identity_df, test_identity_df], axis=0)
        transaction_df = pd.merge(transaction_df, identity_df, on="TransactionID")
        transaction_df.sort_values("TransactionDT")

        # Remove the transactions where isFraud is NaN
        transaction_df = transaction_df[transaction_df["isFraud"].notna()]

        transaction_numeric_features = [
            column
            for column in transaction_df.columns
            if column not in self.non_target_node_types + self.exclude_cols + self.target_cat_feat_cols
        ]

        transaction_feat_df = transaction_df[transaction_numeric_features + self.target_cat_feat_cols].copy()
        transaction_feat_df = transaction_feat_df.fillna(0)
        transaction_feat_df["TransactionAmt"] = np.log10(np.abs(transaction_feat_df["TransactionAmt"]) + 1e-9)

        print("Getting transaction categorical features...")
        transaction_cat_feats = get_categorical_features(transaction_feat_df, self.target_cat_feat_cols)
        print("Getting transaction numerical features...")
        transaction_num_feats = get_numerical_features(transaction_feat_df, transaction_numeric_features)
        transaction_num_feats = normalize(transaction_num_feats)[2]
        transaction_feats = torch.cat((transaction_cat_feats, transaction_num_feats), -1)

        data = HeteroData()
        data["transaction"].num_nodes = len(transaction_df)
        data["transaction"].x = transaction_feats
        data["transaction"].y = torch.tensor(transaction_df["isFraud"].astype("long"))

        for node_type in self.non_target_node_types:
            print(f"Creating edges for {node_type} nodes...")
            edge_list = get_edge_list(transaction_df, node_type)
            data["transaction", "to", node_type].edge_index = edge_list
            data[node_type].num_nodes = int(edge_list[1].max() + 1)
        data.validate()

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(data, self.processed_paths[0])
