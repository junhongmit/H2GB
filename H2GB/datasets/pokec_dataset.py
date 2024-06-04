import re
import os.path as osp
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from torch_geometric.data import (
    HeteroData,
    InMemoryDataset,
    download_url,
    extract_zip,
)
from torch_geometric.utils import index_to_mask

def get_categorical_features(feat_df, cat_cols):
    one_hot_encoded_df = pd.get_dummies(feat_df[cat_cols], columns=cat_cols)
    cat_features = torch.tensor(one_hot_encoded_df.values, dtype=torch.float32)
    
    return cat_features

def get_numerical_features(feat_df, num_cols):
    feat_df[num_cols] = feat_df[num_cols].fillna(0.0)
    num_feats = torch.tensor(feat_df[num_cols].values, dtype=torch.float32)
    
    return num_feats

def normalize(feature_matrix):
    mean = torch.mean(feature_matrix, axis=0)
    stdev = torch.sqrt(torch.sum((feature_matrix - mean)**2, axis=0)/feature_matrix.shape[0]) + 1e-9
    return mean, stdev, (feature_matrix - mean) / stdev

class PokecDataset(InMemoryDataset):
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

    urls = [
        "https://snap.stanford.edu/data/soc-pokec-profiles.txt.gz",
        "https://snap.stanford.edu/data/soc-pokec-relationships.txt.gz"
        ]
    
    node_fields = [
        "public",
        "completion_percentage",
        "gender",
        "region",
        "last_login",
        "registration",
        "AGE",
        "body",
        "I_am_working_in_field",
        "spoken_languages",
        "hobbies",
        "I_most_enjoy_good_food",
        "pets",
        "body_type",
        "my_eyesight",
        "eye_color",
        "hair_color",
        "hair_type",
        "completed_level_of_education",
        "favourite_color",
        "relation_to_smoking",
        "relation_to_alcohol",
        "sign_in_zodiac",
        "on_pokec_i_am_looking_for",
        "love_is_for_me",
        "relation_to_casual_sex",
        "my_partner_should_be",
        "marital_status",
        "children",
        "relation_to_children",
        "I_like_movies",
        "I_like_watching_movie",
        "I_like_music",
        "I_mostly_like_listening_to_music",
        "the_idea_of_good_evening",
        "I_like_specialties_from_kitchen",
        "fun",
        "I_am_going_to_concerts",
        "my_active_sports",
        "my_passive_sports",
        "profession",
        "I_like_books",
        "life_style",
        "music",
        "cars",
        "politics",
        "relationships",
        "art_culture",
        "hobbies_interests",
        "science_technologies",
        "computers_internet",
        "education",
        "sport",
        "movies",
        "travelling",
        "health",
        "companies_brands",
        "more",
        ""
    ]
    
    node_features = ["body",
        "I_am_working_in_field",
        "spoken_languages",
        "hobbies",
        "I_most_enjoy_good_food",
        "pets",
        "body_type",
        "my_eyesight",
        "eye_color",
        "hair_color",
        "hair_type",
        "completed_level_of_education",
        "favourite_color",
        "relation_to_smoking",
        "relation_to_alcohol",
        "sign_in_zodiac",
        "on_pokec_i_am_looking_for",
        "love_is_for_me",
        "relation_to_casual_sex",
        "my_partner_should_be",
        "marital_status",
        "children",
        "relation_to_children",
        "I_like_movies",
        "I_like_watching_movie",
        "I_like_music",
        "I_mostly_like_listening_to_music",
        "the_idea_of_good_evening",
        "I_like_specialties_from_kitchen",
        "fun",
        "I_am_going_to_concerts",
        "my_active_sports",
        "my_passive_sports",
        "profession",
        "I_like_books"
    ]
    
    edge_features = ["life_style",
        "music",
        "cars",
        "politics",
        "relationships",
        "art_culture",
        "hobbies_interests",
        "science_technologies",
        "computers_internet",
        "education",
        "sport",
        "movies",
        "travelling",
        "health",
        "companies_brands"
    ]

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
    def raw_dir(self) -> str:
        return osp.join(self.root, 'raw')

    @property
    def raw_file_names(self):
        return ["soc-pokec-profiles.txt.gz", "soc-pokec-relationships.txt.gz"]

    @property
    def processed_file_names(self):
        return ["data.pt"]

    def download(self):
        if not all([osp.exists(f) for f in self.raw_paths]):
            for url in self.urls:
                path = download_url(url, self.raw_dir)
                extract_zip(path, self.raw_dir)

    def process(self):
        dfn = pd.read_csv(self.raw_paths[0], sep = "\t", names = self.node_fields, nrows = None)
        dfe = pd.read_csv(self.raw_paths[1], sep = "\t", names = ["source", "target"], nrows = None)
        dfn = dfn.sort_index()
        for time_field in ['registration', 'last_login']:
            dfn[time_field] = pd.to_datetime(dfn[time_field], errors='coerce')
            fallback_date = pd.Timestamp('2000-01-01 00:00:00.0')
            dfn[time_field] = dfn[time_field].fillna(fallback_date)
            dfn[time_field] = dfn[time_field].astype('int64') // 10**9
        dfn['AGE'] = dfn['AGE'].replace(0, 23)
        dfn['main_area'] = dfn['region'].str.split(',').str[0]
        dfn['main_area'] = dfn['main_area'].fillna('nan')
        
        entry_dict = {}
        for column in self.edge_features:
            entries = dfn[column].unique()
            entry_set = set()
            for line in entries:
                if line == line: # Check for nan
                    hrefs = re.findall(r'href="/klub/([^"]+)"', line)
                    entry_set |= set(hrefs)
            entry_dict[column] = {entry: idx for idx, entry in enumerate(entry_set)}

        def extract_ids(entry_name, html_data):
            hrefs = re.findall(r'href="/klub/([^"]+)"', html_data)
            return [entry_dict[entry_name].get(name) for name in hrefs if name in entry_dict[entry_name]]

        edge_index_dict = {}
        for column in self.edge_features:
            print(f'Processing {column} ...')
            edge_index = []
            users_data = []
            for idx, line in enumerate(tqdm(dfn[column])):
                if line == line: # Check for nan
                    ids = extract_ids(column, line)
                    users_data.append((idx, ids))

            user_ids = [user_id for user_id, movie_ids in users_data for _ in range(len(movie_ids))]
            movie_ids = [movie_id for _, movie_ids in users_data for movie_id in movie_ids]

            # Convert lists to tensor
            user_tensor = torch.tensor(user_ids, dtype=torch.long)
            movie_tensor = torch.tensor(movie_ids, dtype=torch.long)

            # Create edge_index tensor
            edge_index = torch.stack([user_tensor, movie_tensor])
            edge_index_dict[column] = edge_index
        
        # all_text_dict = torch.load('/nobackup/users/junhong/Data/pokec/embedding.pt')
        # x = torch.zeros((len(dfn), 768))
        # for column in self.node_features:
        #     print(f'Processing {column} ...')
        #     text_dict = all_text_dict[column]
        #     for idx, text in enumerate(tqdm(dfn[column])):
        #         if text == text: # Check for nan
        #             x[idx] += text_dict[text]
        
        user_numeric_features = ['public', 'completion_percentage', 'AGE'] # 'registration', 'last_login'
        user_categorical_features = ['main_area']
        print("Getting user numerical features...")
        user_num_feats = normalize(get_numerical_features(dfn, user_numeric_features))[2]
        print("Getting user categorical features...")
        user_cat_feats = get_categorical_features(dfn, user_categorical_features)
        print("Getting user text features...")
        user_text_feats = dfn[self.node_fields[7:-1]].map(lambda x: 1 if isinstance(x, str) and x != '' else 0).values
        user_text_feats = torch.from_numpy(user_text_feats).float()
        
        data = HeteroData()

        data['user'].num_nodes = len(dfn)
        data['user'].y = torch.tensor(dfn['gender'].fillna(-1).values, dtype=torch.long)
        data['user'].x = torch.cat((user_num_feats, user_cat_feats, user_text_feats), dim=-1)
        for column, entry in entry_dict.items():
            data[column].num_nodes = len(entry_dict[column])

        source_tensor = torch.tensor(dfe['source'].values - 1, dtype=torch.long)
        target_tensor = torch.tensor(dfe['target'].values - 1, dtype=torch.long)
        data['user', 'has_friend', 'user'].edge_index = torch.stack([source_tensor, target_tensor], dim=0)
        for column, entry in entry_dict.items():
            data['user', 'lists', column].edge_index = edge_index_dict[column]

        indices = torch.where(data['user'].y != -1)[0]
        train_size = 0.5 # 50% train
        val_size = 0.5 # 25% val, 25% test
        train_idx, temp_idx = train_test_split(indices, train_size=train_size)
        val_idx, test_idx = train_test_split(temp_idx, train_size=val_size)
        train_mask = index_to_mask(train_idx, data['user'].num_nodes)
        val_mask = index_to_mask(val_idx, data['user'].num_nodes)
        test_mask = index_to_mask(test_idx, data['user'].num_nodes)
        data['user'].train_mask = train_mask
        data['user'].val_mask = val_mask
        data['user'].test_mask = test_mask
        data.validate()
        
        if self.pre_filter is not None:
            data = self.pre_filter(data)

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        torch.save(data, self.processed_paths[0])