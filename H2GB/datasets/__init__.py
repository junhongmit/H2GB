from .mag_dataset import MAGDataset
from .oag_dataset import OAGDataset
from .rcdd_dataset import RCDDDataset
from .ieee_cis_dataset import IeeeCisDataset
from .pokec_dataset import PokecDataset
from .pdns_dataset import PDNSDataset


__all__ = classes = [
    'MAGDataset',
    'OAGDataset',
    'RCDDDataset',
    'IeeeCisDataset',
    'PokecDataset',
    'PDNSDataset',
]

academia_datasets = [
    'MAGDataset',
    'OAGDataset',
]

finance_datasets = [
    'IeeeCisDataset',
]

ecommerce_datasets = [
    'RCDDDataset',
]

social_science_datasets = [
    'PokecDataset',
]

cybersecurity_datasets = [
    'PDNSDataset',
]