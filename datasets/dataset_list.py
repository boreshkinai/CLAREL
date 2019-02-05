from datasets.cvpr2016_cub_loader import Cvpr2016CubLoader
from datasets.fashion_gen import FashionGen
from typing import List

DATASETS = {'cvpr2016_cub': Cvpr2016CubLoader, 'fashion_gen': FashionGen}


def get_dataset_splits(dataset_name, data_dir, splits: List[str] = ['train', 'test', 'validation']):
    dataset = DATASETS[dataset_name]
    dataset_splits = {}
    for split in splits:
        loader = dataset(data_dir=data_dir, split=split)
        loader.load_cached()
        dataset_splits[split] = loader
    return dataset_splits