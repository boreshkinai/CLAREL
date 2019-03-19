from datasets.cvpr2016_cub_loader import Cvpr2016CubLoader
from datasets.xian2017_cub_loader import Xian2017CubLoader
from datasets.fashion_gen import FashionGen
from typing import List

DATASETS = {'cvpr2016_cub': Cvpr2016CubLoader, 'xian2017_cub': Xian2017CubLoader, 'fashion_gen': FashionGen}


def get_dataset_splits(dataset_name, data_dir, splits: List[str] = ['train', 'test', 'validation'], flags=None):
    dataset = DATASETS[dataset_name]
    dataset_splits = {}
    for split in splits:
        loader = dataset(data_dir=data_dir, split=split, image_model=flags.image_feature_extractor)
        loader.load_cached()
        dataset_splits[split] = loader
    return dataset_splits