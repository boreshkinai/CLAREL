import os
import sys
import argparse
import pathlib
import urllib.request as request
import zipfile
from download_cvpr2016_flowers import DEFAULT_FLOWERS_DIR, DEFAULT_DIR

# Downloader for the new splits of the FLOWERS dataset
# https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/feature-generating-networks-for-zero-shot-learning/
PROPOSED_SPLITS_URL = "http://datasets.d2.mpi-inf.mpg.de/xian/cvpr18xian.zip"
FILE_NAME_PROPOSED_SPLITS = 'cvpr18xian.zip'


def download_and_uncompress_dataset(dataset_dir: str):
    """Downloads the splits and features proposed in 
    https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/feature-generating-networks-for-zero-shot-learning/
    Parameters
    ----------
    dataset_dir : str
        The directory where the dataset is stored.
    """

    pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    filepath_flowers = os.path.join(dataset_dir, FILE_NAME_PROPOSED_SPLITS)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading {} {:.1f}%%'.format(
            filepath_flowers,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    
    print()
    print("Downloading FLOWERS from", PROPOSED_SPLITS_URL)
    print("Downloading FLOWERS to", filepath_flowers)
    filepath_flowers, _ = request.urlretrieve(PROPOSED_SPLITS_URL, filepath_flowers, _progress)
    statinfo = os.stat(filepath_flowers)
    print()
    print('Successfully downloaded', FILE_NAME_PROPOSED_SPLITS, statinfo.st_size, 'bytes.')
    print('Uncompressing...')
    print(filepath_flowers)
    with zipfile.ZipFile(filepath_flowers) as file:
        file.extractall(dataset_dir)
    assert os.path.isdir(os.path.join(dataset_dir, "cvpr18xian"))
    print('Successfully downloaded and extracted cvpr18xian')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-dir', type=str,
        default=DEFAULT_DIR,
        help='Path to the raw data')
    args = parser.parse_args()
    download_and_uncompress_dataset(args.dataset_dir)
