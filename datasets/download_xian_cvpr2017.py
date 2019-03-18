import os
import sys
import argparse
import pathlib
import urllib.request as request
import zipfile

# Downloader for the new splits of the CUB dataset
# https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/
PROPOSED_SPLITS_URL = "http://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip"
FILE_NAME_PROPOSED_SPLITS = 'xlsa17.zip.tgz'


DEFAULT_DIR = os.path.join(os.sep, 'mnt', 'datasets', 'public', 'research', 'cvpr2016_cub')
DEFAULT_CUB_DIR = 'CUB_200_2011'



def download_and_uncompress_dataset(dataset_dir: str):
    """Downloads the splits and features proposed in 
    https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/
    Parameters
    ----------
    dataset_dir : str
        The directory where the dataset is stored.
    """

    pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    filepath_cub2011 = os.path.join(dataset_dir, FILE_NAME_PROPOSED_SPLITS)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading {} {:.1f}%%'.format(
            filepath_cub2011,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()
    
    print()
    print("Downloading CUB2011 from", PROPOSED_SPLITS_URL)
    print("Downloading CUB2011 to", filepath_cub2011)
    filepath_cub2011, _ = request.urlretrieve(PROPOSED_SPLITS_URL, filepath_cub2011, _progress)
    statinfo = os.stat(filepath_cub2011)
    print()
    print('Successfully downloaded', FILE_NAME_PROPOSED_SPLITS, statinfo.st_size, 'bytes.')
    print('Uncompressing...')
    print(filepath_cub2011)
    with zipfile.ZipFile(filepath_cub2011) as file:
        file.extractall(dataset_dir)
    assert os.path.isdir(os.path.join(dataset_dir, "xlsa17"))
    print('Successfully downloaded and extracted xlsa17')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-dir', type=str,
        default=DEFAULT_DIR,
        help='Path to the raw data')
    args = parser.parse_args()
    download_and_uncompress_dataset(args.dataset_dir)
