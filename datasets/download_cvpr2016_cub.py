import os
import sys
import tarfile
import argparse
import pathlib
import urllib.request as request
from google_drive_downloader import GoogleDriveDownloader as gdd

# The URL where the CVPR2016 CUB data can be downloaded.
# https://github.com/reedscot/cvpr2016
FILE_ID = '0B0ywwgffWnLLZW9uVHNjb2JmNlE'
FILE_NAME_CVPR2016_CUB = 'cvpr2016_cub.tar.gz'

CUB2011_URL = "http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz"
FILE_NAME_CUB2011 = 'CUB_200_2011.tgz'


DEFAULT_DIR = os.path.join(os.sep, 'mnt', 'datasets', 'public', 'research', 'cvpr2016_cub')
DEFAULT_CUB_DIR = os.path.join(os.sep, DEFAULT_DIR, 'CUB_200_2011')


def download_and_uncompress_dataset(dataset_dir: str):
    """Downloads CVPR2016-CUB dataset, uncompresses it locally.
    Parameters
    ----------
    dataset_dir : str
        The directory where the dataset is stored.
    """

    pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    filename_cvpr2016_cub = FILE_NAME_CVPR2016_CUB
    filepath_cvpr2016_cub = os.path.join(dataset_dir, filename_cvpr2016_cub)

    filepath_cub2011 = os.path.join(dataset_dir, FILE_NAME_CUB2011)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading {} {:.1f}%%'.format(
            filepath_cub2011,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()


    print()
    print("Downloading CVPR2016-CUB from Google Drive file id", FILE_ID)
    print("Downloading CVPR2016-CUB to", filepath_cvpr2016_cub)
    gdd.download_file_from_google_drive(file_id=FILE_ID,
                                        dest_path=filepath_cvpr2016_cub,
                                        unzip=False)
    statinfo = os.stat(filepath_cvpr2016_cub)
    print()
    print('Successfully downloaded', filename_cvpr2016_cub, statinfo.st_size, 'bytes.')
    print('Uncompressing...')
    tarfile.open(filepath_cvpr2016_cub, 'r').extractall(dataset_dir)
    assert os.path.isdir(DEFAULT_DIR)
    print('Successfully downloaded and extracted CVPR2016-CUB')

    print()
    print("Downloading CUB2011 from", CUB2011_URL)
    print("Downloading CUB2011 to", filepath_cvpr2016_cub)
    filepath_cub2011, _ = request.urlretrieve(CUB2011_URL, filepath_cub2011, _progress)
    statinfo = os.stat(filepath_cub2011)
    print()
    print('Successfully downloaded', FILE_NAME_CUB2011, statinfo.st_size, 'bytes.')
    print('Uncompressing...')
    tarfile.open(filepath_cub2011, 'r').extractall(dataset_dir)
    assert os.path.isdir(DEFAULT_CUB_DIR)
    print('Successfully downloaded and extracted CUB2011')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-dir', type=str,
        default=DEFAULT_DIR,
        help='Path to the raw data')
    args = parser.parse_args()
    download_and_uncompress_dataset(args.dataset_dir)
