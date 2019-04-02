import os
import sys
import tarfile
import argparse
import pathlib
import urllib.request as request
from google_drive_downloader import GoogleDriveDownloader as gdd

# The URL where the CVPR2016 FLOWERS data can be downloaded.
# https://github.com/reedscot/cvpr2016
FILE_ID = '0B0ywwgffWnLLcms2WWJQRFNSWXM'
FILE_NAME_CVPR2016_FLOWERS = 'cvpr2016_flowers.tar.gz'

FLOWERS_URL = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
FILE_NAME_FLOWERS = '102flowers.tgz'


DEFAULT_DIR = os.path.join(os.sep, 'mnt', 'datasets', 'public', 'research', 'cvpr2016_flowers')
DEFAULT_FLOWERS_DIR = 'jpg'


def download_and_uncompress_dataset(dataset_dir: str):
    """Downloads CVPR2016-FLOWERS dataset, uncompresses it locally.
    Parameters
    ----------
    dataset_dir : str
        The directory where the dataset is stored.
    """

    pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    filename_cvpr2016_flowers = FILE_NAME_CVPR2016_FLOWERS
    filepath_cvpr2016_flowers = os.path.join(dataset_dir, filename_cvpr2016_flowers)

    filepath_flowers = os.path.join(dataset_dir, FILE_NAME_FLOWERS)

    def _progress(count, block_size, total_size):
        sys.stdout.write('\r>> Downloading {} {:.1f}%%'.format(
            filepath_flowers,
            float(count * block_size) / float(total_size) * 100.0))
        sys.stdout.flush()


    print()
    print("Downloading CVPR2016-FLOWERS from Google Drive file id", FILE_ID)
    print("Downloading CVPR2016-FLOWERS to", filepath_cvpr2016_flowers)
    gdd.download_file_from_google_drive(file_id=FILE_ID,
                                        dest_path=filepath_cvpr2016_flowers,
                                        unzip=False)
    statinfo = os.stat(filepath_cvpr2016_flowers)
    print()
    print('Successfully downloaded', filename_cvpr2016_flowers, statinfo.st_size, 'bytes.')
    print('Uncompressing...')
    tarfile.open(filepath_cvpr2016_flowers, 'r').extractall(dataset_dir)
    assert os.path.isdir(DEFAULT_DIR)
    print('Successfully downloaded and extracted CVPR2016-FLOWERS')

    print()
    print("Downloading FLOWERS from", FLOWERS_URL)
    print("Downloading FLOWERS to", filepath_flowers)
    filepath_flowers, _ = request.urlretrieve(FLOWERS_URL, filepath_flowers, _progress)
    statinfo = os.stat(filepath_flowers)
    print()
    print('Successfully downloaded', FILE_NAME_FLOWERS, statinfo.st_size, 'bytes.')
    print('Uncompressing...')
    tarfile.open(filepath_flowers, 'r').extractall(dataset_dir)
    assert os.path.isdir(os.path.join(dataset_dir, DEFAULT_FLOWERS_DIR))
    print('Successfully downloaded and extracted FLOWERS')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-dir', type=str,
        default=DEFAULT_DIR,
        help='Path to the raw data')
    args = parser.parse_args()
    download_and_uncompress_dataset(args.dataset_dir)
