import os
import sys
import tarfile
import argparse
import pathlib
import urllib.request as request
from google_drive_downloader import GoogleDriveDownloader as gdd
import gzip
import shutil


# The URL where the word2vec pretrained model data can be downloaded.
FILE_ID = '0B7XkCwpI5KDYNlNUTTlSS21pQmM'
FILE_NAME_WORD2VEC = 'GoogleNews-vectors-negative300.bin.gz'

DEFAULT_DIR = os.path.join(os.sep, 'mnt', 'datasets', 'public', 'research', 'GoogleNews_vectors_negative300')


def download_and_uncompress_dataset(dataset_dir: str):
    """Downloads CVPR2016-CUB dataset, uncompresses it locally.
    Parameters
    ----------
    dataset_dir : str
        The directory where the dataset is stored.
    """

    pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    filename_word2vec = FILE_NAME_WORD2VEC
    filepath_word2vec = os.path.join(dataset_dir, filename_word2vec)


    print()
    print("Downloading pretrained word2vec from Google Drive file id", FILE_ID)
    print("Downloading to", filepath_word2vec)
    gdd.download_file_from_google_drive(file_id=FILE_ID, dest_path=filepath_word2vec, unzip=False)
    statinfo = os.stat(filepath_word2vec)
    print()
    print('Successfully downloaded', filename_word2vec, statinfo.st_size, 'bytes.')
    print('Uncompressing...')
    with gzip.open(filepath_word2vec, 'rb') as f_in:
        with open(".".join(filepath_word2vec.split('.')[:-1]), 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)

    print('Successfully downloaded and extracted CVPR2016-CUB')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset-dir', type=str, default=DEFAULT_DIR, help='Path to the raw data')
    args = parser.parse_args()
    download_and_uncompress_dataset(args.dataset_dir)
