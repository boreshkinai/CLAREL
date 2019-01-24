import os
import tarfile
import argparse
import pathlib
from google_drive_downloader import GoogleDriveDownloader as gdd

# The URL where the CVPR2016 CUB data can be downloaded.
# https://github.com/reedscot/cvpr2016
FILE_ID = '0B0ywwgffWnLLZW9uVHNjb2JmNlE'
FILE_NAME = 'cvpr2016_cub.tar.gz'


def download_and_uncompress_dataset(dataset_dir: str):
    """Downloads CVPR2016-CUB dataset, uncompresses it locally.
    Parameters
    ----------
    dataset_dir : str
        The directory where the dataset is stored.
    """
    filename = FILE_NAME
    pathlib.Path(dataset_dir).mkdir(parents=True, exist_ok=True)
    filepath = os.path.join(dataset_dir, filename)


    print()
    print("Downloading CVPR2016-CUB from Google Drive file id", FILE_ID)
    print("Downloading CVPR2016-CUB to", filepath)
    gdd.download_file_from_google_drive(file_id=FILE_ID,
                                        dest_path=filepath,
                                        unzip=False)
    statinfo = os.stat(filepath)
    print()
    print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
    print('Uncompressing...')
    tarfile.open(filepath, 'r').extractall(dataset_dir)
    print('Successfully downloaded and extracted CVPR2016-CUB')

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-dir', type=str,
        default=os.path.join(os.sep, 'mnt', 'datasets', 'public', 'research', 'cvpr2016_cub'),
        help='Path to the raw data')
    args = parser.parse_args()
    download_and_uncompress_dataset(args.dataset_dir)
