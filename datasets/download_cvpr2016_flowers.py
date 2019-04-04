import os
import sys
import tarfile
import argparse
import pathlib
import urllib.request as request
from google_drive_downloader import GoogleDriveDownloader as gdd
from scipy import io
import glob

# The URL where the CVPR2016 FLOWERS data can be downloaded.
# https://github.com/reedscot/cvpr2016
# http://www.robots.ox.ac.uk/~vgg/data/flowers/102/
FILE_ID = '0B0ywwgffWnLLcms2WWJQRFNSWXM'
FILE_NAME_CVPR2016_FLOWERS = 'cvpr2016_flowers.tar.gz'

FLOWERS_IMG_URL = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/102flowers.tgz"
FLOWERS_LBL_URL = "http://www.robots.ox.ac.uk/~vgg/data/flowers/102/imagelabels.mat"
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
    filepath_cvpr2016_flowers_labels = os.path.join(dataset_dir, FLOWERS_LBL_URL.split('/')[-1])
    print(filepath_cvpr2016_flowers_labels)

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
    print("Downloading FLOWERS images from", FLOWERS_IMG_URL)
    print("Downloading FLOWERS images to", filepath_flowers)
    filepath_flowers, _ = request.urlretrieve(FLOWERS_IMG_URL, filepath_flowers, _progress)
    statinfo = os.stat(filepath_flowers)
    print()
    print('Successfully downloaded', FILE_NAME_FLOWERS, statinfo.st_size, 'bytes.')
    print('Uncompressing...')
    tarfile.open(filepath_flowers, 'r').extractall(dataset_dir)
    assert os.path.isdir(os.path.join(dataset_dir, DEFAULT_FLOWERS_DIR))
    print('Downloading FLOWERS image labels...')
    filepath_cvpr2016_flowers_labels, _ = request.urlretrieve(FLOWERS_LBL_URL, filepath_cvpr2016_flowers_labels, _progress)
    statinfo = os.stat(filepath_cvpr2016_flowers_labels)
    print()
    print('Successfully downloaded', FLOWERS_LBL_URL.split('/')[-1], statinfo.st_size, 'bytes.')
    print('Writing mapping class to image mapping into images.txt...')
    image_list = glob.glob(os.path.join(dataset_dir, DEFAULT_FLOWERS_DIR, "*.jpg"))
    image_list.sort()
    matfile = io.loadmat(os.path.join(dataset_dir, "imagelabels.mat"))
    labels = matfile["labels"].ravel()
    
    image_lines = []
    for img_path in image_list:
        img_file_name = img_path.split(os.path.sep)[-1]
        img_idx = int(img_file_name.split("_")[-1].split(".")[0])
        label = labels[img_idx-1]
        img_line = "%d %03d.%s" %(img_idx, label, img_file_name)
        image_lines.append(img_line)
    
    with open(os.path.join(dataset_dir, DEFAULT_FLOWERS_DIR, "images.txt"), 'w') as f:
        f.write("\n".join(image_lines))
    
    print('Successfully downloaded and extracted FLOWERS')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--dataset-dir', type=str,
        default=DEFAULT_DIR,
        help='Path to the raw data')
    args = parser.parse_args()
    download_and_uncompress_dataset(args.dataset_dir)
