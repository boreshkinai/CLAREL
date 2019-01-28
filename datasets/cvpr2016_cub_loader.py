from datasets.download_cvpr2016_cub import DEFAULT_DIR, DEFAULT_CUB_DIR
import os
from PIL import Image
from tqdm import tqdm
import numpy as np


class cvpr2016CubLoader():
    """
    Loads the CVPR2016-CUB dataset
    """

    def _load_split_class_ids(self):
        with open(os.path.join(self.data_dir, self.split + 'classes.txt')) as f:
            split_classes = f.read().splitlines()
        return [c.split(sep='.')[0] for c in split_classes]

    def _load_image_meta(self):
        # Load image names
        with open(os.path.join(self.data_dir, self.cub_dir, 'images.txt')) as f:
            image_lines = f.read().splitlines()
        # LOad train/val/test split file
        split_class_ids = self._load_split_class_ids()

        self.image_ids = []
        self.image_paths = []
        self.image_classes_txt = []
        self.image_classes = []
        self.image_file_names = []
        self.image_names = []
        for line in image_lines:
            line_split = line.split(sep=' ')
            class_id = line_split[1].split(sep='/')[0].split(sep='.')[0]
            if class_id in split_class_ids:
                self.image_ids.append(int(line_split[0]))
                self.image_paths.append(os.path.join(self.data_dir, self.cub_dir, 'images', line_split[1]))
                line_split = line_split[1].split(sep='/')
                self.image_classes_txt.append(line_split[0])
                self.image_file_names.append(line_split[1])
                self.image_classes.append(line_split[0].split(sep='.')[0])
                self.image_names.append(line_split[1].split(sep='.')[0])

        assert len(set(split_class_ids) - set(self.image_classes)) == 0
        assert len(set(self.image_classes) - set(split_class_ids)) == 0

    def _load_raw_images(self):
        print("Load raw image data for split:", self.split)
        self.raw_images = []
        for image_path in tqdm(self.image_paths):
            im = Image.open(image_path)
            scale_factor = ((self.img_target_size + 2*self.img_border_size) / min(im.size))
            im = im.resize((int(scale_factor*im.size[0]), int(scale_factor*im.size[1])), Image.ANTIALIAS)
            self.raw_images.append(im)

    def __init__(self, data_dir=DEFAULT_DIR, cub_dir=DEFAULT_CUB_DIR, split="train",
                 img_target_size=299, img_border_size=16):
        """

        :param data_dir: data in which the main dataset is stored
        :param cub_dir: the folder inside the main dataset where the original CUB-2011 data are stored
        :param split: name of the split, admissible names: train, test, val, trainval, all
        """

        self.data_dir = data_dir
        self.split = split
        self.cub_dir = cub_dir
        self.img_target_size = img_target_size
        self.img_border_size = img_border_size

        self._load_image_meta()
        self._load_raw_images()


        a = 1


cvpr2016CubLoader(data_dir='cvpr2016_cub', cub_dir=DEFAULT_CUB_DIR)
