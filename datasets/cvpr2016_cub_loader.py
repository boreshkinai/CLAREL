from datasets.download_cvpr2016_cub import DEFAULT_DIR, DEFAULT_CUB_DIR
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from torch.utils.serialization import load_lua
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences


class cvpr2016CubLoader():
    """
    Loads the CVPR2016-CUB dataset
    """

    def __init__(self, data_dir=DEFAULT_DIR, cub_dir=DEFAULT_CUB_DIR, split="train",
                 img_target_size=299, img_border_size=16, max_text_len=30, vocab_size=10000):
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
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size

    def load(self):
        print()
        print('Loading split', self.split)
        self._load_image_meta()
        self._load_tokenized_text()
        self._load_raw_images()

    def _load_split_class_ids(self):
        with open(os.path.join(self.data_dir, self.split + 'classes.txt')) as f:
            split_classes = f.read().splitlines()
        return [c.split(sep='.')[0] for c in split_classes]

    def _load_image_meta(self):
        # Load image names
        with open(os.path.join(self.data_dir, self.cub_dir, 'images.txt')) as f:
            image_lines = f.read().splitlines()
        # Load train/val/test split file
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
        print("Load raw image data for split", self.split)
        self.raw_images = []
        for image_path in tqdm(self.image_paths):
            im = Image.open(image_path)
            scale_factor = ((self.img_target_size + 2*self.img_border_size) / min(im.size))
            im = im.resize((int(scale_factor*im.size[0]), int(scale_factor*im.size[1])), Image.ANTIALIAS)
            self.raw_images.append(im)

    def _load_tokenized_text(self):
        # # Load vocabulary
        # self.vocab = load_lua(os.path.join(self.data_dir, 'vocab_c10.t7'))
        # # Load tokenized text from the original dataset
        # self.bow = load_lua(os.path.join(self.data_dir, 'bow_c10', '072.Pomarine_Jaeger.t7'))
        # self.word = load_lua(os.path.join(self.data_dir, 'word_c10', '072.Pomarine_Jaeger.t7')).numpy().astype(dtype=np.int32)
        self._build_vocabulary(vocab_size=self.vocab_size)
        print()
        print("Loading and tokenizing texts in split", self.split)
        self.raw_texts = []
        self.tokenized_texts = []
        self.tokenized_text_lengths = []
        for image_file_name, image_class in tqdm(zip(self.image_file_names, self.image_classes_txt)):
            raw_text_lines = self._load_texts_of_an_imange(image_class, image_file_name)
            tokenized_text_lines = self.tokenizer.texts_to_sequences(raw_text_lines)
            tokenized_text_lengths = [min(len(x), self.max_text_len) for x in tokenized_text_lines]
            tokenized_text_lines = pad_sequences(tokenized_text_lines, maxlen=self.max_text_len, padding='post')
            self.raw_texts.append(raw_text_lines)
            self.tokenized_texts.append(tokenized_text_lines)
            self.tokenized_text_lengths.append(tokenized_text_lengths)

    def _build_vocabulary(self, vocab_size=10000):
        print('Building vocabulary')
        # create the loader object, to be able to go over the 'all' split and create the vocabulary
        all_loader = cvpr2016CubLoader(data_dir=self.data_dir, cub_dir=self.cub_dir, split='all')
        all_loader._load_image_meta()
        print("Loading all raw texts")
        raw_texts = []
        for image_file_name, image_class in tqdm(zip(all_loader.image_file_names, all_loader.image_classes_txt)):
            text_lines = self._load_texts_of_an_imange(image_class, image_file_name)
            raw_texts.extend(text_lines)
        all_loader = None  # this is just for safety to avoid test data leaks

        tokenizer = Tokenizer(num_words=vocab_size, filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(raw_texts)
        self.tokenizer = tokenizer

    def _load_texts_of_an_imange(self, image_class:str=None, image_file_name:str=None):
        with open(os.path.join(self.data_dir, 'text_c10', image_class, image_file_name.split('.')[0] + '.txt')) as f:
            text_lines = f.read().splitlines()
        assert len(text_lines) == 10
        return text_lines



# loader = cvpr2016CubLoader(data_dir='cvpr2016_cub', cub_dir=DEFAULT_CUB_DIR)
# loader.load_data()
