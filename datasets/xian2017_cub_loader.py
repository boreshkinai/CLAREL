from datasets.download_cvpr2016_cub import DEFAULT_DIR, DEFAULT_CUB_DIR
from datasets.download_googlenews_vectors_negative import DEFAULT_WORD2VEC_DIR, FILE_NAME_WORD2VEC
import os
from PIL import Image
from tqdm import tqdm
import numpy as np
from torch.utils.serialization import load_lua
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from datasets import Dataset
import pickle
import gzip
import time
import gensim
import gensim.downloader
from common.pretrained_models import IMAGE_MODELS
import logging
import scipy


class Xian2017CubLoader(Dataset):
    """
    Loads the CVPR2016-CUB dataset with splits proposed in https://www.mpi-inf.mpg.de/departments/computer-vision-and-multimodal-computing/research/zero-shot-learning/zero-shot-learning-the-good-the-bad-and-the-ugly/
    """

    def __init__(self, data_dir: str = DEFAULT_DIR, cub_dir: str = DEFAULT_CUB_DIR, split: str = "train",
                 img_target_size: int = 299, img_border_size: int = 16, max_text_len: int = 30,
                 vocab_size: int = 10000, word_embed_dir=DEFAULT_WORD2VEC_DIR, word_embed_file=FILE_NAME_WORD2VEC,
                 image_model='inception_v2', fold='', shuffle_text_in_batch=True):
        """

        :param data_dir: data in which the main dataset is stored
        :param cub_dir: the folder inside the main dataset where the original CUB-2011 data are stored
        :param split: name of the split, admissible names: train, test, val, trainval, all
        """
        super().__init__()
        if data_dir:
            self.data_dir = data_dir
        else:
            self.data_dir = DEFAULT_DIR
        self.xlsa17_dir = os.path.join(self.data_dir, "xlsa17", "data", "CUB")
        self.split = split
        self.cub_dir = cub_dir
        self.img_target_size = img_target_size
        self.img_border_size = img_border_size
        self.max_text_len = max_text_len
        self.vocab_size = vocab_size
        self.word_embed_dir = word_embed_dir
        self.word_embed_file = word_embed_file
        self.image_model = image_model
        self.fold = fold
        self.shuffle_text_in_batch = shuffle_text_in_batch
        self.crop_options = [self._crop_center, self._crop_bottom_left,
                             self._crop_bottom_right, self._crop_top_left, self._crop_top_right]

    def load(self):
        logging.info('Loading split %s' % self.split)
        self._load_image_meta()
        self._load_tokenized_text()
        self._load_text_embeddings()
        self._load_image_features()

    def load_cached(self):
        logging.info('Loading split %s' % self.split)
        cache_filepath = os.path.join(self.data_dir, 'split_' + self.split + '_xian2017.pkl')
        if os.path.isfile(cache_filepath):
            logging.info("Loading cached file %s" % cache_filepath)
            dt_load = time.time()
            with gzip.GzipFile(cache_filepath, 'r') as f:
                self.__dict__.update(pickle.load(f).__dict__)
            logging.info("Loaded cache in %f sec" % (time.time() - dt_load) )
            return

        self.load()

        with gzip.GzipFile(cache_filepath, 'w') as f:
            pickle.dump(self, f)
            
    def _load_image_features(self):
        image_features_path = os.path.join(self.xlsa17_dir, "res101.mat")
        logging.info("Loading precomputed features from %s ..." % image_features_path)
        matfile = scipy.io.loadmat(image_features_path)
        image_embeddings_precomputed = {n[0].split("/")[-1]: f 
                                            for n,f in zip(matfile['image_files'].ravel(), matfile['features'].transpose())}
        image_embeddings = []
        for im_name in self.image_file_names:
            features = image_embeddings_precomputed[im_name]
            # This is because the previous code would sample from 10 different crops. Backward compatibility.
            features = np.tile(features, (10, 1))
            image_embeddings.append(features)
        self.image_embeddings = np.stack(image_embeddings)

    def _load_text_embeddings(self):
        logging.info("Loading word2vec embedding...")
        word_embedding_model = gensim.models.KeyedVectors.load_word2vec_format(
            os.path.join(self.word_embed_dir, self.word_embed_file), binary=True)

        self.word_vectors_dict = {}
        self.word_vectors_idx = []
        self.word_vectors_len = word_embedding_model.vector_size
        num_misses = 0
        # this corresponds to the 0, unknown, token in the tokenizer
        self.word_vectors_idx.append(np.zeros(shape=(word_embedding_model.vector_size,)))
        self.word_vectors_dict["UNK"] = np.zeros(shape=(word_embedding_model.vector_size,))
        for key, idx in self.tokenizer.word_index.items():
            if key in word_embedding_model.vocab:
                self.word_vectors_dict[key] = word_embedding_model[key]
                self.word_vectors_idx.append(word_embedding_model[key])
            else:
                num_misses += 1
                self.word_vectors_dict[key] = np.zeros(shape=(word_embedding_model.vector_size,))
                self.word_vectors_idx.append(np.zeros(shape=(word_embedding_model.vector_size,)))

        logging.info("Number of tokens not found in the embedding: %d out of %d" %(num_misses, len(self.tokenizer.word_index)))
        self.word_vectors_idx = np.array(self.word_vectors_idx, dtype=np.float32)

    def _load_split_class_ids(self):
        split_map = {"trainval": '%sclasses.txt' % "trainval", 
                     "test_seen": '%sclasses.txt' % "trainval", 
                     "test_unseen": '%sclasses.txt' % "test", 
                     "train": '%sclasses1.txt' % "train", "val": '%sclasses1.txt' % "val",
                     "all": "%sclasses.txt" % "all"}
        with open(os.path.join(self.xlsa17_dir, split_map[self.split]), 'r') as f:
            split_classes = f.read().splitlines()
        self.split_class_ids = [c.split(sep='.')[0] for c in split_classes]
        
    def _load_seen_unseen(self):
        split_map = {"trainval": "trainval_loc", 
                     "test_seen": "test_seen_loc", 
                     "test_unseen": "test_unseen_loc", 
                     "train": "train_loc", "val": "val_loc"}
        self.att_splits_matfile = scipy.io.loadmat(os.path.join(self.xlsa17_dir, "att_splits.mat"))
        self.res101_matfile = scipy.io.loadmat(os.path.join(self.xlsa17_dir, "res101.mat"))
        if self.split in split_map.keys():
            # -1 to correct for the matlab based indexing
            self.xlsa_split_image_idxs = self.att_splits_matfile[split_map[self.split]].ravel()-1

            self.xlsa_split_image_names = self.res101_matfile['image_files'][self.xlsa_split_image_idxs].ravel()
            self.xlsa_split_image_names = np.array([s[0].split('/')[-1].split('.')[0] for s in self.xlsa_split_image_names])
        else:
            self.xlsa_split_image_idxs = []
            self.xlsa_split_image_names = []
        
    def _load_image_meta(self):
        # This is according to https://arxiv.org/pdf/1703.04394.pdf
        number_of_images = {"trainval": 7057, "test_seen": 1764, "test_unseen": 2967, 
                            "all": 11788, "train": 5875, "val": 2946}
        # Load image names
        with open(os.path.join(self.data_dir, self.cub_dir, 'images.txt'), 'r') as f:
            image_lines = f.read().splitlines()
        # Load train/val/test split file
        self._load_split_class_ids()
        self._load_seen_unseen()

        self.image_ids = []
        self.image_paths = []
        self.image_classes_txt = []
        self.image_classes = []
        self.image_file_names = []
        self.image_names = []
        for line in tqdm(image_lines):
            line_split = line.split(sep=' ')
            class_id = line_split[1].split(sep='/')[0].split(sep='.')[0]
            if class_id in self.split_class_ids:
                image_id = int(line_split[0])
                image_path = os.path.join(self.data_dir, self.cub_dir, 'images', line_split[1])
                line_split = line_split[1].split(sep='/')
                image_name = line_split[1].split(sep='.')[0]
                
                if np.isin(image_name, self.xlsa_split_image_names).item() or len(self.xlsa_split_image_names) == 0:
                    self.image_ids.append(image_id)
                    self.image_paths.append(image_path)
                    self.image_classes_txt.append(line_split[0])
                    self.image_file_names.append(line_split[1])
                    self.image_classes.append(line_split[0].split(sep='.')[0])
                    self.image_names.append(image_name)
        
#         print(len(self.image_names))
#         print(set(self.split_class_ids))
#         print(set(self.image_classes))
        
        assert len(set(self.split_class_ids) - set(self.image_classes)) == 0
        assert len(set(self.image_classes) - set(self.split_class_ids)) == 0
        if len(self.xlsa_split_image_names) > 0:
            assert len(set(self.xlsa_split_image_names) - set(self.image_names)) == 0
            assert len(set(self.image_names) - set(self.xlsa_split_image_names)) == 0
        assert number_of_images[self.split] == len(self.image_names)

    def _load_tokenized_text(self):
        # # Load vocabulary
        # self.vocab = load_lua(os.path.join(self.data_dir, 'vocab_c10.t7'))
        # # Load tokenized text from the original dataset
        # self.bow = load_lua(os.path.join(self.data_dir, 'bow_c10', '072.Pomarine_Jaeger.t7'))
        # self.word = load_lua(os.path.join(self.data_dir, 'word_c10', '072.Pomarine_Jaeger.t7'
        #                                   )).numpy().astype(dtype=np.int32)
        self._build_vocabulary(vocab_size=self.vocab_size)
        logging.info("Loading and tokenizing texts in split %s" % self.split)
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
        logging.info('Building vocabulary')
        # create the loader object, to be able to go over the 'all' split and create the vocabulary
        all_loader = Xian2017CubLoader(data_dir=self.data_dir, cub_dir=self.cub_dir, split='all')
        all_loader._load_image_meta()
        logging.info("Loading all raw texts")
        raw_texts = []
        for image_file_name, image_class in tqdm(zip(all_loader.image_file_names, all_loader.image_classes_txt)):
            text_lines = self._load_texts_of_an_imange(image_class, image_file_name)
            raw_texts.extend(text_lines)

        tokenizer = Tokenizer(num_words=vocab_size, filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')
        tokenizer.fit_on_texts(raw_texts)
        self.tokenizer = tokenizer

    def _load_texts_of_an_imange(self, image_class: str = None, image_file_name: str = None):
        with open(os.path.join(self.data_dir, 'text_c10', image_class,
                               image_file_name.split('.')[0] + '.txt'), 'r') as f:
            text_lines = f.read().splitlines()
        assert len(text_lines) == 10
        return text_lines

    def next_batch_features(self, batch_size: int = 64, num_images: int = 2, num_texts: int = 5):
        """
        :param batch_size: number of text/image pairs in the batch
        :param num_images: number of images sampled per pair
        :param num_texts: number of texts per pair
        :return:
        """
        idxs = np.arange(len(self.image_embeddings))
        batch_idxs = np.random.choice(idxs, size=batch_size, replace=False)
        batch_features = []
        batch_texts = []
        batch_text_lengths = []
        class_labels = np.zeros((batch_size,), dtype=np.int64)
        for i, idx in enumerate(batch_idxs):
            batch_features.append(self._sample_features(idx, num_images=num_images))
            texts, text_lengths = self._sample_texts(idx, num_texts=num_texts)
            batch_texts.append(texts)
            batch_text_lengths.append(text_lengths)
            class_labels[i] = int(self.image_classes[idx])
        
        # Shuffling the text to avoid having a trivial relation between image indexes and text indexes
        labels_txt2img = np.arange(batch_size, dtype=np.int32)
        permutation = np.arange(batch_size, dtype=np.int32)
        
        if self.shuffle_text_in_batch:
            permutation = np.random.choice(np.arange(batch_size, dtype=np.int32), size=batch_size, replace=False)
            for i in range(batch_size):
                labels_txt2img[permutation[i]] = i
        labels_img2txt = permutation
        
        return np.array(batch_features), np.array(batch_texts)[permutation], np.array(batch_text_lengths)[permutation], \
               (labels_txt2img, labels_img2txt), class_labels

    def _sample_features(self, idx: int, num_images: int):
        img_embeddings = self.image_embeddings[idx]
        embeddings_batch = []
        for i in range(num_images):
            crop_idx = np.random.choice(np.arange(len(self.crop_options)), size=1, replace=False)[0]
            embeddings_batch.append(img_embeddings[crop_idx])
        return np.array(embeddings_batch)

    def _crop_center(self, img: Image):
        width, height = img.size
        start_w = width // 2 - (self.img_target_size // 2)
        start_h = height // 2 - (self.img_target_size // 2)
        return img.crop((start_w, start_h, start_w + self.img_target_size, start_h + self.img_target_size))

    def _crop_top_left(self, img: Image):
        return img.crop((0, 0, self.img_target_size, self.img_target_size))

    def _crop_top_right(self, img: Image):
        width, height = img.size
        return img.crop((width - self.img_target_size, 0, width, self.img_target_size))

    def _crop_bottom_left(self, img: Image):
        width, height = img.size
        return img.crop((0, height - self.img_target_size, self.img_target_size, height))

    def _crop_bottom_right(self, img: Image):
        width, height = img.size
        return img.crop((width - self.img_target_size, height - self.img_target_size, width, height))

    def _sample_texts(self, idx: int, num_texts: int):
        texts = self.tokenized_texts[idx]
        lengths = np.array(self.tokenized_text_lengths[idx])
        idxs = np.arange(len(texts))
        sampled_idxs = np.random.choice(idxs, size=num_texts, replace=True)
        return texts[sampled_idxs], lengths[sampled_idxs]

    def sequential_evaluation_batches_features(self, batch_size: int = 64, num_images: int = 10, num_texts: int = 10):
        num_batches = int(np.ceil(len(self.image_embeddings) / batch_size))
        for i in range(num_batches):
            idx_max = min((i + 1) * batch_size, len(self.image_embeddings))
            if num_images == 10:
                embeddings_out = self.image_embeddings[i * batch_size:(i + 1) * batch_size]
            else:
                embeddings_out = [self._sample_features(idx, num_images) for idx in range(i * batch_size, idx_max)]

            if num_texts == 10:
                texts_out = self.tokenized_texts[i * batch_size:(i + 1) * batch_size]
                text_lengths_out = self.tokenized_text_lengths[i * batch_size:(i + 1) * batch_size]
            else:
                texts_out, text_lengths_out = [], []
                for idx in range(i * batch_size, idx_max):
                    texts, text_lengths = self._sample_texts(idx, num_texts)
                    texts_out.append(texts)
                    text_lengths_out.append(text_lengths)

            yield np.array(embeddings_out), np.array(texts_out), np.array(text_lengths_out)
            