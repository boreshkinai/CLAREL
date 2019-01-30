import os
import pickle
import h5py
import glob
import numpy as np
import pathlib
from tqdm import tqdm, trange
from keras.preprocessing.sequence import pad_sequences
from shutil import copy
from typing import List, Dict
from common.util import Namespace
import json
from PIL import Image


def get_json_from_ssense_img_name(ssense_dir, img_name):
    json_name = img_name.split('/')[-1].split('_')[0] + '.json'
    with open(os.path.join(ssense_dir, 'images_metadata', json_name)) as f:
        json_content = json.load(f)
    return json_content


class FashionGen(object):
    """ Basic image and text dataset generating batches from a collection of files in a folder """

    def __init__(self, data_path: str = "/mnt/scratch/ssense/data_dumps/images_png_dump_256",
                 tokenizer_path: str = '/mnt/scratch/boris/ssense/tokenizer_embedding.pkl',
                 flags: Namespace = None):
        self.data_path = data_path
        self.tokenizer_path = tokenizer_path

        glob_pattern = os.path.join(data_path, '*.png')
        self.image_filenames = sorted(glob.glob(glob_pattern))
        self.n_samples = len(self.image_filenames)
        self.max_len = flags.text_maxlen
        self.shuffle_text_in_batch = flags.shuffle_text_in_batch

        with open(self.tokenizer_path, 'rb') as f:
            self.tokenizer = pickle.load(f)

    def get_images(self, img_names):
        images = []
        for img_name in img_names:
            images.append(np.asarray(Image.open(img_name)))
        return np.asarray(images)

    def get_text(self, img_names):
        texts = []
        text_length = []
        for img_name in img_names:
            json_content = get_json_from_ssense_img_name(self.data_path, img_name)
            # TODO: Remove b' at the beginnning and ' at the end, this is dirty solution
            text_current = json_content['description'][2:-1]
            text_tokens = self.tokenizer.texts_to_sequences([text_current])[0]
            text_length.append(len(text_tokens))
            texts.append(text_tokens)
        texts = pad_sequences(texts, maxlen=self.max_len, padding='post')
        return texts, np.asarray(text_length, dtype=np.int32)

    def next_batch(self, batch_size=64):
        idxs = np.random.randint(self.n_samples, size=batch_size)
        img_names = [self.image_filenames[i] for i in idxs]

        images = self.get_images(img_names)
        texts, text_length = self.get_text(img_names)
        labels_txt2img = np.arange(batch_size, dtype=np.int32)
        permutation = np.arange(batch_size, dtype=np.int32)
        # Shuffling the text to avoid having a trivial relation between image indexes and text indexes
        if self.shuffle_text_in_batch:
            permutation = np.random.choice(np.arange(batch_size, dtype=np.int32), size=batch_size, replace=False)
            for i in range(batch_size):
                labels_txt2img[permutation[i]] = i
        labels_img2txt = permutation
        return images, texts[permutation], text_length[permutation], (labels_txt2img, labels_img2txt)

    def sequential_batches(self, batch_size, n_batches, rng=np.random):
        """Generator for a random sequence of minibatches with no overlap."""
        permutation = rng.permutation(self.n_samples)
        for batch_idx in range(n_batches):
            start = batch_idx * batch_size
            end = np.minimum((start + batch_size), self.n_samples)
            idxs = permutation[start:end]
            img_names = [self.image_filenames[i] for i in idxs]

            images = self.get_images(img_names)
            texts, text_length = self.get_text(img_names)

            yield images, texts, text_length
            if end == self.n_samples:
                break

def build_tokenizer(flags):
    from keras.preprocessing.text import Tokenizer
    if os.path.exists('/mnt/scratch/boris/ssense/tokenizer_embedding.pkl'):
        with open('/mnt/scratch/boris/ssense/tokenizer_embedding.pkl', 'rb') as input_file:
            tokenizer = pickle.load(input_file)
    else:
        tokenizer = Tokenizer(num_words=flags.vocab_size, filters='\'!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n')

        f = h5py.File('/mnt/scratch/ssense/latest_indexes/unlocked_indexes/ssense_256_256_train.h5', mode='r')
        descriptions = np.char.decode(f['input_description'][:, 0], 'latin')
        tokenizer.fit_on_texts(list(descriptions))
        f.close()
        with open('/mnt/scratch/boris/ssense/tokenizer_embedding.pkl', 'wb') as output_file:
            pickle.dump(tokenizer, output_file)


def get_product_id(img_name: str):
    file_name, ext = os.path.splitext(os.path.basename(img_name))
    product_id, *_, pose_id = file_name.split('_')
    return product_id


def get_product_list(image_filenames: list):
    return list(set([get_product_id(img_name) for img_name in image_filenames]))


def get_product_to_filename_map(image_filenames: list):
    prod_to_filename_map = {}
    for img_name in image_filenames:
        prod_id = get_product_id(img_name)
        prod_to_filename_map[prod_id] = prod_to_filename_map.get(prod_id, []) + [img_name]
    return prod_to_filename_map


def get_splits(image_filenames: List[str], test_split: Dict[str, float]) -> Dict[str, List[str]]:
    if test_split:
        product_ids = get_product_list(image_filenames)
        print("Loaded %d product ids..." % (len(product_ids)))

        print("Create train/test splits in the product id space")
        split_prod_id_dict = {}
        split_idxs = {}
        product_idxs = np.arange(len(product_ids), dtype=np.int64)
        assert sum(test_split.values()) == 1.0, "Split probabilities do not sum up to 1"
        for split_name, split_frac in test_split.items():
            size = int(round(split_frac * len(product_ids)))
            split_idxs[split_name] = np.random.choice(product_idxs, size=size, replace=False)
            split_prod_id_dict[split_name] = [product_ids[idx] for idx in split_idxs[split_name]]
            product_idxs = np.setxor1d(product_idxs, split_idxs[split_name])

        print("Test if product ID splits contain all product IDs ...")
        assert set(np.concatenate(list(split_idxs.values()))) == set(np.arange(len(product_ids), dtype=np.int64))
        print("Test if product ID splits are disjoint...")
        for split_name1, split_idxs1 in split_idxs.items():
            for split_name2, split_idxs2 in split_idxs.items():
                if split_name1 != split_name2:
                    assert len(set.intersection(set(split_idxs1), set(split_idxs2))) == 0, "Splits overlap"

        print("Create train/test splits in the image filename space")
        split_dict = {}
        prod_to_filename_map = get_product_to_filename_map(image_filenames)
        for split_name, split_frac in test_split.items():
            split_product_ids = split_prod_id_dict[split_name]
            split_filenames = [prod_to_filename_map[prod_id] for prod_id in split_product_ids]
            # Flatten list of lists
            split_filenames = [item for sublist in split_filenames for item in sublist]
            split_dict[split_name] = split_filenames

        print("Test if filename splits contain all filenames ...")
        assert set(np.concatenate(list(split_dict.values()))) == set(image_filenames)
        print("Test if filename splits are disjoint...")
        for split_name1, split_idxs1 in split_dict.items():
            for split_name2, split_idxs2 in split_dict.items():
                if split_name1 != split_name2:
                    assert len(set.intersection(set(split_idxs1), set(split_idxs2))) == 0, "Splits overlap"

        for split_name, split_filenames in split_dict.items():
            print("Number of files in split '%s': %d" % (split_name, len(split_filenames)))
    else:
        split_dict = {'': image_filenames}

    return split_dict


def write_splis_to_disk(split_dict: Dict[str, List[str]], ssense_dir: str, ssense_dir_resized: str,
                        resolution: int = 256):
    input_metadata_dir = os.path.join(ssense_dir, 'images_metadata')
    for split_name, split_image_filenames in split_dict.items():
        output_dir = os.path.join(ssense_dir_resized, split_name)
        output_metadata_dir = os.path.join(output_dir, 'images_metadata')
        pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
        pathlib.Path(output_metadata_dir).mkdir(parents=True, exist_ok=True)
        for img_name in tqdm(split_image_filenames):
            img = Image.open(os.path.join(img_name))
            img_size, _ = img.size
            if resolution != img_size:
                img = img.resize((resolution, resolution), Image.ANTIALIAS)
            _, tail = os.path.split(img_name)
            img_name_out = os.path.join(output_dir, tail)
            img.save(img_name_out)
            img.close()

            prod_id = get_product_id(img_name)
            copy(os.path.join(input_metadata_dir, prod_id + '.json'), output_metadata_dir)


def create_png_dump_resized(ssense_dir, ssense_dir_resized, resolution: int = 256, test_split=None):
    glob_pattern = os.path.join(ssense_dir, '*.png')
    print("Loading data from: ", ssense_dir)
    image_filenames = sorted(glob.glob(glob_pattern))

    split_dict = get_splits(image_filenames=image_filenames, test_split=test_split)

    write_splis_to_disk(split_dict=split_dict, ssense_dir=ssense_dir,
                        ssense_dir_resized=ssense_dir_resized, resolution=resolution)