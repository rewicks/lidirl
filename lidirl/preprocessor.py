#!/usr/bin/env python3

"""
    Holds the preprocessor that not only shuffles/stores training data but also extracts ngrams or other features needed during training.
"""

################################ PACKAGING AND LOGGING ################################
import pathlib
import logging
import os, sys
from typing import List, Set, Dict, Tuple

if (__package__ is None or __package__ == "") and __name__ == '__main__':
    parent = pathlib.Path(__file__).absolute().parents[1]
    sys.path.insert(0, str(parent))
    __package__ = 'lidirl'


logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("lidirl")

#################################### FUNCTIONALITY ####################################

import argparse
import random
import shutil
import math
import json
import xxhash

import torch
from collections import Counter

# from git import Repo
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from fontTools.ttLib import TTFont

from .augmentations import Codeswitch, URL, ReplaceEmoji

def generate_ngrams(input_text : str, ngram_order : int = 2) -> List[str]:
    """
        extracts non-overlapping ngrams of specified order
    """
    ngrams = []
    for i in range(len(input_text)-ngram_order+1):
        gram = ' '.join([str(_) for _ in input_text[i:i+ngram_order]])
        ngrams.append(gram)
    return ngrams

class HashXX32(object):
    """
        Class to hash the ngrams. They are seeded sequentially for consistency.
    """
    def __init__(self, seed : int, max_hash_value : int):
        self.h = xxhash.xxh32(seed=seed)
        self.max_hash_value = max_hash_value

    def hash(self, o : str, offset : int):
        """
            Hashes input string and offsets value based on offset (reserved space in the embedding space)
        """
        self.h.reset()
        self.h.update(o)
        
        hash_value = self.h.intdigest() % self.max_hash_value
        offset *= self.max_hash_value
        return hash_value + offset


class TrainingExample():
    """
        Holds one training example.
    """
    def __init__(self, labels : List[int] = None, text : List[int] = None, bytes=False):
        self.labels = labels
        if bytes:
            self.text = [t for t in text]
        self.text = text

    def size(self):
        return len(self.text)

    def save_object(self):
        return (self.labels, self.text)

    def random_truncate(self):
        length = random.choice([_ for _ in range(200, 500)])
        start_index = random.choice([_ for _ in range(0, len(self.text)-length)])
        self.text = self.text[start_index:start_index+length]


class Batch():
    """
        Holds one batch of TrainingExamples (see above)
    """
    def __init__(self):
        self.labels = []
        self.texts = []
        self.size = 0
        self.max_size = 0

    def add(self, example : TrainingExample):
        if example.size() > 0:
            self.labels.append(example.labels)
            self.texts.append(example.text)
            self.size += example.size()
            self.max_size = max(self.max_size, example.size())
            self.size = self.max_size * len(self.labels)


class TrainingShard():
    """
        One shard of training data. Sharding is used in order to shuffle and handle large datasets.
    """
    def __init__(self):
        self.data = []

    def add_example(self, labels, text):
        if len(labels) == 1:
            labels = [labels[0] for _ in text]
        self.data.append(TrainingExample(labels, text))

    def pad_batch(self, batch : Batch):
        """
            Pads batch. Some of this is a little odd because of the odd shape of ngram orders.
        """
        max_size = 0

        pad_value_size = 0
        for batch_item in batch.ngrams:
            for order in batch_item[0]:
                if len(order) > 0:
                    pad_value_size = len(order[0])
                    max_size = max(max_size, len(order))

        padded_ngram_idx = []
        padded_ngram_weights = []
        pad_value = [0 for _ in range(pad_value_size)]
        for batch_item in batch.ngrams:
            padded_item_idx = []
            padded_item_weights = []
            for ids, weights in zip(batch_item[0], batch_item[1]):
                padded_order_idx = []
                padded_order_weights = []

                padded_order_idx += ids
                padded_order_weights += weights
                while len(padded_order_idx) != max_size:
                    padded_order_idx.append(pad_value)
                while len(padded_order_weights) != max_size:
                    padded_order_weights.append(pad_value)
                padded_item_idx.append(padded_order_idx)
                padded_item_weights.append(padded_order_weights)
            padded_ngram_idx.append(padded_item_idx)
            padded_ngram_weights.append(padded_item_weights)

        batch.ngrams = (padded_ngram_idx, padded_ngram_weights)

    def sort_shard(self):
        self.data = sorted(self.data, key=lambda kv: (kv.size(), random.random()), reverse=True)
        # random.shuffle(self.data)

    def get_batch(self, batch_size=2000, augmentations=None, augmentation_prob=0.0):
        """
        This is the main function for generating batches.
        This is where the augmentatinos occur.
        If you add a special type of augmentation then you may need to edit the logic here (similar to Codeswitching)
        """
        batch = Batch()
        for i, training_example in enumerate(self.data):

            # if the training example is too long, we're skipping
            # at some point we should probably log it
            if training_example.size() < batch_size:

                # if there are augmentations, we're going to randomly select one
                if augmentations is not None and random.random() < augmentation_prob:
                    augment = get_augmentation(augmentations)
                else:
                    augment = None
                
                # if an augmentation was selected, we're going to augment the training example
                if augment is not None and batch.size > 0:

                    # these are the different types of augmentations where the label must be None
                    # None labels are treated as uniform outputs
                    if type(augment) in [URL, ReplaceEmoji]:
                        new_text = augment(training_example.text)
                        new_labels = [None]
                        training_example = TrainingExample(labels = new_labels, text=new_text)
                    
                    # another special augmentation is Codeswtiching
                    # codeswitching chooses two training examples so there must be at least 1 previous training example in batch
                    elif type(augment) is Codeswitch and batch.size > 0:

                        # randomly choose a previously (sometimes augmented) training example
                        choice_index = random.choice([_ for _ in range(len(batch.texts))])

                        # if the choice is not one of the non-linguistic augmentations, we'll augment
                        if batch.labels[choice_index][0] is not None:

                            # returns both orders of the concatenation
                            augment_one, augment_two = augment(batch.texts[choice_index], training_example.text)
                            batch.size += (2*len(training_example.text)) + len(batch.texts[choice_index])

                            # adjusts the previous training example and adds the new one
                            batch.texts[choice_index] = augment_one
                            new_labels = training_example.labels + [training_example.labels[0]] + batch.labels[choice_index]
                            batch.labels[choice_index] += [training_example.labels[0]] + training_example.labels

                            training_example = TrainingExample(labels = new_labels, text=augment_two)
                    # every other augmentation is very straightforward
                    else:
                        new_text = augment(training_example.text)
                        new_labels = [training_example.labels[0] for _ in range(len(new_text))]
                        training_example = TrainingExample(labels=new_labels, text=new_text)

                # if there is at least one item in batch and the batch is big enough (with padding), we yield the batch
                if batch.size > 0 and max(batch.size + training_example.size(), training_example.size()*len(batch.texts)) > batch_size: 
                    yield batch.labels, batch.texts
                    batch = Batch()

                # if the training example is too long, we're going to randomly truncate it
                # this is hard coded, but should correspond to the max length of the model
                if training_example.size() > 2048:
                    training_example.random_truncate()
                batch.add(training_example)
        if len(batch.labels) > 0:
            yield batch.labels, batch.texts

    def save_object(self):
        return [_.save_object() for _ in self.data]

    def load_object(self, data):
        self.data = []
        for d in data:
            self.data.append(TrainingExample(d[0], d[1]))
        
        # this makes sure the data is sorted by length
        # I've commented out the version where it's shuffled. I haven't decided which version I like.
        self.data = sorted(self.data, key=lambda kv: (kv.size(), random.random()), reverse=True)
        # random.shuffle(self.data)

    def size(self):
        return len(self.data)

###########################################################################################
#                                      PROCESSORS                                         #   
###########################################################################################

"""
Processors are the objects that take in the data and turn it into the appropriate input 
tensors to give to the model. There's a few variations based on the type of model
"""

class Processor():
    def __init__(self, vocab, labels):
        self.vocab = vocab
        self.labels = labels

    def process_example(self, text, device, is_bytes=False):
        """
        Turns characters (or bytes) into the appropriate input tensors
        """
        tokenized = []
        for t in text:
            if is_bytes:
                tokenized.append([self.vocab[str(_)] for _ in t])
            else:
                tokenized.append([self.vocab.get(_, self.vocab["<unk>"]) for _ in t])
        return self.pad_batch(tokenized, device, is_bytes=is_bytes)

    def process_labels(self, labels, device, token_level=False):
        """
        Maps language codes to the appropriate output tensors.
        """

        # if token level, we need outputs for each input character
        if token_level:
            out_labels = []
            for label in labels:
                if label[0] is None:
                    token_label = []
                    for l in label:
                        token_label.append(0)
                    token_label = torch.tensor(token_label)
                else:
                    token_label = torch.tensor(label)
                out_labels.append(token_label.to(device))
            return out_labels

        # otherwise, we need a single output for the entire input
        else:
            # ratio_labels comes up with the proportion of each language in the input
            # this is used for the non-linguistic augmentations and codeswitching
            # the return value is the distribution
            ratio_labels = []
            for label in labels:
                if label is None:
                    label = [_ for _ in self.labels]
                ratio_labels.append([])
                counts = Counter(label)
                for i, l in enumerate(self.labels):
                    percent = counts.get(i, 0) / len(label)
                    ratio_labels[-1].append(percent)
        labels = torch.tensor(ratio_labels)

        return labels.to(device)

    def convert_bytes(self, batch):
        # turns the bytes into an array of bytes
        out = []
        for item in batch:
            out.append([t for t in item])
        return out

    def pad_batch(self, batch, device, max_size=None, is_bytes=False):
        # pads batch all to the same length
        max_size = 0

        if is_bytes:
            batch = self.convert_bytes(batch)

        for item in batch:
            max_size = max(max_size, len(item))
        
        new_batch = []
        for item in batch:
            new_batch.append(item)
            while len(new_batch[-1]) < max_size:
                new_batch[-1].append(0)
        return torch.tensor(new_batch).to(device)

    def __call__(self, text, labels, device, is_bytes=False, token_level=False):
        return self.process_example(text, device, is_bytes=is_bytes), self.process_labels(labels, device, token_level=token_level)

    def save_object(self):
        return {}

class VisRepProcessor(Processor):
    """
    This processor will render text input into an image.
    Each character is it's own image.
    """
    def __init__(self, 
                labels,
                fonts_path="/home/hltcoe/rwicks/langid/gcld3-pyfork/fonts",
                height=32,
                width=32):
        self.fonts_path = fonts_path
        self.fonts = []
        for _, _, f in os.walk(fonts_path):
            for fi in f:
                if ".ttf" in fi or ".otf" in fi:
                    font_path = os.path.join(fonts_path, fi)
                    self.fonts.append((ImageFont.truetype(font_path, 32), TTFont(font_path), 0))
        self.convert_tensor = transforms.ToTensor()
        self.height = height
        self.width = width
        self.common_char = {}
        self.labels = labels

    def get_font(self, char):
        # looks for the first available font that has the character
        char = ord(char)

        break_out = False
        retval = None
        for id, (i, f, c) in enumerate(self.fonts):
            for cmap in f['cmap'].tables:
                if cmap.isUnicode():
                    if char in cmap.cmap:
                        retval = i
                        break_out = True
                        self.fonts[id] = (i, f, c+1)
                        break
            if break_out:
                break
        if retval is None:
            retval = self.fonts[0][0]
        self.fonts = sorted(self.fonts, key=lambda kv: kv[2], reverse=True)
        return retval

    def build_image(self, char):
        if char in self.common_char:
            self.common_char[char] = (self.common_char[char][0], self.common_char[char][1]+1)
            return self.common_char[char][0]
        image = Image.new('RGB', (self.width, self.height), color = (255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = self.get_font(char)

        _, _, w, h = draw.textbbox((0, 0), char, font=font)
        draw.text(((self.width-w)/2, (self.height-h)/2), char, font=font, fill=(0,0,0))
        image = image.convert('1')
        tensor = self.convert_tensor(image)

        # this is some logic for keeping the most common characters in memory so we don't have to re-render it
        # this is a bit of a hack, but I think it works
        self.common_char[char] = (tensor, self.common_char.get(char, (None, 0))[1] + 1)
        if len(self.common_char) > 10000:
            self.common_char = {k: v for k, v in sorted(self.common_char.items(), key=lambda item: item[1][1])[:10000]}
        return tensor

    def pad_batch(self, batch, device):
        max_size = 0
        for item in batch:
            max_size = max(len(item), max_size)
        
        new_batch = []
        for item in batch:
            new_batch.append([])
            for char in item:
                new_batch[-1].append(self.build_image(char))
            while len(new_batch[-1]) != max_size:
                new_batch[-1].append(self.build_image(" "))

        out = torch.cat([torch.unsqueeze(torch.cat(n), dim=0) for n in new_batch]).to(device)
        return out
        
    def process_example(self, text, device, is_bytes=False):
        return self.pad_batch(text, device)

    def process_labels(self, labels, device, token_level=False):
        # this is re-implemented, but I think it's the same as the superclass so it can probably be deleted.
        if token_level:
            out_labels = []
            for label in labels:
                if label[0] is None:
                    token_label = []
                    for l in label:
                        token_label.append(0)
                    token_label = torch.tensor(token_label)
                else:
                    token_label = torch.tensor(label)
                out_labels.append(token_label.to(device))
            return out_labels
        else:
            ratio_labels = []
            for label in labels:
                if label is None:
                    label = [_ for _ in self.labels]
                ratio_labels.append([])
                counts = Counter(label)
                for i, l in enumerate(self.labels):
                    percent = counts.get(i, 0) / len(label)
                    ratio_labels[-1].append(percent)
        labels = torch.tensor(ratio_labels)

        return labels.to(device)

    def __call__(self, text, labels, device, is_bytes=False, token_level=False):
        return self.process_example(text, device, is_bytes=is_bytes), self.process_labels(labels, device)

    def save_object(self):
        return {
            "fonts_path": self.fonts_path
        }
        

class PaddedProcessor(Processor):
    """
    This processor is for the UNET model. It pads the input to a fixed length.
    """
    def __init__(self, pad_length):
        self.pad_length = pad_length

    def pad_batch(self, batch, device):
        new_batch = []
        for item in batch:
            new_batch.append(item[:self.pad_length])
            while len(new_batch[-1]) < self.pad_length:
                new_batch[-1].append(0)
        return torch.tensor(new_batch).to(device)

    def process_example(self, text, device):
        return self.pad_batch(text, device)

    # def process_labels(self, labels, device):
    #     if len(labels.shape) == 1:
    #         retVal = []
    #         for l in labels:
    #             retVal.append([])
    #             for _ in range(self.pad_length):
    #                 retVal[-1].append(l)
    #         return torch.tensor(retVal).to(device)
    #     return labels

    def __call__(self, text, labels, device, augmentations, augmentation_prob):
        return self.process_example(text, device, augmentations, augmentation_prob), self.process_labels(labels, device)

    def save_object(self):
        return {
            "pad_length": self.pad_length
        }
    

class NGramProcessor(Processor):
    """
    This one is for the NGram model. It extracts ngrams and hashes them.
    """
    def __init__(self, ngram_orders=[1,2,3], num_hashes=3, max_hash_value=128):
        self.ngram_orders = ngram_orders
        self.max_hash_value = max_hash_value
        self.hashes = []
        for h in range(num_hashes):
            self.hashes.append(HashXX32(seed=h, max_hash_value=max_hash_value))

    def process_example(self, langid, text):
        ngrams = self.extract_ngrams(text)
        hashed_ngrams = self.hash_ngrams(ngrams)
        return self.labels.get(langid, 0), hashed_ngrams

    def process_example(self, text, device):
        batched_ngrams = []
        batched_weights = []
        max_size = 0
        for instance in text:

            ngrams = self.extract_ngrams(instance)
            hashed_ngrams = self.hash_ngrams(ngrams)

            ngrams = []
            weights = []
            for _ in hashed_ngrams:
                ngrams.append([])
                weights.append([])

            for idx, ngram_order in enumerate(hashed_ngrams):
                for tok_idx, _ in enumerate(ngram_order):
                    hash_ids = []
                    ngram_weights = []
                    for h, w in hashed_ngrams[idx][tok_idx]:
                        hash_ids.append(h)
                        ngram_weights.append(w)
                    ngrams[idx].append(hash_ids)
                    weights[idx].append(ngram_weights)
                max_size = max(max_size, max([len(n) for n in ngrams]))

            batched_ngrams.append(ngrams)
            batched_weights.append(weights)
        return self.pad_batch((batched_ngrams, batched_weights), device, max_size)

    def pad_batch(self, batch, device, max_size=None):

        batch_padded_ngrams = []
        batch_padded_weights = []
        pad_value = [0 for _ in range(len(self.hashes))]

        for ngrams, weights in zip(batch[0], batch[1]):
            padded_ngrams = []
            padded_weights = []
            for ngram_order, weight_order in zip(ngrams, weights):
                padded_ngram_order = []
                padded_weight_order = []

                padded_ngram_order += ngram_order
                padded_weight_order += weight_order
                while len(padded_ngram_order) != max_size:
                    padded_ngram_order.append(pad_value)
                while len(padded_weight_order) != max_size:
                    padded_weight_order.append(pad_value)
            
                padded_ngrams.append(padded_ngram_order)
                padded_weights.append(padded_weight_order)
            
            batch_padded_ngrams.append(padded_ngrams)
            batch_padded_weights.append(padded_weights)
        


        out = (torch.tensor(batch_padded_ngrams, dtype=torch.long).to(device), torch.tensor(batch_padded_weights).to(device))
        return out


    def extract_ngrams(self, text):
        # Currently paying no regard to whitespace
        # Because I think it's imporant when characters start/end words
        ngrams = {}
        for ngram_order in self.ngram_orders:
            ngrams[ngram_order] = {}
            # counting occurrences of ngrams in text
            extracted_grams = Counter(generate_ngrams(text, ngram_order=ngram_order))
            for gram in extracted_grams:
                # fractional ngram value
                ngrams[ngram_order][gram] = extracted_grams[gram] / sum(extracted_grams.values())
        return ngrams

    def hash_ngrams(self, unhashed_grams):
        ngrams = []
        for idx, ngram_order in enumerate(unhashed_grams):
            ngrams.append([])
            for gram in unhashed_grams[ngram_order]:
                ngrams[-1].append([])
                for h in self.hashes:
                    ngrams[-1][-1].append((h.hash(gram, offset=idx), unhashed_grams[ngram_order][gram]))
        if DEBUG:
            logger.debug(f'Processed example: {ngrams}')
        return ngrams

    def add_label(self, langid):
        if langid not in self.labels:
            self.labels[langid] = len(self.labels.keys())

    def set_itos(self):
        self.itos = ["" for l in self.labels]
        for label in self.labels:
            self.itos[self.labels[label]] = label

    def save_object(self):
        out = {
            "ngram_orders": self.ngram_orders,
            "max_hash_value": self.max_hash_value,
            "num_hashes": len(self.hashes)
        }
        return out

#####################################################################################################
#                                       DATASET CLASS                                               #
#####################################################################################################


class Dataset():
    def __init__(self, 
                    directory,
                    bytes=False,
                    visual=False,
                    max_shard_size=50000,
                    batch_size=2000,
                    type='train',
                    group_name = None,
                    vocab=None,
                    character_coverage=1.0,
                    vocab_length=None):
        """
        Dataset class for loading and processing data

        Args:
            directory (str): Output directory for the data
            bytes (bool, optional): Whether to use byte-level encoding. Defaults to False.
            visual (bool, optional): Whether to use visual encoding. Defaults to False.
            max_shard_size (int, optional): Maximum number of examples per shard. Defaults to 50000.
            batch_size (int, optional): Batch size for training. Defaults to 2000.
            type (str, optional): Type of data to load. Defaults to 'train'.
            group_name (str, optional): Name of the validation group. Used during smart clustering. Defaults to None.
            vocab (dict, optional): Vocabulary to use. Defaults to None.
            character_coverage (float, optional): Character coverage to use. Defaults to 1.0.
            vocab_length (int, optional): Length of the vocabulary. Defaults to None.
        """
            
        self.working_dir = directory
        self.max_shard_size = max_shard_size
        self.batch_size = batch_size
        self.labels = {
            "<unk>": 0
        }
        if bytes:
            self.vocab = dict(enumerate([_ for _ in range(256)]))
            self.locked_vocab = True
        elif visual:
            self.vocab = {}
            self.locked_vocab = True
        elif vocab is None:
            self.vocab = {
                "<unk>": 0
            }
            self.locked_vocab = False
        else:
            self.vocab = vocab
            self.locked_vocab = True
        self.locked_labels = False
        
        self.shards = []
        self.type = type
        self.group_name = group_name
        self.bytes = bytes
        self.visual = visual
        self.size = 0
        self.character_coverage = character_coverage
        self.vocab_length=vocab_length

    def process_data(self, train_files, temperature=1.0):
        """
            Reads in the input files. Assigns labels. Performs sampling if necessary. Writes examples to randomly assigned data shards.
        """

        # Create a temporary directory for processing data
        TMP_DIR = os.path.join(self.working_dir, 'tmp/')
        logger.info(f"Making temp directory at {TMP_DIR}")
        os.makedirs(TMP_DIR, exist_ok=True)

        # get line numbers for shards; randomly shuffle after
        line_ranges = 0
        # also keep track of how many examples for each class exist
        class_counts = {}

        # when the vocabulary is locked, it cannot be adjusted
        if not self.locked_vocab:
            vocab_counter = Counter()

        # iterate over the training files
        for infile in train_files:
            for line in open(infile):

                # split the line into class and text
                line = line.split('\t')

                # catch any odd cases where the lang wasn't added or the line was empty
                if len(line) > 1:
                    line_ranges += 1

                    # class counts are used for temperature sampling
                    class_counts[line[0]] = class_counts.get(line[0], 0) + 1

                    # if we haven't locked the vocabulary, we need to add all these characters to the vocab
                    if not self.locked_vocab:
                        if self.bytes:
                            text = line[1].encode('utf-8')
                        else:
                            text = line[1]
                        vocab_counter.update([t for t in text])

                    # if labels are not locked, add this as a label
                    if not self.locked_labels:
                        self.add_label(line[0])

        # now if the vocab isn't locked, we need to construct it based on character coverage and size
        if not self.locked_vocab:
            vocab_counter = sorted(list(vocab_counter.items()), key=lambda x: x[1], reverse=True)
            if self.bytes:
                vocab_size = len(vocab_counter)
            elif self.vocab_length is None:
                vocab_size = self.character_coverage * len(vocab_counter)
            else:
                vocab_size = self.vocab_length
            for v, _ in vocab_counter[:int(vocab_size)]:
                self.vocab[v] = len(self.vocab.keys())
            self.locked_vocab = True

        # counts the total number of examples
        total = sum(class_counts.values())

        # only training data gets temperature-sampled
        if self.type == "train":

            # keep track of sampling probabilities after adjusted temperature
            class_probabilities = {}
            for class_key, count in class_counts.items():
                class_probabilities[class_key] = math.pow(count/total, temperature)

            # new total based off summed values after upsampling (temperature sampling)
            total = sum(class_probabilities.values())
            
            logger.info(f"Found {line_ranges} {self.type} examples in {len(train_files)} files.")

            # keep track of how many class sampes
            class_samples = {}
            for class_key, prob in class_probabilities.items():
                class_probabilities[class_key] = prob / total
                class_samples[class_key] = (class_probabilities[class_key] * line_ranges) / class_counts[class_key]
                sampled_sentences = int(class_probabilities[class_key] * line_ranges)
                logger.info(f"Sampling {class_key} with {class_probabilities[class_key]:.2f} probability. Will use {sampled_sentences} training examples from a total of {class_counts[class_key]}")
        else:
            for class_key, count in class_counts.items():
                logger.info(f"Found {class_key} with {count} validation examples.")


        # figure out the number of shards (new shuffled files) to be created
        num_shards = int(line_ranges // self.max_shard_size) + 1
        shards = [open(os.path.join(TMP_DIR, f"{self.type}.shard_{n}"), 'w') for n in range(num_shards)]
        logger.info(f"Using {num_shards} shards for {self.type} due to max shard size of {self.max_shard_size}")

        # randomly distribute training file examples amongst shards
        for infile in train_files:
            for line in open(infile):
                if len(line.split('\t')) > 1:
                    class_id = line.split('\t')[0]
                    # if this is a training file, we sample it
                    if self.type == "training":
                        sample = random.random()
                        sampling_rate = class_samples[class_id]
                        # is sampling rate is greater than one, automatically sampled
                        while sampling_rate >= 1.0:
                            random_shard = random.choice(shards)
                            random_shard.write(line)
                            self.size += 1
                            sampling_rate -= 1
                        # for any leftover probability mass, only upsample this example that percent of the tiem
                        if sample <= sampling_rate:
                            random_shard = random.choice(shards)
                            random_shard.write(line)
                            self.size += 1
                    # if it's validation, just copy everything over
                    else:
                        random_shard = random.choice(shards)
                        random_shard.write(line)
                        self.size += 1

        for s in shards:
            s.close()

        logger.info("Proessing Shards...")
        self.process_shards(num_shards, TMP_DIR)

        logger.info(f"Removing temp directory at {TMP_DIR}")
        shutil.rmtree(TMP_DIR)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def process_shards(self, num_shards, TMP_DIR):
        """
        Processes the shards created by the data loader. 
        It used to do more preprocessing with token indexes, but due to the augmentations, we no longer do that.
        """

        for shard_id in range(num_shards):
            logger.info(f"Processing shard id {shard_id} for {self.type}")
            shard = TrainingShard()
            shard_path = os.path.join(TMP_DIR, f"{self.type}.shard_{shard_id}")

            # iterate over plain text examples
            with open(shard_path) as shard_file:
                for example in shard_file:
                    example = example.strip().split('\t')

                    # this is eventually allowing for multiple labels as training
                    # we have not tested, so I would not guarantee it's functionality
                    langid = example[0].split(',')
                    if len(example) > 1:
                        labels = []
                        for l in langid:
                            labels.append(self.labels.get(l, "<unk>"))

                        # if it's bytes, we'll pre-encode (the augmentations will undo this anyway)
                        if self.bytes:
                            text = example[1].encode('utf-8')
                        else:
                            text = example[1]
    
                        shard.add_example(labels, text)
            
            # batch size corresponds better to number of tokens. In order to not have examples with heavy pad values, we sort by length.
            # this may have an unintended effect by putting a significant number of the same language in the same batch. 
            shard.sort_shard() # sorts by length in order to get better batching. Shards are already randomly distributed across data
            
            # save the shard to disk
            if self.group_name is None:
                OUTPUT_PATH = os.path.join(self.working_dir, f"{self.type}.shard_{shard_id}.bin")
            else:
                OUTPUT_PATH = os.path.join(self.working_dir, f"{self.type}.{self.group_name}.shard_{shard_id}.bin")
            torch.save(shard.save_object(), OUTPUT_PATH)
            self.shards.append(OUTPUT_PATH)

    def add_user_defined_ids(self, user_defined_ids):
        for id in user_defined_ids:
            if id not in self.labels:
                self.labels[id] = len(self.labels.keys())

    def set_labels(self, labels):
        self.labels = labels
        self.locked_labels = True

    def add_label(self, langid):
        if langid not in self.labels:
            self.labels[langid] = len(self.labels.keys())

    def tok2id(self, tok):
        if not self.locked_vocab:
            self.vocab[tok] = self.vocab.get(tok, len(self.vocab))
        tok = self.vocab.get(tok, self.vocab["<unk>"])
        return tok

    def enumerate(self, augmentations=None, augmentation_prob=0.0):
        index = 1
        for shard_path in self.shards:
            try:
                shard = TrainingShard()
                shard.load_object(torch.load(shard_path))
            except Exception as e:
                print(e)
                logger.error(f"Couldn't find processed shard at {shard_path}! Reprocess your data")
                sys.exit(-1)
            for label_batch, text_batch in shard.get_batch(self.batch_size, augmentations=augmentations, augmentation_prob=augmentation_prob):
                yield index, (label_batch, text_batch)
                # yield index, (torch.tensor(label_batch, dtype=torch.long), text_batch)
                index += 1

    def save(self):
        output = {
            "bytes": self.bytes,
            "visual": self.visual,
            "working_dir": self.working_dir,
            "max_shard_size": self.max_shard_size,
            "shards": self.shards,
            "batch_size": self.batch_size,
            "labels": self.labels,
            "vocab": self.vocab,
            "locked": self.locked_vocab,
            "size": self.size
        }
        if self.group_name is None:
            output_path = os.path.join(self.working_dir, f"{self.type}.json")
        else:
            output_path = os.path.join(self.working_dir, f"{self.type}.{self.group_name}.json")
        json.dump(output, open(output_path, 'w'), indent=2)


    def load(self):
        if self.group_name is not None:
            input_path = os.path.join(self.working_dir, f"{self.type}.{self.group_name}.json")
        else:
            input_path = os.path.join(self.working_dir, f"{self.type}.json")
        state = json.load(open(input_path))
        self.bytes = state["bytes"]
        self.visual = state.get("visual", False)
        self.working_dir = state["working_dir"]
        self.max_shard_size = state["max_shard_size"]
        self.shards = state["shards"]
        self.batch_size = state["batch_size"]
        self.labels = state["labels"]
        self.vocab = state["vocab"]
        self.locked_vocab = state["locked"]
        self.size = state["size"]

    def __len__(self):
        return self.size


def parse_args():
    parser = argparse.ArgumentParser(
        description="PREPROCESSOR: something something something.\n"
        "      Example: something",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--train_files', nargs='*', required=True,
                            help="Path(s) to tsv training files. Label should be in first column, text in second")
    parser.add_argument('--valid_files', nargs='*', required=True,
                            help="Path(s) to tsv validation files. Label should be first column, text in second.")
    parser.add_argument('--smart_group_validation_files', default=False, action="store_true",
                            help="If passed, will group validation sets into separately logged clusters based on each file's parent directory")

    parser.add_argument('--output_dir', required=True, type=str,
                        help="Output dir to write data files")
    parser.add_argument('--bytes', action="store_true", default=False,
                            help="Stores whether to convert text into bytes for model input.")
    parser.add_argument("--visual", action="store_true", default=False,
                            help="")
    parser.add_argument('--max_shard_size', default=1000000, type=int,
                            help="Stores the maximum size for one shard (an intermediate file loaded during training) of data. \
                                Bigger size means fewer binaries will be written. Smaller means training uses less CPU memory.")
    parser.add_argument('--character_coverage', default=0.95, type=float)
    parser.add_argument('--vocab_length', default=16000, type=int)
    parser.add_argument('--preset_vocabulary', default=None,
                            help="Path to a preset vocabulary. This is just the dictionary mapping of characters to ids for training. \
                                In all likelihood this is unnecessary as we haven't implemented BPE")
    parser.add_argument('--temperature', type=float, default=1.0,
                            help="The value to be used for temperature sampling. Between 0 and 1. Smaller upsamples low resource languages more. 1.0 is no upsampling.")
    parser.add_argument("--seed", type=int, default=14,
                            help="Random seed to be used for reproducibility.")
    parser.add_argument("--debug", default=False, action="store_true",
                            help="Theoretically this allows for more logging, but I haven't used it much...")


    args = parser.parse_args()
    return args


def get_augmentation(augmentations):
    p = random.random()
    for augment, prob in augmentations:
        if p - prob < 0:
            return augment
        p -= prob
    return augment


def main(args):


    # global flag used for debugging
    global DEBUG
    DEBUG = args.debug

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # If a specific vocabulary is required, load it
    if args.preset_vocabulary is not None:
        vocab = json.load(open(args.preset_vocabulary))
    else:
        vocab = None

    # Create the training dataset
    processed_train_dataset = Dataset(args.output_dir,
                                        max_shard_size=args.max_shard_size,
                                        type="train",
                                        bytes=args.bytes,
                                        visual=args.visual,
                                        vocab=vocab,
                                        vocab_length=args.vocab_length)
    processed_train_dataset.process_data(args.train_files, temperature=args.temperature)
    processed_train_dataset.save()

    # We also allow for multiple validation sets. Because of this, we may iterate over each validation set
    if args.smart_group_validation_files:
        clusters = {}
        """
            Smart grouping works by assuming that all files within a parent directory have the same cluster
        """
        for fi in args.valid_files:
            parent_path = os.path.abspath(fi).split('/')[-2]
            clusters[parent_path] = clusters.get(parent_path, []) + [fi]

        # Create a unique validation set for each cluster
        for cluster, valid_files in clusters.items():
            processed_valid_dataset = Dataset(args.output_dir,
                                                max_shard_size=args.max_shard_size,
                                                type="valid",
                                                bytes=args.bytes,
                                                visual=args.visual,
                                                group_name=cluster,
                                                vocab=processed_train_dataset.vocab)
            processed_valid_dataset.set_labels(processed_train_dataset.labels)
            processed_valid_dataset.process_data(valid_files)
            processed_valid_dataset.save()
    # otherwise, all validation data belongs in the same dataset
    else:
        processed_valid_dataset = Dataset(args.output_dir,
                                            max_shard_size=args.max_shard_size,
                                            type="valid",
                                            bytes=args.bytes,
                                            visual=args.visual,
                                            vocab=processed_train_dataset.vocab)
        processed_valid_dataset.set_labels(processed_train_dataset.labels)
        processed_valid_dataset.process_data(args.valid_files)
        processed_valid_dataset.save()



if __name__ == "__main__":
    args = parse_args()
    main(args)
