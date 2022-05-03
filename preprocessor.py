import logging
import os
import argparse
import random
import torch
import shutil
import math
import json
import sys
from collections import Counter

logging.basicConfig(format="%(message)s", level=logging.DEBUG)
logger = logging.getLogger('gcld3')

DEBUG = False

def generate_ngrams(input_text, bytes=False, ngram_order=2):
    ngrams = []
    if not bytes:
        for i in range(len(input_text)-ngram_order+1):
            ngrams.append(input_text[i:i+ngram_order])
    return ngrams

class TrainingExample():
    def __init__(self, langid, id, text, hashed_grams, data=None):
        self.langid = langid
        self.id = id
        self.text = text
        self.hashed_grams = hashed_grams
        # import pdb; pdb.set_trace()
        if data is None:
            self.data = self.compute(hashed_grams)
        else:
            self.data = data

    def compute(self, hashed_grams):
        ngrams = [[] for n in hashed_grams] # an array for each order of ngrams
        weights = [[] for n in hashed_grams] # the weight for averages

        for ngram_order in hashed_grams:
            for textid in hashed_grams[ngram_order]:
                hash_id, ngram_weight = hashed_grams[ngram_order][textid]
                ngrams[ngram_order-1].append(hash_id)
                weights[ngram_order-1].append(ngram_weight)
        return (ngrams, weights)

    def size(self):
        return sum([len(_) for _ in self.data])

    def save_object(self):
        return (self.langid, self.id, self.text, self.hashed_grams, self.data)


class Batch():
    def __init__(self):
        self.langids = []
        self.ids = []
        self.texts = []
        self.hashes = []
        self.ngrams = []
        self.size = 0

    def add(self, example):
        self.langids.append(example.langid)
        self.ids.append(example.id)
        self.texts.append(example.text)
        self.hashes.append(example.hashed_grams)
        self.ngrams.append(example.data)
        self.size += example.size()

class TrainingShard():
    def __init__(self):
        self.data = []

    def add_example(self, langid, id, text, hashed_grams):
        self.data.append(TrainingExample(langid, id, text, hashed_grams))

    def pad_batch(self, batch):
        max_size = 0
        for batch_item in batch.ngrams:
            for order in batch_item[0]:
                max_size = max(max_size, len(order))

        padded_ngram_idx = []
        padded_ngram_weights = []
        for batch_item in batch.ngrams:
            padded_item_idx = []
            padded_item_weights = []
            for ids, weights in zip(batch_item[0], batch_item[1]):
                padded_order_idx = []
                padded_order_weights = []

                padded_order_idx += ids
                padded_order_weights += weights
                while len(padded_order_idx) != max_size:
                    padded_order_idx.append(0)
                while len(padded_order_weights) != max_size:
                    padded_order_weights.append(0)
                padded_item_idx.append(padded_order_idx)
                padded_item_weights.append(padded_order_weights)
            padded_ngram_idx.append(padded_item_idx)
            padded_ngram_weights.append(padded_item_weights)
                # padded_ids.append(padded_order)
            # padded_item.append(padded_ids)
            # padded_ngrams.append(padded_item)
        batch.ngrams = (padded_ngram_idx, padded_ngram_weights)

    def shuffle_shard(self):
        random.shuffle(self.data)

    def get_batch(self, batch_size=2000):
        batch = Batch()
        for training_example in self.data:
            if batch.size + training_example.size() > batch_size:
                batch.ids = torch.tensor(batch.ids, dtype=torch.long)
                self.pad_batch(batch)
                batch.ngrams = (torch.tensor(batch.ngrams[0], dtype=torch.long), torch.tensor(batch.ngrams[1]))
                yield batch.langids, batch.ids, batch.texts, batch.hashes, batch.ngrams
                batch = Batch()
            batch.add(training_example)
        batch.ids = torch.tensor(batch.ids, dtype=torch.long)
        # BATCH SIZE X 2 X MAX ORDER X SEQ LENGTH
        self.pad_batch(batch)
        batch.ngrams = (torch.tensor(batch.ngrams[0], dtype=torch.long), torch.tensor(batch.ngrams[1]))
        yield batch.langids, batch.ids, batch.texts, batch.hashes, batch.ngrams

    def save_object(self):
        return [_.save_object() for _ in self.data]

    def load_object(self, data):
        self.data = [TrainingExample(d[0], d[1], d[2], d[3], data=d[4]) for d in data]




class Dataset():
    def __init__(self, directory, ngram_order=3, max_shard_size=100000, batch_size=2000, max_hash_value=512):
        self.ngram_order = ngram_order
        self.working_dir = directory
        self.max_shard_size = max_shard_size
        self.max_ngram_order = ngram_order
        self.max_hash_value = max_hash_value
        self.batch_size = batch_size
        self.labels = {}
        self.shards = []

    def process_data(self, input_files):

        TMP_DIR = os.path.join(self.working_dir, 'tmp/')
        os.makedirs(TMP_DIR, exist_ok=True)
        # get line numbers for shards
        line_ranges = 0
        for infile in input_files:
            for line in open(infile):
                line_ranges += 1

        logger.info(f"Found {line_ranges} training examples in {len(input_files)} files.")

        num_shards = line_ranges // self.max_shard_size
        shards = [open(os.path.join(TMP_DIR, f"shard_{n}"), 'w') for n in range(num_shards+1)]
        logger.info(f"Using {num_shards} shards for training due to max shard size of {self.max_shard_size}")

        for infile in input_files:
            for line in open(infile):
                random_shard = random.choice(shards)
                random_shard.write(line)

        for s in shards:
            s.close()

        self.binarize_shards(num_shards, TMP_DIR)
        shutil.rmtree(TMP_DIR)

    def extract_ngrams(self, text):
        # Currently paying no regard to whitespace
        # Because I think it's imporant when characters start/end words
        ngrams = {}
        for ngram_order in range(1, self.max_ngram_order+1):
            ngrams[ngram_order] = {}
            # counting occurrences of ngrams in text
            extracted_grams = Counter(generate_ngrams(text, ngram_order=ngram_order))
            for gram in extracted_grams:
                # fractional ngram value
                ngrams[ngram_order][gram] = extracted_grams[gram] / sum(extracted_grams.values())
        return ngrams

    def default_hash(self, input):
        hash_value = hash(input)
        return int(hash_value % self.max_hash_value)

    def hash_ngrams(self, unhashed_grams, hash_fn):
        ngrams = {}
        for ngram_order in unhashed_grams:
            ngrams[ngram_order] = {}
            for gram in unhashed_grams[ngram_order]:
                ngrams[ngram_order][gram] = (hash_fn(gram), unhashed_grams[ngram_order][gram])
        if DEBUG:
            logger.debug(f'Processed example: {ngrams}')
        return ngrams

    def process_example(self, langid, text):
        if langid not in self.labels:
            self.labels[langid] = len(self.labels.keys())
        ngrams = self.extract_ngrams(text)
        hashed_ngrams = self.hash_ngrams(ngrams, self.default_hash)
        return hashed_ngrams

    def binarize_shards(self, num_shards, TMP_DIR):
        for shard_id in range(num_shards+1):
            shard = TrainingShard()
            shard_path = os.path.join(TMP_DIR, f"shard_{shard_id}")
            with open(shard_path) as shard_file:
                for example in shard_file:
                    example = example.strip().split('\t')
                    langid = example[0]
                    try:
                        text = '\t'.join(example[1:])
                    except:
                        logging.error("Data is malformatted.")
                        sys.exit(-1)
                    hashed_grams = self.process_example(langid, text)
                    shard.add_example(langid, self.labels[langid], text, hashed_grams)
            shard.shuffle_shard()
            OUTPUT_PATH = os.path.join(self.working_dir, f"shard_{shard_id}.bin")
            torch.save(shard.save_object(), OUTPUT_PATH)
            self.shards.append(OUTPUT_PATH)

    def add_user_defined_ids(self, user_defined_ids):
        for id in user_defined_ids:
            if id not in self.labels:
                self.labels[id] = len(self.labels.keys())

    def __iter__(self):
        for shard_path in self.shards:
            try:
                shard = TrainingShard()
                shard.load_object(torch.load(shard_path))
            except Exception as e:
                print(e)
                logger.error(f"Couldn't find processed shard at {shard_path}! Reprocess your data")
                sys.exit(-1)
            for langid_batch, id_batch, text_batch, hash_batch, ngram_batch in shard.get_batch(self.batch_size):
                yield langid_batch, id_batch, text_batch, hash_batch, ngram_batch

    def save(self):
        output = {
            "ngram_order": self.max_ngram_order,
            "working_dir": self.working_dir,
            "max_shard_size": self.max_shard_size,
            "labels": self.labels,
            "shards": self.shards,
            "batch_size": self.batch_size,
            "max_hash_value": self.max_hash_value
        }
        output_path = os.path.join(self.working_dir, "dataset.json")
        json.dump(output, open(output_path, 'w'), indent=2)


    def load(self):
        input_path = os.path.join(self.working_dir, "dataset.json")
        state = json.load(open(input_path))
        self.max_ngram_order = state["ngram_order"]
        self.working_dir = state["working_dir"]
        self.max_shard_size = state["max_shard_size"]
        self.labels = state["labels"]
        self.shards = state["shards"]
        self.batch_size = state["batch_size"]
        self.max_hash_value = state["max_hash_value"]





def parse_args():
    parser = argparse.ArgumentParser(
        description="CLD3 PREPROCESSOR: something something something.\n"
        "      Example: cld3 something",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--input_files', nargs='*', required=True)
    parser.add_argument('--output_dir', required=True, type=str,
                        help="Output dir to write data files")
    parser.add_argument('--ngram_order', default=3, type=int)


    args = parser.parse_args()
    return args


def main(args):
    processed_dataset = Dataset(args.output_dir, args.ngram_order)
    processed_dataset.process_data(args.input_files)
    processed_dataset.save()


if __name__ == "__main__":
    args = parse_args()
    main(args)
