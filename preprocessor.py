import logging
import os, sys
import argparse
import random
import torch
import shutil
import math
import json
import sys
from collections import Counter
import xxhash

logging.basicConfig(format="%(message)s", level=logging.DEBUG)
logger = logging.getLogger('gcld3')

DEBUG = False

random.seed(14)
torch.manual_seed(14)

def generate_ngrams(input_text, bytes=False, ngram_order=2):
    ngrams = []
    if not bytes:
        for i in range(len(input_text)-ngram_order+1):
            ngrams.append(' '.join(input_text[i:i+ngram_order]))
    return ngrams

class HashXX32(object):
    def __init__(self, seed, max_hash_value):
        self.h = xxhash.xxh32(seed=seed)
        self.max_hash_value = max_hash_value

    def hash(self, o, offset):
        self.h.reset()
        self.h.update(o)
        hash_value = self.h.intdigest() % self.max_hash_value
        offset *= self.max_hash_value
        return hash_value + offset


class TrainingExample():
    def __init__(self, langid=None, id=None, text=None, hashed_grams=None, data=None):
        self.langid = langid
        self.id = id
        self.text = text
        self.hashed_grams = hashed_grams
        if data is None:
            self.data = self.compute(hashed_grams)
        else:
            self.data = data

    def compute(self, hashed_grams):
        ngrams = [[] for n in hashed_grams] # an array for each order of ngrams
        weights = [[] for n in hashed_grams] # the weight for averages

        for idx, ngram_order in enumerate(hashed_grams):
            if len(hashed_grams[idx]) > 0:
                for t_idx, textid in enumerate(hashed_grams[idx]):
                    hash_ids = [h for h, _ in hashed_grams[idx][t_idx]]
                    ngram_weights = [w for _, w in hashed_grams[idx][t_idx]]
                    ngrams[idx].append(hash_ids)
                    weights[idx].append(ngram_weights)

        return (ngrams, weights)

    def size(self):
        return sum([len(_) for _ in self.data])

    def save_object(self, verbose=True):
        if verbose:
            return (self.langid, self.id, self.text, self.hashed_grams, self.data)
        return (self.id, self.data)


class Batch():
    def __init__(self, verbose=True):
        self.langids = []
        self.ids = []
        self.texts = []
        self.hashes = []
        self.ngrams = []
        self.size = 0

    def add(self, example):
        if example.size() > 0:
            self.langids.append(example.langid)
            self.texts.append(example.text)
            self.ids.append(example.id)
            self.hashes.append(example.hashed_grams)
            self.ngrams.append(example.data)
            self.size += example.size()

class TrainingShard():
    def __init__(self, verbose=True):
        self.data = []
        self.verbose = verbose

    def add_example(self, langid, id, text, hashed_grams):
        self.data.append(TrainingExample(langid, id, text, hashed_grams))

    def pad_batch(self, batch):
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

    def shuffle_shard(self):
        random.shuffle(self.data)

    def get_batch(self, batch_size=2000):
        batch = Batch()
        for training_example in self.data:
            if batch.size + training_example.size() > batch_size:
                batch.ids = torch.tensor(batch.ids, dtype=torch.long)
                self.pad_batch(batch)
                try:
                    batch.ngrams = (torch.tensor(batch.ngrams[0], dtype=torch.long), torch.tensor(batch.ngrams[1]))
                except:
                    for i in batch.ngrams[0]:
                        for j in i:
                            try:
                                torch.tensor(j)
                            except:
                                import pdb; pdb.set_trace()
                yield batch.langids, batch.ids, batch.texts, batch.hashes, batch.ngrams
                batch = Batch()
            batch.add(training_example)
        batch.ids = torch.tensor(batch.ids, dtype=torch.long)
        # BATCH SIZE X 2 X MAX ORDER X SEQ LENGTH
        self.pad_batch(batch)
        batch.ngrams = (torch.tensor(batch.ngrams[0], dtype=torch.long), torch.tensor(batch.ngrams[1]))
        yield batch.langids, batch.ids, batch.texts, batch.hashes, batch.ngrams

    def save_object(self):
        return [_.save_object(verbose=self.verbose) for _ in self.data]

    def load_object(self, data):
        if self.verbose:
            self.data = [TrainingExample(d[0], d[1], d[2], d[3], data=d[4]) for d in data]
        else:
            self.data = [TrainingExample(None, d[0], None, None, data=d[1]) for d in data]


class Preprocessor():
    def __init__(self, ngram_orders=[1,2,3], num_hashes=3, max_hash_value=128):
        self.labels = {"<unk>": 0}
        self.ngram_orders = ngram_orders
        self.max_hash_value = max_hash_value
        self.hashes = []
        for h in range(num_hashes):
            self.hashes.append(HashXX32(seed=h, max_hash_value=max_hash_value))

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

    def process_example(self, langid, text):
        ngrams = self.extract_ngrams(text)
        hashed_ngrams = self.hash_ngrams(ngrams)
        return self.labels.get(langid, 0), hashed_ngrams

    def add_label(self, langid):
        if langid not in self.labels:
            self.labels[langid] = len(self.labels.keys())

    def save_object(self):
        out = {
            "labels": self.labels,
            "ngram_orders": self.ngram_orders,
            "max_hash_value": self.max_hash_value,
            "num_hashes": len(self.hashes)
        }
        return out




class Dataset():
    def __init__(self, directory, ngram_orders=[1,2,3], num_hashes=3, max_shard_size=50000, batch_size=2000, max_hash_value=128, type='train', verbose_data=True):
        self.preprocessor = Preprocessor(ngram_orders=ngram_orders, num_hashes=num_hashes, max_hash_value=max_hash_value)
        self.working_dir = directory
        self.max_shard_size = max_shard_size
        self.max_hash_value = max_hash_value
        self.batch_size = batch_size
        self.labels = {}
        self.shards = []
        self.type = type
        self.verbose_data = verbose_data


    def process_data(self, train_files):

        TMP_DIR = os.path.join(self.working_dir, 'tmp/')
        os.makedirs(TMP_DIR, exist_ok=True)
        # get line numbers for shards
        line_ranges = 0
        for infile in train_files:
            for line in open(infile):
                line_ranges += 1

        logger.info(f"Found {line_ranges} {self.type} examples in {len(train_files)} files.")

        num_shards = (line_ranges // self.max_shard_size) + 1
        shards = [open(os.path.join(TMP_DIR, f"{self.type}.shard_{n}"), 'w') for n in range(num_shards)]
        logger.info(f"Using {num_shards} shards for {self.type} due to max shard size of {self.max_shard_size}")

        for infile in train_files:
            for line in open(infile):
                random_shard = random.choice(shards)
                random_shard.write(line)


        for s in shards:
            s.close()

        logging.info("Binarizing Shards...")
        self.binarize_shards(num_shards, TMP_DIR)
        shutil.rmtree(TMP_DIR)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size


    def binarize_shards(self, num_shards, TMP_DIR):
        for shard_id in range(num_shards):
            logging.info(f"Processing shard id {shard_id} for {self.type}")
            shard = TrainingShard(verbose=self.verbose_data)
            shard_path = os.path.join(TMP_DIR, f"{self.type}.shard_{shard_id}")
            with open(shard_path) as shard_file:
                for example in shard_file:
                    example = example.strip().split('\t')
                    langid = example[0]
                    if len(example) > 1:
                        try:
                            text = example[1].split(' ')
                        except:
                            logging.error("Data is malformatted.")
                            sys.exit(-1)
                        self.preprocessor.add_label(langid)
                        label, hashed_grams = self.preprocessor.process_example(langid, text)
                        shard.add_example(langid, label, text, hashed_grams)
            shard.shuffle_shard()
            OUTPUT_PATH = os.path.join(self.working_dir, f"{self.type}.shard_{shard_id}.bin")
            torch.save(shard.save_object(), OUTPUT_PATH)
            self.shards.append(OUTPUT_PATH)


    def add_user_defined_ids(self, user_defined_ids):
        for id in user_defined_ids:
            if id not in self.labels:
                self.labels[id] = len(self.labels.keys())

    def __iter__(self):
        for shard_path in self.shards:
            try:
                shard = TrainingShard(verbose=self.verbose_data)
                shard.load_object(torch.load(shard_path))
            except Exception as e:
                print(e)
                logger.error(f"Couldn't find processed shard at {shard_path}! Reprocess your data")
                sys.exit(-1)
            for langid_batch, id_batch, text_batch, hash_batch, ngram_batch in shard.get_batch(self.batch_size):
                yield langid_batch, id_batch, text_batch, hash_batch, ngram_batch

    def save(self):
        output = {
            "preprocessor": self.preprocessor.save_object(),
            "working_dir": self.working_dir,
            "max_shard_size": self.max_shard_size,
            "shards": self.shards,
            "batch_size": self.batch_size,
            "verbose": self.verbose_data
        }
        output_path = os.path.join(self.working_dir, f"{self.type}.json")
        json.dump(output, open(output_path, 'w'), indent=2)


    def load(self):
        input_path = os.path.join(self.working_dir, f"{self.type}.json")
        state = json.load(open(input_path))
        prepro_state = state["preprocessor"]
        self.preprocessor = Preprocessor(ngram_orders=prepro_state["ngram_orders"],
                                            num_hashes=prepro_state["num_hashes"],
                                            max_hash_value=prepro_state["max_hash_value"])
        self.preprocessor.labels = prepro_state["labels"]
        self.working_dir = state["working_dir"]
        self.max_shard_size = state["max_shard_size"]
        self.shards = state["shards"]
        self.batch_size = state["batch_size"]
        self.verbose_data = state["verbose"]





def parse_args():
    parser = argparse.ArgumentParser(
        description="CLD3 PREPROCESSOR: something something something.\n"
        "      Example: cld3 something",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--train_files', nargs='*', required=True)
    parser.add_argument('--valid_files', nargs='*', required=True)
    parser.add_argument('--output_dir', required=True, type=str,
                        help="Output dir to write data files")
    parser.add_argument('--ngram_orders', default="1,2,3", type=str)
    parser.add_argument('--max_hash_value', default=128, type=int)
    parser.add_argument('--num_hashes', default=3, type=int)
    parser.add_argument('--max_shard_size', default=200000, type=int)
    parser.add_argument('--verbose_data', default=False, action="store_true")


    args = parser.parse_args()
    return args


def main(args):
    ngram_orders = [int(n) for n in args.ngram_orders.split(',')]

    processed_train_dataset = Dataset(args.output_dir,
                                        ngram_orders=ngram_orders,
                                        num_hashes=args.num_hashes,
                                        max_hash_value=args.max_hash_value,
                                        max_shard_size=args.max_shard_size,
                                        type="train",
                                        verbose_data=args.verbose_data)
    processed_train_dataset.process_data(args.train_files)
    processed_train_dataset.save()

    processed_valid_dataset = Dataset(args.output_dir,
                                        ngram_orders=ngram_orders,
                                        num_hashes=args.num_hashes,
                                        max_hash_value=args.max_hash_value,
                                        max_shard_size=args.max_shard_size,
                                        type="valid",
                                        verbose_data=args.verbose_data)
    processed_valid_dataset.process_data(args.valid_files)
    processed_valid_dataset.save()



if __name__ == "__main__":
    args = parse_args()
    main(args)
