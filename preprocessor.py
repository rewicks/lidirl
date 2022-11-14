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

######################################################################################

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("langid")

######################################################################################

DEBUG = False

random.seed(14)
torch.manual_seed(14)

def generate_ngrams(input_text, ngram_order=2):
    ngrams = []
    for i in range(len(input_text)-ngram_order+1):
        gram = ' '.join([str(_) for _ in input_text[i:i+ngram_order]])
        ngrams.append(gram)
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
    def __init__(self, label=None, text=None):
        self.label = label
        self.text = text

    def size(self):
        return len(self.text)

    def save_object(self):
        return (self.label, self.text)


class Batch():
    def __init__(self):
        self.labels = []
        self.texts = []
        self.size = 0

    def add(self, example):
        if example.size() > 0:
            self.labels.append(example.label)
            self.texts.append(example.text)
            self.size += example.size()

class TrainingShard():
    def __init__(self):
        self.data = []

    def add_example(self, label, text):
        self.data.append(TrainingExample(label, text))

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
                yield batch.labels, batch.texts
                batch = Batch()
            batch.add(training_example)
        if len(batch.labels) > 0:
            yield batch.labels, batch.texts

    def save_object(self):
        return [_.save_object() for _ in self.data]

    def load_object(self, data):
        self.data = []
        for d in data:
            self.data.append(TrainingExample(d[0], d[1]))

    def size(self):
        return len(self.data)

class Processor():
    def __init__(self):
        pass

    def process_example(self, text, device):
        return self.pad_batch(text, device)

    def process_label(self, labels, device):
        return labels.to(device)

    def pad_batch(self, batch, device, max_size=None):
        max_size = 0
        for item in batch:
            max_size = max(max_size, len(item))
        
        new_batch = []
        for item in batch:
            new_batch.append(item)
            while len(new_batch[-1]) < max_size:
                new_batch[-1].append(0)
        return torch.tensor(new_batch).to(device)

    def __call__(self, text, labels, device):
        return self.process_example(text, device), self.process_label(labels, device)

    def save_object(self):
        return {}

class PaddedProcessor(Processor):
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

    def process_labels(self, labels, device):
        if len(labels.shape) == 1:
            retVal = []
            for l in labels:
                retVal.append([])
                for _ in range(self.pad_length):
                    retVal[-1].append(l)
            return torch.tensor(retVal).to(device)
        return labels

    def __call__(self, text, labels, device):
        return self.process_example(text, device), self.process_labels(labels, device)

    def save_object(self):
        return {
            "pad_length": self.pad_length
        }

class NGramProcessor(Processor):
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


class Dataset():
    def __init__(self, directory, bytes=False, max_shard_size=50000, batch_size=2000, type='train', vocab=None):
        self.working_dir = directory
        self.max_shard_size = max_shard_size
        self.batch_size = batch_size
        self.labels = {
            "<unk>": 0
        }
        if vocab is None:
            self.vocab = {
                "<unk>": 0
            }
            self.locked_vocab = False
        else:
            self.vocab = vocab
            self.locked_vocab = True
        self.shards = []
        self.type = type
        self.bytes = bytes
        self.size = 0

    def process_data(self, train_files, temperature=1.0):

        TMP_DIR = os.path.join(self.working_dir, 'tmp/')
        logging.info(f"Making temp directory at {TMP_DIR}")
        os.makedirs(TMP_DIR, exist_ok=True)

        # get line numbers for shards; randomly shuffle after
        line_ranges = 0
        class_counts = {}
        for infile in train_files:
            for line in open(infile):
                line_ranges += 1
                class_counts[line.split('\t')[0]] = class_counts.get(line.split('\t')[0], 0) + 1

        total = sum(class_counts.values())
        class_probabilities = {}
        for class_key, count in class_counts.items():
            class_probabilities[class_key] = math.pow(count/total, temperature)
        
        total = sum(class_probabilities.values())
        class_samples = {}
        for class_key, prob in class_probabilities.items():
            class_probabilities[class_key] = prob / total
            class_samples[class_key] = (class_probabilities[class_key] * line_ranges) / class_counts[class_key]
            sampled_sentences = class_probabilities[class_key] * line_ranges
            logger.info(f"Sampling {class_key} with {class_probabilities[class_key]:.2f} probability. Will use {sampled_sentences} training examples from a total of {class_counts[class_key]}")

        logger.info(f"Found {line_ranges} {self.type} examples in {len(train_files)} files.")

        # figure out the number of shards (new shuffled files) to be created
        num_shards = (line_ranges // self.max_shard_size) + 1
        shards = [open(os.path.join(TMP_DIR, f"{self.type}.shard_{n}"), 'w') for n in range(num_shards)]
        logger.info(f"Using {num_shards} shards for {self.type} due to max shard size of {self.max_shard_size}")

        # randomly distribute training file examples amongst shards
        for infile in train_files:
            for line in open(infile):
                class_id = line.split('\t')[0]
                sample = random.random()
                sampling_rate = class_samples[class_id]
                while sampling_rate >= 1.0:
                    random_shard = random.choice(shards)
                    random_shard.write(line)
                    self.size += 1
                    sampling_rate -= 1
                if sample <= sampling_rate:
                    random_shard = random.choice(shards)
                    random_shard.write(line)
                    self.size += 1

        for s in shards:
            s.close()

        logging.info("Proessing Shards...")
        self.process_shards(num_shards, TMP_DIR)

        logging.info(f"Removing temp directory at {TMP_DIR}")
        shutil.rmtree(TMP_DIR)

    def set_batch_size(self, batch_size):
        self.batch_size = batch_size

    def process_shards(self, num_shards, TMP_DIR):
        for shard_id in range(num_shards):
            logging.info(f"Processing shard id {shard_id} for {self.type}")
            shard = TrainingShard()
            shard_path = os.path.join(TMP_DIR, f"{self.type}.shard_{shard_id}")

            # iterate over plain text examples
            with open(shard_path) as shard_file:
                for example in shard_file:
                    example = example.strip().split('\t')
                    langid = example[0]
                    if len(example) > 1:
                        
                        label = self.labels.get(langid, len(self.labels))
                        self.labels[langid] = label

                        if self.bytes:
                            text = example[1].encode('utf-8')
                        else:
                            text = []
                            for t in example[1]:
                                t = '[SPACE]' if t == " " else t # replace spaces with text for readability
                                self.vocab[t] = self.tok2id(t)
                                text.append(self.vocab[t])
                        
                        shard.add_example(label, text)
            shard.shuffle_shard()
            OUTPUT_PATH = os.path.join(self.working_dir, f"{self.type}.shard_{shard_id}.bin")
            torch.save(shard.save_object(), OUTPUT_PATH)
            self.shards.append(OUTPUT_PATH)

    def add_user_defined_ids(self, user_defined_ids):
        for id in user_defined_ids:
            if id not in self.labels:
                self.labels[id] = len(self.labels.keys())

    def set_labels(self, labels):
        self.labels = labels

    def tok2id(self, tok):
        if not self.locked_vocab:
            self.vocab[tok] = self.vocab.get(tok, len(self.vocab))
        return self.vocab.get(tok, self.vocab["<unk>"])

    def __iter__(self):
        for shard_path in self.shards:
            try:
                shard = TrainingShard()
                shard.load_object(torch.load(shard_path))
            except Exception as e:
                print(e)
                logger.error(f"Couldn't find processed shard at {shard_path}! Reprocess your data")
                sys.exit(-1)
            for label_batch, text_batch in shard.get_batch(self.batch_size):
                yield torch.tensor(label_batch, dtype=torch.long), text_batch

    def save(self):
        output = {
            "bytes": self.bytes,
            "working_dir": self.working_dir,
            "max_shard_size": self.max_shard_size,
            "shards": self.shards,
            "batch_size": self.batch_size,
            "labels": self.labels,
            "vocab": self.vocab,
            "locked": self.locked_vocab,
            "size": self.size
        }
        output_path = os.path.join(self.working_dir, f"{self.type}.json")
        json.dump(output, open(output_path, 'w'), indent=2)


    def load(self):
        input_path = os.path.join(self.working_dir, f"{self.type}.json")
        state = json.load(open(input_path))
        self.bytes = state["labels"]
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
        description="CLD3 PREPROCESSOR: something something something.\n"
        "      Example: cld3 something",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--train_files', nargs='*', required=True)
    parser.add_argument('--valid_files', nargs='*', required=True)
    parser.add_argument('--output_dir', required=True, type=str,
                        help="Output dir to write data files")
    parser.add_argument('--bytes', action="store_true", default=False)
    parser.add_argument('--max_shard_size', default=200000, type=int)
    parser.add_argument('--preset_vocabulary', default=None)
    parser.add_argument('--temperature', type=float, default=1.0)


    args = parser.parse_args()
    return args


def main(args):
    # ngram_orders = [int(n) for n in args.ngram_orders.split(',')]

    if args.preset_vocabulary is not None:
        vocab = json.load(open(args.preset_vocabulary))
    else:
        vocab = None

    processed_train_dataset = Dataset(args.output_dir,
                                        max_shard_size=args.max_shard_size,
                                        type="train",
                                        vocab=None)
    processed_train_dataset.process_data(args.train_files, temperature=args.temperature)
    processed_train_dataset.save()

    processed_valid_dataset = Dataset(args.output_dir,
                                        max_shard_size=args.max_shard_size,
                                        type="valid",
                                        vocab=processed_train_dataset.vocab)
    processed_valid_dataset.set_labels(processed_train_dataset.labels)
    processed_valid_dataset.process_data(args.valid_files)
    processed_valid_dataset.save()



if __name__ == "__main__":
    args = parse_args()
    main(args)
