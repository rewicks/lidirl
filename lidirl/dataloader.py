################################ PACKAGING AND LOGGING ################################
import pathlib
import logging
import os, sys
from typing import List, Set, Dict, Tuple
import time
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

import torch.utils.data
from torch.nn.utils.rnn import pad_sequence
import math
import random
from collections import Counter

from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from fontTools.ttLib import TTFont

from .augmentations import Codeswitch, URL, ReplaceEmoji

################################# VISUAL REPRESENTATION #################################

class VisRepProcessor():
    """
    This will render text input into an image.
    Each character is it's own image.
    """
    def __init__(self,
                    fonts_path = "/home/hltcoe/rwicks/langid/gcld3-pyfork/fonts",
                    height = 32,
                    width = 32
                    ):
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
        image = Image.new('RGB', (self.width, self.height), color = (255, 255, 255))
        draw = ImageDraw.Draw(image)
        font = self.get_font(char)

        _, _, w, h = draw.textbbox((0, 0), char, font=font)
        draw.text(((self.width-w)/2, (self.height-h)/2), char, font=font, fill=(0,0,0))
        image = image.convert('1')
        tensor = self.convert_tensor(image).tolist()

        return tensor

####################################### DATASET ######################################


# def file_gen(file_name):
#     while True:
#         with open(file_name) as inf:
#             for index, line in enumerate(inf):
#                 if len(line.strip().split('\t')) == 2:
#                     yield line, index

class Dataset(torch.utils.data.IterableDataset):
    def __init__(self,
                    train_files,
                    vocab = None,
                    labels = None,
                    temperature=1.0,
                    type='train',
                    input_type="characters",
                    visual_processor = None,
                    num_workers = 1,
                    augmentations = None,
                    augmentation_prob = 0.0):
        super(Dataset).__init__()

        logger.info(f"Creating a {type} dataset with {input_type} inputs.")

        self.type = type
        self.input_type = input_type
        self.visual_processor = visual_processor
        self.augmentations = augmentations
        self.augmentation_prob = augmentation_prob

        logger.info("Reading training files to calculate sampling probability...")
        self.probabilities = self.get_temperature(train_files, temperature)

        # vocab should not be passed for dev data or for pretrained
        # both vocab and labels should either be passed or be None
        if vocab is None:
            logger.info("Vocab was not passed. Builing vocabulary and label index.")
            self.vocab, self.labels = self.build_vocab_and_labels(train_files)
        else:
            logger.info("Using predefined vocabulary and label index.")
            self.vocab = vocab
            self.labels = labels

        self.length = self.get_length(train_files)

        logger.info(f"Set to {num_workers} parallel workers.")
        self.num_workers = num_workers

        self.train_files = train_files
        self.open_files = {}
        if type == "valid":
            lines = []
            for t in train_files:
                with open(t) as inf:
                    for line in inf:
                        lines.append(line)
            self.open_files["all"] = lines


    def get_length(self, train_files):
        count = 0 
        for fi in train_files:
            with open(fi) as inf:
                for _ in inf:
                    count += 1
        return count

    def __len__(self):
        return self.length

    def get_temperature(self, train_files, temperature):
        """
            This function will read in the input files and calculates the new probability assigned to each file.
            This will be the sampling probability to pull an example from this file during training.
        """
        temperature_probs = {}
        class_counts = {}

        # Iterate to find what the original counts and probabilities are
        for t in train_files:
            class_counts[t] = 0
            with open(t) as inf:
                for _ in inf:
                    class_counts[t] += 1
            logger.info(f"Read {t} for {class_counts[t]} total examples.")

        total_lines = sum(class_counts.values())

        # Find the new sampling probability with temperature
        for class_key, count in class_counts.items():
            temperature_probs[class_key] = math.pow(count/total_lines, temperature)
        new_mass = sum(temperature_probs.values())
        for class_key, count in class_counts.items():
            temperature_probs[class_key] /= new_mass
            logger.info(f"Will sample {class_key} with {temperature_probs[class_key]:.3f}. (Original {count/total_lines:.3f}).")

        return temperature_probs

    def build_vocab_and_labels(self, 
                        train_files,
                        vocab_length = None,
                        character_coverage = 1.0):
        """
            This function will build the associated vocabulary if one is not passed
        """
        def add_label(labels, l):
            if l not in labels:
                labels[l] = len(labels.keys())
            return labels

        vocab = {}
        labels = {}

        vocab_counter = Counter()
        for t in train_files:
            with open(t) as inf:
                for line in inf:
                    line = line.strip().split('\t')
                    if len(line) > 1 and len(line[1].strip()) > 0:
                        labels = add_label(labels, line[0])
                        vocab_counter.update([t for t in line[1]])
                        if self.augmentations is not None and random.random() < self.augmentation_prob:
                            aug = self.get_augmentation()
                            if type(aug) is not Codeswitch:
                                new_text = aug(line[1].strip())
                                vocab_counter.update([t for t in new_text])


        vocab_counter = sorted(list(vocab_counter.items()), key=lambda x: x[1], reverse=True)
        if vocab_length is None:
            vocab_length = character_coverage * len(vocab_counter)


        # We only keep the highest frequency characters in our vocabulary
        # add UNK
        if self.input_type == "characters":
            vocab["[PAD]"] = 0
            vocab["[UNK]"] = 1
            logger.info(f"Total vocab size for character model is {int(vocab_length)}")
            for v, _ in vocab_counter[:int(vocab_length)]:
                vocab[v] = len(vocab.keys())
            
        # Bytes obviously ignores and uses the byte number as the index
        elif self.input_type == "bytes":
            vocab["[PAD]"] = 0
            logger.info("Using byte level model. Vocab size 256.")
            for _ in range(256):
                vocab[str(_)] = _

        # for visual representations we will cache the most frequent characters
        else:
            vocab["[PAD]"] = 1
            logger.info(f"Visual representation inputs. Will precache {int(vocab_length)} most frequent characters.")
            for v, _ in vocab_counter[:int(vocab_length)]:
                vocab[v] = self.visual_processor.build_image(v)

        return vocab, labels

    def get_next_line(self):
        """
            This reads until it finds the correct next line for this worker
        """
        if self.type == "train":
            items = self.probabilities.items()
            selected_file = random.choices([_[0] for _ in items], weights = [_[1] for _ in items], k = 1)[0]
            line = None
            line = next(self.open_files[selected_file])
        else:
            line = next(self.open_files["valid"])
        return line

    def get_augmentation(self):
        """
            Randomly selects an augmentation with assigned probabilities
        """
        p = random.random()
        for augment, prob in self.augmentations:
            if p - prob < 0:
                return augment
            p -= prob
        return augment

    def read_text_labels(self, line):
        """
            Parses the input string to get labels and text
        """
        line = line.strip().split('\t')
        # labels = line[0].split(',')
        text = line[1]
        labels = [line[0] for _ in line[1]]
        return labels, text

    def augment(self, text, labels):
        """
            Applies an augmentation to text and labels if necessary
        """

        augmentation = self.get_augmentation()

        # non linguistic context gets replaced with a None label too
        if type(augmentation) in [URL, ReplaceEmoji]:
            new_text = augmentation(text)
            new_labels = [None]

        elif type(augmentation) is Codeswitch:
            line = self.get_next_line()
            codeswitch_labels, codeswitch_text = self.read_text_labels(line)

            new_text, _ = augmentation(text, codeswitch_text)
            new_labels = labels + [labels[0]] + codeswitch_labels

        else:
            new_text = augmentation(text)
            new_labels = [labels[0] for _ in range(len(new_text))]
            if len(new_text) == 0:
                print(text, labels)
                print(type(augmentation))
                breakpoint()

        return new_text, new_labels

    def file_gen(self, file_name, id):
        file = []
        with open(file_name) as inf:
            for index, line in enumerate(inf):
                if len(line.strip().split('\t')) == 2 and index % self.num_workers == id:
                    file.append(line)
        # logger.info(f"here: {file_name}\t{id}\t{len(file)}")
        while True:
            for line in file:
                yield line
    
    def lines_gen(self, lines, id):
        while True:
            for index, line in enumerate(lines):
                if len(line.strip().split('\t')) == 2 and index % self.num_workers == id:
                    yield line

    def __iter__(self):
        """
            Override of PyTorch IteratorDataset's iterator.
            Each worker randomly selects an input file and reads lines skipping every $num_workers lines.
        """
        worker_info = torch.utils.data.get_worker_info()
        id = worker_info.id if worker_info is not None else 0

        if self.type == "train":
            for t in self.train_files:
                self.open_files[t] = self.file_gen(t, id)
        else:
            self.open_files['valid'] = self.lines_gen(self.open_files['all'], id)

        for _ in range(self.length//self.num_workers):
            # logger.info(f"{id}\t{_}\t{self.length//self.num_workers}\t{self.length}\t{self.num_workers}")
            line = self.get_next_line()

            labels, text = self.read_text_labels(line)
            
            if self.augmentations is not None and random.random() < self.augmentation_prob:
                text, labels = self.augment(text, labels)

            labels = [self.labels.get(l, None) for l in labels]

            if len(text) == 0:
                text = " "

            if self.input_type == "characters":
                out = []
                for t in text:
                    out.append(self.vocab.get(t, 0))
                
            elif self.input_type == "bytes":
                out = []
                for t in text.encode('utf-8'):
                    out.append(t)

            # visrep
            else:
                out = []
                for t in text:
                    vis = self.vocab.get(t, self.visual_processor.build_image(t))
                    out.append(vis)

            yield labels, torch.tensor(out)


############################## COLLATOR FOR PYTORCH DATALOADER ##############################

class BatchCollator():
    def __init__(self, vocab, labels, token_level = False):
        self.vocab = vocab
        self.labels = labels
        self.token_level = token_level
        self.pad_idx = vocab["[PAD]"]

    def pad_texts(self, texts, max_size):
        padded_texts = pad_sequence(texts, batch_first=False, padding_value=self.pad_idx).transpose(0,1)
        return padded_texts

        # padded_texts = [text + [self.pad_idx] * (max_size - len(text)) for text in texts]
        # return torch.tensor(padded_texts)

    def process_labels(self, labels):
        """
        Maps language codes to the appropriate output tensors.
        """

        # if token level, we need outputs for each input character
        if self.token_level:
            out_labels = []
            for label in labels:
                if label[0] is None:
                    token_label = []
                    for l in label:
                        token_label.append(0) # this is appending None for each label?
                    token_label = torch.tensor(token_label)
                else:
                    token_label = torch.tensor(label)
                out_labels.append(token_label)
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
                if len(label) == 0:
                    breakpoint()
                ratio_labels.append([])
                counts = Counter(label)
                for i, l in enumerate(self.labels):
                    percent = counts.get(i, 0) / len(label)
                    ratio_labels[-1].append(percent)
        labels = torch.tensor(ratio_labels)

        return labels
   

    def __call__(self, samples):
        max_size = 0

        texts = []
        labels = []
        for label, text in samples:
            max_size = max(max_size, len(text))
            labels.append(label)
            texts.append(text)

        padded_text = self.pad_texts(texts, max_size=max_size)
        processed_labels = self.process_labels(labels)

        return padded_text, processed_labels

            
class ValidationDataLoader():
    def __init__(self, dataset, batch_size, collate_fn) -> None:
        self.dataset = dataset
        self.batch_size = batch_size
        self.collate_fn = collate_fn
        breakpoint()
        self.data = []
        batch = []
        logger.info("Pre-batching validation set for speed")
        for sample in self.dataset:
            batch.append(sample)
            if len(batch) == self.batch_size:
                self.data.append(self.collate_fn(batch))
            batch = []
        if len(batch) > 0:
            self.data.append(self.collate_fn(batch))

    def __iter__(self):
        for batch in self.data:
            yield batch

def build_datasets(args, augmentations):
    if args.input_type == "visual":
        visproc = VisRepProcessor()
    else:
        visproc = None

    train_dataset = Dataset(train_files=args.train_files,
                                num_workers=args.num_workers,
                                input_type=args.input_type,
                                visual_processor=visproc,
                                temperature=args.temperature,
                                augmentations=augmentations,
                                augmentation_prob=args.augmentation_probability)

    token_level = True if args.pred_type == "token_level" else False

    collate_fn = BatchCollator(train_dataset.vocab,
                                train_dataset.labels,
                                token_level = token_level)


    train = torch.utils.data.DataLoader(train_dataset,
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        collate_fn=collate_fn)

    valid_dataset = {

    }

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
            valid_dataset[cluster] = Dataset(train_files=valid_files,
                                            type="valid",
                                            vocab=train_dataset.vocab,
                                            labels=train_dataset.labels,
                                            input_type=args.input_type,
                                            visual_processor=visproc,
                                            num_workers=args.num_workers)
                                        
    else:
        valid_dataset["valid"] = Dataset(train_files=args.valid_files,
                                            type="valid",
                                            vocab=train_dataset.vocab,
                                            labels=train_dataset.labels,
                                            input_type=args.input_type,
                                            visual_processor=visproc,
                                            num_workers=args.num_workers)
    valid = {}
    for v in valid_dataset:
        valid[v] = torch.utils.data.DataLoader(valid_dataset[v],
                                        batch_size=args.batch_size,
                                        num_workers=args.num_workers,
                                        collate_fn=collate_fn)
        # valid[v] = ValidationDataLoader(valid_dataset[v],
        #                                     batch_size=args.batch_size,
        #                                     collate_fn=collate_fn)

    return train, valid, train_dataset.vocab, train_dataset.labels
    
    