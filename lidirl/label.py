#!/usr/bin/env python3

"""
    Labels languages from a pretrained model.
"""

################################ PACKAGING AND LOGGING ################################
import pathlib
import logging
import os, sys

if __package__ is None and __name__ == '__main__':
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
import torch
import json

from . import __version__
from .utils import get_model_path, list_models, MODELS
from .models import CLD3Model, RoformerModel, TransformerModel, ConvModel, FlashModel
from .preprocessor import Processor, NGramProcessor, TrainingShard

class DefaultArgs():
    """
        Default class for arguments if not passed from command line
    """
    def __init__(self):
        pass

def load_from_checkpoint(checkpoint_path : str):
    """
        Loads a model from a checkpoint path.
    """
    model_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    if model_dict["model_type"] == "linear-ngram":
        model = CLD3Model(vocab_size=model_dict['vocab_size'],
                            embedding_dim=model_dict['embedding_dim'],
                            hidden_dim=model_dict['hidden_dim'],
                            label_size=model_dict['label_size'],
                            num_ngram_orders=model_dict['num_ngram_orders'],
                            montecarlo_layer=model_dict['montecarlo_layer'])
        processor = NGramProcessor(
            ngram_orders=model_dict['processor']['ngram_orders'],
            num_hashes=model_dict['processor']['num_hashes'],
            max_hash_value=model_dict['processor']['max_hash_value']
        )
    elif model_dict["model_type"] == "transformer":
        model = TransformerModel(
            vocab_size=model_dict["vocab_size"],
            embedding_dim=model_dict["embedding_size"],
            label_size=model_dict["label_size"],
            num_layers=model_dict["num_layers"],
            max_len=model_dict["max_length"],
            nhead=model_dict["nhead"],
            montecarlo_layer=model_dict['montecarlo_layer']
        )    
        processor = Processor()
    elif model_dict["model_type"] == "roformer":
        model = RoformerModel(
                    vocab_size=model_dict["vocab_size"],
                    embedding_dim=model_dict["embedding_dim"],
                    hidden_dim=model_dict["hidden_dim"],
                    label_size=model_dict["label_size"],
                    num_layers=model_dict["num_layers"],
                    max_len=model_dict["max_len"],
                    nhead=model_dict["nhead"],
                    dropout=model_dict["dropout"],
                    montecarlo_layer=model_dict['montecarlo_layer']
        )
        processor = Processor()
    elif model_dict["model_type"] == "convolutional":
        model = ConvModel(vocab_size=model_dict["vocab_size"],
                            label_size=model_dict["label_size"],
                            embedding_dim=model_dict["embedding_size"],
                            conv_min_width=model_dict["conv_min_width"],
                            conv_max_width=model_dict["conv_max_width"],
                            montecarlo_layer=model_dict['montecarlo_layer'])
        processor = Processor()
    elif model_dict["model_type"] == "flashmodel":
        model = FlashModel(
                    vocab_size=model_dict["vocab_size"],
                    embedding_dim=model_dict["embedding_dim"],
                    hidden_dim=model_dict["hidden_dim"],
                    label_size=model_dict["label_size"],
                    num_layers=model_dict["num_layers"],
                    max_len=model_dict["max_len"],
                    nhead=model_dict["nhead"],
                    dropout=model_dict["dropout"],
                    use_rotary=model_dict["use_rotary"],
                    montecarlo_layer=model_dict['montecarlo_layer'])
        processor = Processor()
    model.load_state_dict(model_dict['weights'])
    model.eval()
    vocab = model_dict["vocab"]
    labels = model_dict["labels"]
    return model, vocab, labels, processor


class EvalModel():
    """
        A wrapper to hold model, vocabulary, labels, and text processor to handle all incoming data.
    """
    def __init__(self, model_path, args):
        """
            Initialization function.

            :param model_path: path to the model checkpoint
            :param args: arguments from command line
        """
        model, vocab, labels, processor = load_from_checkpoint(model_path)
        self.model = model
        self.vocab = vocab
        self.labels = labels
        self.itos = ["" for _ in labels]
        for l in labels:
            self.itos[labels[l]] = l
        self.processor = processor
        self.args = args

    def label_file(self, input_file, output_file, device):
        """
            Takes input file and writes labels to output (can still be stdin/stdout)
        """

        # moves model to the correct device
        self.model = self.model.to(device)

        # treats input file the same as a training data shard with processing/batching
        data = self.build_shard(input_file)

        for labels, texts in data.get_batch(self.args.batch_size):

            # labels are actually just unks but to follow pipeline we keep them
            labels = torch.tensor(labels, dtype=torch.long)
            inputs, labels = self.processor(texts, labels, device)

            # see what it says
            output = self.model(inputs)
            probs = torch.exp(output)

            # formats the output appropriates (either complete or 1-best format)
            outputs = self.build_output(probs)

            # prints labels to file
            for output_line in outputs:
                print(output_line, file=output_file)

    def build_output(self, probs):
        """
            Formats the tensor probabilities into either a json or 1-best output
        """
        outputs = []
        for prediction in probs:
            if self.args.complete:
                output_dict = {}
                for langid in self.labels:
                    output_dict[langid] = prediction[self.labels[langid]].item()
                outputs.append(json.dumps(output_dict))
            else:
                langid_index = torch.argmax(prediction).item()
                prob = prediction[langid_index].item()
                output_line = f'{self.itos[langid_index]}\t{prob}'
                outputs.append(output_line)
        return outputs

    def build_shard(self, input_file):
        """
            Builds out the input data the same as training data.
        """
        data = TrainingShard()
        langid = self.labels.get('<unk>', 0)
        for line in input_file:
            if len(line.strip().split()) > 0:
                line = [l for l in line.strip()]
                text = []
                for t in line:
                    t = '[SPACE]' if t == ' ' else t
                    text.append(self.vocab.get(t, 0))
                data.add_example(langid, text)
            else:
                data.add_example(langid, [0 for _ in range(10)])
        return data

def parse_args():
    parser = argparse.ArgumentParser(
        description="LIDIRL: Labels language of input text.\n"
        "      Example: lidirl --model augment-roformer --input newsdata.txt --output langs.txt",
        usage='%(prog)s [-h] [--model MODEL] [--input INPUT] [--output OUTPUT] [OPTIONS]',
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument('--model', '-m', default=None,
                            help="Either name of or path to a pretrained model")
    parser.add_argument('--input', '-i', default=None,
                            help="Input file. Defaults to stdin.")
    parser.add_argument('--output', '-o', default=None,
                            help="Path to output file. Defaults to stdout.")
    parser.add_argument('--complete', action='store_true', default=False,
                            help="Stores whether or not to output complete probability distribution. Defaults to False.")
    parser.add_argument('--cpu', action='store_true', default=False,
                            help="Uses CPU (GPU is default if available)")
    parser.add_argument('--batch_size', '-b', default=500, type=int,
                            help="Batch size to use for evaluation.")

    options = parser.add_argument_group('additional options')
    options.add_argument('--version', '-V', action='store_true', help="Prints LIDIRL version")
    options.add_argument('--download', '-D', action='store_true',
                        help="Downloads model selected via '--model'")
    options.add_argument('--list', '-l', action='store_true',
                        help="Lists available models.")
    options.add_argument('--quiet', '-q', action='store_true',
                        help="Disables logging.")

    args = parser.parse_args()

    return args

def label_langs(args):
    """
        Labels input based on given arguments
    """
    
    # If no input is specified, defaults to stdin
    if args.input is not None:
        input_file = open(args.input, 'r')
    else:
        input_file = sys.stdin

    # If no output is specified, defaults to stdout
    if args.output is not None:
        output_file = open(args.output, 'w')
    else:
        output_file = sys.stdout

    # If GPU is available and CPU not specified, get cuda device
    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # build wrapper system
    model = EvalModel(args.model, args)

    # label file
    model.label_file(input_file, output_file, device)


def main():

    args = parse_args()

    if args.version:
        print("libirl", __version__)
        sys.exit(0)

    if args.download:
        get_model_path(args.model)
        sys.exit(0)

    if args.list:
        list_models()
        sys.exit(0)

    if args.quiet:
        logger.setLevel(logging.ERROR)

    label_langs(args)

if __name__ == "__main__":
    main()