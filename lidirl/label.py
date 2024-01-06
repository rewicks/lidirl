#!/usr/bin/env python3

"""
    Labels languages from a pretrained model.
"""

################################ PACKAGING AND LOGGING ################################
import pathlib
import logging
import os, sys

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
import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
import json
import time

from lidirl import __version__
from .utils import get_model_path, list_models, MODELS
from .models import CLD3Model, RoformerModel, TransformerModel, Hierarchical
from .dataloader import VisRepProcessor
# from .preprocessor import Processor, NGramProcessor, TrainingShard, VisRepProcessor

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
    processor = None
    if model_dict["model_type"] == "linear-ngram":
        model = CLD3Model(vocab_size=model_dict['vocab_size'],
                            embedding_dim=model_dict['embedding_dim'],
                            hidden_dim=model_dict['hidden_dim'],
                            label_size=model_dict['label_size'],
                            num_ngram_orders=model_dict['num_ngram_orders'],
                            montecarlo_layer=model_dict['montecarlo_layer'])
        # processor = NGramProcessor(
        #     ngram_orders=model_dict['processor']['ngram_orders'],
        #     num_hashes=model_dict['processor']['num_hashes'],
        #     max_hash_value=model_dict['processor']['max_hash_value']
        # )
    elif model_dict["model_type"] == "transformer":
        model = TransformerModel(
            vocab_size=model_dict["vocab_size"],
            embedding_dim=model_dict["embedding_size"],
            hidden_dim=model_dict["hidden_dim"],
            label_size=model_dict["label_size"],
            num_layers=model_dict["num_layers"],
            max_len=model_dict["max_length"],
            nhead=model_dict["nhead"],
            montecarlo_layer=model_dict['montecarlo_layer'],
            visual_inputs=model_dict['input_type'] == "visual",
            convolutions=model_dict["convolutions"]
        )
        if model_dict['input_type'] == "visual":
            processor = VisRepProcessor() 
        else:
            processor = None
            # processor = Processor(vocab=model_dict['vocab'], labels=model_dict['labels'])

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
        if model_dict['input_type'] == "visual":
            processor = VisRepProcessor() 
        else:
            processor = None
    elif model_dict["model_type"] == "hierarchical":
        model = Hierarchical(
                    vocab_size=len(model_dict['vocab']),
                    label_size=model_dict["label_size"],
                    window_size=model_dict["window_size"],
                    stride=model_dict["stride"],
                    embed_dim=model_dict["embed_dim"],
                    hidden_dim=model_dict["hidden_dim"],
                    nhead=model_dict["nhead"],
                    max_length=model_dict["max_length"],
                    num_layers=model_dict["num_layers"],
                    montecarlo_layer=model_dict["montecarlo_layer"]
                    )
    model.load_state_dict(model_dict['weights'])
    model.eval()
    vocab = model_dict["vocab"]
    labels = model_dict["labels"]
    return model, vocab, labels, processor, model_dict['input_type'], model_dict['pred_type']


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
        model, vocab, labels, processor, input_type, pred_type = load_from_checkpoint(model_path)
        self.model = model
        self.vocab = vocab
        self.labels = labels
        self.input_type = input_type
        self.pred_type = pred_type
        self.itos = ["" for _ in labels]
        for l in labels:
            self.itos[labels[l]] = l
        self.processor = processor
        self.args = args
        self.top = args.top

    def label_file(self, input_file, output_file, device):
        """
            Takes input file and writes labels to output (can still be stdin/stdout)
        """

        # moves model to the correct device
        self.model = self.model.to(device)

        # treats input file the same as a training data shard with processing/batching
        batches = self.batches(input_file, batch_size = self.args.batch_size)

        pred_time = 0
        for inputs in batches:
            # labels are actually just unks but to follow pipeline we keep them
            # labels = torch.tensor(labels, dtype=torch.long)

            inputs = inputs.to(device)
            # see what it says

            output = self.model(inputs)

            if self.pred_type == "multilabel":
                probs = F.sigmoid(output)
            else:
                probs = F.softmax(output, dim=-1)
                # probs = torch.exp(output)

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
                ids = torch.topk(prediction, k=self.top).indices
                output_line = ""
                for id in ids:
                    langid_index = id.item()
                    prob = prediction[langid_index].item()
                    output_line += f"{self.itos[langid_index]} {prob:.3f}\t"
                outputs.append(output_line.strip())
        return outputs

    def batches(self, input_file, batch_size=25):
        batch = []
        for line in input_file:
            line = line.strip()
            if len(line) == 0:
                line = " "
            if self.input_type == "bytes":
                batch.append(torch.tensor([int(_) for _ in line.encode('utf-8')]))
            elif self.input_type == "characters":
                batch.append(torch.tensor([self.vocab.get(_, self.vocab["[UNK]"]) for _ in line]))
            else:
                batch.append(torch.tensor([self.vocab.get(_, self.processor.build_image(_)) for _ in line]))
            if len(batch) == batch_size:
                padded_texts = pad_sequence(batch, batch_first=True, padding_value=self.vocab["[PAD]"])
                yield padded_texts
                batch = []
        if len(batch) > 0:
            padded_texts = pad_sequence(batch, batch_first=True, padding_value=self.vocab["[PAD]"])
            yield padded_texts
        

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
    parser.add_argument("--top", default=1, type=int)
    parser.add_argument("--multilabel", default=False, action="store_true")

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
    with torch.no_grad():
        model.label_file(input_file, output_file, device)


def main():

    args = parse_args()

    if args.version:
        print("lidirl", __version__)
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
