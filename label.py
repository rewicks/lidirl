import argparse
import torch
import json
import sys

from models import CLD3Model, RoformerModel, TransformerModel, ConvModel
from preprocessor import Processor, NGramProcessor, TrainingShard


import logging

logging.basicConfig(format="%(message)s", level=logging.DEBUG)
logger = logging.getLogger('gcld3')

class DefaultArgs():
    def __init__(self):
        pass

def load_from_checkpoint(checkpoint_path):
    model_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    if model_dict["model_type"] == "linear-ngram":
        model = CLD3Model(vocab_size=model_dict['vocab_size'],
                            embedding_dim=model_dict['embedding_dim'],
                            hidden_dim=model_dict['hidden_dim'],
                            label_size=model_dict['label_size'],
                            num_ngram_orders=model_dict['num_ngram_orders'])
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
            nhead=model_dict["nhead"]
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
                    dropout=model_dict["dropout"]
        )
        processor = Processor()
    elif model_dict["model_type"] == "convolutional":
        model = ConvModel(vocab_size=model_dict["vocab_size"],
                            label_size=model_dict["label_size"],
                            embedding_dim=model_dict["embedding_size"],
                            conv_min_width=model_dict["conv_min_width"],
                            conv_max_width=model_dict["conv_max_width"])
        processor = Processor()
    model.load_state_dict(model_dict['weights'])
    model.eval()
    vocab = model_dict["vocab"]
    labels = model_dict["labels"]
    return model, vocab, labels, processor


class EvalModel():
    def __init__(self, model_path, args):
        model, vocab, labels, processor = load_from_checkpoint(model_path)
        self.model = model
        self.vocab = vocab
        self.labels = labels
        self.itos = ["" for _ in labels]
        for l in labels:
            self.itos[labels[l]] = l
        self.processor = processor
        # self.preprocessor = load_preprocessor(self.model, preprocessor_path=args.preprocessor_path)
        self.args = args


    def label_file(self, input_file, output_file, device):
        self.model = self.model.to(device)
        data = self.build_shard(input_file)

        for labels, texts in data.get_batch(self.args.batch_size):

            labels = torch.tensor(labels, dtype=torch.long)
            inputs, labels = self.processor(texts, labels, device)

            output = self.model(inputs)
            
            probs = torch.exp(output)
            try:
                outputs = self.build_output(probs)
            except:
                import pdb; pdb.set_trace()
            for output_line in outputs:
                print(output_line, file=output_file)

    def build_output(self, probs):
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
        data = TrainingShard()
        langid = self.labels.get('<unk>', 0)
        for line in input_file:
            line = [l for l in line.strip()]
            text = []
            for t in line:
                t = '[SPACE]' if t == ' ' else t
                text.append(self.vocab.get(t, 0))
            data.add_example(langid, text)
            # label, hashed_grams = self.preprocessor.process_example(langid, line.strip().split())
            # data.add_example(langid, label, line.strip(), hashed_grams)
        return data




def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--model', '-m', default=None)
    parser.add_argument('--preprocessor_path', default=None)
    parser.add_argument('--input', '-i', default=None)
    parser.add_argument('--output', '-o', default=None)
    parser.add_argument('--complete', action='store_true', default=False)
    parser.add_argument('--cpu', action='store_true', default=False)
    parser.add_argument('--batch_size', default=2000, type=int)

    args = parser.parse_args()

    return args

def label_langs(args):
    
    if args.input is not None:
        input_file = open(args.input, 'r')
    else:
        input_file = sys.stdin

    if args.output is not None:
        output_file = open(args.output, 'w')
    else:
        output_file = sys.stdout


    if torch.cuda.is_available() and not args.cpu:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model = EvalModel(args.model, args)
    model.label_file(input_file, output_file, device)
    


def main():
    args = parse_args()
    label_langs(args)

if __name__ == "__main__":
    main()
