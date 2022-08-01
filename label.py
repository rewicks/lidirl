import argparse
import torch
import json
import sys

from models import CLD3Model, TransformerModel
from preprocessor import Processor, NGramProcessor, TrainingShard


import logging

logging.basicConfig(format="%(message)s", level=logging.DEBUG)
logger = logging.getLogger('gcld3')

class DefaultArgs():
    def __init__(self):
        pass

def load_from_checkpoint(checkpoint_path):
    model_dict = torch.load(checkpoint_path, map_location=torch.device('cpu'))
    if "embedding_dim" in model_dict:
        model = CLD3Model(vocab_size=model_dict['vocab_size'],
                            embedding_dim=model_dict['embedding_dim'],
                            hidden_dim=model_dict['hidden_dim'],
                            label_size=model_dict['label_size'],
                            num_ngram_orders=model_dict['num_ngram_orders'])
    else:
        model = TransformerModel(
            vocab_size=model_dict["vocab_size"],
            embedding_dim=model_dict["embedding_size"],
            label_size=model_dict["label_size"],
            num_layers=model_dict["num_layers"],
            max_len=model_dict["max_length"],
            nhead=model_dict["nhead"]
        )    
    model.load_state_dict(model_dict['weights'])
    model.eval()
    vocab = model_dict["vocab"]
    labels = model_dict["labels"]
    if len(model_dict['processor']) == 0:
        processor = Processor()
    else:
        processor = NGramProcessor(
            ngram_orders=model_dict['processor']['ngram_orders'],
            num_hashes=model_dict['processor']['num_hashes'],
            max_hash_value=model_dict['processor']['max_hash_value']
        )

    return model, vocab, labels, processor

# def load_preprocessor(model, preprocessor_path=None):
#     if preprocessor_path is not None:
#         prepro_dict = json.load(open(preprocessor_path))
#         prepro_dict = prepro_dict['preprocessor'] 
#     else:
#         prepro_dict = model.preprocessor_dict

    
#     preprocessor = Preprocessor(ngram_orders=prepro_dict['ngram_orders'],
#                                     num_hashes=prepro_dict['num_hashes'],
#                                     max_hash_value=prepro_dict['max_hash_value']
#                                     )
#     preprocessor.set_labels(prepro_dict['labels'])
#     preprocessor.set_itos()
#     return preprocessor





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

            inputs = self.processor(texts, device)

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
        langid = '<unk>'
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
    
    # model.model = model.model.to(device)

    
        



def main():
    args = parse_args()
    label_langs(args)

if __name__ == "__main__":
    main()
