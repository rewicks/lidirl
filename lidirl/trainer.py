import argparse
import os, sys
from re import M
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import json
import torchmetrics.classification as tmc
import string
import pathlib
from statistics import mean

if (__package__ is None or __package__ == "") and __name__ == '__main__':
    parent = pathlib.Path(__file__).absolute().parents[1]
    sys.path.insert(0, str(parent))
    __package__ = 'lidirl'

from . import __version__
from .preprocessor import Dataset, Processor, PaddedProcessor, NGramProcessor
from .models import CLD3Model, TransformerModel, ConvModel, UNETModel, RoformerModel, FlashModel
from .augmentations import Antspeak, Hashtags, NGrams, Spongebob, Short
from .metrics import Results
from .logger import TrainingLogger

######################################################################################

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("lidirl")

######################################################################################


def save_model(model, model_type, dataset, processor, output_path, bytes, device=None, log_output=None):
    model = model.cpu()
    model_dict = model.save_object()
    model_dict['processor'] = processor.save_object()
    model_dict['labels'] = dataset.labels
    model_dict['vocab'] = dataset.vocab
    model_dict['bytes'] = bytes
    model_dict['model_type'] = model_type
    torch.save(model_dict, output_path)
    model = model.to(device)

class Trainer():
    def __init__(self, args, train, valid, model, processor, augmentations):
        self.with_cuda = torch.cuda.is_available() and not args.cpu
        self.device = torch.device("cuda:0" if self.with_cuda else "cpu")
        logger.info(f"Training on device: {self.device}")

        self.output_path = args.output_path
        self.max_tokens = args.max_tokens

        self.train_dataset = train
        self.train_dataset.set_batch_size(args.max_tokens)

        self.validation_dataset = valid
        for v in self.validation_dataset:
            v.set_batch_size(args.max_tokens)

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)

        self.model = model.to(self.device)
        logger.info(self.model)

        self.processor = processor
        
        self.augmentations = augmentations
        self.augmentation_prob = args.augmentation_probability

        self.criterion = nn.NLLLoss()
        self.lr = args.lr
        self.warmup_lr = args.warmup_lr
        if args.warmup_lr is not None and args.warmup_updates is not None:
            self.warmup_updates = args.warmup_updates
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.warmup_lr)
        else:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

        total_params = sum([p.numel() for p in self.model.parameters()])
        logger.info(f'Training with: {total_params} total parameters')
        if self.with_cuda and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUSs for training")
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()

        self.best_model = None

        # keeps track of all metrics and logging information
        self.results = Results(time.time(), 
                                num_labels=len(self.train_dataset.labels),
                                length=len(self.train_dataset), 
                                device=self.device)
        self.iswarm = False
        logger.info(args)

        wandb_config = {
            "project_name": args.wandb_proj,
            "run_name": args.wandb_run
        }
        self.train_log = TrainingLogger(stdout=True,
                                            wandb_config=wandb_config)

    def run_epoch(self, args, epoch=0):
        completed = 0
        running_loss = 0
        for batch_index, (labels, texts) in self.train_dataset.enumerate(self.augmentations, self.augmentation_prob):
            completed += len(labels)
            inputs, labels = self.processor(texts, labels, self.device)

            output = self.model(inputs)

            probs = torch.exp(output)
            if args.model == "unet":
                loss = 0
                ppl = 0
                for o, l in zip(output.transpose(0,1), labels.transpose(0,1)):
                    loss += self.criterion(o, l)
                    ppl += torch.exp(F.cross_entropy(o, l)).item() 
                running_loss += loss
                
            else:
                loss = self.criterion(output, labels)
                running_loss += loss
                ppl = torch.exp(F.cross_entropy(output, labels)).item()

            self.results.calculate(loss.item(), ppl, probs, labels)

            if not self.iswarm and args.warmup_updates is not None and batch_index / args.update_interval > args.warmup_updates:
                logger.info(f"Model is warmed up. Now changing learning rate to {self.lr}")
                for g in self.optimizer.param_groups:
                    g['lr'] = self.lr
                self.iswarm = True

            if batch_index % args.update_interval == 0:
                self.optimizer.zero_grad()
                running_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                running_loss = 0

            if batch_index % args.log_interval == 0:
                batch_results = self.results.get_results(self.optimizer.param_groups[0]['lr'], completed=completed)
                self.train_log(batch_results)
                # logger.info(json.dumps(self.results.get_results(self.optimizer.param_groups[0]['lr'], completed=completed)))
                self.results.reset(time.time())

            if batch_index % args.validation_interval == 0:
                validation_results = self.validate(args, validation_num = self.results.validations)
                self.train_log.log(validation_results)
                # logger.info(json.dumps(validation_results))
                self.results.validated()
                if self.best_model is not None:
                    if validation_results['accuracy'] > self.best_model['accuracy']:
                        self.best_model = validation_results
                        save_model(self.model, args.model, self.train_dataset, self.processor, os.path.join(self.output_path, 'checkpoint_best.pt'), self.train_dataset.bytes, device=self.device, log_output=self.best_model)
                        logger.info(f"Improved accuracy of {validation_results['accuracy']}")
                    elif validation_results['total_loss'] < self.best_model['total_loss']:
                        self.best_model = validation_results
                        save_model(self.model, args.model, self.train_dataset, self.processor, os.path.join(self.output_path, 'checkpoint_best.pt'), self.train_dataset.bytes, device=self.device, log_output=self.best_model)
                        logger.info(f"Improved loss of {validation_results['total_loss']}")
                    else:
                        if epoch > args.min_epochs and validation_results['validation_num'] - self.best_model['validation_num'] >= args.validation_threshold:
                            logger.info(f"EARLY STOPPING: {json.dumps(self.best_model)}")
                            return 0
                else:
                    self.best_model = validation_results
                    save_model(self.model, args.model, self.train_dataset, self.processor, os.path.join(self.output_path, 'checkpoint_best.pt'), self.train_dataset.bytes, device=self.device, log_output=self.best_model)
                if args.checkpoint_last:
                    save_model(self.model, args.model, self.train_dataset, self.processor, os.path.join(self.output_path, 'checkpoint_last.pt'), self.train_dataset.bytes, device=self.device, log_output=self.best_model)
        if args.save_every_epoch:
            save_model(self.model, args.model, self.train_dataset, self.processor, os.path.join(self.output_path, f'epoch{epoch}.pt'), self.train_dataset.bytes, device=self.device, log_output=self.best_model)
        return 1



    def validate(self, args, validation_num=0):
        ret_results = {}
        accs = []
        losses = []
        self.model.eval()
        for val in self.validation_dataset:
            valid_results = Results(time.time(), 
                                        num_labels=len(val.labels),
                                        length=len(self.validation_dataset),
                                        device=self.device,
                                        type='VALIDATION')
            with torch.no_grad():
                for batch_index, (labels, texts) in enumerate(val):
                    
                    inputs, labels = self.processor(texts, labels, self.device)
                    # labels = labels.to(self.device)

                    output = self.model(inputs)

                    if args.model == "unet":
                        loss = 0
                        ppl = 0
                        for i, o, l in zip(inputs.transpose(0, 1), output.transpose(0,1), labels.transpose(0,1)):
                            if sum(i) == 0:
                                break
                            loss += self.criterion(o, l)
                            ppl += torch.exp(F.cross_entropy(o, l)).item() 
                    else:
                        loss = self.criterion(output, labels)
                        ppl = torch.exp(F.cross_entropy(output, labels)).item()

                    probs = torch.exp(output)

                    valid_results.calculate(loss.item(), ppl, probs, labels)
            ret_results[val.group_name] = valid_results.get_results(self.optimizer.param_groups[0]['lr'])
            accs.append(ret_results[val.group_name]["accuracy"])
            losses.append(ret_results[val.group_name]["total_loss"])
        ret_results["validation_num"] = validation_num
        ret_results["accuracy"] = mean(accs)
        ret_results["total_loss"] = mean(losses)
        self.model.train()
        return ret_results

############################################ BUILDING AND LOADING UTILS ############################################

def load_data(args):
    if not os.path.exists(args.preprocessed_data_dir):
        logger.error(f"Data directory at {args.preprocessed_data_dir} does not exist. Please preprocess the data.")
        sys.exit(-1)
    else:
        train_data =  Dataset(args.preprocessed_data_dir, type="train")
        train_data.load()

        valid_data = []
        for d, s, f in os.walk(args.preprocessed_data_dir):
            for fi in f:
                if "valid" in fi:
                    if len(fi.split('.')) > 2:
                        group = fi.split(".")[1]
                    else:
                        group = None
                    split = Dataset(args.preprocessed_data_dir, type="valid", group_name=group)
                    split.load()
                    valid_data.append(split)

    return train_data, valid_data


def load_model(args):
    logger.info(f"Loading pre-existing model from checkpoint at {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint not found at {args.checkpoint_path}")
    pass

def build_model(args, dataset):
    if args.model == "linear-ngram":
        ngram_orders = [int(n) for n in args.ngram_orders.split(',')]
        model = CLD3Model(vocab_size=args.max_hash_value * len(ngram_orders),
                                    hidden_dim=args.hidden_dim,
                                    embedding_dim=args.embedding_dim,
                                    label_size=len(dataset.labels.keys()),
                                    num_ngram_orders=len(ngram_orders),
                                    montecarlo_layer=args.montecarlo_layer)
    elif args.model == "transformer":
        model = TransformerModel(vocab_size=len(dataset.vocab),
                                    embedding_dim=args.embedding_dim,
                                    label_size=len(dataset.labels.keys()),
                                    num_layers=args.num_layers,
                                    max_len=args.max_length,
                                    nhead=args.nhead,
                                    montecarlo_layer=args.montecarlo_layer)
    elif args.model == "roformer":
        model = RoformerModel(vocab_size=len(dataset.vocab),
                                    embedding_dim=args.embedding_dim,
                                    hidden_dim=args.hidden_dim,
                                    label_size=len(dataset.labels.keys()),
                                    num_layers=args.num_layers,
                                    max_len=args.max_length,
                                    nhead=args.nhead,
                                    dropout=args.dropout,
                                    montecarlo_layer=args.montecarlo_layer)
    elif args.model == "convolutional":
        model = ConvModel(vocab_size=len(dataset.vocab),
                            label_size=len(dataset.labels.keys()),
                            embedding_dim=args.embedding_dim,
                            conv_min_width=args.conv_min_width,
                            conv_max_width=args.conv_max_width,
                            conv_depth=args.conv_depth,
                            montecarlo_layer=args.montecarlo_layer)

    elif args.model == "unet":
        model = UNETModel(vocab_size=len(dataset.vocab),
                            label_size=len(dataset.labels.keys()),
                            embed_size=args.embedding_dim,
                            montecarlo_layer=args.montecarlo_layer
        )
    elif args.model == "flash":
        model = FlashModel(
            vocab_size=len(dataset.vocab),
            embedding_dim=args.embedding_dim,
            hidden_dim=args.hidden_dim,
            label_size=len(dataset.labels.keys()),
            num_layers=args.num_layers,
            max_len=args.max_length,
            nhead=args.nhead,
            dropout=args.dropout,
            montecarlo_layer=args.montecarlo_layer,
            use_rotary=args.use_rotary,
        )

    return model

def build_processor(args):
    if args.model == "linear-ngram":
        logger.info("Buildling an NGramProcessor for an Ngram Linear Model")
        processor = NGramProcessor(
            ngram_orders=[int(n) for n in args.ngram_orders.split(',')],
            num_hashes=args.num_hashes,
            max_hash_value=args.max_hash_value,
        )
    elif args.model == "transformer":  
        logger.info("Building a base Processor for a Transformer model") 
        processor = Processor()
    elif args.model == "roformer":
        logger.info("Building a base Processor for a Roformer Model")
        processor = Processor()
    elif args.model == "flash":
        logger.info("Building a base Processor for a Flash Model")
        processor = Processor()
    elif args.model == "convolutional":
        logger.info("Building a base Processor for a Convolutional model") 
        processor = Processor()
    elif args.model == "unet":
        logger.info("Building a base Processor for a UNet model") 
        processor = PaddedProcessor(args.pad_length)     
    
    return processor

def build_augmentations(args, vocab):
    if args.augmentations is not None:
        augs = []
        probs = []
        punctuation = []
        capitals = {}
        lowers = {}

        for v, id in vocab.items():
            if v in string.punctuation:
                punctuation.append(id)
            if v == "[SPACE]":
                capitals[id] = id
                lowers[id] = id
            else:
                capitals[id] = vocab.get(v.upper(), 0)
                lowers[id] = vocab.get(v.lower(), 0)

        for aug in args.augmentations.split('/'):
            aug = aug.split(',')
            prob = float(aug[1])
            aug = aug[0]
            if aug == "antspeak":
                aug = Antspeak(vocab.get('[SPACE]'))
            elif aug == "ngrams":
                aug = NGrams(disallowed_repeats=punctuation,
                            space_idx=vocab['[SPACE]'])
            elif aug == "hashtag":
                aug = Hashtags(hashtag_idx=vocab.get('#', 0), 
                                    space_idx=vocab['[SPACE]'],
                                    punctuation=punctuation,
                                    capitals=capitals,
                                    lowers=lowers
                                    )
            elif aug == "short":
                aug = Short(space_idx=vocab['[SPACE]'])
            elif aug == "spongebob":
                aug = Spongebob(
                    capitals=capitals,
                    lowers=lowers
                )
            augs.append(aug)
            probs.append(prob)
        total_prob_mass = sum(probs)
        augmentations = []
        for a, p in zip(augs, probs):
            augmentations.append((a, p/total_prob_mass))
        return augmentations
    return None


def parse_args():
    parser = argparse.ArgumentParser(
        description="TRAINER: something something something.\n"
        "      Example: something",
        formatter_class=argparse.RawTextHelpFormatter
    )

    parser.add_argument("--raw_data_dir", 
                                        type=str, 
                                        help="The path to raw data. If this is passed, data directory must have a train and valid subfolder.")
    parser.add_argument("--preprocessed_data_dir",
                                        type=str,
                                        help="The path to the directory where the data is located. This must be the output of the preprocess call")
    parser.add_argument("--output_path", 
                                        required=True,
                                        type=str,
                                        help="The path to the directory will output models will be saved")

    parser.add_argument("--checkpoint_path",
                                        type=str,
                                        default=None,
                                        help="The checkpoint file to continue training from.")

    parser.add_argument("--max_tokens", 
                                        type=int,
                                        default=2000,
                                        help="The batch size in tokens")
    parser.add_argument("--tb_dir", 
                                        type=str,
                                        default=None,
                                        help="The directory path to save tensorboard files")

    parser.add_argument('--warmup_lr', 
                                        type=float,
                                        default=None,
                                        help="The LR to use during the warmup updates")
    parser.add_argument('--warmup_updates',
                                        type=int,
                                        default=None,
                                        help="The number of updates to warm up (different learning rate)")

    parser.add_argument("--lr", 
                                type=float,
                                default=0.0001,
                                help="The learning rate to use")
    parser.add_argument("--embedding_dim",
                                type=int,
                                default=256,
                                help="The size of the embedding dimension")
    parser.add_argument("--hidden_dim", 
                                type=int, 
                                default=256,
                                help="The size of the hidden dimension.")
    parser.add_argument("--dropout", 
                                type=float, 
                                default=0.1,
                                help="Dropout percent to use for regularization")

    parser.add_argument("--cpu", 
                                action="store_true",
                                help="Forces use of cpu even if CUDA is available.")

    parser.add_argument("--min_epochs", 
                                type=int, 
                                default=25,
                                help="Minimum number of epochs to train for.")
    parser.add_argument("--max_epochs", 
                                type=int,
                                default=100,
                                help="Maximum number of epochs to train for.")
    parser.add_argument("--save_every_epoch",
                                action="store_true",
                                default=False,
                                help="If true, saves a model after each epoch")
    parser.add_argument("--checkpoint_last",
                                action="store_true",
                                default=False,
                                help="If true, saves the last model separate from best model.")

    parser.add_argument("--update_interval",
                                type=int,
                                default=1,
                                help="Backprops every N updates")
    parser.add_argument('--log_interval',
                                type=int,
                                default=1000,
                                help="Waits N updates to log information")
    parser.add_argument('--validation_interval',
                                type=int,
                                default=25000,
                                help="Waits N updates to validate")
    parser.add_argument("--patience",
                                type=int,
                                default=-1,
                                help="If loss has not improved in N validations, stops training early.")

    parser.add_argument("--num-layers",
                                default=3,
                                type=int,
                                help="The number of layers to use in model.") # need to add this to linear
    parser.add_argument("--montecarlo_layer",
                                default=False,
                                action="store_true",
                                help="If true, uses a MonteCarlo Layer instead of typical projection layer.")

    parser.add_argument("--augmentations", 
                                default=None,
                                type=str,
                                help="A comma separated list of augmentation (names) and their ratios.")
    parser.add_argument("--augmentation_probability",
                                default=0.2,
                                type=float,
                                help="The probability of augmenting data.")

    parser.add_argument("--wandb_proj", default=None, type=str, help="The project where this run will be logged")
    parser.add_argument("--wandb_run", default=None, type=str, help="The name of the run for wandb logging")

    # A switch for which model architecture to use
    subparsers = parser.add_subparsers(help="Determines the type of model to be built and trained", dest="model")
    
    # SUBPARSER FOR THE LINEAR NGRAM BASED MODEL (SIMILAR TO CLD3)
    linear_parser = subparsers.add_parser("linear-ngram", 
                                                help="a linear ngram style model")
    linear_parser.add_argument("--ngram_orders", 
                                    default="1,2,3", 
                                    type=str,
                                    help="A comma separated list of character n-gram orders to extract from data")
    linear_parser.add_argument("--max_hash_value",
                                    default=128,
                                    type=int,
                                    help="Max hash value when embedding the n-grams")
    linear_parser.add_argument("--num_hashes",
                                    default=1,
                                    type=int,
                                    help="How many distinct hashes to use")

    # A SUBPARSER FOR THE TRANSFORMER (STANDARD) MODEL
    transformer_parser = subparsers.add_parser("transformer", help="a transformer model")
    transformer_parser.add_argument("--max-length", default=1024, type=int)
    transformer_parser.add_argument("--nhead", default=8, type=int)

    # A SUBPARSER FOR THE ROFORMER MODEL
    roformer_parser = subparsers.add_parser("roformer", help="a roformer model")
    roformer_parser.add_argument("--max-length", default=1024, type=int)
    roformer_parser.add_argument("--nhead", default=8, type=int)

    # A SUBPARSER FOR THE CONVOLUTIONAL MODEL
    conv_parser = subparsers.add_parser("convolutional", help="a convolutional model")
    conv_parser.add_argument("--conv_min_width", default=2, type=int)
    conv_parser.add_argument("--conv_max_width", default=5, type=int)
    conv_parser.add_argument("--conv_depth", default=64, type=int)

    # A SUBPARSER FOR THE FLASH (TESTING) MODEL
    flash_parser = subparsers.add_parser("flash", help="the new flash model (for testing)")
    flash_parser.add_argument("--max-length", default=1024, type=int)
    flash_parser.add_argument("--nhead", default=8, type=int)
    flash_parser.add_argument("--use_rotary", default=False, action="store_true")

    # A SUBPARSER FOR THE UNET MODEL
    unet_parser = subparsers.add_parser("unet", help="a unet model")
    unet_parser.add_argument("--pad_length", default=1024, type=int)



    args = parser.parse_args()

    return args


def main(args):

    train, valid = load_data(args)

    if args.checkpoint_path is not None:
        model = load_model(args)
    else:
        model = build_model(args, train)

    processor = build_processor(args)

    augmentations = build_augmentations(args, train.vocab)

    trainer = Trainer(args, train, valid, model, processor, augmentations)
    for ep in range(args.min_epochs):
        logger.info(f"Beginning epoch {ep}")
        epoch_finish = trainer.run_epoch(args, ep)
        trainer.scheduler.step()

    for ep in range(args.min_epochs, args.max_epochs):
        logger.info(f"Beginning epoch {ep}")
        epoch_finish = trainer.run_epoch(args, ep)
        if epoch_finish == 0:
            logger.info("Finished training")
            break
        trainer.scheduler.step()


if __name__ == "__main__":
    args = parse_args()
    main(args)
