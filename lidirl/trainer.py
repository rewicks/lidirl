import argparse
import os, sys
from re import M
import logging
import time
import math
import random
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
from .preprocessor import Dataset, Processor, PaddedProcessor, NGramProcessor, VisRepProcessor
from .models import CLD3Model, TransformerModel, ConvModel, UNETModel, RoformerModel, Hierarchical

from .metrics import Results
from .logger import TrainingLogger
from .dataloader import build_datasets
from .augmentations import build_augmentations

######################################################################################

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("lidirl")

######################################################################################


def save_model(model, 
                    model_type, 
                    input_type,
                    pred_type,
                    dataset, 
                    output_path,
                    training_status,
                    scheduler,
                    optimizer,
                    vocab,
                    labels,
                    device=None, log_output=None):
    model = model.cpu()
    model_dict = model.module.save_object() if type(model) is nn.DataParallel else model.save_object()
    model_dict['pred_type'] = pred_type
    model_dict['labels'] = labels
    model_dict['vocab'] = vocab
    model_dict['input_type'] = input_type
    model_dict['model_type'] = model_type
    model_dict["training_status"] = training_status
    model_dict["scheduler"] = scheduler
    model_dict["optimizer"] = optimizer.state_dict()
    torch.save(model_dict, output_path)
    model = model.to(device)


class Trainer():
    def __init__(self, args, train, valid, model, vocab, labels):
        self.with_cuda = torch.cuda.is_available() and not args.cpu
        self.device = torch.device("cuda:0" if self.with_cuda else "cpu")
        logger.info(f"Training on device: {self.device}")

        self.output_path = args.output_path

        self.train_dataset = train
        self.validation_dataset = valid
        self.vocab = vocab
        self.labels = labels

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)

        self.model = model.to(self.device)
        logger.info(self.model)

        if args.pred_type == "multilabel":
            self.criterion = nn.BCELoss()
        else:
            self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1, reduction='sum')
            
        self.lr = args.lr
        # self.warmup_lr = args.warmup_lr
        # self.warmup_updates = args.warmup_updates

        # lr = self.warmup_lr if (args.warmup_lr is not None and args.warmup_updates is not None) else self.lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.optimizer.zero_grad()

        # if self.args.warmup_lr is not None and self.args.warmup_updates is not None:
        #     scheduler = LinearLR(self.optimizer, total_iters=args.warmup_updates)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 500, gamma=0.99)

        total_params = sum([p.numel() for p in self.model.parameters()])
        logger.info(f'Training with: {total_params} total parameters')
        if self.with_cuda and torch.cuda.device_count() > 1:
            logger.info(f"Using {torch.cuda.device_count()} GPUSs for training")
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()

        self.best_model = None

        # keeps track of all metrics and logging information
        self.results = Results(time.time(), 
                                num_labels=len(self.labels),
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

        self.num_updates = 0

    def run_epoch(self, args, epoch=0):
        for batch_index, (texts, targets) in enumerate(self.train_dataset, 1):
            try:
                # multilabel means that every target class with greater than 0 prob should be correct label
                if args.pred_type == "multilabel":
                    for trg in targets:
                        targets[trg > 0] = 1

                inputs = texts.to(self.device)
                targets = targets.to(self.device)

                output = self.model(inputs) # raw logits
                
                # multiabel--there should be some threshold logic here, but alas
                # by default, we'll just use sigmoid (0.5 threshold)
                if args.pred_type == "multilabel":
                    output = F.sigmoid(output)
                    labels = targets

                # traditional, multiclass classifier
                else:
                    # cross entropy criterion expects no softmax applied so we leave output as is
                    labels = torch.argmax(targets, dim=1)

                # token level outputs
                if args.model == "unet" or args.pred_type == "token_level":
                    loss = 0
                    ppl = 0
                    for b, trg in enumerate(targets):
                        for o, t in zip(output[b], trg):
                            loss += self.criterion(o, t)
                            ppl += torch.exp(F.cross_entropy(o, t)).item() 
                    
                else:
                    loss = self.criterion(output, labels)
                    ppl = torch.exp(F.cross_entropy(output, targets)/args.update_interval).item()

                # keep track of metrics for eventually logging
                self.results.calculate(loss.item()/args.update_interval, ppl, output, targets, args.pred_type == "token_level")


                # update intervals are for effective batch sizes
                loss /= args.update_interval
                loss.backward()

                training_status = {
                    "updates": self.num_updates,
                    "batch_index": batch_index,
                    "epoch": epoch
                }

                if batch_index % args.update_interval == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    self.scheduler.step() # I'm not stepping every epoch bc twitter is quite large---steps every 500 updates in __init__
                    self.num_updates += 1
                    if self.num_updates == args.max_updates:
                        logger.info(f"Reached maximum number of updates ({args.max_updates}) for training. Stopping.")
                        return 0

                if (batch_index) % (args.log_interval * args.update_interval) == 0:
                    batch_results = self.results.get_results(self.optimizer.param_groups[0]['lr'], completed=batch_index+1)
                    self.train_log.log(batch_results)
                    self.results.reset(time.time())

                if (batch_index) % (args.validation_interval * args.update_interval) == 0:
                    validation_results = self.validate(args, validation_num = self.results.validations)
                    self.train_log.log(validation_results)
                    self.results.validated()
                    if self.best_model is not None:
                        if validation_results['accuracy'] > self.best_model['accuracy']:
                            self.best_model = validation_results
                            save_model(self.model, 
                                            args.model,
                                            args.input_type,
                                            args.pred_type,
                                            self.train_dataset, 
                                            os.path.join(self.output_path, 'checkpoint_best.pt'), 
                                            training_status=training_status,
                                            scheduler=self.scheduler,
                                            optimizer=self.optimizer,
                                            vocab=self.vocab,
                                            labels=self.labels,
                                            device=self.device,
                                            log_output=self.best_model)
                            logger.info(f"Improved accuracy of {validation_results['accuracy']}")
                        elif validation_results['total_loss'] < self.best_model['total_loss']:
                            self.best_model = validation_results
                            save_model(self.model, 
                                            args.model,
                                            args.input_type,
                                            args.pred_type,
                                            self.train_dataset, 
                                            os.path.join(self.output_path, 'checkpoint_best.pt'),
                                            training_status=training_status,
                                            scheduler=self.scheduler,
                                            optimizer=self.optimizer,
                                            vocab=self.vocab,
                                            labels=self.labels,
                                            device=self.device, 
                                            log_output=self.best_model)
                            logger.info(f"Improved loss of {validation_results['total_loss']}")
                        else:
                            if epoch > args.min_epochs and validation_results['validation_num'] - self.best_model['validation_num'] >= args.patience:
                                logger.info(f"EARLY STOPPING: {json.dumps(self.best_model)}")
                                return 0
                            if validation_results['validation_num'] - self.best_model['validation_num'] >= 1000000:
                                self.scheduler.step()
                    else:
                        self.best_model = validation_results
                        save_model(self.model, 
                                        args.model,
                                        args.input_type,
                                        args.pred_type,
                                        self.train_dataset, 
                                        os.path.join(self.output_path, 'checkpoint_best.pt'), 
                                        training_status=training_status,
                                        scheduler=self.scheduler,
                                        optimizer=self.optimizer,
                                        vocab=self.vocab,
                                        labels=self.labels,
                                        device=self.device, 
                                        log_output=self.best_model)
                    if args.checkpoint_last:
                        save_model(self.model, 
                                        args.model,
                                        args.input_type,
                                        args.pred_type,
                                        self.train_dataset, 
                                        os.path.join(self.output_path, 'checkpoint_last.pt'),
                                        training_status=training_status,
                                        scheduler=self.scheduler,
                                        optimizer=self.optimizer, 
                                        vocab=self.vocab,
                                        labels=self.labels,
                                        device=self.device, 
                                        log_output=self.best_model)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.info("OOM: Clearing cache and trying again.")
                    self.optimizer.zero_grad()
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise Exception(e)
        if args.save_every_epoch:
            save_model(self.model, 
                            args.model,
                            args.input_type,
                            args.pred_type,
                            self.train_dataset, 
                            os.path.join(self.output_path, f'epoch{epoch}.pt'), 
                            training_status=training_status,
                            scheduler=self.scheduler,
                            optimizer=self.optimizer,
                            vocab=self.vocab,
                            labels=self.labels,
                            device=self.device,
                            log_output=self.best_model)
        return 1



    def validate(self, args, validation_num=0):
        torch.cuda.empty_cache()
        ret_results = {}
        accs = []
        losses = []
        self.model.eval()
        for name, val in self.validation_dataset.items():
            valid_results = Results(time.time(), 
                                        num_labels=len(self.labels),
                                        length=len(self.validation_dataset),
                                        device=self.device,
                                        type='VALIDATION')
            with torch.no_grad():
                for batch_index, (texts, targets) in enumerate(val, 1):
                    inputs = texts.to(self.device)
                    targets = targets.to(self.device)
                    output = self.model(inputs)

                    if args.pred_type != "multilabel":
                        labels = torch.argmax(targets, dim=1)
                    else:
                        labels = targets
                        output = F.sigmoid(output)

                    if args.model == "unet" or args.pred_type == "token_level":
                        loss = 0
                        ppl = 0
                        for b, target in enumerate(targets):
                            for i, o, t in zip(inputs[b], output[b], target):
                                loss += F.cross_entropy(o, t, reduce=False).item()
                                ppl += torch.exp(F.cross_entropy(o, t)).item()
                    else:
                        loss = F.cross_entropy(output, targets, reduction='sum').item()
                        ppl = math.exp(loss)

                    probs = F.softmax(output, dim=1)

                    valid_results.calculate(loss, ppl, probs, targets, args.pred_type == "token_level")
            ret_results[name] = {}
            for key, value in valid_results.get_results(self.optimizer.param_groups[0]['lr']).items():
                if key in ["accuracy", "total_loss", "ppl_per_pred", "num_pred"]:
                    ret_results[name][key] = value
                else:
                    ret_results[key] = value
            accs.append(ret_results[name]["accuracy"])
            losses.append(ret_results[name]["total_loss"])
        ret_results["validation_num"] = validation_num
        ret_results["accuracy"] = mean(accs)
        ret_results["total_loss"] = sum(losses)
        self.model.train()
        return ret_results

############################################ BUILDING AND LOADING UTILS ############################################

def load_model(args):
    logger.info(f"Loading pre-existing model from checkpoint at {args.checkpoint_path}")
    if not os.path.exists(args.checkpoint_path):
        logger.error(f"Checkpoint not found at {args.checkpoint_path}")
    pass

def build_model(args, vocab, labels):

    visual = True if args.input_type == "visual" else False
    token_level = True if args.pred_type == "token_level" else False

    if args.model == "linear-ngram":
        ngram_orders = [int(n) for n in args.ngram_orders.split(',')]
        model = CLD3Model(vocab_size=args.max_hash_value * len(ngram_orders),
                                    hidden_dim=args.hidden_dim,
                                    embedding_dim=args.embedding_dim,
                                    label_size=len(labels),
                                    num_ngram_orders=len(ngram_orders),
                                    montecarlo_layer=args.montecarlo_layer)
    elif args.model == "transformer":
        model = TransformerModel(vocab_size=len(vocab),
                                    hidden_dim=args.hidden_dim,
                                    embedding_dim=args.embedding_dim,
                                    label_size=len(labels.keys()),
                                    num_layers=args.num_layers,
                                    max_len=args.max_length,
                                    nhead=args.nhead,
                                    montecarlo_layer=args.montecarlo_layer,
                                    visual_inputs=visual,
                                    convolutions=[1] + [int(_) for _ in args.convolutions.split(',')])
    elif args.model == "roformer":
        model = RoformerModel(vocab_size=len(vocab),
                                    embedding_dim=args.embedding_dim,
                                    hidden_dim=args.hidden_dim,
                                    label_size=len(labels),
                                    num_layers=args.num_layers,
                                    max_len=args.max_length,
                                    nhead=args.nhead,
                                    dropout=args.dropout,
                                    montecarlo_layer=args.montecarlo_layer,
                                    token_level=token_level)
    elif args.model == "convolutional":
        model = ConvModel(vocab_size=len(vocab),
                            label_size=len(labels),
                            embedding_dim=args.embedding_dim,
                            conv_min_width=args.conv_min_width,
                            conv_max_width=args.conv_max_width,
                            conv_depth=args.conv_depth,
                            montecarlo_layer=args.montecarlo_layer)

    elif args.model == "unet":
        model = UNETModel(vocab_size=len(vocab),
                            label_size=len(labels),
                            embed_size=args.embedding_dim,
                            montecarlo_layer=args.montecarlo_layer
        )
    elif args.model == "hierarchical":
        model = Hierarchical(
                    vocab_size=len(vocab),
                    label_size=len(labels),
                    window_size=args.window_size,
                    stride=args.stride,
                    embed_dim=args.embedding_dim,
                    hidden_dim=args.hidden_dim,
                    nhead=args.nhead,
                    max_length=args.max_length,
                    num_layers=args.num_layers,
        )

    return model

def parse_args():
    parser = argparse.ArgumentParser(
        description="TRAINER: something something something.\n"
        "      Example: something",
        formatter_class=argparse.RawTextHelpFormatter
    )

    # this currently is non-functional, ignore it
    parser.add_argument("--train_files", nargs='+',
                                        type=str,
                                        help="The path to the training data.")
    parser.add_argument("--valid_files", nargs='+',
                                        type=str,
                                        help="The path to the validation data.")
    parser.add_argument('--smart_group_validation_files', default=False, action="store_true",
                            help="If passed, will group validation sets into separately logged clusters based on each file's parent directory")

    parser.add_argument("--output_path", 
                                        required=True,
                                        type=str,
                                        help="The path to the directory will output models will be saved")
    parser.add_argument("--input_type", 
                                            default="characters",
                                            type=str,
                                            choices=["characters", "bytes", "visual"],
                                            help="The type of input to the model.")
    parser.add_argument("--pred_type", 
                                            default="multiclass",
                                            choices=["multiclass", "multilabel", "token_level"],
                                            help="The type of prediction the model will make")
    parser.add_argument("--temperature",
                                            default=0.3,
                                            type=float,
                                            help="The temperature alpha to use for dataset sampling.")
    parser.add_argument("--batch_size", 
                                        type=int,
                                        default=25,
                                        help="The batch size in examples")
    parser.add_argument("--tb_dir", 
                                        type=str,
                                        default=None,
                                        help="The directory path to save tensorboard files")

    parser.add_argument("--num_workers",
                                        type=int,
                                        default=1,
                                        help="The number of workers to use for data loading")

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
    parser.add_argument("--step_rate",
                                type=float,
                                default=1.0,
                                help="The rate at which to decay the learning rate")
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
                                default=0,
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
                                default=math.inf,
                                help="If loss has not improved in N validations, stops training early.")
    parser.add_argument("--max-updates",
                                type=int,
                                default=math.inf,
                                help="The maximum number of updates before halting training.")

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
                                default=0.0,
                                type=float,
                                help="The probability of augmenting data.")
    parser.add_argument('--seed', type=int, default=141414)

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
    transformer_parser.add_argument("--convolutions", default="16,32,64", type=str, help="comma separated list of convolution channels")

    # A SUBPARSER FOR THE ROFORMER MODEL
    roformer_parser = subparsers.add_parser("roformer", help="a roformer model")
    roformer_parser.add_argument("--max-length", default=1024, type=int)
    roformer_parser.add_argument("--nhead", default=8, type=int)

    # A SUBPARSER FOR THE CONVOLUTIONAL MODEL
    conv_parser = subparsers.add_parser("convolutional", help="a convolutional model")
    conv_parser.add_argument("--conv_min_width", default=2, type=int)
    conv_parser.add_argument("--conv_max_width", default=5, type=int)
    conv_parser.add_argument("--conv_depth", default=64, type=int)

    # A SUBPARSER FOR THE UNET MODEL
    unet_parser = subparsers.add_parser("unet", help="a unet model")
    unet_parser.add_argument("--pad_length", default=1024, type=int)

    hierachical_parser = subparsers.add_parser("hierarchical", help="a hierarchical model")
    hierachical_parser.add_argument("--max-length", default=1024, type=int)
    hierachical_parser.add_argument("--nhead", default=8, type=int)
    hierachical_parser.add_argument("--window-size", default=8, type=int)
    hierachical_parser.add_argument("--stride", default=1, type=int)

    args = parser.parse_args()

    return args


def main(args):

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    augmentations = build_augmentations(args)

    train, valid, vocab, labels = build_datasets(args, augmentations)

    model = build_model(args, vocab, labels)

    trainer = Trainer(args, train, valid, model, vocab, labels)
    for ep in range(args.min_epochs):
        logger.info(f"Beginning epoch {ep}")
        epoch_finish = trainer.run_epoch(args, ep)
        # trainer.scheduler.step()

    for ep in range(args.min_epochs, args.max_epochs):
        logger.info(f"Beginning epoch {ep}")
        epoch_finish = trainer.run_epoch(args, ep)
        if epoch_finish == 0:
            logger.info("Finished training")
            break
        # trainer.scheduler.step()

    trainer.train_log.finish_log()

if __name__ == "__main__":
    args = parse_args()
    main(args)
