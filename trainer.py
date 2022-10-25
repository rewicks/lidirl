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
import metrics

from preprocessor import Dataset, Processor, PaddedProcessor, NGramProcessor
from models import CLD3Model, TransformerModel, ConvModel, UNETModel, RoformerModel

######################################################################################

logging.basicConfig(
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stderr,
)
logger = logging.getLogger("langid")

######################################################################################

class Results():
    def __init__(self, time, length=1, device=None, type='TRAINING'):
        self.total_loss = 0
        self.perplexity = 0
        self.accuracy = tmc.Accuracy().to(device)
        self.calibration_error = tmc.CalibrationError(n_bins=10).to(device)
        #self.brier_score = metrics.BrierScore().to(device)
        self.num_pred = 0
        self.update_num = 0
        self.batches = 0
        self.length = length
        self.last_update = time
        self.start = time
        self.validations = 0
        self.type = type

    def calculate(self, loss, ppl, y_hat, labels):
        self.total_loss += loss
        self.perplexity += ppl
        if len(labels.shape) == 2:
            for y_h, l in zip(y_hat.transpose(0, 1), labels.transpose(0, 1)):
                if self.type == "VALIDATION":
                    self.accuracy.update(y_h, l)
                    if self.calibration_error.update(y_h, l) == -1:
                        return -1
                #self.brier_score.update(y_h, l)
                self.num_pred += l.shape[0]
        else:
            if self.type == "VALIDATION":
                self.accuracy.update(y_hat, labels)
                if self.calibration_error.update(y_hat, labels) == -1:
                    return -1
            #self.brier_score.update(y_hat, labels)
            self.num_pred += labels.shape[0]
        self.batches += 1

    def get_results(self, lr, completed=1):
        retVal = {}
        retVal['type'] = self.type
        retVal['update_num'] = self.update_num
        retVal['complete'] = round(completed / self.length, 2)
        if self.type == "VALIDATION":
            retVal['accuracy'] = round(self.accuracy.compute().item(), 4)
            retVal['calibration_error'] = round(self.calibration_error.compute().item(), 4)
        #retVal['brier_score'] = round(self.brier_score.compute().item(), 4)
        retVal['lr'] = lr
        retVal['total_loss'] = round(self.total_loss, 4)
        retVal['average_loss'] = round(self.total_loss/self.num_pred, 4)
        retVal['ppl_per_pred'] = round(self.perplexity/self.num_pred, 4)
        retVal['time_since_last_update'] = round(time.time()-self.last_update, 2)
        retVal['predictions_per_second'] = round(self.num_pred/retVal['time_since_last_update'])
        retVal['time_passed'] = round(time.time()-self.start)
        retVal['validations'] = self.validations
        retVal['num_pred'] = self.num_pred
        return retVal

    # add perplexity
    def reset(self, time):
        self.total_loss = 0
        self.perplexity = 0
        self.num_pred = 0
        if self.type == "VALIDATION":
            self.accuracy.reset()
            self.calibration_error.reset()
        #self.brier_score.reset()
        self.update_num += 1
        self.last_update = time

    def validated(self):
        self.validations += 1


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
    def __init__(self, args, train, valid, model, processor):
        self.with_cuda = torch.cuda.is_available() and not args.cpu
        self.device = torch.device("cuda:0" if self.with_cuda else "cpu")
        logger.info(f"Training on device: {self.device}")

        self.output_path = args.output_path
        self.batch_size = args.batch_size

        self.train_dataset = train
        self.train_dataset.set_batch_size(args.batch_size)

        self.validation_dataset = valid
        self.validation_dataset.set_batch_size(args.batch_size)

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)

        self.model = model.to(self.device)
        logger.info(self.model)

        self.processor = processor

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
        self.results = Results(time.time(), length=len(self.train_dataset), device=self.device)
        logger.info(args)

    def run_epoch(self, args, epoch=0):
        completed = 0
        running_loss = 0
        for batch_index, (labels, texts) in enumerate(self.train_dataset):
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

            if args.warmup_updates is not None and batch_index / args.update_interval == args.warmup_updates:
                for g in self.optimizer.param_groups:
                    g['lr'] = self.lr

            if batch_index % args.update_interval == 0:
                self.optimizer.zero_grad()
                running_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()
                running_loss = 0

            if batch_index % args.log_interval == 0:
                logger.info(json.dumps(self.results.get_results(self.scheduler.get_last_lr()[0], completed=completed)))
                self.results.reset(time.time())

            if batch_index % args.validation_interval == 0:
                validation_results = self.validate(args, validation_num = self.results.validations)
                logger.info(json.dumps(validation_results))
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
        self.model.eval()
        valid_results = Results(time.time(), length=len(self.validation_dataset), device=self.device, type='VALIDATION')
        with torch.no_grad():
            for batch_index, (labels, texts) in enumerate(self.validation_dataset):
                
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

        self.model.train()
        ret_results = valid_results.get_results(self.scheduler.get_last_lr()[0])
        ret_results["validation_num"] = validation_num
        return ret_results

############################################ BUILDING AND LOADING UTILS ############################################

def load_data(args):
    if not os.path.exists(args.data_dir):
        logger.error(f"Data directory at {args.data_dir} does not exist. Please preprocess the data.")
        sys.exit(-1)
    else:
        train_data =  Dataset(args.data_dir, type="train")
        train_data.load()

        valid_data = Dataset(args.data_dir, type="valid")
        valid_data.load()

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

    return model

def build_processor(args):
    if args.model == "linear-ngram":
        logger.info("Buildling an NGramProcessor for an Ngram Linear Model")
        processor = NGramProcessor(
            ngram_orders=[int(n) for n in args.ngram_orders.split(',')],
            num_hashes=args.num_hashes,
            max_hash_value=args.max_hash_value
        )
    elif args.model == "transformer":  
        logger.info("Building a base Processor for a Transformer model") 
        processor = Processor()
    elif args.model == "roformer":
        logger.info("Building a base Processor for a Roformer Model")
        processor = Processor()
    elif args.model == "convolutional":
        logger.info("Building a base Processor for a Convolutional model") 
        processor = Processor()
    elif args.model == "unet":
        logger.info("Building a base Processor for a UNet model") 
        processor = PaddedProcessor(args.pad_length)     
    
    return processor

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)

    parser.add_argument("--checkpoint_path", type=str, default=None)

    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--tb_dir", type=str, default=None)

    parser.add_argument('--warmup_lr', type=float, default=None)
    parser.add_argument('--warmup_updates', type=int, default=None)

    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--dropout", type=float, default=0.1)

    parser.add_argument("--cpu", action="store_true")

    parser.add_argument("--min_epochs", type=int, default=25)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--save_every_epoch", action="store_true", default=False)
    parser.add_argument("--checkpoint_last", action="store_true", default=False)

    parser.add_argument("--update_interval", type=int, default=1)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--validation_interval', type=int, default=25000)
    parser.add_argument("--validation_threshold", type=int, default=10)

    parser.add_argument("--num-layers", default=3, type=int) # need to add this to linear
    parser.add_argument("--montecarlo_layer", default=False, action="store_true")

    subparsers = parser.add_subparsers(help="Determines the type of model to be built and trained", dest="model")
    
    linear_parser = subparsers.add_parser("linear-ngram", help="a linear ngram style model")
    linear_parser.add_argument("--ngram_orders", default="1,2,3", type=str)
    linear_parser.add_argument("--max_hash_value", default=128, type=int)
    linear_parser.add_argument("--num_hashes", default=1, type=int)

    transformer_parser = subparsers.add_parser("transformer", help="a transformer model")
    transformer_parser.add_argument("--max-length", default=1024, type=int)
    transformer_parser.add_argument("--nhead", default=8, type=int)

    roformer_parser = subparsers.add_parser("roformer", help="a roformer model")
    roformer_parser.add_argument("--max-length", default=1024, type=int)
    roformer_parser.add_argument("--nhead", default=8, type=int)

    conv_parser = subparsers.add_parser("convolutional", help="a convolutional model")
    conv_parser.add_argument("--conv_min_width", default=2, type=int)
    conv_parser.add_argument("--conv_max_width", default=5, type=int)
    conv_parser.add_argument("--conv_depth", default=64, type=int)

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

    trainer = Trainer(args, train, valid, model, processor)
    for ep in range(args.min_epochs):
        logger.info(f"Beginning epoch {ep}")
        epoch_finish = trainer.run_epoch(args, ep)
        trainer.scheduler.step()

    for ep in range(args.min_epochs, argxs.max_epochs):
        logger.info(f"Beginning epoch {ep}")
        epoch_finish = trainer.run_epoch(args, ep)
        if epoch_finish == 0:
            logger.info("Finished training")
            break
        trainer.scheduler.step()


if __name__ == "__main__":
    args = parse_args()
    main(args)
