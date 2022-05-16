import argparse
import os
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.tensorboard import SummaryWriter
import json
import torchmetrics.classification as tmc
import metrics

print('here')
from preprocessor import Dataset
from models import CLD3Model

logging.basicConfig(format="%(message)s", level=logging.DEBUG)
logger = logging.getLogger('gcld3')

class Results():
    def __init__(self, time, type='TRAINING'):
        self.total_loss = 0
        self.perplexity = 0
        self.accuracy = tmc.Accuracy()
        self.calibration_error = tmc.CalibrationError(n_bins=10)
        self.brier_score = metrics.BrierScore()
        self.num_pred = 0
        self.update_num = 0
        self.batches = 0
        self.last_update = time
        self.start = time
        self.validations = 0
        self.type = type

    def calculate(self, loss, ppl, y_hat, labels):
        self.total_loss += loss
        self.perplexity += ppl
        self.accuracy.update(y_hat, labels)
        self.calibration_error.update(y_hat, labels)
        self.brier_score.update(y_hat, labels)
        self.num_pred += labels.shape[0]
        self.batches += 1

    def get_results(self, lr):
        retVal = {}
        retVal['type'] = self.type
        retVal['update_num'] = self.update_num
        retVal['accuracy'] = round(self.accuracy.compute().item(), 4)
        retVal['calibration_error'] = round(self.calibration_error.compute().item(), 4)
        retVal['brier_score'] = round(self.brier_score.compute().item(), 4)
        retVal['lr'] = lr
        retVal['total_loss'] = round(self.total_loss, 4)
        retVal['average_loss'] = round(self.total_loss/self.num_pred, 4)
        retVal['ppl_per_pred'] = round(self.perplexity/self.num_pred, 4)
        retVal['time_since_last_update'] = round(time.time()-self.last_update)
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
        self.accuracy.reset()
        self.calibration_error.reset()
        self.brier_score.reset()
        self.update_num += 1
        self.last_update = time

    def validated(self):
        self.validations += 1


def save_model(model, output_path, device=None, log_output=None):
    model = model.cpu()
    model_dict = model.save_object()
    torch.save(model_dict, output_path)
    model = model.to(device)
    logging.info(f"SAVING MODEL: {json.dumps(log_output)}")

class Trainer():
    def __init__(self, args):
        self.with_cuda = torch.cuda.is_available() and not args.cpu
        self.device = torch.device("cuda:0" if self.with_cuda else "cpu")
        logger.info(f"Training on device: {self.device}")

        self.output_path = args.output_path
        self.batch_size = args.batch_size

        self.train_dataset = Dataset(args.data_dir, type='train')
        self.train_dataset.load()
        self.train_dataset.set_batch_size(args.batch_size)

        self.validation_dataset = Dataset(args.data_dir, type='valid')
        self.validation_dataset.load()
        self.validation_dataset.set_batch_size(args.batch_size)

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)

        if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
            logging.info('Loading pre-existing model from checkpoint')
            self.model = load_model(args.checkpoint_path, map_location=self.device)
        else:
            self.model = CLD3Model(vocab_size=self.train_dataset.preprocessor.max_hash_value * len(self.train_dataset.preprocessor.ngram_orders),
                                    hidden_dim=args.hidden_dim,
                                    embedding_dim=args.embedding_dim,
                                    label_size=len(self.train_dataset.preprocessor.labels.keys()),
                                    num_ngram_orders=len(self.train_dataset.preprocessor.ngram_orders)).to(self.device)
        logger.info(self.model)

        self.criterion = nn.NLLLoss()
        self.lr = args.lr
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, 1.0, gamma=0.95)

        total_params = sum([p.numel() for p in self.model.parameters()])
        logging.info(f'Training with: {total_params} total parameters')
        if self.with_cuda and torch.cuda.device_count() > 1:
            logging.info(f"Using {torch.cuda.device_count()} GPUSs for training")
            self.model = nn.DataParallel(self.model)
            self.model = self.model.cuda()

        self.best_model = None
        self.results = Results(time.time())


    def run_epoch(self, args, epoch=0):

        for batch_index, (langids, ids, texts, hashes, inputs) in enumerate(self.train_dataset):

            ids = ids.to(self.device)
            inputs = (inputs[0].to(self.device), inputs[1].to(self.device))
            output = self.model(inputs[0], inputs[1])

            probs = torch.exp(output)
            loss = self.criterion(output, ids)
            ppl = torch.exp(F.cross_entropy(output, ids)).item()

            self.results.calculate(loss.item(), ppl, probs, ids)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            if batch_index % args.log_interval == 0:
                logging.info(self.results.get_results(self.scheduler.get_last_lr()[0]))
                self.results.reset(time.time())

            if batch_index % args.validation_interval == 0:
                validation_results = self.validate(args, validation_num = self.results.validations)
                logging.info(validation_results)
                self.results.validated()
                if self.best_model is not None:
                    if validation_results['accuracy'] > self.best_model['accuracy']:
                        self.best_model = validation_results
                        save_model(self.model, os.path.join(self.output_path, 'checkpoint_best.pt'), device=self.device, log_output=self.best_model)
                        logging.info(f"Improved accuracy of {validation_results['accuracy']}")
                    else:
                        if epoch > args.min_epochs and validation_results['validation_num'] - self.best_model['validation_num'] >= args.validation_threshold:
                            logging.info(f"EARLY STOPPING: {json.dumps(self.best_model)}")
                            return 0
                else:
                    self.best_model = validation_results
                    save_model(self.model, os.path.join(self.output_path, 'checkpoint_best.pt'), device=self.device, log_output=self.best_model)
            if args.save_every_epoch:
                save_model(self.model, os.path.join(self.output_path, f'epoch{epoch}.pt'), device=self.device, log_output=self.best_model)
            return 1



    def validate(self, args, validation_num=0):
        self.model.eval()
        valid_results = Results(time.time(), type='VALIDATION')
        with torch.no_grad():
            for batch_index, (langids, ids, texts, hashes, inputs) in enumerate(self.validation_dataset):
                ids = ids.to(self.device)
                inputs = (inputs[0].to(self.device), inputs[1].to(self.device))
                output = self.model(inputs[0], inputs[1])

                probs = torch.exp(output)
                loss = self.criterion(output, ids)
                ppl = torch.exp(F.cross_entropy(output, ids)).item()

                valid_results.calculate(loss.item(), ppl, probs, ids)

        self.model.train()
        ret_results = valid_results.get_results(self.scheduler.get_last_lr()[0])
        ret_results["validation_num"] = validation_num
        return ret_results


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, type=str)
    parser.add_argument("--output_path", required=True, type=str)
    parser.add_argument("--checkpoint_path", type=str, default=None)
    parser.add_argument('--log_interval', type=int, default=1000)
    parser.add_argument('--validation_interval', type=int, default=25000)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--tb_dir", type=str, default=None)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--embedding_dim", type=int, default=256)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--min_epochs", type=int, default=25)
    parser.add_argument("--max_epochs", type=int, default=100)
    parser.add_argument("--save_every_epoch", action="store_true", default=False)
    parser.add_argument("--hidden_dim", type=int, default=256)

    args = parser.parse_args()
    return args


def main(args):
    trainer = Trainer(args)
    for ep in range(args.min_epochs):
        logger.info(f"Beginning epoch {ep}")
        epoch_finish = trainer.run_epoch(args, ep)
        if epoch_finish == 0:
            break
        trainer.scheduler.step()




if __name__ == "__main__":
    args = parse_args()
    main(args)
