import argparse
import os
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from preprocessor import Dataset
from models import CLD3Model

logging.basicConfig(format="%(message)s", level=logging.DEBUG)
logger = logging.getLogger('gcld3')

class Results():
    def __init__(self, time):
        self.total_loss = 0
        self.perplexity = 0
        self.correct_pred = 0
        self.num_pred = 0
        self.update_num = 0
        self.batches = 0
        self.last_update = time
        self.start = time
        self.validations = 0
        self.type = 'TRAINING'

    def calculate(self, loss, ppl, predictions, labels):
        self.total_loss += loss
        self.perplexity += ppl
        self.num_pred += len(predictions)
        preds = predictions ^ labels
        self.correct_pred = (preds==0).sum().item()
        self.batches += 1

    def get_results(self, lr):
        retVal = {}
        retVal['type'] = self.type
        retVal['update_num'] = self.update_num
        retVal['acc'] = (self.correct_pred)/self.num_pred
        retVal['lr'] = lr
        retVal['total_loss'] = self.total_loss
        retVal['average_loss'] = self.total_loss/self.num_pred
        retVal['ppl_per_pred'] = self.perplexity/self.num_pred
        retVal['time_since_last_update'] = time.time()-self.last_update
        retVal['predictions_per_second'] = self.num_pred/retVal['time_since_last_update']
        retVal['time_passed'] = time.time()-self.start
        retVal['validations'] = self.validations
        retVal['num_pred'] = self.num_pred
        return retVal

    # add perplexity
    def reset(self, time):
        self.total_loss = 0
        self.perplexity = 0
        self.correct_pred = 0
        self.num_pred = 0
        self.update_num += 1
        self.last_update = time

    def validated(self):
        self.validations += 1


class Trainer():
    def __init__(self, args):
        self.with_cuda = torch.cuda.is_available() and not args.cpu
        self.device = torch.device("cuda:0" if self.with_cuda else "cpu")
        logger.info(f"Training on device: {self.device}")

        self.output_path = args.output_path
        self.batch_size = args.batch_size

        self.dataset = Dataset(args.data_dir)
        self.dataset.load()

        if not os.path.exists(args.output_path):
            os.makedirs(args.output_path, exist_ok=True)

        if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
            logging.info('Loading pre-existing model from checkpoint')
            self.model = load_model(args.checkpoint_path, map_location=self.device)
        else:
            self.model = CLD3Model(vocab_size=self.dataset.preprocessor.max_hash_value * len(self.dataset.preprocessor.ngram_orders),
                                    hidden_dim=args.hidden_dim,
                                    embedding_dim=args.embedding_dim,
                                    label_size=len(self.dataset.preprocessor.labels.keys()),
                                    num_ngram_orders=len(self.dataset.preprocessor.ngram_orders)).to(self.device)
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

        self.results = Results(time.time())


    def run_epoch(self, args):

        for batch_index, (langids, ids, texts, hashes, inputs) in enumerate(self.dataset):

            ids = ids.to(self.device)
            inputs = (inputs[0].to(self.device), inputs[1].to(self.device))
            output = self.model(inputs[0], inputs[1])

            loss = self.criterion(output, ids)
            ppl = torch.exp(F.cross_entropy(output, ids)).item()
            pred = output.argmax(1)

            self.results.calculate(loss.item(), ppl, pred, ids)

            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()

            if batch_index % 100 == 0:
                print(self.results.get_results(self.scheduler.get_last_lr()[0]))
                self.results.reset(time.time())




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
    parser.add_argument("--hidden_dim", type=int, default=256)

    args = parser.parse_args()
    return args


def main(args):
    trainer = Trainer(args)
    for ep in range(10):
        trainer.run_epoch(args)



if __name__ == "__main__":
    args = parse_args()
    main(args)
