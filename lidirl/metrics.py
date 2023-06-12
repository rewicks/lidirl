#!/usr/bin/env python3

"""
    Keeps track of any additional/custom metrics for scoring during training.

    Right now limited to Brier Score which isn't implemented by torchmetrics
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

import torchmetrics
import torch
import torch.nn.functional as F

class BrierScore(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("sum_of_squares", default=torch.tensor(0.0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):

        one_hot_target = F.one_hot(target)
       
        try: 
            self.sum_of_squares += torch.sum(torch.pow((preds-one_hot_target), 2))
        except:
            return -1
        self.total += target.numel()

    def compute(self):
        return self.sum_of_squares / self.total

#################################### FUNCTIONALITY ####################################
import time
import torchmetrics.classification as tmc

class Results():
    def __init__(self, time, num_labels, length=1, device=None, type='TRAINING'):
        self.total_loss = 0
        self.perplexity = 0
        self.accuracy = tmc.Accuracy(task="multiclass", num_classes=num_labels).to(device)
        self.calibration_error = tmc.CalibrationError(task="multiclass", 
                                                            num_classes=num_labels, 
                                                            n_bins=10).to(device)
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
        retVal['step'] = self.update_num
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

    def log(self, log_string):
        logger.log(log_string)
