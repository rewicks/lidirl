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
