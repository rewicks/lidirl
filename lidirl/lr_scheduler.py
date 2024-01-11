class NoamOpt():
    "Optim wrapper that implements rate."
    def __init__(self, lr, warmup, warmup_lr, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.lr = lr
        self.warmup = warmup if warmup is not None else 1
        self.warmup_lr = warmup_lr if (warmup_lr is not None and warmup is not None) else lr

        self.warmup_step_size = (self.lr - self.warmup_lr) / self.warmup
        self.decay_factor = self.lr * self.warmup**0.5

        self._rate = self.warmup_lr 
    
    def state_dict(self):
        """Returns the state of the warmup scheduler as a :class:`dict`.
        It contains an entry for every variable in self.__dict__ which
        is not the optimizer.
        """
        return {key: value for key, value in self.__dict__.items() if key != 'optimizer'}
    
    def load_state_dict(self, state_dict):
        """Loads the warmup scheduler's state.
        Arguments:
            state_dict (dict): warmup scheduler state. Should be an object returned
                from a call to :meth:`state_dict`.
        """
        self.__dict__.update(state_dict) 
        
    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()
        
    def rate(self, step = None):
        "Implement `lrate` above"
        if step is None:
            step = self._step
        if step < self.warmup:
            return self.warmup_lr + (self.warmup_step_size * step)
        else:
            # then, decay prop. to the inverse square root of the update number
            return self.decay_factor * step**-0.5