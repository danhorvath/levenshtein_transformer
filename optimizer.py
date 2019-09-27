class NoamOpt(object):
    """
    Optim wrapper that implements rate.
    """

    def __init__(self, warmup_init_lr, warmup_end_lr, warmup_updates, optimizer):
        self.optimizer = optimizer
        self._step = 0
        self.warmup_init_lr = warmup_init_lr
        self.warmup_end_lr = warmup_end_lr
        self.warmup_updates = warmup_updates
        self.decay_factor = warmup_end_lr * warmup_updates**0.5

        self._rate = warmup_init_lr

        self.warmup_lr_step = (warmup_end_lr - warmup_init_lr) / warmup_updates

    def step(self):
        """
        Update parameters and rate
        """
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step=None):
        """
        Implement `lrate` above
        """
        if step is None:
            step = self._step

        if step < self.warmup_updates:
            return self.warmup_init_lr + self.step*self.warmup_lr_step
        else:
            return self.decay_factor * self.step**-0.5