# ------------------------------------------------------------------------------#
#
# File name                 : scheduler_switcher.py
# Purpose                   : Switch between different learning rate schedulers for LDM training
# Usage                     : Imported by train.py
#
# Authors                   : Talha Ahmed, Nehal Ahmed Shaikh, Hassan Mohy-ud-Din
# Email                     : 24100033@lums.edu.pk, 202410001@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Last Modified             : May 23, 2025
# ------------------------------------------------------------------------------#
import numpy as np
import torch
from   torch.optim.lr_scheduler        import CosineAnnealingLR
from   diffusers.optimization          import get_cosine_schedule_with_warmup

# ------------------------------------------------------------------------------ #
class LambdaWarmUpCosineScheduler:
    def __init__(self, warm_up_steps, lr_min, lr_max, lr_start, max_decay_steps, verbosity_interval=0):
        self.lr_warm_up_steps = warm_up_steps
        self.lr_min = lr_min
        self.lr_max = lr_max
        self.lr_start = lr_start
        self.lr_max_decay_steps = max_decay_steps
        self.last_lr = 0.
        self.verbosity_interval = verbosity_interval

    def schedule(self, n):
        n = np.asarray(n)
        if n < self.lr_warm_up_steps:
            lr = (self.lr_max - self.lr_start) / self.lr_warm_up_steps * n + self.lr_start
        else:
            t = (n - self.lr_warm_up_steps) / (self.lr_max_decay_steps - self.lr_warm_up_steps)
            t = min(t, 1.0)
            lr = self.lr_min + 0.5 * (self.lr_max - self.lr_min) * (1 + np.cos(t * np.pi))
        self.last_lr = lr
        return lr

    def __call__(self, n):
        return self.schedule(n)
    
# ------------------------------------------------------------------------------ #
class SchedulerSwitcher:
    def __init__(self, optimizer, total_steps, warmup_steps, scheduler_type="cosine_warmup", **kwargs):
        self.optimizer      = optimizer
        self.total_steps    = total_steps
        self.warmup_steps   = warmup_steps
        self.scheduler_type = scheduler_type.lower()
        self.kwargs         = kwargs
        self.scheduler      = self._initialize_scheduler()

    def _initialize_scheduler(self):
        if self.scheduler_type == "cosine_warmup":
            return get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps,
                **self.kwargs
            )
        elif self.scheduler_type == "lambda_warmup_cosine":
            return torch.optim.lr_scheduler.LambdaLR(
                self.optimizer,
                lr_lambda=LambdaWarmUpCosineScheduler(
                    warm_up_steps=self.warmup_steps,
                    lr_min=self.kwargs.get("lr_min", 0.0),
                    lr_max=self.kwargs.get("lr_max", 1.0),
                    lr_start=self.kwargs.get("lr_start", 0.0),
                    max_decay_steps=self.total_steps,
                    verbosity_interval=self.kwargs.get("verbosity_interval", 0)
                )
            )
        elif self.scheduler_type == "cosine_annealing":
            return CosineAnnealingLR(
                self.optimizer,
                T_max=self.total_steps - self.warmup_steps,
                eta_min=self.kwargs.get("eta_min", 0.0),
                **self.kwargs
            )
        elif self.scheduler_type == "ddpm":
            return get_cosine_schedule_with_warmup(
                optimizer=self.optimizer,
                num_warmup_steps=self.warmup_steps,
                num_training_steps=self.total_steps,
                **self.kwargs
            )
        else:
            raise ValueError(f"Unsupported scheduler type: {self.scheduler_type}")

    def step(self):
        self.scheduler.step()

    def get_last_lr(self):
        return self.scheduler.get_last_lr()
    
# ------------------------------------------------------------------------------ #