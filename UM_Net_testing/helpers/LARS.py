#------------------------------------------------------------------------------#
# File name         : LARS.py
# Purpose           : This file runs Large Batch Training of CNNs
#                     Read paper: https://arxiv.org/abs/1708.03888
#
# Authors           : Shujah Ur Rehman, Dr. Hassan Mohy-ud-Din
# Email             : 21060003@lums.edu.pk, hassan.mohyuddin@lums.edu.pk
#
# Notes:            : Code is borrowed from github.com/haooyuee/BarlowTwins-CXR/
#
# Last Date         : March 13, 2025
#------------------------------------------------------------------------------#

import os, sys, random, torch, math
import numpy                                                    as np
import torch.optim                                              as optim

#------------------------------------------------------------------------------#
class LARS(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9, eta=0.001,
                 weight_decay_filter=False, lars_adaptation_filter=False):

        defaults = dict(lr                      = lr,
                        weight_decay            = weight_decay,
                        momentum                = momentum,
                        eta                     = eta,
                        weight_decay_filter     = weight_decay_filter,
                        lars_adaptation_filter  = lars_adaptation_filter)

        super().__init__(params, defaults)

    #--------------------------------------------------------------------------#
    def exclude_bias_and_norm(self, p):
        return p.ndim == 1

    #--------------------------------------------------------------------------#
    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp              = p.grad

                if dp is None:
                    continue

                if not g['weight_decay_filter'] or not self.exclude_bias_and_norm(p):
                    dp          = dp.add(p, alpha=g['weight_decay'])

                if not g['lars_adaptation_filter'] or not self.exclude_bias_and_norm(p):
                    param_norm  = torch.norm(p)
                    update_norm = torch.norm(dp)
                    one         = torch.ones_like(param_norm)
                    q           = torch.where(param_norm > 0.,
                                              torch.where(update_norm > 0,
                                              (g['eta'] * param_norm / update_norm), one), one)
                    dp          = dp.mul(q)

                param_state     = self.state[p]
                if 'mu' not in param_state:
                    param_state['mu'] = torch.zeros_like(p)

                mu              = param_state['mu']
                mu.mul_(g['momentum']).add_(dp)

                p.add_(mu, alpha=-g['lr'])

#------------------------------------------------------------------------------#
def adjust_learning_rate(self, optimizer, step):
    warmup_steps    = 10 * self.num_batches
    base_lr         = self.batch_size / 256

    if step < warmup_steps:
        lr          = base_lr * step / warmup_steps
    else:
        step                    -= warmup_steps
        self.total_iterations   -= warmup_steps

        q           = 0.5 * (1 + math.cos(math.pi * step / self.total_iterations))
        end_lr      = base_lr * 0.001
        lr          = base_lr * q + end_lr * (1 - q)

    #optimizer.param_groups[0]['lr'] = lr * self.learning_rate_weights
    #optimizer.param_groups[1]['lr'] = lr * self.learning_rate_biases

    optimizer.param_groups[0]['lr'] = lr * 0.20
    optimizer.param_groups[1]['lr'] = lr * 0.0048

#------------------------------------------------------------------------------#
