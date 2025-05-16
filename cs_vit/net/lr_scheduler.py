from typing import *

import math
import numpy as np
import torch
from torch.optim.lr_scheduler import LambdaLR


def gen_cosine_scheduler_array(
    base_value, final_value, epochs, niter_per_ep, warmup_epochs=0, start_warmup_value=0
):
    warmup_schedule = np.array([])
    warmup_iters = warmup_epochs * niter_per_ep
    if warmup_epochs > 0:
        warmup_schedule = np.linspace(start_warmup_value, base_value, warmup_iters)

    iters = np.arange(epochs * niter_per_ep - warmup_iters)
    schedule = final_value + 0.5 * (base_value - final_value) * (
        1 + np.cos(np.pi * iters / len(iters))
    )

    schedule = np.concatenate((warmup_schedule, schedule))
    assert len(schedule) == epochs * niter_per_ep
    return schedule


def warmup_scheduler(
    optimizer: torch.optim.Optimizer,
    max_lr: float,
    min_lr: float,
    warmup_epochs: int,
    annealing_epochs: int,
    steps_per_epoch: int,
) -> LambdaLR:
    """Warmup lr scheduler."""
    assert warmup_epochs >= 0, "warmup_epochs>=0"
    assert annealing_epochs >= 0, "annealing_epochs>=0"
    assert max_lr > min_lr >= 0.0, "max_lr>min_lr>=0.0"
    assert steps_per_epoch > 0

    warmup_steps = warmup_epochs * steps_per_epoch
    annealing_steps = annealing_epochs * steps_per_epoch

    def lr_lambda(current_step: int) -> float:
        # linear increase
        if current_step < warmup_steps:
            if warmup_steps == 0:
                return 1.0
            return current_step / warmup_steps
        # cosine annealing
        elif current_step < (warmup_steps + annealing_steps):
            progress = (current_step - warmup_steps) / annealing_steps
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            scaled_min = min_lr / max_lr
            return scaled_min + (1 - scaled_min) * cosine_decay

        # constant
        return min_lr / max_lr

    return LambdaLR(optimizer, lr_lambda=lr_lambda, last_epoch=-1)
