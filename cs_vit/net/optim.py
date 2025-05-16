import torch
import torch.optim as optim
import random


class ColumnRandomUpdateOptimizer(optim.AdamW):
    def __init__(self, params, num_columns_to_update, **kwargs):
        super().__init__(params, **kwargs)
        self.num_columns_to_update = num_columns_to_update

    def step(self, closure=None):
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if p.dim() == 2:
                    grad = p.grad.data
                    in_features = grad.size(1)

                    cols_to_update = random.sample(
                        range(in_features),
                        min(self.num_columns_to_update, in_features)
                    )

                    mask = torch.zeros_like(grad)
                    mask[:, cols_to_update] = 1

                    p.grad.data.mul_(mask)

        super().step(closure)