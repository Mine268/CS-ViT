from typing import *
from functools import partial

import torch
import torch.nn as nn
from einops import *

from .transformer_module import EncoderBlock, ContinuousAngleEmbedding, RoPE2DPositionalEncoding


class ImageLatentTransformerGroup(nn.Module):
    def __init__(
        self,
        num_layers: int=1,
        embed_dim: int=768,
        num_heads: int=12,
    ):
        """
        Args:
            num_layers (int): Number of transformer block for latent transformations. Defaults to 3.
            embed_dim (int): Image token dim. Defaults to 768.
        """
        super(ImageLatentTransformerGroup, self).__init__()

        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # HF: horizontal flip
        self.hf = nn.Sequential(*[EncoderBlock(dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)])
        # CR: center-oriented rotation
        self.cr = nn.Sequential(*[EncoderBlock(dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)])
        # HR: horizontal flip then center-oriented rotation
        self.hr = nn.Sequential(*[EncoderBlock(dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)])

        # Rotation embedding
        self.angle_embedder = ContinuousAngleEmbedding(output_dim=embed_dim, num_freq=32)

        # key: (first_op, second_op)
        self.composition_law = {
            (self.do_hf, self.do_hf): (self.do_cr, 0, 0),
            (self.do_hf, self.do_cr): (self.do_hr, 0, 1),
            (self.do_hf, self.do_hr): (self.do_cr, 0, 1),
            (self.do_cr, self.do_hf): (self.do_hr, -1, 0),
            (self.do_cr, self.do_cr): (self.do_cr, 1, 1),
            (self.do_cr, self.do_hr): (self.do_hr, -1, 1),
            (self.do_hr, self.do_hf): (self.do_cr, -1, 0),
            (self.do_hr, self.do_cr): (self.do_hr, 1, 1),
            (self.do_hr, self.do_hr): (self.do_cr, -1, 1),
        }

    def __repr__(self):
        return f"ImageLatentTransformerGroup(" + \
            f"num_layer={self.num_layers}, embed_dim={self.embed_dim}, num_heads={self.num_heads})"

    def do_hf(self, patches: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        _, _ = args, kwargs
        """
        Args:
            patches (torch.Tensor): Size=(N,L,D), output size=(N,L,D).
        """
        return self.hf(patches)

    def do_cr(self,
        patches: torch.Tensor, angle_rad: Union[torch.Tensor, List, None], *args, **kwargs
    ) -> torch.Tensor:
        _, _ = args, kwargs
        """
        Args:
            patches (torch.Tensor): As `do_hf`.
            angle_rad (Union[torch.Tensor, List, None]): Angles rotated for each image in radius, \
                size=(N,)
        """
        if angle_rad is None:
            angle_rad = torch.zeros(
                size=(patches.shape[0],),
                device=patches.device,
                dtype=patches.dtype)
        elif isinstance(angle_rad, List):
            angle_rad = torch.tensor(angle_rad, device=patches.device, dtype=patches.dtype)

        angle_embeds = self.angle_embedder(angle_rad)
        patches = torch.concat([angle_embeds[:, None], patches], dim=1)

        return self.cr(patches)[:, 1:]

    def do_hr(self,
        patches: torch.Tensor, angle_rad: Union[torch.Tensor, List, None], *args, **kwargs
    ) -> torch.Tensor:
        _, _ = args, kwargs
        """
        Args:
            patches (torch.Tensor): As `do_hf`.
            angle_rad (Union[torch.Tensor, List, None]): Angles rotated for each image in radius, \
                size=(N,)
        """
        if angle_rad is None:
            angle_rad = torch.zeros(
                size=(patches.shape[0],),
                device=patches.device,
                dtype=patches.dtype)
        elif isinstance(angle_rad, List):
            angle_rad = torch.tensor(angle_rad, device=patches.device, dtype=patches.device)

        angle_embeds = self.angle_embedder(angle_rad)
        patches = torch.concat([angle_embeds[:, None], patches], dim=1)

        return self.hr(patches)[:, 1:]


    def _unwrap_partial(self, op: partial) -> Tuple[Callable, torch.Tensor]:
        if op.func != self.do_hf:
            return op.func, op.keywords['angle_rad']
        else:
            return op.func, None

    def _mix_param(self,
        factor1: int, param1: Optional[torch.Tensor],
        factor2: int, param2: Optional[torch.Tensor],
    ) -> Optional[torch.Tensor]:
        if param1 is not None and param2 is not None:
            return factor1 * param1 + factor2 * param2
        if param1 is None and param2 is None:
            return None
        if param1 is not None:
            return factor1 * param1
        return factor2 * param2

    def compose(self,
        first_op: partial,
        second_op: partial,
    ) -> partial:
        """Compose the latent transformation to produce new one

        Args:
            first_op/second_op (partial): one of `do_{hf,cf,hr}`.

        Returns:
            partial: one of `do_{hf,cf,hr}`, with argument being partialed.
        """
        op1, param1 = self._unwrap_partial(first_op)
        op2, param2 = self._unwrap_partial(second_op)

        result_op, factor1, factor2 = self.composition_law[(op1, op2)]
        result_param = self._mix_param(factor1, param1, factor2, param2)

        return partial(result_op, angle_rad=result_param)

    def get_parameterized_hf(self) -> partial:
        return partial(self.do_hf, angle_rad=None)

    def get_parameterized_cr(self, angle_rad: Union[torch.Tensor, List]) -> partial:
        if isinstance(angle_rad, List):
            angle_rad = torch.tensor(angle_rad)
        return partial(self.do_cr, angle_rad=angle_rad)

    def get_parameterized_hr(self, angle_rad: Union[torch.Tensor, List]) -> partial:
        if isinstance(angle_rad, List):
            angle_rad = torch.tensor(angle_rad)
        return partial(self.do_hr, angle_rad=angle_rad)


class ScaleRotTransformationGroup(nn.Module):
    def __init__(
        self,
        num_layers: int=1,
        embed_dim: int=768,
        num_heads: int=12,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # sr: scaled then rotate
        self.sr = nn.Sequential(*[EncoderBlock(dim=embed_dim, num_heads=num_heads)
            for _ in range(num_layers)])

        # Scale coeff embedding
        self.scale_embedder = ContinuousAngleEmbedding(output_dim=embed_dim, num_freq=32)
        # Rotation embedding
        self.angle_embedder = ContinuousAngleEmbedding(output_dim=embed_dim, num_freq=32)

    def __repr__(self):
        return f"ImageLatentTransformerGroup(" + \
            f"num_layer={self.num_layers}, embed_dim={self.embed_dim}, num_heads={self.num_heads})"

    def do_sr(
        self,
        patches: torch.Tensor,
        scale_ratio: Union[torch.Tensor, List, None],
        angle_rad: Union[torch.Tensor, List, None]
    ) -> torch.Tensor:
        if angle_rad is None:
            angle_rad = torch.zeros(
                size=(patches.shape[0],),
                device=patches.device,
                dtype=patches.dtype)
        elif isinstance(angle_rad, List):
            angle_rad = torch.tensor(angle_rad, device=patches.device, dtype=patches.dtype)

        if scale_ratio is None:
            scale_ratio = torch.zeros(
                size=(patches.shape[0],),
                device=patches.device,
                dtype=patches.dtype)
        elif isinstance(scale_ratio, List):
            scale_ratio = torch.tensor(scale_ratio, device=patches.device, dtype=patches.dtype)

        angle_embeds = self.angle_embedder(angle_rad)
        scale_embeds = self.scale_embedder(scale_ratio)
        patches = torch.concat([scale_embeds[:, None], angle_embeds[:, None], patches], dim=1)

        return self.sr(patches)[:, 2:]

    def _unwrap_partial(self, op: partial) -> Tuple[torch.Tensor, torch.Tensor]:
        return op.keywords["scale_ratio"], op.keywords["angle_rad"]

    def compose(
        self,
        first_op: partial,
        second_op: partial,
    ) -> partial:
        s1, r1 = self._unwrap_partial(first_op)
        s2, r2 = self._unwrap_partial(second_op)
        return partial(
            self.do_sr,
            scale_ratio=s1 * s2,
            angle_rad=r1 + r2,
        )

    def get_parameterized_sr(
        self,
        scale_ratio: Union[torch.Tensor, List],
        angle_rad: Union[torch.Tensor, List]
    ) -> partial:
        if isinstance(scale_ratio, List):
            scale_ratio = torch.Tensor(scale_ratio)
        if isinstance(angle_rad, List):
            angle_rad = torch.Tensor(angle_rad)
        return partial(self.do_sr, scale_ratio=scale_ratio, angle_rad=angle_rad)


class ScaleRotComplexEmbedTransformationGroup(nn.Module):
    def __init__(
        self,
        num_layers: int = 1,
        embed_dim: int = 768,
        num_heads: int = 12,
        num_p: int = 16,
        num_q: int = 16,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.truncated = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # sr: scaled then rotate
        self.rope2d = RoPE2DPositionalEncoding(embed_dim, num_p, num_q, 32)
        self.sr = nn.Sequential(
            *[
                EncoderBlock(dim=embed_dim, num_heads=num_heads)
                for _ in range(num_layers)
            ]
        )

        # Scale coeff embedding
        self.scale_embedder = ContinuousAngleEmbedding(
            output_dim=embed_dim, num_freq=32
        )
        self.scale_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )
        # Rotation embedding
        self.angle_embedder = ContinuousAngleEmbedding(
            output_dim=embed_dim, num_freq=32
        )
        self.angle_linear = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
        )

    def __repr__(self):
        return (
            f"ImageLatentTransformerGroup("
            + f"num_layer={self.num_layers}, embed_dim={self.embed_dim}, num_heads={self.num_heads})"
        )

    def truncate(self, l):
        self.truncated = min(1, max(l, self.num_layers))

    def do_sr(
        self,
        patches: torch.Tensor,
        scale_ratio: Union[torch.Tensor, List, None],
        angle_rad: Union[torch.Tensor, List, None],
    ) -> torch.Tensor:
        if angle_rad is None:
            angle_rad = torch.zeros(
                size=(patches.shape[0],), device=patches.device, dtype=patches.dtype
            )
        elif isinstance(angle_rad, List):
            angle_rad = torch.tensor(
                angle_rad, device=patches.device, dtype=patches.dtype
            )

        if scale_ratio is None:
            scale_ratio = torch.zeros(
                size=(patches.shape[0],), device=patches.device, dtype=patches.dtype
            )
        elif isinstance(scale_ratio, List):
            scale_ratio = torch.tensor(
                scale_ratio, device=patches.device, dtype=patches.dtype
            )

        patches = self.rope2d(patches)

        angle_embeds = self.scale_linear(self.angle_embedder(angle_rad))
        scale_embeds = self.angle_linear(self.scale_embedder(scale_ratio))
        # patches = torch.concat(
        #     [scale_embeds[:, None], angle_embeds[:, None], patches], dim=1
        # )
        patches = scale_embeds[:, None] * patches + angle_embeds[:, None]

        return self.sr[:self.truncated](patches)  # [:, 2:]

    def _unwrap_partial(self, op: partial) -> Tuple[torch.Tensor, torch.Tensor]:
        return op.keywords["scale_ratio"], op.keywords["angle_rad"]

    def compose(
        self,
        first_op: partial,
        second_op: partial,
    ) -> partial:
        s1, r1 = self._unwrap_partial(first_op)
        s2, r2 = self._unwrap_partial(second_op)
        return partial(
            self.do_sr,
            scale_ratio=s1 * s2,
            angle_rad=r1 + r2,
        )

    def get_parameterized_sr(
        self,
        scale_ratio: Union[torch.Tensor, List],
        angle_rad: Union[torch.Tensor, List],
    ) -> partial:
        if isinstance(scale_ratio, List):
            scale_ratio = torch.Tensor(scale_ratio)
        if isinstance(angle_rad, List):
            angle_rad = torch.Tensor(angle_rad)
        return partial(self.do_sr, scale_ratio=scale_ratio, angle_rad=angle_rad)
