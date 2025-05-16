import math
from copy import deepcopy
from warnings import warn

from einops import *
import torch
import torch.nn as nn
from transformers import ViTModel
from transformers.models.vit_mae.modeling_vit_mae import (
    ViTMAELayer,
    get_2d_sincos_pos_embed,
    ViTMAEDecoderOutput
)


class PositionalEncoding(nn.Module):
    """PE module"""
    def __init__(self, d_model: int, max_len: int = 512, mode: str = 'absolute'):
        super(PositionalEncoding, self).__init__()
        self.mode = mode
        self.d_model = d_model

        if mode == 'absolute':
            self.pe = nn.Embedding(max_len, d_model)
            self.register_buffer('positions', torch.arange(max_len))
        elif mode == 'relative':
            self.max_rel_dist = max_len
            self.rel_k = nn.Parameter(torch.randn(2*max_len+1, d_model)//math.sqrt(d_model))
        elif mode == "trope":
            if d_model % 2 != 0:
                raise ValueError(f"d_model must be even for RoPE, but got {d_model}")
            inv_freq = 1.0 / (10000 ** (torch.arange(0, d_model, 2).float() / d_model))
            self.register_buffer("inv_freq", inv_freq)
        else:
            raise ValueError(f"Unsupported position mode: {mode}")

    def forward(self, x: torch.Tensor, t: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            x (Tensor): [batch, seq, dim]
            t (Tensor): [batch, seq], is `None` in abs and rel mode
        """
        seq_len = x.size(1)

        if self.mode == 'absolute':
            positions = self.positions[:seq_len].unsqueeze(0)  # [1, seq]
            pos_embed = self.pe(positions)  # [1, seq, dim]
            return x + pos_embed
        elif self.mode == 'relative':
            rel_dist = torch.arange(seq_len)[:, None] - torch.arange(seq_len)[None, :]  # [seq, seq]
            rel_dist = rel_dist.clamp(-self.max_rel_dist, self.max_rel_dist) + self.max_rel_dist
            rel_bias = self.rel_k[rel_dist]  # [seq, seq, dim]
            return x + rel_bias[None,:,:,:].sum(dim=2)
        elif self.mode == "trope":
            if t is None:
                raise ValueError("t must be provided for 'trope' mode")
            _, seq_len = t.size()
            # Preprocess t: t_processed = last_t - t
            t_last = t[:, -1].unsqueeze(1)  # [batch, 1]
            t_processed = t_last - t  # [batch, seq_len]
            # Compute frequencies
            # [batch, seq, d_model//2]
            freqs = t_processed.float().unsqueeze(-1) * self.inv_freq.unsqueeze(0)
            # Apply RoPE
            cos_vals = torch.cos(freqs)  # [batch, seq, d_model//2]
            sin_vals = torch.sin(freqs)
            x_rotated = self._apply_rope(x, cos_vals, sin_vals)
            return x_rotated

    def _apply_rope(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
        """Applies RoPE to input tensor x using precomputed cos/sin values."""
        # Reshape x to [batch, seq, d_model//2, 2]
        x_reshaped = x.view(*x.shape[:-1], -1, 2)
        # Split into components
        x1, x2 = x_reshaped.unbind(dim=-1)
        # Apply rotation
        x1_rot = x1 * cos - x2 * sin
        x2_rot = x1 * sin + x2 * cos
        # Recombine and flatten
        x_rotated = torch.stack([x1_rot, x2_rot], dim=-1)
        return x_rotated.flatten(start_dim=-2)


class RoPE2DPositionalEncoding(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_p: int, num_q: int,
        num_point: int
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_p = num_p
        self.num_q = num_q
        self.num_point = num_point

        self.embedding = nn.Parameter(
            torch.randn(size=(self.num_point, self.embed_dim), dtype=torch.float32),
            requires_grad=True
        )

        self.center_p = (num_p - 1) / 2
        self.center_q = (num_q - 1) / 2
        self.freq_base = 10000.0

        self.post_init()

    def post_init(self):
        p, q = torch.meshgrid(
            torch.arange(self.num_p), torch.arange(self.num_q), indexing="ij"
        )

        delta_p = p.float() - self.center_p
        delta_q = q.float() - self.center_q
        distances = torch.sqrt(delta_p**2 + delta_q**2)
        max_distance = torch.sqrt(torch.tensor([self.center_p**2 + self.center_q**2]))
        norm_distances = torch.clamp(distances / max_distance, 0.0, 1.0)

        sample_coords = norm_distances * (self.num_point - 1)

        theta = torch.atan2(delta_q, delta_p)  # [-π, π]

        half_dim = self.embed_dim // 2
        freq = 1.0 / (
            self.freq_base ** (torch.arange(0, half_dim, 1).float() / half_dim)
        )

        pos_theta = torch.einsum("pq,d->pqd", theta, freq)
        rot_cos = torch.cos(pos_theta).unsqueeze(-1)
        rot_sin = torch.sin(pos_theta).unsqueeze(-1)
        rot_matrix = torch.cat([rot_cos, -rot_sin, rot_sin, rot_cos], dim=-1)
        rot_matrix = rot_matrix.view(self.num_p, self.num_q, -1, 2, 2)

        self.register_buffer("sample_coords", sample_coords)
        self.register_buffer("rot_matrix", rot_matrix)
        self.register_buffer("pos_floor", torch.floor(sample_coords).long())
        self.register_buffer("pos_ceil", torch.ceil(sample_coords).long())
        self.register_buffer(
            "alpha", (sample_coords - torch.floor(sample_coords)).unsqueeze(-1)
        )

    def _interpolate_embeddings(self):
        pos_floor = torch.clamp(self.pos_floor, 0, self.num_point - 1)
        pos_ceil = torch.clamp(self.pos_ceil, 0, self.num_point - 1)

        emb_floor = self.embedding[pos_floor]
        emb_ceil = self.embedding[pos_ceil]

        return emb_floor * (1 - self.alpha) + emb_ceil * self.alpha

    def forward(self, patches: torch.Tensor):
        x = rearrange(patches, "b (p q) c -> b p q c", p=self.num_p, q=self.num_q)
        dist_emb = self._interpolate_embeddings()  # [p, q, c]
        encoded = x + dist_emb.unsqueeze(0)
        half_dim = self.embed_dim // 2
        x_rot = encoded.view(*encoded.shape[:3], half_dim, 2)  # [b,p,q,d/2,2]
        rotated = torch.einsum("pqdrc,bpqdc->bpqdr", self.rot_matrix, x_rot)
        return rearrange(rotated, "b p q d r -> b (p q) (d r)")


class ContinuousAngleEmbedding(nn.Module):
    def __init__(
        self,
        output_dim=64,
        num_freq=16,
        learnable_freq=True,
        max_angle=2*math.pi,
        epsilon=1e-6
    ):
        super().__init__()
        self.output_dim = output_dim
        self.num_freq = num_freq
        self.max_angle = max_angle
        self.epsilon = epsilon

        self.freq_base = nn.Parameter(
            torch.logspace(0, 1, num_freq, base=10).float(),
            requires_grad=learnable_freq
        )

        self.proj = nn.Sequential(
            nn.Linear(2 * num_freq, output_dim),
            nn.GELU(),
            nn.LayerNorm(output_dim)
        )

    def forward(self, angles):
        """
        Args:
            angles (torch.Tensor): Size=[...]

        Returns:
            (torch.Tensor): Size=[..., output_dim]
        """
        # normalize to [0,1], then scale to [0,2 pi]
        angles %= self.max_angle
        angles = angles / self.max_angle * 2 * math.pi

        scaled_angles = angles.unsqueeze(-1) * self.freq_base

        sin_enc = torch.sin(scaled_angles)  # [..., num_freq]
        cos_enc = torch.cos(scaled_angles)  # [..., num_freq]
        raw_enc = torch.cat([sin_enc, cos_enc], dim=-1)

        embeddings = self.proj(raw_enc)
        return embeddings


class LoraCompatibleMHA(nn.Module):
    """LoRA compatible multi-head attention
    """
    def __init__(self, embed_dim: int, num_heads: int):
        super(LoraCompatibleMHA, self).__init__()
        warn(
            f"{LoraCompatibleMHA.__name__} has been deprecated. Use {MHA.__name__} instead.",
            DeprecationWarning,
            stacklevel=2
        )
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.mha = nn.MultiheadAttention(
            embed_dim, num_heads,
            kdim=embed_dim, vdim=embed_dim,
            batch_first=True
        )

    def forward(self, query, key, value):
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        return self.mha(q, k, v, need_weights=False)[0]


class MHA(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(MHA, self).__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.inv_sqrt_head_dim = 1 / (self.head_dim ** 0.5)

        self.query = nn.Linear(embed_dim, embed_dim)
        self.key = nn.Linear(embed_dim, embed_dim)
        self.value = nn.Linear(embed_dim, embed_dim)
        self.output = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): `(B,L,D)`
            ctx (torch.Tensor): `(B,S,D)`

        Returns:
            torch.Tensor: `(B,L,D)`
        """
        B, L, _ = x.shape
        _, S, _ = ctx.shape

        q = self.query(x)    # (B, L, D)
        k = self.key(ctx)    # (B, S, D)
        v = self.value(ctx)  # (B, S, D)

        # [B, L, D] -> [B, num_heads, L, head_dim]
        q = q.view(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        # score
        attn_scores = torch.matmul(q, k.transpose(-2, -1))  # (B, H, L, S)
        attn_scores = attn_scores / self.inv_sqrt_head_dim  # scale
        attn_weights = torch.softmax(attn_scores, dim=-1)  # (B, H, L, S)

        # atten fuse
        context = torch.matmul(attn_weights, v)  # (B, H, L, head_dim)

        # [B, H, L, D/H] -> [B, L, D]
        context = context.transpose(1, 2).contiguous().view(B, L, self.embed_dim)

        return self.output(context)  # (B, L, D)


class FeedForwardNetwork(nn.Module):
    """FFN
    """
    def __init__(self, dim):
        super(FeedForwardNetwork, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, 4 * dim),
            nn.GELU(),
            nn.Linear(4 * dim, dim)
        )

    def forward(self, x):
        return self.net(x)


class EncoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super(EncoderBlock, self).__init__()
        # self.pe = PositionalEncoding(dim, mode='absolute')
        self.attn = MHA(dim, num_heads)
        self.ffn = FeedForwardNetwork(dim)
        self.norm1 = nn.BatchNorm1d(dim)
        self.norm2 = nn.BatchNorm1d(dim)

    def forward(self, x):
        # x = self.pe(x)

        y = self.norm1(x.transpose(-1, -2)).transpose(-1, -2)
        y = self.attn(y, y)
        x = x + y

        y = self.norm2(x.transpose(-1, -2)).transpose(-1, -2)
        y = self.ffn(y)
        x = x + y
        return x


class DecoderBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super(DecoderBlock, self).__init__()
        # self.pe = PositionalEncoding(dim, mode='absolute')
        self.self_atten = MHA(dim, num_heads)
        self.cross_atten = MHA(dim, num_heads)
        self.ffn = FeedForwardNetwork(dim)
        self.norm1 = nn.BatchNorm1d(dim)
        self.norm2 = nn.BatchNorm1d(dim)
        self.norm3 = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor, ref: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Shape=[N,L1,D], used as value
            ref (torch.Tensor): Shape=[N,L2,D], used as key & query
        """
        # x = self.pe(x)

        y = self.norm1(x.transpose(-1, -2)).transpose(-1, -2)
        y = self.self_atten(y, y)
        x = x + y

        y = self.norm2(x.transpose(-1, -2)).transpose(-1, -2)
        y = self.cross_atten(y, ref)
        x = x + y

        y = self.norm3(x.transpose(-1, -2)).transpose(-1, -2)
        y = self.ffn(y)
        x = x + y

        return x


class CrossAttnDecoder(nn.Module):
    def __init__(self, dim: int, num_heads: int):
        super(CrossAttnDecoder, self).__init__()
        self.cross_atten = MHA(dim, num_heads)
        self.ffn = FeedForwardNetwork(dim)
        self.norm1 = nn.BatchNorm1d(dim)
        self.norm2 = nn.BatchNorm1d(dim)

    def forward(self, x: torch.Tensor, ref: torch.Tensor):
        """
        Args:
            x (torch.Tensor): Shape=[N,L1,D], used as value
            ref (torch.Tensor): Shape=[N,L2,D], used as key & query
        """
        y = self.norm1(x.transpose(-1, -2)).transpose(-1, -2)
        y = self.cross_atten(y, ref)
        x = x + y

        y = self.norm2(x.transpose(-1, -2)).transpose(-1, -2)
        y = self.ffn(y)
        x = x + y

        return x


# ref: huggingface transformer
# manually remove mask mechanics, compatible with origin ViTMAEDecoder checkpoint
class ViTMAEDecoder_NoMask(nn.Module):
    def __init__(self, config, num_patches):
        super().__init__()
        self.decoder_embed = nn.Linear(config.hidden_size, config.decoder_hidden_size, bias=True)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, config.decoder_hidden_size))
        self.decoder_pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, config.decoder_hidden_size), requires_grad=False
        )  # fixed sin-cos embedding

        decoder_config = deepcopy(config)
        decoder_config.hidden_size = config.decoder_hidden_size
        decoder_config.num_hidden_layers = config.decoder_num_hidden_layers
        decoder_config.num_attention_heads = config.decoder_num_attention_heads
        decoder_config.intermediate_size = config.decoder_intermediate_size
        self.decoder_layers = nn.ModuleList(
            [ViTMAELayer(decoder_config) for _ in range(config.decoder_num_hidden_layers)]
        )

        self.decoder_norm = nn.LayerNorm(config.decoder_hidden_size, eps=config.layer_norm_eps)
        self.decoder_pred = nn.Linear(
            config.decoder_hidden_size, config.patch_size**2 * config.num_channels, bias=True
        )  # encoder to decoder
        self.gradient_checkpointing = False
        self.config = config
        self.initialize_weights(num_patches)

    def interpolate_pos_encoding(self, embeddings: torch.Tensor) -> torch.Tensor:
        """
        This method is a modified version of the interpolation function for ViT-mae model at the decoder, that
        allows to interpolate the pre-trained decoder position encodings, to be able to use the model on higher
        resolution images.

        Adapted from:
        https://github.com/facebookresearch/dino/blob/de9ee3df6cf39fac952ab558447af1fa1365362a/vision_transformer.py#L174
        """

        # -1 removes the class dimension since we later append it without interpolation
        embeddings_positions = embeddings.shape[1] - 1

        # Separation of class token and patch tokens
        class_pos_embed = self.decoder_pos_embed[:, :1]
        patch_pos_embed = self.decoder_pos_embed[:, 1:]

        # To retain the final 3d tensor with the required dimensions
        dim = self.decoder_pos_embed.shape[-1]

        # Increasing a dimension to enable bicubic interpolation
        patch_pos_embed = patch_pos_embed.reshape(1, 1, -1, dim)

        # permute to bring the dimension to be interpolated, to the last
        patch_pos_embed = patch_pos_embed.permute(0, 3, 1, 2)

        # Interpolating the decoder position embeddings shape wrt embeddings shape i.e (x).
        # we keep the second last dimension constant
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed,
            size=(patch_pos_embed.shape[-2], embeddings_positions),
            mode="bicubic",
            align_corners=False,
        )

        # Converting back to the original shape
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        # Adding the class token back
        return torch.cat((class_pos_embed, patch_pos_embed), dim=1)

    def initialize_weights(self, num_patches):
        # initialize (and freeze) position embeddings by sin-cos embedding
        decoder_pos_embed = get_2d_sincos_pos_embed(
            self.decoder_pos_embed.shape[-1], int(num_patches**0.5), add_cls_token=True
        )
        self.decoder_pos_embed.data.copy_(torch.from_numpy(decoder_pos_embed).float().unsqueeze(0))

        # timm's trunc_normal_(std=.02) is effectively normal_(std=0.02) as cutoff is too big (2.)
        torch.nn.init.normal_(self.mask_token, std=self.config.initializer_range)

    def forward(
        self,
        hidden_states,
        # ids_restore,
        output_attentions=False,
        output_hidden_states=False,
        return_dict=True,
        interpolate_pos_encoding: bool = False,
    ):
        # embed tokens
        x = self.decoder_embed(hidden_states)

        # mask unshuffle is ignored

        # add pos embed
        if interpolate_pos_encoding:
            decoder_pos_embed = self.interpolate_pos_encoding(x)
        else:
            decoder_pos_embed = self.decoder_pos_embed
        hidden_states = x + decoder_pos_embed

        # apply Transformer layers (blocks)
        all_hidden_states = () if output_hidden_states else None
        all_self_attentions = () if output_attentions else None
        for i, layer_module in enumerate(self.decoder_layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)

            if self.gradient_checkpointing and self.training:
                layer_outputs = self._gradient_checkpointing_func(
                    layer_module.__call__,
                    hidden_states,
                    None,
                    output_attentions,
                )
            else:
                layer_outputs = layer_module(hidden_states, head_mask=None, output_attentions=output_attentions)

            hidden_states = layer_outputs[0]

            if output_attentions:
                all_self_attentions = all_self_attentions + (layer_outputs[1],)

        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)

        hidden_states = self.decoder_norm(hidden_states)

        # predictor projection
        logits = self.decoder_pred(hidden_states)

        # remove cls token
        logits = logits[:, 1:, :]

        if not return_dict:
            return tuple(v for v in [logits, all_hidden_states, all_self_attentions] if v is not None)
        return ViTMAEDecoderOutput(
            logits=logits,
            hidden_states=all_hidden_states,
            attentions=all_self_attentions,
        )


# vit model converted from mae
class ViTModelFromMAE(ViTModel):
    def __init__(self, *args, **kwargs):
        super(ViTModelFromMAE, self).__init__(*args, **kwargs)
        # remove pooler, align with MAE vit
        self.pooler = nn.Identity()
