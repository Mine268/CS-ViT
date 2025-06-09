from dataclasses import dataclass, asdict, field
from typing import *
import json


@dataclass
class FinetuneConfig():
    # Experiments
    exp: str = field(default=None)
    epoch: int = 30

    # Model
    backbone: str = field(default=None)
    num_joints: int = 16
    num_spatial_layer: int = 6
    spatial_layer_type: str = "decoder"
    num_temporal_layer: int = 2
    temporal_init_method: str = "zero"
    img_size: int = 256
    expansion_ratio: float = 1.25
    trope_scalar: float = 20.0  # 20 ms -> 1 step
    num_latent_layer: int = field(default=None)
    persp_embed_method: str = "dense"
    persp_decorate: str = "query"

    # Dataset
    data: List[str] = field(default=None)
    seq_len: int = field(default=None)
    batch_size: int = field(default=None)
    # source
    ih26mseq_root: str = "/data_1/datasets_temp/InterHand2.6M_5fps_batch1"
    ho3d_root: str = "/data_1/datasets_temp/HO3D_v3"
    dexycb_root: str = "/data_1/datasets_temp/dexycb"

    # Train
    phase: str = "inference"
    temporal_supervision: str = "full"
    spatial_ckpt: str = field(default=None)
    lr: float = 1e-4
    lr_min: float = 1e-6
    lr_scheduler: str = field(default=None)
    warmup_epoch: int = 1
    cooldown_epoch: int = 10

    # Evaluation
    eval_ckpt: str = field(default=None)

    # member functions
    def update(self, other: Union['FinetuneConfig', Dict[str, Any]]):
        if isinstance(other, FinetuneConfig):
            merge_dict = other.to_dict()
        elif isinstance(other, dict):
            merge_dict = other
        else:
            raise TypeError("can only merge from Config/dict")

        for key, value in merge_dict.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise KeyError(f"Unexpected key: {key}.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=4)


default_finetune_cfg = FinetuneConfig()