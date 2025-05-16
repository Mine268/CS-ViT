from dataclasses import dataclass, asdict, field
from typing import *
import json


@dataclass
class PretrainConfig:
    # members
    # experiment
    exp: str
    data: str
    model_dir: str
    decoder_config: str = ""
    decoder_ckpt: str = ""
    epoch: int = 30

    # data
    COCO_root: str = r"/mnt/qnap/data/datasets/coco2017/train/images"
    ego4d_root: str = r"/mnt/qnap/data/datasets/ego4d_hand_sep60"
    ih26m_root: str = r"/mnt/qnap/data/datasets/InterHand2.6M_5fps_batch1"
    img_size: int = 224
    expansion_ratio: float = 2.0

    # train
    ti_loss: bool = True
    ft_method: str = "full_param"
    secondary_loss: bool=True
    num_sample: int=4
    batch_size: int = 84
    lr: float = 1e-4
    lr_min: float = 1e-5
    optimizer: str = "adamw"
    lr_scheduler: str = "warmup"
    lora_rank: int = 4
    teacher_momentum_base: float = 1 - 1e-6

    # cosine anneling
    T_0: int = 10
    T_mult: int = 2

    # warmup
    warmup_epoch: int = 1
    cooldown_epoch: int = 10

    def __post_init__(self):
        if self.decoder_config == "":
            self.decoder_config = None
        if self.decoder_ckpt == "":
            self.decoder_ckpt = None

    # member functions
    def update(self, other: Union['PretrainConfig', Dict[str, Any]]):
        if isinstance(other, PretrainConfig):
            merge_dict = other.to_dict()
        elif isinstance(other, dict):
            merge_dict = other
        else:
            raise TypeError("can only merge from Config/dict")

        for key, value in merge_dict.items():
            if hasattr(self, key):
                # current_type = type(getattr(self, key))
                # if not isinstance(value, current_type):
                #     raise ValueError(f"Type mismatched: '{key}' expects {current_type.__name__}, "
                #         f"but received {type(value).__name__}.")
                setattr(self, key, value)
            else:
                raise KeyError(f"Unexpected key: {key}.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=False, indent=4)


@dataclass
class FinetuneConfig():
    # Experiments
    exp: str = field(default=None)
    epoch: int = 30

    # Model
    backbone_arch: str = field(default=None)
    backbone_ckpt: str = field(default=None)
    num_joints: int = 16
    num_spatial_layer: int = 6
    num_temporal_layer: int = 2
    img_size: int = 224
    expansion_ratio: float = 1.25
    trope_scalar: float = 20.0  # 20 ms -> 1 step
    num_latent_layer: int = 2

    # Dataset
    data: str = field(default=None)
    seq_len: int = field(default=None)
    batch_size: int = field(default=None)
    # source
    ih26mseq_root: str = "/mnt/qnap/data/datasets/InterHand2.6M_5fps_batch1"
    ho3d_root: str = "/mnt/qnap/data/datasets/ho3d_v3/ho3d_v3"

    # Train
    phase: str = "inference"
    temporal_supervision: str = "full"
    spatial_ckpt: str = field(default=None)
    lr: float = 1e-4
    lr_min: float = 1e-6
    lr_scheduler: str = field(default=None)
    warmup_epoch: int = 1
    cooldown_epoch: int = 10
    lora_backbone: int = -1

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


default_pretrain_cfg = PretrainConfig(
    exp="debug_train",
    data="COCO",
    model_dir="",
)

default_finetune_cfg = FinetuneConfig()