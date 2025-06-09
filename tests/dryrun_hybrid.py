from tqdm import tqdm
from copy import deepcopy
import os
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data.dataloader import DataLoader

from cs_vit.dataset import DexYCB, InterHand26MSeq, HO3D
from cs_vit.config import default_finetune_cfg as cfg


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(
        backend="nccl",
        rank=int(os.environ["LOCAL_RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )


# ddp_setup()

dataset_ih26m = InterHand26MSeq(
    root=cfg.ih26mseq_root,
    num_frames=7,
    data_split="train",
    img_size=cfg.img_size,
    expansion_ratio=cfg.expansion_ratio,
)
dataset_dex = DexYCB(
    root=cfg.dexycb_root,
    num_frames=7,
    protocol="s1",
    data_split="train",
    img_size=cfg.img_size,
    expansion_ratio=cfg.expansion_ratio,
)
dataloader_ho3d = HO3D(
    root=cfg.ho3d_root,
    num_frames=7,
    data_split="train",
    img_size=cfg.img_size,
    expansion_ratio=cfg.expansion_ratio,
)
dataset = ConcatDataset([dataset_ih26m, dataset_dex, dataloader_ho3d])

dataloader = DataLoader(
    dataset,
    batch_size=16,
    pin_memory=False,
    drop_last=False,
    num_workers=32,
    collate_fn=InterHand26MSeq.collate_fn,
    # sampler=DistributedSampler(dataset, shuffle=True, drop_last=False)
)

for batch in tqdm(dataloader, ncols=50):
    print(list(batch.keys()))

# torch.distributed.destroy_process_group()
