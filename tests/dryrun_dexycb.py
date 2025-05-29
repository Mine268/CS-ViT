from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader

from cs_vit.dataset import DexYCB, InterHand26MSeq
from cs_vit.config import default_finetune_cfg as cfg


dataset = DexYCB(
    root=cfg.dexycb_root,
    num_frames=7,
    protocol="s1",
    data_split="train",
    img_size=cfg.img_size,
    expansion_ratio=cfg.expansion_ratio
)
dataloader = DataLoader(
    dataset,
    batch_size=10,
    pin_memory=False,
    drop_last=False,
    shuffle=False,
    num_workers=8,
    collate_fn=InterHand26MSeq.collate_fn
)


for batch in tqdm(dataloader, ncols=50):
    pass