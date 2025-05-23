import argparse
from typing import *
import os
from copy import deepcopy
from tqdm import tqdm
from datetime import datetime
import json

import h5py
import math
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.utils import make_grid
import numpy as np

from cs_vit.net import Poser, warmup_scheduler
from cs_vit.dataset import InterHand26MSeq, HO3D, DexYCB
from cs_vit.config import *
from cs_vit.utils.misc import move_to_device, flatten_dict, wrap_prefix_print, print_grouped_losses


def nop(*a, **k):
    _, _ = a, k


def ddp_setup():
    torch.cuda.set_device(int(os.environ["LOCAL_RANK"]))
    torch.distributed.init_process_group(
        backend="nccl",
        rank=int(os.environ["LOCAL_RANK"]),
        world_size=int(os.environ["WORLD_SIZE"]),
    )


def tensor_item(x: Union[torch.Tensor, float]):
    if isinstance(x, torch.Tensor):
        return x.item()
    else:
        return x


def get_rank() -> int:
    return int(os.environ["LOCAL_RANK"])


def get_world_size() -> int:
    return int(os.environ["WORLD_SIZE"])


def gather_strings(img_path: list, rank: int, world_size: int):
    max_len = max(len(s.encode('utf-8')) for s in img_path) if img_path else 0
    tensor = torch.zeros(len(img_path), max_len, dtype=torch.uint8, device=f"cuda:{get_rank()}")
    for i, s in enumerate(img_path):
        encoded = s.encode('utf-8')
        tensor[i, :len(encoded)] = torch.tensor(list(encoded), dtype=torch.uint8)

    gathered_tensors = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered_tensors, tensor)

    if rank == 0:
        all_paths = []
        for t in gathered_tensors:
            paths = []
            for row in t:
                bytes_data = row[row != 0].tolist()
                paths.append(bytes(bytes_data).decode('utf-8'))
            all_paths.extend(paths)
        return all_paths
    return None


def gather_tensors(tensor: torch.Tensor, rank: int, world_size: int):
    if rank == 0:
        gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
        dist.gather(tensor, gathered, dst=0)
        return torch.cat(gathered, dim=0)
    else:
        dist.gather(tensor, dst=0)
        return None


def setup(rank: int, cfg: FinetuneConfig, print_: Callable = print):
    # 0. basic setup
    device = torch.device(f"cuda:{rank}")
    start_epoch = 1
    end_epoch = cfg.epoch

    summary_writer = None

    # 1. init dataset
    if cfg.data == "interhand26m":
        dataset = InterHand26MSeq(
            root=cfg.ih26mseq_root,
            num_frames=1 if cfg.phase == "spatial" else cfg.seq_len,
            data_split="test",
            img_size=cfg.img_size,
            expansion_ratio=cfg.expansion_ratio,
        )
        collate_fn = InterHand26MSeq.collate_fn
        shuffle = False
    elif cfg.data == "evaluation":
        dataset = HO3D(
            root=cfg.ho3d_root,
            num_frames=1 if cfg.phase == "spatial" else cfg.seq_len,
            data_split="train",
            img_size=cfg.img_size,
            expansion_ratio=cfg.expansion_ratio
        )
        collate_fn = InterHand26MSeq.collate_fn
        shuffle = False
    elif cfg.data == "dexycb":
        dataset = DexYCB(
            root=cfg.dexycb_root,
            num_frames=1 if cfg.phase == "spatial" else cfg.seq_len,
            protocol="s1",
            data_split="test",
            img_size=cfg.img_size,
            expansion_ratio=cfg.expansion_ratio
        )
        collate_fn = InterHand26MSeq.collate_fn
        shuffle=True
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        pin_memory=False,
        drop_last=False,
        num_workers=4,
        sampler=DistributedSampler(dataset, shuffle=shuffle, drop_last=False),
        collate_fn=collate_fn
    )

    # 2. setup model
    model: Poser = Poser(
        backbone=cfg.backbone,
        num_pose_query=cfg.num_joints,
        num_spatial_layer=cfg.num_spatial_layer,
        spatial_layer_type=cfg.spatial_layer_type,
        num_temporal_layer=cfg.num_temporal_layer,
        expansion_ratio=cfg.expansion_ratio,
        temporal_supervision=cfg.temporal_supervision,
        trope_scalar=cfg.trope_scalar,
        num_latent_layer=cfg.num_latent_layer,
        persp_decorate=cfg.persp_decorate,
    )
    model.phase(Poser.TrainingPhase(cfg.phase))
    model.load_state_dict(torch.load(cfg.eval_ckpt, map_location="cpu")["merged"], strict=False)
    model.to(rank)
    model.eval()
    model = DistributedDataParallel(
        model, device_ids=[rank], output_device=rank, find_unused_parameters=False
    )

    # 3. optimizer
    max_lr = math.sqrt(get_world_size() * cfg.batch_size / 44) * cfg.lr
    min_lr = math.sqrt(get_world_size() * cfg.batch_size / 44) * cfg.lr_min
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max_lr
    )

    # 4. scheduler
    scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda _: 1.0
    )
    in_epoch_scheduler: bool = False
    if cfg.lr_scheduler == "warmup":
        scheduler = warmup_scheduler(
            optimizer=optimizer,
            max_lr=max_lr,
            min_lr=min_lr,
            warmup_epochs=cfg.warmup_epoch,
            annealing_epochs=cfg.cooldown_epoch,
            steps_per_epoch=len(dataloader),
        )
        in_epoch_scheduler = True
    elif cfg.lr_scheduler == "constant":
        scheduler = scheduler
        in_epoch_scheduler = None

    return (
        start_epoch,
        end_epoch,
        summary_writer,
        dataloader,
        optimizer,
        scheduler,
        in_epoch_scheduler,
        model,
    )


def main(rank: int, cfg: FinetuneConfig, print_: Callable = print):
    assert (
        cfg.phase == "temporal" and cfg.temporal_supervision == "realtime"
        or cfg.phase == "spatial"
    )

    world_size = get_world_size()
    if rank == 0:
        date_str = datetime.now().strftime("%Y%m%d")
        h5_path = os.path.join(
            f"checkpoints/{cfg.exp}/"
            f"eval_{cfg.data}_{cfg.phase}_{cfg.temporal_supervision}_{date_str}.h5"
        )
        h5file = h5py.File(h5_path, 'w')
        str_dtype = h5py.special_dtype(vlen=str)
        h5file.create_dataset(
            "img_paths",
            shape=(0,),
            maxshape=(None,),
            dtype=str_dtype
        )
        h5file.create_dataset(
            "joint_cam_gt",
            shape=(0, 21, 3),
            maxshape=(None, 21, 3),
            dtype='float32',
            chunks=(1000, 21, 3),
            compression='gzip'
        )
        h5file.create_dataset(
            "joint_cam_pred",
            shape=(0, 21, 3),
            maxshape=(None, 21, 3),
            dtype='float32',
            chunks=(1000, 21, 3),
            compression='gzip'
        )
        h5file.create_dataset(
            "joint_reproj_pred",
            shape=(0, 21, 2),
            maxshape=(None, 21, 2),
            dtype='float32',
            chunks=(1000, 21, 2),
            compression='gzip'
        )
        h5file.create_dataset(
            "joint_reproj_gt",
            shape=(0, 21, 2),
            maxshape=(None, 21, 2),
            dtype='float32',
            chunks=(1000, 21, 2),
            compression='gzip'
        )

    # 1. setup
    (_, _, _, dataloader, _, _, _, model) = setup(rank, cfg, print_)

    # 2. eval
    print_("evaluation starts")
    device = torch.device(f"cuda:{rank}")
    for _, batch_ in enumerate(tqdm(dataloader, ncols=100)):
        batch = move_to_device(deepcopy(batch_), device)
        with torch.inference_mode():
            predict = model.module.predict_batch(
                img_tensor=batch["patches"],
                square_bboxes=batch["square_bboxes"],
                timestamp=batch["timestamp"],
                focal=batch["focal"],
                princpt=batch["princpt"],
            )

        img_path = [x[-1] for x in batch["imgs_path"]]
        joint_cam_gt = batch["joint_cam"][:, -1]
        joint_cam_pred = predict["joint_cam"][:, -1]

        # reproj
        joint_reproj_pred_u = (
            batch["focal"][..., :1] * predict["joint_cam"][..., 0] +
            batch["princpt"][..., :1] * predict["joint_cam"][..., 2]
        )
        joint_reproj_pred_v = (
            batch["focal"][..., 1:] * predict["joint_cam"][..., 1] +
            batch["princpt"][..., 1:] * predict["joint_cam"][..., 2]
        )
        # [B,T,J=21,2]
        joint_reproj_pred = torch.stack([joint_reproj_pred_u, joint_reproj_pred_v], dim=-1)
        joint_reproj_pred = joint_reproj_pred / predict["joint_cam"][..., -1:]
        joint_reproj_gt = batch["joint_img"]
        joint_reproj_pred = joint_reproj_pred[:, -1]
        joint_reproj_gt = joint_reproj_gt[:, -1]

        all_img_paths = gather_strings(img_path, rank, world_size)
        all_joint_cam_gt = gather_tensors(joint_cam_gt, rank, world_size)
        all_joint_cam_pred = gather_tensors(joint_cam_pred, rank, world_size)
        all_joint_reproj_gt = gather_tensors(joint_reproj_gt, rank, world_size)
        all_joint_reproj_pred = gather_tensors(joint_reproj_pred, rank, world_size)

        if rank == 0:
            img_paths_np = np.array(all_img_paths, dtype=object)
            gt_np = all_joint_cam_gt.detach().cpu().numpy().astype(np.float32)
            pred_np = all_joint_cam_pred.detach().cpu().numpy().astype(np.float32)
            reproj_gt_np = all_joint_reproj_gt.detach().cpu().numpy().astype(np.float32)
            reproj_pred_np = all_joint_reproj_pred.detach().cpu().numpy().astype(np.float32)

            current_size = h5file['img_paths'].shape[0]
            new_size = current_size + len(img_paths_np)

            h5file['img_paths'].resize((new_size,))
            h5file['joint_cam_gt'].resize((new_size, 21, 3))
            h5file['joint_cam_pred'].resize((new_size, 21, 3))
            h5file['joint_reproj_gt'].resize((new_size, 21, 2))
            h5file['joint_reproj_pred'].resize((new_size, 21, 2))

            h5file['img_paths'][current_size:new_size] = img_paths_np
            h5file['joint_cam_gt'][current_size:new_size, :] = gt_np
            h5file['joint_cam_pred'][current_size:new_size, :] = pred_np
            h5file['joint_reproj_gt'][current_size:new_size, :] = reproj_gt_np
            h5file['joint_reproj_pred'][current_size:new_size, :] = reproj_pred_np
        torch.distributed.barrier()

    torch.distributed.barrier()
    if rank == 0:
        h5file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="eval in gpu")
    parser.add_argument("--exp", type=str, required=True, help="Exp name")
    parser.add_argument("--data", type=str, required=True, help="Dataset",
        choices=["interhand26m", "ho3d", "dexycb"]
    )
    parser.add_argument("--seq_len", type=int, required=False, default=1, help="Sequence length")
    parser.add_argument("--batch_size", type=int, required=False, default=16)
    parser.add_argument("--eval_ckpt", type=str, required=True, help="Checkpoint to eval")

    args = parser.parse_args()
    exp_name: str = args.exp

    # load from config
    assert os.path.exists(f"./checkpoints/{exp_name}/config.json")
    with open(f"./checkpoints/{exp_name}/config.json", "r") as f:
        json_obj = json.loads(f.read())
    cfg = FinetuneConfig(**json_obj)
    cfg.update(vars(args))

    ddp_setup()
    torch.manual_seed(42)
    np.random.seed(42)
    rank = get_rank()

    print_ = wrap_prefix_print(prefix=f"[{rank}] ") if rank == 0 else nop

    cfg.update(vars(args))
    print_("Config loaded from command")

    main(rank, cfg, print_=print_)
    torch.distributed.destroy_process_group()
