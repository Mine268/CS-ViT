import argparse
import datetime
from typing import *
import os
import json
from copy import deepcopy

import random
import math
import torch
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import ConcatDataset
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid
import numpy as np

from cs_vit.net import Poser, warmup_scheduler
from cs_vit.dataset import InterHand26MSeq, HO3D, DexYCB
from cs_vit.config import *
from cs_vit.utils.misc import move_to_device, flatten_dict, wrap_prefix_print, print_grouped_losses
from cs_vit.utils.tensor import calculate_gradient_norm

os.environ['CUDA_LAUNCH_BLOCKING'] = "1"


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


def setup(rank: int, cfg: FinetuneConfig, print_: Callable = print):
    # 0. basic setup
    device = torch.device(f"cuda:{rank}")
    start_epoch = 1
    end_epoch = cfg.epoch

    summary_writer: Optional[SummaryWriter] = None
    if rank == 0:
        summary_writer = SummaryWriter(log_dir=f"./checkpoints/{cfg.exp}/tb_logs")

    # 1. init dataset
    dataset_list = []
    if "interhand26m" in cfg.data:
        dataset_list.append(
            InterHand26MSeq(
                root=cfg.ih26mseq_root,
                num_frames=1 if cfg.phase == "spatial" else cfg.seq_len,
                data_split="train",
                img_size=cfg.img_size,
                expansion_ratio=cfg.expansion_ratio,
            )
        )
        print_("Added interhand26m")
    if "ho3d" in cfg.data:
        dataset_list.append(
            HO3D(
                root=cfg.ho3d_root,
                num_frames=1 if cfg.phase == "spatial" else cfg.seq_len,
                data_split="train",
                img_size=cfg.img_size,
                expansion_ratio=cfg.expansion_ratio,
            )
        )
        print_("Added ho3d")
    if "dexycb" in cfg.data:
        dataset_list.append(
            DexYCB(
                root=cfg.dexycb_root,
                num_frames=1 if cfg.phase == "spatial" else cfg.seq_len,
                protocol="s1",
                data_split="train",
                img_size=cfg.img_size,
                expansion_ratio=cfg.expansion_ratio,
            )
        )
        print_("Added dexycb")
    dataset = ConcatDataset(datasets=dataset_list)
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=cfg.batch_size,
        pin_memory=False,
        drop_last=False,
        num_workers=24,
        sampler=DistributedSampler(dataset, shuffle=True, drop_last=False),
        collate_fn=InterHand26MSeq.collate_fn
    )

    # 2. setup model
    model = Poser(
        backbone=cfg.backbone,
        num_pose_query=cfg.num_joints,
        num_spatial_layer=cfg.num_spatial_layer,
        spatial_layer_type=cfg.spatial_layer_type,
        num_temporal_layer=cfg.num_temporal_layer,
        temporal_init_method=cfg.temporal_init_method,
        expansion_ratio=cfg.expansion_ratio,
        temporal_supervision=cfg.temporal_supervision,
        trope_scalar=cfg.trope_scalar,
        num_latent_layer=cfg.num_latent_layer,
        persp_embed_method=cfg.persp_embed_method,
        persp_decorate=cfg.persp_decorate,
        image_size=cfg.img_size,
    )
    model.phase(Poser.TrainingPhase(cfg.phase))
    if (cfg.phase == "temporal"):
        model.load_state_dict(torch.load(cfg.spatial_ckpt)["merged"], strict=False)
    model.to(rank)
    model = DistributedDataParallel(
        model, device_ids=[rank], output_device=rank, find_unused_parameters=True
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

    # 5. restore from checkpoint if exists
    if os.path.exists(f"./checkpoints/{cfg.exp}/checkpoint.pt"):
        print_(
            f"found checkpoints,"
            f" trying reading from ./checkpoints/{cfg.exp}/checkpoint.pt"
        )
        ckpt: Dict[str, Any] = torch.load(
            f"./checkpoints/{cfg.exp}/checkpoint.pt",
            map_location=device,
            weights_only=False,
        )

        start_epoch = ckpt["epoch"] + 1
        model.module.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        scheduler.load_state_dict(ckpt["scheduler"])

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


def train_one_epoch(
    rank: int,
    cfg: FinetuneConfig,
    epoch: int,
    model: DistributedDataParallel,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    scheduler: Tuple[bool, torch.optim.lr_scheduler.LRScheduler],
    log_every: int = 20,
    print_: Callable = None,
    summary_writer: SummaryWriter = None,
):
    device = torch.device(f"cuda:{rank}")
    start_log_time, end_log_time = datetime.datetime.now(), None
    in_step_scheduler, scheduler_ = scheduler

    for it, batch_ in enumerate(dataloader):
        # move to device
        batch = move_to_device(deepcopy(batch_), device)
        del batch_
        # batch = move_to_device(batch_, device)

        # forward
        forward_result = model(batch)

        # backward
        loss = forward_result["loss"]
        optimizer.zero_grad(set_to_none=True)
        if torch.isnan(loss):
            print_("loss is nan, skipping this batch")
            continue
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        # lr update
        if in_step_scheduler is not None and in_step_scheduler:
            scheduler_.step()

        # log
        if log_every is not None and (it + 1) % log_every == 0:
            global_step = epoch * len(dataloader) + it + 1
            if summary_writer is not None:
                # plot scalars to tb
                for group, value in flatten_dict(forward_result["logs"]["scalar"]):
                    summary_writer.add_scalar(
                        f"train/{group}",
                        value,
                        global_step=global_step,
                    )
                # plot imgs
                T = forward_result["logs"]["image"]["img_reproj"].shape[0]
                img_grid = make_grid(
                    forward_result["logs"]["image"]["img_reproj"],
                    nrow=T
                )
                summary_writer.add_image(
                    f"train/img_reproj",
                    img_grid,
                    global_step=global_step,
                )
                del img_grid
                # plot lr
                summary_writer.add_scalar(
                    "train/lr",
                    optimizer.param_groups[0]["lr"],
                    global_step=global_step,
                )
                # plot grad norm
                grad_norm = calculate_gradient_norm(model.module)
                summary_writer.add_scalar(
                    "train/grad",
                    grad_norm,
                    global_step=global_step,
                )

            # output to terminal
            end_log_time = datetime.datetime.now()
            iter_time = (end_log_time - start_log_time) / log_every
            print_grouped_losses(
                epoch=epoch,
                iteration=it,
                total_iters=len(dataloader),
                iter_time=iter_time,
                lr=scheduler_.get_last_lr()[0],
                forward_result=forward_result,
                print_=print_
            )
            start_log_time = datetime.datetime.now()

        del batch, forward_result

    # lr update
    if in_step_scheduler is not None and not in_step_scheduler:
        scheduler_.step()


def main(rank: int, cfg: FinetuneConfig, print_: Callable = print):
    # 1. setup
    (
        start_epoch,
        end_epoch,
        summary_writer,
        dataloader,
        optimizer,
        scheduler,
        in_step_scheduler,
        model
    ) = setup(rank, cfg, print_)

    # 2. train
    for epoch in range(start_epoch, end_epoch + 1):
        start_time = datetime.datetime.now()
        print_(
            f"training for epoch {epoch}/{end_epoch},"
            f" start time {start_time.strftime('%Y-%m-%d_%H:%M:%S')}."
        )

        dataloader.sampler.set_epoch(epoch)
        train_one_epoch(
            rank=rank,
            cfg=cfg,
            epoch=epoch,
            model=model,
            dataloader=dataloader,
            optimizer=optimizer,
            scheduler=(in_step_scheduler, scheduler),
            print_=print_,
            summary_writer=summary_writer,
        )

        end_time = datetime.datetime.now()
        print_(
            f"epoch {epoch} ends at {end_time.strftime('%Y-%m-%d_%H:%M:%S')},"
            f" time costs {end_time - start_time}."
        )

        # dump checkpoints to file
        if rank == 0 and (epoch % 1 == 0 or epoch == end_epoch):
            print_(f"writing checkpoint for epoch {epoch}.")
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.module.state_dict(),
                    "merged": model.module.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                },
                f"./checkpoints/{cfg.exp}/checkpoint_{epoch}.pt",
            )
            if os.path.exists(f"./checkpoints/{cfg.exp}/checkpoint.pt"):
                os.remove(f"./checkpoints/{cfg.exp}/checkpoint.pt")
            os.symlink(
                f"./checkpoint_{epoch}.pt", f"./checkpoints/{cfg.exp}/checkpoint.pt"
            )
        torch.distributed.barrier()

        print_()  # print a blank line


if __name__ == "__main__":
    parser = argparse.ArgumentParser(prog="fit in gpu")
    parser.add_argument("--exp", type=str, required=True, help="Exp name")
    parser.add_argument("--epoch", type=int, required=False, default=30)
    parser.add_argument("--phase", type=str, required=True, help="Training phase",
        choices=["spatial", "temporal", "inference"]
    )
    parser.add_argument("--spatial_ckpt", type=str, required=False, default=None,
        help="Spatial checkpoint path"
    )
    parser.add_argument("--temporal_supervision", type=str, required=True,
        help="How the temporal outputs are supervised",
        choices=["full", "realtime"]
    )
    parser.add_argument("--backbone", type=str, required=True,
        help="Backbone path (huggingface checkpoint)"
    )
    parser.add_argument("--num_latent_layer", type=int, required=False, default=None,
        help="if None, no latent constraints applied"
    )
    parser.add_argument("--spatial_layer_type", type=str, required=False, default="decoder",
        help="Type of spatial encoder layer",
        choices=["decoder", "encoder"]
    )
    parser.add_argument("--temporal_init_method", type=str, required=False, default="zero",
        help="Initialization method for temporal layers",
        choices=["zero", "random"]
    )
    parser.add_argument("--persp_embed_method", type=str, required=False, default="dense",
        help=
            "How the model produce PEE embeding vector. "
            "dense: using perspective vector map. "
            "sparse: using normalized coordiantes of four corners of bounding box",
        choices=["dense", "sparse"]
    )
    parser.add_argument("--persp_decorate", type=str, required=False, default="query",
        help="Perpective embedding decoration approach",
        choices=["query", "patch"]
    )
    parser.add_argument("--data", type=str, required=True, help="Dataset", nargs="+",
        choices=["interhand26m", "ho3d", "dexycb"]
    )
    parser.add_argument("--seq_len", type=int, required=False, default=7, help="Sequence length")
    parser.add_argument("--batch_size", type=int, required=False, default=16)
    parser.add_argument("--lr", type=float, required=False, default=1e-4)
    parser.add_argument("--lr_min", type=float, required=False, default=1e-6)
    parser.add_argument("--lr_scheduler", type=str, required=False, default="warmup",
        help="Learning rate scheduler",
        choices=["warmup", "constant"]
    )

    args = parser.parse_args()
    exp_name: str = args.exp

    cfg = deepcopy(default_finetune_cfg)

    ddp_setup()
    random.seed(42)
    torch.manual_seed(42)
    np.random.seed(42)
    rank = get_rank()

    print_ = wrap_prefix_print(prefix=f"[{rank}] ") if rank == 0 else nop

    # read from existing json if exists, otherwise udpate default with input args
    if os.path.exists(f"./checkpoints/{exp_name}/config.json"):
        with open(f"./checkpoints/{exp_name}/config.json", "r") as f:
            json_obj = json.loads(f.read())
        cfg = FinetuneConfig(**json_obj)
        # train new epoch if bigger
        cfg.epoch = args.epoch
        print_("Config loaded from file")
    else:
        cfg.update(vars(args))
        # save experiement setup to file
        if rank == 0:
            os.makedirs(f"./checkpoints/{exp_name}", exist_ok=True)
            with open(f"./checkpoints/{exp_name}/config.json", "w") as f:
                f.write(cfg.to_json())
        print_("Config loaded from command")

    main(rank, cfg, print_=print_)
    torch.distributed.destroy_process_group()
