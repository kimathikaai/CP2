import argparse
import logging
import math
import os
import random
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path

import albumentations as A
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import wandb
# from mmcv.utils import Config
from mmengine.config import Config
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import builder
import loader
from datasets.pretrain_dataset import DatasetType, get_pretrain_dataset
from networks.segment_network import PretrainType

DEFAULT_QUEUE_SIZE = 65536


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Copy-Paste Contrastive Pretraining on ImageNet')

    parser.add_argument('--config', help='path to configuration file')
    parser.add_argument("--run_id", required=True, type=str, help='Unique identifier for a run')
    parser.add_argument("--tags", nargs='+', default=[], help='Tags to include for logging')

    parser.add_argument('--offline_wandb', action='store_true', help='Run wandb offline')

    parser.add_argument('--debug', action='store_true', help='Debug')

    parser.add_argument('--pretrain_from_scratch', action='store_true', help='Whether to initialize with ImageNet weights')

    # Logging
    parser.add_argument("--log_dir", type=str, required=True, help='Where to store logs')
    parser.add_argument("--wandb_project", type=str, default='ssl-pretraining', help='Wandb project name')
    parser.add_argument("--wandb_team", type=str, default='critical-ml-dg', help='Wandb team name')

    # Data
    parser.add_argument("--data_dirs", metavar='DIR', nargs='+', help='Folder(s) containing image data', required=True)
    parser.add_argument("--directory_type", type=str, choices=[x.name for x in DatasetType], default=DatasetType.FILENAME.name)

    # Method
    parser.add_argument("--backbone_type", type=str, choices=[x.name for x in builder.BackboneType], default=builder.BackboneType.DEEPLABV3.name)
    parser.add_argument("--pretrain_type", type=str, choices=[x.name for x in PretrainType], default=PretrainType.CP2.name)
    parser.add_argument("--mapping_type", type=str, choices=[x.name for x in builder.MappingType], default=builder.MappingType.CP2.name)
    parser.add_argument("--negative_type", type=str, choices=[x.name for x in builder.NegativeType], default=builder.NegativeType.NONE.name)
    parser.add_argument("--negative_scale", type=float, default=2)
    parser.add_argument('--num-workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')

    # Custom experimental hyper-parameters
    parser.add_argument('--lmbd_cp2_dense_loss', default=0.2, type=float)
    parser.add_argument('--lmbd_region_corr_weight', default=1, type=float)
    parser.add_argument('--lmbd_pixel_corr_weight', default=1, type=float)
    parser.add_argument('--lmbd_not_corr_weight', default=1, type=float)
    parser.add_argument('--pixel_ids_stride', default=1, type=int)
    parser.add_argument('--unet_truncated_dec_blocks', default=2, type=int)
    parser.add_argument('--same_foreground', action='store_true', help='Use the same foreground images for both bacgrounds')
    parser.add_argument('--cap_queue', action='store_true', help='Cap queue size to dataset size')
    parser.add_argument('--include_background', action='store_true', help='Include background aggregate pixels as negative pairs')

    parser.add_argument('--dense_logits_temp', default=1, type=float)
    parser.add_argument('--instance_logits_temp', default=0.2, type=float)

    parser.add_argument('--lemon_data', action='store_true', help='Running with lemon data')

    parser.add_argument('--img_height', default=224, type=int)
    parser.add_argument('--img_width', default=224, type=int)

    parser.add_argument('--foreground_min', default=0.5, type=float, help='Minimum size of foreground images')
    parser.add_argument('--foreground_max', default=0.8, type=float, help='Maximum size of foreground images')

    # Distributed training
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')

    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--max_steps', default=np.inf, type=int)
    parser.add_argument('--num-images', default=1281167, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='total batch size over all GPUs')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--remove_lr_scheduler', action='store_true', help='Stop using the lr scheduler')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum of SGD solver')
    parser.add_argument('--optim', default='sgd', help='optimizer')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--scalar-freq', default=100, type=int,
                        help='metrics writing frequency')
    parser.add_argument('--ckpt-freq', default=10, type=int,
                        help='checkpoint saving frequency')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')

    # fmt: on

    args = parser.parse_args()
    # convert to enum
    args.directory_type = DatasetType[args.directory_type]
    args.pretrain_type = PretrainType[args.pretrain_type]
    args.backbone_type = builder.BackboneType[args.backbone_type]
    args.mapping_type = builder.MappingType[args.mapping_type]
    args.negative_type = builder.NegativeType[args.negative_type]

    # lemon data
    if args.lemon_data:
        args.directory_type = DatasetType.CSV
        args.img_height = 512
        args.img_width = 512

    # if args.pretrain_type == PretrainType.PROPOSED:
    #     args.mapping_type = builder.MappingType.PIXEL_REGION_ID
    #     args.lmbd_not_corr_weight = 0
    #     args.lmbd_pixel_corr_weight = 10
    #     args.lmbd_region_corr_weight = 1
    #     args.pixel_ids_stride = 1

    return args


def cleanup():
    dist.destroy_process_group()


def setup(rank, args):
    dist.init_process_group(
        backend=args.dist_backend,
        init_method=args.dist_url,
        world_size=args.world_size,
        rank=rank,
    )


def prepare_data(rank, num_workers, args):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )

    # augmentation = loader.TwoCropsTransform(
    #     transforms.Compose(
    #         [
    #             transforms.RandomResizedCrop(
    #                 (args.img_height, args.img_width), scale=(0.2, 1.0)
    #             ),
    #             transforms.RandomApply(
    #                 [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
    #             ),
    #             transforms.RandomGrayscale(p=0.2),
    #             transforms.RandomApply([loader.GaussianBlur([0.1, 2.0])], p=0.5),
    #             transforms.RandomHorizontalFlip(),
    #             transforms.ToTensor(),
    #             # normalize,
    #         ]
    #     )
    # )

    # simply use RandomErasing for Copy-Paste implementation:
    # erase a random block of background image and replace the erased positions by foreground
    augmentation_bg = loader.BackgroundTransform(
        transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    (args.img_height, args.img_width), scale=(0.2, 1.0)
                ),
                transforms.RandomApply(
                    [transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8
                ),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([loader.GaussianBlur([0.1, 2.0])], p=0.5),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                # normalize,
                transforms.RandomErasing(
                    p=1.0,
                    scale=(args.foreground_min, args.foreground_max),
                    ratio=(0.8, 1.25),
                    value=0.0,
                ),
            ]
        )
    )

    augmentation = loader.A_TwoCropsTransform(
        A.Compose(
            [
                A.RandomResizedCrop(
                    (args.img_height, args.img_width), scale=(0.2, 1.0)
                ),
                A.ColorJitter(0.4, 0.4, 0.4, 0.1, p=0.8),
                A.ToGray(p=0.2),
                loader.AGaussianBlur(p=0.5),
                A.HorizontalFlip(),
            ],
            additional_targets={"region_ids": "mask"},
        ),
        mapping_type=args.mapping_type,
        pixel_ids_stride=args.pixel_ids_stride,
    )

    train_dataset = get_pretrain_dataset(
        image_directory_list=args.data_dirs,
        transform=augmentation,
        directory_type=args.directory_type,
        split_name="train",
    )
    train_dataset_bg = get_pretrain_dataset(
        image_directory_list=args.data_dirs,
        transform=augmentation_bg,
        directory_type=args.directory_type,
        split_name="train",
    )

    def get_dataloader(dataset, seed):
        sampler = DistributedSampler(
            dataset=train_dataset_bg,
            num_replicas=args.world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
            seed=seed,
        )
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size // args.world_size,
            shuffle=(sampler is None),
            num_workers=num_workers,
            pin_memory=False,
            persistent_workers=True,
            drop_last=True,
            sampler=sampler,
        )
        return dataloader, sampler

    train_loader, train_sampler = get_dataloader(train_dataset, 0)
    train_loader_bg0, train_sampler_bg0 = get_dataloader(train_dataset_bg, 1024)
    train_loader_bg1, train_sampler_bg1 = get_dataloader(train_dataset_bg, 2048)

    return (
        (train_loader, train_sampler),
        (train_loader_bg0, train_sampler_bg0),
        (train_loader_bg1, train_sampler_bg1),
    )


def setup_logger(rank, args):
    logger = logging.getLogger(__name__ + f"-rank{rank}")
    logger.setLevel(level=logging.INFO)
    formatter = logging.Formatter(
        "%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(funcName)s:%(lineno)d]-2s %(message)s"
    )

    # Rank 0 can output  to console
    if rank == 0:
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        stream_handler.setLevel(level=logging.INFO)
        logger.addHandler(stream_handler)

    # All logs should be sent to the files
    handler = logging.FileHandler(os.path.join(args.run_log_dir, f"log-rank{rank}.txt"))
    handler.setLevel(level=logging.INFO)
    handler.setFormatter(formatter)
    logger.addHandler(handler)

    return logger


def main_worker(rank, args):
    torch.cuda.set_device(rank)
    device = f"cuda:{rank}"

    # Seed everything
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    # If your model does not change and your input sizes remain the same
    # - then you may benefit from setting torch.backends.cudnn.benchmark = True.
    # However, if your model changes: for instance, if you have layers that
    # are only "activated" when certain conditions are met, or you have
    # layers inside a loop that can be iterated a different number of times,
    # then setting torch.backends.cudnn.benchmark = True might stall your execution.
    # Source: https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
    # cudnn.benchmark = True
    # Setting cudnn.deterministic as True will use the default algorithms, i.e.,
    # setting cudnn.benchmark as True will have no effect
    cudnn.deterministic = True
    cudnn.benchmark = False

    # get configuration file
    cfg = Config.fromfile(args.config)

    # initialize wandb for the main process
    if rank == 0:
        tags = ["pretrain"] + args.tags
        wandb.init(
            name=args.run_id,
            project=args.wandb_project,
            entity=args.wandb_team,
            dir=args.log_dir,
            tags=tags,
            mode="offline" if args.offline_wandb else "online",
        )
        # Add hyperparameters to config
        wandb.config.update({"hyper-parameters": vars(args)})
        wandb.config.update({"config_file": cfg})
        wandb.config.update(
            {"nvidia-smi": subprocess.check_output(["nvidia-smi"]).decode()}
        )
        # define our custom x axis metric
        # wandb.define_metric("step")
        # # define which metrics will be plotted against it (e.g. all metrics
        # # under 'train')
        # wandb.define_metric("train/*", step_metric="step")
        # wandb.define_metric("learning_rate", step_metric="step")

    # setup process groups
    setup(rank, args)
    # get logger
    logger = setup_logger(rank, args)

    #
    # Data
    #
    data_loaders = prepare_data(
        rank=rank,
        num_workers=args.num_workers_per_dataset,
        args=args,
    )
    (
        (train_loader, train_sampler),
        (train_loader_bg0, train_sampler_bg0),
        (train_loader_bg1, train_sampler_bg1),
    ) = data_loaders
    logger.info(f"Initialized data loaders ({rank = })")
    len_dataset = len(train_loader.dataset)
    logger.info(f"{len_dataset}")

    #
    # Model
    #
    # instantiate the model(it's your own model) and move it to the right device
    model = builder.MODEL(
        cfg,
        m=0.999
        if args.pretrain_type
        in [PretrainType.CP2, PretrainType.PROPOSED, PretrainType.DENSECL]
        else 0.996,
        K=min(len_dataset, DEFAULT_QUEUE_SIZE)
        if args.cap_queue
        else DEFAULT_QUEUE_SIZE,
        dim=128
        if args.pretrain_type
        in [PretrainType.CP2, PretrainType.PROPOSED, PretrainType.DENSECL]
        else 256,
        pretrain_from_scratch=args.pretrain_from_scratch,
        include_background=args.include_background,
        lmbd_cp2_dense_loss=args.lmbd_cp2_dense_loss,
        pretrain_type=args.pretrain_type,
        backbone_type=args.backbone_type,
        mapping_type=args.mapping_type,
        negative_type=args.negative_type,
        negative_scale=args.negative_scale,
        lmbd_pixel_corr_weight=args.lmbd_pixel_corr_weight,
        lmbd_region_corr_weight=args.lmbd_region_corr_weight,
        lmbd_not_corr_weight=args.lmbd_not_corr_weight,
        dense_logits_temp=args.dense_logits_temp,
        instance_logits_temp=args.instance_logits_temp,
        unet_truncated_dec_blocks=args.unet_truncated_dec_blocks,
        device=device,
        rank=rank,
    )
    model.to(device)
    logger.info(model)

    if rank == 0:
        wandb.config.update({"output_stride": model.output_stride})

    # import copy
    # # Initialize the model pretrained ImageNet weights
    # weights_before_q = copy.deepcopy(model.encoder_q.backbone)
    # model.encoder_q.backbone.init_weights()
    # weights_after_q = copy.deepcopy(model.encoder_q.backbone)
    # # print(f"{weights_before_q = }\n{weights_after_q = }")
    # weights_before_k = copy.deepcopy(model.encoder_k.backbone)
    # model.encoder_k.backbone.init_weights()
    # weights_after_k = copy.deepcopy(model.encoder_k.backbone)
    # # print(f"{weights_before_k = }\n{weights_after_k = }")
    # print(f"{torch.all(weights_before_q.layer4[2].bn3.bias == weights_after_q.layer4[2].bn3.bias) = }")

    # wrap the model with DDP
    # device_ids tell DDP where is your model
    # output_device tells DDP where to output, in our case, it is rank
    # find_unused_parameters=True instructs DDP to find unused output of the forward() function of any module in the model
    model = DistributedDataParallel(
        model,
        device_ids=[rank],
        output_device=rank,
        find_unused_parameters=True,
    )
    logger.info(f"Initialized the model ({rank = })")

    #
    # Optimizers
    #
    if args.optim == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), args.lr, weight_decay=0.01)
    elif args.optim == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    else:
        raise NotImplementedError("Only sgd and adamw optimizers are supported.")

    logger.info(f"Initialized optimizer ({rank = })")

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info("loading checkpoint '{}'".format(args.resume))
            dist.barrier()
            # Map model to be loaded to specified single gpu.
            loc = "cuda:{}".format(rank)
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            logger.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logger.info("=> no checkpoint found at '{}'".format(args.resume))

    step = 0
    for epoch in range(args.start_epoch, args.epochs):
        # When using DistributedSampler, we have to specify the epoch
        train_sampler.set_epoch(epoch)
        train_sampler_bg0.set_epoch(epoch)
        train_sampler_bg1.set_epoch(epoch)
        lr = (
            args.lr
            if args.remove_lr_scheduler
            else adjust_learning_rate(optimizer, epoch, args)
        )
        logger.info(f"Beginning {epoch = }")

        if rank == 0:
            wandb.log({"epoch": epoch, "update-step": step})
            wandb.log({"learning_rate": lr})

        # train for one epoch
        step = train(
            [train_loader, train_loader_bg0, train_loader_bg1],
            model,
            optimizer,
            epoch,
            args,
            step,
            logger,
            rank,
        )
        if epoch % args.ckpt_freq == args.ckpt_freq - 1 or step > args.max_steps:
            if rank == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        # 'arch': args.arch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "pretrain_type": args.pretrain_type.name,
                        "backbone_type": args.backbone_type.name,
                    },
                    epoch=epoch,
                    is_best=True,
                    filename=os.path.join(
                        args.log_dir,
                        args.run_id,
                        f"{step}_{epoch}_checkpoint.ckpt",
                    ),
                )
        if step > args.max_steps:
            break
    cleanup()


def train(train_loader_list, model, optimizer, epoch, args, step, logger, rank):
    train_loader, train_loader_bg0, train_loader_bg1 = train_loader_list

    batch_time = builder.AverageMeter("Time", ":6.3f")
    # data_time = AverageMeter('Data', ':6.3f')
    loss_log = builder.AverageMeter("Loss", ":.4f")
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, loss_log],
        logger,
        prefix="Epoch: [{}]".format(epoch),
    )

    model.train()
    end = time.time()

    for i, (images, bg0, bg1) in enumerate(
        zip(train_loader, train_loader_bg0, train_loader_bg1)
    ):
        if step > args.max_steps:
            return step

        if rank == 0:
            wandb.log({"update-step": step})

        # data_time.update(time.time() - end)
        idx_a = 0
        idx_b = idx_a if args.same_foreground else 1
        ids_a, ids_b = None, None

        # Sample A
        sample_a = images[idx_a]
        img_a, pixel_ids_a, region_ids_a = sample_a
        img_a, pixel_ids_a, region_ids_a = (
            img_a.to(model.device),
            pixel_ids_a.to(model.device),
            region_ids_a.to(model.device),
        )

        # Sample B
        sample_b = images[idx_b]
        img_b, pixel_ids_b, region_ids_b = sample_b
        img_b, pixel_ids_b, region_ids_b = (
            img_b.to(model.device),
            pixel_ids_b.to(model.device),
            region_ids_b.to(model.device),
        )

        # validate ids and images
        bg0 = bg0.to(model.device)
        bg1 = bg1.to(model.device)

        visualize = epoch == 0 and i == 0
        new_epoch = i == 0
        if args.pretrain_type in [PretrainType.CP2, PretrainType.PROPOSED]:
            loss = model(
                img_a=img_a,
                img_b=img_b,
                pixel_ids_a=pixel_ids_a,
                pixel_ids_b=pixel_ids_b,
                region_ids_a=region_ids_a,
                region_ids_b=region_ids_b,
                bg0=bg0,
                bg1=bg1,
                visualize=visualize,
                step=step,
                new_epoch=new_epoch,
            )
        else:
            loss = model(
                img_a=img_a,
                img_b=img_b,
                bg0=bg0,
                bg1=bg1,
                visualize=visualize,
                step=step,
                new_epoch=new_epoch,
            )
        # logger.info(f"{epoch = }, {step = }, {loss.item() = }")

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # logging
        loss_log.update(loss.item(), img_a.size(0))
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

        step += 1

    model.module.on_train_epoch_end(step)

    return step


def save_checkpoint(
    state,
    is_best,
    epoch,
    filename="checkpoint.ckpt",
):
    torch.save(state, filename)
    if is_best:
        copy_file = os.path.join(Path(filename).parent, "checkpoint.ckpt")
        shutil.copyfile(filename, copy_file)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, logger, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix
        self.logger = logger

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        self.logger.info("    ".join(entries))
        # if torch.distributed.get_rank() == 0:
        #     self.logger.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    return lr


if __name__ == "__main__":
    # parse command line
    args = get_args()
    print(f"Args: f{vars(args) = }")

    # create logging dir
    args.run_log_dir = os.path.join(args.log_dir, args.run_id)
    os.mkdir(args.run_log_dir)
    print(f"{args.run_log_dir = }")

    print(f"{torch.cuda.device_count() = }")
    # assert args.world_size <= torch.cuda.device_count()

    # get the number of workers
    args.num_workers_per_dataset = args.num_workers // (args.world_size * 3)
    print(f"{args.num_workers_per_dataset = }")

    # Seed everything
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    if args.debug:
        args.batch_size = 8
        args.world_size = 1
        args.num_workers_per_dataset = args.num_workers // (args.world_size * 3)
        print(f"{args.num_workers_per_dataset = }")
        main_worker(0, args)
    else:
        # spawn parallel process
        mp.spawn(main_worker, args=[args], nprocs=args.world_size)
