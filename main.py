import argparse
import builtins
import logging
import math
import os
import random
import shutil
import time
from datetime import datetime

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
# from mmcv.utils import Config
from mmengine.config import Config
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

import builder
import loader
from datasets.pretrain_dataset import get_pretrain_dataset


def get_args():
    # fmt: off
    parser = argparse.ArgumentParser(description='Copy-Paste Contrastive Pretraining on ImageNet')

    parser.add_argument('--config', help='path to configuration file')
    parser.add_argument("--run_id", type=str, required=True, help='Unique identifier for a run')
    parser.add_argument("--log_dir", type=str, required=True, help='Where to store logs')

    # Data
    parser.add_argument("--data_dirs", metavar='DIR', nargs='+', help='Folder(s) containing image data')
    parser.add_argument("--train_csv_paths", nargs="+", help="CSVs with training data paths")
    parser.add_argument("--val_csv_path", nargs="+", help="CSVs with validation data paths")
    parser.add_argument("--test_csv_path", nargs="+", help="CSVs with test data paths")
    parser.add_argument('--num-workers', default=32, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')

    # Distributed training
    parser.add_argument('--dist-url', default='tcp://localhost:10001', type=str,
                        help='url used to set up distributed training')
    parser.add_argument('--dist-backend', default='nccl', type=str,
                        help='distributed backend')
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of nodes for distributed training')

    parser.add_argument('--epochs', default=200, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--num-images', default=1281167, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='manual epoch number (useful on restarts)')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
                        metavar='N', help='total batch size over all GPUs')
    parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
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
    parser.add_argument('--ckpt-freq', default=1, type=int,
                        help='checkpoint saving frequency')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to latest checkpoint (default: none)')
    parser.add_argument('--seed', default=0, type=int,
                        help='seed for initializing training. ')

    parser.add_argument('--output-stride', default=16, type=int,
                        help='output stride of encoder')
    # fmt: on

    args = parser.parse_args()

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
    augmentation = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([loader.GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,
    ]

    # simply use RandomErasing for Copy-Paste implementation:
    # erase a random block of background image and replace the erased positions by foreground
    augmentation_bg = [
        transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
        transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomApply([loader.GaussianBlur([0.1, 2.0])], p=0.5),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # normalize,
        transforms.RandomErasing(p=1.0, scale=(0.5, 0.8), ratio=(0.8, 1.25), value=0.0),
    ]

    train_dataset = get_pretrain_dataset(
        image_directory_list=args.data_dirs,
        image_csv_list=args.train_csv_paths,
        transform=loader.TwoCropsTransform(transforms.Compose(augmentation)),
    )
    train_dataset_bg = get_pretrain_dataset(
        image_directory_list=args.data_dirs,
        image_csv_list=args.train_csv_paths,
        transform=transforms.Compose(augmentation_bg),
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


def main_worker(rank, args):
    torch.cuda.set_device(rank)

    device = f"cuda:{rank}"

    # configure logger
    logging.basicConfig(
        format=f"[%(levelname)][{device}][%(filename)s] %(message)s",
        datefmt="%Y-%m-%d:%H:%M:%S",
        level=logging.INFO,
    )


    # setup process groups
    setup(rank, args)

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
    logging.info(f"Initialized data loaders ({rank = })")

    #
    # Model
    #
    # instantiate the model(it's your own model) and move it to the right device
    cfg = Config.fromfile(args.config)
    model = builder.CP2_MOCO(cfg)
    model.cuda(rank)
    if rank == 0:
        logging.info(model)

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
    logging.info(f"Initialized the model ({rank = })")

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

    logging.info(f"Initialized optimizer ({rank = })")

    criterion = nn.CrossEntropyLoss().cuda(rank)

    # If your model does not change and your input sizes remain the same
    # - then you may benefit from setting torch.backends.cudnn.benchmark = True.
    # However, if your model changes: for instance, if you have layers that
    # are only "activated" when certain conditions are met, or you have
    # layers inside a loop that can be iterated a different number of times,
    # then setting torch.backends.cudnn.benchmark = True might stall your execution.
    # Source: https://stackoverflow.com/questions/58961768/set-torch-backends-cudnn-benchmark-true-or-not
    cudnn.benchmark = True

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            logging.info("loading checkpoint '{}'".format(args.resume))
            dist.barrier()
            # Map model to be loaded to specified single gpu.
            loc = "cuda:{}".format(rank)
            checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint["epoch"]
            model.load_state_dict(checkpoint["state_dict"])
            optimizer.load_state_dict(checkpoint["optimizer"])
            logging.info(
                "=> loaded checkpoint '{}' (epoch {})".format(
                    args.resume, checkpoint["epoch"]
                )
            )
        else:
            logging.info("=> no checkpoint found at '{}'".format(args.resume))

    for epoch in range(args.start_epoch, args.epochs):
        # When using DistributedSampler, we have to specify the epoch
        train_sampler.set_epoch(epoch)
        train_sampler_bg0.set_epoch(epoch)
        train_sampler_bg1.set_epoch(epoch)
        adjust_learning_rate(optimizer, epoch, args)

        # train for one epoch
        train(
            [train_loader, train_loader_bg0, train_loader_bg1],
            model,
            criterion,
            optimizer,
            epoch,
            args,
            device,
        )
        if epoch % args.ckpt_freq == args.ckpt_freq - 1:
            if rank == 0:
                save_checkpoint(
                    {
                        "epoch": epoch + 1,
                        # 'arch': args.arch,
                        "state_dict": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                    },
                    is_best=False,
                    filename=os.path.join(
                        args.log_dir,
                        args.run_id,
                        "checkpoint_{:04d}.pth.tar".format(epoch),
                    ),
                )
    cleanup()


def train(train_loader_list, model, criterion, optimizer, epoch, args, device):
    batch_time = AverageMeter("Time", ":6.3f")
    # data_time = AverageMeter('Data', ':6.3f')
    loss_i = AverageMeter("Loss_ins", ":.4f")
    loss_d = AverageMeter("Loss_den", ":.4f")
    acc_ins = AverageMeter("Acc_ins", ":6.2f")
    acc_seg = AverageMeter("Acc_seg", ":6.2f")
    train_loader, train_loader_bg0, train_loader_bg1 = train_loader_list
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, loss_i, loss_d, acc_ins, acc_seg],
        prefix="Epoch: [{}]".format(epoch),
    )

    # cre_dense = nn.LogSoftmax(dim=1)

    model.train()

    end = time.time()
    for i, (images, bg0, bg1) in enumerate(
        zip(train_loader, train_loader_bg0, train_loader_bg1)
    ):
        # data_time.update(time.time() - end)

        images[0] = images[0].to(device)
        images[1] = images[1].to(device)
        bg0 = bg0.to(device)
        bg1 = bg1.to(device)
        # mask_q = mask_q.cuda(args.gpu, non_blocking=True)
        # mask_k = mask_k.cuda(args.gpu, non_blocking=True)

        mask_q, mask_k = (bg0[:, 0] == 0).float(), (bg1[:, 0] == 0).float()
        image_q = images[0] * mask_q.unsqueeze(1) + bg0
        image_k = images[1] * mask_k.unsqueeze(1) + bg1

        # compute output
        stride = args.output_stride
        (
            output_instance,
            output_dense,
            target_instance,
            target_dense,
            mask_dense,
        ) = model(
            image_q,
            image_k,
            mask_q[:, stride // 2 :: stride, stride // 2 :: stride],
            mask_k[:, stride // 2 :: stride, stride // 2 :: stride],
        )
        loss_instance = criterion(output_instance, target_instance)

        # dense loss of softmax
        output_dense_log = (-1.0) * nn.LogSoftmax(dim=1)(output_dense)
        output_dense_log = output_dense_log.reshape(output_dense_log.shape[0], -1)
        loss_dense = torch.mean(
            torch.mul(output_dense_log, target_dense).sum(dim=1)
            / target_dense.sum(dim=1)
        )

        loss = loss_instance + loss_dense * 0.2

        acc1, acc5 = accuracy(output_instance, target_instance, topk=(1, 5))
        acc_dense_pos = output_dense.reshape(output_dense.shape[0], -1).argmax(dim=1)
        acc_dense = (
            target_dense[torch.arange(0, target_dense.shape[0]), acc_dense_pos]
            .float()
            .mean()
            * 100.0
        )
        loss_i.update(loss_instance.item(), images[0].size(0))
        loss_d.update(loss_dense.item(), images[0].size(0))
        acc_ins.update(acc1[0], images[0].size(0))
        acc_seg.update(acc_dense.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, "model_best.pth.tar")


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":f"):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = "{name} {val" + self.fmt + "} ({avg" + self.fmt + "})"
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        logging.info("    ".join(entries))
        if torch.distributed.get_rank() == 0:
            logging.info("\t".join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = "{:" + str(num_digits) + "d}"
        return "[" + fmt + "/" + fmt.format(num_batches) + "]"


def adjust_learning_rate(optimizer, epoch, args):
    lr = args.lr
    lr *= 0.5 * (1.0 + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


if __name__ == "__main__":
    # parse command line
    args = get_args()

    # create logging dir
    log_path = os.path.join(args.log_dir, args.run_id)
    os.mkdir(log_path)
    logging.info(f"{log_path = }")

    # set seed
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        cudnn.deterministic = True

    logging.info(f"{torch.cuda.device_count() = }")
    assert args.world_size <= torch.cuda.device_count()

    # get the number of workers
    args.num_workers_per_dataset = args.num_workers // (args.world_size * 3)
    logging.info(f"{args.num_workers_per_dataset = }")

    # spawn parallel process
    mp.spawn(main_worker, args=[args], nprocs=args.world_size)
