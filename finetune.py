import argparse
import os
import subprocess

import lightning as L
import numpy as np
import torch
import torchvision
import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import (Callback, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import WandbLogger
from mmengine.config import Config

from datasets.finetune_dataset import DataSplitType, PolypDataModule
from networks.segment_network import PretrainType, SegmentationModule


def get_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()

    # fmt:off
    parser.add_argument('--config', default='configs/config_finetune.py', help='path to configuration file')
    parser.add_argument("--seed", type=int, default=0, help='Set global seed')
    parser.add_argument("--run_id", type=str, required=True, help='Unique identifier for a run')
    parser.add_argument("--tags", nargs='+', default=[], help='Tags to include for logging')

    parser.add_argument('--offline_wandb', action='store_true', help='Run wandb offline')

    parser.add_argument("--img_dirs", nargs='+', help='Folder(s) containing image data')
    parser.add_argument("--mask_dirs", nargs='+', help='Folder(s) containing segmentation masks')
    parser.add_argument("--train_data_ratio", type=float, default=1.0, help='Amount of finetuning data')
    parser.add_argument("--data_split_type", type=str, choices=[x.name for x in DataSplitType], default=DataSplitType.FILENAME.name)

    parser.add_argument("--log_dir", type=str, required=True, help='For storing artifacts')
    parser.add_argument("--wandb_project", type=str, default='ssl-pretraining', help='Wandb project name')
    parser.add_argument("--wandb_team", type=str, default='critical-ml-dg', help='Wandb team name')
    parser.add_argument("--num_gpus", type=int, default=2, help='number of gpus')
    parser.add_argument("--num_workers", type=int, default=0, help='number of workers')
    parser.add_argument("--fast_dev_run", action='store_true', help="For debugging")
    parser.add_argument("--use_profiler", action='store_true', help="For debugging")

    parser.add_argument("--num_classes", type=int, default=2)

    parser.add_argument('--lemon_data', action='store_true', help='Running with lemon data')

    parser.add_argument('--img_height', default=512, type=int)
    parser.add_argument('--img_width', default=512, type=int)

    parser.add_argument("--batch_size", type=int, default=10, help='Batch size to train with')
    parser.add_argument("--learning_rate", type=float, default=0.0001, help='Max learning rate used during training') 
    parser.add_argument("--epochs", type=int, default=20, help='Number of training epochs') 
    parser.add_argument("--weight_decay", type=float, default=0.0001, help='weight decay of optimizer')  ## from centralai codebase

    parser.add_argument("--pretrain_path", type=str, default='', help="If starting training from a pretrained checkpoint, list the full path to the model with this flag.")
    parser.add_argument("--pretrain_type", type=str, choices=[x.name for x in PretrainType], required=True)

    parser.add_argument("--linear_evaluation", action='store_true', help="Freeze the encoder")
    # fmt:on

    args = parser.parse_args()

    # only 1 for now
    assert len(args.img_dirs) == 1
    assert len(args.mask_dirs) == 1
    # convert to enum
    args.pretrain_type = PretrainType[args.pretrain_type]
    args.data_split_type = DataSplitType[args.data_split_type]

    # lemon data
    if args.lemon_data:
        args.img_height = 544
        args.img_width = 1024
        args.num_classes = 12
        args.epochs = 200

    return args


class CustomCallback(Callback):
    """
    During training, we want to keep track of the learning progress and augmentations
    """

    def __init__(self, images, masks, every_n_epochs=10):
        super().__init__()
        self.images = images
        self.masks = masks
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_start(self, trainer, model):
        if (
            trainer.current_epoch % self.every_n_epochs == 0
            and trainer.global_rank == 0
        ):
            # Reconstruct images
            images = self.images.to(model.device)
            masks = self.masks.to(model.device)
            with torch.no_grad():
                model.eval()
                _, masks_pred = model(images)
                model.train()

            # create torch grids
            image_grid = (
                torchvision.utils.make_grid(images, nrow=1)
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
            )
            mask_grid = (
                torchvision.utils.make_grid(masks.unsqueeze(1), nrow=1)
                .detach()
                .cpu()
                .numpy()
            )
            pred_mask_grid = (
                torchvision.utils.make_grid(masks_pred.unsqueeze(1), nrow=1)
                .detach()
                .cpu()
                .numpy()
            )

            # torch grid makes 1-channel masks 3 channel
            wandb_image = wandb.Image(
                image_grid,
                masks={
                    "predictions": {"mask_data": pred_mask_grid[0, :, :]},
                    "ground_truth": {"mask_data": mask_grid[0, :, :]},
                },
            )
            wandb.log({"Segmentations": wandb_image})


def main(args):
    # Setup data loaders
    datamodule = PolypDataModule(
        data_split_type=args.data_split_type,
        image_directory=args.img_dirs[0],
        mask_directory=args.mask_dirs[0],
        num_classes=args.num_classes,
        image_height=args.img_height,
        image_width=args.img_width,
        lemon_data=args.lemon_data,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_gpus=args.num_gpus,
        train_data_ratio=args.train_data_ratio,
    )

    # logging directory
    args.run_dir = os.path.join(args.log_dir, args.run_id)
    os.makedirs(args.run_dir, exist_ok=True)

    # setup callbacks
    lr_callback = LearningRateMonitor("epoch")
    prefix = "Binary" if args.num_classes == 2 else "Multiclass"
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.run_dir,
        filename="{epoch}-{step}-{val_micro_iou:.2f}",
        save_top_k=1,
        monitor=f"val_{prefix}JaccardIndex",
        mode="max",
    )

    # setup custom callback
    num = 10
    example_images_masks = [datamodule.dataset_test[i] for i in range(num)]
    images = torch.stack([x for x, _ in example_images_masks], dim=0)
    masks = torch.stack([y for _, y in example_images_masks], dim=0)
    custom_callback = CustomCallback(images=images, masks=masks)

    # wandb logger
    tags = ["finetune", "deterministic"] + args.tags
    if args.linear_evaluation:
        tags += ["linear-evaluation"]
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_team,
        tags=tags,
        name=args.run_id,
        save_dir=args.log_dir,
        mode="offline" if args.offline_wandb else "online",
    )

    #
    # Setup the model
    #
    cfg = Config.fromfile(args.config)
    # If no pretraining was used then assert that a path wasn't provided
    if args.pretrain_path:
        assert args.pretrain_type != PretrainType.NONE
        print(f"[INFO] Updating the pretrain_path to {args.pretrain_path = }")
        cfg.model.backbone.init_cfg.checkpoint = args.pretrain_path
    else:
        # Using empty string to denote no path
        assert args.pretrain_path == "", f"{args.pretrain_path = }"

    cfg.model.decode_head.num_classes = args.num_classes
    model = SegmentationModule(
        model_config=cfg,
        pretrain_type=args.pretrain_type,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        num_classes=args.num_classes,
        image_shape=datamodule.image_shape,
    )

    # if linear evaluation freeze backbone
    if args.linear_evaluation:
        for param in model.model.backbone.parameters():
            # not updated by gradient
            param.requires_grad = False

    # setup trainer
    trainer = L.Trainer(
        deterministic=True,
        strategy="ddp" if args.num_gpus > 1 else "auto",
        accelerator="gpu",
        devices=args.num_gpus,
        sync_batchnorm=args.num_gpus > 1,
        precision=32,
        max_epochs=args.epochs,
        logger=wandb_logger,
        profiler="simple" if args.use_profiler else None,
        fast_dev_run=args.fast_dev_run,
        callbacks=[checkpoint_callback, lr_callback, custom_callback],
        log_every_n_steps=1,
    )

    # log additional parameters
    if trainer.global_rank == 0:
        # wandb_logger.watch(model, log="all", log_graph=True)
        wandb_logger.experiment.config.update({"hyper-parameters": vars(args)})
        wandb_logger.experiment.config.update(
            {"nvidia-smi": subprocess.check_output(["nvidia-smi"]).decode()}
        )

    # Train
    trainer.fit(model, datamodule=datamodule)

    # The watch method adds hooks to the model which can be removed at the end of training
    # wandb_logger.experiment.unwatch(model)

    #
    # Test the model
    #
    if trainer.global_rank == 0:
        if args.num_gpus > 1:
            torch.distributed.destroy_process_group()
        # Setup a test trainer
        test_trainer = L.Trainer(
            devices=1,
            num_nodes=1,
            logger=wandb_logger,
            fast_dev_run=args.fast_dev_run,
            accelerator="gpu",
        )

        print(f"{checkpoint_callback.best_model_path = }")
        test_trainer.test(
            model,
            datamodule=datamodule,
            ckpt_path=checkpoint_callback.best_model_path,
        )


if __name__ == "__main__":
    args = get_args()
    print("Module Command Line Arguments: ", vars(args))
    load_dotenv()
    L.seed_everything(args.seed, workers=True)

    # To speed up training
    torch.set_float32_matmul_precision("medium")

    main(args)
