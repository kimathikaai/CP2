import argparse
import os

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

from networks.mirror_network import MirrorModule
from datasets.pretrain_dataset import CutPasteDataModule, MirrorVariant
from networks.segment_network import PretrainType
from finetune import CustomCallback


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

    parser.add_argument("--data_dirs", nargs='+', help='Folder(s) containing image data')

    parser.add_argument("--log_dir", type=str, required=True, help='For storing artifacts')
    parser.add_argument("--wandb_project", type=str, default='ssl-pretraining', help='Wandb project name')
    parser.add_argument("--wandb_team", type=str, default='critical-ml-dg', help='Wandb team name')
    parser.add_argument("--num_gpus", type=int, default=2, help='number of gpus')
    parser.add_argument("--num-workers", type=int, default=0, help='number of workers')
    parser.add_argument("--fast_dev_run", action='store_true', help="For debugging")
    parser.add_argument("--use_profiler", action='store_true', help="For debugging")

    parser.add_argument("-x", "--img_x_size", type=int, default=512, help='height of image')
    parser.add_argument("-y", "--img_y_size", type=int, default=512, help='width of image')
    parser.add_argument("--num_classes", type=int, default=2)

    parser.add_argument('--lemon_data', action='store_true', help='Running with lemon data')

    # cutpaste
    parser.add_argument('--softmax_temp', type=float, default=2)
    parser.add_argument("--lmbd_compare_loss", type=float, default=0.01, help='Loss coefficient')
    parser.add_argument('--variant', choices=[x.name for x in MirrorVariant], default=MirrorVariant.OUTPUT.name)
    parser.add_argument("--max_num_patches", type=int, default=1, help='Maximum number of cutpastes')
    parser.add_argument("--min_area_scale", type=float, default=0.02, help='minimum area of patch')
    parser.add_argument("--max_area_scale", type=float, default=0.15, help='maximum area of patch')
    parser.add_argument("--min_aspect_ratio", type=float, default=1/3, help='minimum aspect ratio of patch')
    parser.add_argument("--max_aspect_ratio", type=float, default=4/3, help='maximum aspect ratio of patch')
    parser.add_argument("--min_rotation", type=int, default=0, help='minimum rotation angle of patch')
    parser.add_argument("--max_rotation", type=int, default=0, help='max rotation angle of patch')


    parser.add_argument("--batch-size", type=int, default=10, help='Batch size to train with')
    parser.add_argument("--lr", type=float, default=0.001, help='Max learning rate used during training') 
    parser.add_argument("--epochs", type=int, default=200, help='Number of training epochs') 
    parser.add_argument("--weight_decay", type=float, default=0.0001, help='weight decay of optimizer')  ## from centralai codebase


    # fmt:on

    args = parser.parse_args()

    args.log_dir = os.path.abspath(os.path.expanduser(args.log_dir))
    # convert to enum
    args.variant = MirrorVariant[args.variant]

    # lemon data
    if args.lemon_data:
        args.img_x_size = 544
        args.img_y_size = 1024
        args.epochs = 200
        args.max_area_scale = 0.007
        args.min_area_scale = 0.0003
        args.max_num_patches = 1

    return args


class MirrorCallback(Callback):
    """
    During training, we want to keep track of the learning progress and augmentations
    """

    def __init__(self, images_a, images_b, masks, every_n_epochs=5):
        super().__init__()
        self.images_a = images_a
        self.images_b = images_b
        self.masks = masks
        self.every_n_epochs = every_n_epochs

    def on_train_epoch_start(self, trainer, model):
        if (
            trainer.current_epoch % self.every_n_epochs == 0
            and trainer.global_rank == 0
        ):
            # Reconstruct images
            images = torch.cat([self.images_a, self.images_b]).to(model.device)
            masks = torch.cat([self.masks, self.masks]).to(model.device)
            with torch.no_grad():
                model.eval()
                _, masks_pred = model(images)
                model.train()

            # create torch grids
            image_grid = (
                torchvision.utils.make_grid(images, nrow=2)
                .permute(1, 2, 0)
                .detach()
                .cpu()
                .numpy()
            )
            mask_grid = (
                torchvision.utils.make_grid(masks.unsqueeze(1), nrow=2)
                .detach()
                .cpu()
                .numpy()
            )
            pred_mask_grid = (
                torchvision.utils.make_grid(masks_pred.unsqueeze(1), nrow=2)
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
    datamodule = CutPasteDataModule(
        img_dir_list=args.data_dirs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        num_classes=args.num_classes,
        max_num_patches=args.max_num_patches,
        variant=args.variant,
        img_x_size=args.img_x_size,
        img_y_size=args.img_y_size,
        min_area_scale=args.min_area_scale,
        max_area_scale=args.max_area_scale,
        min_aspect_ratio=args.min_aspect_ratio,
        max_aspect_ratio=args.max_aspect_ratio,
        min_rotation=args.min_rotation,
        max_rotation=args.max_rotation,
    )
    print("CPDataModule loaded")

    # logging directory
    args.run_dir = os.path.join(args.log_dir, args.run_id)
    os.makedirs(args.run_dir, exist_ok=True)

    # setup callbacks
    lr_callback = LearningRateMonitor("epoch")
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.run_dir,
        filename="checkpoint",
        save_top_k=1,
        monitor="val_loss_epoch",
        mode="min",
    )

    # setup custom callback
    num = 10
    if args.variant == MirrorVariant.OUTPUT:
        samples = [datamodule.dataset_val[i] for i in range(num)]
        images_a = torch.stack([a for a,_, _ in samples], dim=0)
        images_b = torch.stack([b for _,b, _ in samples], dim=0)
        masks = torch.stack([y for _,_, y in samples], dim=0)
        custom_callback = MirrorCallback(images_a=images_a, images_b=images_b, masks=masks)
    elif args.variant == MirrorVariant.NONE:
        samples = [datamodule.dataset_val[i] for i in range(num)]
        images = torch.stack([x for x, _ in samples], dim=0)
        masks = torch.stack([y for _, y in samples], dim=0)
        custom_callback = CustomCallback(images=images, masks=masks)
    else:
        raise NotImplementedError(f"{args.variant =}")

    # wandb logger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        entity=args.wandb_team,
        tags=["pretrain"] + args.tags,
        name=args.run_id,
        save_dir=args.run_dir,
    )

    #
    # Setup the model
    #
    cfg = Config.fromfile(args.config)

    cfg.model.decode_head.num_classes = args.num_classes
    cfg.model.decode_head.contrast = False
    model = MirrorModule(
        model_config=cfg,
        pretrain_type=PretrainType.IMAGENET,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        num_classes=args.num_classes,
        image_shape=datamodule.image_shape,
        lmbd_compare_loss=args.lmbd_compare_loss,
        softmax_temp=args.softmax_temp,
        mirror_variant=args.variant
    )

    # setup trainer
    trainer = L.Trainer(
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

    # Train
    trainer.fit(model, datamodule=datamodule)

    # The watch method adds hooks to the model which can be removed at the end of training
    # wandb_logger.experiment.unwatch(model)
    print(f"{checkpoint_callback.best_model_path = }")


if __name__ == "__main__":
    args = get_args()
    print("Module Command Line Arguments: ", vars(args))
    load_dotenv()
    L.seed_everything(args.seed, workers=True)

    # To speed up training
    torch.set_float32_matmul_precision("medium")

    main(args)
