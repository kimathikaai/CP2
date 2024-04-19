import argparse
import os

import lightning as L
import torch
import torchvision
import wandb
from dotenv import load_dotenv
from lightning.pytorch.callbacks import (Callback, LearningRateMonitor,
                                         ModelCheckpoint)
from lightning.pytorch.loggers import WandbLogger

from datasets.finetune_dataset import GLASDataModule


def get_args():
    """
    Parse arguments.
    """
    parser = argparse.ArgumentParser()

    # fmt:off
    parser.add_argument('--config', help='path to configuration file')
    parser.add_argument("--seed", type=int, default=0, help='Set global seed')
    parser.add_argument("--run_id", type=str, required=True, help='Unique identifier for a run')

    parser.add_argument("--img_dirs", nargs='+', help='Folder(s) containing image data')
    parser.add_argument("--mask_dirs", nargs='+', help='Folder(s) containing segmentation masks')
    parser.add_argument("--train_data_ratio", type=float, default=1.0, help='Amount of finetuning data')

    parser.add_argument("--log_dir", type=str, help='For storing artifacts')
    parser.add_argument("--wandb_project", type=str, required=True, help='Wandb project name')
    parser.add_argument("--num_gpus", type=int, default=4, help='number of gpus')
    parser.add_argument("--num_workers", type=int, default=0, help='number of workers')
    parser.add_argument("--fast_dev_run", action='store_true', help="For debugging")
    parser.add_argument("--use_profiler", action='store_true', help="For debugging")

    parser.add_argument("--img_size", type=int, default=512)
    parser.add_argument("--num_classes", type=int, required=True)

    parser.add_argument("--batch_size", type=int, default=10, help='Batch size to train with')
    parser.add_argument("--lr", type=float, default=0.0001, help='Max learning rate used during training') 
    parser.add_argument("--epochs", type=int, default=200, help='Number of training epochs') 
    parser.add_argument("--weight_decay", type=float, default=0.05, help='weight decay of optimizer')  ## from centralai codebase
    parser.add_argument("--pretrain_path", type=str, default=None, help="If starting training from a pretrained checkpoint, list the full path to the model with this flag.")
    # fmt:on

    args = parser.parse_args()

    # only 1 for now
    assert len(args.img_dirs) == 1
    assert len(args.mask_dirs) == 1

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

    def on_train_epoch_end(self, trainer, model):
        if (
            trainer.current_epoch % self.every_n_epochs == 0
            and trainer.global_rank == 0
        ):
            # Reconstruct images
            images = self.images.to(model.device)
            masks = self.masks.to(model.device)
            with torch.no_grad():
                model.eval()
                masks_pred = model(images)
                model.train()

            # Plot one image for now
            mask_img = wandb.Image(
                images[0],
                masks={
                    "predictions": {"mask_data": masks_pred[0]},
                    "ground_truth": {"mask_data": masks[0]},
                },
            )
            wandb.log({"Predictions/Augmentations": mask_img})
            # trainer.logger.experiment.log_image()

            # Plot and add to tensorboard
            # imgs = torch.stack([images, masks, masks_pred], dim=1).flatten(0, 1)
            # grid = torchvision.utils.make_grid(imgs, nrow=3)
            # trainer.logger.experiment.log_image(
            #     "Reconstructions", grid, global_step=trainer.global_step
            # )


def main(args):
    # Setup data loaders
    datamodule = GLASDataModule(
        image_directory=args.img_dirs[0],
        mask_directory=args.mask_dirs[0],
        num_classes=args.num_classes,
        image_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_data_ratio=args.train_data_ratio,
    )

    # logging directory
    args.run_dir = os.path.join(args.log_dir, args.run_id)
    os.mkdir(args.run_dir)

    # setup callbacks
    lr_callback = LearningRateMonitor("epoch")
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.run_dir,
        filename="{epoch}-{step}-{val_micro_iou:.2f}",
        save_top_k=1,
        monitor="val_micro_iou",
        mode="max",
    )

    # setup custom callback
    num = 1
    example_images_masks = [datamodule.dataset_train[i] for i in range(num)]
    images = torch.stack([x for x, _ in example_images_masks], dim=0)
    masks = torch.stack([y for _, y in example_images_masks], dim=0)
    custom_callback = CustomCallback(images=images, masks=masks)

    # wandb logger
    wandb_logger = WandbLogger(
        project=args.wandb_project,
        tags=["finetune"],
        name=args.run_id,
        save_dir=args.run_dir,
    )

    # setup trainer
    trainer = L.Trainer(
        strategy="ddp",
        accelerator="gpu",
        devices=args.num_gpus,
        sync_batchnorm=args.num_gpus > 1,
        precision=32,
        max_epochs=args.epochs,
        logger=wandb_logger,
        profiler="simple" if args.use_profiler else None,
        fast_dev_run=args.fast_dev_run,
        callbacks=[checkpoint_callback, lr_callback, custom_callback],
    )

    # log additional parameters
    if trainer.global_rank == 0:
        wandb_logger.experiment.config.update({"hyper-parameters": vars(args)})


if __name__ == "__main__":
    args = get_args()
    print("Module Command Line Arguments: ", vars(args))
    load_dotenv()
    L.seed_everything(args.seed, workers=True)

    # To speed up training
    torch.set_float32_matmul_precision("medium")

    main(args)
