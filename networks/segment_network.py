from enum import Enum

import lightning as L
import torch
import torch.nn as nn
from mmseg.models import build_segmentor
from mmseg.models.utils import resize
from torchmetrics import Accuracy, Dice, JaccardIndex, Precision, Recall, F1Score
import torch.nn as nn

BACKGROUND_CLASS = 0


class PretrainType(Enum):
    """
    Used to determine how to load pretrained weights
    """

    RANDOM = 0
    IMAGENET = 1
    CP2 = 2


class SegmentationModule(L.LightningModule):
    def __init__(
        self,
        model_config,
        pretrain_type: PretrainType,
        learning_rate,
        weight_decay,
        num_classes,
        image_shape,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize the model
        self.model = build_segmentor(model_config.model)
        assert pretrain_type in PretrainType
        if pretrain_type == PretrainType.IMAGENET:
            self.model.backbone.init_cfg.checkpoint = "torchvision://resnet50"
            self.model.backbone.init_weights()
        elif pretrain_type == PretrainType.CP2:
            checkpoint_path = self.model.backbone.init_cfg.checkpoint
            checkpoint = torch.load(checkpoint_path)
            state_dict = {
                x.replace("module.encoder_k.", ""): y
                for x, y in checkpoint["state_dict"].items()
                if "encoder_k" in x
            }
            # Remove the conv_seg weights for now (mismatch in num_classes)
            state_dict = {x: y for x, y in state_dict.items() if "conv_seg" not in x}
            print(self.model.load_state_dict(state_dict, strict=False))
        elif pretrain_type == PretrainType.RANDOM:
            pass
        else:
            raise NotImplementedError(f"{pretrain_type = }")

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.image_shape = image_shape

        self.loss = nn.CrossEntropyLoss()

        self.metrics = nn.ModuleDict(
            {
                x
                + "_micro_iou": JaccardIndex(
                    task="binary" if num_classes == 2 else "multiclass",
                    average="micro",
                    ignore_index=BACKGROUND_CLASS,
                    num_classes=self.num_classes,
                )
                for x in ["train", "val", "test"]
            }
        )
        self.metrics.update(
            {
                x
                + "_micro_dice": Dice(
                    average="micro",
                    ignore_index=BACKGROUND_CLASS,
                    num_classes=self.num_classes,
                )
                for x in ["train", "val", "test"]
            }
        )
        self.metrics.update(
            {
                x
                + "_micro_precision": Precision(
                    task="binary" if num_classes == 2 else "multiclass",
                    multidim_average="global",
                    ignore_index=BACKGROUND_CLASS,
                    num_classes=self.num_classes,
                )
                for x in ["train", "val", "test"]
            }
        )
        self.metrics.update(
            {
                x
                + "_micro_recall": Recall(
                    task="binary" if num_classes == 2 else "multiclass",
                    multidim_average="global",
                    ignore_index=BACKGROUND_CLASS,
                    num_classes=self.num_classes,
                )
                for x in ["train", "val", "test"]
            }
        )
        self.metrics.update(
            {
                x
                + "_micro_accuracy": Accuracy(
                    task="binary" if num_classes == 2 else "multiclass",
                    multidim_average="global",
                    ignore_index=BACKGROUND_CLASS,
                    num_classes=self.num_classes,
                )
                for x in ["train", "val", "test"]
            }
        )
        self.metrics.update(
            {
                x
                + "_micro_f1score": Accuracy(
                    task="binary" if num_classes == 2 else "multiclass",
                    multidim_average="global",
                    ignore_index=BACKGROUND_CLASS,
                    num_classes=self.num_classes,
                )
                for x in ["train", "val", "test"]
            }
        )

        # self.metrics = {
        #     "train_micro_iou": JaccardIndex(
        #         task="multiclass",
        #         average="micro",
        #         ignore_index=BACKGROUND_CLASS,
        #         num_classes=self.num_classes,
        #     ),
        #     "val_micro_iou": JaccardIndex(
        #         task="multiclass",
        #         average="micro",
        #         ignore_index=BACKGROUND_CLASS,
        #         num_classes=self.num_classes,
        #     ),
        #     "test_micro_iou": JaccardIndex(
        #         task="multiclass",
        #         average="micro",
        #         ignore_index=BACKGROUND_CLASS,
        #         num_classes=self.num_classes,
        #     ),
        #     "train_micro_dice": Dice(
        #         average="micro",
        #         ignore_index=BACKGROUND_CLASS,
        #         num_classes=self.num_classes,
        #     ),
        #     "val_micro_dice": Dice(
        #         average="micro",
        #         ignore_index=BACKGROUND_CLASS,
        #         num_classes=self.num_classes,
        #     ),
        #     "test_micro_dice": Dice(
        #         average="micro",
        #         ignore_index=BACKGROUND_CLASS,
        #         num_classes=self.num_classes,
        #     ),
        # }

    def forward(self, images):
        logits = self.model(images)

        # resize from 32 -> 512
        logits = resize(
            input=logits,
            size=self.image_shape[1:],
            mode="bilinear",
            align_corners=False,
        )
        argmax_logits = logits.argmax(dim=1)

        return logits, argmax_logits

    def shared_step(self, batch, stage):
        assert stage in ["train", "val", "test"]

        images, masks = batch
        logits, argmax_logits = self.forward(images)
        loss = self.loss(logits, masks)

        # update logs
        self.log(f"{stage}_loss", loss, sync_dist=True, on_epoch=True, on_step=True)
        for name, metric in self.metrics.items():
            if stage in name:
                metric.update(argmax_logits, masks)
                self.log(name, metric, on_epoch=True, on_step=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self.shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, "test")

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        # Using a scheduler is optional but can be helpful.
        # The scheduler reduces the LR if the validation performance hasn't improved for the last N epochs
        # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        #     optimizer, mode="min", factor=0.2, patience=10, min_lr=5e-5
        # )
        return {
            "optimizer": optimizer,
            # "lr_scheduler": scheduler,
            # "monitor": "val_loss",
        }
