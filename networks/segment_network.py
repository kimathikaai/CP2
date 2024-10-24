from enum import Enum

import lightning as L
import torch
import torch.nn as nn
from mmseg.models import build_segmentor
from mmseg.models.utils import resize
from torchmetrics import (Accuracy, Dice, F1Score, JaccardIndex,
                          MetricCollection, Precision, Recall)

BACKGROUND_CLASS = 0


class PretrainType(Enum):
    """
    Used to determine how to load pretrained weights
    """

    RANDOM = 0
    NONE = 1
    CP2 = 2
    MIRROR = 3
    BYOL = 4
    MOCO = 5
    PROPOSED = 6
    PIXPRO = 7
    DENSECL_IMGNET = 8
    DINO_IMGNET = 9
    BARLOWTWINS_IMGNET = 10
    VICEREGL_IMGNET = 11
    MOCO_IMGNET = 12
    PIXPRO_IMGNET = 13
    BYOL_IMGNET = 14
    CP2_IMGNET = 15
    MOSREP_IMGNET = 16
    CLOVE_IMGNET = 17
    DENSECL = 18
    PROPOSED_V2 = 19


class Stage(Enum):
    TRAIN = 0
    VAL = 1
    TEST = 2
    PSEUDOTEST = 3


class SegmentationModule(L.LightningModule):
    def __init__(
        self,
        model_config,
        pretrain_type: PretrainType,
        learning_rate,
        weight_decay,
        num_classes,
        image_shape,
        use_backbone_only
    ):
        super().__init__()
        self.save_hyperparameters()

        # Initialize the model
        self.model = build_segmentor(model_config.model)
        assert pretrain_type in PretrainType
        if pretrain_type == PretrainType.NONE:
            # ImageNet initialization
            self.model.backbone.init_cfg.checkpoint = "torchvision://resnet50"
            self.model.backbone.init_weights()
        elif pretrain_type == PretrainType.RANDOM:
            pass
        elif pretrain_type in [
            PretrainType.CP2,
            PretrainType.MOCO,
            PretrainType.BYOL,
            PretrainType.PROPOSED,
            PretrainType.DENSECL,
            PretrainType.PROPOSED_V2
        ]:
            checkpoint_path = self.model.backbone.init_cfg.checkpoint
            checkpoint = torch.load(checkpoint_path)
            assert (
                checkpoint["pretrain_type"] == pretrain_type.name
            ), f"{checkpoint['pretrain_type']} != {pretrain_type}"
            filter = 'encoder_q.backbone' if use_backbone_only else 'encoder_q.'
            state_dict = {
                x.replace("module.encoder_q.", ""): y
                for x, y in checkpoint["state_dict"].items()
                if filter in x
            }
            # Remove the conv_seg weights for now (mismatch in num_classes)
            state_dict = {x: y for x, y in state_dict.items() if "conv_seg" not in x}
            print(self.model.load_state_dict(state_dict, strict=False))
            print(f"[INFO] {use_backbone_only = }")

        elif pretrain_type == PretrainType.MIRROR:
            checkpoint_path = self.model.backbone.init_cfg.checkpoint
            checkpoint = torch.load(checkpoint_path)
            state_dict = checkpoint["state_dict"]
            # Remove the conv_seg weights for now (mismatch in num_classes)
            state_dict = {x: y for x, y in state_dict.items() if "conv_seg" not in x}
            print(self.load_state_dict(state_dict, strict=False))

        elif pretrain_type == PretrainType.PIXPRO:
            checkpoint_path = self.model.backbone.init_cfg.checkpoint
            checkpoint = torch.load(checkpoint_path)
            # Check that the weights match the type
            assert (
                checkpoint["pretrain_type"] == pretrain_type.name
            ), f"{checkpoint['pretrain_type']} != {pretrain_type}"
            state_dict = {
                x.replace("module.encoder.", ""): y
                for x, y in checkpoint["model"].items()
                if "module.encoder." in x
            }
            print(self.model.backbone.load_state_dict(state_dict, strict=True))
        elif pretrain_type in [PretrainType.PIXPRO_IMGNET, PretrainType.CLOVE_IMGNET]:
            checkpoint_path = self.model.backbone.init_cfg.checkpoint
            checkpoint = torch.load(checkpoint_path)
            # Check that the weights match the type
            # assert (
            #     checkpoint["pretrain_type"] == pretrain_type.name
            # ), f"{checkpoint['pretrain_type']} != {pretrain_type}"
            state_dict = {
                x.replace("module.encoder.", ""): y
                for x, y in checkpoint["model"].items()
                if "module.encoder." in x
            }

            print(self.model.backbone.load_state_dict(state_dict, strict=True))
        elif pretrain_type == PretrainType.DENSECL_IMGNET:
            checkpoint_path = self.model.backbone.init_cfg.checkpoint
            checkpoint = torch.load(checkpoint_path)
            state_dict = checkpoint["state_dict"]

            # assert (
            #     checkpoint["pretrain_type"] == pretrain_type.name
            # ), f"{checkpoint['pretrain_type']} != {pretrain_type}"

            print(self.model.backbone.load_state_dict(state_dict, strict=False))

        elif pretrain_type in [
            PretrainType.BYOL_IMGNET,
            PretrainType.CP2_IMGNET,
            PretrainType.VICEREGL_IMGNET,
            PretrainType.BARLOWTWINS_IMGNET,
            PretrainType.DINO_IMGNET,
        ]:
            checkpoint_path = self.model.backbone.init_cfg.checkpoint
            checkpoint = torch.load(checkpoint_path)
            print(self.model.backbone.load_state_dict(checkpoint, strict=False))

        elif pretrain_type in [PretrainType.MOSREP_IMGNET, PretrainType.MOCO_IMGNET]:
            checkpoint_path = self.model.backbone.init_cfg.checkpoint
            checkpoint = torch.load(checkpoint_path)
            state_dict = {
                x.replace("module.encoder_q.", ""): y
                for x, y in checkpoint["state_dict"].items()
                if "encoder_q" in x
            }
            print(self.model.backbone.load_state_dict(state_dict, strict=False))
        else:
            raise NotImplementedError(f"{pretrain_type = }")

        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.num_classes = num_classes
        self.image_shape = image_shape

        # RuntimeError: nll_loss2d_forward_out_cuda_template does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'
        # https://discuss.pytorch.org/t/pytorchs-non-deterministic-cross-entropy-loss-and-the-problem-of-reproducibility/172180/9
        self.loss = nn.CrossEntropyLoss(reduction="none")

        #
        # Metrics
        #
        self.binary_task = self.num_classes == 2
        self.metric_task = "binary" if self.binary_task else "multiclass"
        self.ignore_index = None if self.binary_task else BACKGROUND_CLASS
        metrics = MetricCollection(
            [
                JaccardIndex(
                    task=self.metric_task,
                    average="micro",
                    ignore_index=self.ignore_index,
                    num_classes=self.num_classes,
                ),
                Dice(
                    average="micro",
                    ignore_index=BACKGROUND_CLASS,
                    num_classes=self.num_classes,
                ),
                Precision(
                    task=self.metric_task,
                    multidim_average="global",
                    ignore_index=self.ignore_index,
                    num_classes=self.num_classes,
                ),
                Recall(
                    task=self.metric_task,
                    multidim_average="global",
                    ignore_index=self.ignore_index,
                    num_classes=self.num_classes,
                ),
                F1Score(
                    task=self.metric_task,
                    multidim_average="global",
                    ignore_index=self.ignore_index,
                    num_classes=self.num_classes,
                ),
            ]
        )
        self.train_metrics = metrics.clone(prefix=f"{Stage.TRAIN.name.lower()}_")
        self.val_metrics = metrics.clone(prefix=f"{Stage.VAL.name.lower()}_")
        self.test_metrics = metrics.clone(prefix=f"{Stage.TEST.name.lower()}_")
        self.pseudo_test_metrics = metrics.clone(
            prefix=f"{Stage.PSEUDOTEST.name.lower()}_"
        )

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

    def shared_step(self, batch, stage: Stage):
        images, masks = batch
        logits, argmax_logits = self.forward(images)
        # argmax_logits.shape = BxCxHxW
        loss = self.loss(logits, masks)
        # For reproducability
        loss = loss.mean()

        # update logs
        self.log(
            f"{stage.name.lower()}_loss",
            loss,
            sync_dist=True,
            on_epoch=True,
            on_step=True,
            add_dataloader_idx=False,
        )
        if stage == Stage.TRAIN:
            self.train_metrics.update(argmax_logits, masks)
            self.log_dict(
                {k: v for k, v in self.train_metrics.items()},
                on_epoch=True,
                on_step=True,
            )
        elif stage == Stage.VAL:
            self.val_metrics.update(argmax_logits, masks)
            self.log_dict(
                {k: v for k, v in self.val_metrics.items()},
                on_epoch=True,
                on_step=False,
                add_dataloader_idx=False,
            )
        elif stage == Stage.PSEUDOTEST:
            self.pseudo_test_metrics.update(argmax_logits, masks)
            self.log_dict(
                {k: v for k, v in self.pseudo_test_metrics.items()},
                on_epoch=True,
                on_step=False,
                add_dataloader_idx=False,
            )
        elif stage == Stage.TEST:
            self.test_metrics.update(argmax_logits, masks)
            self.log_dict(
                {k: v for k, v in self.test_metrics.items()},
                on_epoch=True,
                on_step=False,
            )

        return loss

    def training_step(self, batch, batch_idx):
        return self.shared_step(batch, Stage.TRAIN)

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        return self.shared_step(
            batch, Stage.VAL if dataloader_idx == 0 else Stage.PSEUDOTEST
        )

    def test_step(self, batch, batch_idx):
        return self.shared_step(batch, Stage.TEST)

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
