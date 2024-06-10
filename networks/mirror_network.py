import torch
import torch.nn as nn

from networks.segment_network import PretrainType, SegmentationModule, Stage
from datasets.pretrain_dataset import MirrorVariant


class MirrorModule(SegmentationModule):
    def __init__(
        self,
        model_config,
        pretrain_type: PretrainType,
        learning_rate,
        weight_decay,
        num_classes,
        image_shape,
        lmbd_compare_loss,
        softmax_temp,
        mirror_variant,
    ):
        super().__init__(
            model_config=model_config,
            pretrain_type=pretrain_type,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            num_classes=num_classes,
            image_shape=image_shape,
        )

        self.class_loss = nn.CrossEntropyLoss()
        # self.compare_loss = nn.MSELoss(reduction="mean")
        self.compare_loss = nn.CrossEntropyLoss()
        self.lmbd_compare_loss = lmbd_compare_loss
        # https://intellabs.github.io/distiller/knowledge_distillation.html
        self.softmax_temp = softmax_temp
        self.softmax = nn.Softmax(dim=1)
        assert mirror_variant in MirrorVariant
        self.mirror_variant = mirror_variant

    def shared_step(self, batch, stage: Stage):
        compare_loss = 0
        if self.mirror_variant == MirrorVariant.OUTPUT:
            s_img, t_img, masks = batch
            s_logits, s_argmax = self.forward(s_img)
            t_logits, t_argmax = self.forward(t_img)

            all_logits = torch.cat([s_logits, t_logits])
            all_masks = torch.cat([masks, masks])
            compare_loss = self.compare_loss(
                self.softmax(s_logits / self.softmax_temp),
                self.softmax(t_logits / self.softmax_temp),
            )
        elif self.mirror_variant == MirrorVariant.NONE:
            img, all_masks = batch
            all_logits, _ = self.forward(img)
        else:
            raise NotImplementedError(f"{self.mirror_variant = }")

        all_argmax = all_logits.argmax(dim=1)

        # losses
        class_loss = self.class_loss(all_logits, all_masks)
        loss = class_loss + self.lmbd_compare_loss * compare_loss

        # fmt:off
        self.log(f"{stage.name.lower()}_loss", loss, sync_dist=True, on_epoch=True, on_step=True)
        self.log(f"{stage.name.lower()}_compare_loss", compare_loss, sync_dist=True, on_epoch=True, on_step=True)
        self.log(f"{stage.name.lower()}_class_loss", class_loss, sync_dist=True, on_epoch=True, on_step=True)
        # fmt:on

        # update logs
        if stage == Stage.TRAIN:
            self.train_metrics.update(all_argmax, all_masks)
            self.log_dict(
                {k: v for k, v in self.train_metrics.items()},
                on_epoch=True,
                on_step=True,
            )
        elif stage == Stage.VAL:
            self.val_metrics.update(all_argmax, all_masks)
            self.log_dict(
                {k: v for k, v in self.val_metrics.items()},
                on_epoch=True,
                on_step=False,
            )
        return loss
