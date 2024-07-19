# The CP2_MoCo model is built upon moco v2 code base:
# https://github.com/facebookresearch/moco
# Copyright (c) Facebook, Inc. and its affilates. All Rights Reserved
import copy
from enum import Enum

import cv2
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import wandb
from mmseg.models import build_segmentor
from mmseg.models.decode_heads import FCNHead

from networks.segment_network import PretrainType
from tools.correlation_mapping import (get_correlation_map,
                                       get_masked_correlation_map)


class BackboneType(Enum):
    """
    Used to differentiate which backbone architecture is used
    """

    DEEPLABV3 = 0
    UNET_ENCODER_ONLY = 1
    UNET_TRUNCATED = 2


class MappingType(Enum):
    """
    Used to determine how pixel level comparisons are made
    """

    CP2 = 0
    PIXEL_PIXEL_REGION = 1


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


class UNET_TRUNCATED(nn.Module):
    def __init__(self, projector_dim, num_decoder_blocks=2) -> None:
        super().__init__()
        decoder_channels = [256, 128, 64, 32, 16]
        self.model = smp.Unet(
            "resnet50",
            encoder_weights="imagenet",
            classes=2,
            in_channels=3,
            activation=None,
            encoder_depth=5,
            decoder_channels=decoder_channels,
        )

        assert num_decoder_blocks > 0, f"{num_decoder_blocks = }"
        self.num_decoder_blocks = num_decoder_blocks

        # self.channels = self.model.encoder.out_channels[-1]
        self.channels = decoder_channels[self.num_decoder_blocks - 1]
        self.backbone = self.model.encoder
        # TODO: determine the size of this decoder output
        self.projector = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 1),
            nn.ReLU(),
            nn.Conv2d(self.channels, projector_dim, 1),
        )

        blocks = [self.model.decoder.blocks[i] for i in range(self.num_decoder_blocks)]
        self.model.decoder.blocks = nn.ModuleList(blocks)

    def forward(self, x):
        features = self.backbone(x)
        features = self.model.decoder(*features)
        projection = self.projector(features)
        return projection


class UNET_ENCODER_ONLY(nn.Module):
    def __init__(self, projector_dim) -> None:
        super().__init__()
        self.model = smp.Unet(
            "resnet50",
            encoder_weights="imagenet",
            classes=2,
            in_channels=3,
            activation=None,
            encoder_depth=5,
            decoder_channels=[256, 128, 64, 32, 16],
        )

        self.channels = self.model.encoder.out_channels[-1]
        self.backbone = self.model.encoder
        self.projector = nn.Sequential(
            nn.Conv2d(self.channels, self.channels, 1),
            nn.ReLU(),
            nn.Conv2d(self.channels, projector_dim, 1),
        )

    def forward(self, x):
        features = self.backbone(x)
        projection = self.projector(features[-1])
        return projection


class CP2_MOCO(nn.Module):
    def __init__(
        self,
        cfg,
        rank,
        dim=128,
        K=65536,
        m=0.999,
        T=0.2,
        include_background=False,
        lmbd_cp2_dense_loss=0.2,
        lmbd_corr_weight=1,
        pretrain_type=PretrainType.CP2,
        backbone_type=BackboneType.DEEPLABV3,
        mapping_type=MappingType.CP2,
        unet_truncated_dec_blocks=2,
        device=None,
    ):
        super(CP2_MOCO, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.dim = dim
        self.include_background = include_background
        self.lmbd_cp2_dense_loss = lmbd_cp2_dense_loss
        self.device = device
        self.rank = rank
        self.epoch = 0

        assert mapping_type in MappingType
        self.mapping_type = mapping_type

        # Validate the correlation map weight
        if mapping_type == MappingType.CP2:
            assert lmbd_corr_weight == 1
        elif mapping_type == MappingType.PIXEL_PIXEL_REGION:
            assert lmbd_corr_weight >= 1
        else:
            raise NotImplementedError(f"{mapping_type = }")
        self.lmbd_corr_weight = lmbd_corr_weight

        assert pretrain_type in PretrainType
        self.pretrain_type = pretrain_type

        assert backbone_type in BackboneType
        self.backbone_type = backbone_type

        # No current handling for both
        if backbone_type != BackboneType.DEEPLABV3:
            assert (
                pretrain_type == PretrainType.CP2
            ), f"{backbone_type = }, {pretrain_type = }"

        if backbone_type == BackboneType.DEEPLABV3:
            self.encoder_q = build_segmentor(
                cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
            )
            self.encoder_k = build_segmentor(
                cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
            )
            # Initialize the model pretrained ImageNet weights
            self.encoder_q.backbone.init_weights()
            self.encoder_k.backbone.init_weights()
        elif backbone_type == BackboneType.UNET_ENCODER_ONLY:
            self.encoder_q = UNET_ENCODER_ONLY(projector_dim=dim)
            self.encoder_k = UNET_ENCODER_ONLY(projector_dim=dim)
        elif backbone_type == BackboneType.UNET_TRUNCATED:
            self.encoder_q = UNET_TRUNCATED(
                projector_dim=dim, num_decoder_blocks=unet_truncated_dec_blocks
            )
            self.encoder_k = UNET_TRUNCATED(
                projector_dim=dim, num_decoder_blocks=unet_truncated_dec_blocks
            )
        else:
            raise NotImplementedError(f"{backbone_type = }")

        # Get the output stride
        test_sample = torch.rand(2, 3, 224, 224)
        output_shape = self.encoder_q(test_sample).shape
        self.output_stride = int(test_sample.shape[2] / output_shape[2])
        print(f"{self.output_stride = }")

        if pretrain_type in [PretrainType.BYOL, PretrainType.MOCO]:
            # Projection/prediction networks
            # backbone_features = 2048 * 7 * 7  # if imgs are 224x224
            backbone_features = 2048 * 14 * 14  # if imgs are 224x224
            hidden_features = 2048
            batch_norm = (
                nn.BatchNorm1d(hidden_features)
                if pretrain_type == PretrainType.BYOL
                else nn.Identity()
            )
            self.encoder_q.projector = nn.Sequential(
                nn.Linear(in_features=backbone_features, out_features=hidden_features),
                batch_norm,
                nn.ReLU(inplace=True),
                nn.Linear(in_features=hidden_features, out_features=self.dim),
            )
            self.encoder_k.projector = copy.deepcopy(self.encoder_q.projector)

            self.predictor = nn.Sequential(
                nn.Linear(in_features=self.dim, out_features=hidden_features),
                batch_norm,
                nn.ReLU(inplace=True),
                nn.Linear(in_features=hidden_features, out_features=self.dim),
            )

        # Exact copy parameters
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # confirm config
        # if self.pretrain_type in [PretrainType.MOCO, PretrainType.BYOL]:
        #     assert isinstance(self.encoder_q.decode_head, FCNHead)

        # create the queue
        self.register_buffer("queue", torch.randn(self.dim, K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Metrics
        self.loss_o = AverageMeter("Loss_overall", ":.4f")
        self.loss_i = AverageMeter("Loss_ins", ":.4f")
        self.loss_d = AverageMeter("Loss_den", ":.4f")
        self.acc_ins = AverageMeter("Acc_ins", ":6.2f")
        self.acc_seg = AverageMeter("Acc_seg", ":6.2f")
        self.cross_image_variance_source = AverageMeter(
            "Cross_Image_Variance_Source", ":6.2f"
        )
        self.cross_image_variance_target = AverageMeter(
            "Cross_Image_Variance_Target", ":6.2f"
        )
        self.correlation_ious = []
        self.masked_correlation_ious = []

    @torch.no_grad()
    def _momentum_update_key_encoder(self):
        """
        Momentum update of the key encoder
        """
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data = param_k.data * self.m + param_q.data * (1.0 - self.m)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.K:
            self.queue[:, ptr : self.K] = keys[0 : self.K - ptr].T
            self.queue[:, 0 : ptr + batch_size - self.K] = keys[
                self.K - ptr : batch_size
            ].T
        else:
            self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.K  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, **kwargs):
        if self.pretrain_type == PretrainType.CP2:
            return self.forward_cp2(**kwargs)
        elif self.pretrain_type == PretrainType.BYOL:
            return self.forward_byol(**kwargs)
        elif self.pretrain_type == PretrainType.MOCO:
            return self.forward_moco(**kwargs)

    def forward_moco(
        self, img_a, img_b, bg0, bg1, visualize, step, new_epoch, ids_a, ids_b
    ):
        if visualize and self.rank == 0:
            log_imgs = torch.stack([img_a, img_b], dim=1).flatten(0, 1)
            log_grid = torchvision.utils.make_grid(log_imgs, nrow=2)
            wandb.log(
                {
                    "train-examples": wandb.Image(
                        log_grid, caption=self.pretrain_type.name
                    )
                }
            )

        # compute query features
        # fmt:off
        embd_q = self.encoder_q.backbone(img_a)[3]
        q = self.encoder_q.projector(embd_q.flatten(1))
        q = F.normalize(q, dim=-1)

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            img_b, idx_unshuffle = self._batch_shuffle_ddp(img_b)
            k = self.encoder_k.projector(self.encoder_k.backbone(img_b)[3].flatten(1))
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle) 
            k = F.normalize(k, dim=-1)

        # moco logits
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        logits_moco = torch.cat([l_pos, l_neg], dim=1)
        labels_moco = torch.zeros(logits_moco.shape[0], dtype=torch.long).to(self.device)
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        loss = F.cross_entropy(logits_moco/self.T, labels_moco).mean()

        acc1, acc5 = self._accuracy(logits_moco, labels_moco, topk=(1, 5))

        # Update logs
        self.loss_o.update(loss.item(), img_a.size(0))
        self.acc_ins.update(acc1[0], img_a.size(0))

        if self.rank == 0:
            wandb.log(
                {
                    "train/loss_step": self.loss_o.val,
                    "train/acc_ins_step": self.acc_ins.val,
                }
            )


        return loss

    def forward_byol(
        self, img_a, img_b, bg0, bg1, visualize, step, new_epoch, ids_a, ids_b
    ):
        def loss_byol(x, y):
            x = F.normalize(x, dim=-1, p=2)
            y = F.normalize(y, dim=-1, p=2)
            return 2 - 2 * torch.einsum("nc,nc->n", [x, y])

        if visualize and self.rank == 0:
            log_imgs = torch.stack([img_a, img_b], dim=1).flatten(0, 1)
            log_grid = torchvision.utils.make_grid(log_imgs, nrow=2)
            wandb.log(
                {
                    "train-examples": wandb.Image(
                        log_grid, caption=self.pretrain_type.name
                    )
                }
            )

        # compute query features
        # fmt:off
        embd_a = self.encoder_q.backbone(img_a)[3]
        embd_b = self.encoder_q.backbone(img_b)[3]
        q_a = self.predictor(self.encoder_q.projector(embd_a.flatten(1)))
        q_b = self.predictor(self.encoder_q.projector(embd_b.flatten(1)))
        # fmt:on

        # compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder
            k_a = self.encoder_k.projector(self.encoder_k.backbone(img_a)[3].flatten(1))
            k_b = self.encoder_k.projector(self.encoder_k.backbone(img_b)[3].flatten(1))

        loss = (loss_byol(q_a, k_b) + loss_byol(q_b, k_a)).mean()

        # Update logs
        self.loss_o.update(loss.item(), img_a.size(0))

        if self.rank == 0:
            wandb.log(
                {
                    "train/loss_step": self.loss_o.val,
                }
            )

        return loss

    def forward_cp2(
        self, img_a, img_b, bg0, bg1, visualize, step, new_epoch, ids_a, ids_b
    ):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        mask_a, mask_b = (bg0[:, 0] == 0).float(), (bg1[:, 0] == 0).float()
        _img_a = img_a.clone()
        _img_b = img_b.clone()
        img_a = img_a * mask_a.unsqueeze(1) + bg0
        img_b = img_b * mask_b.unsqueeze(1) + bg1

        # update map to correct size
        mask_a = mask_a[
            :,
            self.output_stride // 2 :: self.output_stride,
            self.output_stride // 2 :: self.output_stride,
        ]
        ids_a = ids_a[
            :,
            self.output_stride // 2 :: self.output_stride,
            self.output_stride // 2 :: self.output_stride,
        ]
        mask_b = mask_b[
            :,
            self.output_stride // 2 :: self.output_stride,
            self.output_stride // 2 :: self.output_stride,
        ]
        ids_b = ids_b[
            :,
            self.output_stride // 2 :: self.output_stride,
            self.output_stride // 2 :: self.output_stride,
        ]

        if visualize and self.rank == 0:
            log_imgs = torch.stack(
                [_img_a, _img_b, bg0, bg1, img_a, img_b], dim=1
            ).flatten(0, 1)
            log_grid = torchvision.utils.make_grid(log_imgs, nrow=6)
            wandb.log(
                {
                    "train-examples": wandb.Image(
                        log_grid, caption=self.pretrain_type.name
                    )
                }
            )

        #
        # Generate the pixel correspondence map
        #
        results = get_masked_correlation_map(
            map_a=ids_a,
            map_b=ids_b,
            mask_a=mask_a,
            mask_b=mask_b,
        )
        corr_map = results["corr_map"]
        corr_weights = (self.lmbd_corr_weight * corr_map + ~corr_map).detach()

        corr_map_a = results["corr_map_a"]
        corr_map_b = results["corr_map_b"]
        corr_map_a_masked = results["corr_map_a_masked"]
        corr_map_b_masked = results["corr_map_b_masked"]

        # Flatten the masks
        current_bs = img_a.size(0)
        hidden_image_size = mask_a.shape[1:]
        mask_a = mask_a.reshape(current_bs, -1)
        mask_b = mask_b.reshape(current_bs, -1)

        # Calculate the masked correlation ious

        # Update correlation_ious
        ious = list(results["iou"].detach().cpu().numpy())
        ious_masked = list(results["iou_masked"].detach().cpu().numpy())
        self.correlation_ious.extend(ious)
        self.masked_correlation_ious.extend(ious_masked)

        # compute query features
        q = self.encoder_q(img_a)  # queries: NxCx14x14
        q = q.reshape(q.shape[0], q.shape[1], -1)  # queries: NxCx196
        q_dense = F.normalize(q, dim=1)  # normalize each pixel

        q_pos = F.normalize(torch.einsum("ncx,nx->nc", [q_dense, mask_a]), dim=1)
        cross_image_variance_source = q_pos.std(0).mean()
        q_neg = F.normalize(
            torch.einsum("ncx,nx->nc", [q_dense, (~mask_a.bool()).float()]), dim=1
        )

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder
            # shuffle for making use of BN
            img_b, idx_unshuffle = self._batch_shuffle_ddp(img_b)
            k = self.encoder_k(img_b)  # keys: NxC
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle)

            k = k.reshape(k.shape[0], k.shape[1], -1)  # keys: NxCx196
            k_dense = F.normalize(k, dim=1)  # NxCx120
            k_pos = F.normalize(torch.einsum("ncx,nx->nc", [k_dense, mask_b]), dim=1)
            cross_image_variance_target = k_pos.std(0).mean()
            k_neg = F.normalize(
                torch.einsum("ncx,nx->nc", [k_dense, (~mask_b.bool()).float()]), dim=1
            )

        # dense logits
        # pixel to pixel cosine similarities
        logits_dense = torch.einsum("ncx,ncy->nxy", [q_dense, k_dense])  # Nx196x196
        # a correspondenc map between all pixel pairs
        labels_dense = torch.einsum("nx,ny->nxy", [mask_a, mask_b])
        labels_dense = labels_dense.reshape(labels_dense.shape[0], -1)
        # all the pixels in k versus all the pixels in q
        # mask_dense = torch.einsum("x,ny->nxy", [torch.ones().cuda(), mask_b])
        # mask_dense = mask_dense.reshape(mask_dense.shape[0], -1)

        # Apply the weight mask
        assert (
            corr_weights.shape == logits_dense.shape
        ), f"{corr_weights.shape = }, {logits_dense = }"
        logits_dense *= corr_weights

        # moco logits
        l_pos = torch.einsum("nc,nc->n", [q_pos, k_pos]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q_pos, self.queue.clone().detach()])
        logits_moco = torch.cat([l_pos, l_neg], dim=1)

        if self.include_background:
            # including background pixels
            l_neg_q = torch.einsum("nc,nc->n", [q_pos, q_neg]).unsqueeze(-1)
            l_neg_k = torch.einsum("nc,nc->n", [q_pos, k_neg]).unsqueeze(-1)
            logits_moco = torch.cat([l_pos, l_neg, l_neg_q, l_neg_k], dim=1)

        labels_moco = torch.zeros(logits_moco.shape[0], dtype=torch.long).cuda()

        # apply temperature
        logits_moco /= self.T

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_pos)

        loss_instance = F.cross_entropy(logits_moco, labels_moco)

        # dense loss of softmax
        output_dense_log = (-1.0) * nn.LogSoftmax(dim=1)(logits_dense)
        output_dense_log = output_dense_log.reshape(output_dense_log.shape[0], -1)
        loss_dense = torch.mean(
            torch.mul(output_dense_log, labels_dense).sum(dim=1)
            / labels_dense.sum(dim=1)
        )

        loss = loss_instance + loss_dense * self.lmbd_cp2_dense_loss

        acc1, acc5 = self._accuracy(logits_moco, labels_moco, topk=(1, 5))
        acc_dense_pos = logits_dense.reshape(logits_dense.shape[0], -1).argmax(dim=1)
        acc_dense = (
            labels_dense[torch.arange(0, labels_dense.shape[0]), acc_dense_pos]
            .float()
            .mean()
            * 100.0
        )

        if new_epoch and self.rank == 0:
            #
            # Create the correlation map iou histogram
            #
            plt.figure(figsize=(10, 4))
            plt.hist(self.correlation_ious, bins="auto")
            plt.title("Histogram of IoU values")
            plt.xlabel("IoU")
            plt.ylabel("Frequency")
            wandb.log({"feature_space_iou": wandb.Image(plt)})
            # Number of non zero ious
            non_zero_iou_count = np.count_nonzero(self.correlation_ious)
            size = len(self.correlation_ious)
            wandb.log({"feature_space_non_zero_iou": non_zero_iou_count})
            wandb.log({"feature_space_non_zero_iou_ratio": non_zero_iou_count / size})
            self.correlation_ious = []

            #
            # Create the masked correlation map iou histogram
            #
            plt.figure(figsize=(10, 4))
            plt.hist(self.masked_correlation_ious, bins="auto")
            plt.title("Histogram of Masked IoU values")
            plt.xlabel("IoU")
            plt.ylabel("Frequency")
            wandb.log({"feature_space_masked_iou": wandb.Image(plt)})
            # Number of non zero ious
            non_zero_masked_iou_count = np.count_nonzero(self.masked_correlation_ious)
            size = len(self.masked_correlation_ious)
            wandb.log(
                {
                    "feature_space_non_zero_masked_iou": np.count_nonzero(
                        non_zero_masked_iou_count
                    )
                }
            )
            wandb.log(
                {
                    "feature_space_non_zero_masked_iou_ratio": non_zero_masked_iou_count
                    / size
                }
            )
            self.masked_correlation_ious = []

            #
            # Create the foreground similarity score heatmap
            #
            heatmaps_a = []
            heatmaps_b = []
            for i in range(len(logits_dense)):
                # get similarity scores with respect
                # order depends on how the labels are generated
                # i.e., labels_dense = torch.einsum("nx,ny->nxy", [mask_a, mask_b])
                hm_b = logits_dense[i][mask_a[i].bool(), :]
                hm_a = logits_dense[i][:, mask_b[i].bool()]

                # take the average scores
                hm_b = hm_b.sum(0) / hm_b.shape[0]
                hm_a = hm_a.sum(1) / hm_a.shape[1]
                assert hm_a.shape == hm_b.shape
                # reshape to the original image
                hm_b = hm_b.reshape(*hidden_image_size)
                hm_a = hm_a.reshape(*hidden_image_size)

                # append
                heatmaps_a.append(hm_a)
                heatmaps_b.append(hm_b)

            heatmaps_a = torch.stack(heatmaps_a).unsqueeze(1)
            heatmaps_b = torch.stack(heatmaps_b).unsqueeze(1)

            m_a = mask_a.reshape(current_bs, *hidden_image_size).unsqueeze(1)
            m_b = mask_b.reshape(current_bs, *hidden_image_size).unsqueeze(1)

            # Shift the heatmap range [0,1] to [-1,1]
            m_a = 2 * m_a - 1
            m_b = 2 * m_b - 1

            # Log the heatmaps
            log_imgs = torch.stack([m_a, heatmaps_a, m_b, heatmaps_b], dim=1).flatten(
                0, 1
            )
            log_grid = torchvision.utils.make_grid(log_imgs, nrow=4)
            # rescale the images

            scale_factor = 224 // hidden_image_size[0]
            grid_h = log_grid.shape[1]
            grid_w = log_grid.shape[2]

            resize = T.Resize(
                (grid_h * scale_factor, grid_w * scale_factor),
                T.InterpolationMode.NEAREST_EXACT,
            )
            log_grid = resize(log_grid)

            # Grab only the first channel of the grayscale image
            log_grid = log_grid.detach().cpu().numpy()[0]
            norm = matplotlib.colors.Normalize(
                vmin=log_grid.min(), vmax=log_grid.max(), clip=True
            )
            mapper = cm.ScalarMappable(norm=norm, cmap="viridis")
            log_grid = mapper.to_rgba(log_grid)  # HxWxC
            # only grab the rgb channels
            log_grid = log_grid[:, :, :3]

            wandb.log({"dense-heatmaps": wandb.Image(log_grid)})
            self.epoch += 1

        # Update logs
        self.loss_o.update(loss.item(), img_a.size(0))
        self.loss_i.update(loss_instance.item(), img_a.size(0))
        self.loss_d.update(loss_dense.item(), img_a.size(0))
        self.acc_ins.update(acc1[0], img_a.size(0))
        self.acc_seg.update(acc_dense.item(), img_a.size(0))
        self.cross_image_variance_source.update(
            cross_image_variance_source, img_a.size(0)
        )
        self.cross_image_variance_target.update(
            cross_image_variance_target, img_a.size(0)
        )

        if self.rank == 0:
            wandb.log({"train/loss_step": self.loss_o.val})

            if self.pretrain_type in [PretrainType.MOCO, PretrainType.CP2]:
                wandb.log({"train/acc_ins_step": self.acc_ins.val})

            if self.pretrain_type == PretrainType.CP2:
                wandb.log(
                    {
                        "train/loss_ins_step": self.loss_i.val,
                        "train/loss_dense_step": self.loss_d.val,
                        "train/acc_seg_step": self.acc_seg.val,
                        "train/cross_image_variance_source_step": self.cross_image_variance_source.val,
                        "train/cross_image_variance_target_step": self.cross_image_variance_target.val,
                    }
                )

        return loss

    def on_train_epoch_end(self, step):
        if self.rank == 0:
            wandb.log({"train/loss": self.loss_o.avg})

            if self.pretrain_type in [PretrainType.MOCO, PretrainType.CP2]:
                wandb.log({"train/acc_ins": self.acc_ins.avg})

            if self.pretrain_type == PretrainType.CP2:
                wandb.log(
                    {
                        "train/loss_ins": self.loss_i.avg,
                        "train/loss_dense": self.loss_d.avg,
                        "train/acc_seg": self.acc_seg.avg,
                        "train/cross_image_variance_source": self.cross_image_variance_source.avg,
                        "train/cross_image_variance_target": self.cross_image_variance_target.avg,
                    }
                )
        self.reset_metrics()

    def reset_metrics(self):
        self.loss_o.reset()
        self.loss_i.reset()
        self.loss_d.reset()
        self.acc_ins.reset()
        self.acc_seg.reset()
        self.cross_image_variance_source.reset()

    def _accuracy(self, output, target, topk=(1,)):
        """Computes the accuracy over the k top predictions for the specified values of k"""
        with torch.no_grad():
            maxk = max(topk)
            batch_size = target.size(0)

            _, pred = output.topk(maxk, 1, True, True)
            pred = pred.t()
            correct = pred.eq(target.view(1, -1).expand_as(pred))

            res = []
            for k in topk:
                correct_k = (
                    correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
                )
                res.append(correct_k.mul_(100.0 / batch_size))
            return res


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [
        torch.ones_like(tensor) for _ in range(torch.distributed.get_world_size())
    ]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
