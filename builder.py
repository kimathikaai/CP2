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
from torch.cuda import temperature
from torchmetrics.aggregation import MeanMetric

from networks.segment_network import PretrainType
from tools.correlation_mapping import (calcuate_dense_loss_stats,
                                       get_correlation_map,
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
    PIXEL_ID = 1
    REGION_ID = 2
    PIXEL_REGION_ID = 3


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


class NegativeType(Enum):
    """Determine the type of dense negative similarity post processing"""

    NONE = 0
    FIXED = 1
    AVERAGE = 2
    MEDIAN = 3
    HARD = 4


class ContrastiveHead(nn.Module):
    """
    Head for contrastive learning.
    """

    def __init__(self):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, pos, neg, temperature):
        """Forward head.

        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.
            temperature (float): The temperature hyper-parameter that
                controls the concentration level of the distribution.
                Default: 0.1.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        N = pos.size(0)
        logits = torch.cat((pos, neg), dim=1)
        logits /= temperature
        labels = torch.zeros((N,), dtype=torch.long).cuda()
        return self.criterion(logits, labels)


class DenseCLNeck(nn.Module):
    """The non-linear neck in DenseCL.
    Single and dense in parallel: fc-relu-fc, conv-relu-conv
    Reimplementing: https://github.com/WXinlong/DenseCL/blob/main/openselfsup/models/densecl.py
    """

    def __init__(self, in_channels, hid_channels, out_channels, num_grid=None):
        super(DenseCLNeck, self).__init__()

        self.avgpool_global = nn.AdaptiveAvgPool2d((1, 1))
        self.global_projector = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels),
        )
        self.global_predictor = nn.Sequential(
            nn.Linear(out_channels, hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels),
        )

        self.with_pool = num_grid != None
        if self.with_pool:
            self.pool = nn.AdaptiveAvgPool2d((num_grid, num_grid))

        self.local_projector = nn.Sequential(
            nn.Conv2d(in_channels, hid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1),
        )
        self.local_predictor = nn.Sequential(
            nn.Conv2d(out_channels, hid_channels, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(hid_channels, out_channels, 1),
        )
        self.avgpool_local = nn.AdaptiveAvgPool2d((1, 1))

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(
                m,
                (
                    nn.BatchNorm1d,
                    nn.BatchNorm2d,
                    nn.GroupNorm,
                    nn.SyncBatchNorm,
                ),
            ):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # assert len(x) == 1
        # x = x[0]

        x_global = self.avgpool_global(x)
        x_global_proj = self.global_projector(x_global.view(x_global.size(0), -1))
        x_global_pred = self.global_predictor(x_global_proj)

        if self.with_pool:
            x = self.pool(x)  # sxs

        x_local_proj = self.local_projector(x)  # sxs: bxdxsxs
        x_local_pred = self.local_predictor(x_local_proj)

        x_avgpool_local_proj = self.avgpool_local(x_local_proj)  # 1x1: bxdx1x1
        x_avgpool_local_proj = x_avgpool_local_proj.view(
            x_avgpool_local_proj.size(0), -1
        )  # bxd

        x_avgpool_local_pred = self.avgpool_local(x_local_pred)  # 1x1: bxdx1x1
        x_avgpool_local_pred = x_avgpool_local_pred.view(
            x_avgpool_local_pred.size(0), -1
        )  # bxd

        return {
            "x_global_proj": x_global_proj,
            "x_local_proj": x_local_proj,
            "x_global_pred": x_global_pred,
            "x_local_pred": x_local_pred,
            "x_avgpool_local_pred": x_avgpool_local_pred,
            "x_avgpool_local_proj": x_avgpool_local_proj,
        }


class MODEL(nn.Module):
    def __init__(
        self,
        cfg,
        rank,
        dim=128,
        K=65536,
        m=0.999,
        instance_logits_temp=0.2,
        pretrain_from_scratch=False,
        include_background=False,
        lmbd_cp2_dense_loss=0.2,
        lmbd_pixel_corr_weight=1,
        lmbd_region_corr_weight=1,
        lmbd_not_corr_weight=1,
        negative_type=NegativeType.NONE,
        negative_scale=2,
        pretrain_type=PretrainType.CP2,
        backbone_type=BackboneType.DEEPLABV3,
        mapping_type=MappingType.CP2,
        dense_logits_temp=1,
        unet_truncated_dec_blocks=2,
        use_predictor=False,
        use_avgpool_global=False,
        use_symmetrical_loss=False,
        lmbd_coordinate=0,
        device=None,
    ):
        super(MODEL, self).__init__()

        self.queue_len = K
        self.momentum = m
        self.temp_global = instance_logits_temp
        self.dim = dim
        self.include_background = include_background
        self.lmbd_dense_loss = lmbd_cp2_dense_loss
        self.device = device
        self.rank = rank
        self.epoch = 0
        self.temp_local = dense_logits_temp
        self.contrastive_head = ContrastiveHead()
        self.use_predictor = use_predictor
        self.use_avgpool_global = use_avgpool_global
        self.use_symmetrical_loss = use_symmetrical_loss

        assert lmbd_coordinate >= 0 and lmbd_coordinate <= 1, f"{lmbd_coordinate = }"
        self.lmbd_coordinate = lmbd_coordinate

        assert mapping_type in MappingType
        self.mapping_type = mapping_type

        # Validate the correlation map weight
        if mapping_type == MappingType.CP2:
            assert lmbd_pixel_corr_weight == 1
            assert lmbd_region_corr_weight == 1
            assert lmbd_not_corr_weight == 1
        elif mapping_type == MappingType.PIXEL_ID:
            assert lmbd_region_corr_weight == 1
            assert lmbd_pixel_corr_weight > 1
        elif mapping_type == MappingType.REGION_ID:
            assert lmbd_pixel_corr_weight == 1
            assert lmbd_region_corr_weight > 1
        elif mapping_type == MappingType.PIXEL_REGION_ID:
            pass
            # assert lmbd_pixel_corr_weight >= 1
            # assert lmbd_region_corr_weight >= 1
        else:
            raise NotImplementedError(f"{mapping_type = }")
        self.lmbd_pixel_corr_weight = lmbd_pixel_corr_weight
        self.lmbd_region_corr_weight = lmbd_region_corr_weight
        self.lmbd_not_corr_weight = lmbd_not_corr_weight

        assert pretrain_type in PretrainType
        self.pretrain_type = pretrain_type

        assert negative_type in NegativeType
        self.negative_type = negative_type
        self.negative_scale = negative_scale

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
            print(
                f"[INFO] Initializing with imagenet weights: {not pretrain_from_scratch}"
            )
            if not pretrain_from_scratch:
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

        # Get the backbone output stride
        test_sample = torch.rand(2, 3, 224, 224)
        output_shape = self.encoder_q.backbone(test_sample)[3].shape
        self.backbone_output_stride = int(test_sample.shape[2] / output_shape[2])
        print(f"{self.backbone_output_stride = }")

        if pretrain_type in [PretrainType.BYOL, PretrainType.MOCO]:
            # Projection/prediction networks
            # backbone_features = 2048 * 7 * 7  # if imgs are 224x224
            backbone_features = (
                2048 * self.backbone_output_stride**2
            )  # if imgs are 224x224
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

        elif pretrain_type == PretrainType.CP2:
            assert self.negative_type == NegativeType.NONE
            assert self.mapping_type == MappingType.CP2

        elif pretrain_type == PretrainType.DENSECL:
            self.encoder_q.neck = DenseCLNeck(
                in_channels=2048, hid_channels=2048, out_channels=self.dim
            )
            self.encoder_k.neck = DenseCLNeck(
                in_channels=2048, hid_channels=2048, out_channels=self.dim
            )
            # Parameters
            assert self.momentum == 0.999, f"{self.momentum = }"
            assert self.lmbd_dense_loss == 0.5, f"{self.lmbd_dense_loss = }"
            assert self.temp_global == 0.2, f"{self.temp_global = }"
            assert self.temp_local == 0.2, f"{self.temp_local = }"
            assert self.use_predictor == False
            assert self.use_avgpool_global == False
            assert self.use_symmetrical_loss == False
            assert self.lmbd_coordinate == 0
        elif pretrain_type == PretrainType.PROPOSED_V2:
            self.encoder_q.neck = DenseCLNeck(
                in_channels=2048, hid_channels=2048, out_channels=self.dim
            )
            self.encoder_k.neck = DenseCLNeck(
                in_channels=2048, hid_channels=2048, out_channels=self.dim
            )
            # Parameters
            assert self.momentum == 0.999, f"{self.momentum = }"
            assert self.lmbd_dense_loss == 0.5, f"{self.lmbd_dense_loss = }"
            assert self.temp_global == 0.2, f"{self.temp_global = }"
            assert self.temp_local == 0.2, f"{self.temp_local = }"

        # Exact copy parameters
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # confirm config
        # if self.pretrain_type in [PretrainType.MOCO, PretrainType.BYOL]:
        #     assert isinstance(self.encoder_q.decode_head, FCNHead)

        # create the global image queue
        self.register_buffer("queue", torch.randn(self.dim, self.queue_len))
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        # create the local averaged pooled queue
        self.register_buffer("queue2", torch.randn(self.dim, self.queue_len))
        self.queue2 = F.normalize(self.queue2, dim=0)
        self.register_buffer("queue2_ptr", torch.zeros(1, dtype=torch.long))

        #
        # Metrics
        #

        # contrastive
        self.dense_per_sample_average_positive_scores = MeanMetric()
        self.dense_per_sample_lower_positive_scores = MeanMetric()
        self.dense_per_sample_median_positive_scores = MeanMetric()
        self.dense_per_sample_upper_positive_scores = MeanMetric()

        self.dense_per_sample_average_negative_scores = MeanMetric()
        self.dense_per_sample_lower_negative_scores = MeanMetric()
        self.dense_per_sample_median_negative_scores = MeanMetric()
        self.dense_per_sample_upper_negative_scores = MeanMetric()

        if self.rank == 0:
            # fmt:off
            wandb.define_metric("step/dense_per_sample_average_positive_scores", summary='last')
            wandb.define_metric("step/dense_per_sample_lower_positive_scores", summary='last')
            wandb.define_metric("step/dense_per_sample_median_positive_scores", summary='last')
            wandb.define_metric("step/dense_per_sample_upper_positive_scores", summary='last')
            wandb.define_metric("step/dense_per_sample_average_negative_scores", summary='last')
            wandb.define_metric("step/dense_per_sample_lower_negative_scores", summary='last')
            wandb.define_metric("step/dense_per_sample_median_negative_scores", summary='last')
            wandb.define_metric("step/dense_per_sample_upper_negative_scores", summary='last')

            wandb.define_metric("dense_per_sample_average_positive_scores", summary='last')
            wandb.define_metric("dense_per_sample_lower_positive_scores", summary='last')
            wandb.define_metric("dense_per_sample_median_positive_scores", summary='last')
            wandb.define_metric("dense_per_sample_upper_positive_scores", summary='last')
            wandb.define_metric("dense_per_sample_average_negative_scores", summary='last')
            wandb.define_metric("dense_per_sample_lower_negative_scores", summary='last')
            wandb.define_metric("dense_per_sample_median_negative_scores", summary='last')
            wandb.define_metric("dense_per_sample_upper_negative_scores", summary='last')
            # fmt:on

        #
        # Instance level stats
        #

        self.instance_average_positive_scores = MeanMetric()
        self.instance_average_negative_scores = MeanMetric()
        self.instance_lower_negative_scores = MeanMetric()
        self.instance_median_negative_scores = MeanMetric()
        self.instance_upper_negative_scores = MeanMetric()

        if self.rank == 0:
            wandb.define_metric("step/instance_average_positive_scores", summary="last")
            wandb.define_metric("step/instance_average_negative_scores", summary="last")
            wandb.define_metric("step/instance_lower_negative_scores", summary="last")
            wandb.define_metric("step/instance_median_negative_scores", summary="last")
            wandb.define_metric("step/instance_upper_negative_scores", summary="last")

            wandb.define_metric("instance_average_positive_scores", summary="last")
            wandb.define_metric("instance_average_negative_scores", summary="last")
            wandb.define_metric("instance_lower_negative_scores", summary="last")
            wandb.define_metric("instance_median_negative_scores", summary="last")
            wandb.define_metric("instance_upper_negative_scores", summary="last")

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
            param_k.data = param_k.data * self.momentum + param_q.data * (
                1.0 - self.momentum
            )

    @torch.no_grad()
    def _dequeue_and_enqueue(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue_ptr)

        if ptr + batch_size > self.queue_len:
            self.queue[:, ptr : self.queue_len] = keys[0 : self.queue_len - ptr].T
            self.queue[:, 0 : ptr + batch_size - self.queue_len] = keys[
                self.queue_len - ptr : batch_size
            ].T
        else:
            self.queue[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue_ptr[0] = ptr

    @torch.no_grad()
    def _dequeue_and_enqueue2(self, keys):
        # gather keys before updating queue
        keys = concat_all_gather(keys)

        batch_size = keys.shape[0]

        ptr = int(self.queue2_ptr)

        if ptr + batch_size > self.queue_len:
            self.queue2[:, ptr : self.queue_len] = keys[0 : self.queue_len - ptr].T
            self.queue2[:, 0 : ptr + batch_size - self.queue_len] = keys[
                self.queue_len - ptr : batch_size
            ].T
        else:
            self.queue2[:, ptr : ptr + batch_size] = keys.T
        ptr = (ptr + batch_size) % self.queue_len  # move pointer

        self.queue2_ptr[0] = ptr

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
        elif self.pretrain_type == PretrainType.PROPOSED:
            return self.forward_cp2(**kwargs)
        elif self.pretrain_type == PretrainType.DENSECL:
            return self.forward_densecl(**kwargs)
        elif self.pretrain_type == PretrainType.PROPOSED_V2:
            return self.forward_densecl(**kwargs)
        else:
            raise NotImplementedError(f"{self.pretrain_type = }")

    def forward_densecl(
        self,
        img_a,
        img_b,
        bg0,
        bg1,
        visualize,
        step,
        new_epoch,
        pixel_ids_a,
        pixel_ids_b,
        region_ids_a,
        region_ids_b,
    ):
        """
        Reimplementing: https://github.com/WXinlong/DenseCL/blob/main/openselfsup/models/densecl.py
        """
        batch_size = len(img_a)

        # Visualize the data
        if visualize and self.rank == 0:
            log_imgs = torch.stack(
                [img_a[0 : batch_size // 2], img_b[0 : batch_size // 2]], dim=1
            ).flatten(0, 1)
            log_grid = torchvision.utils.make_grid(log_imgs, nrow=2)
            wandb.log(
                {
                    "train-examples": wandb.Image(
                        log_grid, caption=self.pretrain_type.name
                    )
                }
            )

        def get_query_features(img):
            # compute query features
            embd_q = self.encoder_q.backbone(img)[3]
            _q = self.encoder_q.neck(embd_q)
            q_local = _q["x_local_pred"] if self.use_predictor else _q["x_local_proj"]
            q_global = (
                _q["x_global_pred"] if self.use_predictor else _q["x_global_proj"]
            )
            if self.use_avgpool_global:
                q_global = (
                    _q["x_avgpool_local_pred"]
                    if self.use_predictor
                    else _q["x_avgpool_local_proj"]
                )
            # normalize query features
            q_local = F.normalize(
                q_local.view(q_local.size(0), q_local.size(1), -1), dim=1
            )  # NxS^2
            q_global = F.normalize(q_global, dim=1)
            embd_q = F.normalize(embd_q.view(embd_q.size(0), embd_q.size(1), -1), dim=1)

            return embd_q, q_local, q_global

        def get_key_features(img):
            # compute key features
            with torch.no_grad():  # no gradient to keys
                self._momentum_update_key_encoder()  # update the key encoder

                # shuffle for making use of BN
                img, idx_unshuffle = self._batch_shuffle_ddp(img)

                # compute query features
                embd_k = self.encoder_k.backbone(img)[3]
                _k = self.encoder_k.neck(embd_k)
                k_local = _k["x_local_proj"]
                k_local_proj_pooled = _k["x_avgpool_local_proj"]
                k_global = _k["x_global_proj"]
                if self.use_avgpool_global:
                    k_global = k_local_proj_pooled

                # normalize query features
                k_local = F.normalize(
                    k_local.view(k_local.size(0), k_local.size(1), -1), dim=1
                )  # NxS^2
                k_local_proj_pooled = F.normalize(k_local_proj_pooled, dim=1)
                k_global = F.normalize(k_global, dim=1)
                embd_k = F.normalize(
                    embd_k.view(embd_k.size(0), embd_k.size(1), -1), dim=1
                )

                # undo shuffle
                k_local = self._batch_unshuffle_ddp(k_local, idx_unshuffle)
                k_local_proj_pooled = self._batch_unshuffle_ddp(
                    k_local_proj_pooled, idx_unshuffle
                )
                k_global = self._batch_unshuffle_ddp(k_global, idx_unshuffle)
                embd_k = self._batch_unshuffle_ddp(embd_k, idx_unshuffle)

                return embd_k, k_local, k_global, k_local_proj_pooled

        def compute_global_loss(q, k, log_metrics=False):
            # compute global similarities
            pos_global = torch.einsum(
                "nc,nc->n",
                [q, k],
            ).unsqueeze(-1)
            neg_global = torch.einsum(
                "nc,ck->nk",
                [q, self.queue.clone().detach()],
            )
            loss_global = self.contrastive_head(
                pos=pos_global, neg=neg_global, temperature=self.temp_global
            )

            # metrics
            if log_metrics and self.rank == 0:
                # Compute variance metrics
                cross_image_variance_source = q.std(0).mean()
                cross_image_variance_target = k.std(0).mean()

                # Compute score metrics
                instance_average_positive_scores = pos_global
                instance_average_negative_scores = neg_global.mean(1)
                instance_negative_quartiles = torch.quantile(
                    neg_global,
                    q=torch.Tensor([0.25, 0.5, 0.75]).to(neg_global.device),
                    dim=1,
                )
                instance_lower_negative_scores = instance_negative_quartiles[0]
                instance_median_negative_scores = instance_negative_quartiles[1]
                instance_upper_negative_scores = instance_negative_quartiles[2]

                # fmt:off
                wandb.log(
                    {
                        "step/instance_average_positive_scores": instance_average_positive_scores.mean().item(),
                        "step/instance_average_negative_scores": instance_average_negative_scores.mean().item(),
                        "step/instance_lower_negative_scores": instance_lower_negative_scores.mean().item(),
                        "step/instance_median_negative_scores": instance_median_negative_scores.mean().item(),
                        "step/instance_upper_negative_scores": instance_upper_negative_scores.mean().item(),
                        "step/cross_image_variance_source_step": cross_image_variance_source.item(),
                        "step/cross_image_variance_target_step": cross_image_variance_target.item()
                    }
                )
                # fmt:on

            return loss_global

        def compute_local_loss(
            q_embed,
            k_embed,
            q_local,
            k_local,
            q_pixel_ids,
            k_pixel_ids,
            log_metrics=False,
        ):
            # local backbone feature similarity
            backbone_sim_matrix = torch.einsum(
                "ncx,ncy->nxy", [q_embed, k_embed]
            )  # NxS^2xS^2
            pos_global_k_idx = backbone_sim_matrix.max(dim=2)[1]  # NxS^2

            # local projection head feature similarity
            local_sim_matrix = torch.einsum(
                "ncx,ncy->nxy", [q_local, k_local]
            )  # NxS^2xS^2

            # identify top positive scores using similarity scores
            pos_local = torch.gather(
                local_sim_matrix,
                2,
                pos_global_k_idx.unsqueeze(2),
            ).squeeze(
                -1
            )  # NxS^2
            assert len(pos_local.shape) == 2, f"{pos_local.shape = }"

            # identify the coordinate correspondences
            _correlations = get_correlation_map(q_pixel_ids, k_pixel_ids)
            iou = _correlations["iou"]  # size N
            corr_map = _correlations["corr_map"].detach()  # NxS^2xS^2

            # get pixels in overlapping regions
            overlap_pixels = corr_map.sum(-1) > 0
            # mask out scores for non_overlapping regions
            overlap_scores = local_sim_matrix * corr_map  # NxS^2xS^2
            # get scores for pixels in overlapping regions
            coord_overlap_scores = overlap_scores.sum(-1)[overlap_pixels]  # NxK
            # find the scores (using sim not coord) that are in overlapping regions
            pos_local_overlap = pos_local[overlap_pixels]  # NxK
            # get the mix of the sim based and coord based scores
            pos_local[overlap_pixels] = (
                pos_local_overlap * (1 - self.lmbd_coordinate)
                + coord_overlap_scores * self.lmbd_coordinate
            )

            # Calculate how often the max is the same as ground truth
            matching_positives_rate = -1
            if corr_map.sum() > 0:
                # find the overlapping pixels
                corr_max = corr_map[overlap_pixels, :].max(dim=2)[1].flatten()
                sim_max = local_sim_matrix[overlap_pixels, :].max(dim=2)[1].flatten()
                assert len(corr_max) == len(sim_max), f"{corr_max.shape = }, {sim_max.shape = }"
                matching_positives_rate = (corr_max == sim_max).float().mean().item()

            # Move pixels to batch dimension
            q_local = q_local.permute(0, 2, 1)  # NxS^2xC
            q_local = q_local.reshape(-1, q_local.size(2))  # NS^2xC
            pos_local = pos_local.view(-1).unsqueeze(-1)  # NS^2x1

            neg_local = torch.einsum(
                "nc,ck->nk", [q_local, self.queue2.clone().detach()]
            )

            if log_metrics and self.rank == 0:
                dense_average_positive_scores = pos_local
                dense_average_negative_scores = neg_local.mean(1)
                dense_negative_quartiles = torch.quantile(
                    neg_local,
                    q=torch.Tensor([0.25, 0.5, 0.75]).to(neg_local.device),
                    dim=1,
                )
                dense_lower_negative_scores = dense_negative_quartiles[0]
                dense_median_negative_scores = dense_negative_quartiles[1]
                dense_upper_negative_scores = dense_negative_quartiles[2]

                # calculate the number of non zero ious
                non_zero_iou = torch.count_nonzero(iou) / len(iou)

                # fmt:off
                wandb.log(
                    {
                        "step/dense_average_positive_scores": dense_average_positive_scores.mean().item(),
                        "step/dense_average_negative_scores": dense_average_negative_scores.mean().item(),
                        "step/dense_lower_negative_scores": dense_lower_negative_scores.mean().item(),
                        "step/dense_median_negative_scores": dense_median_negative_scores.mean().item(),
                        "step/dense_upper_negative_scores": dense_upper_negative_scores.mean().item(),
                        "step/average_iou": iou.mean().item(),
                        "step/non_zero_iou_ratio": non_zero_iou.item(),
                        "step/matching_positives_rate": matching_positives_rate
                    }
                )
                # fmt:on

            # losses
            loss_local = self.contrastive_head(
                pos=pos_local, neg=neg_local, temperature=self.temp_local
            )

            return loss_local

        # Downsample the pixel ids
        pixel_ids_a = pixel_ids_a[
            :,
            self.backbone_output_stride // 2 :: self.backbone_output_stride,
            self.backbone_output_stride // 2 :: self.backbone_output_stride,
        ]
        pixel_ids_b = pixel_ids_b[
            :,
            self.backbone_output_stride // 2 :: self.backbone_output_stride,
            self.backbone_output_stride // 2 :: self.backbone_output_stride,
        ]

        embd_q_1, q_local_1, q_global_1 = get_query_features(img_a)
        embd_k_1, k_local_1, k_global_1, k_local_proj_pooled_1 = get_key_features(img_b)
        loss_global = compute_global_loss(
            q=q_global_1,
            k=k_global_1,
            log_metrics=True,
        )
        loss_local = compute_local_loss(
            q_embed=embd_q_1,
            k_embed=embd_k_1,
            q_local=q_local_1,
            k_local=k_local_1,
            q_pixel_ids=pixel_ids_a,
            k_pixel_ids=pixel_ids_b,
            log_metrics=True,
        )

        # features for updating the momentum queue
        to_update_queues = {"global": k_global_1, "local": k_local_proj_pooled_1}

        if self.use_symmetrical_loss:
            embd_q_2, q_local_2, q_global_2 = get_query_features(img_b)
            embd_k_2, k_local_2, k_global_2, k_local_proj_pooled_2 = get_key_features(
                img_a
            )
            loss_global_2 = compute_global_loss(
                q=q_global_2,
                k=k_global_2,
            )
            loss_local_2 = compute_local_loss(
                q_embed=embd_q_2,
                k_embed=embd_k_2,
                q_local=q_local_2,
                k_local=k_local_2,
                q_pixel_ids=pixel_ids_b,
                k_pixel_ids=pixel_ids_a,
            )

            # combine losses
            loss_global = (loss_global + loss_global_2).mean()
            loss_local = (loss_local + loss_local_2).mean()

            # features for updating the momentum queue
            # inspired from https://github.com/megvii-research/revisitAIRL/blob/7afeb0299d354de873808cf6ed7848924e72c1d9/exp/mocov2_plus.py#L57
            if step % 2 == 0:
                to_update_queues = {
                    "global": k_global_2,
                    "local": k_local_proj_pooled_2,
                }

        # Loss
        loss = (
            1 - self.lmbd_dense_loss
        ) * loss_global + self.lmbd_dense_loss * loss_local

        # Update momentum queue
        self._dequeue_and_enqueue(to_update_queues["global"])
        self._dequeue_and_enqueue2(to_update_queues["local"])

        # Update logs
        self.loss_o.update(loss.item(), batch_size)
        self.loss_i.update(loss_global.item(), batch_size)
        self.loss_d.update(loss_local.item(), batch_size)

        if self.rank == 0:
            # fmt:off
            wandb.log(
                {
                    "train/loss_step": self.loss_o.val,
                    "train/loss_ins_step": self.loss_i.val,
                    "train/loss_dense_step": self.loss_d.val,
                }
            )
            # fmt:on

        return loss

    def forward_moco(self, img_a, img_b, bg0, bg1, visualize, step, new_epoch):
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

        loss = F.cross_entropy(logits_moco/self.temp_global, labels_moco).mean()

        acc1, acc5 = self._accuracy(logits_moco, labels_moco, topk=(1, 5))

        # Update logs
        self.loss_o.update(loss.item(), img_a.size(0))
        self.acc_ins.update(acc1[0], img_a.size(0))

        instance_average_positive_scores = l_pos
        instance_average_negative_scores = l_neg.mean(1)
        instance_negative_quartiles = torch.quantile(
            l_neg, q=torch.Tensor([0.25, 0.5, 0.75]).to(l_neg.device), dim=1
        )
        instance_lower_negative_scores = instance_negative_quartiles[0]
        instance_median_negative_scores = instance_negative_quartiles[1]
        instance_upper_negative_scores = instance_negative_quartiles[2]

        # negative_threshold = torch.quantile(l_neg, q = 0.75, dim = 1)
        # hard_negative_mask = l_neg > negative_threshold.unsqueeze(1)
        # mask_negatives = l_neg.clone()
        # mask_negatives[hard_negative_mask] = 0 # Set values above 75th percentile to 0

        if self.rank == 0:
            # fmt:off
            wandb.log(
                {
                    "train/loss_step": self.loss_o.val,
                    "train/acc_ins_step": self.acc_ins.val,
                    "step/instance_average_positive_scores": instance_average_positive_scores.mean().item(),
                    "step/instance_average_negative_scores": instance_average_negative_scores.mean().item(),
                    "step/instance_lower_negative_scores": instance_lower_negative_scores.mean().item(),
                    "step/instance_median_negative_scores": instance_median_negative_scores.mean().item(),
                    "step/instance_upper_negative_scores": instance_upper_negative_scores.mean().item(),
                    # "step/masked_negative_scores": mask_negatives.mean().item()
                }
            )
            # fmt:on


        return loss

    def forward_byol(self, img_a, img_b, bg0, bg1, visualize, step, new_epoch):
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
        self,
        img_a,
        img_b,
        bg0,
        bg1,
        visualize,
        step,
        new_epoch,
        pixel_ids_a,
        pixel_ids_b,
        region_ids_a,
        region_ids_b,
    ):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        batch_size = img_a.size(0)
        mask_a, mask_b = (bg0[:, 0] == 0).float(), (bg1[:, 0] == 0).float()
        mask_a.to(bg0.device)
        mask_b.to(bg1.device)
        _img_a = img_a.clone()
        _img_b = img_b.clone()
        img_a = img_a * mask_a.unsqueeze(1) + bg0
        img_b = img_b * mask_b.unsqueeze(1) + bg1

        # Update A
        mask_a = mask_a[
            :,
            self.output_stride // 2 :: self.output_stride,
            self.output_stride // 2 :: self.output_stride,
        ]
        pixel_ids_a = pixel_ids_a[
            :,
            self.output_stride // 2 :: self.output_stride,
            self.output_stride // 2 :: self.output_stride,
        ]
        region_ids_a = region_ids_a[
            :,
            self.output_stride // 2 :: self.output_stride,
            self.output_stride // 2 :: self.output_stride,
        ]

        # Update B
        mask_b = mask_b[
            :,
            self.output_stride // 2 :: self.output_stride,
            self.output_stride // 2 :: self.output_stride,
        ]
        pixel_ids_b = pixel_ids_b[
            :,
            self.output_stride // 2 :: self.output_stride,
            self.output_stride // 2 :: self.output_stride,
        ]
        region_ids_b = region_ids_b[
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
        pixel_corr_results = get_masked_correlation_map(
            map_a=pixel_ids_a,
            map_b=pixel_ids_b,
            mask_a=mask_a,
            mask_b=mask_b,
        )
        pixel_corr_map = pixel_corr_results["corr_map"]
        #
        # Generation the region correspondence map
        region_corr_results = get_masked_correlation_map(
            map_a=region_ids_a,
            map_b=region_ids_b,
            mask_a=mask_a,
            mask_b=mask_b,
        )
        region_corr_map = region_corr_results["corr_map"]

        # If using pre-generated ids based on unsupervised region proposals
        # we don't want to correlated pixels that have the 0 id
        # since it denotes unkown regions. Therefore we need to remove them
        # from the corr_map
        known_region_ids = torch.einsum(
            "nx,ny->nxy",
            [
                region_ids_a.reshape(batch_size, -1),
                region_ids_b.reshape(batch_size, -1),
            ],
        ).bool()
        region_corr_map *= known_region_ids

        #
        # Get the correspondence weights
        #
        # Calculate the region level values
        corr_weights = self.lmbd_region_corr_weight * region_corr_map
        # Then apply the pixel level values
        corr_weights[torch.where(pixel_corr_map)] = self.lmbd_pixel_corr_weight
        # Then apple the other pixel values
        # corr_weights = (~pixel_corr_map * self.lmbd_not_corr_weight).detach()
        corr_weights += (~(corr_weights.bool()) * self.lmbd_not_corr_weight).detach()
        # pix = 10, region = 1, not=0 | pix = 10, region=0, not=0 | pix=10, region=0, not=1

        # Flatten the masks
        hidden_image_size = mask_a.shape[1:]
        mask_a = mask_a.reshape(batch_size, -1)
        mask_b = mask_b.reshape(batch_size, -1)

        # Calculate the masked correlation ious

        # Update correlation_ious
        ious = list(region_corr_results["iou"].detach().cpu().numpy())
        ious_masked = list(region_corr_results["iou_masked"].detach().cpu().numpy())
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
        _logits_dense = torch.einsum("ncx,ncy->nxy", [q_dense, k_dense])  # Nx196x196
        # a correspondenc map between all pixel pairs
        _labels_dense = torch.einsum("nx,ny->nxy", [mask_a, mask_b])
        labels_dense = _labels_dense.reshape(_labels_dense.shape[0], -1)
        # all the pixels in k versus all the pixels in q
        # mask_dense = torch.einsum("x,ny->nxy", [torch.ones().cuda(), mask_b])
        # mask_dense = mask_dense.reshape(mask_dense.shape[0], -1)

        # Other stats
        contrast_stats = calcuate_dense_loss_stats(_logits_dense, _labels_dense)
        positive_scores_average = contrast_stats["positive"]["average"]
        negative_scores_average = contrast_stats["negative"]["average"]

        #
        # Update dense stats
        #
        # self.dense_per_sample_average_positive_scores.update(
        #     contrast_stats["positive"]["average"]
        # )
        # self.dense_per_sample_lower_positive_scores.update(
        #     contrast_stats["positive"]["quartiles"][0]
        # )
        # self.dense_per_sample_median_positive_scores.update(
        #     contrast_stats["positive"]["quartiles"][1]
        # )
        # self.dense_per_sample_upper_positive_scores.update(
        #     contrast_stats["positive"]["quartiles"][2]
        # )
        #
        # self.dense_per_sample_average_negative_scores.update(
        #     contrast_stats["negative"]["average"]
        # )
        # self.dense_per_sample_lower_negative_scores.update(
        #     contrast_stats["negative"]["quartiles"][0]
        # )
        # self.dense_per_sample_median_negative_scores.update(
        #     contrast_stats["negative"]["quartiles"][1]
        # )
        # self.dense_per_sample_upper_negative_scores.update(
        #     contrast_stats["negative"]["quartiles"][2]
        # )

        # Don't focus on negatives that are too similar or already pretty far
        if self.negative_type == NegativeType.FIXED:
            negative_scores = torch.where(~(_labels_dense.bool()))
            _logits_dense[negative_scores] = (
                2
                / (1 + torch.exp(_logits_dense[negative_scores] * -self.negative_scale))
                - 1
            )
        # Shift the center based on the average score
        elif self.negative_type == NegativeType.AVERAGE:
            negative_scores = torch.where(~(_labels_dense.bool()))
            _logits_dense[negative_scores] = (
                2
                / (
                    1
                    + torch.exp(
                        (
                            _logits_dense
                            - negative_scores_average.detach().reshape(-1, 1, 1)
                        )[negative_scores]
                        * -self.negative_scale
                    )
                )
                - 1
            )
        elif self.negative_type == NegativeType.MEDIAN:
            negative_scores = torch.where(~(_labels_dense.bool()))
            _logits_dense[negative_scores] = (
                2
                / (
                    1
                    + torch.exp(
                        (
                            _logits_dense
                            - contrast_stats["negative"]["quartiles"][1]
                            .detach()
                            .reshape(-1, 1, 1)
                        )[negative_scores]
                        * -self.negative_scale
                    )
                )
                - 1
            )

        elif self.negative_type == NegativeType.HARD:
            # _logits_dense -> N x 196 x 196
            negatives = _logits_dense[~(_labels_dense.bool())] # Select only the negatives
            third_quartile = torch.quantile(negatives, q = 0.75)
            hard_negative_mask = negatives > third_quartile

            _logits_dense[~(_labels_dense.bool())][hard_negative_mask] *= 1.5

        elif self.negative_type == NegativeType.NONE:
            pass
        else:
            raise NotImplemented(f"{self.negative_type = }")

        # Apply the weight mask
        assert (
            corr_weights.shape == _logits_dense.shape
        ), f"{corr_weights.shape = }, {_logits_dense.shape = }"
        logits_dense = _logits_dense * corr_weights

        # moco logits
        l_pos = torch.einsum("nc,nc->n", [q_pos, k_pos]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q_pos, self.queue.clone().detach()])
        logits_moco = torch.cat([l_pos, l_neg], dim=1)

        instance_average_positive_scores = l_pos
        instance_average_negative_scores = l_neg.mean(1)
        instance_negative_quartiles = torch.quantile(
            l_neg, q=torch.Tensor([0.25, 0.5, 0.75]).to(l_neg.device), dim=1
        )
        instance_lower_negative_scores = instance_negative_quartiles[0]
        instance_median_negative_scores = instance_negative_quartiles[1]
        instance_upper_negative_scores = instance_negative_quartiles[2]

        # self.instance_average_positive_scores.update(instance_average_positive_scores)
        # self.instance_average_negative_scores.update(instance_average_negative_scores)
        # self.instance_lower_negative_scores.update(instance_lower_negative_scores)
        # self.instance_median_negative_scores.update(instance_median_negative_scores)
        # self.instance_upper_negative_scores.update(instance_upper_negative_scores)

        if self.include_background:
            # including background pixels
            l_neg_q = torch.einsum("nc,nc->n", [q_pos, q_neg]).unsqueeze(-1)
            l_neg_k = torch.einsum("nc,nc->n", [q_pos, k_neg]).unsqueeze(-1)
            logits_moco = torch.cat([l_pos, l_neg, l_neg_q, l_neg_k], dim=1)

        labels_moco = torch.zeros(logits_moco.shape[0], dtype=torch.long).cuda()

        # apply temperature
        logits_moco /= self.temp_global

        # dequeue and enqueue
        self._dequeue_and_enqueue(k_pos)

        loss_instance = F.cross_entropy(logits_moco, labels_moco)

        # dense loss of softmax
        logits_dense /= self.temp_local
        output_dense_log = (-1.0) * nn.LogSoftmax(dim=1)(logits_dense)
        output_dense_log = output_dense_log.reshape(output_dense_log.shape[0], -1)
        loss_dense = torch.mean(
            torch.mul(output_dense_log, labels_dense).sum(dim=1)
            / labels_dense.sum(dim=1)
        )

        loss = loss_instance + loss_dense * self.lmbd_dense_loss

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
            wandb.log({"feature_space_non_zero_masked_iou": non_zero_masked_iou_count})
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

            m_a = mask_a.reshape(batch_size, *hidden_image_size).unsqueeze(1)
            m_b = mask_b.reshape(batch_size, *hidden_image_size).unsqueeze(1)

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

            if self.pretrain_type in [
                PretrainType.MOCO,
                PretrainType.CP2,
                PretrainType.PROPOSED,
            ]:
                wandb.log({"train/acc_ins_step": self.acc_ins.val})

            if (
                self.pretrain_type == PretrainType.CP2
                or self.pretrain_type == PretrainType.PROPOSED
            ):
                # fmt:off
                wandb.log(
                    {
                        "train/loss_ins_step": self.loss_i.val,
                        "train/loss_dense_step": self.loss_d.val,
                        "train/acc_seg_step": self.acc_seg.val,
                        "train/cross_image_variance_source_step": self.cross_image_variance_source.val,
                        "train/cross_image_variance_target_step": self.cross_image_variance_target.val,
                        "train/+ive_scores_step": positive_scores_average.mean(),
                        "train/-ive_scores_step": negative_scores_average.mean(),
                        "step/dense_per_sample_average_positive_scores": contrast_stats["positive"]["average"].mean().item(),
                        "step/dense_per_sample_lower_positive_scores": contrast_stats["positive"]["quartiles"][0].mean().item(),
                        "step/dense_per_sample_median_positive_scores": contrast_stats["positive"]["quartiles"][1].mean().item(),
                        "step/dense_per_sample_upper_positive_scores": contrast_stats["positive"]["quartiles"][2].mean().item(),
                        "step/dense_per_sample_average_negative_scores": contrast_stats["negative"]["average"].mean().item(),
                        "step/dense_per_sample_lower_negative_scores": contrast_stats["negative"]["quartiles"][0].mean().item(),
                        "step/dense_per_sample_median_negative_scores": contrast_stats["negative"]["quartiles"][1].mean().item(),
                        "step/dense_per_sample_upper_negative_scores": contrast_stats["negative"]["quartiles"][2].mean().item(),
                        "step/instance_average_positive_scores": instance_average_positive_scores.mean().item(),
                        "step/instance_average_negative_scores": instance_average_negative_scores.mean().item(),
                        "step/instance_lower_negative_scores": instance_lower_negative_scores.mean().item(),
                        "step/instance_median_negative_scores": instance_median_negative_scores.mean().item(),
                        "step/instance_upper_negative_scores": instance_upper_negative_scores.mean().item(),
                    }
                )
                # fmt:on

        return loss

    def on_train_epoch_end(self, step):
        if self.rank == 0:
            wandb.log({"train/loss": self.loss_o.avg})

            if self.pretrain_type in [
                PretrainType.MOCO,
                PretrainType.CP2,
                PretrainType.PROPOSED,
            ]:
                wandb.log({"train/acc_ins": self.acc_ins.avg})

            if self.pretrain_type == PretrainType.DENSECL:
                # fmt:off
                wandb.log(
                    {
                        "train/loss_ins": self.loss_i.avg,
                        "train/loss_dense": self.loss_d.avg
                    })
                # fmt:on

            if self.pretrain_type == PretrainType.PROPOSED_V2:
                # fmt:off
                wandb.log(
                    {
                        "train/loss_ins": self.loss_i.avg,
                        "train/loss_dense": self.loss_d.avg,
                        "train/cross_image_variance_source": self.cross_image_variance_source.avg,
                        "train/cross_image_variance_target": self.cross_image_variance_target.avg,
                    })
                # fmt:on

            if self.pretrain_type == PretrainType.CP2:
                # fmt:off
                wandb.log(
                    {
                        "train/loss_ins": self.loss_i.avg,
                        "train/loss_dense": self.loss_d.avg,
                        "train/acc_seg": self.acc_seg.avg,
                        "train/cross_image_variance_source": self.cross_image_variance_source.avg,
                        "train/cross_image_variance_target": self.cross_image_variance_target.avg,
                        # "dense_per_sample_average_positive_scores": self.dense_per_sample_average_positive_scores.compute(),
                        # "dense_per_sample_lower_positive_scores": self.dense_per_sample_lower_positive_scores.compute(),
                        # "dense_per_sample_median_positive_scores": self.dense_per_sample_median_positive_scores.compute(),
                        # "dense_per_sample_upper_positive_scores": self.dense_per_sample_upper_positive_scores.compute(),
                        # "dense_per_sample_average_negative_scores": self.dense_per_sample_average_negative_scores.compute(),
                        # "dense_per_sample_lower_negative_scores": self.dense_per_sample_lower_negative_scores.compute(),
                        # "dense_per_sample_median_negative_scores": self.dense_per_sample_median_negative_scores.compute(),
                        # "dense_per_sample_upper_negative_scores": self.dense_per_sample_upper_negative_scores.compute(),
                        # "instance_average_positive_scores": self.instance_average_positive_scores.compute(),
                        # "instance_average_negative_scores": self.instance_average_negative_scores.compute(),
                        # "instance_lower_negative_scores": self.instance_lower_negative_scores.compute(),
                        # "instance_median_negative_scores": self.instance_median_negative_scores.compute(),
                        # "instance_upper_negative_scores": self.instance_upper_negative_scores.compute(),
                    }
                )
                # fmt:on
        self.reset_metrics()

    def reset_metrics(self):
        self.loss_o.reset()
        self.loss_i.reset()
        self.loss_d.reset()
        self.acc_ins.reset()
        self.acc_seg.reset()
        self.cross_image_variance_source.reset()
        self.cross_image_variance_target.reset()

        # self.dense_per_sample_average_positive_scores.reset()
        # self.dense_per_sample_lower_positive_scores.reset()
        # self.dense_per_sample_median_positive_scores.reset()
        # self.dense_per_sample_upper_positive_scores.reset()
        # self.dense_per_sample_average_negative_scores.reset()
        # self.dense_per_sample_lower_negative_scores.reset()
        # self.dense_per_sample_median_negative_scores.reset()
        # self.dense_per_sample_upper_negative_scores.reset()
        #
        # self.instance_average_positive_scores.reset()
        # self.instance_average_negative_scores.reset()
        # self.instance_lower_negative_scores.reset()
        # self.instance_median_negative_scores.reset()
        # self.instance_upper_negative_scores.reset()

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
