# The CP2_MoCo model is built upon moco v2 code base:
# https://github.com/facebookresearch/moco
# Copyright (c) Facebook, Inc. and its affilates. All Rights Reserved
import copy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import wandb
from mmseg.models import build_segmentor
from mmseg.models.decode_heads import FCNHead

from networks.segment_network import PretrainType


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
        pretrain_type=PretrainType.CP2,
        device=None,
    ):
        super(CP2_MOCO, self).__init__()

        self.K = K
        self.m = m
        self.T = T
        self.include_background = include_background
        self.lmbd_cp2_dense_loss = lmbd_cp2_dense_loss
        self.device = device
        self.rank = rank

        assert pretrain_type in PretrainType
        self.pretrain_type = pretrain_type

        self.encoder_q = build_segmentor(
            cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
        )
        self.encoder_k = build_segmentor(
            cfg.model, train_cfg=cfg.get("train_cfg"), test_cfg=cfg.get("test_cfg")
        )

        # Projection/prediction networks
        backbone_features = 2048  # if imgs are 224x224
        hidden_features = 2048
        out_features = 256
        batch_norm = (
            nn.BatchNorm1d(hidden_features)
            if pretrain_type == PretrainType.BYOL
            else nn.Identity()
        )
        self.encoder_q.projector = nn.Sequential(
            nn.Linear(in_features=backbone_features, out_features=hidden_features),
            batch_norm,
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_features, out_features=out_features),
        )
        self.encoder_k.projector = copy.deepcopy(self.encoder_q.projector)

        self.predictor = nn.Sequential(
            nn.Linear(in_features=out_features, out_features=hidden_features),
            batch_norm,
            nn.ReLU(inplace=True),
            nn.Linear(in_features=hidden_features, out_features=out_features),
        )

        # Exact copy parameters
        for param_q, param_k in zip(
            self.encoder_q.parameters(), self.encoder_k.parameters()
        ):
            param_k.data.copy_(param_q.data)  # initialize
            param_k.requires_grad = False  # not update by gradient

        # confirm config
        if self.pretrain_type in [PretrainType.MOCO, PretrainType.BYOL]:
            assert isinstance(self.encoder_q.decode_head, FCNHead)

        # create the queue
        self.register_buffer("queue", torch.randn(dim, K))
        self.queue = F.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        # Metrics
        self.loss_o = AverageMeter("Loss_overall", ":.4f")
        self.loss_i = AverageMeter("Loss_ins", ":.4f")
        self.loss_d = AverageMeter("Loss_den", ":.4f")
        self.acc_ins = AverageMeter("Acc_ins", ":6.2f")
        self.acc_seg = AverageMeter("Acc_seg", ":6.2f")

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

    def forward_moco(self, img_a, img_b, bg0, bg1, visualize):
        def loss_moco(x, y):
            x = F.normalize(x, dim=-1, p=2)
            y = F.normalize(y, dim=-1, p=2)
            return F.cross_entropy(x / self.T, y).mean()

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
        embd_q = self.encoder_q(img_a)
        q = self.encoder_q.projector(embd_q.flatten(1))

        # compute key features
        with torch.no_grad():  # no gradient to keys
            self._momentum_update_key_encoder()  # update the key encoder

            # shuffle for making use of BN
            img_b, idx_unshuffle = self._batch_shuffle_ddp(img_b)
            k = self.encoder_k.projector(self.encoder_k(img_b).flatten(1))
            # undo shuffle
            k = self._batch_unshuffle_ddp(k, idx_unshuffle) 

        # moco logits
        l_pos = torch.einsum("nc,nc->n", [q, k]).unsqueeze(-1)
        l_neg = torch.einsum("nc,ck->nk", [q, self.queue.clone().detach()])
        logits_moco = torch.cat([l_pos, l_neg], dim=1)
        labels_moco = torch.zeros(logits_moco.shape[0], dtype=torch.long).cuda()
        # dequeue and enqueue
        self._dequeue_and_enqueue(k)

        loss = loss_moco(logits_moco, labels_moco)

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

    def forward_byol(self, img_a, img_b, bg0, bg1, visualize):
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
        embd_a = self.encoder_q(img_a)
        embd_b = self.encoder_q(img_b)
        q_a = self.predictor(self.encoder_q.projector(embd_a.flatten(1)))
        q_b = self.predictor(self.encoder_q.projector(embd_b.flatten(1)))
        # fmt:on

        # compute key features
        with torch.no_grad():
            self._momentum_update_key_encoder()  # update the key encoder
            k_a = self.encoder_k.projector(self.encoder_k(img_a).flatten(1))
            k_b = self.encoder_k.projector(self.encoder_k(img_b).flatten(1))

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

    def forward_cp2(self, img_a, img_b, bg0, bg1, visualize):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        mask_a, mask_b = (bg0[:, 0] == 0).float(), (bg1[:, 0] == 0).float()
        img_a = img_a * mask_a.unsqueeze(1) + bg0
        img_b = img_b * mask_b.unsqueeze(1) + bg1

        if visualize and self.rank == 0:
            log_imgs = torch.stack([bg0, bg1, img_a, img_b], dim=1).flatten(0, 1)
            log_grid = torchvision.utils.make_grid(log_imgs, nrow=4)
            wandb.log(
                {
                    "train-examples": wandb.Image(
                        log_grid, caption=self.pretrain_type.name
                    )
                }
            )

        current_bs = img_a.size(0)

        mask_a = mask_a.reshape(current_bs, -1)
        mask_b = mask_b.reshape(current_bs, -1)

        # compute query features
        q = self.encoder_q(img_a)  # queries: NxCx14x14
        q = q.reshape(q.shape[0], q.shape[1], -1)  # queries: NxCx196
        q_dense = F.normalize(q, dim=1)  # normalize each pixel

        q_pos = F.normalize(torch.einsum("ncx,nx->nc", [q_dense, mask_a]), dim=1)
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
        mask_dense = torch.einsum("x,ny->nxy", [torch.ones(196).cuda(), mask_b])
        mask_dense = mask_dense.reshape(mask_dense.shape[0], -1)

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

        # Update logs
        self.loss_o.update(loss.item(), img_a.size(0))
        self.loss_i.update(loss_instance.item(), img_a.size(0))
        self.loss_d.update(loss_dense.item(), img_a.size(0))
        self.acc_ins.update(acc1[0], img_a.size(0))
        self.acc_seg.update(acc_dense.item(), img_a.size(0))

        if self.rank == 0:
            wandb.log(
                {
                    "train/loss_step": self.loss_o.val,
                    "train/loss_ins_step": self.loss_i.val,
                    "train/loss_dense_step": self.loss_d.val,
                    "train/acc_ins_step": self.acc_ins.val,
                    "train/acc_seg_step": self.acc_seg.val,
                    "train/batch_time_step": self.batch_time.val,
                }
            )

        return loss

    def on_train_epoch_end(self):
        if self.rank == 0:
            wandb.log(
                {
                    "train/loss": self.loss_o.avg,
                    "train/loss_ins": self.loss_i.avg,
                    "train/loss_dense": self.loss_d.avg,
                    "train/acc_ins": self.acc_ins.avg,
                    "train/acc_seg": self.acc_seg.avg,
                    "train/batch_time": self.batch_time.avg,
                }
            )
        self.reset_metrics()

    def reset_metrics(self):
        self.loss_o.reset()
        self.loss_i.reset()
        self.loss_d.reset()
        self.acc_ins.reset()
        self.acc_seg.reset()

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
