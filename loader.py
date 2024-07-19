# tool functions from moco v2 code base:
# https://github.com/facebookresearch/moco
# Copyright (c) Facebook, Inc. and its affilates. All Rights Reserved
import random

import albumentations as A
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms as T
from torchvision.transforms import functional as F


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class A_TwoCropsTransform:
    """
    Albumentations style implemenation
    Take two random crops of one image as the query and key.
    """

    def __init__(self, base_transform):
        self.base_transform = base_transform
        self.to_tensor = T.ToTensor()

    def __call__(self, x):
        #
        # Generate pixel ids
        #
        sample = np.asarray(x)
        # Get the image sizes
        print(f"{sample.shape = }")
        h, w, c = sample.shape
        # Get an array of ids
        pixel_ids = np.arange(start=1, stop=h * w + 1).reshape((h, w))
        print(f"{pixel_ids.shape = }")

        #
        # Get the query and key images
        #
        aug_q = self.base_transform(image=sample, mask=pixel_ids)
        q = self.to_tensor(aug_q["image"])
        q_ids = torch.from_numpy(aug_q["mask"])

        aug_k = self.base_transform(image=sample, mask=pixel_ids)
        k = self.to_tensor(aug_k["image"])
        k_ids = torch.from_numpy(aug_k["mask"])

        return (q, q_ids), (k, k_ids)


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x


# https://github.com/albumentations-team/albumentations/issues/938
# https://github.com/albumentations-team/albumentations/issues/938#issuecomment-1612522697
class AGaussianBlur(A.ImageOnlyTransform):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709"""

    def __init__(self, p=0.5, sigma=[0.1, 2.0], always_apply=False):
        super().__init__(always_apply, p)
        self._sigma = sigma

    def apply(self, x, copy=True, **params):
        if np.random.uniform(0, 1) > self.p:
            return x
        if copy:
            x = x.copy()

        # convert to PIL image
        x = Image.fromarray(x)
        sigma = random.uniform(self._sigma[0], self._sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return np.asarray(x)
