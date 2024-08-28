# tool functions from moco v2 code base:
# https://github.com/facebookresearch/moco
# Copyright (c) Facebook, Inc. and its affilates. All Rights Reserved
import os
import random
from pathlib import Path

import albumentations as A
import cv2
import numpy as np
import torch
from PIL import Image, ImageFilter
from torchvision import transforms as T
from torchvision.transforms import functional as F

from builder import MappingType


class TwoCropsTransform:
    """Take two random crops of one image as the query and key."""

    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x):
        q = self.base_transform(x)
        k = self.base_transform(x)
        return [q, k]


class BackgroundTransform:
    def __init__(self, base_transform):
        self.base_transform = base_transform

    def __call__(self, x, path):
        return self.base_transform(x)


def rescale_ids(pixel_ids, stride):
    return pixel_ids[
        stride // 2 :: stride,
        stride // 2 :: stride,
    ]


MASK_DIR = "SAM_Masks"
MASK_EXT = ".png"


class A_TwoCropsTransform:
    """
    Albumentations style implemenation
    Take two random crops of one image as the query and key.
    """

    def __init__(
        self, base_transform, mapping_type: MappingType, pixel_ids_stride: int = 1
    ):
        self.base_transform = base_transform
        self.to_tensor = T.ToTensor()
        assert pixel_ids_stride > 0
        self.pixel_ids_stride = pixel_ids_stride
        assert mapping_type in MappingType
        self.mapping_type = mapping_type

    def get_pixel_ids(self, height, width, path):
        pixel_ids = np.arange(start=1, stop=height * width + 1).reshape((height, width))
        # change the pixel id resolution
        pixel_ids = rescale_ids(pixel_ids, self.pixel_ids_stride)
        pixel_ids = cv2.resize(
            pixel_ids, dsize=(width, height), interpolation=cv2.INTER_NEAREST_EXACT
        )

        # Get the SAM generated pixel id map
        if self.mapping_type in [MappingType.REGION_ID, MappingType.PIXEL_REGION_ID]:
            mask_name = Path(path).stem + MASK_EXT
            mask_path = os.path.join(Path(path).parents[1], MASK_DIR, mask_name)
            region_ids = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            # change the pixel id resolution
            region_ids = rescale_ids(region_ids, self.pixel_ids_stride)
            region_ids = cv2.resize(
                region_ids, dsize=(width, height), interpolation=cv2.INTER_NEAREST_EXACT
            )
        else:
            region_ids = pixel_ids

        return pixel_ids, region_ids

    def __call__(self, x, path):
        #
        # Generate pixel ids
        #
        sample = np.asarray(x)
        # Get the image sizes
        # print(f"{sample.shape = }")
        h, w, c = sample.shape
        # Get an array of ids
        pixel_ids, region_ids = self.get_pixel_ids(h, w, path)
        #
        # Get the query and key images
        #
        aug_q = self.base_transform(
            image=sample, mask=pixel_ids, region_ids=region_ids
        )
        q = aug_q["image"]
        q = self.to_tensor(np.array(q))
        q_pixel_ids = torch.from_numpy(aug_q["mask"])
        q_region_ids = torch.from_numpy(aug_q["region_ids"])

        aug_k = self.base_transform(
            image=sample, mask=pixel_ids, region_ids=region_ids
        )
        k = aug_k["image"]
        k = self.to_tensor(np.array(k))
        k_pixel_ids = torch.from_numpy(aug_k["mask"])
        k_region_ids = torch.from_numpy(aug_k["region_ids"])

        return (q, q_pixel_ids, q_region_ids), (k, k_pixel_ids, k_region_ids)


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
