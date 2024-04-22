import os
from glob import glob
from pathlib import Path

import albumentations as A
import cv2
import lightning as L
import numpy as np
import torch
from torchvision import transforms as T
from torch.utils.data import DataLoader, Dataset

from datasets.pretrain_dataset import pil_image_loader, pil_mask_loader

TRAIN_SPLIT_SEED = 0


class SegmentationDataset(Dataset):
    """
    Base dataset for fine-tuning
    """

    def __init__(self, image_mask_paths, transform, num_classes):
        super().__init__()
        self.image_mask_paths = image_mask_paths
        self.transform = transform
        self.num_classes = num_classes
        self.to_tensor = T.ToTensor()

    def __len__(self):
        return len(self.image_mask_paths)

    def __getitem__(self, index):
        image_path, mask_path = self.image_mask_paths[index]
        image = np.array(pil_image_loader(image_path))
        mask = np.array(pil_mask_loader(mask_path))

        if self.transform is not None:
            aug = self.transform(image=image, mask=mask)
            image, mask = aug["image"], aug["mask"]

        if self.num_classes == 2:
            # Binarize detection mask
            mask = np.array(mask, dtype=bool)

        return self.to_tensor(image), torch.from_numpy(mask).long()


class SegmentationDataModule(L.LightningDataModule):
    def __init__(
        self,
        image_directory: str,
        mask_directory: str,
        batch_size: int,
        num_workers: int,
        num_classes: int,
        image_width: float,
        image_height: float,
    ) -> None:
        super().__init__()

        self.train_image_mask_paths = []
        self.val_image_mask_paths = []
        self.test_image_mask_paths = []

        self.transform_train = None
        self.transform_val = None
        self.transform_test = None

        # Validate the directories
        self.image_directory = os.path.abspath(os.path.expanduser(image_directory))
        self.mask_directory = os.path.abspath(os.path.expanduser(mask_directory))
        assert os.path.isdir(self.image_directory)
        assert os.path.isdir(self.mask_directory)

        # image information
        self.image_width = image_width
        self.image_height = image_height
        self.image_shape = (3, self.image_height, self.image_width)

        # dataloading
        self.num_workers = num_workers
        self.batch_size = batch_size
        self.num_classes = num_classes

        # Get images
        self.image_paths = sorted(glob(os.path.join(self.image_directory, "*")))
        self.mask_paths = sorted(glob(os.path.join(self.mask_directory, "*")))
        print(f"[info] {len(self.image_paths) = } {len(self.mask_paths) = }")
        assert len(self.image_paths) > 0
        assert len(self.mask_paths) == len(self.mask_paths)

        # Make sure all images have corresponding masks
        self.image_mask_paths = []
        for img, mask in zip(self.image_paths, self.mask_paths):
            assert Path(img).stem == Path(mask).stem, "{} and {} do not match".format(
                img, mask
            )
            self.image_mask_paths.append((img, mask))

    def setup(self, stage=None) -> None:
        self.dataset_train = SegmentationDataset(
            self.train_image_mask_paths, self.transform_train, self.num_classes
        )
        self.dataset_val = SegmentationDataset(
            self.val_image_mask_paths, self.transform_val, self.num_classes
        )
        self.dataset_test = SegmentationDataset(
            self.test_image_mask_paths, self.transform_test, self.num_classes
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset_train,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            batch_size=self.batch_size,
            num_workers=int(self.num_workers * 0.5),
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset_val,
            shuffle=False,
            pin_memory=True,
            persistent_workers=False,
            batch_size=self.batch_size,
            num_workers=int(self.num_workers * 0.25),
        )

    def test_dataloader(self):
        return DataLoader(
            dataset=self.dataset_test,
            shuffle=False,
            pin_memory=True,
            persistent_workers=False,
            batch_size=self.batch_size,
            num_workers=int(self.num_workers * 0.25),
        )


class GLASDataModule(SegmentationDataModule):
    def __init__(
        self,
        image_directory: str,
        mask_directory: str,
        batch_size: int,
        num_workers: int,
        num_classes: int,
        image_size: int,
        train_data_ratio: float,
    ) -> None:
        super().__init__(
            image_directory,
            mask_directory,
            batch_size,
            num_workers,
            num_classes,
            image_size,
            image_size,
        )

        # Create the datasplits
        self.train_image_mask_paths = [
            (x, y) for x, y in self.image_mask_paths if "train" in x
        ]
        self.val_image_mask_paths = [
            (x, y) for x, y in self.image_mask_paths if "testB" in x
        ]
        self.test_image_mask_paths = [
            (x, y) for x, y in self.image_mask_paths if "testA" in x
        ]
        # Validate initial splits
        print(f"{len(self.train_image_mask_paths) = }")
        print(f"{len(self.val_image_mask_paths) = }")
        print(f"{len(self.test_image_mask_paths) = }")
        assert len(self.train_image_mask_paths) + len(self.val_image_mask_paths) + len(
            self.test_image_mask_paths
        ) == len(self.image_mask_paths)

        # Reduce train data
        self.train_data_ratio = train_data_ratio
        num_train_samples = int(
            len(self.train_image_mask_paths) * self.train_data_ratio
        )
        assert num_train_samples > 0 and num_train_samples <= len(
            self.train_image_mask_paths
        )
        train_idxs = np.random.RandomState(
            abs(hash(f"train-split-{TRAIN_SPLIT_SEED}")) % (2**31)
        ).choice(
            len(self.train_image_mask_paths), size=num_train_samples, replace=False
        )
        self.train_image_mask_paths = [
            self.train_image_mask_paths[i] for i in train_idxs
        ]
        print(f"[updated] {len(self.train_image_mask_paths) = }")
        assert len(self.train_image_mask_paths) == num_train_samples

        # setup transforms
        self.image_size = image_size
        self.transform_train = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=self.image_size,
                    interpolation=cv2.INTER_NEAREST,
                    always_apply=True,
                ),
                A.RandomCrop(
                    height=self.image_size, width=self.image_size, always_apply=True
                ),
                A.HorizontalFlip(),
                A.VerticalFlip(),
                A.ColorJitter(
                    brightness=(0.65, 1.35),
                    contrast=(0.5, 1.5),
                    saturation=(0, 1),
                    hue=(-0.1, 0.1),
                    p=0.75,
                ),
                A.GridDistortion(p=0.2),
                A.GaussNoise(p=0.5),
            ]
        )
        self.transform_val = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=self.image_size,
                    interpolation=cv2.INTER_NEAREST,
                    always_apply=True,
                ),
                A.RandomCrop(
                    height=self.image_size, width=self.image_size, always_apply=True
                ),
                A.HorizontalFlip(),
                A.VerticalFlip(),
            ]
        )
        self.transform_test = A.Compose(
            [
                A.SmallestMaxSize(
                    max_size=self.image_size,
                    interpolation=cv2.INTER_NEAREST,
                    always_apply=True,
                ),
                A.CenterCrop(
                    height=self.image_size, width=self.image_size, always_apply=True
                ),
            ]
        )

        self.setup()
