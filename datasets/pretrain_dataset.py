import csv
import logging
import os
from enum import Enum
from glob import glob
from pathlib import Path
from random import sample
from typing import List

import albumentations as A
import cv2
import lightning as L
import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T


class DatasetType(Enum):
    """
    Used to differentiate between expected data structures
    """

    CSV = 0
    CLASSIFICATION = 1
    # Uses indicators in the filenames to split
    FILENAME = 2


def pil_image_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def pil_mask_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("L")


def read_paths_csv(csv_path: str) -> List[str]:
    """
    Given a csv of paths, read them as a comma separated row
    """
    path_list = []

    with open(csv_path, "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            path_list.extend(row)

    print(f"Read {len(path_list)} filenames from {csv_path}")

    return path_list


def get_file_stem(path_list: List[str]):
    """
    Remove the exensions from file name and return the
    extension free file name. This function is required
    because some file names include '.'
    which naively splitting based on '.' on file names
    without extensions will produce non existent filenames
    """
    _path_list = []
    for path in path_list:
        if Path(path).suffix in [".png", ".jpg", ".bmp", ".tif", ".tiff"]:
            _path_list.append(Path(path).stem)
        else:
            _path_list.append(Path(path).name)

    return _path_list


class PretrainDataset(Dataset):
    """
    Base dataset for pre-training
    """

    def __init__(self, images_list, transform):
        super().__init__()
        self.images_list = images_list
        self.transform = transform

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, index):
        path = self.images_list[index]
        sample = pil_image_loader(path)
        if self.transform is not None:
            return self.transform(sample)


def get_custom_pretrain_dataset(image_directory_list: List[str], split_name, transform):
    # Assumes that the split csv file is in the same directory as the image data

    # for each directory find the split csv file
    # then get the full paths of the images in that csv file
    # the assert that the number of collected and viable paths is the same as the csv List
    # output the number of files used from this directory

    # then extend a larger list of files that will be used to create the dataset

    sample_paths = []

    for img_dir in image_directory_list:
        assert os.path.exists(img_dir), "DNE: {}".format(img_dir)
        csv_path = os.path.join(img_dir, f"{split_name}.csv")
        included_paths = read_paths_csv(csv_path)
        included_paths_stems = get_file_stem(included_paths)
        file_paths = glob(os.path.join(img_dir, "*"))
        _sample_paths = [x for x in file_paths if Path(x).stem in included_paths_stems]
        _sample_stems = [Path(x).stem for x in _sample_paths]
        # validate filtering
        print(
            f"[info] Path differences: {set(_sample_stems).symmetric_difference(set(included_paths_stems))}"
        )
        assert len(included_paths) == len(
            _sample_paths
        ), f"{len(_sample_paths) = }, {len(included_paths) = }"
        print(f"[info] Loading {len(_sample_paths) = } from {img_dir}")
        sample_paths.extend(_sample_paths)

    print(f"[info] Using {len(sample_paths) = } total files")
    return PretrainDataset(sample_paths, transform)


def get_classification_pretrain_dataset(image_directory_list: List[str], transform):
    # validate directory existence and get images
    sample_paths = []
    for img_dir in image_directory_list:
        assert os.path.exists(img_dir), "DNE: {}".format(img_dir)
        files = glob(os.path.join(img_dir, "*"))
        sample_paths.extend(files)

    # sort based on file names
    sample_paths = sorted(sample_paths, key=lambda x: Path(x).stem)
    print(f"Found {len(sample_paths) = } images")

    return PretrainDataset(sample_paths, transform)


def get_filename_pretrain_dataset(dataset: PretrainDataset, split_name):
    assert split_name in ["train", "val", "test"]
    orig_len = len(dataset)
    dataset.images_list = [
        x for x in dataset.images_list if split_name in x and ".csv" not in x
    ]
    print(f"{orig_len = }, {len(dataset) = }")
    return dataset


def get_pretrain_dataset(
    image_directory_list: List[str],
    directory_type: DatasetType,
    transform,
    split_name=None,
) -> PretrainDataset:
    """
    Returns an initialized PretrainDataset
    """
    # Get the full paths
    image_directory_list = [
        os.path.abspath(os.path.expanduser(x)) for x in image_directory_list
    ]

    if directory_type == DatasetType.CSV:
        return get_custom_pretrain_dataset(image_directory_list, split_name, transform)
    elif directory_type == DatasetType.CLASSIFICATION:
        return get_classification_pretrain_dataset(image_directory_list, transform)
    elif directory_type == DatasetType.FILENAME:
        dataset = get_classification_pretrain_dataset(image_directory_list, transform)
        return get_filename_pretrain_dataset(dataset, split_name)


class CutPastePatchType(Enum):
    NONE = 0
    REGULAR = 1
    SCAR = 2


class MirrorVariant(Enum):
    NONE = 0
    OUTPUT = 1


class CutPasteDataset(Dataset):
    """
    Dataset for implementing the CutPaste pre-training
    strategy. This dataset takes in input images, applies the CutPaste Augmentation
    and returns the image and its corresponding label
    """

    def __init__(
        self,
        images_list,
        min_area_scale: float,
        max_area_scale: float,
        min_aspect_ratio: float,
        max_aspect_ratio: float,
        min_rotation: float,
        max_rotation: float,
        mirror_variant: MirrorVariant,
        num_classes: int,
        max_num_patches: int,
        base_transform,
        debug=False,
    ):
        super().__init__()
        self.images_list = images_list
        self.base_transform = base_transform
        self.to_tensor = T.ToTensor()

        # Parameters
        self.debug = debug
        self.min_rotation = min_rotation
        self.max_rotation = max_rotation
        self.min_area_scale = min_area_scale
        self.max_area_scale = max_area_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

        # variant
        assert mirror_variant in MirrorVariant
        self.mirror_variant = mirror_variant

        assert max_num_patches >= 1
        self.max_num_patches = max_num_patches
        # no current handling for scars and regular with multiple patches
        assert max_num_patches == 1 or num_classes <= 2

        # List of the potential transformations. In the order of: Original Image, CutPaste, CutPaste Scar
        self.transforms_list = [
            T.Lambda(
                lambda img_mirror: (
                    img_mirror[0],
                    img_mirror[1],
                    np.zeros(shape=img_mirror[0].shape[:2], dtype=np.uint8),
                )
            ),
            T.Lambda(
                lambda img_mirror: self.cutpaste(
                    image=img_mirror[0],
                    mirror_image=img_mirror[1],
                    patch_type=CutPastePatchType.REGULAR,
                )
            ),
            T.Lambda(
                lambda img_mirror: self.cutpaste(
                    image=img_mirror[0],
                    mirror_image=img_mirror[1],
                    patch_type=CutPastePatchType.SCAR,
                )
            ),
        ]

        # classes
        self.classes = list(range(num_classes))
        self.targets = np.random.choice(
            self.classes,
            size=len(self.images_list),
            replace=True,
            p=[0.1, 0.45, 0.45] if num_classes == 3 else [0.1, 0.9],
        )

        print(f"Dataset target values: {np.unique(self.targets, return_counts=True)}")

    def cutpaste(
        self,
        image: np.ndarray,
        mirror_image: np.ndarray,
        patch_type: CutPastePatchType,
    ):
        """"""

        assert patch_type in CutPastePatchType

        img_h, img_w, img_c = image.shape
        assert img_h > img_c
        assert img_h <= img_w

        #
        # Get patch details
        #
        if patch_type == CutPastePatchType.REGULAR:
            patch_area_scale = np.random.uniform(
                high=self.max_area_scale,
                low=self.min_area_scale,
            )
            patch_aspect_ratio = np.random.uniform(
                high=self.max_aspect_ratio,
                low=self.min_aspect_ratio,
            )
            patch_rotation = 0
        elif patch_type == CutPastePatchType.SCAR:
            patch_area_scale = np.random.uniform(
                high=self.max_area_scale * 0.5,
                low=self.min_area_scale,
            )
            patch_aspect_ratio = np.random.uniform(3, 6)
            patch_rotation = np.random.uniform(
                low=self.min_rotation,
                high=self.max_rotation,
            )
        else:
            raise Exception(f"No handling for patch type {patch_type}")

        patch_area = int(img_h * img_w * patch_area_scale)
        patch_height = int(np.sqrt(patch_area / patch_aspect_ratio))
        patch_width = int(patch_height * patch_aspect_ratio)

        # get random patch
        x_pos_patch = np.random.randint(0, img_w - patch_width)
        y_pos_patch = np.random.randint(0, img_h - patch_height)
        image_patch = image[
            y_pos_patch : y_pos_patch + patch_height,
            x_pos_patch : x_pos_patch + patch_width,
            :,
        ]

        # create the mask
        patch_mask = Image.new("L", (patch_width, patch_height), 255)
        # rotate the patch image
        rotated_patch = Image.fromarray(image_patch).rotate(patch_rotation, expand=True)
        # also rotate the mask
        patch_mask = patch_mask.rotate(patch_rotation, expand=True)
        # Get pasting position
        x_pos = np.random.randint(0, img_w - rotated_patch.width)
        y_pos = np.random.randint(0, img_h - rotated_patch.height)

        # paste patch onto original image
        pil_image = Image.fromarray(image)
        pil_image.paste(rotated_patch, (x_pos, y_pos), patch_mask)
        if mirror_image is not None:
            mirror_image = Image.fromarray(mirror_image)
            mirror_image.paste(rotated_patch, (x_pos, y_pos), patch_mask)

        # change mask value to class
        patch_mask = np.array(patch_mask, dtype=bool) * patch_type.value
        # create the full image mask
        mask = np.zeros(shape=(img_h, img_w), dtype=int)
        mask[
            y_pos : y_pos + patch_mask.shape[0],
            x_pos : x_pos + patch_mask.shape[1],
        ] = patch_mask

        return pil_image, mirror_image, mask

    def __len__(self):
        return len(self.images_list)

    def __getitem__(self, idx):
        img_path = self.images_list[idx]
        img_class = self.targets[idx]

        img = cv2.imread(img_path)
        # opencv reads images in BGR format by default. Need to switch for transforms
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        # resizing the image
        resized_img = self.base_transform(image=img)["image"]

        mirror_img = None
        if self.mirror_variant == MirrorVariant.OUTPUT:
            # select a new index
            mirror_idx = np.random.randint(len(self.images_list))
            mirror_path = self.images_list[mirror_idx]
            mirror_img = cv2.imread(mirror_path)
            # opencv reads images in BGR format by default. Need to switch for transforms
            mirror_img = cv2.cvtColor(
                mirror_img, cv2.COLOR_BGR2RGB
            )  # convert BGR to RGB
            mirror_img = self.base_transform(image=mirror_img)["image"]

        img, mirror_img, mask = self.transforms_list[img_class](
            (resized_img, mirror_img)
        )

        if img_class != 0:
            num_additional_patches = np.random.randint(self.max_num_patches)
            for _ in range(num_additional_patches):
                mirror_img = None if mirror_img is None else np.asarray(mirror_img)
                img = np.asarray(img)
                old_mask = mask

                # add a new cutpaste region
                img, mirror_img, mask = self.transforms_list[img_class](
                    (img, mirror_img)
                )

                # update the cutpaste mask
                mask = np.logical_or(mask, old_mask)

        img, mask = self.to_tensor(img), torch.from_numpy(mask).long()

        if self.mirror_variant == MirrorVariant.OUTPUT:
            mirror_img = self.to_tensor(mirror_img)
            return (
                (img, mirror_img, mask, img_class)
                if self.debug
                else (img, mirror_img, mask)
            )

        else:
            return (img, mask, img_class) if self.debug else (img, mask)


class CutPasteDataModule(L.LightningDataModule):
    """
    LightningDataModule for instantiating training, validation and testing
    DataLoaders for the CutPaste Dataset
    """

    def __init__(
        self,
        img_dir_list: List[str],
        batch_size: int,
        num_workers: int,
        num_classes: int,
        max_num_patches: int,
        img_x_size: int,
        img_y_size: int,
        min_area_scale: float,
        max_area_scale: float,
        min_aspect_ratio: float,
        max_aspect_ratio: float,
        min_rotation: float,
        max_rotation: float,
        variant: MirrorVariant,
        debug=False,
    ):
        super().__init__()

        #
        # data loading
        #
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.num_classes = num_classes
        self.max_num_patches = max_num_patches
        assert variant in MirrorVariant
        self.mirror_variant = variant
        self.debug = debug

        #
        # transformation parameters
        #
        self.img_x_size = img_x_size
        self.img_y_size = img_y_size
        self.image_shape = (3, img_x_size, img_y_size)

        self.min_rotation = min_rotation
        self.max_rotation = max_rotation
        self.min_area_scale = min_area_scale
        self.max_area_scale = max_area_scale
        self.min_aspect_ratio = min_aspect_ratio
        self.max_aspect_ratio = max_aspect_ratio

        #
        # get full list of images and masks
        #
        self.img_dir_list = [
            os.path.abspath(os.path.expanduser(img_dir)) for img_dir in img_dir_list
        ]
        print(f"[info] {self.img_dir_list =  }")

        # get images for each split
        self.images_dict = {"train": [], "val": []}
        for img_dir in self.img_dir_list:
            assert os.path.exists(img_dir), "DNE: {}".format(img_dir)
            for split in self.images_dict.keys():
                csv_path = os.path.join(img_dir, split + ".csv")
                # read the split image paths
                included_paths = read_paths_csv(csv_path)
                included_paths_stems = get_file_stem(included_paths)
                # get all the images in the directory
                file_paths = glob(os.path.join(img_dir, "*"))
                # get only the images included in the split csv
                _sample_paths = [
                    x for x in file_paths if Path(x).stem in included_paths_stems
                ]
                _sample_stems = [Path(x).stem for x in _sample_paths]
                # validate filtering
                print(
                    f"[info][{split}] Path differences: ",
                    set(_sample_stems).symmetric_difference(set(included_paths_stems)),
                )
                assert len(included_paths) == len(
                    _sample_paths
                ), f"{len(_sample_paths) = }, {len(included_paths) = }"
                print(
                    f"[info][{split}] Loading  {len(_sample_paths) = } from {img_dir}"
                )
                self.images_dict[split].extend(_sample_paths)

        for name, paths in self.images_dict.items():
            assert len(paths) > 0
            print(f"[info][{name}] {len(paths) = }")

        self.setup()

    def setup(self, stage=None):
        #
        # Initialize Resizing
        #
        train_transform = A.Compose(
            [
                # A.Resize(
                #     self.img_x_size, self.img_y_size, interpolation=cv2.INTER_NEAREST
                # ),
                A.RandomResizedCrop(
                    size=(self.img_x_size, self.img_y_size),
                    scale=(0.2, 1.0),
                    ratio=(3 / 4, 4 / 3),
                    interpolation=cv2.INTER_NEAREST,
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
                # A.RandomBrightnessContrast((0, 0.5), (0, 0.5)),
                A.GaussNoise(),
            ]
        )

        #
        # Initialize Datasets
        #
        self.dataset_train = CutPasteDataset(
            images_list=self.images_dict["train"],
            num_classes=self.num_classes,
            max_num_patches=self.max_num_patches,
            mirror_variant=self.mirror_variant,
            min_rotation=self.min_rotation,
            max_rotation=self.max_rotation,
            min_area_scale=self.min_area_scale,
            max_area_scale=self.max_area_scale,
            min_aspect_ratio=self.min_aspect_ratio,
            max_aspect_ratio=self.max_aspect_ratio,
            base_transform=train_transform,
            debug=self.debug,
        )
        self.dataset_val = CutPasteDataset(
            images_list=self.images_dict["val"],
            num_classes=self.num_classes,
            max_num_patches=self.max_num_patches,
            mirror_variant=self.mirror_variant,
            min_rotation=self.min_rotation,
            max_rotation=self.max_rotation,
            min_area_scale=self.min_area_scale,
            max_area_scale=self.max_area_scale,
            min_aspect_ratio=self.min_aspect_ratio,
            max_aspect_ratio=self.max_aspect_ratio,
            base_transform=train_transform,
            debug=self.debug,
        )

    def train_dataloader(self):
        return DataLoader(
            dataset=self.dataset_train,
            shuffle=True,
            pin_memory=True,
            persistent_workers=True,
            batch_size=self.batch_size,
            num_workers=int(self.num_workers * 0.75),
            drop_last=False,
        )

    def val_dataloader(self):
        return DataLoader(
            dataset=self.dataset_val,
            pin_memory=False,
            batch_size=self.batch_size,
            num_workers=int(self.num_workers * 0.25),
        )
