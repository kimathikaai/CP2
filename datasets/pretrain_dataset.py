import csv
import logging
import os
from enum import Enum
from glob import glob
from pathlib import Path
from random import sample
from typing import List

from PIL import Image
from torch.utils.data import DataLoader, Dataset


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
            sample = self.transform(sample)
        return sample


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
    dataset.images_list = [x for x in dataset.images_list if split_name in x]
    print(f"{orig_len = }, {len(dataset) = }")
    return dataset


def get_pretrain_dataset(
    image_directory_list: List[str],
    directory_type: DatasetType,
    transform,
    split_name=None,
):
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
