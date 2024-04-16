import csv
import logging
import os
from glob import glob
from pathlib import Path
from typing import List

from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import ImageFolder


def pil_loader(path: str) -> Image.Image:
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, "rb") as f:
        img = Image.open(f)
        return img.convert("RGB")


def read_paths_csv(csv_path: str) -> List[str]:
    """
    Given a csv of paths, read them as a comma separated row
    """
    path_list = []

    with open(csv_path, "r") as file:
        csvreader = csv.reader(file)
        for row in csvreader:
            path_list.extend(row)

    logging.info(f"Read {len(path_list)} filenames from {csv_path}")

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
        sample = pil_loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample


def get_pretrain_dataset(
    image_directory_list: List[str], image_csv_list: List[str], transform
):
    """
    Returns an initialized PretrainDataset
    """
    # Get the full paths
    image_directory_list = [
        os.path.abspath(os.path.expanduser(x)) for x in image_directory_list
    ]
    image_csv_list = [os.path.abspath(os.path.expanduser(x)) for x in image_csv_list]
    logging.info(
        f"[info] Loading image from\n{image_directory_list}\naccording to\n {image_csv_list}"
    )

    # validate directory existence and get images
    sample_paths = []
    for img_dir in image_directory_list:
        assert os.path.exists(img_dir), "DNE: {}".format(img_dir)
        files = glob(os.path.join(img_dir, "*"))
        sample_paths.extend(files)
    # sort based on file names
    sample_paths = sorted(sample_paths, key=lambda x: Path(x).stem)
    logging.info(f"Found {len(sample_paths) = } images")

    # assert that no files share the same name
    sample_names = [Path(x).stem for x in sample_paths]
    assert len(sample_names) == len(set(sample_names))

    # get the included files
    included_samples = []
    for csv_path in image_csv_list:
        included_samples.extend(read_paths_csv(csv_path))

    # filter out the images not included
    included_samples_stems = get_file_stem(included_samples)
    _sample_paths = [x for x in sample_paths if Path(x).stem in included_samples_stems]

    return PretrainDataset(_sample_paths, transform)
