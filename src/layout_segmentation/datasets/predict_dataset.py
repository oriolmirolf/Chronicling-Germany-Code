"""
module for Dataset class
"""
from __future__ import annotations

import glob
import os
from typing import Tuple, Optional, Union

import numpy as np
import torch
from PIL import Image
from PIL.Image import BICUBIC  # pylint: disable=no-name-in-module # type: ignore
# pylint thinks torch has no name randperm this is wrong
# pylint: disable-next=no-name-in-module
from torch.utils.data import Dataset
from torchvision import transforms

from src.layout_segmentation.utils import pad_image, calculate_padding

IMAGE_PATH = "data/images"
TARGET_PATH = "data/targets/"


class PredictDataset(Dataset):
    """
    A dataset class for the newspaper datasets
    """

    def __init__(
            self,
            image_path: str,
            scale: float,
            pad: Tuple[int, int],
            target_path: Optional[str] = None,
    ) -> None:
        """
        load images and targets from folder
        :param image_path: image path
        :param scale: factor for scaling the images
        :param pad: right and bottom pad for images in pixel
        :param target_path: target path
        """

        self.image_path = image_path
        self.target_path = target_path
        self.scale = scale
        self.pad = pad

        self.file_names = [file.split(os.sep)[-1] for file in glob.glob(f"{image_path}*.png") +
                           glob.glob(f"{image_path}*.jpg")]

    def load_image(self, file: str) -> torch.Tensor:
        """
        Loads image and applies necessary transformation for prdiction.
        :param file: path to image
        :return: Tensor of dimensions (BxCxHxW). In this case, the number of batches will always be 1.
        """
        image = Image.open(self.image_path + file).convert("RGB")
        shape = int(image.size[0] * self.scale), int(image.size[1] * self.scale)
        image = image.resize(shape, resample=BICUBIC)
        transform = transforms.PILToTensor()
        data: torch.Tensor = transform(image).float() / 255
        return data

    def load_target(self, file: str) -> torch.Tensor:
        """
        Loads target and applies necessary transformation for debugging function.
        :param file: path to target
        :return: Tensor of dimensions (BxCxHxW). In this case, the number of batches will always be 1.
        """
        target:  torch.Tensor = np.load(f"{self.target_path}{file[:-4]}.npy")
        shape = int(target.shape[0] * self.scale), int(target.shape[1] * self.scale)
        target = torch.nn.functional.interpolate(torch.tensor(target[None, :, :]), size=shape, mode='nearest')

        return target

    def __getitem__(self, item: int) -> Tuple[torch.Tensor, Union[torch.Tensor, None], str]:
        """
        returns one datapoint
        :param item: number of the datapoint
        :return (tuple): torch tensor of image, torch tensor of annotation, tuple of mask
        """
        file_name = self.file_names[item]
        image = self.load_image(file_name)

        pad = calculate_padding(self.pad, image.shape, self.scale)
        image = pad_image(pad, image)

        if self.target_path is not None:
            target = self.load_target(file_name)
            target = pad_image(pad, target)
        else:
            target = torch.zeros((image.shape[0], image.shape[1]))

        return image, target, file_name

    def __len__(self) -> int:
        """
        standard len function
        :return: number of items in dateset
        """
        return len(self.file_names)
