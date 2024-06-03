"""Module for small dhsegment https://arxiv.org/abs/1804.10371"""
# coding=utf-8
from __future__ import absolute_import, division, print_function

import logging
from typing import Dict, Iterator, Union

import torch
from torch import nn
from torch.nn.parameter import Parameter

from src.layout_segmentation.models.dh_segment import DhSegment, conv1x1

logger = logging.getLogger(__name__)


class DilationModule(nn.Module):
    """Dilation Module consisting of a 5x5 convolution followed by a 9x9 dilated convolution. It contains a residual
    connection, so the identity will be added to the conv results before the  ReLu activation function.
    This module is intended to greatly increase the receptive field. It does not change the number of channels."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv_5 = nn.Conv2d(
            channels,
            channels,
            kernel_size=5,
            stride=1,
            padding="same",
            groups=1,
            bias=False,
        )
        self.bn_1 = nn.BatchNorm2d(channels)
        self.conv_dilated = nn.Conv2d(
            channels,
            channels,
            kernel_size=9,
            stride=1,
            padding="same",
            groups=1,
            bias=False,
            dilation=2,
        )
        self.bn_2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Execute conv and batch norm layers, as well as residual connection before ReLu."""
        identity = inputs

        results = self.conv_5(inputs)
        results = self.bn_1(results)
        results = self.conv_dilated(results)
        results = self.bn_2(results)
        return self.relu(results + identity)  # type: ignore


class Encoder(nn.Module):
    """
    CNN Encoder Class, corresponding to the first resnet50 layers.
    """

    def __init__(self, dhsegment: DhSegment, in_channels: int):
        super().__init__()
        self.conv1 = dhsegment.conv1
        self.bn1 = dhsegment.bn1
        self.relu = dhsegment.relu
        self.maxpool = dhsegment.maxpool

        self.block1 = dhsegment.block1
        self.block2 = dhsegment.block2
        self.dilated_block_1 = self.make_dilation_layers(3, 512)
        self.maxpool_2 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.dilated_block_2 = self.make_dilation_layers(2, 512)

        # initialize normalization
        # pylint: disable=duplicate-code
        self.register_buffer("means", torch.tensor([0] * in_channels))
        self.register_buffer("stds", torch.tensor([1] * in_channels))
        self.normalize = dhsegment.normalize

    def forward(self, inputs: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Encoder forward
        :param inputs: input tensor
        :return: dictionary with result and scip-connections
        """
        result = self.normalize(inputs, self.means, self.stds)
        result = self.conv1(result)
        result = self.bn1(result)
        copy_0 = self.relu(result)
        result = self.maxpool(copy_0)

        result, copy_1 = self.block1(result)
        result, copy_2 = self.block2(result)

        return {
            "copy_0": copy_0,
            "copy_1": copy_1,
            "copy_2": copy_2
        }

    def make_dilation_layers(self, number: int, channels: int) -> nn.Sequential:
        """Arange multiple dilation modules in nn.Sequential."""
        layers = []
        for _ in range(0, number):
            layers.append(
                DilationModule(
                    channels
                )
            )
        return nn.Sequential(*layers)

    def freeze_encoder(self, requires_grad: bool = False) -> None:
        """
        Set requires grad of encoder to True or False. Freezes encoder weights
        :param requires_grad: freezes encoder weights if false else unfreezes the weights
        """

        # noinspection DuplicatedCode
        # pylint: disable=duplicate-code
        def freeze(params: Iterator[Parameter]) -> None:
            for param in params:
                param.requires_grad_(requires_grad)

        freeze(self.conv1.parameters())
        freeze(self.bn1.parameters())
        freeze(self.block1.parameters())
        freeze(self.block2.parameters())


class Decoder(nn.Module):
    """
    CNN Decoder class, corresponding to DhSegment Decoder
    """

    def __init__(self, dhsegment: DhSegment):
        super().__init__()
        self.up_block3 = dhsegment.up_block3
        self.up_block4 = dhsegment.up_block4
        self.up_block5 = dhsegment.up_block5

        self.conv2 = dhsegment.conv2

    def forward(self, encoder_results: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        forward path of cnn decoder
        :param transformer_result: transformer output, as matrix??
        :param encoder_results: contains saved values for scip connections of unet
        :return: a decoder result
        """
        # pylint: disable=duplicate-code
        tensor_x: torch.Tensor = self.up_block3(encoder_results["copy_2"], encoder_results["copy_1"])
        tensor_x = self.up_block4(tensor_x, encoder_results["copy_0"])
        tensor_x = self.up_block5(tensor_x, encoder_results["identity"])

        tensor_x = self.conv2(tensor_x)

        return tensor_x


class DhSegmentDilated(nn.Module):
    """Implements dilated DhSegment by replacing the lower 2 layers."""

    def __init__(
            self, in_channels: int = 3, out_channel: int = 3, load_resnet_weights: bool = True
    ) -> None:
        """
        :param config:
        :param in_channels:
        :param out_channel:
        :param zero_head:
        """
        super().__init__()
        dhsegment = DhSegment(
            [3, 4, 6, 1],
            in_channels=in_channels,
            out_channel=out_channel,
            load_resnet_weights=load_resnet_weights,
        )
        self.encoder = Encoder(dhsegment, in_channels)
        self.decoder = Decoder(dhsegment)

    def forward(self, x_tensor: torch.Tensor) -> torch.Tensor:
        """
        :param x_tensor: input
        :return: unet result
        """
        encoder_results = self.encoder(x_tensor)
        x_tensor = self.decoder(encoder_results)
        return x_tensor

    def freeze_encoder(self, requires_grad: bool = False) -> None:
        """
        Set requires grad of encoder to True or False. Freezes encoder weights
        :param requires_grad: freezes encoder weights if false else unfreezes the weights
        """

        self.encoder.freeze_encoder(requires_grad)

    def save(self, path: Union[str, None]) -> None:
        """
        saves the model weights
        :param path: path to savepoint
        :return: None
        """
        # pylint: disable=duplicate-code
        if path is None:
            return
        torch.save(self.state_dict(), path + ".pt")

    def load(self, path: Union[str, None], device: str) -> None:
        """
        load the model weights
        :param device: mapping device
        :param path: path to savepoint
        :return: None
        """
        if path is None:
            return
        self.load_state_dict(torch.load(path, map_location=device))
        self.eval()
