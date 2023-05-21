# Code here is taken from Pytorch torchvision.models repository and then edited by Modestas Sukarevicius

from functools import partial
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.init as init


from torchvision.models.squeezenet import Fire



class SqueezeNet(nn.Module):
    def __init__(
            self,
            version: str = "1_0",
            num_outputs: int = 6,
            dropout: float = 0.5,
            extra_layers: bool = False
    ) -> None:
        super().__init__()
        self.num_outputs = num_outputs
        self.extra_layers = extra_layers
        if version == "1_0":
            self.features = nn.Sequential(
                nn.Conv2d(1, 96, kernel_size=7, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(512, 64, 256, 256),
            )
        elif version == "1_1":
            self.features = nn.Sequential(
                nn.Conv2d(1, 64, kernel_size=3, stride=2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256),
            )
        else:
            raise ValueError(f"Unsupported SqueezeNet version {version}: 1_0 or 1_1 expected")

        # Final convolution is initialized differently from the rest
        if extra_layers:
            final_conv = nn.Conv2d(512, 256, kernel_size=3, padding=1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                final_conv,
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=2),
                nn.Flatten(),
                nn.Linear(1024 * 6 * 6, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, self.num_outputs)
            )
        else:
            final_conv = nn.Conv2d(512, self.num_outputs, kernel_size=1)
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout),
                final_conv,
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1))
            )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if m is final_conv:
                    init.normal_(m.weight, mean=0.0, std=0.01)
                else:
                    init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x)
        if self.extra_layers==False:
            x = torch.flatten(x, 1)
        return x


def _squeezenet(version: str, **kwargs: Any) -> SqueezeNet:
    model = SqueezeNet(version, **kwargs)
    return model

def squeezenet1_0(
    *, progress: bool = True, **kwargs: Any
) -> SqueezeNet:
    """SqueezeNet model architecture from the `SqueezeNet: AlexNet-level
    accuracy with 50x fewer parameters and <0.5MB model size
    <https://arxiv.org/abs/1602.07360>`_ paper.
    Args:
        **kwargs: parameters passed to the ``torchvision.models.squeezenet.SqueezeNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py>`_
            for more details about this class.
    """
    return _squeezenet("1_0", **kwargs)

def squeezenet1_1(
    *, progress: bool = True, **kwargs: Any
) -> SqueezeNet:
    """SqueezeNet 1.1 model from the `official SqueezeNet repo
    <https://github.com/DeepScale/SqueezeNet/tree/master/SqueezeNet_v1.1>`_.
    SqueezeNet 1.1 has 2.4x less computation and slightly fewer parameters
    than SqueezeNet 1.0, without sacrificing accuracy.
    Args:
        **kwargs: parameters passed to the ``torchvision.models.squeezenet.SqueezeNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/squeezenet.py>`_
            for more details about this class.
    """
    return _squeezenet("1_1", **kwargs)