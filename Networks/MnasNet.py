import warnings
from functools import partial
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch import Tensor

from torchvision.models._api import register_model
from torchvision.models._utils import _ovewrite_named_param, handle_legacy_interface
from torchvision.models.mnasnet import  _stack, _round_to_multiple_of, _get_depths, _InvertedResidual



# Paper suggests 0.9997 momentum, for TensorFlow. Equivalent PyTorch momentum is
# 1.0 - tensorflow.
_BN_MOMENTUM = 1 - 0.9997

class MNASNet(torch.nn.Module):
    """MNASNet, as described in https://arxiv.org/pdf/1807.11626.pdf. This
    implements the B1 variant of the model.
    >>> model = MNASNet(1.0, num_outputs=1000)
    >>> x = torch.rand(1, 3, 224, 224)
    >>> y = model(x)
    >>> y.dim()
    2
    >>> y.nelement()
    1000
    """

    # Version 2 adds depth scaling in the initial stages of the network.
    _version = 2

    def __init__(
            self,
            alpha: float,
            num_outputs: int = 6,
            dropout: float = 0.2,
            extra_layers: bool = False,
    ) -> None:
        super().__init__()
        if alpha <= 0.0:
            raise ValueError(f"alpha should be greater than 0.0 instead of {alpha}")
        self.alpha = alpha
        self.num_outputs = num_outputs
        depths = _get_depths(alpha)
        layers = [
            # First layer: regular conv.
            nn.Conv2d(1, depths[0], 3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # Depthwise separable, no skip.
            nn.Conv2d(depths[0], depths[0], 3, padding=1, stride=1, groups=depths[0], bias=False),
            nn.BatchNorm2d(depths[0], momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(depths[0], depths[1], 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(depths[1], momentum=_BN_MOMENTUM),
            # MNASNet blocks: stacks of inverted residuals.
            _stack(depths[1], depths[2], 3, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[2], depths[3], 5, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[3], depths[4], 5, 2, 6, 3, _BN_MOMENTUM),
            _stack(depths[4], depths[5], 3, 1, 6, 2, _BN_MOMENTUM),
            _stack(depths[5], depths[6], 5, 2, 6, 4, _BN_MOMENTUM),
            _stack(depths[6], depths[7], 3, 1, 6, 1, _BN_MOMENTUM),
            # Final mapping to classifier input.
            nn.Conv2d(depths[7], 1280, 1, padding=0, stride=1, bias=False),
            nn.BatchNorm2d(1280, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
        ]
        self.layers = nn.Sequential(*layers)
        if extra_layers:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(1280, 1024),
                nn.ReLU(inplace=True),
                nn.Linear(1024, 512),
                nn.ReLU(inplace=True),
                nn.Linear(512, num_outputs))
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(1280, num_outputs))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, mode="fan_out", nonlinearity="sigmoid")
                nn.init.zeros_(m.bias)

    def forward(self, x: Tensor) -> Tensor:
        x = self.layers(x)
        # Equivalent to global avgpool and removing H and W dimensions.
        x = x.mean([2, 3])
        return self.classifier(x)

    def _load_from_state_dict(
        self,
        state_dict: Dict,
        prefix: str,
        local_metadata: Dict,
        strict: bool,
        missing_keys: List[str],
        unexpected_keys: List[str],
        error_msgs: List[str],
    ) -> None:
        version = local_metadata.get("version", None)
        if version not in [1, 2]:
            raise ValueError(f"version shluld be set to 1 or 2 instead of {version}")

        if version == 1 and not self.alpha == 1.0:
            # In the initial version of the model (v1), stem was fixed-size.
            # All other layer configurations were the same. This will patch
            # the model so that it's identical to v1. Model with alpha 1.0 is
            # unaffected.
            depths = _get_depths(self.alpha)
            v1_stem = [
                nn.Conv2d(3, 32, 3, padding=1, stride=2, bias=False),
                nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 32, 3, padding=1, stride=1, groups=32, bias=False),
                nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
                nn.ReLU(inplace=True),
                nn.Conv2d(32, 16, 1, padding=0, stride=1, bias=False),
                nn.BatchNorm2d(16, momentum=_BN_MOMENTUM),
                _stack(16, depths[2], 3, 2, 3, 3, _BN_MOMENTUM),
            ]
            for idx, layer in enumerate(v1_stem):
                self.layers[idx] = layer

            # The model is now identical to v1, and must be saved as such.
            self._version = 1
            warnings.warn(
                "A new version of MNASNet model has been implemented. "
                "Your checkpoint was saved using the previous version. "
                "This checkpoint will load and work as before, but "
                "you may want to upgrade by training a newer model or "
                "transfer learning from an updated ImageNet checkpoint.",
                UserWarning,
            )

        super()._load_from_state_dict(
            state_dict, prefix, local_metadata, strict, missing_keys, unexpected_keys, error_msgs
        )

def _mnasnet(alpha: float, **kwargs: Any) -> MNASNet:
    model = MNASNet(alpha, **kwargs)
    return model


def mnasnet1_0(*, progress: bool = True, **kwargs: Any) -> MNASNet:
    """MNASNet with depth multiplier of 0.5 from
    `MnasNet: Platform-Aware Neural Architecture Search for Mobile
    <https://arxiv.org/pdf/1807.11626.pdf>`_ paper.
    Args:
        **kwargs: parameters passed to the ``torchvision.models.mnasnet.MNASNet``
            base class. Please refer to the `source code
            <https://github.com/pytorch/vision/blob/main/torchvision/models/mnasnet.py>`_
            for more details about this class.
    .. autoclass:: torchvision.models.MNASNet0_5_Weights
        :members:
    """
    return _mnasnet(1.0, **kwargs)

def mnasnet3_8(*, progress: bool = True, **kwargs: Any) -> MNASNet:
    return _mnasnet(3.8, **kwargs)

def mnasnet6_0(*, progress: bool = True, **kwargs: Any) -> MNASNet:
    return _mnasnet(6, **kwargs)