# Copyright 2018 University of Basel, Center for medical Image Analysis and Navigation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import os
import sys

import torch as th
import torch.nn.functional as F
from torchvision import transforms

import numpy as np
import torch

from .. import transformation as T
from ..transformation import utils as tu
from ..utils import kernelFunction as utils
from ..dino import FeatureExtractor
import matplotlib.pyplot as plt

# Loss base class (standard from PyTorch)

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning)
sys.stderr = open(os.devnull, 'w')


class _PairwiseImageLoss(th.nn.modules.Module):
    def __init__(self, fixed_image, moving_image, fixed_mask=None, moving_mask=None, size_average=True, reduce=True):
        super(_PairwiseImageLoss, self).__init__()
        self._size_average = size_average
        self._reduce = reduce
        self._name = "parent"

        self._warped_moving_image = None
        self._warped_moving_mask = None
        self._weight = 1

        self._moving_image = moving_image
        self._moving_mask = moving_mask
        self._fixed_image = fixed_image
        self._fixed_mask = fixed_mask
        self._grid = None

        assert self._moving_image != None and self._fixed_image != None
        # TODO allow different image size for each image in the future
        assert self._moving_image.size == self._fixed_image.size
        assert self._moving_image.device == self._fixed_image.device
        assert len(self._moving_image.size) == 2 or len(
            self._moving_image.size) == 3

        self._grid = T.utils.compute_grid(self._moving_image.size, dtype=self._moving_image.dtype,
                                          device=self._moving_image.device)

        self._dtype = self._moving_image.dtype
        self._device = self._moving_image.device

    @property
    def name(self):
        return self._name

    def GetWarpedImage(self):
        return self._warped_moving_image[0, 0, ...].detach().cpu()

    def GetCurrentMask(self, displacement):
        """
        Computes a mask defining if pixels are warped outside the image domain, or if they fall into
        a fixed image mask or a warped moving image mask.
        return (Tensor): maks array
        """
        # exclude points which are transformed outside the image domain
        mask = th.zeros_like(self._fixed_image.image,
                             dtype=th.uint8, device=self._device)
        for dim in range(displacement.size()[-1]):
            mask += displacement[...,
                                 dim].gt(1) + displacement[..., dim].lt(-1)

        mask = mask == 0

        # and exclude points which are masked by the warped moving and the fixed mask
        if not self._moving_mask is None:
            self._warped_moving_mask = F.grid_sample(
                self._moving_mask.image, displacement)
            self._warped_moving_mask = self._warped_moving_mask >= 0.5

            # if either the warped moving mask or the fixed mask is zero take zero,
            # otherwise take the value of mask
            if not self._fixed_mask is None:
                mask = th.where(((self._warped_moving_mask == 0) | (
                    self._fixed_mask == 0)), th.zeros_like(mask), mask)
            else:
                mask = th.where((self._warped_moving_mask == 0),
                                th.zeros_like(mask), mask)

        return mask

    def set_loss_weight(self, weight):
        self._weight = weight

    # conditional return
    def return_loss(self, tensor):
        if self._size_average and self._reduce:
            return tensor.mean()*self._weight
        if not self._size_average and self._reduce:
            return tensor.sum()*self._weight
        if not self.reduce:
            return tensor*self._weight


class FeatureSpaceMSE:
    """
    Mean squared error loss of the features.
    """

    def __init__(self, feature_extractor: FeatureExtractor.FeatureExtractor):
        self.feature_extractor = feature_extractor
        self.f_true = None

    def loss(self, y_true, y_pred):
        if self.f_true == None:
            self.f_true = self.feature_extractor.compute_features(
                y_true)  # TODO perform feature extraction in torch not np
        f_pred = self.feature_extractor.compute_features(y_pred)

        # fixed = self.feature_extractor.features_pca(self.f_true, 1)
        # moving = self.feature_extractor.features_pca(f_pred, 1)

        # return np.mean((fixed - moving) ** 2)

        return torch.mean((self.f_true - f_pred) ** 2)


class Dino(_PairwiseImageLoss):
    def __init__(self, fixed_image, moving_image, dimensions, fixed_mask=None, moving_mask=None, size_average=True, reduce=True):
        super(Dino, self).__init__(fixed_image, moving_image,
                                   fixed_mask, moving_mask, size_average, reduce)

        self.dimensions = dimensions

        self._name = "dino"

        self.warped_moving_image = None

        self.feature_extractor = FeatureExtractor.FeatureExtractor("DINOv2", {
                                                                   "slice_dist": 0})
        self.loss = FeatureSpaceMSE(self.feature_extractor).loss

    def forward(self, displacement):

        # compute displacement field
        displacement = self._grid + displacement

        # warp moving image with dispalcement field
        self.warped_moving_image = F.grid_sample(
            self._moving_image.image, displacement)  # .squeeze()

        # upsample both with torch
        val = 14
        scale_factor = [val, val]

        fixed = F.interpolate(
            self._fixed_image.image, scale_factor=scale_factor, mode='bilinear').squeeze()
        moving = F.interpolate(
            self.warped_moving_image, scale_factor=scale_factor, mode='bilinear').squeeze()

        final_loss = self.loss(fixed.detach().cpu().numpy(),
                               moving.detach().cpu().numpy()) * self._weight

        return final_loss


class Random(_PairwiseImageLoss):

    def __init__(self, fixed_image, moving_image, fixed_mask=None, moving_mask=None, size_average=True, reduce=True):
        super(Random, self).__init__(fixed_image, moving_image,
                                     fixed_mask, moving_mask, size_average, reduce)

        self._name = "random"

        self.warped_moving_image = None

        self.feature_extractor = FeatureExtractor.FeatureExtractor("DINOv2", self._device, {
                                                                   "slice_dist": 0})
        self.loss = FeatureSpaceMSE(self.feature_extractor).loss

    def forward(self, displacement):

        # compute displacement field
        displacement = self._grid + displacement

        # warp moving image with dispalcement field
        self.warped_moving_image = F.grid_sample(
            self._moving_image.image, displacement)

        moving = self.warped_moving_image
        fixed = self._fixed_image.image

        # upsample both with torch
        val = 1
        scale_factor = [val, val]

        fixed = F.interpolate(
            self._fixed_image.image, scale_factor=scale_factor, mode='bilinear').squeeze()
        moving = F.interpolate(
            self.warped_moving_image, scale_factor=scale_factor, mode='bilinear').squeeze()

        return_val2 = self.loss(fixed, moving) * self._weight

        # compute squared differences
        value = (moving - fixed).pow(2)
        return_val1 = torch.mean(value)

        return return_val2

        # return


class MSE(_PairwiseImageLoss):

    r""" The mean square error loss is a simple and fast to compute point-wise measure
    which is well suited for monomodal image registration.

    .. math::
         \mathcal{S}_{\text{MSE}} := \frac{1}{\vert \mathcal{X} \vert}\sum_{x\in\mathcal{X}}
          \Big(I_M\big(x+f(x)\big) - I_F\big(x\big)\Big)^2

    Args:
        fixed_image (Image): Fixed image for the registration
        moving_image (Image): Moving image for the registration
        size_average (bool): Average loss function
        reduce (bool): Reduce loss function to a single value

    """

    def __init__(self, fixed_image, moving_image, fixed_mask=None, moving_mask=None, size_average=True, reduce=True):
        super(MSE, self).__init__(fixed_image, moving_image,
                                  fixed_mask, moving_mask, size_average, reduce)

        self._name = "mse"

        self.warped_moving_image = None

    def forward(self, displacement):

        # compute displacement field
        displacement = self._grid + displacement

        # compute current mask
        mask = super(MSE, self).GetCurrentMask(displacement)

        # warp moving image with dispalcement field
        self.warped_moving_image = F.grid_sample(
            self._moving_image.image, displacement)

        # compute squared differences
        value = (self.warped_moving_image - self._fixed_image.image).pow(2)

        # mask values
        value = th.masked_select(value, mask)

        return self.return_loss(value)
