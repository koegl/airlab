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

import torch
import torchvision
import torch.nn.functional as F

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


class _PairwiseImageLoss(torch.nn.modules.Module):
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
        mask = torch.zeros_like(self._fixed_image.image,
                                dtype=torch.uint8, device=self._device)
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
                mask = torch.where(((self._warped_moving_mask == 0) | (
                    self._fixed_mask == 0)), torch.zeros_like(mask), mask)
            else:
                mask = torch.where((self._warped_moving_mask == 0),
                                   torch.zeros_like(mask), mask)

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


class MI:
    """
    Mutual information.
    """

    def mi(self,
           fixed: torch.Tensor,
           moving: torch.Tensor,
           num_bins: int = 32) -> torch.Tensor:
        """
            Compute the mutual information between two images.

            Args:
            - fixed (torch.Tensor): The fixed image.
            - moving (torch.Tensor): The moving image.
            - num_bins (int): The number of bins in the histogram.

            Returns:
            - mi (torch.Tensor): The mutual information.
            """

        def soft_binning(values, num_bins):
            """
            Convert continuous values to discrete bins using a softmax approximation.

            Args:
            - values (torch.Tensor): Continuous values.
            - num_bins (int): Number of bins.

            Returns:
            - binned (torch.Tensor): Soft binned values.
            """
            min_val, _ = torch.min(values), torch.max(values)
            max_val, _ = torch.max(values), torch.min(values)
            values = (values - min_val) / (max_val -
                                           min_val + 1e-10)  # Normalize to [0, 1]
            values = values * (num_bins - 1)  # Scale to [0, num_bins-1]

            bin_edges = torch.linspace(
                0, num_bins - 1, num_bins).to(values.device)
            binned = torch.softmax(-((values.unsqueeze(1) -
                                   bin_edges.unsqueeze(0))**2), dim=1)

            return binned

        def joint_histogram_soft(fixed, moving, num_bins=32):
            """
            Compute the joint histogram for two images using soft binning.

            Args:
            - fixed (torch.Tensor): The fixed image.
            - moving (torch.Tensor): The moving image.
            - num_bins (int): The number of bins in the histogram.

            Returns:
            - joint_hist (torch.Tensor): The joint histogram.
            """
            assert fixed.shape == moving.shape, "Images must have the same shape"

            # Soft binning
            fixed_binned = soft_binning(fixed.flatten(), num_bins)
            moving_binned = soft_binning(moving.flatten(), num_bins)

            joint_hist = torch.matmul(fixed_binned.t(), moving_binned)
            joint_hist /= torch.sum(joint_hist)

            return joint_hist

        def marginal_histograms_soft(joint_hist):
            """
            Compute the marginal histograms from the joint histogram.

            Args:
            - joint_hist (torch.Tensor): The joint histogram.

            Returns:
            - fixed_hist (torch.Tensor): The marginal histogram for the fixed image.
            - moving_hist (torch.Tensor): The marginal histogram for the moving image.
            """
            fixed_hist = torch.sum(joint_hist, dim=1)
            moving_hist = torch.sum(joint_hist, dim=0)
            return fixed_hist, moving_hist

        def entropy(hist):
            """
            Compute the entropy of a histogram.

            Args:
            - hist (torch.Tensor): The histogram.

            Returns:
            - entropy (torch.Tensor): The entropy.
            """
            p = hist / torch.sum(hist)
            p = p[p > 0]
            return -torch.sum(p * torch.log(p + 1e-10))

        joint_hist = joint_histogram_soft(fixed, moving, num_bins)
        fixed_hist, moving_hist = marginal_histograms_soft(joint_hist)
        h_joint = entropy(joint_hist)
        h_fixed = entropy(fixed_hist)
        h_moving = entropy(moving_hist)
        mi_score = h_fixed + h_moving - h_joint

        return mi_score

    def loss(self, t1: torch.Tensor, t2: torch.Tensor):

        loss = - self.mi(t1.flatten(), t2.flatten())

        return loss


class LatentSpaceFeatureLoss(_PairwiseImageLoss):

    def __init__(self, fixed_image, moving_image, extractor: str, loss_type: str, fixed_mask=None, moving_mask=None, size_average=True, reduce=True):
        super(LatentSpaceFeatureLoss, self).__init__(fixed_image, moving_image,
                                                     fixed_mask, moving_mask, size_average, reduce)

        self._name = f"{extractor} with {loss_type} loss"

        self.warped_moving_image = None

        self.feature_extractor = FeatureExtractor.FeatureExtractor(extractor, self._device, {
                                                                   "slice_dist": 0})

        if loss_type == "MI":
            self.loss = MI().loss
        else:
            raise ValueError("Loss type not supported")

    def forward(self, displacement):

        # compute displacement field
        displacement = self._grid + displacement

        # warp moving image with dispalcement field
        self.warped_moving_image = F.grid_sample(
            self._moving_image.image, displacement)

        moving = self.warped_moving_image
        fixed = self._fixed_image.image

        dino_upscale_factor = 5

        # upsample both with torch
        scale_factor = [dino_upscale_factor,
                        dino_upscale_factor]

        fixed = F.interpolate(
            self._fixed_image.image, scale_factor=scale_factor, mode='bilinear').squeeze()
        moving = F.interpolate(
            self.warped_moving_image, scale_factor=scale_factor, mode='bilinear').squeeze()

        features_fixed = self.feature_extractor.compute_features(fixed)
        features_warped = self.feature_extractor.compute_features(moving)

        features_fixed = self.feature_extractor.min_max_normalization(
            features_fixed)
        features_warped = self.feature_extractor.min_max_normalization(
            features_warped)

        return_val2 = self.loss(features_fixed, features_warped) * self._weight

        return return_val2


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
        value = torch.masked_select(value, mask)

        return self.return_loss(value)
