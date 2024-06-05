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

from .. import transformation as T
from ..transformation import utils as tu
from ..utils import kernelFunction as utils
from ..dino import utils_dino, utils_feature_reduction, utils_visualisation
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


class Dino(_PairwiseImageLoss):
    def __init__(self, fixed_image, moving_image, model, dimensions, transform=None, fixed_mask=None, moving_mask=None, size_average=True, reduce=True):
        super(Dino, self).__init__(fixed_image, moving_image,
                                   fixed_mask, moving_mask, size_average, reduce)

        self.model = model
        self.dimensions = dimensions

        self._name = "dino"

        self.warped_moving_image = None

        if transform is None:
            self.transform = transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize(mean=0.5, std=0.2)
            ])
        else:
            self.transform = transform

    def forward(self, displacement):

        # compute displacement field
        displacement = self._grid + displacement

        # compute current mask
        mask = super(Dino, self).GetCurrentMask(displacement)

        # warp moving image with dispalcement field
        self.warped_moving_image = F.grid_sample(
            self._moving_image.image, displacement).squeeze()

        # value = th.sqrt(F.mse_loss(
        #     self.warped_moving_image, self._fixed_image.image))

        # # MSE
        # value_MSE = (self.warped_moving_image - self._fixed_image.image).pow(2)

        # # mask values

        # final_MSE = self.return_loss(value_MSE)

        # DINO STUFF

        encoded_image_fixed = utils_dino.prepare_input_2d_rgb(self._fixed_image.numpy(),
                                                              self.model.patch_size)
        encoded_image_warped = utils_dino.prepare_input_2d_rgb(self.warped_moving_image.detach().cpu().numpy(),
                                                               self.model.patch_size)

        features_fixed = utils_dino.run_dino_inference(self.model,
                                                       self.transform,
                                                       encoded_image_fixed)
        features_moving = utils_dino.run_dino_inference(self.model,
                                                        self.transform,
                                                        encoded_image_warped)

        features_extracted_fixed = utils_feature_reduction.get_dimensionality_reduction_features(features_fixed,
                                                                                                 self.dimensions,
                                                                                                 **{"method": utils_feature_reduction.DimensionalityReductionType.PCA})
        features_extracted_moving = utils_feature_reduction.get_dimensionality_reduction_features(features_moving,
                                                                                                  self.dimensions,
                                                                                                  **{"method": utils_feature_reduction.DimensionalityReductionType.PCA})

        for i in range(self.dimensions):
            features_extracted_fixed[:, i] = utils_visualisation.min_max_normalization(
                features_extracted_fixed[:, i])
            features_extracted_moving[:, i] = utils_visualisation.min_max_normalization(
                features_extracted_moving[:, i])

        # compute squared differences
        # move to gpu
        features_extracted_fixed = th.from_numpy(
            features_extracted_fixed).to(self._device)
        features_extracted_moving = th.from_numpy(
            features_extracted_moving).to(self._device)

        value = (features_extracted_fixed - features_extracted_moving).pow(2)

        patch_h = self._fixed_image.size[0] // self.model.patch_size
        patch_w = self._fixed_image.size[1] // self.model.patch_size

        # utils_visualisation.show_two_feature_maps((self._fixed_image.numpy(), features_extracted_fixed.detach().cpu().numpy()),
        #                                           (self.warped_moving_image.detach().cpu().numpy(),
        #                                            features_extracted_moving.detach().cpu().numpy()),
        #                                           patch_h,
        #                                           patch_w,
        #                                           utils_feature_reduction.DimensionalityReductionType.PCA,
        #                                           "dino",
        #                                           save=True)

        """
        value = value.reshape(patch_h, patch_w, -1)

        # Reshape to (1, dim, patch_h, patch_w) for interpolation
        value = value.permute(2, 0, 1).unsqueeze(0)

        # upscale value to the image size
        new_shape = (self._fixed_image.size[0], self._fixed_image.size[1])
        upscaled_tensor = F.interpolate(
            value, size=new_shape, mode='bilinear', align_corners=False)
        value = upscaled_tensor.squeeze(0).permute(1, 2, 0)

        # Broadcast the mask to match the shape of upscaled_tensor
        broadcasted_mask = mask.squeeze().unsqueeze(-1).expand(-1, -1, value.size(2))

        # mask values
        value = th.masked_select(value, broadcasted_mask)
        """

        return value.mean() * self._weight

        # return self.return_loss(value)
        # return value.mean()*self._weight


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


class NCC(_PairwiseImageLoss):
    r""" The normalized cross correlation loss is a measure for image pairs with a linear
         intensity relation.

        .. math::
            \mathcal{S}_{\text{NCC}} := \frac{\sum I_F\cdot (I_M\circ f)
                   - \sum\text{E}(I_F)\text{E}(I_M\circ f)}
                   {\vert\mathcal{X}\vert\cdot\sum\text{Var}(I_F)\text{Var}(I_M\circ f)}


        Args:
            fixed_image (Image): Fixed image for the registration
            moving_image (Image): Moving image for the registration

    """

    def __init__(self, fixed_image, moving_image, fixed_mask=None, moving_mask=None):
        super(NCC, self).__init__(fixed_image, moving_image,
                                  fixed_mask, moving_mask, False, False)

        self._name = "ncc"

        self.warped_moving_image = th.empty_like(
            self._moving_image.image, dtype=self._dtype, device=self._device)

    def forward(self, displacement):

        # compute displacement field
        displacement = self._grid + displacement

        # compute current mask
        mask = super(NCC, self).GetCurrentMask(displacement)

        self._warped_moving_image = F.grid_sample(
            self._moving_image.image, displacement)

        moving_image_valid = th.masked_select(self._warped_moving_image, mask)
        fixed_image_valid = th.masked_select(self._fixed_image.image, mask)

        value = -1.*th.sum((fixed_image_valid - th.mean(fixed_image_valid))*(moving_image_valid - th.mean(moving_image_valid)))\
            / th.sqrt(th.sum((fixed_image_valid - th.mean(fixed_image_valid))**2)*th.sum((moving_image_valid - th.mean(moving_image_valid))**2) + 1e-10)

        return value


class MI(_PairwiseImageLoss):
    r""" Implementation of the Mutual Information image loss.

         .. math::
            \mathcal{S}_{\text{MI}} := H(F, M) - H(F|M) - H(M|F)

        Args:
            fixed_image (Image): Fixed image for the registration
            moving_image (Image): Moving image for the registration
            bins (int): Number of bins for the intensity distribution
            sigma (float): Kernel sigma for the intensity distribution approximation
            spatial_samples (float): Percentage of pixels used for the intensity distribution approximation
            background: Method to handle background pixels. None: Set background to the min value of image
                                                            "mean": Set the background to the mean value of the image
                                                            float: Set the background value to the input value
            size_average (bool): Average loss function
            reduce (bool): Reduce loss function to a single value

    """

    def __init__(self, fixed_image, moving_image, fixed_mask=None, moving_mask=None, bins=64, sigma=3,
                 spatial_samples=0.1, background=None, size_average=True, reduce=True):
        super(MI, self).__init__(fixed_image, moving_image,
                                 fixed_mask, moving_mask, size_average, reduce)

        self._name = "mi"

        self._dim = fixed_image.ndim
        self._bins = bins
        self._sigma = 2*sigma**2
        self._normalizer_1d = np.sqrt(2.0 * np.pi) * sigma
        self._normalizer_2d = 2.0 * np.pi*sigma**2

        if background is None:
            self._background_fixed = th.min(fixed_image.image)
            self._background_moving = th.min(moving_image.image)
        elif background == "mean":
            self._background_fixed = th.mean(fixed_image.image)
            self._background_moving = th.mean(moving_image.image)
        else:
            self._background_fixed = background
            self._background_moving = background

        self._max_f = th.max(fixed_image.image)
        self._max_m = th.max(moving_image.image)

        self._spatial_samples = spatial_samples

        self._bins_fixed_image = th.linspace(self._background_fixed, self._max_f, self.bins,
                                             device=fixed_image.device, dtype=fixed_image.dtype).unsqueeze(1)

        self._bins_moving_image = th.linspace(self._background_moving, self._max_m, self.bins,
                                              device=fixed_image.device, dtype=fixed_image.dtype).unsqueeze(1)

    @ property
    def sigma(self):
        return self._sigma

    @ property
    def bins(self):
        return self._bins

    @ property
    def bins_fixed_image(self):
        return self._bins_fixed_image

    def _compute_marginal_entropy(self, values, bins):

        p = th.exp(-((values - bins).pow(2).div(self._sigma))
                   ).div(self._normalizer_1d)
        p_n = p.mean(dim=1)
        p_n = p_n/(th.sum(p_n) + 1e-10)

        return -(p_n * th.log2(p_n + 1e-10)).sum(), p

    def forward(self, displacement):

        # compute displacement field
        displacement = self._grid + displacement

        # compute current mask
        mask = super(MI, self).GetCurrentMask(displacement)

        self._warped_moving_image = F.grid_sample(
            self._moving_image.image, displacement)

        moving_image_valid = th.masked_select(self._warped_moving_image, mask)
        fixed_image_valid = th.masked_select(self._fixed_image.image, mask)

        mask = (fixed_image_valid > self._background_fixed) & (
            moving_image_valid > self._background_moving)

        fixed_image_valid = th.masked_select(fixed_image_valid, mask)
        moving_image_valid = th.masked_select(moving_image_valid, mask)

        number_of_pixel = moving_image_valid.shape[0]

        sample = th.zeros(number_of_pixel, device=self._fixed_image.device,
                          dtype=self._fixed_image.dtype).uniform_() < self._spatial_samples

        # compute marginal entropy fixed image
        image_samples_fixed = th.masked_select(
            fixed_image_valid.view(-1), sample)

        ent_fixed_image, p_f = self._compute_marginal_entropy(
            image_samples_fixed, self._bins_fixed_image)

        # compute marginal entropy moving image
        image_samples_moving = th.masked_select(
            moving_image_valid.view(-1), sample)

        ent_moving_image, p_m = self._compute_marginal_entropy(
            image_samples_moving, self._bins_moving_image)

        # compute joint entropy
        p_joint = th.mm(p_f, p_m.transpose(0, 1)).div(self._normalizer_2d)
        p_joint = p_joint / (th.sum(p_joint) + 1e-10)

        ent_joint = -(p_joint * th.log2(p_joint + 1e-10)).sum()

        return -(ent_fixed_image + ent_moving_image - ent_joint)


class NGF(_PairwiseImageLoss):
    r""" Implementation of the Normalized Gradient Fields image loss.

            Args:
                fixed_image (Image): Fixed image for the registration
                moving_image (Image): Moving image for the registration
                fixed_mask (Tensor): Mask for the fixed image
                moving_mask (Tensor): Mask for the moving image
                epsilon (float): Regulariser for the gradient amplitude
                size_average (bool): Average loss function
                reduce (bool): Reduce loss function to a single value

    """

    def __init__(self, fixed_image, moving_image, fixed_mask=None, moving_mask=None, epsilon=1e-5,
                 size_average=True,
                 reduce=True):
        super(NGF, self).__init__(fixed_image, moving_image,
                                  fixed_mask, moving_mask, size_average, reduce)

        self._name = "ngf"

        self._dim = fixed_image.ndim
        self._epsilon = epsilon

        if self._dim == 2:
            dx = (fixed_image.image[..., 1:, 1:] -
                  fixed_image.image[..., :-1, 1:]) * fixed_image.spacing[0]
            dy = (fixed_image.image[..., 1:, 1:] -
                  fixed_image.image[..., 1:, :-1]) * fixed_image.spacing[1]

            if self._epsilon is None:
                with th.no_grad():
                    self._epsilon = th.mean(th.abs(dx) + th.abs(dy))

            norm = th.sqrt(dx.pow(2) + dy.pow(2) + self._epsilon ** 2)

            self._ng_fixed_image = F.pad(
                th.cat((dx, dy), dim=1) / norm, (0, 1, 0, 1))

            self._ngf_loss = self._ngf_loss_2d
        else:
            dx = (fixed_image.image[..., 1:, 1:, 1:] -
                  fixed_image.image[..., :-1, 1:, 1:]) * fixed_image.spacing[0]
            dy = (fixed_image.image[..., 1:, 1:, 1:] -
                  fixed_image.image[..., 1:, :-1, 1:]) * fixed_image.spacing[1]
            dz = (fixed_image.image[..., 1:, 1:, 1:] -
                  fixed_image.image[..., 1:, 1:, :-1]) * fixed_image.spacing[2]

            if self._epsilon is None:
                with th.no_grad():
                    self._epsilon = th.mean(
                        th.abs(dx) + th.abs(dy) + th.abs(dz))

            norm = th.sqrt(dx.pow(2) + dy.pow(2) +
                           dz.pow(2) + self._epsilon ** 2)

            self._ng_fixed_image = F.pad(
                th.cat((dx, dy, dz), dim=1) / norm, (0, 1, 0, 1, 0, 1))

            self._ngf_loss = self._ngf_loss_3d

    def _ngf_loss_2d(self, warped_image):

        dx = (warped_image[..., 1:, 1:] - warped_image[...,
                                                       :-1, 1:]) * self._moving_image.spacing[0]
        dy = (warped_image[..., 1:, 1:] - warped_image[...,
                                                       1:, :-1]) * self._moving_image.spacing[1]

        norm = th.sqrt(dx.pow(2) + dy.pow(2) + self._epsilon ** 2)

        return F.pad(th.cat((dx, dy), dim=1) / norm, (0, 1, 0, 1))

    def _ngf_loss_3d(self, warped_image):

        dx = (warped_image[..., 1:, 1:, 1:] - warped_image[...,
                                                           :-1, 1:, 1:]) * self._moving_image.spacing[0]
        dy = (warped_image[..., 1:, 1:, 1:] - warped_image[...,
                                                           1:, :-1, 1:]) * self._moving_image.spacing[1]
        dz = (warped_image[..., 1:, 1:, 1:] - warped_image[...,
                                                           1:, 1:, :-1]) * self._moving_image.spacing[2]

        norm = th.sqrt(dx.pow(2) + dy.pow(2) + dz.pow(2) + self._epsilon ** 2)

        return F.pad(th.cat((dx, dy, dz), dim=1) / norm, (0, 1, 0, 1, 0, 1))

    def forward(self, displacement):

        # compute displacement field
        displacement = self._grid + displacement

        # compute current mask
        mask = super(NGF, self).GetCurrentMask(displacement)

        self._warped_moving_image = F.grid_sample(
            self._moving_image.image, displacement)

        # compute the gradient of the warped image
        ng_warped_image = self._ngf_loss(self._warped_moving_image)

        value = 0
        for dim in range(self._dim):
            value = value + \
                ng_warped_image[:, dim, ...] * \
                self._ng_fixed_image[:, dim, ...]

        value = 0.5 * th.masked_select(-value.pow(2), mask)

        return self.return_loss(value)
