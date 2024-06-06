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

import torch as th
from torch.nn.parameter import Parameter
import torch.nn.functional as F
import numpy as np

from ..utils import kernelFunction as utils

from . import utils as tu

"""
    Base class for a transformation
"""


class _Transformation(th.nn.Module):
    def __init__(self, image_size, diffeomorphic=False, dtype=th.float32, device='cpu'):
        super(_Transformation, self).__init__()

        self._dtype = dtype
        self._device = device
        self._dim = len(image_size)
        self._image_size = np.array(image_size)
        self._constant_displacement = None
        self._diffeomorphic = diffeomorphic
        self._constant_flow = None

        self._compute_flow = None

        if self._diffeomorphic:
            self._diffeomorphic_calculater = tu.Diffeomorphic(
                image_size, dtype=dtype, device=device)
        else:
            self._diffeomorphic_calculater = None

    def get_flow(self):

        if self._constant_flow is None:
            return self._compute_flow().detach()
        else:
            return self._compute_flow().detach() + self._constant_flow

    def set_constant_flow(self, flow):
        self._constant_flow = flow

    def get_displacement_numpy(self):

        if self._dim == 2:
            return th.unsqueeze(self().detach(), 0).cpu().numpy()
        elif self._dim == 3:
            return self().detach().cpu().numpy()

    def get_displacement(self):
        return self().detach()

    # def get_current_displacement(self):
    #
    #     if self._dim == 2:
    #         return th.unsqueeze(self().detach(), 0).cpu().numpy()
    #     elif self._dim == 3:
    #         return self().detach().cpu().numpy()

    # def set_constant_displacement(self, displacement):
    #
    #     self._constant_displacement = displacement

    # def get_inverse_transformation(self, displacement):
    #     if self._diffeomorphic:
    #         if self._dim == 2:
    #             inv_displacement = self._diffeomorphic_calculater.calculate(displacement * -1)
    #         else:
    #             inv_displacement = self._diffeomorphic_calculater.calculate(displacement * -1)
    #     else:
    #         print("error displacement ")
    #         inv_displacement = None
    #
    #     return inv_displacement

    def get_inverse_displacement(self):

        flow = self._concatenate_flows(self._compute_flow()).detach()

        if self._diffeomorphic:
            inv_displacement = self._diffeomorphic_calculater.calculate(
                flow * -1)
        else:
            print("error displacement ")
            inv_displacement = None

        return inv_displacement

    def _compute_diffeomorphic_displacement(self, flow):

        return self._diffeomorphic_calculater.calculate(flow)

    def _concatenate_flows(self, flow):

        if self._constant_flow is None:
            return flow
        else:
            return flow + self._constant_flow


"""
    Base class for kernel transformations
"""


class _KernelTransformation(_Transformation):
    def __init__(self, image_size, diffeomorphic=False, dtype=th.float32, device='cpu'):
        super(_KernelTransformation, self).__init__(
            image_size, diffeomorphic, dtype, device)

        self._kernel = None
        self._stride = 1
        self._padding = 0
        self._displacement_tmp = None
        self._displacement = None

        assert self._dim == 2 or self._dim == 3

        if self._dim == 2:
            self._compute_flow = self._compute_flow_2d
        else:
            self._compute_flow = self._compute_flow_3d

    def get_current_displacement(self):

        if self._dim == 2:
            return th.unsqueeze(self._compute_displacement().detach(), 0).cpu().numpy()
        elif self._dim == 3:
            return self._compute_displacement().detach().cpu().numpy()

    def _initialize(self):

        cp_grid = np.ceil(np.divide(self._image_size,
                          self._stride)).astype(dtype=int)

        # new image size after convolution
        inner_image_size = np.multiply(
            self._stride, cp_grid) - (self._stride - 1)

        # add one control point at each side
        cp_grid = cp_grid + 2

        # image size with additional control points
        new_image_size = np.multiply(
            self._stride, cp_grid) - (self._stride - 1)

        # center image between control points
        image_size_diff = inner_image_size - self._image_size
        image_size_diff_floor = np.floor(
            (np.abs(image_size_diff)/2))*np.sign(image_size_diff)

        self._crop_start = image_size_diff_floor + \
            np.remainder(image_size_diff, 2)*np.sign(image_size_diff)
        self._crop_end = image_size_diff_floor

        cp_grid = [1, self._dim] + cp_grid.tolist()

        # create transformation parameters
        self.trans_parameters = Parameter(th.Tensor(*cp_grid))
        self.trans_parameters.data.fill_(0)

        # copy to gpu if needed
        self.to(dtype=self._dtype, device=self._device)

        # convert to integer
        self._padding = self._padding.astype(dtype=int).tolist()
        self._stride = self._stride.astype(dtype=int).tolist()

        self._crop_start = self._crop_start.astype(dtype=int)
        self._crop_end = self._crop_end.astype(dtype=int)

        size = [1, 1] + new_image_size.astype(dtype=int).tolist()
        self._displacement_tmp = th.empty(
            *size, dtype=self._dtype, device=self._device)

        size = [1, 1] + self._image_size.astype(dtype=int).tolist()
        self._displacement = th.empty(
            *size, dtype=self._dtype, device=self._device)

    def _compute_flow_2d(self):
        displacement_tmp = F.conv_transpose2d(self.trans_parameters, self._kernel,
                                              padding=self._padding, stride=self._stride, groups=2)

        def print_2d_tensor(tensor):
            """
            Print a 2D tensor in a readable format.

            Args:
            tensor (torch.Tensor): The 2D tensor to print.
            """
            if tensor.dim() != 2:
                raise ValueError("The input tensor must be 2-dimensional")

            # Convert to NumPy array if it's a PyTorch tensor
            tensor = tensor.cpu().detach().numpy()
            for row in tensor:
                print(" ".join(f"{elem:.4f}" for elem in row))
        # print("================================================================================================")
        # print_2d_tensor(self.trans_parameters.squeeze(0)[0, :, :])
        # print("------------------------------------------------------------------------------------------------")
        # print_2d_tensor(self.trans_parameters.squeeze(0)[1, :, :])
        # print("================================================================================================")
        # print("\n")

        # # Print gradients if they exist
        # if self.trans_parameters.grad is not None:
        #     print("Gradients for trans_parameters:")
        #     print_2d_tensor(self.trans_parameters.grad.squeeze(0)[0, :, :])
        #     print("------------------------------------------------------------------------------------------------")
        #     print_2d_tensor(self.trans_parameters.grad.squeeze(0)[1, :, :])
        #     print("================================================================================================")
        #     print("\n")

        # crop displacement
        return th.squeeze(displacement_tmp[:, :,
                                           self._stride[0] + self._crop_start[0]:-self._stride[0] - self._crop_end[0],
                                           self._stride[1] + self._crop_start[1]:-self._stride[1] - self._crop_end[1]].transpose_(1, 3).transpose(1, 2))

    def _compute_flow_3d(self):

        # compute dense displacement
        displacement = F.conv_transpose3d(self.trans_parameters, self._kernel,
                                          padding=self._padding, stride=self._stride, groups=3)

        # crop displacement
        return th.squeeze(displacement[:, :, self._stride[0] + self._crop_start[0]:-self._stride[0] - self._crop_end[0],
                                       self._stride[1] + self._crop_start[1]:-self._stride[1] - self._crop_end[1],
                                       self._stride[2] + self._crop_start[2]:-self._stride[2] - self._crop_end[2]
                                       ].transpose_(1, 4).transpose_(1, 3).transpose_(1, 2))

    def forward(self):

        flow = self._concatenate_flows(self._compute_flow())

        if self._diffeomorphic:
            displacement = self._compute_diffeomorphic_displacement(flow)
        else:
            displacement = flow

        return displacement


"""
    bspline kernel transformation
"""


class BsplineTransformation(_KernelTransformation):
    def __init__(self, image_size, sigma, diffeomorphic=False, order=2, dtype=th.float32, device='cpu'):
        super(BsplineTransformation, self).__init__(
            image_size, diffeomorphic, dtype, device)

        self._stride = np.array(sigma)

        # compute bspline kernel
        self._kernel = utils.bspline_kernel(
            sigma, dim=self._dim, order=order, asTensor=True, dtype=dtype)

        self._padding = (np.array(self._kernel.size()) - 1) / 2

        self._kernel.unsqueeze_(0).unsqueeze_(0)
        self._kernel = self._kernel.expand(
            self._dim, *((np.ones(self._dim + 1, dtype=int)*-1).tolist()))
        self._kernel = self._kernel.to(dtype=dtype, device=self._device)

        self._initialize()
