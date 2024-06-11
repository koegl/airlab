
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

import torch
import sys
import os
import time
from pathlib import Path
import warnings

from typing import Optional

import matplotlib.pyplot as plt
import torch as th
import numpy as np
from PIL import Image
import myutils as mu

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # nopep8

import airlab as al

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning)
sys.stderr = open(os.devnull, 'w')


torch.autograd.set_detect_anomaly(True)


def _get_image_pair(path_fixed: Path, path_moving: Path, dtype, device):

    # load the images
    fac = 4
    image_fixed = np.array(Image.open(
        path_fixed).convert('L').resize((28*fac, 28*fac)))
    image_moving = np.array(Image.open(
        path_moving).convert('L').resize((28*fac, 28*fac)))
    # image_fixed = np.array(Image.open(
    #     path_fixed).convert('L'))
    # image_moving = np.array(Image.open(
    #     path_moving).convert('L'))

    # min max normalization 0-1
    image_fixed = (image_fixed - np.min(image_fixed)) / \
        (np.max(image_fixed) - np.min(image_fixed))
    image_moving = (image_moving - np.min(image_moving)) / \
        (np.max(image_moving) - np.min(image_moving))

    image_fixed = al.utils.image_from_numpy(
        image_fixed, [1, 1], [0, 0], dtype=dtype, device=device)
    image_moving = al.utils.image_from_numpy(
        image_moving, [1, 1], [0, 0], dtype=dtype, device=device)

    return image_fixed, image_moving


def main():

    start = time.time()

    # set the used data type
    dtype = th.float32
    # device = th.device("cpu")
    device = th.device("cuda:0")

    fixed_image, moving_image = _get_image_pair("/u/home/koeglf/Documents/code/airlab/airlab/data/fixed.png",
                                                "/u/home/koeglf/Documents/code/airlab/airlab/data/moving.png",
                                                dtype,
                                                device)

    regularisation_weight = 1
    number_of_iterations = 40

    sigma = [15, 15]

    registration = al.PairwiseRegistration(verbose=True)

    # define the transformation
    transformation = al.transformation.pairwise.RigidTransformation(
        moving_image)
    # transformation = al.transformation.pairwise.BsplineTransformation(moving_image.size,
    #                                                                   sigma=sigma,
    #                                                                   order=1,
    #                                                                   dtype=dtype,
    #                                                                   device=device,
    #                                                                   diffeomorphic=True)

    registration.set_transformation(transformation)

    # choose the Mean Squared Error as image loss
    # image_loss = al.loss.pairwise.MSE(fixed_image, moving_image)
    image_loss = al.loss.pairwise.LatentSpaceFeatureLoss(fixed_image,
                                                         moving_image,
                                                         extractor="DINOv2",
                                                         loss_type="CKA")

    registration.set_image_loss([image_loss])

    # define the regulariser for the displacement
    regulariser = al.regulariser.displacement.DiffusionRegulariser(
        moving_image.spacing)
    regulariser.SetWeight(regularisation_weight)
    registration.set_regulariser_displacement([regulariser])

    # define the optimizer
    optimizer = th.optim.Adam(
        transformation.parameters(), lr=0.01)

    registration.set_optimizer(optimizer)
    registration.set_number_of_iterations(number_of_iterations)

    registration.start()

    # create final result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(
        moving_image, displacement)
    displacement = al.create_displacement_image_from_image(
        displacement, moving_image)

    # plot the results
    plt.figure(figsize=(12, 6))

    plt.subplot(231)
    plt.imshow(fixed_image.numpy(), cmap='gray')
    plt.title('Fixed Image')

    plt.subplot(232)
    plt.imshow(warped_image.numpy(), cmap='gray')
    plt.title('Warped Moving Image')

    plt.subplot(233)
    plt.imshow(fixed_image.numpy() - moving_image.numpy(),
               cmap='gray', vmin=-0.5, vmax=0.5)
    plt.title('Difference Fixed - Moving')

    plt.subplot(234)
    plt.imshow(moving_image.numpy(), cmap='gray')
    plt.title('Moving Image')

    plt.subplot(235)
    plt.imshow(displacement.magnitude().numpy(), cmap='jet')
    plt.title('Magnitude Displacement')
    plt.colorbar(shrink=0.9)  # Decrease the size of the colorbar

    # plot the results
    plt.subplot(236)
    plt.imshow(fixed_image.numpy() - warped_image.numpy(),
               cmap='gray', vmin=-0.5, vmax=0.5)
    plt.title('Difference Fixed - Warped')

    # Increase horizontal distance between subplots
    plt.subplots_adjust(wspace=0.1, hspace=0.3)

    plt.show()

    plt.savefig(
        "/u/home/koeglf/Documents/code/airlab/tmp/resulting_registration.jpg")

    end = time.time()

    print("=================================================================")

    print("Registration done in: ", end - start)

    x = 0


if __name__ == '__main__':

    main()
