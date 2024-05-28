
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

import sys
import os
import time
from pathlib import Path

from typing import Optional

import matplotlib.pyplot as plt
import torch as th
import numpy as np
from scipy.ndimage import map_coordinates

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # nopep8

import airlab as al


def deform_landmarks(moving_landmarks: np.ndarray, displacement: np.ndarray) -> np.ndarray:
    # Map the moving landmarks to the fixed landmarks using the displacement field
    mov_lms_disp_x = map_coordinates(
        displacement[:, :, 0], moving_landmarks.transpose())
    mov_lms_disp_y = map_coordinates(
        displacement[:, :, 1], moving_landmarks.transpose())
    mov_lms_disp = np.array(
        (mov_lms_disp_x, mov_lms_disp_y)).transpose()
    return moving_landmarks + mov_lms_disp


def tre(landmarks_fixed: np.ndarray,
        landmarks_moving: np.ndarray,
        displacement: np.ndarray,
        spacing_moving: list[float],
        percentile: Optional[float] = None) -> float:
    """
    Calculate the Target Registration Error (TRE) between two sets of landmarks.

    Args:
        landmarks_fixed (np.ndarray): The fixed landmarks. landmarks in shape (N,3)
        landmarks_moving (np.ndarray): The moving landmarks. landmarks in shape (N,3)
        displacement (np.ndarray): The displacement field. displacement in shape (h,w,d,[1], 3)
        spacing_moving (Tuple[float, float, float]): The spacing of the moving image.

    Returns:
        float: The mean TRE.
    """

    assert landmarks_moving.shape == landmarks_fixed.shape
    assert landmarks_fixed.shape[-1] == 2

    mov_lms_warped = deform_landmarks(landmarks_moving, displacement)

    # Calculate the TRE
    all_errors = np.linalg.norm((mov_lms_warped - landmarks_fixed)
                                * spacing_moving, axis=1)

    return all_errors


def main():
    start = time.time()

    # set the used data type
    dtype = th.float32
    # set the device for the computaion to CPU
    device = th.device("cpu")

    # In order to use a GPU uncomment the following line. The number is the device index of the used GPU
    # Here, the GPU with the index 0 is used.
    # device = th.device("cuda:0")

    loader = al.dataloading.loader.Fire(path_base=Path("/home/fryderyk/Documents/data/FIRE"),
                                        dtype=dtype,
                                        device=device,
                                        output_type="images")

    laoder_points = al.dataloading.loader.Fire(path_base=Path("/home/fryderyk/Documents/data/FIRE"),
                                               dtype=dtype,
                                               device=device,
                                               output_type="keypoints")

    fixed_image, moving_image = loader[0]
    points_fixed, points_moving = laoder_points[0]

    # create image pyramide size/4, size/2, size/1
    fixed_image_pyramid = al.create_image_pyramid(
        fixed_image, [[4, 4], [2, 2]])
    moving_image_pyramid = al.create_image_pyramid(
        moving_image, [[4, 4], [2, 2]])

    regularisation_weight = [1, 5, 50]
    number_of_iterations = [75, 50, 25]
    # number_of_iterations = [1, 1, 1]

    sigma = [[11, 11], [11, 11], [3, 3]]

    once = True

    for level, (mov_im_level, fix_im_level) in enumerate(zip(moving_image_pyramid, fixed_image_pyramid)):

        registration = al.PairwiseRegistration(verbose=True)

        # define the transformation
        transformation = al.transformation.pairwise.BsplineTransformation(mov_im_level.size,
                                                                          sigma=sigma[level],
                                                                          order=3,
                                                                          dtype=dtype,
                                                                          device=device,
                                                                          diffeomorphic=True)

        if once:
            once = False
            displacement_once = transformation.get_displacement().detach().numpy().squeeze()
            tre_before = tre(points_fixed, points_moving,
                             displacement_once, [1, 1])

        if level > 0:
            constant_flow = al.transformation.utils.upsample_displacement(constant_flow,
                                                                          mov_im_level.size,
                                                                          interpolation="linear")
            transformation.set_constant_flow(constant_flow)

        registration.set_transformation(transformation)

        # choose the Mean Squared Error as image loss
        image_loss = al.loss.pairwise.MSE(fix_im_level, mov_im_level)

        registration.set_image_loss([image_loss])

        # define the regulariser for the displacement
        regulariser = al.regulariser.displacement.DiffusionRegulariser(
            mov_im_level.spacing)
        regulariser.SetWeight(regularisation_weight[level])
        registration.set_regulariser_displacement([regulariser])

        # define the optimizer
        optimizer = th.optim.Adam(transformation.parameters())

        registration.set_optimizer(optimizer)
        registration.set_number_of_iterations(number_of_iterations[level])

        registration.start()

        constant_flow = transformation.get_flow()

    # create final result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(
        moving_image, displacement)
    displacement = al.create_displacement_image_from_image(
        displacement, moving_image)

    end = time.time()

    print("=================================================================")

    print("Registration done in: ", end - start)

    new_disp = transformation.get_displacement().detach().numpy().squeeze()

    tre_after = tre(points_fixed,
                    points_moving,
                    new_disp,
                    [1, 1])

    # print(
    #     f"TRE before:\t\tmin: {tre_before.min():.4f}\tmax: {tre_before.max():.4f}\tmean: {tre_before.mean():.4f}")
    # print(
    #     f"TRE after:\t\tmin: {tre_after.min():.4f}\tmax: {tre_after.max():.4f}\tmean: {tre_after.mean():.4f}")
    improvement_min = (tre_before.mean() - tre_after.mean()
                       ) / tre_before.mean() * 100
    improvement_max = (tre_before.max() - tre_after.max()) / \
        tre_before.max() * 100
    improvement_mean = (tre_before.mean() - tre_after.mean()
                        ) / tre_before.mean() * 100
    print(
        f"TRE improvements:\tmin: {improvement_min:.4f}%\tmax: {improvement_max:.4f}%\tmean: {improvement_mean:.4f}%")

    # plot the results
    plt.subplot(231)
    plt.imshow(fixed_image.numpy(), cmap='gray')
    plt.title('Fixed Image')

    plt.subplot(232)
    plt.imshow(warped_image.numpy(), cmap='gray')
    plt.title('Warped Moving Image')

    plt.subplot(233)
    plt.imshow(fixed_image.numpy() - moving_image.numpy(), cmap='gray')
    plt.title('Difference Fixed - Moving')

    plt.subplot(234)
    plt.imshow(moving_image.numpy(), cmap='gray')
    plt.title('Moving Image')

    plt.subplot(235)
    plt.imshow(displacement.magnitude().numpy(), cmap='jet')
    plt.title('Magnitude Displacement')

    # plot the results
    plt.subplot(236)
    plt.imshow(fixed_image.numpy() - warped_image.numpy(), cmap='gray')
    plt.title('Difference Fixed - Warped')

    plt.show()

    x = 0


if __name__ == '__main__':
    main()
