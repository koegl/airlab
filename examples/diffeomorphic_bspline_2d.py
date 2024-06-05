
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


def _get_image_pair(path_fixed: Path, path_moving: Path, dtype, device):

    # load the images
    image_fixed = np.array(Image.open(
        path_fixed).convert('L').resize((392, 392)))
    image_moving = np.array(Image.open(
        path_moving).convert('L').resize((392, 392)))

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


def _get_image_pair_np(path_fixed: Path, path_moving: Path, dtype, device):

    # load the images
    image_fixed = np.array(
        Image.fromarray(np.load(path_fixed)).resize((392, 392)))
    image_moving = np.array(
        Image.fromarray(np.load(path_moving)).resize((392, 392)))

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

    file_index = -1

    start = time.time()

    # set the used data type
    dtype = th.float32
    # set the device for the computaion to CPU
    device = th.device("cpu")

    # In order to use a GPU uncomment the following line. The number is the device index of the used GPU
    # Here, the GPU with the index 0 is used.
    device = th.device("cuda:0")

    """
    loader = al.dataloading.loader.Fire(path_base=Path("/home/fryderyk/Documents/data/FIRE"),
                                        dtype=dtype,
                                        device=device,
                                        output_type="images")

    laoder_points = al.dataloading.loader.Fire(path_base=Path("/home/fryderyk/Documents/data/FIRE"),
                                            dtype=dtype,
                                            device=device,
                                            output_type="keypoints")

    fixed_image, moving_image = loader[file_index]
    points_fixed, points_moving = laoder_points[file_index]
    """

    fixed_image, moving_image = _get_image_pair("/data/mnist/trainingSample/img_5.jpg",
                                                "/data/mnist/trainingSample/img_1.jpg",
                                                dtype,
                                                device)

    # fixed_image, moving_image = _get_image_pair_np("/u/home/koeglf/Documents/code/airlab/tmp/fixed.npy",
    #                                                "/u/home/koeglf/Documents/code/airlab/tmp/moving.npy",
    #                                                dtype,
    #                                                device)

    # create image pyramide size/4, size/2, size/1
    fixed_image_pyramid = al.create_image_pyramid(
        fixed_image, [[4, 4], [2, 2]])
    moving_image_pyramid = al.create_image_pyramid(
        moving_image, [[4, 4], [2, 2]])

    regularisation_weight = [10, 50, 500]
    number_of_iterations = list(np.array([4, 2, 1]) * 32)
    # number_of_iterations = [1, 1, 1]

    sigma = [[20, 20], [12, 12], [4, 4]]
    # sigma = [[20, 20], [12, 12], [4, 4]]

    model = al.dino.utils_dino.get_model()
    for level, (mov_im_level, fix_im_level) in enumerate(zip(moving_image_pyramid, fixed_image_pyramid)):

        registration = al.PairwiseRegistration(verbose=True)

        # define the transformation
        transformation = al.transformation.pairwise.BsplineTransformation(mov_im_level.size,
                                                                          sigma=sigma[level],
                                                                          order=1,
                                                                          dtype=dtype,
                                                                          device=device,
                                                                          diffeomorphic=True)

        if level > 0:
            constant_flow = al.transformation.utils.upsample_displacement(constant_flow,
                                                                          mov_im_level.size,
                                                                          interpolation="linear")
            transformation.set_constant_flow(constant_flow)

        registration.set_transformation(transformation)

        # choose the Mean Squared Error as image loss
        # image_loss = al.loss.pairwise.MSE(fix_im_level, mov_im_level)
        image_loss = al.loss.pairwise.Dino(fix_im_level,
                                           mov_im_level,
                                           model,
                                           dimensions=1)

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

    new_disp = transformation.get_displacement().detach().cpu().numpy().squeeze()

    # tre_after = mu.tre(points_fixed,
    #                    points_moving,
    #                    new_disp,
    #                    [1, 1])

    # print(
    #     f"TRE before:\t\tmin: {tre_before.min():.4f}\tmax: {tre_before.max():.4f}\tmean: {tre_before.mean():.4f}")
    # print(
    #     f"TRE after:\t\tmin: {tre_after.min():.4f}\tmax: {tre_after.max():.4f}\tmean: {tre_after.mean():.4f}")
    # improvement_min = (tre_before.mean() - tre_after.mean()
    #                    ) / tre_before.mean() * 100
    # improvement_max = (tre_before.max() - tre_after.max()) / \
    #     tre_before.max() * 100
    # improvement_mean = (tre_before.mean() - tre_after.mean()
    #                     ) / tre_before.mean() * 100
    # print(
    #     f"TRE improvements:\tmin: {improvement_min:.4f}%\tmax: {improvement_max:.4f}%\tmean: {improvement_mean:.4f}%")

    # mu.plot_all_registration_results("/home/fryderyk/Documents/fig.jpg",
    #                                  moving_image.numpy(),
    #                                  fixed_image.numpy(),
    #                                  warped_image.numpy(),
    #                                  new_disp,
    #                                  moving_keypoints=points_moving,
    #                                  fixed_keypoints=points_fixed,
    #                                  pred_keypoints=mu.deform_landmarks(points_moving, new_disp))

    # plot the results
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

    # plot the results
    plt.subplot(236)
    plt.imshow(fixed_image.numpy() - warped_image.numpy(),
               cmap='gray', vmin=-0.5, vmax=0.5)
    plt.title('Difference Fixed - Warped')

    plt.show()

    plt.savefig("/u/home/koeglf/Documents/code/airlab/tmp/fig.jpg")

    x = 0


if __name__ == '__main__':

    main()
