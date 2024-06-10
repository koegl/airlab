
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
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # nopep8

import airlab as al

import warnings
warnings.filterwarnings('ignore')
warnings.filterwarnings("ignore", category=UserWarning)
sys.stderr = open(os.devnull, 'w')


torch.autograd.set_detect_anomaly(True)


def _get_image_pair(path_fixed: Path, path_moving: Path, dtype, device):

    # load the images
    fac = 2.0
    image_fixed = np.array(Image.open(
        path_fixed).convert('L').resize((int(28*fac), int(28*fac))))
    image_moving = np.array(Image.open(
        path_moving).convert('L').resize((int(28*fac), int(28*fac))))
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


def deform_image(dtype, device, moving_image):
    sigma = [50, 50]

    # define the transformation
    transformation = al.transformation.pairwise.BsplineTransformation(moving_image.size,
                                                                      sigma=sigma,
                                                                      order=1,
                                                                      dtype=dtype,
                                                                      device=device)

    # change values of transformation.trans_parameters
    trans_parameters_clone = transformation.trans_parameters.clone()
    trans_parameters_clone[0, 0, 2, 2] = 0.5
    trans_parameters_clone[0, 1, 2, 2] = 0.5
    transformation.trans_parameters = torch.nn.Parameter(
        trans_parameters_clone)

    # create final result
    displacement = transformation.get_displacement()
    warped_image = al.transformation.utils.warp_image(
        moving_image, displacement)

    return warped_image


def pca(X: torch.Tensor, k: int) -> torch.Tensor:
    """
    Perform PCA on the input data X.

    Args:
    - X (torch.Tensor): The input data matrix of shape (n_samples, n_features).
    - k (int): The number of principal components to compute.

    Returns:
    - projected (torch.Tensor): The data projected onto the first k principal components.
    - components (torch.Tensor): The first k principal components.
    """
    # Center the data
    X_mean = torch.mean(X, dim=0)
    X_centered = X - X_mean

    # Compute covariance matrix
    covariance_matrix = torch.mm(
        X_centered.T, X_centered) / (X_centered.size(0) - 1)

    # Eigen decomposition
    eigenvalues, eigenvectors = torch.linalg.eig(
        covariance_matrix)
    eigenvalues = eigenvalues.real
    eigenvectors = eigenvectors.real

    # Sort eigenvalues and eigenvectors
    sorted_indices = torch.argsort(eigenvalues, descending=True)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Select the first k principal components
    components = sorted_eigenvectors[:, :k]

    # Project the data onto the first k principal components
    projected = torch.mm(X_centered, components)

    return projected


def main():

    start = time.time()

    # set the used data type
    dtype = th.float32
    # device = th.device("cpu")
    device = th.device("cuda:0")

    feature_extractor = al.dino.FeatureExtractor.FeatureExtractor("DINOv2",
                                                                  device,
                                                                  {"slice_dist": 0})

    _, image = _get_image_pair("/data/mnist/trainingSample/img_1.jpg",
                               "/data/mnist/trainingSample/img_1.jpg",
                               dtype,
                               device)

    image_size = image.size

    image_upscaled = F.interpolate(image.image,
                                   scale_factor=14,
                                   mode='bilinear').squeeze()

    features = feature_extractor.compute_features(image_upscaled)
    feature_dim1 = pca(features, 1)
    feature_dim1 = feature_extractor.min_max_normalization(feature_dim1)
    feature_dim1 = al.utils.image_from_numpy(feature_dim1.detach().cpu().numpy().reshape(image_size),
                                             [1, 1],
                                             [0, 0],
                                             dtype=dtype,
                                             device=device)

    image_deformed = deform_image(dtype, device, image)
    image_deformed_upscaled = F.interpolate(image_deformed.image,
                                            scale_factor=14,
                                            mode='bilinear').squeeze()

    # DONE HERE enocde image, deform feature
    feature_dim1_deformed = deform_image(dtype, device, feature_dim1)

    features_of_image_deformed = feature_extractor.compute_features(
        torch.Tensor(image_deformed_upscaled))
    features_of_image_deformed_dim1 = pca(features_of_image_deformed, 1)
    features_of_image_deformed_dim1 = feature_extractor.min_max_normalization(
        features_of_image_deformed_dim1)

    # DONE HERE deform image, encode image
    features_of_image_deformed_dim1 = al.utils.image_from_numpy(features_of_image_deformed_dim1.detach().cpu().numpy().reshape(image_size),
                                                                [1, 1],
                                                                [0, 0],
                                                                dtype=dtype,
                                                                device=device)

    # plot the results
    plt.figure(figsize=(20, 4))

    plt.subplot(151)
    plt.imshow(image.numpy(), cmap='gray')
    plt.title('Fixed Image')

    plt.subplot(152)
    plt.imshow(feature_dim1.numpy(), cmap='gray')
    plt.title('Feature')

    plt.subplot(153)
    plt.imshow(feature_dim1_deformed.numpy())
    plt.title('encode->deform')

    plt.subplot(154)
    plt.imshow(features_of_image_deformed_dim1.numpy())
    plt.title('deform->encode')

    plt.subplot(155)
    plt.imshow(feature_dim1_deformed.numpy() -
               features_of_image_deformed_dim1.numpy(), cmap='gray')
    plt.title('Difference')

    # Increase horizontal distance between subplots
    # plt.subplots_adjust(wspace=0.5)

    plt.show()

    plt.savefig(
        "/u/home/koeglf/Documents/code/airlab/tmp/symmetric_features.jpg")

    end = time.time()

    print("=================================================================")

    print("Registration done in: ", end - start)

    x = 0


if __name__ == '__main__':

    main()
