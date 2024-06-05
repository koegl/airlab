from enum import Enum

import matplotlib.pyplot as plt
import numpy as np


class DimensionalityReductionType(Enum):
    # linear
    PCA = "PCA"
    ICA = "ICA"

    # non-linear
    ISOMAP = "Isomap"
    TSNE = "TSNE"


def show_two_feature_maps(input_1: tuple[np.ndarray, np.ndarray],
                          input_2: tuple[np.ndarray, np.ndarray],
                          patch_h: int,
                          patch_w: int,
                          method: DimensionalityReductionType,
                          case_name: str,
                          save: bool = False) -> None:
    """
    Show the fixed and moving features.s
    """

    image_1 = input_1[0]
    features_1 = input_1[1]

    image_2 = input_2[0]
    features_2 = input_2[1]

    assert features_1.shape == features_2.shape
    assert image_1.shape == image_2.shape

    num_dims = features_1.shape[1]

    windows = num_dims + 1

    _, axs = plt.subplots(2, windows, figsize=(windows * 3, 6))

    axs[0, 0].imshow(image_1, cmap='gray')
    axs[1, 0].imshow(image_2, cmap='gray')

    for i in range(num_dims):

        features_1[:, i] = min_max_normalization(features_1[:, i])
        features_2[:, i] = min_max_normalization(features_2[:, i])

        reshaped_1 = features_1[: patch_h*patch_w, i].reshape(patch_h, patch_w)
        reshaped_2 = features_2[: patch_h*patch_w, i].reshape(patch_h, patch_w)

        # +1 because in the 0th position the image is shown
        axs[0, i+1].imshow(reshaped_1)
        axs[1, i+1].imshow(reshaped_2)

    plt.suptitle(
        f"Dimensionality reduction method: {method.value} [{case_name}]")

    plt.tight_layout()

    if save:
        plt.savefig(
            f'/u/home/koeglf/Documents/code/featureReg/copy/feature_maps_{method.value}_{case_name}_d{num_dims}.jpg')
    plt.show()


def min_max_normalization(features: np.ndarray) -> np.ndarray:
    """
    Normalize the features to the range [0, 1].
    """

    min_val = features.min()
    max_val = features.max()

    features = (features - min_val) / (max_val - min_val)

    return features
