from enum import Enum
from warnings import warn

from sklearn.decomposition import PCA, FastICA
from sklearn.manifold import Isomap, TSNE
import numpy as np


class DimensionalityReductionType(Enum):
    # linear
    PCA = "PCA"
    ICA = "ICA"

    # non-linear
    ISOMAP = "Isomap"
    TSNE = "TSNE"


def get_dimensionality_reduction_features(features: np.ndarray,
                                          n_components: int,
                                          **kwargs) -> np.ndarray:
    """
    Perform dimensionality reduction on the features.
    """

    items = list(kwargs.items())
    method = items[0][1]
    remaining = dict(items[1:])

    assert isinstance(method, DimensionalityReductionType)

    if method == DimensionalityReductionType.PCA:
        return get_pca_features(features, n_components, **remaining)
    if method == DimensionalityReductionType.ICA:
        return get_ica_features(features, n_components, **remaining)
    elif method == DimensionalityReductionType.ISOMAP:
        return get_isomap_features(features, n_components, **remaining)
    elif method == DimensionalityReductionType.TSNE:
        return get_tsne_features(features, n_components, **remaining)

    else:
        warn(
            f"Dimensionality reduction method {method.value} not implemented.")
        return features


def get_pca_features(features: np.ndarray, n_components: int, **kwargs) -> np.ndarray:
    """
    Perform PCA on the passed array with the specified number of output components.

    :param features: Original high-dimensional data as a NumPy array.
    :param n_components: Number of dimensions in the output.
    :return: Data transformed into a lower-dimensional space.
    """

    pca = PCA(n_components=n_components, **kwargs)
    pca.fit(features)

    features_pca = pca.transform(features)

    return features_pca


def get_isomap_features(features: np.ndarray, n_components: int, **kwargs) -> np.ndarray:
    """
    Perform Isomap on the passed array with the specified number of output components.

    :param features: Original high-dimensional data as a NumPy array.
    :param n_components: Number of dimensions in the output.
    :param **kwargs: Additional arguments for the Isomap model.
    :return: Data transformed into a lower-dimensional space.
    """

    # Initialize the Isomap model with the specified number of components and neighbors
    isomap = Isomap(n_components=n_components, **kwargs)

    # Fit the Isomap model to the data and transform it
    features_isomap = isomap.fit_transform(features)

    return features_isomap


def get_ica_features(features: np.ndarray, n_components: int, **kwargs) -> np.ndarray:
    """
    Perform ICA on the features to extract independent components.
    :param features: Original high-dimensional data as a NumPy array.
    :param n_components: Number of dimensions in the output.
    :return: Data transformed into a lower-dimensional space.
    """

    # Initialize the ICA model
    ica = FastICA(n_components=n_components, **kwargs)

    # Fit the ICA model and transform the data to get independent components
    features_ica = ica.fit_transform(features)

    return features_ica


def get_tsne_features(features: np.ndarray, n_components: int, **kwargs) -> np.ndarray:
    """
    Perform t-SNE on the passed array with the specified number of output components.

    :param features: Original high-dimensional data as a NumPy array.
    :param n_components: Number of dimensions in the output (typically 2 or 3 for visualization).
    :param kwargs: Additional keyword arguments for the TSNE model.
    :return: Data transformed into a lower-dimensional space.
    """
    # Initialize the TSNE model
    tsne = TSNE(n_components=n_components, **kwargs)

    # Fit and transform the data to the new lower-dimensional space
    features_tsne = tsne.fit_transform(features)

    return features_tsne
