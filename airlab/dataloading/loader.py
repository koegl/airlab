from pathlib import Path

from typing import List, Tuple

import numpy as np
from PIL import Image
import torch as th

from ..utils import image as iutils


class Fire():
    """
    Dataloader for the FIRE dataset.
    """

    def __init__(self, path_base: Path, dtype: th.dtype, device: th.device, output_type: str):
        """
        """

        if output_type not in ["images", "keypoints"]:
            raise ValueError(
                "output_type must be either 'images' or 'keypoints'")

        self.dtype = dtype
        self.device = device

        self.output_type = output_type

        self.list_of_image_pair_paths = []
        self.list_of_keypoint_paths = []

        self._load_paths(path_base)

    def __len__(self):
        """
            Return the number of transformations.
        """

        if self.output_type == "images":
            return len(self.list_of_image_pair_paths)
        else:
            return len(self.list_of_keypoint_paths)

    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """
              Return the path to the transformation at index idx.
          """
        if self.output_type == "images":
            return self._get_image_pair(idx)
        else:
            return self._get_keypoints_pair(idx)

    def _get_image_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        path_fixed = self.list_of_image_pair_paths[idx][0]
        path_moving = self.list_of_image_pair_paths[idx][1]

        # load the images
        image_fixed = np.array(Image.open(
            path_fixed).convert('L').resize((256, 256)))
        image_moving = np.array(Image.open(
            path_moving).convert('L').resize((256, 256)))

        # load the images
        image_fixed = np.array(Image.open(
            path_fixed).convert('L'))
        image_moving = np.array(Image.open(
            path_moving).convert('L'))

        # min max normalization 0-1
        image_fixed = (image_fixed - np.min(image_fixed)) / \
            (np.max(image_fixed) - np.min(image_fixed))
        image_moving = (image_moving - np.min(image_moving)) / \
            (np.max(image_moving) - np.min(image_moving))

        image_fixed = iutils.image_from_numpy(
            image_fixed, [1, 1], [0, 0], dtype=self.dtype, device=self.device)
        image_moving = iutils.image_from_numpy(
            image_moving, [1, 1], [0, 0], dtype=self.dtype, device=self.device)

        return image_fixed, image_moving

    def _get_keypoints_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        file_path = self.list_of_keypoint_paths[idx]

        data = np.loadtxt(file_path.as_posix())

        # Assuming the original dimensions are known, you can hardcode them or make them configurable
        # replace with actual dimensions if different
        original_width, original_height = 2912, 2912

        # Scaling factors
        scale_x = 256 / original_width
        scale_y = 256 / original_height

        coords_fixed = data[:, [0, 1]]
        coords_moving = data[:, [2, 3]]

        # Scale the coordinates
        coords_fixed[:, 0] *= scale_x
        coords_fixed[:, 1] *= scale_y
        coords_moving[:, 0] *= scale_x
        coords_moving[:, 1] *= scale_y

        return coords_fixed, coords_moving

    def _load_paths(self, path_base: Path):

        if self.output_type == "images":
            self.list_of_image_pair_paths = self._load_path_pairs_images(
                path_base)
        else:
            self.list_of_keypoint_paths = self._load_paths_keypoints(
                path_base)

    def _load_path_pairs_images(self, path_base: Path) -> List[Tuple[Path, Path]]:
        """
            Load the list of transformations.
        """

        # get all deformations from path_result/deformations
        path_images = path_base / "Images"

        all_paths = list(path_images.glob("*.*"))

        # sort the list alphabetically
        all_paths.sort()

        # match consequitive images into pairs (tuples)
        path_pairs = [(all_paths[i], all_paths[i+1])
                      for i in range(0, len(all_paths), 2)]

        return path_pairs

    def _load_paths_keypoints(self, path_base: Path) -> List[Path]:
        # get all deformations from path_result/deformations
        path_images = path_base / "Ground Truth"

        all_paths = list(path_images.glob("*.*"))

        # sort the list alphabetically
        all_paths.sort()

        return all_paths
