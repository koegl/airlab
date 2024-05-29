from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
from typing import Optional

import pandas as pd
# plt.switch_backend('agg')
from matplotlib.colors import ListedColormap
from matplotlib.colors import Normalize
from scipy.ndimage import binary_erosion, map_coordinates
import matplotlib.colors as colors
import matplotlib.cm as cm
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
from scipy.spatial.distance import dice, directed_hausdorff
from scipy.spatial import KDTree
import nibabel as nib
import SimpleITK as sitk


# matplotlib.rcParams['text.usetex'] = True


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


def jacobian_determinant_from_displacement(displacement: np.ndarray) -> np.ndarray:
    displacement_image = sitk.GetImageFromArray(
        displacement.squeeze(), isVector=True)
    jacobian_determinant_image = sitk.DisplacementFieldJacobianDeterminant(
        displacement_image)
    return sitk.GetArrayFromImage(jacobian_determinant_image)


def plot_quiverplot(u: np.ndarray, axis: Optional[int] = None, ax=None) -> None:
    """
    :param u: (h,d,w,3)
    """

    def flow(slices_in,  # the 2D slices
             ax,
             titles=None,  # list of titles
             cmaps=None,  # list of colormaps
             width=15,  # width in in
             indexing='ij',  # plot vecs w/ matrix indexing 'ij' or cartesian indexing 'xy'
             img_indexing=True,  # whether to match the image view, i.e. flip y axis
             grid=False,  # option to plot the images in a grid or a single row
             show=True,  # option to actually show the plot (plt.show())
             quiver_width=None,
             plot_block=True,  # option to plt.show()
             scale=1):  # note quiver essentially draws quiver length = 1/scale
        '''
        plot a grid of flows (2d+2 images)
        '''

        # input processing
        nb_plots = len(slices_in)
        for slice_in in slices_in:
            assert len(
                slice_in.shape) == 3, 'each slice has to be 3d: 2d+2 channels'
            assert slice_in.shape[-1] == 2, 'each slice has to be 3d: 2d+2 channels'

        def input_check(inputs, nb_plots, name):
            ''' change input from None/single-link '''
            if not isinstance(inputs, (list, tuple)):
                inputs = [inputs]
            assert (inputs is None) or (len(inputs) == nb_plots) or (len(inputs) == 1), \
                'number of %s is incorrect' % name
            if inputs is None:
                inputs = [None]
            if len(inputs) == 1:
                inputs = [inputs[0] for i in range(nb_plots)]
            return inputs

        assert indexing in ['ij', 'xy']
        # Since img_indexing, indexing may modify slices_in in memory
        slices_in = np.copy(slices_in)

        if indexing == 'ij':
            for si, slc in enumerate(slices_in):
                # Make y values negative so y-axis will point down in plot
                slices_in[si][:, :, 1] = -slices_in[si][:, :, 1]

        if img_indexing:
            for si, slc in enumerate(slices_in):
                # Flip vertical order of y values
                slices_in[si] = np.flipud(slc)

        titles = input_check(titles, nb_plots, 'titles')
        cmaps = input_check(cmaps, nb_plots, 'cmaps')
        scale = input_check(scale, nb_plots, 'scale')

        # figure out the number of rows and columns
        if grid:
            if isinstance(grid, bool):
                rows = np.floor(np.sqrt(nb_plots)).astype(int)
                cols = np.ceil(nb_plots / rows).astype(int)
            else:
                assert isinstance(grid, (list, tuple)), \
                    "grid should either be bool or [rows,cols]"
                rows, cols = grid
        else:
            rows = 1
            cols = nb_plots

        # # prepare the subplot
        # fig, axs = plt.subplots(rows, cols)
        # if rows == 1 and cols == 1:
        #     axs = [axs]

        for i in range(nb_plots):
            col = np.remainder(i, cols)
            row = np.floor(i / cols).astype(int)

            # get row and column axes
            # row_axs = axs if rows == 1 else axs[row]
            # ax = row_axs[col]

            # turn off axis
            ax.axis('off')

            # add titles
            if titles is not None and titles[i] is not None:
                ax.title.set_text(titles[i])

            u, v = slices_in[i][..., 0], slices_in[i][..., 1]
            colors = np.arctan2(u, v)
            colors[np.isnan(colors)] = 0
            norm = Normalize()
            norm.autoscale(colors)
            if cmaps[i] is None:
                colormap = cm.winter
            else:
                raise Exception(
                    "custom cmaps not currently implemented for plt.flow()")

            # show figure
            colormap = cm.hsv
            ax.quiver(u, v,
                      color=colormap(norm(colors).flatten()),
                      angles='xy',
                      units='xy',
                      width=quiver_width,
                      scale=scale[i])
            ax.axis('equal')

    if u.shape[-1] == 3:
        assert axis is not None
        axes = [0, 1, 2]
        axes.remove(axis)
        u = u[..., axes].take(u.shape[axis] // 2, axis=axis)
        # print(u.shape)

    flow([u], show=False, ax=ax, scale=3)
    # f, axs =neurite.plot.flow([u_slice], show=False, ax=ax)


def plot_deformation_field(ax: plt.Axes,
                           disp: np.ndarray,
                           background: Optional[np.ndarray] = None,
                           interval: Optional[int] = 3,
                           title: Optional[str] = None,
                           color: Optional[str] = 'cornflowerblue') -> None:
    """
    Plots 2d warped grid from a displacement field to a given matplotlib axis. source: https://github.com/qiuhuaqi/midir
    :param ax: axis of a plt plot
    :param disp: displacement field of size (2,H,W)
    :param background:
    :param interval:
    :param background:  background image plotted behind the warped grid
    :param color: color of grid lines
    :return: None
    """
    if background is not None:
        background = background
    else:
        background = np.zeros(disp.shape[1:])

    id_grid_H, id_grid_W = np.meshgrid(range(0, background.shape[0] - 1, interval),
                                       range(
                                           0, background.shape[1] - 1, interval),
                                       indexing='ij')

    new_grid_H = id_grid_H + disp[0, id_grid_H, id_grid_W]
    new_grid_W = id_grid_W + disp[1, id_grid_H, id_grid_W]

    kwargs = {"linewidth": 0.34, "color": color}
    for i in range(new_grid_H.shape[0]):
        ax.plot(new_grid_W[i, :], new_grid_H[i, :], **
                kwargs)  # each draws a horizontal line
    for i in range(new_grid_H.shape[1]):
        ax.plot(new_grid_W[:, i], new_grid_H[:, i], **
                kwargs)  # each draws a vertical line

    ax.set_title(title)
    ax.imshow(background, cmap='gray')
    # ax.set_aspect(1.25/1.75)
    ax.grid(False)
    ax.margins(x=0, y=0)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)


def multilabel_to_boundary(label_map: np.ndarray):
    label_map = label_map.astype(int)
    num_labels = np.max(label_map)
    boundary_label_map = np.zeros_like(label_map)

    # Iterate over each label
    for label in range(1, num_labels + 1):
        # Create a binary mask for the current label
        label_mask = (label_map == label)

        # Apply binary erosion to the binary mask
        eroded_label_mask = binary_erosion(label_mask)

        # Compute the boundary mask for the current label
        boundary_mask = label_mask.astype(
            np.int8) - eroded_label_mask.astype(np.int8)

        # Assign the boundary mask to the boundary label map
        boundary_label_map += boundary_mask * label
    return boundary_label_map


def plot_all_registration_results(save_path: Path,
                                  moving_image: np.ndarray,
                                  fixed_image: np.ndarray,
                                  pred_image: np.ndarray,
                                  displacement: np.ndarray,
                                  fixed_labels: Optional[np.array] = None,
                                  pred_labels: Optional[np.array] = None,
                                  moving_keypoints: Optional[np.array] = None,
                                  fixed_keypoints: Optional[np.array] = None,
                                  pred_keypoints: Optional[np.ndarray] = None,
                                  title: Optional[str] = None) -> None:
    """
    plots a figure with 9x3 subplots. Half-slices used for plots in each dimension.
    rows: dims
    cols: M,F, warpedM,phi_grid, phi_quiver, diff_before, diff_after, labels, jacdet

    @param moving_image:
    @param fixed_image:
    @param pred_image:
    @param displacement: (h,w,d,3)
    @param fixed_labels:
    @param pred_labels:
    @param moving_keypoints:
    @param fixed_keypoints:
    @param pred_keypoints:
    @param title:
    @return: plot
    """

    assert displacement.shape[-1] == 2
    assert displacement.ndim == 3

    fig = plt.figure(figsize=(40, 7))
    if title:
        fig.suptitle(title)
    image_size = moving_image.shape
    image_dim = len(image_size)

    jacobian_determinant = jacobian_determinant_from_displacement(displacement)

    toprow = True
    # moving image

    d = 1

    ax = fig.add_subplot(3, 9, (9 * d) + 1)
    ax.imshow(moving_image, cmap='gray')
    if moving_keypoints is not None:
        ax.scatter(moving_keypoints[:, 0],
                   moving_keypoints[:, 1], marker='x', c='red')
    if toprow:
        ax.title.set_text("M")
    plt.axis('off')

    # fixed image
    ax = fig.add_subplot(3, 9, (9 * d) + 2)
    ax.imshow(fixed_image, cmap='gray')
    if fixed_keypoints is not None:
        ax.scatter(fixed_keypoints[:, 0],
                   fixed_keypoints[:, 1], marker='x', c='red')
    if toprow:
        ax.title.set_text("F")
    plt.axis('off')

    # deformed image
    ax = fig.add_subplot(3, 9, (9 * d) + 3)
    ax.imshow(pred_image, cmap='gray')
    if pred_keypoints is not None:
        ax.scatter(pred_keypoints[:, 0],
                   pred_keypoints[:, 1], marker='x', c='red')
    if toprow:
        ax.title.set_text("warped M")
    plt.axis('off')

    # displacement field
    ax = fig.add_subplot(3, 9, (9 * d) + 4)
    fieldAx = displacement
    # plot_quiverplot(fieldAx, ax=ax)
    plot_deformation_field(
        ax, 1 * fieldAx.transpose(2, 0, 1), pred_image, interval=5, color="white")
    ax.set_frame_on(False)
    if toprow:
        ax.title.set_text("deformation")
    plt.axis('off')
    # fig.tight_layout()
    """
    # difference image before registration
    ax = fig.add_subplot(3, 9, (9 * d) + 5)
    diff_image = fixed_image - moving_image
    ax.imshow(diff_image, cmap='gray')
    ax.set_frame_on(False)
    if toprow:
        ax.title.set_text("diff image")
    plt.axis('off')
    # fig.tight_layout()

    # difference image after registration
    ax = fig.add_subplot(3, 9, (9 * d) + 6)
    diff_image = fixed_image - pred_image
    ax.imshow(diff_image, cmap='gray')
    ax.set_frame_on(False)
    if toprow:
        ax.title.set_text("diff image after")
    plt.axis('off')

    # boundaries
    ax = fig.add_subplot(3, 9, (9 * d) + 7)
    if toprow:
        ax.title.set_text("segmentations")

    # jacobian determinant, negative values shown in red
    ax = fig.add_subplot(3, 9, (9 * d) + 8)
    jacdet_d = jacobian_determinant
    jacdet_d[jacdet_d < 0] = np.min(jacdet_d)
    norm = colors.TwoSlopeNorm(
        vmin=-np.max(jacdet_d), vmax=np.max(jacdet_d), vcenter=0)
    im1 = ax.imshow(jacdet_d, cmap="RdBu", norm=norm)
    ax.set_frame_on(False)
    if toprow:
        ax.title.set_text("jac det")
    plt.axis('off')
    plt.colorbar(im1, ax=ax)
    """

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.01, hspace=0.01)

    plt.show()
    fig.savefig(save_path)

    x = 0


def plot_quantitative_results(df: pd.DataFrame, plot_path: Path):
    df = df.drop(["min", "max", "mean", "std"])
    df = df.astype(float)

    # box plots
    df.plot(kind='box', subplots=True, figsize=(40, 7))
    plt.savefig(plot_path, dpi=900)
