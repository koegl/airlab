from pathlib import Path
import warnings

import numpy as np

import torch
import torchio as tio


from PIL import Image


# Factor by which to reduce the images to reduce the memory usage.
MEMORY_FACTOR = 2


def get_model() -> torch.nn.Module:
    """
    Load the DINOv2 model from the Facebook Research hub.
    """

    # Suppress SparseEfficiencyWarning
    warnings.filterwarnings("ignore", category=UserWarning)

    dinov2_vitl14 = torch.hub.load(
        'facebookresearch/dinov2', 'dinov2_vitl14', pretrained=False)
    model_path = r"/u/home/koeglf/Documents/models/dinov2_vitl14_pretrain.pth"
    dinov2_vitl14.load_state_dict(torch.load(model_path))

    # this would download a 'fresh' model, in the code above we load a pretrained model (the weights are the same)
    # dinov2_vitl14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')

    # un ignore UserWarning
    warnings.filterwarnings("default", category=UserWarning)

    return dinov2_vitl14


def prepare_input_2d_rgb(image: np.ndarray,
                         patch_size: int) -> np.ndarray:
    """
    Prepare the input for the DINOv2 model. Each image is repeated 3 times in the last dimension to create a RGB image.

    @param image: image to prepare for DINOv2.
    @param patch_size: Patch size used in the DINOv2 model.
    """
    # create a tio subject with the passed image
    image = image[np.newaxis, ...,  np.newaxis]
    subject = tio.Subject({"image": tio.ScalarImage(tensor=image)})

    assert subject["image"].shape[0] == 1
    assert len(subject["image"].shape) == 4

    """
    # RESIZE THE SUBJECT - so we get higher resolution features
    # with those dimensions the output of the model is would be the same as the original input
    # but we are constrained by memory, so we need the memory_factor
    x_dim = subject["image"].shape[1] * patch_size
    y_dim = subject["image"].shape[2] * patch_size
    z_dim = subject["image"].shape[3]

    resize = tio.Resize((x_dim // MEMORY_FACTOR,
                         y_dim // MEMORY_FACTOR,
                         z_dim))  # this doesn't need the facotr, we don't upscale it

    subject = resize(subject)
    """

    # do min max scaling
    rescale = tio.RescaleIntensity(out_min_max=(0, 1), percentiles=(0, 100), in_min_max=(
        subject["image"].numpy().min(), subject["image"].numpy().max()))
    subject["image"] = rescale(subject["image"])

    img = subject["image"].numpy().squeeze()

    assert img.max() == 1
    assert img.min() == 0

    # repeat image 3times in last dimension
    rgb_img = (np.array(img)*255).astype(np.uint8)
    rgb_img = np.repeat(rgb_img[:, :, np.newaxis], 3, axis=2)

    return rgb_img


def run_dino_inference(model,
                       transform,
                       image: np.ndarray) -> np.ndarray:
    """
    Extract features from the DINOv2 model.
    """

    patch_size = model.patch_size  # patchsize=14
    feat_dim = 1024  # vitl14

    shape = image.shape

    patch_h = shape[0] // patch_size
    patch_w = shape[1] // patch_size

    features = None

    with torch.no_grad():
        image = transform(image)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
        image = image.to(device)

        features_dict = model.forward_features(image.unsqueeze(0))
        features = features_dict['x_norm_patchtokens']

    # move features to cpu
    features = features.cpu().numpy()

    features = features.reshape(patch_h * patch_w, feat_dim)

    return features


def load_and_scale_volume(path_volume: Path, dino_patch_size: int) -> np.ndarray:
    """
    Prepare the input volume by upscaling it and rescaling the intensities.

    params:
    path_volume: Path to the volume
    dino_patch_size: Patch size used in the DINOv2 model.
    """

    subject = tio.Subject({"image": tio.ScalarImage(path_volume)})

    assert subject["image"].shape[0] == 1
    assert len(subject["image"].shape) == 4

    # RESIZE THE SUBJECT - so we get higher resolution features
    # with those dimensions the output of the model is would be the same as the original input
    # but we are constrained by memory, so we need the memory_factor
    x_dim = subject["image"].shape[1] * dino_patch_size
    y_dim = subject["image"].shape[2] * dino_patch_size
    z_dim = subject["image"].shape[3]

    resize = tio.Resize((x_dim // MEMORY_FACTOR,
                         y_dim // MEMORY_FACTOR,
                         z_dim))  # this doesn't need the facotr, we don't upscale it

    subject = resize(subject)

    rescale = tio.RescaleIntensity(out_min_max=(0, 1),
                                   percentiles=(0, 100),
                                   in_min_max=(subject["image"].numpy().min(),
                                               subject["image"].numpy().max()))
    subject["image"] = rescale(subject["image"])

    data = subject["image"].numpy().squeeze()

    data = np.array(data*255).astype(np.uint8)

    return data


def encode_volume_into_rgb_slices(volume: np.ndarray) -> list[np.ndarray]:
    """
    Encode the volume into RGB slices.
    """

    shape = volume.shape

    list_of_rgbs = []

    start = 0
    stop = shape[-1] - (shape[-1] % 3)
    step = 3
    for i in range(start, stop, step):

        list_of_rgbs.append(volume[:, :, i:i+3])

    # if we have a remainder we have to repeat some slices
    # if we have a remainder of 1, we simply repeat the last slice 3 times
    # if we have a remainder of 2, we repeat the last slice 2 times
    remainder = shape[-1] % 3
    if remainder != 0:
        repeated = volume[:, :, -1]
        repeated = np.repeat(repeated[:, :, np.newaxis], 3, axis=2)

        if remainder == 2:
            repeated[:, :, 0] = volume[:, :, -2]

        list_of_rgbs.append(repeated)

    return list_of_rgbs
