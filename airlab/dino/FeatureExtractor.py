from PIL import Image
from torchvision import transforms
import torch
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional


class FeatureExtractor():

    def __init__(self, encoder_type: str, device, args: Optional[dict] = None):
        self.features = None
        self.name = encoder_type
        assert encoder_type in ["DINOv2", "CLIP", "SAM", "MedSAM", "BLIP"]

        self.device = device
        self.args = args

        self.encoder = self._load_encoder_model()

    def _load_encoder_model(self):
        if self.name == "DINOv2":
            model = torch.hub.load('facebookresearch/dinov2',
                                   'dinov2_vitl14',
                                   pretrained=False)
            model_path = r"/u/home/koeglf/Documents/models/dinov2_vitl14_pretrain.pth"
            model.load_state_dict(torch.load(model_path))
        else:
            model = None
            print("Not implemented")
        return model.to(self.device)

    def min_max_normalization(self, features: torch.Tensor) -> torch.Tensor:
        """
        Normalize the features to the range [0, 1].
        """

        min_val = features.min()
        max_val = features.max()

        features = (features - min_val) / (max_val - min_val)

        return features

    def compute_features(self, image: torch.Tensor):
        """

        :param image: (h,w,d), ideally (h,w,3) in RGB
        :return: features of shape (x, 1024)
        """
        # assert image.ndim == 3, image.shape  # batch size not
        if self.name == "DINOv2":
            # print("ddd", image.shape)

            image = self.min_max_normalization(image)

            if image.shape[-1] != 3:
                image = self._convert_to_three_channels(
                    image, self.args["slice_dist"])
            # print("ddd", image.shape)
            return self._compute_DINOv2_features(image)

        elif self.name == "CLIP":
            return self._compute_CLIP_features(image)
        else:
            print("Not implemented")

    def compute_features_batch_RGB(self, image_batch: np.ndarray):
        """

        :param image_batch: (bs, 3,h,w), batch of RGB images
        :return: features of shape (x, 1024)
        """
        assert image_batch.ndim == 4, image_batch.shape  # batch size not

        features_batch = np.zeros(
            (image_batch.shape[0], 1024, image_batch.shape[2], image_batch.shape[3]))
        print("f", features_batch.shape, list(image_batch.shape[2:]))
        shape_new = [(x + (14 - x % 14)) *
                     4 for x in list(image_batch.shape[2:])]

        for i in range(image_batch.shape[0]):
            image = image_batch[i]
            if self.name == "DINOv2":
                image = (np.array(image) * 255).astype(np.uint8).squeeze()
                features = self._compute_DINOv2_features(
                    image.transpose(1, 2, 0))
                print(features.shape, shape_new)
                features_batch[i] = self._reshape_DINOv2_features(
                    features, shape_new)

            elif self.name == "CLIP":
                return self._compute_CLIP_features(image)
            else:
                print("Not implemented")

        return features_batch

    def plot_features(self, image: np.ndarray, num_dims: Optional[int] = 10, save_path: Optional[str] = None, plot_pca: Optional[bool] = True):
        # prepare image to fit RGB natural image shape
        shape_new = [(x + (14 - x % 14))*4 for x in list(image.shape[:2])]
        features = self.compute_features(image)

        if plot_pca:
            plt.figure(figsize=(6*num_dims, 12))
            rows = 2
            features_pca = self.features_pca(features, num_dims)
            print("PCA feature shape:", features_pca.shape)
            features_pca = self._reshape_DINOv2_features(
                features_pca, shape_new)
        else:
            plt.figure(figsize=(6*num_dims, 6))
            rows = 1

        features = self._reshape_DINOv2_features(features, shape_new)
        print("feature shape:", features.shape)

        # plot orig image
        if image.shape[-1] != 3:
            image_3channel = self._convert_to_three_channels(
                image, self.args["slice_dist"])
        else:
            image_3channel = image
        plt.subplot(rows, num_dims + 1, 1)
        plt.imshow(image_3channel)
        plt.axis("off")
        plt.title("image")

        # plot features
        for i in range(num_dims):
            plt.subplot(rows, num_dims+1, i + 2)
            plt.imshow(features[..., i])
            plt.axis("off")
            plt.title("feature dim "+str(i))

        if plot_pca:
            # plot pca of features
            for i in range(num_dims):
                plt.subplot(rows, num_dims + 1, num_dims+i + 3)
                plt.imshow(features_pca[..., i])
                plt.axis("off")
                plt.title("PCA dim " + str(i))
        plt.show()
        if save_path is not None:
            plt.savefig(save_path)
        plt.close()

        print(f"Features -- min: {features.min()},max: {features.max()}")
        if plot_pca:
            print(
                f"PCA Features -- min: {features_pca.min()},max: {features_pca.max()}")

    def features_pca(self, features: torch.Tensor, num_dims: Optional[int] = 6):
        pca = PCA(n_components=num_dims)
        pca.fit(features)
        pca_features = pca.transform(features)
        return pca_features

    def _convert_to_three_channels(self, image: torch.Tensor, slice_distance: int):
        """
        Converts an image of shape (h,w,d) to (h,w,3)
        :param image:
        :param slice_distance:
        :return:
        """

        shape = image.shape
        mid_slice = int(shape[-2] / 2)
        if slice_distance == 0:
            # repeat images 3times in last dimension
            image = torch.repeat_interleave(image.unsqueeze(0), 3, dim=0)
        else:
            print("Extracting slices:", mid_slice - slice_distance,
                  mid_slice, mid_slice + slice_distance)
            image = image[..., [mid_slice - slice_distance,
                                mid_slice, mid_slice + slice_distance]]
        return image

    def _compute_DINOv2_features(self, image: np.ndarray):
        total_features = []
        features_dict = self.encoder.forward_features(
            image.unsqueeze(0))
        features = features_dict['x_norm_patchtokens']
        total_features.append(features)

        features = torch.cat(total_features, dim=0).squeeze()
        # print("Features shape:", features.shape)
        return features

    def _reshape_DINOv2_features(self, features: torch.Tensor, image_shape: list[int]):
        # print("before",features.shape)
        patch_size = self.encoder.patch_size  # patchsize=14

        patch_h = image_shape[0] // patch_size  # 37
        patch_w = image_shape[1] // patch_size  # 37
        feat_dim = features.shape[-1]  # vitl14
        # print(patch_h, patch_w, patch_h * patch_w, patch_size)

        total_features = features.reshape(
            patch_h, patch_w, feat_dim)  # (*H*w, 1024)
        # print(total_features.shape)
        return total_features

    def _compute_CLIP_features(self, image: np.ndarray):
        pass
