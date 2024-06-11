from abc import ABC, abstractmethod


import torch


class Metric(ABC):
    """
    Base class for all metrics.
    """

    @abstractmethod
    def loss(self, t1: torch.Tensor, t2: torch.Tensor) -> torch.Tensor:
        """
        Compute the loss between two tensors.

        Args:
        - t1 (torch.Tensor): The first tensor.
        - t2 (torch.Tensor): The second tensor.

        Returns:
        - loss (torch.Tensor): The loss.
        """


class MI(Metric):
    """
    Mutual information.
    """

    def mi(self,
           fixed: torch.Tensor,
           moving: torch.Tensor,
           num_bins: int = 32) -> torch.Tensor:
        """
            Compute the mutual information between two images.

            Args:
            - fixed (torch.Tensor): The fixed image.
            - moving (torch.Tensor): The moving image.
            - num_bins (int): The number of bins in the histogram.

            Returns:
            - mi (torch.Tensor): The mutual information.
            """

        def soft_binning(values, num_bins):
            """
            Convert continuous values to discrete bins using a softmax approximation.

            Args:
            - values (torch.Tensor): Continuous values.
            - num_bins (int): Number of bins.

            Returns:
            - binned (torch.Tensor): Soft binned values.
            """
            min_val, _ = torch.min(values), torch.max(values)
            max_val, _ = torch.max(values), torch.min(values)
            values = (values - min_val) / (max_val -
                                           min_val + 1e-10)  # Normalize to [0, 1]
            values = values * (num_bins - 1)  # Scale to [0, num_bins-1]

            bin_edges = torch.linspace(
                0, num_bins - 1, num_bins).to(values.device)
            binned = torch.softmax(-((values.unsqueeze(1) -
                                   bin_edges.unsqueeze(0))**2), dim=1)

            return binned

        def joint_histogram_soft(fixed, moving, num_bins=32):
            """
            Compute the joint histogram for two images using soft binning.

            Args:
            - fixed (torch.Tensor): The fixed image.
            - moving (torch.Tensor): The moving image.
            - num_bins (int): The number of bins in the histogram.

            Returns:
            - joint_hist (torch.Tensor): The joint histogram.
            """
            assert fixed.shape == moving.shape, "Images must have the same shape"

            # Soft binning
            fixed_binned = soft_binning(fixed.flatten(), num_bins)
            moving_binned = soft_binning(moving.flatten(), num_bins)

            joint_hist = torch.matmul(fixed_binned.t(), moving_binned)
            joint_hist /= torch.sum(joint_hist)

            return joint_hist

        def marginal_histograms_soft(joint_hist):
            """
            Compute the marginal histograms from the joint histogram.

            Args:
            - joint_hist (torch.Tensor): The joint histogram.

            Returns:
            - fixed_hist (torch.Tensor): The marginal histogram for the fixed image.
            - moving_hist (torch.Tensor): The marginal histogram for the moving image.
            """
            fixed_hist = torch.sum(joint_hist, dim=1)
            moving_hist = torch.sum(joint_hist, dim=0)
            return fixed_hist, moving_hist

        def entropy(hist):
            """
            Compute the entropy of a histogram.

            Args:
            - hist (torch.Tensor): The histogram.

            Returns:
            - entropy (torch.Tensor): The entropy.
            """
            p = hist / torch.sum(hist)
            p = p[p > 0]
            return -torch.sum(p * torch.log(p + 1e-10))

        joint_hist = joint_histogram_soft(fixed, moving, num_bins)
        fixed_hist, moving_hist = marginal_histograms_soft(joint_hist)
        h_joint = entropy(joint_hist)
        h_fixed = entropy(fixed_hist)
        h_moving = entropy(moving_hist)
        mi_score = h_fixed + h_moving - h_joint

        return mi_score

    def loss(self, t1: torch.Tensor, t2: torch.Tensor):

        loss = - self.mi(t1.flatten(), t2.flatten())

        return loss


class CKA(Metric):
    """
    https://github.com/yuanli2333/CKA-Centered-Kernel-Alignment
    """

    def centering(self, K: torch.Tensor):
        n = K.shape[0]
        unit = torch.ones([n, n], device=K.device)
        I = torch.eye(n, device=K.device)
        H = I - unit / n

        # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
        return torch.matmul(torch.matmul(H, K), H)
        # return np.dot(H, K)  # KH

    def rbf(self, X, sigma=None):
        GX = torch.matmul(X, X.t())
        KX = torch.diag(GX) - GX + (torch.diag(GX) - GX).T
        if sigma is None:
            mdist = torch.median(KX[KX != 0])
            sigma = torch.sqrt(mdist)
        KX *= - 0.5 / (sigma * sigma)
        KX = torch.exp(KX)
        return KX

    def kernel_HSIC(self, X, Y, sigma):
        return torch.sum(self.centering(self.rbf(X, sigma)) * self.centering(self.rbf(Y, sigma)))

    def kernel_CKA(self, X, Y, sigma=None):
        hsic = self.kernel_HSIC(X, Y, sigma)
        var1 = torch.sqrt(self.kernel_HSIC(X, X, sigma))
        var2 = torch.sqrt(self.kernel_HSIC(Y, Y, sigma))

        return hsic / (var1 * var2)

    def loss(self, t1: torch.Tensor, t2: torch.Tensor):

        loss = - self.kernel_CKA(t1, t2)

        return loss
