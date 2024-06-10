import torch
import torch.nn as nn
import numpy as np
from time import time


class BatchKMeans(nn.Module):
    r"""Run multiple independent K-means algorithms in parallel.

    Args:
        n_clusters (int): Number of clusters
        max_iter (int): Maximum number of iterations (default: 100)
        tol (float): Tolerance (default: 0.0001)
        n_redo (int): Number of time k-means will be run with differently initialized centroids.
            the centroids with the lowest inertia will be selected as a final result. (default: 1)
        init_mode (str): Initialization method.
            'random': randomly chose initial centroids from input data.
            'kmeans++': use k-means++ algorithm to initialize centroids. (default: 'kmeans++')
    """

    def __init__(self, n_clusters, n_redo=1, max_iter=100, tol=1e-4, init_mode="kmeans++", verbose=False):
        super(BatchKMeans, self).__init__()
        self.n_redo = n_redo
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.init_mode = init_mode
        self.verbose = verbose

        self.register_buffer("centroids", None)

    def load_state_dict(self, state_dict, **kwargs):
        r"""Override the default load_state_dict() to load custom buffers."""

        for k, v in state_dict.items():
            if "." not in k:
                assert hasattr(self, k), f"attribute {k} does not exist"
                delattr(self, k)
                self.register_buffer(k, v)

        for name, module in self.named_children():
            sd = {k.replace(name + ".", ""): v for k, v in state_dict.items() if k.startswith(name + ".")}
            module.load_state_dict(sd)

    @staticmethod
    def calculate_error(a, b):
        r"""Compute L2 error between a and b"""

        diff = a - b
        diff.pow_(2)
        return diff.sum()

    @staticmethod
    def calculate_inertia(a):
        r"""Compute inertia of a"""

        return (-a).mean()

    @staticmethod
    def euc_sim(a, b):
        r"""Compute batched negative squared Euclidean distance between 'a' and 'b'

        Args:
            a (torch.Tensor): Vector of shape (..., d_vector, m)
            b (torch.Tensor): Vector of shape (..., d_vector, n)

        Returns:
            y (torch.Tensor): Vector of shape (..., m, n)
        """

        y = a.transpose(-2, -1) @ b
        y.mul_(2)
        y.sub_(a.pow(2).sum(dim=-2)[..., :, None])
        y.sub_(b.pow(2).sum(dim=-2)[..., None, :])

        return y

    def kmeanspp(self, data):
        r"""Initialize centroids with k-means++ algorithm

        Args:
            data (torch.Tensor): Vector of shape (..., d_vector, n_data)

        Returns:
            centroids (torch.Tensor): Vector of shape (..., d_vector, n_clusters)
        """

        d_vector, n_data = data.shape[-2:]
        centroids = torch.zeros(*data.shape[:-2], d_vector, self.n_clusters, device=data.device, dtype=data.dtype)

        # Select initial centroid
        centroids[..., 0] = data[..., np.random.randint(n_data)]
        for i in range(1, self.n_clusters):
            current_centroids = centroids[..., :i].contiguous()
            sims = self.euc_sim(data, current_centroids)
            max_sims_v, max_sims_i = sims.max(dim=-1)
            index = max_sims_v.argmin(dim=-1)  # (batch,)

            if data.dim() == 2:
                new_centroid = data[:, index]
            elif data.dim() == 3:
                arange = torch.arange(data.size(0), device=data.device)
                new_centroid = data[arange, :, index]  # (batch, d_vector)
            elif data.dim() == 4:
                arange_w = torch.arange(data.size(0), device=data.device).unsqueeze(dim=1)
                arange_h = torch.arange(data.size(1), device=data.device).unsqueeze(dim=0)
                new_centroid = data[arange_w, arange_h, :, index]
            else:
                raise NotImplementedError

            centroids[..., i] = new_centroid
        return centroids

    def initialize_centroids(self, data):
        r"""
        Initialize centroids with init_method specified in __init__

        Args:
            data (torch.Tensor) Vector of shape (..., d_vector, n_data)

        Returns:
            centroids (torch.Tensor) Vector of shape (..., d_vector, n_clusters)
        """
        n_data = data.size(-1)
        if self.init_mode == "random":
            random_index = np.random.choice(n_data, size=[self.n_clusters], replace=False)
            centroids = data[:, :, random_index].clone()

            if self.verbose:
                print("centroids are randomly initialized.")

        elif self.init_mode == "kmeans++":
            centroids = self.kmeanspp(data).clone()

            if self.verbose:
                print("centroids are initialized with kmeans++.")

        else:
            raise NotImplementedError

        return centroids

    def get_labels(self, data, centroids):
        r"""Compute labels of data

        Args:
            data (torch.Tensor): Vector of shape (..., d_vector, n_data)
            centroids (torch.Tensor): Vector of shape (..., d_vector, n_clusters)

        Returns:
            maxsims (torch.Tensor): Vector of shape (..., n_data)
            labels (torch.Tensor): Vector of shape (..., n_data)
        """

        sims = self.euc_sim(data, centroids)
        maxsims, labels = sims.max(dim=-1)

        return maxsims, labels

    def compute_centroids_loop(self, data, labels):
        r"""Compute centroids of data

        Args:
            data (torch.Tensor): Vector of shape (..., d_vector, n_data)
            labels (torch.Tensor): Vector of shape (..., n_data)

        Returns:
            centroids (torch.Tensor): Vector of shape (..., d_vector, n_clusters)
        """

        ### Naive method with loop ###
        # l, d, m = data.shape
        # centroids = torch.zeros(l, d, self.n_clusters, device=data.device, dtype=data.dtype)
        # for j in range(l):
        #     unique_labels, counts = labels[j].unique(return_counts=True)
        #     for i, count in zip(unique_labels, counts):
        #         centroids[j, :, i] = data[j, :, labels[j] == i].sum(dim=1) / count

        ### Fastest method ###
        mask = [labels == i for i in range(self.n_clusters)]
        mask = torch.stack(mask, dim=-1)  # (..., d_vector, n_clusters)
        centroids = (data.unsqueeze(dim=-1) * mask.unsqueeze(dim=-3)).sum(dim=-2) / mask.sum(dim=-2, keepdim=True)

        return centroids

    def compute_centroids(self, data, labels):
        r"""Compute centroids of data

        Args:
            data (torch.Tensor): Vector of shape (..., d_vector, n_data)
            labels (torch.Tensor): Vector of shape (..., n_data)

        Returns:
            centroids (torch.Tensor): Vector of shape (..., d_vector, n_clusters)
        """

        centroids = self.compute_centroids_loop(data, labels)
        return centroids

    def fit(self, data, centroids=None):
        r"""Perform K-means clustering, and return final labels

        Args:
            data (torch.Tensor): data to be clustered, shape (l, d_vector, n_data)
            centroids (torch.Tensor): initial centroids, shape (l, d_vector, n_clusters)

        Returns:
            best_labels (torch.Tensor): final labels, shape (l, n_data)
        """

        assert data.is_contiguous(), "use .contiguous()"

        best_centroids = None
        best_error = 1e32
        best_labels = None
        best_inertia = 1e32

        if self.verbose:
            tm = time()

        for i in range(self.n_redo):
            if self.verbose:
                tm_i = time()

            if centroids is None:
                centroids = self.initialize_centroids(data)

            for j in range(self.max_iter):
                # clustering iteration
                maxsims, labels = self.get_labels(data, centroids)
                new_centroids = self.compute_centroids(data, labels)
                error = self.calculate_error(centroids, new_centroids)
                centroids = new_centroids
                inertia = self.calculate_inertia(maxsims)

                if self.verbose:
                    print(f"----iteration {j} of {i}th redo, error={error.item()}, inertia={inertia.item()}")

                if error <= self.tol:
                    break

            if inertia < best_inertia:
                best_centroids = centroids
                best_error = error
                best_labels = labels
                best_inertia = inertia

            centroids = None

            if self.verbose:
                print(
                    f"--{i}th redo finished, error: {error.item()}, inertia: {inertia.item()}time spent:{round(time() - tm_i, 4)} sec")

        self.register_buffer("centroids", best_centroids)

        if self.verbose:
            print(f"finished {self.n_redo} redos in {round(time() - tm, 4)} sec, final_inertia: {best_inertia}")

        return best_labels

    def predict(self, query):
        r"""Predict the closest cluster center each sample in query belongs to.

        Args:
            query (torch.Tensor): Vector of shape (l, d_vector, n_query)

        Returns:
            labels (torch.Tensor): Vector of shape (l, n_query)
        """

        _, labels = self.get_labels(query, self.centroids)
        return labels


if __name__ == "__main__":
    x = torch.randn(13, 29, 2, 1000).cuda()
    multi_k_means = BatchKMeans(n_clusters=20, n_redo=1)
    multi_k_means.fit(x)
    print(multi_k_means.centroids.shape)
