import torch
import torch.nn as nn
from .normalizer import TrajNorm
import numpy as np
from sklearn.cluster import KMeans
from scipy.interpolate import BSpline


class SingularSpace(nn.Module):
    r"""Singular space model

    Args:
        hyper_params (DotDict): The hyper-parameters
        norm_ori (bool): Whether to normalize the trajectory with the origin
        norm_rot (bool): Whether to normalize the trajectory with the rotation
        norm_sca (bool): Whether to normalize the trajectory with the scale"""

    def __init__(self, hyper_params, norm_ori=True, norm_rot=True, norm_sca=True):
        super().__init__()

        self.hyper_params = hyper_params
        self.t_obs, self.t_pred = hyper_params.obs_len, hyper_params.pred_len
        self.obs_svd, self.pred_svd = hyper_params.obs_svd, hyper_params.pred_svd
        self.k = hyper_params.k
        self.s = hyper_params.num_samples
        self.dim = hyper_params.traj_dim
        self.traj_normalizer = TrajNorm(ori=norm_ori, rot=norm_rot, sca=norm_sca)

        self.V_trunc = nn.Parameter(torch.zeros((self.t_pred * self.dim, self.k)))
        self.V_obs_trunc = nn.Parameter(torch.zeros((self.t_obs * self.dim, self.k)))
        self.V_pred_trunc = nn.Parameter(torch.zeros((self.t_pred * self.dim, self.k)))

    def normalize_trajectory(self, obs_traj, pred_traj=None):
        r"""Trajectory normalization

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory (Optional, for training only)

        Returns:
            obs_traj_norm (torch.Tensor): The normalized observed trajectory
            pred_traj_norm (torch.Tensor): The normalized predicted trajectory
        """

        self.traj_normalizer.calculate_params(obs_traj)
        obs_traj_norm = self.traj_normalizer.normalize(obs_traj)
        pred_traj_norm = self.traj_normalizer.normalize(pred_traj) if pred_traj is not None else None
        return obs_traj_norm, pred_traj_norm

    def denormalize_trajectory(self, traj_norm):
        r"""Trajectory denormalization

        Args:
            traj_norm (torch.Tensor): The trajectory to be denormalized

        Returns:
            traj (torch.Tensor): The denormalized trajectory
        """

        traj = self.traj_normalizer.denormalize(traj_norm)
        return traj

    def to_Singular_space(self, traj, evec):
        r"""Transform Euclidean trajectories to Singular space coordinates

        Args:
            traj (torch.Tensor): The trajectory to be transformed
            evec (torch.Tensor): The Singular space basis vectors

        Returns:
            C (torch.Tensor): The Singular space coordinates"""

        # Euclidean space -> Singular space
        tdim = evec.size(0)
        M = traj.reshape(-1, tdim).T
        C = evec.T.detach() @ M
        return C

    def batch_to_Singular_space(self, traj, evec):
        # Euclidean space -> Singular space
        tdim = evec.size(0)
        M = traj.reshape(traj.size(0), traj.size(1), tdim).transpose(1, 2)
        C = evec.T.detach() @ M
        return C

    def to_Euclidean_space(self, C, evec):
        r"""Transform Singular space coordinates to Euclidean trajectories

        Args:
            C (torch.Tensor): The Singular space coordinates
            evec (torch.Tensor): The Singular space basis vectors

        Returns:
            traj (torch.Tensor): The Euclidean trajectory"""

        # Singular space -> Euclidean
        t = evec.size(0) // self.dim
        M = evec.detach() @ C
        traj = M.T.reshape(-1, t, self.dim)
        return traj
    
    def batch_to_Euclidean_space(self, C, evec):
        # Singular space -> Euclidean
        b = C.size(0)
        t = evec.size(0) // self.dim
        M = evec.detach() @ C
        traj = M.transpose(1, 2).reshape(b, -1, t, self.dim)
        return traj

    def truncated_SVD(self, traj, k=None, full_matrices=False):
        r"""Truncated Singular Value Decomposition

        Args:
            traj (torch.Tensor): The trajectory to be decomposed
            k (int): The number of singular values and vectors to be computed
            full_matrices (bool): Whether to compute full-sized matrices

        Returns:
            U_trunc (torch.Tensor): The truncated left singular vectors
            S_trunc (torch.Tensor): The truncated singular values
            Vt_trunc (torch.Tensor): The truncated right singular vectors
        """

        assert traj.size(2) == self.dim  # NTC
        k = self.k if k is None else k

        # Singular Value Decomposition
        M = traj.reshape(-1, traj.size(1) * self.dim).T
        U, S, Vt = torch.linalg.svd(M, full_matrices=full_matrices)

        # Truncated SVD
        U_trunc, S_trunc, Vt_trunc = U[:, :k], S[:k], Vt[:k, :]
        return U_trunc, S_trunc, Vt_trunc.T

    def parameter_initialization(self, obs_traj, pred_traj):
        r"""Initialize the Singular space basis vectors parameters (for training only)

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory

        Returns:
            pred_traj_norm (torch.Tensor): The normalized predicted trajectory
            V_pred_trunc (torch.Tensor): The truncated eigenvectors of the predicted trajectory

        Note:
            This function should be called once before training the model."""

        # Normalize trajectory
        obs_traj_norm, pred_traj_norm = self.normalize_trajectory(obs_traj, pred_traj)
        V_trunc, _, _ = self.truncated_SVD(pred_traj_norm)

        # Pre-calculate the transformation matrix
        # Here, we use Irwin–Hall polynomial function
        degree=2
        twot_win = self.dim * self.t_pred
        twot_hist=self.dim * self.t_obs
        steps = np.linspace(0., 1., twot_hist)
        knot = twot_win - degree + 1
        knots_qu = np.concatenate([np.zeros(degree), np.linspace(0, 1, knot), np.ones(degree)])
        C_hist = np.zeros([twot_hist, twot_win])
        for i in range(twot_win):
            C_hist[:, i] = BSpline(knots_qu, (np.arange(twot_win) == i).astype(float), degree, extrapolate=False)(steps)
        C_hist = torch.FloatTensor(C_hist) 

        V_obs_trunc = C_hist @ V_trunc
        V_pred_trunc = V_trunc

        # Register basis vectors as model parameters
        self.V_trunc = nn.Parameter(V_trunc.to(self.V_trunc.device))
        self.V_obs_trunc = nn.Parameter(V_obs_trunc.to(self.V_obs_trunc.device))
        self.V_pred_trunc = nn.Parameter(V_pred_trunc.to(self.V_pred_trunc.device))

        # Reuse values for anchor generation
        return pred_traj_norm, V_pred_trunc

    def projection(self, obs_traj, pred_traj=None):
        r"""Trajectory projection to the Singular space

        Args:
            obs_traj (torch.Tensor): The observed trajectory
            pred_traj (torch.Tensor): The predicted trajectory (optional, for training only)

        Returns:
            C_obs (torch.Tensor): The observed trajectory in the Singular space
            C_pred (torch.Tensor): The predicted trajectory in the Singular space (optional, for training only)
        """

        # Trajectory Projection
        obs_traj_norm, pred_traj_norm = self.normalize_trajectory(obs_traj, pred_traj)
        C_obs = self.to_Singular_space(obs_traj_norm, evec=self.V_obs_trunc).detach()
        C_pred = self.to_Singular_space(pred_traj_norm, evec=self.V_pred_trunc).detach() if pred_traj is not None else None
        return C_obs, C_pred

    def reconstruction(self, C_pred):
        r"""Trajectory reconstruction from the Singular space

        Args:
            C_pred (torch.Tensor): The predicted trajectory in the Singular space

        Returns:
            pred_traj (torch.Tensor): The predicted trajectory in the Euclidean space
        """

        C_pred = C_pred.permute(2, 0, 1)
        pred_traj = self.batch_to_Euclidean_space(C_pred, evec=self.V_pred_trunc)
        pred_traj = self.denormalize_trajectory(pred_traj)

        return pred_traj

    def forward(self, C_pred):
        r"""Alias for reconstruction"""

        return self.reconstruction(C_pred)
