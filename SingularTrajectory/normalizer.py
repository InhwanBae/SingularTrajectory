import torch


class TrajNorm:
    r"""Normalize trajectory with shape (num_peds, length_of_time, 2)

    Args:
        ori (bool): Whether to normalize the trajectory with the origin
        rot (bool): Whether to normalize the trajectory with the rotation
        sca (bool): Whether to normalize the trajectory with the scale
    """

    def __init__(self, ori=True, rot=True, sca=True):
        self.ori, self.rot, self.sca = ori, rot, sca
        self.traj_ori, self.traj_rot, self.traj_sca = None, None, None

    def calculate_params(self, traj):
        r"""Calculate the normalization parameters"""

        if self.ori:
            self.traj_ori = traj[:, [-1]]
        if self.rot:
            if traj.size(1) <= 2:
                dir = traj[:, -1] - traj[:, -2]
            else:
                dir = traj[:, -1] - traj[:, -3]
            rot = torch.atan2(dir[:, 1], dir[:, 0])
            self.traj_rot = torch.stack([torch.stack([rot.cos(), -rot.sin()], dim=1),
                                         torch.stack([rot.sin(), rot.cos()], dim=1)], dim=1)
        if self.sca:
            if traj.size(1) <= 2:
                self.traj_sca = 1. / (traj[:, -1] - traj[:, -2]).norm(p=2, dim=-1)[:, None, None] * 1
            else:
                self.traj_sca = 1. / (traj[:, -1] - traj[:, -3]).norm(p=2, dim=-1)[:, None, None] * 2
            # self.traj_sca[self.traj_sca.isnan() | self.traj_sca.isinf()] = 1e2

    def get_params(self):
        r"""Get the normalization parameters"""

        return self.ori, self.rot, self.sca, self.traj_ori, self.traj_rot, self.traj_sca

    def set_params(self, ori, rot, sca, traj_ori, traj_rot, traj_sca):
        r"""Set the normalization parameters"""

        self.ori, self.rot, self.sca = ori, rot, sca
        self.traj_ori, self.traj_rot, self.traj_sca = traj_ori, traj_rot, traj_sca

    def normalize(self, traj):
        r"""Normalize the trajectory"""

        if self.ori:
            traj = traj - self.traj_ori
        if self.rot:
            traj = traj @ self.traj_rot
        if self.sca:
            traj = traj * self.traj_sca
        return traj

    def denormalize(self, traj):
        r"""Denormalize the trajectory"""

        if self.sca:
            traj = traj / self.traj_sca
        if self.rot:
            traj = traj @ self.traj_rot.transpose(-1, -2)
        if self.ori:
            traj = traj + self.traj_ori
        return traj
