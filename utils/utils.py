import os
import json
import random
import numpy as np
import torch


def reproducibility_settings(seed: int = 0):
    r"""Set the random seed for reproducibility"""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False  # Settings for 3090
    torch.backends.cudnn.allow_tf32 = False  # Settings for 3090


def get_exp_config(file: str):
    r"""Load the configuration files"""

    assert os.path.exists(file), f"File {file} does not exist!"
    file = open(file)
    config = json.load(file)
    for k in config.keys():
        if type(config[k]) == dict:
            config[k] = DotDict(config[k])
    return DotDict(config)


class DotDict(dict):
    r"""dot.notation access to dictionary attributes"""

    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__
    __getstate__ = dict
    __setstate__ = dict.update


def print_arguments(args, length=100, sep=': ', delim=' | '):
    r"""Print the arguments in a nice format

    Args:
        args (dict): arguments
        length (int): maximum length of each line
        sep (str): separator between key and value
        delim (str): delimiter between lines
    """

    text = []
    for key in args.keys():
        text.append('{}{}{}'.format(key, sep, args[key]))

    cl = 0
    for n, line in enumerate(text):
        if cl + len(line) > length:
            print('')
            cl = 0
        print(line, end='')
        cl += len(line)
        if n != len(text) - 1:
            print(delim, end='')
            cl += len(delim)
    print('')


def augment_trajectory(obs_traj, pred_traj, flip=True, reverse=True):
    r"""Flip and reverse the trajectory

    Args:
        obs_traj (torch.Tensor): observed trajectory with shape (num_peds, obs_len, 2)
        pred_traj (torch.Tensor): predicted trajectory with shape (num_peds, pred_len, 2)
        flip (bool): whether to flip the trajectory
        reverse (bool): whether to reverse the trajectory
    """

    if flip:
        obs_traj = torch.cat([obs_traj, obs_traj * torch.FloatTensor([[[1, -1]]])], dim=0)
        pred_traj = torch.cat([pred_traj, pred_traj * torch.FloatTensor([[[1, -1]]])], dim=0)
    elif reverse:
        full_traj = torch.cat([obs_traj, pred_traj], dim=1)  # NTC
        obs_traj = torch.cat([obs_traj, full_traj.flip(1)[:, :obs_traj.size(1)]], dim=0)
        pred_traj = torch.cat([pred_traj, full_traj.flip(1)[:, obs_traj.size(1):]], dim=0)
    return obs_traj, pred_traj


if __name__ == '__main__':
    cfg = get_exp_config("../config/stochastic/singulartrajectory-transformerdiffusion-eth.json")
    print(cfg)
