import numpy as np
import torch


class AverageMeter(object):
    r"""Stores the results of a metric and computes its average"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.data = list()

    def append(self, value):
        self.data.append([value])

    def extend(self, values):
        self.data.append(values)

    def mean(self):
        return np.concatenate(self.data, axis=0).mean()

    def sum(self):
        return np.concatenate(self.data, axis=0).sum()

    def __len__(self):
        return np.concatenate(self.data, axis=0).shape[0]


def compute_batch_ade(pred, gt):
    r"""Compute ADE(average displacement error) scores for each pedestrian

    Args:
        pred (torch.Tensor): (num_samples, num_ped, seq_len, 2)
        gt (torch.Tensor): (1, num_ped, seq_len, 2) or (num_ped, seq_len, 2)

    Returns:
        ADEs (np.ndarray): (num_ped,)
    """
    temp = (pred - gt).norm(p=2, dim=-1)
    ADEs = temp.mean(dim=2).min(dim=0)[0]
    return ADEs.detach().cpu().numpy()


def compute_batch_fde(pred, gt):
    r"""Compute FDE(final displacement error) scores for each pedestrian

    Args:
        pred (torch.Tensor): (num_samples, num_ped, seq_len, 2)
        gt (torch.Tensor): (1, num_ped, seq_len, 2) or (num_ped, seq_len, 2)

    Returns:
        FDEs (np.ndarray): (num_ped,)
    """
    temp = (pred - gt).norm(p=2, dim=-1)
    FDEs = temp[:, :, -1].min(dim=0)[0]
    return FDEs.detach().cpu().numpy()


def compute_batch_tcc(pred, gt):
    r"""Compute TCC(temporal correlation coefficient) scores for each pedestrian

    Args:
        pred (torch.Tensor): (num_samples, num_ped, seq_len, 2)
        gt (torch.Tensor): (1, num_ped, seq_len, 2) or (num_ped, seq_len, 2)

    Returns:
        TCCs (np.ndarray): (num_ped,)
    """

    gt = gt.squeeze(dim=0) if gt.dim() == 4 else gt
    temp = (pred - gt).norm(p=2, dim=-1)
    pred_best = pred[temp[:, :, -1].argmin(dim=0), range(pred.size(1)), :, :]
    pred_gt_stack = torch.stack([pred_best, gt], dim=0)
    pred_gt_stack = pred_gt_stack.permute(3, 1, 0, 2)
    covariance = pred_gt_stack - pred_gt_stack.mean(dim=-1, keepdim=True)
    factor = 1 / (covariance.shape[-1] - 1)
    covariance = factor * covariance @ covariance.transpose(-1, -2)
    variance = covariance.diagonal(offset=0, dim1=-2, dim2=-1)
    stddev = variance.sqrt()
    corrcoef = covariance / stddev.unsqueeze(-1) / stddev.unsqueeze(-2)
    corrcoef = corrcoef.clamp(-1, 1)
    corrcoef[torch.isnan(corrcoef)] = 0
    TCCs = corrcoef[:, :, 0, 1].mean(dim=0)
    return TCCs.detach().cpu().numpy()


def compute_batch_col(pred, gt):
    r"""Compute COL(collision rate) scores for each pedestrian

    Args:
        pred (torch.Tensor): (num_samples, num_ped, seq_len, 2)
        gt (torch.Tensor): (1, num_ped, seq_len, 2) or (num_ped, seq_len, 2)

    Returns:
        COLs (np.ndarray): (num_ped,)
    """

    pred = pred.permute(0, 2, 1, 3)
    num_interp, thres = 4, 0.2
    pred_fp = pred[:, [0], :, :]
    pred_rel = pred[:, 1:] - pred[:, :-1]
    pred_rel_dense = pred_rel.div(num_interp).unsqueeze(dim=2).repeat_interleave(repeats=num_interp, dim=2).contiguous()
    pred_rel_dense = pred_rel_dense.reshape(pred.size(0), num_interp * (pred.size(1) - 1), pred.size(2), pred.size(3))
    pred_dense = torch.cat([pred_fp, pred_rel_dense], dim=1).cumsum(dim=1)
    col_mask = pred_dense[:, :3 * num_interp + 2].unsqueeze(dim=2).repeat_interleave(repeats=pred.size(2), dim=2)
    col_mask = (col_mask - col_mask.transpose(2, 3)).norm(p=2, dim=-1)
    col_mask = col_mask.add(torch.eye(n=pred.size(2), device=pred.device)[None, None, :, :]).min(dim=1)[0].lt(thres)
    COLs = col_mask.sum(dim=1).gt(0).type(pred.type()).mean(dim=0).mul(100)
    return COLs.detach().cpu().numpy()
