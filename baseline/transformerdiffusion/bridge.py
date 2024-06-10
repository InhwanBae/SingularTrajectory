from collections import defaultdict
import torch


def model_forward_pre_hook(obs_data, obs_ori, addl_info):
    # Pre-process input data for the baseline model
    if obs_ori is not None:
        obs_data = torch.cat([obs_data, obs_ori], dim=0)
    
    scene_mask = addl_info["scene_mask"]
    num_samples = addl_info["num_samples"]
    anchor = addl_info["anchor"]
    
    obs_data = torch.cat([obs_data.transpose(1, 0), anchor.permute(1, 2, 0).flatten(start_dim=1)], dim=1)
    obs_data = obs_data.unsqueeze(dim=-1)
    
    loc = anchor.permute(1, 2, 0).unsqueeze(dim=-1)
    input_data = [obs_data, scene_mask, loc]
    
    return input_data


def model_forward(input_data, baseline_model):
    # Forward the baseline model with input data
    output_data = baseline_model(*input_data)
    return output_data


def model_forward_post_hook(output_data, addl_info=None):
    # Post-process output data of the baseline model
    pred_data = output_data.squeeze(dim=-1).permute(2, 0, 1)

    return pred_data
