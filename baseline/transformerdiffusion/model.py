import math
import torch
import torch.nn as nn
from torch.nn import Module, Linear
import numpy as np
from .layers import PositionalEncoding, ConcatSquashLinear


class st_encoder(nn.Module):
    """Transformer Denoising Model
    codebase borrowed from https://github.com/MediaBrain-SJTU/LED"""
    def __init__(self):
        super().__init__()
        channel_in = 2
        channel_out = 32
        dim_kernel = 3
        self.dim_embedding_key = 256
        self.spatial_conv = nn.Conv1d(channel_in, channel_out, dim_kernel, stride=1, padding=1)
        self.temporal_encoder = nn.GRU(channel_out, self.dim_embedding_key, 1, batch_first=True)
        self.relu = nn.ReLU()
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_normal_(self.spatial_conv.weight)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_ih_l0)
        nn.init.kaiming_normal_(self.temporal_encoder.weight_hh_l0)
        nn.init.zeros_(self.spatial_conv.bias)
        nn.init.zeros_(self.temporal_encoder.bias_ih_l0)
        nn.init.zeros_(self.temporal_encoder.bias_hh_l0)

    def forward(self, X):
        X_t = torch.transpose(X, 1, 2)
        X_after_spatial = self.relu(self.spatial_conv(X_t))
        X_embed = torch.transpose(X_after_spatial, 1, 2)
        output_x, state_x = self.temporal_encoder(X_embed)
        state_x = state_x.squeeze(0)
        return state_x


class social_transformer(nn.Module):
    """Transformer Denoising Model
    codebase borrowed from https://github.com/MediaBrain-SJTU/LED"""
    def __init__(self, cfg):
        super(social_transformer, self).__init__()
        self.encode_past = nn.Linear(cfg.k*cfg.s+6, 256, bias=False)
        self.layer = nn.TransformerEncoderLayer(d_model=256, nhead=2, dim_feedforward=256)
        self.transformer_encoder = nn.TransformerEncoder(self.layer, num_layers=2)

    def forward(self, h, mask):
        h_feat = self.encode_past(h.reshape(h.size(0), -1)).unsqueeze(1)
        h_feat_ = self.transformer_encoder(h_feat, mask)
        h_feat = h_feat + h_feat_

        return h_feat


class TransformerDenoisingModel(Module):
    """Transformer Denoising Model
    codebase borrowed from https://github.com/MediaBrain-SJTU/LED"""
    def __init__(self, context_dim=256, cfg=None):
        super().__init__()
        self.context_dim = context_dim
        self.spatial_dim = 1
        self.temporal_dim = cfg.k
        self.n_samples = cfg.s
        self.encoder_context = social_transformer(cfg)
        self.pos_emb = PositionalEncoding(d_model=2*context_dim, dropout=0.1, max_len=24)
        self.concat1 = ConcatSquashLinear(self.n_samples*self.spatial_dim*self.temporal_dim, 2*context_dim, context_dim+3)
        self.concat3 = ConcatSquashLinear(2*context_dim,context_dim,context_dim+3)
        self.concat4 = ConcatSquashLinear(context_dim,context_dim//2,context_dim+3)
        self.linear = ConcatSquashLinear(context_dim//2, self.n_samples*self.spatial_dim*self.temporal_dim, context_dim+3)

    def forward(self, x, beta, context, mask):
        batch_size = x.size(0)
        beta = beta.view(batch_size, 1, 1)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1)
        ctx_emb = torch.cat([time_emb, context], dim=-1)
        x = self.concat1(ctx_emb, x)
        final_emb = x.permute(1,0,2)
        final_emb = self.pos_emb(final_emb)
        trans = self.transformer_encoder(final_emb).permute(1,0,2)
        trans = self.concat3(ctx_emb, trans)
        trans = self.concat4(ctx_emb, trans)
        return self.linear(ctx_emb, trans)

    def encode_context(self, context, mask):
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        context = self.encoder_context(context, mask)
        return context

    def generate_accelerate(self, x, beta, context, mask):
        beta = beta.view(beta.size(0), 1)
        time_emb = torch.cat([beta, torch.sin(beta), torch.cos(beta)], dim=-1) 
        ctx_emb = torch.cat([time_emb, context.view(-1, self.context_dim*self.spatial_dim)], dim=-1)

        trans = self.concat1.batch_generate(ctx_emb, x.view(-1, self.n_samples*self.temporal_dim*self.spatial_dim))
        trans = self.concat3.batch_generate(ctx_emb, trans)  
        trans = self.concat4.batch_generate(ctx_emb, trans)
        return self.linear.batch_generate(ctx_emb, trans).view(-1, self.n_samples, self.temporal_dim, self.spatial_dim)


class DiffusionModel(Module):
    """Transformer Denoising Model
    codebase borrowed from https://github.com/MediaBrain-SJTU/LED"""
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.model = TransformerDenoisingModel(context_dim=256, cfg=cfg)

        self.betas = self.make_beta_schedule(
            schedule=self.cfg.beta_schedule, n_timesteps=self.cfg.steps, 
            start=self.cfg.beta_start, end=self.cfg.beta_end).cuda()
        
        self.alphas = 1 - self.betas
        self.alphas_prod = torch.cumprod(self.alphas, 0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_prod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_prod)
        
    def make_beta_schedule(self, schedule: str = 'linear', 
            n_timesteps: int = 1000, 
            start: float = 1e-5, end: float = 1e-2) -> torch.Tensor:
        if schedule == 'linear':
            betas = torch.linspace(start, end, n_timesteps)
        elif schedule == "quad":
            betas = torch.linspace(start ** 0.5, end ** 0.5, n_timesteps) ** 2
        elif schedule == "sigmoid":
            betas = torch.linspace(-6, 6, n_timesteps)
            betas = torch.sigmoid(betas) * (end - start) + start
        return betas

    def extract(self, input, t, x):
        shape = x.shape
        out = torch.gather(input, 0, t.to(input.device))
        reshape = [t.shape[0]] + [1] * (len(shape) - 1)
        return out.reshape(*reshape)

    def forward(self, past_traj, traj_mask, loc):
        pred_traj = self.p_sample_forward(past_traj, traj_mask, loc)
        return pred_traj

    def p_sample(self, x, mask, cur_y, t, context):
        t = torch.tensor([t]).cuda()
        beta = self.extract(self.betas, t.repeat(x.shape[0]), cur_y)
        eps_theta = self.model.generate_accelerate(cur_y, beta, context, mask)
        eps_factor = ((1 - self.extract(self.alphas, t, cur_y)) / self.extract(self.one_minus_alphas_bar_sqrt, t, cur_y))
        mean = (1 / self.extract(self.alphas, t, cur_y).sqrt()) * (cur_y - (eps_factor * eps_theta))

        # Fix the random seed for reproducibility
        if False:
            z = torch.randn_like(cur_y).to(x.device)
        else:
            rng = np.random.default_rng(seed=0)
            z = torch.Tensor(rng.normal(loc=0, scale=1.0, size=cur_y.shape)).cuda()
        
        sigma_t = self.extract(self.betas, t, cur_y).sqrt()
        sample = mean + sigma_t * z * 0.00001
        return (sample)

    def p_sample_forward(self, x, mask, loc):
        prediction_total = torch.Tensor().cuda()

        # Fix the random seed for reproducibility
        if False:
            cur_y = torch.randn_like(loc)
        else:
            rng = np.random.default_rng(seed=0)
            cur_y = torch.Tensor(rng.normal(loc=0, scale=1.0, size=loc.shape)).cuda()

        context = self.model.encode_context(x, mask)
        for i in reversed(range(self.cfg.steps)):
            cur_y = self.p_sample(x, mask, cur_y, i, context)
        prediction_total = cur_y
        return prediction_total
