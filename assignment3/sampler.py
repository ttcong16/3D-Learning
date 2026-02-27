import math
from typing import List
import torch.nn as nn
import torch
from ray_utils import RayBundle
from pytorch3d.renderer.cameras import CamerasBase


# Sampler which implements stratified (uniform) point sampling along rays
class StratifiedRaysampler(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.n_pts_per_ray = cfg.n_pts_per_ray
        self.min_depth = cfg.min_depth
        self.max_depth = cfg.max_depth

    def forward(
        self,
        ray_bundle,
    ):
        # TODO (Q1.4): Compute z values for self.n_pts_per_ray points uniformly sampled between [near, far]
        # z_vals = None
        device = ray_bundle.origins.device
        dtype = ray_bundle.origins.dtype
        K = self.n_pts_per_ray
        ray_shape = ray_bundle.origins.shape[:-1]  #(N,) or (B,H,W)

        # Generate per-bin distances z in [min_depth, max_depth] 
        near = torch.as_tensor(self.min_depth, device=device, dtype=dtype)
        far  = torch.as_tensor(self.max_depth,  device=device, dtype=dtype)
        step = (far - near) / K  # scalar bin width

        # bin indices shaped for broadcasting to ray batch shape
        idx = torch.arange(K, device=device, dtype=dtype).view(*([1] * len(ray_shape)), K)  # (..., K)

        jitter = torch.rand(*ray_shape, K, device=device, dtype=dtype) if self.training else 0.5
        z_vals = near + (idx + jitter) * step  # (..., K)


        # TODO (Q1.4): Sample points from z values
        # sample_points = None
        sample_points = (
            ray_bundle.origins[..., None, :] + z_vals[..., None] * ray_bundle.directions[..., None, :]
        )  # (..., K, 3)

        # Return
        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals[..., None] * torch.ones_like(sample_points[..., :1])
        )

    def fine_sampling(self, ray_bundle, weights):
        device, dtype = weights.device, weights.dtype
        pdf = weights + 1e-5
        pdf = pdf / torch.sum(pdf, dim=-1, keepdim=True)
        cdf = torch.cumsum(pdf, dim=-1)
        cdf = torch.cat([torch.zeros_like(cdf[..., :1]), cdf], dim=-1)
        cdf = cdf / (cdf[..., -1:].clamp(min=1e-5))

        N_fine = self.n_pts_per_ray
        u = torch.rand(*cdf.shape[:-1], N_fine, device=device, dtype=dtype)

        inds = torch.searchsorted(cdf.contiguous(), u, right=True)
        inds = torch.clamp(inds, 1, cdf.shape[-1] - 1)
        below, above = inds - 1, inds
        inds_g = torch.stack([below, above], dim=-1)

        N_coarse = weights.shape[-1]
        bins = torch.linspace(self.min_depth, self.max_depth, N_coarse, device=device, dtype=dtype)

        # [N_rays, N_coarse]
        bins = bins.unsqueeze(0).expand(weights.shape[0], -1)
        cdf = cdf[..., :-1]  # đảm bảo cùng length với bins

        # [N_rays, N_fine, N_coarse]
        bins_exp = bins.unsqueeze(1).expand(-1, N_fine, -1)
        cdf = cdf.squeeze(-1)
        cdf_exp = cdf.unsqueeze(1).expand(-1, N_fine, -1)

        inds_g = torch.clamp(inds_g, 0, N_coarse - 1)

        cdf_g = torch.gather(cdf_exp, -1, inds_g)
        bins_g = torch.gather(bins_exp, -1, inds_g)

        denom = (cdf_g[..., 1] - cdf_g[..., 0]).clamp(min=1e-5)
        t = (u - cdf_g[..., 0]) / denom
        z_vals = bins_g[..., 0] + t * (bins_g[..., 1] - bins_g[..., 0])
        z_vals = torch.clamp(z_vals, self.min_depth, self.max_depth)

        sample_points = ray_bundle.origins[..., None, :] + z_vals[..., None] * ray_bundle.directions[..., None, :]

        return ray_bundle._replace(
            sample_points=sample_points,
            sample_lengths=z_vals[..., None]
        )


sampler_dict = {
    'stratified': StratifiedRaysampler
}