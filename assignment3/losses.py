import torch
import torch.nn.functional as F

def eikonal_loss(gradients):
    # TODO (Q6): Implement eikonal loss
     # Compute L2 norm of gradients along last dimension
    eps = 1e-8
    grad_norm = torch.sqrt(torch.sum(gradients ** 2, dim=-1) + eps)
    loss = ((grad_norm - 1.0) ** 2).mean()
    return loss


def sphere_loss(signed_distance, points, radius=1.0):
    return torch.square(signed_distance[..., 0] - (torch.norm(points, dim=-1) - radius)).mean()

def get_random_points(num_points, bounds, device):
    min_bound = torch.tensor(bounds[0], device=device).unsqueeze(0)
    max_bound = torch.tensor(bounds[1], device=device).unsqueeze(0)

    return torch.rand((num_points, 3), device=device) * (max_bound - min_bound) + min_bound

def select_random_points(points, n_points):
    points_sub = points[torch.randperm(points.shape[0])]
    return points_sub.reshape(-1, 3)[:n_points]
