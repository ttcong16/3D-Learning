import argparse
import torch
import torch.fft
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import numpy as np

def normal_to_depth(normal: torch.Tensor, scale: float = 0.6) -> torch.Tensor:
    """
    Convert normal map (Tensor [3, H, W], range [-1,1]) to depth map (Tensor [1, H, W], range [0,1]).
    """
    n_x, n_y, n_z = normal[0], normal[1], normal[2]

    # p = dz/dx, q = dz/dy
    p = -n_x / (n_z + 1e-8)
    q = -n_y / (n_z + 1e-8)

    # Gradient and divergence
    dp_dx = p[:, 1:] - p[:, :-1]
    dp_dx = F.pad(dp_dx, (1, 0))  # original shape
    dq_dy = q[1:, :] - q[:-1, :]
    dq_dy = F.pad(dq_dy, (0, 0, 1, 0))
    div_pq = dp_dx + dq_dy

    # FFT to solve Poisson equation
    h, w = normal.shape[1], normal.shape[2]
    freq_y = torch.fft.fftfreq(h, device=normal.device).reshape(h, 1)
    freq_x = torch.fft.fftfreq(w, device=normal.device).reshape(1, w)
    denom = 4 * (torch.sin(np.pi * freq_x) ** 2 + torch.sin(np.pi * freq_y) ** 2)
    denom[0, 0] = 1e-8

    div_fft = torch.fft.fft2(div_pq)
    depth_fft = div_fft / denom
    depth = torch.fft.ifft2(depth_fft).real

    # Normalize to [0,1]
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    depth = 1.0 - depth
    depth = torch.clamp(depth * scale, 0.0, 1.0)

    return depth.unsqueeze(0)  # [1, H, W]


def main():
    parser = argparse.ArgumentParser(description="Convert normal map to depth map")
    parser.add_argument("--input", type=str, required=True, help="Path to input normal map (RGB)")
    parser.add_argument("--output", type=str, required=True, help="Path to save output depth map (grayscale)")
    args = parser.parse_args()

    # Load normal map
    normal_img = Image.open(args.input).convert("RGB")
    transform = transforms.ToTensor()  # [0,1]
    normal = transform(normal_img).float() * 2 - 1  # scale về [-1,1]

    # Convert
    depth = normal_to_depth(normal)

    # Convert into numpy and save
    depth_np = depth.squeeze(0).detach().cpu().numpy()  # [H,W]
    depth_np = (depth_np * 255.0).astype(np.uint8)

    depth_img = Image.fromarray(depth_np, mode="L")
    depth_img.save(args.output)



if __name__ == "__main__":
    main()
