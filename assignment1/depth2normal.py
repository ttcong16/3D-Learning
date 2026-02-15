import argparse
import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
def depth_to_normal(depth: torch.Tensor, scale: float = 50.0) -> torch.Tensor:
    # Normalize depth
    depth = (depth - depth.min()) / (depth.max() - depth.min()) #[1, H, W]
    device = depth.device

    #
    sobel_x = torch.tensor([[-1., 0., 1.],
                            [-2., 0., 2.],
                            [-1., 0., 1.]], device=device).view(1, 1, 3, 3)
    
    sobel_y = torch.tensor([[-1., -2., -1.],
                            [ 0.,  0.,  0.],
                            [ 1.,  2.,  1.]], device=device).view(1, 1, 3, 3)

    dzdx = F.conv2d(depth.unsqueeze(0), sobel_x, padding=1)
    dzdy = F.conv2d(depth.unsqueeze(0), sobel_y, padding=1)

    nx = -dzdx * scale
    ny = -dzdy * scale
    nz = torch.ones_like(dzdx)

    normal = torch.cat([nx, ny, nz], dim=1) #[1, 3, H, W]
    normal = normal / (torch.norm(normal, dim=1, keepdim=True)) 

    return normal.squeeze(0) #[3, H, W]



def main():
    parser = argparse.ArgumentParser(description="Convert depth map to normal map")
    parser.add_argument("--input", type=str, required=True, help="Path to input depth image (grayscale)")
    parser.add_argument("--output", type=str, required=True, help="Path to save output normal map (RGB)")
    args = parser.parse_args()

    # Load depth image
    depth_img = Image.open(args.input).convert("L")  # grayscale
    transform = transforms.ToTensor()  # [0,1]
    depth = transform(depth_img).float()  # [1, H, W]

    # Convert to normal map
    normal = depth_to_normal(depth)  # [3, H, W]

    # Convert to numpy for saving
    normal_np = normal.permute(1, 2, 0).detach().cpu().numpy()  # [H, W, 3]
    normal_np = ((normal_np + 1.0) * 0.5 * 255.0).astype(np.uint8)  # [-1,1] -> [0,255]

    # Save image
    normal_img = Image.fromarray(normal_np)
    normal_img.save(args.output)
   


if __name__ == "__main__":
    main()


