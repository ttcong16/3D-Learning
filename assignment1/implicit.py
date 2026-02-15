import torch
import numpy as np
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PointLights,
    PointsRasterizationSettings, PointsRasterizer,
    AlphaCompositor, PointsRenderer,
    look_at_view_transform
)
from pytorch3d.structures import Pointclouds
from PIL import Image
import imageio

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def implicit_torus(x, y, z, R=1.0, r=0.4):
    return (torch.sqrt(x**2 + y**2) - R)**2 + z**2 - r**2

def sample_torus_points(grid_size=128, threshold=0.02, R=1.0, r=0.4, device="cpu"):
    lin = torch.linspace(-1.5, 1.5, grid_size, device=device)
    X, Y, Z = torch.meshgrid(lin, lin, lin, indexing="ij")
    F = implicit_torus(X, Y, Z, R=R, r=r)
    
    mask = torch.abs(F) < threshold
    points = torch.stack([X[mask], Y[mask], Z[mask]], dim=1)
    colors = torch.tensor([0.2, 0.6, 1.0], device=device).repeat(points.shape[0], 1)
    return points, colors

def get_renderer(image_size=512):
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.01,
        points_per_pixel=10
    )
    rasterizer = PointsRasterizer(cameras=None, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor()
    )
    return renderer

def render_rotation(point_cloud, colors, num_frames=36, output_path="output/torus_pointcloud.gif"):
    frames = []
    point_cloud = Pointclouds(points=[point_cloud], features=[colors])
    renderer = get_renderer(image_size=512)

    for azim in torch.linspace(0, 360, steps=num_frames):
        R, T = look_at_view_transform(dist=3.0, elev=0.0, azim=azim.item())
        cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
        image = renderer(point_cloud, cameras=cameras)
        img = (image[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        frames.append(Image.fromarray(img))

    frames[0].save(output_path, save_all=True, append_images=frames[1:], duration=100, loop=0)

def main():
    points, colors = sample_torus_points(grid_size=128, threshold=0.02, R=1.0, r=0.4, device=device)
    render_rotation(points, colors, num_frames=36, output_path="output/torus_pointcloud.gif")

if __name__ == "__main__":
    main()
