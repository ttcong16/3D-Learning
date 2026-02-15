import torch
import numpy as np
import imageio
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    AlphaCompositor,
    PointsRasterizer,
    FoVPerspectiveCameras,
    look_at_view_transform
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create point cloud of Torus
def generate_torus(R=2.0, r=0.7, num_samples=100):
    """
    R: large radius 
    r: small radius 
    num_samples: num of sampling points from 2 dimension
    """
    theta = torch.linspace(0, 2 * np.pi, num_samples)
    phi = torch.linspace(0, 2 * np.pi, num_samples)
    theta, phi = torch.meshgrid(theta, phi, indexing="ij")

    x = (R + r * torch.cos(theta)) * torch.cos(phi)
    y = (R + r * torch.cos(theta)) * torch.sin(phi)
    z = r * torch.sin(theta)

    points = torch.stack([x, y, z], dim=-1).reshape(-1, 3)

    # Normalize -> [0,1]
    colors = (points - points.min(0).values) / (points.max(0).values - points.min(0).values)

    return points.to(device), colors.to(device)

# Create renderer
def create_renderer(image_size=512, radius=0.01, points_per_pixel=5):
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=radius,
        points_per_pixel=points_per_pixel
    )
    rasterizer = PointsRasterizer(raster_settings=raster_settings)
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=AlphaCompositor())
    return renderer

# Render 360° rotation
def render_rotation(point_cloud, renderer, num_frames=36, dist=6.0, elev=20.0, out_gif="output/torus.gif"):
    images = []
    angles = torch.linspace(0, 360, num_frames)
    for azim in angles:
        R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

        image = renderer(point_cloud, cameras=cameras)
        img = (image[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        images.append(img)

    imageio.mimsave(out_gif, images, duration=0.1)  #10 FPS


if __name__ == "__main__":
    points, colors = generate_torus(R=2.0, r=0.7, num_samples=100)
    point_cloud = Pointclouds(points=[points], features=[colors])
    renderer = create_renderer(image_size=512, radius=0.01, points_per_pixel=5)
    render_rotation(point_cloud, renderer, num_frames=36, dist=6.0, elev=20.0, out_gif="output/torus.gif")
