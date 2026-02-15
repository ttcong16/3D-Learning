import argparse
import pickle
import matplotlib.pyplot as plt
import numpy as np
import torch
import pytorch3d
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRasterizer,
    AlphaCompositor,
    PointsRenderer,
    look_at_view_transform,
    FoVPerspectiveCameras
)



# ======================
# Utils thay cho starter.utils
# ======================

def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def get_points_renderer(image_size=256, background_color=(1, 1, 1), device=None):
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.003,
        points_per_pixel=10
    )
    rasterizer = PointsRasterizer(raster_settings=raster_settings)
    compositor = AlphaCompositor(background_color=torch.tensor(background_color, device=device))
    renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)
    return renderer


def unproject_depth_image(image, mask, depth, camera):
    """
    Unprojects a depth image into a 3D point cloud.

    Args:
        image (torch.Tensor): (S, S, 3)
        mask (torch.Tensor): (S, S)
        depth (torch.Tensor): (S, S)
        camera: A Pytorch3D FoVPerspectiveCameras

    Returns:
        points (torch.Tensor): (N, 3)
        rgba   (torch.Tensor): (N, 4)
    """
    device = camera.device
    assert image.shape[0] == image.shape[1], "Image must be square."
    image_shape = image.shape[0]
    ndc_pixel_coordinates = torch.linspace(1, -1, image_shape)
    Y, X = torch.meshgrid(ndc_pixel_coordinates, ndc_pixel_coordinates, indexing="ij")
    xy_depth = torch.dstack([X, Y, depth])
    points = camera.unproject_points(
        xy_depth.to(device), in_ndc=False, from_ndc=False, world_coordinates=True,
    )
    points = points[mask > 0.5]
    rgb = image[mask > 0.5]
    rgb = rgb.to(device)

    alpha = torch.ones_like(rgb)[..., :1]
    rgb = torch.cat([rgb, alpha], dim=1)

    return points, rgb



def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)
    return data

def render_rgbd_pointclouds(
    data_path="data/rgbd_data.pkl",
    image_size=256,
    background_color=(1, 1, 1),
    device=None,
):
    if device is None:
        device = get_device()

    # Load the RGB-D data
    data = load_rgbd_data(data_path)

    # Extract data
    rgb1 = torch.tensor(data['rgb1'], dtype=torch.float32) / 255.0
    depth1 = torch.tensor(data['depth1'], dtype=torch.float32)
    mask1 = torch.tensor(data['mask1'], dtype=torch.float32)
    camera1 = data['cameras1'].to(device)

    rgb2 = torch.tensor(data['rgb2'], dtype=torch.float32) / 255.0
    depth2 = torch.tensor(data['depth2'], dtype=torch.float32)
    mask2 = torch.tensor(data['mask2'], dtype=torch.float32)
    camera2 = data['cameras2'].to(device)

    # Show original RGB, depth, mask
    fig, axs = plt.subplots(2, 3, figsize=(12, 8))
    axs[0, 0].imshow(rgb1.cpu().numpy())
    axs[0, 0].set_title("RGB1")
    axs[0, 1].imshow(depth1.cpu().numpy(), cmap="plasma")
    axs[0, 1].set_title("Depth1")
    axs[0, 2].imshow(mask1.cpu().numpy(), cmap="gray")
    axs[0, 2].set_title("Mask1")

    axs[1, 0].imshow(rgb2.cpu().numpy())
    axs[1, 0].set_title("RGB2")
    axs[1, 1].imshow(depth2.cpu().numpy(), cmap="plasma")
    axs[1, 1].set_title("Depth2")
    axs[1, 2].imshow(mask2.cpu().numpy(), cmap="gray")
    axs[1, 2].set_title("Mask2")
    for ax in axs.ravel():
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # Unproject
    points1, colors1 = unproject_depth_image(rgb1, mask1, depth1, camera1)
    points2, colors2 = unproject_depth_image(rgb2, mask2, depth2, camera2)

    # Union
    points_union = torch.cat([points1, points2], dim=0)
    colors_union = torch.cat([colors1, colors2], dim=0)

    # Create Pointclouds
    pc1 = Pointclouds(points=[points1], features=[colors1[:, :3]])
    pc2 = Pointclouds(points=[points2], features=[colors2[:, :3]])
    pc_union = Pointclouds(points=[points_union], features=[colors_union[:, :3]])

    # Renderer
    renderer = get_points_renderer(
        image_size=image_size, background_color=background_color, device=device
    )

    # Render from multiple viewpoints
    azimuth_values = [0, 120, 240]
    for pc, name in [(pc1, "first"), (pc2, "second"), (pc_union, "union")]:
        fig, axs = plt.subplots(1, len(azimuth_values), figsize=(12, 4))
        for i, azim in enumerate(azimuth_values):
            R, T = look_at_view_transform(dist=6.0, elev=10, azim=azim)
            cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
            rend = renderer(pc.to(device), cameras=cameras)
            img = rend[0, ..., :3].cpu().numpy()
            axs[i].imshow(img)
            axs[i].axis("off")
            axs[i].set_title(f"{name} azim={azim}")
        plt.show()


if __name__ == "__main__":
    device = get_device()
    render_rgbd_pointclouds(device=device)
