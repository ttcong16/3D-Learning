import argparse
import pickle
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch3d.structures import Pointclouds
from pytorch3d.renderer import (
    PointsRasterizationSettings,
    PointsRenderer,
    PointsRasterizer,
    AlphaCompositor,
    FoVPerspectiveCameras,
    look_at_view_transform,
)


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def unproject_depth_image(image, mask, depth, camera):
    image = image.to(device)
    mask = mask.to(device)
    depth = depth.to(device)
    camera = camera.to(device)

    assert image.shape[0] == image.shape[1], "Image must be square."
    image_shape = image.shape[0]

    ndc_pixel_coordinates = torch.linspace(1, -1, image_shape, device=device)
    Y, X = torch.meshgrid(ndc_pixel_coordinates, ndc_pixel_coordinates, indexing="ij")

    xy_depth = torch.dstack([X, Y, depth]).to(device)

    points = camera.unproject_points(
        xy_depth, in_ndc=False, from_ndc=False, world_coordinates=True,
    )
    points = points[mask > 0.5]

    rgb = image[mask > 0.5].to(device)
    alpha = torch.ones_like(rgb, device=device)[..., :1]
    rgb = torch.cat([rgb, alpha], dim=1)

    return points, rgb



def load_rgbd_data(path="data/rgbd_data.pkl"):
    with open(path, "rb") as f:
        data = pickle.load(f)

    rgb1 = torch.tensor(data["rgb1"], dtype=torch.float32, device=device) / 255.0
    depth1 = torch.tensor(data["depth1"], dtype=torch.float32, device=device)
    mask1 = torch.tensor(data["mask1"], dtype=torch.float32, device=device)

    rgb2 = torch.tensor(data["rgb2"], dtype=torch.float32, device=device) / 255.0
    depth2 = torch.tensor(data["depth2"], dtype=torch.float32, device=device)
    mask2 = torch.tensor(data["mask2"], dtype=torch.float32, device=device)

    cam1 = data["cameras1"].to(device)
    cam2 = data["cameras2"].to(device)

    return rgb1, depth1, mask1, cam1, rgb2, depth2, mask2, cam2



def get_points_renderer(image_size=256, background_color=(1, 1, 1)):
    raster_settings = PointsRasterizationSettings(
        image_size=image_size,
        radius=0.01,
        points_per_pixel=10,
    )
    rasterizer = PointsRasterizer(cameras=None, raster_settings=raster_settings)
    renderer = PointsRenderer(
        rasterizer=rasterizer,
        compositor=AlphaCompositor(background_color=background_color),
    )
    return renderer



def render_rgbd_pointclouds(data_path="data/rgbd_data.pkl"):
    rgb1, depth1, mask1, cam1, rgb2, depth2, mask2, cam2 = load_rgbd_data(data_path)

    points1, colors1 = unproject_depth_image(rgb1, mask1, depth1, cam1)
    points2, colors2 = unproject_depth_image(rgb2, mask2, depth2, cam2)

    points_union = torch.cat([points1, points2], dim=0).to(device)
    colors_union = torch.cat([colors1, colors2], dim=0).to(device)

    pointcloud_union = Pointclouds(points=[points_union], features=[colors_union]).to(device)

    renderer = get_points_renderer()

    azimuth_values = [0, 120, 240]
    rendered_images = []
    for azim in azimuth_values:
        R, T = look_at_view_transform(dist=2.5, elev=10, azim=azim)  # dist nhỏ hơn
        cameras = FoVPerspectiveCameras(R=R, T=T, device=device)
        image = renderer(pointcloud_union, cameras=cameras)
        img = image[0, ..., :3].detach().cpu().numpy()
        rendered_images.append(img)

    fig, axes = plt.subplots(1, len(rendered_images), figsize=(8, 3))  # khung nhỏ hơn
    for i, img in enumerate(rendered_images):
        axes[i].imshow(img.clip(0, 1))
        axes[i].set_title(f"View {i}", fontsize=8)   # chữ nhỏ hơn
        axes[i].axis("off")
    plt.tight_layout()
    plt.show()

# ---------------------------
# Entry point
# ---------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/rgbd_data.pkl")
    args = parser.parse_args()

    render_rgbd_pointclouds(args.data_path)
