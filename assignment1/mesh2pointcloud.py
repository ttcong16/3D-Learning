import torch
import numpy as np
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Pointclouds
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    look_at_view_transform,
    AlphaCompositor,
    PointsRasterizationSettings,
    PointsRasterizer,
    PointsRenderer
)
import imageio

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Load mesh
mesh = load_objs_as_meshes(["data/cow.obj"], device=device)


# Set camera
R, T = look_at_view_transform(dist=3.0, elev=0.0, azim=180)
cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

# Create renderer
raster_settings = PointsRasterizationSettings(image_size=512, radius=0.03)
rasterizer = PointsRasterizer(raster_settings=raster_settings)
compositor = AlphaCompositor(background_color=[1.0, 1.0, 1.0])
renderer = PointsRenderer(rasterizer=rasterizer, compositor=compositor)


# Sample points from mesh, normals, textures
points, normals, textures = sample_points_from_meshes(
    mesh, num_samples=50000, return_normals=True, return_textures=True
)

np.savez(
    "data/cow_pointcloud.npz",
    verts=points[0].cpu().numpy(),
    normals=normals[0].cpu().numpy(),
    rgb=textures[0].cpu().numpy()
)

# Create Pointclouds from file .npz
cow_pointclouds = np.load("data/cow_pointcloud.npz")
points_new = torch.tensor(cow_pointclouds["verts"], dtype=torch.float32).unsqueeze(0).to(device)
textures_new = torch.tensor(cow_pointclouds["rgb"], dtype=torch.float32).unsqueeze(0).to(device)
point_clouds = Pointclouds(points=points_new, features=textures_new)

# Render point cloud
images = renderer(point_clouds, cameras=cameras)  # [1,H,W,4] 

# Save image
image_np = images[0, ..., :3].cpu().numpy()
image_np = (image_np * 255).astype(np.uint8)

imageio.imwrite("output/cow_pointcloud.png", image_np)

# # Render multiple views for GIF
# frames = []
# for azim in range(0, 360, 5):   # rotation 0->355, step 5
#     R, T = look_at_view_transform(dist=3.0, elev=20.0, azim=azim)
#     cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

#     images = renderer(point_clouds, cameras=cameras)  # [1,H,W,4]
#     image_np = images[0, ..., :3].cpu().numpy()
#     image_np = (image_np * 255).astype(np.uint8)

#     frames.append(image_np)

# # Save GIF
# imageio.mimsave("output/cow_pointcloud.gif", frames, fps=20)