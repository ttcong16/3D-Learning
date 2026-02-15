import torch
import numpy as np
import imageio
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer
)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load mesh 
def load_mesh(obj_path, device):
    return load_objs_as_meshes([obj_path], device=device)


# Set camera
def set_camera(device, dist=2.7, elev=10, azim=150):
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    return cameras


# Set rasterizer
def set_rasterizer(cameras, image_size=512):
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1
    )
    return MeshRasterizer(cameras=cameras, raster_settings=raster_settings)


# Render depth map (numpy array)
def render_depthmap(mesh, rasterizer):
    fragments = rasterizer(mesh)
    depth_map = fragments.zbuf[0, ..., 0].cpu().numpy()
    depth_map[np.isinf(depth_map)] = 0  # remove background
    return depth_map


# Save depth to PNG
def save_image(depth_map, out_path="output/depth_map.png"):
    depth_min, depth_max = depth_map.min(), depth_map.max()
    depth_norm = (depth_map - depth_min) / (depth_max - depth_min)
    depth_img = (depth_norm * 255).astype(np.uint8)
    imageio.imwrite(out_path, depth_img)


if __name__ == "__main__":
    mesh = load_mesh("data/cow.obj", device)
    cameras = set_camera(device, dist=2.7, elev=10, azim=150)
    rasterizer = set_rasterizer(cameras, image_size=512)
    depth_map = render_depthmap(mesh, rasterizer)
    save_image(depth_map, "output/cow_depth.png")
