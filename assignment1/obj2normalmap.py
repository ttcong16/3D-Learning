import torch
import numpy as np
import matplotlib.pyplot as plt
import imageio

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRasterizer,
)

# Load mesh
def load_mesh(obj_path, device):
    mesh = load_objs_as_meshes([obj_path], device=device)
    return mesh

# Set camera
def set_camera(device):
    R, T = look_at_view_transform(2.7, 0, 180)  
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    return cameras

# Set rasterizer
def set_rasterizer(cameras, image_size, device):
    raster_settings = RasterizationSettings(
        image_size=image_size,
        blur_radius=0.0,
        faces_per_pixel=1
    )
    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    return rasterizer

# Render normal map
def render_normalmap(mesh, rasterizer, cameras, device, image_size=256):
    fragments = rasterizer(mesh)

    faces = mesh.faces_packed()        # (F, 3)
    verts_normals = mesh.verts_normals_packed()
    face_normals = verts_normals[faces].mean(dim=1)  # (F, 3)

    face_idx = fragments.pix_to_face[0, ..., 0]  # (H, W)

    H, W = face_idx.shape
    normal_map = torch.zeros((H, W, 3), device=device)

    mask = face_idx >= 0
    normal_map[mask] = face_normals[face_idx[mask]]

    normal_map = (normal_map + 1.0) / 2.0  # [-1,1] -> [0,1]
    normal_map = (normal_map.cpu().numpy() * 255).astype(np.uint8)

    return normal_map

def main():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    obj_path = "data/cow.obj"

    mesh = load_mesh(obj_path, device)
    cameras = set_camera(device)
    rasterizer = set_rasterizer(cameras, 256, device)

    normal_map = render_normalmap(mesh, rasterizer, cameras, device, 256)

    # Save image
    imageio.imwrite("output/cow_normalmap.png", normal_map)
    # plt.imshow(normal_map)
    # plt.axis("off")
    # plt.show()

if __name__ == "__main__":
    main()
