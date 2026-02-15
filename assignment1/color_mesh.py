import torch
import numpy as np
import imageio
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    PointLights,
    RasterizationSettings,
    MeshRasterizer,
    MeshRenderer,
    HardPhongShader,
    TexturesVertex
)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


# Load cow.obj, texture follow z
def load_mesh(obj_path, color_front, color_back, device):
    mesh = load_objs_as_meshes([obj_path], device=device)
    verts = mesh.verts_packed()
    faces = mesh.faces_packed()

    z_vals = verts[:, 2]
    z_min, z_max = z_vals.min(), z_vals.max()
    alpha = (z_vals - z_min) / (z_max - z_min)

    colors = (1 - alpha).unsqueeze(1) * torch.tensor(color_front, device=device) \
             + alpha.unsqueeze(1) * torch.tensor(color_back, device=device)

    textures = TexturesVertex(colors.unsqueeze(0))
    mesh_colored = Meshes(verts=[verts], faces=[faces], textures=textures)
    return mesh_colored


# Create renderer
def create_renderer(image_size=512):
    lights = PointLights(location=[[2.0, 2.0, -2.0]], device=device)
    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)
    cameras = FoVPerspectiveCameras(device=device)
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
        shader=HardPhongShader(device=device, cameras=cameras, lights=lights)
    )
    return renderer


# Make gif
def make_gif(mesh, renderer, n_frames=36, distance=3.0, out_path="output/cow_color.gif"):
    frames = []
    for azim in np.linspace(0, 360, n_frames):
        R, T = look_at_view_transform(distance, 0, azim)
        R, T = R.to(device), T.to(device)
        img = renderer(mesh, R=R, T=T)
        img = (img[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
        frames.append(img)
    imageio.mimsave(out_path, frames, duration=100)


if __name__ == "__main__":
    obj_path = "data/cow.obj"
    color1 = [0.0, 0.0, 1.0]  # front: blue
    color2 = [1.0, 0.0, 0.0]  # back: red

    cow_mesh = load_mesh(obj_path, color1, color2, device)
    renderer = create_renderer(image_size=512)
    make_gif(cow_mesh, renderer, n_frames=48, distance=3.0, out_path="output/cow_color.gif")
