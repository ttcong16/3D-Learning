import torch
import imageio
import numpy as np
from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras, PointLights,
    RasterizationSettings, MeshRasterizer,
    SoftPhongShader, MeshRenderer, TexturesVertex,
    look_at_view_transform
)
from PIL import Image


# Create renderer
R, T = look_at_view_transform(3.0, 0, 0)
cameras = FoVPerspectiveCameras(device="cpu", R=R, T=T)


lights = PointLights(device="cpu", location=[[2.0, 2.0, 2.0]])

raster_settings = RasterizationSettings(
    image_size=512,      
    blur_radius=1e-6,  
    faces_per_pixel=1,
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=SoftPhongShader(cameras=cameras, lights=lights)
)

# Define vertices cube (8 vertex, length 1)
verts = torch.tensor([
    [-0.5, -0.5, -0.5],  # 0
    [ 0.5, -0.5, -0.5],  # 1
    [ 0.5,  0.5, -0.5],  # 2
    [-0.5,  0.5, -0.5],  # 3
    [-0.5, -0.5,  0.5],  # 4
    [ 0.5, -0.5,  0.5],  # 5
    [ 0.5,  0.5,  0.5],  # 6
    [-0.5,  0.5,  0.5],  # 7
], dtype=torch.float32)

# Define faces (12 triangle from 6 square face)
faces = torch.tensor([
    [0, 1, 2], [0, 2, 3],  # Back side
    [4, 5, 6], [4, 6, 7],  # Front side
    [0, 4, 7], [0, 7, 3],  # Left side
    [1, 5, 6], [1, 6, 2],  # Right side
    [0, 1, 5], [0, 5, 4],  # Bottom
    [3, 2, 6], [3, 6, 7],  # Top side
], dtype=torch.int64)

# Create texture
verts_rgb = torch.tensor([[0.1, 0.7, 1.0]] * verts.shape[0])  # blue
textures = TexturesVertex(verts_features=verts_rgb.unsqueeze(0))

# Create mesh
mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

# Render multiple views for gif
frames = []
num_views = 36  

for i in range(num_views):
    angle = 360.0 * i / num_views
    R, T = look_at_view_transform(3.0, 20, angle)  
    image = renderer(mesh, R=R, T=T)
    img = (image[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
    frames.append(Image.fromarray(img))

# GIF
frames[0].save("output/cube.gif", save_all=True, append_images=frames[1:], duration=100, loop=0)
