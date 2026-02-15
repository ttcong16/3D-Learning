import torch
import numpy as np
import imageio
import matplotlib.pyplot as plt

from pytorch3d.structures import Meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    HardPhongShader,
    PointLights,
    TexturesVertex
)

# Create renderer
R, T = look_at_view_transform(2.7, 0, 0)  
cameras = FoVPerspectiveCameras(device="cpu", R=R, T=T)

lights = PointLights(device="cpu", location=[[2.0, 2.0, 2.0]])

raster_settings = RasterizationSettings(
    image_size=256,
    blur_radius=0.0,
    faces_per_pixel=1,
)

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(cameras=cameras, raster_settings=raster_settings),
    shader=HardPhongShader(cameras=cameras, lights=lights)
)


# Define vertices, faces of tetrahedron
# Create 4 vertex (x,y,z)
verts = torch.tensor([
    [0.0, 0.0, 0.0],   # vertex 0
    [1.0, 0.0, 0.0],   # vertex 1
    [0.5, np.sqrt(3)/2, 0.0],  # Vertex 2
    [0.5, np.sqrt(3)/6, np.sqrt(6)/3]  # vertex 3 (Highest vertex)
], dtype=torch.float32)

# Create 4 faces of triangle by index of each vertex
faces = torch.tensor([
    [0, 1, 2],  # Bottom
    [0, 1, 3],  # Side face
    [1, 2, 3],  # Side face
    [2, 0, 3]   # Side face
], dtype=torch.int64)

# All vertices the same color
verts_rgb = torch.tensor([[0.2, 0.7, 1.0]] * verts.shape[0])  # blue
textures = TexturesVertex(verts_features=verts_rgb[None])

# Create mesh
mesh = Meshes(verts=[verts], faces=[faces], textures=textures)

# Render multiple views for gif
frames = []
num_views = 36  # 36 perspectives → rotation 360 - step 10

for i in range(num_views):
    angle = 360.0 * i / num_views
    R, T = look_at_view_transform(2.7, 0, angle)  # rotation about Y
    image = renderer(mesh, R=R, T=T) #[1, H, W, 4]
    img = image[0, ..., :3].cpu().numpy()   #[0,1]
    frames.append((img * 255).astype(np.uint8))

# Save gif
imageio.mimsave("output/tetrahedron.gif", frames, fps=10)

 