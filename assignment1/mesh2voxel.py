import argparse
import imageio
import torch
import numpy as np
from PIL import Image
import math
from pytorch3d.structures import Meshes
from pytorch3d.renderer import TexturesVertex

from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    HardPhongShader,
    look_at_view_transform
)
from pytorch3d.ops import sample_points_from_meshes

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Load mesh
mesh = load_objs_as_meshes(["data/cow.obj"], device=device)

# Create renderer
raster_settings = RasterizationSettings(image_size=512)
lights = PointLights(device=device, location=[[2.0, 2.0, 2.0]])

renderer = MeshRenderer(
    rasterizer=MeshRasterizer(raster_settings=raster_settings),
    shader=HardPhongShader(device=device, lights=lights)
)


# Sample points (approx voxelization)
num_points = 80000   
points, normals, textures = sample_points_from_meshes(
    mesh, num_points, return_normals=True, return_textures=True
)

points_np = points[0].cpu().numpy()
colors_np = textures[0].cpu().numpy() if textures is not None else np.ones((num_points,3))

# Discretize into voxel grid (follow original size)
grid_size = 128
min_p, max_p = points_np.min(0), points_np.max(0)
bbox_size = max_p - min_p
voxel_size = bbox_size.max() / grid_size   

idx = ((points_np - min_p) / voxel_size).astype(int)
idx = np.clip(idx, 0, grid_size-1)

voxel_dict = {}
for i, voxel in enumerate(map(tuple, idx)):
    if voxel not in voxel_dict:
        voxel_dict[voxel] = []
    voxel_dict[voxel].append(colors_np[i])

coords, colors = [], []
for v, c_list in voxel_dict.items():
    coords.append(v)
    colors.append(np.mean(c_list, axis=0))
coords = np.array(coords, dtype=np.float32)
colors = np.array(colors, dtype=np.float32)

# Create cube mesh for each voxel
cube_verts = torch.tensor([
    [-0.5,-0.5,-0.5],[0.5,-0.5,-0.5],[0.5,0.5,-0.5],[-0.5,0.5,-0.5],
    [-0.5,-0.5,0.5],[0.5,-0.5,0.5],[0.5,0.5,0.5],[-0.5,0.5,0.5]
], dtype=torch.float32)

cube_faces = torch.tensor([
    [0,1,2],[0,2,3],
    [4,5,6],[4,6,7],
    [0,1,5],[0,5,4],
    [2,3,7],[2,7,6],
    [1,2,6],[1,6,5],
    [0,3,7],[0,7,4]
], dtype=torch.int64)

all_verts, all_faces, all_colors = [], [], []
for i, (c, col) in enumerate(zip(coords, colors)):
    v = cube_verts * voxel_size + (torch.tensor(c, dtype=torch.float32) * voxel_size + min_p)
    all_verts.append(v)
    all_faces.append(cube_faces + 8*i)
    all_colors.append(torch.tensor(col).repeat(v.shape[0],1))

all_verts = torch.cat(all_verts, dim=0)
all_faces = torch.cat(all_faces, dim=0)
all_colors = torch.cat(all_colors, dim=0)

voxel_mesh = Meshes(
    verts=[all_verts.to(device)],
    faces=[all_faces.to(device)],
    textures=TexturesVertex(verts_features=[all_colors.to(device)])
)




# 6. Render multiple views for GIF
frames = []
for angle in range(0, 360, 5):
    R, T = look_at_view_transform(dist=5.0, elev=30, azim=angle)  
    cameras = FoVPerspectiveCameras(R=R, T=T, device=device)

    image = renderer(voxel_mesh, cameras=cameras)
    img = (image[0, ..., :3].cpu().numpy() * 255).astype(np.uint8)
    frames.append(img)

imageio.mimsave("output/cow_voxel.gif", frames, fps=20)
