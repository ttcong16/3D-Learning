import torch
import numpy as np
import pytorch3d
import pytorch3d.io
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    TexturesVertex,
    SoftPhongShader,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    PointLights
)
from pytorch3d.structures import Meshes
import imageio
import argparse

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Load mesh 
 
def load_mesh(path: str):
    verts, faces_obj, _ = pytorch3d.io.load_obj(path)
    faces = faces_obj.verts_idx  #(F, 3)
    verts = verts.unsqueeze(0)   # (1, N, 3)
    faces = faces.unsqueeze(0)   # (1, F, 3)

    # create white for mesh
    verts_rgb = torch.ones_like(verts)
    textures = TexturesVertex(verts_rgb)
    mesh = Meshes(verts=verts, faces=faces, textures=textures)
    return mesh.to(device)


# Set camera
def set_camera(dist=3.0, elev=0.0, azim=0.0):
    R, T = look_at_view_transform(dist=dist, elev=elev, azim=azim)
    cameras = FoVPerspectiveCameras(device=device, R=R, T=T)
    return cameras

# Create renderer
def create_renderer(image_size=512):
    lights = PointLights(location=[[0, 0, -3]], device=device)
    raster_settings = RasterizationSettings(image_size=image_size)
    rasterizer = MeshRasterizer(raster_settings=raster_settings)
    shader = SoftPhongShader(device=device, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)
    return renderer, lights

# Render a frame
def render_frame(renderer, mesh, cameras, lights):
    image = renderer(mesh, cameras=cameras, lights=lights) #(1, H, W, 4)
    image = image[0, ..., :3].cpu().numpy() #(H, W, 3)
    return (image * 255).astype(np.uint8)

# Render 360 GIF
def render_360(path, num_frames=12, dist=3.0, elev=0.0, image_size=512):
    mesh = load_mesh(path)
    renderer, lights = create_renderer(image_size=image_size)

    images = []
    angles = torch.linspace(0, 360, steps=num_frames, dtype=torch.float32)

    for azim in angles:
        cameras = set_camera(dist=dist, elev=elev, azim=azim.item())
        frame = render_frame(renderer, mesh, cameras, lights)
        images.append(frame)

    return images

# Save gif
def save_gif(images, output="output/render_3D.gif", fps=15):
    duration = 1000 // fps
    imageio.mimsave(output, images, duration=duration)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Render 360 GIF")
    parser.add_argument("--input", type=str, default="data/cow.obj", help="Path to OBJ file")
    parser.add_argument("--output", type=str, default="output/render_3D.gif", help="Output GIF path")
    parser.add_argument("--num_frames", type=int, default=12, help="Number of frames in GIF")
    parser.add_argument("--dist", type=float, default=3.0, help="Distance camera-object")
    parser.add_argument("--elev", type=float, default=0.0, help="Elevation angle")
    parser.add_argument("--size", type=int, default=512, help="Image size")
    args = parser.parse_args()

    images = render_360(args.input, args.num_frames, args.dist, args.elev, args.size)
    save_gif(images, args.output)

