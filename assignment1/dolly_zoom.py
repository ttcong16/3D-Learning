import argparse
import imageio
import torch
import pytorch3d
import numpy as np
from PIL import Image, ImageDraw
from pytorch3d.renderer import (
    FoVPerspectiveCameras,
    PointLights,
    MeshRenderer,
    MeshRasterizer,
    RasterizationSettings,
    HardPhongShader,
)
from pytorch3d.io import load_objs_as_meshes
import math

def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

# Create render
def get_mesh_renderer(image_size=512, lights=None, device=None):
    raster_settings = RasterizationSettings(
        image_size=image_size, blur_radius=0.0, faces_per_pixel=1,
    )
    renderer = MeshRenderer(
        rasterizer=MeshRasterizer(raster_settings=raster_settings),
        shader=HardPhongShader(device=device, lights=lights),
    )
    return renderer

def dolly_zoom(image_size=256, num_frames=30, duration=3, output="output/dolly.gif"):
    device = get_device()

    # Load mesh
    mesh = pytorch3d.io.load_objs_as_meshes(["data/cow_on_plane.obj"]).to(device)

    # Renderer , lights
    lights = PointLights(location=[[0.0, 0.0, -3.0]], device=device)
    renderer = get_mesh_renderer(image_size=image_size, lights=lights, device=device)

    # FOV 
    fovs = torch.linspace(5, 120, num_frames)

    images = []
    for fov in fovs:
        distance = 150 / fov.item()
        T = [[0, 0, distance]]

        cameras = FoVPerspectiveCameras(fov=fov, T=T, device=device)
        image = renderer(mesh, cameras=cameras, lights=lights)[0, ..., :3].cpu().numpy()
        image = (image * 255).astype(np.uint8)

        im = Image.fromarray(image)
        draw = ImageDraw.Draw(im)
        draw.text((20, 20), f"fov: {fov:.2f}", fill=(255, 0, 0))
        images.append(np.array(im))

    imageio.mimsave(output, images, duration=duration, loop=0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_frames", type=int, default=30)
    parser.add_argument("--duration", type=float, default=3)
    parser.add_argument("--output", type=str, default="output/dolly.gif")
    parser.add_argument("--size", type=int, default=256)
    args = parser.parse_args()

    dolly_zoom(
        image_size=args.size,
        num_frames=args.num_frames,
        duration=args.duration,
        output=args.output,
    )