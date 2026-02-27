import os
import warnings
import sys
import hydra
import numpy as np
import torch
import tqdm
import imageio
import torch.nn.functional as F
from omegaconf import DictConfig
from PIL import Image
from pytorch3d.renderer import (
    PerspectiveCameras,
    look_at_view_transform
)
import matplotlib.pyplot as plt

from implicit import implicit_dict
from sampler import sampler_dict
from renderer import renderer_dict
from ray_utils import (
    sample_images_at_xy,
    get_pixels_from_image,
    get_random_pixels_from_image,
    get_rays_from_pixels
)
from data_utils import (
    dataset_from_config,
    create_surround_cameras,
    vis_grid,
    vis_rays,
)
from dataset import (
    get_nerf_datasets,
    trivial_collate,
)
from render_functions import render_points


# Model class containing:
#   1) Implicit volume defining the scene
#   2) Sampling scheme which generates sample points along rays
#   3) Renderer which can render an implicit volume given a sampling scheme

class Model(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()
        

        # Get implicit function from config
        self.implicit_fn = implicit_dict[cfg.implicit_function.type](
            cfg.implicit_function
        )

        # Point sampling (raymarching) scheme
        self.sampler = sampler_dict[cfg.sampler.type](
            cfg.sampler
        )

        # Initialize volume renderer
        self.renderer = renderer_dict[cfg.renderer.type](
            cfg.renderer
        )
    
    def forward(
        self,
        ray_bundle
    ):
        # Call renderer with
        #  a) Implicit volume
        #  b) Sampling routine

        return self.renderer(
            self.sampler,
            self.implicit_fn,
            ray_bundle
        )

#Q4.2
class Modelcoarse_fine(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()

        # Two implicit functions: coarse and fine
        self.implicit_fn_coarse = implicit_dict[cfg.implicit_function.type](
            cfg.implicit_function
        )
        self.implicit_fn_fine = implicit_dict[cfg.implicit_function.type](
            cfg.implicit_function
        )

        # Samplers (reuse same config)
        self.sampler_coarse = sampler_dict[cfg.sampler.type](cfg.sampler)
        self.sampler_fine = sampler_dict[cfg.sampler.type](cfg.sampler)

        # One renderer shared
        self.renderer = renderer_dict[cfg.renderer.type](cfg.renderer)
        self.sampler = self.sampler_coarse

    def forward(self, ray_bundle):
        """Two-stage coarse_fine NeRF forward pass"""
        # ---- 1. Coarse pass ----
        out_coarse = self.renderer(
            self.sampler_coarse,
            self.implicit_fn_coarse,
            ray_bundle
        )
   

        # ---- 2. Fine pass ----
        # use importance sampling if weights available
        weights = out_coarse.get("weights", None)
        if weights is not None and hasattr(self.sampler_fine, "fine_sampling"):
            ray_bundle_fine = self.sampler_fine.fine_sampling(ray_bundle, weights)
        else:
            # fallback to same uniform sampling
            ray_bundle_fine = self.sampler_fine(ray_bundle)

        out_fine = self.renderer(
            self.sampler_fine,
            self.implicit_fn_fine,
            ray_bundle_fine
        )

        return {
            "coarse": out_coarse,
            "fine": out_fine
        }

def render_images(
    model,
    cameras,
    image_size,
    save=False,
    file_prefix=''
):
    all_images = []
    device = list(model.parameters())[0].device

    for cam_idx, camera in enumerate(cameras):
        print(f'Rendering image {cam_idx}')

        torch.cuda.empty_cache()
        camera = camera.to(device)
        xy_grid = get_pixels_from_image(image_size, camera) # TODO (Q1.3): implement in ray_utils.py
        ray_bundle = get_rays_from_pixels(xy_grid, image_size, camera) # TODO (Q1.3): implement in ray_utils.py

        # TODO (Q1.3): Visualize xy grid using vis_grid
        if cam_idx == 0 and file_prefix == '':
            image = vis_grid(xy_grid= xy_grid, image_size= image_size)
            plt.imsave('images/xy_grid.png', image)

        # TODO (Q1.3): Visualize rays using vis_rays
        if cam_idx == 0 and file_prefix == '':
            image = vis_rays(ray_bundle= ray_bundle, image_size= image_size)
            plt.imsave('images/ray_bundle.png', image)
        
        # TODO (Q1.4): Implement point sampling along rays in sampler.py
        ray_bundle = model.sampler(ray_bundle)

        # TODO (Q1.4): Visualize sample points as point cloud
        if cam_idx == 0 and file_prefix == '':
            from render_functions import render_points
            pts = ray_bundle.sample_points  # (n_rays, n_pts, 3)
            points = pts.view(1, -1, 3)
            render_points(filename="images/sampled_points.png", points=points)   

        # TODO (Q1.5): Implement rendering in renderer.py
        out = model(ray_bundle)

        # Return rendered features (colors)
        # image = np.array(
        #     out['feature'].view(
        #         image_size[1], image_size[0], 3
        #     ).detach().cpu()
        # )

        # single-network and coarse_fine models
        if "feature" in out:
            image = out["feature"]
        elif "fine" in out and "feature" in out["fine"]:
            image = out["fine"]["feature"]
        elif "coarse" in out and "feature" in out["coarse"]:
            image = out["coarse"]["feature"]
        image = np.array(
            image.view(image_size[1], image_size[0], 3).detach().cpu()
        )
        all_images.append(image)

        # TODO (Q1.5): Visualize depth
        if cam_idx == 2 and file_prefix == '':
            depth = out['depth'].detach().cpu().reshape(image_size[1], image_size[0]).numpy()
            depth_min, depth_max = np.percentile(depth, [1, 99])
            depth = np.clip((depth - depth_min) / (depth_max - depth_min + 1e-8), 0, 1)
            plt.imsave('images/depth_map.png', depth, cmap='plasma') 

        image = np.clip(image, 0.0, 1.0)


        # Save
        if save:
            plt.imsave(
                f'{file_prefix}_{cam_idx}.png',
                image
            )
    
    return all_images


def render(
    cfg,
):
    # Create model
    model = Model(cfg)
    model = model.cuda(); model.eval()

    # Render spiral
    cameras = create_surround_cameras(3.0, n_poses=20)
    all_images = render_images(
        model, cameras, cfg.data.image_size
    )
    imageio.mimsave('images/part_1.gif', [np.uint8(im * 255) for im in all_images], loop=0)


def train(
    cfg
):
    # Create model
    model = Model(cfg)
    model = model.cuda(); model.train()

    # Create dataset 
    train_dataset = dataset_from_config(cfg.data)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=lambda batch: batch,
    )
    image_size = cfg.data.image_size

    # Create optimizer 
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr
    )

    # Render images before training
    cameras = [item['camera'] for item in train_dataset]
    render_images(
        model, cameras, image_size,
        save=True, file_prefix='images/part_2_before_training'
    )

    # Train
    t_range = tqdm.tqdm(range(cfg.training.num_epochs))

    for epoch in t_range:
        for iteration, batch in enumerate(train_dataloader):
            image, camera, camera_idx = batch[0].values()
            image = image.cuda()
            camera = camera.cuda()

            # Sample rays
            xy_grid = get_random_pixels_from_image(cfg.training.batch_size, image_size, camera) # TODO (Q2.1): implement in ray_utils.py
            ray_bundle = get_rays_from_pixels(xy_grid, image_size, camera)
            rgb_gt = sample_images_at_xy(image, xy_grid)

            # Run model forward
            out = model(ray_bundle)

            # TODO (Q2.2): Calculate loss

            loss = F.mse_loss(out['feature'], rgb_gt)

            # Backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if (epoch % 10) == 0:
            t_range.set_description(f'Epoch: {epoch:04d}, Loss: {loss:.06f}')
            t_range.refresh()

    # Print center and side lengths
    print("Box center:", tuple(np.array(model.implicit_fn.sdf.center.data.detach().cpu()).tolist()[0]))
    print("Box side lengths:", tuple(np.array(model.implicit_fn.sdf.side_lengths.data.detach().cpu()).tolist()[0]))

    # Render images after training
    render_images(
        model, cameras, image_size,
        save=True, file_prefix='images/part_2_after_training'
    )
    all_images = render_images(
        model, create_surround_cameras(3.0, n_poses=20), image_size, file_prefix='part_2'
    )
    imageio.mimsave('images/part_2.gif', [np.uint8(im * 255) for im in all_images], loop=0)


def create_model(cfg):
    # Create model
    model = Model(cfg)
    model.cuda(); model.train()

    # Load checkpoints
    optimizer_state_dict = None
    start_epoch = 0

    checkpoint_path = os.path.join(
        hydra.utils.get_original_cwd(),
        cfg.training.checkpoint_path
    )

    if len(cfg.training.checkpoint_path) > 0:
        # Make the root of the experiment directory.
        checkpoint_dir = os.path.split(checkpoint_path)[0]
        os.makedirs(checkpoint_dir, exist_ok=True)

        # Resume training if requested.
        if cfg.training.resume and os.path.isfile(checkpoint_path):
            print(f"Resuming from checkpoint {checkpoint_path}.")
            loaded_data = torch.load(checkpoint_path)
            model.load_state_dict(loaded_data["model"])
            start_epoch = loaded_data["epoch"]

            print(f"   => resuming from epoch {start_epoch}.")
            optimizer_state_dict = loaded_data["optimizer"]

    # Initialize the optimizer.
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.training.lr,
    )

    # Load the optimizer state dict in case we are resuming.
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
        optimizer.last_epoch = start_epoch

    # The learning rate scheduling is implemented with LambdaLR PyTorch scheduler.
    def lr_lambda(epoch):
        return cfg.training.lr_scheduler_gamma ** (
            epoch / cfg.training.lr_scheduler_step_size
        )

    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda, last_epoch=start_epoch - 1, verbose=False
    )

    return model, optimizer, lr_scheduler, start_epoch, checkpoint_path

def train_nerf(
    cfg
):
    # Create model
    model, optimizer, lr_scheduler, start_epoch, checkpoint_path = create_model(cfg)

    # Load the training/validation data.
    train_dataset, val_dataset, _ = get_nerf_datasets(
        dataset_name=cfg.data.dataset_name,
        image_size=[cfg.data.image_size[1], cfg.data.image_size[0]],
    )

    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=trivial_collate,
    )

    # Run the main training loop.
    for epoch in range(start_epoch, cfg.training.num_epochs):
        t_range = tqdm.tqdm(enumerate(train_dataloader))

        for iteration, batch in t_range:
            image, camera, camera_idx = batch[0].values()
            image = image.cuda().unsqueeze(0)
            camera = camera.cuda()

            # Sample rays
            xy_grid = get_random_pixels_from_image(
                cfg.training.batch_size, cfg.data.image_size, camera
            )
            ray_bundle = get_rays_from_pixels(
                xy_grid, cfg.data.image_size, camera
            )
            rgb_gt = sample_images_at_xy(image, xy_grid)

            # Run model forward
            out = model(ray_bundle)

            # TODO (Q3.1): Calculate loss
            rgb_pred = out["feature"]
            loss = F.mse_loss(rgb_pred, rgb_gt)

            # Take the training step.
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_range.set_description(f'Epoch: {epoch:04d}, Loss: {loss:.06f}')
            t_range.refresh()

        # Adjust the learning rate.
        lr_scheduler.step()

        # Checkpoint.
        if (
            epoch % cfg.training.checkpoint_interval == 0
            and len(cfg.training.checkpoint_path) > 0
            and epoch > 0
        ):
            print(f"Storing checkpoint {checkpoint_path}.")

            data_to_store = {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch,
            }

            torch.save(data_to_store, checkpoint_path)

        # Render
        if (
            epoch % cfg.training.render_interval == 0
            and epoch > 0
        ):
            with torch.no_grad():
                test_images = render_images(
                    model, create_surround_cameras(4.0, n_poses=20, up=(0.0, 0.0, 1.0), focal_length=2.0),
                    cfg.data.image_size, file_prefix='nerf'
                )
                imageio.mimsave('images/part_3.gif', [np.uint8(im * 255) for im in test_images], loop=0)

        with torch.no_grad():
            model.eval()
            cameras = create_surround_cameras(
                radius=4.0,
                n_poses=40,
                up=(0.0, 0.0, 1.0),
                focal_length=2.0
            )
            final_images = render_images(
                model, cameras, cfg.data.image_size, file_prefix='part_4'
            )
            os.makedirs("images", exist_ok=True)
            # gif_path = os.path.join("images", "part_4.gif")
            args = sys.argv
            # Tìm xem có chứa tên config nào
            config_name = "unknown"
            for a in args:
                if "--config-name=" in a:
                    config_name = a.split("=")[-1]
                    break

            if "highres" in config_name.lower():
                gif_name = "part_42.gif"
            else:
                gif_name = "part_41.gif"

            gif_path = os.path.join("images", gif_name)
            imageio.mimsave(gif_path, [np.uint8(im * 255) for im in final_images], loop=0)

#Q4.2
def train_coarse_fine(cfg):

    # Create coarse_fine model
    model = Modelcoarse_fine(cfg)
    model = model.cuda(); model.train()

    # Dataloader reuse
    train_dataset, _, _ = get_nerf_datasets(
        dataset_name=cfg.data.dataset_name,
        image_size=[cfg.data.image_size[1], cfg.data.image_size[0]],
    )
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        collate_fn=trivial_collate,
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.training.lr)

    # Training loop
    for epoch in range(cfg.training.num_epochs):
        t_range = tqdm.tqdm(enumerate(train_dataloader))
        for iteration, batch in t_range:
            image, camera, camera_idx = batch[0].values()
            image = image.cuda().unsqueeze(0)
            camera = camera.cuda()

            xy_grid = get_random_pixels_from_image(
                cfg.training.batch_size, cfg.data.image_size, camera
            )
            ray_bundle = get_rays_from_pixels(xy_grid, cfg.data.image_size, camera)
            rgb_gt = sample_images_at_xy(image, xy_grid)

            # Forward coarse_fine model
            out = model(ray_bundle)
            rgb_coarse = out["coarse"]["feature"]
            rgb_fine = out["fine"]["feature"]

            # Loss = coarse + fine
            loss = F.mse_loss(rgb_coarse, rgb_gt) + F.mse_loss(rgb_fine, rgb_gt)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_range.set_description(f'Epoch {epoch:03d} | Loss: {loss:.6f}')

    # Render example output
    with torch.no_grad():
        model.eval()
        cameras = create_surround_cameras(
                radius=4.0,
                n_poses=20,
                up=(0.0, 0.0, 1.0),  # <-- This is the key fix
                focal_length=2.0
            )

        imgs = render_images(
            model,
            cameras,
            cfg.data.image_size,
            file_prefix="coarse_fine"
        )

        os.makedirs("images", exist_ok=True)
        imageio.mimsave(
            "images/part_4.gif",
            [np.uint8(im * 255) for im in imgs],
            loop=0
        )

def train_coarse(cfg):
    model = Modelcoarse_fine(cfg).cuda()
    model.train()

    for p in model.implicit_fn_coarse.parameters():
        p.requires_grad = True
    for p in model.implicit_fn_fine.parameters():
        p.requires_grad = False

    optimizer = torch.optim.Adam(model.implicit_fn_coarse.parameters(), lr=cfg.training.lr)

    # Dataset
    train_dataset, _, _ = get_nerf_datasets(
        dataset_name=cfg.data.dataset_name,
        image_size=[cfg.data.image_size[1], cfg.data.image_size[0]],
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=trivial_collate
    )

    #Training loop
    for epoch in range(cfg.training.num_epochs):
        t_range = tqdm.tqdm(enumerate(dataloader))
        for iteration, batch in t_range:
            image, camera, _ = batch[0].values()
            image = image.cuda().unsqueeze(0)
            camera = camera.cuda()

            xy_grid = get_random_pixels_from_image(cfg.training.batch_size, cfg.data.image_size, camera)
            ray_bundle = get_rays_from_pixels(xy_grid, cfg.data.image_size, camera)
            rgb_gt = sample_images_at_xy(image, xy_grid)

            out = model.renderer(model.sampler_coarse, model.implicit_fn_coarse, ray_bundle)
            rgb_coarse = out["feature"]

            loss = F.mse_loss(rgb_coarse, rgb_gt)
            optimizer.zero_grad(); loss.backward(); optimizer.step()

            t_range.set_description(f"[COARSE] Epoch {epoch:03d} | Loss: {loss:.6f}")

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.implicit_fn_coarse.state_dict(), "checkpoints/coarse_model.pth")

    # Render
    print("Rendering coarse results ...")
    model.eval()
    cameras = create_surround_cameras(radius=4.0, n_poses=40, up=(0.0,0.0,1.0), focal_length=2.0)
    imgs = []
    for cam_idx, cam in enumerate(cameras):
        torch.cuda.empty_cache()
        xy_grid = get_pixels_from_image(cfg.data.image_size, cam.to("cuda"))
        rays = get_rays_from_pixels(xy_grid, cfg.data.image_size, cam.to("cuda"))
        rays = model.sampler_coarse(rays)
        out = model.renderer(model.sampler_coarse, model.implicit_fn_coarse, rays)
        img = np.array(out["feature"].view(cfg.data.image_size[1], cfg.data.image_size[0], 3).detach().cpu())
        imgs.append(np.clip(img,0,1))
    os.makedirs("images", exist_ok=True)
    imageio.mimsave("images/coarse.gif", [np.uint8(im*255) for im in imgs], loop=0)



def train_fine(cfg):

    # Create model
    model = Modelcoarse_fine(cfg).cuda()
    model.train()

    # Load trained coarse
    coarse_path = "checkpoints/coarse_model.pth"
    if not os.path.exists(coarse_path):
        raise FileNotFoundError("coarse_model.pth not found — train coarse trước!")
    model.implicit_fn_coarse.load_state_dict(torch.load(coarse_path, weights_only=True))
    for p in model.implicit_fn_coarse.parameters():
        p.requires_grad = False
    for p in model.implicit_fn_fine.parameters():
        p.requires_grad = True

    optimizer = torch.optim.Adam(model.implicit_fn_fine.parameters(), lr=cfg.training.lr)

    # Dataset
    train_dataset, _, _ = get_nerf_datasets(
        dataset_name=cfg.data.dataset_name,
        image_size=[cfg.data.image_size[1], cfg.data.image_size[0]],
    )
    dataloader = torch.utils.data.DataLoader(
        train_dataset, batch_size=1, shuffle=True, num_workers=0, collate_fn=trivial_collate
    )

    #Training loop
    for epoch in range(cfg.training.num_epochs):
        t_range = tqdm.tqdm(enumerate(dataloader))
        for iteration, batch in t_range:
            image, camera, _ = batch[0].values()
            image = image.cuda().unsqueeze(0)
            camera = camera.cuda()

            xy_grid = get_random_pixels_from_image(cfg.training.batch_size, cfg.data.image_size, camera)
            ray_bundle = get_rays_from_pixels(xy_grid, cfg.data.image_size, camera)
            rgb_gt = sample_images_at_xy(image, xy_grid)

            # Coarse to get weights
            with torch.no_grad():
                out_coarse = model.renderer(model.sampler_coarse, model.implicit_fn_coarse, ray_bundle)
                weights = out_coarse["weights"]

                weights = weights.squeeze()
                if weights.ndim == 1:
                    weights = weights.unsqueeze(0)
                if weights.ndim > 2:
                    weights = weights.view(weights.shape[0], -1)
                n_rays = ray_bundle.origins.shape[0]
                n_coarse = weights.numel() // n_rays
                weights = weights.view(n_rays, n_coarse)

                weights = torch.nan_to_num(weights, nan=0.0, posinf=0.0, neginf=0.0)
                weights = weights / (weights.sum(dim=-1, keepdim=True) + 1e-5)

                if torch.isnan(weights).any() or torch.isinf(weights).any():
                    weights = torch.ones_like(weights) / weights.shape[-1]

            #Fine pass
            try:
                ray_bundle_fine = model.sampler_fine.fine_sampling(ray_bundle, weights)
            except RuntimeError as e:
                print(f"[WARN] Fine sampling shape error: {e}")
                weights = weights.squeeze()
                if weights.ndim == 1:
                    weights = weights.unsqueeze(0)
                if weights.ndim > 2:
                    weights = weights.view(weights.shape[0], -1)
                n_rays = ray_bundle.origins.shape[0]
                n_coarse = weights.shape[-1]
                weights = weights.view(n_rays, n_coarse)
                ray_bundle_fine = model.sampler_fine.fine_sampling(ray_bundle, weights)

            out_fine = model.renderer(model.sampler_fine, model.implicit_fn_fine, ray_bundle_fine)
            rgb_fine = out_fine["feature"]

            #Loss
            loss = F.mse_loss(rgb_fine, rgb_gt)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            t_range.set_description(f"[FINE] Epoch {epoch:03d} | Loss: {loss:.6f}")
            del out_coarse, out_fine, ray_bundle_fine
            torch.cuda.empty_cache()

    os.makedirs("checkpoints", exist_ok=True)
    torch.save(model.implicit_fn_fine.state_dict(), "checkpoints/fine_model.pth")

     # Render
    model.eval()
    model.implicit_fn_fine.load_state_dict(torch.load("checkpoints/fine_model.pth", weights_only=True))
    cameras = create_surround_cameras(radius=4.0, n_poses=10, up=(0.0,0.0,1.0), focal_length=2.0)

    def forward_(ray_bundle):
        out_coarse = model.renderer(model.sampler_coarse, model.implicit_fn_coarse, ray_bundle)
        weights = out_coarse["weights"]

        weights = weights.squeeze()
        if weights.ndim == 1:
            weights = weights.unsqueeze(0)
        if weights.ndim > 2:
            weights = weights.view(weights.shape[0], -1)
        ray_bundle_fine = model.sampler_fine.fine_sampling(ray_bundle, weights)
        out_fine = model.renderer(model.sampler_fine, model.implicit_fn_fine, ray_bundle_fine)
        return out_fine
    model.forward = forward_


    imgs = render_images(model, cameras, [64, 64], file_prefix="fine")
    os.makedirs("images", exist_ok=True)
    imageio.mimsave(
        "images/fine.gif",
        [np.uint8(im * 255) for im in imgs],
        duration=1.0, loop = 0
    )


@hydra.main(config_path='./configs', config_name='sphere')
def main(cfg: DictConfig):
    os.chdir(hydra.utils.get_original_cwd())

    if cfg.type == 'render':
        render(cfg)
    elif cfg.type == 'train':
        train(cfg)
    elif cfg.type == 'train_nerf':
        train_nerf(cfg)
    elif cfg.type == 'train_coarse_fine':
        train_coarse_fine(cfg)
    elif cfg.type == 'train_coarse':
        train_coarse(cfg)
    elif cfg.type == 'train_fine':
        train_fine(cfg)


if __name__ == "__main__":
    main()

