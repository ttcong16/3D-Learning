import argparse
import time
import dataset_location
import losses
import torch
from model import SingleViewto3D
from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from pytorch3d.ops import sample_points_from_meshes
from r2n2_custom import R2N2
import wandb
import os


def get_args_parser():
    parser = argparse.ArgumentParser("Singleto3D", add_help=False)
    # Model parameters
    parser.add_argument("--arch", default="resnet18", type=str)
    parser.add_argument("--lr", default=4e-4, type=float)
    parser.add_argument("--max_iter", default=750, type=int)
    parser.add_argument("--batch_size", default=32, type=int)
    parser.add_argument("--num_workers", default=4, type=int)
    parser.add_argument(
        "--type", default="vox", choices=["vox", "point", "mesh", "implicit", "parametric"], type=str
    )
    parser.add_argument("--n_points", default=1000, type=int)
    parser.add_argument("--w_chamfer", default=1.0, type=float)
    parser.add_argument("--w_smooth", default=0.1, type=float)
    parser.add_argument("--save_freq", default=500, type=int)
    parser.add_argument("--load_checkpoint", action="store_true")
    parser.add_argument('--device', default='cuda', type=str) 
    parser.add_argument('--load_feat', action='store_true') 
    parser.add_argument('--wandb', action='store_true', help='Enable Weights & Biases logging')
    parser.add_argument('--wandb_project', default='singleview3d', type=str)
    parser.add_argument('--wandb_run_name', default=None, type=str)
    parser.add_argument('--wandb_log_every', default=10, type=int, help='log scalars every N steps')
    parser.add_argument('--wandb_vis_every', default=100, type=int, help='log 3D viz every N steps')
    return parser


def preprocess(feed_dict, args):
    images = feed_dict["images"].squeeze(1)
    if args.type == "vox":
        voxels = feed_dict["voxels"].float()
        ground_truth_3d = voxels
    elif args.type == "point":
        mesh = feed_dict["mesh"]
        pointclouds_tgt = sample_points_from_meshes(mesh, args.n_points)
        ground_truth_3d = pointclouds_tgt
    elif args.type == "mesh":
        ground_truth_3d = feed_dict["mesh"]

    elif args.type == "implicit":
        voxels = feed_dict["voxels"].float()  # occupancy GT in {0,1}
        ground_truth_3d = voxels

    elif args.type == "parametric":
        mesh = feed_dict["mesh"]
        pointclouds_tgt = sample_points_from_meshes(mesh, args.n_points)
        ground_truth_3d = pointclouds_tgt


    if args.load_feat:
        feats = torch.stack(feed_dict["feats"])
        return feats.to(args.device), ground_truth_3d.to(args.device)
    else:
        return images.to(args.device), ground_truth_3d.to(args.device)
    
    


def calculate_loss(predictions, ground_truth, args):
    if args.type == "vox":
        loss = losses.voxel_loss(predictions, ground_truth)
    elif args.type == "point":
        loss = losses.chamfer_loss(predictions, ground_truth)
    elif args.type == "mesh":
        sample_trg = sample_points_from_meshes(ground_truth, args.n_points)
        sample_pred = sample_points_from_meshes(predictions, args.n_points)

        loss_reg = losses.chamfer_loss(sample_pred, sample_trg)
        loss_smooth = losses.smoothness_loss(predictions)

        loss = args.w_chamfer * loss_reg + args.w_smooth * loss_smooth

    elif args.type == "implicit":
        loss = losses.voxel_loss(predictions, ground_truth)
    elif args.type == "parametric":
        loss = losses.chamfer_loss(predictions, ground_truth)


    return loss

 
def _wandb_init(args, model):
    if not args.wandb:
        return None
    if wandb is None:
        print('[WARN] wandb not installed; disable --wandb or pip install wandb')
        return None
    run = wandb.init(
        project=args.wandb_project,
        name=args.wandb_run_name,
        config={
            'decoder_type': args.type,
            'arch': args.arch,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'max_iter': args.max_iter,
            'n_points': args.n_points,
            'w_chamfer': args.w_chamfer,
            'w_smooth': args.w_smooth,
            'dataset': 'R2N2'
        },
        resume='allow'
    )
    # log grads every 100 steps (safe/default)
    wandb.watch(model, log='gradients', log_freq=100)
    # useful tags for filtering
    run.tags = [args.type, args.arch]
    return run


def _wandb_log_scalars(step, loss_val, lr, read_t, iter_t):
    if wandb is None:
        return
    wandb.log({
        'step': step,
        'train/loss': float(loss_val),
        'train/lr': float(lr),
        'time/read_s': float(read_t),
        'time/iter_s': float(iter_t),
    }, step=step)

def _wandb_log_pointcloud(tag, pts_tensor):
    if wandb is None:
        return
    try:
        pts = pts_tensor.detach().cpu().numpy()
        wandb.log({tag: wandb.Object3D(pts)})
    except Exception as e:
        print(f"[W&B] pointcloud log failed: {e}")

def _wandb_log_mesh_obj(tag, obj_path):
    if wandb is None:
        return
    if os.path.exists(obj_path):
        try:
           
           wandb.log({tag: wandb.Object3D(obj_path)})
        except Exception as e:
           print(f"[W&B] mesh log failed: {e}")

def _wandb_log_checkpoint(args, step, model, optimizer):
    if wandb is None:
        return
# save locally first (keeps original behavior)
    os.makedirs('.', exist_ok=True)
    ckpt_path = f"checkpoint_{args.type}.pth"
    torch.save({
        'step': step,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, ckpt_path)
# log as artifact with unique name per step to avoid collisions
    try:
        art = wandb.Artifact(
        name=f"ckpt-{args.type}-{wandb.run.id}-{step}",
        type='model',
        metadata={'decoder': args.type, 'arch': args.arch, 'step': step}
        )
        art.add_file(ckpt_path)
        wandb.log_artifact(art)
    except Exception as e:
        print(f"[W&B] artifact log failed: {e}")

def train_model(args):
    r2n2_dataset = R2N2(
        "train",
        dataset_location.SHAPENET_PATH,
        dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH,
        return_voxels=True,
        return_feats=args.load_feat,
    )

    loader = torch.utils.data.DataLoader(
        r2n2_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        drop_last=True,
        shuffle=True,
    )
    train_loader = iter(loader)

    model = SingleViewto3D(args)
    model.to(args.device)
    model.train()

    # ============ preparing optimizer ... ============
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)  # to use with ViTs
    start_iter = 0
    start_time = time.time()

    if args.load_checkpoint:
        checkpoint = torch.load(f"checkpoint_{args.type}.pth")
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        start_iter = checkpoint["step"]
        print(f"Succesfully loaded iter {start_iter}")


    run = _wandb_init(args, model)
    print("Starting training !")
    for step in range(start_iter, args.max_iter):
        iter_start_time = time.time()

        if step % len(train_loader) == 0:  # restart after one epoch
            train_loader = iter(loader)

        read_start_time = time.time()

        feed_dict = next(train_loader)

        images_gt, ground_truth_3d = preprocess(feed_dict, args)
        read_time = time.time() - read_start_time

        prediction_3d = model(images_gt, args)

        loss = calculate_loss(prediction_3d, ground_truth_3d, args)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_time = time.time() - start_time
        iter_time = time.time() - iter_start_time

        loss_vis = loss.cpu().item()

        # --- W&B: scalar logs ---
        if args.wandb and (step % args.wandb_log_every == 0):
            lr = optimizer.param_groups[0]['lr']
            _wandb_log_scalars(step, loss_vis, lr, read_time, iter_time)


        # --- W&B: occasional 3D visualization ---
        if args.wandb and (step % args.wandb_vis_every == 0) and step > 0:
            try:
                if args.type in ("point", "parametric"):
        # prediction_3d is expected to be (B, N, 3) or a structure convertible to that
                    _wandb_log_pointcloud("viz/pred_pointcloud", prediction_3d[0])
                    _wandb_log_pointcloud("viz/gt_pointcloud", ground_truth_3d[0])
                elif args.type == "mesh":
        # Sample to point clouds for quick preview
                    sample_pred = sample_points_from_meshes(prediction_3d, min(args.n_points, 2048))
                    sample_trg = sample_points_from_meshes(ground_truth_3d, min(args.n_points, 2048))
                    _wandb_log_pointcloud("viz/pred_mesh_points", sample_pred[0])
                    _wandb_log_pointcloud("viz/gt_mesh_points", sample_trg[0])
    # For vox/implicit you could add marching-cubes and log mesh if you already export it
            except Exception as e:
                print(f"[W&B] viz log failed: {e}")

        if (step % args.save_freq) == 0 and step > 0:
            print(f"Saving checkpoint at step {step}")
            torch.save(
                {
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                },
                f"checkpoint_{args.type}.pth",
            )
            if args.wandb:
                _wandb_log_checkpoint(args, step, model, optimizer)

        print(
            "[%4d/%4d]; ttime: %.0f (%.2f, %.2f); loss: %.3f"
            % (step, args.max_iter, total_time, read_time, iter_time, loss_vis)
        )

    print("Done!")
    if args.wandb and wandb is not None:
        wandb.finish()

if __name__ == "__main__":
    parser = argparse.ArgumentParser("Singleto3D", parents=[get_args_parser()])
    args = parser.parse_args()
    train_model(args)
