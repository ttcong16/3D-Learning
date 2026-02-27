import argparse
import os
import math
import numpy as np
from typing import Tuple
import csv

import torch
from torch.utils.data import DataLoader

from pytorch3d.datasets.r2n2.utils import collate_batched_R2N2
from model import SingleViewto3D
import dataset_location
from r2n2_custom import R2N2


def extract_model_id(feed: dict, b: int):
    for k in ["model_id", "model", "modelname"]:
        if k in feed:
            try:
                return str(feed[k][b])
            except Exception:
                pass

    for k in ["image_path", "img_path", "cad_path", "path"]:
        if k in feed:
            val = feed[k][b]
            if isinstance(val, (list, tuple)) and len(val) > 0:
                val = val[0]
            import os
            return os.path.splitext(os.path.basename(str(val)))[0]
    return None

def extract_tag_from_feed(feed: dict, b: int):
    
    synset = None
    view   = None
    for k in ["synset_id", "synset"]:
        if k in feed:
            try: synset = str(feed[k][b])
            except: pass
    for k in ["view_id", "view", "camera_idx", "az_el_t"]:
        if k in feed:
            try: view = str(feed[k][b])
            except: pass
    model = extract_model_id(feed, b)
    parts = [p for p in [synset, model, view] if p]
    return ("_".join(parts) if parts else None), model


def save_obj_raw(verts: np.ndarray, faces: np.ndarray, path: str):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {float(v[0])} {float(v[1])} {float(v[2])}\n")
        for tri in faces:
            f.write(f"f {int(tri[0])+1} {int(tri[1])+1} {int(tri[2])+1}\n")


try:
    import mcubes
    HAS_MCUBES = True
except Exception:
    HAS_MCUBES = False

try:
    from pytorch3d.io import save_obj
    HAS_P3D = True
except Exception:
    HAS_P3D = False

try:
    import open3d as o3d
    HAS_O3D = True
except Exception:
    HAS_O3D = False



def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def save_pointcloud_ply(xyz: np.ndarray, out_path: str):
    N = xyz.shape[0]
    with open(out_path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {N}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("end_header\n")
        for p in xyz:
            f.write(f"{p[0]} {p[1]} {p[2]}\n")

def save_pointcloud_xyz(xyz: np.ndarray, out_path: str):
    np.savetxt(out_path, xyz, fmt="%.6f")


def volume_to_mesh_obj(volume: np.ndarray, out_obj: str, isovalue: float = 0.5):
    if not HAS_MCUBES:
        raise RuntimeError("Thiếu PyMCubes: pip install PyMCubes")

    vol = np.asarray(volume, dtype=np.float32)
    vol = np.nan_to_num(vol, nan=0.0, posinf=1.0, neginf=0.0)
    vol = np.clip(vol, 0.0, 1.0)

    vmin, vmax = float(vol.min()), float(vol.max())

    if not (vmin < isovalue < vmax):
        if vmin == vmax:
            raise RuntimeError(f"Volume constant={vmin:.3f} → không trích được bề mặt.")
        isovalue = 0.5 * (vmin + vmax)

    verts, faces = mcubes.marching_cubes(vol, isovalue)
    if verts.size == 0 or faces.size == 0:
        raise RuntimeError(
            f"MC ra lưới rỗng (min={vmin:.3f}, max={vmax:.3f}, thr={isovalue:.3f})."
        )

    R = vol.shape[0]
    verts = (verts / (R - 1)) * 2.0 - 1.0
    mcubes.export_obj(verts, faces, out_obj)

def pointcloud_to_mesh_o3d(xyz: np.ndarray,
                           method: str = "bpa",
                           radii: list = None,
                           poisson_depth: int = 9) -> Tuple[np.ndarray, np.ndarray]:
    if not HAS_O3D:
        raise RuntimeError("Thiếu open3d: pip install open3d")
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(xyz))
    pcd.estimate_normals()
    pcd.orient_normals_consistent_tangent_plane(10)

    if method == "bpa":
        if radii is None:
            bbox = pcd.get_axis_aligned_bounding_box()
            extent = np.max(bbox.get_extent())
            base = extent * 0.02
            radii = [base, base * 2, base * 4]
        mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
            pcd, o3d.utility.DoubleVector(radii)
        )
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
    elif method == "poisson":
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=poisson_depth
        )
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)
        mesh.remove_degenerate_triangles()
        mesh.remove_duplicated_triangles()
        mesh.remove_duplicated_vertices()
        mesh.remove_non_manifold_edges()
    else:
        raise ValueError("Error")

    V = np.asarray(mesh.vertices, dtype=np.float32)
    F = np.asarray(mesh.triangles, dtype=np.int64)
    return V, F


# ===================== FEED PREPROCESS =====================
def preprocess(feed_dict, args):
   
    images = feed_dict["images"].squeeze(1)  
    if args.load_feat:
        feats = torch.stack(feed_dict["feats"])
        return feats.to(args.device)
    else:
        return images.to(args.device)



def load_checkpoint_maybe(model: torch.nn.Module, ckpt_path: str, device: torch.device):
    if not ckpt_path:
        print("[INFO] Not provide --checkpoint, use weight init.")
        return
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"[WARN] Missing keys: {len(missing)} ")
    if unexpected:
        print(f"[WARN] Unexpected keys: {len(unexpected)}")
    print("[OK] Đã nạp checkpoint.")


# ===================== Main inference loop =====================
@torch.no_grad()
def run(args):
    device = torch.device(args.device)
    dataset = R2N2(
        "test",
        dataset_location.SHAPENET_PATH,
        dataset_location.R2N2_PATH,
        dataset_location.SPLITS_PATH,
        return_voxels=True,
        return_feats=args.load_feat,
    )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        collate_fn=collate_batched_R2N2,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )

    class ModelArgs:
        pass
    margs = ModelArgs()
    margs.arch         = args.arch
    margs.device       = device
    margs.load_feat    = args.load_feat
    margs.type         = args.type
    margs.batch_size   = args.batch_size
    margs.n_points     = args.n_points
    margs.w_chamfer    = 1.0
    margs.w_smooth     = 0.0
    # Parametric
    margs.patches      = args.patches
    margs.hidden       = args.hidden
    margs.uv_mode      = args.uv_mode

    model = SingleViewto3D(margs).to(device).eval()
    load_checkpoint_maybe(model, args.checkpoint, device)

    ensure_dir(args.outdir)

    audit_csv = os.path.join(args.outdir, "order_audit.csv")
    with open(audit_csv, "w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["global_idx", "tag_used_in_filename", "model_id", "saved_filename"])

    global_idx = 0

    for batch_idx, feed in enumerate(loader):

        images_or_feats = preprocess(feed, margs)  # (B,H,W,3) hoặc (B,512)


        out = model(images_or_feats, margs)

      
        B = images_or_feats.size(0)

        if args.type in ["vox", "implicit"]:
    
    
            occ = out.detach().cpu().numpy()
            for b in range(B):
                tag_from_feed, model_id = extract_tag_from_feed(feed, b)
                tag = tag_from_feed if tag_from_feed is not None else f"test_{global_idx:06d}"
                vol_path = os.path.join(args.outdir, f"{tag}_{args.type}.npy")
                np.save(vol_path, occ[b])

                if HAS_MCUBES:
                    obj_path = os.path.join(args.outdir, f"{tag}_{args.type}_thr{args.thr:.2f}.obj")
                    try:
                        volume_to_mesh_obj(occ[b], obj_path, isovalue=args.thr)
                        # ghi audit cho file OBJ 
                        with open(audit_csv, "a", newline="") as fcsv:
                            csv.writer(fcsv).writerow([global_idx, tag, (model_id or ""), os.path.basename(obj_path)])
                    except Exception as e:
                        print(f"[WARN] Marching Cubes sample error {tag}: {e}")
                else:
                    print("[WARN] Only save .npy")

        elif args.type == "mesh":
            # out: pytorch3d.structures.Meshes
       
            meshes = out
            for b in range(B):
                tag_from_feed, model_id = extract_tag_from_feed(feed, b)
                tag = tag_from_feed if tag_from_feed is not None else f"test_{global_idx:06d}"

                V_t = meshes.verts_list()[b].detach().cpu()
                F_t = meshes.faces_list()[b].detach().cpu()
                obj_path = os.path.join(args.outdir, f"{tag}_mesh.obj")

                try:
                    if HAS_P3D:
                        save_obj(obj_path, V_t, F_t)
                    else:
                        V = V_t.numpy().astype(np.float32)
                        F = F_t.numpy().astype(np.int64)
                        save_obj_raw(V, F, obj_path)
                    with open(audit_csv, "a", newline="") as fcsv:
                        csv.writer(fcsv).writerow([global_idx, tag, (model_id or ""), os.path.basename(obj_path)])
                except Exception as e:
                    print(f"[WARN] Fail to save mesh {tag}: {e}")

                global_idx += 1



        elif args.type in ["point", "parametric"]:
            # # out: (B,N,3) hoặc tuple(...,(B,N,3))
            
            if isinstance(out, (list, tuple)):
                pts_batch = out[0].detach().cpu().numpy()
            else:
                pts_batch = out.detach().cpu().numpy()

            for b in range(B):
                tag_from_feed, model_id = extract_tag_from_feed(feed, b)
                tag = tag_from_feed if tag_from_feed is not None else f"test_{global_idx:06d}"
                pts = pts_batch[b].astype(np.float32)

                # RAW: .ply + .xyz
                ply_path = os.path.join(args.outdir, f"{tag}_{args.type}.ply")
                xyz_path = os.path.join(args.outdir, f"{tag}_{args.type}.xyz")
                save_pointcloud_ply(pts, ply_path)
                save_pointcloud_xyz(pts, xyz_path)
                with open(audit_csv, "a", newline="") as fcsv:
                    csv.writer(fcsv).writerow([global_idx, tag, (model_id or ""), os.path.basename(ply_path)])
                    csv.writer(fcsv).writerow([global_idx, tag, (model_id or ""), os.path.basename(xyz_path)])

                if HAS_O3D:
                    try:
                        V, F = pointcloud_to_mesh_o3d(pts, method=args.pc2mesh, poisson_depth=args.poisson_depth)
                        obj_path = os.path.join(args.outdir, f"{tag}_{args.type}_mesh.obj")
                        if HAS_P3D:
                            save_obj(obj_path, torch.from_numpy(V), torch.from_numpy(F))
                        else:
                            save_obj_raw(V, F, obj_path)
                        with open(audit_csv, "a", newline="") as fcsv:
                            csv.writer(fcsv).writerow([global_idx, tag, (model_id or ""), os.path.basename(obj_path)])
                    except Exception as e:
                        print(f"[WARN] point->mesh sample error {tag}: {e}")
                else:
                    print("[WARN] only save point cloud.")

                global_idx += 1


        else:
            raise ValueError(f"Unknown type {args.type}")

        if (batch_idx + 1) % args.log_freq == 0:
            print(f"[{batch_idx+1}/{len(loader)}] exported up to index {global_idx-1}")

    print("Done inference on test split.")


def main():
    parser = argparse.ArgumentParser("R2N2 Test Inference")
    parser.add_argument("--type", required=True, choices=["vox", "point", "mesh", "implicit", "parametric"])
    parser.add_argument("--checkpoint", default=f"checkpoint_vox.pth", help="path to checkpoint .pth")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--arch", default="resnet18")

    # Loader
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--load_feat", action="store_true")

    # Output
    parser.add_argument("--outdir", default="inference_out")
    parser.add_argument("--log_freq", type=int, default=1)

    # Vox/Implicit
    parser.add_argument("--thr", type=float, default=0.5, help="Isovalue cho Marching Cubes")

    # Point/Parametric
    parser.add_argument("--n_points", type=int, default=1000)  
    parser.add_argument("--pc2mesh", default="bpa", choices=["bpa", "poisson"])
    parser.add_argument("--poisson_depth", type=int, default=9)
    parser.add_argument("--patches", type=int, default=25)
    parser.add_argument("--hidden", type=int, default=256)
    parser.add_argument("--uv_mode", default="random", choices=["random", "grid"])

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
