# visualization.py
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def draw_voxel(vox, *, thresh: float = 0.5, save_as: str = "voxel.png") -> None:
    os.makedirs(os.path.dirname(save_as) or ".", exist_ok=True)

    if isinstance(vox, np.ndarray):
        t = torch.from_numpy(vox)
    else:
        t = vox
    if t.dim() == 5:   # [B,C,R,R,R]
        t = t[0]
        if t.size(0) == 1:  # C=1
            t = t[0]
    elif t.dim() == 4:  # [B,R,R,R]
        t = t[0]
    if t.dim() != 3:
        raise ValueError(f"Voxel phải có dạng [R,R,R] (hoặc [B,*,R,R,R]), nhận {tuple(t.shape)}")

    vol = t.detach().cpu().float().numpy()
    idx = np.argwhere(vol > thresh)  # (K,3)

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    if idx.size > 0:
        ax.scatter(idx[:, 0], idx[:, 1], idx[:, 2], s=18, marker="s", alpha=0.9)
    else:
        ax.text(0.5, 0.5, 0.5, f"No voxels > {thresh}", transform=ax.transAxes,
                ha="center", va="center")
    ax.set_title("Voxel View")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    plt.close()


def draw_pointcloud(pts, *, save_as: str = "pointcloud.png", dot_size: int = 2) -> None:

    os.makedirs(os.path.dirname(save_as) or ".", exist_ok=True)

    if isinstance(pts, np.ndarray):
        P = torch.from_numpy(pts)
    else:
        P = pts
    if P.dim() == 3:   # [B,N,3]
        P = P[0]
    if P.dim() != 2 or P.size(-1) != 3:
        raise ValueError(f"Point cloud phải có dạng [N,3] (hoặc [B,N,3]), nhận {tuple(P.shape)}")

    arr = P.detach().cpu().float().numpy()

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(arr[:, 0], arr[:, 1], arr[:, 2], s=dot_size, alpha=0.95)
    ax.set_title("Point Cloud View")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    plt.close()


def draw_mesh(mesh_or_tuple, *, save_as: str = "mesh.png") -> None:

    os.makedirs(os.path.dirname(save_as) or ".", exist_ok=True)

    V_np = None; F_np = None

    try:
        from pytorch3d.structures import Meshes
        if isinstance(mesh_or_tuple, Meshes):
            V_np = mesh_or_tuple.verts_list()[0].detach().cpu().float().numpy()
            F_np = mesh_or_tuple.faces_list()[0].detach().cpu().long().numpy()
    except Exception:
        pass
    if V_np is None or F_np is None:
        assert isinstance(mesh_or_tuple, (tuple, list)) and len(mesh_or_tuple) == 2, \
            "mesh_or_tuple phải là (verts, faces) hoặc Meshes"
        V, F = mesh_or_tuple
        V = torch.as_tensor(V)
        F = torch.as_tensor(F)
        if V.dim() != 2 or V.size(-1) != 3:
            raise ValueError(f"Verts phải [Nv,3], nhận {tuple(V.shape)}")
        if F.dim() != 2 or F.size(-1) != 3:
            raise ValueError(f"Faces phải [Nf,3], nhận {tuple(F.shape)}")
        V_np = V.detach().cpu().float().numpy()
        F_np = F.detach().cpu().long().numpy()

    tris = V_np[F_np]  # [Nf,3,3]

    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    poly = Poly3DCollection(tris, alpha=0.75)
    poly.set_edgecolor("k")
    ax.add_collection3d(poly)
    ax.auto_scale_xyz(V_np[:, 0], V_np[:, 1], V_np[:, 2])
    ax.set_title("Mesh View")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_box_aspect((1, 1, 1))
    plt.tight_layout()
    plt.savefig(save_as, dpi=300)
    plt.close()
