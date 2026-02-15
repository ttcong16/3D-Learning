import sys
from typing import Tuple, Union

import torch
from pytorch3d.ops.mesh_face_areas_normals import mesh_face_areas_normals
from pytorch3d.ops.packed_to_padded import packed_to_padded
from pytorch3d.renderer.mesh.rasterizer import Fragments as MeshFragments


def _rand_barycentric_coords(
    N: int, S: int, dtype: torch.dtype, device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Sinh ra tọa độ barycentric ngẫu nhiên cho N mesh với S sample mỗi mesh.
    """
    w0 = torch.rand((N, S), dtype=dtype, device=device)
    w1 = torch.rand((N, S), dtype=dtype, device=device)
    mask = w0 + w1 > 1
    w0[mask] = 1 - w0[mask]
    w1[mask] = 1 - w1[mask]
    w2 = 1 - (w0 + w1)
    return w0, w1, w2


def sample_points_from_meshes(
    meshes,
    num_samples: int = 10000,
    return_normals: bool = False,
    return_textures: bool = False,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, torch.Tensor],
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor],
]:
    """
    Lấy mẫu ngẫu nhiên từ surface của mesh mà KHÔNG dùng trọng số diện tích,
    cũng không dùng randint, mà random thủ công bằng torch.rand.
    """
    if meshes.isempty():
        raise ValueError("Meshes are empty.")

    verts = meshes.verts_packed()
    if not torch.isfinite(verts).all():
        raise ValueError("Meshes contain nan or inf.")

    if return_textures and meshes.textures is None:
        raise ValueError("Meshes do not contain textures.")

    faces = meshes.faces_packed()
    mesh_to_face = meshes.mesh_to_faces_packed_first_idx()
    num_meshes = len(meshes)
    num_valid_meshes = torch.sum(meshes.valid)

    # Khởi tạo tensor kết quả
    samples = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)

    # Lấy mẫu ngẫu nhiên trên từng mesh
    sample_face_idxs = []
    for i, f in enumerate(meshes.num_faces_per_mesh()):
        if f == 0:
            sample_face_idxs.append(
                torch.zeros((num_samples,), dtype=torch.long, device=meshes.device)
            )
        else:
            # random thủ công thay cho randint
            r = torch.rand((num_samples,), device=meshes.device)  # [0,1)
            face_idx = torch.floor(r * f).long()                  # [0, f-1]
            face_idx = face_idx + mesh_to_face[i]                 # offset packed
            sample_face_idxs.append(face_idx)

    sample_face_idxs = torch.stack(sample_face_idxs, dim=0)  # (N, num_samples)

    # Lấy các đỉnh tương ứng
    face_verts = verts[faces]
    v0, v1, v2 = face_verts[:, 0], face_verts[:, 1], face_verts[:, 2]

    # Sinh barycentric ngẫu nhiên
    w0, w1, w2 = _rand_barycentric_coords(
        num_valid_meshes, num_samples, verts.dtype, verts.device
    )

    # Tính tọa độ sample theo barycentric
    a = v0[sample_face_idxs]
    b = v1[sample_face_idxs]
    c = v2[sample_face_idxs]
    samples[meshes.valid] = w0[:, :, None] * a + w1[:, :, None] * b + w2[:, :, None] * c

    if return_normals:
        normals = torch.zeros((num_meshes, num_samples, 3), device=meshes.device)
        vert_normals = (v1 - v0).cross(v2 - v1, dim=1)
        vert_normals = vert_normals / vert_normals.norm(dim=1, p=2, keepdim=True).clamp(
            min=sys.float_info.epsilon
        )
        vert_normals = vert_normals[sample_face_idxs]
        normals[meshes.valid] = vert_normals

    if return_textures:
        pix_to_face = sample_face_idxs.view(len(meshes), num_samples, 1, 1)  # NxSx1x1
        bary = torch.stack((w0, w1, w2), dim=2).unsqueeze(2).unsqueeze(2)    # NxSx1x1x3
        dummy = torch.zeros(
            (len(meshes), num_samples, 1, 1), device=meshes.device, dtype=torch.float32
        )
        fragments = MeshFragments(
            pix_to_face=pix_to_face, zbuf=dummy, bary_coords=bary, dists=dummy
        )
        textures = meshes.sample_textures(fragments)  # NxSx1x1xC
        textures = textures[:, :, 0, 0, :]            # NxSxC

    # Return tuỳ chọn
    if return_normals and return_textures:
        return samples, normals, textures
    if return_normals:
        return samples, normals
    if return_textures:
        return samples, textures
    return samples
