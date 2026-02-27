import torch
import torch.nn as nn
from pytorch3d.ops import knn_points
from pytorch3d.loss import mesh_laplacian_smoothing
# define losses
def voxel_loss(voxel_src,voxel_tgt):
	# voxel_src: b x h x w x d
	# voxel_tgt: b x h x w x d
	# loss = 
	# implement some loss for binary voxel grids
	eps = 1e-5
	# breakpoint()
	voxel_src = torch.clamp(voxel_src, eps, 1 - eps)
	loss = -((voxel_tgt * torch.log(voxel_src)) + (1 - voxel_tgt) * torch.log(1 - voxel_src )).mean()
	# breakpoint()
	return loss
	
def chamfer_loss(point_cloud_src,point_cloud_tgt):
	# point_cloud_src, point_cloud_src: b x n_points x 3  
	# loss_chamfer = 
	# implement chamfer loss from scratch
	src = point_cloud_src.float()
	tgt = point_cloud_tgt.float()
	dst = knn_points(src, tgt, K = 1).dists.squeeze(-1) # (B, Ns)
	dts = knn_points(tgt, src, K = 1).dists.squeeze(-1) # (B, Nt)
	loss_chamfer = (dst.mean(dim=1) + dts.mean(dim=1)).mean()
	return loss_chamfer

def smoothness_loss(mesh_src):
	# loss_laplacian = 
	# implement laplacian smoothening loss

	loss_laplacian = mesh_laplacian_smoothing(mesh_src)

	return loss_laplacian