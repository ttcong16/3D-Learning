from torchvision import models as torchvision_models
from torchvision import transforms
import time
import torch.nn as nn
import torch
from pytorch3d.utils import ico_sphere
import pytorch3d
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Sequence
import math

class VoxDecoder(nn.Module):
    def __init__(self, latent_dim=512, hidden_dim=256, resolution=32):
        super().__init__()
        self.resolution = resolution
        self.fc1_z = nn.Linear(latent_dim, hidden_dim, bias=False)  # Wz
        self.fc1_c = nn.Linear(3, hidden_dim)                       # Wc + b
        self.fc2   = nn.Linear(hidden_dim, hidden_dim)
        self.fc3   = nn.Linear(hidden_dim, 1)
        x = torch.linspace(-1, 1, resolution)
        y = torch.linspace(-1, 1, resolution)
        z = torch.linspace(-1, 1, resolution)
        grid = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)  # (R,R,R,3)
        self.register_buffer("coords", grid.reshape(-1, 3))  # (N,3)
    
    def forward(self, z):
        B = z.size(0)
        R = self.resolution
        N = R * R * R
        z_term = self.fc1_z(z)                  # (B,H)
        output = z.new_zeros(B, N)
        chunk_size = 16384  
        for start in range(0, N, chunk_size):
            end = min(start + chunk_size, N)
            c_term = self.fc1_c(self.coords[start:end])     # (n,H)
            x = F.relu(c_term.unsqueeze(0) + z_term.unsqueeze(1), inplace=True)  # (B,n,H)
            x = F.relu(self.fc2(x), inplace=True)
            x = torch.sigmoid(self.fc3(x)).squeeze(-1)      # (B,n)
            output[:, start:end] = x
        return output.view(B, R, R, R)

    


class PointCloudDecoder(nn.Module):
    def __init__(self, n_points, latent_dim=512, hidden_dim=512):
        super().__init__()
        self.n_points = n_points
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim * 2, n_points * 3),
            nn.Tanh()
        )

    def forward(self, z):                     # z: (B, latent_dim)
        coords = self.mlp(z)                  # (B, n_points*3)
        return coords.view(z.size(0), self.n_points, 3)  # (B, n_points, 3)
    
    
class MeshDecoder(nn.Module):
    def __init__(self, num_verts, latent_dim=512, hidden_dim1=1024, hidden_dim2=2048):
        super().__init__()
        self.num_verts  = num_verts
        self.fc1 = nn.Linear(latent_dim, hidden_dim1)   
        self.fc2 = nn.Linear(hidden_dim1, hidden_dim2)
        self.fc3 = nn.Linear(hidden_dim2, num_verts * 3)
        nn.init.zeros_(self.fc3.weight)
        nn.init.zeros_(self.fc3.bias)

    def forward(self, z):
        x  = F.relu(self.fc1(z), inplace=True)      # (B, hidden_dim1)
        x  = F.relu(self.fc2(x), inplace=True)      # (B, hidden_dim2)
        y = self.fc3(x).view(z.size(0), self.num_verts, 3)
        return y



class ImplicitDecoder(nn.Module):
    def __init__(self, latent_dim=512, hidden_dim=256):
        super().__init__()
        H = hidden_dim
        self.fc1_z = nn.Linear(latent_dim, H, bias=False)
        self.fc1_p = nn.Linear(3, H)      # p in (-1,1)^3
        self.fc2   = nn.Linear(H, H)
        self.fc3   = nn.Linear(H, 1)

    def forward(self, z, p):
        zt = self.fc1_z(z)                      # (B,H)
        pt = self.fc1_p(p)                      # (N,H)
        h  = F.relu(zt[:,None,:] + pt[None,:,:], inplace=True)  # (B,N,H)
        h  = F.relu(self.fc2(h), inplace=True)
        return torch.sigmoid(self.fc3(h)).squeeze(-1)           # (B,N)

def grid(device):
    R = 32
    x = torch.linspace(-1, 1, R, device=device)
    y = torch.linspace(-1, 1, R, device=device)
    z = torch.linspace(-1, 1, R, device=device)
    g = torch.stack(torch.meshgrid(x, y, z, indexing="ij"), dim=-1)  # (R,R,R,3)
    return g.reshape(-1,3), R
def infer_volume(model: ImplicitDecoder, z, chunk=8192):
    points, R = grid(z.device)             # (32768,3)
    B, N = z.size(0), points.size(0)
    outs = []
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        outs.append(model(z, points[s:e]))

    occ_flat = torch.cat(outs, dim=1)      # (B, 32768)
    return occ_flat.view(B, R, R, R)       # (B,32,32,32)


class Parametric(nn.Module):
    def __init__(self, K: int = 25, latent_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.K = K
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim

        # K MLPs 
        self.mlps = nn.ModuleList([
            nn.Sequential(
                nn.Linear(latent_dim + 2, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Linear(hidden_dim, 3)        
            ) for _ in range(K)
        ])

    @staticmethod
    @torch.no_grad()
    def make_uv(m_per_patch: int, mode: str = "grid", device=None) -> torch.Tensor:
        if mode == "random":
            return torch.rand(m_per_patch, 2, device=device) * 2 - 1
        elif mode == "grid": 
            s = int(math.ceil(m_per_patch ** 0.5))
            lin = torch.linspace(-1, 1, s, device=device)
            u, v = torch.meshgrid(lin, lin, indexing="ij")
            uv = torch.stack([u, v], dim=-1).reshape(-1, 2)
            return uv[:m_per_patch, :]

    def patch_forward(self, k: int, z: torch.Tensor, uv_k: torch.Tensor) -> torch.Tensor:
        if uv_k.dim() == 2:  
            uv_k = uv_k.unsqueeze(0)
        B, M, _ = uv_k.shape
       
        z_expand = z[:, None, :].expand(B, M, self.latent_dim)
        inp = torch.cat([z_expand, uv_k], dim=-1)
        return self.mlps[k](inp)  # (B,M,3)

    def forward(self, z: torch.Tensor, uv_patches: Sequence[torch.Tensor]) -> torch.Tensor:
        assert isinstance(uv_patches, (list, tuple)) and len(uv_patches) == self.K
        outs = [self.patch_forward(k, z, uv_patches[k]) for k in range(self.K)]
        return torch.cat(outs, dim=1)  

    

class SingleViewto3D(nn.Module):
    def __init__(self, args):
        super(SingleViewto3D, self).__init__()
        self.device = args.device
        if not args.load_feat:
            vision_model = torchvision_models.__dict__[args.arch](pretrained=True)
            self.encoder = torch.nn.Sequential(*(list(vision_model.children())[:-1]))
            self.normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])


        # define decoder
        if args.type == "vox":
            # Input: b x 512
            # Output: b x 32 x 32 x 32
            # self.decoder =    
            self.decoder = VoxDecoder(latent_dim=512, hidden_dim=256, resolution=32).to(self.device)         
        elif args.type == "point":
            # Input: b x 512
            # Output: b x args.n_points x 3  

            # self.decoder =  
            self.n_point = args.n_points
            self.decoder = PointCloudDecoder(self.n_point).to(self.device)

        elif args.type == "mesh":
            # Input: b x 512
            # Output: b x mesh_pred.verts_packed().shape[0] x 3  
            # try different mesh initializations 
            mesh_pred = ico_sphere(4, self.device)
            self.mesh_pred = pytorch3d.structures.Meshes(mesh_pred.verts_list()*args.batch_size, mesh_pred.faces_list()*args.batch_size)
            self.V = mesh_pred.verts_list()[0].shape[0]

            # self.decoder =       
            self.decoder = MeshDecoder(num_verts=self.V).to(self.device)      



        elif args.type == "implicit":
            self.decoder = ImplicitDecoder(latent_dim=512, hidden_dim=256).to(self.device)

        elif args.type == "parametric":
            patches = getattr(args, "patches", 25)
            hidden = getattr(args, "hidden", 256)
            self.n_point = getattr(args, "n_points", 2048)  
            self.decoder = Parametric(
                K=patches,
               latent_dim=512,
               hidden_dim=hidden,
            ).to(self.device)


    def forward(self, images, args):
        results = dict()

        total_loss = 0.0
        start_time = time.time()

        B = images.shape[0]

        if not args.load_feat:
            images_normalize = self.normalize(images.permute(0,3,1,2))
            encoded_feat = self.encoder(images_normalize).squeeze(-1).squeeze(-1) # b x 512
        else:
            encoded_feat = images 

        # call decoder
        if args.type == "vox":
            # voxels_pred =  
            voxels_pred = self.decoder(encoded_feat)           
            return voxels_pred

        elif args.type == "point":

            # pointclouds_pred =
            pointclouds_pred = self.decoder(encoded_feat)              
            return pointclouds_pred

        elif args.type == "mesh":
            # deform_vertices_pred =         
            deform_vertices_pred = self.decoder(encoded_feat)    
            mesh_pred = self.mesh_pred.offset_verts(deform_vertices_pred.reshape([-1,3]))
            return  mesh_pred          


        elif args.type == "implicit":
            occ = infer_volume(self.decoder, encoded_feat)          # (B,32,32,32), in (0,1)
            return occ

        elif args.type == "parametric":
            z = encoded_feat  # (B,512) latent từ encoder

            K = self.decoder.K
            B = z.size(0)
            M_total = getattr(args, "n_points", getattr(self, "n_point", 2048))
            uv_mode = getattr(args, "uv_mode", "random") 

      
            base = M_total // K
            rem  = M_total % K
            sizes = [base + 1] * rem + [base] * (K - rem)  

            uv_patches = []
            for m_k in sizes:
               
                uv_k = self.decoder.make_uv(m_k, mode=uv_mode, device=self.device)  # (m_k,2)
                uv_k = uv_k.unsqueeze(0).expand(B, -1, -1).contiguous()             # (B,m_k,2)
                uv_patches.append(uv_k)

            xyz = self.decoder(z, uv_patches)   
            return xyz