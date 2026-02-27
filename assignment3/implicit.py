import torch
import torch.nn.functional as F
from torch import autograd

from ray_utils import RayBundle
import torch.nn as nn


# Sphere SDF class
class SphereSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.radius = torch.nn.Parameter(
            torch.tensor(cfg.radius.val).float(), requires_grad=cfg.radius.opt
        )
        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)

        return torch.linalg.norm(
            points - self.center,
            dim=-1,
            keepdim=True
        ) - self.radius


# Box SDF class
class BoxSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.side_lengths = torch.nn.Parameter(
            torch.tensor(cfg.side_lengths.val).float().unsqueeze(0), requires_grad=cfg.side_lengths.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = torch.abs(points - self.center) - self.side_lengths / 2.0

        signed_distance = torch.linalg.norm(
            torch.maximum(diff, torch.zeros_like(diff)),
            dim=-1
        ) + torch.minimum(torch.max(diff, dim=-1)[0], torch.zeros_like(diff[..., 0]))

        return signed_distance.unsqueeze(-1)

# Torus SDF class
class TorusSDF(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.center = torch.nn.Parameter(
            torch.tensor(cfg.center.val).float().unsqueeze(0), requires_grad=cfg.center.opt
        )
        self.radii = torch.nn.Parameter(
            torch.tensor(cfg.radii.val).float().unsqueeze(0), requires_grad=cfg.radii.opt
        )

    def forward(self, points):
        points = points.view(-1, 3)
        diff = points - self.center
        q = torch.stack(
            [
                torch.linalg.norm(diff[..., :2], dim=-1) - self.radii[..., 0],
                diff[..., -1],
            ],
            dim=-1
        )
        return (torch.linalg.norm(q, dim=-1) - self.radii[..., 1]).unsqueeze(-1)

sdf_dict = {
    'sphere': SphereSDF,
    'box': BoxSDF,
    'torus': TorusSDF,
}


# Converts SDF into density/feature volume
class SDFVolume(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )

        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )

        self.alpha = torch.nn.Parameter(
            torch.tensor(cfg.alpha.val).float(), requires_grad=cfg.alpha.opt
        )
        self.beta = torch.nn.Parameter(
            torch.tensor(cfg.beta.val).float(), requires_grad=cfg.beta.opt
        )

    def _sdf_to_density(self, signed_distance):
        # Convert signed distance to density with alpha, beta parameters
        return torch.where(
            signed_distance > 0,
            0.5 * torch.exp(-signed_distance / self.beta),
            1 - 0.5 * torch.exp(signed_distance / self.beta),
        ) * self.alpha

    def forward(self, ray_bundle):
        sample_points = ray_bundle.sample_points.view(-1, 3)
        depth_values = ray_bundle.sample_lengths[..., 0]
        deltas = torch.cat(
            (
                depth_values[..., 1:] - depth_values[..., :-1],
                1e10 * torch.ones_like(depth_values[..., :1]),
            ),
            dim=-1,
        ).view(-1, 1)

        # Transform SDF to density
        signed_distance = self.sdf(ray_bundle.sample_points)
        density = self._sdf_to_density(signed_distance)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(sample_points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        out = {
            'density': -torch.log(1.0 - density) / deltas,
            'feature': base_color * self.feature * density.new_ones(sample_points.shape[0], 1)
        }

        return out


# Converts SDF into density/feature volume
class SDFSurface(torch.nn.Module):
    def __init__(
        self,
        cfg
    ):
        super().__init__()

        self.sdf = sdf_dict[cfg.sdf.type](
            cfg.sdf
        )
        self.rainbow = cfg.feature.rainbow if 'rainbow' in cfg.feature else False
        self.feature = torch.nn.Parameter(
            torch.ones_like(torch.tensor(cfg.feature.val).float().unsqueeze(0)), requires_grad=cfg.feature.opt
        )
    
    def get_distance(self, points):
        points = points.view(-1, 3)
        return self.sdf(points)

    def get_color(self, points):
        points = points.view(-1, 3)

        # Outputs
        if self.rainbow:
            base_color = torch.clamp(
                torch.abs(points - self.sdf.center),
                0.02,
                0.98
            )
        else:
            base_color = 1.0

        return base_color * self.feature * points.new_ones(points.shape[0], 1)
    
    def forward(self, points):
        return self.get_distance(points)

class HarmonicEmbedding(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        n_harmonic_functions: int = 6,
        omega0: float = 1.0,
        logspace: bool = True,
        include_input: bool = True,
    ) -> None:
        super().__init__()

        if logspace:
            frequencies = 2.0 ** torch.arange(
                n_harmonic_functions,
                dtype=torch.float32,
            )
        else:
            frequencies = torch.linspace(
                1.0,
                2.0 ** (n_harmonic_functions - 1),
                n_harmonic_functions,
                dtype=torch.float32,
            )

        self.register_buffer("_frequencies", omega0 * frequencies, persistent=False)
        self.include_input = include_input
        self.output_dim = n_harmonic_functions * 2 * in_channels

        if self.include_input:
            self.output_dim += in_channels

    def forward(self, x: torch.Tensor):
        embed = (x[..., None] * self._frequencies).view(*x.shape[:-1], -1)

        if self.include_input:
            return torch.cat((embed.sin(), embed.cos(), x), dim=-1)
        else:
            return torch.cat((embed.sin(), embed.cos()), dim=-1)


class LinearWithRepeat(torch.nn.Linear):
    def forward(self, input):
        n1 = input[0].shape[-1]
        output1 = F.linear(input[0], self.weight[:, :n1], self.bias)
        output2 = F.linear(input[1], self.weight[:, n1:], None)
        return output1 + output2.unsqueeze(-2)


class MLPWithInputSkips(torch.nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_dim: int,
        output_dim: int,
        skip_dim: int,
        hidden_dim: int,
        input_skips,
    ):
        super().__init__()

        layers = []

        for layeri in range(n_layers):
            if layeri == 0:
                dimin = input_dim
                dimout = hidden_dim
            elif layeri in input_skips:
                dimin = hidden_dim + skip_dim
                dimout = hidden_dim
            else:
                dimin = hidden_dim
                dimout = hidden_dim

            linear = torch.nn.Linear(dimin, dimout)
            layers.append(torch.nn.Sequential(linear, torch.nn.ReLU(True)))

        self.mlp = torch.nn.ModuleList(layers)
        self._input_skips = set(input_skips)

    def forward(self, x: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
        y = x

        for li, layer in enumerate(self.mlp):
            if li in self._input_skips:
                y = torch.cat((y, z), dim=-1)

            y = layer(y)

        return y


# TODO (Q3.1): Implement NeRF MLP
class NeuralRadianceField(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()

        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        self.harmonic_embedding_dir = HarmonicEmbedding(3, cfg.n_harmonic_functions_dir)

        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim
        embedding_dim_dir = self.harmonic_embedding_dir.output_dim


        self.n_neurons_xyz = cfg.n_hidden_neurons_xyz
        self.n_neurons_dir = cfg.n_hidden_neurons_dir
        self.n_xyz_layers = cfg.n_layers_xyz
        self.concat = cfg.append_xyz                 
        self.n_dir_layers = 1

        # MLP location
        self.xyz_layers = nn.ModuleList([nn.Linear(embedding_dim_xyz, self.n_neurons_xyz)])
        for i in range(self.n_xyz_layers - 1):
            if i in self.concat:
                self.xyz_layers.append(nn.Linear(embedding_dim_xyz + self.n_neurons_xyz, self.n_neurons_xyz))
            else:
                self.xyz_layers.append(nn.Linear(self.n_neurons_xyz, self.n_neurons_xyz))

        # MLP direction
        self.dir_layers = nn.ModuleList(
            [nn.Linear(embedding_dim_dir + self.n_neurons_xyz, self.n_neurons_dir)] +
            [nn.Linear(self.n_neurons_dir, self.n_neurons_dir) for _ in range(self.n_dir_layers - 1)]
        )

        # === Output heads ===
        self.density_layer = nn.Linear(self.n_neurons_xyz, 1)          # sigma(x)
        self.feat_layer = nn.Linear(self.n_neurons_xyz, self.n_neurons_xyz)  # feature bottleneck
        self.rgb_layer = nn.Linear(self.n_neurons_dir, 3)              # RGB color

    def forward(self, ray_bundle):
        # Encoding location
        xyz = ray_bundle.sample_points                                # (N_rays, N_samples, 3)
        xyz_embed = self.harmonic_embedding_xyz(xyz)                  # (N_rays, N_samples, embed_dim_xyz)
        add = xyz_embed.clone()

        
        for i in range(self.n_xyz_layers):
            xyz_embed = F.relu(self.xyz_layers[i](xyz_embed))
            if i in self.concat:
                xyz_embed = torch.cat([xyz_embed, add], dim=-1)

        # Density
        raw_density = self.density_layer(xyz_embed)
        density = F.softplus(raw_density)  

        # Feature bottleneck
        feat = F.relu(self.feat_layer(xyz_embed))

        # Encoding dir
        dirs = ray_bundle.directions
        dirs = dirs / (torch.norm(dirs, dim=-1, keepdim=True) + 1e-8)
        dirs = dirs.unsqueeze(1).expand(-1, feat.shape[1], -1)
        dir_embed = self.harmonic_embedding_dir(dirs)

        # MLP dir + feature + dir 
        h = torch.cat([feat, dir_embed], dim=-1)
        for layer in self.dir_layers:
            h = F.relu(layer(h))

        # RGB 
        feature = torch.sigmoid(self.rgb_layer(h))

        return {
            "density": density,
            "feature": feature
        }


class NeuralSurface(torch.nn.Module):
    def __init__(
        self,
        cfg,
    ):
        super().__init__()
        # TODO (Q6): Implement Neural Surface MLP to output per-point SDF
        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim

        # ===== MLP cho SDF (Signed Distance Function) =====
        self.n_hidden_dist = cfg.n_hidden_neurons_distance
        self.n_layers_dist = cfg.n_layers_distance
        self.concat = cfg.append_distance  # skip connections

        dist_layers = [nn.Linear(embedding_dim_xyz, self.n_hidden_dist)]
        for i in range(self.n_layers_dist - 1):
            if i in self.concat:
                dist_layers.append(nn.Linear(embedding_dim_xyz + self.n_hidden_dist, self.n_hidden_dist))
            else:
                dist_layers.append(nn.Linear(self.n_hidden_dist, self.n_hidden_dist))
        self.mlp = nn.ModuleList(dist_layers)

        self.sdf_layer = nn.Linear(self.n_hidden_dist, 1)
        self.linear_feat = nn.Linear(self.n_hidden_dist, self.n_hidden_dist)
        # TODO (Q7): Implement Neural Surface MLP to output per-point color
        self.harmonic_embedding_xyz = HarmonicEmbedding(3, cfg.n_harmonic_functions_xyz)
        embedding_dim_xyz = self.harmonic_embedding_xyz.output_dim

        #  MLP for SDF
        self.n_hidden_dist = cfg.n_hidden_neurons_distance
        self.n_layers_dist = cfg.n_layers_distance
        self.concat = cfg.append_distance  # skip connections

        dist_layers = [nn.Linear(embedding_dim_xyz, self.n_hidden_dist)]
        for i in range(self.n_layers_dist - 1):
            if i in self.concat:
                dist_layers.append(nn.Linear(embedding_dim_xyz + self.n_hidden_dist, self.n_hidden_dist))
            else:
                dist_layers.append(nn.Linear(self.n_hidden_dist, self.n_hidden_dist))
        self.mlp = nn.ModuleList(dist_layers)

        self.sdf_layer = nn.Linear(self.n_hidden_dist, 1)
        self.linear_feat = nn.Linear(self.n_hidden_dist, self.n_hidden_dist)

        #  MLP for RGB 
        self.n_hidden_color = cfg.n_hidden_neurons_color
        self.n_layers_color = cfg.n_layers_color
        self.concat_color = cfg.append_color  # skip connections cho color

        rgb_layers = [nn.Linear(self.n_hidden_dist + embedding_dim_xyz, self.n_hidden_color)]
        for i in range(self.n_layers_color - 1):
            if i in self.concat_color:
                rgb_layers.append(nn.Linear(embedding_dim_xyz + self.n_hidden_color, self.n_hidden_color))
            else:
                rgb_layers.append(nn.Linear(self.n_hidden_color, self.n_hidden_color))
        self.mlp_color = nn.ModuleList(rgb_layers)
        self.color_out = nn.Linear(self.n_hidden_color, 3)

    def get_distance(
        self,
        points
    ):
        '''
        TODO: Q6
        Output:
            distance: N X 1 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3)
        xyz_emb = self.harmonic_embedding_xyz(points)
        residual = xyz_emb.clone()

        x = xyz_emb
        for i in range(self.n_layers_dist):
            x = F.relu(self.mlp[i](x))
            if i in self.concat:
                x = torch.cat([x, residual], dim=-1)

        distance = self.sdf_layer(x)
        return distance
    
    def get_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Output:
            distance: N X 3 Tensor, where N is number of input points
        '''
        points = points.view(-1, 3)
        xyz_emb = self.harmonic_embedding_xyz(points)

        x = F.relu(self.linear_feat(xyz_emb))
        x = torch.cat([x, xyz_emb], dim=-1)
        for i in range(self.n_layers_color):
            x = F.relu(self.mlp_color[i](x))
            if i in self.concat_color:
                x = torch.cat([x, xyz_emb], dim=-1)
        color = torch.sigmoid(self.color_out(x))
        return color
    def get_distance_color(
        self,
        points
    ):
        '''
        TODO: Q7
        Output:
            distance, points: N X 1, N X 3 Tensors, where N is number of input points
        You may just implement this by independent calls to get_distance, get_color
            but, depending on your MLP implementation, it maybe more efficient to share some computation
        '''
        points = points.view(-1, 3)
        xyz_emb = self.harmonic_embedding_xyz(points)
        skip = xyz_emb.clone()

        # SDF 
        x = xyz_emb
        for i in range(self.n_layers_dist):
            x = F.relu(self.mlp[i](x))
            if i in self.concat:
                x = torch.cat([x, skip], dim=-1)
        distance = self.sdf_layer(x)
        # Color 
        x = F.relu(self.linear_feat(x))
        x = torch.cat([x, xyz_emb], dim=-1)
        for i in range(self.n_layers_color):
            x = F.relu(self.mlp_color[i](x))
            if i in self.concat_color:
                x = torch.cat([x, xyz_emb], dim=-1)
        color = torch.sigmoid(self.color_out(x))

        return distance, color
    def forward(self, points):
        return self.get_distance(points)

    def get_distance_and_gradient(
        self,
        points
    ):
        has_grad = torch.is_grad_enabled()
        points = points.view(-1, 3)

        # Calculate gradient with respect to points
        with torch.enable_grad():
            points = points.requires_grad_(True)
            distance = self.get_distance(points)
            gradient = autograd.grad(
                distance,
                points,
                torch.ones_like(distance, device=points.device),
                create_graph=has_grad,
                retain_graph=has_grad,
                only_inputs=True
            )[0]
        
        return distance, gradient


implicit_dict = {
    'sdf_volume': SDFVolume,
    'nerf': NeuralRadianceField,
    'sdf_surface': SDFSurface,
    'neural_surface': NeuralSurface,
}
