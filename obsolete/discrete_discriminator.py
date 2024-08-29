from pytorch3d.io import load_obj, save_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency

import collections
import cProfile
import datetime
import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import random
import time
import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torchvision.models import vgg19, VGG19_Weights
from typing import Iterable

import plotting_utils


def get_parallel_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def load_meshes(obj_files: Iterable[str]) -> Iterable[Meshes]:
    meshes = []
    for obj_file in obj_files:
        verts, faces, _ = load_obj(obj_file, load_textures=False)
        meshes.append(Meshes(verts=[verts], faces=[faces.verts_idx]))
    
    return meshes

class DiscreteDiscriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        parallel_device = get_parallel_device()

        # input (N, 16^3)
        n_filters_1 = 8
        n_filters_2 = 16
        n_filters_3 = 32

        self.conv_1_1 = torch.nn.Conv3d(1, n_filters_1, 3, padding=1, device=parallel_device)
        self.conv_1_2 = torch.nn.Conv3d(n_filters_1, n_filters_1, 3, padding=1, device=parallel_device)

        self.conv_2_1 = torch.nn.Conv3d(n_filters_1, n_filters_2, 3, padding=1, device=parallel_device)
        self.conv_2_2 = torch.nn.Conv3d(n_filters_2, n_filters_2, 3, padding=1, device=parallel_device)

        self.conv_3_1 = torch.nn.Conv3d(n_filters_2, n_filters_3, 3, padding=1, device=parallel_device)
        self.conv_3_2 = torch.nn.Conv3d(n_filters_3, n_filters_3, 3, padding=1, device=parallel_device)

        self.dense_1 = torch.nn.Linear(n_filters_3 * 2 * 2 * 2, 64, device=parallel_device)
        self.dense_2 = torch.nn.Linear(64, 32, device=parallel_device)
        self.dense_3 = torch.nn.Linear(32, 1, device=parallel_device)

    def forward(self, samples_grid_batch: torch.Tensor):
        """
        samples_grid has shape (N, 1, 16, 16, 16)
        cell entries are all nonnegative, representing the approx. number of samples at that cell
        """
        x = samples_grid_batch
        x = F.relu(self.conv_1_1(x))  # (N, C_1, 16, 16, 16)
        x = F.relu(self.conv_1_2(x))
        x = F.max_pool3d(x, kernel_size=2)  # (N, C_1, 8, 8, 8)
        x = F.relu(self.conv_2_1(x))
        x = F.relu(self.conv_2_2(x))
        x = F.max_pool3d(x, kernel_size=2)  # (N, C_2, 4, 4, 4)
        x = F.relu(self.conv_3_1(x))
        x = F.relu(self.conv_3_2(x))
        x = F.max_pool3d(x, 2)  # (N, C_3, 2, 2, 2)
        x = x.flatten(start_dim=1)  # (N, C_3 * 2 * 2 * 2)
        x = F.relu(self.dense_1(x))
        x = F.relu(self.dense_2(x))
        x = F.sigmoid(self.dense_3(x)) # (N, 1)
        return x

def create_grid_from_samples(local_samples: torch.Tensor, size: int) -> torch.Tensor:
    """
    expects input of shape (S, 3), in local coordinates (-1 to 1)
    output (1, size, size, size)
    uses GPU, decently optimized
    """
    parallel_device= get_parallel_device()
    local_samples = local_samples.to(parallel_device)
    grid = torch.zeros([size, size, size], device=parallel_device)

    idx_space_samples = (local_samples + 1.) / 2. * float(size - 1)
    idx_space_samples = idx_space_samples
    
    sample_idxs = idx_space_samples.floor().int().clamp(0, size - 2)
    sample_idx_fracts = (idx_space_samples - sample_idxs.float()).clamp(0.0, 1.0)

    tot_weight = 0.0
    
    # workspace for index_put_
    grid_copy = torch.zeros(grid.shape, device=parallel_device)
    
    # loop over the 8 corners of a cube
    for i, j, k in itertools.product(range(2), range(2), range(2)):
        weights_i = (1.0 - float(i) - sample_idx_fracts[:, 0]).abs()
        weights_j = (1.0 - float(j) - sample_idx_fracts[:, 1]).abs()
        weights_k = (1.0 - float(k) - sample_idx_fracts[:, 2]).abs()
        weights = weights_i * weights_j * weights_k  # (n_samples)
        offsets = torch.tensor([[i, j, k]], device=parallel_device).int()  # (1, 3)
        idxs = sample_idxs + offsets  # (n_samples, 3)

        tot_weight += weights.sum()

        idxs_tuple = (idxs[:, 0], idxs[:, 1], idxs[:, 2])
        grid_copy.zero_()
        grid_copy.index_put_(idxs_tuple, weights, accumulate=True)  # accumulate=True needed due to dupe indices
        grid += grid_copy

    return grid.unsqueeze(0)

def normalize_sample_coordinates(samples: torch.Tensor, center: torch.Tensor, box_sidelength: float) -> torch.Tensor:
    # samples is (S, 3)
    # normalize to [-1, 1]
    centered = samples - center
    scaled = centered / (box_sidelength / 2.)
    clamped = scaled.clamp(-1.0, 1.0)
    return clamped


def create_local_grids_from_mesh(mesh: Meshes, grid_span: float, grid_size: int, n_grids: int, n_samples: int) -> Iterable[torch.Tensor]:
    """
    mesh: Mesh
    grid_span: fraction of the sidelength (relative to a minimum enclosing cube)
    """
    all_samples = sample_points_from_meshes(mesh, n_samples).squeeze().to(get_parallel_device())  # (n_samples, 3)
    
    # Normalize to within [0, 1]
    min_sample = all_samples.min(dim=0)[0]
    max_sample = all_samples.max(dim=0)[0]
    mid = 0.5 * (min_sample + max_sample)
    sidelength = (max_sample - min_sample).max()
    grid_extents = sidelength / 2. * 1.001
    grid_min = mid - grid_extents
    grid_max = mid + grid_extents
    all_samples = (all_samples - grid_min) / (grid_max - grid_min)  # within [0, 1]

    # Brute force, but far faster than an "intelligent" spatial hashing algorithm on the CPU for 5000-1000 samples
    center_sample_idxs = np.random.choice(all_samples.shape[0], n_grids, False)
    grids = []
    for i in center_sample_idxs:
        center_sample = all_samples[i]
        rel_pos = all_samples - center_sample
        is_sample_nearby = rel_pos.abs().max(dim=1)[0] < 0.5 * grid_span
        local_samples = normalize_sample_coordinates(all_samples[is_sample_nearby, :], center_sample, grid_span)

        grid = create_grid_from_samples(local_samples, grid_size)
        grids.append(grid)

    return grids

def compute_loss(discr: DiscreteDiscriminator, 
                 meshes: Iterable[Meshes], 
                 targets: Iterable[torch.tensor], update: bool) -> float:
    """
    meshes: Iterable[Meshes]
    targets: Iterable[torch.tensor of single element of 0 or 1]
    update: True if training, False if just evaluating

    returns avg loss per batch (not that precise)
    """
    assert(len(meshes) == len(targets))
    opt = torch.optim.Adam(discr.parameters(), lr=0.01)
    
    idx_perm = torch.randperm(len(meshes))
    batch_size = 64
    
    parallel_device = get_parallel_device()
    
    total_loss_numer = 0.0
    total_loss_denom = 0.0

    for i in range(0, len(meshes), batch_size):
        j = min(i + batch_size, len(meshes))
        batch_grids_list = []
        batch_targets_list = []
        for idx in idx_perm[i:j]:
            mesh_local_grids = create_local_grids_from_mesh(meshes[idx], 0.1, 16, 32, 5000)
            batch_grids_list.extend(mesh_local_grids)
            batch_targets_list.extend((targets[idx] for _ in range(len(mesh_local_grids))))
        
        batch_grids = torch.stack(batch_grids_list)
        batch_targets = torch.stack(batch_targets_list)
        batch_predictions = discr(batch_grids)
        batch_weight = torch.ones(batch_predictions.shape, device=parallel_device)
        loss_function = torch.nn.BCELoss(batch_weight)
        batch_loss = loss_function(batch_predictions, batch_targets.to(parallel_device))
        
        total_loss_numer += batch_loss.item() * (j - i)
        total_loss_denom += j - i

        if update:
            opt.zero_grad()
            batch_loss.backward()
            opt.step()
    
    return total_loss_numer / total_loss_denom

def train():
    device = get_parallel_device()

    lowpoly_dir = os.path.join(".", "data", "lowpoly")
    realistic_dir = os.path.join(".", "data", "realistic")
    lowpoly_obj_files = list(Path(lowpoly_dir).glob("**/*.obj"))
    realistic_obj_files = list(Path(realistic_dir).glob("**/*.obj"))

    lowpoly_meshes = load_meshes(lowpoly_obj_files)[0:32]
    realistic_meshes = load_meshes(realistic_obj_files)[0:32]

    discr = DiscreteDiscriminator()

    print("PARAMETERS:")
    for name, param in discr.named_parameters():
        print(f"\tname: {name}, shape: {param.shape}, requires_grad={param.requires_grad}")

    max_sidelength = 0.4
    n_samples = 5000  # number of samples to take per mesh, when sampling
    n_steps = 1  # number of batches learned on
    n_grids_per_mesh = 32
    grid_span = 0.2
    grid_size = 8

    # create grids
    meshes = lowpoly_meshes + realistic_meshes
    n_lp = len(lowpoly_meshes)
    n_r = len(realistic_meshes)
    mesh_targets = [torch.tensor([0.]) for _ in range(n_lp)] + [torch.tensor([1.]) for _ in range(n_r)]

    grids_and_targets = []

    for i in range(len(meshes)):

        mesh = meshes[i]
        target = mesh_targets[i]
        this_mesh_grids = create_local_grids_from_mesh(mesh, grid_span, grid_size, n_grids_per_mesh, n_samples)

        this_mesh_data = [(g, target.detach().clone()) for g in this_mesh_grids]
        grids_and_targets.extend(this_mesh_data)
    
    print(f"finished loading meshes ({len(grids_and_targets)})")

    for step in range(10):
        loss_value = compute_loss(discr, meshes, mesh_targets, True)
        print(f"loss value: {loss_value}")

if __name__ == "__main__":
    train()
    # cProfile.run("train()", sort='tottime')