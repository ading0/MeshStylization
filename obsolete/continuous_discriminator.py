from pytorch3d.io import load_obj, save_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes, join_meshes_as_batch
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency

import collections
import datetime
import glob
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path
import plotting_utils
import random
import torch
import torch.nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from typing import Iterable


def get_parallel_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def load_normalized_meshes(obj_files: Iterable[str]) -> Iterable[Meshes]:
    device = get_parallel_device()
    meshes = []

    for obj_file in obj_files:
        
        verts, faces, _ = load_obj(obj_file, load_textures=False)
        faces_idx = faces.verts_idx.to(device)
        verts = verts.to(device)

        center = verts.mean(0)
        verts = verts - center

        scale = max(verts.abs().max(0)[0])
        verts = verts / scale * 0.999  # strictly within [-1, 1]

        mesh = Meshes(verts=[verts], faces=[faces_idx])
        meshes.append(mesh)
    
    return meshes

class ContinuousDiscriminator(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.device = get_parallel_device()
        self.weight_conv_dense1 = torch.nn.Linear(3, 32, device=device)
        self.weight_conv_dense2 = torch.nn.Linear(32, 96, device=device)

        self.final_dense = torch.nn.Linear(32, 1, device=device)

    def forward(self, samples: torch.Tensor):
        """
        'samples' has shape (n_samples, 3)
        points coordinates are local; are expected to be in the range [-1, 1] where the neighborhood has radius ~1 or so
        """
        local_positions = samples  # (n_samples, 3)
        features = samples  # (n_samples, c_in)
        n_samples = samples.shape[0]

        # Weight predictor
        w: torch.Tensor = self.weight_conv_dense1(local_positions)
        w = F.relu(w)
        w = self.weight_conv_dense2(w)
        w = w.reshape([n_samples, 3, 32])  # (n_samples, c_in, c_out)

        tiled_features = features.unsqueeze(2).expand(-1, -1, 32)  # (n_samples, c_in, c_out)
        
        prod = torch.mul(tiled_features, w)

        conv_out = prod.mean(dim=[0, 1])  # (c_out)

        x = F.relu(conv_out)
        x = self.final_dense(x)  # (1)
        x = F.sigmoid(x)  # (1), within [0, 1]
        return x
    
def run_discriminator(discr: ContinuousDiscriminator, update: bool, n_epochs: int,
                      lowpoly_meshes: Iterable[Meshes], realistic_meshes: Iterable[Meshes], 
                      local_region_radius: float, n_local_regions_per_mesh: int, n_samples_per_mesh: int) -> float:
    parallel_device = get_parallel_device()

    opt = torch.optim.Adam(discr.parameters(), lr=0.01)
    loss_function = torch.nn.BCELoss()

    last_epoch_loss_values = []
    
    for step in range(n_epochs):
        last_epoch_loss_values.clear()

        # Choose meshes of each type
        meshes = [m for m in lowpoly_meshes] + [m for m in realistic_meshes]
        mesh_targets = [0.0] * len(lowpoly_meshes) + [1.0] * len(realistic_meshes)

        batch = zip(meshes, mesh_targets)
        
        local_examples = []  # list of (local_samples, local_target)

        for mesh, mesh_target in batch:
            all_samples = sample_points_from_meshes(mesh, n_samples_per_mesh).to(parallel_device).squeeze()
            for _ in range(n_local_regions_per_mesh):
                sample_idx = np.random.randint(0, len(all_samples))
                chosen_sample = all_samples[sample_idx, :]
                nearby = (all_samples - chosen_sample).square().sum(dim=1) <= local_region_radius * local_region_radius

                nearby_samples = all_samples[nearby]
                local_center = nearby_samples.mean(0, keepdim=True)
                local_samples = (nearby_samples - local_center) / local_region_radius

                local_target = torch.Tensor([mesh_target]).to(parallel_device)
                local_examples.append((local_samples, local_target))


        # Randomize order of cloud inputs/targets
        for local_ex_idx in torch.randperm(len(local_examples)):
            local_samples, local_target = local_examples[local_ex_idx]
        
            local_output = discr(local_samples)
            loss = loss_function(local_output, local_target)
            last_epoch_loss_values.append(loss.item())
            # print(f"loss value: {loss_value.item()}")
            
            if update:
                opt.zero_grad()
                loss.backward()
                opt.step()
        
    return np.mean(last_epoch_loss_values)

if __name__ == "__main__":
    device = get_parallel_device()

    lowpoly_dir = os.path.join(".", "data", "lowpoly")
    realistic_dir = os.path.join(".", "data", "realistic")
    lowpoly_obj_files = list(Path(lowpoly_dir).glob("**/*.obj"))
    realistic_obj_files = list(Path(realistic_dir).glob("**/*.obj"))
    random.shuffle(lowpoly_obj_files)
    random.shuffle(realistic_obj_files)

    n_files = min(len(lowpoly_obj_files), len(realistic_obj_files))
    n_tr_files = int(0.8 * n_files)
    n_val_files = n_files - n_tr_files

    training_lowpoly_meshes = load_normalized_meshes(lowpoly_obj_files[:n_tr_files])
    training_realistic_meshes = load_normalized_meshes(realistic_obj_files[:n_tr_files])

    validation_lowpoly_meshes = load_normalized_meshes(lowpoly_obj_files[n_tr_files:n_tr_files+n_val_files])
    validation_realistic_meshes = load_normalized_meshes(realistic_obj_files[n_tr_files:n_tr_files+n_val_files])

    print(f"TRAINING: {len(training_lowpoly_meshes)} lowpoly and {len(training_realistic_meshes)} realistic meshes")
    print(f"VALIDATION: {len(validation_lowpoly_meshes)} lowpoly and {len(validation_realistic_meshes)} realistic meshes")

    discr = ContinuousDiscriminator()


    print("PARAMETERS:")
    for name, param in discr.named_parameters():
        print(f"\tname: {name}, shape: {param.shape}, requires_grad={param.requires_grad}")

    for epoch_number in range(51):
        training_lv = run_discriminator(discr, True, 1, training_lowpoly_meshes, training_realistic_meshes, 0.4, 100, 5000)
        print(f"epoch_number {epoch_number}, training loss: {training_lv}")
        if epoch_number % 5 == 0:
            validation_lv = run_discriminator(discr, False, 1, validation_lowpoly_meshes, validation_realistic_meshes, 0.4, 100, 5000)
            print(f"epoch_number {epoch_number}, validation loss: {validation_lv}")
    
    