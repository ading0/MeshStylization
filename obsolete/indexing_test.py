from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency

import plotting_utils
import torch
import matplotlib.pyplot as plt
import datetime
import itertools

def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    device = get_device()
    grid = torch.zeros([4, 4])

    i_tensor = torch.tensor([0, 0, 0]).int()
    j_tensor = torch.tensor([0, 0, 1]).int()
    val_tensor = torch.tensor([1.0, 2.0, 3.0]).float()

    grid.index_add_((i_tensor, j_tensor), val_tensor, accumulate=True)
    print(grid)

