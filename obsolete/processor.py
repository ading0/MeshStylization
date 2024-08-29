from pytorch3d.io import load_obj, save_obj, load_objs_as_meshes
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency

import datetime
import glob
import matplotlib.pyplot as plt
import os
from pathlib import Path
import torch
from torch.nn.functional import grid_sample
from typing import Iterable

import plotting_utils

def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

def normalize_path(path: str):
    return

def extract_mesh_from_files(obj_file: str) -> Meshes:
    verts_list = []
    faces_idx_list = []
    device = get_device()

    verts, faces, aux = load_obj(obj_file)
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)

    center = verts.mean(0)
    verts = verts - center

    scale = max(verts.abs().max(0)[0])
    verts = verts / scale * 0.999  # strictly within [-1, 1]

    verts_list.append(verts)
    faces_idx_list.append(faces_idx)
    
    return Meshes(verts=verts_list, faces=faces_idx_list)

if __name__ == "__main__":
    device = get_device()

    raw_dirs = [os.path.join(".", "data", "lowpoly"), os.path.join(".", "data", "realistic")]
    obj_paths = []

    # Find all obj file paths
    for dir in raw_dirs:
        path = Path(dir)
        obj_paths.extend(path.glob("**/*.obj"))

    for obj_path in obj_paths:
        mesh = load_objs_as_meshes([obj_path], device=device, load_textures=False)
        n_vertices = mesh.verts_packed().shape[0]
        if n_vertices >= 50000 or n_vertices < 10:
            Path.unlink(obj_path)
            print(f"deleting: {obj_path}")
    
    obj_paths = []
    for dir in raw_dirs:
        path = Path(dir)
        obj_paths.extend(path.glob("**/*.obj"))

    meshes = load_objs_as_meshes(obj_paths, device=device, load_textures=False)

    print(meshes.verts_packed().shape)


        