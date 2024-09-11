import math
import matplotlib.pyplot as plt
import os
import sys
import torch

import pytorch3d
# Data structures and functions for rendering
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights, 
    Materials, 
    RasterizationSettings, 
    MeshRenderer, 
    MeshRasterizer,  
    SoftPhongShader,
    TexturesUV,
    TexturesVertex
)
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib


import io_utils
import plotting_utils
import utils

def sample_points_on_unit_sphere(n_samples: int) -> torch.Tensor:
    """
    output shape: (n_samples, 3)
    """
    raw_points = torch.randn((n_samples, 3))
    raw_points[raw_points == 0] = 0.001
    raw_points_scale = raw_points.square().sum(1, keepdim=True).sqrt()
    points = raw_points / raw_points_scale
    return points

def get_random_cameras(mesh: Meshes, n_samples: int, camera_distance: float, device: torch.device) -> FoVPerspectiveCameras:
    points, normals = sample_points_from_meshes(mesh, n_samples, return_normals=True)
    points = points.squeeze()
    normals = normals.squeeze()
    rot_list = []
    tra_list = []
    for i in range(n_samples):
        eye = (points[i] + camera_distance * normals[i]).unsqueeze(0)  # (1, 3)
        at = points[i].unsqueeze(0)
        up = torch.tensor([0, 1, 0], device=device).float().unsqueeze(0)
        
        rot, tra = look_at_view_transform(eye=eye, at=at, up=up)

        rot_list.append(rot)
        tra_list.append(tra)

    rots = torch.concat(rot_list, dim=0)
    tras = torch.concat(tra_list, dim=0)

    # rots, tras = look_at_view_transform(camera_distance, elev=elevations, azim=azimuths, degrees=False)  # use radians
    cameras = FoVPerspectiveCameras(device=device, R=rots, T=tras)
    return cameras

def render_views(normalized_mesh: Meshes, cameras: FoVPerspectiveCameras, image_size: int, device: torch.device) -> torch.Tensor:
    """
    return shape: (n_samples, image_size, image_size, 4)
    """
    n_views = cameras.get_camera_center().shape[0]
    meshes = normalized_mesh.extend(n_views)

    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)

    light_ambient_colors = [(0.5, 0, 0), (0, 0.5, 0), (0, 0, 0.5)]
    light_diffuse_colors = [(0.3, 0, 0), (0, 0.3, 0), (0, 0, 0.3)]
    light_specular_colors = [(0.2, 0, 0), (0, 0.2, 0), (0, 0, 0.2)]
    light_directions = [(1, 0, 0), (0, 1, 0), (0, 0, 1)]
    lights = DirectionalLights(direction=[(1, 0, 0)],
                               device=device)

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader =  SoftPhongShader(device=device, cameras=cameras, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    images_rgba = renderer(meshes)  # n_samples, C, D, 4
    images_rgb_raw = images_rgba[:, :, :, 0:3]
    images_alpha = images_rgba[:, :, :, 3:4]

    images_rgb = images_rgb_raw * images_alpha + (1. - images_alpha)  # n_samples, C, D, 3
    images_rgb_reshaped = torch.permute(images_rgb, [0, 3, 1, 2])  # n_samples, 3, C, D

    return images_rgb_reshaped

if __name__ == "__main__":
    parallel_device = utils.get_parallel_device()
    torch.cuda.set_device(parallel_device)
    print(f"device: {parallel_device}")

    obj_file_name = "./data/meshes/cat/model.obj"
    mesh = io_utils.load_obj_as_normalized_mesh(obj_file_name, parallel_device)

    n_samples = 20

    cameras = get_random_cameras(mesh, 20, 0.1, parallel_device)
    images = render_views(mesh, cameras, 512, parallel_device)
    print(images.shape)

    plotting_utils.plot_image_grid(images.cpu().numpy(), rows=4, cols=5)
    plt.show()
