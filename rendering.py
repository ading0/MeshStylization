import math
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import torch
from typing import Optional

import pytorch3d
# Data structures and functions for rendering
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    BlendParams,
    FoVPerspectiveCameras, 
    PointLights, 
    DirectionalLights,
    HardFlatShader,
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

def get_random_cameras_and_lights(
    mesh: Meshes,
    n_samples: int,
    camera_distance: float, 
    device: torch.device
) -> tuple[FoVPerspectiveCameras, PointLights]:
    
    points, normals = sample_points_from_meshes(mesh, n_samples, return_normals=True)
    points = points.squeeze()
    normals = normals.squeeze()
    rot_list = []
    tra_list = []
    light_loc_list = []
    for i in range(n_samples):
        eye = (points[i] + camera_distance * normals[i]).unsqueeze(0)  # (1, 3)
        at = points[i].unsqueeze(0)
        up = torch.tensor([0, 1, 0], device=device).float().unsqueeze(0)  # (1, 3)
        
        rot, tra = look_at_view_transform(eye=eye, at=at, up=up)
        
        rot_list.append(rot)
        tra_list.append(tra)
        light_loc_list.append((eye + at) * 0.5)

    # all shapes below are (n_samples, 3)
    rots = torch.concat(rot_list, dim=0)
    tras = torch.concat(tra_list, dim=0)
    light_locs = torch.concat(light_loc_list, dim=0)

    # rots, tras = look_at_view_transform(camera_distance, elev=elevations, azim=azimuths, degrees=False)  # use radians
    cameras = FoVPerspectiveCameras(R=rots, T=tras, device=device)
    lights = PointLights(location=light_locs, device=device)
    return cameras, lights

def assemble_rgb_images(rgba_images: torch.tensor, background: torch.tensor, alpha_threshold=0.01) -> torch.tensor:
    """
    expected shape of rgba images, (n_samples, 4, C, D)
    background: (n_sampoes, 3, C, D)
    """
    assert rgba_images.shape[1] == 4

    rgb_raw = rgba_images[:, 0:3, :, :]
    alpha = rgba_images[:, 3:4, :, :]
    foreground_condition = torch.ge(alpha, alpha_threshold).expand(-1, 3, -1, -1)
    return torch.where(foreground_condition, rgb_raw, background)


def render_views(
    normalized_mesh: Meshes, 
    cameras: FoVPerspectiveCameras, 
    lights: DirectionalLights, 
    image_size: int,
    device: torch.device,
    background: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """
    return shape: (n_samples, image_size, image_size, 4)
    """
    n_views = len(cameras)

    if background is None:
        background = torch.ones(3, image_size, image_size).float().to(device)  # should broadcast to (n_samples, 3, image_size, image_size)

    meshes = normalized_mesh.extend(n_views)

    raster_settings = RasterizationSettings(image_size=image_size)

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader =  SoftPhongShader(device=device, cameras=cameras, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    materials = Materials(device=device)

    rgba_images_nonstandard_shape = renderer(meshes, materials=materials)  # (n_samples, C, D, 4)
    rgba_images = torch.permute(rgba_images_nonstandard_shape, [0, 3, 1, 2])  # (n_samples, 4, C, D)
    rgb_images = assemble_rgb_images(rgba_images, background)  # (n_samples, 3, C, D)
    return rgb_images

if __name__ == "__main__":
    parallel_device = utils.get_parallel_device()
    torch.cuda.set_device(parallel_device)
    print(f"device: {parallel_device}")

    mesh_fn = "./data/meshes/treefrog.obj"
    mesh = io_utils.load_obj_as_normalized_mesh(mesh_fn, parallel_device)

    n_samples = 20

    cameras, lights = get_random_cameras_and_lights(mesh, 20, 2.0, parallel_device)
    images = render_views(mesh, cameras, lights, 256, parallel_device)
    print(f"[DEBUG] images tensor shape: {images.shape}")

    plotting_utils.plot_image_grid(images.cpu().numpy(), rows=4, cols=5)
    plt.show()
