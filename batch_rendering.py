import math
import matplotlib.pyplot as plt
import os
import sys
import torch

import pytorch3d
# Data structures and functions for rendering
from pytorch3d.structures import Meshes
from pytorch3d.vis.plotly_vis import AxisArgs, plot_batch_individually, plot_scene
from pytorch3d.vis.texture_vis import texturesuv_image_matplotlib
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

def get_random_cameras(n_samples: int, camera_distance: float, device: torch.device) -> FoVPerspectiveCameras:
    assert(camera_distance > 1.0)  # since the mesh is in a unit sphere, distance should be substantially above 1

    sphere_points = sample_points_on_unit_sphere(n_samples)
    elevations = torch.atan(sphere_points[:, 1] / sphere_points[:, 2])  # radians
    azimuths = torch.atan(sphere_points[:, 0] / sphere_points[:, 1])  # radians
    rots, tras = look_at_view_transform(camera_distance, elev=elevations, azim=azimuths, degrees=False)  # use radians
    cameras = FoVPerspectiveCameras(device=device, R=rots, T=tras)
    return cameras

def render_views(normalized_mesh: Meshes, cameras: FoVPerspectiveCameras, image_size: int, device: torch.device) -> torch.Tensor:
    """
    return shape: (n_samples, image_size, image_size, 4)
    """
    n_views = cameras.get_camera_center().shape[0]
    meshes = normalized_mesh.extend(n_views)

    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)

    lights = PointLights(device=device, location=[[0.0, 0.0, -3.0]])

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

    cameras = get_random_cameras(20, 2.0, parallel_device)
    images = render_views(mesh, cameras, 512, parallel_device)
    print(images.shape)

    plotting_utils.plot_image_grid(images.cpu().numpy(), rows=4, cols=5)
    plt.show()
