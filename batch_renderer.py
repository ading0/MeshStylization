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

from mesh_io import load_obj_as_normalized_mesh
from plotting_utils import plot_image_grid
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

def sample_and_render_views(normalized_mesh: Meshes, n_samples: int, camera_distance: float, image_size: int) -> torch.Tensor:
    """
    return shape: (n_samples, image_size, image_size, 4)
    """
    assert(camera_distance > 1.0)  # since the mesh is in a unit sphere, distance should be substantially above 1

    meshes = normalized_mesh.extend(n_samples)

    sphere_points = sample_points_on_unit_sphere(n_samples)
    elevations = torch.atan(sphere_points[:, 1] / sphere_points[:, 2])  # radians
    azimuths = torch.atan(sphere_points[:, 0] / sphere_points[:, 1])  # radians

    rots, tras = look_at_view_transform(camera_distance, elev=elevations, azim=azimuths, degrees=False)  # use radians
    # rots shape (n_samples, 3, 3); tras shape (n_samples, 3)
    cameras = FoVPerspectiveCameras(device=parallel_device, R=rots, T=tras)
    raster_settings = RasterizationSettings(image_size=image_size, blur_radius=0.0, faces_per_pixel=1)

    lights = PointLights(device=parallel_device, location=[[0.0, 0.0, -3.0]])

    rasterizer = MeshRasterizer(cameras=cameras, raster_settings=raster_settings)
    shader =  SoftPhongShader(device=parallel_device, cameras=cameras, lights=lights)
    renderer = MeshRenderer(rasterizer=rasterizer, shader=shader)

    images = renderer(meshes)
    return images

if __name__ == "__main__":
    parallel_device = utils.get_parallel_device()
    torch.cuda.set_device(parallel_device)
    print(f"device: {parallel_device}")

    obj_file_name = "./data/meshes/cat/model.obj"
    mesh = load_obj_as_normalized_mesh(obj_file_name, parallel_device)

    n_samples = 20

    images = sample_and_render_views(mesh, n_samples, 2, 512)
    print(images.shape)

    plot_image_grid(images.cpu().numpy(), rows=4, cols=5, rgb=True)
    plt.show()
