import math
import matplotlib.pyplot as plt
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchviz

from pytorch3d.loss import mesh_normal_consistency, mesh_edge_loss, chamfer_distance, mesh_laplacian_smoothing
from pytorch3d.renderer.mesh.textures import Textures

import vgg_losses
import rendering
import utils
import plotting_utils
import io_utils

if __name__ == "__main__":
    parallel_device = utils.get_parallel_device()
    torch.cuda.set_device(parallel_device)
    print(f"device: {parallel_device}")

    mesh_fn = "./data/meshes/bumpy_cube.obj"
    style_image_fn = "./data/images/abstract_flowers.png"
    mesh = io_utils.load_obj_as_normalized_mesh(mesh_fn, parallel_device)

    # PARAMETERS
    image_size = 256
    n_views = 2
    camera_distance = 1.5

    sigma_values =[]
    sigma_exp = -2.4
    while sigma_exp >= -3.5:
        sigma_values.append(10 ** sigma_exp)
        sigma_exp -= 0.2

    style_image = io_utils.load_rgb_image(style_image_fn, image_size, device=parallel_device)
    style_images = style_image.unsqueeze(0).expand(n_views, -1, -1, -1)
    image_style_loss = vgg_losses.ImageStyleLoss(style_images)

    cameras, lights = rendering.get_random_cameras_and_lights(mesh, n_views, camera_distance, parallel_device)
    
    renders_per_sigma = []
    labels = []
    
    for sigma in sigma_values:

        content_images = rendering.render_views(mesh, cameras, lights, image_size, sigma, parallel_device)

        mesh_verts_deformation = torch.zeros_like(mesh.verts_packed())
        mesh_verts_deformation.requires_grad_(True)
        mesh_verts_deformation.retain_grad()
        optimizer = torch.optim.SGD([mesh_verts_deformation], lr=0.005)

        deformed_mesh = mesh.offset_verts(mesh_verts_deformation)
        
        image_content_loss = vgg_losses.ImageContentLoss(content_images)
        rendered_images = rendering.render_views(deformed_mesh, cameras, lights, image_size, sigma, parallel_device)

        content_score = 1.0 * image_content_loss(rendered_images)
        style_score = 1e4 * image_style_loss(rendered_images)
        normal_consistency_score = 1e2 * mesh_normal_consistency(deformed_mesh)
        loss_value = content_score + style_score + normal_consistency_score

        optimizer.zero_grad()
        loss_value.backward()
        optimizer.step()

        # CODE TO VISUALIZE GRADIENT
        debug_image_size = 256

        vert_grads = mesh_verts_deformation.grad.clone().detach()
        grad_mesh = mesh.offset_verts(torch.zeros_like(mesh.verts_packed()))
        vert_grads_magnitudes = vert_grads.square().sum(dim=1).sqrt()  # shape = (n_verts)
        max_grad_magnitude = vert_grads_magnitudes.max()
        red = pow(vert_grads_magnitudes / max_grad_magnitude, 0.5)
        green = 1.0 - red
        colors = torch.zeros_like(vert_grads)
        colors[:, 0] = pow(vert_grads_magnitudes / max_grad_magnitude, 0.25)
        colors[:, 1] = 1.0 - colors[:, 0]
        colors = colors.unsqueeze(0)
        grad_mesh.textures = Textures(verts_rgb=colors)

        display_sigma = 1e-8

        grad_renders = rendering.render_views(grad_mesh, cameras, lights, debug_image_size, display_sigma, parallel_device)
        normal_renders = rendering.render_views(mesh, cameras, lights, debug_image_size, display_sigma, parallel_device)

        renders = torch.cat([grad_renders, normal_renders], dim=0)
        renders_per_sigma.append(renders)

        label = f"sigma={sigma:.2E}"
        for _ in range(renders.shape[0]):
            labels.append(label)
    

    plot_images = torch.cat(renders_per_sigma, dim=0).cpu().detach().numpy()
    print(plot_images.shape)

    n_images = plot_images.shape[0]

    plotting_utils.plot_image_grid(plot_images, n_images // 4, 4, labels)
    plt.show()

