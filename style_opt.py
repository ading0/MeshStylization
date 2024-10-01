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
import batch_rendering
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
    n_views = 16
    n_iters_per_view_set = 10
    n_view_sets = 1
    camera_distance = 1.0


    style_image = io_utils.load_rgb_image(style_image_fn, image_size, device=parallel_device)
    style_images = style_image.unsqueeze(0).expand(n_views, -1, -1, -1)

    image_style_loss = vgg_losses.ImageStyleLoss(style_images)

    mesh_verts_deformation = torch.zeros_like(mesh.verts_packed())
    mesh_verts_deformation.requires_grad_(True)
    mesh_verts_deformation.retain_grad()

    optimizer = torch.optim.SGD([mesh_verts_deformation], lr=0.005)
    
    for vs_idx in range(n_view_sets):
        
        cameras, lights = batch_rendering.get_random_cameras_and_lights(mesh, n_views, camera_distance, parallel_device)
        content_images = batch_rendering.render_views(mesh, cameras, lights, image_size, parallel_device, background=style_images)

        for it_idx in range(n_iters_per_view_set):

            deformed_mesh = mesh.offset_verts(mesh_verts_deformation)
            
            image_content_loss = vgg_losses.ImageContentLoss(content_images)
            rendered_images = batch_rendering.render_views(deformed_mesh, cameras, lights, image_size, parallel_device, background=style_images)

            content_score = 1.0 * image_content_loss(rendered_images)
            style_score = 1e4 * image_style_loss(rendered_images)
            normal_consistency_score = 1e2 * mesh_normal_consistency(deformed_mesh)
            loss_value = content_score + style_score + normal_consistency_score

            print(f"iter: {it_idx}, loss: {loss_value.item()}; content: {content_score.item()}, style: {style_score.item()}, "
                  f"normal consistency: {normal_consistency_score}")
            

            optimizer.zero_grad()
            loss_value.backward()
            optimizer.step()


            if it_idx == 0:
                # TEMPORARY CODE TO VISUALIZE GRADIENT
                vert_grads = mesh_verts_deformation.grad.clone().detach()
                print(f"grad shape: {mesh_verts_deformation.grad.shape}")
                gradient_display_mesh = mesh.offset_verts(torch.zeros_like(mesh.verts_packed()))
                vert_grads_magnitudes = vert_grads.square().sum(dim=1).sqrt()  # shape = (n_verts)
                max_grad_magnitude = vert_grads_magnitudes.max()
                red = pow(vert_grads_magnitudes / max_grad_magnitude, 0.5)
                green = 1.0 - red
                colors = torch.zeros_like(vert_grads)
                colors[:, 0] = pow(vert_grads_magnitudes / max_grad_magnitude, 0.25)
                colors[:, 1] = 1.0 - colors[:, 0]
                colors = colors.unsqueeze(0)
                gradient_display_mesh.textures = Textures(verts_rgb=colors)

                display_renders = batch_rendering.render_views(gradient_display_mesh, cameras, lights, image_size, parallel_device)
                plotting_utils.plot_image_grid(display_renders[0:16].cpu().detach().numpy(), rows=4, cols=4)
                plt.show()
    
        if vs_idx == n_view_sets - 1:
            
            deformed_mesh = mesh.offset_verts(mesh_verts_deformation)
            final_rendered_images = batch_rendering.render_views(deformed_mesh, cameras, lights, image_size, parallel_device)
            plotting_utils.plot_image_grid(final_rendered_images[0:16].cpu().detach().numpy(), rows=4, cols=4)
            plt.show()

