import math
import matplotlib.pyplot as plt
import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import torchviz


import vgg_stylization
import batch_rendering
import utils
import plotting_utils
import io_utils

if __name__ == "__main__":
    parallel_device = utils.get_parallel_device()
    torch.cuda.set_device(parallel_device)
    print(f"device: {parallel_device}")

    mesh_fn = "./data/meshes/treefrog.obj"
    style_image_fn = "./data/images/picasso.jpg"
    mesh = io_utils.load_obj_as_normalized_mesh(mesh_fn, parallel_device)

    # PARAMETERS
    image_size = 256
    n_views = 16
    n_iters = 500

    style_image = io_utils.load_rgb_image(style_image_fn, image_size, device=parallel_device)
    style_images = style_image.unsqueeze(0).expand(n_views, -1, -1, -1)

    image_style_loss = vgg_stylization.ImageStyleLoss(style_images)

    mesh_verts_deformation = torch.zeros_like(mesh.verts_packed())
    mesh_verts_deformation.requires_grad_(True)

    optimizer = torch.optim.SGD([mesh_verts_deformation], lr=0.001)
    
    for i in range(n_iters):
        deformed_mesh = mesh.offset_verts(mesh_verts_deformation)
        cameras, lights = batch_rendering.get_random_cameras_and_lights(deformed_mesh, n_views, 1.0, parallel_device)
        content_images = batch_rendering.render_views(mesh, cameras, lights, image_size, parallel_device, background=style_images)
        image_content_loss = vgg_stylization.ImageContentLoss(content_images)
        
        rendered_images = batch_rendering.render_views(deformed_mesh, cameras, lights, image_size, parallel_device, background=style_images)

        content_score = 1.0 * image_content_loss(rendered_images)
        style_score = 1e4 * image_style_loss(rendered_images)
        loss = content_score + style_score

        print(f"iteration: {i}, loss: {loss.item()}, content score: {content_score.item()}, style_score: {style_score.item()}")

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i == n_iters - 1:
            plotting_utils.plot_image_grid(rendered_images.cpu().detach().numpy(), rows=4, cols=4)
            plt.show()

