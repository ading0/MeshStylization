from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency

import plotting_utils
import torch
from torch.nn.functional import grid_sample
import matplotlib.pyplot as plt
import datetime

def get_device():
    return torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

if __name__ == "__main__":
    device = get_device()
    plotting_utils.set_dpi()

    file = "./data/dolphin.obj"
    verts, faces, aux = load_obj(file)
    faces_idx = faces.verts_idx.to(device)
    verts = verts.to(device)

    center = verts.mean(0)
    verts = verts - center

    scale = max(verts.abs().max(0)[0])
    verts = verts / scale

    original_mesh = Meshes(verts=[verts], faces=[faces_idx])
    work_mesh = original_mesh.detach().clone()

    vert_offsets = torch.full(work_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([vert_offsets], lr=1.0, momentum=0.9)

    conv_filters = torch.zeros(3, 5, 5, 5, requires_grad=False)
    conv_filters[0, :, 2, 2] = 1.0
    conv_filters[0, 2, :, 2] = 1.0
    conv_filters[0, 2, 2, :] = 1.0

    n_steps = 400
    plot_steps = set(range(0, n_steps, 200)).union(set([10, 100]))
    cur_mesh = None
    labeled_meshes = []
    
    grid_size = (5, 5, 5)
    filter_grid = conv_filters.repeat(1, 5, 5, 5)
    filter_grid_gs = filter_grid.reshape((1, 3, 25, 25, 25)).to(device)
    print(f"filter_grid shape: {filter_grid.shape} (for grid-sample {filter_grid_gs.shape})")

    n_point_samples_per_step = 1000
    print(f"# point samples per step: {n_point_samples_per_step}")

    for i in range(n_steps):
        optimizer.zero_grad()
        cur_mesh = work_mesh.offset_verts(vert_offsets)
        original_samples = sample_points_from_meshes(original_mesh, n_point_samples_per_step)
        cur_samples = sample_points_from_meshes(cur_mesh, n_point_samples_per_step)
        chamfer_loss, _ = chamfer_distance(original_samples, cur_samples)
        edge_loss = mesh_edge_loss(cur_mesh)
        normal_loss = mesh_normal_consistency(cur_mesh)
        laplacian_loss = mesh_laplacian_smoothing(cur_mesh)

        loss = torch.tensor([0.0]).to(device)
        loss += 1.0 * chamfer_loss
        loss += 1.0 * edge_loss
        loss += 0.01 * normal_loss
        loss += 0.1 * laplacian_loss

        """
        Compute convolutional loss
        """
        # random offset
        grid_offset = torch.rand(3, requires_grad=False).to(device) * 2. / 25.
        cur_samples_gs = (cur_samples + grid_offset).view(1, 1, cur_samples.shape[0], cur_samples.shape[1], cur_samples.shape[2])
        

        
        if i == 0:
            print(f"cur samples shape: {cur_samples.shape} (for gs: {cur_samples_gs.shape})")

        gs_output = grid_sample(filter_grid_gs, cur_samples_gs, padding_mode='border', align_corners=False)
        gs_output_squeezed = gs_output.squeeze()
        if i == 0:
            print(f"grid-sample output shape: {gs_output.shape}, squeezed shape: {gs_output_squeezed.shape}")

        loss += 1.0 * gs_output.mean()  # won't scale by number of samples

        if i in plot_steps:
            print(f"{i} steps, loss: {loss.item()}")
            labeled_meshes.append((cur_mesh.detach().clone(), f"iteration {i}"))

        loss.backward()
        optimizer.step()

    final_mesh = work_mesh.offset_verts(vert_offsets)
    labeled_meshes.append((final_mesh.detach().clone(), f"final, {n_steps} steps"))

    final_verts, final_faces = final_mesh.get_mesh_verts_faces(0)
    final_verts = final_verts * scale + center
    
    plotting_utils.plot_meshes(labeled_meshes)

    unix_time = datetime.datetime.now(datetime.UTC).timestamp()
    time_label = int(unix_time * 100)

    saved_figure_name = f"./results/figure_{time_label}.png"
    plt.savefig(saved_figure_name)
    print(f"Saved training plot at: {saved_figure_name}")

    saved_model_name = f"./results/output_{time_label}.obj"
    save_obj(saved_model_name, final_verts, final_faces)
    print(f"Saved final model's obj file at: {saved_model_name}")