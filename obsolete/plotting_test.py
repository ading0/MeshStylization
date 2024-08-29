from pytorch3d.io import load_obj, save_obj
from pytorch3d.structures import Meshes
from pytorch3d.utils import ico_sphere
from pytorch3d.ops import sample_points_from_meshes
from pytorch3d.loss import chamfer_distance, mesh_edge_loss, mesh_laplacian_smoothing, mesh_normal_consistency

import plotting_utils
import torch
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

    verts = verts - verts.mean(0)
    verts = verts / max(verts.abs().max(0)[0])

    target_mesh = Meshes(verts=[verts], faces=[faces_idx])
    work_mesh = ico_sphere(4, device)

    target_point_cloud = plotting_utils.sample_point_cloud(target_mesh).squeeze()
    source_point_cloud = plotting_utils.sample_point_cloud(work_mesh).squeeze()
    labeled_point_clouds = [(target_point_cloud, "target"), (source_point_cloud, "source")]

    vert_offsets = torch.full(work_mesh.verts_packed().shape, 0.0, device=device, requires_grad=True)
    optimizer = torch.optim.SGD([vert_offsets], lr=1.0, momentum=0.9)

    n_iters = 1000
    w_chamfer = 1.0
    w_edge = 1.0
    w_normal = 0.01
    w_laplacian = 0.1
    plot_period = 200
    cur_work_mesh = None
    for i in range(1000):
            
        optimizer.zero_grad()
        cur_work_mesh = work_mesh.offset_verts(vert_offsets)
        target_sample = sample_points_from_meshes(target_mesh, 500)
        work_sample = sample_points_from_meshes(cur_work_mesh, 500)

        chamfer_loss, _ = chamfer_distance(target_sample, work_sample)
        edge_loss = mesh_edge_loss(cur_work_mesh)
        normal_loss = mesh_normal_consistency(cur_work_mesh)
        laplacian_loss = mesh_laplacian_smoothing(cur_work_mesh)

        loss = chamfer_loss * w_chamfer + edge_loss * w_edge + normal_loss * w_normal + laplacian_loss * w_laplacian

        loss.backward()
        optimizer.step()
        if i % plot_period == 0:
            print(f"iteration #{i}, loss: {loss.item()}")
            labeled_point_clouds.append((work_sample.squeeze(), f"iteration {i}"))

    labeled_point_clouds.append((sample_points_from_meshes(cur_work_mesh, 5000).squeeze(), "final point cloud"))
    
    plotting_utils.plot_meshes(labeled_point_clouds)

    unix_time = datetime.datetime.now(datetime.UTC).timestamp()
    time_label = int(unix_time * 100)
    saved_figure_name = f"./results/figure_{time_label}.png"
    plt.savefig(saved_figure_name)
    print(f"Saved figure at: {saved_figure_name}")