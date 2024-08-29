import torch
import pytorch3d.ops
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from pytorch3d.structures import Meshes
from pytorch3d.ops import sample_points_from_meshes

from typing import Iterable

def set_dpi():
    matplotlib.rcParams['savefig.dpi'] = 80
    matplotlib.rcParams['figure.dpi'] = 80

def sample_point_cloud(mesh, n_points=5000):
    return pytorch3d.ops.sample_points_from_meshes(mesh, n_points)

def sample_and_plot_point_cloud(mesh, title=""):
    # Sample points uniformly from the surface of the mesh.
    points = pytorch3d.ops.sample_points_from_meshes(mesh, 5000)
    x, y, z = points.clone().detach().cpu().squeeze().unbind(1)    
    fig = plt.figure(figsize=(5, 5))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter3D(x, z, -y)
    ax.set_xlabel('x')
    ax.set_ylabel('z')
    ax.set_zlabel('y')
    ax.set_title(title)
    ax.view_init(190, 30)
    plt.show()

def plot_meshes(labeled_meshes: Iterable[tuple[Meshes, str]], n_points_per_cloud: int = 10000):
    px = 1.0 / plt.rcParams['figure.dpi']

    n_rows = len(labeled_meshes)
    n_cols = 2
    figure = plt.figure()
    figure.set_size_inches(2 * 400 * px, 400 * px * n_rows)

    for i in range(len(labeled_meshes)):
        meshes: Meshes = labeled_meshes[i][0]
        label: str = labeled_meshes[i][1]

        text_axes: Axes3D = figure.add_subplot(n_rows, n_cols, n_cols * i + 1)
        point_cloud_axes: Axes3D = figure.add_subplot(n_rows, n_cols, n_cols * i + 2, projection='3d')

        point_samples, normals = sample_points_from_meshes(meshes, n_points_per_cloud, return_normals=True)
        point_samples: torch.Tensor = point_samples.detach().squeeze()
        normals: torch.Tensor = normals.detach().squeeze()

        assert point_samples.shape[0] == normals.shape[0]

        
        # text plot
        text_axes.set_axis_off()
        full_desc = label + "\n" + f"({point_samples.shape[0]} points)"
        text_axes.text(0.5, 0.5, full_desc, ha='center', va='center')
        
        # point cloud plot
        x, y, z = point_samples.cpu().unbind(1)
        colors = normals.abs()  # in range [0, 1]
        colors = 0.25 + 0.75 * colors # in range [0.25, 1]
        colors = colors.clamp(0., 1.).cpu()
        point_cloud_axes.scatter3D(x, y, z, c=colors)
        point_cloud_axes.view_init(190, 30)
        point_cloud_axes.set_xticks([])
        point_cloud_axes.set_yticks([])
        point_cloud_axes.set_zticks([])
