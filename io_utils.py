import torch
import torchvision.transforms as transforms

from PIL import Image
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer.mesh.textures import Textures
from pytorch3d.structures import Meshes


def load_obj_as_normalized_mesh(file_name: str, device: torch.device, default_texture_color=(0.5, 0.5, 0.5)) -> Meshes:
    """
    Load a mesh from an obj file
    Normalization: centroid should be at the origin and the mesh should fit within a unit sphere
    Note that the return type is a Meshes of 1 mesh
    """
    assert len(default_texture_color) == 3

    mesh = load_objs_as_meshes([file_name], device=device)

    center = mesh.verts_packed().squeeze().mean(0)
    mesh.offset_verts_(-center)  # in-place offset; move centroid to origin
    scale = mesh.verts_packed().squeeze().square().sum(1).max().sqrt().item()
    mesh.scale_verts_(1.0 / scale * 0.999)  # in-place scale; fit in unit sphere with some tolerance

    if mesh.textures is None:
        rgb_color = torch.tensor(default_texture_color).to(device).unsqueeze(0).unsqueeze(0)  # (1, 1, 3)
        verts_rgb_colors = rgb_color.expand(-1, mesh.verts_packed().shape[0], -1)
        mesh.textures = Textures(verts_rgb=verts_rgb_colors)

    return mesh

def load_rgb_image(file_name: str, image_size: int, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """output should be (3, image_size, image_size)"""
    
    loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    pil_image = Image.open(file_name)
    image_tensor = loader(pil_image).to(device, torch.float)[0:3, :, :]
    return image_tensor


