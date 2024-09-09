import torch
import torchvision.transforms as transforms

from PIL import Image
from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.structures import Meshes


def load_obj_as_normalized_mesh(file_name: str, device: torch.device) -> Meshes:
    """
    Load a mesh from an obj file
    Normalization: centroid should be at the origin and the mesh should fit within a unit sphere
    Note that the return type is a Meshes of 1 mesh
    """
    mesh = load_objs_as_meshes([file_name], device=device)

    center = mesh.verts_packed().squeeze().mean(0)
    mesh.offset_verts_(-center)  # in-place offset; move centroid to origin
    scale = mesh.verts_packed().squeeze().square().sum(1).max().sqrt().item()
    mesh.scale_verts_(1.0 / scale * 0.999)  # in-place scale; fit in unit sphere with some tolerance

    return mesh

def load_rgb_image(file_name: str, image_size: int = 512, device: torch.device = torch.device("cpu")) -> torch.Tensor:
    """output should be (image_size, image_size, 3)"""
    
    loader = transforms.Compose([transforms.Resize(image_size), transforms.ToTensor()])
    pil_image = Image.open(file_name)
    image_tensor = loader(pil_image).to(device, torch.float)[0:3, :, :]
    return image_tensor


