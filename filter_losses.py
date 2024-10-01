import torch
import torch.nn.functional as F

def superpixel_loss(images: torch.Tensor, reduced_image_size: int) -> torch.Tensor:
    """
    batch size x C x H x W
    interpret as a lower-resolution image (3, lower resolution, lower_resolution)
    """
    B, C, S, S2 = images.shape
    assert S == S2
    assert C == 3
    assert S % reduced_image_size == 0

    superpixel_size = S // reduced_image_size
    superpixel_averages = F.avg_pool2d(images, superpixel_size).repeat(1, 1, superpixel_size, superpixel_size)
    return F.mse_loss(images, superpixel_averages)