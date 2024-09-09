import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from PIL import Image
import matplotlib.pyplot as plt

import torchvision.transforms as transforms
from torchvision.models import VGG, vgg19, VGG19_Weights

import copy

import io_utils
import plotting_utils
import utils

class ContentLossRecorder(nn.Module):
    def __init__(self, target,):
        super(ContentLossRecorder, self).__init__()
        # we 'detach' the target content from the tree used
        # to dynamically compute the gradient: this is a stated value,
        # not a variable. Otherwise the forward method of the criterion
        # will throw an error.
        self.target = target.detach()

    def forward(self, input):
        self.loss = F.mse_loss(input, self.target)
        return input

class ImageContentLoss(nn.Module):
    def __init__(self, content_image):
        super(ImageContentLoss, self).__init__()
        parallel_device = utils.get_parallel_device()

        cnn: VGG = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(parallel_device)
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406], device=parallel_device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225], device=parallel_device)
        normalization = Normalization(cnn_normalization_mean, cnn_normalization_std)

        self.model = nn.Sequential(normalization).to(parallel_device)
        self.content_loss_recorders = []

        content_layers = ['conv_4']

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ``ContentLoss``
                # and ``StyleLoss`` we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            self.model.add_module(name, layer)

            if name in content_layers:
                # add content loss:
                target = self.model(content_image).detach()
                content_loss_recorder = ContentLossRecorder(target)
                self.model.add_module("content_loss_recorder_{}".format(i), content_loss_recorder)
                self.content_loss_recorders.append(content_loss_recorder)
        
        # trim
        i = len(self.model) - 1
        while not isinstance(self.model[i], ContentLossRecorder):
            i -= 1

        self.model = self.model[:(i + 1)]
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.model(image)
        
        total_loss = 0
        for content_loss_recorder in self.content_loss_recorders:
            total_loss += content_loss_recorder.loss
        
        return total_loss
        


def gram_matrix(input):
    b, f, c, d = input.size()  # b=batch size(=1)
    # f=number of feature maps
    # (c,d)=dimensions of a f. map (N=c*d)

    features = input.view(b, f, c * d)  # flatten matrices
    G = torch.bmm(features, features.permute(0, 2, 1))  # batch Gram product, permutation here is matrix transposition

    # 'normalize' the values of the gram matrix by dividing by the number of elements in feature maps
    return G.div(f * c * d)

class StyleLossRecorder(nn.Module):

    def __init__(self, target_feature):
        super(StyleLossRecorder, self).__init__()
        self.target = gram_matrix(target_feature).detach()

    def forward(self, input):
        G = gram_matrix(input)
        self.loss = F.mse_loss(G, self.target)
        return input

class ImageStyleLoss(nn.Module):
    def __init__(self, style_image):
        super(ImageStyleLoss, self).__init__()
        parallel_device = utils.get_parallel_device()

        cnn: VGG = vgg19(weights=VGG19_Weights.DEFAULT).features.eval().to(parallel_device)
        cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406], device=parallel_device)
        cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225], device=parallel_device)
        normalization = Normalization(cnn_normalization_mean, cnn_normalization_std)

        self.model = nn.Sequential(normalization).to(parallel_device)
        self.style_loss_recorders = []

        style_layers = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']

        i = 0  # increment every time we see a conv
        for layer in cnn.children():
            if isinstance(layer, nn.Conv2d):
                i += 1
                name = 'conv_{}'.format(i)
            elif isinstance(layer, nn.ReLU):
                name = 'relu_{}'.format(i)
                # The in-place version doesn't play very nicely with the ``ContentLoss``
                # and ``StyleLoss`` we insert below. So we replace with out-of-place
                # ones here.
                layer = nn.ReLU(inplace=False)
            elif isinstance(layer, nn.MaxPool2d):
                name = 'pool_{}'.format(i)
            elif isinstance(layer, nn.BatchNorm2d):
                name = 'bn_{}'.format(i)
            else:
                raise RuntimeError('Unrecognized layer: {}'.format(layer.__class__.__name__))

            self.model.add_module(name, layer)

            if name in style_layers:
                # add style loss
                target = self.model(style_image).detach()
                style_loss_recorder = StyleLossRecorder(target)
                self.model.add_module("style_loss_recorder_{}".format(i), style_loss_recorder)
                self.style_loss_recorders.append(style_loss_recorder)
        
        # trim
        i = len(self.model) - 1
        while not isinstance(self.model[i], StyleLossRecorder):
            i -= 1

        self.model = self.model[:(i + 1)]
    
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.model(image)
        
        total_loss = 0
        for style_loss_recorder in self.style_loss_recorders:
            total_loss += style_loss_recorder.loss
        
        return total_loss

class Normalization(nn.Module):

    def __init__(self, mean: torch.tensor, std: torch.tensor):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = mean.clone().detach().view(-1, 1, 1)
        self.std = std.clone().detach().view(-1, 1, 1)

    def forward(self, img):
        # normalize ``img``
        return (img - self.mean) / self.std

def run_style_transfer(input_images, content_images, style_images, num_steps,
                       content_weight=1.0, style_weight=1e6):
    """Run the style transfer."""
    print('Building the style transfer model..')

    # We want to optimize the input and not the model parameters so we
    # update all the requires_grad fields accordingly
    input_images.requires_grad_(True)
    # We also put the model in evaluation mode, so that specific layers
    # such as dropout or batch normalization layers behave correctly.

    optimizer = optim.LBFGS([input_images])

    image_content_loss = ImageContentLoss(content_images)
    image_style_loss = ImageStyleLoss(style_images)

    print('Optimizing..')
    run = [0]
    while run[0] <= num_steps:

        def closure():
            # correct the values of updated input image
            with torch.no_grad():
                input_images.clamp_(0, 1)

            style_score = 0
            
            content_score = content_weight * image_content_loss(input_images)
            style_score = style_weight * image_style_loss(input_images)

            loss = style_score + content_score
            
            optimizer.zero_grad()
            loss.backward()

            run[0] += 1
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
                print()

            return style_score + content_score

        optimizer.step(closure)

    # a last correction...
    with torch.no_grad():
        input_images.clamp_(0, 1)

    return input_images

if __name__ == "__main__":
    parallel_device = utils.get_parallel_device()
    print(f"Using default device: {parallel_device}")

    imsize = 512 if torch.cuda.is_available() else 128  # use small size if no GPU

    content_image_fns = ["./data/images/dancing.jpg", "./data/images/cat.png"]
    style_image_fns = ["./data/images/picasso.jpg"] * 2

    content_images = torch.stack([io_utils.load_rgb_image(fn, device=parallel_device) for fn in content_image_fns])
    style_images = torch.stack([io_utils.load_rgb_image(fn, device=parallel_device) for fn in style_image_fns])

    print(f"content images shape: {content_images.shape}, style images shape: {style_images.shape}")
    
    unloader = transforms.ToPILImage()  # reconvert into PIL image

    input_images = content_images.clone()

    output_images = run_style_transfer(input_images, content_images, style_images, 300)

    # sphinx_gallery_thumbnail_number = 4

    images = [output_images[0], output_images[1]]
    plot_images = [unloader(img.cpu().clone().squeeze(0)) for img in images]

    plotting_utils.plot_image_row(plot_images)
    plt.show()
    