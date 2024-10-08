# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import matplotlib.pyplot as plt
import numpy as np


def plot_image_grid(images, rows=None, cols=None, labels=None, fill: bool = True, show_axes: bool = False):
    """
    A util function for plotting a grid of images.

    Args:
        images: (N, 3, H, W) array of RGBA images
        rows: number of rows in the grid
        cols: number of columns in the grid
        fill: boolean indicating if the space between images should be filled
        show_axes: boolean indicating if the axes of the plots should be visible
        rgb: boolean, If True, only RGB channels are plotted.
            If False, only the alpha channel is plotted.

    Returns:
        None
    """
    if (rows is None) != (cols is None):
        raise ValueError("Specify either both rows and cols or neither.")
    
    if labels is None:
        labels = []
    while len(labels) < images.shape[0]:
        labels.append("")
    labels = labels[0:images.shape[0]]

    if rows is None:
        rows = len(images)
        cols = 1

    gridspec_kw = {"wspace": 0.0, "hspace": 0.0} if fill else {}
    fig, axarr = plt.subplots(rows, cols, gridspec_kw=gridspec_kw, figsize=(15, 9))
    bleed = 0
    fig.subplots_adjust(left=bleed, bottom=bleed, right=(1 - bleed), top=(1 - bleed))

    for ax, im, label in zip(axarr.ravel(), images, labels):
        im = np.transpose(im, [1, 2, 0])
        ax.imshow(im)
        ax.set_title(label)
        if not show_axes:
            ax.set_axis_off()

def plot_image_row(images, labels=None):
    n_images = len(images)
    fig, axarr = plt.subplots(1, n_images, squeeze=False)
    fig.set_figheight(360 / fig.dpi)
    fig.set_figwidth(300 / fig.dpi * n_images)
    for i in range(n_images):
        axarr[0, i].imshow(images[i])
        axarr[0, i].set_axis_off()
        if labels is not None and i < len(labels):
            axarr[0, i].title.set_text(labels[i])

