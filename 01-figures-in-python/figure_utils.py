"""
MIT License

Copyright (c) 2021 Friedrich Miescher Institute for Biomedical Research - FAIM.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

Author: Tim-Oliver Buchholz
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib import cm
from matplotlib_scalebar.scalebar import ScaleBar

import colorsys

from os.path import splitext

def cm2in(x):
    """
    Convert cm to inch.
    """
    return x / 2.54


def add_scalebar(ax, length, pixel_size, units="um", location="lower right", show_label=False, color="white", box_color="black"):
    """
    Add scalebar to an axes. 
    
    Parameters:
    ax: Axes
        Add scalebar to this axes.
    length: float
        Length of the scalebar in units.
    pixel_size: float
        Size of a pixel in units.
    units: String
        Unit of the pixel size. Default: "um"
    location: String
        One of the location codes: "upper right", "upper left", 
        "lower left", "lower right", "right", "center left", 
        "center right", "lower center", "upper center" or "center"
    show_label: Bool
        Display the length of the scalebar. Default: False
    color: String
        Scalebar color. Default: "white"
    box_color: String
        Background color. Default: "black"
    """
    if show_label:
        sb = ScaleBar(fixed_value=length, dx=pixel_size, units=units, location=location, frameon=True, 
                  color=color, box_color=box_color)
    else:
        sb = ScaleBar(fixed_value=length, dx=pixel_size, units=units, location=location, frameon=True, 
                  color=color, box_color=box_color, scale_loc="none", label_loc="none")
        
    ax.add_artist(sb)
    
    
def add_img(ax, img, cmap="gray", vmin=0, vmax=255):
    """
    Add an image to a matplotlib.Axes.
    
    Parameters:
    -----------
    ax: matplotlib.Axes
        Axes object used to display the image.
    img: numpy Array 
        The image data (Y, X). (X, Y, 3) is without a cmap is treated as RGB image.
    cmap: Colormap or String
        Colormap of the display, if set to `None` the default colormap is used or RGB if the image shape is (X, Y, 3).
    vmin: float
        Display range minimum value.
    vmax: float
        Display range maximum value.
    """
    if cmap is None:
        ax.imshow(img, interpolation="nearest")
    else:
        ax.imshow(img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.axis('off')

    
def append_figure_size_in_cm(save_path, height, width):
    """
    Append "{height}x{width}cm" string to save_path.
    
    Parameters:
    save_path: String
        Output path.
    height: float
        Height in cm.
    width: float
        Width in cm.
    """
    path, ext = splitext(save_path)
    return path + "_{}x{}cm".format(np.round(height, 2), np.round(width, 2)) + ext


def create_img_figure(save_path, img, width, height, pixel_size, scalebar_length, scalebar_units="um", dpi=300, cmap="gray", vmin=0, vmax=255):
    """
    Save an image with a scalebar as a figure.
    
    Parameters:
    save_path: String
        Output path.
    img: numpy array
        The image data (X, Y) or (X, Y, 3).
    width: float
        Figure width in cm.
    height: float
        Figure height in cm.
    pixel_size: float
        Pixel size of the image data.
    scalebar_length: float
        Length of the scalebar in scalebar_units.
    scalebar_units: String
        Pixel size units. `Default: 'um'`
    dpi: int
        DPI at which this figure is saved. The data will be upscaled with nearest neighbor interpolation 
        if the image data has not sufficient amounts of pixels. `Default: 300`
    cmap: Colormap or String
        Colormap of the display, if set to `None` the default colormap is used or RGB if the image shape is (X, Y, 3).
    vmin: float
        Display range minimum value. `Default: vmin=0`
    vmax: float
        Display range maximum value. `Default: vmax=255`
    """
    fig = plt.figure(figsize=(cm2in(width), cm2in(height)), dpi=dpi)
    
    ax = fig.add_axes([0, 0, 1, 1])
    
    add_img(ax, img, cmap=cmap, vmin=vmin, vmax=vmax)
    if scalebar_length > 0:
        add_scalebar(ax, length=scalebar_length, pixel_size=pixel_size, units=scalebar_units)
    
    new_path = append_figure_size_in_cm(save_path, height, width)
    
    fig.savefig(new_path, bbox_inches="tight", pad_inches=0)
    
    
def create_img_figure_300dpi(save_path, img, pixel_size, scalebar_length, scalebar_units="um", cmap="gray", vmin=0, vmax=255):
    """
    Save an image at 300dpi with a scalebar as a figure.
    
    The figure size is computed based on the number of pixels provided by `img`.
    
    Parameters:
    save_path: String
        Output path.
    img: numpy array
        The image data (X, Y) or (X, Y, 3).
    pixel_size: float
        Pixel size of the image data.
    scalebar_length: float
        Length of the scalebar in scalebar_units.
    scalebar_units: String
        Pixel size units. `Default: 'um'`
    cmap: Colormap or String
        Colormap of the display, if set to `None` the default colormap is used or RGB if the image shape is (X, Y, 3).
    vmin: float
        Display range minimum value. `Default: vmin=0`
    vmax: float
        Display range maximum value. `Default: vmax=255`
    """
    height, width = img.shape[:2]
    fig_height = (height / 300) * 2.54
    fig_width = (width / 300) * 2.54
    
    create_img_figure(save_path=save_path,
                      img=img,
                      width=fig_width,
                      height=fig_height,
                      pixel_size=pixel_size,
                      scalebar_length=scalebar_length,
                      scalebar_units=scalebar_units,
                      dpi=300,
                      cmap=cmap,
                      vmin=vmin,
                      vmax=vmax)
    
    
def plot_color_range():
    """
    Plot colors according to their hue.
    """
    fig = plt.figure(figsize=(cm2in(10), cm2in(2)), dpi=300)
    ax = fig.add_axes([0, 0, 1, 1])
    
    color_range = ListedColormap([colorsys.hsv_to_rgb(i, 1, 1) for i in np.linspace(0, 1, 360)], name='color_range')
    colors, _ = np.meshgrid(range(360), range(10))
    
    ax.imshow(colors, cmap=color_range)
    ax.set_yticks([])
    ax.set_xticks(range(0, 361, 30))
    ax.set_xlabel('Hue value')
    
    ax.set_title('Color Range')
    
    
def get_gray_cmap():
    """
    Get the gray (black to white) colormap.
    """
    return cm.get_cmap('gray')


def create_colormap_from_hue(hue, name):
    """
    Create a linear colormap from dark to bright of a given hue with 256 steps.
    
    Parameters:
    hue: float [0, 360]
        Hue value between 0 and 360.
    name: String
        Name of the colormap.
        
    Returns:
        ListedColormap
    """
    return ListedColormap([colorsys.hsv_to_rgb(hue/360, 1, i) for i in np.linspace(0, 1, 256)])


def normalize_min_max(img, min_max):
    """
    Normalize an image to min_max = (min, max).
    
    Parameters:
    img: numpy array
        The image to normalize.
    min_max: tuple([float, float])
        Min and max value of the target image.
        
    Returns:
        Normalized image
    """
    vmin, vmax = min_max
    return (img - vmin)/(vmax-vmin)


def merge_rgba_images(imgs):
    """
    Merge RGBA images.
    
    Paremters:
    imgs: List(numpy arrays)
        List of RGBA images
        
    Returns:
        Merged RGBA image.
    """
    merge = imgs[0]
    for i in range(1, len(imgs)):
        merge = merge + imgs[i]
    return np.minimum(merge, 1)


def invert_rgba(rgba):
    """
    Invert RGBA image.
    
    Parameter:
    rgba: numpy array
        RGBA image
        
    Returns:
        Inverted rgba image.
    """
    inv = np.abs(1-rgba)
    inv[...,3] = rgba[...,3]
    return inv


def merge_channels(channels, cmaps):
    """
    Given a list of channels and corresponding colormaps a composit 
    image is created with all channels merged in a single RGBA image.
    
    Parameters:
    channels: List(numpy arrays)
        List of single channel 2D images (Y, X).
    cmaps: Colormaps
        List of colormaps.
        
    Returns:
        Merged channel image (Y, X, 4)
    """
    assert len(channels) == len(cmaps)
    
    color_channels = [lut(normalize_min_max(channel, (0, len(lut.colors)))) for channel, lut in zip(channels, cmaps)]
    
    return merge_rgba_images(color_channels)


def get_circles():
    """
    Create 3 images each containing a circle at a different position.
    
    Returns:
        List([numpy array,]) of 3 circle images
    """
    circles = []

    x, y = np.meshgrid(range(256), range(256))

    for d in range(30, 390, 120):
        img = np.zeros((256, 256), dtype=np.uint8)
        c = (np.sin(np.deg2rad(d)) * 50 + 128, -np.cos(np.deg2rad(d)) * 50 + 128)
        r = 70

        y_ = y - c[0]
        x_ = x - c[1]

        img[np.sqrt(y_**2 + x_**2) < r] = 255
        
        circles.append(img)
    return circles


def add_inset(ax, img, cmap, center, size, relative_inset_pos, relative_inset_size, vmin=0, vmax=255, color="gray", alpha=1):
    """
    Add an inset to an axes.
    
    Parameters:
    ax: matplotlib.Axes
        The axes to which the inset is added.
    img: numpy array
        The image data displayed in `ax`.
    cmap: Colormap or String
        Colormap of the display, if set to `None` the default colormap is used or RGB if the image shape is (X, Y, 3).
    center: [y, x]
        Center of the inset.
    size: [height, width]
        Size of the inset region.
    relative_inset_pos: [x0, y0]
        The relative position of the lower left corner of the inset. 
        Values must be between 0 and 1.
    relative_inset_size: [width, height]
        The relative size of the displayed inset.
        Values must be between 0 and 1.
    vmin: float
        Display range minimum value. `Default: vmax=255`
    vmax: float
        Display range maximum value. `Default: vmin=0`
    color: String
        Color of the inset outline. `Default: color="gray"`
    alpha: float
        Alpha of the inset outline. `Default: alpha=1`
    """
    axins = ax.inset_axes(relative_inset_pos + relative_inset_size)
    
    add_img(axins, img, cmap=cmap, vmin=vmin, vmax=vmax)
    axins.axis("on")
    axins.set_yticks([])
    axins.set_xticks([])
    axins.spines["bottom"].set_color(color)
    axins.spines["bottom"].set_alpha(alpha)
    axins.spines["top"].set_color(color) 
    axins.spines["top"].set_alpha(alpha)
    axins.spines["right"].set_color(color)
    axins.spines["right"].set_alpha(alpha)
    axins.spines["left"].set_color(color)
    axins.spines["left"].set_alpha(alpha)
    
    axins.set_ylim([center[0] - size[0]//2, center[0] + size[0]//2])
    axins.set_xlim([center[1] - size[1]//2, center[1] + size[1]//2])
    
    ax.indicate_inset_zoom(axins, edgecolor=color, alpha=alpha, clip_on=True);

    axins.invert_yaxis();
    
    
def create_img_figure_with_inset(save_path, img, width, height, pixel_size, scalebar_length, 
                                 inset_center, inset_size, inset_relative_pos, inset_relative_size,
                                 inset_indicator_color="gray", inset_indicator_alpha=1,
                                 scalebar_units="um", dpi=300, cmap="gray", vmin=0, vmax=255):
    """
    Save an image with a scalebar and an inset as a figure.
    
    Parameters:
    save_path: String
        Output path.
    img: numpy array
        The image data (X, Y) or (X, Y, 3).
    width: float
        Figure width in cm.
    height: float
        Figure height in cm.
    pixel_size: float
        Pixel size of the image data.
    scalebar_length: float
        Length of the scalebar in scalebar_units.
    inset_center: [y, x]
        Center of the inset.
    inset_size: [height, width]
        Size of the inset region.
    inset_relative_pos: [x0, y0]
        The relative position of the lower left corner of the inset. 
        Values must be between 0 and 1.
    inset_relative_size: [width, height]
        The relative size of the displayed inset.
        Values must be between 0 and 1.
    inset_indicator_color: String
        Color of the inset outline. `Default: color="gray"`
    inset_indicator_alpha: float
        Alpha of the inset outline. `Default: alpha=1`
    scalebar_units: String
        Pixel size units. `Default: 'um'`
    dpi: int
        DPI at which this figure is saved. The data will be upscaled with nearest neighbor interpolation 
        if the image data has not sufficient amounts of pixels. `Default: 300`
    cmap: Colormap or String
        Colormap of the display, if set to `None` the default colormap is used or RGB if the image shape is (X, Y, 3).
    vmin: float
        Display range minimum value. `Default: vmin=0`
    vmax: float
        Display range maximum value. `Default: vmax=255`
    """
    fig = plt.figure(figsize=(cm2in(width), cm2in(height)), dpi=dpi)
    
    ax = fig.add_axes([0, 0, 1, 1])
    
    add_img(ax, img, cmap=cmap, vmin=vmin, vmax=vmax)
    if scalebar_length > 0:
        add_scalebar(ax, length=scalebar_length, pixel_size=pixel_size, units=scalebar_units)
        
    add_inset(ax, img, cmap=cmap, center=inset_center, size=inset_size, relative_inset_pos=inset_relative_pos, relative_inset_size=inset_relative_size,
              vmin=vmin, vmax=vmax, color=inset_indicator_color, alpha=inset_indicator_alpha)
    
    new_path = append_figure_size_in_cm(save_path, height, width)
    
    fig.savefig(new_path, bbox_inches="tight", pad_inches=0)
    
    
def create_img_figure_with_inset_300dpi(save_path, img, pixel_size, scalebar_length, 
                                 inset_center, inset_size, inset_relative_pos, inset_relative_size,
                                 inset_indicator_color="gray", inset_indicator_alpha=1,
                                 scalebar_units="um", dpi=300, cmap="gray", vmin=0, vmax=255):
    """
    Save an image at 300dpi with a scalebar and an inset as a figure.
    
    The figure size is computed based on the number of pixels provided by `img`.
    Parameters:
    save_path: String
        Output path.
    img: numpy array
        The image data (X, Y) or (X, Y, 3).
    pixel_size: float
        Pixel size of the image data.
    scalebar_length: float
        Length of the scalebar in scalebar_units.
    inset_center: [y, x]
        Center of the inset.
    inset_size: [height, width]
        Size of the inset region.
    inset_relative_pos: [x0, y0]
        The relative position of the lower left corner of the inset. 
        Values must be between 0 and 1.
    inset_relative_size: [width, height]
        The relative size of the displayed inset.
        Values must be between 0 and 1.
    inset_indicator_color: String
        Color of the inset outline. `Default: color="gray"`
    inset_indicator_alpha: float
        Alpha of the inset outline. `Default: alpha=1`
    scalebar_units: String
        Pixel size units. `Default: 'um'`
    dpi: int
        DPI at which this figure is saved. The data will be upscaled with nearest neighbor interpolation 
        if the image data has not sufficient amounts of pixels. `Default: 300`
    cmap: Colormap or String
        Colormap of the display, if set to `None` the default colormap is used or RGB if the image shape is (X, Y, 3).
    vmin: float
        Display range minimum value. `Default: vmin=0`
    vmax: float
        Display range maximum value. `Default: vmax=255`
    """
    height, width = img.shape[:2]
    fig_height = (height / 300) * 2.54
    fig_width = (width / 300) * 2.54
    
    create_img_figure_with_inset(save_path=save_path,
                                 img=img,
                                 width=fig_width,
                                 height=fig_height,
                                 pixel_size=pixel_size,
                                 scalebar_length=scalebar_length,
                                 inset_center=inset_center,
                                 inset_size=inset_size,
                                 inset_relative_pos=inset_relative_pos,
                                 inset_relative_size=inset_relative_size,
                                 inset_indicator_color=inset_indicator_color,
                                 inset_indicator_alpha=inset_indicator_alpha,
                                 scalebar_units=scalebar_units,
                                 dpi=300,
                                 cmap=cmap,
                                 vmin=vmin,
                                 vmax=vmax)