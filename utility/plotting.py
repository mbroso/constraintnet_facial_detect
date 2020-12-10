import torch
import torchvision
import numpy as np
import matplotlib
import matplotlib.pyplot as plt 
import io
import cv2


def get_img_from_plt(fig, dpi=180):
    """Returns matplotlip figure as numpy array.

    Args:
        fig (obj): Matplotlib figure.

    Returns:
        img (obj): Figure as numpy array.
    """
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    return img


def polygon_on_img(img, polygon_xy=None, linewidth=5, scatter_pts=None):
    """Plot arbitrary number of landmarks as scatter points on images of a 
    batch and save the plot as image.

    Args:
        img (obj): Image [H, W, C].
        polygon_xy (obj): Numpy array with shape (n_v, 2) to plot 
            2d-polygons on img. E.g. to mark region of interests.
        *scatter_pts (obj): Arbitrary number of scatter points 
            for plotting on image (e.g.landmarks) in format 
            (pts, s, marker, c). pts is a tensors with shape 
            [N, 2], s/marker/c are format information for scatter points.
    """

    plt.clf()
    plt.imshow(img)
    
    #plot polygon
    ax = plt.gca()
    if not type(polygon_xy) == type(None):
        polygon = matplotlib.patches.Polygon(
                polygon_xy,
                edgecolor='r',
                fill=False,
                linewidth=linewidth)
        ax.add_patch(polygon)

    #plot scatter points
    if not scatter_pts == None:
        pts =  scatter_pts[0]
        s, marker, c = scatter_pts[1], scatter_pts[2], scatter_pts[3]
        for i, xy in enumerate(pts):
            x = xy[0]
            y = xy[1]
            plt.scatter(x, y, s=s, marker=marker, c=c, linewidth=linewidth)

    img_plt = get_img_from_plt(plt)
    return img_plt
