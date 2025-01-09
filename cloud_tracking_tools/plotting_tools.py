import matplotlib.colors as mcolors
import numpy as np

def random_colormap(n=256,first_white=True):
    """
    Generate a random colormap with n colors.
    """
    colors = np.random.rand(n, 3)
    if first_white:
        colors[0,:] = [1,1,1]
    return mcolors.LinearSegmentedColormap.from_list("random_colormap", colors)