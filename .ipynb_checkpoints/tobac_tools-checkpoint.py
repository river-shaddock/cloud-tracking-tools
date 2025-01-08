import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from contourpy import contour_generator
import cartopy.crs as ccrs
import xarray as xr
import tobac


def get_timestr(mask):
    return mask.time.dt.strftime('%Y-%m-%d %H:%M:%S').data[()]



def plot_segmentation(ax, Mask, Track, index, c, drawtrails=True, drawmarkers=True):
    # select timestep from mask
    mask = Mask.segmentation_mask.isel(time=index)
    mask_prev = Mask.segmentation_mask.isel(time=index-1)

    # get time from mask as string
    timestr = get_timestr(mask)
    timestr_prev = get_timestr(mask_prev)

    # subset Track table to relevant times
    track = Track[Track["timestr"]==timestr]
    track_prev = Track[Track["timestr"]==timestr_prev]

    # initialise empty lists for drawing
    outlines = []; outlinecolors = []
    if drawtrails:
        trails = []; trailcolors = []

    # iterate over features
    for i, t in track.iterrows():
        cell = t["cell"]

        # row of cell in previous timestep
        t_prev = track_prev[track_prev["cell"]==cell]

        # generate contours and add to draw list
        cont_gen = contour_generator(x=Mask.longitude, y=Mask.latitude,
                                     z=xr.where(mask==t["feature"],1,0))
        l = cont_gen.lines(0.5)
        outlines += l
        outlinecolors += [c[cell]]*len(l)

        # draw trails and markers
        if t_prev['longitude'].values.size == 1:
            if drawtrails:
                trails.append(np.column_stack([
                    [t_prev['longitude'].values[0],t['longitude']],
                    [t_prev['latitude'].values[0],t['latitude']]]))
                trailcolors.append(c[cell])
            if drawmarkers:
                plt.plot(t['longitude'],
                     t['latitude'],
                     'x',color=c[cell],linewidth=0.5,markersize=3)
        elif drawmarkers:
            plt.plot(t['longitude'],
                     t['latitude'],
                     'o',color=c[cell],linewidth=0.2,markersize=2)

    # add outlines and trails if enabled
    ax.add_collection(LineCollection(outlines,colors=outlinecolors,linewidths=0.5))
    if drawtrails:
        ax.add_collection(LineCollection(trails,colors=trailcolors,
                                         linewidths=0.5,linestyle="--"))
    
    
