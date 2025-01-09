import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.collections import LineCollection
from contourpy import contour_generator
import cartopy.crs as ccrs
import xarray as xr
import tobac
import os
import json


def get_timestr(mask):
    # gets time string from mask with no time dimension
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
                ax.plot(t['longitude'],
                     t['latitude'],
                     'x',color=c[cell],markersize=6)
        elif drawmarkers:
            ax.plot(t['longitude'],
                     t['latitude'],
                     'o',color=c[cell],markersize=4)

    # add outlines and trails if enabled
    ax.add_collection(LineCollection(outlines,colors=outlinecolors,linewidths=1))
    if drawtrails:
        ax.add_collection(LineCollection(trails,colors=trailcolors,
                                         linewidths=1,linestyle="--"))


def _get_cells(MergeSplit,track):
    cells = []
    for f in MergeSplit.feature.values:
        if MergeSplit.sel(feature=f).cell_parent_track_id == track:
            cells.append(f)

def cells_by_track(MergeSplit):
    cbt = []
    for t in MergeSplit.track.values:
        cbt.append(_get_cells(MergeSplit,t))

def plot_segmentation_mergesplit(ax, Mask, Track, MergeSplit, index, c, drawtrails=True, drawmarkers=True, cbt=None):
    
    plot_segmentation(ax,Mask,Track,index,c,drawtrails=drawtrails,drawmarkers=drawmarkers)
    
    # select timestep from mask
    mask = Mask.segmentation_mask.isel(time=index)
    mask_prev = Mask.segmentation_mask.isel(time=index-1)

    # get time from mask as string
    timestr = get_timestr(mask)
    timestr_prev = get_timestr(mask_prev)
    
    if cbt is None:
        cbt = cells_by_track(MergeSplit)

    # subset Track table to relevant times
    track = Track[Track["timestr"]==timestr]
    track_prev = Track[Track["timestr"]==timestr_prev]

    
    # iterate over features
    for cell in MergeSplit.cell.values:
        track_id = int(MergeSplit.sel(cell=cell).cell_parent_track_id.values)
        
        # row of cell in current and previous timestep
        t = track[track["cell"]==cell]
        t_prev = track_prev[track_prev["cell"]==cell]
        
        # identify start of cell with split
        if t_prev.size == 0 and t.size > 0 and MergeSplit.sel(cell=cell).cell_starts_with_split:
            pass
        
        # identify end of cell with merge
        if t_prev.size > 0 and t.size == 0 and MergeSplit.sel(cell=cell).cell_ends_with_merge:
            pass
        

def run_tobac(ds, setting, datapath, resolution=0.02):
    r_Earth = 6378e3
    
    if not os.path.isdir(datapath):
        os.mkdir(datapath)
        
    # load parameters
    with open(f"tobac-parameters/{setting}.json") as f:
        tobac_params = json.load(f)
    
    # Determine temporal and spatial sampling of the input data:
    dxy,dt = tobac.get_spacings(ds.OLR,grid_spacing=resolution*r_Earth*np.pi/180)
    
    print('starting feature detection')
    Features=tobac.feature_detection_multithreshold(ds.OLR,dxy,**tobac_params["parameters_features"])
    Features.to_hdf(datapath + 'Features.h5',key='table')
    print('feature detection performed and saved')
    
    # Perform segmentation and save results to files:
    Mask_OLR,Features_OLR=tobac.segmentation_2D(Features,ds.OLR,dxy,**tobac_params["parameters_segmentation"])
    print('segmentation OLR performed, start saving results to files')
    Mask_OLR.to_netcdf(datapath+'Mask_Segmentation_OLR.nc')
    Features_OLR.to_hdf(datapath+'Features_OLR.h5', key='table')
    print('segmentation OLR performed and saved')
    
    # Perform linking and save results to file:
    Track=tobac.linking_trackpy(Features,ds.OLR,dt=dt,dxy=dxy,**tobac_params["parameters_linking"])
    Track.to_hdf(datapath+'Track.h5',key='table')
    
    MergeSplit = tobac.merge_split.merge_split_MEST(Track,dxy)
    MergeSplit.to_netcdf(datapath+'MergeSplit.nc')
    print('mergesplit complete')
    print('tobac complete')
    
    