from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import xarray as xr
from contourpy import contour_generator
from matplotlib.collections import LineCollection, PatchCollection
from matplotlib.patches import Polygon
import matplotlib.pyplot as plt

def random_colormap(n=256,first_white=True,blue_background=False):
    """
    Generate a random colormap with n colors.
    """
    colors = np.random.rand(n, 3)
    if first_white:
        colors[0,:] = [1,1,1]
        
    # shifts colors away from blue when background is blue
    if blue_background:
        colors[:,2] = colors[:,2]*0.5
        colors[:,1] = colors[:,1]
        colors[:,0] = colors[:,0]*0.6+0.4
    return LinearSegmentedColormap.from_list("random_colormap", colors), colors


def plot_segmentation(
    ax, 
    cloudid, 
    T, 
    trackstats=None,
    feature_nums=None, 
    track_ids=None, 
    c=None, 
    linewidth=1, 
    blue_background=False, 
    patch=False, 
    patch_alpha=0.4
    ):
    
    """Plots segmentation mask on given axis.

    Args:
        ax (matplotlib.axes.Axes): Axis to plot on.
        cloudid (xarray.DataSet): mfdatabase of cloudid files.
        feature_nums (list): List of feature numbers to plot.
        track_ids (list, optional): List of track ids corresponding to feature_nums. If not supplied feature_nums is used.
        c (Any, optional): Colormap as array or None. If None, a random colormap is generated. Defaults to None.
        linewidth (int, optional): Width of lines. Defaults to 1.
        blue_background (bool, optional): Whether to use a blue background when generating colormap. Defaults to False
        patch (bool, optional): Whether to plot patches. Defaults to False.
        patch_alpha (float, optional): Alpha value for patches. Defaults to 0.4.
    """
    
    mask = cloudid.sel(time=T,method='nearest').feature_number
    
    
    if trackstats is not None:
        if feature_nums is None:
            feature_nums = trackstats.cloudnumber.values[trackstats.base_time.values==T].astype(int)
        if track_ids is None:
            track_ids = trackstats.track_id.values[np.sum(trackstats.base_time.values==T,axis=1)>0]
        if c is None:
            _, c = random_colormap(trackstats.track_id.max().values+1,blue_background=blue_background)
    
    if track_ids is None:
        track_ids = feature_nums
        
    if len(track_ids) != len(feature_nums):
        raise ValueError('feature_nums and track_ids must have the same length')

    # initialise empty lists for drawing
    outlines = []; outlinecolors = []
    
    coords = list(mask.coords) # should be time, lat, lon

    if patch:
        mask = mask.pad(lat=1,lon=1,mode='constant',constant_values=-1)
        newlon = mask.coords['lon'].values.copy()
        newlat = mask.coords['lat'].values.copy()
        newlon[0] = newlon[1] - (newlon[2]-newlon[1])
        newlat[0] = newlat[1] - (newlat[2]-newlat[1])
        newlon[-1] = newlon[-2] + (newlon[-2]-newlon[-3])
        newlat[-1] = newlat[-2] + (newlat[-2]-newlat[-3])
        
        
        mask.coords['lon'] = newlon
        mask.coords['lat'] = newlat
    # iterate over features
    for i in range(len(feature_nums)):
        feature = feature_nums[i]
    
        # generate contours and add to draw list
        cont_gen = contour_generator(x=mask[coords[2]], y=mask[coords[1]],
                                     z=xr.where(mask.squeeze()==feature,1,0))
        l = cont_gen.lines(0.5)
        outlines += l
        outlinecolors += [c[track_ids[i]]]*len(l)
        
     # Set axis limits
    ax.set_xlim(mask[coords[2]].min().values, mask[coords[2]].max().values)
    ax.set_ylim(mask[coords[1]].min().values, mask[coords[1]].max().values)

    
    if patch:
        # add patches to plot
        patches = [Polygon(line) for line in outlines]
        ax.add_collection(PatchCollection(patches,edgecolors=outlinecolors,facecolors=outlinecolors,alpha=patch_alpha))
        
    # add outlines to plot
    ax.add_collection(LineCollection(outlines,colors=outlinecolors,linewidths=linewidth))
    
    

def plot_track_flexrkr(ax,trackstats,track_id,T,color='b',recursive=False):
    
    if recursive:
        color = 'gray'
    
    track = np.argwhere(trackstats.track_id.values == track_id)[0,0]
    
    
    merge_tracks = np.argwhere(trackstats.end_merge_tracknumber.values == track_id)[:,0]
    
    for merging_track in merge_tracks:
        merge_time = trackstats.end_basetime[merging_track]
        if (merge_time < T) and (merge_time > (T - np.timedelta64(1,'h') - np.timedelta64(30,'m'))):
            ax.plot([trackstats.meanlon[track,trackstats.base_time[track,:]==merge_time],
                    trackstats.meanlon[merging_track,trackstats.base_time[merging_track,:]==merge_time]],
                    [trackstats.meanlat[track,trackstats.base_time[track,:]==merge_time],
                    trackstats.meanlat[merging_track,trackstats.base_time[merging_track,:]==merge_time]],
                    '--',
                    color='gray',
                    linewidth=0.5,
                    )
            plot_track_flexrkr(ax,trackstats,trackstats.track_id.values[merging_track],T,color=color,recursive=True)
    
    ax.plot(
        trackstats.meanlon.values[track,trackstats.base_time.values[track,:] <= T],
        trackstats.meanlat.values[track,trackstats.base_time.values[track,:] <= T],
        color=color)
    
    
    # print('Drawing track',track,':',starttime,'-',min(endtime,tindex))
    
    if (trackstats.start_basetime[track] <= T) and (trackstats.end_basetime[track] >= T):
        # if track still exists at this time
        # print('track',track,'still exists at time',T)
        ax.plot(trackstats.meanlon[track,trackstats.base_time[track,:] == T],
                trackstats.meanlat[track,trackstats.base_time[track,:] == T],
                'o',markersize=4,
                color=color)
        ax.text(trackstats.meanlon[track,trackstats.base_time[track,:] == T],
                trackstats.meanlat[track,trackstats.base_time[track,:] == T],
                str(track_id),color='black',in_layout=False)
    elif trackstats.start_basetime[track] > T:
        pass
    elif trackstats.end_basetime[track] < T:
        # if track has ended, mark last position
        ax.plot(trackstats.meanlon[track,trackstats.base_time[track,:] == trackstats.end_basetime[track]],
                trackstats.meanlat[track,trackstats.base_time[track,:] == trackstats.end_basetime[track]],
                'x',color='gray')
    
    
    
        
    
def plot_tracks_flexrkr(ax,trackstats,T,c='b',omit=[]):
    active_tracks = trackstats.track_id.values[np.sum(trackstats.base_time.values == T,axis=1)>0]
    # print('active tracks',active_tracks)
    omitted = []
    for o in omit:
        if o in active_tracks:
            omitted.append(o)
    if len(omitted) > 0:
        # print('omitted',omitted)
        pass
    for track_id in active_tracks:
        if track_id not in omit:
            if np.array(c).size > 1:
               plot_track_flexrkr(ax,trackstats,track_id,T,color=c[track_id])
            else:
                plot_track_flexrkr(ax,trackstats,track_id,T,color=c)
            
    
