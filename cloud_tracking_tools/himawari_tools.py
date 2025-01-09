import numpy as np
import cartopy.crs as ccrs
import dask.array as da
import xarray as xr
import xesmf as xe
import json
from scipy.constants import h,k,c


# Loading file
def load_himawari(yyyy:int, mm:int, dd:int, time:int, roi=[-30, -10, 140, 155], 
                  padding=5, resolution=0.02, regridder=None):

    basepath = "/g/data/gy85/Himawari8_AusGeo1-0-3/L1/summer_2021-2022/"
    # construct path to file
    datedirectory = f"{yyyy:04d}{mm:02d}/"
    yearday = dd
    if mm > 1:
        yearday += 31
    if mm > 2:
        yearday += 28
    file_prefix = "geocatL1.HIMAWARI-8."
    file_postfix = "00.FLDK.R20.nc"
    filename = f"{file_prefix}{yyyy:04d}{yearday:03d}.{time:04d}{file_postfix}"
    path = basepath + datedirectory + filename

    # open dataset
    ds = xr.open_dataset(path)

    # housekeeping
    ds = ds.rename({
        "pixel_latitude":"lat",
        "pixel_longitude":"lon",
        "himawari_8_ahi_channel_8_brightness_temperature":"ch8",
        "himawari_8_ahi_channel_13_brightness_temperature":"ch12",
        "himawari_8_ahi_channel_15_brightness_temperature":"ch15",
        "himawari_8_ahi_channel_16_brightness_temperature":"ch16"
    }).drop_vars([
        "himawari_8_ahi_channel_9_brightness_temperature",
        "himawari_8_ahi_channel_10_brightness_temperature",
        "himawari_8_ahi_channel_11_brightness_temperature",
        "himawari_8_ahi_channel_12_brightness_temperature",
        "himawari_8_ahi_channel_14_brightness_temperature",
        "himawari_8_ahi_channel_1_reflectance",
        "himawari_8_ahi_channel_2_reflectance",
        "himawari_8_ahi_channel_3_reflectance",
        "himawari_8_ahi_channel_4_reflectance",
        "himawari_8_ahi_channel_5_reflectance",
        "himawari_8_ahi_channel_6_reflectance",
        "himawari_8_ahi_channel_7_reflectance",
        "himawari_8_ahi_channel_7_brightness_temperature",
        "himawari_8_ahi_channel_7_emissivity",
        "pixel_ecosystem_type",
        "pixel_relative_azimuth_angle",
        "pixel_solar_zenith_angle",
        "pixel_surface_type"
    ])
    ds = ds.set_coords(("lat","lon"))
    
    
    
    # apply roi with padding (speeds up regrid)
    ds = ds.where(
        (ds.lat>roi[0]-padding) & (ds.lat<roi[1]+padding) &
        (ds.lon>roi[2]-padding) & (ds.lon<roi[3]+padding),
        drop = True
    )
    # regrid to lat/lon
    if regridder is None:
        regridder = compute_regridder(ds,roi,resolution)
    
    ds = regridder(ds)
    
    return ds, regridder


def compute_regridder(ds, roi, resolution):
    ds_out = xe.util.grid_2d(roi[2],roi[3],resolution,roi[0],roi[1],resolution)
    return xe.Regridder(ds, ds_out, "bilinear",weights="bilinear_1484x1353_1000x750.nc")



# Conversion to OLR using coefficients from Kim 2019
temp_to_narrowband_radiance = lambda T, lam: 2*h*c**2/lam**5/(np.exp(h*c/(k*T*lam))-1) * 1e-6

def compute_OLR_kim(ds):
    with open('coefficients-kim.json') as f:
        coeff_kim = json.load(f)
    
    theta = ds.pixel_satellite_zenith_angle*np.pi/180
    for ch in ["ch8","ch12","ch15","ch16"]:
        L = temp_to_narrowband_radiance(ds[ch],coeff_kim["central_wavelength"][ch]*1e-6)
        A = (coeff_kim["k"][ch][0] + coeff_kim["k"][ch][1]*(1/np.cos(theta)-1) 
             + coeff_kim["k"][ch][2]*(1/np.cos(theta)-1)**2)
        B = (coeff_kim["k"][ch][3] + coeff_kim["k"][ch][4]*(1/np.cos(theta)-1) 
             + coeff_kim["k"][ch][5]*(1/np.cos(theta)-1)**2)
        F = A*L + B
        ds[ch+"_F"] = (["y","x"],F.data)
    a = coeff_kim["a"]
    OLR = (a[0] + a[1]*ds.ch8_F + a[2]*ds.ch8_F**2 + 
           a[3]*ds.ch12_F + a[4]*ds.ch12_F**2 + 
           a[5]*np.log(ds.ch15_F) + a[6]*np.log(ds.ch15_F)**2 + 
           a[7]*ds.ch16_F + a[8]*ds.ch16_F**2)
    ds["OLR"] = (["y","x"],OLR.data)