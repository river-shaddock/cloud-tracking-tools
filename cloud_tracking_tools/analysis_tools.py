import numpy as np
from statsmodels.nonparametric.kernel_regression import KernelReg


def gaussian_kernel_regression(var, X, Xq, bw):
    kr = KernelReg(var, X, var_type='cc', reg_type='lc', bw=[bw,bw])
    return kr.fit(Xq)[0]


def gkern_reg_histgrid(var,X,extent,bins,bw):
    if np.array(bins).ndim == 0:
        bins = (bins,bins)
    padx = (extent[1]-extent[0])/(2*bins[0])
    pady = (extent[3]-extent[2])/(2*bins[1])
    x = np.linspace(extent[0]+padx,extent[1]-padx,bins[0])
    y = np.linspace(extent[2]+pady,extent[3]-pady,bins[1])
    Xq = np.array(np.meshgrid(x,y)).T.reshape(-1,2)
    Xplot,Yplot = np.meshgrid(
        np.linspace(extent[0],extent[1],bins[0]+1),
        np.linspace(extent[2],extent[3],bins[1]+1)
    )
    return Xplot, Yplot, gaussian_kernel_regression(var, X, Xq, bw).reshape(bins[1],bins[0]).T