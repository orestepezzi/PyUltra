#! /usr/bin/python
from math import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm,BoundaryNorm
from matplotlib.ticker import MaxNLocator

import matplotlib as mpl

fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 8,
        "font.size": 8,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 7,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
}
mpl.rcParams.update(fonts)

###########################################################################################
def maps2D(field,x,y,strx,stry,strtitle,zmin,zmax,colormap,nb,isave,strsave):
    fig, ax = plt.subplots()
    fig.subplots_adjust(left=.1, bottom=.1, right=.85, top=.85)

    field=field.T
    levels=MaxNLocator(nbins=nb).tick_values(zmin,zmax)
    cmap = plt.get_cmap(colormap)

    cax = fig.add_axes([0.86, 0.1, 0.03, 0.75])
    cax.tick_params(labelsize=6)

    im = ax.contourf(x,y,field,levels=levels,cmap=cmap)
    cbar= plt.colorbar(im, cax=cax, orientation='vertical')

    ax.set_xlabel(strx)
    ax.set_ylabel(stry)
    ax.set_title(strtitle)
    ax.set_aspect('equal')

    width = 3.487 # inch
    height = width


    fig.set_size_inches(width, height)
    plt.show()
    if isave == 0:
        print("saving in: ", strsave)
        fig.savefig(strsave)


