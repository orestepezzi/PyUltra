#! /usr/bin/python
from math import *
import numpy as np
import matplotlib.pyplot as plt

###########################################################################################
def autocorr_3D(x,y,z,fx,fy,fz,Nlags):
    Nx = np.size(x)
    Ny = np.size(y)
    Nz = np.size(z)

    dx = x[1]

    fx0 = np.mean(fx)
    fy0 = np.mean(fy)
    fz0 = np.mean(fz)
    dfrms = np.sqrt(np.mean((fx-fx0)**2+(fy-fy0)**2+(fz-fz0)**2))

    fx = fx - fx0
    fy = fy - fy0
    fz = fz - fz0

    Nel = min(Nx,Ny,Nz)

    dl = np.unique(np.logspace(0,int(np.round(np.log10(Nel))),Nlags).astype(int))

    Rb_x = np.zeros(shape=(np.size(dl)))
    Rb_y = np.zeros(shape=(np.size(dl)))
    Rb_z = np.zeros(shape=(np.size(dl)))

    for il in range( np.size(dl) ):
    #x increments
        Rb_x[il] =  np.mean( np.ravel( fx*np.roll(fx,-int(dl[il]),axis=0) + fy*np.roll(fy,-int(dl[il]),axis=0) + fz*np.roll(fz,-int(dl[il]),axis=0) ) )
    #y increments
        Rb_y[il] =  np.mean( np.ravel( fx*np.roll(fx,-int(dl[il]),axis=1) + fy*np.roll(fy,-int(dl[il]),axis=1) + fz*np.roll(fz,-int(dl[il]),axis=1) ) )
    #y increments
        Rb_z[il] =  np.mean( np.ravel( fx*np.roll(fx,-int(dl[il]),axis=2) + fy*np.roll(fy,-int(dl[il]),axis=2) + fz*np.roll(fz,-int(dl[il]),axis=2) ) )


    lc_x = np.sum(Rb_x)*dx*0.5/(dfrms**2.0)
    lc_y = np.sum(Rb_y)*dx*0.5/(dfrms**2.0)
    lc_z = np.sum(Rb_z)*dx*0.5/(dfrms**2.0)

    return(lc_x, lc_y, lc_z)

###########################################################################################
def structfunc_3D(x,y,z,f,Nlags,NSFs):
    Nx = np.size(x)
    Ny = np.size(y)
    Nz = np.size(z)

    dx = x[1]

    Nel = min(Nx,Ny,Nz)

    dl = np.unique(np.logspace(0,int(np.round(np.log10(Nel))),Nlags).astype(int))

    SFdf_x = np.zeros(shape=(np.size(dl),NSFs),dtype=np.float32)
    SFdf_y = np.zeros(shape=(np.size(dl),NSFs),dtype=np.float32)
    SFdf_z = np.zeros(shape=(np.size(dl),NSFs),dtype=np.float32)

    for il in range(np.size(dl)):
        for isf in range(NSFs):
                SFdf_x[il,isf] = np.mean(np.abs( np.ravel(f - np.roll(f,-int(dl[il]),axis=0) ) )**(isf+1))
                SFdf_y[il,isf] = np.mean(np.abs( np.ravel(f - np.roll(f,-int(dl[il]),axis=2) ) )**(isf+1))
                SFdf_z[il,isf] = np.mean(np.abs( np.ravel(f - np.roll(f,-int(dl[il]),axis=2) ) )**(isf+1))

    dl = dl*dx
    return(dl,SFdf_x,SFdf_y,SFdf_z)


###########################################################################################
def PP(x,y,z,zpx,zpy,zpz,zmx,zmy,zmz,Nlags):
    Nx = np.size(x)
    Ny = np.size(y)
    Nz = np.size(z)

    dx = x[1]

    Nel = min(Nx,Ny,Nz)

    dl = np.unique(np.logspace(0,int(np.round(np.log10(Nel))),Nlags).astype(int))

    Yp = np.zeros(shape=(np.size(dl),3))
    Ym = np.zeros(shape=(np.size(dl),3))


    for il in range(np.size(dl)):
    #Yp
        Yp[il,0] = np.mean( (np.ravel(zpx-np.roll(zpx,-int(dl[il]),axis=0))**2 + \
                np.ravel(zpy-np.roll(zpy,-int(dl[il]),axis=0))**2 + \
                np.ravel(zpz-np.roll(zpz,-int(dl[il]),axis=0))**2)* \
                np.ravel(zmx-np.roll(zmx,-int(dl[il]),axis=0)) )

        Yp[il,1] = np.mean( (np.ravel(zpx-np.roll(zpx,-int(dl[il]),axis=1))**2 + \
                np.ravel(zpy-np.roll(zpy,-int(dl[il]),axis=1))**2 + \
                np.ravel(zpz-np.roll(zpz,-int(dl[il]),axis=1))**2)* \
                np.ravel(zmy-np.roll(zmy,-int(dl[il]),axis=1)))

        Yp[il,2] = np.mean( (np.ravel(zpx-np.roll(zpx,-int(dl[il]),axis=2))**2 + \
                np.ravel(zpy-np.roll(zpy,-int(dl[il]),axis=2))**2 + \
                np.ravel(zpz-np.roll(zpz,-int(dl[il]),axis=2))**2)* \
                np.ravel(zmz-np.roll(zmz,-int(dl[il]),axis=2)))

    #Ym
        Ym[il,0] = np.mean( (np.ravel(zmx-np.roll(zmx,-int(dl[il]),axis=0))**2 + \
                np.ravel(zmy-np.roll(zmy,-int(dl[il]),axis=0))**2 + \
                np.ravel(zmz-np.roll(zmz,-int(dl[il]),axis=0))**2)* \
                np.ravel(zpx-np.roll(zpx,-int(dl[il]),axis=0)) )

        Ym[il,1] = np.mean( (np.ravel(zmx-np.roll(zmx,-int(dl[il]),axis=1))**2 + \
                np.ravel(zmy-np.roll(zmy,-int(dl[il]),axis=1))**2 + \
                np.ravel(zmz-np.roll(zmz,-int(dl[il]),axis=1))**2)* \
                np.ravel(zpy-np.roll(zpy,-int(dl[il]),axis=1)))

        Ym[il,2] = np.mean( (np.ravel(zmx-np.roll(zmx,-int(dl[il]),axis=2))**2 + \
                np.ravel(zmy-np.roll(zmy,-int(dl[il]),axis=2))**2 + \
                np.ravel(zmz-np.roll(zmz,-int(dl[il]),axis=2))**2)* \
                np.ravel(zpz-np.roll(zpz,-int(dl[il]),axis=2)))

    dl = dl * dx
    return(dl,Yp,Ym)


