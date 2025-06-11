#! /usr/bin/python
from math import *
import numpy as np
import matplotlib.pyplot as plt


###########################################################################################
###########################################################################################
def PiD(x,y,z,Pixx,Piyy,Pizz,Pixy,Pixz,Piyz,ux,uy,uz):
    Nx = np.size(x)
    Ny = np.size(y)
    Nz = np.size(z)

    if ( (Nz != 1) and (Ny != 1)):
        dux = PyUltra.FFTderiv.gradf_3D(x,y,z,ux)
        duy = PyUltra.FFTderiv.gradf_3D(x,y,z,uy)
        duz = PyUltra.FFTderiv.gradf_3D(x,y,z,uz)

        Pscalar=( Pixx + Piyy + Pizz)/3.0

        Pixx= Pixx - Pscalar
        Piyy= Piyy - Pscalar
        Pizz= Pizz - Pscalar

        divu = dux[0,:,:,:]+ duy[1,:,:,:] + duz[2,:,:,:]

        Ptheta = Pscalar*divu

        PiD = Pixx*(dux[0,:,:,:] - divu/3.0) + Pixy*(dux[1,:,:,:]+duy[0,:,:,:]) + Pixz*(dux[2,:,:,:]+duz[0,:,:,:]) + \
              Piyy*(duy[1,:,:,:] - divu/3.0) + Piyz*(duy[2,:,:,:]+duz[1,:,:,:]) + Pizz*(duz[2,:,:,:] - divu/3.0)

        return(PiD, Ptheta)

    elif (Ny != 1):
        dux = PyUltra.FFTderiv.gradf_2D(x,y,ux)
        duy = PyUltra.FFTderiv.gradf_2D(x,y,uy)
        duz = PyUltra.FFTderiv.gradf_2D(x,y,uz)

        Pscalar=( Pixx + Piyy + Pizz)/3.0

        Pixx= Pixx - Pscalar
        Piyy= Piyy - Pscalar
        Pizz= Pizz - Pscalar

        divu = dux[0,:,:]+ duy[1,:,:]

        Ptheta = Pscalar*divu

        PiD = Pixx*(dux[0,:,:] - divu/3.0) + Pixy*(dux[1,:,:]+duy[0,:,:]) + Pixz*(duz[0,:,:]) + \
              Piyy*(duy[1,:,:] - divu/3.0) + Piyz*(duz[1,:,:]) + Pizz*( - divu/3.0)

        return(PiD, Ptheta)


###########################################################################################
def Zenitani(Bx,By,Bz,Ex,Ey,Ez,jx,jy,jz,ux,uy,uz,rhoc,c):
    jE = jx*Ex + jy*Ey + jz*Ez 
    jE = jE + jx * (uy*Bz - uz*By)/c + jy * (uz*Bx - ux*Bz)/c + jz * (ux*By - uy*Bx)/c
    jE = jE - rhoc*(ux*Ex + uy*Ey + uz*Ez)

    return(jE)
###########################################################################################
def Epar(Bx,By,Bz,Ex,Ey,Ez):
    Epar = (Ex*Bx + Ey*By + Ez*Bz)/np.sqrt(Bx**2.0+ By**2.0 + Bz**2.0)
    return(Epar)

