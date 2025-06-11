#! /usr/bin/python
from math import *
import numpy as np
import matplotlib.pyplot as plt
#import re

###########################################################################################
def omnispectra_3D(x,y,z,fx,fy,fz):
    Nx = np.size(x)
    Ny = np.size(y)
    Nz = np.size(z)

    if ( (Nz != 1) and (Ny != 1)):
        Lx = x[Nx-1]+x[1]
        Ly = y[Ny-1]+y[1]
        Lz = z[Nz-1]+z[1]

        wavenumbers_x = np.fft.fftfreq(Nx, d=1/Nx) * 2 * np.pi / Lx
        wavenumbers_y = np.fft.fftfreq(Ny, d=1/Ny) * 2 * np.pi / Ly
        wavenumbers_z = np.fft.rfftfreq(Nz, d=1/Nz) * 2 * np.pi / Lz

        wavenumbers = np.stack(np.meshgrid(wavenumbers_x, wavenumbers_y, wavenumbers_z, indexing='ij') )
        k = np.sqrt( np.sum(wavenumbers**2.,axis=0) )

        ishell = np.rint(k)

        mask = (ishell > 0) & (ishell <= np.size(wavenumbers_z))

        fc  = np.fft.rfftn(fx)/(Nx*Ny*Nz)
        E3D = np.real(fc*np.conjugate(fc))

        fc  = np.fft.rfftn(fy)/(Nx*Ny*Nz)
        E3D = E3D + np.real(fc*np.conjugate(fc))

        fc  = np.fft.rfftn(fz)/(Nx*Ny*Nz)
        E3D = E3D + np.real(fc*np.conjugate(fc))

        E1D = np.zeros(np.size(wavenumbers_z))
        for ik in range(np.size(wavenumbers_z)):
            mask2 = mask & (ishell == wavenumbers_z[ik])
            E1D[ik] = np.sum(E3D,where= mask2)

        return(wavenumbers_z,E1D)

    elif (Ny != 1):
        Lx = x[Nx-1]+x[1]
        Ly = y[Ny-1]+y[1]

        wavenumbers_x = np.fft.fftfreq(Nx, d=1/Nx) * 2 * np.pi / Lx
        wavenumbers_y = np.fft.fftfreq(Ny, d=1/Ny) * 2 * np.pi / Ly

        wavenumbers = np.stack(np.meshgrid(wavenumbers_x, wavenumbers_y) )
        k = np.sqrt( np.sum(wavenumbers**2.,axis=0) )

        ishell = np.rint(k/wavenumbers_y[1])

        Nk = int(np.size(wavenumbers_y)/2)
        mask = (ishell > 0) & (ishell <= Nk)
        print(np.shape(mask))

        fc  = np.fft.fft2(fx)/(Nx*Ny)
        print(np.shape(fc))
        E3D = np.real(fc*np.conjugate(fc))

        fc  = np.fft.fft2(fy)/(Nx*Ny)
        E3D = E3D + np.real(fc*np.conjugate(fc))

        fc  = np.fft.fft2(fz)/(Nx*Ny)
        E3D = E3D + np.real(fc*np.conjugate(fc))

        E1D = np.zeros(Nk)
        for ik in range(Nk):
            mask2 = mask & (ishell == ik)#wavenumbers_y[ik])
            E1D[ik] = np.sum(E3D,where= mask2)

        return(wavenumbers_y[0:Nk],E1D)
        
    else:
        print("implement the 1D case")

###########################################################################################
def parperpspectra_3D(x,y,z,fx,fy,fz):
    Nx = np.size(x)
    Ny = np.size(y)
    Nz = np.size(z)

    if ( (Nz != 1) and (Ny != 1)):
        Lx = x[Nx-1]+x[1]
        Ly = y[Ny-1]+y[1]
        Lz = z[Nz-1]+z[1]

        wavenumbers_x = np.fft.fftfreq(Nx, d=1/Nx) * 2 * np.pi / Lx
        wavenumbers_y = np.fft.fftfreq(Ny, d=1/Ny) * 2 * np.pi / Ly
        wavenumbers_z = np.fft.rfftfreq(Nz, d=1/Nz) * 2 * np.pi / Lz

        wavenumbers = np.stack(np.meshgrid(wavenumbers_x, wavenumbers_y, wavenumbers_z, indexing='ij') )

        k_perp = np.sqrt( np.sum(wavenumbers[0:2,:]**2.,axis=0) )
        k_par = wavenumbers[2]

        ishell_perp = np.rint(k_perp)
        ishell_par  = np.rint(k_par)

        mask_perp = (ishell_perp > 0) & (ishell_perp <= np.size(wavenumbers_z))
        mask_par = (ishell_par > 0) & (ishell_par <= np.size(wavenumbers_z))

        fc  = np.fft.rfftn(fx)/(Nx*Ny*Nz)
        E3D = np.real(fc*np.conjugate(fc))

        fc  = np.fft.rfftn(fy)/(Nx*Ny*Nz)
        E3D = E3D + np.real(fc*np.conjugate(fc))

        fc  = np.fft.rfftn(fz)/(Nx*Ny*Nz)
        E3D = E3D + np.real(fc*np.conjugate(fc))

        Eperp = np.zeros(np.size(wavenumbers_z ))
        for ik_perp in range(np.size(wavenumbers_z)):
            mask2_perp = mask_perp & (ishell_perp == wavenumbers_z[ik_perp])
            Eperp[ik_perp] = np.sum(E3D, where= mask2_perp )

        Epar = np.zeros(np.size(wavenumbers_z ))
        for ik_par in range(np.size(wavenumbers_z)):
            mask2_par = mask_par & (ishell_par == wavenumbers_z[ik_par])
            Epar[ik_par] = np.sum(E3D, where= mask2_par )

        return(wavenumbers_z,wavenumbers_z,Eperp,Epar)

    elif (Ny != 1):
        print("implement the 2D case")

###########################################################################################
