#! /usr/bin/python
from math import *
import numpy as np
import matplotlib.pyplot as plt

###########################################################################################
#############
#############           3D Functions
#############
###########################################################################################


###########################################################################################
def curl_3D(x,y,z,fx,fy,fz):
    Nx = np.size(x)
    Ny = np.size(y)
    Nz = np.size(z)

    Lx = x[Nx-1]+x[1]
    Ly = y[Ny-1]+y[1]
    Lz = z[Nz-1]+z[1]

    wavenumbers_x = np.fft.fftfreq(Nx, d=1/Nx) * 2 * np.pi / Lx
    wavenumbers_y = np.fft.fftfreq(Ny, d=1/Ny) * 2 * np.pi / Ly
    wavenumbers_z = np.fft.rfftfreq(Nz, d=1/Nz) * 2 * np.pi / Lz

    wavenumbers = np.stack(np.meshgrid(wavenumbers_x, wavenumbers_y, wavenumbers_z, indexing='ij') )

    derivative_operator = 1j * wavenumbers

    curlf_x = np.fft.irfftn(derivative_operator[1,:] * np.fft.rfftn(fz) - derivative_operator[2,:] * np.fft.rfftn(fy) ,s=(Nx,Ny,Nz))
    curlf_y = np.fft.irfftn(derivative_operator[2,:] * np.fft.rfftn(fx) - derivative_operator[0,:] * np.fft.rfftn(fz) ,s=(Nx,Ny,Nz))
    curlf_z = np.fft.irfftn(derivative_operator[0,:] * np.fft.rfftn(fy) - derivative_operator[1,:] * np.fft.rfftn(fx) ,s=(Nx,Ny,Nz))

    return(curlf_x, curlf_y, curlf_z)

###########################################################################################
def strain_3D(x,y,z,fx,fy,fz):
    Nx = np.size(x)
    Ny = np.size(y)
    Nz = np.size(z)

    Lx = x[Nx-1]+x[1]
    Ly = y[Ny-1]+y[1]
    Lz = z[Nz-1]+z[1]

    wavenumbers_x = np.fft.fftfreq(Nx, d=1/Nx) * 2 * np.pi / Lx
    wavenumbers_y = np.fft.fftfreq(Ny, d=1/Ny) * 2 * np.pi / Ly
    wavenumbers_z = np.fft.rfftfreq(Nz, d=1/Nz) * 2 * np.pi / Lz

    wavenumbers = np.stack(np.meshgrid(wavenumbers_x, wavenumbers_y, wavenumbers_z, indexing='ij') )

    derivative_operator = 1j * wavenumbers

    dfx = gradf_3D(x,y,z,fx)
    dfy = gradf_3D(x,y,z,fy)
    dfz = gradf_3D(x,y,z,fz)

    divf = dfx[0,:,:,:] + dfy[1,:,:,:] + dfz[2,:,:,:]

    Dij2 = (dfx[0,:,:,:] - divf/3.0)**2 + (dfy[1,:,:,:]-divf/3.0)**2.0 + (dfz[2,:,:,:] - divf/3.0)**2.0 + \
            0.5*(dfx[1,:,:,:]+dfy[0,:,:,:])**2.0 + 0.5*(dfz[0,:,:,:] + dfx[2,:,:,:])**2.0 + 0.5*(dfz[1,:,:,:]+dfy[2,:,:,:])**2.0

    return(Dij2)
###########################################################################################
def div_3D(x,y,z,fx,fy,fz):
    Nx = np.size(x)
    Ny = np.size(y)
    Nz = np.size(z)

    Lx = x[Nx-1]+x[1]
    Ly = y[Ny-1]+y[1]
    Lz = z[Nz-1]+z[1]

    wavenumbers_x = np.fft.fftfreq(Nx, d=1/Nx) * 2 * np.pi / Lx
    wavenumbers_y = np.fft.fftfreq(Ny, d=1/Ny) * 2 * np.pi / Ly
    wavenumbers_z = np.fft.rfftfreq(Nz, d=1/Nz) * 2 * np.pi / Lz

    wavenumbers = np.stack(np.meshgrid(wavenumbers_x, wavenumbers_y, wavenumbers_z, indexing='ij') )

    derivative_operator = 1j * wavenumbers

    divf = np.fft.irfftn(derivative_operator[0,:] * np.fft.rfftn(fx) + \
            derivative_operator[1,:] * np.fft.rfftn(fy) + \
            derivative_operator[2,:] * np.fft.rfftn(fz), s=(Nx,Ny,Nz))

    return(divf)


###########################################################################################
def gradf_3D(x,y,z,f):
    Nx = np.size(x)
    Ny = np.size(y)
    Nz = np.size(z)

    Lx = x[Nx-1]+x[1]
    Ly = y[Ny-1]+y[1]
    Lz = z[Nz-1]+z[1]

    wavenumbers_x = np.fft.fftfreq(Nx, d=1/Nx) * 2 * np.pi / Lx
    wavenumbers_y = np.fft.fftfreq(Ny, d=1/Ny) * 2 * np.pi / Ly
    wavenumbers_z = np.fft.rfftfreq(Nz, d=1/Nz) * 2 * np.pi / Lz

    wavenumbers = np.stack(np.meshgrid(wavenumbers_x, wavenumbers_y, wavenumbers_z, indexing='ij') )

    derivative_operator = 1j * wavenumbers

    gradf = np.fft.irfftn(derivative_operator * np.fft.rfftn(f),s=(Nx,Ny,Nz))

    return(gradf)

###########################################################################################
def curvature3D(x,y,z,fx,fy,fz):
    fmod = np.sqrt(fx**2. + fy**2. + fz**2.)

    vfx = fx/fmod
    vfy = fy/fmod
    vfz = fz/fmod

    gradfx = gradf_3D(x,y,z,vfx)
    gradfy = gradf_3D(x,y,z,vfy)
    gradfz = gradf_3D(x,y,z,vfz)

    fgradfx = vfx*gradfx[0,:,:,:] + vfy*gradfx[1,:,:,:] + vfz*gradfx[2,:,:,:]
    fgradfy = vfx*gradfy[0,:,:,:] + vfy*gradfy[1,:,:,:] + vfz*gradfy[2,:,:,:]
    fgradfz = vfx*gradfz[0,:,:,:] + vfy*gradfz[1,:,:,:] + vfz*gradfz[2,:,:,:]

    k_curv  = np.sqrt(fgradfx**2.0 + fgradfy**2.0 + fgradfz**2.0)
    fn_curv = k_curv*fmod**2.0

    return(k_curv,fn_curv)

###########################################################################################
def MHDisoinv3D_PQRS(x,y,z,fx,fy,fz):

    gradfx = gradf_3D(x,y,z,fx)
    gradfy = gradf_3D(x,y,z,fy)
    gradfz = gradf_3D(x,y,z,fz)
    #print(np.shape(gradfx))

    A = np.stack((gradfx,gradfy,gradfz),axis=1)
    #print(np.shape(A))

    #print(np.max(A[0,0,:,:,:]-gradfx[0,:,:,:]))
    #print(np.max(A[1,0,:,:,:]-gradfx[1,:,:,:]))
    #print(np.max(A[2,0,:,:,:]-gradfx[2,:,:,:]))
    #print(np.max(A[0,1,:,:,:]-gradfy[0,:,:,:]))
    #print(np.max(A[1,1,:,:,:]-gradfy[1,:,:,:]))
    #print(np.max(A[2,1,:,:,:]-gradfy[2,:,:,:]))
    #print(np.max(A[0,2,:,:,:]-gradfz[0,:,:,:]))
    #print(np.max(A[1,2,:,:,:]-gradfz[1,:,:,:]))
    #print(np.max(A[2,2,:,:,:]-gradfz[2,:,:,:]))
    del gradfx
    del gradfy
    del gradfz

    A2 = np.matmul(A,A, axes=[(0,1),(0,1),(0,1)] )
    A3 = np.matmul(A2,A, axes=[(0,1),(0,1),(0,1)] )

    P = - np.trace(A)
    Q = - np.trace(A2)*(1.0/2.0)
    R = - np.trace(A3)*(1.0/3.0)

    At = np.transpose(A,(1,0,2,3,4))
    Sij = (A + At)/2.
    del At

    sij = Sij - np.mean(Sij,axis=(2,3,4))[:,:,None,None,None]

    sijsij = np.matmul(sij,sij, axes=[(0,1),(0,1),(0,1)] )
    S = np.trace(sijsij)

    (wx,wy,wz) = curl_3D(x,y,z,fx,fy,fz)

    Nx = np.size(x)
    Ny = np.size(y)
    Nz = np.size(z)

    S_star = np.zeros(shape=(Nx,Ny,Nz))
    costh  = np.zeros(shape=(Nx,Ny,Nz))
    for ix in range(Nx):
        for iy in range(Ny):
            for iz in range(Nz):
#The normalized (unit “length”) eigenvectors, such that the column eigenvectors[:,i] is the eigenvector corresponding to the eigenvalue eigenvalues[i].

                (egvals, egvecs) = np.linalg.eig(sij[:,:,ix,iy,iz])
                idx = np.argsort(egvals)[::-1]
                egvals = egvals[idx]
                egvecs = egvecs[:,idx]
                S_star[ix,iy,iz] = -3.0/np.sqrt(6.0)*egvals[0]*egvals[1]*egvals[2]/(egvals[0]**2.0 + egvals[1]**2.0 + egvals[2]**2.0)**(1.5)

                wmod = np.sqrt( wx[ix,iy,iz]**2.0 + wy[ix,iy,iz]**2.0 + wz[ix,iy,iz]**2.0 )
                costh[ix,iy,iz] = egvecs[0,1]*wx[ix,iy,iz] + egvecs[1,1]*wy[ix,iy,iz] + egvecs[2,1]*wz[ix,iy,iz]/wmod

    return(P,Q,R,S,S_star,costh)


###########################################################################################
#############
#############           2D Functions
#############
###########################################################################################

###########################################################################################
def curl_2D(x,y,fx,fy,fz):
    Nx = np.size(x)
    Ny = np.size(y)

    Lx = x[Nx-1]+x[1]
    Ly = y[Ny-1]+y[1]
    
    wavenumbers_x = np.fft.fftfreq(Nx, d=1/Nx) * 2 * np.pi / Lx
    wavenumbers_y = np.fft.fftfreq(Ny, d=1/Ny) * 2 * np.pi / Ly

    KX, KY = np.meshgrid(wavenumbers_x, wavenumbers_y)
    wavenumbers = np.stack((KX, KY))

    derivative_operator = 1j * wavenumbers

    curlf_x = np.fft.ifft2(derivative_operator[1,:] * np.fft.fft2(fz)).real
    curlf_y = np.fft.ifft2(-derivative_operator[0,:] * np.fft.fft2(fz)).real
    curlf_z = np.fft.ifft2(derivative_operator[0,:] * np.fft.fft2(fy) - derivative_operator[1,:] * np.fft.fft2(fx)).real
    
    return(curlf_x, curlf_y, curlf_z)

###########################################################################################
def div_2D(x,y,fx,fy,fz):
    Nx = np.size(x)
    Ny = np.size(y)

    Lx = x[Nx-1]+x[1]
    Ly = y[Ny-1]+y[1]

    wavenumbers_x = np.fft.fftfreq(Nx, d=1/Nx) * 2 * np.pi / Lx
    wavenumbers_y = np.fft.fftfreq(Ny, d=1/Ny) * 2 * np.pi / Ly

    KX, KY = np.meshgrid(wavenumbers_x, wavenumbers_y)
    wavenumbers = np.stack((KX, KY))

    derivative_operator = 1j * wavenumbers

    divf = np.fft.ifft2(derivative_operator[0,:] * np.fft.fft2(fx) + derivative_operator[1,:] * np.fft.fft2(fy)).real

    return(divf)

###########################################################################################
def strain_2D(x,y,fx,fy,fz):
    print("NB: Check! /3 or /2?")
    Nx = np.size(x)
    Ny = np.size(y)

    Lx = x[Nx-1]+x[1]
    Ly = y[Ny-1]+y[1]

    wavenumbers_x = np.fft.fftfreq(Nx, d=1/Nx) * 2 * np.pi / Lx
    wavenumbers_y = np.fft.fftfreq(Ny, d=1/Ny) * 2 * np.pi / Ly

    KX, KY = np.meshgrid(wavenumbers_x, wavenumbers_y)
    wavenumbers = np.stack((KX, KY))

    derivative_operator = 1j * wavenumbers
    
    dfx = gradf_2D(x,y,fx)
    dfy = gradf_2D(x,y,fy)
    dfz = gradf_2D(x,y,fz)

    divf = dfx[0,:,:] + dfy[1,:,:]
    Dij2 = (dfx[0,:,:] - divf/3.0)**2 + (dfy[1,:,:]-divf/3.0)**2.0 + \
            0.5*(dfx[1,:,:]+dfy[0,:,:])**2.0 + 0.5*dfz[0,:,:]**2.0 + 0.5*dfz[1,:,:]**2.0

    return(Dij2)


###########################################################################################
def gradf_2D(x,y,f):
    Nx = np.size(x)
    Ny = np.size(y)

    Lx = x[Nx-1]+x[1]
    Ly = y[Ny-1]+y[1]

    wavenumbers_x = np.fft.fftfreq(Nx, d=1/Nx) * 2 * np.pi / Lx
    wavenumbers_y = np.fft.fftfreq(Ny, d=1/Ny) * 2 * np.pi / Ly

    KX, KY = np.meshgrid(wavenumbers_x, wavenumbers_y)
    wavenumbers = np.stack((KX, KY))

    derivative_operator = 1j * wavenumbers

    gradf = np.fft.ifft2(derivative_operator * np.fft.fft2(f)).real

    return(gradf)

###########################################################################################
def Poisson_2D(x,y,f):
    print("NB: Check it!!!")

    Nx = np.size(x)
    Ny = np.size(y)

    Lx = x[Nx-1]+x[1]
    Ly = y[Ny-1]+y[1]

    wavenumbers_x = np.fft.fftfreq(Nx, d=1/Nx) * 2 * np.pi / Lx
    wavenumbers_y = np.fft.fftfreq(Ny, d=1/Ny) * 2 * np.pi / Ly

#    KX, KY = np.meshgrid(wavenumbers_x, wavenumbers_y)
#    wavenumbers = np.stack((KX, KY))

#    derivative_operator = 1j * wavenumbers

    fc = np.fft.fft2(f)
    dfc = np.zeros(fc.shape, dtype=np.complex128)

    for mx in range(int(Nx/2)):
        for my in range(int(Ny/2)):
            k2 = wavenumbers_x[mx]**2+ wavenumbers_y[my]**2
            if (k2 != 0):
                dfc[mx,my] = - fc[mx,my]/k2


    sol = np.fft.ifft2(dfc ).real

    return(sol)
