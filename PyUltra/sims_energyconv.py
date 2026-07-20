#! /usr/bin/python
from math import *
import numpy as np
import matplotlib.pyplot as plt
import PyUltra.FFTderiv

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


#        PiD = Pixx*dux[0,:,:] + Piyy*duy[1,:,:] + \
#               Pixy*(dux[1,:,:]+duy[0,:,:]) + \
#               Pixz*(duz[0,:,:] )  + \
#               Piyz*(duz[1,:,:] )  
#        PiD = PiD - Ptheta     

        PiD = Pixx*(dux[0,:,:] - divu/3.0) + Pixy*(dux[1,:,:]+duy[0,:,:]) + Pixz*(duz[0,:,:]) + \
              Piyy*(duy[1,:,:] - divu/3.0) + Piyz*(duz[1,:,:]) + Pizz*( - divu/3.0)

        return(PiD, Ptheta)

#############################################################
def PiD_components(x, y, z, Pixx, Piyy, Pizz, Pixy, Pixz, Piyz, ux, uy, uz):
    """
    Compute pressure-strain interaction split into:
      - PiD_total
      - Ptheta   = p * div(u)
      - PiD_normal
      - PiD_shear
      """

    Nx = np.size(x)
    Ny = np.size(y)
    Nz = np.size(z)

    # ----------------------------
    # 3D case
    # ----------------------------
    if ((Nz != 1) and (Ny != 1)):
        dux = PyUltra.FFTderiv.gradf_3D(x, y, z, ux)
        duy = PyUltra.FFTderiv.gradf_3D(x, y, z, uy)
        duz = PyUltra.FFTderiv.gradf_3D(x, y, z, uz)

        # Scalar pressure p = Tr(P)/3
        Pscalar = (Pixx + Piyy + Pizz) / 3.0

        # Deviatoric diagonal components Π_ii = P_ii - p
        Pixx_dev = Pixx - Pscalar
        Piyy_dev = Piyy - Pscalar
        Pizz_dev = Pizz - Pscalar

        # Divergence of velocity
        divu = dux[0, :, :, :] + duy[1, :, :, :] + duz[2, :, :, :]

        # Compressional heating term
        Ptheta = Pscalar * divu

        # Normal (diagonal) part
        PiD_normal = (
            Pixx_dev * (dux[0, :, :, :] - divu / 3.0)
            + Piyy_dev * (duy[1, :, :, :] - divu / 3.0)
            + Pizz_dev * (duz[2, :, :, :] - divu / 3.0)
        )

        # Shear (off-diagonal) part
        PiD_shear = (
            Pixy * (dux[1, :, :, :] + duy[0, :, :, :])
            + Pixz * (dux[2, :, :, :] + duz[0, :, :, :])
            + Piyz * (duy[2, :, :, :] + duz[1, :, :, :])
        )

        PiD_total = PiD_normal + PiD_shear

        return PiD_total, Ptheta, PiD_normal, PiD_shear

    # ----------------------------
    # 2D case
    # ----------------------------
    elif (Ny != 1):
        dux = PyUltra.FFTderiv.gradf_2D(x, y, ux)
        duy = PyUltra.FFTderiv.gradf_2D(x, y, uy)
        duz = PyUltra.FFTderiv.gradf_2D(x, y, uz)

        # Scalar pressure p = Tr(P)/3
        Pscalar = (Pixx + Piyy + Pizz) / 3.0

        # Deviatoric diagonal components Π_ii = P_ii - p
        Pixx_dev = Pixx - Pscalar
        Piyy_dev = Piyy - Pscalar
        Pizz_dev = Pizz - Pscalar

        # In 2D: div u = d ux/dx + d uy/dy
        divu = dux[0, :, :] + duy[1, :, :]

        # Compressional heating term
        Ptheta = Pscalar * divu

        # Normal (diagonal) part
        PiD_normal = (
            Pixx_dev * (dux[0, :, :] - divu / 3.0)
            + Piyy_dev * (duy[1, :, :] - divu / 3.0)
            + Pizz_dev * (-divu / 3.0)
        )

        # Shear (off-diagonal) part
        PiD_shear = (
            Pixy * (dux[1, :, :] + duy[0, :, :])
            + Pixz * duz[0, :, :]
            + Piyz * duz[1, :, :]
        )

        PiD_total = PiD_normal + PiD_shear

        return PiD_total, Ptheta, PiD_normal, PiD_shear

    else:
        raise ValueError("PiD_components currently supports 2D and 3D cases only.")
######################################################################
##########Test Megnetic Analog Pressure- Strain 

def TDB_components(x, y, z, Bx, By, Bz, ux, uy, uz):
    """
    Magnetic analog of pressure-strain decomposition.

    Returns
    -------
    TD_total
    TD_normal
    TD_shear
    eps
    divu_perp
    ux_perp, uy_perp, uz_perp
    """
    Nx = np.size(x)
    Ny = np.size(y)
    Nz = np.size(z)
# --- magnetic field magnitude and unit vector
    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

    bx = Bx / Bmag
    by = By / Bmag
    bz = Bz / Bmag
    # --- perpendicular velocity
    bdotu = bx*ux + by*uy + bz*uz

    ux_perp = ux - bx*bdotu
    uy_perp = uy - by*bdotu
    uz_perp = uz - bz*bdotu

    # --- magnetic energy density
    epsM = (Bx**2 + By**2 + Bz**2) /2.0 #(8.0*np.pi)

    # ----------------------------
    # 3D case
    # ----------------------------
    if ((Nz != 1) and (Ny != 1)):
        dux = PyUltra.FFTderiv.gradf_3D(x, y, z, ux_perp)
        duy = PyUltra.FFTderiv.gradf_3D(x, y, z, uy_perp)
        duz = PyUltra.FFTderiv.gradf_3D(x, y, z, uz_perp)

        divu_perp = dux[0,:,:,:] + duy[1,:,:,:] + duz[2,:,:,:]

        TD_normal = ((Bx**2)*dux[0,:,:,:] 
            + (By**2)*duy[1,:,:,:]
            + (Bz**2)*duz[2,:,:,:]
        ) - (2.0/3.0)*epsM*divu_perp

        TD_shear = (
            Bx*By*(dux[1,:,:,:] + duy[0,:,:,:])
            + Bx*Bz*(dux[2,:,:,:] + duz[0,:,:,:])
            + By*Bz*(duy[2,:,:,:] + duz[1,:,:,:])
        )

        TD_total = TD_normal + TD_shear

        return TD_total, TD_normal, TD_shear, epsM

    # ----------------------------
    # 2D case
    # ----------------------------
    elif (Ny != 1):
        dux = PyUltra.FFTderiv.gradf_2D(x, y, ux_perp)
        duy = PyUltra.FFTderiv.gradf_2D(x, y, uy_perp)
        duz = PyUltra.FFTderiv.gradf_2D(x, y, uz_perp)

        divu_perp = dux[0,:,:] + duy[1,:,:]

        TD_normal = (
            (Bx**2)*dux[0,:,:]
            + (By**2)*duy[1,:,:]
            + (Bz**2)*0.0
        ) - (2.0/3.0)*epsM*divu_perp

        TD_shear = (
            Bx*By*(dux[1,:,:] + duy[0,:,:])
            + Bx*Bz*(duz[0,:,:])
            + By*Bz*(duz[1,:,:])
        )

        TD_total = TD_normal + TD_shear

        return TD_total, TD_normal, TD_shear, epsM

    else:
        raise ValueError("TDB_components currently supports 2D and 3D cases only.")




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
###############################################################################################
###########################################################################################
def P_Hornet(
    x, y, z,
    phi_m, phi_c, phi_p,
    time_m, time_p,
    Pxx, Pyy, Pzz,
    ux, uy, uz
):
    """
    Compute the HORNET term:

        P_HORNET = -p * D(phi)/Dt

    where:

        p = (Pxx + Pyy + Pzz) / 3

        D(phi)/Dt = partial(phi)/partial(t) + u . grad(phi)

    The temporal derivative is computed with a centered difference:

        partial(phi)/partial(t)
            = (phi_p - phi_m) / (time_p - time_m)
    """

    Nx = np.size(x)
    Ny = np.size(y)
    Nz = np.size(z)

    dt = time_p - time_m

    if dt == 0.0:
        raise ValueError("P_Hornet: time_p and time_m produce dt = 0.")

    # Centered temporal derivative
    dphi_dt = (phi_p - phi_m) / dt

    # Scalar pressure: p = Tr(P)/3 = n*T
    Pscalar = (Pxx + Pyy + Pzz) / 3.0

    # ----------------------------
    # 3D case
    # ----------------------------
    if ((Nz != 1) and (Ny != 1)):

        grad_phi = PyUltra.FFTderiv.gradf_3D(x, y, z, phi_c)

        convection = (
              ux * grad_phi[0, :, :, :]
            + uy * grad_phi[1, :, :, :]
            + uz * grad_phi[2, :, :, :]
        )

    # ----------------------------
    # 2D case
    # ----------------------------
    elif (Ny != 1):

        grad_phi = PyUltra.FFTderiv.gradf_2D(x, y, phi_c)

        convection = (
              ux * grad_phi[0, :, :]
            + uy * grad_phi[1, :, :]
        )

    else:
        raise ValueError(
            "P_Hornet currently supports 2D and 3D cases only."
        )

    # Material derivative
    DphiDt = dphi_dt + convection

    # HORNET term
    PHornet = -Pscalar * DphiDt

    return PHornet, dphi_dt, convection, DphiDt










