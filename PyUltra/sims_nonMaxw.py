#! /usr/bin/python
from math import *
import numpy as np
import matplotlib.pyplot as plt


###########################################################################################
def PparPerp(Pixx,Piyy,Pizz,Pixy,Pixz,Piyz,fx,fy,fz):
    Ppar = fx**2.0*Pixx +fy**2.0*Piyy + fz**2.0*Pizz + 2.0*(fx*fy*Pixy + fx*fz*Pixz + fy*fz*Piyz )
    Ppar = Ppar/(fx**2.0 + fy**2.0 + fz**2.0)

    Pperp = Pixx + Piyy + Pizz - Ppar
    Pperp = Pperp*0.5

    return(Ppar,Pperp)

###########################################################################################
def sqrtQ(Pixx,Piyy,Pizz,Pixy,Pixz,Piyz,fx,fy,fz):
    I1=Pixx+Piyy+Pizz
    I2=Pixx*Piyy + Pixx*Pizz + Piyy*Pizz - (Pixy**2.0 + Pixz**2.0 + Piyz**2.0)

    Ppar = fx**2.0*Pixx +fy**2.0*Piyy + fz**2.0*Pizz + 2.0*(fx*fy*Pixy + fx*fz*Pixz + fy*fz*Piyz )
    Ppar = Ppar/(fx**2.0 + fy**2.0 + fz**2.0)

    Q=1.0 - 4.0*I2/((I1-Ppar)*(I1+3.0*Ppar))

    return(np.sqrt(Q))


###########################################################################################
#Ppar, Pperp should be already known. fx,fy,fz -> parallel direction 
# LBF -> fx,fy,fz=Bx,By,Bz + Ppar, Pperp standard
# MVF -> fx,fy,fz=e1x,e1y,e1z + Ppar =dl1/dens ; Pperp = 0.5*(dl2+dl3)/dens
def Dng(Pixx,Piyy,Pizz,Pixy,Pixz,Piyz,Ppar,Pperp,fx,fy,fz):
    fmod2 = fx**2.0 + fy**2.0 + fz**2.0

    traceP = Pixx + Piyy + Pizz
    
    Gxx = Pixx - (Pperp + (Ppar - Pperp)*fx*fx/fmod2)
    Gyy = Piyy - (Pperp + (Ppar - Pperp)*fy*fy/fmod2)
    Gzz = Pizz - (Pperp + (Ppar - Pperp)*fz*fz/fmod2)

    Gxy = Pixy - (Ppar - Pperp)*fx*fy/fmod2
    Gxz = Pixz - (Ppar - Pperp)*fx*fz/fmod2
    Gyz = Piyz - (Ppar - Pperp)*fy*fz/fmod2

    Dng = np.sqrt( (Gxx**2.0 + Gyy**2.0 + Gzz**2.0) - 2.0*(Gxy**2.0 + Gxz**2.0 + Gyz**2.0))/traceP

    return(Dng)
