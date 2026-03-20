# -*- coding: utf-8 -*-
"""
Created on Fri Jul  5 12:42:19 2024

@author: Andrea
"""
import numpy as np
import pandas as pd
import datetime
import wget
import cdflib
import numpy as np
#from datetime import datetime
import datetime
import os.path
import bisect
import matplotlib.pyplot as plt
from matplotlib import ticker, cm
import warnings 
from scipy.interpolate import LinearNDInterpolator
warnings.filterwarnings("ignore")

######################  ###################################

def compute_d3V(vel, thetaReshaped, phiReshaped):
    """
    compute the volume element for the SPAN ion instrument (see Livi+2020)
   
    vel is the energy grid in velocity units
   
    thetaReshaped contains the values of the angle elevation of 
    the instrument

    phiReshaped contains the azimuth of the instrument
    
    phiReshaped, vel and thetaReshaped have 3 dimensions
    the first one corresponds to phi, the second to the velocity or energy
    and the third to theta

    """
    cos_theta = np.cos(thetaReshaped*np.pi/180)
    
    # steps in velocity
    dV_single = np.concatenate((np.diff(vel[0, :, 0]), [np.diff(vel[0, :, 0])[-1]]))
    
    # step in azimuth in radiants
    dphi = np.diff(phiReshaped[:, 0, 0])[0]*np.pi/180    
    # step in theta in radiants
    # mean because dtheta changes a bit but the moments computations
    # agrees with the L3 data products
    dtheta = np.mean(np.diff(thetaReshaped[0, 0, :]))*np.pi/180 
    
    # volume elements                   
    dV_Int = np.tile(dV_single, (8, 8)).reshape(8, 32, 8, order='F')
    

    return cos_theta*dtheta*(vel**2)*dV_Int*dphi
    



def computeVmean1comp(vdf, vel, v_comp, thetaReshaped, phiReshaped):
    
    # d3V for the instrument
    cos_theta = np.cos(thetaReshaped*np.pi/180)

    dV_single = np.concatenate((np.diff(vel[0, :, 0]), [np.diff(vel[0, :, 0])[-1]]))

    dphi = np.diff(phiReshaped[:, 0, 0])[0]*np.pi/180 # delta phi in radianti
    dtheta = np.mean(np.diff(thetaReshaped[0, 0, :]))*np.pi/180 # mean because dtheta changes a bit,
    
    #                     
    dV_Int = np.tile(dV_single, (8, 8)).reshape(8, 32, 8, order='F')


    num_int = np.sum(v_comp*vdf*cos_theta*dtheta*(vel**2)*dV_Int*dphi)

    den_int = np.sum(vdf*cos_theta*dtheta*(vel**2)*dV_Int*dphi)
    # questo funziona bene, sbaglio le velocita medie id qulche km/s


    v_comp_mean = num_int/den_int
    
    return v_comp_mean



def computeVmean(vdf, vel, vx, vy, vz, thetaReshaped, phiReshaped):
        
    vx_mean = computeVmean1comp(vdf, vel, vx, thetaReshaped, phiReshaped)
    vy_mean = computeVmean1comp(vdf, vel, vy, thetaReshaped, phiReshaped)
    vz_mean = computeVmean1comp(vdf, vel, vz, thetaReshaped, phiReshaped)
    
    return np.array((vx_mean, vy_mean, vz_mean))
    

def computeThermalSpeed(vdf, vel, vx, vy, vz, thetaReshaped, phiReshaped):
    """compute the full thermal speed tensor"""
    
    vx_mean, vy_mean, vz_mean = computeVmean(vdf, vel, vx, vy, vz,
                                             thetaReshaped, phiReshaped)

    d3V = compute_d3V(vel, thetaReshaped, phiReshaped)
    
    den_int = np.sum(vdf*d3V)
    # diagonal 
    vthxx_square = np.sum((vx-vx_mean)*(vx-vx_mean)*vdf*d3V)/den_int
    vthyy_square = np.sum((vy-vy_mean)*(vy-vy_mean)*vdf*d3V)/den_int
    vthzz_square = np.sum((vz-vz_mean)*(vz-vz_mean)*vdf*d3V)/den_int
    
    # off diagonal
    vthxy_square = np.sum((vx-vx_mean)*(vy-vy_mean)*vdf*d3V)/den_int
    vthxz_square = np.sum((vx-vx_mean)*(vz-vz_mean)*vdf*d3V)/den_int
    vthyz_square = np.sum((vy-vy_mean)*(vz-vz_mean)*vdf*d3V)/den_int
    
    
    # to have the temperatures multiply by mass_p
    # take the square root to have the thermal speed
    return np.array((vthxx_square, vthyy_square, vthzz_square, vthxy_square,
                      vthxz_square, vthyz_square))
    



def spp_swp_spi_VDF_data(timeslice):
    """Given the desired timeslice gives back the vdf data from
       span-ion with the velocities and the angle bins and and energy
       (in velocity units) in order to be able to compute integrals
    """
    
    # !pip install wget
    import wget
    import cdflib
    import numpy as np
    from datetime import datetime
    import os.path
    import bisect
    import matplotlib.pyplot as plt

    from matplotlib import ticker, cm
    import warnings 
    warnings.filterwarnings("ignore")


    from warnings import simplefilter 
    simplefilter(action='ignore', category=DeprecationWarning)
    
    year  = timeslice.year
    month = timeslice.month
    day   = timeslice.day
    
    if len(str(month)) < 2:
        month = '0'+str(month)

    
    #This is not the best way to do this, but it should at least work (for a while, anyway)
    versionList = ['09', '08', '07', '06', '05', '04', '03', '02', '01', '00']

    versionTest = 'notFound'
    for version in versionList:
        if versionTest == 'notFound':
            VDfile_directoryRemote = f'http://w3sweap.cfa.harvard.edu/pub/data/sci/sweap/spi/L2/spi_sf00/{year}/{month}/'
            VDfile_filename = f'psp_swp_spi_sf00_L2_8Dx32Ex8A_{year}{month}{day}_v{version}.cdf'
            
            
            if os.path.isfile(VDfile_filename):
                print("Version {version} exists")
                VDfile = VDfile_filename
                versionTest = 'found'
            else:
                try:
                    #print(f"Version {version} doesn't exist locally, searching online..")
                    VDfile = wget.download(VDfile_directoryRemote + VDfile_filename)
                    versionTest = 'found'
                    print(f'Grabbed version {version}')
                except:
                    continue
        elif versionTest == 'found':
            break
          

    cdf_VDfile = cdflib.CDF(VDfile)

    epoch           = cdf_VDfile['Epoch']
    theta           = cdf_VDfile['THETA']
    phi             = cdf_VDfile['PHI']
    energy          = cdf_VDfile['ENERGY']
    eflux           = cdf_VDfile['EFLUX']
    rotMat          = cdf_VDfile['ROTMAT_SC_INST']
    
    import datetime
    datetime_t0 = datetime.datetime(2000, 1, 1, 12, 0, 0)
    epoch = cdflib.cdfepoch.to_datetime(cdf_VDfile.varget('Epoch'))

    print('Desired timeslice:',timeslice)
    tSliceIndex  = bisect.bisect_left(epoch, timeslice)
    print('time Index:',tSliceIndex)
    print('Time of closest data point:',epoch[tSliceIndex])

    epochSlice  = epoch[tSliceIndex]
    thetaSlice  = theta[tSliceIndex,:]
    phiSlice    = phi[tSliceIndex,:]
    energySlice = energy[tSliceIndex,:]
    efluxSlice  = eflux[tSliceIndex,:]

    thetaReshaped = thetaSlice.reshape((8,32,8))
    phiReshaped = phiSlice.reshape((8,32,8))
    energyReshaped = energySlice.reshape((8,32,8))
    efluxReshaped = efluxSlice.reshape((8,32,8))

    mass_p = 0.010438870      #eV/c^2 where c = 299792 km/s
    charge_p = 1              #eV

    #Define VDF
    numberFlux = efluxReshaped/energyReshaped
    vdf = numberFlux*(mass_p**2)/((2E-5)*energyReshaped)

    #Convert to velocity units in each energy channel
    vel = np.sqrt(2*charge_p*energyReshaped/mass_p)

    vx = vel * np.cos(np.radians(phiReshaped)) * np.cos(np.radians(thetaReshaped))
    vy = vel * np.sin(np.radians(phiReshaped)) * np.cos(np.radians(thetaReshaped))
    vz = vel *                                   np.sin(np.radians(thetaReshaped))

    return vx, vy, vz, vdf, vel, thetaReshaped, phiReshaped


def fieldAlignedCoordinates(Bx, By, Bz):
    '''
    INPUTS:
         Bx, By, Bz = rank1 arrays of magnetic field measurements in instrument frame
         
    OUTPUT: parallel and perpendicular versor to B     
    '''

    Bmag = np.sqrt(Bx**2 + By**2 + Bz**2)

    # Define field-aligned vector
    Nx = Bx/Bmag
    Ny = By/Bmag
    Nz = Bz/Bmag

    # Make up some unit vector
    if np.isscalar(Nx):
        Rx = 0
        Ry = 1.
        Rz = 0
    else:
        Rx = np.zeros(Nx.len())
        Ry = np.ones(len(Nx))
        Rz = np.zeros(len(Nx))

    # Find some vector perpendicular to field NxR 
    TEMP_Px = ( Ny * Rz ) - ( Nz * Ry )  # P = NxR
    TEMP_Py = ( Nz * Rx ) - ( Nx * Rz )  # This is temporary in case we choose a vector R that is not unitary
    TEMP_Pz = ( Nx * Ry ) - ( Ny * Rx )


    Pmag = np.sqrt( TEMP_Px**2 + TEMP_Py**2 + TEMP_Pz**2 ) #Have to normalize, since previous definition does not imply unitarity, just orthogonality
  
    Px = TEMP_Px / Pmag # for R=(0,1,0), NxR = P ~= RTN_N
    Py = TEMP_Py / Pmag
    Pz = TEMP_Pz / Pmag


    Qx = ( Pz * Ny ) - ( Py * Nz )   # N x P
    Qy = ( Px * Nz ) - ( Pz * Nx )  
    Qz = ( Py * Nx ) - ( Px * Ny )  

    return(Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz)


# ###
# ### TRANSFORM VECTOR DATA INTO FIELD-ALIGNED COORDINATES
# ###

def rotateVectorIntoFieldAligned(Ax, Ay, Az, Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz):
    # For some Vector A in the SAME COORDINATE SYSTEM AS THE ORIGINAL B-FIELD VECTOR:

    An = (Ax * Nx) + (Ay * Ny) + (Az * Nz)  # A dot N = A_parallel
    Ap = (Ax * Px) + (Ay * Py) + (Az * Pz)  # A dot P = A_perp (~RTN_N (+/- depending on B), perpendicular to s/c y)
    Aq = (Ax * Qx) + (Ay * Qy) + (Az * Qz)  # 

    return(An, Ap, Aq)


def get_single_vdf(thetaSlice, phiSlice, energySlice, efluxSlice):
    
    """ get vdf and velocity for a given time/array index"""
    
    thetaReshaped = thetaSlice.reshape((8,32,8))
    phiReshaped = phiSlice.reshape((8,32,8))
    energyReshaped = energySlice.reshape((8,32,8))
    efluxReshaped = efluxSlice.reshape((8,32,8))
    
    mass_p = 0.010438870      #eV/c^2 where c = 299792 km/s
    charge_p = 1              #eV
    
    #Define VDF
    numberFlux = efluxReshaped/energyReshaped
    vdf = numberFlux*(mass_p**2)/((2E-5)*energyReshaped)
    
    #Convert to velocity units in each energy channel
    vel = np.sqrt(2*charge_p*energyReshaped/mass_p)
    
    vx = vel * np.cos(np.radians(phiReshaped)) * np.cos(np.radians(thetaReshaped))
    vy = vel * np.sin(np.radians(phiReshaped)) * np.cos(np.radians(thetaReshaped))
    vz = vel *                                   np.sin(np.radians(thetaReshaped))
    
    return vx, vy, vz, vdf, vel, thetaReshaped, phiReshaped, energyReshaped


def TtensAlongBVerniero(T_Tens, B_inst):
    """
    Compute Tpar e Tperp
    from the projection of the Temperature tensor
    in the B direction as in Jaye's notebook
    
    """
    T_XX = T_Tens[:,0]
    T_YY = T_Tens[:,1]
    T_ZZ = T_Tens[:,2]
    T_XY = T_Tens[:,3]
    T_XZ = T_Tens[:,4]
    T_YZ = T_Tens[:,5]
    
    T_YX = T_XY
    T_ZX = T_XZ
    T_ZY = T_YZ
    
    
    B_X = B_inst[:,0]
    B_Y = B_inst[:,1]
    B_Z = B_inst[:,2]
    B_mag_XYZ = np.sqrt(B_X**2 + B_Y**2 + B_Z**2)

    
    T_parallel=[]
    T_perpendicular=[]
    Anisotropy=[]
    for i in range(len(B_X)):
        Sum_1=B_X[i]*B_X[i]*T_XX[i]
        Sum_2=B_X[i]*B_Y[i]*T_XY[i]
        Sum_3=B_X[i]*B_Z[i]*T_XZ[i]
        Sum_4=B_Y[i]*B_X[i]*T_YX[i]
        Sum_5=B_Y[i]*B_Y[i]*T_YY[i]
        Sum_6=B_Y[i]*B_Z[i]*T_YZ[i]
        Sum_7=B_Z[i]*B_X[i]*T_ZX[i]
        Sum_8=B_Z[i]*B_Y[i]*T_ZY[i]
        Sum_9=B_Z[i]*B_Z[i]*T_ZZ[i]    
        T_para=((Sum_1+Sum_2+Sum_3+Sum_4+Sum_5+Sum_6+Sum_7+Sum_8+Sum_9)/(B_mag_XYZ[i])**2)
        Trace_Temp=(T_XX[i]+T_YY[i]+T_ZZ[i])
        T_perp=(Trace_Temp-T_para)/2.0
        T_parallel.append((Sum_1+Sum_2+Sum_3+Sum_4+Sum_5+Sum_6+Sum_7+Sum_8+Sum_9)/(B_mag_XYZ[i])**2)
        T_perpendicular.append(T_perp)
        Anisotropy.append(T_perp/T_para)
    
    return np.asarray(T_parallel), np.asarray(T_perpendicular), np.asarray(Anisotropy)




import subprocess
import os

def download_with_system_wget(url, download_dir):
    filename = url.split("/")[-1]
    output_path = os.path.join(download_dir, filename)

    cmd = [
        "wget",
        "-q",                 # quiet
        "--show-progress",
        "--tries=1",
        "--timeout=10",
        "--no-clobber",
        "--retry-connrefused",
        "-O", output_path,
        url,
    ]

    result = subprocess.run(cmd)

    # if wget failed → remove file
    if result.returncode != 0:
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"wget failed for {url}")

    #  check file size (very important)
    if not os.path.exists(output_path) or os.path.getsize(output_path) < 10000:
        # too small → likely HTML or corrupted
        if os.path.exists(output_path):
            os.remove(output_path)
        raise RuntimeError(f"Invalid file (too small): {filename}")

    return output_path



def find_or_download_spi_vdf(timeSliceStart, timeSliceEnd, download_dir="."):
    """
    Download PSP SPAN-I L2 SPI SF00 file (same day only)

    Returns local file path
    """

    # --- format date ---
    year_str  = f"{timeSliceStart.year}"
    month_str = f"{timeSliceStart.month:02d}"
    day_str   = f"{timeSliceStart.day:02d}"

    # --- base URL ---
    base_url = (
        f"https://sweap.cfa.harvard.edu/pub/data/sci/sweap/"
        f"spi/L2/spi_sf00/{year_str}/{month_str}/"
    )

    # --- try versions (newest first) ---
    version_list = [f"{i:02d}" for i in range(9, -1, -1)]

    for version in version_list:
        filename = f"psp_swp_spi_sf00_L2_8Dx32Ex8A_{year_str}{month_str}{day_str}_v{version}.cdf"
        local_path = os.path.join(download_dir, filename)

        # already exists
        if os.path.isfile(local_path):
            print(f"✔ Found locally: {filename}")
            return local_path

        url = base_url + filename
        print(f"Trying: {url}")

        try:
            downloaded_file = download_with_system_wget(url, download_dir)
            print(f"\n✔ Downloaded version v{version}")
            return downloaded_file

        except Exception as e:
            print(f"✘ Failed v{version}: {e}")

    raise FileNotFoundError(
        f"No valid version found for {year_str}-{month_str}-{day_str}"
    )


def get_theta_phi_energy_eflux_trange(timeSliceStart, timeSliceEnd, filename):
    
    """
    filename is the output of find_or_download_spi_vdf
    timeSliceStart, timeSliceEnd are the start and end data of interest
    for a given day
    """
    #open CDF file
    dat = cdflib.CDF(filename)
	
	#print variable names in CDF files
    print(dat._get_varnames())
    cdf_VDfile=dat
	
	#check variable formats in cdf file
    print(cdf_VDfile)
    epoch_ns        = cdf_VDfile['Epoch']
    epoch = cdflib.cdfepoch.to_datetime(epoch_ns)
	
	# select range through slices
    tSlice1Index = bisect.bisect_left(epoch, np.datetime64(timeSliceStart))
    tSlice2Index = bisect.bisect_left(epoch, np.datetime64(timeSliceEnd))
	
	
	
    theta           = cdf_VDfile['THETA'][tSlice1Index:tSlice2Index, :]
    phi             = cdf_VDfile['PHI'][tSlice1Index:tSlice2Index, :]
    energy          = cdf_VDfile['ENERGY'][tSlice1Index:tSlice2Index, :]
    eflux           = cdf_VDfile['EFLUX'][tSlice1Index:tSlice2Index, :]
	# rotMat          = cdf_VDfile['ROTMAT_SC_INST'][tSlice1Index:tSlice2Index, :]
    counts          = cdf_VDfile['DATA'][tSlice1Index:tSlice2Index, :]
	
    return theta, phi, energy, eflux	



