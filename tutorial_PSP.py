import matplotlib.pyplot as plt
import numpy as np
import pyspedas
from pytplot import tplot, store_data, get_data ,tlimit,xlim,ylim,tplot_options,options,split_vec
import pickle
from numpy.polynomial.hermite import hermgauss, Hermite

from PyUltra.insitu_PSP_hermite2D import * 
from PyUltra.insitu_PSP_VDFroutines import * 

#This directories mustbe created before executing this tutorial.
save_folder = 'testPSP/'
save_folder_vdfs = 'vdfFolder/'

# =============================================================================
# Hermite grid and order, used for all the vdfs
# =============================================================================

order = 50
n_roots = 51
radici, weights = hermgauss(n_roots)

grid_x, grid_y = np.meshgrid(radici, radici)

# =============================================================================
# Interval of interest
# =============================================================================

############ Extract B and T_tensor from L3 file#############################
trange=['2023-03-01/21:00:00','2023-03-01/21:21:00']

# Parse start and end time using datetime
start_dt = datetime.datetime.strptime(trange[0], '%Y-%m-%d/%H:%M:%S')
end_dt   = datetime.datetime.strptime(trange[1], '%Y-%m-%d/%H:%M:%S')

# Extract components
year  = start_dt.year
month = start_dt.month
day   = start_dt.day
hour1 = start_dt.hour
minute1 = start_dt.minute

hour2 = end_dt.hour
minute2 = end_dt.minute

# Optional: extract seconds too
second1 = start_dt.second
second2 = end_dt.second


timeSlice1  = datetime.datetime(year, month, day, hour1, minute1)
timeSlice2 = datetime.datetime(year, month, day, hour2, minute2)


prefix = 'psp_spi_'
datatype = 'spi_sf00_l3_mom'

# get L3 moments
spi_vars = pyspedas.psp.spi(trange=trange, datatype=datatype, level='l3',
                            time_clip=True, no_update=False)


# just o get the time
aus_dens = get_data('psp_spi_DENS', metadata=False)
tSpan = aus_dens.times

# temp tensor
T_tens = get_data('psp_spi_T_TENSOR_INST', metadata=False).y

# B in the instrument frame
B_inst = get_data('psp_spi_MAGF_INST', metadata=False).y

############ Get par and perp temperatures#############################
Tpar, Tperp, T_Anisotropy = TtensAlongBVerniero(T_tens, B_inst)

Ttrace = (Tpar+Tperp*2)/3

# proton mass in eV/c^2 where c = 299792 km/s
mass_p = 0.010438870

vth_perp = np.sqrt(Tperp/mass_p)
vth_par = np.sqrt(Tpar/mass_p)

# get vbulk
vBulk =  get_data('psp_spi_VEL_INST', metadata=False).y

############ vdf part#############################
fileVDF = find_or_download_spi_vdf(timeSlice1, timeSlice2, download_dir=save_folder_vdfs)

# get span data for the interval
theta, phi, energy, eflux = get_theta_phi_energy_eflux_trange(timeSlice1, timeSlice2, fileVDF)

# =============================================================================
# Loop to work on each vdf of the range
# =============================================================================

diz = {}

diz['tSpan'] = tSpan 
diz['gkl'] = []
diz['Mkp'] = []


loop_step = 1
print(theta.shape[0])

for idx in range(0, theta.shape[0], loop_step):

    print(idx)

    # get velocity grids and vdf
    vx, vy, vz, vdf, vel, thetaReshaped, phiReshaped, energyReshaped = get_single_vdf(theta[idx], phi[idx], energy[idx], eflux[idx])
    
    # center the grid
    vx_centered = vx - vBulk[idx, 0]
    vy_centered = vy - vBulk[idx, 1]
    vz_centered = vz - vBulk[idx, 2]

    # get the magnetic field at SPAN cadence
    Bx_SPAN, By_SPAN, Bz_SPAN = B_inst[idx, :]

    # get parallel and perp1 perp2 versor (courtesy of J. Verniero)
    (Nx, Ny, Nz, Px, Py, Pz, Qx, Qy, Qz) = fieldAlignedCoordinates(Bx_SPAN, By_SPAN, Bz_SPAN)

    # velocity grids in field aligned coordinates (courtesy of J. Verniero)
    
    (vn_aus, vp_aus, vq_aus) = rotateVectorIntoFieldAligned(vx_centered,
                                                                vy_centered,
                                                                vz_centered,
                                                                Nx, Ny, Nz,
                                                                Px, Py, Pz,
                                                                Qx, Qy, Qz)
    v_par = vn_aus
    # define single perp speed following Bowen+PRL 2020
    v_perp = np.sqrt(vp_aus**2+vq_aus**2)

    # gyrotropy assumption as in Bowen+PRL 2020
    v_perp_gyro, v_par_gyro, vdf_gyro = goToGyroSystem(v_perp, v_par, vdf, vth_perp[idx], vth_par[idx])


    # interp VDF onto the hermite quadrature grid
    vdf_on_quad_grid = interp_vdf_to_herm_grid(v_perp_gyro, v_par_gyro, vdf_gyro, radici)

    # compute hermite coefficients
    gkl = compute_gkl_fast(radici, radici, vdf_on_quad_grid, order, weights)
    
    # store in python dictionary

    diz['gkl'].append(gkl)

    ##########################################################################
    # Part to compute the Kauffman Paterson measure
    ##########################################################################
   
    # get volume element for integrals
    d3V = compute_d3V(vel, thetaReshaped, phiReshaped)
    
    # compute density

    n = np.sum(vdf*d3V) # density in 1/km**3
    
   # build maxwellian
    f_maxw = build_3d_Maxwellian(vdf, Ttrace[idx], n,  vx_centered, vy_centered, vz_centered)
   
   # compute KP
    M_KP = compute_kauffmann_paterson(vdf, f_maxw, d3V, n)    


    diz['Mkp'].append(M_KP)




string_save_diz = f'dict_{timeSlice1}__{timeSlice2}'.replace(" ", "_").replace("-","_").replace(":","_")

with open(save_folder+string_save_diz+'.pkl', 'wb') as file:
    pickle.dump(diz, file)






































