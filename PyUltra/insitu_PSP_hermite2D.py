import numpy as np
import matplotlib.pyplot as plt
from scipy.special import hermite
from math import factorial
from scipy.special import roots_hermite
from scipy.interpolate import griddata
import numpy as np
from numpy.polynomial.hermite import hermgauss, Hermite, hermval
from scipy.interpolate import NearestNDInterpolator, LinearNDInterpolator
import pandas as pd
from matplotlib.colors import LogNorm


def hermite_poly_2d_numpy(n, m, x, y):
    """ 
    Computes the matrix given by phi_n(x)*phi_m(y)
    
    phi_n(x) and phi_m(y) are the basis functions 
    of the Hermite trasform see Larosa+2025 ApJL
    """
    #Hermite polynomial of order n in x   
    Hn_x = Hermite([0]*n + [1])(x)    
    phi_x = Hn_x*np.exp(-0.5*x**2)/np.sqrt(2**n*np.sqrt(np.pi)*factorial(n))  
    
    # Hermite polynomial of order m in y
    Hm_y = Hermite([0]*m + [1])(y)    
    phi_y = Hm_y*np.exp(-0.5*y**2)/np.sqrt(2**m*np.sqrt(np.pi)*factorial(m))  
    
    return np.outer(phi_x, phi_y)



def compute_gkl(v_perp, v_par, input_signal, order, weights):
    """
    Computes the Hermite coefficients by using the Gauss-Hermite
    quadrature

    v_perp = roots of the Hermite polynial Hn with n = order+1 
    v_par = roots of the Hermite polynial Hn with n = order+1 

    v_par = v_perp since I use the same order in both directions for the 
    Hermite expansion.    

    input_signal is the vdf interpolated on the Hermite grid
    
    "order" is the desired order for the Hermite transform 
    in both directions

    weights are the weights of ther Hermite-Gauss quadrature 
         
    N.B
    radici and weights are obtained as follow:
    from numpy.polynomial.hermite import hermgauss 
    radici, weights = hermgauss(n_roots) 
    
    """

    coefficients = np.zeros((order, order))
    for n in range(order):
        for m in range(order):
            
            Phi_nm = hermite_poly_2d_numpy(n, m, v_perp, v_par)
            
            aus=0
            
            for j in range(len(weights)):
                for i in range(len(weights)):
                   aus+= weights[i]*weights[j]*input_signal[i,j]*Phi_nm[i,j]*np.exp(v_perp[i]**2+v_par[j]**2)
            coefficients[n, m] = aus
            
            
    return coefficients
       

def compute_gkl_phi(v_perp, v_par, input_signal, order, weights):
    """
    Same as compute_gkl but returns also phi_full which is needed
    to have the reconstructed VDF through recons_f_Hgrid 
    """

    # Compute the 2D Hermite transform
    coefficients = np.zeros((order, order))
    
    # multidimenional array to put all the hermite polynomials for each order 
    phi_full = np.zeros([order, order, weights.shape[0], weights.shape[0]])
    for n in range(order):
        for m in range(order):
            
            Phi_nm = hermite_poly_2d_numpy(n, m, v_perp, v_par)
            
            # store Phi_nm
            phi_full[n, m, :, :] = Phi_nm 

            aus=0
            for j in range(len(weights)):
                for i in range(len(weights)):
                   aus+= weights[i]*weights[j]*input_signal[i,j]*Phi_nm[i,j]*np.exp(v_perp[i]**2+v_par[j]**2)
            coefficients[n, m] = aus
            
            
    return coefficients, phi_full


def recons_f_Hgrid(gkl, phi_full):
    """
    compute the reconstructed vdf
    
    gkl is a H_order*H_order matrix that contains the hermite coefficients

    phi_full in a tensor of shape [H_order, H_order, radici, radici]

    the functions computes f(vperp, vpar) = sum gkl*phi_k(vperp)*phi_l(vpar)

    it is equivalent to 

    f_recons = np.zeros([100, 100])

    for i in range(weights.shape[0]):
        for j in range(weights.shape[0]):
            for n in range(order):
                for m in range(order):
                    f_recons[i, j] += gkl[n, m]*phi[n, m, i, j]

    """
    f_recons = np.tensordot(gkl, phi_full, axes=([0, 1], [0, 1]))


    return f_recons


def interp_vdf_to_herm_grid(v_perp_gyro, v_par_gyro, vdf_gyro, radici):
    
    """Interpolate vdf to hermite grid. The input is the output of goToGyroSystem
       
       v_perp_gyro and v_par_gyro are centered and normalized to the thermal speed

       v_perp_gyro, v_par_gyro and vdf_gyro previosly mask with mask=vdf>0

       radici are the roots of the Hermite polynomials of order N+1, where N is 
       the desired order of the transform. The roots are obtained from
       numpy.polynomial.hermite.hermgauss

    """
        
    grid_x, grid_y = np.meshgrid(radici, radici)
    
    # vdf values
    values = vdf_gyro

    # points on which the vdf was originally known
    points = np.array([v_perp_gyro, v_par_gyro])

    # linear interpolation of the vdf onto the new grid
    vdf_on_quad_grid = griddata(points.T, values, (grid_x, grid_y), method='linear')
    
    # 0. instead of nan. This gives no contribution to the gkl
    # once the hermite trasform is computed
    vdf_on_quad_grid[np.isnan(vdf_on_quad_grid)] = 0.

    return vdf_on_quad_grid



def goToGyroSystem(v_perp, v_par, vdf, vth_perp, vth_par):

    """
    Gives the input to interp_vdf_to_herm_grid

    this treatement of the VDF is based on Bowen+PRL 2022 
    
    v_perp and v_par are the original instrument grid
    centered by removing the bulk speed

    """
    # faltten the grids
    v_perp = v_perp.reshape(-1)
    v_par  = v_par.reshape(-1)
    vdf    = vdf.reshape(-1)
    
    # to mask empty bins
    mask = vdf>0
	
    v_perp = v_perp[mask]
    v_par  = v_par[mask]
    vdf    = vdf[mask]
	
	# normalize grids to thermal speeds
    v_perp_gyro = np.concatenate([-v_perp, v_perp])/vth_perp
    v_par_gyro = np.concatenate([v_par, v_par])/vth_par
    vdf_gyro = np.concatenate([vdf, vdf])
	
    return v_perp_gyro, v_par_gyro, vdf_gyro



def plot_vdf_FAC_noTS(vperp, vpar, vdf):

    x = vperp.reshape(-1)
    y = vpar.reshape(-1)
    z = vdf.reshape(-1)
    
    X = np.linspace(min(x), max(x))
    Y = np.linspace(min(y), max(y))        
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    
    interp = LinearNDInterpolator(list(zip(x, y)), z)

    Z = interp(X, Y)
    return X, Y, Z



def plot_vdf_FAC_noTS_MaxgridPoints(vperp, vpar, vdf, max_grid_point, choice):
    
    x = vperp.reshape(-1)
    y = vpar.reshape(-1)
    z = vdf.reshape(-1)
    
    X = np.linspace(min(x), max(x), max_grid_point)
    Y = np.linspace(min(y), max(y), max_grid_point)        
    X, Y = np.meshgrid(X, Y)  # 2D grid for interpolation
    
    if choice == "linear":
        interp = LinearNDInterpolator(list(zip(x, y)), z)
    elif choice == "nearest":
        interp = NearestNDInterpolator(list(zip(x, y)), z)
    else:
        print("Choice is linear or nearest")
        return None
    Z = interp(X, Y)
    return X, Y, Z




def plot_par_perp_H_spectrum(gkl, order, slope):

    """
    Plot par and perp Hermite spectrum
    only even modes in the perpendicular direction.

    gkl is the matrix of the Hermite coefficients
    
    order is the order of the Hermite Transform

    slope in the slope to plot for reference
    
    """
    
    # power in the order (0, 0) hermite mode.
    # used to normalize the spectra
    sp_0 = gkl[0, 0]**2
    
    # square of the Hermite coefficients
    sp_2d = gkl**2
   
    # Perpendicular spectrum
    sp_vs_m_perp = np.sum(sp_2d, axis=0)/order 

    # to avoid very low values. Due to the girotropy assumptions
    # the odd modes in the perp direction are close to zero.
    aus_perp = sp_vs_m_perp.copy()
    aus_perp[1::2]=np.nan
    
    # Parallel spectrum
    sp_vs_m_par = np.sum(sp_2d, axis = 1)/order # sommo i par

    
    fig, ax = plt.subplots(1)

    ax.loglog(np.arange(order)[1:], aus_perp[1:]/sp_0, c='b', label='Perp', marker='o')
    ax.loglog(np.arange(order)[1:], sp_vs_m_par[1:]/sp_0, c='m', label='Par')
    
    # slopes
    x = np.arange(order)[1:].astype(float)
    y = np.max(sp_vs_m_par/sp_0)*x**slope

    ax.plot(x, y, ls="--", c='cyan', label=str(slope))

    ax.set_ylabel(r"$P/P_0$", fontsize=15)
    ax.set_xlabel("Hermite Order", fontsize=15)
    ax.legend()
    plt.show()

    return ax




def concentric_shells_avg_of_matrix(size_x, size_y, array, bin_width):
    """
    Compute the average on concentric shells for the given matrix (array)
    neglecting the [0, 0] element

    The bin edges are chosen to average over the closest values to the radius corresponding 
    to each 

    The goal is to compute the unidirectional H spectrum as in Pezzi+2018

    Some points are left outside because their radius is greater than the highest
    H order that corresponds to size_x

    """ 
    # Generate coordinate grid
    x = np.arange(size_x)
    y = np.arange(size_y)
   
    # Create a meshgrid for x and y
    # with indexing="ij" X and Y have the same row, column convention of python 
    X, Y = np.meshgrid(x, y, indexing="ij")     
    
    # Compute the radius for each grid point
    radii = np.sqrt(X**2 + Y**2)
    
    max_radius = radii.max()  # Maximum radius in the grid
    num_bins = int(np.ceil(max_radius / bin_width))  # Calculate the number of bins
    bin_edges = np.arange(0.5, max_radius + bin_width, bin_width)  # Define bin edges
    bins = np.digitize(radii, bin_edges)-1   # Assign each radius to a bin (0-indexed)


    # Sum values in the array for each shell
    max_bin = bins.max()  # Largest radius bin
    shell_avg = np.zeros(max_bin + 1)
    
    count = 1
    for b in range(max_bin + 1):
        
        
        mask = bins == b

        shell_avg[b] = array[bins == b].mean()

        count+=1

    return bin_edges, bins, shell_avg[:size_x]


def HermitegramEnstrophy(gkl_stream, order):
    """
    Given gkl_stream = gkl(t, m, n)         

    computes for each time t the 1D spectrum and stores it in matr_1d_spec
    the enstrophy is computed at each t as the sum of the correspodning 1d spectrum neglecting 
    the m=0 term    

    See Pezzi 2018
    """
    matr_1d_spec = np.zeros([gkl_stream.shape[0], order])
    for it in range(gkl_stream.shape[0]):
        # compute shell avg
        _, _, shell_avg = concentric_shells_avg_of_matrix(50, 50, gkl_stream[it, :, :]**2, 1)
        
        # remember that this spectra does not contain g00
        # g00 is already discarded in concentric_shells_avg_of_matrix
        matr_1d_spec[it, :] = shell_avg/gkl_stream[it, 0, 0]**2
        

        # NB for better visualization matr_1d_spec must be plotted as follow
        # figure(); plt.pcolormesh(np.arange(matr_1d_spec.shape[0]), np.arange(order)[1::2], np.log10(matr_1d_spec[:, 1::2].T))
        # Here starting from 1 correspond to start from H order = 2 and than we move only along the
        # even orders

    enstrophy = np.sum(matr_1d_spec, axis = 1)
    
    return matr_1d_spec, enstrophy


def concentric_shells_sum_of_matrix(size_x, size_y, array, bin_width):
    """
    Compute SUM on concentric shells for the given matrix (array)
    neglecting the [0, 0] element

    The bin edges are chosen to average over the closest values to the radius corresponding 
    to each 

    The goal is to compute the unidirectional H spectrum as in Pezzi+2018

    Some points are left outside because their radius is greater than the highest
    H order that corresponds to size_x

    """ 
    # Generate coordinate grid
    x = np.arange(size_x)
    y = np.arange(size_y)
   
    # Create a meshgrid for x and y
    # with indexing="ij" X and Y have the same row, column convention of python 
    X, Y = np.meshgrid(x, y, indexing="ij")     
    
    # Compute the radius for each grid point
    radii = np.sqrt(X**2 + Y**2)
    
    max_radius = radii.max()  # Maximum radius in the grid
    num_bins = int(np.ceil(max_radius / bin_width))  # Calculate the number of bins
    bin_edges = np.arange(0.5, max_radius + bin_width, bin_width)  # Define bin edges
    bins = np.digitize(radii, bin_edges)-1   # Assign each radius to a bin (0-indexed)


    # Sum values in the array for each shell
    max_bin = bins.max()  # Largest radius bin
    shell_sum = np.zeros(max_bin + 1)
    
    #print("Start averaging current matrix")
    count = 1
    for b in range(max_bin + 1):
        
        
        mask = bins == b
     #   print(count, "# averaged points: ", np.sum(mask))


        shell_sum[b] = array[bins == b].sum()

        count+=1
   
    #print("End averaging current matrix")

    return bin_edges, bins, shell_sum[:size_x]


def HermitegramEnstrophySum(gkl_stream, order):
    """
    Given gkl_stream = gkl(t, m, n)         

    computes for each t the 1D spectrum as the sum on concentric shells and stores it in matr_1d_spec
    the enstrophy is computed at each t as the sum of the correspodning 1d spectrum neglecting 
    the m=0 term    

    See Pezzi 2018
    """
    matr_1d_spec = np.zeros([gkl_stream.shape[0], order])
    for it in range(gkl_stream.shape[0]):
        # compute shell avg
        _, _, shell_sum = concentric_shells_sum_of_matrix(order, order, gkl_stream[it, :, :]**2, 1)
        
        # remember that this spectra does not contain g00
        # g00 is already discarded in concentric_shells_avg_of_matrix
        matr_1d_spec[it, :] = shell_sum/gkl_stream[it, 0, 0]**2
        

        # NB for better visualization matr_1d_spec must be plotted as follow
        # figure(); plt.pcolormesh(np.arange(matr_1d_spec.shape[0]), np.arange(order)[1::2], np.log10(matr_1d_spec[:, 1::2].T))
        # Here starting from 1 correspond to start from H order = 2 and than we move only along the
        # even orders

    enstrophy = np.sum(matr_1d_spec, axis = 1)
    
    return matr_1d_spec, enstrophy


def HermitegramEnstrophySum_noNorm(gkl_stream, order):
    """
    same as HermitegramEnstrophySum but the spectra are not normalized 
    to gkl[0, 0]
    This is useful for a posteriori normalization

    """
    matr_1d_spec = np.zeros([gkl_stream.shape[0], order])
    for it in range(gkl_stream.shape[0]):
        # compute shell avg
        _, _, shell_sum = concentric_shells_sum_of_matrix(order, order, gkl_stream[it, :, :]**2, 1)
        
        # remember that this spectra does not contain g00
        # g00 is already discarded in concentric_shells_avg_of_matrix
        matr_1d_spec[it, :] = shell_sum#/gkl_stream[it, 0, 0]**2
        

        # NB for better visualization matr_1d_spec must be plotted as follow
        # figure(); plt.pcolormesh(np.arange(matr_1d_spec.shape[0]), np.arange(order)[1::2], np.log10(matr_1d_spec[:, 1::2].T))
        # Here starting from 1 correspond to start from H order = 2 and than we move only along the
        # even orders

    enstrophy = np.sum(matr_1d_spec, axis = 1)
    
    return matr_1d_spec, enstrophy


def HermitegramEnstrophySumAvgGklBeforeShellSum(gkl_stream, order):
    """
    This was only to check that averaging the matrices before computing the 1D spectrum
    or computing for each matrix the 1D spectrum and then averagin does not change the result
    """

    matr_1d_spec = np.zeros([gkl_stream.shape[0], order])
     
    gkl_mean = np.mean(gkl_stream**2, axis=0)/np.mean(gkl_stream[:, 0, 0]**2, axis=0)
 
    _, _, shell_sum = concentric_shells_sum_of_matrix(order, order, gkl_mean, 1)
    
    return shell_sum 


###############################################################
# Faster routines thanks to ChatGPT
###############################################################


def hermite_basis_1d(x, order):
    """
    Compute 1D Hermite basis functions φ_n(x) for n = 0,...,order-1
    Returns array of shape (order, len(x))
    """
    N = len(x)
    phi = np.zeros((order, N))

    exp_term = np.exp(-0.5 * x**2)

    for n in range(order):
        coeffs = [0]*n + [1]  # H_n
        Hn = hermval(x, coeffs)

        norm = np.sqrt(2**n * np.sqrt(np.pi) * factorial(n))
        phi[n] = Hn * exp_term / norm

    return phi


def hermite_poly_2d_fast(phi_x, phi_y):
    """
    Build full 2D Hermite basis from 1D bases
    Returns Phi[n, m, i, j] = φ_n(x_i) * φ_m(y_j)
    """
    return np.einsum('ni,mj->nmij', phi_x, phi_y)


def compute_gkl_fast(v_perp, v_par, input_signal, order, weights):
    """
    Fast computation of Hermite coefficients using full vectorization
    """

    # Precompute 1D Hermite bases
    phi_x = hermite_basis_1d(v_perp, order)
    phi_y = hermite_basis_1d(v_par, order)

    # Build 2D basis
    Phi = hermite_poly_2d_fast(phi_x, phi_y)

    # Precompute static terms
    W = np.outer(weights, weights)
    exp_term = np.exp(v_perp[:, None]**2 + v_par[None, :]**2)
    base = W * input_signal * exp_term

    # Final contraction (fully vectorized)
    coefficients = np.einsum('ij,nmij->nm', base, Phi)

    return coefficients

##############################################################################
############################## Non hermite stuff #############################


def compute_increments(tau, B):
    """
    Calculate increments, optimized for speed and memory.

    Args:
        tau (int): Time lag.
        B np.ndarray: Input field.

    Returns:
        increments
    """
    # Only keep what you need from df
    increments = np.linalg.norm(B[:-tau]-B[tau:], axis=1)

    return increments

def compute_increments_df(tau, B):
    """
    B must be a pandas dataframe. 
    Calculate increments, optimized for speed and memory.

    Args:
        tau (int): Time lag.
        B dataframe: Input field.

    Returns:
        increments
    """
    # Only keep what you need from df
    dB                      = (B.iloc[:-tau].values - B.iloc[tau:].values)
    dB_shape                = B.shape
    dB_filled               = pd.DataFrame(np.nan, index=B.index, columns=B.columns)
    dB_filled.iloc[:-tau,:] = dB
    dB                      = dB_filled#.iloc[tau:,:]

    return dB


def compute_PVI(step, av_window, B):
    """
    Compute the Partial Variance of Increments (PVI), see Greco+2008
    This function was adapted from the corresponding routine in 
    MHDTurbPy see https://github.com/nsioulas/MHDTurbPy
    
    step is an integer, step*dt = time scale of the increment
    where dt is the cadence of the measurements
    
    av_window is the scale over which the denominator 
    of\ is averaged

    B is defined as follow
    B = pd.DataFrame(b_rtn, index = timeB, columns=["Br", "Bt", "Bn"])
    
    timeB is the vector of the timestamps
    b_rtn are the magnetic field components measurements with respect to time
    b_rtn.shape = (timeB.shape[0], 3)
    """

    keys    = list(B.keys())
    dB = compute_increments_df(step, B)
    
    B['DBtotal']         = np.sqrt(sum((dB[key])**2 for key in keys))
    B['DBtotal_squared'] = B['DBtotal']**2
    
    denominator = np.sqrt(B['DBtotal_squared'].rolling(av_window, center=True).mean())

    # add PVI values to the dataframe
    B[f'PVI_{str(step)}'] = B['DBtotal'] / denominator
    
    # drop intermediate products from the dataframe
    B.drop(columns=['DBtotal_squared', 'DBtotal'], inplace=True)

    return B




def build_3d_Maxwellian(vdf, T_trace, n, vx_centered, vy_centered, vz_centered):

    """
    Buuild a 3d Maxwellian on the grid given by
    vx_centered, vy_centered, vz_centered

    T_trace is in eV
    n in km**-3
    velocities are in km/s
    """

    mass_p = 0.010438870      #eV/c^2 where c = 299792 km/s
    
    const_maxw = (mass_p/(2*np.pi*T_trace))**1.5 # T_trace is in eV so kb not needed

    const_exp = mass_p/(2*T_trace)

    f_maxw = n*const_maxw*np.exp(-const_exp*(vx_centered*vx_centered+vy_centered*vy_centered+vz_centered*vz_centered))

    return f_maxw



def compute_kauffmann_paterson(vdf, f_maxw, d3V, n):
    
    """
    Compute the kauffmann Paterson measure which
    quantify how far the entropy of the velocity distribution
    function under consideration (vdf) is with respect to
    a Maxwellian with the same temperature and densityi (f_maxw).
    See R. L. Kaufmann & W. R. Paterson 2009 and Liang et al. 2020
    
    For a discussion on taking the logarithm of a dimensional 
    quantity see the above references. This is safe of the entropy
    density differences.

    d3V is the volume element defined on the same grid on which both 
    f_maxw and vdf are defined

    Regarding the units, any units of the of vdf and f_maxw work for this
    function as long as n = \int vdf d3V . The normalization by n cancel out
    units and M_KP is left adimensional.


    """

    # n = np.sum(vdf*d3V)   density in 1/km**3
    
    M_KP = (-np.nansum(f_maxw*np.log(f_maxw)*d3V)+np.nansum(vdf*np.log(vdf)*d3V))/(1.5*n)

    return M_KP



def LET(v_rtn, b_rtn, n_step, dt, v_mean):
    
    """
    see doi: 10.3389/fphy.2019.00108
    
    v_rtn and b_rtn are the velocity and magnetic field vectors 
    as pandas dataframes in velocity units (e.g. 
    v_df = pd.DataFrame(vBulk, index = time, columns=["Vr", "Vt", "Vn"],
    where vBulk.shape = (time.shape[0], 3))

    
    n_step is an integer, n_step*dt = time scale of the increment
    where dt is the cadence of the measurements
    
    v_mean is the average flow speed
    v_mean = np.mean(np.linalg.norm(vBulk, axis=1))
    
    The longitudinal direction is assumed to be along the radial

    """
    db = compute_increments_df(n_step, b_rtn).to_numpy()
    dv = compute_increments_df(n_step, v_rtn).to_numpy()
    
    dv_par = dv[:, 0]
    db_par = db[:, 0]
    
    tau = n_step*dt

    ev = np.linalg.norm(dv, axis=1)**2*dv_par
    eb = np.linalg.norm(db, axis=1)**2*dv_par
    ec = -2*np.sum(db*dv, axis=1)*db_par

    LET = (ev+eb+ec)/(4*tau*v_mean/3)
    
    LET_cross = ec/(4*tau*v_mean/3)
    LET_en = (ev+eb)/(4*tau*v_mean/3)

    return LET, LET_en, LET_cross


