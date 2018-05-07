import numpy as np
import random
import matplotlib.pyplot as plt


######### Non Conditional Simulation Functions ############
def band_coef_gen(a, l):
    """
    Generates band's coefficients.

    Arguments:
        l : float
            Total length of bands.
        a : numpy.array
            range of spherical covariance.
    """
    t = np.arange(-l/2,l/2,a)
    f = np.random.choice([-1, 1],len(t) )
    a0 = f*2*(3**0.5)/a

    return np.column_stack((a0,t))



def band_eval(t,band_coef,a):
    """
    Evaluates bands at time t.

    Arguments:
        t : numpy.array
            time.
        band_coef : numpy.array
            coefficients of bands.
        a : numpy.array
            range of spherical covariance.
    """

    index_t = np.searchsorted(band_coef[:,1],t) -1
    #index_t = sum(band_coef[:, 1]<=t)-1
    t = t - band_coef[index_t,1]-a/2
    f = band_coef[index_t,0]*t*(np.absolute(t)<=(a/2))
    return f


def line_eq(x,m,a):
    """
    Line equation.

    Arguments:
        x : numpy.array
            x coordinate.
        m : numpy.array
            slope of band.
        a : numpy.array
            y axis-bands intersection.
    """
    y = m*x + a
    return y

def random_bands(xf, yf, nbands):
    """
    Generates random bands.

    Arguments:
        xf : float
            maximum x coordinate for target grid.
        yf : float
            maximum y coordinate for target grid.
        nbands : integer
            number of bands.
    """
    x0 = np.random.uniform(0,xf, nbands)
    y0 = np.random.uniform(0, yf, nbands)

    theta = np.random.uniform(0,np.pi, nbands)

    x1 = x0 + np.sin(theta)
    y1 = y0 + np.cos(theta)

    m = (y1 -y0)/(x1-x0)
    a = y0 - m*x0

    mn = -1/m

    bands = np.column_stack((x0,y0,x1,y1,theta,m,a,mn))

    return bands



def band_target_proj(x0,y0,m,a,mn):
    """
    Projection of target point on band.

    Arguments:
        x0 : numpy.array
            x coordinate of targets.
        y0 : numpy.array
            y coordinate of targets.
        m : numpy.array
            slope of band.
        a : numpy.array
            y axis-bands intersection.
        mn : integer
            normal of bands.
    """
    nband = m.shape[0]
    npoint = x0.shape[0]

    x0 = np.transpose(np.tile(x0, nband))
    y0 = np.transpose(np.tile(y0, nband))

    m = m.repeat(npoint)
    a = a.repeat(npoint)
    mn = mn.repeat(npoint)

    an = y0 - mn * x0
    x= (a-an) / (mn-m)
    y = mn*x + an

    return x, y



def intcoord_to_time(bx0,by0,px0,py0, npoint):
    """
    Transforms targets projection on band coordinates to time.

    Arguments:
        bx0 : numpy.array
            x coordinate of bands.
        by0 : numpy.array
            y coordinate of bands.
        py0 : numpy.array
            x coordinate of targets.
        py0 : numpy.array
            y coordinate of targets.
        npoint : integer
            number of target points.
    """

    nband = bx0.shape[0]
    bx0 = bx0.repeat(npoint)
    by0 = by0.repeat(npoint)

    sign = -1*(bx0>=px0) + (bx0<px0)

    t = sign*((bx0-px0)**2+(by0-py0)**2)**0.5



    return np.transpose(np.reshape(t,(nband, npoint)))



def non_cond_sim(nbands, xf, yf, nx, a, l, nsim, scatter):  #Non-conditional simulations
    """
    Turning Bands 2D Un-conditional Simulation.

    Arguments:
        nbands : integer
            number of bands.
        xf : float
            maximum x coordinate for target grid.
        yf : float
            maximum y coordinate for target grid.
        nx : integer
            number of nodes of the x-axis
        a : float
            range of spherical covariance.
        l : float
            Total length of bands.
        nsim : integer
            Number of simulation to be performed.
        scatter: numpy.ndarray
            Optional. User target coordinates.
    """

    # Define target locations
    if 'numpy.ndarray' in str(type(scatter)):
        # Read scattered locations
        d1 = scatter
    else:
        # Create Grid
        dx = xf / nx
        dy = yf / nx
        d1 = np.mgrid[0:(xf + dx):dx, 0:(yf + dy):dy]
        d1 = np.append(d1[0].reshape(-1, 1), d1[1].reshape(-1, 1), axis=1)

    npoint = d1.shape[0]
    y_star = np.zeros((npoint, nsim))

    for j in range(0, nsim):

        # Random Direction: ec. line and normal

        bands = random_bands(xf,yf,nbands)

        # Points Projection in all bands, transform to time and evaluate stochastic process.

        target_line_proj = band_target_proj(d1[:, 0], d1[:, 1], bands[:, 5], bands[:, 6], bands[:, 7])
        bands_eval_time = intcoord_to_time(bands[:, 0], bands[:, 1], target_line_proj[0], target_line_proj[1], npoint)

        y_i = np.zeros((npoint, nbands))

        for i in range(0, nbands):

            bands_coef = band_coef_gen(a, l)
            y_i[:, i] = band_eval(bands_eval_time[:, i], bands_coef, a)


        ### Sum  all stocastic process and calculate y*
        y = y_i.sum(axis=1)
        y = y / nbands**0.5

        y_star[:,j] = y

    simu = np.column_stack((d1,y_star))
    return simu

######### Sampling of Simulation #############

def sampling(reality,n_points):
    """
    Random sampling.

    Arguments:
        reality : numpy.ndarray
            (x , y, Y(x,y) ) x n array.
        n_points : integer
            number of samples required.
    """
    samp_sel = np.array(random.sample(range(np.shape(reality)[0]), n_points))
    data = reality[samp_sel, :]

    return data


######### Simple Kriging Functions ############

def dist(data_coord,target_coord):
    """
    Calculates all distances between two 2D matrices.

    Arguments:
        data_coord : numpy.ndarray
            (x ,y) x nsamples array.
        target_coord : numpy.ndarray
            (x ,y) x ntarget array.
    """
    dx = np.subtract.outer(data_coord[:,0],target_coord[:,0])
    dy = np.subtract.outer(data_coord[:,1],target_coord[:,1])
    d = np.sqrt(np.power(dx,2)+np.power(dy,2))

    return d


def cova(h,a):
    """
    Spherical Covariance.

    Arguments:
        h : numpy.ndarray
            (h) x ndistances array. Distances between points.
        a : float
            Range of spherical covariance.
    """
    C = (1 - 1.5*(h/a)+0.5*np.power(h/a,3))*(h<=a)+ 0.
    return C

def skrig(samples,target,a):
    """
    2D Simple Kriging.

    Arguments:
        samples : numpy.ndarray
            (x , y, Y(x,y) ) x nsamples array. Samples to condition with.
        target : numpy.ndarray
            (x , y) x ntargets array. Target coordinates.
        a : float
            Range of spherical covariance.
    """

    target_coord = target[:,[0,1]]

    data_values = samples[:, 2]
    #data_values = data_values.reshape(1,np.shape(data)[0])

    data_coord = samples[:, [0, 1]]

    C = dist(data_coord, data_coord)  # Left hand distances matrix
    C = cova(C, a)  # Left hand Covariance Matrix

    T = dist(data_coord, target_coord)  # Right hand distance vector
    T = cova(T, a)  # Right hand covariance vector

    #C_inv = np.linalg.cholesky(C)  # Cholesky Decomposition
    #C_inv = np.linalg.inv(C_inv)
    #C_inv = np.matmul(C_inv, np.transpose(C_inv))  # Multiplication of Lower and Upper Cholesky matrix to calculate inv(C)
    C_inv = np.linalg.inv(C)

    Lambdas = np.matmul(C_inv, T)
    y_sk = np.matmul(data_values, Lambdas)

    y_sk = np.column_stack((target_coord, y_sk))

    return Lambdas, y_sk


######### Conditional Simulation ############
def cond_sim(samples, y_sk, Lambdas, a, l, nbands, nsim):
    """
    Turning Bands 2D Conditional Simulation.

    Arguments:
        samples : numpy.ndarray
            (x , y, Y(x,y) ) x nsamples array. Samples to condition with.
        y_sk : numpy.ndarray
            (x , y, Y(x,y) ) x ntargets array. Simple Kriging estimates on target.
        Lambdas : numpy.ndarray
            nsamples x ntargets array. Samples weights for each target.
        a : float
            Range of spherical covariance.
        l : string
            Total length of bands.
        nsim : integer
            Number of simulation to be performed.
    """

    xf = y_sk[:,0].max()
    yf = y_sk[:,1].max()

    n_target = y_sk.shape[0]
    n_data = samples.shape[0]

    reality_coords = y_sk[:, [0, 1]]
    samples_coords = samples[:, [0, 1]]


    nc_sim = np.concatenate((reality_coords, samples_coords))  ## Join grid + samples location
    nc_sim = non_cond_sim(nbands, xf = xf, yf = yf, nx = 0, a = a, l = l, nsim = nsim, scatter = nc_sim)  ## Non conditional simulation on grid and points

    nc_sim_target = nc_sim[range(0, n_target), :]   ## Divide the output into grid and data points
    nc_sim_data = nc_sim[range(n_target, n_target + n_data), :]

    y_cs = y_sk[:,2] + nc_sim_target[:,2] - np.matmul(np.transpose(Lambdas), nc_sim_data[:,2])  ## y_cs(x) = y_sk(x) + s(x) - s_sk(x)

    y_cs = np.column_stack((reality_coords,y_cs))

    return y_cs


######### Plot ############
def plot_3subplots(V1, V2, V3, nx, title1, title2, title3):

    """
    Shows 3 mesh subplots of grids in vectorial format.

    Arguments:
        V1 : numpy.ndarray
            (x , y, Y(x,y) ) x npoints array.
        V2 : numpy.ndarray
            (x , y, Y(x,y) ) x npoints array.
        V2 : numpy.ndarray
            (x , y, Y(x,y) ) x npoints array.
        nx : integer
            Number of nodes of the x-axis.
        title1 : string
            Title for V1 plot.
        title2 : string
            Title for V2 plot.
        title3 : string
            Title for V3 plot.
    """

    #Vector to matrix form
    V1x = np.transpose(np.reshape(V1[:, 0], (int(nx) + 1, int(nx) + 1)))
    V1y = np.transpose(np.reshape(V1[:, 1], (int(nx) + 1, int(nx) + 1)))
    V1z = np.transpose(np.reshape(V1[:, 2], (int(nx) + 1, int(nx) + 1)))

    V2x = np.transpose(np.reshape(V2[:, 0], (int(nx) + 1, int(nx) + 1)))
    V2y = np.transpose(np.reshape(V2[:, 1], (int(nx) + 1, int(nx) + 1)))
    V2z = np.transpose(np.reshape(V2[:, 2], (int(nx) + 1, int(nx) + 1)))

    V3x = np.transpose(np.reshape(V3[:, 0], (int(nx) + 1, int(nx) + 1)))
    V3y = np.transpose(np.reshape(V3[:, 1], (int(nx) + 1, int(nx) + 1)))
    V3z = np.transpose(np.reshape(V3[:, 2], (int(nx) + 1, int(nx) + 1)))

    f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15,15))

    ax1.set_xlabel('x [m]')
    ax1.set_ylabel('y [m]')
    ax2.set_xlabel('x [m]')
    ax2.set_ylabel('y [m]')
    ax3.set_xlabel('x [m]')
    ax3.set_ylabel('y [m]')

    ax1.set_title(title1)
    ax2.set_title(title2)
    ax3.set_title(title3)

    ax1.set(adjustable='box-forced', aspect='equal')
    ax2.set(adjustable='box-forced', aspect='equal')
    ax3.set(adjustable='box-forced', aspect='equal')

    ax1.pcolormesh(V1x, V1y, V1z, vmin = -3, vmax = 3)
    ax2.pcolormesh(V2x, V2y, V2z, vmin = -3, vmax = 3)
    f3 = ax3.pcolormesh(V3x, V3y, V3z, vmin = -3, vmax = 3)

    f.subplots_adjust(right=0.8)
    cbar_ax = f.add_axes([0.85, 0.4, 0.01, 0.20])
    f.colorbar(f3, cax = cbar_ax )