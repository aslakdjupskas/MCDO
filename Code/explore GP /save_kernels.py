import numpy as np
import matplotlib.pyplot as plt

import matplotlib as mpl
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from scipy.special import kv, gamma

# Use LaTeX-like font (Computer Modern)
mpl.rcParams['text.usetex'] = False  # Don't use full LaTeX
mpl.rcParams['mathtext.fontset'] = 'cm'  # Use Computer Modern for math
mpl.rcParams['font.family'] = 'STIXGeneral'  # Close match to LaTeX text
mpl.rcParams['font.size'] = 20
mpl.rcParams['axes.titlesize'] = 20
mpl.rcParams['axes.labelsize'] = 16
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16
mpl.rcParams['legend.fontsize'] = 14
mpl.rcParams['figure.titlesize'] = 16
mpl.rcParams['axes.labelweight'] = 'bold'
mpl.rcParams['axes.titleweight'] = 'bold'

import jax.numpy as jnp


def rbf_kernel(X1, X2, length_scale, sigma):
    """Exponentiated quadratic (RBF) kernel."""
    dist_sq = jnp.sum((X1[:, None] - X2)**2, axis=-1)
    return sigma**2 * jnp.exp(-0.5 * dist_sq / length_scale**2)


def matern_kernel(r, nu, ell=1.0, sigma=1.0):
    """
    Compute the Matérn kernel.

    Parameters:
        r (float or np.ndarray): The distance(s) between input points.
        nu (float): Smoothness parameter.
        ell (float): Length scale parameter (default is 1.0).
        sigma (float): Signal variance (default is 1.0).
    
    Returns:
        np.ndarray: Matérn kernel values.
    """
    r = np.abs(r) / ell
    if nu == 0.5:
        # Exponential kernel
        return sigma**2 * np.exp(-r)
    
    elif nu == np.inf:
        # Squared exponential kernel
        return sigma**2 * np.exp(-r**2 / 2)
    
    else:
        factor = (2**(1 - nu)) / gamma(nu)
        scaled_r = np.sqrt(2 * nu) * r
        matern_value = factor * (scaled_r**nu) * kv(nu, scaled_r)
        matern_value[r == 0] = sigma**2  # Avoid NaN at r = 0
        return sigma**2 * matern_value
    
def gradient_matern_kernel(r, nu, ell=1.0, sigma=1.0):
    """
    Compute the gradient of the Matérn kernel with respect to the length scale.

    Parameters:
        r (float or np.ndarray): The distance(s) between input points.
        nu (float): Smoothness parameter.
        ell (float): Length scale parameter (default is 1.0).
        sigma (float): Signal variance (default is 1.0).
    
    Returns:
        np.ndarray: Gradient of the Matérn kernel with respect to the length scale.
    """
    r = np.abs(r) / ell
    if nu == 0.5:
        return sigma**2 * r * np.exp(-r)
    
    elif nu == np.inf:
        return sigma**2 * r * np.exp(-r**2 / 2)
    
    else:
        factor = (2**(1 - nu)) / gamma(nu)
        scaled_r = np.sqrt(2 * nu) * r
        gradient_value = factor * (scaled_r**nu) * (kv(nu - 1, scaled_r) - kv(nu, scaled_r))
        return sigma**2 * gradient_value

def plot_matern_kernel(nu, ell=1.0, sigma=1.0):
    """
    Plot the Matérn kernel for a given smoothness parameter.

    Parameters:
        nu (float): Smoothness parameter.
        ell (float): Length scale parameter (default is 1.0).
        sigma (float): Signal variance (default is 1.0).
    """
    r = np.linspace(0, 3 * ell, 1000)
    kernel = matern_kernel(r, nu, ell, sigma)
    plt.plot(r, kernel, label=f"Matérn ($\\nu = {nu}$)")
    plt.xlabel("$r$")
    plt.ylabel("$k(r)$")
    plt.legend()
    plt.show()

def plot_matern_kernels():
    """
    Plot the Matérn kernel for different smoothness parameters.
    """
    for nu in [0.5, 1.5, 2.5]:
        plot_matern_kernel(nu)
#plot_matern_kernels()

def periodic_kernel(x1, x2, ell=1.0, p=1.0, sigma=1.0):
    """
    Compute the periodic kernel.

    Parameters:
        x1 (float or np.ndarray): The first input point(s).
        x2 (float or np.ndarray): The second input point(s).
        ell (float): Length scale parameter (default is 1.0).
        p (float): Period parameter (default is 1.0).
        sigma (float): Signal variance (default is 1.0).
    
    Returns:
        np.ndarray: Periodic kernel values.
    """
    r = np.abs(x1 - x2) / ell
    return sigma**2 * np.exp(-2 * np.sin(np.pi * r / p)**2)

def gradient_periodic_kernel(x1, x2, ell=1.0, p=1.0, sigma=1.0):
    """
    Compute the gradient of the periodic kernel with respect to the length scale.

    Parameters:
        x1 (float or np.ndarray): The first input point(s).
        x2 (float or np.ndarray): The second input point(s).
        ell (float): Length scale parameter (default is 1.0).
        p (float): Period parameter (default is 1.0).
        sigma (float): Signal variance (default is 1.0).
    
    Returns:
        np.ndarray: Gradient of the periodic kernel with respect to the length scale.
    """
    r = np.abs(x1 - x2) / ell
    return -2 * sigma**2 * np.exp(-2 * np.sin(np.pi * r / p)**2) * np.sin(2 * np.pi * r / p) * np.pi / p

x = np.linspace(-5, 5, 1000)
X1, X2 = np.meshgrid(x, x)
# Add periodic kernel
K_p = periodic_kernel(X1, X2, ell=2.0, p=2.0)

# parameters
w_m = 0.5; w_p = 0.5; 
ell_m = 1.0; sigma_m = 1.0; nu_m = 2.5
ell_p = 2.0; p_p = 2.0; sigma_p = 1.0
# K = w_m * matern_kernel(np.abs(X1 - X2), ell=ell_m, sigma=sigma_m, nu=nu_m) + w_p * periodic_kernel(X1, X2, ell=ell_p, p=p_p, sigma=sigma_p)

plt.figure(figsize=(6, 5))
plt.imshow(K_p, cmap="jet", extent=(x[0], x[-1], x[-1], x[0]))
plt.colorbar()
plt.xlabel("$x_i$")
plt.ylabel("$x_j$")
# plt.title("Kernel matrix $K$")
# plt.show()
plt.savefig("Periodic_kernel.pdf")


# Define the RBF kernel function
def rbf_kernel(x, y, length_scale=1.0):
    sqdist = np.subtract.outer(x, y)**2
    return np.exp(-0.5 * sqdist / length_scale**2)

# Create input points
x = np.linspace(-5, 5, 1000)

# Compute the kernel matrix
K = rbf_kernel(x, x, length_scale=1.0)

# Plot the kernel matrix
plt.figure(figsize=(6, 5))
plt.imshow(K, cmap="jet", extent=(x[0], x[-1], x[-1], x[0]))
plt.colorbar()
plt.xlabel("$x_i$")
plt.ylabel("$x_j$")
# plt.title("RBF Kernel Matrix")
plt.savefig("RBF_kernel.pdf")
plt.show()

