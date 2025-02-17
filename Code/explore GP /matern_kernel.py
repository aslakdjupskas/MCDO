import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns
from scipy.special import kv, gamma

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
    
import numpy as np
from scipy.special import gamma, kv

def grad_matern_kernel_ell(r, nu, ell=1.0, sigma=1.0):
    """
    Compute the gradient of the Matérn kernel with respect to the length scale.

    Parameters:
        r (float or np.ndarray): The distance(s) between input points.
        nu (float): Smoothness parameter.
        ell (float): Length scale parameter (default is 1.0).
        sigma (float): Signal variance (default is 1.0).
    
    Returns:
        np.ndarray or float: Gradient of the Matérn kernel with respect to the length scale.
    """
    if ell <= 0:
        raise ValueError("The length scale 'ell' must be positive.")
    
    r = np.asarray(r)
    scaled_r = np.abs(r) / ell
    
    if nu == 0.5:
        # Exponential kernel
        kernel = sigma**2 * np.exp(-scaled_r)
        gradient = kernel * (scaled_r / ell)
    
    elif nu == np.inf:
        # Squared exponential kernel
        kernel = sigma**2 * np.exp(-scaled_r**2 / 2)
        gradient = kernel * (scaled_r**2 / ell)
    
    else:
        # General Matérn kernel
        factor = (2**(1 - nu)) / gamma(nu)
        matern_value = factor * ((np.sqrt(2 * nu) * scaled_r)**nu) * kv(nu, np.sqrt(2 * nu) * scaled_r)
        matern_value[scaled_r == 0] = sigma**2  # Avoid NaN for r = 0
        kernel = sigma**2 * matern_value
        gradient = kernel * (nu / ell - scaled_r / ell)
    
    return gradient

    
# Function to calculate the gradient of the Matérn kernel with respect to the signal variance
def grad_matern_kernel_sigma(r, nu, ell=1.0, sigma=1.0):
    """
    Compute the gradient of the Matérn kernel with respect to the signal variance.

    Parameters:
        r (float or np.ndarray): The distance(s) between input points.
        nu (float): Smoothness parameter.
        ell (float): Length scale parameter (default is 1.0).
        sigma (float): Signal variance (default is 1.0).
    
    Returns:
        np.ndarray: Gradient of the Matérn kernel with respect to the signal variance.
    """
    r = np.abs(r) / ell
    if nu == 0.5:
        # Exponential kernel
        return 2 * sigma * np.exp(-r)
    
    elif nu == np.inf:
        # Squared exponential kernel
        return 2 * sigma * np.exp(-r**2 / 2)
    
    else:
        factor = (2**(1 - nu)) / gamma(nu)
        scaled_r = np.sqrt(2 * nu) * r
        matern_value = factor * (scaled_r**nu) * kv(nu, scaled_r)
        matern_value[r == 0] = sigma**2  # Avoid NaN at r = 0
        return 2 * sigma * matern_value
    
# Function to calculate the gradient of the Matérn kernel with respect to the smoothness parameter
def grad_matern_kernel_nu(r, nu, ell=1.0, sigma=1.0):
    """
    Compute the gradient of the Matérn kernel with respect to the smoothness parameter.

    Parameters:
        r (float or np.ndarray): The distance(s) between input points.
        nu (float): Smoothness parameter.
        ell (float): Length scale parameter (default is 1.0).
        sigma (float): Signal variance (default is 1.0).
    
    Returns:
        np.ndarray: Gradient of the Matérn kernel with respect to the smoothness parameter.
    """
    r = np.abs(r) / ell
    if nu == 0.5:
        # Exponential kernel
        return 0
    
    elif nu == np.inf:
        # Squared exponential kernel
        return 0
    
    else:
        factor = (2**(1 - nu)) / gamma(nu)
        scaled_r = np.sqrt(2 * nu) * r
        matern_value = factor * (scaled_r**nu) * kv(nu, scaled_r)
        matern_value[r == 0] = sigma**2  # Avoid NaN at r = 0
        grad = matern_value * np.log(scaled_r) - kv(nu + 1, scaled_r)
        return sigma**2 * grad
    
