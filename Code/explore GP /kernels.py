import numpy as np
import scipy.spatial

# Define the exponentiated quadratic 
def exponentiated_quadratic(xa, xb):
    """Exponentiated quadratic  with σ=1"""
    # L2 distance (Squared Euclidian)
    sq_norm = -0.5 * scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(sq_norm)

def linear_kernel(xa, xb):
    """Linear kernel: K(x, x') = x.T * x'"""
    return np.dot(xa, xb.T)

def polynomial_kernel(xa, xb, degree=3, coef0=1):
    """Polynomial kernel: K(x, x') = (x.T * x' + coef0)^degree"""
    return (np.dot(xa, xb.T) + coef0) ** degree

def sigmoid_kernel(xa, xb, coef0=1, alpha=0.1):
    """Sigmoid kernel: K(x, x') = tanh(alpha * (x.T * x') + coef0)"""
    return np.tanh(alpha * np.dot(xa, xb.T) + coef0)

def rational_quadratic_kernel(xa, xb, alpha=1.0):
    """Rational quadratic kernel: K(x, x') = (1 + ||x - x'||^2 / (2 * alpha))^(-alpha)"""
    sq_norm = scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return (1 + sq_norm / (2 * alpha)) ** (-alpha)

def cosine_similarity_kernel(xa, xb):
    """Cosine similarity kernel: K(x, x') = cos(theta)"""
    cos_sim = np.dot(xa, xb.T) / (np.linalg.norm(xa, axis=1)[:, np.newaxis] * np.linalg.norm(xb, axis=1))
    return cos_sim

def rbf_kernel(xa, xb, sigma=1.0):
    """Radial Basis Function (RBF) kernel: K(x, x') = exp(-||x - x'||^2 / (2 * sigma^2))"""
    sq_norm = scipy.spatial.distance.cdist(xa, xb, 'sqeuclidean')
    return np.exp(-sq_norm / (2 * sigma**2))

def white_noise_kernel(xa, xb, sigma_n=1.0):
    """
    White Noise Kernel: Returns sigma_n^2 if xa == xb, otherwise 0.
    
    Parameters:
    - xa, xb: Input data points (can be scalars or vectors).
    - sigma_n: The noise variance (default is 1.0).
    
    Returns:
    - The kernel value (covariance).
    """
    n_points = xa.shape[0]
    m_points = xb.shape[0]
    # The covariance matrix is a diagonal matrix with sigma_n^2 on the diagonal
    K = sigma_n**2 * np.eye(n_points)
    
    return K


