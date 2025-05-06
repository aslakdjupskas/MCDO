import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad
from jax.scipy.linalg import cholesky, solve
import numpy as np
import jax

import matplotlib as mpl

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


from jax import config
config.update("jax_enable_x64", True)


def periodic_kernel(X1, X2, ell, p, sigma_p):
    """Periodic kernel."""
    r = jnp.abs(X1[:, None] - X2[None, :]) / ell
    r = r[:,:,0]
    K = sigma_p**2 * jnp.exp(-2 * jnp.sin(jnp.pi * r / p)**2)
    return K.astype(jnp.float64)


def rbf_kernel(X1, X2, length_scale, sigma):
    """Exponentiated quadratic (RBF) kernel."""
    dist_sq = jnp.sum((X1[:, None] - X2)**2, axis=-1)
    K = sigma**2 * jnp.exp(-0.5 * dist_sq / length_scale**2)
    return K.astype(jnp.float64)


def kernel(X1, X2, length_scale, sigma, ell, p, sigma_p):
    rbf = rbf_kernel(X1, X2, length_scale, sigma)
    periodic = periodic_kernel(X1, X2, ell, p, sigma_p)
    # return w1 * rbf + w2 * periodic
    return rbf * periodic


def log_marginal_likelihood(log_params, X_train, y_train, noise_variance=1e-6):
    """
    Computes the log-marginal likelihood for GP with a given set of hyperparameters.
    
    Parameters:
    - params: (length_scale, sigma, ell, p, sigma_p) tuple of kernel parameters
    - X_train: Training inputs (shape: n, d)
    - y_train: Training outputs (shape: n,)
    - noise_variance: Noise variance (default 1e-6)
    
    Returns:
    - log-likelihood: Log marginal likelihood (loss)
    """
    length_scale, sigma, ell, p, sigma_p = jnp.exp(jnp.array(log_params))


    y_norm = y_train/jnp.linalg.norm(y_train)
    # Compute the kernel matrix
    K = kernel(X_train, X_train, length_scale, sigma, ell, p, sigma_p)
    K += noise_variance * jnp.eye(X_train.shape[0])

    # Cholesky decomposition (for stability)
    L = cholesky(K, lower=True)

    # Compute alpha (same as K^(-1)*y)
    alpha = solve(L.T, solve(L, y_norm))
    
    # Compute log marginal likelihood
    log_lml = (-0.5 * jnp.dot(y_norm, alpha) - jnp.sum(jnp.log(jnp.diagonal(L))) - 0.5 * X_train.shape[0] * jnp.log(2 * jnp.pi))
    
    return -log_lml  # Return negative LML for minimization


def optimize_hyperparameters(X_train, y_train, init_params, learning_rate=0.01, num_iters=200):
    """
    Optimize hyperparameters using gradient descent.
    
    Parameters:
    - X_train: Training inputs
    - y_train: Training outputs
    - init_params: Initial kernel hyperparameters (length_scale, sigma, ell, p, sigma_p)
    - learning_rate: Learning rate
    - num_iters: Number of iterations for optimization
    
    Returns:
    - optimized_params: Optimized kernel hyperparameters (length_scale, sigma, ell, p, sigma_p)
    """

    # Transform initial parameters to log-space
    log_params = jnp.log(jnp.array(init_params))
    # Compute the gradient w.r.t. the log marginal likelihood
    grad_log_lml = grad(log_marginal_likelihood)
    
    
    # Perform gradient descent
    for i in range(num_iters):
        grads = grad_log_lml(log_params, X_train, y_train)
        
        # Update parameters 
        length_scale, sigma, ell, p, sigma_p = log_params
        grad_length_scale, grad_sigma, grad_ell, grad_p,grad_sigma_p = grads
        
        length_scale = length_scale - learning_rate * grad_length_scale
        sigma        = sigma - learning_rate * grad_sigma
        ell          = ell - learning_rate * grad_ell
        p            = p - learning_rate * grad_p
        sigma_p      = sigma_p - learning_rate * grad_sigma_p
        
        log_params = (length_scale, sigma, ell, p, sigma_p)
        loss = log_marginal_likelihood(log_params, X_train, y_train)

        if i % 100 == 0:    
            print(f"Iteration {i}: params = {jnp.exp(jnp.array(log_params))}, loss = {loss}")
    optimized_params = jnp.exp(jnp.array(log_params))
    return optimized_params


np.random.seed(1)
n_samples = 200
X = np.linspace(-3, 3, n_samples).flatten()[:, None]
y = np.sin(2*X).flatten() + 0.01 * np.random.randn(n_samples)
y = np.sin(4*X).flatten() + X.flatten()/3 + (X.flatten()**2)/5 


# Randomly sample indices
n_train = 10  # Number of training points
random_indices = np.random.choice(len(X), size=n_train, replace=False)
# random_indices = [i for i in range(0,n_samples, int(np.floor(n_samples/(n_train-1))))]
# Training data based on random indices
X_train = X[random_indices]
y_train = y[random_indices] + 0.01 * np.random.randn(len(random_indices))

# Initialize kernel hyperparameters (length_scale, sigma, p, sigma_p)
init_params = (1.0, 1.0, 1.0, 1.0, 1.0)

# Optimize hyperparameters
optimized_params = optimize_hyperparameters(X_train, y_train, init_params, learning_rate=0.001, num_iters=2000)

print(f"""Optimized hyperparameters: length_scale = {optimized_params[0]}, sigma = {optimized_params[1]}
      ell = {optimized_params[2]}, p = {optimized_params[3]}, sigma_p = {optimized_params[4]}""")

K = kernel(X_train, X_train, *optimized_params)
L = np.linalg.cholesky(K + 0.000001 * np.eye(len(X_train)))

# Step 3: Compute alpha, same as K^(-1)*y
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

# Step 4: Predictive mean
K_train_test = kernel(X_train, X, *optimized_params)  # Cross-covariance between train and test points
f_mean_opt = K_train_test.T @ alpha

# Step 4: Predictive variance
K_test_test = kernel(X, X, *optimized_params)  # Covariance of test points
v = np.linalg.solve(L, K_train_test)
f_var_opt = K_test_test - v.T@v

# Non optimized parameters
K = kernel(X_train, X_train, *init_params)
L = np.linalg.cholesky(K + 0.000001 * np.eye(len(X_train)))

# Mean
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))
K_train_test = kernel(X_train, X, *init_params)  # Cross-covariance between train and test points
f_mean = K_train_test.T @ alpha

# VCV
K_test_test = kernel(X, X, *init_params)  # Covariance of test points
v = np.linalg.solve(L, K_train_test)
f_var = K_test_test - v.T@v

# plot the posterior samples
X = X.flatten()
fig, axes = plt.subplots(2, 1, figsize=(10, 12))
# fig.suptitle("Posterior Samples from Gaussian Process", fontsize=16)

# Generate samples and plot from the posterior
num_samples = 50
for i in range(num_samples):
    axes[0].plot(X, np.random.multivariate_normal(f_mean_opt, f_var_opt), color="gray", alpha=0.05)  # Plot the posterior samples
axes[0].plot(X, f_mean_opt, color='black', label="Predictive Mean", linewidth=2)
axes[0].fill_between(
    X, 
    f_mean_opt - 2 * np.sqrt(np.diagonal(f_var_opt)), 
    f_mean_opt + 2 * np.sqrt(np.diagonal(f_var_opt)), 
    color='lightblue', label="95% Confidence Interval"
)
axes[0].plot(X_train, y_train, "ro", label="Training Points")
axes[0].plot(X, y, "m-", label="True function")
axes[0].legend()
axes[0].set_title("Optimized RBF and Periodic Kernel")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")
# grid
axes[0].grid()

# Second plot
for i in range(num_samples):
    axes[1].plot(X, np.random.multivariate_normal(f_mean, f_var),  color="gray", alpha=0.05)  # Plot the posterior samples
axes[1].plot(X, f_mean, color='black', label="Predictive Mean", linewidth=2)
axes[1].fill_between(
    X, 
    f_mean - 2 * np.sqrt(np.diagonal(f_var)), 
    f_mean + 2 * np.sqrt(np.diagonal(f_var)), 
    color='lightblue', label="95% Confidence Interval"
)
axes[1].plot(X_train, y_train, "ro", label="Training Points")
axes[1].plot(X, y, "m-", label="True function")
axes[1].legend()
axes[1].set_title("Non optimized RBF and Periodic Kernel")
axes[1].set_xlabel("x")
axes[1].set_ylabel("f(x)")
axes[1].grid()

plt.tight_layout()
plt.savefig("plots/GP_npyrofcn.pdf")
plt.show()
print("OK")
