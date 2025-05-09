import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad
from jax.scipy.linalg import cholesky, solve
import numpy as np
from scipy.linalg import solve_triangular

from jax import config
config.update("jax_enable_x64", True)

import gpjax as gpx
from jax import random as jr
import jax.numpy as jnp
import optax as ox

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


def kernel(X1, X2, length_scale, sigma):
    """Exponentiated quadratic (RBF) kernel."""

    X1 = X1[:, None] if X1.ndim == 1 else X1  # Ensure 2D
    X2 = X2[:, None] if X2.ndim == 1 else X2  # Ensure 2D

    dist_sq = jnp.sum((X1[:, None, :] - X2[None, :, :])**2, axis=-1)
    K = sigma**2 * jnp.exp(-0.5 * dist_sq / length_scale**2)
    return K


def log_marginal_likelihood(log_params, X_train, y_train, noise_variance=1e-3):
    """
    Compute the log-marginal likelihood for GP with a given set of hyperparameters.
    
    Parameters:
    - log_params: Log-space parameters (log_length_scale, log_sigma)
    - X_train: Training inputs (shape: n, d)
    - y_train: Training outputs (shape: n,)
    - noise_variance: Noise variance (default 1e-6)
    
    Returns:
    - log-likelihood: Log marginal likelihood (scalar)
    """
    # Transform back to original space
    length_scale, sigma = jnp.exp(jnp.array(log_params))

    # Kernel matrix
    K = kernel(X_train, X_train, length_scale, sigma) + noise_variance * jnp.eye(X_train.shape[0])

    # Cholesky decomposition for stability
    L = cholesky(K, lower=True) #+ np.random.uniform(-1,1)
    
    y_train_scaled = y_train/jnp.linalg.norm(y_train)

    # Compute alpha (same as K^(-1)*y)
    alpha = solve(L.T, solve(L, y_train))

    # Compute log marginal likelihood
    
    log_lml = -0.5 * jnp.dot(y_train, alpha) - jnp.sum(jnp.log(jnp.diagonal(L) + 1e-6)) - 0.5 * X_train.shape[0] * jnp.log(2 * jnp.pi)+ 1/(length_scale+sigma)
    
    return -log_lml  # Return negative LML for minimization


def optimize_hyperparameters(X_train, y_train, X_test, init_params, learning_rate=0.01, num_iters=100):
    """
    Optimize hyperparameters using gradient descent in log-space.
    
    Parameters:
    - X_train: Training inputs
    - y_train: Training outputs
    - init_params: Initial kernel hyperparameters (length_scale, sigma)
    - learning_rate: Learning rate for gradient descent
    - num_iters: Number of iterations for optimization
    
    Returns:
    - optimized_params: Optimized kernel hyperparameters (length_scale, sigma)
    """

    # Transform initial parameters to log-space
    log_params = jnp.log(jnp.array(init_params))  # (log_length_scale, log_sigma)
    
    # Compute the gradient of the log marginal likelihood
    grad_log_lml = grad(log_marginal_likelihood)
    # Perform gradient descent
    for i in range(num_iters):
        grads = grad_log_lml(log_params, X_train, y_train)
        
        # Update parameters individually
        length_scale, sigma = log_params
        grad_length_scale, grad_sigma = grads
        length_scale = length_scale - learning_rate * grad_length_scale #/(jnp.abs(grad_length_scale)+1e-6)
        sigma        = sigma - learning_rate * grad_sigma #/(jnp.abs(grad_sigma)+1e-6)

        # Compute current loss for logging
        log_params = (length_scale, sigma)
        loss = log_marginal_likelihood(log_params, X_train, y_train)

        if i % 2 == 0:    
            print(f"Iteration {i}: params = {jnp.exp(jnp.array(log_params))}, loss = {loss}")
            params_i = jnp.exp(jnp.array(log_params))
            # fm, fv = run_posterior(X_train, y_train, X_test, *params_i)
            # save_posterior_plot(fm, fv, X_train, y_train, X_test.flatten(), loss, i)
    
    # Transform back to original space
    optimized_params = jnp.exp(jnp.array(log_params))
    return optimized_params

def run_posterior(X_train, y_train, X_test, *params):
    K = kernel(X_train, X_train, *params)
    L = cholesky(K + 0.002 * np.eye(len(X_train)), lower=True)
    

    # Step 3: Compute alpha, same as K^(-1)*y
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    # Step 4: Predictive mean
    K_train_test = kernel(X_train, X_test, *params)  # Cross-covariance between train and test points
    f_mean_i = K_train_test.T @ alpha

    # Step 4: Predictive variance
    K_test_test = kernel(X_test, X_test, *params)  # Covariance of test points
    v = np.linalg.solve(L, K_train_test)
    f_var_i = K_test_test - v.T@v
    return f_mean_i, f_var_i

def save_posterior_plot(fm, fv, X_train, y_train, X_test, loss, j):
    plt.figure(figsize=(10, 12))
    plt.title(f"Posterior Samples. Loss: {loss:.2f}")

    # Generate samples and plot from the posterior
    num_samples = 20
    for _ in range(num_samples):
        plt.plot(X_test, np.random.multivariate_normal(fm, fv), alpha=0.1, color='gray')  # Plot the posterior samples
    plt.fill_between(
        X_test, 
        fm - 2.96 * np.sqrt(np.diagonal(fv)), 
        fm + 2.96 * np.sqrt(np.diagonal(fv)), 
        color='gray', alpha=0.2, label="99% Confidence Interval"
    )

    plt.plot(X_train, y_train, "r.", label="Training Points")
    plt.plot(X_test, fm, color='black', label="Predictive Mean", linewidth=0.8)
    
    plt.legend(loc='upper left')
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.ylim(top=8, bottom=-6)
    plt.savefig(f"Code/Clean code/opt_vid/opt_vid_{j}.png")
    plt.close()

# create artificial regression dataset
def get_data(N=50, D_X=3, sigma_obs=0.05, N_test=500, gap=False):
    D_Y = 1  # Create 1D outputs
    np.random.seed(0)
    
    # Generate test data first
    X_test = jnp.linspace(-1.3, 1.3, N_test)
    X_test = jnp.power(X_test[:, np.newaxis], jnp.arange(D_X))
    
    # Define ground truth function
    W = 0.5 * np.random.randn(D_X)
    Y_true = jnp.dot(X_test, W) + 0.5 * jnp.power(0.5 + X_test[:, 1], 2.0) * jnp.sin(4.0 * X_test[:, 1])
    Y_true = Y_true[:, np.newaxis]
    Y_true -= jnp.mean(Y_true)
    Y_true /= jnp.std(Y_true)
    
    # Apply gap only if enabled
    if gap:
        mask = (X_test[:, 1] < -0.5) | (X_test[:, 1] > 0.5)
        X_available = X_test[mask]
        Y_available = Y_true[mask]
    else:
        X_available = X_test
        Y_available = Y_true
    
    # Ensure X and Y are within the range [-1,1] **before selecting indices**
    valid_mask = (X_available[:, 1] >= -1.0) & (X_available[:, 1] <= 1.0)
    X_available = X_available[valid_mask]
    Y_available = Y_available[valid_mask]

    # Now we are guaranteed to sample only from valid points
    if len(X_available) < N:
        raise ValueError(f"Not enough valid samples ({len(X_available)}) to select N={N}.")

    indices = np.linspace(0, len(X_available) - 1, N, dtype=int)
    X = X_available[indices]
    Y = Y_available[indices] + sigma_obs * np.random.randn(N, D_Y)
    assert X.shape == (N, D_X)
    assert Y.shape == (N, D_Y)
    assert X_test.shape == (N_test, D_X)
    assert Y_true.shape == (N_test, D_Y)
    
    return X, Y, X_test, Y_true




if __name__ == "__main__":
    # Generate data
    np.random.seed(8)
    n_samples = 1000
    # X = np.linspace(-60, 60, n_samples).flatten()[:, None]

    # y = np.sin(X).flatten() *  (X.flatten()**2/5) # True function
    # y = np.sin(X).flatten() + (X.flatten()/9) + np.random.uniform(-1,1) # True function

    N, D_X, D_H = 50, 3, 5
    X, Y, X_test, Y_true = get_data(N=N, D_X=D_X, N_test=n_samples, gap=True)
    X_train = X[:,1]; y_train = Y[:,0]; X_test = X_test[:,1]
    
    # plt.figure()
    # plt.plot(X[:,1], Y[:,0], 'kx')

    # plt.show()


    # n_train = 50  # Number of training pointsNumber of training points
    # random_indices = np.random.choice(len(X), size=n_train, replace=False)
    # random_indices = [i for i in range(0,n_samples, int(np.floor(n_samples/(n_train-1))))]

    # # Training data based on random indices
    # X_train = X[random_indices]
    # y_train = y[random_indices] + np.random.normal(0, 0.01, X_train.shape[0]) 
    # X_test = np.delete(X, random_indices, axis=0)

    # Initialize kernel hyperparameters
    init_params = (10.0, 10.0) # (length_scale, sigma)

    # Optimize hyperparameters
    optimized_params = optimize_hyperparameters(X_train, y_train, X_test, init_params, learning_rate=0.000075, num_iters=2000)

    print(f"Optimized hyperparameters: length_scale = {optimized_params[0]}, sigma = {optimized_params[1]}")

    K_train = kernel(X_train, X_train, *optimized_params)
    L = cholesky(K_train + 0.002 * np.eye(len(X_train)) , lower=True) #+ 0.0005 * np.eye(len(X_train))

    # Step 3: Compute alpha, same as K^(-1)*y
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

    # Step 4: Predictive mean
    K_train_test = kernel(X_train, X_test, *optimized_params)  # Cross-covariance between train and test points
    f_mean_opt = K_train_test.T @ alpha

    # Step 4: Predictive variance
    K_test_test = kernel(X_test, X_test, *optimized_params)  # Covariance of test points
    v = np.linalg.solve(L, K_train_test)
    f_var_opt = K_test_test - v.T@v
    var_norm = np.linalg.norm(f_var_opt)
    f_var_opt = (f_var_opt @ f_var_opt.T)/(var_norm**2)

    D = np.diag(1/(2 *np.sqrt(np.diag(f_var_opt))))
    f_var_opt = (f_var_opt @ D)


    # Without optimization
    Kw = kernel(X_train, X_train, *init_params)
    Lw = cholesky(Kw + 0.001 * np.eye(len(X_train)), lower=True)

    # Mean
    alpha = np.linalg.solve(Lw.T, np.linalg.solve(Lw, y_train))
    K_train_test = kernel(X_train, X_test, *init_params)
    f_mean = K_train_test.T @ alpha

    # VCV
    K_test_test = kernel(X_test, X_test, *init_params)  # Covariance of test points
    vw = np.linalg.solve(Lw, K_train_test)
    f_var = K_test_test - vw.T@vw
    f_var = (f_var @ f_var.T)

    # GP Jax
    # dataset
    key = jr.PRNGKey(123)
    np.random.seed(8)
    # X_train = X[:,1].reshape(-1,1); y_train = Y[:,0].reshape(-1,1); X_test = X_test[:,1].reshape(-1,1)

    # dataset
    D = gpx.Dataset(X=X_train.reshape(-1,1), y=y_train.reshape(-1,1))

    # prior
    meanf = gpx.mean_functions.Zero()  # Zero mean 
    kernel = gpx.kernels.RBF()  # RBF kernel
    # kernel = gpx.kernels.Matern32(lengthscale=10., variance=10.)


    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

    # Define the likelihood 
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=15)

    # Combine prior and likelihood to form posterior
    posterior = prior * likelihood

    # Define optimizer
    optimizer = ox.adam(learning_rate=1e-2)
    # optimizer = ox.sgd(learning_rate=0.002)


    # Fit the model to the training data using MLE optimization
    opt_posterior, history = gpx.fit(
        model=posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data=D,
        optim=optimizer,
        num_iters=8000,
        safe=True,
        key=key,
    )

    # Predict on test points
    latent_dist = opt_posterior(X_test.reshape(-1,1), D)
    predictive_dist = opt_posterior.likelihood(latent_dist)

    # Obtain predictive mean and std
    pred_mean = predictive_dist.mean()
    pred_std = predictive_dist.stddev()
    # cov matrix
    cov_matrix = predictive_dist.covariance()
    
    opt_params = (opt_posterior.prior.kernel.lengthscale.value, opt_posterior.prior.kernel.variance.value)


    # plot the posterior samples
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 12))
    # fig.suptitle("Posterior Samples from Gaussian Process", fontsize=16, fontweight='bold')

    # Generate samples and plot from the posterior
    num_samples = 50
    for i in range(num_samples):
        axes[0].plot(X_test, np.random.multivariate_normal(f_mean_opt, f_var_opt), color='gray', alpha=0.1)
          # Plot the posterior samples
    axes[0].plot(X_test, f_mean_opt, color='black', label="Predictive Mean", linewidth=2)
    axes[0].fill_between(X_test, f_mean_opt - 2 * np.sqrt(var_norm*np.diagonal(f_var_opt)), f_mean_opt + 2 * np.sqrt(var_norm * np.diagonal(f_var_opt)), color="lightblue", label="95% CI")
    axes[0].plot(X_test, Y_true, "k--", lw=2.0, label="True function")
    axes[0].plot(X_train, y_train, "r.", label="Train Data", alpha=0.5)
    # axes[0].plot(X, y, "m-", label="True function")
    axes[0].legend()
    axes[0].set_title(f"\n\nMyGP \nLengthscale: {optimized_params[0]:.2f}, Variance: {optimized_params[1]:.2f}\n")
    axes[0].set_xlabel("X")
    axes[0].set_ylabel("Y")
    axes[0].set_ylim(top=4, bottom=-4)
    axes[0].grid()

    axes[1].set_title(f"\n\nGP Jax \nLengthscale: {opt_params[0]:.2f}, Variance: {opt_params[1]:.2f}\n")
    for _ in range(num_samples):
        axes[1].plot(X_test, np.random.multivariate_normal(pred_mean.flatten(), cov_matrix), alpha=0.1, color='gray')  # Plot the posterior samples
    axes[1].plot(X[:,1], Y[:,0], 'r.', label="Train Data", alpha=0.5)
    axes[1].plot(X_test, pred_mean, label="Mean Prediction", color="black", linewidth=2)
    axes[1].plot(X_test, Y_true, "k--", lw=2.0, label="True function")
    axes[1].fill_between(X_test, pred_mean - 2*pred_std, pred_mean + 2*pred_std, color="lightblue", label="95% CI")
    axes[1].legend()
    axes[1].set_ylim(top=4, bottom=-4)
    axes[1].set_xlabel("X")
    axes[1].set_ylabel("Y")
    axes[1].grid()


    # Second plot
    # for i in range(num_samples):
    #     axes[1].plot(X_test, np.random.multivariate_normal(f_mean, f_var), color='black', alpha=0.1)  # Plot the posterior samples
    # axes[1].plot(X_test, f_mean, color='black', label="Predictive Mean", linewidth=2)
    # axes[1].fill_between(
    #     X_test, 
    #     f_mean - 2 * np.sqrt(np.diagonal(f_var)), 
    #     f_mean + 2 * np.sqrt(np.diagonal(f_var)), 
    #     color='gray', alpha=0.2, label="95% Confidence Interval"
    # )
    # axes[1].plot(X_train, y_train, "r.", label="Training Points")
    # # axes[1].plot(X, y, "m-", label="True function")
    # axes[1].legend()
    # axes[1].set_title("\nNon optimized RBF Kernel")
    # axes[1].set_xlabel("x")
    # axes[1].set_ylabel("f(x)")
    # axes[1].set_ylim(top=4, bottom=-4)

    plt.tight_layout()
    
    plt.savefig("plots/posGP_rbf_gap.pdf")
    plt.show()
    print("ok")
    plt.close()