
import jax.numpy as jnp
from jax import grad
from jax.scipy.linalg import cholesky, solve


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

        if i % 10 == 0:    
            print(f"Iteration {i}: params = {jnp.exp(jnp.array(log_params))}, loss = {loss}")
    
    # Transform back to original space
    optimized_params = jnp.exp(jnp.array(log_params))
    return optimized_params

def run_posterior(X_train, y_train, X_test, init_params, noise_variance=1e-3):

    # Optimize hyperparameters
    optimized_params = optimize_hyperparameters(X_train, y_train, X_test, init_params, learning_rate=0.00015, num_iters=100)
    print(f"Optimized hyperparameters: length_scale = {optimized_params[0]}, sigma = {optimized_params[1]}")

    K_train = kernel(X_train, X_train, *optimized_params)
    L = cholesky(K_train + noise_variance * jnp.eye(len(X_train)) , lower=True) #+ 0.0005 * np.eye(len(X_train))

    # Compute alpha, same as K^(-1)*y
    alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y_train))

    # Conditional posterior Mean function
    K_train_test = kernel(X_train, X_test, *optimized_params)  # Cross-covariance between train and test points
    f_mean_opt = K_train_test.T @ alpha

    # Conditional posterior variance function
    K_test_test = kernel(X_test, X_test, *optimized_params)
    v = jnp.linalg.solve(L, K_train_test)
    f_var_opt = K_test_test - v.T@v
    f_var_opt = (f_var_opt @ f_var_opt.T) /2
    return f_mean_opt, f_var_opt
