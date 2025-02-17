import matplotlib.pyplot as plt
import jax.numpy as jnp
from jax import grad
from jax.scipy.linalg import cholesky, solve
import numpy as np
from scipy.linalg import solve_triangular

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
    
    log_lml = -0.5 * jnp.dot(y_train, alpha) - jnp.sum(jnp.log(jnp.diagonal(L) + 1e-6)) - 0.5 * X_train.shape[0] * jnp.log(2 * jnp.pi)
    
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
def get_data(N=50, D_X=3, sigma_obs=0.05, N_test=500):
    D_Y = 1  # create 1d outputs
    np.random.seed(0)
    X = jnp.linspace(-1, 1, N)
    X = jnp.power(X[:, np.newaxis], jnp.arange(D_X))
    W = 0.5 * np.random.randn(D_X)
    Y = jnp.dot(X, W) + 0.5 * jnp.power(0.5 + X[:, 1], 2.0) * jnp.sin(4.0 * X[:, 1])
    Y += sigma_obs * np.random.randn(N)
    Y = Y[:, np.newaxis]
    Y -= jnp.mean(Y)
    Y /= jnp.std(Y)

    assert X.shape == (N, D_X)
    assert Y.shape == (N, D_Y)

    X_test = jnp.linspace(-1.3, 1.3, N_test)
    X_test = jnp.power(X_test[:, np.newaxis], jnp.arange(D_X))

    return X, Y, X_test




if __name__ == "__main__":
    # Generate data
    np.random.seed(8)
    n_samples = 1000
    # X = np.linspace(-60, 60, n_samples).flatten()[:, None]

    # y = np.sin(X).flatten() *  (X.flatten()**2/5) # True function
    # y = np.sin(X).flatten() + (X.flatten()/9) + np.random.uniform(-1,1) # True function

    N, D_X, D_H = 100, 3, 5
    X, Y, X_test = get_data(N=N, D_X=D_X, N_test=n_samples)
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
    optimized_params = optimize_hyperparameters(X_train, y_train, X_test, init_params, learning_rate=0.00015, num_iters=500)

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
    f_var_opt = (f_var_opt @ f_var_opt.T) /2

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

    # plot the posterior samples
    X = X.flatten()
    X_test = X_test.flatten()
    
    fig, axes = plt.subplots(2, 1, figsize=(10, 12))
    fig.suptitle("Posterior Samples from Gaussian Process", fontsize=16)

    # Generate samples and plot from the posterior
    num_samples = 15
    for i in range(num_samples):
        axes[0].plot(X_test, np.random.multivariate_normal(f_mean_opt, f_var_opt), color='black', alpha=0.1)  # Plot the posterior samples
    axes[0].plot(X_test, f_mean_opt, color='black', label="Predictive Mean", linewidth=2)
    axes[0].fill_between(
        X_test, 
        f_mean_opt - 2 * np.sqrt(np.diagonal(f_var_opt)), 
        f_mean_opt + 2 * np.sqrt(np.diagonal(f_var_opt)), 
        color='gray', alpha=0.2, label="95% Confidence Interval"
    )
    axes[0].plot(X_train, y_train, "r.", label="Training Points")
    # axes[0].plot(X, y, "m-", label="True function")
    axes[0].legend()
    axes[0].set_title("\nOptimized RBF Kernel")
    axes[0].set_xlabel("x")
    axes[0].set_ylabel("f(x)")
    axes[0].set_ylim(top=8, bottom=-6)


    # Second plot
    for i in range(num_samples):
        axes[1].plot(X_test, np.random.multivariate_normal(f_mean, f_var), color='black', alpha=0.1)  # Plot the posterior samples
    axes[1].plot(X_test, f_mean, color='black', label="Predictive Mean", linewidth=2)
    axes[1].fill_between(
        X_test, 
        f_mean - 2 * np.sqrt(np.diagonal(f_var)), 
        f_mean + 2 * np.sqrt(np.diagonal(f_var)), 
        color='gray', alpha=0.2, label="95% Confidence Interval"
    )
    axes[1].plot(X_train, y_train, "r.", label="Training Points")
    # axes[1].plot(X, y, "m-", label="True function")
    axes[1].legend()
    axes[1].set_title("\nNon optimized RBF Kernel")
    axes[1].set_xlabel("x")
    axes[1].set_ylabel("f(x)")

    plt.tight_layout()
    
    plt.savefig("plots/posGP_rbf.pdf")
    plt.show()
    print("ok")
    plt.close()