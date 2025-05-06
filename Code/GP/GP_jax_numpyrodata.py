from jax import config
config.update("jax_enable_x64", True)

import gpjax as gpx
from jax import random as jr
import jax.numpy as jnp
import optax as ox
import matplotlib.pyplot as plt
from GP_RBF import optimize_hyperparameters, kernel as Knl, log_marginal_likelihood
import numpy as np

def get_data(N=50, D_X=3, sigma_obs=0.05, N_test=500, N_val=20, gap=True, seed=0, sinc_noise_bool=False):
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
    valid_mask = (X_available[:, 1] >= -1.3) & (X_available[:, 1] <= 1.3)
    X_available = X_available[valid_mask]
    Y_available = Y_available[valid_mask]

    # Now we are guaranteed to sample only from valid points
    if len(X_available) < N:
        raise ValueError(f"Not enough valid samples ({len(X_available)}) to select N={N}.")

    indices = np.linspace(0, len(X_available) - 1, N, dtype=int)

    np.random.seed(seed)
    # indices = np.sort(np.random.choice(len(X_available), N, replace=False))
    X = X_available[indices]

    if sinc_noise_bool:
        sinc_noise = np.sinc(X[:,1])
        Y = Y_available[indices] + sinc_noise[:, None] * np.random.randn(N, D_Y)
    else:
        Y = Y_available[indices] + sigma_obs * np.random.randn(N, D_Y)

    assert X.shape == (N, D_X)
    assert Y.shape == (N, D_Y)
    assert X_test.shape == (N_test, D_X)
    assert Y_true.shape == (N_test, D_Y)

    # random indices in X_available for validation
    # np.random.seed(seed)
    # indices_val = np.sort(np.random.choice(len(X_available), N_val, replace=False))
    indices_val = np.linspace(0, len(X_available) - 1, N_val, dtype=int)
    X_val = X_available[indices_val]
    Y_val = Y_available[indices_val]
    if sinc_noise_bool:
        sinc_noise = np.sinc(X_val[:,1])
        Y_val = Y_val + sinc_noise[:, None] * np.random.randn(N_val, D_Y)
    else:
        Y_val = Y_val + sigma_obs * np.random.randn(N_val, D_Y)

    assert X_val.shape == (N_val, D_X)
    assert Y_val.shape == (N_val, D_Y)
    return X, Y, X_test, Y_true, X_val, Y_val





key = jr.PRNGKey(123)
np.random.seed(8)
n_samples = 500
N, D_X, D_H = 1500, 3, 5
sigma = 0.35
X, Y, X_test, Y_test, _, _ = get_data(N=1000, D_X=3, N_test=1000, sigma_obs=sigma, gap=False, sinc_noise_bool=True, seed=0)

# X_train = X[:,1].reshape(-1,1); y_train = Y[:,0].reshape(-1,1); X_test = X_test[:,1].reshape(-1,1)

# dataset
D = gpx.Dataset(X=X, y=Y)

# prior
meanf = gpx.mean_functions.Zero()  # Zero mean 
kernel_rbf = gpx.kernels.RBF()  # RBF kernelx
kernel_per = gpx.kernels.Periodic(period=5.0, lengthscale=2.0, variance=10.)
kernel_32 = gpx.kernels.Matern32()
kernel_12 = gpx.kernels.Matern12()
kernel_52 = gpx.kernels.Matern52()
kernel_pol = gpx.kernels.Polynomial(variance=10., degree=3)


kernels = {
    "RBF": kernel_rbf,
    #"Periodic": kernel_per,
    # "Matern12": kernel_12,
    # "Matern32": kernel_32,
    # "Matern52": kernel_52,
    #"Polynomial 3": kernel_pol
}

plt.figure()
plt.xticks([])
plt.yticks([])
plt.box(False)
plt.suptitle("Gaussian Process Regression with different kernels\n\n", fontsize=16, fontweight='bold')
# plt.suptitle("Gaussian Process Regression with different kernels\n\n")
# plt.subplot(3, 2, 1)

for idx, (kernel_name, kernel) in enumerate(kernels.items()):

    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

    # Define the likelihood 
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=15)

    # Combine prior and likelihood to form posterior
    posterior = prior * likelihood

    # Define optimizer
    optimizer = ox.adam(learning_rate=1e-3)
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
    latent_dist = opt_posterior(X_test, D)
    predictive_dist = opt_posterior.likelihood(latent_dist)

    # Obtain predictive mean and std
    pred_mean = predictive_dist.mean()
    pred_std = predictive_dist.stddev()
    # cov matrix
    cov_matrix = predictive_dist.covariance()
    
    opt_params = (opt_posterior.prior.kernel.lengthscale.value, opt_posterior.prior.kernel.variance.value)

    # plt.subplot(2, 2, idx+1)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"\n\n\n\nGP with {kernel_name} kernel \nLengthscale: {opt_params[0]:.2f}, Variance: {opt_params[1]:.2f}\n")

    for _ in range(200):
        plt.plot(X_test[:,1], np.random.multivariate_normal(pred_mean.flatten(), cov_matrix), alpha=0.1, color='gray')  # Plot the posterior samples
    plt.plot(X[:,1], Y[:,0], 'r.', label="Train Data", alpha=0.5)
    plt.plot(X_test[:,1], pred_mean, label="Mean Prediction", color="blue")
    plt.plot(X_test[:,1], Y_test, "k--", lw=2.0, label="True function")
    plt.fill_between(X_test[:,1].flatten(), pred_mean - 2*pred_std, pred_mean + 2*pred_std, color="lightblue", label="95% CI")
    plt.legend()
    plt.ylim(-4, 4)
    plt.grid()
    plt.savefig("MCDO/Code/GP/plots/GP_sincData2.pdf")
plt.show()
print("Done")


#plot
plt.figure(figsize=(8, 6))
plt.title("GP with RBF kernel")
plt.xlabel("X")
plt.ylabel("Y")

for _ in range(50):
    plt.plot(X_test[:,1], np.random.multivariate_normal(pred_mean.flatten(), cov_matrix), alpha=0.1, color='gray')  # Plot the posterior samples
plt.plot(X[:,1], Y[:,0], 'r.', label="Train Data", alpha=0.5)
plt.plot(X_test[:,1], pred_mean, label="Mean Prediction", color="blue")
plt.plot(X_test[:,1], Y_test, "k--", lw=2.0, label="True function")
plt.fill_between(X_test[:,1].flatten(), pred_mean - 2*pred_std, pred_mean + 2*pred_std, color="lightblue", label="95% CI")
plt.legend()
plt.grid()
# plt.savefig("MCDO/Code/GP/plots/gp_plot_numpyrodata.pdf")
plt.show()
print("Done")

