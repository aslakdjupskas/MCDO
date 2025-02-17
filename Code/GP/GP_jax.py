from jax import config
config.update("jax_enable_x64", True)

import gpjax as gpx
from jax import random as jr
import jax.numpy as jnp
import optax as ox
import matplotlib.pyplot as plt
from GP_RBF import optimize_hyperparameters, kernel as Knl, log_marginal_likelihood
import numpy as np

# true function
# f = lambda x: 10 * jnp.sin(x)
f = lambda x: jnp.sin(x) + x/3 + (x**2)/5 

# Generate data
key = jr.PRNGKey(123)
n_samples = 400
X = jr.uniform(key, shape=(n_samples, 1), minval=-4.0, maxval=4.0).sort()
y = f(X) + jr.normal(key, shape=(n_samples, 1)) * 0.1  # Added noise

# training points
n_train = 10
random_indices = [i for i in range(0, n_samples, int(np.floor(n_samples / (n_train - 1))))]

# Convert to JAX array
X_train = X[jnp.array(random_indices)]
y_train = y[jnp.array(random_indices)]

# dataset
D = gpx.Dataset(X=X_train, y=y_train)

# prior
meanf = gpx.mean_functions.Zero()  # Zero mean 
kernel = gpx.kernels.RBF()  # RBF kernel
prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

# Define the likelihood 
likelihood = gpx.likelihoods.Gaussian(num_datapoints=n_train)

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
xtest = jnp.linspace(-4., 4., 100).reshape(-1, 1)
latent_dist = opt_posterior(xtest, D)
predictive_dist = opt_posterior.likelihood(latent_dist)

# Obtain predictive mean and std
pred_mean = predictive_dist.mean()
pred_std = predictive_dist.stddev()

# Initialize kernel hyperparameters
init_params = (1.0, 0.50) # (length_scale, sigma)

# Optimize hyperparameters
optimized_params = optimize_hyperparameters(X_train, y_train.flatten(), xtest, init_params, learning_rate=0.002, num_iters=5001)
log_params = jnp.log(jnp.array(optimized_params))
lml = log_marginal_likelihood(log_params, X_train, y_train.flatten())

print(f"MyGP: Optimized hyperparameters: length_scale = {optimized_params[0]}, sigma = {optimized_params[1]}")

K = Knl(X_train, X_train, *optimized_params)
L = np.linalg.cholesky(K + 0.000001 * np.eye(len(X_train)))

# Step 3: Compute alpha, same as K^(-1)*y
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

# Step 4: Predictive mean
K_train_test = Knl(X_train, xtest, *optimized_params)  # Cross-covariance between train and test points
f_mean_opt = K_train_test.T @ alpha

# Step 4: Predictive variance
K_test_test = Knl(xtest, xtest, *optimized_params)  # Covariance of test points
v = np.linalg.solve(L, K_train_test)
f_var_opt = (K_test_test - v.T@v)

plt.figure()
plt.plot(xtest, f_mean_opt, 'r', label="MyGP")
plt.plot(xtest, pred_mean,  'b', label="GPJax")
plt.plot(xtest, f(xtest),  'y-', label="True")
plt.plot(X_train, y_train, "ro", label="Training Points")
plt.legend(loc='upper right')
plt.grid()
plt.savefig('plots/MyGPvsGPJax2.pdf')
plt.show()
plt.close()

X = X.flatten()
fig, axes = plt.subplots(2, 1, figsize=(10, 12))
fig.suptitle("Posterior Samples from Gaussian Process", fontsize=16)

# Generate samples and plot from the posterior
num_samples = 5
for i in range(num_samples):
    axes[0].plot(xtest, np.random.multivariate_normal(f_mean_opt.flatten(), f_var_opt), alpha=0.5)  # Plot the posterior samples
axes[0].plot(xtest, f_mean_opt, color='black', label="Predictive Mean", linewidth=2)
axes[0].fill_between(
    xtest.flatten(), 
    f_mean_opt.flatten() - 2 * np.sqrt(np.diagonal(f_var_opt)).flatten(), 
    f_mean_opt.flatten() + 2 * np.sqrt(np.diagonal(f_var_opt)).flatten(), 
    color='gray', alpha=0.2, label="95% Confidence Interval"
)
axes[0].plot(X_train, y_train, "ro", label="Training Points")
axes[0].plot(xtest, f(xtest), "m-", label="True function")
axes[0].legend()
axes[0].set_title(f"\nOptimized RBF Kernel:  σ={optimized_params[1]:.2f}, l={optimized_params[0]:.2f}, NLML={lml:.2f}")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")

# Second plot
for i in range(num_samples):
    axes[1].plot(xtest, np.random.multivariate_normal(pred_mean, latent_dist.covariance()), alpha=0.5)  # Plot the posterior samples
axes[1].plot(xtest, pred_mean, color='black', label="Predictive Mean", linewidth=2)
axes[1].fill_between(
    xtest.flatten(), 
    pred_mean - 2 * pred_std, 
    pred_mean + 2 * pred_std, 
    color='gray', alpha=0.2, label="95% Confidence Interval"
)
axes[1].plot(X_train, y_train, "ro", label="Training Points")
axes[1].plot(xtest, f(xtest).flatten(), label="True function", color='blue')
axes[1].legend()
axes[1].set_title(f"\nGP-Jax:  σ={np.sqrt(opt_posterior.prior.kernel.variance.value):.2f}, l = {opt_posterior.prior.kernel.lengthscale.value:.2f}, NLML={history._value[-1]:.2f}")
axes[1].set_xlabel("x")
axes[1].set_ylabel("f(x)")

plt.tight_layout()
plt.savefig("plots/compare_jax2.pdf")
plt.show()
print('ok')


