import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from GP_RBF import optimize_hyperparameters, kernel as Knl, log_marginal_likelihood
import jax.numpy as jnp


# Generate data
np.random.seed(9)
n_samples = 400
X = np.linspace(-4, 4, n_samples).flatten()[:, None]
y = np.sin(X).flatten() + X.flatten()/3 + (X.flatten()**2)/5  # True function


n_train = 6  # Number of training points
random_indices = np.random.choice(len(X), size=n_train, replace=False)
random_indices = [i for i in range(0,n_samples, int(np.floor(n_samples/(n_train-1))))]

# Training data 
X_train = X[random_indices]
y_train = y[random_indices] + 0.1 * np.random.randn(len(random_indices))
X_test = np.delete(X, random_indices, axis=0)
# Define the kernel: RBF kernel with length scale and constant term (sigma)
kernel = C(1.0, (1e-3, 1e3)) * RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e2))

# Create and train the Gaussian Process
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)
gp.fit(X_train, y_train)

# Make predictions (Mean and STD)
y_pred, sigma = gp.predict(X_test, return_std=True)

# Sample 5 functions from the posterior
y_samples = gp.sample_y(X_test, n_samples=5, random_state=42)


# Initialize kernel hyperparameters
init_params = (1.0, 0.50) # (length_scale, sigma)

# Optimize hyperparameters
optimized_params = optimize_hyperparameters(X_train, y_train, X_test, init_params, learning_rate=0.002, num_iters=1501)
log_params = jnp.log(jnp.array(optimized_params))
lml = log_marginal_likelihood(log_params, X_train, y_train)

print(f"MyGP: Optimized hyperparameters: length_scale = {optimized_params[0]}, sigma = {optimized_params[1]}")

# Display optimized hyperparameters
print("Scikit-learn: Optimized kernel:", gp.kernel_)

K = Knl(X_train, X_train, *optimized_params)
L = np.linalg.cholesky(K + 0.000001 * np.eye(len(X_train)))

# Step 3: Compute alpha, same as K^(-1)*y
alpha = np.linalg.solve(L.T, np.linalg.solve(L, y_train))

# Step 4: Predictive mean
K_train_test = Knl(X_train, X_test, *optimized_params)  # Cross-covariance between train and test points
f_mean_opt = K_train_test.T @ alpha

# Step 4: Predictive variance
K_test_test = Knl(X_test, X_test, *optimized_params)  # Covariance of test points
v = np.linalg.solve(L, K_train_test)
f_var_opt = K_test_test - v.T@v


X = X.flatten()
X_test = X_test.flatten()
fig, axes = plt.subplots(2, 1, figsize=(10, 12))
fig.suptitle("Posterior Samples from Gaussian Process", fontsize=16)

# Generate samples and plot from the posterior
num_samples = 5
for i in range(num_samples):
    axes[0].plot(X_test, np.random.multivariate_normal(f_mean_opt, f_var_opt), alpha=0.5)  # Plot the posterior samples
axes[0].plot(X_test, f_mean_opt, color='black', label="Predictive Mean", linewidth=2)
axes[0].fill_between(
    X_test, 
    f_mean_opt - 2 * np.sqrt(np.diagonal(f_var_opt)), 
    f_mean_opt + 2 * np.sqrt(np.diagonal(f_var_opt)), 
    color='gray', alpha=0.2, label="95% Confidence Interval"
)
axes[0].plot(X_train, y_train, "ro", label="Training Points")
axes[0].plot(X, y, "m-", label="True function")
axes[0].legend()
axes[0].set_title(f"\nOptimized RBF Kernel:  σ={optimized_params[1]:.2f}, l={optimized_params[0]:.2f}, , NLML={lml:.2f}")
axes[0].set_xlabel("x")
axes[0].set_ylabel("f(x)")

# Second plot
for i in range(num_samples):
    axes[1].plot(X_test, y_samples[:,i], alpha=0.5)  # Plot the posterior samples
axes[1].plot(X_test, y_pred, color='black', label="Predictive Mean", linewidth=2)
axes[1].fill_between(X_test, y_pred - 1.96 * sigma, y_pred + 1.96 * sigma, color='gray', alpha=0.2, label='95% Confidence interval')
axes[1].plot(X_train, y_train, "ro", label="Training Points")
axes[1].plot(X, y, "m-", label="True function")
axes[1].legend()
axes[1].set_title(f"\nScikit-learn:  σ={np.sqrt(gp.kernel_.k1.constant_value):.2f}, l={gp.kernel_.k2.length_scale:.2f}, NLML={-gp.log_marginal_likelihood_value_:.2f}")
axes[1].set_xlabel("x")
axes[1].set_ylabel("f(x)")

plt.tight_layout()
plt.savefig("plots/compare_sklearn2.pdf")
plt.show()
print('ok')


