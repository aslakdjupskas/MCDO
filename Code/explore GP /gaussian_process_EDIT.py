from matern_kernel import matern_kernel, grad_matern_kernel_ell, grad_matern_kernel_sigma, grad_matern_kernel_nu
import numpy as np
 
x = np.linspace(-5, 5, 1000)
y_known = np.sin(x) + 0.01 * np.random.randn(len(x))
random_set = np.random.choice(len(x), 20, replace=False)
x_known = y_known[random_set]

# Initialize hyperparameters
ell = 1.0
sigma = 1.0
nu = 1.5
# Compute the kernel and its gradient

# Write without loops
X1, X2 = np.meshgrid(x, x)
K = matern_kernel(np.abs(X1 - X2), nu, ell, sigma)

K11 = K[random_set][:, random_set]
K12 = K[random_set][:, :]
K21 = K[:, random_set]
K22 = K
mu_post = K21 @ np.linalg.inv(K11) @ x_known
cov_post = K22 - K21 @ np.linalg.inv(K11) @ K12
learning_rate = 0.01
import numpy as np

for i in range(10):
    # Compute the kernel matrix
    K = matern_kernel(np.abs(X1 - X2), nu=1.5, ell, sigma)
    K11 = K[random_set][:, random_set]
    K12 = K[random_set][:, :]
    K21 = K[:, random_set]
    K22 = K

    # Compute Cholesky decomposition for K11
    L = np.linalg.cholesky(K11 + 1e-6 * np.eye(len(K11)))  # Add jitter for numerical stability
    
    # Calculate posterior mean and covariance
    alpha = np.linalg.solve(L.T, np.linalg.solve(L, x_known))  # Solves L.T @ (L @ alpha) = x_known
    mu_post = K21 @ alpha
    v = np.linalg.solve(L, K12)  # Corrected step
    cov_post = K22 - v.T @ v


    # Calculate the log marginal likelihood
    log_det = 2 * np.sum(np.log(np.diag(L)))
    LML = -0.5 * x_known @ alpha - 0.5 * log_det - 0.5 * len(x_known) * np.log(2 * np.pi)
    print(f"Log marginal likelihood: {LML}")

    # Compute the gradient of the LML with respect to ell
    grad_K = grad_matern_kernel_ell(np.abs(X1 - X2), nu, ell, sigma)
    alpha_outer = np.outer(alpha, alpha)
    Q = alpha_outer - np.linalg.inv(K11)  # Precomputed for gradient use
    grad_ell = -0.5 * np.trace(Q @ grad_K[random_set][:, random_set])

    # Update the hyperparameters
    ell = max(ell + learning_rate * grad_ell, 1e-5)  # Ensure ell > 0
    print(f"Updated hyperparameters: ell={ell}, sigma={sigma}, nu={nu}")

# Plot the posterior mean and covariance
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.plot(x, y_known, 'b', label='True function')
plt.plot(x_known, x_known, 'ro', label='Observed data')
plt.plot(x, mu_post, 'r', label='Posterior mean')
for i in range(10):
    plt.plot(x, np.random.multivariate_normal(mu_post, cov_post), alpha=0.1)
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.title('Posterior mean and covariance')
plt.grid()
plt.show()

