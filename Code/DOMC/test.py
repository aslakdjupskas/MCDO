import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt


class VariationalDense(nn.Module):
    """Variational Dense Layer Class"""
    def __init__(self, n_in, n_out, model_prob, model_lam):
        super(VariationalDense, self).__init__()
        self.model_prob = model_prob
        self.model_lam = model_lam
        self.model_bern = torch.distributions.Bernoulli(probs=self.model_prob)
        self.model_M = nn.Parameter(torch.randn(n_in, n_out) * 0.01)
        self.model_m = nn.Parameter(torch.zeros(n_out))

    def forward(self, X):
        bernoulli_mask = self.model_bern.sample((self.model_M.shape[0],)).to(X.device)
        model_W = torch.mm(torch.diag(bernoulli_mask), self.model_M)
        output = torch.mm(X, model_W) + self.model_m
        return output

    def regularization(self):
        return self.model_lam * (
            self.model_prob * torch.sum(self.model_M ** 2) + torch.sum(self.model_m ** 2)
        )


# Create sample data.
n_samples = 20
X = np.random.normal(size=(n_samples, 1))
y = np.random.normal(np.cos(5. * X) / (np.abs(X) + 1.), 0.1).ravel()
X_pred = np.atleast_2d(np.linspace(-3., 3., num=100)).T
X = np.hstack((X, X**2, X**3))
X_pred = np.hstack((X_pred, X_pred**2, X_pred**3))

# Convert data to PyTorch tensors.
X_torch = torch.tensor(X, dtype=torch.float32)
y_torch = torch.tensor(y, dtype=torch.float32)
X_pred_torch = torch.tensor(X_pred, dtype=torch.float32)

# Define the model parameters.
n_feats = X.shape[1]
n_hidden = 10
model_prob = 0.8
model_lam = 1e-2

# Create the model layers.
L_1 = VariationalDense(n_feats, n_hidden, model_prob, model_lam)
L_2 = VariationalDense(n_hidden, n_hidden, model_prob, model_lam)
L_3 = VariationalDense(n_hidden, 1, model_prob, model_lam)
layers = [L_1, L_2, L_3]

# Define the optimizer.
optimizer = optim.Adam(
    params=[param for layer in layers for param in layer.parameters()], lr=1e-3
)

# Training loop.
n_iterations = 10000
for i in range(n_iterations):
    # Forward pass.
    out_1 = torch.relu(L_1(X_torch))
    out_2 = torch.relu(L_2(out_1))
    pred = L_3(out_2).squeeze()

    # Compute loss.
    sse = torch.sum((y_torch - pred) ** 2)
    loss = (
        sse +
        L_1.regularization() +
        L_2.regularization() +
        L_3.regularization()
    ) / n_samples

    # Backward pass and optimization.
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        mse = sse.item() / n_samples
        print(f"Iteration {i}. Mean squared error: {mse:.4f}.")

# Sampling from the posterior.
n_post = 1000
Y_post = np.zeros((n_post, X_pred.shape[0]))
with torch.no_grad():
    for i in range(n_post):
        out_1 = torch.relu(L_1(X_pred_torch))
        out_2 = torch.relu(L_2(out_1))
        Y_post[i] = L_3(out_2).squeeze().numpy()

# Plot the results.
if True:
    plt.figure(figsize=(8, 6))
    for i in range(n_post):
        plt.plot(X_pred[:, 0], Y_post[i], "b-", alpha=1. / 200)
    plt.plot(X[:, 0], y, "r.")
    plt.grid()
    plt.show()
