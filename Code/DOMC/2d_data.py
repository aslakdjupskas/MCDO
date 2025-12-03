import numpy as np
from scipy.stats import multivariate_normal



import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import jax.numpy as jnp
import numpy as np

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


# MC Variational Dense Layer
class MCVariationalDense(nn.Module):
    def __init__(self, n_in, n_out, model_prob, model_lam, layer, dropout_layers):
        super(MCVariationalDense, self).__init__()
        self.model_prob = model_prob
        self.model_lam = model_lam
        self.dropout = nn.Dropout(p=self.model_prob) 
        self.model_M = nn.Parameter(torch.randn(n_in, n_out) * 0.1)
        self.model_m = nn.Parameter(torch.zeros(n_out))
        self.layer = layer
        self.dropout_layers = dropout_layers

    def forward(self, X):
        "Dropout only on the selected layers and during training"
        # if (self.training and self.layer in self.dropout_layers) or self.layer in self.dropout_layers:
        if not self.training and self.layer in self.dropout_layers:
            if self.layer in self.dropout_layers:
                model_W = self.model_M * torch.bernoulli(torch.full_like(self.model_M, 1 - self.model_prob)) / (1 - self.model_prob)      
            elif self.training: # Only scale in layers with dropout
                model_W = self.model_M * torch.bernoulli(torch.full_like(self.model_M, 1 - self.model_prob))

        else:
            model_W = self.model_M
        output = torch.mm(X, model_W) + self.model_m
        return output

    def regularization(self):
        return self.model_lam * (
            torch.sum(self.model_M ** 2) + torch.sum(self.model_m ** 2)
        )
        # return self.model_lam * (
        #     torch.sum(torch.abs(self.model_M)) + torch.sum(torch.abs(self.model_m))
        # )

class MCVariationalNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model_prob, model_lam, dropout_layers):
        super(MCVariationalNN, self).__init__()
        self.dropout_layers = dropout_layers
        self.layer1 = MCVariationalDense(input_size, hidden_size, model_prob, model_lam, layer=1, dropout_layers=dropout_layers)
        self.layer2 = MCVariationalDense(hidden_size, hidden_size, model_prob, model_lam, layer=2, dropout_layers=dropout_layers)
        self.layer3 = MCVariationalDense(hidden_size, hidden_size, model_prob, model_lam, layer=3, dropout_layers=dropout_layers)
        self.layer4 = MCVariationalDense(hidden_size, output_size, model_prob, model_lam, layer=4, dropout_layers=dropout_layers)
        

    def forward(self, X):
        X = torch.relu(self.layer1(X))
        X = torch.relu(self.layer2(X))
        X = torch.relu(self.layer3(X))  # New layer added
        X = self.layer4(X)  # Output layer
        return X

    def regularization(self):
        return (
            self.layer1.regularization() +
            self.layer2.regularization() +
            self.layer3.regularization() +  # Include new layer
            self.layer4.regularization()
        )



# MC Dropout inference
def mc_inference(model, X, n_samples=100):
    # model.eval()  # Keep dropout active
    preds = torch.zeros((n_samples, X.size(0), 1))
    with torch.no_grad():
        for i in range(n_samples):
            preds[i] = model(X)
    return preds.mean(dim=0), preds.std(dim=0), preds


import numpy as np
import jax.numpy as jnp

# -------------------------------------------------
# Function f
# -------------------------------------------------
def f(x, y):
    return np.sin(x)**8 + np.cos(20 + y * x) * np.cos(y)

# -------------------------------------------------
# DATASET 1 — f(x, y)
# -------------------------------------------------
def get_data_f(N=50, N_test=500, sigma_obs=0.05):
    """
    Returns train/test subsets for f(x, y):
        X_train: (N, 2)
        Y_train: (N, 1)
        X_test: (N_test, 2)
        Y_test: (N_test, 1)
        Z_full: full grid values (100x100)
    """
    # Grid
    x = np.linspace(0, 5, 100)
    y = np.linspace(0, 5, 100)
    Xg, Yg = np.meshgrid(x, y)

    Z = f(Xg, Yg)
    Z_flat = Z.ravel()
    XY = np.column_stack([Xg.ravel(), Yg.ravel()])

    # Training subset
    idx = np.random.choice(len(XY), N, replace=False)
    X_train = XY[idx]
    Y_train = Z_flat[idx] + sigma_obs * np.random.randn(N)
    Y_train = Y_train.reshape(-1, 1)

    # Test subset
    idx_test = np.random.choice(len(XY), N_test, replace=False)
    X_test = XY[idx_test]
    Y_test = Z_flat[idx_test].reshape(-1, 1)

    return X_train, Y_train, X_test, Y_test, Z

# -------------------------------------------------
# Function f2
# -------------------------------------------------
def f2(x, y):
    return 2. * multivariate_normal.pdf(
        np.dstack((x, y)),
        mean=[2.5, 2.5],
        cov=[[1.5, 0], [0, 1.5]]
    )

# -------------------------------------------------
# DATASET 2 — f2(x, y)
# -------------------------------------------------
def get_data_f2(N=50, N_test=500, sigma_obs=0.05):
    """
    Returns train/test subsets for f2(x, y):
        X_train: (N, 2)
        Y_train: (N, 1)
        X_test: (N_test, 2)
        Y_test: (N_test, 1)
        Z2_full: full grid (100x100)
    """
    # Grid
    x = np.linspace(0, 5, 100)
    y = np.linspace(0, 5, 100)
    Xg, Yg = np.meshgrid(x, y)

    Z2 = f2(Xg, Yg)
    Z_flat = Z2.ravel()
    XY = np.column_stack([Xg.ravel(), Yg.ravel()])

    # Training subset
    idx = np.random.choice(len(XY), N, replace=False)
    X_train = XY[idx]
    Y_train = Z_flat[idx] + sigma_obs * np.random.randn(N)
    Y_train = Y_train.reshape(-1, 1)

    # Test subset
    idx_test = np.random.choice(len(XY), N_test, replace=False)
    X_test = XY[idx_test]
    Y_test = Z_flat[idx_test].reshape(-1, 1)

    return X_train, Y_train, X_test, Y_test, Z2


sigma = 0.05
X, Y, X_test, Y_true, _ = get_data_f(N=50, N_test=500, sigma_obs=sigma)

## PARAMETERS
input_size = 2
output_size = 1      # FIXED
hidden_size = 32
model_prob  = 0.15
model_lam   = 1e-4
lr          = 0.001


# Convert to PyTorch tensors
X_train_torch = torch.tensor(np.array(X), dtype=torch.float32)
Y_train_torch = torch.tensor(np.array(Y), dtype=torch.float32)
X_test_torch = torch.tensor(np.array(X_test), dtype=torch.float32)
seeds = [100]
for idx, seed in enumerate(seeds):

    # random seed
    # seed = np.random.randint(0, 10000)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # Initialize model, loss function, and optimizer
    model = MCVariationalNN(input_size, hidden_size, output_size, model_prob, model_lam, [2,3])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    validation_error = []
    loss_history = []  
    patience = 12
    min_delta = 1e-10  # Minimum change in loss to qualify as improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0


    # Training
    epochs = 20000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_torch)
        loss = criterion(outputs, Y_train_torch) + model.regularization()
        loss.backward()
        loss_history.append(loss.item())
        optimizer.step()
        
    model.eval()

    plt.figure()
    plt.plot(validation_error, label="Validation Loss")
    plt.plot(loss_history, label="Training loss")
    plt.title(f"Training error for {seed}")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.yscale("log")
    plt.legend()
    plt.grid()
    # plt.show()
    
    mean_pred, std_pred, predictions = mc_inference(model, X_test_torch, n_samples=20000)
   
    # Convert to numpy
    mean_pred = mean_pred.numpy()
    std_pred = std_pred.numpy()




# Rebuild grid from test points
X_grid = X_test[:,0]
Y_grid = X_test[:,1]

order = np.lexsort((Y_grid, X_grid))

from scipy.interpolate import griddata

# Create regular grid
x = np.linspace(0, 5, 100)
y = np.linspace(0, 5, 100)
Xg, Yg = np.meshgrid(x, y)

# Interpolate data onto grid
Z_true_mean = griddata(X_test, Y_true[:,0], (Xg, Yg), method='cubic')
Z_pred_mean = griddata(X_test, mean_pred[:,0], (Xg, Yg), method='cubic')
Z_pred_std  = griddata(X_test, std_pred[:,0],  (Xg, Yg), method='cubic')

# True observation noise
Z_true_std = np.full_like(Z_pred_std, sigma)

# ============================
#     PLOTS
# ============================

fig, axs = plt.subplots(2, 2, figsize=(14, 10))

c1 = axs[0,0].contourf(Xg, Yg, Z_true_mean, levels=100)
axs[0,0].set_title("True Mean Function f(x,y)")
fig.colorbar(c1, ax=axs[0,0])

c2 = axs[0,1].contourf(Xg, Yg, Z_pred_mean, levels=100)
axs[0,1].set_title("Predicted Mean")
fig.colorbar(c2, ax=axs[0,1])

c3 = axs[1,0].contourf(Xg, Yg, Z_true_std, levels=100)
axs[1,0].set_title(f"True Std (sigma={sigma})")
fig.colorbar(c3, ax=axs[1,0])

c4 = axs[1,1].contourf(Xg, Yg, Z_pred_std, levels=100)
axs[1,1].set_title("Predicted Std (Uncertainty)")
fig.colorbar(c4, ax=axs[1,1])

plt.tight_layout()
plt.show()

