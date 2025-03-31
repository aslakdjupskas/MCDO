import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter


import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp


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
        # return self.model_lam * (
        #     torch.sum(self.model_M ** 2) + torch.sum(self.model_m ** 2)
        # )
        return self.model_lam * (
            torch.sum(torch.abs(self.model_M)) + torch.sum(torch.abs(self.model_m))
        )

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



def get_data(N=50, D_X=3, sigma_obs=0.05, N_test=500, N_val=20, gap=True, seed=0):
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

    # np.random.seed(seed)
    # indices = np.random.choice(len(X_available), N, replace=False)
    X = X_available[indices]
    Y = Y_available[indices] + sigma_obs * np.random.randn(N, D_Y)
    assert X.shape == (N, D_X)
    assert Y.shape == (N, D_Y)
    assert X_test.shape == (N_test, D_X)
    assert Y_true.shape == (N_test, D_Y)

    # random indices in X_available for validation
    np.random.seed(seed)
    indices_val = np.random.choice(len(X_available), N_val, replace=False)
    X_val = X_available[indices_val]
    Y_val = Y_available[indices_val]
    Y_val = Y_val + sigma_obs * np.random.randn(N_val, D_Y)
    assert X_val.shape == (N_val, D_X)
    assert Y_val.shape == (N_val, D_Y)
    
    return X, Y, X_test, Y_true, X_val, Y_val


sigma = 0.05
X, Y, X_test, Y_true, _, _ = get_data(N=150, D_X=3, N_test=500, sigma_obs=sigma, gap=True)

# plot data
# plt.figure()
# plt.plot(X[:, 1], Y, "r.", label="Train Data", alpha=0.5
# plt.plot(X_val[:,1], Y_val, "bo", lw=2.0, label="True mean")
# plt.plot(X_test[:,1], Y_true, "k--", lw=2.0, label="True mean")
# plt.grid()
# plt.show()


## PARAMETERS
input_size = 3
output_size = 1
hidden_size = 32
model_prob  = 0.1    # Dropout probability
model_lam   = 1e-3   # Regularization coefficient
lr          = 0.001  # Learning rate
seed        = 169      # Random seed
model_path  = f"MCDO/Code/DOMC/models/hidden_size_{hidden_size}_model_prob_{model_prob}_seed_{seed}.pth"

# Convert to PyTorch tensors
X_train_torch = torch.tensor(np.array(X), dtype=torch.float32)
Y_train_torch = torch.tensor(np.array(Y), dtype=torch.float32)
X_test_torch = torch.tensor(np.array(X_test), dtype=torch.float32)

# Initialize TensorBoard
writer = SummaryWriter(f"runs/model_prob_{model_prob}_hidden_size_{hidden_size}_seed_{seed}")



def train_and_save_model():
    model = MCVariationalNN(input_size, hidden_size, output_size, model_prob, model_lam, [2,3])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Dummy training loop
    for epoch in range(10000):

        optimizer.zero_grad()
        outputs = model(X_train_torch)
        loss = criterion(outputs, Y_train_torch)
        loss.backward()
        optimizer.step()
        
        # Log loss to TensorBoard
        writer.add_scalar("Loss/train", loss.item(), epoch)
        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

        if epoch % 100 == 0:
            # Run model in evaluation mode
            model.eval()
            mean_pred, std_pred, preds = mc_inference(model, X_test_torch, n_samples=100)

            # Convert tensors to NumPy (ensuring they are on CPU)
            mean_pred = mean_pred.cpu().numpy()
            std_pred = std_pred.cpu().numpy()
            preds = preds.cpu().numpy().reshape(-1)  # Flatten predictions

            # Log predictions to TensorBoard
            writer.add_scalar("Prediction/mean", mean_pred.mean().item(), epoch)
            writer.add_scalar("Prediction/std", std_pred.mean().item(), epoch)
            writer.add_histogram("Prediction/preds", preds, epoch)


        model.train()
    
    # Save model
    torch.save(model.state_dict(), model_path)
    print("Model saved.")

def load_and_run_model():
    model = MCVariationalNN(input_size, hidden_size, output_size, model_prob, model_lam, [2,3])
    model.load_state_dict(torch.load(model_path))
    model.eval()
    print("Model loaded and running.")
    preds = torch.zeros((100, X_test.shape[0], 1))
    with torch.no_grad():
        for i in range(100):
            preds[i] = model(X_test_torch)
    mean_pred = preds.mean(dim=0)
    std_pred = preds.std(dim=0)
    # Convert to numpy
    mean_pred = mean_pred.numpy()
    std_pred = std_pred.numpy()
    return mean_pred, std_pred, preds

import matplotlib.pyplot as plt
import numpy as np
from torch.utils.tensorboard import SummaryWriter

# Initialize TensorBoard writer
writer = SummaryWriter("runs/numpyro_bnn_logs")


if __name__ == "__main__":
    if not os.path.exists(model_path):
        train_and_save_model()

    if os.path.exists(model_path):
        mean_pred, std_pred, preds = load_and_run_model()
       




writer.close()
