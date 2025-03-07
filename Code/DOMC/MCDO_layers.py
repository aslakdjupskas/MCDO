import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import jax.numpy as jnp
import numpy as np

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
        if self.training or self.layer in self.dropout_layers:
            model_W = self.model_M * torch.bernoulli(torch.full_like(self.model_M, 1 - self.dropout.p))       
        else:
            model_W = self.model_M
        output = torch.mm(X, model_W) + self.model_m
        return output

    def regularization(self):
        return self.model_lam * (
            self.model_prob * torch.sum(self.model_M ** 2) + torch.sum(self.model_m ** 2)
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



np.random.seed(0)
torch.manual_seed(0)

def get_data(N=50, D_X=3, sigma_obs=0.05, N_test=500, gap=True):
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


X, Y, X_test, Y_true = get_data(N=150, D_X=3, N_test=500, gap=True)
input_size = 3
output_size = 1

# Convert to PyTorch tensors
X_train_torch = torch.tensor(np.array(X), dtype=torch.float32)
Y_train_torch = torch.tensor(np.array(Y), dtype=torch.float32)
X_test_torch = torch.tensor(np.array(X_test), dtype=torch.float32)

# Subplot 3x3
plt.figure(figsize=(15, 15))
plt.title("MC Dropout on different layers\nDropout probability: 0.1\n\n")
plt.xticks([])
plt.yticks([])
plt.box(False)

dropout_layers = {
    'All layers':[1, 2, 3, 4],
    'First layer':[1],
    'Second layer':[2],
    'Third layer':[3],
    'Last layer':[4],
    "Middle layers":[2, 3],
    'First two layers':[1, 2],
    'Last two layers':[3, 4],
    'First and last layer':[1, 4]
}

for idx, (name, layers) in enumerate(dropout_layers.items()):


    hidden_size = 64
    model_prob  = 0.1   # Dropout probability
    model_lam   = 0.01  # Regularization coefficient
    lr          =  0.015 # Learning rate

    # Initialize model, loss function, and optimizer
    model = MCVariationalNN(input_size, hidden_size, output_size, model_prob, model_lam, layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    epochs = 10000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_torch)
        loss = criterion(outputs, Y_train_torch) + model.regularization()
        loss.backward()
        optimizer.step()
        
        if epoch % 10000 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # MC Dropout Inference
    model.eval()

    mean_pred, std_pred, predictions = mc_inference(model, X_test_torch, n_samples=100000)

    # Convert to numpy
    mean_pred = mean_pred.numpy()
    std_pred = std_pred.numpy()

    # plot in subplot   
    plt.subplot(3, 3, idx+1)
    plt.plot(X[:, 1], Y, "r.", label="Train Data", alpha=0.5)
    plt.plot(X_test[:,1], mean_pred, label="MC Mean Prediction", color="blue")
    for i in range(250):
        plt.plot(X_test[:,1], predictions[5000+i], color="gray", alpha=0.05)
    plt.fill_between(
        X_test[:,1],
        (mean_pred - 2 * std_pred).flatten(),
        (mean_pred + 2 * std_pred).flatten(),
        color="lightblue",
        label="95% CI",
    )
    plt.plot(X_test[:,1], Y_true, "k--", lw=2.0, label="True mean")
    plt.grid()
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim(-4, 4)
    plt.title(f"\nDropout on: {name}")
plt.tight_layout()

# plt.show()

plt.savefig("MCDO/Code/DOMC/plots/MCDO_pytorch_layersdiffer.pdf")
print("Done")