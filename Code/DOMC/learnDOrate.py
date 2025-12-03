import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import torch.optim as optim
import jax.numpy as jnp
import numpy as np
import optuna
import pandas as pd
import os

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


import torch
import torch.nn as nn

# Variational Dense Layer with Learnable Dropout Probability
class MCVariationalDense(nn.Module):
    def __init__(self, n_in, n_out, model_prob, model_lam, layer, dropout_layers):
        super(MCVariationalDense, self).__init__()
        # self.model_prob = nn.Parameter(torch.tensor(model_prob))  # Directly learnable prob
        self.model_lam = model_lam
        self.model_prob = model_prob  # Fixed dropout probability
        self.model_M = nn.Parameter(torch.randn(n_in, n_out) * 0.1)
        self.model_m = nn.Parameter(torch.zeros(n_out))
        self.layer = layer
        self.dropout_layers = dropout_layers

    def forward(self, X):
        # Clamp to avoid numerical issues

        if not self.training and self.layer in self.dropout_layers:
            p = float(1 - self.model_prob.item())  # only works if scalar
            p = max(0.0, min(1.0, p))              # clamp manually
            mask = torch.bernoulli(torch.full_like(self.model_M, p))

            model_W = self.model_M * mask / (1 - self.model_prob)
        elif not self.training and self.layer in self.dropout_layers:
            model_W = self.model_M * (1 - self.model_prob)
        else:
            model_W = self.model_M

        output = torch.mm(X, model_W) + self.model_m
        return output

    def regularization(self):
        return self.model_lam * (torch.sum(self.model_M ** 2) + torch.sum(self.model_m ** 2))


# Variational Neural Network
class MCVariationalNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model_prob, model_lam, dropout_layers):
        super(MCVariationalNN, self).__init__()
        self.dropout_layers = dropout_layers
        self.model_prob = nn.Parameter(torch.tensor(model_prob))
        self.layer1 = MCVariationalDense(input_size, hidden_size, self.model_prob, model_lam, layer=1, dropout_layers=dropout_layers)
        self.layer2 = MCVariationalDense(hidden_size, hidden_size, self.model_prob, model_lam, layer=2, dropout_layers=dropout_layers)
        self.layer3 = MCVariationalDense(hidden_size, hidden_size, self.model_prob, model_lam, layer=3, dropout_layers=dropout_layers)
        self.layer4 = MCVariationalDense(hidden_size, output_size, self.model_prob, model_lam, layer=4, dropout_layers=dropout_layers)

    def forward(self, X):
        X = torch.relu(self.layer1(X))
        X = torch.relu(self.layer2(X))
        X = torch.relu(self.layer3(X))
        X = self.layer4(X)  # Output layer
        return X

    def regularization(self):
        return (
            self.layer1.regularization() +
            self.layer2.regularization() +
            self.layer3.regularization() +
            self.layer4.regularization()
        )



# MC Variational Dense Layer
# class MCVariationalDense(nn.Module):
#     def __init__(self, n_in, n_out, init_model_prob, model_lam, layer, dropout_layers):
#         super(MCVariationalDense, self).__init__()
#         self.logit_model_prob = nn.Parameter(torch.tensor(init_model_prob))
#         self.model_lam = model_lam
#         self.model_M = nn.Parameter(torch.randn(n_in, n_out) * 0.1)
#         self.model_m = nn.Parameter(torch.zeros(n_out))
#         self.layer = layer
#         self.dropout_layers = dropout_layers

#     # def forward(self, X):
#     #     model_prob = torch.sigmoid(self.logit_model_prob)

#     #     if not self.training and self.layer in self.dropout_layers:
#     #         mask = torch.bernoulli(torch.full_like(self.model_M, 1 - model_prob))
#     #         model_W = self.model_M * mask

#     #     else:
#     #         model_W = self.model_M

#     #     output = torch.mm(X, model_W) + self.model_m
#     #     return output

#     def forward(self, X):
#         model_prob = self.logit_model_prob

#         if self.training and self.layer in self.dropout_layers:
#             mask = torch.bernoulli(torch.full_like(self.model_M, (1 - model_prob).item()))
#             model_W = self.model_M * mask / (1 - model_prob)  # Scale for MC dropout
#         elif not self.training and self.layer in self.dropout_layers:
#             model_W = self.model_M * (1 - model_prob)
#         else:
#             model_W = self.model_M

#         output = torch.mm(X, model_W) + self.model_m
#         return output


#     def regularization(self):
#         return self.model_lam * (torch.sum(self.model_M ** 2) + torch.sum(self.model_m ** 2))


# class MCVariationalNN(nn.Module):
#     def __init__(self, input_size, hidden_size, output_size, model_prob, model_lam, dropout_layers):
#         super(MCVariationalNN, self).__init__()
#         self.dropout_layers = dropout_layers
#         self.layer1 = MCVariationalDense(input_size, hidden_size, model_prob, model_lam, layer=1, dropout_layers=dropout_layers)
#         self.layer2 = MCVariationalDense(hidden_size, hidden_size, model_prob, model_lam, layer=2, dropout_layers=dropout_layers)
#         self.layer3 = MCVariationalDense(hidden_size, hidden_size, model_prob, model_lam, layer=3, dropout_layers=dropout_layers)
#         # self.layer4 = MCVariationalDense(hidden_size, hidden_size, model_prob, model_lam, layer=4, dropout_layers=dropout_layers)
#         self.layer4 = MCVariationalDense(hidden_size, output_size, model_prob, model_lam, layer=4, dropout_layers=dropout_layers)
        

#     def forward(self, X):
#         X = torch.relu(self.layer1(X))
#         X = torch.relu(self.layer2(X))
#         X = torch.relu(self.layer3(X))  # New layer added
#         # X = torch.tanh(self.layer4(X))
#         X = self.layer4(X)  # Output layer
#         return X

#     def regularization(self):
#         return (
#             self.layer1.regularization() +
#             self.layer2.regularization() +
#             self.layer3.regularization() +  # Include new layer
#             # self.layer4.regularization() +
#             self.layer4.regularization()
#         )


def mc_inference_train(model, N, n_samples=100):
    model.eval()  # Keep dropout active
    preds = torch.zeros((n_samples, N, 1))
    with torch.no_grad():
        for i in range(n_samples):
            X, _, _, _, _, _ = get_data(N=100, D_X=3, N_test=500, sigma_obs=sigma, gap=False, sinc_noise_bool=sinc_noise_bool, seed=i)
            X_train_torch = torch.tensor(np.array(X), dtype=torch.float32)
            preds[i] = model(X_train_torch)
    return preds.mean(dim=0), preds.std(dim=0), preds

# MC Dropout inference
def mc_inference(model, X, n_samples=100):
    model.eval()  # Keep dropout active
    preds = torch.zeros((n_samples, X.size(0), 1))
    with torch.no_grad():
        for i in range(n_samples):
            preds[i] = model(X)
    return preds.mean(dim=0), preds.std(dim=0), preds


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
        Y = Y_available[indices] + sigma_obs * sinc_noise[:, None] * np.random.randn(N, D_Y)
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


## PARAMETERS
input_size = 3
output_size = 1
hidden_size = 32
model_prob  = 0.39    # Dropout probability
model_lam   = 0.00069   # Regularization coefficient
lr          = 0.001  # Learning rate
seed = 416       # Random seed
sigma = 0.35
sinc_noise_bool = True
_, _, X_test, Y_true, _, _ = get_data(N=150, D_X=3, N_test=500, sigma_obs=sigma, gap=False, sinc_noise_bool=sinc_noise_bool)



# # Save the study
# study.trials_dataframe().to_csv("optuna_trials_relu.csv")
# Load the study
# optuna_df = pd.read_csv("optuna_trials.csv")
# optimal = optuna_df.loc[14,["value", 'params_dropout_rate', 'params_model_lam']]
# study = optuna.load_study(study_name="optuna_trials.csv", storage="sqlite:///example.db")

# Convert to PyTorch tensors
X_test_torch = torch.tensor(np.array(X_test), dtype=torch.float32)

# Subplot 4x4
plt.figure()
# plt.title(f"MC Dropout with different seeds\nDropout probability: {model_prob}\n\n")
plt.xticks([])
plt.yticks([])
plt.box(False)


mean_fcn = {
    # 'Middle layers':[2, 3],
    'All layers':None,
    'First layer':None,
    'Second layer':None,
    'Third layer':None,
    'Last layer':None,
    'Middle layers':None,
    'First two layers':None,
    'Last two layers':None,
    'First and last layer':None,
}

# 16, seeds 
seeds = [12, 123, 23, 234, 21, 321, 32, 11, 22, 33, 44, 55, 12772, 5543, 12356, 66642]
# seeds = [i for i in range(50)]
# 9 seeds
# seeds = [20, 21, 22, 23, 24, 52, 62, 27, 82]


ys = np.zeros((200,100))
xs = np.zeros((200,100))
for i in range(ys.shape[0]):
    X, Y, _, _, _, _ = get_data(N=100, D_X=3, N_test=500, sigma_obs=sigma, gap=False, sinc_noise_bool=sinc_noise_bool, seed=i)
    ys[i,:] = np.array(Y[:,0])
    xs[i,:] = np.array(X[:,1])
Y_std  = ys.std(axis=0)
Y_mean = ys.mean(axis=0)
# Make tensor
Y_std = torch.tensor(np.array(Y_std), dtype=torch.float32)
Y_mean = torch.tensor(np.array(Y_mean), dtype=torch.float32)

model_prob = 0.39


plt.figure(figsize=(8,5))
plt.grid(True)
for idx, trial in enumerate([1]):
    
    # random seed
    np.random.seed(seed)
    torch.manual_seed(seed)


    # Initialize model, loss function, and optimizer
    model = MCVariationalNN(input_size, hidden_size, output_size, model_prob, model_lam, [2,3])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    validation_error = []
    loss_history = []
    patience = 10
    min_delta = 1e-10  # Minimum change in loss to qualify as improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training
    variance_loss = []
    mean_loss = []
    epochs = 250
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        X, Y, _, _, _, _ = get_data(N=100, D_X=3, N_test=500, sigma_obs=sigma, gap=False, sinc_noise_bool=sinc_noise_bool, seed=epoch)
        X_train_torch = torch.tensor(np.array(X), dtype=torch.float32)
        Y_train_torch = torch.tensor(np.array(Y), dtype=torch.float32)

        # plt.plot(X_train_torch[:,1], Y_train_torch, "r.", alpha=0.5)

        outputs = model(X_train_torch)
        y_pred_mean, y_pred_std, _ = mc_inference_train(model, N=100, n_samples=100)
        std_loss = criterion(y_pred_std[:, 0], Y_std)
        variance_loss.append(std_loss.item())
        mean_loss.append(criterion(y_pred_mean[:, 0], Y_mean).item())

        loss = criterion(outputs, Y_train_torch) + model.regularization() + 0.1 *std_loss * model.model_prob

        loss.backward()
        optimizer.step()
        
        if epoch % 10 == 0:
            print(model.model_prob)
            model.eval()  # Set model to evaluation mode
            # print(model.layer3.logit_model_prob)
            with torch.no_grad():  # Disable gradient computation
                _, _, _, _, X_val, Y_val = get_data(N=100, D_X=3, N_test=500, sigma_obs=sigma, gap=False, sinc_noise_bool=sinc_noise_bool, seed=epoch)
                val_outputs = model(torch.tensor(np.array(X_val), dtype=torch.float32))
                val_loss = criterion(val_outputs, torch.tensor(np.array(Y_val), dtype=torch.float32))
                validation_error.append(val_loss.item())
                loss_history.append(loss.item())


            print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

            # Check early stopping condition
            if val_loss < best_val_loss - min_delta:
                best_val_loss = val_loss
                epochs_no_improve = 0  # Reset counter
            else:
                epochs_no_improve += 1  # Increment counter
                print(epochs_no_improve)

            if epochs_no_improve >= patience and epoch > 2*patience:
                print(f"Early stopping at epoch {epoch}, best validation loss: {best_val_loss:.4f}")
                break  # Stop training
            

    # plt.plot(X_test[:,1], Y_true[:,0], "k--", lw=2.0, label="True mean")
    # plt.grid()
    # plt.legend()
    # plt.show()
    # MC Dropout Inference
    # torch.save(model.state_dict(), f"MCDO/Code/DOMC/models/model_lambda{model_lam}_doRate{model_prob}_seed{seed}_noise{sigma}.pth")
    model.eval()

    # plt.figure()
    # plt.plot(validation_error, label="Validation Loss")
    # plt.plot(loss_history, label="Training loss")
    # plt.title(f"Training error for dropout rate {dr}")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.yscale("log")
    # plt.legend()
    # plt.grid()
    # plt.show()

    mean_pred, std_pred, predictions = mc_inference(model, X_test_torch, n_samples=10000)


    # Convert to numpy
    mean_pred = mean_pred.numpy()
    std_pred = std_pred.numpy()

    # plot in subplot   
    # plt.subplot(2, 2, idx+1)
    # plt.plot(X[:, 1], Y, "r.", label="Train Data", alpha=0.5)# Compute fixed shot noise standard deviation
    std_shot_noise_fixed = (sigma * np.random.randn(10000, 1)).std()


    if sinc_noise_bool:
        # Compute sinc function values (avoid division by zero at x=0)
        sinc_values = np.sinc(X_test[:, 1])  # sinc(x) = sin(pi*x) / (pi*x)

        # Scale sinc values to match noise scale
        std_shot_noise = std_shot_noise_fixed * sinc_values
    else:
        std_shot_noise = std_shot_noise_fixed * np.ones_like(X_test[:, 1])

    # mask for zero out std_shot_noise when x>1 or x<-1
    mask = (X_test[:, 1] < -1) | (X_test[:, 1] > 1)
    std_shot_noise[mask] = 0


    plt.plot(X_test[:,1], mean_pred, label="MC Mean Prediction", color="blue")
    for i in range(250):
        plt.plot(X_test[:,1], predictions[5000+i], color="gray", alpha=0.05)
    plt.fill_between(
        X_test[:,1],
        (mean_pred - 2 * std_pred).flatten(),
        (mean_pred + 2 * std_pred).flatten(),
        color="lightblue",
        label="95% CI prediction",
        alpha=0.5
    )
    plt.fill_between(
        X_test[:, 1],
        (Y_true[:,0] - 2 * std_shot_noise).flatten(),
        (Y_true[:,0] + 2 * std_shot_noise).flatten(),
        color="orange",
        label="95% CI training data",
        alpha=0.5

    )
    plt.plot(X_test[:,1], Y_true, "k--", lw=2.0, label="True mean")
    plt.grid(True)
    plt.legend()
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.ylim(-4, 4)
    # plt.title(f"Dropout rate: {model_prob:.2f}| Lambda: {model_lam:.5f}")
plt.tight_layout()
plt.savefig(f"MCDO/Code/DOMC/plots/MCDO_pytorch_sinc_unlimdata_RegPoster.pdf")
# plt.savefig(f"MCDO/Code/DOMC/plots/MCDO_pytorch_sinc_unlimdata_Reg{model_prob}.pdf")

# plot loss history
plt.figure(figsize=(8, 5))
plt.plot(variance_loss, label="Variance Loss")
# plt.plot(mean_loss, label="Mean Loss")
plt.title(f"Training error for dropout rate {model_prob:.2f} | Lambda: {model_lam:.5f}")
plt.xlabel("Epoch")
plt.ylabel("Loss")
# plt.yscale("log")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig(f"MCDO/Code/DOMC/plots/MCDO_pytorch_sinc_unlimdata_RegPoster_loss.pdf")


plt.show()



print("Done")