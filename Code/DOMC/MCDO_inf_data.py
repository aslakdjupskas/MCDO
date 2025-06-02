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
        # if not self.training and self.layer in self.dropout_layers:
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
        # self.layer4 = MCVariationalDense(hidden_size, hidden_size, model_prob, model_lam, layer=4, dropout_layers=dropout_layers)
        self.layer4 = MCVariationalDense(hidden_size, output_size, model_prob, model_lam, layer=4, dropout_layers=dropout_layers)
        

    def forward(self, X):
        X = torch.relu(self.layer1(X))
        X = torch.relu(self.layer2(X))
        X = torch.relu(self.layer3(X))  # New layer added
        # X = torch.tanh(self.layer4(X))
        X = self.layer4(X)  # Output layer
        return X

    def regularization(self):
        return (
            self.layer1.regularization() +
            self.layer2.regularization() +
            self.layer3.regularization() +  # Include new layer
            # self.layer4.regularization() +
            self.layer4.regularization()
        )


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



def objective(trial):
    # Sample a dropout rate
    model_prob = trial.suggest_float("dropout_rate", 0.1, 0.8)
    # hidden_size = trial.suggest_int("hidden_size", 16, 1024, step=16)
    # model_lam = trial.suggest_float("model_lam", 1e-5, 5e-3, log=True)
    # lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)

    # Create model
    model = MCVariationalNN(
        input_size=3,
        hidden_size=32,
        output_size=1,
        model_prob=model_prob,
        model_lam=1e-4,
        dropout_layers=[2, 3]
    )

    # Define loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Get fixed training data
    X, Y, _, _, _, _ = get_data(N=100, D_X=3, N_test=500, sigma_obs=sigma, gap=False, sinc_noise_bool=sinc_noise_bool, seed=np.random.randint(0, 1000))
    # X_train_torch = torch.tensor(np.array(X), dtype=torch.float32)
    # Y_train_torch = torch.tensor(np.array(Y), dtype=torch.float32)


    # _, _, X_true, Y_true, _, _ = get_data(N=100, D_X=3, N_test=500, sigma_obs=sigma, gap=False, sinc_noise_bool=sinc_noise_bool, seed=42)
    # Compute true mean and std using sinc_noise
    # Y_mean_true = torch.tensor(np.array(Y_true), dtype=torch.float32).squeeze()

    ys = np.zeros((10000,100))
    xs = np.zeros((10000,100))
    for i in range(ys.shape[0]):
        X, Y, _, _, _, _ = get_data(N=100, D_X=3, N_test=200, sigma_obs=sigma, gap=False, sinc_noise_bool=sinc_noise_bool, seed=i)
        ys[i,:] = np.array(Y[:,0])
        xs[i,:] = np.array(X[:,1])
    Y_std  = ys.std(axis=0)
    Y_mean = ys.mean(axis=0)
    # Make tensor
    Y_std = torch.tensor(np.array(Y_std), dtype=torch.float32)
    Y_mean = torch.tensor(np.array(Y_mean), dtype=torch.float32)
    
    # Track losses
    mean_losses = []
    std_losses = []
    total_losses = []

    for epoch in range(1000):
        model.train()
        optimizer.zero_grad()
        X, Y, _, _, _, _ = get_data(N=100, D_X=3, N_test=200, sigma_obs=sigma, gap=False, sinc_noise_bool=sinc_noise_bool, seed=epoch)
        for i in range(1):
            # Perform MC Inference after training step
            model.eval()
            y_pred_mean, y_pred_std, _ = mc_inference_train(model, N=100, n_samples=500)

            # Compute additional losses
            mean_loss = criterion(y_pred_mean[:, 0], Y_mean) + model.regularization()
            std_loss = criterion(y_pred_std[:, 0], Y_std)

            # Combine all losses
            total_loss = mean_loss + std_loss  

            mean_losses.append(mean_loss.item())
            std_losses.append(std_loss.item())
            total_losses.append(total_loss.item())

            # Backpropagation
            total_loss.backward()
            optimizer.step()

            # Report to Optuna
            trial.report(total_loss.item(), epoch)

            # Check for early stopping
            if trial.should_prune():
                raise optuna.TrialPruned()

    # Store tracked losses for post-analysis
    trial.set_user_attr("mean_losses", mean_losses)
    trial.set_user_attr("std_losses", std_losses)
    trial.set_user_attr("total_losses", total_losses)

    # model_dir = "saved_models"
    # os.makedirs(model_dir, exist_ok=True)
    # model_path = os.path.join(model_dir, f"model_trial_{trial.number}.pt")
    # torch.save(model.state_dict(), model_path)

    # Link model path to trial
    # trial.set_user_attr("model_path", model_path)



    return total_loss.item()


# X, Y, X_test, Y_true, X_val, Y_val = get_data(N=150, D_X=3, N_test=500, sigma_obs=0.35, gap=False, sinc_noise_bool=True)


# # plot data
# plt.figure()
# plt.plot(X[:, 1], Y, "r.", label="Train Data", alpha=0.5)
# plt.plot(X_val[:,1], Y_val, "bo", lw=2.0, label="True mean")
# plt.plot(X_test[:,1], Y_true, "k--", lw=2.0, label="True mean")
# plt.grid()
# plt.show()


## PARAMETERS
input_size = 3
output_size = 1
hidden_size = 32
model_prob  = 0.2    # Dropout probability
model_lam   = 1e-4   # Regularization coefficient
lr          = 0.001  # Learning rate
seed = 416       # Random seed
sigma = 1.35
sinc_noise_bool = True
_, _, X_test, Y_true, _, _ = get_data(N=150, D_X=3, N_test=500, sigma_obs=sigma, gap=False, sinc_noise_bool=sinc_noise_bool)

# # Estimate mean and variance for the data
# plt.figure()
# ys = np.zeros((10000,150))
# for i in range(ys.shape[0]):
#     X, Y, _, _, _, _ = get_data(N=150, D_X=3, N_test=500, sigma_obs=sigma, gap=False, sinc_noise_bool=sinc_noise_bool, seed=i)
#     # _, _, _, _, X, Y = get_data(N=150, D_X=3, N_test=500, sigma_obs=sigma, gap=False, sinc_noise_bool=sinc_noise_bool, seed=i)
#     plt.plot(X[:,1], Y, "r.")
#     ys[i,:] = np.array(Y[:,0])
# plt.plot(X_test[:,1], Y_true[:,0], 'k-')
# # plot ys
# plt.figure()
# # for i in range(1000):
#     # plt.plot(X[:,1], ys[i,:], 'r.')
# plt.plot(X_test[:,1], Y_true[:,0], 'k-')
# # get a dist for Y for each X
# Y_mean = ys.mean(axis=0)
# Y_std  = ys.std(axis=0)
# plt.plot(X[:,1], Y_mean)
# plt.fill_between(
#     X[:,1],
#     (Y_mean - 2 * Y_std).flatten(),
#     (Y_mean + 2 * Y_std).flatten(),
#     color="lightblue",
#     label="95% CI",
#     alpha=0.75
# )
# # plt.show()

# # Run the optimization
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=1)

study.trials_dataframe().to_csv("optuna_trials_relu_trackloss.csv")


best_trial = study.best_trial
dropout_rate = best_trial.params["dropout_rate"]
mean_losses = best_trial.user_attrs["mean_losses"]
std_losses = best_trial.user_attrs["std_losses"]
total_losses = best_trial.user_attrs["total_losses"]

import matplotlib.pyplot as plt

plt.plot(mean_losses, label="Mean Loss")
plt.plot(std_losses, label="Std Loss")
plt.plot(total_losses, label="Total Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid()
plt.title("Loss components over training epochs")
plt.savefig("MCDO/Code/DOMC/plots/loss_components.pdf")
# plt.show()


# # Save the study
# study.trials_dataframe().to_csv("optuna_trials_relu.csv")
# Load the study
optuna_df = pd.read_csv("optuna_trials.csv")
# optimal = optuna_df.loc[14,["value", 'params_dropout_rate', 'params_model_lam']]
# study = optuna.load_study(study_name="optuna_trials.csv", storage="sqlite:///example.db")

# Convert to PyTorch tensors
X_test_torch = torch.tensor(np.array(X_test), dtype=torch.float32)

# Subplot 4x4
plt.figure(figsize=(15, 15))
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

dropout_rates_dict = {
    0.1:None,
    }
trials = [0]

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



best_trial = study.best_trial
model_path = best_trial.user_attrs["model_path"]

# Recreate the model architecture
best_model = MCVariationalNN(
    input_size=3,
    hidden_size=32,
    output_size=1,
    model_prob=best_trial.params["dropout_rate"],  # Use best dropout rate
    model_lam=1e-4,
    dropout_layers=[2, 3]
)

# Load saved weights
best_model.load_state_dict(torch.load(model_path))
best_model.eval()
mean_pred, std_pred, predictions = mc_inference(best_model, X_test_torch, n_samples=5000)

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
plt.fill_between(
    X_test[:, 1],
    (Y_true[:,0] - 2 * std_shot_noise).flatten(),
    (Y_true[:,0] + 2 * std_shot_noise).flatten(),
    color="orange",
    label="95% CI training data",
    alpha=0.5

)

plt.plot(X_test[:,1], mean_pred, label="MC Mean Prediction", color="blue")
for i in range(250):
    plt.plot(X_test[:,1], predictions[1000+i], color="gray", alpha=0.05)
plt.fill_between(
    X_test[:,1],
    (mean_pred - 2 * std_pred).flatten(),
    (mean_pred + 2 * std_pred).flatten(),
    color="lightblue",
    label="95% CI prediction",
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
# plt.savefig(f"MCDO/Code/DOMC/plots/MCDO_pytorch_sinc_unlimdata_RegPoster.pdf")
# plt.savefig(f"MCDO/Code/DOMC/plots/MCDO_pytorch_sinc_unlimdata_Reg{model_prob}.pdf")

# plt.show()


# plot ys and xs as datapoints
# plt.figure()
# for i in range(200):
#     plt.plot(xs[i,:], ys[i,:], 'r.')
# plt.grid()
# # plt.title("Data points")
# plt.xlabel("X")
# plt.ylabel("Y")
# plt.savefig("data_points_constant.pdf")
# plt.show()
plt.figure(figsize=(8,5))
plt.grid(True)
for idx, trial in enumerate(trials):
    
    # random seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    # optuna_df = pd.read_csv("optuna_trials.csv")
    # optimal = optuna_df.loc[trial,["value", 'params_dropout_rate', 'params_model_lam']]
    # model_prob  = optimal['params_dropout_rate']    # Dropout probability
    # model_lam   = optimal['params_model_lam'] 
    model_prob  = dropout_rate    # Dropout probability

    # Initialize model, loss function, and optimizer
    model = MCVariationalNN(input_size, hidden_size, output_size, model_prob, model_lam, [2,3])
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    validation_error = []
    loss_history = []
    patience = 30
    min_delta = 1e-10  # Minimum change in loss to qualify as improvement
    best_val_loss = float('inf')
    epochs_no_improve = 0

    # Training
    variance_loss = []
    epochs = 10000
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        
        X, Y, _, _, _, _ = get_data(N=100, D_X=3, N_test=500, sigma_obs=sigma, gap=False, sinc_noise_bool=sinc_noise_bool, seed=epoch)
        X_train_torch = torch.tensor(np.array(X), dtype=torch.float32)
        Y_train_torch = torch.tensor(np.array(Y), dtype=torch.float32)

        # plt.plot(X_train_torch[:,1], Y_train_torch, "r.", alpha=0.5)

        outputs = model(X_train_torch)
        loss = criterion(outputs, Y_train_torch) + model.regularization()
        y_pred_mean, y_pred_std, _ = mc_inference_train(model, N=100, n_samples=500)
        std_loss = criterion(y_pred_std[:, 0], Y_std)

        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            model.eval()  # Set model to evaluation mode
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


    # Measure size of std_pred
    dropout_rates_dict[model_prob] = std_pred.mean()

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
    plt.fill_between(
        X_test[:, 1],
        (Y_true[:,0] - 2 * std_shot_noise).flatten(),
        (Y_true[:,0] + 2 * std_shot_noise).flatten(),
        color="orange",
        label="95% CI training data",
        alpha=0.5

    )

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

plt.show()

# Extract keys (dropout rates) and values (uncertainty measures)
dropout_rates = list(dropout_rates_dict.keys())
uncertainties = list(dropout_rates_dict.values())

# Bar plot
plt.figure(figsize=(8,5))
plt.bar(dropout_rates, uncertainties, width=0.05, color='b', alpha=0.7, edgecolor='black')

# Labels and title
plt.xlabel("Dropout Rate")
plt.ylabel("Uncertainty Measure")
plt.title("Uncertainty vs. Dropout Rate")
plt.xticks(dropout_rates)  # Ensure correct ticks on x-axis
plt.grid(axis='y', linestyle='--', alpha=0.7)
# plt.savefig(f"MCDO/Code/DOMC/plots/MCDO_pytorch_unlimdata_DOvsuncertainty.pdf")
plt.savefig(f"MCDO/Code/DOMC/plots/MCDO_pytorch_sinc_seed2.pdf")
plt.show()

print("Done")