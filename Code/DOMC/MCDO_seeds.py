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

def get_data(N=50, D_X=3, sigma_obs=0.05, N_test=500, N_val=20, gap=True, seed=0):
    D_Y = 1  # Create 1D outputs
    np.random.seed(0)
    
    # Generate test data first
    X_test = jnp.linspace(-1.3, 1.3, N_test)
    X_test = jnp.power(X_test[:, np.newaxis], jnp.arange(D_X))
    
    # Define ground truth function
    W = 0.5 * np.random.randn(D_X)
    Y_true = jnp.dot(X_test, W) + 0.5 * jnp.power(0.5 + X_test[:, 1], 2.0) * jnp.sin(4.0 * X_test[:, 1])
    
    # Y_true =  jnp.power(0.5 + X_test[:, 1], 2.0) * jnp.sin(4.0 * X_test[:, 1])
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
X, Y, X_test, Y_true, _, _ = get_data(N=50, D_X=3, N_test=500, sigma_obs=sigma, gap=True)

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
model_prob  = 0.15    # Dropout probability
model_lam   = 1e-4   # Regularization coefficient
lr          = 0.001  # Learning rate



# Convert to PyTorch tensors
X_train_torch = torch.tensor(np.array(X), dtype=torch.float32)
Y_train_torch = torch.tensor(np.array(Y), dtype=torch.float32)
X_test_torch = torch.tensor(np.array(X_test), dtype=torch.float32)

# # Subplot 4x4
# plt.figure(figsize=(15, 15))
# # plt.title(f"MC Dropout with different seeds\nDropout probability: {model_prob}\n\n")
# plt.xticks([])
# plt.yticks([])
# plt.box(False)


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
# seeds = [12, 123, 23, 234, 21, 321, 32, 11, 22, 33, 44, 55, 12772, 5543, 12356, 66642]
# seeds = [i for i in range(50)]
# 9 seeds
# seeds = [32, 21, 123456, 66642, 294, 5932, 24456, 68998, 3422] #, 24, 52, 62, 27, 82]
predictions_all = []
MSEs = []
MSEs_std = []
# seeds = [11, 22, 33, 44]
# seeds = [21, 32, 43, 54]
seeds = [5, 6, 7, 8]
seeds = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99]

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
        optimizer.step()
        
        if epoch % 1000 == 0:
            model.eval()  # Set model to evaluation mode
            # with torch.no_grad():  # Disable gradient 
                # seed = np.random.randint(0, 10000)
                # _, _, _, _, X_val, Y_val = get_data(N=150, D_X=3, N_test=500, gap=True, seed=seed)
                # np.random.seed(seed)
                # torch.manual_seed(seed)

                # # TODO: Add read training data
                # val_outputs = model(torch.tensor(np.array(X_val), dtype=torch.float32))
                # val_loss = criterion(val_outputs, torch.tensor(np.array(Y_val), dtype=torch.float32))
                # validation_error.append(val_loss.item())
                # loss_history.append(loss.item())

            # print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}")

            # # Check early stopping condition
            # if val_loss < best_val_loss - min_delta:
            #     best_val_loss = val_loss
            #     epochs_no_improve = 0  # Reset counter
            # else:
            #     epochs_no_improve += 1  # Increment counter

            # if epochs_no_improve >= patience and epoch > 2*patience:
            #     print(f"Early stopping at epoch {epoch}, best validation loss: {best_val_loss:.4f}")
            #     break  # Stop training
            
            # validation_error.append()
    # MC Dropout Inference
    # torch.save(model.state_dict(), f"MCDO/Code/DOMC/models/model_lambda{model_lam}_doRate{model_prob}_seed{seed}_noise{sigma}.pth")
    model.eval()

    # plt.figure()
    # plt.plot(validation_error, label="Validation Loss")
    # plt.plot(loss_history, label="Training loss")
    # plt.title(f"Training error for {name}")
    # plt.xlabel("Epoch")
    # plt.ylabel("Loss")
    # plt.yscale("log")
    # plt.legend()
    # plt.grid()
    # plt.show()


    mean_pred, std_pred, predictions = mc_inference(model, X_test_torch, n_samples=20000)
    # print(f"Mean MSE error: {np.mean((np.array(mean_pred) - np.array(Y_true))**2)}\n seed: {seed}")
    # print(f"Mean STD error: {np.mean((np.array(std_pred) - np.ones_like(std_pred)*sigma)**2)} \n seed: {seed}")



    predictions_all.append(predictions)

    # Convert to numpy
    mean_pred = mean_pred.numpy()
    std_pred = std_pred.numpy()

    MSEs.append(np.mean((mean_pred - Y_true)**2))
    # sigma_obs is the noise added to the data
    MSEs_std.append(np.mean((std_pred - np.ones_like(std_pred)*sigma)**2))

    # plot in subplot   
    # plt.subplot(2, 2, idx+1)
    # plt.plot(X[:, 1], Y, "r.", label="Train Data", alpha=0.5)
    # plt.plot(X_test[:,1], mean_pred, label="MC Mean Prediction", color="blue")
    # for i in range(250):
    #     plt.plot(X_test[:,1], predictions[5000+i], color="gray", alpha=0.05)
    # plt.fill_between(
    #     X_test[:,1],
    #     (mean_pred - 2 * std_pred).flatten(),
    #     (mean_pred + 2 * std_pred).flatten(),
    #     color="lightblue",
    #     label="95% CI",
    # )
    # plt.plot(X_test[:,1], Y_true, "k--", lw=2.0, label="True mean")
    # plt.grid()
    # plt.legend()
    # plt.xlabel("X")
    # plt.ylabel("Y")
    # plt.ylim(-4, 4)
    # # plt.title(f"Seed {seed}\n")
    # print(i+1)
# plt.tight_layout()
# plt.savefig(f"MCDO/Code/DOMC/plots/MCDO_pytorch_SEEDSdifferDO{model_prob}_Reg{model_lam}_NoDoTraining.pdf")
# plt.savefig(f"MCDO/Code/DOMC/plots/MCDO_pytorch_SEEDSdifferDO{model_prob}_Reg{model_lam}_sigma{sigma}_report_paper.pdf")
# plt.show()



# plot all predictions mean, std, and samples from predictions_all
predictions_all = np.array(predictions_all)
predictions_all = predictions_all.reshape(predictions_all.shape[0]*predictions_all.shape[1], predictions_all.shape[2])
mean_pred = np.mean(predictions_all, axis=0)
std_pred = np.std(predictions_all, axis=0)
plt.figure(figsize=(8, 5))
plt.plot(X_test[:, 1], Y_true[:,0], "k--", lw=2.0, label="True mean")
plt.plot(X[:, 1], Y[:,0], "r.", label="Train Data", alpha=0.5)
plt.plot(X_test[:, 1], mean_pred, label="MC Mean Prediction", color="blue")
n = predictions_all.shape[0]
for i in range(0, n, int(np.ceil(n/150))):
    plt.plot(X_test[:,1], predictions_all[i], color="gray", alpha=0.05)
plt.fill_between(
    X_test[:, 1],
    (mean_pred - 2 * std_pred).flatten(),
    (mean_pred + 2 * std_pred).flatten(),
    color="lightblue",
    label="95% CI",
)
plt.grid()
plt.legend()
plt.xlabel("X")
plt.ylabel("Y")
# plt.title("MC Dropout Ensemble seeds")
# plt.savefig(f"MCDO/Code/DOMC/plots/mergedModels{model_prob}_Reg{model_lam}_NoDoTraining.pdf")
# plt.savefig(f"MCDO/Code/DOMC/plots/mergedModels{model_prob}_Reg{model_lam}4x4.pdf")
# plt.savefig(f"MCDO/Code/DOMC/plots/Ensemble_mergedModels{model_prob}_Reg{model_lam}_sigma{sigma}_report_seedis2.pdf")
plt.savefig(f"MCDO/Code/DOMC/plots/Ensemble_mergedModels{model_prob}_Reg{model_lam}_sigma{sigma}ENSAMBLE_0DO.pdf")
#  plt.show()

print(f"Mean MSE error: {np.array(MSEs).mean()}")
print(f"Mean MSE error std: {np.array(MSEs).std()}")

print(f"Mean MSE std error: {np.array(MSEs_std).mean()}")
print(f"Mean MSE std error std: {np.array(MSEs_std).std()}")

print("Done")