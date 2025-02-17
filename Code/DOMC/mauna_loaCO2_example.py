import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import requests
import io

class MCVariationalDense(nn.Module):
    """Variational Dense Layer with MC Dropout"""
    def __init__(self, n_in, n_out, model_prob, model_lam):
        super(MCVariationalDense, self).__init__()
        self.model_prob = model_prob
        self.model_lam = model_lam
        self.dropout = nn.Dropout(p=1 - self.model_prob)
        self.model_M = nn.Parameter(torch.randn(n_in, n_out) * 0.01)
        self.model_m = nn.Parameter(torch.zeros(n_out))

    def forward(self, X):
        # Apply dropout mask directly to the weights
        model_W = self.dropout(self.model_M)
        output = torch.mm(X, model_W) + self.model_m
        return output

    def regularization(self):
        return self.model_lam * (
            self.model_prob * torch.sum(self.model_M ** 2) + torch.sum(self.model_m ** 2)
        )


# Model definition using MCVariationalDense
class MCVariationalNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model_prob, model_lam):
        super(MCVariationalNN, self).__init__()
        self.layer1 = MCVariationalDense(input_size, hidden_size, model_prob, model_lam)
        self.layer2 = MCVariationalDense(hidden_size, hidden_size, model_prob, model_lam)
        self.layer3 = MCVariationalDense(hidden_size, output_size, model_prob, model_lam)


    def forward(self, X):
        X = torch.relu(self.layer1(X))
        X = torch.relu(self.layer2(X))
        X = self.layer3(X)
        return X
    

    def regularization(self):
        return (
            self.layer1.regularization() +
            self.layer2.regularization() +
            self.layer3.regularization()
        )
# MC Dropout inference
def mc_inference(model, X, n_samples=100):
    model.train()  # Enable dropout during inference
    preds = torch.zeros((n_samples, X.size(0), 10))
    for i in range(n_samples):
        preds[i] = model(X)
    return preds.mean(dim=0), preds.std(dim=0), preds  # Mean and uncertainty    

data = pd.read_csv('Code/MLO/co2_daily_mlo.csv')
data['date'] = pd.to_datetime(data[['Year', 'Month', 'Day']])
data.set_index('date', inplace=True)


# Create lagged features
data['Lag_1'] = data['Interpolated'].shift(1)
data = data.dropna()

# Normalize the data
scaler = MinMaxScaler()
data[['Interpolated', 'Lag_1']] = data[['Interpolated', 'Lag_1']].fillna(0)

# Split into training and testing sets
X = data[['Lag_1']].values
y = data['Interpolated'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).unsqueeze(1)

# Define the model
input_size = 1
hidden_size = 10
output_size = 1
model_prob = 0.9
model_lam = 0.01

model = MCVariationalNN(input_size, hidden_size, output_size, model_prob, model_lam)

# Define optimizer and loss function
optimizer = optim.Adam(model.parameters(), lr=0.001)
loss_fn = nn.MSELoss()

# Training loop
n_epochs = 100
for epoch in range(n_epochs):
    model.train()
    optimizer.zero_grad()
    y_pred = model(X_train)
    loss = loss_fn(y_pred, y_train) + model.regularization()
    loss.backward()
    optimizer.step()

    # Print progress
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch + 1}/{n_epochs}, Loss: {loss.item()}")

# Evaluation
model.eval()
n_mc_samples = 100

# Perform MC Dropout on test data
mc_predictions = torch.stack([model(X_test) for _ in range(n_mc_samples)])
mean_prediction = mc_predictions.mean(dim=0)
uncertainty = mc_predictions.std(dim=0)

# Convert predictions back to the original scale
mean_prediction = scaler.inverse_transform(mean_prediction.detach().numpy())
uncertainty = scaler.inverse_transform(uncertainty.detach().numpy())


plt.figure(figsize=(10, 6))
plt.plot(data['Date'][-len(y_test):], scaler.inverse_transform(y_test), label="True Values")
plt.plot(data['Date'][-len(y_test):], mean_prediction, label="Mean Prediction")
plt.fill_between(
    data['Date'][-len(y_test):],
    mean_prediction.squeeze() - 2 * uncertainty.squeeze(),
    mean_prediction.squeeze() + 2 * uncertainty.squeeze(),
    color='gray',
    alpha=0.3,
    label="Uncertainty (95% CI)"
)
plt.legend()
plt.title("MC Dropout Predictions with Uncertainty")
plt.xlabel("Date")
plt.ylabel("CO2 Emissions")
plt.show()
print("Done")