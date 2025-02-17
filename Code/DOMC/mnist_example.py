import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import numpy as np

def plot_sample_prediction(image, mean_list, std_list, correct, savefig=None):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(image, cmap='gray')
    plt.title(f"Original MNIST Image (Correct: {correct})")
    plt.axis('off')

    # Plotting the predictions with uncertainty
    plt.subplot(1, 2, 2)
    digits = np.arange(10)
    plt.bar(digits, mean_list, yerr=std_list, capsize=5, color='blue', alpha=0.7)
    plt.xticks(digits)
    plt.xlabel("Digits")
    plt.ylabel("Prediction Probability")
    plt.title("Predictions with Uncertainty")
    plt.tight_layout()

    if savefig:
        plt.savefig(f"Code/plots/{savefig}")
    else:
        plt.show()

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


# Load and preprocess MNIST dataset
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root="./data", train=True, transform=transform, download=True)
test_dataset = datasets.MNIST(root="./data", train=False, transform=transform, download=True)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Model parameters
input_size = 28 * 28
hidden_size = 256
output_size = 10
model_prob = 0.9
model_lam = 1e-2
n_epochs = 5

# Initialize model, loss, and optimizer
model = MCVariationalNN(input_size, hidden_size, output_size, model_prob, model_lam)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


# Training loop
for epoch in range(n_epochs):
    model.train()
    for images, labels in train_loader:
        # Flatten images
        images = images.view(-1, 28 * 28)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels) + model.regularization()

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch [{epoch + 1}/{n_epochs}], Loss: {loss.item():.4f}")


# MC Dropout inference
def mc_inference(model, X, n_samples=100):
    model.train()  # Enable dropout during inference
    preds = torch.zeros((n_samples, X.size(0), 10))
    for i in range(n_samples):
        preds[i] = model(X)
    return preds.mean(dim=0), preds.std(dim=0), preds  # Mean and uncertainty


# Evaluate with MC Dropout
wrong_storage = {0:{"Image":[], "Mean list":[], "std list":[], "correct": [], "preds":[]},
                 1:{"Image":[], "Mean list":[], "std list":[], "correct": [], "preds":[]},
                 2:{"Image":[], "Mean list":[], "std list":[], "correct": [], "preds":[]},
                 3:{"Image":[], "Mean list":[], "std list":[], "correct": [], "preds":[]},
                 4:{"Image":[], "Mean list":[], "std list":[], "correct": [], "preds":[]},
                 5:{"Image":[], "Mean list":[], "std list":[], "correct": [], "preds":[]},
                 6:{"Image":[], "Mean list":[], "std list":[], "correct": [], "preds":[]},
                 7:{"Image":[], "Mean list":[], "std list":[], "correct": [], "preds":[]},
                 8:{"Image":[], "Mean list":[], "std list":[], "correct": [], "preds":[]},
                 9:{"Image":[], "Mean list":[], "std list":[], "correct": [], "preds":[]},}
model.eval()
correct = 0
total = 0
uncertainties = []
with torch.no_grad():
    for images, labels in test_loader:
        image = images.clone()
        images = images.view(-1, 28 * 28)
        mean_preds, uncertainty, preds = mc_inference(model, images, n_samples=10)
        # mean_preds = (1/uncertainty*10)*mean_preds
        _, predicted = torch.max(mean_preds, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        for i, (pred, cor) in enumerate(zip(predicted, labels)):
            if pred != cor:
                wrong_storage[int(cor)]["Image"].append(image[i][0])
                wrong_storage[int(cor)]["Mean list"].append(mean_preds[i,:])
                wrong_storage[int(cor)]["std list"].append(uncertainty[i,:])
                wrong_storage[int(cor)]["correct"].append(cor)
                wrong_storage[int(cor)]["preds"].append(preds)

        uncertainties.append(uncertainty.mean().item())  # Collect uncertainty for analysis
print(f"Test Accuracy: {100 * correct / total:.2f}%")
print(f"Average Uncertainty: {sum(uncertainties) / len(uncertainties):.4f}")

for digit, data in wrong_storage.items():
    for i, _ in enumerate(data["Image"]):
        plot_sample_prediction(data["Image"][i], data["Mean list"][i], data["std list"][i], data["correct"][i], savefig=f"wrong_{digit}_number_{i+1}")



# Collect uncertainties, predictions, and images
all_uncertainties = []
all_images = []
all_labels = []
all_confidences = []

model.eval()  # Ensure dropout is off by default
for images, labels in test_loader:
    images = images.view(-1, 28 * 28)
    mean_preds, uncertainty = mc_inference(model, images, n_samples=50)
    softmax_preds = torch.nn.functional.softmax(mean_preds, dim=1)
    
    all_uncertainties.append(uncertainty.mean(dim=1).detach())  # Average uncertainty per sample
    all_confidences.append(softmax_preds.max(dim=1)[0].detach())  # Max probability (confidence)
    all_images.append(images.detach())
    all_labels.append(labels.detach())

# Flatten lists
all_uncertainties = torch.cat(all_uncertainties).cpu().numpy()
all_confidences = torch.cat(all_confidences).cpu().numpy()
all_images = torch.cat(all_images).view(-1, 28, 28).cpu().numpy()
all_labels = torch.cat(all_labels).cpu().numpy()

# 1. Uncertainty Heatmap for Sample Images
plt.figure(figsize=(10, 5))
for i in range(10):
    plt.subplot(2, 5, i + 1)
    plt.imshow(all_images[i], cmap="gray")
    plt.title(f"Unc: {all_uncertainties[i]:.2f}")
    plt.axis("off")
plt.suptitle("Uncertainty Heatmap for Sample Test Images")
plt.show()

# 2. Confidence vs. Uncertainty
plt.figure(figsize=(8, 6))
plt.scatter(all_confidences, all_uncertainties, alpha=0.5, color="blue")
plt.xlabel("Confidence (Max Softmax Probability)")
plt.ylabel("Uncertainty (Standard Deviation)")
plt.title("Confidence vs. Uncertainty")
plt.grid(True)
plt.show()

# 3. Mean Uncertainty per Digit Class
uncertainty_per_class = {i: [] for i in range(10)}
for img, label, unc in zip(all_images, all_labels, all_uncertainties):
    uncertainty_per_class[label].append(unc)

class_mean_uncertainty = {k: np.mean(v) for k, v in uncertainty_per_class.items()}
plt.figure(figsize=(8, 6))
plt.bar(class_mean_uncertainty.keys(), class_mean_uncertainty.values())
plt.xlabel("Digit Class")
plt.ylabel("Mean Uncertainty")
plt.title("Mean Uncertainty per Digit Class")
plt.xticks(range(10))
plt.show()

# 4. Incorrect Predictions with High Uncertainty
high_uncertainty_samples = []
uncertainty_threshold = 0.1  # Lower the threshold to include more samples

# Collect incorrect predictions with high uncertainty
for images, labels in test_loader:
    images = images.view(-1, 28 * 28)
    mean_preds, uncertainty = mc_inference(model, images, n_samples=50)
    predicted = mean_preds.argmax(dim=1)
    
    # Add those with high uncertainty and incorrect predictions
    high_uncertainty_samples.extend(
        [(img, true, pred, unc) for img, true, pred, unc in zip(images, labels, predicted, uncertainty.mean(dim=1))
         if true != pred and unc > uncertainty_threshold]
    )

# Debugging: Check how many samples met the criteria
print(f"Number of incorrect predictions with high uncertainty: {len(high_uncertainty_samples)}")

# Plot only if there are enough samples
if len(high_uncertainty_samples) > 0:
    plt.figure(figsize=(10, 5))
    for i, (img, true, pred, unc) in enumerate(high_uncertainty_samples[:10]):
        plt.subplot(2, 5, i + 1)
        plt.imshow(img.view(28, 28).cpu().numpy(), cmap="gray")
        plt.title(f"True: {true}\nPred: {pred}\nUnc: {unc:.2f}")
        plt.axis("off")
    plt.suptitle("Incorrect Predictions with High Uncertainty")
    plt.show()
else:
    print("No incorrect predictions with high uncertainty found.")


# 5. Uncertainty Distribution
plt.figure(figsize=(8, 6))
plt.hist(all_uncertainties, bins=50, alpha=0.7, color="blue")
plt.xlabel("Uncertainty")
plt.ylabel("Frequency")
plt.title("Uncertainty Distribution Across Test Dataset")
plt.grid(True)
plt.show()
print('ok')

