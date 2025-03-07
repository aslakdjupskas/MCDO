from jax import config
config.update("jax_enable_x64", True)

import gpjax as gpx
from jax import random as jr
import jax.numpy as jnp
import optax as ox
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import time
from jax import vmap
import jax.random as random
import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import torch
import torch.nn as nn
import torch.optim as optim


results = {
    "BNN": {
        15: None,
        50: None,
        1500: None
    },
    "GP": {
        15: None,
        50: None,
        1500: None
    },
    "MCDO": {
        15: None,
        50: None,
        1500: None
    }
}

MSE = {
    "BNN": {
        15: None,
        50: None,
        1500: None
    },
    "GP": {
        15: None,
        50: None,
        1500: None
    },
    "MCDO": {
        15: None,
        50: None,
        1500: None
    }
}



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
    
    indices = np.linspace(0, len(X_available) - 1, N, dtype=int)
    X = X_available[indices]
    Y = Y_available[indices] + sigma_obs * np.random.randn(N, D_Y)
    
    assert X.shape == (N, D_X)
    assert Y.shape == (N, D_Y)
    assert X_test.shape == (N_test, D_X)
    assert Y_true.shape == (N_test, D_Y)
    
    return X, Y, X_test, Y_true


## GAUSSIAN PROCESS
def runGP(N):
    key = jr.PRNGKey(123)
    np.random.seed(8)
    D_X, D_H = 3, 5
    X, Y, X_test, Y_true = get_data(N=N, D_X=D_X)
    # X_train = X[:,1].reshape(-1,1); y_train = Y[:,0].reshape(-1,1); X_test = X_test[:,1].reshape(-1,1)

    # dataset
    D = gpx.Dataset(X=X, y=Y)

    # prior
    meanf = gpx.mean_functions.Zero()  # Zero mean 
    kernel = gpx.kernels.RBF()  # RBF kernel
    # kernel = gpx.kernels.Matern32(lengthscale=10., variance=10.)


    prior = gpx.gps.Prior(mean_function=meanf, kernel=kernel)

    # Define the likelihood 
    likelihood = gpx.likelihoods.Gaussian(num_datapoints=15)

    # Combine prior and likelihood to form posterior
    posterior = prior * likelihood

    # Define optimizer
    optimizer = ox.adam(learning_rate=1e-2)
    # optimizer = ox.sgd(learning_rate=0.002)


    # Fit the model to the training data using MLE optimization
    opt_posterior, history = gpx.fit(
        model=posterior,
        objective=lambda p, d: -gpx.objectives.conjugate_mll(p, d),
        train_data=D,
        optim=optimizer,
        num_iters=8000,
        safe=True,
        key=key,
    )

    # Predict on test points
    latent_dist = opt_posterior(X_test, D)
    predictive_dist = opt_posterior.likelihood(latent_dist)

    # Obtain predictive mean and std
    pred_mean = predictive_dist.mean()
    pred_std = predictive_dist.stddev()
    # cov matrix
    cov_matrix = predictive_dist.covariance()
    
    mse = []
    for i in range(1000):
        pred = np.random.multivariate_normal(pred_mean.flatten(), cov_matrix)
        mse.append(np.mean((pred - Y_true[:,0]) ** 2))

    opt_params = (opt_posterior.prior.kernel.lengthscale.value, opt_posterior.prior.kernel.variance.value)
    return pred_mean, pred_std, cov_matrix, X, Y, X_test, Y_true, opt_params, mse



## BNN

# a two-layer bayesian neural network with computational flow
# given by D_X => D_H => D_H => D_Y where D_H is the number of
# hidden units. (note we indicate tensor dimensions in the comments)
def model(X, Y, D_H, D_Y=1):
    N, D_X = X.shape

    # sample first layer (we put unit normal priors on all weights)
    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((D_X, D_H)), jnp.ones((D_X, D_H))))
    assert w1.shape == (D_X, D_H)
    z1 = jnp.tanh(jnp.matmul(X, w1))  # <= first layer of activations
    assert z1.shape == (N, D_H)

    # sample second layer
    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H))))
    assert w2.shape == (D_H, D_H)
    z2 = jnp.tanh(jnp.matmul(z1, w2))  # <= second layer of activations
    assert z2.shape == (N, D_H)

    # sample final layer of weights and neural network output
    w3 = numpyro.sample("w3", dist.Normal(jnp.zeros((D_H, D_Y)), jnp.ones((D_H, D_Y))))
    assert w3.shape == (D_H, D_Y)
    z3 = jnp.matmul(z2, w3)  # <= output of the neural network
    assert z3.shape == (N, D_Y)

    if Y is not None:
        assert z3.shape == Y.shape

    # we put a prior on the observation noise
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3.0, 1.0))
    sigma_obs = 1.0 / jnp.sqrt(prec_obs)

    # observe data
    with numpyro.plate("data", N):
        # note we use to_event(1) because each observation has shape (1,)
        numpyro.sample("Y", dist.Normal(z3, sigma_obs).to_event(1), obs=Y)


# helper function for HMC inference
def run_inference(model, args, rng_key, X, Y, D_H):
    start = time.time()
    kernel = NUTS(model)
    mcmc = MCMC(
        kernel,
        num_warmup=args.num_warmup,
        num_samples=args.num_samples,
        num_chains=args.num_chains,
        progress_bar=False if "NUMPYRO_SPHINXBUILD" in os.environ else True,
    )
    mcmc.run(rng_key, X, Y, D_H)
    mcmc.print_summary()
    print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()


# helper function for prediction
def predict(model, rng_key, samples, X, D_H):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None, D_H=D_H)
    return model_trace["Y"]["value"]

def runBNN(args):

    N, D_X, D_H = args.num_data, 3, args.num_hidden
    X, Y, X_test, Y_true = get_data(N=N, D_X=D_X)
    
    # do inference
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    samples = run_inference(model, args, rng_key, X, Y, D_H)

    # # predict Y_test at inputs X_test
    # vmap_args = (
    #     samples,
    #     random.split(rng_key_predict, args.num_samples * args.num_chains),
    # )
    # predictions = vmap(
    #     lambda samples, rng_key: predict(model, rng_key, samples, X_test, D_H)
    # )(*vmap_args)
    # predictions = predictions[..., 0]

    predictions = []
    for i in range(args.num_samples * args.num_chains):
        sample = {k: v[i] for k, v in samples.items()} 

        # Forward pass with activation functions
        z1 = jnp.tanh(X_test @ sample["w1"])
        z2 = jnp.tanh(z1 @ sample["w2"])
        y_pred = z2 @ sample["w3"]  # No activation here for regression

        predictions.append(y_pred)

    # Convert list to JAX array
    predictions = jnp.stack(predictions)

    # Ensure correct shape (num_samples, num_test_points)
    predictions = predictions[..., 0]  # Assuming output shape is (N, 1)


    mse = []
    # mse for 1000 predictions
    for pred in predictions[-1000:]:
        mse.append(np.mean((pred - Y_true[:,0]) ** 2))

    # compute mean prediction and confidence interval around median
    mean_prediction = jnp.mean(predictions, axis=0)
    #percentiles = np.percentile(predictions, [5.0, 95.0], axis=0)
    std_prediction = jnp.std(predictions, axis=0)
    cov_matrix = np.cov(predictions, rowvar=False)
    return mean_prediction, std_prediction, predictions, X, Y, X_test, Y_true, mse



## MONTE CARLO DROPOUT
# MC Variational Dense Layer
class MCVariationalDense(nn.Module):
    def __init__(self, n_in, n_out, model_prob, model_lam):
        super(MCVariationalDense, self).__init__()
        self.model_prob = model_prob
        self.model_lam = model_lam
        self.dropout = nn.Dropout(p=self.model_prob)
        self.model_M = nn.Parameter(torch.randn(n_in, n_out) * 0.1)
        self.model_m = nn.Parameter(torch.zeros(n_out))

    def forward(self, X):
        if not self.training:
            model_W = self.model_M
        else:
            model_W = self.dropout(self.model_M)
        output = torch.mm(X, model_W) + self.model_m
        return output

    def regularization(self):
        return self.model_lam * (
            self.model_prob * torch.sum(self.model_M ** 2) + torch.sum(self.model_m ** 2)
        )

# MC Variational Neural Network
class MCVariationalNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, model_prob, model_lam):
        super(MCVariationalNN, self).__init__()
        self.layer1 = MCVariationalDense(input_size, hidden_size, model_prob, model_lam)
        self.layer2 = MCVariationalDense(hidden_size, hidden_size, model_prob, model_lam)
        self.layer3 = MCVariationalDense(hidden_size, hidden_size, model_prob, model_lam)
        self.layer4 = MCVariationalDense(hidden_size, output_size, model_prob, model_lam)

    def forward(self, X):
        X = torch.relu(self.layer1(X))
        X = torch.relu(self.layer2(X))
        X = torch.relu(self.layer3(X))
        X = self.layer4(X)
        return X

    def regularization(self):
        return (
            self.layer1.regularization() +
            self.layer2.regularization() +
            self.layer3.regularization() +
            self.layer4.regularization()
        )

# MC Dropout inference
def mc_inference(model, X, n_samples=100):
    model.train()  # Keep dropout active
    preds = torch.zeros((n_samples, X.size(0), 1))
    with torch.no_grad():
        for i in range(n_samples):
            preds[i] = model(X)
    return preds.mean(dim=0), preds.std(dim=0), preds

def runMCDO(N, hidden_size=32, model_prob=0.1, model_lam=1e-2, lr=1e-2, num_epochs=1000):
    np.random.seed(0)
    torch.manual_seed(0)
    # Convert to PyTorch tensors
    X, Y, X_test, Y_true = get_data(N=N, D_X=3)
    # X_train = X[:,1].reshape(-1,1); y_train = Y[:,0].reshape(-1,1); X_test = X_test[:,1].reshape(-1,1)
    X_train_torch = torch.tensor(X.tolist(), dtype=torch.float32)
    Y_train_torch = torch.tensor(Y.tolist(), dtype=torch.float32)
    X_test_torch = torch.tensor(X_test.tolist(), dtype=torch.float32)

    # Model parameters
    input_size  = 3
    output_size = 1

    # Initialize model, loss function, and optimizer
    model = MCVariationalNN(input_size, hidden_size, output_size, model_prob, model_lam)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # Training
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_torch)
        loss = criterion(outputs, Y_train_torch) + model.regularization()
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")

    # MC Dropout Inference
    model.eval()
    mean_pred, std_pred, predictions = mc_inference(model, X_test_torch, n_samples=10000)
    mse = []
    # MSE for 1000 predictions
    for pred in predictions[-1000:]: 
        mse.append(np.mean((jnp.array(pred) - Y_true) ** 2))

    # Convert to numpy
    mean_pred = mean_pred.numpy()
    std_pred = std_pred.numpy()
    return mean_pred, std_pred, predictions, X, Y, X_test, Y_true, mse



samples = 12000
warmup = 2000
chains = 1
num_hidden = 4

num_data_values = [15, 50, 150]  # Three different values for num_data


for num_data in num_data_values:
    args = argparse.Namespace(
        num_samples=samples, 
        num_warmup=warmup,
        num_chains=chains,
        num_hidden=num_hidden,
        num_data=num_data
    )
    

    pred_mean, pred_std, cov_matrix, X, Y, X_test, Y_true, opt_params, mse = runGP(num_data)
    results["GP"][num_data] = (pred_mean, pred_std, cov_matrix, X, Y, X_test, Y_true, opt_params, mse)

    mean_prediction, std_prediction, preds, X, Y, X_test, Y_true, mse = runBNN(args)
    results["BNN"][num_data] = (mean_prediction, std_prediction, preds, X, Y, X_test, Y_true, mse)

    mean_pred, std_pred, preds, X, Y, X_test, Y_true, mse = runMCDO(num_data, hidden_size=64, model_prob=0.2, model_lam=0.01, lr=0.001, num_epochs=10000)
    results["MCDO"][num_data] = (mean_pred, std_pred, preds, X, Y, X_test, Y_true, mse)


# Sample data (Replace these with actual data from your results dictionary)
models = ["GP", "BNN", "MCDO"]

fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for row, model in enumerate(models):
    for col, num_data in enumerate(num_data_values):
        ax = axes[row, col]
        
        # Extract stored results
        if model == "GP":
            mean_prediction, std_prediction, cov_matrix, X_train, y_train, X_test, Y_true, opt_params, mse = results[model][num_data]
        else:
            mean_prediction, std_prediction, preds, X_train, y_train, X_test, Y_true, mse = results[model][num_data]

        # Plot individual predictions for BNN and MCDO
        if model in ["BNN", "MCDO"]:
            for pred in preds[-100:]:  # Plot fewer samples for clarity
                ax.plot(X_test[:, 1], pred, color='gray', alpha=0.1)
            # if model == "BNN":
            #     ax.plot(X_test[:, 1], preds[-51], color='black', alpha=0.5)

        else:
            for _ in range(100):
                ax.plot(X_test[:, 1], np.random.multivariate_normal(mean_prediction.flatten(), cov_matrix), alpha=0.1, color='gray')
        
        # Compute 90% confidence intervals        
        lower_bound = np.squeeze(mean_prediction - 1.645 * std_prediction)
        upper_bound = np.squeeze(mean_prediction + 1.645 * std_prediction)
        ax.fill_between(X_test[:, 1], lower_bound, upper_bound, color="lightblue", label="90% CI")
        
        # Plot mean prediction
        ax.plot(X_test[:, 1], mean_prediction, "blue", ls="solid", lw=2.0, label="Mean Prediction")
        
        # Plot true function
        ax.plot(X_test[:, 1], Y_true, "g--", label="True Function")

        # Plot training data
        ax.plot(X_train[:, 1], y_train[:, 0], "r.", alpha=0.5, label="Training Data")
        ax.set_ylim(-4,4) 
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        if model == "GP":
            ax.set_title(f"\n{model} Predictions (n={num_data})\nLengthscale: {opt_params[0]:.2f}, Variance: {opt_params[1]:.2f}\nMean Squared Error: {np.mean(mse):.2f}")
        else:
            ax.set_title(f"\n{model} Predictions (n={num_data})\nMean Squared Error: {np.mean(mse):.2f}")
        ax.legend()
        ax.grid()

plt.tight_layout()
plt.savefig("plots/AllModelsComparison_uniformDO_smoothbnn.pdf")
plt.show()

# plot MSE
fig, axes = plt.subplots(3, 3, figsize=(15, 15))

for row, model in enumerate(models):
    for col, num_data in enumerate(num_data_values):
        ax = axes[row, col]
        
        # Extract stored results
        if model == "GP":
            _, _, _, _, _, _, _, _, mse = results[model][num_data]
        else:
            _, _, _, _, _, _, _, mse = results[model][num_data]

        ax.hist(mse, bins=50, color="skyblue", edgecolor="black")
        ax.set_title(f"{model} MSE (n={num_data})")
        ax.set_xlabel("Mean Squared Error")
        ax.set_ylabel("Frequency")
        ax.grid()
        # Title
        ax.set_title(f"{model} MSE (n={num_data})\nMean: {np.mean(mse):.2f}, Std: {np.std(mse):.2f}")


plt.tight_layout()
plt.savefig("plots/MSE_ALL_DO_smoothbnn.pdf")
plt.show()
