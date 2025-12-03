import argparse
import os
import time

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from jax import vmap
import jax.numpy as jnp
import jax.random as random

import numpyro
from numpyro import handlers
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro import render_model

from scipy.stats import ks_2samp

# matplotlib.use("Agg")  # noqa: E402
from scipy.stats import gamma

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


# the non-linearity we use in our neural network
def nonlin(x):
    return jnp.tanh(x)


# a two-layer bayesian neural network with computational flow
# given by D_X => D_H => D_H => D_Y where D_H is the number of
# hidden units. (note we indicate tensor dimensions in the comments)
def model(X, Y, D_H, D_Y=1):
    N, D_X = X.shape

    # sample first layer (we put unit normal priors on all weights)
    # Normal
    w1 = numpyro.sample("w1", dist.Normal(jnp.zeros((D_X, D_H)), np.ones((D_X, D_H))))
    w2 = numpyro.sample("w2", dist.Normal(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H))))
    w3 = numpyro.sample("w3", dist.Normal(jnp.zeros((D_H, D_Y)), jnp.ones((D_H, D_Y))))
    
    # Laplace
    # w1 = numpyro.sample("w1", dist.Laplace(jnp.zeros((D_X, D_H)), jnp.ones((D_X, D_H))))
    # w2 = numpyro.sample("w2", dist.Laplace(jnp.zeros((D_H, D_H)), jnp.ones((D_H, D_H))))
    # w3 = numpyro.sample("w3", dist.Laplace(jnp.zeros((D_H, D_Y)), jnp.ones((D_H, D_Y))))


    assert w1.shape == (D_X, D_H)
    z1 = nonlin(jnp.matmul(X, w1))  # <= first layer of activations
    assert z1.shape == (N, D_H)

    assert w2.shape == (D_H, D_H)
    z2 = nonlin(jnp.matmul(z1, w2))  # <= second layer of activations
    assert z2.shape == (N, D_H)

    assert w3.shape == (D_H, D_Y)
    z3 = jnp.matmul(z2, w3)  # <= output of the neural network
    assert z3.shape == (N, D_Y)

    if Y is not None:
        assert z3.shape == Y.shape


    # we put a prior on the observation noise
    prec_obs = numpyro.sample("prec_obs", dist.Gamma(3, 1))
    # prec_obs = numpyro.sample("prec_obs", dist .Gamma(1000, 0.1))
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
    # mcmc.print_summary()
    # print("\nMCMC elapsed time:", time.time() - start)
    return mcmc.get_samples()

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
    valid_mask = (X_available[:, 1] >= -1.0) & (X_available[:, 1] <= 1.0)
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
        Y = Y_available[indices] + sinc_noise[:, None] * np.random.randn(N, D_Y)
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



# def get_data(N=50, D_X=3, sigma_obs=0.05, N_test=500, gap=False):
#     D_Y = 1  # Create 1D outputs
#     np.random.seed(0)
    
#     # Generate test data first
#     X_test = jnp.linspace(-1.3, 1.3, N_test)
#     X_test = jnp.power(X_test[:, np.newaxis], jnp.arange(D_X))
    
#     # Define ground truth function
#     W = 0.5 * np.random.randn(D_X)
#     Y_true = jnp.dot(X_test, W) + 0.5 * jnp.power(0.5 + X_test[:, 1], 2.0) * jnp.sin(4.0 * X_test[:, 1])
#     Y_true = Y_true[:, np.newaxis]
#     Y_true -= jnp.mean(Y_true)
#     Y_true /= jnp.std(Y_true)
    
#     # Apply gap only if enabled
#     if gap:
#         mask = (X_test[:, 1] < -0.5) | (X_test[:, 1] > 0.5)
#         X_available = X_test[mask]
#         Y_available = Y_true[mask]
#     else:
#         X_available = X_test
#         Y_available = Y_true
    
#     # Ensure X and Y are within the range [-1,1] **before selecting indices**
#     valid_mask = (X_available[:, 1] >= -1.0) & (X_available[:, 1] <= 1.0)
#     X_available = X_available[valid_mask]
#     Y_available = Y_available[valid_mask]

#     # Now we are guaranteed to sample only from valid points
#     if len(X_available) < N:
#         raise ValueError(f"Not enough valid samples ({len(X_available)}) to select N={N}.")

#     indices = np.linspace(0, len(X_available) - 1, N, dtype=int)
#     X = X_available[indices]
#     Y = Y_available[indices] + sigma_obs * np.random.randn(N, D_Y)
#     assert X.shape == (N, D_X)
#     assert Y.shape == (N, D_Y)
#     assert X_test.shape == (N_test, D_X)
#     assert Y_true.shape == (N_test, D_Y)
    
#     return X, Y, X_test, Y_true


#  helper function for prediction
def predict(model, rng_key, samples, X, D_H):
    model = handlers.substitute(handlers.seed(model, rng_key), samples)
    # note that Y will be sampled in the model because we pass Y=None here
    model_trace = handlers.trace(model).get_trace(X=X, Y=None, D_H=D_H)

    return model_trace["Y"]["value"]


def forward(X_test, samples, n):
    predictions = []
    for i in range(n):
        sample = {k: v[i] for k, v in samples.items()} 

        # Forward pass with activation functions
        z1 = nonlin(X_test @ sample["w1"])
        z2 = nonlin(z1 @ sample["w2"])
        z3 = z2 @ sample["w3"] 
        sigma_obs = 1.0 / jnp.sqrt(sample["prec_obs"])
        noise = dist.Normal(0, sigma_obs).sample(random.PRNGKey(i))
        y_pred = z3 + noise

        predictions.append(y_pred)

    # Convert list to JAX array
    predictions = jnp.stack(predictions)

    # Ensure correct shape (num_samples, num_test_points)
    predictions = predictions[..., 0]  # Assuming output shape is (N, 1)
    return predictions

def main(args):
    N, D_X, D_H = args.num_data, 3, args.num_hidden
    # X, Y, X_test, Y_true = get_data(N=N, D_X=D_X, N_test=500, gap=False, sigma_obs=args.sigma_obs)

    X, Y, X_test, Y_true, _, _ = get_data(N=N, D_X=3, N_test=500, sigma_obs=0.05, gap=False, sinc_noise_bool=False, seed=0)

    # plt.figure()
    # # plt.subplot(1, 3, 1)
    # # plt.plot(X[:, 0], Y[:, 0], lw=2.0, label="Training Data")
    # # plt.subplot(1, 3, 2)
    # plt.plot(X_test[:, 1], Y_true[:, 0], lw=2.0, label="Training Data")
    # # plt.subplot(1, 3, 3)
    # # plt.plot(X[:, 2], Y[:, 0], lw=2.0, label="Training Data")
    # plt.show()

    # X_test = jnp.ones_like(X_test)
    # do inference
    rng_key, rng_key_predict = random.split(random.PRNGKey(0))
    samples = run_inference(model, args, rng_key, X, Y, D_H)

    # Generate x values
    x_new2 = np.linspace(0, 1, 10000)

    # Compute PDF
    pdf_new2 = gamma.pdf(x_new2, a=3, scale=1)

    
    predicted_noise = 1.0 / jnp.sqrt(samples["prec_obs"])

    true_noise = (args.sigma_obs * np.random.randn(len(predicted_noise)))
    ks_stat, p_value = ks_2samp(predicted_noise, true_noise)
    print(f"KS Statistic: {ks_stat}, p-value: {p_value}")

    error = np.loadtxt("MCDO_True_Model_Error.txt", delimiter=",")

    # normalize the error, predicted_noise and true_noise
    # predicted_noise = (predicted_noise - np.mean(predicted_noise)) / np.std(predicted_noise)
    # true_noise = (true_noise - np.mean(true_noise)) / np.std(true_noise)
    # error = (error - np.mean(error)) / np.std(error)


    # plot the histogram
    plt.figure(figsize=(8, 6))
    plt.hist(predicted_noise, bins=50, density=True, alpha=0.5, label="Predicted Shot noise", color='blue')
    plt.hist(true_noise, bins=50, density=True, alpha=0.5, label="True Noise", color='red')
    plt.hist(np.sqrt(error), bins=50, density=True, alpha=0.5, label="True Model Error", color='green')
    # plt.plot(x_new2, pdf_new2, label="Gamma PDF", color='green')
    plt.xlabel("Noise")
    plt.ylabel("Density")
    # plt.title("Predicted Noise vs True Noise")
    plt.legend()
    plt.grid()
    plt.savefig(f"KS_test.pdf")
    plt.show()



    samples["prec_obs"] += 1e6
    
    if args.vmapped:
        # predict Y_test at inputs X_test
        vmap_args = (
            samples,
            random.split(rng_key_predict, args.num_samples * args.num_chains),
        )
        predictions = vmap(
            lambda samples, rng_key: predict(model, rng_key, samples, X_test, D_H)
        )(*vmap_args)
        predictions = predictions[..., 0]
    else:
        # predict Y_test at inputs X_test
        predictions = forward(X_test, samples, args.num_samples * args.num_chains)
    
    # compute mean prediction and confidence interval around median

    mean_prediction = jnp.mean(predictions, axis=0)
    percentiles = np.percentile(predictions, [2.5, 97.5], axis=0)
    cov_matrix = np.cov(predictions, rowvar=False)

    # make plots
    plt.figure(figsize=(8, 6)) #, constrained_layout=True)
    for i in range(6000,6150):
        plt.plot(X_test[:, 1], predictions[i], color='gray', alpha=0.1)
        # plt.plot(X_test[:, 1], np.random.multivariate_normal(mean_prediction, cov_matrix), alpha=0.1, color='gray')  # Plot the posterior samples
    # plot 90% confidence level of predictions
    plt.fill_between(
        X_test[:, 1], percentiles[0, :], percentiles[1, :], color="lightblue", label="95% CI",
    )
    plt.plot(X_test[:, 1], Y_true, "k--", lw=2.0, label="True mean")
    # plot mean prediction
    plt.plot(X_test[:, 1], mean_prediction, "blue", ls="solid", lw=2.0, label="Mean Prediction")
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.title("Mean predictions with 95% CI")
    plt.plot(X[:, 1], Y[:, 0], "r.", alpha=0.5, label="Training Data")
    plt.legend()
    plt.grid()
    plt.savefig(f"bnn_smooth_plot_{N}_notPredNoise_gap.pdf")
    plt.show()


if __name__ == "__main__":
    assert numpyro.__version__.startswith("0.16.1")
    parser = argparse.ArgumentParser(description="Bayesian neural network example")
    parser.add_argument("-n", "--num-samples", nargs="?", default=15000, type=int)
    parser.add_argument("--num-warmup", nargs="?", default=5000, type=int)
    parser.add_argument("--num-chains", nargs="?", default=1, type=int)
    parser.add_argument("--num-data", nargs="?", default=50, type=int)
    parser.add_argument("--num-hidden", nargs="?", default=5, type=int)
    parser.add_argument("--device", default="cpu", type=str, help='use "cpu" or "gpu".')
    parser.add_argument("--vmapped", action="store_true", default=True)
    parser.add_argument("--sigma-obs", nargs="?", default=0.05, type=float)
    args = parser.parse_args()

    numpyro.set_platform(args.device)
    numpyro.set_host_device_count(args.num_chains)
    main(args)