import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Define a simple 2D Gaussian distribution
mu = np.array([2, -1])  # Mean
cov = np.array([[1, 0.5], [0.5, 1]])  # Covariance matrix

# Define a Bayesian model using PyMC
with pm.Model():
    # Prior: Define multivariate normal distribution
    x = pm.MvNormal("x", mu=mu, cov=cov, shape=2)
    
    # Use NUTS sampler
    trace = pm.sample(2000, tune=1000, return_inferencedata=True, target_accept=0.9)

# Plot results
# az.plot_pair(trace, kind="kde", marginals=True)
# plt.show()
import pymc as pm

# Taking draws from a normal distribution
seed = 42
x_dist = pm.Normal.dist(shape=(100, 3))
x_data = pm.draw(x_dist, random_seed=seed)

# Define coordinate values for all dimensions of the data
coords={
 "trial": range(100),
 "features": ["sunlight hours", "water amount", "soil nitrogen"],
}

# Define generative model
with pm.Model(coords=coords) as generative_model:
   x = pm.Data("x", x_data, dims=["trial", "features"])

   # Model parameters
   betas = pm.Normal("betas", dims="features")
   sigma = pm.HalfNormal("sigma")

   # Linear model
   mu = x @ betas

   # Likelihood
   # Assuming we measure deviation of each plant from baseline
   plant_growth = pm.Normal("plant growth", mu, sigma, dims="trial")


# Generating data from model by fixing parameters
fixed_parameters = {
 "betas": [5, 20, 2],
 "sigma": 0.5,
}
with pm.do(generative_model, fixed_parameters) as synthetic_model:
   idata = pm.sample_prior_predictive(random_seed=seed) # Sample from prior predictive distribution.
   synthetic_y = idata.prior["plant growth"].sel(draw=0, chain=0)


# Infer parameters conditioned on observed data
with pm.observe(generative_model, {"plant growth": synthetic_y}) as inference_model:
   idata = pm.sample(random_seed=seed)

   summary = pm.stats.summary(idata, var_names=["betas", "sigma"])
   print(summary)