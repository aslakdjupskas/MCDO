import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Target distribution: Normal(3, 2)
def target_distribution(x):
    return np.exp(-0.5 * ((x - 3) / 2) ** 2)  # Unnormalized normal PDF

# Metropolis-Hastings MCMC
def metropolis_hastings(iterations, proposal_std=1.0):
    samples = []
    x = 0  # Initial value

    for _ in range(iterations):
        x_new = x + np.random.normal(0, proposal_std)  # Propose a new sample
        acceptance_ratio = target_distribution(x_new) / target_distribution(x)
        
        if np.random.rand() < acceptance_ratio:
          x = x_new  # Accept the new sample
            
        samples.append(x)
    
    return np.array(samples)

# Run MCMC
samples_fine = metropolis_hastings(1500000)[500000:]
samples_few = metropolis_hastings(501000)[500000:]

# Define x values for the true distribution
x = np.linspace(-5, 12, 1000)
pdf_vals = target_distribution(x)
pdf_vals /= np.trapz(pdf_vals, x)  # Normalize using numerical integration


# Plot for many samples
sns.histplot(samples_few, bins=50, stat="density", kde=True, alpha=0.6, label="MCMC Samples")
plt.plot(x, pdf_vals, label="True Distribution", color="red")
# plt.title("Metropolis-Hastings (500 Samples)")
plt.xlabel("x")
plt.ylabel("Density")
plt.legend()
plt.savefig('MCMC_illustration_1e3.pdf')
plt.close()

# Plot for few samples
sns.histplot(samples_fine, bins=150, stat="density", kde=True, alpha=0.6, label="MCMC Samples")
plt.plot(x, pdf_vals, label="True Distribution", color="red")
# plt.title("Metropolis-Hastings (1,000,000 Samples)")
plt.ylabel("Density")
plt.xlabel("x")
plt.legend()
plt.savefig('MCMC_illustration_1e6.pdf')
plt.close()