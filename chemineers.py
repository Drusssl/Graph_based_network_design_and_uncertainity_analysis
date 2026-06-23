import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint
import emcee
import corner

# ==============================================================================
# 1. Define the Chemical Kinetic Model 
# (Based on Fig 1 from the paper: Parent -> Metabolite -> Sink)
# ==============================================================================

def kinetic_model(y, t, k_p, k_m, f):
    """
    Ordinary Differential Equations (ODEs) for the chemical transformation.
    y[0] = Parent compound concentration
    y[1] = Metabolite concentration
    
    Parameters:
    k_p = degradation rate of the parent compound
    k_m = degradation rate of the metabolite
    f   = formation fraction (fraction of degraded parent that becomes the metabolite)
    """
    C_p, C_m = y
    dCp_dt = -k_p * C_p
    dCm_dt = f * k_p * C_p - k_m * C_m
    return [dCp_dt, dCm_dt]

def solve_kinetics(t, theta, y0):
    k_p, k_m, f = theta
    # Numerically integrate the ODEs over time 't'
    sol = odeint(kinetic_model, y0, t, args=(k_p, k_m, f))
    return sol

# ==============================================================================
# 2. Generate Synthetic Experimental Data 
# (As performed in the paper to validate the MCMC approach)
# ==============================================================================

# True parameters to simulate a realistic experiment
true_k_p = 0.1   # True degradation rate of parent
true_k_m = 0.05  # True degradation rate of metabolite
true_f = 0.7     # True fraction forming the metabolite
true_sigma = 2.0 # Measurement noise standard deviation

true_theta = [true_k_p, true_k_m, true_f]
y0 = [100.0, 0.0]  # Initial concentrations (100% parent, 0% metabolite)

# Simulated measurement days
t_obs = np.array([0, 2, 5, 10, 20, 30, 50])
sol_true = solve_kinetics(t_obs, true_theta, y0)

# Add Gaussian measurement noise (constant variance error model)
np.random.seed(42)
data_obs = sol_true + np.random.normal(0, true_sigma, sol_true.shape)
data_obs = np.maximum(data_obs, 0) # Concentrations cannot be negative physically

# ==============================================================================
# 3. Define the Bayesian Framework for MCMC
# ==============================================================================

def log_prior(theta):
    """
    Defines uninformative uniform priors with physical constraints.
    Rates and fractions cannot be negative. Fractions must be <= 1.0.
    """
    k_p, k_m, f, sigma = theta
    if 0 < k_p < 1.0 and 0 < k_m < 1.0 and 0 <= f <= 1.0 and 0 < sigma < 20.0:
        return 0.0
    return -np.inf

def log_likelihood(theta, t, y0, data):
    """
    Calculates the Gaussian Log-Likelihood of the data given the model parameters.
    """
    k_p, k_m, f, sigma = theta
    model_theta = [k_p, k_m, f]
    
    # Run the model with the current proposed parameters
    model_pred = solve_kinetics(t, model_theta, y0)
    
    # Calculate Gaussian Log-Likelihood
    sigma2 = sigma ** 2
    ll = -0.5 * np.sum((data - model_pred) ** 2 / sigma2 + np.log(sigma2))
    return ll

def log_probability(theta, t, y0, data):
    """
    Combines the Prior and Likelihood to get the unnormalized Posterior probability.
    """
    lp = log_prior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + log_likelihood(theta, t, y0, data)

# ==============================================================================
# 4. Run Markov-Chain Monte Carlo (MCMC) Sampling
# ==============================================================================

ndim = 4         # k_p, k_m, f, sigma
nwalkers = 32    # Number of independent MCMC chains
nsteps = 3000    # Number of iterations

# Initialize walkers slightly perturbed around a general guess
initial_guess = np.array([0.05, 0.02, 0.5, 5.0]) 
pos = initial_guess + 1e-4 * np.random.randn(nwalkers, ndim)

print("Running MCMC Sampling...")
sampler = emcee.EnsembleSampler(nwalkers, ndim, log_probability, args=(t_obs, y0, data_obs))
sampler.run_mcmc(pos, nsteps, progress=True)

# ==============================================================================
# 5. Statistical Analysis & Post-Processing (Uncertainty Estimation)
# ==============================================================================

# Discard the initial "burn-in" phase and flatten the chains
flat_samples = sampler.get_chain(discard=1000, thin=15, flat=True)

# Print parameter estimates based on the resulting probability distributions
labels = ["k_parent", "k_metabolite", "f", "sigma"]
print("\nReal Probability Distribution Estimates (Median with 95% Credible Intervals):")
for i in range(ndim):
    mcmc = np.percentile(flat_samples[:, i], [2.5, 50, 97.5])
    q = np.diff(mcmc)
    print(f"{labels[i]:<15}: {mcmc[1]:.4f} (-{q[0]:.4f}, +{q[1]:.4f})")

# Generate the corner plot to visualize parameter correlations and marginal distributions
fig = corner.corner(
    flat_samples, 
    labels=[r"$k_{parent}$", r"$k_{metabolite}$", r"$f$", r"$\sigma$"], 
    truths=[true_k_p, true_k_m, true_f, true_sigma],
    quantiles=[0.025, 0.5, 0.975],
    show_titles=True,
    title_kwargs={"fontsize": 12}
)
plt.suptitle("MCMC Posterior Probability Distributions", y=1.02, fontsize=14)
plt.show()

# Plot the predictive model uncertainty against the simulated data
plt.figure(figsize=(10, 5))
t_eval = np.linspace(0, 50, 100)

# Plot 100 random samples from the posterior to visualize uncertainty
for param_sample in flat_samples[np.random.randint(len(flat_samples), size=100)]:
    sol_sample = solve_kinetics(t_eval, param_sample[:3], y0)
    plt.plot(t_eval, sol_sample[:, 0], color='blue', alpha=0.05)
    plt.plot(t_eval, sol_sample[:, 1], color='orange', alpha=0.05)

plt.plot(t_obs, data_obs[:, 0], 'bo', label='Parent (Observed)')
plt.plot(t_obs, data_obs[:, 1], 'ro', label='Metabolite (Observed)')
plt.xlabel('Time')
plt.ylabel('Concentration')
plt.title('Kinetic Fit with MCMC Uncertainty Bounds')
plt.legend()
plt.show()