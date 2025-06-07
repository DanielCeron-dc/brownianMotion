# Imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Seed the random number generator
rng = np.random.default_rng(10)

# Determine the number of paths and points per path
points = 1000
paths = 5
# Create the initial set of random normal draws
mu, sigma = 0.0, 1.0
Z = rng.normal(mu, sigma, (paths, points))

# Define the time step size and t-axis
interval = [0.0, 1.0]
dt = (interval[1] - interval[0]) / (points - 1)
t_axis = np.linspace(interval[0], interval[1], points)

mu_c, sigma_c = 5.0, 2.0

# Use Equation 3.3 from [Glasserman, 2003] to sample 50 brownian motion paths
X = np.zeros((paths, points))

print('X size = ', X.shape)

for idx in range(points - 1):
    real_idx = idx + 1
    X[:, real_idx] = X[:, real_idx - 1] + mu_c * dt + sigma_c * np.sqrt(dt) * Z[:, idx]



print('x = ', X)

# Plot these paths
fig, ax = plt.subplots(1, 1, figsize=(12, 8))
for path in range(paths):
    ax.plot(t_axis, X[path, :])
ax.set_title("Constant mean and standard deviation Brownian Motion sample paths")
ax.set_xlabel("Time")
ax.set_ylabel("Asset Value")
plt.show()