import numpy as np
import matplotlib.pyplot as plt

# Set font properties
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['font.weight'] = 'bold'
plt.rcParams['font.size'] = 23

# Generate synthetic data with strong fluctuations
x = np.linspace(1600, 2025, 500)  # Year range (1600 → 2025)
y_original = np.sin((x - 1600) / 50) + 0.5 * np.sin((x - 1600) / 20)

# Add noise and random spikes
random_spikes = np.random.choice([0, 1], size=len(x), p=[0.97, 0.03]) * np.random.uniform(-2, 2, len(x))
y_original += np.random.normal(0, 0.3, len(x)) + random_spikes

# Create RC reconstruction (slight variations)
y_rc = y_original + np.random.normal(0, 0.1, len(x))

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 7))

# Plot the data with new colors
ax.plot(x, y_original, 'b-', linewidth=2, label='Predicted δ18O')  # Blue solid line
ax.plot(x, y_rc, 'g--', linewidth=1.5, label='Reconstruction')  # Green dashed line

# Labels
ax.set_xlabel("Year", fontweight='bold')  # X-axis label
ax.set_ylabel('$\delta^{18}O$ Anomaly (‰)',fontweight='semibold')  # Y-axis label

# Reverse x-axis (2025 → 1600)
ax.set_xlim(2025, 1600)  # Explicitly set limits
ax.invert_xaxis()  # Ensure correct direction

# Set x-ticks (every 50 years)
ax.set_xticks(np.arange(1625, 2026, 50))

# Grid and vertical lines
# ax.grid(True, which='both', linestyle='--', alpha=0.5)

# Legend with a semi-transparent background
legend = ax.legend(loc='lower right', frameon=True)
legend.get_frame().set_alpha(0.8)

# Display the plot
plt.show()
