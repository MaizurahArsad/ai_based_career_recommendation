import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# Data
days = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
temperatures = [20, 25, 30, 5, 40, 45, -5]

# Create line graph using plt.plot()
plt.figure(figsize=(9, 5))
plt.plot(
    days,
    temperatures,
    c="red",
    marker="o",
    markersize=10
)

# Graph labels and title
plt.title("Temperature Variation Over 7 Days")
plt.xlabel("Day")
plt.ylabel("Temperature (°C)")

# Set y-ticks as required
plt.yticks([0, 25, 30, 40])

# Improve layout
plt.xticks(rotation=30)
plt.grid(True, linestyle="--", alpha=0.5)
plt.tight_layout()

# Save graph as PNG
plt.savefig("lecture10/temperature_line_graph.png", dpi=300)
plt.close()
