import matplotlib.pyplot as plt
import numpy as np

# ---- Data ----
formulations = ["First Formulation", "Second Formulation", "Third Formulation"]
success_rates = [4.5, 71.67, 86.33]  # <-- replace with your actual results

# ---- Plot ----
plt.figure(figsize=(8, 5))

bars = plt.bar(formulations, success_rates, color=["#4C72B0", "#DD8452", "#55A868"], width=0.6)

# Y-axis range and ticks
plt.ylim(0, 100)
plt.yticks(np.arange(0, 101, 10))

# Axis labels
plt.xlabel("Formulations", fontsize=12)
plt.ylabel("Success Rate (%)", fontsize=12)

# Adding value labels on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(
        bar.get_x() + bar.get_width()/2,
        yval + 1,          # slight offset above the bar
        f"{yval}%",
        ha="center",
        va="bottom",
        fontsize=11
    )

# Title (optional)
plt.title("Success Rate Comparison Across Formulations", fontsize=14)

plt.tight_layout()
plt.show()

