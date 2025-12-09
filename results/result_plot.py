import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def smooth(data, window_size=50):
    """Stronger smoothing (default = 50). Adjust if needed."""
    return np.convolve(data, np.ones(window_size) / window_size, mode='valid')


def plot_multi_logs():
    log_dir = ""   # change if logs are stored elsewhere

    # Your three log files
    files = {
        #"Simple Formulation": "kinova_env_first.csv",
        #"Elaborate state space and incremental action space formulation": "SAC_final_without_curriculum_2nd_run.csv",
        "Elaborate state and incremental action space with curriculum training": "SAC_Final_formulation_logs.csv",
    }

    MAX_STEP = 350000  # <--- LIMIT PLOT UNTIL THIS STEP

    plt.figure(figsize=(12, 6))

    for label, file_name in files.items():
        file_path = os.path.join(log_dir, file_name)

        if not os.path.exists(file_path):
            print(f"[WARNING] File not found: {file_path}")
            continue

        # Load file
        data = pd.read_csv(file_path, skiprows=1)

        # Compute cumulative timesteps
        data["timestep"] = np.cumsum(data["l"])

        # Truncate to 350k steps
        data = data[data["timestep"] <= MAX_STEP]

        # Smooth rewards
        smoothed_rewards = smooth(data["r"])

        # Truncate x to match smoothed length
        x = data["timestep"].to_numpy()[:len(smoothed_rewards)]
        y = smoothed_rewards

        # Plot
        plt.plot(x, y, label=label)

    # Final plot formatting
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward Comparison Across Training Runs")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_multi_logs()

