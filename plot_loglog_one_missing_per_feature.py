import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def plot_loglog_from_folder(save_folder, out_file=None):
    # loads the raw timings saved by figure2_new_one_missing_per_feature.py and plots
    # log(time) vs log(n) with a fitted power-law slope for each method.
    folder = Path(save_folder)
    total_time_gibb_sampl = np.load(folder / "total_time_gibb_sampl.npy")
    total_time_ridge = np.load(folder / "total_time_ridge.npy")
    list_n = np.load(folder / "list_n.npy")
    d, lbd, R = np.load(folder / "params.npy")

    log_n = np.log(list_n)
    log_gibb = np.log(total_time_gibb_sampl)
    log_ridge = np.log(total_time_ridge)

    slope_gibb, intercept_gibb = np.polyfit(log_n, log_gibb, 1)
    slope_ridge, intercept_ridge = np.polyfit(log_n, log_ridge, 1)
    print(f"fitted slope (power-law exponent) gibb sampl: {slope_gibb:.4f}")
    print(f"fitted slope (power-law exponent) IterativeImputer(Ridge): {slope_ridge:.4f}")

    plt.scatter(log_n, log_gibb, label="our gibb sampl (under-param, sampling)", marker="o", color="blue")
    plt.plot(log_n, slope_gibb * log_n + intercept_gibb, color="blue", linestyle="--",
              label=f"fit gibb: slope={slope_gibb:.2f}")
    plt.scatter(log_n, log_ridge, label="IterativeImputer(Ridge, intercept=True)", marker="*", color="green")
    plt.plot(log_n, slope_ridge * log_n + intercept_ridge, color="green", linestyle="--",
              label=f"fit ridge: slope={slope_ridge:.2f}")
    plt.xlabel("log(n)")
    plt.ylabel("log(time)")
    plt.title(f"log(time) vs log(n) (d={int(d)}, one missing component per column, n>d always)")
    plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
    plt.tight_layout()

    if out_file is not None:
        plt.savefig(out_file, dpi=150)
        print(f"saved plot to {out_file}")
    else:
        plt.show()


if __name__ == "__main__":
    save_folder = "results/experiment_2_one_missing_per_feature"
    plot_loglog_from_folder(save_folder)
