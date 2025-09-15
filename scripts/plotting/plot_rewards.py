# * Plotting script for visualizing data from a pickle file.

import matplotlib.pyplot as plt
import numpy as np
import pickle
import os
import seaborn as sns
from pathlib import Path
import matplotlib as mpl

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

font = {'size': 14}
mpl.rc('font', **font)
line = {'linewidth': 2}
mpl.rc('lines', **line)

def _stack_and_stats(runs, key, start = None, end = None):
    """Return (mean, std) over seeds for the given dict key."""
    series = [np.asarray(run[key]) for run in runs.values() if key in run]
    arr    = np.stack(series)  # shape (n_seeds, T)
    arr    = arr[:, start:end]
    return arr.mean(0), arr.std(0)

def plot_advantage_variance(decentralized_data, centralized_data):
    """ Make plot for the variance of the advantage function for centralized vs decentralized critics. """

    # Compute statistics
    end = 400 # None
    dec_leg_mu, dec_leg_std = _stack_and_stats(decentralized_data, "Policy/leg/advantage_variance", end=end)
    cen_leg_mu, cen_leg_std = _stack_and_stats(centralized_data, "Policy/leg/advantage_variance", end=end)
    dec_arm_mu, dec_arm_std = _stack_and_stats(decentralized_data, "Policy/arm/advantage_variance", end=end)
    cen_arm_mu, cen_arm_std = _stack_and_stats(centralized_data, "Policy/arm/advantage_variance", end=end)

    fig, ax = plt.subplots(1, 2, figsize=(12, 5), sharey=False)
    colors  = {"cen": "#1f77b4", "dec": "#d62728"}

    t_leg = np.arange(len(cen_leg_mu))
    t_arm = np.arange(len(cen_arm_mu))

    # ---- leg ----------------------------------------------------
    ax[0].plot(t_leg, dec_leg_mu, color=colors["dec"], label="Decentralised critic")
    ax[0].fill_between(t_leg, dec_leg_mu-dec_leg_std, dec_leg_mu+dec_leg_std, color=colors["dec"], alpha=0.25)

    ax[0].plot(t_leg, cen_leg_mu, color=colors["cen"], label="Centralised critic")
    ax[0].fill_between(t_leg, cen_leg_mu-cen_leg_std, cen_leg_mu+cen_leg_std, color=colors["cen"], alpha=0.25)
                       
    ax[0].set_title("Leg advantage variance")
    ax[0].set_xlabel("Iterations")
    ax[0].set_ylabel("Variance")
    ax[0].legend()
    ax[0].set_xlim(0, len(cen_leg_mu)-1)
    ax[0].set_ylim(0, 1.1*max(max(cen_leg_mu), max(dec_leg_mu)))  # Set y-limits for better visibility

    # ---- arm ----------------------------------------------------
    ax[1].plot(t_arm, dec_arm_mu, color=colors["dec"], label="Decentralised critic")
    ax[1].fill_between(t_arm, dec_arm_mu-dec_arm_std, dec_arm_mu+dec_arm_std, color=colors["dec"], alpha=0.25)

    ax[1].plot(t_arm, cen_arm_mu, color=colors["cen"], label="Centralised critic")
    ax[1].fill_between(t_arm, cen_arm_mu-cen_arm_std, cen_arm_mu+cen_arm_std, color=colors["cen"], alpha=0.25)

    ax[1].set_title("Arm advantage variance")
    ax[1].set_xlabel("Iterations")
    ax[1].set_ylabel("Variance")
    ax[1].legend()
    ax[1].set_xlim(0, len(cen_arm_mu)-1)
    ax[1].set_ylim(0, 1.1*max(max(cen_arm_mu), max(dec_arm_mu)))  # Set y-limits for better visibility

    fig.tight_layout()
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), "figures", "fig_advantage_variance.pdf")
    plt.savefig(output_path)

    # * Plot arm advantage variance alone
    fig, ax = plt.subplots(figsize=(10, 5))

    colors = {"dec": "#d62728", "cen": "#1f77b4"}

    ax.plot(t_arm, dec_arm_mu, color=colors["dec"], label="Multi-Agent DTDE")
    ax.fill_between(t_arm, dec_arm_mu-dec_arm_std, dec_arm_mu+dec_arm_std, color=colors["dec"], alpha=0.25)

    ax.plot(t_arm, cen_arm_mu, color=colors["cen"], label=r"Multi-Agent CTDE $\mathbf{(Ours)}$")
    ax.fill_between(t_arm, cen_arm_mu-cen_arm_std, cen_arm_mu+cen_arm_std, color=colors["cen"], alpha=0.25)

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Variance")
    ax.legend()
    ax.set_xlim(0, len(cen_arm_mu)-1)
    ax.set_ylim(0, 1.1*max(max(cen_arm_mu), max(dec_arm_mu)))  # Set y-limits for better visibility
    tick_locs  = np.arange(0, end, 100)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([f"{v}" for v in tick_locs])
    
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), "figures", "fig_arm_advantage_variance.pdf")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.)

def plot_rewards(vanilla_data, full_vanilla_data, modular_dc_da_data, modular_cc_da_data, modular_cc_ca_data):
    """ Make plot for the rewards of the full vanilla vs modular centralized agent. """

    # Compute statistics
    end = 400 # None
    key = "Episode Reward/leg/tracking_lin_vel_xy"
    vanilla_mu, vanilla_std               = _stack_and_stats(vanilla_data, key, end=end)
    full_vanilla_mu, full_vanilla_std     = _stack_and_stats(full_vanilla_data, key, end=end)
    modular_dc_da_mu, modular_dc_da_std   = _stack_and_stats(modular_dc_da_data, key, end=end)
    modular_cc_da_mu, modular_cc_da_std   = _stack_and_stats(modular_cc_da_data, key, end=end)
    modular_cc_ca_mu, modular_cc_ca_std   = _stack_and_stats(modular_cc_ca_data, key, end=end)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"vanilla": "#2ca02c", "full_vanilla": "#d68910", "modular_dc_da": "#e74c3c", "modular_cc_da": "#3385FF", "modular_cc_ca": "tab:purple"}

    t = np.arange(len(full_vanilla_mu))

    # # ---- Vanilla -------------------------------------------------
    # ax.plot(t, vanilla_mu, color=colors["vanilla"], label="Vanilla")
    # ax.fill_between(t, vanilla_mu-vanilla_std, vanilla_mu+vanilla_std, color=colors["vanilla"], alpha=0.25)

    # ---- Full Vanilla --------------------------------------------
    # ax.plot(t, full_vanilla_mu, color=colors["full_vanilla"], label="Full Vanilla")
    ax.plot(t, full_vanilla_mu, color=colors["full_vanilla"], label="Single Agent")
    ax.fill_between(t, full_vanilla_mu-full_vanilla_std, full_vanilla_mu+full_vanilla_std, color=colors["full_vanilla"], alpha=0.25)

    # ---- Modular Decentralized Critic / Decentralized Actor -----------------------------------
    # ax.plot(t, modular_dc_da_mu, color=colors["modular_dc_da"], label="Modular Decentralized-Critic / Decentralized-Actor")
    ax.plot(t, modular_dc_da_mu, color=colors["modular_dc_da"], label="Multi-Agent DTDE")
    ax.fill_between(t, modular_dc_da_mu-modular_dc_da_std, modular_dc_da_mu+modular_dc_da_std, color=colors["modular_dc_da"], alpha=0.25)

    # ---- Modular Centralized Critic / Decentralized Actor -------------------------------------
    # ax.plot(t, modular_cc_da_mu, color=colors["modular_cc_da"], label="Modular Centralized-Critic / Decentralized-Actor")
    ax.plot(t, modular_cc_da_mu, color=colors["modular_cc_da"], label=r"Multi-Agent CTDE $\mathbf{(Ours)}$")
    ax.fill_between(t, modular_cc_da_mu-modular_cc_da_std, modular_cc_da_mu+modular_cc_da_std, color=colors["modular_cc_da"], alpha=0.25)

    # ---- Modular Centralized Critic / Centralized Actor -------------------------------------
    # ax.plot(t, modular_cc_ca_mu, color=colors["modular_cc_ca"], label="Modular Centralized-Critic / Centralized-Actor")
    ax.plot(t, modular_cc_ca_mu, color=colors["modular_cc_ca"], label="Multi-Agent CTCE")
    ax.fill_between(t, modular_cc_ca_mu-modular_cc_ca_std, modular_cc_ca_mu+modular_cc_ca_std, color=colors["modular_cc_ca"], alpha=0.25)

    # ax.set_title("Rewards Comparison")
    ax.set_xlabel("Iterations")
    ax.set_ylabel("Rewards")
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.01), ncol=2)
    # ax.legend()
    ax.set_xlim(0, len(full_vanilla_mu)-1)
    ax.set_ylim(0, None)
    tick_locs  = np.arange(0, end, 100)
    ax.set_xticks(tick_locs)
    ax.set_xticklabels([f"{v}" for v in tick_locs])
    
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), "figures", "rewards_comparison.pdf")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.)

    # Compute statistics
    start = 80 # None
    end = 160 # None
    key = "Episode Reward/leg/tracking_lin_vel_xy"
    vanilla_mu, vanilla_std               = _stack_and_stats(vanilla_data, key, start=start, end=end)
    full_vanilla_mu, full_vanilla_std     = _stack_and_stats(full_vanilla_data, key, start=start, end=end)
    modular_dc_da_mu, modular_dc_da_std   = _stack_and_stats(modular_dc_da_data, key, start=start, end=end)
    modular_cc_da_mu, modular_cc_da_std   = _stack_and_stats(modular_cc_da_data, key, start=start, end=end)
    modular_cc_ca_mu, modular_cc_ca_std   = _stack_and_stats(modular_cc_ca_data, key, start=start, end=end)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"vanilla": "#2ca02c", "full_vanilla": "#d68910", "modular_dc_da": "#e74c3c", "modular_cc_da": "#3385FF", "modular_cc_ca": "tab:purple"}

    t = np.arange(len(full_vanilla_mu))

    # ---- Full Vanilla --------------------------------------------
    ax.plot(t, full_vanilla_mu, color=colors["full_vanilla"], label="Single Agent", linewidth=4)
    ax.fill_between(t, full_vanilla_mu-full_vanilla_std, full_vanilla_mu+full_vanilla_std, color=colors["full_vanilla"], alpha=0.25)

    # ---- Modular Decentralized Critic / Decentralized Actor -----------------------------------
    ax.plot(t, modular_dc_da_mu, color=colors["modular_dc_da"], label="Multi-Agent DTDE", linewidth=4)
    ax.fill_between(t, modular_dc_da_mu-modular_dc_da_std, modular_dc_da_mu+modular_dc_da_std, color=colors["modular_dc_da"], alpha=0.25)

    # ---- Modular Centralized Critic / Decentralized Actor -------------------------------------
    ax.plot(t, modular_cc_da_mu, color=colors["modular_cc_da"], label=r"Multi-Agent CTDE $\mathbf{(Ours)}$", linewidth=5)
    ax.fill_between(t, modular_cc_da_mu-modular_cc_da_std, modular_cc_da_mu+modular_cc_da_std, color=colors["modular_cc_da"], alpha=0.25)

    # ---- Modular Centralized Critic / Centralized Actor -------------------------------------
    ax.plot(t, modular_cc_ca_mu, color=colors["modular_cc_ca"], label="Multi-Agent CTCE", linewidth=4)
    ax.fill_between(t, modular_cc_ca_mu-modular_cc_ca_std, modular_cc_ca_mu+modular_cc_ca_std, color=colors["modular_cc_ca"], alpha=0.25)

    ax.set_xlabel("Iterations")
    ax.set_ylabel("Rewards")
    # ax.legend()
    # ax.set_xlim(0, len(full_vanilla_mu)-1)
    ax.set_xlim(0, end-start-1)
    ax.set_ylim(1, None)
    # tick_locs  = np.arange(0, end, 100)
    # ax.set_xticks(tick_locs)
    # ax.set_xticklabels([f"{v}" for v in tick_locs])
    
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), "figures", "rewards_comparison_zoom.pdf")
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.)


if __name__ == "__main__":
    DATA_DIR = Path(__file__).resolve().parent / "data"

    # Make sure the data directory exists
    experiment_names = ["Humanoid_Vanilla", 
                        "Humanoid_Full_Vanilla", 
                        "Humanoid_Full_Modular_DecCritic_DecActor",
                        "Humanoid_Full_Modular_CenCritic_DecActor",
                        "Humanoid_Full_Modular_CenCritic_CenActor"]

    for exp in experiment_names:
        (DATA_DIR / exp).mkdir(parents=True, exist_ok=True)

    # Load data from pickle file
    seeds = list(range(10))
    data: dict[str, dict[int, object]] = {}

    for exp in experiment_names:
        exp_dir  = DATA_DIR / exp
        exp_dict = {}

        for seed in seeds:
            file_name = f"{exp}_seed_{seed}_log_buffer.pkl"
            pkl_path  = exp_dir / file_name

            if not pkl_path.exists():
                print(f"[WARN] Missing file: {pkl_path}")
                continue

            with pkl_path.open("rb") as f:
                exp_dict[seed] = pickle.load(f)

        data[exp] = exp_dict

    # Plot the data
    plot_advantage_variance(data['Humanoid_Full_Modular_DecCritic_DecActor'],
                            data['Humanoid_Full_Modular_CenCritic_DecActor'])
    
    plot_rewards(data['Humanoid_Vanilla'],
                 data['Humanoid_Full_Vanilla'],
                 data['Humanoid_Full_Modular_DecCritic_DecActor'],
                 data['Humanoid_Full_Modular_CenCritic_DecActor'],
                 data['Humanoid_Full_Modular_CenCritic_CenActor'])