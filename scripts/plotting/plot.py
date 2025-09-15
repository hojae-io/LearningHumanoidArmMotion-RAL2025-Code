# * Plotting script for visualizing data from a pickle file.

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import os
import seaborn as sns
from pathlib import Path
from scipy.stats import gaussian_kde
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import MultipleLocator
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap, BoundaryNorm
from scipy.ndimage import gaussian_filter, zoom

np.set_printoptions(threshold=np.inf, linewidth=np.inf)

mpl.rcParams['pdf.fonttype'] = 42
mpl.rcParams['ps.fonttype'] = 42
mpl.rcParams['mathtext.fontset'] = 'stix'
mpl.rcParams['font.family'] = 'STIXGeneral'

font = {'size': 14}
mpl.rc('font', **font)
line = {'linewidth': 3}
mpl.rc('lines', **line)


def trim(data, q=5):
    data = np.array(data)
    lo, hi = np.percentile(data, [q, 100-q])
    return data[(data > lo) & (data < hi)]       

def plot_GRM_distribution(vanilla_data, modular_data, modular_noCAM_data):
    """ Make plot for Ground Reaction Moment distribution for right and left foot"""
    # Extract data
    vanilla_rf_GRM_z = trim(np.array(vanilla_data["rf_GRM_z"]))
    vanilla_lf_GRM_z = trim(np.array(vanilla_data["lf_GRM_z"]))
    modular_rf_GRM_z = trim(np.array(modular_data["rf_GRM_z"]))
    modular_lf_GRM_z = trim(np.array(modular_data["lf_GRM_z"]))
    modular_noCAM_rf_GRM_z = trim(np.array(modular_noCAM_data["rf_GRM_z"]))
    modular_noCAM_lf_GRM_z = trim(np.array(modular_noCAM_data["lf_GRM_z"]))

    labels = ["Vanilla Right Foot", "Vanilla Left Foot",
              "Modular Right Foot", "Modular Left Foot",
              "Modular NoCAM Right Foot", "Modular NoCAM Left Foot"]
    colors = ["tab:pink", "tab:cyan",  # vanilla
              "tab:red", "tab:blue",
              "tab:green", "tab:orange"]  # modular

    # Create subplots
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Left Foot GRM Distribution
    sns.kdeplot(vanilla_lf_GRM_z, ax=axs[0], fill=True, color=colors[1], label=labels[1])
    sns.kdeplot(modular_lf_GRM_z, ax=axs[0], fill=True, color=colors[3], label=labels[3])
    sns.kdeplot(modular_noCAM_lf_GRM_z, ax=axs[0], fill=True, color=colors[5], label=labels[5])
    axs[0].set_title("Left Foot GRM Distribution")
    axs[0].set_xlabel("GRM (Nm)")
    axs[0].set_ylabel("Density")
    axs[0].legend()

    # Right Foot GRM Distribution
    sns.kdeplot(vanilla_rf_GRM_z, ax=axs[1], fill=True, color=colors[0], label=labels[0])
    sns.kdeplot(modular_rf_GRM_z, ax=axs[1], fill=True, color=colors[2], label=labels[2])
    sns.kdeplot(modular_noCAM_rf_GRM_z, ax=axs[1], fill=True, color=colors[4], label=labels[4])
    axs[1].set_title("Right Foot GRM Distribution")
    axs[1].set_xlabel("GRM (Nm)")
    axs[1].set_ylabel("Density")
    axs[1].legend()

    # Adjust layout
    plt.tight_layout()
    
    # Save the plot
    output_path = os.path.join(os.path.dirname(__file__), "figures", "GRM_distribution.pdf")
    plt.savefig(output_path)

    # Plot violin plot
    vanilla_all   = np.abs(vanilla_lf_GRM_z + vanilla_rf_GRM_z)
    nocam_all     = np.abs(modular_noCAM_lf_GRM_z + modular_noCAM_rf_GRM_z)
    ours_all      = np.abs(modular_lf_GRM_z + modular_rf_GRM_z)

    df = pd.DataFrame({
        "GRM":   np.concatenate([ours_all, vanilla_all, nocam_all]),
        "Model": (["With Arms (Ours)"]  * len(ours_all) + 
                  ["Fixed Arms"]        * len(vanilla_all) +
                  ["With Arms w/o CAM"] * len(nocam_all)
                 )
    })

    # colour palette
    palette = {"With Arms (Ours)":  "#3385FF",
               "Fixed Arms":        "#FF6666",
               "With Arms w/o CAM": "#FF9900",
               }

    # ----------------------------------------------------------------------
    #  plotting  -----------------------------------------------------------
    # ----------------------------------------------------------------------
    plt.figure(figsize=(6, 3.1))
    ax = sns.violinplot(
            data=df,
            x="Model", y="GRM",
            hue="Model",          # tell seaborn who owns the colours  ← NEW
            dodge=False,          # hue share the same x-position
            inner=None, cut=0,
            palette=palette,
            linewidth=1.0,
            legend=False,)         # seaborn won’t add a legend now
    for pc in ax.collections:        # each violin is a PolyCollection
        pc.set_alpha(0.7)

    # put star at the top of each violin -------------------------------
    for x_coord, values in enumerate([ours_all, vanilla_all, nocam_all]):
        ax.scatter([x_coord], [values.max()],       # coordinates
                marker="*", s=150, color=list(palette.values())[x_coord],
                edgecolor="none", zorder=20, alpha=0.8)

    # white box-and-whisker overlay ----------------------------------------
    sns.boxplot(data=df, x="Model", y="GRM",
                showcaps=True,  width=0.20,
                boxprops=dict(facecolor="white", edgecolor="black", linewidth=1.5),
                whiskerprops=dict(color="black", linewidth=1.5),
                medianprops=dict(color="black", linewidth=1.5),
                linewidth=1.5,
                showfliers=False, ax=ax)

    # cosmetic tweaks -------------------------------------------------------
    ax.set_xlabel("")
    ax.set_ylabel(r"$M^{z}$ [$N \cdot m$]")
    ax.set_ylim(0, None)                           # start at 0
    ax.tick_params(axis="x")
    for tick in ax.get_xticklabels():
        if tick.get_text() == "With Arms (Ours)":
            tick.set_fontweight("bold")
    ax.tick_params(axis="x",           # affect x-axis
               which="both",       # major and minor
               length=0)           # zero-length → invisible

    ax.spines["top"].set_visible(False)     # hide the top border
    ax.spines["right"].set_visible(False)   # hide the right border

    out_path = FIG_PATH / "fig_GRM_violin_full.pdf"
    plt.savefig(out_path, bbox_inches='tight', pad_inches=0.0)
    print(f"✓ saved {out_path}")

def _domain_box(ax, x_coord, values, *, dx=0.58, color="#111111", alpha=0.15):
    """
    Draw a flat rectangle (x-range dx) × (y-range [min,max]) at z=0
    to visualise the domain of the sample distribution.
    """
    sigma  = values.std(ddof=1)
    tail   = 1
    y_min, y_max = float(values.min()), float(values.max())
    verts = [
        (x_coord - dx/2, y_min - tail * sigma, 0),
        (x_coord + dx/2, y_min - tail * sigma, 0),
        (x_coord + dx/2, y_max + tail * sigma, 0),
        (x_coord - dx/2, y_max + tail * sigma, 0),
    ]
    poly = Poly3DCollection([verts], facecolor=color,
                            edgecolor=color, linewidths=0.8,
                            alpha=alpha, zorder=1)
    ax.add_collection3d(poly)

def _domain_box_abs(ax, x_coord, values, *, dx=0.58,
                color="#111111", alpha=0.15, tail=1.0):
    sigma = values.std(ddof=1)
    y_max = float(values.max()) + tail * sigma       # upper edge

    verts = [
        (x_coord - dx/2, 0.0, 0),
        (x_coord + dx/2, 0.0, 0),
        (x_coord + dx/2, y_max, 0),
        (x_coord - dx/2, y_max, 0),
    ]
    poly = Poly3DCollection([verts], facecolor=color,
                            edgecolor=color, linewidths=0.8,
                            alpha=alpha, zorder=1)
    ax.add_collection3d(poly)

    ax.scatter([x_coord], [y_max], [0.0],      # coordinates
               marker="*", s=200, color=color,
               edgecolor="none", zorder=20)
    
    q75 = np.percentile(values, 75)

    # horizontal dotted line across the floor patch (z = 0)
    ax.plot([x_coord - dx/2, x_coord + dx/2],   # x-span
            [q75, q75],             # y is the quantile
            [0, 0],                 # z = 0 → on the domain plane
            ls="--",  lw=1.2, color="gray", zorder=9)

def _kde_1d(values, n_grid=300, bw="scott", tail=1.0):
    """
    Return (y, z) where y spans [min-τ·σ, max+τ·σ] and z is the KDE density.
    `tail` controls how far beyond the data range to draw (in std-dev units).
    """
    values = np.asarray(values, dtype=float)
    kde    = gaussian_kde(values, bw_method=bw)

    # Span the data range plus a tail on each side
    sigma  = values.std(ddof=1)
    y_min  = values.min() - tail * sigma
    y_max  = values.max() + tail * sigma
    y      = np.linspace(y_min, y_max, n_grid)
    z      = kde(y)
    return y, z

def _kde_1d_abs(values, n_grid=300, bw="scott", tail=1.0):
    """
    Evaluate KDE only for y >= 0.
    """
    values = np.asarray(values, dtype=float)
    kde    = gaussian_kde(values, bw_method=bw)

    sigma  = values.std(ddof=1)
    y_max  = float(values.max()) + tail * sigma
    y      = np.linspace(0.0, y_max, n_grid)   # start at 0
    z      = kde(y)
    return y, z

def _plot_ridge(ax, x, y, z, color, alpha=0.5, linewidth=1.5):
    """
    Draw a KDE ridge at fixed x-coordinate.

    * line  : black curve on top
    * patch : translucent curtain down to z=0
    """
    # -- outline -------------------------------------------------------------
    ax.plot([x] * len(y), y, z, color="k", linewidth=linewidth, zorder=10)

    # -- build a closed polygon (x,y,z)  -------------------------------------
    verts = []
    for yi, zi in zip(y, z):
        verts.append((x, yi, zi))
    for yi in y[::-1]:                    # go back at z = 0
        verts.append((x, yi, 0.0))

    poly = Poly3DCollection([verts], facecolor=color, alpha=alpha, edgecolor=None)
    ax.add_collection3d(poly, zs=x, zdir="x")  # anchor at x = const

def _style_3d_grid(ax, *, lw=0.4, ls=":", color="#A0A0A0", alpha=0.6):
    """Make pane grid lines thin & dotted."""
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update(
            {"linewidth": lw,
             "linestyle": ls,
             "color":     color,
             "alpha":     alpha}
        )

def plot_GRM_3D_distribution(vanilla_data, modular_data, modular_noCAM_data,
                             save_path="figures/GRM_3D_KDE.pdf"):
    """
    Create a ridgeline-style 3-D KDE plot:

    X-axis  → Category (Vanilla / Modular-NoCAM / Modular)
    Y-axis  → GRM (Nm)
    Z-axis  → Probability density
    """

    # --- unpack -----------------------------------------------------------------
    sets = [
        ("Fixed Arms",        trim(np.asarray(vanilla_data["lf_GRM_z"])),
                             trim(np.asarray(vanilla_data["rf_GRM_z"])),
                             ("#FF6666", "#FF6666")),       # light red / red
        ("With Arms w/o CAM", trim(np.asarray(modular_noCAM_data["lf_GRM_z"])),
                             trim(np.asarray(modular_noCAM_data["rf_GRM_z"])),
                             ("#FF9900", "#FF9900")),       # light orange / orange
        (r"With Arms $\mathbf{(Ours)}$",  trim(np.asarray(modular_data["lf_GRM_z"])),
                                         trim(np.asarray(modular_data["rf_GRM_z"])),
                                         ("#3385FF", "#3385FF")),       # light blue / blue
    ]

    fig = plt.figure(figsize=(10, 5))
    ax_left  = fig.add_subplot(1, 2, 1, projection="3d")
    ax_right = fig.add_subplot(1, 2, 2, projection="3d")
    width = 0.6

    for x, (label, lf_vals, rf_vals, (c_light, c_dark)) in enumerate(sets):
        # -- left foot ridge -----------------------------------------------------
        _domain_box(ax_left,  x, lf_vals, dx=0.55, color=c_light, alpha=0.25)
        y_lf, z_lf = _kde_1d(lf_vals, n_grid=300)
        _plot_ridge(ax_left, x, y_lf, z_lf, color=c_light)
        # -- right foot ridge ----------------------------------------------------
        _domain_box(ax_right, x, rf_vals, dx=0.55, color=c_dark, alpha=0.25)
        y_rf, z_rf = _kde_1d(rf_vals, n_grid=300)
        _plot_ridge(ax_right, x, y_rf, z_rf, color=c_dark)

    # ---- cosmetics -------------------------------------------------------------
    for ax, title in zip((ax_left, ax_right), ("Left Foot", "Right Foot")):
        ax.set_facecolor("white")          # axes pane
        ax.xaxis.set_pane_color((1, 1, 1, 1))
        ax.yaxis.set_pane_color((1, 1, 1, 1))
        ax.zaxis.set_pane_color((1, 1, 1, 1))
        ax.set_xticks(range(len(sets)))
        ax.set_xticklabels([s[0] for s in sets], rotation=15, ha="right")
        ax.set_ylabel("GRM (Nm)", labelpad=-2)
        ax.set_zlabel("Density", labelpad=-2)
        ax.tick_params(axis='x', pad=0)
        ax.tick_params(axis='y', pad=-2)
        ax.tick_params(axis='z', pad=1)
        ax.set_title(f"{title} GRM Distribution")
        ax.view_init(elev=25, azim=-35)
        _style_3d_grid(ax, lw=0.5, ls=":", color="#999999")

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_path = FIG_PATH / "GRM_3D_KDE.pdf"
    plt.savefig(save_path, dpi=300)
    print(f"[✓] Saved 3-D KDE plot to {save_path}")

    # plot combined graph
    fig = plt.figure(figsize=(10, 6))
    ax: Axes3D = fig.add_subplot(111, projection="3d")

    for x, (label, lf_vals, rf_vals, (c_light, c_dark)) in enumerate(sets):
        vals = np.abs(lf_vals + rf_vals)
        _domain_box_abs(ax,  x, vals, dx=0.55, color=c_light, alpha=0.25)
        y, z = _kde_1d_abs(vals, n_grid=300)
        _plot_ridge(ax, x, y, z, color=c_light)

    ax.set_facecolor("white")          # axes pane
    ax.xaxis.set_pane_color((1, 1, 1, 1))
    ax.yaxis.set_pane_color((1, 1, 1, 1))
    ax.zaxis.set_pane_color((1, 1, 1, 1))
    ax.set_xticks(range(len(sets)))
    ax.set_xticklabels([s[0] for s in sets], rotation=15, ha="right", fontsize=10)
    ax.set_ylabel(r"$M^z$ [$N \cdot m$]", labelpad=-2, fontsize=12)
    ax.set_zlabel("Density", labelpad=-1, fontsize=10)
    ax.zaxis.set_major_locator(MultipleLocator(0.5))
    ax.set_box_aspect((1, 1, 0.5))
    ax.tick_params(axis='x', pad=-10)
    ax.tick_params(axis='y', pad=-2, labelsize=12)
    ax.tick_params(axis='z', pad=2, labelsize=12)
    # ax.set_title(f"{title} GRM Distribution")
    ax.view_init(elev=25, azim=130)
    ax.invert_xaxis()
    ax.set_ylim(0, None)   # automatic upper limit
    ax.set_zlim(0, None)   # automatic upper limit
    _style_3d_grid(ax, lw=0.5, ls=":", color="#999999")
    ax.xaxis.line.set_visible(False)

    ax.xaxis._axinfo["tick"]["inward_factor"]  = 0.
    ax.xaxis._axinfo["tick"]["outward_factor"] = 0.

    save_path = FIG_PATH / "fig_GRM_3D_KDE_combined.pdf"
    plt.savefig(save_path, bbox_inches='tight', pad_inches=1)
    print(f"[✓] Saved 3-D KDE plot to {save_path}")

def plot_CAM(vanilla_CAM_data, modular_CAM_data):
    """ Plot the CAM z data from the experiment """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True, gridspec_kw={"wspace": 0.1})
    
    # Extract data
    # start, end = 340, 465
    start, end = 340, 449
    t = np.arange(start, end) * CONTROL_DT   # time [s] for the slice
    vanilla_CAM_z = np.array(vanilla_CAM_data["CAM_z"])[start:end]
    vanilla_CAM_base_z = np.array(vanilla_CAM_data["CAM_base_z"])[start:end]
    vanilla_CAM_leg_z  = np.array(vanilla_CAM_data["CAM_leg_z"])[start:end]
    vanilla_CAM_arm_z  = np.array(vanilla_CAM_data["CAM_arm_z"])[start:end]
    vanilla_CAM_des_z  = np.array(vanilla_CAM_data["CAM_des_z"])[start:end]

    modular_CAM_z = np.array(modular_CAM_data["CAM_z"])[start:end]
    modular_CAM_base_z = np.array(modular_CAM_data["CAM_base_z"])[start:end]
    modular_CAM_leg_z  = np.array(modular_CAM_data["CAM_leg_z"])[start:end]
    modular_CAM_arm_z  = np.array(modular_CAM_data["CAM_arm_z"])[start:end]
    modular_CAM_des_z  = np.array(modular_CAM_data["CAM_des_z"])[start:end]

    # colors = ['k', 'g', 'b', 'r', 'gray']
    colors = ['#000000', 'tab:green', 'tab:blue', 'tab:red', '#c1bfbf']

    # plot
    axes[0].plot(t, vanilla_CAM_z, color=colors[0], linewidth=4, label='Total')
    axes[0].plot(t, vanilla_CAM_base_z, color=colors[1], label='Base')
    axes[0].plot(t, vanilla_CAM_leg_z, color=colors[2], label='Leg')
    axes[0].plot(t, vanilla_CAM_arm_z, color=colors[3], label='Arm')
    axes[0].plot(t, vanilla_CAM_des_z, color=colors[4], linestyle='--', label='Reference')

    axes[1].plot(t, modular_CAM_z, color=colors[0], linewidth=4, label='Total')
    axes[1].plot(t, modular_CAM_base_z, color=colors[1], label='Base')
    axes[1].plot(t, modular_CAM_leg_z, color=colors[2], label='Leg')
    axes[1].plot(t, modular_CAM_arm_z, color=colors[3], label='Arm')
    axes[1].plot(t, modular_CAM_des_z, color=colors[4], linestyle='--', label='Reference')

    axes[0].legend(fontsize=14)
    tick_locs  = np.arange(t[0], t[-1] + TICK_STEP, TICK_STEP)
    for ax in axes:                          # axes[0] and axes[1]
        ax.set_xlabel(r"Time [$s$]")
        ax.set_xticks(tick_locs)
        ax.set_xticklabels([f"{v:.1f}" for v in tick_locs])

    axes[0].set_title("Fixed Arms")
    axes[1].set_title("With Arms (Ours)", fontweight='bold')
    axes[0].set_ylabel(r"$\mathbf{k}_G^z$ [$kg \cdot m^2/s$]")

    # * Save the plot
    plt.savefig(FIG_PATH / "fig_CAM_separate.pdf", bbox_inches='tight')
    print(f"[✓] Saved CAM z plot to {FIG_PATH / 'fig_CAM_separate.pdf'}")

    # Plot another figure with the same data but in two vertical subplots
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True, sharey=True, gridspec_kw={"hspace": 0.15})
    axes[0].plot(t, modular_CAM_des_z, color=colors[4], linestyle='--', label='Reference')
    axes[0].plot(t, vanilla_CAM_z, color='#9a9e9c', linewidth=4, label='Fixed Arms')
    axes[0].plot(t, modular_CAM_z, color=colors[0], linewidth=4, label='With Arms (Ours)')
    # plot
    axes[1].plot(t, modular_CAM_des_z, color=colors[4], linestyle='--')
    axes[1].plot(t, modular_CAM_base_z, color=colors[1], label='Base')
    axes[1].plot(t, modular_CAM_leg_z, color=colors[2], label='Legs')
    axes[1].plot(t, modular_CAM_arm_z, color=colors[3], label='Arms')

    leg = axes[0].legend(fontsize=15, bbox_to_anchor=(0., 1.05))   # all entries bold
    leg.get_texts()[2].set_fontweight('bold')  # make the second entry bold
    axes[1].legend(fontsize=15, bbox_to_anchor=(0., 1.05))
    tick_locs  = np.arange(t[0], t[-1] + TICK_STEP, TICK_STEP)
    for ax in axes:                          # axes[0] and axes[1]
        ax.set_xticks(tick_locs)
        ax.set_xticklabels([f"{v:.1f}" for v in tick_locs])
        ax.set_xlim(t[0], t[-1])
        ax.spines["top"].set_visible(False)     # hide the top border
        ax.spines["right"].set_visible(False)   # hide the right border

    axes[1].set_xlabel(r"Time [$s$]", fontsize=15)
    # fig.subplots_adjust(hspace=0.15)      # very small vertical gap

    axes[0].text(-0.12, 1.0, "(a)", transform=axes[0].transAxes,
             ha="left", va="top", fontsize=15)
    axes[1].text(-0.12, 1.0, "(b)", transform=axes[1].transAxes,
             ha="left", va="top", fontsize=15)
    axes[0].set_ylabel(r"$\mathbf{k}_G^z$ [$kg \cdot m^2/s$]", fontsize=16)
    axes[1].set_ylabel(r"$\mathbf{k}_G^z$ [$kg \cdot m^2/s$]", fontsize=16)

    # * Save the plot
    plt.savefig(FIG_PATH / "fig_CAM.pdf", bbox_inches='tight')
    print(f"[✓] Saved CAM z plot to {FIG_PATH / 'fig_CAM.pdf'}")
    fig.canvas.draw()                            # ensure positions are final
    renderer = fig.canvas.get_renderer()

    for idx, ax in enumerate(axes, start=1):
        # bounding box in figure inches
        bbox = ax.get_tightbbox(renderer, call_axes_locator=True)
        bbox = bbox.transformed(fig.dpi_scale_trans.inverted())

        # enlarge 2 % horizontally & vertically
        bbox = bbox.expanded(1.0, 1.0)
        ax.spines["top"].set_visible(False)     # hide the top border
        ax.spines["right"].set_visible(False)   # hide the right border

        out_name = FIG_PATH / f"fig_CAM_{idx}.pdf"
        fig.savefig(out_name, bbox_inches=bbox, pad_inches=0)   # extra 0.02 in pad
        print(f"✓ saved {out_name}")

def plot_push_recovery(push_data):
    """ Plot the push recovery data from the experiment """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8))
    # Extract data
    start, end = 150, 230
    push_t = 3.35

    t = np.arange(start, end) * CONTROL_DT   # time [s] for the slice
    CAM_z = np.array(push_data["CAM_z"])[start:end]
    CAM_base_z = np.array(push_data["CAM_base_z"])[start:end]
    CAM_leg_z  = np.array(push_data["CAM_leg_z"])[start:end]
    CAM_arm_z  = np.array(push_data["CAM_arm_z"])[start:end]
    CAM_des_z  = np.array(push_data["CAM_des_z"])[start:end]

    dCAM_z = np.array(push_data["dCAM_z"])[start:end]
    dCAM_base_z = np.array(push_data["dCAM_base_z"])[start:end]
    dCAM_leg_z  = np.array(push_data["dCAM_leg_z"])[start:end]
    dCAM_arm_z  = np.array(push_data["dCAM_arm_z"])[start:end]
    dCAM_des_z  = np.array(push_data["dCAM_des_z"])[start:end]

    colors = ['#000000', 'tab:green', 'tab:blue', 'tab:red', '#c1bfbf']

    # Plot the CAM data
    axes[0].plot(t, CAM_z, color=colors[0], linewidth=4, label='Total')
    axes[0].plot(t, CAM_base_z, color=colors[1], label='Base')
    axes[0].plot(t, CAM_leg_z, color=colors[2], label='Leg')
    axes[0].plot(t, CAM_arm_z, color=colors[3], label='Arm')
    axes[0].plot(t, CAM_des_z, color=colors[4], linestyle='--', label='Reference')

    # Plot the time derivative of CAM data
    axes[1].plot(t, dCAM_z, color=colors[0], linewidth=4, label='Total')
    axes[1].plot(t, dCAM_base_z, color=colors[1], label='Base')
    axes[1].plot(t, dCAM_leg_z, color=colors[2], label='Leg')
    axes[1].plot(t, dCAM_arm_z, color=colors[3], label='Arm')
    axes[1].plot(t, dCAM_des_z, color=colors[4], linestyle='--', label='Reference')
    axes[1].plot(t, dCAM_des_z, color=colors[4], linestyle='--', linewidth=2)
    axes[1].legend()

    tick_locs  = np.arange(t[0], t[-1] + TICK_STEP, TICK_STEP)
    for ax in axes:                          # axes[0] and axes[1]
        ax.set_xlabel(r"Time [$s$]")
        ax.set_xticks(tick_locs)
        ax.set_xticklabels([f"{v:.1f}" for v in tick_locs])
        # vertical dotted line
        ax.axvline(push_t, color="k", ls="--", lw=1.5)
        # annotation slightly above the top of the data
        ymax = ax.get_ylim()[1]
        ax.annotate("Push",
                    xy=(push_t, ymax),            # arrow tail
                    xytext=(push_t, ymax*0.9),  # text offset
                    textcoords="data",
                    rotation=90,                   # make the text vertical
                    ha="right", va="top",
                    fontsize=14,
                    arrowprops=None)
        
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=5)

    axes[0].set_ylabel(r"$\mathbf{k}_G^z$ [$kg \cdot m^2/s$]")
    axes[1].set_ylabel(r"$\dot{\mathbf{k}}_G^z$ [$N \cdot m$]")

    plt.tight_layout()

    # * Save the plot
    plt.savefig(FIG_PATH / "fig_push_recovery_plot.pdf", bbox_inches='tight')
    print(f"[✓] Saved Push recovery plot to {FIG_PATH / 'fig_push_recovery_plot.pdf'}")

    # * Plot the push sequence thumbnails with time derivative of CAM data
    fig   = plt.figure(figsize=(8, 3))
    ax_ts = fig.add_subplot()

    ax_ts.plot(t, dCAM_base_z, color=colors[1], lw=2, label="Base")
    ax_ts.plot(t, dCAM_arm_z , color=colors[3], lw=2, label="Arm")
    ax_ts.plot(t, dCAM_des_z , color=colors[4], ls="--", lw=2)

    ax_ts.set_xlabel(r"Time [$s$]")
    ax_ts.set_ylabel(r"$\dot{\mathbf{k}}_G^{z}$ [$N \cdot m$]")

    tick_locs  = np.arange(t[0], t[-1] + TICK_STEP, TICK_STEP)
    ax_ts.set_xticks(tick_locs)
    ax_ts.set_xticklabels([f"{v:.1f}" for v in tick_locs])
    ax_ts.set_xlim(t[0], t[-1])  # set x-limits to the time range
    # vertical dotted line
    ax_ts.axvline(push_t, color="k", ls="--", lw=1.5)
    # annotation slightly above the top of the data
    ymax = ax_ts.get_ylim()[1]
    ax_ts.annotate("Push",
                xy=(push_t, ymax),            # arrow tail
                xytext=(push_t, ymax*0.9),  # text offset
                textcoords="data",
                rotation=90,                   # make the text vertical
                ha="right", va="top",
                fontsize=14,
                arrowprops=None)

    ax_ts.legend()
    for spine in ("top", "right"):
        ax_ts.spines[spine].set_visible(False)

    plt.tight_layout(pad=0.3)     # small padding between thumbnails
    plt.savefig(FIG_PATH / "fig_dCM.pdf", bbox_inches='tight', pad_inches=0.)
    print(f"[✓] Saved Push sequence plot to {FIG_PATH / 'fig_dCM.pdf'}")

def plot_survival_heatmaps(survival_data):
    """ Plot the survival rates heatmaps """

    # Extract data
    CMAP = "Blues"
    vanilla_force  = survival_data["vanilla_force"]
    vanilla_torque = survival_data["vanilla_torque"]

    modular_force  = survival_data["modular_force"]
    modular_torque = survival_data["modular_torque"]

    fig, axes = plt.subplots(1, 2, figsize=(8, 4), sharex=False, sharey=True)

    FORCE_RANGE  = (-70.0, 70.0)  # Fx / Fy (N)
    TORQUE_RANGE = (-15.0, 15.0)  # τx / τy (Nm)
    
    # Plot the heatmaps
    im = axes[0].imshow(vanilla_torque, aspect='auto', origin="lower",
                        extent=(TORQUE_RANGE[0], TORQUE_RANGE[1], TORQUE_RANGE[0], TORQUE_RANGE[1]),
                        cmap=CMAP, vmin=0, vmax=1)
    axes[0].set_title("Vanilla Humanoid")
    axes[0].set_xlabel("Torque (Nm)")
    axes[0].set_ylabel("Torque (Nm)")
    fig.colorbar(im, ax=axes[0], label="Survival Rate", fraction=0.046, pad=0.04)

    im = axes[1].imshow(modular_torque, aspect='auto', origin="lower",
                        extent=(TORQUE_RANGE[0], TORQUE_RANGE[1], TORQUE_RANGE[0], TORQUE_RANGE[1]),
                        cmap=CMAP, vmin=0, vmax=1)
    axes[1].set_title("Modular Humanoid")
    axes[1].set_xlabel("Torque (Nm)")
    axes[1].set_ylabel("Torque (Nm)")
    fig.colorbar(im, ax=axes[1], label="Survival Rate", fraction=0.046, pad=0.04)

    # Combined
    THR = 0.45           # survival threshold
    SIG = 0.7            # Gaussian std-dev in *cells*
    UPS = 2              # 2× in each direction → 140×140 grid

    # ------------------------------------------------------------------ #
    # 1) blur the two probability maps
    vanilla_s = gaussian_filter(vanilla_torque, SIG, mode="nearest")
    modular_s = gaussian_filter(modular_torque, SIG, mode="nearest")

    vanilla_hi = zoom(vanilla_s, UPS, order=1, mode="nearest")
    modular_hi = zoom(modular_s, UPS, order=1, mode="nearest")

    # 2) classify on the *smoothed* fields
    succ_van = vanilla_hi >= THR
    succ_mod = modular_hi >= THR

    cat = np.zeros_like(vanilla_hi, dtype=int)          # 0: both fail
    cat[(~succ_van) &  succ_mod] = 1                  # modular only
    cat[ succ_van  &  succ_mod] = 2                  # both succeed
    # ------------------------------------------------------------------ #

    # separate masks so the grid can go between grey & colours
    fail_mask    = np.ma.masked_where(cat != 0, cat)            # only 0
    succ_mask    = np.ma.masked_where(cat == 0, cat)            # 1 & 2

    # colour maps
    succ_cmap  = ListedColormap(["#3385FF", "#FF6666"])         # blue & red
    succ_norm  = BoundaryNorm([0.5, 1.5, 2.5],  succ_cmap.N)

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.set_aspect("equal", adjustable="box")        #  ⟵  key line

    # 3-c  coloured success overlay
    ax.imshow(succ_mask, origin="lower",
            extent=(*TORQUE_RANGE,*TORQUE_RANGE),
            cmap=succ_cmap, norm=succ_norm,
            interpolation="none", zorder=2)

    ax.set_xlabel(r"$\tau_x$ [$N \cdot m$]", labelpad=1)
    ax.set_ylabel(r"$\tau_y$ [$N \cdot m$]", labelpad=-10)
    
    patches = [mpatches.Patch(color="#FF6666", label="Fixed Arms"),
               mpatches.Patch(color="#3385FF", label="With Arms (Ours)")]
               
    ax_handle = ax.legend(handles=patches, loc="upper left", frameon=True, fontsize=12, ncols=2, bbox_to_anchor=(0., 1.02), )
    ax_handle.get_texts()[1].set_fontweight('bold')  # make the second entry bold

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    success_set_ratio = 100 * (cat == 1).sum() / ((cat == 1).sum()+(cat == 2).sum())
    print('Success set ratio:', success_set_ratio, '%')

    plt.savefig(FIG_PATH / "fig_survival_heatmap.pdf", bbox_inches="tight", pad_inches=0.)


if __name__ == "__main__":
    FIG_PATH  = Path(__file__).resolve().parent / "figures"
    DATA_PATH = Path(__file__).resolve().parent / "data"
    CONTROL_DT = 0.02  # Control time step in seconds
    TICK_STEP = 0.5 # Ticks every 0.5 seconds
    # Load data from pickle file
    vanilla = "Humanoid_Vanilla_model_1000_log_buffer.pkl"
    modular = "Humanoid_Full_Modular_model_1000_log_buffer.pkl"
    modular_noCAM = "Humanoid_Full_Modular_noCAM_model_1000_log_buffer.pkl"

    vanilla_path = DATA_PATH / vanilla
    modular_path = DATA_PATH / modular
    modular_noCAM_path = DATA_PATH / modular_noCAM

    with vanilla_path.open("rb") as f:
        vanilla_data = pickle.load(f)

    with modular_path.open("rb") as f:
        modular_data = pickle.load(f)

    with modular_noCAM_path.open("rb") as f:
        modular_noCAM_data = pickle.load(f)

    # Plot the data
    plot_GRM_distribution(vanilla_data, modular_data, modular_noCAM_data)
    plot_GRM_3D_distribution(vanilla_data, modular_data, modular_noCAM_data)

    # * Plot CAM data
    vanilla_CAM_exp = "Humanoid_Vanilla_2025-06-09_16-48-25_CAM_log_buffer.pkl"
    modular_CAM_exp = "Humanoid_Full_Modular_2025-06-10_02-44-32_CAM_log_buffer.pkl"
    push_exp = "Humanoid_Full_Modular_2025-06-11_22-07-21_push_log_buffer.pkl"

    vanilla_CAM_path = DATA_PATH / vanilla_CAM_exp
    modular_CAM_path = DATA_PATH / modular_CAM_exp
    push_path = DATA_PATH / push_exp

    with vanilla_CAM_path.open("rb") as f:
        vanilla_CAM_data = pickle.load(f)
    with modular_CAM_path.open("rb") as f:
        modular_CAM_data = pickle.load(f)
    with push_path.open("rb") as f:
        push_data = pickle.load(f)

    # Plot the command data
    plot_CAM(vanilla_CAM_data, modular_CAM_data)
    plot_push_recovery(push_data)

    # * Plot survival rates
    survival_exp = "humanoid_survival_probs.pkl"

    survival_path = DATA_PATH / survival_exp

    with survival_path.open("rb") as f:
        survival_data = pickle.load(f)

    plot_survival_heatmaps(survival_data)
