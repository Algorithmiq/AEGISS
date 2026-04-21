
from __future__ import annotations

import os
from typing import Any, Union

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.patches as patches
import matplotlib.lines as mlines
from collections import Counter


def plot_pie_contribution(ao_contributions: list[tuple[int, float]],
                          ao_labels: Union[list[str], str],
                          save_path: str) -> None:
    pass

def plot_ao_contribution(ao_contributions: list[tuple[int, float]],
                         ao_labels: Union[list[str], str],
                         save_path: str) -> None:
    pass



def SOE_heatmap(
    so1_entropies: list[tuple[int, float]],
    n_occ: int = 0,
    folder: str = './',
    filename: str = "SOE_plot",
) -> None:
    """Plot a single-orbital entropy (SOE) heatmap.

    Displays a 5-row matrix where the top three rows show the relative S(1)
    values as a continuous colormap, and the bottom two rows show binary
    threshold indicators. Saves the figure as a PDF.

    Parameters
    ----------
    so1_entropies:
        List of (orbital_index, relative_entropy) tuples, where the entropy
        is already normalised by ``max_so1``.
    n_occ:
        Number of occupied orbitals. When non-zero, a dashed vertical line is
        drawn at ``n_occ + 0.5`` to mark the occupied/virtual boundary.
    folder:
        Directory where the PDF is saved.
    filename:
        Base name of the output file (without extension).
    """
    so1_abs = np.array([x[1] for x in so1_entropies])
    max_so1 = max(so1_abs)
    so1_rel = so1_abs / max_so1
    large_cas = len(so1_entropies)

    data = np.vstack([
        so1_abs,
        so1_abs,
        so1_abs,
        (so1_rel >= 0.1).astype(float),
        (so1_abs > 0.06).astype(float),
    ])
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(data, cmap='magma', aspect='auto', vmin=0, vmax=1, interpolation='nearest')
    fig.colorbar(im, ax=ax, orientation='vertical', fraction=0.02, pad=0.04,
                 label=r"$S(1)_i\,/\,S(1)_\mathrm{max}$")

    ax.set_yticks([1, 3, 4])
    ax.set_yticklabels([r"$S(1)_i$", r"$\tau_{E}$", r"$Emp$"], fontsize=16)
    ax.set_xticks(np.arange(large_cas))
    ax.set_xticklabels([str(i[0]) for i in so1_entropies], fontsize=10, rotation=90)
    ax.set_xlim(-0.5, large_cas - 0.5)
    ax.xaxis.set_ticks_position('bottom')

    if n_occ:
        ax.axvline(x=n_occ + 0.5, ymin=0.4, color='red', linestyle='--', linewidth=4)
    ax.axhline(y=2.5, color='red', linewidth=2)
    ax.axhline(y=3.5, color='red', linewidth=2)

    ax.set_title(f"Max SOE: {max_so1:.5f}", fontsize=16)
    fig.tight_layout()
    fig.savefig(os.path.join(folder, filename + ".pdf"), format='pdf', bbox_inches='tight')
    plt.close(fig)

def _detect_plateaus(
    y_values: np.ndarray,
    indices_above: dict[int, np.ndarray],
    plateau_size: int,
) -> tuple[list[float], list[float], list[int], list[list[float]]]:
    tmp_val = y_values[0]
    plateau_vector: list[float] = []
    plateau_index: list[float] = []
    starting_plateaus_indices: list[int] = []
    thresh_count = 1
    is_plateau = False
    rect_size: list[list[float]] = []
    current_bounds: list[float] = []

    for i, y_value in enumerate(y_values):
        if y_value == tmp_val:
            thresh_count += 1
            if not current_bounds:
                current_bounds.append((i - 1.5) / 100)
        else:
            if thresh_count >= plateau_size:
                current_bounds.extend([(i - 0.5) / 100, len(indices_above[i - 1])])
                rect_size.append(current_bounds)
            current_bounds = []
            thresh_count = 1
            tmp_val = y_value
            is_plateau = False

        if thresh_count >= plateau_size:
            if not is_plateau:
                starting_plateaus_indices.append(i)
                is_plateau = True
            plateau_vector.append(y_value)
            plateau_index.append(i / 100)

    return plateau_vector, plateau_index, starting_plateaus_indices, rect_size


def _define_cas_size(selected_ones: np.ndarray, occ_index: int) -> tuple[int, int]:
    return (len(selected_ones), 2 * int(np.sum(selected_ones < occ_index)))

def TH_plots(
    so1_entropies: list[tuple[int, float]],
    folder: str = './',
    occ_index: int = 0,
    plateau_size: int = 10,
    filename: str = "TH_plot",
) -> None:
    """Plot the threshold curve for single-orbital entropy (TH) analysis.

    For each threshold value τ ∈ [0, 1] (as a fraction of S(1)_max), computes
    the number of orbitals whose relative entropy exceeds τ and plots the
    resulting staircase curve. Plateau regions — where the orbital count is
    stable across a range of thresholds — are highlighted and annotated with
    the implied CAS(n_elec, n_orbs) sizes. Saves the figure as a PDF.

    Parameters
    ----------
    so1_entropies:
        List of (orbital_index, relative_entropy) tuples, where the entropy
        is already normalised by ``max_so1``.
    folder:
        Directory where the PDF is saved.
    plateau_size:
        Minimum number of consecutive threshold steps (out of 101) for a
        flat region to be considered a plateau and annotated on the plot.
    filename:
        Base name of the output file (without extension).
    """


    so1_abs = np.array([x[1] for x in so1_entropies])
    max_so1 = max(so1_abs)
    so1_rel = so1_abs / max_so1
    large_cas = len(so1_entropies)
    orbitals_index = np.array([x[0] for x in so1_entropies])

    thumb_rule = int(np.sum(so1_abs > 0.06))
    shade = int(np.sum(so1_rel >= 0.1))

    x_values = np.arange(0, 1.01, 0.01)
    y_values = np.zeros(101)
    indices_above: dict[int, np.ndarray] = {}
    for i, x_value in enumerate(x_values):
        mask = so1_rel > x_value
        indices_above[i] = orbitals_index[mask]
        y_values[i] = int(np.sum(mask))

    plateau_vector, plateau_index, starting_plateaus_indices, rect_size = (
        _detect_plateaus(y_values, indices_above, plateau_size)
    )

    fig, ax = plt.subplots(figsize=(10, 5))
    c = sns.color_palette("magma")

    ax.fill_between(np.arange(-0.05, 1.06, 0.01), shade, 100, color='grey', alpha=0.3)
    ax.plot(x_values, y_values, marker='s', markersize=6, ls="", color=c[0])
    ax.plot(plateau_index, plateau_vector, ls="", marker="x", color='teal', markersize=5)
    ax.set_ylabel("# Selected MOs", fontsize=20)
    ax.set_xlabel(r"Threshold as % of largest $S(1)_\mathrm{max}$", fontsize=18)
    ax.grid(True, alpha=0.3)

    if large_cas > 50:
        ax.set_yticks(np.arange(0, large_cas + 1, 5))
        ax.tick_params(axis='y', labelsize=14)
        ax.set_ylim(0, shade)
    else:
        ax.set_yticks(np.arange(0, large_cas + 1, 2))
        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylim(0, large_cas + 2)

    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis='x', labelsize=18)
    ax.set_xlim(-0.05, 1.05)
    ax.annotate(xy=(0.82, thumb_rule + 0.6), text="Empirical", fontsize=16, color='purple')
    ax.annotate(xy=(0.82, shade + 0.6), text=r"$0.1\cdot S(1)_\mathrm{max}$", fontsize=16, color='grey')
    ax.axhline(y=thumb_rule, color='purple', linestyle='--', zorder=0)
    ax.axhline(y=y_values[10], color='grey', linestyle='--', zorder=0)

    for i, idx in enumerate(starting_plateaus_indices):
        cas = _define_cas_size(indices_above[idx], occ_index=occ_index)
        x_start, x_end, n_orbs = rect_size[i]
        width = x_end - x_start
        d = 1.5 if large_cas > 40 else 0.5
        rect = patches.Rectangle(
            (x_start, n_orbs - d), width, 2 * d,
            edgecolor='teal', facecolor='none',
        )
        ax.add_patch(rect)
        ax.annotate(
            xy=(x_start + width / 2, y_values[idx] + d * 2.5),
            text=f"P#{i+1}\nCAS({cas[1]},{cas[0]})",
            ha='center', fontsize=14, color='teal', backgroundcolor='white',
            bbox={'edgecolor': 'teal', 'facecolor': 'white'},
        )
        print(f"\nPlateau #{i+1}:\nN_MOs: {len(indices_above[idx])}\nList of MOs: {indices_above[idx]}\n")

    fig.tight_layout()
    fig.savefig(os.path.join(folder, filename + ".pdf"), format='pdf')
    plt.close(fig)
