"""Plotting utilities for AEGISS active space diagnostics."""

from __future__ import annotations

import logging
import string
from pathlib import Path
from typing import TYPE_CHECKING, Any

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import patches

if TYPE_CHECKING:
    import pyscf

logger = logging.getLogger(__name__)


def _draw_pie_subplot(
    ax: plt.Axes,
    data: np.ndarray,
    labels_names: list[str],
    orb_id: int,
    min_pct: float,
) -> None:
    """Draw a single pie chart for one orbital's AO composition.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes on which to draw the pie chart.
    data : np.ndarray
        Fractional contributions for each AO group (must sum to ≤ 1).
    labels_names : list of str
        Slice labels corresponding to each entry in *data*.
    orb_id : int
        Orbital index displayed as the subplot title.
    min_pct : float
        Minimum fractional contribution below which the slice label and
        percentage text are hidden.
    """
    n_labels = len(data)
    explode = np.zeros(n_labels)
    explode[np.argmax(data)] = 0.1

    _, texts, autotexts = ax.pie(  # pyright: ignore[reportAssignmentType]
        data,
        labels=labels_names,
        explode=explode,
        autopct="%.00f%%",
        colors=sns.color_palette("crest", n_labels),
        startangle=0,
    )

    for k, (text, autotext) in enumerate(zip(texts, autotexts, strict=False)):
        text.set_fontsize(14)
        autotext.set_fontsize(16)
        if data[k] < min_pct:
            text.set_text("")
            autotext.set_text("")

    ax.set_title(f"Orbital: {orb_id}", fontsize=16)


def _save_contributions(
    shape: tuple[int, int],
    mol: pyscf.gto.Mole,
    orbs: np.ndarray | list[int],
    ao_labels: list[str | list[str]],
    labels_names: list[str],
    mo: np.ndarray,
    folder: str = "./",
    filename: str = "major_composition",
    min_pct: float = 0.01,
) -> None:
    """Compute AO compositions and save a grid of pie charts as a PDF.

    Computes the fractional contribution of each AO group to every orbital in
    *orbs*, lays out the charts in a *shape* grid, and writes a single PDF file.

    Parameters
    ----------
    shape : tuple[int, int]
        Grid dimensions ``(n_rows, n_cols)`` for the subplot layout.
    mol : pyscf.gto.Mole
        PySCF Mole object used to resolve AO labels.
    orbs : np.ndarray or list of int
        Indices of the molecular orbitals to plot.
    ao_labels : list of str or list of str
        AO label strings recognised by ``mol.search_ao_label``.
    labels_names : list of str
        Human-readable names for each AO group, used as pie slice labels.
    mo : np.ndarray, shape (n_ao, n_mo)
        MO coefficient matrix for one spin channel.
    folder : str
        Directory where the PDF is saved.
    filename : str
        Output filename without extension.
    min_pct : float
        Minimum fractional contribution below which the slice label and
        percentage are hidden.
    """
    n_orbs, n_labels = len(orbs), len(ao_labels)
    total_ao_idx = np.unique(
        np.concatenate([mol.search_ao_label(lbl) for lbl in ao_labels])
    )
    mo_orbs = mo[:, list(orbs)]
    total_norm = np.sum(mo_orbs[total_ao_idx, :] ** 2, axis=0)

    data_list = np.zeros((n_orbs, n_labels))
    for i, label in enumerate(ao_labels):
        idx = mol.search_ao_label(label)
        data_list[:, i] = np.sum(mo_orbs[idx, :] ** 2, axis=0) / total_norm

    x, y = shape
    fig, axes = plt.subplots(x, y, figsize=(5 * y, 5 * x))

    for i in range(x):
        for j in range(y):
            ax = axes[j] if x == 1 else axes[i, j]

            if i * y + j >= n_orbs:
                ax.axis("off")
                continue

            _draw_pie_subplot(
                ax,
                data=data_list[i * y + j],
                labels_names=labels_names,
                orb_id=orbs[i * y + j],
                min_pct=min_pct,
            )

    logger.info("Saving %s.pdf", filename)
    fig.tight_layout()
    fig.savefig(Path(folder) / (filename + ".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def plot_pie_contribution(
    mol: pyscf.gto.Mole,
    mf: Any,
    ao_labels: list[str | list[str]] | str,
    labels_names: list[str | list[str]],
    orbs: list[int] | np.ndarray,
    filename: str = "composition",
    folder: str = "./",
    orb_for_row: int = 4,
    min_pct: float = 0.01,
) -> None:
    """Plot per-orbital AO composition as a grid of pie charts.

    For each orbital in *orbs*, draws a pie chart showing the fractional
    contribution of each AO group defined in *ao_labels*. Slices whose
    contribution is below *min_pct* are rendered without label or percentage.
    Saves one PDF per spin channel (``_alpha`` / ``_beta`` for unrestricted,
    ``_alpha`` only for restricted).

    Parameters
    ----------
    mol:
        PySCF Mole object, used to resolve AO labels.
    mf:
        Converged mean-field object. ``mf.mo_coeff`` may be ``(n_ao, n_mo)``
        (restricted) or ``(2, n_ao, n_mo)`` (unrestricted).
    ao_labels:
        AO label strings recognised by ``mol.search_ao_label``,
        e.g. ``["C 2pz", "Fe 3d"]``. A bare string is wrapped into a list.
    labels_names:
        Human-readable names for each entry in *ao_labels*, used as pie
        slice labels. Must have the same length as *ao_labels*.
    orbs:
        Indices of the molecular orbitals to plot.
    filename:
        Base name for the output PDF (spin suffix appended automatically).
    folder:
        Directory where the PDF is saved.
    orb_for_row:
        Number of orbitals per row in the subplot grid.
    min_pct:
        Minimum fractional contribution (0-1) below which the slice label
        and percentage are hidden. Default is 0.01 (1 %).
    """
    if isinstance(ao_labels, str):
        ao_labels = [ao_labels]
    if len(ao_labels) != len(labels_names):
        raise ValueError(
            f"ao_labels and labels_names must have the same length, "
            f"got {len(ao_labels)} and {len(labels_names)}."
        )
    if len(orbs) == 0:
        raise ValueError("orbs must not be empty.")
    if orb_for_row <= 0:
        raise ValueError(f"orb_for_row must be a positive integer, got {orb_for_row}.")

    str_labels_names = [" ".join(n) if isinstance(n, list) else n for n in labels_names]
    shape = (int(np.ceil(len(orbs) / orb_for_row)), orb_for_row)
    common = {
        "shape": shape,
        "mol": mol,
        "orbs": orbs,
        "ao_labels": ao_labels,
        "labels_names": str_labels_names,
        "folder": folder,
        "min_pct": min_pct,
    }

    if np.ndim(mf.mo_coeff) == 3:
        moa, mob = mf.mo_coeff
        _save_contributions(**common, mo=moa, filename=filename + "_alpha")
        _save_contributions(**common, mo=mob, filename=filename + "_beta")
    else:
        _save_contributions(**common, mo=mf.mo_coeff, filename=filename + "_alpha")


def plot_ao_contribution(
    ao_contributions: dict[int, list[tuple[int, float]]],
    save_path: str,
    ao_labels: dict[int, str | list[str]] | None = None,
    mo_labels: list[int] | np.ndarray | None = None,
    mos_per_plot: int = 200,
) -> None:
    """Plot AO contributions as a heatmap, one row per AO group.

    Each row is the contribution array for one AO group, normalised by its own
    maximum so all rows share the same [0, 1] colour scale.

    Parameters
    ----------
    ao_contributions:
        Dict mapping integer group index to a list of (orbital_index, weight)
        pairs covering all orbitals in the search space.
    save_path:
        Output path/stem; ``_heatmap.pdf`` is appended automatically.
    ao_labels:
        Dict mapping each group index to its label string (or list of strings,
        which are joined with a space). When *None* (default), letter
        identifiers A, B, C, … are used as y-tick labels.
    mo_labels:
        Orbital indices to use as x-tick labels. Defaults to sequential
        integers when *None*.
    mos_per_plot:
        Maximum number of MOs shown in each subplot panel.
    """
    keys = list(ao_contributions.keys())

    def _ytick(group_idx: int, position: int) -> str:
        if ao_labels is not None:
            raw = ao_labels.get(group_idx, str(group_idx))
            return "\n".join(raw) if isinstance(raw, list) else raw
        return (
            string.ascii_uppercase[position]
            if position < 26
            else string.ascii_uppercase[position // 26 - 1]
            + string.ascii_uppercase[position % 26]
        )

    heat_map_matrix: list[np.ndarray] = []
    valid_yticks: list[str] = []
    for i, group_idx in enumerate(keys):
        weights = ao_contributions[group_idx]
        if not weights:
            continue
        vals = np.array([w for _, w in weights])
        max_val = np.max(vals)
        heat_map_matrix.append(vals if max_val == 0 else vals / max_val)
        valid_yticks.append(_ytick(group_idx, i))

    if not heat_map_matrix:
        return

    matrix = np.array(heat_map_matrix)  # (n_groups, n_mos)
    n_mos = matrix.shape[1]
    n_plots = int(np.ceil(n_mos / mos_per_plot))
    vmin, vmax = 0.0, 1.0

    fig = plt.figure(figsize=(25, 3 * n_plots))
    gs = fig.add_gridspec(n_plots, 1, wspace=0.3, hspace=0.25)

    for i in range(n_plots):
        start = i * mos_per_plot
        end = min(start + mos_per_plot, n_mos)
        subset = matrix[:, start:end]

        ax = fig.add_subplot(gs[i, 0])
        sns.heatmap(
            subset,
            ax=ax,
            cbar=False,
            cmap="magma",
            vmin=vmin,
            vmax=vmax,
            yticklabels=valid_yticks,
        )
        ax.set_ylabel("Atomic Orbitals")
        ax.set_xticks(np.arange(end - start) + 0.5)
        if mo_labels is not None:
            ax.set_xticklabels([str(m) for m in mo_labels[start:end]], rotation=90)

    cbar_ax = fig.add_axes((0.92, 0.15, 0.02, 0.7))
    sm = plt.cm.ScalarMappable(
        cmap="magma", norm=mcolors.Normalize(vmin=vmin, vmax=vmax)
    )
    sm.set_array([])
    fig.colorbar(sm, cax=cbar_ax)

    plt.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    fig.savefig(Path(save_path + "_heatmap.pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def soe_heatmap(
    so1_entropies: list[tuple[int, float]],
    n_occ: int = 0,
    folder: str = "./",
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

    data = np.vstack(
        [
            so1_abs,
            so1_abs,
            so1_abs,
            (so1_rel >= 0.1).astype(float),
            (so1_abs > 0.06).astype(float),
        ]
    )
    fig, ax = plt.subplots(figsize=(8, 3))
    im = ax.imshow(
        data, cmap="magma", aspect="auto", vmin=0, vmax=1, interpolation="nearest"
    )
    fig.colorbar(
        im,
        ax=ax,
        orientation="vertical",
        fraction=0.02,
        pad=0.04,
        label=r"$S(1)_i\,/\,S(1)_\mathrm{max}$",
    )

    ax.set_yticks([1, 3, 4])
    ax.set_yticklabels([r"$S(1)_i$", r"$\tau_{E}$", r"$Emp$"], fontsize=16)
    ax.set_xticks(np.arange(large_cas))
    ax.set_xticklabels([str(i[0]) for i in so1_entropies], fontsize=10, rotation=90)
    ax.set_xlim(-0.5, large_cas - 0.5)
    ax.xaxis.set_ticks_position("bottom")

    if n_occ:
        ax.axvline(x=n_occ + 0.5, ymin=0.4, color="red", linestyle="--", linewidth=4)
    ax.axhline(y=2.5, color="red", linewidth=2)
    ax.axhline(y=3.5, color="red", linewidth=2)

    ax.set_title(f"Max SOE: {max_so1:.5f}", fontsize=16)
    fig.tight_layout()
    fig.savefig(Path(folder) / (filename + ".pdf"), format="pdf", bbox_inches="tight")
    plt.close(fig)


def _detect_plateaus(
    y_values: np.ndarray,
    indices_above: dict[int, np.ndarray],
    plateau_size: int,
) -> tuple[list[float], list[float], list[int], list[list[float]]]:
    """Identify flat regions (plateaus) in a threshold curve.

    Scans *y_values* for runs of identical consecutive values and marks those
    of length ≥ *plateau_size* as plateaus. Returns the data needed to annotate
    the threshold curve plot.

    Parameters
    ----------
    y_values : np.ndarray, shape (n_thresholds,)
        Number of orbitals above each threshold step.
    indices_above : dict[int, np.ndarray]
        Mapping from threshold step index to the array of orbital indices above
        that threshold.
    plateau_size : int
        Minimum run length (in threshold steps) for a flat region to be
        considered a plateau.

    Returns:
    -------
    plateau_vector : list of float
        y-values belonging to plateau points.
    plateau_index : list of float
        x-coordinates (threshold fractions) of plateau points.
    starting_plateaus_indices : list of int
        Index into *y_values* at which each plateau begins.
    rect_size : list of list of float
        For each plateau: ``[x_start, x_end, n_orbs]``, used to draw bounding
        rectangles on the plot.
    """
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
    """Return ``(n_orbs, n_elec)`` for a CAS defined by a set of orbital indices.

    Parameters
    ----------
    selected_ones : np.ndarray
        Orbital indices in the candidate active space.
    occ_index : int
        Index of HOMO + 1 (first virtual orbital). Orbitals with index strictly
        below this value are treated as occupied.

    Returns:
    -------
    cas_size : tuple[int, int]
        ``(n_orbs, n_elec)`` where *n_orbs* is the total number of selected
        orbitals and *n_elec* is twice the number of selected occupied orbitals.
    """
    return (len(selected_ones), 2 * int(np.sum(selected_ones < occ_index)))


def th_plots(
    so1_entropies: list[tuple[int, float]],
    folder: str = "./",
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

    ax.fill_between(np.arange(-0.05, 1.06, 0.01), shade, 100, color="grey", alpha=0.3)
    ax.plot(x_values, y_values, marker="s", markersize=6, ls="", color=c[0])
    ax.plot(
        plateau_index, plateau_vector, ls="", marker="x", color="teal", markersize=5
    )
    ax.set_ylabel("# Selected MOs", fontsize=20)
    ax.set_xlabel(r"Threshold as % of largest $S(1)_\mathrm{max}$", fontsize=18)
    ax.grid(visible=True, alpha=0.3)

    if large_cas > 50:
        ax.set_yticks(np.arange(0, large_cas + 1, 5))
        ax.tick_params(axis="y", labelsize=14)
        ax.set_ylim(0, shade)
    else:
        ax.set_yticks(np.arange(0, large_cas + 1, 2))
        ax.tick_params(axis="y", labelsize=12)
        ax.set_ylim(0, large_cas + 2)

    ax.set_xticks(np.arange(0, 1.1, 0.1))
    ax.tick_params(axis="x", labelsize=18)
    ax.set_xlim(-0.05, 1.05)
    ax.annotate(
        xy=(0.82, thumb_rule + 0.6), text="Empirical", fontsize=16, color="purple"
    )
    ax.annotate(
        xy=(0.82, shade + 0.6),
        text=r"$0.1\cdot S(1)_\mathrm{max}$",
        fontsize=16,
        color="grey",
    )
    ax.axhline(y=thumb_rule, color="purple", linestyle="--", zorder=0)
    ax.axhline(y=y_values[10], color="grey", linestyle="--", zorder=0)

    for i, idx in enumerate(starting_plateaus_indices):
        cas = _define_cas_size(indices_above[idx], occ_index=occ_index)
        x_start, x_end, n_orbs = rect_size[i]
        width = x_end - x_start
        d = 1.5 if large_cas > 40 else 0.5
        rect = patches.Rectangle(
            (x_start, n_orbs - d),
            width,
            2 * d,
            edgecolor="teal",
            facecolor="none",
        )
        ax.add_patch(rect)
        ax.annotate(
            xy=(x_start + width / 2, y_values[idx] + d * 2.5),
            text=f"P#{i + 1}\nCAS({cas[1]},{cas[0]})",
            ha="center",
            fontsize=14,
            color="teal",
            backgroundcolor="white",
            bbox={"edgecolor": "teal", "facecolor": "white"},
        )
        logger.info(
            "\nPlateau #%d:\nN_MOs: %d\nList of MOs: %s\n",
            i + 1,
            len(indices_above[idx]),
            indices_above[idx],
        )

    fig.tight_layout()
    fig.savefig(Path(folder) / (filename + ".pdf"), format="pdf")
    plt.close(fig)


def _helper_mi(
    minfo: np.ndarray, x: np.ndarray, y: np.ndarray, ax: plt.Axes
) -> list[plt.Line2D]:
    """Helper function to plot lines between orbitals based on mutual information (MI) values.

    Parameters:
    -----------
    minfo: np.ndarray
        Mutual Information matrix between orbitals.
    x: np.ndarray
        x-coordinates of the orbital positions on the plot.
    y: np.ndarray
        y-coordinates of the orbital positions on the plot.
    ax: plt.Axes
        Matplotlib Axes object on which to plot the lines.

    Returns:
    --------
    list of plt.Line2D
        Legend handles for the different MI thresholds.
    """
    # Set up line styles and alpha values for different MI thresholds
    N = minfo.shape[0]
    mi_thresholds = [1e-1, 1e-2, 1e-3]
    mi_linestyles = ["-", "--", ":"]
    alphas = [0.9, 0.6, 0.3]

    # Draw lines between orbitals based on MI values and thresholds
    for i in range(N):
        for j in range(i + 1, N):
            mi_value = minfo[i, j]
            for thresh, ls, a in zip(
                mi_thresholds, mi_linestyles, alphas, strict=False
            ):
                if mi_value >= thresh:
                    ax.plot(
                        [x[i], x[j]],
                        [y[i], y[j]],
                        color="grey",
                        linewidth=2,
                        linestyle=ls,
                        alpha=a,
                    )
                    break

    return [
        plt.Line2D([0], [0], color="grey", linestyle=ls, label=f"MI ≥ {thresh}")
        for thresh, ls in zip(mi_thresholds, mi_linestyles, strict=False)
    ]


def _helper_soe(
    soe1: list[tuple[int, float]], x: np.ndarray, y: np.ndarray, ax: plt.Axes
) -> list[plt.Line2D]:
    """Helper function to plot orbital markers based on single-orbital entropy (SOE) values.

    Parameters:
    -----------
    soe1: list of tuples
        List of (orbital_index, single_orbital_entropy) tuples for all orbitals.
    x: np.ndarray
        x-coordinates of the orbital positions on the plot.
    y: np.ndarray
        y-coordinates of the orbital positions on the plot.
    ax: plt.Axes
        Matplotlib Axes object on which to plot the orbital markers.

    Returns:
    --------
    list of plt.Line2D
        Legend handles for the different SOE thresholds.
    """
    # Set up marker sizes and colors for different SOE thresholds
    N = len(soe1)
    c = sns.color_palette("magma", n_colors=1)
    orb_indices = np.array([orb[0] for orb in soe1])
    soe_values = np.array([orb[1] for orb in soe1])
    soe_values = soe_values / max(soe_values)  # Normalize SOE values to [0, 1]
    soe1_thresholds = [x / 20 for x in range(20)][:0:-1]  # 3 thresholds excluding 0
    soe1_sizes = [x * 10 for x in np.linspace(5, 35, 20)][
        :0:-1
    ]  # Fixed marker size for all thresholds

    for xi, yi, soe in zip(x, y, soe_values, strict=False):
        for thresh, size in zip(soe1_thresholds, soe1_sizes, strict=False):
            if soe >= thresh:
                ax.scatter(
                    xi,
                    yi,
                    s=size,
                    color=c[0],
                    edgecolors="black",
                    zorder=3,
                    linewidths=2,
                )
                break
    for i in range(N):
        ax.text(
            1.15 * x[i],
            1.15 * y[i],
            str(orb_indices[i]),
            ha="center",
            va="center",
            fontsize=12,
            zorder=4,
        )
    return [
        plt.Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            label="SOE",
            markerfacecolor=c[0],
            markeredgecolor="black",
            markeredgewidth=2,
            markersize=10,
        )
    ]


def entanglement_diagrams(
    soe1: list[tuple[Any, Any]],
    minfo: np.ndarray,
    folder: str = "./",
    filename: str = "entanglement_diagrams",
) -> None:
    """Plot entanglement diagrams for active and virtual orbitals.

    Each orbital is represented by an individual marker (different colors for different irreps).
    The size of the marker is proportional to its single-orbital entropy (SOE).
    Solid to dotted is the gradient for the Mutual Information (MI) between orbitals.
    The plot is saved as a PDF file.

    Parameters:
    -----------
    soe1: list of tuples
        List of (orbital_index, single_orbital_entropy) tuples for all orbitals.
    minfo: np.ndarray
        Mutual Information matrix between orbitals.
    folder: str, default "./"
        Directory where the PDF file will be saved.
    filename: str, default "entanglement_diagrams"
        Base name of the output PDF file (without extension).
    """
    N = len(soe1)

    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)
    x = np.cos(angles)
    y = np.sin(angles)

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    ax.axis("off")

    circle1 = plt.Circle((0, 0), 1.25, color="black", fill=False, linewidth=1)
    circle2 = plt.Circle((0, 0), 1, color="black", fill=False, linewidth=2)
    ax.add_patch(circle1)
    ax.add_patch(circle2)

    # mi thresholds

    handles_mi = _helper_mi(minfo, x, y, ax)
    handler_soe = _helper_soe(soe1, x, y, ax)
    handles = handles_mi + handler_soe

    leg = plt.legend(handles=handles, loc="center")
    leg.get_frame().set_alpha(None)
    fig.tight_layout()
    fig.savefig(Path(folder) / (filename + ".pdf"), format="pdf", bbox_inches="tight")
    logger.info("Entanglement diagram saved as %s.pdf", filename)
    plt.close(fig)
