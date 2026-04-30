"""Active space selection via AO projection onto a target set of atoms/AOs."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pyscf import gto

from aegiss.utils.orbital_indices import combine_candidate_entropies
from aegiss.utils.plotting import plot_ao_contribution


def _compute_contribution(
    mf: Any,
    label: str | list[str],
    full_mo_coeff: np.ndarray,
    reduced_space_indices: list[tuple[int, float]],
    contribution: float,
    contribution_type: Literal["r", "a"],
    secondary_basis: str | None = "minao",
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Compute the AO contribution of each orbital to a given label.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        Mean-field object containing the molecule and MO coefficients.
    label : str or list of str
        AO label(s) to compute contributions for.
    full_mo_coeff : np.ndarray
        Coefficients of the active MOs.
    reduced_space_indices : list of tuple[int, float]
        Orbital index/entropy pairs in the reduced space.
    contribution : float
        Threshold for selecting orbitals.
    contribution_type : {'r', 'a'}
        Relative or absolute contribution type.
    secondary_basis : str or None
        Basis for projection; ``None`` uses the original molecule basis.

    Returns:
    -------
    selected : list of tuple[int, float]
        Orbital index/weight pairs above the contribution threshold.
    all_contributions : list of tuple[int, float]
        Orbital index/weight pairs for all orbitals, ordered by index, for potential further analysis or plotting.
    """
    mol2 = mf.mol  # original basis
    projection_mol = mol2.copy()  # extended basis
    if secondary_basis is not None:
        projection_mol.atom = mol2._atom
        projection_mol.unit = (
            "B"  # in Bohr because `mol._atom` returns the coordinates in Bohr
        )
        projection_mol.symmetry = False
        projection_mol.basis = secondary_basis
        projection_mol.build(False, False)  # noqa: FBT003
    else:
        print("No secondary basis provided. Using original molecule for projection.")
    baslst = projection_mol.search_ao_label(label)
    s21 = gto.intor_cross("int1e_ovlp", projection_mol, mol2)[baslst]
    if len(baslst) == 0:
        return [], []
    if s21.shape[1] != full_mo_coeff.shape[0]:
        msg = "Mismatch in number of orbitals between s21 and active_mo_coeff"
        raise ValueError(msg)

    reduced_mos = full_mo_coeff[:, np.array([idx for idx, _ in reduced_space_indices])]

    s22 = np.dot(s21, reduced_mos[:, :])
    raw_weights: list[tuple[int, float]] = []
    print(f"\nComputing contributions for AO label(s): {label}")
    for idx in range(len(s22[0, :])):
        weight = np.sum(np.abs(s22[:, idx]))
        orb = reduced_space_indices[idx][0]
        raw_weights.append((orb, weight))

    projected_weights: list[tuple[int, float]]
    if contribution_type == "r":
        max_contribution = max(weight for _, weight in raw_weights)
        if max_contribution == 0.0:
            projected_weights = [(i, 0.0) for i, _ in raw_weights]
        else:
            projected_weights = [(i, w / max_contribution) for i, w in raw_weights]
    else:
        projected_weights = raw_weights
    sorted_weights = sorted(projected_weights, key=lambda x: x[1], reverse=True)

    first_below = False
    print("\t" + "-" * 10 + "above" + "-" * 10)
    for idx, tmp in sorted_weights:
        if tmp < contribution and not first_below:
            print("\t" + "-" * 10 + "below" + "-" * 10)
            first_below = True
        print(f"\tOrbital {idx}: contribution = {tmp:.8f}")

    return [
        (idx, tmp) for idx, tmp in sorted_weights if tmp >= contribution
    ], projected_weights


def select_by_label(
    label: str | list[str],
    mf: Any,
    full_mo_coeff: np.ndarray,
    space: list[tuple[int, float]],
    contribution: float,
    projected_basis: str | None,
    contribution_type: Literal["r", "a"],
) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Select orbitals based on their contribution to a specific AO label.

    Parameters
    ----------
    label : str or list of str
        AO label(s) to compute contributions for. Each label should be in the
        format ``"Element.Index AtomicOrbital"``, e.g. ``"O.1 1s"`` for the
        second oxygen atom. If a list is provided, contributions are computed
        for each label separately and combined.
    mf : pyscf.scf.hf.SCF
        Mean-field object containing the molecule and MO coefficients.
    full_mo_coeff : np.ndarray
        Coefficients of the full MOs.
    space : list of tuple[int, float]
        Orbital index/entropy pairs to restrict the selection to.
    contribution : float
        Threshold for selecting orbitals based on their contribution.
    projected_basis : str or None
        Basis set used for projection; ``None`` uses the original basis.
    contribution_type : {'r', 'a'}
        Relative (``'r'``) or absolute (``'a'``) contribution type.

    Returns:
    -------
    selected_indices : list of tuple[int, float]
        Orbital index/weight pairs selected based on their contribution.
    projected_weights : list of tuple[int, float]
        Orbital index/weight pairs for all orbitals, ordered by index, for potential further analysis or plotting.
    """
    return _compute_contribution(
        mf,
        label,
        full_mo_coeff,
        space,
        contribution,
        contribution_type,
        projected_basis,
    )


def _resolve_space(
    mf: Any,
    restricted_space: list[tuple[int, float]]
    | tuple[list[tuple[int, float]], list[tuple[int, float]]]
    | None,
) -> list[tuple[int, float]]:
    if restricted_space is None:
        print("No restriction on orbital indices for selection. Using all orbitals.")
        return [(idx, 0.0) for idx in range(mf.mo_coeff.shape[1])]
    if isinstance(restricted_space, tuple):
        space = combine_candidate_entropies(*restricted_space)
        print(
            "Combining candidate orbitals from different groups: ",
            [int(idx[0]) for idx in space],
        )
        return space
    return restricted_space


def _resolve_contribution(
    contribution: float | list[float],
    n_labels: int,
    top_n: int | list[int] | None,
) -> list[float]:
    """Helper function to resolve contribution thresholds for AO projection selection.

    Parameters:
    ----------
    contribution : float or list of float
        Contribution threshold(s) for selecting orbitals.
        A single float applies to all labels; a list specifies per-label thresholds.
    n_labels : int
        Number of AO labels being considered for selection.
    top_n : int or list of int or None
        If specified, selects the top N orbitals per label instead of using contribution thresholds.
        A single int applies to all labels; a list specifies per-label counts.
        If None, uses contribution thresholds for selection.

    Returns:
    -------
    resolved_contribution : list of float
        List of contribution thresholds for each AO label, resolved based on the input parameters.
    """
    if top_n is not None:
        print(
            "\nSelecting top orbitals based on contribution. Ignoring contribution threshold."
        )
        if isinstance(contribution, list):
            print(
                "Warning: contribution is a list but top_n is specified."
                " Ignoring contribution thresholds and using top_n for selection."
            )
            return [0.0] * len(contribution)
        return [0.0] * n_labels

    if isinstance(contribution, list):
        print(
            "\nSelecting orbitals based on contribution thresholds per AO label: ",
            contribution,
        )
        if len(contribution) != n_labels:
            print(
                "Warning: Length of contribution list does not match number of AO labels. "
                "Using the first contribution value for all labels."
            )
            return [contribution[0]] * n_labels
        return contribution
    print(
        "\nSelecting orbitals based on a single contribution threshold for all AO labels: ",
        contribution,
    )
    return [contribution] * n_labels


def select_by_ao_projection(
    mf: Any,
    ao_labels: list[str | list[str]],
    projected_basis: str | None = "minao",
    contribution_type: Literal["r", "a"] = "a",
    contribution: float | list[float] = 0.0,
    top_n: int | list[int] | None = None,
    restricted_space: list[tuple[int, float]]
    | tuple[list[tuple[int, float]], list[tuple[int, float]]]
    | None = None,
    save_path: str | None = None,
) -> np.ndarray:
    """Select active orbitals by their projection onto a set of target AOs.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF or pyscf.dft.rks.KohnShamDFT
        Mean-field object containing the molecular orbitals and occupation numbers.
    ao_labels : list[str | list[str]]
        List of AO label entries. Each entry is either a single label string
        (e.g. ``"Fe 3d"``) or a list of strings treated as one group
        (e.g. ``["Fe 3d", "Fe 4d"]``). Entries can be freely mixed, e.g.
        ``["C 2pz", "Fe 3d", ["Fe 3d", "Fe 4d"]]``.
    projected_basis : str or None, default 'minao'
        Basis set to use for the projection. ``None`` uses the original basis.
        A minimal basis (e.g. ``'minao'``) helps focus the selection on
        chemically relevant orbitals and reduces noise from diffuse functions.
    contribution_type : {'r', 'a'}
        Relative (``'r'``) or absolute (``'a'``) contribution. Required when
        *contribution* is specified.
    contribution : float or list of float, default 0.0
        Threshold for selecting orbitals based on their contribution.
        A single float applies to all AO groups; a list specifies per-group
        thresholds and must match the length of *ao_labels*.
    top_n : int or list of int, optional
        Alternative to *contribution* thresholding. Selects the top N orbitals
        per AO group. A list specifies per-group counts.
    restricted_space : list of tuple or tuple of lists, optional
        Orbital indices (with entropy scores) to restrict selection to.
        ``None`` considers all orbitals.
    save_path : str or None, optional
        If provided, saves a heatmap of AO contributions to the specified path.

    Returns:
    -------
    active_indices : np.ndarray of int
        Array of orbital indices selected for the active space.
    """
    print("\n----------------AO Projection Screening for AEGISS----------------\n")
    print("AO labels for projection:", ao_labels)
    if contribution is not None:
        print("Contribution threshold:", contribution)
        print("Contribution type:", contribution_type)
    if top_n is not None:
        print("Top N orbitals to select:", top_n)

    space = _resolve_space(mf, restricted_space)
    resolved_contribution = _resolve_contribution(contribution, len(ao_labels), top_n)

    if projected_basis is not None:
        print("\nUsing projected basis for AO projection:", projected_basis)
    else:
        print("\nUsing molecule for AO projection")

    selected_indices: set[int] = set()
    projected_total_weights: dict[int, list[tuple[int, float]]] = {}
    labels_idx: dict[int, str | list[str]] = dict(enumerate(ao_labels))
    for loop_idx, label in enumerate(ao_labels):
        selected_indices_for_label, projected_weights = select_by_label(
            label=label,
            projected_basis=projected_basis,
            mf=mf,
            full_mo_coeff=mf.mo_coeff,
            space=space,
            contribution=resolved_contribution[loop_idx],
            contribution_type=contribution_type,
        )
        if top_n is not None and isinstance(top_n, int):
            selected_indices_for_label = selected_indices_for_label[:top_n]
        selected_indices.update(orb_idx for orb_idx, _ in selected_indices_for_label)
        if save_path is not None:
            projected_total_weights[loop_idx] = projected_weights

    if save_path is not None and projected_total_weights:
        plot_ao_contribution(
            ao_contributions=projected_total_weights,
            save_path=save_path,
            ao_labels=labels_idx,
            mo_labels=[orb_idx for orb_idx, _ in space],
        )

    print("\nFinal selected orbital indices:")
    for idx in sorted(selected_indices):
        print(f"\tOrbital {idx}")
    return np.asarray(sorted(selected_indices))
