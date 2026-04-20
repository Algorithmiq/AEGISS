"""Active space selection via AO projection onto a target set of atoms/AOs."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np
from pyscf import gto

from aegiss.utils.orbital_indices import combine_candidate_entropies


def _compute_contribution(
    mf: Any,
    label: str | list[str],
    active_mo_coeff: np.ndarray,
    reduced_space_indices: list[tuple[int, float]],
    contribution: float,
    contribution_type: Literal["r", "a"],
    secondary_basis: str | None = "minao",
) -> list[tuple[int, float]]:
    """Compute the AO contribution of each orbital to a given label.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        Mean-field object containing the molecule and MO coefficients.
    label : str or list of str
        AO label(s) to compute contributions for.
    active_mo_coeff : np.ndarray
        Coefficients of the active MOs in the AO basis.
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
        return []
    if s21.shape[1] != active_mo_coeff.shape[0]:
        msg = "Mismatch in number of orbitals between s21 and active_mo_coeff"
        raise ValueError(msg)

    reduced_mos = active_mo_coeff[
        :, np.array([idx for idx, _ in reduced_space_indices])
    ]

    s22 = np.dot(s21, reduced_mos[:, :])
    raw_weights: list[tuple[int, float]] = []
    print(f"\nComputing contributions for AO label(s): {label}")
    for idx in range(len(s22[0, :])):
        tmp: float = 0.0
        for j in s22[:, idx]:
            tmp += abs(j)
        orb = reduced_space_indices[idx][0]
        raw_weights.append((orb, tmp))

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

    return [(idx, tmp) for idx, tmp in sorted_weights if tmp >= contribution]


def select_by_label(
    label: str | list[str],
    mf: Any,
    active_mo_coeff: np.ndarray,
    space: list[tuple[int, float]],
    contribution: float,
    projected_basis: str | None,
    contribution_type: Literal["r", "a"],
) -> list[tuple[int, float]]:
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
    active_mo_coeff : np.ndarray
        Coefficients of the active MOs in the AO basis.
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
    """
    return _compute_contribution(
        mf,
        label,
        active_mo_coeff,
        space,
        contribution,
        contribution_type,
        projected_basis,
    )


def select_by_ao_projection(
    mf: Any,
    ao_labels: list[str] | list[list[str]],
    projected_basis: str | None = "minao",
    contribution: float | list[float] | None = None,
    contribution_type: Literal["r", "a"] | None = None,
    top_n: int | list[int] | None = None,
    restricted_space: list[tuple[int, float]]
    | tuple[list[tuple[int, float]], list[tuple[int, float]]]
    | None = None,
) -> np.ndarray:
    """Select active orbitals by their projection onto a set of target AOs.

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF or pyscf.dft.rks.KohnShamDFT
        Mean-field object containing the molecular orbitals and occupation numbers.
    ao_labels : list of str or list of list of str
        List of AO labels to project onto. Each label should be in the format
        ``"Element.Index AtomicOrbital"``, e.g. ``"O.1 1s"`` for the second
        oxygen atom. If a list of lists is provided, each sublist is treated as
        a separate group and contributions are computed per group.
    projected_basis : str or None, default 'minao'
        Basis set to use for the projection. ``None`` uses the original basis.
        A minimal basis (e.g. ``'minao'``) helps focus the selection on
        chemically relevant orbitals and reduces noise from diffuse functions.
    contribution : float or list of float, optional
        Threshold for selecting orbitals based on their contribution.
        A single float applies to all AO groups; a list specifies per-group
        thresholds and must match the length of *ao_labels*.
    contribution_type : {'r', 'a'}, optional
        Relative (``'r'``) or absolute (``'a'``) contribution. Required when
        *contribution* is specified.
    top_n : int or list of int, optional
        Alternative to *contribution* thresholding. Selects the top N orbitals
        per AO group. A list specifies per-group counts.
    restricted_space : list of tuple or tuple of lists, optional
        Orbital indices (with entropy scores) to restrict selection to.
        ``None`` considers all orbitals.

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

    if restricted_space is not None:
        if isinstance(restricted_space, tuple):
            restricted_space = combine_candidate_entropies(*restricted_space)
            print(
                "Combining candidate orbitals from different groups: ",
                [int(idx[0]) for idx in restricted_space],
            )
    else:
        print("No restriction on orbital indices for selection. Using all orbitals.")
        restricted_space = [(idx, 0.0) for idx in range(mf.mo_coeff.shape[1])]

    if projected_basis is not None:
        print("\nUsing projected basis for AO projection:", projected_basis)
    else:
        print("\nUsing molecule for AO projection")

    # Resolve contribution: default to 0.0 so top_n slice handles the cut-off
    resolved_contribution: float
    if top_n is not None:
        print(
            "\nSelecting top orbitals based on contribution."
            " Ignoring contribution threshold."
        )
        resolved_contribution = 0.0
    else:
        resolved_contribution = contribution if isinstance(contribution, float) else 0.0

    resolved_type: Literal["r", "a"] = (
        contribution_type if contribution_type is not None else "a"
    )

    selected_indices: set[int] = set()
    for label in ao_labels:
        selected_indices_for_label = select_by_label(
            label=label,
            projected_basis=projected_basis,
            mf=mf,
            active_mo_coeff=mf.mo_coeff,
            space=restricted_space,
            contribution=resolved_contribution,
            contribution_type=resolved_type,
        )
        if top_n is not None and isinstance(top_n, int):
            selected_indices_for_label = selected_indices_for_label[:top_n]
        selected_indices.update(idx for idx, _ in selected_indices_for_label)

    print("\nFinal selected orbital indices:")
    for idx in sorted(selected_indices):
        print(f"\tOrbital {idx}")
    return np.asarray(sorted(selected_indices))
