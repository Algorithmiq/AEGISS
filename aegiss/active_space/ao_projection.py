"""Active space selection via AO projection onto a target set of atoms/AOs."""

from __future__ import annotations

import numpy as np
from pyscf import gto
from aegiss.utils.orbital_indices import combine_candidate_entropies
from typing import List, Union, Literal


def _get_secondary_basis_s21(
    mol, secondary_basis: str | None = 'minao'
) -> np.ndarray:

    projection_mol = mol.copy()  # extended basis
    if secondary_basis is not None:
        projection_mol.atom = mol._atom
        projection_mol.unit = (
            "B"  # in Bohr because `mol._atom` returns the coordinates in Bohr
        )
        projection_mol.symmetry = False
        projection_mol.basis = secondary_basis
        projection_mol.build(False, False)
    return gto.intor_cross('int1e_ovlp', projection_mol, mol)

def _get_s22_from_ao_labels(projected_mol, 
                            label: Union[str, list[str]], 
                            s21: np.ndarray, 
                            active_mo_coeff: np.ndarray) -> np.ndarray:
    baslst = projected_mol.search_ao_label(label)
    if len(baslst) == 0:
        return np.zeros(active_mo_coeff)
    
    return np.dot(s21[baslst], active_mo_coeff)

def _compute_contribution(s22: np.ndarray, 
                          label: Union[str, list[str]],
                          reduced_space_indices: list[int],
                          contribution: float, 
                          contribution_type: Literal["r", "a"]) -> np.ndarray:
    projected_weights = []
    print(f"\nComputing contributions for AO label(s): {label}")
    for idx, _ in reduced_space_indices:
        tmp = 0
        for j in s22[:, idx]:
            tmp += abs(j)
        projected_weights.append((idx, tmp))
    
    if contribution_type == "r":
        max_contribution = max(weight for _, weight in projected_weights)
        projected_weights = [(idx, tmp / max_contribution) for idx, tmp in projected_weights]
    sorted_weights = sorted(projected_weights, key=lambda x: x[1], reverse=True)
    
    first_below = False
    print('\t' + '-'*10 + 'above' + '-'*10)
    for idx, tmp in sorted_weights:
        if tmp < contribution and not first_below:
            print('\t' + '-'*10 + 'below' + '-'*10)
            first_below = True
        print(f"\tOrbital {idx}: contribution = {tmp:.4f}")

    return  [(idx, tmp) for idx, tmp in sorted_weights if tmp >= contribution]

def select_by_ao_projection(
    mol,
    mf,
    ao_labels: Union[List[str], List[List[str]]],
    projected_basis: str | None = 'minao',
    contribution: Union[float, List[float]] | None = None,
    contribution_type: Literal["r", "a"] | None = None,
    top_n: Union[int, List[int]] | None = None,
    restricted_space: list[tuple[int, float]]
    | tuple[list[tuple[int, float]], list[tuple[int, float]]]
    | None = None,
) -> np.ndarray:
    """Select active orbitals by their projection onto a set of target AOs.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object containing the geometry and basis set information.
    mf : pyscf.scf.hf.SCF or pyscf.dft.rks.KohnShamDFT
        Mean-field object containing the molecular orbitals and occupation numbers.
    restricted_space : list of int, optional
        List of orbital indices to restrict the selection to. If None, all orbitals will be considered.
    ao_labels : list of str or list of list of str
        List of AO labels to project onto. Each label should be in the format "Element.Index AtomicOrbital", e.g. "O.1 1s" for the second oxygen atom.
        If a list of lists is provided, each sublist will be treated as a separate group of AOs, and the contributions will be computed separately for each group.
    projected_basis : str or None, default 'minao'
        Basis set to use for the projection. If None, the original basis of the molecule will be used. 
        Using a minimal basis (e.g. 'minao') can help to focus the selection on the most chemically relevant orbitals and reduce noise from diffuse functions.
    contribution : float or list of float, optional
        Threshold for selecting orbitals based on their contribution to the target AOs.
        If a single float is provided, it will be applied to all groups of AOs.
        If a list is provided, it should have the same length as *ao_labels* and specify the threshold for each group separately.
    contribution_type : {'r', 'a'}, optional
        Type of contribution to use for selection. 'r' for relative contribution (normalized by the total AO contribution), 'a' for absolute contribution. Required if *contribution* is specified.
    top_n : int or list of int, optional
        Alternative to *contribution* thresholding. If specified, selects the top N orbitals with the highest contribution to the target AOs.
        If a list is provided, it should have the same length as *ao_labels* and specify the number of orbitals to select for each group separately.

    Returns
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
            "Restricting selection to orbitals with indices: ",
            [int(idx[0]) for idx in restricted_space],
        )

    if projected_basis is not None:
        print("Using projected basis for AO projection:", projected_basis)
    else:
        print("Using molecule for AO projection")

    s21 = _get_secondary_basis_s21(mol, projected_basis)
    
    if top_n is not None:
        print("\nSelecting top orbitals based on contribution. Ignoring contribution threshold.")
        contribution = 0.0  # set to 0 to include all orbitals in the top_n selection
    for label in ao_labels:
        s22 = _get_s22_from_ao_labels(mol, label, s21, active_mo_coeff=mf.mo_coeff)
        above_threshold = _compute_contribution(s22, label, restricted_space, contribution, contribution_type)
        