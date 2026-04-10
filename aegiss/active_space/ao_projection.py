"""Active space selection via AO projection onto a target set of atoms/AOs."""

from __future__ import annotations

import numpy as np
from pyscf import gto
from aegiss.utils.orbital_indices import combine_candidate_entropies
from typing import List, Union, Literal


def _compute_contribution( mf, 
                            label: Union[str, list[str]], 
                            active_mo_coeff: np.ndarray,
                            reduced_space_indices: list[int],
                            contribution: float, 
                            contribution_type: Literal["r", "a"],
                            secondary_basis: str | None = 'minao',
                          ) -> np.ndarray:
    
    mol2 = mf.mol  # original basis
    projection_mol = mol2.copy()  # extended basis
    if secondary_basis is not None:
        projection_mol.atom = mol2._atom
        projection_mol.unit = (
            "B"  # in Bohr because `mol._atom` returns the coordinates in Bohr
        )
        projection_mol.symmetry = False
        projection_mol.basis = secondary_basis
        projection_mol.build(False, False)
    else:
        print("No secondary basis provided. Using original molecule for projection.")
    baslst = projection_mol.search_ao_label(label)
    s21 = gto.intor_cross('int1e_ovlp', projection_mol, mol2)[baslst]
    if len(baslst) == 0:
        return np.zeros(active_mo_coeff)
    assert s21.shape[1] == active_mo_coeff.shape[0], "Mismatch in number of orbitals between s21 and active_mo_coeff"
    
    reduced_mos = active_mo_coeff[:, np.array([idx for idx, _ in reduced_space_indices])]
    
    s22 = np.dot(s21, reduced_mos[:,:])
    projected_weights = []
    print(f"\nComputing contributions for AO label(s): {label}")
    for idx in range(len(s22[0,:])):
        tmp = 0
        for j in s22[:, idx]:
            tmp += abs(j)
        orb = reduced_space_indices[idx][0]
        projected_weights.append((orb, tmp))
    
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
        print(f"\tOrbital {idx}: contribution = {tmp:.8f}")

    return  [(idx, tmp) for idx, tmp in sorted_weights if tmp >= contribution]


def select_by_label(label: str | list[str],
                    mf, 
                    active_mo_coeff,
                    space,
                    contribution, 
                    projected_basis: str | None,
                    contribution_type) -> List[int]:
    """Select orbitals based on their contribution to a specific AO label.
    
    Parameters
    ----------
    label : str or list of str
        AO label(s) to compute contributions for. Each label should be in the format "Element.Index AtomicOrbital", e.g. "O.1 1s" for the second oxygen
        atom. If a list of labels is provided, the contributions will be computed for each label separately and combined.
    s21 : np.ndarray
        Overlap matrix between the secondary basis (used for projection) and the original basis of the molecule.
    mol : pyscf.gto.Mole
        Molecule object containing the geometry and basis set information.
    active_mo_coeff : np.ndarray
        Coefficients of the active molecular orbitals in the original basis.
    space : list of int
        List of orbital indices to restrict the selection to. Only orbitals in this list will be considered for selection.
    contribution : float
        Threshold for selecting orbitals based on their contribution to the target AO label(s).
    contribution_type : {'r', 'a'}
        Type of contribution to use for selection. 'r' for relative contribution (normalized by the maximum contribution among the orbitals), 'a' for absolute contribution.        
    
    Returns
    -------
    selected_indices : list of int
        List of orbital indices selected based on their contribution to the target AO label(s).
    """
    
    return  _compute_contribution(mf,
                                 label, 
                                 active_mo_coeff,
                                 space,
                                 contribution, 
                                 contribution_type,
                                 projected_basis)
    
def select_by_ao_projection(
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
    
    if top_n is not None:
        print("\nSelecting top orbitals based on contribution. Ignoring contribution threshold.")
        contribution = 0.0  # set to 0 to include all orbitals in the top_n selection
    
    selected_indices = set()
    for label in ao_labels:
        selected_indices_for_label = select_by_label(label=label, 
                                                     projected_basis=projected_basis, 
                                                     mf=mf, 
                                                     active_mo_coeff=mf.mo_coeff, 
                                                     space=restricted_space, 
                                                     contribution=contribution, 
                                                     contribution_type=contribution_type)
        if top_n is not None:
            selected_indices_for_label = selected_indices_for_label[:top_n]  # select only the top N orbitals for this label
        selected_indices.update(idx for idx, _ in selected_indices_for_label)
        
    print("\nFinal selected orbital indices:", sorted(selected_indices))
    return np.array(sorted(selected_indices))
    

    
    
        
    