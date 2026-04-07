"""Active space selection via AO projection onto a target set of atoms/AOs."""
from __future__ import annotations

import numpy as np

from typing import List, Union, Literal


def _get_secondary_basis_s22(mol, 
                             secondary_basis: str | None,
                             slice_labels: np.ndarray) -> np.ndarray:
    
    projection_mol = mol.copy() # extended basis
    if secondary_basis is not None:
    
        print("Using secondary basis for AO projection:", secondary_basis)
        projection_mol.atom = mol._atom
        projection_mol.unit = 'B'
        projection_mol.symmetry = False
        projection_mol.basis = secondary_basis
        projection_mol.build(False, False)
    else:
        print("Using original basis for AO projection")
        
def select_by_ao_projection(
    mol,
    mf,
    ao_labels: Union[List[str], List[List[str]]],
    projected_basis: str | None = None,
    contribution: Union[float, List[float]] | None = None,
    contribution_type: Literal['r', 'a'] | None = None,
    top_n: Union[int, List[int]] | None = None,
    restricted_space: list[int] | None = None
    
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
    projected_basis : str, optional
        If specified, use this basis set for the AO projection instead of the original basis of the
    contribution : float or list of float, optional
        Threshold for selecting orbitals based on their contribution to the target AOs.
        If a single float is provided, it will be applied to all groups of AOs. 
        If a list is provided, it should have the same length as *ao_labels* and specify the threshold for each group separately.
    contribution_type : {'r', 'a'}, optional
        Type of contribution to use for selection. 'r' for relative contribution (normalized by the total AO contribution), 'a' for absolute contribution. Required if *contribution* is specified.
    top_n : int or list of int, optional
        Alternative to *contribution* thresholding. If specified, selects the top N orbitals with the highest contribution to the target AOs. 
        If a list is provided, it should have the same length as *ao_labels* and specify the number of orbitals to select for each group separately.
    
    """
    print("\n----------------AO Projection Screening for AEGISS----------------\n")
    print("AO labels for projection:", ao_labels)
    if contribution is not None:
        print("Contribution threshold:", contribution)
        print("Contribution type:", contribution_type)
    if top_n is not None:
        print("Top N orbitals to select:", top_n)
    if restricted_space is not None:
        print("Restricting selection to orbitals with indices: ", [int(idx[0]) for idx in restricted_space])
        
    if projected_basis is not None:
        print("Using projected basis for AO projection:", projected_basis)
    else:
        print("Using original basis for AO projection")
