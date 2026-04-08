"""Active space selection via single-orbital entropy screening."""
from __future__ import annotations

import numpy as np
from typing import Literal


def select_by_entropy(occ_orbs: np.ndarray,
                      virt_orbs: np.ndarray,
                      soe1: np.ndarray,
                      fraction: float =  0.1,
                      fraction_type: Literal['r', 'a'] = 'r'
                      ) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Select active orbitals by single-orbital entanglement entropy.
    
    Parameters
    ----------
    occ_orbs : np.ndarray, shape (n_occ,)
        Indices of occupied orbitals.
    virt_orbs : np.ndarray, shape (n_virt,)
        Indices of virtual orbitals.
    soe1 : np.ndarray, shape (n_occ + n_virt,)
        Single-orbital entropies for all orbitals.
    fraction : float or None, optional
        If a float in (0, 1), select the top fraction of orbitals by entropy.
        If None, use an absolute threshold (see `fraction_type`).
    fraction_type : {'r', 'a'}, optional
        If 'r', `fraction` is interpreted as a fraction of the total number of orbitals.
        If 'a', `fraction` is interpreted as an absolute threshold on the entropy value.
    
    Returns
    -------
    active_indices : np.ndarray of int
        Indices of orbitals selected by entropy screening.
    
    """
    
    print("\n----------------Entropy Screening for AEGISS----------------\n")
    n_orbs = len(soe1)
    
    orbitals_index = []
    for i in occ_orbs:
        orbitals_index.append(i)
    for i in virt_orbs:
        orbitals_index.append(i)
    orbitals_index = np.array(orbitals_index)
    
    max_s1 = np.max(soe1)
    max_s1_index = np.argmax(soe1)
    
    ths_lists = soe1
    print("S(1) values for candidate orbitals:")
    for idx, s1 in zip(orbitals_index, ths_lists):
        print(f"\tOrbital {idx}: S(1) = {s1:.4f}")
        
    print(f"MAX S(1): {max_s1:.4f} (orbital {orbitals_index[max_s1_index]})\n")
    
    if fraction_type == 'r':
        print("Using relative threshold")
        ths_lists = ths_lists / max_s1
    else:
        print("Using absolute threshold")
    sortkey = np.argsort(ths_lists)[::-1]
    ths_lists = ths_lists[sortkey]
    orbitals_index = orbitals_index[sortkey]
    for idx, s1 in zip(orbitals_index, ths_lists):
        print(f"\tOrbital {idx}: S(1) = {s1:.4f}")
    
    stop_idx = None
    for k, ths in enumerate(ths_lists):
        if ths < fraction:
            stop_idx = k
            break
    if stop_idx is None:
        print(f'All {n_orbs} orbitals are selected by the entropy threshold {fraction}.')
    else:
        ths_lists = ths_lists[:stop_idx]
        orbitals_index = orbitals_index[:stop_idx]
        print(f"\nSelected {len(orbitals_index)} orbitals by entropy threshold {fraction} (fraction type: {fraction_type}).")
        # for idx, s1 in zip(orbitals_index, ths_lists):
        #     print(f"\tOrbital {idx}: S(1) = {s1:.4f}")
    
    occ_selected = []
    virt_selected = []
    for idx, n_orb in enumerate(orbitals_index):
        if n_orb in occ_orbs:
            occ_selected.append((n_orb, ths_lists[idx]))
        elif n_orb in virt_orbs:
            virt_selected.append((n_orb, ths_lists[idx]))
    
    for i in occ_selected:
        print(f"\tOccupied orbital {i[0]}: S(1) = {i[1]:.4f}")
    for i in virt_selected:
        print(f"\tVirtual orbital {i[0]}:  S(1) = {i[1]:.4f}")
    return occ_selected, virt_selected 
    
        
