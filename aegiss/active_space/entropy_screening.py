"""Active space selection via single-orbital entropy screening."""
from __future__ import annotations

import numpy as np
from typing import Literal


def _global_entropy_ranking(soe1,
                            fraction_type, 
                            fraction,
                            n_orbs,
                            occ_orbs, 
                            virt_orbs) ->  tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    
    max_s1 = np.max(soe1)
    max_s1_index = np.argmax(soe1)
    orbitals_index = np.concatenate((occ_orbs, virt_orbs))
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
    
    return occ_selected, virt_selected 

def _relative_entropy_ranking(soe1,
                            fraction_type, 
                            fraction,
                            subspace) ->  tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    
    
    max_s1 = np.max(soe1)
    max_s1_index = np.argmax(soe1)
    
    n_orbs = len(soe1)
    
    ths_lists = soe1
    print("S(1) values for candidate orbitals:")
    for idx, s1 in zip(subspace, ths_lists):
        print(f"\tOrbital {idx}: S(1) = {s1:.4f}")
        
    print(f"MAX S(1): {max_s1:.4f} (orbital {subspace[max_s1_index]})\n")
    
    if fraction_type == 'r':
        print("Using relative threshold")
        ths_lists = ths_lists / max_s1
    else:
        print("Using absolute threshold")
    sortkey = np.argsort(ths_lists)[::-1]
    ths_lists = ths_lists[sortkey]
    orbitals_index = subspace[sortkey]
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
    
    selected = []
    for idx, n_orb in enumerate(orbitals_index):
        selected.append((n_orb, ths_lists[idx]))
    
    return selected

def select_by_entropy(occ_orbs: np.ndarray,
                      virt_orbs: np.ndarray,
                      soe1: np.ndarray,
                      fraction: float =  0.1,
                      fraction_type: Literal['r', 'a'] = 'r', 
                      ranking: Literal['global', 'relative'] = 'global',
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
    ranking : {'global', 'relative'}, optional
        If 'global', rank orbitals by their absolute entropy values.
        If 'relative', rank orbitals by their entropy values relative to the maximum entropy in the subspace.
    
    Returns
    -------
    active_indices : np.ndarray of int
        Indices of orbitals selected by entropy screening.
    
    """
    
    print("\n----------------Entropy Screening for AEGISS----------------\n")
    n_orbs = len(soe1)
    
    if ranking == 'global':
        print("Using global ranking of orbitals by entropy values.")
        selected_occ, selected_virt = _global_entropy_ranking(soe1, fraction_type, fraction, n_orbs, occ_orbs, virt_orbs)
    else:
        print("Using relative ranking of orbitals by entropy values within the subspace.")
        soe1_occ = []
        for idx in range(len(occ_orbs)):
            soe1_occ.append(soe1[idx])
        soe1_occ = np.array(soe1_occ)
        
        soe1_virt = []
        off_set = len(occ_orbs)
        for idx in range(len(virt_orbs)):
            soe1_virt.append(soe1[off_set + idx])
        soe1_virt = np.array(soe1_virt)
        
        selected_occ = _relative_entropy_ranking(soe1_occ, fraction_type, fraction, occ_orbs)
        selected_virt = _relative_entropy_ranking(soe1_virt, fraction_type, fraction, virt_orbs)
    
    for i in selected_occ:
        print(f"\tSelected occupied orbital {i[0]}: S(1) = {i[1]:.4f}")
    for i in selected_virt:
        print(f"\tSelected virtual orbital {i[0]}:  S(1) = {i[1]:.4f}")
    
    return selected_occ, selected_virt
