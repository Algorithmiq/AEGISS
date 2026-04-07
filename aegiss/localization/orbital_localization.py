"""Orbital localization procedures with a unified dispatcher."""
from __future__ import annotations

from typing import Any, Literal, Optional
import numpy as np
from pyscf import lo, gto
from aegiss.wavefunction.methods import build_correlated_wfn
from aegiss.utils.orbital_indices import get_orbital_spaces


# ---------------------------------------------------------------------------
# Per-method functions
# ---------------------------------------------------------------------------

def localize_boys(
    mol: gto.Mole,
    mf,
    n_occ: int | None = None,
    on_virtual: bool = False,
) -> np.ndarray:
    """Boys (Foster-Boys) localization.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    mf : SCF object
    n_occ : int, optional
        Number of occupied orbitals. Inferred from *mf* if not given.
    on_virtual : bool
        Whether to also localize the virtual space.

    Returns
    -------
    new_mos : np.ndarray, shape (nao, nmo)
        Localized MO coefficients.
    """
    if n_occ is None:
        n_occ = int(np.count_nonzero(mf.mo_occ > 0))
    loc_occ = lo.Boys(mol, mf.mo_coeff[:, :n_occ]).kernel()
    loc_virt = lo.Boys(mol, mf.mo_coeff[:, n_occ:]).kernel() if on_virtual else mf.mo_coeff[:, n_occ:]
    return np.hstack((loc_occ, loc_virt))


def localize_edmiston(
    mol: gto.Mole,
    mf,
    n_occ: int | None = None,
    on_virtual: bool = False,
) -> np.ndarray:
    """Edmiston-Ruedenberg localization.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    mf : SCF object
    n_occ : int, optional
        Number of occupied orbitals. Inferred from *mf* if not given.
    on_virtual : bool
        Whether to also localize the virtual space.

    Returns
    -------
    new_mos : np.ndarray, shape (nao, nmo)
        Localized MO coefficients.
    """
    if n_occ is None:
        n_occ = int(np.count_nonzero(mf.mo_occ > 0))
    loc_occ = lo.ER(mol, mf.mo_coeff[:, :n_occ]).kernel()
    loc_virt = lo.ER(mol, mf.mo_coeff[:, n_occ:]).kernel() if on_virtual else mf.mo_coeff[:, n_occ:]
    return np.hstack((loc_occ, loc_virt))


def localize_pipek_mezey(
    mol: gto.Mole,
    mf,
    n_occ: int | None = None,
    on_virtual: bool = False,
    pop_method: str = "becke",
) -> np.ndarray:
    """Pipek-Mezey localization.

    Parameters
    ----------
    mol : pyscf.gto.Mole
    mf : SCF object
    n_occ : int, optional
        Number of occupied orbitals. Inferred from *mf* if not given.
    on_virtual : bool
        Whether to also localize the virtual space.
    pop_method : str
        Population analysis scheme used for the virtual block (default ``"becke"``).

    Returns
    -------
    new_mos : np.ndarray, shape (nao, nmo)
        Localized MO coefficients.
    """
    if n_occ is None:
        n_occ = int(np.count_nonzero(mf.mo_occ > 0))
    loc_occ = lo.PM(mol, mf.mo_coeff[:, :n_occ]).kernel()
    loc_virt = lo.PM(mol, mf.mo_coeff[:, n_occ:], pop_method=pop_method).kernel() if on_virtual else mf.mo_coeff[:, n_occ:]
    return np.hstack((loc_occ, loc_virt))


def localize_fnos(
    mf,
    correlated_method: Literal['mp2', 'cisd', 'ccsd'],
    active: bool | list[int] = True
) -> np.ndarray:
    """Frozen Natural orbital (FNO) localization.

    Parameters
    ----------
    mf: SCF object
    correlated_method : {'mp2', 'cisd', 'ccsd'}
        Correlated method used to compute the NOs. See *build_correlated_wfn* for details.
    active : bool | list[int]
        If ``True``, all the occupied orbitals will be used for the NO costruction. 
        If a list of integers is provided, these will be the indices of the orbitals to be frozen (0-based indexing).
    Returns
    -------
    occ : np.ndarray
        Natural occupation numbers (active NOs only when ``frozen=True``).
    nos : np.ndarray, shape (nao, n_active)
        Natural orbital coefficients in the AO basis.
    """
    from aegiss.localization.fnos import compute_fnos

    
    print(f"Computing FNOs using {correlated_method.upper()}")
    
    frozen_orbs, active_orbs, _,frozen_virt = get_orbital_spaces(mf, orbital_list=active if isinstance(active, list) else [], unique_virt_space=True)
    wfn_obj = build_correlated_wfn(mf, correlated_method, frozen=frozen_orbs)
    print("Frozen orbitals: ", frozen_orbs)
    print("Active orbitals: ", active_orbs)
    print("Frozen virtual orbitals: ", frozen_virt)
          
    
    return compute_fnos(wfn_obj, frozen=frozen_orbs, active=active_orbs)


def localize_nos(
    mf,
    correlated_method: Literal['mp2', 'cisd', 'ccsd'],
    *,
    split: bool = False,
) -> np.ndarray:
    """Natural orbital (NO) localization.

    Parameters
    ----------
    mf: SCF object
    correlated_method : {'mp2', 'cisd', 'ccsd'}
        Correlated method used to compute the NOs. See *build_correlated_wfn* for details.
    split : bool
        Obtain the Natural Orbital basis by diagonalizing the 1-RDM in the occupied and virtual subspaces separately. 
        This can be useful for certain post-HF methods where the occupied and virtual spaces are treated differently.

    Returns
    -------
    occ : np.ndarray
        Natural occupation numbers (active NOs only when ``frozen=True``).
    nos : np.ndarray, shape (nao, n_active)
        Natural orbital coefficients in the AO basis.
    """
    from aegiss.localization.nos import compute_nos

    print(f"Computing NOs using {correlated_method.upper()}")
    wfn_obj = build_correlated_wfn(mf, correlated_method)
    print("Using standard NOs without freezing. ", "With Splitted NOs." if split else "Without Split NOs.")
    return compute_nos(wfn_obj, split_nos=split)
    # if not frozen:
    #     return occ, nos, None


_METHODS: dict[str, Any] = {
    "boys":     localize_boys,
    "edmiston": localize_edmiston,
    "er":       localize_edmiston,
    "pm":       localize_pipek_mezey,
    "pipek":    localize_pipek_mezey,
    "nos":      localize_nos,
    "fnos":     localize_fnos,         
}


def localize(method: str, *args, **kwargs):
    """Dispatch to a localization method by name.

    Parameters
    ----------
    method : str
        One of ``'boys'``, ``'edmiston'`` / ``'er'``, ``'pm'`` / ``'pipek'``,
        ``'nos'``, ``'fnos'``.
    *args, **kwargs
        Forwarded to the selected method function.

    Examples
    --------
    >>> localize('pm', mol, mf, on_virtual=True)
    >>> localize('nos', wfn, frozen=True)
    >>> localize('boys', mol, mf, n_occ=5, on_virtual=False)
    """
    if method not in _METHODS:
        raise ValueError(
            f"Unknown localization method {method!r}. "
            f"Choose from: {sorted(_METHODS)}"
        )
    return _METHODS[method](*args, **kwargs)
