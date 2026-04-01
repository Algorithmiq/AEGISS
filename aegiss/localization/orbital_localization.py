"""Orbital localization procedures with a unified dispatcher."""
from __future__ import annotations

from typing import Any, Literal, Optional
import numpy as np
from pyscf import lo, gto
from aegiss.wavefunction.methods import build_correlated_wfn


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


def localize_nos(
    mf,
    correlated_method: Literal['mp2', 'cisd', 'ccsd', 'casci', 'casscf'],
    frozen: bool = False,
    occ_threshold: float = 1e-4,
    cas_size: Optional[tuple[int, int]] = (6, 6),
):
    """Natural orbital (NO) localization.

    Parameters
    ----------
    mf: SCF object
    correlated_method : {'mp2', 'cisd', 'ccsd', 'casci', 'casscf'}
        Correlated method used to compute the NOs. See *build_correlated_wfn* for details.
    frozen : bool
        If ``True``, return frozen natural orbitals (FNOs): weakly occupied
        NOs below *occ_threshold* are removed.
    occ_threshold : float
        Occupation cutoff used when ``frozen=True``.

    Returns
    -------
    occ : np.ndarray
        Natural occupation numbers (active NOs only when ``frozen=True``).
    nos : np.ndarray, shape (nao, n_active)
        Natural orbital coefficients in the AO basis.
    frozen_mask : np.ndarray of bool or None
        Boolean mask of frozen NOs when ``frozen=True``, else ``None``.
    """
    from aegiss.localization.nos import compute_nos

    print(mf, correlated_method)
    wfn_obj = build_correlated_wfn(mf,correlated_method)
    occ, nos = compute_nos(wfn_obj)
    # if not frozen:
    #     return occ, nos, None



# ---------------------------------------------------------------------------
# Registry and dispatcher
# ---------------------------------------------------------------------------

_METHODS: dict[str, Any] = {
    "boys":     localize_boys,
    "edmiston": localize_edmiston,
    "er":       localize_edmiston,
    "pm":       localize_pipek_mezey,
    "pipek":    localize_pipek_mezey,
    "nos":      localize_nos,
    "fnos":     localize_nos,          # alias: fnos always uses frozen=True
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
    if method == "fnos":
        kwargs.setdefault("frozen", True)
    return _METHODS[method](*args, **kwargs)
