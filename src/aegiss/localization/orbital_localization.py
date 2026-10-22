"""Orbital localization procedures with a unified dispatcher."""

from __future__ import annotations

import logging
from typing import Any, Literal

import numpy as np
from pyscf import gto, lo

from aegiss.utils.orb_display import from_display_all, to_display_all
from aegiss.utils.orbital_indices import get_orbital_spaces
from aegiss.wavefunction.methods import build_correlated_wfn

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Per-method functions
# ---------------------------------------------------------------------------


def localize_boys(
    mol: gto.Mole,
    mf: Any,
    n_occ: int | None = None,
    *,
    on_virtual: bool = False,
) -> np.ndarray:
    """Boys (Foster-Boys) localization — minimises the spatial spread of each orbital.

    Args:
        mol (pyscf.gto.Mole):
            Molecular geometry and basis set.
        mf (pyscf.scf.hf.SCF):
            Converged mean-field object supplying ``mo_coeff`` and ``mo_occ``.
        n_occ (int, optional):
            Number of occupied orbitals. Inferred from *mf* if not given.
        on_virtual (bool):
            Whether to also localize the virtual space.

    Returns:
        new_mos (np.ndarray, shape (nao, nmo)):
            Localized MO coefficients.
    """
    if n_occ is None:
        n_occ = int(np.count_nonzero(mf.mo_occ > 0))
    loc_occ = lo.Boys(mol, mf.mo_coeff[:, :n_occ]).kernel()
    loc_virt = (
        lo.Boys(mol, mf.mo_coeff[:, n_occ:]).kernel()
        if on_virtual
        else mf.mo_coeff[:, n_occ:]
    )
    return np.hstack((loc_occ, loc_virt))


def localize_edmiston(
    mol: gto.Mole,
    mf: Any,
    n_occ: int | None = None,
    *,
    on_virtual: bool = False,
) -> np.ndarray:
    """Edmiston-Ruedenberg localization — maximises orbital self-repulsion energy.

    Args:
        mol (pyscf.gto.Mole):
            Molecular geometry and basis set.
        mf (pyscf.scf.hf.SCF):
            Converged mean-field object supplying ``mo_coeff`` and ``mo_occ``.
        n_occ (int, optional):
            Number of occupied orbitals. Inferred from *mf* if not given.
        on_virtual (bool):
            Whether to also localize the virtual space.

    Returns:
        new_mos (np.ndarray, shape (nao, nmo)):
            Localized MO coefficients.
    """
    if n_occ is None:
        n_occ = int(np.count_nonzero(mf.mo_occ > 0))
    loc_occ = lo.ER(mol, mf.mo_coeff[:, :n_occ]).kernel()
    loc_virt = (
        lo.ER(mol, mf.mo_coeff[:, n_occ:]).kernel()
        if on_virtual
        else mf.mo_coeff[:, n_occ:]
    )
    return np.hstack((loc_occ, loc_virt))


def localize_pipek_mezey(
    mol: gto.Mole,
    mf: Any,
    n_occ: int | None = None,
    *,
    on_virtual: bool = False,
    pop_method: str = "mulliken",
) -> np.ndarray:
    """Pipek-Mezey localization — maximises orbital partial charges on atoms.

    Args:
        mol (pyscf.gto.Mole):
            Molecular geometry and basis set.
        mf (pyscf.scf.hf.SCF):
            Converged mean-field object supplying ``mo_coeff`` and ``mo_occ``.
        n_occ (int, optional):
            Number of occupied orbitals. Inferred from *mf* if not given.
        on_virtual (bool):
            Whether to also localize the virtual space.
        pop_method (str):
            Population analysis scheme used for the virtual block (default ``"mulliken"``).

    Returns:
        new_mos (np.ndarray, shape (nao, nmo)):
            Localized MO coefficients.
    """
    if n_occ is None:
        n_occ = int(np.count_nonzero(mf.mo_occ > 0))
    loc_occ = lo.PM(mol, mf.mo_coeff[:, :n_occ]).kernel()
    loc_virt = (
        lo.PM(mol, mf.mo_coeff[:, n_occ:], pop_method=pop_method).kernel()
        if on_virtual
        else mf.mo_coeff[:, n_occ:]
    )
    return np.hstack((loc_occ, loc_virt))


def localize_ibos(
    mol: gto.Mole,
    mf: Any,
    n_occ: int | None = None,
    *,
    minao: str = "minao",
    exponent: int = 4,
) -> np.ndarray:
    """Intrinsic Bond Orbital (IBO) localization.

    Localizes occupied MOs by maximising their population on intrinsic atomic
    orbitals (IAOs). IBOs tend to give chemically intuitive bond and lone-pair
    orbitals, and handle systems with mixed ionic/covalent character well.

    Args:
        mol (pyscf.gto.Mole):
            Molecular geometry and basis set.
        mf (pyscf.scf.hf.SCF):
            Converged mean-field object supplying ``mo_coeff`` and ``mo_occ``.
        n_occ (int, optional):
            Number of occupied orbitals. Inferred from *mf* if not given.
        minao (str):
            Reference minimal basis used for IAO construction (default ``"minao"``).
        exponent (int):
            Exponent in the IBO localization criterion (default ``4``).

    Returns:
        new_mos (np.ndarray, shape (nao, nmo)):
            IBO coefficients for the occupied block; virtual block is unchanged.
    """
    from pyscf.lo import ibo as _ibo  # noqa: PLC0415

    if mol.natm == 1:
        logger.warning(
            "IBO localization skipped: single-atom molecule has no bonds to localize."
        )
        return np.array(mf.mo_coeff)
    if n_occ is None:
        n_occ = int(np.count_nonzero(mf.mo_occ > 0))
    loc_occ = _ibo.ibo(mol, mf.mo_coeff[:, :n_occ], minao=minao, exponent=exponent)
    return np.hstack((loc_occ, mf.mo_coeff[:, n_occ:]))


def localize_iaos(
    mol: gto.Mole,
    mf: Any,
    n_occ: int | None = None,
    *,
    minao: str = "minao",
) -> np.ndarray:
    """Intrinsic Atomic Orbital (IAO) construction.

    Constructs orthonormal IAOs that closely resemble minimal-basis atomic
    orbitals while spanning the occupied MO space. IAOs are useful as a
    chemically meaningful occupied basis for population analysis and as a
    starting point for IBO construction.

    Args:
        mol (pyscf.gto.Mole):
            Molecular geometry and basis set.
        mf (pyscf.scf.hf.SCF):
            Converged mean-field object supplying ``mo_coeff`` and ``mo_occ``.
        n_occ (int, optional):
            Number of occupied orbitals. Inferred from *mf* if not given.
        minao (str):
            Reference minimal basis (default ``"minao"``).

    Returns:
        new_mos (np.ndarray, shape (nao, nmo)):
            Löwdin-orthogonalized IAOs for the occupied block; virtual block is
            unchanged.
    """
    from pyscf.lo import iao as _iao  # noqa: PLC0415
    from pyscf.lo import orth as _orth  # noqa: PLC0415

    if mol.natm == 1:
        logger.warning(
            "IAO localization skipped: single-atom molecule has no off-site contributions."
        )
        return np.array(mf.mo_coeff)
    if n_occ is None:
        n_occ = int(np.count_nonzero(mf.mo_occ > 0))
    iaos = _iao.iao(mol, mf.mo_coeff[:, :n_occ], minao=minao)
    s = mol.intor_symmetric("int1e_ovlp")
    loc_occ = _orth.vec_lowdin(iaos, s)
    return np.hstack((loc_occ, mf.mo_coeff[:, n_occ:]))


def localize_fnos(
    mf: Any,
    correlated_method: Literal["mp2", "cisd", "ccsd"],
    active: bool | list[int] = True,  # noqa: FBT001, FBT002
    *,
    frozen: list[int] | None = None,
) -> np.ndarray:
    """Frozen natural orbital (FNO) transformation.

    Builds a correlated wavefunction, partitions orbitals into a frozen core and
    an active set, and returns the FNO coefficient matrix.

    Args:
        mf (pyscf.scf.hf.SCF):
            Converged mean-field object.
        correlated_method ({'mp2', 'cisd', 'ccsd'}):
            Correlated method used to build the 1-RDM for the NO transformation.
        active (True or list of int):
            Controls which occupied orbitals enter the correlated calculation.
            ``True`` uses all occupied orbitals (no freezing). A list of integers
            specifies the indices of the active occupied orbitals; all others
            are frozen. Ignored when *frozen* is provided. Indices are given in
            the configured display base (see :func:`aegiss.utils.orb_display.orbital_base`):
            0-based by default, or 1-based when ``AEGISS_ORB_BASE=1``.
        frozen (list of int, optional):
            Indices of occupied orbitals to freeze, in the configured display
            base (0-based by default; 1-based when ``AEGISS_ORB_BASE=1``). Takes
            precedence over *active* when provided.

    Returns:
        mo_coeff (np.ndarray, shape (nao, nmo)):
            FNO coefficients in the AO basis.
    """
    from aegiss.localization.fnos import compute_fnos  # noqa: PLC0415

    logger.info("Computing FNOs using %s", correlated_method.upper())

    orbital_list: list[int]
    if frozen is not None:
        # `frozen` is user input in the display base; convert to 0-based first.
        # `frozen` names the orbitals to exclude; active for get_orbital_spaces
        # is the complement.
        frozen_set = set(from_display_all(frozen))
        orbital_list = [
            int(i) for i in np.nonzero(mf.mo_occ > 0)[0] if i not in frozen_set
        ]
    elif isinstance(active, list):
        # `active` is user input in the display base; convert to 0-based.
        orbital_list = from_display_all(active)
    elif active is True:
        orbital_list = []
    else:
        raise ValueError(
            "active=False is not supported; pass True to use all occupied orbitals or a list of active orbital indices."
        )
    frozen_orbs, active_orbs, _, frozen_virt = get_orbital_spaces(
        mf,
        orbital_list=orbital_list,
        unique_virt_space=True,
    )
    wfn_obj = build_correlated_wfn(mf, correlated_method, frozen=frozen_orbs)
    logger.info("Frozen orbitals: %s", to_display_all(frozen_orbs))
    logger.info("Active orbitals: %s", to_display_all(active_orbs))
    logger.info("Frozen virtual orbitals: %s", to_display_all(frozen_virt))

    return compute_fnos(
        wfn_obj,
        frozen=frozen_orbs.tolist(),
        active=active_orbs.tolist(),
    )


def localize_nos(
    mf: Any,
    correlated_method: Literal["mp2", "cisd", "ccsd"],
    *,
    split: bool = False,
) -> np.ndarray:
    """Natural orbital (NO) transformation from a correlated 1-RDM.

    Always uses the full molecular orbital space. To freeze core orbitals use
    :func:`localize_fnos` instead.

    Args:
        mf (pyscf.scf.hf.SCF):
            Converged mean-field object.
        correlated_method ({'mp2', 'cisd', 'ccsd'}):
            Correlated method used to build the 1-RDM for the NO transformation.
        split (bool):
            If ``True``, diagonalise the occupied and virtual blocks of the 1-RDM
            separately instead of the full matrix.

    Returns:
        nos (np.ndarray, shape (nao, nmo)):
            Natural orbital coefficients in the AO basis.
    """
    from aegiss.localization.nos import compute_nos  # noqa: PLC0415

    logger.info("Computing NOs using %s", correlated_method.upper())
    wfn_obj = build_correlated_wfn(mf, correlated_method)
    logger.info("%s NOs over the full orbital space.", "Split" if split else "Standard")
    result: np.ndarray = compute_nos(wfn_obj, split_nos=split)
    return result


_METHODS: dict[str, Any] = {
    "boys": localize_boys,
    "edmiston": localize_edmiston,
    "er": localize_edmiston,
    "pm": localize_pipek_mezey,
    "pipek": localize_pipek_mezey,
    "ibos": localize_ibos,
    "ibo": localize_ibos,
    "iaos": localize_iaos,
    "iao": localize_iaos,
    "nos": localize_nos,
    "fnos": localize_fnos,
}


def localize(method: str, *args: Any, **kwargs: Any) -> Any:
    """Dispatch to a localization method by name.

    Args:
        method (str):
            One of ``'boys'``, ``'edmiston'`` / ``'er'``, ``'pm'`` / ``'pipek'``,
            ``'ibos'`` / ``'ibo'``, ``'iaos'`` / ``'iao'``, ``'nos'``, ``'fnos'``.
        *args:
            Positional arguments forwarded to the selected method.
        **kwargs:
            Keyword arguments forwarded to the selected method.

    Note:
        Localized orbitals have no orbital energies, so the index-based DMRG window
        (``ncas`` orbitals around the occupied/virtual boundary) is no longer a
        Fermi-level window when built on them.

    Example:
        >>> localize('pm', mol, mf, on_virtual=True)
    >>> localize('ibos', mol, mf)
    >>> localize('iaos', mol, mf, minao='minao')
    >>> localize('nos', mf, correlated_method='mp2', frozen=[0])
    >>> localize('boys', mol, mf, n_occ=5, on_virtual=False)
    """
    if method not in _METHODS:
        raise ValueError(
            f"Unknown localization method {method!r}. Choose from: {sorted(_METHODS)}"
        )
    return _METHODS[method](*args, **kwargs)
