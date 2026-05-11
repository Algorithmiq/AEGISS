"""HF and DFT initial guesses via PySCF."""

from __future__ import annotations

import logging
from typing import Any, Literal, overload

from pyscf import dft, scf

from aegiss.utils.env import pyscf_max_memory

logger = logging.getLogger(__name__)


def build_mol(atom: Any, **kwargs: Any) -> Any:
    """Helper function to build a PySCF molecule object.

    Args:
        atom (str or list of str): Atomic geometry in PySCF format, e.g. ``"N 0 0 0; N 0 0 1.1"`` or
            ``["N 0 0 0", "N 0 0 1.1"]``.
        **kwargs: Additional keyword arguments to pass to the Mole object,
            e.g. ``basis="ccpvdz"``.

    Returns:
        mol (pyscf.gto.Mole): Molecule object with specified geometry and basis set."""
    logger.info("----------Building Molecule----------")
    mol = scf.gto.Mole()
    mol.atom = atom
    if "max_memory" not in kwargs:
        mol.max_memory = pyscf_max_memory()
    logger.info("Molecular geometry:")
    logger.info("%s", mol.atom)
    for key, val in kwargs.items():
        logger.debug("Setting mol.%s = %s", key, val)
        setattr(mol, key, val)
    mol.build()
    return mol


@overload
def run_scf(
    mol: scf.hf.SCF,
    method: Literal["RHF", "rhf", "UHF", "uhf"] = ...,
    **kwargs: Any,
) -> scf.hf.SCF: ...


@overload
def run_scf(
    mol: scf.hf.SCF,
    method: Literal["RKS", "rks", "UKS", "uks"] = ...,
    xc: str = ...,
    **kwargs: Any,
) -> dft.rks.KohnShamDFT: ...


def run_scf(
    mol: Any,
    method: str = "RHF",
    xc: str = "b3lyp",
    solvent_info: tuple[str, float] | None = None,
    **kwargs: Any,
) -> Any:
    """Run an HF or KS-DFT mean-field calculation.

    Args:
        mol (pyscf.gto.Mole): Molecular geometry and basis set.
        method ({'RHF', 'UHF', 'RKS', 'UKS'}, optional): Level of theory. Default is ``'RHF'``.
        xc (str, optional): Exchange-correlation functional; only used when *method* is
            ``'RKS'`` or ``'UKS'``. Default is ``'b3lyp'``.
        solvent_info (tuple[str, float], optional): Solvent model to use, specified as ``(model, dielectric_constant)``,
            e.g. ``('PCM', 78.4)`` for water. Default is ``None`` (no solvent).
        **kwargs: Additional attributes forwarded to the PySCF mean-field object
            (e.g. ``max_cycle=50``).

    Returns:
        mf (pyscf.scf.SCF or pyscf.dft.KohnShamDFT): Converged mean-field object."""
    method = method.upper()

    logger.info("----------SCF Calculation (Method %s)----------", method)

    if method in ("RHF", "UHF"):
        mf = getattr(scf, method)(mol)
    elif method in ("RKS", "UKS"):
        mf = getattr(dft, method)(mol)
        mf.xc = xc
    else:
        raise ValueError(f"Unknown method '{method}'. Choose from RHF, UHF, RKS, UKS.")

    for key, val in kwargs.items():
        setattr(mf, key, val)

    if solvent_info is not None:
        from pyscf.solvent import DDCOSMO, PCM  # noqa: PLC0415, F401

        model, eps = solvent_info
        model = model.upper()
        if "PCM" in model:
            mf = mf.PCM()
            mf.with_solvent.method = model
            mf.with_solvent.eps = eps
        elif "DDCOSMO" in model:
            mf = mf.DDCOSMO()
            mf.with_solvent.method = model
            mf.with_solvent.eps = eps
    mf.kernel()
    return mf
