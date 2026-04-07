"""HF and DFT initial guesses via PySCF."""
from __future__ import annotations

from typing import Literal, overload

from pyscf import dft, scf

def build_mol(atom, **kwargs):
    """Helper function to build a PySCF molecule object.

    Parameters
    ----------
    atom : str or list of str
        Atomic geometry in PySCF format, e.g. "N 0 0 0; N 0 0 1.1" or
        ["N 0 0 0", "N 0 0 1.1"].
    **kwargs
        Additional keyword arguments to pass to the Mole object, e.g. ``basis="ccpvdz"``.
        
    Returns
    -------
    mol : pyscf.gto.Mole
        Molecule object with specified geometry and basis set.
    """
    mol = scf.gto.Mole()
    mol.atom = atom
    for key, val in kwargs.items():
        print(f"Setting mol.{key} = {val}")
        setattr(mol, key, val)
    mol.build()
    return mol

@overload
def run_scf(
    mol: scf.hf.SCF,
    method: Literal["RHF", "rhf", "UHF", "uhf"] = ...,
    **kwargs,
) -> scf.hf.SCF: ...


@overload
def run_scf(
    mol: scf.hf.SCF,
    method: Literal["RKS", "rks", "UKS", "uks"] = ...,
    xc: str = ...,
    **kwargs,
) -> dft.rks.KohnShamDFT: ...


def run_scf(mol, method="RHF", xc="b3lyp", solvent_info: tuple[str, float] | None = None, **kwargs):
    """Run an HF or KS-DFT mean-field calculation.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecular geometry and basis set.
    method : {'RHF', 'UHF', 'RKS', 'UKS'}, optional
        Level of theory. Default is ``'RHF'``.
    xc : str, optional
        Exchange-correlation functional; only used when *method* is
        ``'RKS'`` or ``'UKS'``. Default is ``'b3lyp'``.
    solvent_info : tuple[str, float], optional
        Solvent model to use, specified as a tuple of (model, dielectric constant),
        e.g. ``('PCM', 78.4)`` for water. Default is ``None`` (no solvent).
        
    **kwargs
        Additional attributes forwarded to the PySCF mean-field object
        (e.g. ``max_cycle=50``).

    Returns
    -------
    mf : pyscf.scf.SCF or pyscf.dft.KohnShamDFT
        Converged mean-field object.
    """
    method = method.upper()

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
        from pyscf import solvent
        from pyscf.solvent import PCM, DDCOSMO
        model, eps = solvent_info
        model = model.upper()
        if 'PCM' in model:
            mf = mf.PCM()
            mf.with_solvent.method = model
            mf.with_solvent.eps = eps
        elif 'DDCOSMO' in model:
            mf = mf.DDCOSMO()
            mf.with_solvent.method = model
            mf.with_solvent.eps = eps
    mf.kernel()
    return mf
