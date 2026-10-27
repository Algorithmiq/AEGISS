# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
        mol (pyscf.gto.Mole): Molecule object with specified geometry and basis set.
    """
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
    xc: str = ...,
    solvent_info: tuple[str, float] | None = ...,
    relativistic: str | None = ...,
    **kwargs: Any,
) -> scf.hf.SCF: ...


@overload
def run_scf(
    mol: scf.hf.SCF,
    method: Literal["RKS", "rks", "UKS", "uks"] = ...,
    xc: str = ...,
    solvent_info: tuple[str, float] | None = ...,
    relativistic: str | None = ...,
    **kwargs: Any,
) -> dft.rks.KohnShamDFT: ...


def run_scf(
    mol: Any,
    method: str = "RHF",
    xc: str = "b3lyp",
    solvent_info: tuple[str, float] | None = None,
    relativistic: str | None = None,
    **kwargs: Any,
) -> Any:
    """Run an HF or KS-DFT mean-field calculation.

    Args:
        mol (pyscf.gto.Mole): Molecular geometry and basis set.
        method ({'RHF', 'UHF', 'RKS', 'UKS'}, optional): Level of theory. Default is ``'RHF'``.
        xc (str, optional): Exchange-correlation functional; only used when *method* is
            ``'RKS'`` or ``'UKS'``. Default is ``'b3lyp'``.
        solvent_info (tuple[str, float], optional): Solvent model to use, specified as
            ``(model, dielectric_constant)``, e.g. ``('IEF-PCM', 78.4)`` for water.
            *model* is case-insensitive. ``'C-PCM'`` / ``'CPCM'`` and
            ``'IEF-PCM'`` / ``'IEFPCM'`` select PySCF's PCM solver with that
            variant; ``'DDCOSMO'`` selects ddCOSMO. Any other name containing
            ``'PCM'`` (including plain ``'PCM'``) makes PySCF raise
            ``RuntimeError`` when the SCF runs; any name containing neither
            ``'PCM'`` nor ``'DDCOSMO'`` (e.g. ``'COSMO'``) is silently ignored
            and the calculation runs without solvent. Default is ``None`` (no
            solvent).
        relativistic ({'x2c'} or None, optional): Relativistic correction to apply.
            ``'x2c'`` applies the scalar exact two-component (X2C) transformation via
            ``mf.x2c()``. Default is ``None`` (no relativistic correction).
        **kwargs: Additional attributes forwarded to the PySCF mean-field object
            (e.g. ``max_cycle=50``).

    Returns:
        mf (pyscf.scf.SCF or pyscf.dft.KohnShamDFT): Converged mean-field object.

    Raises:
        ValueError: If *method* is not one of ``'RHF'``, ``'UHF'``, ``'RKS'``,
            ``'UKS'``, or if *relativistic* is empty or not ``'x2c'``.
        RuntimeError: Raised by PySCF if *solvent_info* names an unsupported
            PCM variant (see *solvent_info*).
    """
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

    if relativistic is not None:
        relativistic = relativistic.strip().lower()
        if not relativistic:
            raise ValueError("Relativistic method must be specified. Choose from: x2c.")
        if relativistic == "x2c":
            logger.info("Applying X2C scalar relativistic correction.")
            mf = mf.x2c()
        else:
            raise ValueError(
                f"Unknown relativistic method '{relativistic}'. Choose from: x2c."
            )

    mf.kernel()
    _log_scf_result(mf, method)
    return mf


def _log_scf_result(mf: Any, method: str) -> None:
    """Log the mean-field energy, or warn when the SCF stopped unconverged."""
    if mf.converged:
        logger.info("SCF converged: E(%s) = %.10f Eh", method, mf.e_tot)
    else:
        logger.warning(
            "SCF did NOT converge after %d cycles: last E(%s) = %.10f Eh",
            mf.max_cycle,
            method,
            mf.e_tot,
        )
