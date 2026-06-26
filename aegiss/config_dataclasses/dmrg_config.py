"""DMRG configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from aegiss.utils.orbital_indices import resolve_nelecas


@dataclass(frozen=True)
class DMRGConfig:
    """Configuration dataclass for DMRG calculations.

    Parameters:
    ----------
    ncas : int
        Number of active orbitals for the DMRG calculation.
    nelecas : int or tuple[int, int]
        Number of active electrons, either as a total count or an explicit
        ``(nalpha, nbeta)`` pair (same convention as PySCF's CASCI).
    max_bond_dim : int, default 500
        Target DMRG bond dimension (must be a positive multiple of 4).
        The sweep schedule starts at ``max_bond_dim // 2`` and ramps up to ``max_bond_dim``.
    symm_type : {'SZ', 'SU2'}, default 'SZ'
        Spin symmetry used by block2.  ``'SZ'`` is required for entropy
        calculations; ``'SU2'`` is faster but needs MPS conversion for entropies.
    verbosity : int, default 0
        Verbosity level for block2 output.  Higher values produce more detailed logs.
    unrestricted : bool or None, default None
        Whether to emit an unrestricted (UHF) FCIDUMP. ``None`` autodetects from
        the mean-field type (``isinstance(mf, scf.uhf.UHF)``); pass ``True``/
        ``False`` to override. UHF requires ``symm_type='SZ'``.
    reorder : {'noreorder', 'fiedler'}, default 'noreorder'
        Orbital ordering for block2main. ``'noreorder'`` keeps the FCIDUMP order
        and enables ``full_fci_space``; ``'fiedler'`` reorders by the Fiedler
        vector and disables ``full_fci_space``.
    """

    ncas: int
    nelecas: int | tuple[int, int]
    max_bond_dim: int = 500
    symm_type: Literal["SZ", "SU2"] = "SZ"
    verbosity: int = 0
    unrestricted: bool | None = None
    reorder: Literal["noreorder", "fiedler"] = "noreorder"

    def __post_init__(self) -> None:
        """Validate the configuration parameters.

        Raises:
        ------
        ValueError
            If any of the parameters are out of their expected ranges or have invalid values.
        """
        self._validate_cas()
        self._validate_solver()

    def _validate_cas(self) -> None:
        """Validate the active-space size and electron count."""
        if self.ncas <= 0:
            raise ValueError(
                f"DMRGConfig: Number of active orbitals (`ncas`) must be positive, got {self.ncas}."
            )
        if isinstance(self.nelecas, tuple) and len(self.nelecas) != 2:
            raise ValueError(
                "DMRGConfig: `nelecas` tuple must have exactly two elements (nalpha, nbeta)."
            )
        if isinstance(self.nelecas, int) and self.nelecas <= 0:
            raise ValueError(
                f"DMRGConfig: Number of active electrons (`nelecas`) must be positive, got {self.nelecas}."
            )
        nalpha, nbeta = resolve_nelecas(self.nelecas)
        if nalpha < nbeta or nbeta < 0:
            raise ValueError(
                f"DMRGConfig: `nelecas` must satisfy nalpha >= nbeta >= 0, got {self.nelecas}."
            )

    def _validate_solver(self) -> None:
        """Validate the bond dimension, symmetry, verbosity, and ordering options."""
        if self.max_bond_dim <= 0:
            raise ValueError(
                f"DMRGConfig: Target bond dimension (`max_bond_dim`) must be positive, got {self.max_bond_dim}."
            )
        if self.max_bond_dim % 4 != 0:
            raise ValueError(
                f"DMRGConfig: `max_bond_dim` must be a multiple of 4, got {self.max_bond_dim}."
            )
        if self.symm_type not in ("SZ", "SU2"):
            raise ValueError(
                f"DMRGConfig: Invalid symmetry type (`symm_type`) {self.symm_type!r}. Must be 'SZ' or 'SU2'."
            )
        if self.verbosity < 0:
            raise ValueError(
                f"DMRGConfig: `verbosity` must be non-negative, got {self.verbosity}."
            )
        if self.reorder not in ("noreorder", "fiedler"):
            raise ValueError(
                f"DMRGConfig: Invalid reorder {self.reorder!r}. "
                "Must be 'noreorder' or 'fiedler'."
            )
        if self.unrestricted is True and self.symm_type != "SZ":
            raise ValueError(
                "DMRGConfig: `unrestricted=True` requires `symm_type='SZ'`, "
                f"got {self.symm_type!r}."
            )
