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
    init_bond_dim : int, default 10
        Initial DMRG bond dimension; grows automatically during the sweep schedule.
    symm_type : {'SZ', 'SU2'}, default 'SZ'
        Spin symmetry used by block2.  ``'SZ'`` is required for entropy
        calculations; ``'SU2'`` is faster but needs MPS conversion for entropies.
    verbosity : int, default 0
        Verbosity level for block2 output.  Higher values produce more detailed logs.
    backend : {'pyblock2', 'subprocess'}, default 'pyblock2'
        DMRG execution backend.  ``'pyblock2'`` uses the Python API directly;
        ``'subprocess'`` calls the ``block2main`` executable and reads numpy
        output files (no Python import of block2).
    """

    ncas: int
    nelecas: int | tuple[int, int]
    init_bond_dim: int = 10
    symm_type: Literal["SZ", "SU2"] = "SZ"
    verbosity: int = 0
    backend: Literal["pyblock2", "subprocess"] = "pyblock2"

    def __post_init__(self) -> None:
        """Validate the configuration parameters.

        Raises:
        ------
        ValueError
            If any of the parameters are out of their expected ranges or have invalid values.
        """
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
        if self.init_bond_dim <= 0:
            raise ValueError(
                f"DMRGConfig: Initial bond dimension (`init_bond_dim`) must be positive, got {self.init_bond_dim}."
            )
        if self.symm_type not in ("SZ", "SU2"):
            raise ValueError(
                f"DMRGConfig: Invalid symmetry type (`symm_type`) {self.symm_type!r}. Must be 'SZ' or 'SU2'."
            )
        if self.verbosity < 0:
            raise ValueError(
                f"DMRGConfig: `verbosity` must be non-negative, got {self.verbosity}."
            )
        if self.backend not in ("pyblock2", "subprocess"):
            raise ValueError(
                f"DMRGConfig: Invalid backend {self.backend!r}. Must be 'pyblock2' or 'subprocess'."
            )
