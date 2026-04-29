"""DMRG configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class DMRGConfig:
    """Configuration dataclass for DMRG calculations."""

    ncas: int
    nelecas: int
    init_bond_dim: int = 10
    symm_type: Literal["SZ", "SU2"] = "SZ"

    def __post_init__(self) -> None:
        """Validate the configuration parameters.

        Raises:
        ------
        ValueError
            If any of the parameters are out of their expected ranges or have invalid values.
        """
        if self.ncas <= 0:
            raise ValueError(
                f"Number of active orbitals (ncas) must be positive, got {self.ncas}."
            )
        if self.nelecas <= 0:
            raise ValueError(
                f"Number of active electrons (nelecas) must be positive, got {self.nelecas}."
            )
        if self.init_bond_dim <= 0:
            raise ValueError(
                f"Initial bond dimension must be positive, got {self.init_bond_dim}."
            )
        if self.symm_type not in ("SZ", "SU2"):
            raise ValueError(
                f"Invalid symmetry type {self.symm_type!r}. Must be 'SZ' or 'SU2'."
            )
