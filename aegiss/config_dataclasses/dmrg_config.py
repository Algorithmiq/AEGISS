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
    nroots : int, default 1
        Number of electronic states to target.  ``1`` runs a standard ground-state
        DMRG; values greater than ``1`` activate the SA-DMRG path (state-averaged
        optimization followed by state-specific refinement sweeps).
    weights : tuple of float or None, default None
        Per-state weights for the state-averaged DMRG objective.  Must have
        ``len(weights) == nroots`` when provided.  ``None`` uses uniform weights
        ``1 / nroots`` for each state.  Ignored when ``nroots == 1``.
    n_refinement_sweeps : int, default 4
        Number of state-specific refinement sweeps to run after SA-DMRG
        convergence.  ``0`` skips refinement and computes entropies directly from
        the SA-MPS.  Ignored when ``nroots == 1``.
    """

    ncas: int
    nelecas: int | tuple[int, int]
    init_bond_dim: int = 10
    symm_type: Literal["SZ", "SU2"] = "SZ"
    verbosity: int = 0
    nroots: int = 1
    weights: tuple[float, ...] | None = None
    n_refinement_sweeps: int = 8

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
        self._validate_sa_params()

    def _validate_sa_params(self) -> None:
        """Validate SA-DMRG-specific fields (nroots, weights, n_refinement_sweeps)."""
        if self.nroots < 1:
            raise ValueError(
                f"DMRGConfig: `nroots` must be at least 1, got {self.nroots}."
            )
        if self.weights is not None:
            if len(self.weights) != self.nroots:
                raise ValueError(
                    f"DMRGConfig: `weights` must have exactly `nroots` ({self.nroots}) entries, "
                    f"got {len(self.weights)}."
                )
            if any(w <= 0 for w in self.weights):
                raise ValueError(
                    "DMRGConfig: All entries in `weights` must be positive."
                )
        if self.n_refinement_sweeps < 0:
            raise ValueError(
                f"DMRGConfig: `n_refinement_sweeps` must be non-negative, got {self.n_refinement_sweeps}."
            )
