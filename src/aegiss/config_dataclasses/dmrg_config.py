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

"""DMRG configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from aegiss.utils.orbital_indices import resolve_nelecas


@dataclass(frozen=True)
class DMRGConfig:
    """Configuration dataclass for DMRG calculations.

    Args:
        ncas (int):
            Number of active orbitals for the DMRG calculation.
        nelecas (int or tuple[int, int]):
            Number of active electrons, either as a total count or an explicit
            ``(nalpha, nbeta)`` pair (same convention as PySCF's CASCI).
        max_bond_dim (int, default 500):
            Target DMRG bond dimension (must be a positive multiple of 4).
            The sweep schedule starts at ``max_bond_dim // 2`` and ramps up to ``max_bond_dim``.
        symm_type ({'SZ', 'SU2'}, default 'SZ'):
            Spin symmetry used by block2. Both give spin-summed single-orbital
            entropies directly from the ``1pdm``/``1npc`` output; ``'SU2'`` is
            spin-adapted and faster. ``'SZ'`` is required only for
            ``spin_soe='resolved'`` and for unrestricted (UHF) integrals.
        verbosity (int, default 0):
            Verbosity level for block2 output.  Higher values produce more detailed logs.
        unrestricted (bool or None, default None):
            Whether to emit an unrestricted (UHF) FCIDUMP. ``None`` autodetects from
            the mean-field type (``isinstance(mf, scf.uhf.UHF)``); pass ``True``/
            ``False`` to override. UHF requires ``symm_type='SZ'``.
        reorder ({'noreorder', 'fiedler'}, default 'noreorder'):
            Orbital ordering for block2main. ``'noreorder'`` keeps the FCIDUMP order
            and enables ``full_fci_space``; ``'fiedler'`` reorders by the Fiedler
            vector and disables ``full_fci_space``.
        spin_soe ({'auto', 'summed', 'resolved'}, default 'auto'):
            Whether single-orbital entropies (and the selection built on them) are
            spin-summed or spin-resolved. ``'summed'`` uses the 4-state spatial
            entropy, so both spin channels share one entropy window; for a UHF
            reference the AO projection still returns a split ``{'alpha', 'beta'}``
            active space. ``'resolved'`` uses the
            per-spin-orbital entropies ``H(n_alpha)`` / ``H(n_beta)`` and screens the
            alpha and beta channels independently, yielding a split
            ``{'alpha', 'beta'}`` active space. ``'auto'`` picks ``'resolved'`` when
            the target is spin-polarised (``nalpha != nbeta``) and ``symm_type='SZ'``,
            else ``'summed'``. ``'resolved'`` requires ``symm_type='SZ'`` (the
            spin-resolved 1-PDM is only available without spin adaptation).
    """

    ncas: int
    nelecas: int | tuple[int, int]
    max_bond_dim: int = 500
    symm_type: Literal["SZ", "SU2"] = "SZ"
    verbosity: int = 0
    unrestricted: bool | None = None
    reorder: Literal["noreorder", "fiedler"] = "noreorder"
    spin_soe: Literal["auto", "summed", "resolved"] = "auto"

    def __post_init__(self) -> None:
        """Validate the configuration parameters.

        Raises:
            ValueError: If any of the parameters are out of their expected ranges or have invalid values.
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
        if self.spin_soe not in ("auto", "summed", "resolved"):
            raise ValueError(
                f"DMRGConfig: Invalid spin_soe {self.spin_soe!r}. "
                "Must be 'auto', 'summed', or 'resolved'."
            )
        if self.spin_soe == "resolved" and self.symm_type != "SZ":
            raise ValueError(
                "DMRGConfig: `spin_soe='resolved'` requires `symm_type='SZ'`, "
                f"got {self.symm_type!r} (the spin-resolved 1-PDM is only "
                "available without spin adaptation)."
            )

    def spin_resolved(self) -> bool:
        """Resolve *spin_soe* to a concrete spin-resolved / spin-summed choice.

        ``'auto'`` selects spin-resolved only for a spin-polarised target
        (``nalpha != nbeta``) computed with ``symm_type='SZ'``; otherwise the
        spin-summed spatial entropy is used.

        Returns:
            spin_resolved (bool): ``True`` if the single-orbital entropies
                should be spin-resolved.
        """
        if self.spin_soe == "resolved":
            return True
        if self.spin_soe == "summed":
            return False
        nalpha, nbeta = resolve_nelecas(self.nelecas)
        return nalpha != nbeta and self.symm_type == "SZ"
