"""Per-orbital AO composition reporting (PySCF ``analyze``-style)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


_MOS_PER_ROW = 4
_COEFF_WIDTH = 9
_COL_SEP = "  |  "
_RULE_SEP = "--+--"


def _format_composition_table(
    composition: dict[int, dict[str, list[tuple[int, str, float]]]],
    top_n: int,
    *,
    unrestricted: bool,
) -> str:
    """Lay out the per-orbital contributions as side-by-side columns.

    Each molecular orbital becomes one column (restricted) or two adjacent
    columns — alpha then beta — when *unrestricted*. Orbitals are grouped into
    blocks of ``_MOS_PER_ROW``; within a block every column shares the same
    label width so coefficients line up, and a rule separates the headers from
    the values.
    """
    # One list of (header, entries) per molecular orbital.
    per_mo: list[list[tuple[str, list[tuple[int, str, float]]]]] = []
    for mo_idx, per_spin in composition.items():
        per_mo.append(
            [
                (f"MO {mo_idx} {spin}" if unrestricted else f"MO {mo_idx}", entries)
                for spin, entries in per_spin.items()
            ]
        )

    flat = [col for mo_cols in per_mo for col in mo_cols]
    label_w = max(
        (len(label) for _, entries in flat for _, label, _ in entries), default=1
    )
    cell_w = max(label_w + 2 + _COEFF_WIDTH, max((len(h) for h, _ in flat), default=0))

    def cell(entry: tuple[int, str, float] | None) -> str:
        if entry is None:
            return " " * cell_w
        _, label, coeff = entry
        return f"{label:<{label_w}}  {coeff:>+{_COEFF_WIDTH}.4f}".ljust(cell_w)

    blocks: list[str] = []
    for start in range(0, len(per_mo), _MOS_PER_ROW):
        cols = [c for mo_cols in per_mo[start : start + _MOS_PER_ROW] for c in mo_cols]
        header = _COL_SEP.join(h.ljust(cell_w) for h, _ in cols)
        rule = _RULE_SEP.join("-" * cell_w for _ in cols)
        rows = [
            _COL_SEP.join(
                cell(entries[r] if r < len(entries) else None) for _, entries in cols
            )
            for r in range(top_n)
        ]
        blocks.append("\n".join([header, rule, *rows]))

    return "\n\n".join(blocks)


def _grouped_entries(
    coeff: np.ndarray,
    ao_struct: Sequence[tuple[int, str, str, str]],
    top_n: int,
) -> list[tuple[int, str, float]]:
    """Collapse AOs by element + basis label, summing ``|coeff|``.

    The atom index is dropped, so every ``C 2pz`` across the molecule is merged
    into a single ``"C 2pz"`` group whose weight is the sum of the absolute
    coefficients of its members. Groups are ranked by that summed magnitude and
    the *top_n* largest returned. The leading tuple element is the number of AOs
    aggregated into the group (in place of an AO index).
    """
    weights: dict[str, float] = {}
    counts: dict[str, int] = {}
    for c, (_, symbol, nl, m) in zip(coeff, ao_struct, strict=True):
        label = f"{symbol} {nl}{m}"
        weights[label] = weights.get(label, 0.0) + abs(float(c))
        counts[label] = counts.get(label, 0) + 1
    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [(counts[label], label, weight) for label, weight in ranked]


def mo_ao_contributions(
    mf: Any,
    mo_indices: int | Sequence[int],
    *,
    top_n: int = 10,
    group_by_element: bool = False,
) -> dict[int, dict[str, list[tuple[int, str, float]]]]:
    """Print the top-N atomic-orbital contributions to one or more MOs.

    For each requested molecular orbital the AO basis functions are ranked by
    the magnitude of their expansion coefficient ``mo_coeff[mu, i]`` and the
    *top_n* largest are kept. The signed coefficient is reported, so the
    orbital's phase is preserved. Useful for inspecting an MO that shows only a
    small weight in an AO-projection heatmap: it reveals which AOs the orbital
    is actually built from.

    When *group_by_element* is set, contributions are instead collapsed by
    element symbol and basis label (the atom index is dropped), so every
    ``C 2pz`` across the molecule is merged into one ``"C 2pz"`` group whose
    weight is the sum of the absolute coefficients of its members. Groups are
    ranked by that summed magnitude.

    The result is always logged as a table with one column per orbital
    (restricted) or a pair of alpha/beta columns per orbital (unrestricted).

    Parameters
    ----------
    mf : pyscf.scf.hf.SCF
        Converged mean-field object exposing ``mo_coeff`` and ``mol``.
        ``mo_coeff`` may be restricted ``(n_ao, n_mo)`` or unrestricted
        ``(2, n_ao, n_mo)``; in the latter case both spins are reported.
    mo_indices : int or sequence of int
        Index (or indices) of the molecular orbital(s) to decompose.
    top_n : int
        Number of largest AO contributions (or element groups) to keep per
        orbital.
    group_by_element : bool
        If ``True``, sum ``|coefficient|`` over all atoms of the same element
        sharing a basis label, ranking the resulting groups instead of
        individual AOs.

    Returns:
    -------
    composition : dict[int, dict[str, list[tuple[int, str, float]]]]
        Maps each requested MO index to a ``{spin: entries}`` dict, where
        *spin* is ``"alpha"`` (and ``"beta"`` for unrestricted inputs) and
        *entries* is a list of ``(key, label, weight)`` triples ordered from
        largest to smallest weight. Ungrouped: ``key`` is the AO index and
        ``weight`` is the signed coefficient. Grouped: ``key`` is the number of
        AOs in the group and ``weight`` is the summed ``|coefficient|``.
    """
    indices = [mo_indices] if isinstance(mo_indices, int) else list(mo_indices)

    mo_coeff = mf.mo_coeff
    unrestricted = np.ndim(mo_coeff) == 3
    channels = (
        (("alpha", mo_coeff[0]), ("beta", mo_coeff[1]))
        if unrestricted
        else (("alpha", mo_coeff),)
    )
    ao_labels = mf.mol.ao_labels()
    ao_struct = mf.mol.ao_labels(fmt=False) if group_by_element else None
    n_mo = channels[0][1].shape[1]

    composition: dict[int, dict[str, list[tuple[int, str, float]]]] = {}
    for mo_idx in indices:
        if not 0 <= mo_idx < n_mo:
            raise ValueError(f"MO index {mo_idx} out of range [0, {n_mo}).")
        per_spin: dict[str, list[tuple[int, str, float]]] = {}
        for spin, coeff_matrix in channels:
            coeff = coeff_matrix[:, mo_idx]
            if ao_struct is not None:
                per_spin[spin] = _grouped_entries(coeff, ao_struct, top_n)
            else:
                order = np.argsort(np.abs(coeff))[::-1][:top_n]
                per_spin[spin] = [
                    (int(i), ao_labels[i], float(coeff[i])) for i in order
                ]
        composition[mo_idx] = per_spin

    detail = (
        "summed |mo_coeff| per element+label"
        if group_by_element
        else "signed mo_coeff entries"
    )
    logger.info(
        "Top-%d AO contributions per MO (%s):\n%s",
        top_n,
        detail,
        _format_composition_table(composition, top_n, unrestricted=unrestricted),
    )
    return composition
