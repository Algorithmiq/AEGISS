"""Per-orbital AO composition reporting (PySCF ``analyze``-style)."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

logger = logging.getLogger(__name__)


_UNASSIGNED = object()  # sentinel cluster key for atoms in no cluster


def _atom_group_str(atom_ids: Sequence[int], symbol_of: Callable[[int], str]) -> str:
    """List atoms, collapsing those of the same element: ``"C(0,1,2),N(3)"``."""
    by_symbol: dict[str, list[int]] = {}
    for a in sorted(atom_ids):
        by_symbol.setdefault(symbol_of(a), []).append(a)
    return ",".join(
        f"{sym}({','.join(str(i) for i in ids)})" for sym, ids in by_symbol.items()
    )


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


def _clustered_entries(
    coeff: np.ndarray,
    ao_struct: Sequence[tuple[int, str, str, str]],
    clusters: Mapping[int, Sequence[int]],
    top_n: int,
) -> list[tuple[int, str, float]]:
    """Collapse AOs by cluster + basis label, summing ``|coeff|``.

    Each atom is mapped to the cluster containing it (or a catch-all group keyed
    ``U`` for atoms in no cluster); AOs are then merged by
    ``(cluster, basis label)`` and weighted by the sum of the absolute
    coefficients of their members. Each group label reads
    ``"(<cluster id>) <elements> <ao_label>"`` — for example ``"(0) C 2pz"`` (or
    ``"(0) C,O 2pz"`` for a mixed cluster, ``"(U) H 1s"`` for unassigned atoms) —
    listing the distinct element symbols of the cluster, so the same basis
    function on different clusters stays separate. Groups are ranked by summed
    magnitude and the *top_n* largest returned; the leading tuple element is the
    number of AOs aggregated.
    """
    atom_cluster: dict[int, object] = {}
    for cid, atom_ids in clusters.items():
        for aid in atom_ids:
            atom_cluster[aid] = cid

    atom_symbol: dict[int, str] = {}
    for atom_id, symbol, _, _ in ao_struct:
        atom_symbol.setdefault(atom_id, symbol)

    weights: dict[tuple[object, str], float] = {}
    counts: dict[tuple[object, str], int] = {}
    members: dict[object, set[int]] = {}
    for c, (atom_id, _, nl, m) in zip(coeff, ao_struct, strict=True):
        cluster = atom_cluster.get(atom_id, _UNASSIGNED)
        key = (cluster, f"{nl}{m}")
        weights[key] = weights.get(key, 0.0) + abs(float(c))
        counts[key] = counts.get(key, 0) + 1
        members.setdefault(cluster, set()).add(atom_id)

    def group_label(cluster: object, basis: str) -> str:
        elements: list[str] = []
        for a in sorted(members[cluster]):
            if atom_symbol[a] not in elements:
                elements.append(atom_symbol[a])
        cid = "U" if cluster is _UNASSIGNED else cluster
        return f"({cid}) {','.join(elements)} {basis}"

    ranked = sorted(weights.items(), key=lambda kv: kv[1], reverse=True)[:top_n]
    return [
        (counts[key], group_label(key[0], key[1]), weight) for key, weight in ranked
    ]


def _entries_for_coeff(
    coeff: np.ndarray,
    top_n: int,
    *,
    ao_labels: Sequence[str],
    ao_struct: Sequence[tuple[int, str, str, str]] | None,
    clusters: Mapping[int, Sequence[int]] | None,
) -> list[tuple[int, str, float]]:
    """Rank one MO's contributions, optionally collapsed by cluster/element."""
    if ao_struct is None:
        order = np.argsort(np.abs(coeff))[::-1][:top_n]
        return [(int(i), ao_labels[i], float(coeff[i])) for i in order]
    if clusters is not None:
        return _clustered_entries(coeff, ao_struct, clusters, top_n)
    return _grouped_entries(coeff, ao_struct, top_n)


def _log_cluster_membership(mol: Any, clusters: Mapping[int, Sequence[int]]) -> None:
    """Log each cluster's member atoms, collapsing those of the same element."""
    symbol_of = mol.atom_symbol
    assigned: set[int] = set()
    lines: list[str] = []
    for cid in sorted(clusters):
        members = list(clusters[cid])
        assigned.update(members)
        lines.append(f"  cluster {cid}: {_atom_group_str(members, symbol_of)}")
    unassigned = [a for a in range(mol.natm) if a not in assigned]
    if unassigned:
        lines.append(f"  unassigned: {_atom_group_str(unassigned, symbol_of)}")
    logger.info("Cluster membership:\n%s", "\n".join(lines))


def _validate_clusters(clusters: Mapping[int, Sequence[int]], n_atoms: int) -> None:
    """Check cluster atom indices are in range and assigned to one cluster."""
    seen: dict[int, int] = {}
    for cid, atom_ids in clusters.items():
        for aid in atom_ids:
            if not 0 <= aid < n_atoms:
                raise ValueError(
                    f"Atom index {aid} in cluster {cid} out of range [0, {n_atoms})."
                )
            if aid in seen:
                raise ValueError(
                    f"Atom index {aid} appears in clusters {seen[aid]} and {cid}."
                )
            seen[aid] = cid


def mo_ao_contributions(
    mf: Any,
    mo_indices: int | Sequence[int],
    *,
    top_n: int = 10,
    group_by_element: bool = False,
    clusters: Mapping[int, Sequence[int]] | None = None,
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

    When *clusters* is given, contributions are collapsed by cluster and basis
    label instead: each atom is mapped to the cluster containing it (atoms in no
    cluster fall into a catch-all group keyed ``U``), and ``|coefficient|`` is
    summed over all atoms of a cluster sharing a basis label. The group label
    reads ``"(<cluster id>) <elements> <ao_label>"``, e.g. ``"(0) C 2pz"`` (or
    ``"(U) H 1s"`` for unassigned atoms). The cluster membership — each cluster's
    atoms with indices — is logged first. *clusters* and *group_by_element* are
    mutually exclusive.

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
    clusters : mapping of int to sequence of int, optional
        Maps a cluster id to the atom indices it contains. When given, sum
        ``|coefficient|`` over the atoms of each cluster sharing a basis label.
        Atoms must each belong to at most one cluster; any not listed fall into
        the catch-all ``U`` group. Mutually exclusive with *group_by_element*.

    Returns:
    -------
    composition : dict[int, dict[str, list[tuple[int, str, float]]]]
        Maps each requested MO index to a ``{spin: entries}`` dict, where
        *spin* is ``"alpha"`` (and ``"beta"`` for unrestricted inputs) and
        *entries* is a list of ``(key, label, weight)`` triples ordered from
        largest to smallest weight. Ungrouped: ``key`` is the AO index, *label*
        the PySCF AO label and ``weight`` the signed coefficient. Grouped (by
        element or cluster): ``key`` is the number of AOs in the group and
        ``weight`` the summed ``|coefficient|``.
    """
    if clusters is not None and group_by_element:
        raise ValueError("Pass at most one of `group_by_element` and `clusters`.")

    indices = [mo_indices] if isinstance(mo_indices, int) else list(mo_indices)

    mo_coeff = mf.mo_coeff
    unrestricted = np.ndim(mo_coeff) == 3
    channels = (
        (("alpha", mo_coeff[0]), ("beta", mo_coeff[1]))
        if unrestricted
        else (("alpha", mo_coeff),)
    )
    ao_labels = mf.mol.ao_labels()
    grouped = group_by_element or clusters is not None
    ao_struct = mf.mol.ao_labels(fmt=False) if grouped else None
    n_mo = channels[0][1].shape[1]

    if clusters is not None:
        _validate_clusters(clusters, mf.mol.natm)
        _log_cluster_membership(mf.mol, clusters)

    composition: dict[int, dict[str, list[tuple[int, str, float]]]] = {}
    for mo_idx in indices:
        if not 0 <= mo_idx < n_mo:
            raise ValueError(f"MO index {mo_idx} out of range [0, {n_mo}).")
        per_spin = {
            spin: _entries_for_coeff(
                coeff_matrix[:, mo_idx],
                top_n,
                ao_labels=ao_labels,
                ao_struct=ao_struct,
                clusters=clusters,
            )
            for spin, coeff_matrix in channels
        }
        composition[mo_idx] = per_spin

    if clusters is not None:
        detail = "summed |mo_coeff| per cluster+label"
    elif group_by_element:
        detail = "summed |mo_coeff| per element+label"
    else:
        detail = "signed mo_coeff entries"
    logger.info(
        "Top-%d AO contributions per MO (%s):\n%s",
        top_n,
        detail,
        _format_composition_table(composition, top_n, unrestricted=unrestricted),
    )
    return composition
