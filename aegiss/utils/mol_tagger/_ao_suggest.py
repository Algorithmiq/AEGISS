"""AO label suggestion for MolTagger's chemical character modes."""

from __future__ import annotations

import numpy as np
from ase.data import atomic_numbers as _atomic_numbers

_PLANARITY_THRESHOLD: float = 0.25
_MIN_PLANE_ATOMS: int = 3
_SVD_EPS: float = 1e-10

# Transition metal d/f shell: atomic number -> primary shell label
_D_SHELL: dict[int, str] = {
    **dict.fromkeys(range(21, 31), "3d"),  # Sc-Zn
    **dict.fromkeys(range(39, 49), "4d"),  # Y-Cd (includes Ru=44)
    57: "5d",  # La
    **dict.fromkeys(range(58, 72), "4f"),  # Ce-Lu (lanthanides)
    **dict.fromkeys(range(72, 81), "5d"),  # Hf-Hg
    89: "6d",  # Ac
    **dict.fromkeys(range(90, 104), "5f"),  # Th-Lr (actinides)
    **dict.fromkeys(range(104, 113), "6d"),  # Rf-Cn
}

# f-block elements also need the (n+1)d shell in CASSCF active spaces
_D_SHELL_SECONDARY: dict[int, str] = {
    **dict.fromkeys(range(58, 72), "5d"),  # Ce-Lu: 4f primary + 5d secondary
    **dict.fromkeys(range(90, 104), "6d"),  # Th-Lr: 5f primary + 6d secondary
}

_Z_P1: int = 2  # He  — last period-1 element (H, He have only 1s, no p)
_Z_P2: int = 10  # Ne  — last period-2 element
_Z_P3: int = 18  # Ar  — last period-3 element
_Z_P4: int = 36  # Kr  — last period-4 element
_Z_P5: int = 54  # Xe  — last period-5 element

# Internal atom record: (atom_index, atomic_number, base_symbol, label_prefix)
# label_prefix is the tag string (e.g. "C.1") when the atom is tagged, else the
# bare symbol (e.g. "C").  base_symbol is always the chemical element and is
# used for valence-number look-ups.
_Atom = tuple[int, int, str, str]


def _valence_n(sym: str) -> int:
    """Return the principal quantum number of the valence p/s shell for *sym*.

    Args:
        sym: Chemical symbol (e.g. ``"C"``, ``"N"``).

    Returns:
        Principal quantum number (1 for H/He, 2 for period 2, 3 for period 3, ...).
    """
    Z = _atomic_numbers.get(sym, 6)
    if Z <= _Z_P1:
        return 1
    if Z <= _Z_P2:
        return 2
    if Z <= _Z_P3:
        return 3
    if Z <= _Z_P4:
        return 4
    if Z <= _Z_P5:
        return 5
    return 6


def _pi_axis(org_pos: np.ndarray) -> tuple[str, float]:
    """Return the best-fit-plane normal axis and planarity score for *org_pos*.

    Uses singular value decomposition of the centred position matrix.  The
    third right singular vector is perpendicular to the best-fit plane.

    Args:
        org_pos: Positions of the atoms, shape ``(n, 3)``.

    Returns:
        axis: Which of ``'x'``, ``'y'``, ``'z'`` is closest to the plane normal.
        score: ``s[2] / s[1]`` from SVD — near 0 means flat, near 1 means
            spherical.  Values below ``_PLANARITY_THRESHOLD`` are planar.
    """
    if len(org_pos) < _MIN_PLANE_ATOMS:
        return "z", 0.0
    centered = org_pos - org_pos.mean(axis=0)
    _, s, vt = np.linalg.svd(centered, full_matrices=False)
    normal = vt[2]
    score = float(s[2] / s[1]) if s[1] > _SVD_EPS else 0.0
    return ["x", "y", "z"][int(np.argmax(np.abs(normal)))], score


def _unique_pairs(atoms: list[_Atom]) -> list[tuple[str, str]]:
    """Return sorted unique (label_prefix, base_symbol) pairs from *atoms*.

    Deduplication is by label_prefix; alphabetical sort ensures a stable,
    consistent ordering regardless of atom-index order in the selection.

    Args:
        atoms: List of ``(atom_index, Z, base_symbol, label_prefix)`` records.

    Returns:
        Sorted list of ``(label_prefix, base_symbol)`` pairs.
    """
    seen: dict[str, str] = {}
    for _, _, sym, label in atoms:
        if label not in seen:
            seen[label] = sym
    return sorted(seen.items())


def suggest_ao_labels(
    nums: np.ndarray,
    syms: list[str],
    pos: np.ndarray,
    selected: set[int],
    character: str,
    tags: dict[int, str] | None = None,
) -> tuple[list[str], str | None, str]:
    """Suggest PySCF-compatible ``ao_labels`` strings for the selected atoms.

    Classifies each selected atom as a transition metal or a main-group element
    and applies the appropriate orbital selection rule for the requested
    *character* mode.

    When *tags* is provided, tagged atoms use their tag string (e.g. ``"C.1"``)
    as the label prefix instead of the bare element symbol, so the output
    distinguishes symmetry-inequivalent atoms that carry different tags.

    Args:
        nums: Atomic numbers, shape ``(n_atoms,)``.
        syms: Chemical symbols, length ``n_atoms``.
        pos: Atomic positions, shape ``(n_atoms, 3)``.
        selected: Indices of currently selected atoms.
        character: One of ``"auto"`` (pi + d-shell auto), ``"d-shell"``,
            ``"sigma"``, or ``"custom"``.
        tags: Optional mapping from atom index to tag string (e.g. ``"C.1"``).
            Untagged atoms fall back to their bare chemical symbol.

    Returns:
        labels: Ready-to-use ``ao_labels`` strings for ``ProjectionConfig``.
        warning: Non-fatal warning (e.g. non-planar atoms) or ``None``.
        info: One-line explanation of the suggestion.
    """
    if not selected:
        return [], None, "No atoms selected."
    if character == "custom":
        return [], None, "Enter labels manually."

    _tags: dict[int, str] = tags or {}
    sel = sorted(selected)
    sel_nums = [int(nums[i]) for i in sel]
    sel_syms = [syms[i] for i in sel]
    sel_labels = [_tags.get(i, syms[i]) for i in sel]

    metals: list[_Atom] = [
        (sel[k], Z, sel_syms[k], sel_labels[k])
        for k, Z in enumerate(sel_nums)
        if Z in _D_SHELL
    ]
    organics: list[_Atom] = [
        (sel[k], Z, sel_syms[k], sel_labels[k])
        for k, Z in enumerate(sel_nums)
        if Z not in _D_SHELL
    ]

    if character == "d-shell":
        return _d_shell_labels(metals, organics)
    if character == "sigma":
        return _sigma_labels(organics, pos)
    return _pi_labels(metals, organics, pos)


def _collect_s_labels(pairs: list[tuple[str, str]], seen: set[str]) -> list[str]:
    """Collect ``ns`` labels for *pairs*, skipping duplicates already in *seen*.

    Args:
        pairs: ``(label_prefix, base_symbol)`` pairs for unique atom identifiers.
        seen: Already-used label set; updated in place to avoid duplicates.

    Returns:
        New ``"prefix ns"`` labels added.
    """
    labels: list[str] = []
    for label, sym in pairs:
        lbl = f"{label} {_valence_n(sym)}s"
        if lbl not in seen:
            labels.append(lbl)
            seen.add(lbl)
    return labels


def _collect_p_labels(
    pairs: list[tuple[str, str]], suffix: str, seen: set[str]
) -> list[str]:
    """Collect p-orbital labels for *pairs*, appending *suffix* to the angular part.

    Args:
        pairs: ``(label_prefix, base_symbol)`` pairs for unique atom identifiers.
        suffix: Axis qualifier appended to ``p``, e.g. ``"z"`` → ``"2pz"``,
            or ``""`` for all-p (``"2p"``).
        seen: Already-used label set; updated in place to avoid duplicates.

    Returns:
        New labels added; H/He (n=1) are silently skipped — they have no p orbitals.
    """
    labels: list[str] = []
    for label, sym in pairs:
        n = _valence_n(sym)
        if n < 2:
            continue
        lbl = f"{label} {n}p{suffix}"
        if lbl not in seen:
            labels.append(lbl)
            seen.add(lbl)
    return labels


def _d_shell_labels(
    metals: list[_Atom],
    organics: list[_Atom],
) -> tuple[list[str], str | None, str]:
    """Suggest d/f-shell labels for transition metals only.

    Args:
        metals: Atom records for metal atoms.
        organics: Atom records for non-metal atoms.

    Returns:
        labels, warning, info — see :func:`suggest_ao_labels`.
    """
    if not metals:
        return [], "No transition metals in selection.", "d-shell: no metals found."
    seen: set[str] = set()
    labels: list[str] = []
    for _, Z, _, label in metals:
        for shell_dict in (_D_SHELL, _D_SHELL_SECONDARY):
            if Z in shell_dict:
                lbl = f"{label} {shell_dict[Z]}"
                if lbl not in seen:
                    labels.append(lbl)
                    seen.add(lbl)
    warning: str | None = None
    if organics:
        lig_labels = sorted({label for _, _, _, label in organics})
        warning = f"Ligand atoms ({', '.join(lig_labels)}) ignored in d-shell mode."
    return labels, warning, f"d-shell: {', '.join(labels)}"


def _pi_labels(
    metals: list[_Atom],
    organics: list[_Atom],
    pos: np.ndarray,
) -> tuple[list[str], str | None, str]:
    """Suggest pi-system labels: d-shell for metals, p{axis} for planar organics.

    Args:
        metals: Atom records for metal atoms.
        organics: Atom records for non-metal atoms.
        pos: Full position array for all atoms (indexed by atom index).

    Returns:
        labels, warning, info — see :func:`suggest_ao_labels`.
    """
    seen: set[str] = set()
    labels: list[str] = []
    info_parts: list[str] = []
    warning: str | None = None

    metal_label_set: set[str] = set()
    for _, Z, _, label in metals:
        for shell_dict in (_D_SHELL, _D_SHELL_SECONDARY):
            if Z in shell_dict:
                lbl = f"{label} {shell_dict[Z]}"
                if lbl not in seen:
                    labels.append(lbl)
                    seen.add(lbl)
                metal_label_set.add(lbl)
    if metals:
        info_parts.append(f"metals: {', '.join(sorted(metal_label_set))}")

    if organics:
        org_labels, warning, org_info = _organic_pi_labels(organics, pos, seen)
        labels.extend(org_labels)
        info_parts.append(org_info)

    return labels, warning, " | ".join(info_parts) or "No atoms classified."


def _organic_pi_labels(
    organics: list[_Atom],
    pos: np.ndarray,
    seen: set[str],
) -> tuple[list[str], str | None, str]:
    """Determine pi-orbital labels for main-group atoms using planarity detection.

    Args:
        organics: Atom records for non-metal atoms.
        pos: Full position array for all atoms (indexed by atom index).
        seen: Already-used label set; updated in place to avoid duplicates.

    Returns:
        labels, warning, info — see :func:`suggest_ao_labels`.
    """
    org_pos = pos[[i for i, _, _, _ in organics]]
    pairs = _unique_pairs(organics)
    pair_labels = [lbl for lbl, _ in pairs]

    if len(organics) < _MIN_PLANE_ATOMS:
        return (
            _collect_p_labels(pairs, "", seen),
            "Fewer than 3 organic atoms; pi axis undetermined - using all p.",
            "organics: all p",
        )

    axis, score = _pi_axis(org_pos)
    if score < _PLANARITY_THRESHOLD:
        return (
            _collect_p_labels(pairs, axis, seen),
            None,
            f"pi-normal ~ {axis.upper()} (score {score:.2f})",
        )

    return (
        _collect_p_labels(pairs, "", seen),
        f"Atoms not coplanar (score {score:.2f} >= {_PLANARITY_THRESHOLD}); using all p.",
        f"non-planar ({', '.join(pair_labels)})",
    )


def _sigma_labels(
    organics: list[_Atom],
    pos: np.ndarray,
) -> tuple[list[str], str | None, str]:
    """Suggest sigma-framework labels: ns + in-plane np for main-group atoms.

    Args:
        organics: Atom records for non-metal atoms.
        pos: Full position array for all atoms.

    Returns:
        labels, warning, info — see :func:`suggest_ao_labels`.
    """
    if not organics:
        return [], "No main-group atoms in selection.", ""

    org_pos = pos[[i for i, _, _, _ in organics]]
    pairs = _unique_pairs(organics)

    if len(organics) < _MIN_PLANE_ATOMS:
        seen: set[str] = set()
        labels = (
            _collect_s_labels(pairs, seen)
            + _collect_p_labels(pairs, "x", seen)
            + _collect_p_labels(pairs, "y", seen)
        )
        return (
            labels,
            "Fewer than 3 atoms; sigma axis undetermined.",
            f"sigma: {', '.join(labels)}",
        )

    axis, score = _pi_axis(org_pos)
    in_plane = [a for a in ("x", "y", "z") if a != axis]
    seen = set()
    labels = (
        _collect_s_labels(pairs, seen)
        + _collect_p_labels(pairs, in_plane[0], seen)
        + _collect_p_labels(pairs, in_plane[1], seen)
    )
    warning = (
        None
        if score < _PLANARITY_THRESHOLD
        else f"Atoms not coplanar (score {score:.2f}); sigma axis uncertain."
    )
    info = (
        f"sigma: in-plane {in_plane[0].upper()},{in_plane[1].upper()} | "
        f"pi-normal ~ {axis.upper()} (score {score:.2f})"
    )
    return labels, warning, info
