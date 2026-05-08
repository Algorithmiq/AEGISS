"""Unit tests for aegiss.utils.mol_tagger pure-logic modules.

Covers ``_atoms`` (load_xyz, detect_bonds, align_selection_to_xy, get_basis)
and ``_ao_suggest`` (suggest_ao_labels and all character modes) without
touching any Qt/PyVista code.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from ase import Atoms

from aegiss.utils.mol_tagger._ao_suggest import (
    _pi_axis,
    _valence_n,
    suggest_ao_labels,
)
from aegiss.utils.mol_tagger._atoms import (
    align_selection_to_xy,
    detect_bonds,
    get_basis,
    load_xyz,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_XYZ_H2O = """\
3
water
O   0.000  0.000  0.119
H   0.000  0.757 -0.477
H   0.000 -0.757 -0.477
"""

_XYZ_TAGGED = """\
3
tagged water
O.0   0.000  0.000  0.119
H.1   0.000  0.757 -0.477
H     0.000 -0.757 -0.477
"""

# Benzene-like ring flat in XZ plane (y≈0, x/z vary)
_BENZENE_XZ = np.array(
    [
        [1.4,  0.0,  0.0],
        [0.7,  0.0,  1.21],
        [-0.7, 0.0,  1.21],
        [-1.4, 0.0,  0.0],
        [-0.7, 0.0, -1.21],
        [0.7,  0.0, -1.21],
    ]
)

# Regular tetrahedral vertices — score ≈ 1.0 (perfectly non-planar)
_TETRAHEDRAL = np.array(
    [
        [1.0,  1.0,  1.0],
        [1.0,  -1.0, -1.0],
        [-1.0,  1.0, -1.0],
        [-1.0, -1.0,  1.0],
    ],
    dtype=float,
)


def _write_xyz(tmp_path: Path, content: str) -> Path:
    p = tmp_path / "mol.xyz"
    p.write_text(content)
    return p


# ===========================================================================
# get_basis
# ===========================================================================


class TestGetBasis:
    def test_hydrogen(self) -> None:
        b = get_basis(1)
        assert b["all"] == ["1s"]
        assert b["valence"] == ["1s"]

    def test_carbon(self) -> None:
        b = get_basis(6)
        assert b["all"] == ["1s", "2s", "2p"]
        assert b["valence"] == ["2s", "2p"]

    def test_iron(self) -> None:
        b = get_basis(26)
        assert "3d" in b["all"]
        assert "3d" in b["valence"]

    def test_krypton_period4(self) -> None:
        b = get_basis(36)
        assert "4s" in b["all"]
        assert "4p" in b["all"]

    def test_rubidium_fallback(self) -> None:
        """Z=37 (Rb) is not in MINIMAL_BASIS; _guess_basis is used."""
        b = get_basis(37)
        assert len(b["all"]) > 0
        assert len(b["valence"]) > 0

    def test_valence_subset_of_all(self) -> None:
        for Z in [1, 6, 8, 16, 26, 36, 44]:
            b = get_basis(Z)
            for shell in b["valence"]:
                assert shell in b["all"], f"Z={Z}: {shell!r} not in all"


# ===========================================================================
# load_xyz
# ===========================================================================


class TestLoadXyz:
    def test_plain_xyz_loads_correct_symbols(self, tmp_path: Path) -> None:
        p = _write_xyz(tmp_path, _XYZ_H2O)
        atoms, tags = load_xyz(p)
        assert list(atoms.symbols) == ["O", "H", "H"]
        assert tags == {}

    def test_plain_xyz_loads_correct_positions(self, tmp_path: Path) -> None:
        p = _write_xyz(tmp_path, _XYZ_H2O)
        atoms, _ = load_xyz(p)
        assert atoms.positions.shape == (3, 3)
        np.testing.assert_allclose(atoms.positions[0], [0.0, 0.0, 0.119])

    def test_tagged_xyz_recovers_tags(self, tmp_path: Path) -> None:
        p = _write_xyz(tmp_path, _XYZ_TAGGED)
        atoms, tags = load_xyz(p)
        assert tags[0] == "O.0"
        assert tags[1] == "H.1"
        assert 2 not in tags  # untagged H has no entry

    def test_tagged_xyz_strips_suffix_from_element(self, tmp_path: Path) -> None:
        p = _write_xyz(tmp_path, _XYZ_TAGGED)
        atoms, _ = load_xyz(p)
        assert list(atoms.symbols) == ["O", "H", "H"]

    def test_round_trip_via_manual_write(self, tmp_path: Path) -> None:
        """Write a tagged XYZ by hand, read it back, and verify round-trip."""
        content = "2\ntest\nC.0  0.0 0.0 0.0\nN.3  1.2 0.0 0.0\n"
        p = tmp_path / "rt.xyz"
        p.write_text(content)
        atoms, tags = load_xyz(p)
        assert list(atoms.symbols) == ["C", "N"]
        assert tags == {0: "C.0", 1: "N.3"}

    def test_string_path_accepted(self, tmp_path: Path) -> None:
        p = _write_xyz(tmp_path, _XYZ_H2O)
        atoms, tags = load_xyz(str(p))
        assert len(atoms) == 3


# ===========================================================================
# detect_bonds
# ===========================================================================


class TestDetectBonds:
    def test_h2_one_bond(self) -> None:
        atoms = Atoms("HH", positions=[[0, 0, 0], [0, 0, 0.74]])
        bonds = detect_bonds(atoms)
        assert bonds == [(0, 1)]

    def test_h2_too_far_no_bond(self) -> None:
        atoms = Atoms("HH", positions=[[0, 0, 0], [0, 0, 10.0]])
        bonds = detect_bonds(atoms)
        assert bonds == []

    def test_h2_too_close_no_bond(self) -> None:
        """Distances < 0.4 Å are excluded (unphysical overlap)."""
        atoms = Atoms("HH", positions=[[0, 0, 0], [0, 0, 0.1]])
        bonds = detect_bonds(atoms)
        assert bonds == []

    def test_water_two_bonds(self) -> None:
        atoms = Atoms(
            "OHH",
            positions=[[0.0, 0.0, 0.119], [0.0, 0.757, -0.477], [0.0, -0.757, -0.477]],
        )
        bonds = detect_bonds(atoms)
        # O-H bonds; H-H too far
        assert len(bonds) == 2
        indices = {frozenset(b) for b in bonds}
        assert frozenset({0, 1}) in indices
        assert frozenset({0, 2}) in indices

    def test_bonds_are_ordered_pairs(self) -> None:
        """Each bond (i, j) must satisfy i < j."""
        atoms = Atoms("CH4", positions=[[0, 0, 0]] + [[1, 0, 0], [0, 1, 0], [0, 0, 1], [-1, 0, 0]])
        bonds = detect_bonds(atoms)
        for i, j in bonds:
            assert i < j

    def test_no_self_bonds(self) -> None:
        atoms = Atoms("CCC", positions=[[0, 0, 0], [1.5, 0, 0], [3.0, 0, 0]])
        bonds = detect_bonds(atoms)
        for i, j in bonds:
            assert i != j


# ===========================================================================
# align_selection_to_xy
# ===========================================================================


class TestAlignSelectionToXy:
    def test_fewer_than_3_returns_false(self) -> None:
        pos = np.eye(3)
        new_pos, ok = align_selection_to_xy(pos, [0, 1])
        assert ok is False
        np.testing.assert_array_equal(new_pos, pos)

    def test_collinear_atoms_returns_false(self) -> None:
        pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
        _, ok = align_selection_to_xy(pos, [0, 1, 2, 3])
        assert ok is False

    def test_planar_ring_z_becomes_zero(self) -> None:
        """Atoms already in XY plane — mean Z of selection should be ~0."""
        pos = np.array(
            [
                [1.4, 0.0, 0.0],
                [0.7, 1.21, 0.0],
                [-0.7, 1.21, 0.0],
                [-1.4, 0.0, 0.0],
                [-0.7, -1.21, 0.0],
                [0.7, -1.21, 0.0],
            ]
        )
        indices = list(range(6))
        new_pos, ok = align_selection_to_xy(pos, indices)
        assert ok is True
        np.testing.assert_allclose(new_pos[indices, 2].mean(), 0.0, atol=1e-9)

    def test_xz_ring_aligned_to_xy(self) -> None:
        """Ring in XZ plane (y=0) should be rotated so its normal points to Z."""
        indices = list(range(6))
        new_pos, ok = align_selection_to_xy(_BENZENE_XZ.copy(), indices)
        assert ok is True
        # After alignment, mean Z of selection should be ~0
        np.testing.assert_allclose(new_pos[indices, 2].mean(), 0.0, atol=1e-8)

    def test_output_shape_preserved(self) -> None:
        pos = np.random.default_rng(0).random((10, 3))
        new_pos, _ = align_selection_to_xy(pos, list(range(6)))
        assert new_pos.shape == pos.shape

    def test_molecule_centroid_stable(self) -> None:
        """Rotation is about centroid; overall centroid should not shift much."""
        rng = np.random.default_rng(1)
        pos = rng.random((8, 3))
        centroid_before = pos.mean(axis=0)
        new_pos, ok = align_selection_to_xy(pos, [0, 1, 2, 3])
        if ok:
            # Centroid can shift because Z is then adjusted; only X/Y should be stable
            np.testing.assert_allclose(
                new_pos.mean(axis=0)[:2], centroid_before[:2], atol=1e-9
            )

    def test_already_xy_plane_unchanged_xy(self) -> None:
        """Selection already in XY plane: XY coordinates unaffected."""
        pos = np.array(
            [
                [1.0, 0.0, 0.0],
                [0.0, 1.0, 0.0],
                [-1.0, 0.0, 0.0],
                [0.0, -1.0, 0.0],
                [0.0, 0.0, 5.0],  # out-of-plane spectator
            ]
        )
        indices = [0, 1, 2, 3]
        new_pos, ok = align_selection_to_xy(pos, indices)
        assert ok is True
        np.testing.assert_allclose(new_pos[indices, :2], pos[indices, :2], atol=1e-9)


# ===========================================================================
# _valence_n  (internal helper — tested for correctness)
# ===========================================================================


class TestValenceN:
    @pytest.mark.parametrize(
        "sym,expected",
        [
            ("H", 1),
            ("He", 1),
            ("C", 2),
            ("N", 2),
            ("O", 2),
            ("Si", 3),
            ("Fe", 4),
            ("Xe", 5),
            ("Pb", 6),
        ],
    )
    def test_period_mapping(self, sym: str, expected: int) -> None:
        assert _valence_n(sym) == expected


# ===========================================================================
# _pi_axis  (internal helper)
# ===========================================================================


class TestPiAxis:
    def test_xy_plane_normal_is_z(self) -> None:
        pos = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0]], dtype=float)
        axis, score = _pi_axis(pos)
        assert axis == "z"
        assert score < 0.05

    def test_xz_plane_normal_is_y(self) -> None:
        axis, score = _pi_axis(_BENZENE_XZ)
        assert axis == "y"
        assert score < 0.05

    def test_fewer_than_3_returns_zero_score(self) -> None:
        pos = np.array([[0, 0, 0], [1, 0, 0]], dtype=float)
        _, score = _pi_axis(pos)
        assert score == 0.0

    def test_non_planar_high_score(self) -> None:
        _, score = _pi_axis(_TETRAHEDRAL.astype(float))
        assert score > 0.2


# ===========================================================================
# suggest_ao_labels
# ===========================================================================


def _make_inputs(
    syms: list[str],
    pos: np.ndarray,
) -> tuple[np.ndarray, list[str], np.ndarray]:
    """Return (nums, syms, pos) from a list of chemical symbols and positions."""
    from ase.data import atomic_numbers

    nums = np.array([atomic_numbers[s] for s in syms])
    return nums, syms, pos


class TestSuggestAoLabels:
    # --- trivial cases ---

    def test_empty_selection_returns_empty(self) -> None:
        nums, syms, pos = _make_inputs(["C", "C"], np.zeros((2, 3)))
        labels, warning, info = suggest_ao_labels(nums, syms, pos, set(), "auto")
        assert labels == []
        assert "No atoms selected" in info

    def test_custom_mode_returns_empty(self) -> None:
        nums, syms, pos = _make_inputs(["C"], np.zeros((1, 3)))
        labels, _, info = suggest_ao_labels(nums, syms, pos, {0}, "custom")
        assert labels == []
        assert "manually" in info.lower()

    # --- d-shell mode ---

    def test_d_shell_iron_selection(self) -> None:
        nums, syms, pos = _make_inputs(["Fe"], np.zeros((1, 3)))
        labels, warning, _ = suggest_ao_labels(nums, syms, pos, {0}, "d-shell")
        assert any("3d" in lbl for lbl in labels)
        assert warning is None

    def test_d_shell_no_metals_emits_warning(self) -> None:
        nums, syms, pos = _make_inputs(["C", "N"], np.zeros((2, 3)))
        labels, warning, _ = suggest_ao_labels(nums, syms, pos, {0, 1}, "d-shell")
        assert labels == []
        assert warning is not None

    def test_d_shell_mixed_metal_ligand_warns_ligands(self) -> None:
        nums, syms, pos = _make_inputs(["Fe", "C", "N"], np.zeros((3, 3)))
        labels, warning, _ = suggest_ao_labels(nums, syms, pos, {0, 1, 2}, "d-shell")
        assert any("3d" in lbl for lbl in labels)
        assert warning is not None  # ligand warning

    def test_d_shell_lanthanide_emits_secondary_shell(self) -> None:
        """Ce (Z=58) should produce both 4f (primary) and 5d (secondary)."""
        nums, syms, pos = _make_inputs(["Ce"], np.zeros((1, 3)))
        labels, _, _ = suggest_ao_labels(nums, syms, pos, {0}, "d-shell")
        shells = {lbl.split()[-1] for lbl in labels}
        assert "4f" in shells
        assert "5d" in shells

    # --- pi / auto mode ---

    def test_pi_planar_benzene_carbon_selection(self) -> None:
        """6 C atoms in XZ plane → normal Y → labels should be 'C 2py'."""
        syms = ["C"] * 6
        nums, syms, pos = _make_inputs(syms, _BENZENE_XZ)
        labels, warning, info = suggest_ao_labels(nums, syms, pos, set(range(6)), "auto")
        assert warning is None
        # Exactly one unique label since all atoms share the same bare symbol
        assert labels == ["C 2py"]

    def test_pi_non_planar_falls_back_to_all_p(self) -> None:
        syms = ["C"] * 4
        nums, syms, pos = _make_inputs(syms, _TETRAHEDRAL)
        labels, warning, _ = suggest_ao_labels(nums, syms, pos, set(range(4)), "auto")
        assert "C 2p" in labels  # all-p (no axis suffix) for non-planar
        assert warning is not None

    def test_pi_hydrogen_skipped_no_p(self) -> None:
        """H has no p orbitals; suggest_ao_labels should not include them."""
        syms = ["H"] * 3
        nums, syms, pos = _make_inputs(syms, np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], float))
        labels, _, _ = suggest_ao_labels(nums, syms, pos, {0, 1, 2}, "auto")
        # H (n=1) has no p → no labels in pi mode
        assert labels == []

    def test_pi_metal_plus_organic(self) -> None:
        atom_syms = ["Fe", "C", "C", "C", "C", "C", "C"]
        atom_pos = np.vstack([[0, 0, 0], _BENZENE_XZ])
        nums, _, _ = _make_inputs(atom_syms, atom_pos)
        labels, _, _ = suggest_ao_labels(nums, atom_syms, atom_pos, set(range(7)), "auto")
        assert any("3d" in lbl for lbl in labels)
        assert any("2p" in lbl for lbl in labels)

    # --- sigma mode ---

    def test_sigma_planar_has_s_and_inplane_p(self) -> None:
        syms = ["C"] * 6
        nums, syms, pos = _make_inputs(syms, _BENZENE_XZ)
        labels, warning, info = suggest_ao_labels(nums, syms, pos, set(range(6)), "sigma")
        assert warning is None
        # Normal is Y, so in-plane axes are X and Z
        assert "C 2s" in labels
        assert "C 2px" in labels
        assert "C 2pz" in labels
        # Out-of-plane py should NOT appear
        assert "C 2py" not in labels

    def test_sigma_no_organics_emits_warning(self) -> None:
        nums, syms, pos = _make_inputs(["Fe"], np.zeros((1, 3)))
        labels, warning, _ = suggest_ao_labels(nums, syms, pos, {0}, "sigma")
        assert labels == []
        assert warning is not None

    # --- tagged atoms use label prefix ---

    def test_tagged_atoms_use_label_prefix(self) -> None:
        """When tags are present the label prefix should be the tag string."""
        syms = ["C", "C"]
        nums, syms, pos = _make_inputs(syms, np.array([[0, 0, 0], [1, 0, 0]], float))
        tags = {0: "C.0", 1: "C.1"}
        labels, _, _ = suggest_ao_labels(
            nums, syms, pos, {0, 1}, "d-shell", tags=tags
        )
        # d-shell for C → no metal warning, but we're testing that the labels
        # (even if empty for non-metals) don't crash; also test via auto mode
        fe_syms = ["Fe", "Fe"]
        fe_pos = np.zeros((2, 3))
        nums2, _, _ = _make_inputs(fe_syms, fe_pos)
        tags2 = {0: "Fe.0", 1: "Fe.1"}
        labels2, _, _ = suggest_ao_labels(nums2, fe_syms, fe_pos, {0, 1}, "d-shell", tags=tags2)
        assert "Fe.0 3d" in labels2
        assert "Fe.1 3d" in labels2

    def test_no_tags_uses_bare_symbol(self) -> None:
        nums, syms, pos = _make_inputs(["Fe"], np.zeros((1, 3)))
        labels, _, _ = suggest_ao_labels(nums, syms, pos, {0}, "d-shell")
        assert labels == ["Fe 3d"]

    # --- label uniqueness ---

    def test_no_duplicate_labels(self) -> None:
        syms = ["C", "C", "C", "N", "N"]
        pos = np.array(
            [[0, 0, 0], [1.4, 0, 0], [2.8, 0, 0], [4.2, 0, 0], [5.6, 0, 0]], float
        )
        nums, syms, pos = _make_inputs(syms, pos)
        for mode in ("auto", "sigma", "d-shell"):
            labels, _, _ = suggest_ao_labels(nums, syms, pos, set(range(5)), mode)
            assert len(labels) == len(set(labels)), f"Duplicates in {mode}: {labels}"
