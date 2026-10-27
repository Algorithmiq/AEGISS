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

"""Tests for aegiss.utils.mo_composition."""

from __future__ import annotations

import logging
from typing import Any

import pytest
from pyscf import gto, scf

from aegiss.utils.mo_composition import mo_ao_contributions


class TestMoAoContributions:
    def test_single_int_index_accepted(self, h2o_rhf: Any) -> None:
        result = mo_ao_contributions(h2o_rhf, 4, top_n=3)
        assert set(result.keys()) == {4}

    def test_multiple_indices(self, h2o_rhf: Any) -> None:
        result = mo_ao_contributions(h2o_rhf, [2, 4, 5], top_n=3)
        assert set(result.keys()) == {2, 4, 5}

    def test_restricted_has_alpha_channel_only(self, h2o_rhf: Any) -> None:
        result = mo_ao_contributions(h2o_rhf, 4, top_n=3)
        assert list(result[4].keys()) == ["alpha"]

    def test_top_n_respected(self, h2o_rhf: Any) -> None:
        result = mo_ao_contributions(h2o_rhf, 4, top_n=2)
        assert len(result[4]["alpha"]) == 2

    def test_entries_sorted_by_descending_magnitude(self, h2o_rhf: Any) -> None:
        entries = mo_ao_contributions(h2o_rhf, 4, top_n=5)[4]["alpha"]
        magnitudes = [abs(coeff) for _, _, coeff in entries]
        assert magnitudes == sorted(magnitudes, reverse=True)

    def test_entry_shape_and_types(self, h2o_rhf: Any) -> None:
        ao_idx, label, coeff = mo_ao_contributions(h2o_rhf, 4, top_n=1)[4]["alpha"][0]
        assert isinstance(ao_idx, int)
        assert isinstance(label, str)
        assert isinstance(coeff, float)

    def test_signed_coefficient_preserved(self, h2o_rhf: Any) -> None:
        """Coefficients are reported signed, not as magnitudes."""
        entries = mo_ao_contributions(h2o_rhf, range(h2o_rhf.mo_coeff.shape[1]))
        all_coeffs = [c for per in entries.values() for _, _, c in per["alpha"]]
        assert any(c < 0 for c in all_coeffs)

    def test_label_matches_pyscf_ao_labels(self, h2o_rhf: Any) -> None:
        ao_idx, label, _ = mo_ao_contributions(h2o_rhf, 4, top_n=1)[4]["alpha"][0]
        assert label == h2o_rhf.mol.ao_labels()[ao_idx]

    def test_out_of_range_raises(self, h2o_rhf: Any) -> None:
        n_mo = h2o_rhf.mo_coeff.shape[1]
        with pytest.raises(ValueError, match="out of range"):
            mo_ao_contributions(h2o_rhf, n_mo)

    def test_top_n_larger_than_basis_is_clamped(self, h2o_rhf: Any) -> None:
        n_ao = h2o_rhf.mo_coeff.shape[0]
        entries = mo_ao_contributions(h2o_rhf, 4, top_n=n_ao + 50)[4]["alpha"]
        assert len(entries) == n_ao

    def test_unrestricted_reports_both_spins(self, o_sto3g_uhf: Any) -> None:
        result = mo_ao_contributions(o_sto3g_uhf, 2, top_n=3)
        assert list(result[2].keys()) == ["alpha", "beta"]
        assert len(result[2]["alpha"]) == 3
        assert len(result[2]["beta"]) == 3

    def test_unrestricted_alpha_beta_can_differ(self, o_sto3g_uhf: Any) -> None:
        """Open-shell O: the SOMO region differs between spin channels."""
        result = mo_ao_contributions(
            o_sto3g_uhf, range(o_sto3g_uhf.mo_coeff.shape[2]), top_n=1
        )
        alpha_tops = [per["alpha"][0][0] for per in result.values()]
        beta_tops = [per["beta"][0][0] for per in result.values()]
        assert alpha_tops != beta_tops

    def test_returns_without_error_on_default_top_n(self, h2o_rhf: Any) -> None:
        result = mo_ao_contributions(h2o_rhf, 0)
        assert len(result[0]["alpha"]) == min(10, h2o_rhf.mo_coeff.shape[0])

    def test_coeffients_match_mo_coeff(self, h2o_rhf: Any) -> None:
        """The reported coefficient equals the raw mo_coeff entry."""
        ao_idx, _, coeff = mo_ao_contributions(h2o_rhf, 3, top_n=1)[3]["alpha"][0]
        assert coeff == pytest.approx(h2o_rhf.mo_coeff[ao_idx, 3])

    def test_grouped_labels_drop_atom_index(self, h2o_rhf: Any) -> None:
        """Grouped labels are 'Element basis' with no leading atom id."""
        entries = mo_ao_contributions(h2o_rhf, 4, top_n=5, group_by_element=True)
        for _, label, _ in entries[4]["alpha"]:
            assert label[0].isalpha()  # not a digit-prefixed AO label
            assert label.split()[0].isalpha()

    def test_grouped_weights_are_summed_magnitudes(self, h2o_rhf: Any) -> None:
        """Each group's weight is the sum of |coeff| over its member AOs."""
        labels = h2o_rhf.mol.ao_labels(fmt=False)
        coeff = h2o_rhf.mo_coeff[:, 4]
        expected: dict[str, float] = {}
        for c, (_, sym, nl, m) in zip(coeff, labels, strict=True):
            expected[f"{sym} {nl}{m}"] = expected.get(f"{sym} {nl}{m}", 0.0) + abs(c)
        for count, label, weight in mo_ao_contributions(
            h2o_rhf, 4, top_n=99, group_by_element=True
        )[4]["alpha"]:
            assert weight == pytest.approx(expected[label])
            assert count >= 1
            assert weight >= 0.0

    def test_grouped_merges_atoms_of_same_element(self) -> None:
        """Two O + four H collapse to one group per (element, basis label)."""
        mol = gto.M(
            atom="O 0 0 0; H 0 0 1; H 0 1 0; O 2 0 0; H 2 0 1; H 2 1 0",
            basis="sto-3g",
        )
        mf = scf.RHF(mol).run(verbose=0)
        entries = mo_ao_contributions(mf, 5, top_n=99, group_by_element=True)[5][
            "alpha"
        ]
        # 12 H 1s AOs merge into a single "H 1s" group of count 4.
        h_groups = [(count, lbl) for count, lbl, _ in entries if lbl == "H 1s"]
        assert h_groups == [(4, "H 1s")]

    def test_grouped_entries_ranked_descending(self, h2o_rhf: Any) -> None:
        entries = mo_ao_contributions(h2o_rhf, 4, top_n=5, group_by_element=True)
        weights = [w for _, _, w in entries[4]["alpha"]]
        assert weights == sorted(weights, reverse=True)

    def test_clusters_and_group_by_element_are_exclusive(self, h2o_rhf: Any) -> None:
        with pytest.raises(ValueError, match="at most one"):
            mo_ao_contributions(h2o_rhf, 4, group_by_element=True, clusters={0: [0]})

    def test_clusters_out_of_range_raises(self, h2o_rhf: Any) -> None:
        n_atoms = h2o_rhf.mol.natm
        with pytest.raises(ValueError, match="out of range"):
            mo_ao_contributions(h2o_rhf, 4, clusters={0: [n_atoms]})

    def test_clusters_overlap_raises(self, h2o_rhf: Any) -> None:
        with pytest.raises(ValueError, match="appears in clusters"):
            mo_ao_contributions(h2o_rhf, 4, clusters={0: [0], 1: [0]})

    def test_clustered_labels_use_cluster_id_and_elements(self, h2o_rhf: Any) -> None:
        """Labels read '(<id>) <elements> <ao_label>' (no per-atom indices)."""
        # H2O: atom 0 = O, atoms 1,2 = H.
        entries = mo_ao_contributions(h2o_rhf, 4, top_n=99, clusters={0: [1, 2]})[4][
            "alpha"
        ]
        labels = [lbl for _, lbl, _ in entries]
        assert any(lbl.startswith("(0) H ") for lbl in labels)

    def test_clustered_unassigned_group(self, h2o_rhf: Any) -> None:
        """Atoms in no cluster fall into an 'unassigned' group."""
        entries = mo_ao_contributions(h2o_rhf, 4, top_n=99, clusters={0: [1, 2]})[4][
            "alpha"
        ]
        labels = [lbl for _, lbl, _ in entries]
        assert any(lbl.startswith("(U) ") for lbl in labels)

    def test_clustered_weights_are_summed_magnitudes(self, h2o_rhf: Any) -> None:
        """Each cluster group sums |coeff| over its atoms sharing a basis label."""
        clusters = {0: [0], 1: [1, 2]}
        prefix_of = {0: "(0) O", 1: "(1) H", 2: "(1) H"}
        struct = h2o_rhf.mol.ao_labels(fmt=False)
        coeff = h2o_rhf.mo_coeff[:, 4]
        expected: dict[str, float] = {}
        for c, (atom_id, _, nl, m) in zip(coeff, struct, strict=True):
            key = f"{prefix_of[atom_id]} {nl}{m}"
            expected[key] = expected.get(key, 0.0) + abs(c)
        for count, label, weight in mo_ao_contributions(
            h2o_rhf, 4, top_n=99, clusters=clusters
        )[4]["alpha"]:
            assert weight == pytest.approx(expected[label])
            assert count >= 1

    def test_clustered_mixed_cluster_lists_distinct_elements(self) -> None:
        """A mixed cluster lists its distinct elements in order: 'C,O'."""
        mol = gto.M(
            atom="C 0 0 0; C 1.4 0 0; O 2.8 0 0; H 3.8 0 0; H -1.0 0 0",
            basis="sto-3g",
        )
        mf = scf.RHF(mol).run(verbose=0)
        # Cluster 0 mixes two C and one O; the two H atoms are left unassigned.
        entries = mo_ao_contributions(mf, 5, top_n=99, clusters={0: [0, 1, 2]})[5][
            "alpha"
        ]
        labels = [lbl for _, lbl, _ in entries]
        assert any(lbl.startswith("(0) C,O ") for lbl in labels)
        assert any(lbl.startswith("(U) H ") for lbl in labels)

    def test_clustered_entries_ranked_descending(self, h2o_rhf: Any) -> None:
        entries = mo_ao_contributions(h2o_rhf, 4, top_n=5, clusters={0: [0]})
        weights = [w for _, _, w in entries[4]["alpha"]]
        assert weights == sorted(weights, reverse=True)

    def test_clustered_logs_membership(
        self, h2o_rhf: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Cluster membership (assigned + unassigned) is logged before the table."""
        with caplog.at_level(logging.INFO, logger="aegiss.utils.mo_composition"):
            mo_ao_contributions(h2o_rhf, 4, clusters={0: [0], 1: [1, 2]})
        text = caplog.text
        assert "Cluster membership:" in text
        assert "cluster 0: O(0)" in text
        assert "cluster 1: H(1,2)" in text

    def test_clustered_logs_unassigned_members(
        self, h2o_rhf: Any, caplog: pytest.LogCaptureFixture
    ) -> None:
        with caplog.at_level(logging.INFO, logger="aegiss.utils.mo_composition"):
            mo_ao_contributions(h2o_rhf, 4, clusters={0: [1, 2]})
        assert "unassigned: O(0)" in caplog.text

    def test_pyscf_atom_tags_kept_with_paren_index(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """Tagged atoms keep their tag with the index in parens, e.g. 'H.1(1)'."""
        mol = gto.M(atom="O 0 0 0; H.1 0 0 1; H.2 0 1 0", basis="sto-3g")
        mf = scf.RHF(mol).run(verbose=0)
        with caplog.at_level(logging.INFO, logger="aegiss.utils.mo_composition"):
            entries = mo_ao_contributions(mf, 4, top_n=99, clusters={0: [1, 2]})
        # Membership log keeps the tag and parenthesises the atom index.
        # Distinct tags ('H.1' vs 'H.2') are different symbols, so no collapse.
        assert "cluster 0: H.1(1),H.2(2)" in caplog.text
        # Table labels list the distinct tagged element symbols (no indices).
        labels = [lbl for _, lbl, _ in entries[4]["alpha"]]
        assert any(lbl.startswith("(0) H.1,H.2 ") for lbl in labels)
