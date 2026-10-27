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

"""Tests for aegiss.utils.plotting."""

from __future__ import annotations

from unittest.mock import MagicMock

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

from aegiss.utils.plotting import (
    _build_ao_heatmap_matrix,
    _define_cas_size,
    _detect_plateaus,
    entanglement_diagrams,
    mi_plot,
    plot_ao_contribution,
    plot_pie_contribution,
    soe_heatmap,
    th_plots,
)


@pytest.fixture
def simple_entropies() -> list[tuple[int, float]]:
    """10 orbitals with linearly increasing absolute entropy values."""
    return [(i, 0.1 * (i + 1)) for i in range(10)]


@pytest.fixture
def mock_mol() -> MagicMock:
    mol = MagicMock()
    mol.search_ao_label.return_value = np.array([0, 1, 2])
    return mol


@pytest.fixture
def mock_mf_restricted() -> MagicMock:
    rng = np.random.default_rng(42)
    mf = MagicMock()
    mf.mo_coeff = rng.random((10, 6))
    return mf


@pytest.fixture
def mock_mf_unrestricted() -> MagicMock:
    rng = np.random.default_rng(0)
    mo = rng.random((10, 6))
    mf = MagicMock()
    mf.mo_coeff = np.stack([mo, mo])
    return mf


class TestDefineCasSize:
    def test_all_occupied(self) -> None:
        n_orbs, n_elec = _define_cas_size(np.array([0, 1, 2]), occ_index=4)
        assert n_orbs == 3
        assert n_elec == 6

    def test_all_virtual(self) -> None:
        n_orbs, n_elec = _define_cas_size(np.array([5, 6]), occ_index=4)
        assert n_orbs == 2
        assert n_elec == 0

    def test_mixed_occ_and_virt(self) -> None:
        n_orbs, n_elec = _define_cas_size(np.array([2, 3, 5]), occ_index=4)
        assert n_orbs == 3
        assert n_elec == 4

    def test_empty(self) -> None:
        n_orbs, n_elec = _define_cas_size(np.array([], dtype=int), occ_index=4)
        assert n_orbs == 0
        assert n_elec == 0


class TestDetectPlateaus:
    @staticmethod
    def _indices_above(y: np.ndarray) -> dict[int, np.ndarray]:
        return {i: np.arange(int(v)) for i, v in enumerate(y)}

    def test_no_plateau_when_all_steps_differ(self) -> None:
        y = np.array([float(100 - i) for i in range(101)])
        _, _, starts, _ = _detect_plateaus(y, self._indices_above(y), plateau_size=10)
        assert len(starts) == 0

    def test_detects_long_flat_region(self) -> None:
        y = np.concatenate([np.full(10, 8.0), np.full(20, 5.0), np.full(71, 2.0)])
        _, _, starts, _ = _detect_plateaus(y, self._indices_above(y), plateau_size=10)
        assert len(starts) >= 1

    def test_short_flat_region_below_threshold_not_detected(self) -> None:
        y = np.concatenate([np.full(5, 8.0), np.full(96, 2.0)])
        _, _, starts, _ = _detect_plateaus(y, self._indices_above(y), plateau_size=10)
        plateau_at_8 = [s for s in starts if np.isclose(y[s], 8.0)]
        assert len(plateau_at_8) == 0

    def test_plateau_vector_matches_starts(self) -> None:
        y = np.concatenate([np.full(50, 4.0), np.full(51, 1.0)])
        plateau_vector, plateau_index, starts, _ = _detect_plateaus(
            y, self._indices_above(y), plateau_size=10
        )
        assert len(plateau_vector) == len(plateau_index)
        assert len(starts) >= 1


class TestSoeHeatmap:
    def test_creates_pdf(
        self, tmp_path: pytest.FixturePath, simple_entropies: list
    ) -> None:
        soe_heatmap(
            so1_entropies=simple_entropies, folder=str(tmp_path), filename="soe"
        )
        assert (tmp_path / "soe.pdf").exists()

    def test_with_n_occ_draws_boundary(
        self, tmp_path: pytest.FixturePath, simple_entropies: list
    ) -> None:
        soe_heatmap(
            so1_entropies=simple_entropies,
            n_occ=4,
            folder=str(tmp_path),
            filename="soe_occ",
        )
        assert (tmp_path / "soe_occ.pdf").exists()

    def test_single_orbital(self, tmp_path: pytest.FixturePath) -> None:
        soe_heatmap(
            so1_entropies=[(0, 0.5)], folder=str(tmp_path), filename="soe_single"
        )
        assert (tmp_path / "soe_single.pdf").exists()

    def test_default_filename(
        self, tmp_path: pytest.FixturePath, simple_entropies: list
    ) -> None:
        soe_heatmap(so1_entropies=simple_entropies, folder=str(tmp_path))
        assert (tmp_path / "SOE_plot.pdf").exists()


class TestThPlots:
    def test_creates_pdf(
        self, tmp_path: pytest.FixturePath, simple_entropies: list
    ) -> None:
        th_plots(so1_entropies=simple_entropies, folder=str(tmp_path), filename="th")
        assert (tmp_path / "th.pdf").exists()

    def test_large_cas_branch(self, tmp_path: pytest.FixturePath) -> None:
        entropies = [(i, 0.01 * i) for i in range(1, 61)]
        th_plots(so1_entropies=entropies, folder=str(tmp_path), filename="th_large")
        assert (tmp_path / "th_large.pdf").exists()

    def test_with_plateau(self, tmp_path: pytest.FixturePath) -> None:
        # Step function → long plateau at both levels → plateau detection runs
        entropies = [(i, 1.0 if i < 5 else 0.1) for i in range(10)]
        th_plots(
            so1_entropies=entropies,
            plateau_size=5,
            folder=str(tmp_path),
            filename="th_plateau",
        )
        assert (tmp_path / "th_plateau.pdf").exists()

    def test_default_filename(
        self, tmp_path: pytest.FixturePath, simple_entropies: list
    ) -> None:
        th_plots(so1_entropies=simple_entropies, folder=str(tmp_path))
        assert (tmp_path / "TH_plot.pdf").exists()


@pytest.fixture
def external_mi() -> tuple[list[tuple[int, float]], np.ndarray]:
    """SOE pairs and an MI matrix built the way block2 does, I = (S_i + S_j - S_ij) / 2."""
    rng = np.random.default_rng(7)
    window = [16, 19, 20, 21, 22, 29]
    s1 = rng.uniform(0.2, 0.6, size=len(window))
    s2 = s1[:, None] + s1[None, :] - rng.uniform(0.0, 0.3, size=(6, 6))
    s2 = (s2 + s2.T) / 2
    minfo = 0.5 * (s1[:, None] + s1[None, :] - s2) * (1 - np.identity(len(s1)))
    return list(zip(window, s1, strict=True)), minfo


class TestMiPlot:
    def test_creates_pdf_from_external_matrix(
        self, tmp_path: pytest.FixturePath, external_mi: tuple
    ) -> None:
        soe1, minfo = external_mi
        mi_plot(minfo, soe1, folder=str(tmp_path), filename="mi")
        assert (tmp_path / "mi.pdf").exists()

    def test_png_format(self, tmp_path: pytest.FixturePath, external_mi: tuple) -> None:
        soe1, minfo = external_mi
        mi_plot(minfo, soe1, folder=str(tmp_path), filename="mi", fmt="png")
        assert (tmp_path / "mi.png").exists()


class TestEntanglementDiagrams:
    def test_creates_pdf_from_external_matrix(
        self, tmp_path: pytest.FixturePath, external_mi: tuple
    ) -> None:
        soe1, minfo = external_mi
        entanglement_diagrams(soe1, minfo, folder=str(tmp_path), filename="ent")
        assert (tmp_path / "ent.pdf").exists()

    def test_mi_convention_and_diagonal_do_not_matter(
        self, tmp_path: pytest.FixturePath, external_mi: tuple
    ) -> None:
        # Without the factor 1/2 and with a non-zero diagonal, the diagram still
        # renders: MI is normalised by its maximum and only i < j pairs are drawn.
        soe1, minfo = external_mi
        unhalved = 2 * minfo + np.diag(np.ones(len(soe1)))
        entanglement_diagrams(soe1, unhalved, folder=str(tmp_path), filename="ent2")
        assert (tmp_path / "ent2.pdf").exists()


class TestPlotPieContribution:
    def test_raises_on_mismatched_label_lengths(
        self, mock_mol: MagicMock, mock_mf_restricted: MagicMock
    ) -> None:
        with pytest.raises(ValueError, match="same length"):
            plot_pie_contribution(
                mock_mol, mock_mf_restricted, ["C 2pz", "Fe 3d"], ["Carbon"], orbs=[0]
            )

    def test_raises_on_empty_orbs(
        self, mock_mol: MagicMock, mock_mf_restricted: MagicMock
    ) -> None:
        with pytest.raises(ValueError, match="empty"):
            plot_pie_contribution(
                mock_mol, mock_mf_restricted, ["C 2pz"], ["Carbon"], orbs=[]
            )

    def test_raises_on_non_positive_orb_for_row(
        self, mock_mol: MagicMock, mock_mf_restricted: MagicMock
    ) -> None:
        with pytest.raises(ValueError, match="positive"):
            plot_pie_contribution(
                mock_mol,
                mock_mf_restricted,
                ["C 2pz"],
                ["Carbon"],
                orbs=[0],
                orb_for_row=0,
            )

    def test_string_ao_label_accepted(
        self,
        tmp_path: pytest.FixturePath,
        mock_mol: MagicMock,
        mock_mf_restricted: MagicMock,
    ) -> None:
        plot_pie_contribution(
            mock_mol,
            mock_mf_restricted,
            "C 2pz",
            ["Carbon"],
            orbs=[0, 1],
            folder=str(tmp_path),
        )
        assert (tmp_path / "composition_alpha.pdf").exists()

    def test_restricted_produces_alpha_only(
        self,
        tmp_path: pytest.FixturePath,
        mock_mol: MagicMock,
        mock_mf_restricted: MagicMock,
    ) -> None:
        plot_pie_contribution(
            mock_mol,
            mock_mf_restricted,
            ["C 2pz"],
            ["Carbon"],
            orbs=[0, 1, 2],
            folder=str(tmp_path),
        )
        assert (tmp_path / "composition_alpha.pdf").exists()
        assert not (tmp_path / "composition_beta.pdf").exists()

    def test_unrestricted_produces_alpha_and_beta(
        self,
        tmp_path: pytest.FixturePath,
        mock_mol: MagicMock,
        mock_mf_unrestricted: MagicMock,
    ) -> None:
        plot_pie_contribution(
            mock_mol,
            mock_mf_unrestricted,
            ["C 2pz"],
            ["Carbon"],
            orbs=[0, 1],
            folder=str(tmp_path),
        )
        assert (tmp_path / "composition_alpha.pdf").exists()
        assert (tmp_path / "composition_beta.pdf").exists()

    def test_min_pct_zero_shows_all_labels(
        self,
        tmp_path: pytest.FixturePath,
        mock_mol: MagicMock,
        mock_mf_restricted: MagicMock,
    ) -> None:
        plot_pie_contribution(
            mock_mol,
            mock_mf_restricted,
            ["C 2pz"],
            ["Carbon"],
            orbs=[0],
            min_pct=0.0,
            folder=str(tmp_path),
        )
        assert (tmp_path / "composition_alpha.pdf").exists()

    def test_grouped_ao_label_entry_accepted(
        self,
        tmp_path: pytest.FixturePath,
        mock_mol: MagicMock,
        mock_mf_restricted: MagicMock,
    ) -> None:
        """list[str] entry in ao_labels is searched as a group; list entry in labels_names is joined."""
        plot_pie_contribution(
            mock_mol,
            mock_mf_restricted,
            [["C 2px", "C 2py"]],
            [["C px", "C py"]],
            orbs=[0, 1],
            folder=str(tmp_path),
        )
        assert (tmp_path / "composition_alpha.pdf").exists()


class TestPlotAoContribution:
    @pytest.fixture
    def simple_contributions(self) -> dict[int, list[tuple[int, float]]]:
        """3 AO groups, 5 MOs each."""
        rng = np.random.default_rng(0)
        return {i: [(j, float(rng.random())) for j in range(5)] for i in range(3)}

    def test_creates_pdf(
        self,
        tmp_path: pytest.FixturePath,
        simple_contributions: dict,
    ) -> None:
        plot_ao_contribution(simple_contributions, folder=str(tmp_path), filename="ao")
        assert (tmp_path / "ao_heatmap.pdf").exists()

    def test_empty_contributions_returns_silently(
        self, tmp_path: pytest.FixturePath
    ) -> None:
        plot_ao_contribution({}, folder=str(tmp_path), filename="ao")
        assert not (tmp_path / "ao_heatmap.pdf").exists()

    def test_empty_group_is_skipped(self, tmp_path: pytest.FixturePath) -> None:
        contributions = {0: [(0, 0.5), (1, 0.3)], 1: []}
        plot_ao_contribution(contributions, folder=str(tmp_path), filename="ao")
        assert (tmp_path / "ao_heatmap.pdf").exists()

    def test_default_letter_labels(
        self,
        tmp_path: pytest.FixturePath,
        simple_contributions: dict,
    ) -> None:
        """No ao_labels → should complete without error using A/B/C labels."""
        plot_ao_contribution(
            simple_contributions, folder=str(tmp_path), filename="ao", ao_labels=None
        )
        assert (tmp_path / "ao_heatmap.pdf").exists()

    def test_string_ao_labels(
        self,
        tmp_path: pytest.FixturePath,
        simple_contributions: dict,
    ) -> None:
        labels = {0: "Fe 3d", 1: "C 2pz", 2: "C 2px"}
        plot_ao_contribution(
            simple_contributions,
            folder=str(tmp_path),
            filename="ao",
            ao_labels=labels,
        )
        assert (tmp_path / "ao_heatmap.pdf").exists()

    def test_list_str_ao_labels_joined(
        self,
        tmp_path: pytest.FixturePath,
        simple_contributions: dict,
    ) -> None:
        labels = {0: ["Fe 3d", "Fe 4d"], 1: "C 2pz", 2: "C 2px"}
        plot_ao_contribution(
            simple_contributions,
            folder=str(tmp_path),
            filename="ao",
            ao_labels=labels,
        )
        assert (tmp_path / "ao_heatmap.pdf").exists()

    def test_mo_labels_accepted(
        self,
        tmp_path: pytest.FixturePath,
        simple_contributions: dict,
    ) -> None:
        plot_ao_contribution(
            simple_contributions,
            folder=str(tmp_path),
            filename="ao",
            mo_labels=[10, 11, 12, 13, 14],
        )
        assert (tmp_path / "ao_heatmap.pdf").exists()

    def test_multiple_panels_when_many_mos(self, tmp_path: pytest.FixturePath) -> None:
        """mos_per_plot=3 on 5 MOs should produce 2 panels without error."""
        contributions = {0: [(j, float(j) / 5) for j in range(5)]}
        plot_ao_contribution(
            contributions,
            folder=str(tmp_path),
            filename="ao",
            mos_per_plot=3,
        )
        assert (tmp_path / "ao_heatmap.pdf").exists()

    def test_combined_alpha_beta_creates_pdf(
        self,
        tmp_path: pytest.FixturePath,
        simple_contributions: dict,
    ) -> None:
        """Passing ao_contributions_beta produces the combined heatmap."""
        beta = {
            i: [(j, v * 0.5) for j, v in simple_contributions[i]]
            for i in simple_contributions
        }
        plot_ao_contribution(
            simple_contributions,
            beta,
            folder=str(tmp_path),
            filename="ao",
        )
        assert (tmp_path / "ao_heatmap.pdf").exists()

    def test_relative_normalization_creates_pdf(
        self,
        tmp_path: pytest.FixturePath,
        simple_contributions: dict,
    ) -> None:
        plot_ao_contribution(
            simple_contributions,
            folder=str(tmp_path),
            filename="ao",
            normalization="relative",
        )
        assert (tmp_path / "ao_heatmap.pdf").exists()


class TestBuildAoHeatmapMatrix:
    def test_restricted_one_row_per_group(self) -> None:
        contributions = {0: [(0, 0.5), (1, 0.3)], 1: [(0, 0.2), (1, 0.8)]}
        matrix, yticks, boundaries = _build_ao_heatmap_matrix(contributions, None, None)
        assert matrix.shape == (2, 2)
        assert boundaries == [1, 2]
        assert len(yticks) == 2

    def test_empty_returns_zero_size_matrix(self) -> None:
        matrix, yticks, boundaries = _build_ao_heatmap_matrix({}, None, None)
        assert matrix.size == 0
        assert yticks == []
        assert boundaries == []

    def test_empty_group_skipped(self) -> None:
        contributions = {0: [(0, 0.5), (1, 0.3)], 1: []}
        matrix, _, boundaries = _build_ao_heatmap_matrix(contributions, None, None)
        assert matrix.shape == (1, 2)
        assert boundaries == [1]

    def test_combined_interleaves_alpha_beta(self) -> None:
        alpha = {0: [(0, 1.0), (1, 0.5)], 1: [(0, 0.4), (1, 0.2)]}
        beta = {0: [(0, 0.6), (1, 0.3)], 1: [(0, 0.8), (1, 0.1)]}
        matrix, yticks, boundaries = _build_ao_heatmap_matrix(alpha, beta, None)
        # 2 groups x (alpha, beta) = 4 rows; group boundaries after rows 2 and 4
        assert matrix.shape == (4, 2)
        assert boundaries == [2, 4]
        assert yticks[0].endswith(r"($\alpha$)")
        assert yticks[1].endswith(r"($\beta$)")

    def test_missing_beta_group_treated_as_zeros(self) -> None:
        alpha = {0: [(0, 1.0), (1, 0.5)]}
        beta: dict[int, list[tuple[int, float]]] = {}
        matrix, _, _ = _build_ao_heatmap_matrix(alpha, beta, None)
        assert matrix.shape == (2, 2)
        np.testing.assert_array_equal(matrix[1], [0.0, 0.0])  # beta row all zeros

    def test_global_normalization_preserves_cross_group_magnitude(self) -> None:
        """Global mode: a weak group stays below 1; the dominant group hits 1."""
        contributions = {0: [(0, 0.1), (1, 0.05)], 1: [(0, 1.0), (1, 0.5)]}
        matrix, _, _ = _build_ao_heatmap_matrix(
            contributions, None, None, normalization="global"
        )
        assert matrix[0].max() < 1.0
        assert matrix[1].max() == pytest.approx(1.0)

    def test_relative_normalization_each_group_maxes_to_one(self) -> None:
        contributions = {0: [(0, 0.1), (1, 0.05)], 1: [(0, 1.0), (1, 0.5)]}
        matrix, _, _ = _build_ao_heatmap_matrix(
            contributions, None, None, normalization="relative"
        )
        assert matrix[0].max() == pytest.approx(1.0)
        assert matrix[1].max() == pytest.approx(1.0)

    def test_combined_relative_uses_joint_group_max(self) -> None:
        """Relative mode normalises alpha and beta of a group by their shared max."""
        alpha = {0: [(0, 1.0), (1, 0.5)]}
        beta = {0: [(0, 0.5), (1, 0.25)]}
        matrix, _, _ = _build_ao_heatmap_matrix(
            alpha, beta, None, normalization="relative"
        )
        # joint max is 1.0 (alpha) → beta scaled by the same 1.0, not its own 0.5
        assert matrix[0].max() == pytest.approx(1.0)
        assert matrix[1].max() == pytest.approx(0.5)

    def test_all_zero_weights_do_not_divide(self) -> None:
        contributions = {0: [(0, 0.0), (1, 0.0)]}
        matrix, _, _ = _build_ao_heatmap_matrix(contributions, None, None)
        np.testing.assert_array_equal(matrix[0], [0.0, 0.0])
