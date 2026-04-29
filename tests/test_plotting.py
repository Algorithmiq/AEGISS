"""Tests for aegiss.utils.plotting."""

from __future__ import annotations

from unittest.mock import MagicMock

import matplotlib as mpl
import numpy as np
import pytest

mpl.use("Agg")

from aegiss.utils.plotting import (
    _define_cas_size,
    _detect_plateaus,
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
        plot_ao_contribution(simple_contributions, save_path=str(tmp_path / "ao"))
        assert (tmp_path / "ao_heatmap.pdf").exists()

    def test_empty_contributions_returns_silently(
        self, tmp_path: pytest.FixturePath
    ) -> None:
        plot_ao_contribution({}, save_path=str(tmp_path / "ao"))
        assert not (tmp_path / "ao_heatmap.pdf").exists()

    def test_empty_group_is_skipped(self, tmp_path: pytest.FixturePath) -> None:
        contributions = {0: [(0, 0.5), (1, 0.3)], 1: []}
        plot_ao_contribution(contributions, save_path=str(tmp_path / "ao"))
        assert (tmp_path / "ao_heatmap.pdf").exists()

    def test_default_letter_labels(
        self,
        tmp_path: pytest.FixturePath,
        simple_contributions: dict,
    ) -> None:
        """No ao_labels → should complete without error using A/B/C labels."""
        plot_ao_contribution(
            simple_contributions, save_path=str(tmp_path / "ao"), ao_labels=None
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
            save_path=str(tmp_path / "ao"),
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
            save_path=str(tmp_path / "ao"),
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
            save_path=str(tmp_path / "ao"),
            mo_labels=[10, 11, 12, 13, 14],
        )
        assert (tmp_path / "ao_heatmap.pdf").exists()

    def test_multiple_panels_when_many_mos(self, tmp_path: pytest.FixturePath) -> None:
        """mos_per_plot=3 on 5 MOs should produce 2 panels without error."""
        contributions = {0: [(j, float(j) / 5) for j in range(5)]}
        plot_ao_contribution(
            contributions,
            save_path=str(tmp_path / "ao"),
            mos_per_plot=3,
        )
        assert (tmp_path / "ao_heatmap.pdf").exists()
