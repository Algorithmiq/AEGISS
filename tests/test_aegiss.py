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

"""Tests for the AEGISS active space selection pipeline."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from aegiss.active_space.aegiss import select_aegiss
from aegiss.config_dataclasses import (
    DMRGConfig,
    EntropyConfig,
    OutputConfig,
    ProjectionConfig,
)

_MODULE = "aegiss.active_space.aegiss"

_SCREENED_OCC = [(2, 0.8), (3, 0.6)]
_SCREENED_VIRT = [(5, 0.4)]

# Spin-resolved entropy screening: one (occ, virt) window per spin channel.
_SPIN_SCREENED = {
    "alpha": ([(2, 0.8)], [(5, 0.4)]),
    "beta": ([(3, 0.7)], []),
}


@pytest.fixture(autouse=True)
def _skip_label_check():
    """The mocked ``mf`` has no real basis to check labels against."""
    with patch(f"{_MODULE}._check_ao_labels"):
        yield


@pytest.fixture
def mf_mock() -> MagicMock:
    mf = MagicMock()
    mf.mo_coeff = np.eye(8)
    return mf


@pytest.fixture
def mol_mock() -> MagicMock:
    return MagicMock()


@pytest.fixture
def dmrg_config() -> DMRGConfig:
    return DMRGConfig(ncas=4, nelecas=4)


@pytest.fixture
def entropy_config() -> EntropyConfig:
    return EntropyConfig()


@pytest.fixture
def projection_config_subset() -> ProjectionConfig:
    return ProjectionConfig(ao_labels=["C 2pz"], ranking="subset")


@pytest.fixture
def projection_config_global() -> ProjectionConfig:
    return ProjectionConfig(ao_labels=["C 2pz"], ranking="global")


@pytest.fixture
def pipeline_patches():
    """Patch all heavy dependencies except select_by_ao_projection."""
    with (
        patch(
            f"{_MODULE}.get_subspaces_from_cas",
            return_value=(np.array([2, 3]), np.array([5])),
        ),
        patch(
            "aegiss.active_space.aegiss.run_dmrg_subprocess",
            return_value=(-75.0, np.ones(3)),
        ),
        patch(
            f"{_MODULE}.select_by_entropy",
            return_value=(_SCREENED_OCC, _SCREENED_VIRT),
        ),
        patch(f"{_MODULE}.save_orbitals") as mock_save,
        patch(f"{_MODULE}.save_orbitals_from_entropies") as mock_save_ent,
        patch(f"{_MODULE}._make_soe_plots"),
        patch(f"{_MODULE}.plot_pie_contribution"),
    ):
        yield {"save": mock_save, "save_ent": mock_save_ent}


@pytest.mark.usefixtures("pipeline_patches")
class TestSubsetProjectionRanking:
    def test_ao_projection_called_per_subspace(
        self,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        dmrg_config: DMRGConfig,
        entropy_config: EntropyConfig,
        projection_config_subset: ProjectionConfig,
    ) -> None:
        with patch(
            f"{_MODULE}.select_by_ao_projection",
            side_effect=[np.array([2, 3]), np.array([5])],
        ) as mock_proj:
            select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=entropy_config,
                projection_config=projection_config_subset,
                output_config=OutputConfig(save_orb=False),
            )

        assert mock_proj.call_count == 2

    def test_passes_occ_and_virt_separately(
        self,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        dmrg_config: DMRGConfig,
        entropy_config: EntropyConfig,
        projection_config_subset: ProjectionConfig,
    ) -> None:
        with patch(
            f"{_MODULE}.select_by_ao_projection",
            side_effect=[np.array([2, 3]), np.array([5])],
        ) as mock_proj:
            select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=entropy_config,
                projection_config=projection_config_subset,
                output_config=OutputConfig(save_orb=False),
            )

        assert mock_proj.call_args_list[0].kwargs["restricted_space"] == _SCREENED_OCC
        assert mock_proj.call_args_list[1].kwargs["restricted_space"] == _SCREENED_VIRT

    def test_returns_sorted_ndarray(
        self,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        dmrg_config: DMRGConfig,
        entropy_config: EntropyConfig,
        projection_config_subset: ProjectionConfig,
    ) -> None:
        with patch(
            f"{_MODULE}.select_by_ao_projection",
            side_effect=[np.array([3, 2]), np.array([5])],
        ):
            result = select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=entropy_config,
                projection_config=projection_config_subset,
                output_config=OutputConfig(save_orb=False),
            )

        assert isinstance(result, np.ndarray)
        assert list(result) == [2, 3, 5]

    def test_duplicate_indices_deduplicated(
        self,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        dmrg_config: DMRGConfig,
        entropy_config: EntropyConfig,
        projection_config_subset: ProjectionConfig,
    ) -> None:
        with patch(
            f"{_MODULE}.select_by_ao_projection",
            side_effect=[np.array([2, 3]), np.array([3, 5])],
        ):
            result = select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=entropy_config,
                projection_config=projection_config_subset,
                output_config=OutputConfig(save_orb=False),
            )

        assert list(result) == [2, 3, 5]

    def test_save_path_none_when_save_plots_false(
        self,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        dmrg_config: DMRGConfig,
        entropy_config: EntropyConfig,
        projection_config_subset: ProjectionConfig,
    ) -> None:
        with patch(
            f"{_MODULE}.select_by_ao_projection",
            side_effect=[np.array([2, 3]), np.array([5])],
        ) as mock_proj:
            select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=entropy_config,
                projection_config=projection_config_subset,
                output_config=OutputConfig(save_orb=False, save_plots=False),
            )

        for call in mock_proj.call_args_list:
            assert call.kwargs["save_path"] is None

    def test_save_path_has_occ_virt_suffixes_when_save_plots_true(
        self,
        tmp_path: pytest.TempPathFactory,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        dmrg_config: DMRGConfig,
        entropy_config: EntropyConfig,
        projection_config_subset: ProjectionConfig,
    ) -> None:
        with patch(
            f"{_MODULE}.select_by_ao_projection",
            side_effect=[np.array([2, 3]), np.array([5])],
        ) as mock_proj:
            select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=entropy_config,
                projection_config=projection_config_subset,
                output_config=OutputConfig(
                    save_orb=False,
                    save_plots=True,
                    saving_folder=str(tmp_path) + "/",
                    saving_prefix="mytest",
                ),
            )

        paths = [call.kwargs["save_path"] for call in mock_proj.call_args_list]
        assert any("_occ" in p for p in paths if p)
        assert any("_virt" in p for p in paths if p)

    def test_pie_plot_uses_saving_prefix(
        self,
        tmp_path: pytest.TempPathFactory,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        dmrg_config: DMRGConfig,
        entropy_config: EntropyConfig,
        projection_config_subset: ProjectionConfig,
    ) -> None:
        with (
            patch(
                f"{_MODULE}.select_by_ao_projection",
                side_effect=[np.array([2, 3]), np.array([5])],
            ),
            patch(f"{_MODULE}.plot_pie_contribution") as mock_pie,
        ):
            select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=entropy_config,
                projection_config=projection_config_subset,
                output_config=OutputConfig(
                    save_orb=False,
                    save_plots=True,
                    saving_folder=str(tmp_path) + "/",
                    saving_prefix="mytest",
                ),
            )

        assert mock_pie.call_args.kwargs["filename"] == "mytest_composition"


@pytest.mark.usefixtures("pipeline_patches")
class TestGlobalProjectionRanking:
    def test_ao_projection_called_once(
        self,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        dmrg_config: DMRGConfig,
        entropy_config: EntropyConfig,
        projection_config_global: ProjectionConfig,
    ) -> None:
        with patch(
            f"{_MODULE}.select_by_ao_projection",
            return_value=np.array([2, 3, 5]),
        ) as mock_proj:
            select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=entropy_config,
                projection_config=projection_config_global,
                output_config=OutputConfig(save_orb=False),
            )

        mock_proj.assert_called_once()

    def test_passes_combined_tuple(
        self,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        dmrg_config: DMRGConfig,
        entropy_config: EntropyConfig,
        projection_config_global: ProjectionConfig,
    ) -> None:
        with patch(
            f"{_MODULE}.select_by_ao_projection",
            return_value=np.array([2, 3, 5]),
        ) as mock_proj:
            select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=entropy_config,
                projection_config=projection_config_global,
                output_config=OutputConfig(save_orb=False),
            )

        assert mock_proj.call_args.kwargs["restricted_space"] == (
            _SCREENED_OCC,
            _SCREENED_VIRT,
        )

    def test_returns_sorted_ndarray(
        self,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        dmrg_config: DMRGConfig,
        entropy_config: EntropyConfig,
        projection_config_global: ProjectionConfig,
    ) -> None:
        with patch(
            f"{_MODULE}.select_by_ao_projection",
            return_value=np.array([5, 2, 3]),
        ):
            result = select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=entropy_config,
                projection_config=projection_config_global,
                output_config=OutputConfig(save_orb=False),
            )

        assert isinstance(result, np.ndarray)
        assert list(result) == [2, 3, 5]

    def test_save_path_no_suffix_for_global_when_save_plots_true(
        self,
        tmp_path: pytest.TempPathFactory,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        dmrg_config: DMRGConfig,
        entropy_config: EntropyConfig,
        projection_config_global: ProjectionConfig,
    ) -> None:
        with patch(
            f"{_MODULE}.select_by_ao_projection",
            return_value=np.array([2, 3, 5]),
        ) as mock_proj:
            select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=entropy_config,
                projection_config=projection_config_global,
                output_config=OutputConfig(
                    save_orb=False,
                    save_plots=True,
                    saving_folder=str(tmp_path) + "/",
                    saving_prefix="mytest",
                ),
            )

        save_path = mock_proj.call_args.kwargs["save_path"]
        assert save_path is not None
        assert "_occ" not in save_path
        assert "_virt" not in save_path


class TestOrbitalSaving:
    def test_save_orb_true_calls_save_functions(
        self,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        pipeline_patches: dict,
        dmrg_config: DMRGConfig,
        entropy_config: EntropyConfig,
        projection_config_subset: ProjectionConfig,
    ) -> None:
        with patch(
            f"{_MODULE}.select_by_ao_projection",
            return_value=np.array([2, 3, 5]),
        ):
            select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=entropy_config,
                projection_config=projection_config_subset,
                output_config=OutputConfig(save_orb=True),
            )

        assert pipeline_patches["save_ent"].call_count == 2
        pipeline_patches["save"].assert_called_once()

    def test_save_orb_false_skips_save_functions(
        self,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        pipeline_patches: dict,
        dmrg_config: DMRGConfig,
        entropy_config: EntropyConfig,
        projection_config_subset: ProjectionConfig,
    ) -> None:
        with patch(
            f"{_MODULE}.select_by_ao_projection",
            return_value=np.array([2, 3, 5]),
        ):
            select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=entropy_config,
                projection_config=projection_config_subset,
                output_config=OutputConfig(save_orb=False),
            )

        pipeline_patches["save_ent"].assert_not_called()
        pipeline_patches["save"].assert_not_called()

    def test_saving_prefix_used_in_filenames(
        self,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        pipeline_patches: dict,
        dmrg_config: DMRGConfig,
        entropy_config: EntropyConfig,
        projection_config_subset: ProjectionConfig,
    ) -> None:
        with patch(
            f"{_MODULE}.select_by_ao_projection",
            return_value=np.array([2]),
        ):
            select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=entropy_config,
                projection_config=projection_config_subset,
                output_config=OutputConfig(saving_prefix="mytest", save_orb=True),
            )

        for call_args in pipeline_patches["save_ent"].call_args_list:
            assert call_args.kwargs["filename_prefix"].startswith("mytest")
        assert (
            pipeline_patches["save"]
            .call_args.kwargs["filename_prefix"]
            .startswith("mytest")
        )


class TestEntropyScreeningEarlyReturn:
    def test_both_empty_returns_empty_array(
        self,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        dmrg_config: DMRGConfig,
        entropy_config: EntropyConfig,
        projection_config_subset: ProjectionConfig,
    ) -> None:
        with (
            patch(
                f"{_MODULE}.get_subspaces_from_cas",
                return_value=(np.array([2, 3]), np.array([5])),
            ),
            patch(
                "aegiss.active_space.aegiss.run_dmrg_subprocess",
                return_value=(-75.0, np.ones(3)),
            ),
            patch(f"{_MODULE}.select_by_entropy", return_value=([], [])),
            patch(f"{_MODULE}.select_by_ao_projection") as mock_proj,
            patch(f"{_MODULE}.save_orbitals"),
            patch(f"{_MODULE}.save_orbitals_from_entropies"),
        ):
            result = select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=entropy_config,
                projection_config=projection_config_subset,
                output_config=OutputConfig(save_orb=False),
            )

        assert isinstance(result, np.ndarray)
        assert len(result) == 0
        mock_proj.assert_not_called()

    def test_only_occ_empty_continues_to_projection(
        self,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        dmrg_config: DMRGConfig,
        entropy_config: EntropyConfig,
        projection_config_subset: ProjectionConfig,
    ) -> None:
        with (
            patch(
                f"{_MODULE}.get_subspaces_from_cas",
                return_value=(np.array([2, 3]), np.array([5])),
            ),
            patch(
                "aegiss.active_space.aegiss.run_dmrg_subprocess",
                return_value=(-75.0, np.ones(3)),
            ),
            patch(f"{_MODULE}.select_by_entropy", return_value=([], [(5, 0.4)])),
            patch(
                f"{_MODULE}.select_by_ao_projection",
                return_value=np.array([5]),
            ) as mock_proj,
            patch(f"{_MODULE}.save_orbitals"),
            patch(f"{_MODULE}.save_orbitals_from_entropies"),
        ):
            result = select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=entropy_config,
                projection_config=projection_config_subset,
                output_config=OutputConfig(save_orb=False),
            )

        mock_proj.assert_called_once()
        assert mock_proj.call_args.kwargs["restricted_space"] == [(5, 0.4)]
        assert list(result) == [5]

    def test_only_virt_empty_continues_to_projection(
        self,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        dmrg_config: DMRGConfig,
        entropy_config: EntropyConfig,
        projection_config_subset: ProjectionConfig,
    ) -> None:
        with (
            patch(
                f"{_MODULE}.get_subspaces_from_cas",
                return_value=(np.array([2, 3]), np.array([5])),
            ),
            patch(
                "aegiss.active_space.aegiss.run_dmrg_subprocess",
                return_value=(-75.0, np.ones(3)),
            ),
            patch(f"{_MODULE}.select_by_entropy", return_value=([(2, 0.8)], [])),
            patch(
                f"{_MODULE}.select_by_ao_projection",
                return_value=np.array([2]),
            ) as mock_proj,
            patch(f"{_MODULE}.save_orbitals"),
            patch(f"{_MODULE}.save_orbitals_from_entropies"),
        ):
            result = select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=entropy_config,
                projection_config=projection_config_subset,
                output_config=OutputConfig(save_orb=False),
            )

        mock_proj.assert_called_once()
        assert mock_proj.call_args.kwargs["restricted_space"] == [(2, 0.8)]
        assert list(result) == [2]


class TestDefaultEntropyRanking:
    """The default relative threshold is a fraction of the largest S(1) in the window."""

    def test_weak_virtual_dropped_by_default(
        self,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        dmrg_config: DMRGConfig,
        projection_config_subset: ProjectionConfig,
    ) -> None:
        # The virtual is at 5% of the window maximum: below the default 10% cut
        # globally, but at 100% of its own subspace maximum.
        with (
            patch(
                f"{_MODULE}.get_subspaces_from_cas",
                return_value=(np.array([2, 3]), np.array([5])),
            ),
            patch(
                "aegiss.active_space.aegiss.run_dmrg_subprocess",
                return_value=(-75.0, np.array([1.0, 0.5, 0.05])),
            ),
            patch(
                f"{_MODULE}.select_by_ao_projection",
                return_value=np.array([2, 3]),
            ) as mock_proj,
            patch(f"{_MODULE}.save_orbitals"),
            patch(f"{_MODULE}.save_orbitals_from_entropies"),
        ):
            result = select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=EntropyConfig(),
                projection_config=projection_config_subset,
                output_config=OutputConfig(save_orb=False),
            )

        mock_proj.assert_called_once()
        assert mock_proj.call_args.kwargs["restricted_space"] == [(2, 1.0), (3, 0.5)]
        assert list(result) == [2, 3]


class TestSpinResolvedPipeline:
    """UHF + ``spin_soe='resolved'``: split alpha/beta pipeline, no reconciliation."""

    def _run(
        self,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        proj_side_effect: list,
        ranking: str = "global",
    ):
        dmrg_config = DMRGConfig(ncas=4, nelecas=4, symm_type="SZ", spin_soe="resolved")
        with (
            patch(
                f"{_MODULE}.get_subspaces_from_cas",
                return_value=(np.array([2, 3]), np.array([5])),
            ),
            patch(
                f"{_MODULE}.run_dmrg_subprocess",
                return_value=(-75.0, np.ones((2, 3))),
            ) as mock_dmrg,
            patch(
                f"{_MODULE}.select_by_spin_entropy",
                return_value=_SPIN_SCREENED,
            ) as mock_spin,
            patch(
                f"{_MODULE}.select_by_ao_projection",
                side_effect=proj_side_effect,
            ) as mock_proj,
        ):
            result = select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=EntropyConfig(),
                projection_config=ProjectionConfig(
                    ao_labels=["C 2pz"], ranking=ranking
                ),
                output_config=OutputConfig(save_orb=False, save_plots=False),
            )
        return result, mock_dmrg, mock_spin, mock_proj

    def test_requests_spin_resolved_soe(
        self, mol_mock: MagicMock, mf_mock: MagicMock
    ) -> None:
        _, mock_dmrg, mock_spin, _ = self._run(
            mol_mock,
            mf_mock,
            [{"alpha": np.array([2]), "beta": np.array([3])}],
        )
        assert mock_dmrg.call_args.kwargs["spin_resolved"] is True
        mock_spin.assert_called_once()

    def test_returns_split_not_reconciled(
        self, mol_mock: MagicMock, mf_mock: MagicMock
    ) -> None:
        result, _, _, _ = self._run(
            mol_mock,
            mf_mock,
            [{"alpha": np.array([2, 5]), "beta": np.array([3])}],
        )
        assert isinstance(result, dict)
        assert result["alpha"].tolist() == [2, 5]
        assert result["beta"].tolist() == [3]

    def test_uses_per_channel_restricted_space(
        self, mol_mock: MagicMock, mf_mock: MagicMock
    ) -> None:
        _, _, _, mock_proj = self._run(
            mol_mock,
            mf_mock,
            [{"alpha": np.array([2]), "beta": np.array([3])}],
        )
        restricted_space = mock_proj.call_args.kwargs["restricted_space"]
        assert set(restricted_space) == {"alpha", "beta"}
        assert restricted_space["alpha"] == _SPIN_SCREENED["alpha"]


class TestParameterValidation:
    def test_invalid_entropy_ranking_raises(self) -> None:
        with pytest.raises(ValueError, match="ranking"):
            EntropyConfig(ranking="invalid")  # type: ignore[arg-type]

    def test_invalid_projection_ranking_raises(self) -> None:
        with pytest.raises(ValueError, match="ranking"):
            ProjectionConfig(ao_labels=["C 2pz"], ranking="bad")  # type: ignore[arg-type]

    def test_unmatched_ao_label_raises_before_dmrg(
        self,
        mol_mock: MagicMock,
        mf_mock: MagicMock,
        dmrg_config: DMRGConfig,
        projection_config_subset: ProjectionConfig,
    ) -> None:
        with (
            patch(
                f"{_MODULE}._check_ao_labels", side_effect=ValueError("matches no AO")
            ),
            patch(f"{_MODULE}.run_dmrg_subprocess") as mock_dmrg,
            pytest.raises(ValueError, match="matches no AO"),
        ):
            select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_config=dmrg_config,
                entropy_config=EntropyConfig(),
                projection_config=projection_config_subset,
                output_config=OutputConfig(save_orb=False),
            )
        mock_dmrg.assert_not_called()
