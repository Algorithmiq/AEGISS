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
            f"{_MODULE}.run_dmrg",
            return_value=(-75.0, np.ones(3), None, None),
        ),
        patch(
            f"{_MODULE}.select_by_entropy",
            return_value=(_SCREENED_OCC, _SCREENED_VIRT),
        ),
        patch(f"{_MODULE}.save_orbitals") as mock_save,
        patch(f"{_MODULE}.save_orbitals_from_entropies") as mock_save_ent,
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
        self, mol_mock: MagicMock, mf_mock: MagicMock
    ) -> None:
        with (
            patch(
                f"{_MODULE}.get_subspaces_from_cas",
                return_value=(np.array([2, 3]), np.array([5])),
            ),
            patch(f"{_MODULE}.run_dmrg", return_value=(-75.0, np.ones(3), None, None)),
            patch(f"{_MODULE}.select_by_entropy", return_value=([], [])),
            patch(f"{_MODULE}.select_by_ao_projection") as mock_proj,
            patch(f"{_MODULE}.save_orbitals"),
            patch(f"{_MODULE}.save_orbitals_from_entropies"),
        ):
            result = select_aegiss(
                mol_mock, mf_mock, dmrg_ncas=4, dmrg_nelecas=4, save_orb=False
            )

        assert isinstance(result, np.ndarray)
        assert len(result) == 0
        mock_proj.assert_not_called()

    def test_only_occ_empty_continues_to_projection(
        self, mol_mock: MagicMock, mf_mock: MagicMock
    ) -> None:
        with (
            patch(
                f"{_MODULE}.get_subspaces_from_cas",
                return_value=(np.array([2, 3]), np.array([5])),
            ),
            patch(f"{_MODULE}.run_dmrg", return_value=(-75.0, np.ones(3), None, None)),
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
                dmrg_ncas=4,
                dmrg_nelecas=4,
                projection_ranking="subset",
                save_orb=False,
            )

        mock_proj.assert_called_once()
        assert mock_proj.call_args.kwargs["restricted_space"] == [(5, 0.4)]
        assert list(result) == [5]

    def test_only_virt_empty_continues_to_projection(
        self, mol_mock: MagicMock, mf_mock: MagicMock
    ) -> None:
        with (
            patch(
                f"{_MODULE}.get_subspaces_from_cas",
                return_value=(np.array([2, 3]), np.array([5])),
            ),
            patch(f"{_MODULE}.run_dmrg", return_value=(-75.0, np.ones(3), None, None)),
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
                dmrg_ncas=4,
                dmrg_nelecas=4,
                projection_ranking="subset",
                save_orb=False,
            )

        mock_proj.assert_called_once()
        assert mock_proj.call_args.kwargs["restricted_space"] == [(2, 0.8)]
        assert list(result) == [2]


class TestParameterValidation:
    def test_invalid_entropy_ranking_raises(
        self, mol_mock: MagicMock, mf_mock: MagicMock
    ) -> None:
        with pytest.raises(ValueError, match="entropy_ranking"):
            select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_ncas=4,
                dmrg_nelecas=4,
                entropy_ranking="invalid",
                save_orb=False,
            )

    def test_invalid_projection_ranking_raises(
        self, mol_mock: MagicMock, mf_mock: MagicMock
    ) -> None:
        with pytest.raises(ValueError, match="projection_ranking"):
            select_aegiss(
                mol_mock,
                mf_mock,
                dmrg_ncas=4,
                dmrg_nelecas=4,
                projection_ranking="bad",
                save_orb=False,
            )
