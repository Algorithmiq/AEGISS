"""Tests for config dataclass validation."""

from __future__ import annotations

import pytest

from aegiss.config_dataclasses import (
    DMRGConfig,
    EntropyConfig,
    OutputConfig,
    ProjectionConfig,
)


class TestDMRGConfig:
    def test_valid_config(self) -> None:
        cfg = DMRGConfig(ncas=4, nelecas=4)
        assert cfg.ncas == 4
        assert cfg.nelecas == 4
        assert cfg.max_bond_dim == 500
        assert cfg.symm_type == "SZ"
        assert cfg.verbosity == 0
        assert cfg.unrestricted is None
        assert cfg.reorder == "noreorder"

    def test_reorder_fiedler_valid(self) -> None:
        cfg = DMRGConfig(ncas=4, nelecas=4, reorder="fiedler")
        assert cfg.reorder == "fiedler"

    def test_invalid_reorder_raises(self) -> None:
        with pytest.raises(ValueError, match="reorder"):
            DMRGConfig(ncas=4, nelecas=4, reorder="bogus")  # type: ignore[arg-type]

    def test_unrestricted_override_valid(self) -> None:
        cfg = DMRGConfig(ncas=4, nelecas=(3, 1), unrestricted=True)
        assert cfg.unrestricted is True

    def test_unrestricted_with_su2_raises(self) -> None:
        with pytest.raises(ValueError, match="symm_type='SZ'"):
            DMRGConfig(ncas=4, nelecas=4, unrestricted=True, symm_type="SU2")

    def test_zero_ncas_raises(self) -> None:
        with pytest.raises(ValueError, match="ncas"):
            DMRGConfig(ncas=0, nelecas=4)

    def test_negative_ncas_raises(self) -> None:
        with pytest.raises(ValueError, match="ncas"):
            DMRGConfig(ncas=-1, nelecas=4)

    def test_zero_nelecas_raises(self) -> None:
        with pytest.raises(ValueError, match="nelecas"):
            DMRGConfig(ncas=4, nelecas=0)

    def test_negative_nelecas_raises(self) -> None:
        with pytest.raises(ValueError, match="nelecas"):
            DMRGConfig(ncas=4, nelecas=-2)

    def test_nelecas_tuple_valid(self) -> None:
        cfg = DMRGConfig(ncas=10, nelecas=(6, 4))
        assert cfg.nelecas == (6, 4)

    def test_nelecas_tuple_closed_shell_valid(self) -> None:
        cfg = DMRGConfig(ncas=10, nelecas=(5, 5))
        assert cfg.nelecas == (5, 5)

    def test_nelecas_tuple_nalpha_lt_nbeta_raises(self) -> None:
        with pytest.raises(ValueError, match="nelecas"):
            DMRGConfig(ncas=10, nelecas=(2, 4))

    def test_nelecas_tuple_negative_nbeta_raises(self) -> None:
        with pytest.raises(ValueError, match="nelecas"):
            DMRGConfig(ncas=10, nelecas=(3, -1))

    def test_zero_max_bond_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="max_bond_dim"):
            DMRGConfig(ncas=4, nelecas=4, max_bond_dim=0)

    def test_negative_max_bond_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="max_bond_dim"):
            DMRGConfig(ncas=4, nelecas=4, max_bond_dim=-4)

    def test_non_multiple_of_4_max_bond_dim_raises(self) -> None:
        with pytest.raises(ValueError, match="max_bond_dim"):
            DMRGConfig(ncas=4, nelecas=4, max_bond_dim=250)

    def test_valid_max_bond_dim(self) -> None:
        cfg = DMRGConfig(ncas=4, nelecas=4, max_bond_dim=500)
        assert cfg.max_bond_dim == 500

    def test_invalid_symm_type_raises(self) -> None:
        with pytest.raises(ValueError, match="symm_type"):
            DMRGConfig(ncas=4, nelecas=4, symm_type="SU3")  # type: ignore[arg-type]

    def test_su2_symm_type_valid(self) -> None:
        cfg = DMRGConfig(ncas=4, nelecas=4, symm_type="SU2")
        assert cfg.symm_type == "SU2"

    def test_negative_verbosity_raises(self) -> None:
        with pytest.raises(ValueError, match="verbosity"):
            DMRGConfig(ncas=4, nelecas=4, verbosity=-1)

    def test_zero_verbosity_valid(self) -> None:
        cfg = DMRGConfig(ncas=4, nelecas=4, verbosity=0)
        assert cfg.verbosity == 0


class TestEntropyConfig:
    def test_valid_default_config(self) -> None:
        cfg = EntropyConfig()
        assert cfg.fraction == 0.1
        assert cfg.fraction_type == "r"
        assert cfg.ranking == "subset"

    def test_relative_fraction_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="fraction"):
            EntropyConfig(fraction=1.5, fraction_type="r")

    def test_relative_fraction_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="fraction"):
            EntropyConfig(fraction=-0.1, fraction_type="r")

    def test_absolute_fraction_above_one_valid(self) -> None:
        cfg = EntropyConfig(fraction=1.5, fraction_type="a")
        assert cfg.fraction == 1.5

    def test_absolute_fraction_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="fraction"):
            EntropyConfig(fraction=-0.1, fraction_type="a")

    def test_invalid_fraction_type_raises(self) -> None:
        with pytest.raises(ValueError, match="fraction_type"):
            EntropyConfig(fraction_type="x")  # type: ignore[arg-type]

    def test_invalid_empty_fraction_raises(self) -> None:
        with pytest.raises(ValueError, match="fraction"):
            EntropyConfig(fraction="")  # type: ignore[arg-type]

    def test_invalid_fraction_type_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="fraction_type"):
            EntropyConfig(fraction_type="")  # type: ignore[arg-type]

    def test_invalid_ranking_raises(self) -> None:
        with pytest.raises(ValueError, match="ranking"):
            EntropyConfig(ranking="invalid")  # type: ignore[arg-type]

    def test_invalid_empty_ranking_raises(self) -> None:
        with pytest.raises(ValueError, match="ranking"):
            EntropyConfig(ranking="")  # type: ignore[arg-type]

    def test_global_ranking_valid(self) -> None:
        cfg = EntropyConfig(ranking="global")
        assert cfg.ranking == "global"


class TestProjectionConfig:
    def test_valid_config_defaults(self) -> None:
        cfg = ProjectionConfig(ao_labels=["C 2pz"])
        assert cfg.ao_labels == ["C 2pz"]
        assert cfg.basis == "minao"
        assert cfg.contribution == 0.5
        assert cfg.contribution_type == "a"
        assert cfg.top_n is None
        assert cfg.ranking == "subset"

    def test_valid_config_all_fields(self) -> None:
        cfg = ProjectionConfig(
            ao_labels=["C 2pz", "Fe 3d"],
            basis="ccpvdz",
            contribution=[0.3, 0.5],
            contribution_type="r",
            top_n=5,
            ranking="global",
        )
        assert cfg.ao_labels == ["C 2pz", "Fe 3d"]
        assert cfg.basis == "ccpvdz"
        assert cfg.contribution == [0.3, 0.5]
        assert cfg.contribution_type == "r"
        assert cfg.top_n == 5
        assert cfg.ranking == "global"

    def test_ao_labels_non_string_raises(self) -> None:
        with pytest.raises(ValueError, match="ao_labels"):
            ProjectionConfig(ao_labels=["C 2pz", 123])  # type: ignore[list-item]

    def test_ao_labels_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="ao_labels"):
            ProjectionConfig(ao_labels=["C 2pz", ""])  # type: ignore[list-item]

    def test_ao_labels_non_string_inner_list_raises(self) -> None:
        with pytest.raises(ValueError, match="ao_labels"):
            ProjectionConfig(ao_labels=[["C 2pz", 123]])  # type: ignore[list-item]

    def test_ao_labels_empty_inner_list_raises(self) -> None:
        with pytest.raises(ValueError, match="ao_labels"):
            ProjectionConfig(ao_labels=[[]])  # type: ignore[list-item]

    def test_ao_labels_non_string_grouped_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="ao_labels"):
            ProjectionConfig(ao_labels=["C 2pz", ["C 2px", 123]])  # type: ignore[list-item]

    def test_ao_labels_empty_grouped_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="ao_labels"):
            ProjectionConfig(ao_labels=["C 2pz", []])  # type: ignore[list-item]

    def test_ao_labels_empty_inner_grouped_entry_raises(self) -> None:
        with pytest.raises(ValueError, match="ao_labels"):
            ProjectionConfig(ao_labels=["C 2pz", ["C 2px", []]])  # type: ignore[list-item]

    def test_ao_labels_non_list_raises(self) -> None:
        with pytest.raises(ValueError, match="ao_labels"):
            ProjectionConfig(ao_labels="C 2pz")  # type: ignore[arg-type]

    def test_empty_ao_labels_raises(self) -> None:
        with pytest.raises(ValueError, match="ao_labels"):
            ProjectionConfig(ao_labels=[])

    def test_invalid_contribution_type_raises(self) -> None:
        with pytest.raises(ValueError, match="contribution_type"):
            ProjectionConfig(ao_labels=["C 2pz"], contribution_type="x")  # type: ignore[arg-type]

    def test_invalid_ranking_raises(self) -> None:
        with pytest.raises(ValueError, match="ranking"):
            ProjectionConfig(ao_labels=["C 2pz"], ranking="bad")  # type: ignore[arg-type]

    def test_relative_contribution_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="contribution"):
            ProjectionConfig(
                ao_labels=["C 2pz"], contribution=1.5, contribution_type="r"
            )

    def test_relative_contribution_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="contribution"):
            ProjectionConfig(
                ao_labels=["C 2pz"], contribution=-0.1, contribution_type="r"
            )

    def test_absolute_contribution_above_one_valid(self) -> None:
        cfg = ProjectionConfig(
            ao_labels=["C 2pz"], contribution=1.5, contribution_type="a"
        )
        assert cfg.contribution == 1.5

    def test_absolute_contribution_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="contribution"):
            ProjectionConfig(
                ao_labels=["C 2pz"], contribution=-0.1, contribution_type="a"
            )

    def test_contribution_list_valid(self) -> None:
        cfg = ProjectionConfig(ao_labels=["C 2pz", "Fe 3d"], contribution=[0.3, 0.5])
        assert cfg.contribution == [0.3, 0.5]

    def test_contribution_empty_list_raises(self) -> None:
        with pytest.raises(ValueError, match="contribution"):
            ProjectionConfig(ao_labels=["C 2pz"], contribution=[])  # type: ignore[arg-type]

    def test_contribution_list_relative_above_one_raises(self) -> None:
        with pytest.raises(ValueError, match="contribution"):
            ProjectionConfig(
                ao_labels=["C 2pz", "Fe 3d"],
                contribution=[0.5, 1.5],
                contribution_type="r",
            )

    def test_contribution_list_absolute_negative_raises(self) -> None:
        with pytest.raises(ValueError, match="contribution"):
            ProjectionConfig(
                ao_labels=["C 2pz", "Fe 3d"],
                contribution=[0.5, -0.1],
                contribution_type="a",
            )

    def test_zero_top_n_raises(self) -> None:
        with pytest.raises(ValueError, match="top_n"):
            ProjectionConfig(ao_labels=["C 2pz"], top_n=0)

    def test_negative_top_n_raises(self) -> None:
        with pytest.raises(ValueError, match="top_n"):
            ProjectionConfig(ao_labels=["C 2pz"], top_n=-1)

    def test_positive_top_n_valid(self) -> None:
        cfg = ProjectionConfig(ao_labels=["C 2pz"], top_n=5)
        assert cfg.top_n == 5

    def test_multiple_ao_labels_valid(self) -> None:
        cfg = ProjectionConfig(ao_labels=["C 2pz", "C 2px", "C 2py"])
        assert len(cfg.ao_labels) == 3

    def test_grouped_ao_label_entry_valid(self) -> None:
        cfg = ProjectionConfig(ao_labels=["C 2pz", ["C 2px", "C 2py"]])
        assert len(cfg.ao_labels) == 2

    def test_all_grouped_ao_labels_valid(self) -> None:
        cfg = ProjectionConfig(ao_labels=[["C 2pz", "C 2px"], ["Fe 3d", "Fe 4d"]])
        assert len(cfg.ao_labels) == 2


class TestOutputConfig:
    def test_valid_default_config(self) -> None:
        cfg = OutputConfig()
        assert cfg.saving_folder == "./aegiss_output/"
        assert cfg.saving_prefix == "aegiss_selection"
        assert cfg.save_orb is True
        assert cfg.save_plots is False

    def test_empty_saving_folder_raises(self) -> None:
        with pytest.raises(ValueError, match="saving_folder"):
            OutputConfig(saving_folder="")

    def test_empty_saving_prefix_raises(self) -> None:
        with pytest.raises(ValueError, match="saving_prefix"):
            OutputConfig(saving_prefix="")
