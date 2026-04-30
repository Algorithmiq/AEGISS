"""Full AEGISS active space selection procedure."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from aegiss.active_space.ao_projection import select_by_ao_projection
from aegiss.active_space.entropy_screening import select_by_entropy
from aegiss.utils.dump_info import save_orbitals, save_orbitals_from_entropies
from aegiss.utils.orbital_indices import get_subspaces_from_cas
from aegiss.utils.plotting import plot_pie_contribution, soe_heatmap, th_plots
from aegiss.wavefunction.dmrg import run_dmrg

if TYPE_CHECKING:
    from aegiss.config_dataclasses import (
        DMRGConfig,
        EntropyConfig,
        OutputConfig,
        ProjectionConfig,
    )

logger = logging.getLogger(__name__)


def _make_soe_plots(
    active_orbs: np.ndarray,
    virt_orbs: np.ndarray,
    soe1: np.ndarray,
    saving_folder: str,
    saving_prefix: str,
) -> None:
    """Generate and save single-orbital entropy diagnostic plots.

    Saves both an SOE heatmap and a threshold curve (TH plot) to *saving_folder*.

    Parameters
    ----------
    active_orbs : np.ndarray
        Indices of the occupied (active) orbitals in the CAS.
    virt_orbs : np.ndarray
        Indices of the virtual orbitals in the CAS.
    soe1 : np.ndarray
        Single-orbital entropies for all CAS orbitals, ordered as
        ``[active_orbs..., virt_orbs...]``.
    saving_folder : str
        Directory where the PDF files are written.
    saving_prefix : str
        Filename prefix; ``_SOE_heatmap.pdf`` and ``_TH_plot.pdf`` are appended.
    """
    tagged = [(orb_id, soe1[idx]) for idx, orb_id in enumerate(active_orbs)]
    tagged += [
        (orb_id, soe1[len(active_orbs) + idx]) for idx, orb_id in enumerate(virt_orbs)
    ]
    soe_heatmap(
        so1_entropies=tagged,
        n_occ=len(active_orbs),
        folder=saving_folder,
        filename=saving_prefix + "_SOE_heatmap",
    )
    th_plots(
        so1_entropies=tagged,
        occ_index=max(active_orbs) + 1,
        folder=saving_folder,
        filename=saving_prefix + "_TH_plot",
    )


def select_aegiss(
    mol: Any,
    mf: Any,
    dmrg_config: DMRGConfig,
    entropy_config: EntropyConfig,
    projection_config: ProjectionConfig,
    output_config: OutputConfig,
) -> np.ndarray:
    """Run the full AEGISS active space selection pipeline.

    Runs DMRG on the specified CAS, then filters the resulting orbitals
    through entropy screening and AO projection.  An orbital is included
    in the final active space when it has been selected by *both* criteria.
    Therefore, only highly entangled orbitals are kept. These orbitals
    must also have significant contribution on the specified AOs.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecular geometry and basis set.
    mf : pyscf.scf.hf.SCF
        Converged mean-field object.
    dmrg_config : DMRGConfig
        Configuration for the DMRG calculation (CAS size, bond dimension, symmetry).
    entropy_config : EntropyConfig
        Configuration for the entropy-based screening step.
    projection_config : ProjectionConfig
        Configuration for the AO projection-based screening step.
    output_config : OutputConfig
        Configuration for output options (file saving, plots).

    Returns:
    -------
    active_indices : np.ndarray of int
        Sorted array of orbital indices forming the selected active space.
    """
    active_orbs, virt_orbs = get_subspaces_from_cas(
        mf, ncas=dmrg_config.ncas, nelecas=dmrg_config.nelecas
    )
    _energy, soe1, _soe2, _minfo = run_dmrg(
        mf,
        ncas=dmrg_config.ncas,
        nelecas=dmrg_config.nelecas,
        symm_type=dmrg_config.symm_type,
        init_bond_dim=dmrg_config.init_bond_dim,
        verbosity=dmrg_config.verbosity,
    )

    if output_config.save_plots and soe1 is not None:
        _make_soe_plots(
            active_orbs,
            virt_orbs,
            soe1,
            output_config.saving_folder,
            output_config.saving_prefix,
        )

    screened_occ, screened_virt = select_by_entropy(
        occ_orbs=active_orbs,
        virt_orbs=virt_orbs,
        soe1=soe1,
        ranking=entropy_config.ranking,
        fraction=entropy_config.fraction,
        fraction_type=entropy_config.fraction_type,
    )

    if len(screened_occ) == 0 and len(screened_virt) == 0:
        logger.info(
            "No orbitals passed the entropy screening step. Consider adjusting the "
            "entropy_fraction or using 'global' ranking for a more inclusive selection."
        )
        return np.array([], dtype=int)

    if output_config.save_orb:
        for subspace in (screened_occ, screened_virt):
            save_orbitals_from_entropies(
                mol,
                mf.mo_coeff,
                subspace,
                filename_prefix=output_config.saving_prefix + "_entropy",
                folder=output_config.saving_folder,
            )

    base_path = (
        str(Path(output_config.saving_folder) / output_config.saving_prefix)
        if output_config.save_plots
        else None
    )

    final_space: list[int] = []
    if projection_config.ranking == "subset":
        logger.info("Ranking orbitals for AO projection based on subset contribution.")
        for suffix, sub in (("_occ", screened_occ), ("_virt", screened_virt)):
            if not sub:
                continue
            final_space.extend(
                select_by_ao_projection(
                    mf,
                    ao_labels=projection_config.ao_labels,
                    contribution=projection_config.contribution,
                    projected_basis=projection_config.basis,
                    contribution_type=projection_config.contribution_type,
                    top_n=projection_config.top_n,
                    restricted_space=sub,
                    save_path=base_path + suffix if base_path else None,
                )
            )
    else:
        logger.info("Ranking orbitals for AO projection based on global contribution.")
        final_space.extend(
            select_by_ao_projection(
                mf,
                ao_labels=projection_config.ao_labels,
                contribution=projection_config.contribution,
                projected_basis=projection_config.basis,
                contribution_type=projection_config.contribution_type,
                top_n=projection_config.top_n,
                restricted_space=(screened_occ, screened_virt),
                save_path=base_path,
            )
        )

    active_indices = np.asarray(sorted(set(final_space)))

    if output_config.save_orb:
        save_orbitals(
            mol,
            mf.mo_coeff,
            active_indices,
            filename_prefix=output_config.saving_prefix + "_AEGISS",
            folder=output_config.saving_folder,
        )

    if output_config.save_plots:
        plot_pie_contribution(
            mol,
            mf,
            ao_labels=projection_config.ao_labels,
            labels_names=projection_config.ao_labels,
            orbs=active_indices,
            folder=output_config.saving_folder,
        )

    return active_indices
