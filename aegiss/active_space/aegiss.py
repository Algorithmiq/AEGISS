"""Full AEGISS active space selection procedure."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from aegiss.active_space.ao_projection import select_by_ao_projection
from aegiss.active_space.entropy_screening import (
    select_by_entropy,
    select_by_spin_entropy,
)
from aegiss.utils.dump import save_orbitals, save_orbitals_from_entropies
from aegiss.utils.orbital_indices import get_subspaces_from_cas
from aegiss.utils.plotting import (
    _make_soe_plots,
    plot_pie_contribution,
)
from aegiss.wavefunction.dmrg_subprocess import run_dmrg_subprocess

if TYPE_CHECKING:
    from aegiss.config_dataclasses import (
        DMRGConfig,
        EntropyConfig,
        OutputConfig,
        ProjectionConfig,
    )

logger = logging.getLogger(__name__)

# A per-spin entropy window: (occupied, virtual) index/entropy pairs.
_OccVirt = tuple[list[tuple[int, float]], list[tuple[int, float]]]
# Entropy screening output: one window (spin-summed) or one window per spin
# channel (spin-resolved).
_Screened = _OccVirt | dict[str, _OccVirt]
# A restricted_space handed to select_by_ao_projection: a single occupied/virtual
# list (spin-summed subset), a whole (occ, virt) window, or a per-spin window.
_RestrictedSpace = list[tuple[int, float]] | _OccVirt | dict[str, _OccVirt]
# Maps a spin-channel name to its row in an unrestricted ``mo_coeff``.
_CHANNEL_SPIN = {"alpha": 0, "beta": 1}


def _screen_by_entropy(
    entropy_config: EntropyConfig,
    active_orbs: np.ndarray,
    virt_orbs: np.ndarray,
    soe1: np.ndarray | None,
    *,
    spin_resolved: bool,
) -> _Screened:
    """Screen the DMRG active window by single-orbital entropy.

    Dispatches to the spin-summed or spin-resolved screener according to
    *spin_resolved* (from :meth:`DMRGConfig.spin_resolved`).

    Args:
        entropy_config (EntropyConfig): Screening fraction / ranking settings.
        active_orbs (np.ndarray): Occupied indices of the DMRG active window.
        virt_orbs (np.ndarray): Virtual indices of the DMRG active window.
        soe1 (np.ndarray or None): Single-orbital entropies -- 1-D (spin-summed)
            or ``(2, ncas)`` (spin-resolved).
        spin_resolved (bool): Whether to screen the alpha/beta channels apart.

    Returns:
        screened (_Screened): A spin-summed ``(occ, virt)`` window, or a
            spin-resolved ``{"alpha": (occ, virt), "beta": (occ, virt)}`` mapping.
    """
    if spin_resolved:
        return select_by_spin_entropy(
            occ_orbs=active_orbs,
            virt_orbs=virt_orbs,
            soe1=soe1,
            fraction=entropy_config.fraction,
            fraction_type=entropy_config.fraction_type,
            ranking=entropy_config.ranking,
        )
    return select_by_entropy(
        occ_orbs=active_orbs,
        virt_orbs=virt_orbs,
        soe1=soe1,
        ranking=entropy_config.ranking,
        fraction=entropy_config.fraction,
        fraction_type=entropy_config.fraction_type,
    )


def _screening_empty(screened: _Screened) -> bool:
    """Return ``True`` if no orbital survived entropy screening (any channel)."""
    if isinstance(screened, dict):
        return all(not occ and not virt for occ, virt in screened.values())
    occ, virt = screened
    return not occ and not virt


def _ao_projection_passes(
    screened: _Screened,
    ranking: str,
) -> list[tuple[str, _RestrictedSpace]]:
    """Build the ``(save_suffix, restricted_space)`` AO-projection passes.

    For ``'global'`` ranking the occupied and virtual windows are projected in a
    single pass; for ``'subset'`` they are projected separately. Spin-resolved
    input keeps its ``{"alpha", "beta"}`` structure so each pass projects both
    channels on their own window (each half wrapped as an ``(occ, virt)`` window
    with the other half empty, to keep a uniform per-channel value type).

    Args:
        screened (_Screened): Entropy-screening output.
        ranking (str): ``'global'`` or ``'subset'``.

    Returns:
        passes (list of tuple[str, _RestrictedSpace]): Save-path suffix and the
            *restricted_space* to hand to :func:`select_by_ao_projection`.
    """
    if ranking != "subset":
        return [("", screened)]

    passes: list[tuple[str, _RestrictedSpace]] = []
    for pos, suffix in ((0, "_occ"), (1, "_virt")):
        space: _RestrictedSpace
        if isinstance(screened, dict):
            space = {ch: _half_window(win, pos) for ch, win in screened.items()}
            nonempty = any(win[pos] for win in screened.values())
        else:
            space = screened[pos]
            nonempty = bool(space)
        if nonempty:
            passes.append((suffix, space))
    return passes


def _half_window(window: _OccVirt, pos: int) -> _OccVirt:
    """Keep only the occupied (``pos=0``) or virtual (``pos=1``) half of a window."""
    return (window[0], []) if pos == 0 else ([], window[1])


def _collect_active_indices(
    mf: Any,
    projection_config: ProjectionConfig,
    passes: list[tuple[str, _RestrictedSpace]],
    base_path: str | None,
) -> np.ndarray | dict[str, np.ndarray]:
    """Run the AO-projection passes and merge their selected indices.

    Indices from different passes (e.g. the occupied and virtual subsets) are
    unioned *within* each spin channel. For an unrestricted mean-field the alpha
    and beta selections are returned as a split ``{"alpha", "beta"}`` mapping and
    are deliberately **not** unioned across channels: index *k* refers to a
    different MO in each spin channel, so a single spatial active space is not
    defined here (the caller consumes the split directly).

    Args:
        mf (pyscf.scf.hf.SCF): Mean-field object.
        projection_config (ProjectionConfig): AO-projection settings.
        passes (list of tuple[str, _RestrictedSpace]): Output of
            :func:`_ao_projection_passes`.
        base_path (str or None): Plot path stem, or ``None`` for no plotting.

    Returns:
        active_indices (np.ndarray, or dict[str, np.ndarray]): Sorted spatial
            indices (restricted), or a split alpha/beta mapping (unrestricted).
    """
    channels: dict[str, set[int]] = {"alpha": set(), "beta": set()}
    flat: set[int] = set()
    is_split = False
    for suffix, restricted_space in passes:
        save_path = base_path + suffix if base_path else None
        result = select_by_ao_projection(
            mf,
            ao_labels=projection_config.ao_labels,
            contribution=projection_config.contribution,
            projected_basis=projection_config.basis,
            contribution_type=projection_config.contribution_type,
            top_n=projection_config.top_n,
            restricted_space=restricted_space,
            save_path=save_path,
        )
        if isinstance(result, dict):
            is_split = True
            for channel, indices in result.items():
                channels[channel].update(int(i) for i in indices)
        else:
            flat.update(int(i) for i in result)

    if is_split:
        return {
            channel: np.asarray(sorted(indices))
            for channel, indices in channels.items()
        }
    return np.asarray(sorted(flat))


def _maybe_plot_soe(
    output_config: OutputConfig,
    active_orbs: np.ndarray,
    virt_orbs: np.ndarray,
    soe1: np.ndarray | None,
) -> None:
    """Write the single-orbital-entropy plot when enabled and spin-summed.

    Spin-resolved ``(2, ncas)`` entropies have no plotter yet, so that case is
    skipped with a log note rather than raising.

    Args:
        output_config (OutputConfig): Output settings (gates on ``save_plots``).
        active_orbs (np.ndarray): Occupied indices of the DMRG active window.
        virt_orbs (np.ndarray): Virtual indices of the DMRG active window.
        soe1 (np.ndarray or None): Single-orbital entropies.
    """
    if not output_config.save_plots or soe1 is None:
        return
    if np.ndim(soe1) != 1:
        logger.info(
            "Spin-resolved single-orbital entropies are not plotted; skipping "
            "the SOE plot."
        )
        return
    _make_soe_plots(
        active_orbs,
        virt_orbs,
        soe1,
        output_config.saving_folder,
        output_config.saving_prefix,
    )


def _save_entropy_orbitals(
    mol: Any,
    mf: Any,
    screened: _Screened,
    output_config: OutputConfig,
) -> None:
    """Dump cube files for the entropy-screened orbitals of every window.

    For an unrestricted reference each channel is saved from its own spin block
    of ``mf.mo_coeff`` with a channel-tagged prefix.

    Args:
        mol (pyscf.gto.Mole): Molecule.
        mf (pyscf.scf.hf.SCF): Mean-field object.
        screened (_Screened): Entropy-screening output.
        output_config (OutputConfig): Output settings.
    """
    mo_coeff = np.asarray(mf.mo_coeff)
    if isinstance(screened, dict):
        for channel, window in screened.items():
            coeff = mo_coeff[_CHANNEL_SPIN[channel]]
            for subspace in window:
                save_orbitals_from_entropies(
                    mol,
                    coeff,
                    subspace,
                    filename_prefix=f"{output_config.saving_prefix}_entropy_{channel}",
                    folder=output_config.saving_folder,
                )
        return
    for subspace in screened:
        save_orbitals_from_entropies(
            mol,
            mo_coeff,
            subspace,
            filename_prefix=output_config.saving_prefix + "_entropy",
            folder=output_config.saving_folder,
        )


def _save_active_orbitals(
    mol: Any,
    mf: Any,
    active_indices: np.ndarray | dict[str, np.ndarray],
    output_config: OutputConfig,
) -> None:
    """Dump cube files for the final AEGISS active orbitals.

    Args:
        mol (pyscf.gto.Mole): Molecule.
        mf (pyscf.scf.hf.SCF): Mean-field object.
        active_indices (np.ndarray, or dict[str, np.ndarray]): Selected indices;
            a split alpha/beta mapping for an unrestricted reference.
        output_config (OutputConfig): Output settings.
    """
    mo_coeff = np.asarray(mf.mo_coeff)
    if isinstance(active_indices, dict):
        for channel, indices in active_indices.items():
            save_orbitals(
                mol,
                mo_coeff[_CHANNEL_SPIN[channel]],
                indices,
                filename_prefix=f"{output_config.saving_prefix}_AEGISS_{channel}",
                folder=output_config.saving_folder,
            )
        return
    save_orbitals(
        mol,
        mo_coeff,
        active_indices,
        filename_prefix=output_config.saving_prefix + "_AEGISS",
        folder=output_config.saving_folder,
    )


def _plot_active(
    mol: Any,
    mf: Any,
    active_indices: np.ndarray | dict[str, np.ndarray],
    projection_config: ProjectionConfig,
    output_config: OutputConfig,
) -> None:
    """Plot the AO pie contribution for the final active orbitals.

    The pie plotter takes a single flat orbital list, so the split alpha/beta
    selection of an unrestricted reference is skipped with a log note.

    Args:
        mol (pyscf.gto.Mole): Molecule.
        mf (pyscf.scf.hf.SCF): Mean-field object.
        active_indices (np.ndarray, or dict[str, np.ndarray]): Selected indices.
        projection_config (ProjectionConfig): AO-projection settings.
        output_config (OutputConfig): Output settings.
    """
    if isinstance(active_indices, dict):
        logger.info(
            "AO pie-contribution plotting is not available for the split "
            "alpha/beta selection; skipping."
        )
        return
    plot_pie_contribution(
        mol,
        mf,
        ao_labels=projection_config.ao_labels,
        labels_names=projection_config.ao_labels,
        orbs=active_indices,
        folder=output_config.saving_folder,
    )


def select_aegiss(
    mol: Any,
    mf: Any,
    dmrg_config: DMRGConfig,
    entropy_config: EntropyConfig,
    projection_config: ProjectionConfig,
    output_config: OutputConfig,
) -> np.ndarray | dict[str, np.ndarray]:
    """Run the full AEGISS active space selection pipeline.

    Runs DMRG on the specified CAS, then filters the resulting orbitals
    through entropy screening and AO projection.  An orbital is included
    in the final active space when it has been selected by *both* criteria.
    Therefore, only highly entangled orbitals are kept. These orbitals
    must also have significant contribution on the specified AOs.

    For an unrestricted (UHF) mean-field the whole pipeline runs per spin
    channel: when ``dmrg_config.spin_soe`` resolves to spin-resolved the alpha
    and beta channels are screened on their own entropy windows, otherwise they
    share one spin-summed window; either way the AO projection is done per spin.
    The alpha and beta selections are returned split and are **not** reconciled
    into a single spatial active space -- index *k* denotes a different MO in
    each channel, so no common spatial index set is defined here.

    Args:
        mol (pyscf.gto.Mole): Molecular geometry and basis set.
        mf (pyscf.scf.hf.SCF): Converged mean-field object.
        dmrg_config (DMRGConfig): Configuration for the DMRG calculation (CAS size, bond dimension, symmetry).
        entropy_config (EntropyConfig): Configuration for the entropy-based screening step.
        projection_config (ProjectionConfig): Configuration for the AO projection-based screening step.
        output_config (OutputConfig): Configuration for output options (file saving, plots).

    Returns:
        active_indices (np.ndarray of int, or dict[str, np.ndarray]): Sorted
            orbital indices of the selected active space. A single array for a
            restricted mean-field; a split ``{"alpha": ..., "beta": ...}``
            mapping (kept separate, not reconciled) for an unrestricted one.
    """
    active_orbs, virt_orbs = get_subspaces_from_cas(
        mf, ncas=dmrg_config.ncas, nelecas=dmrg_config.nelecas
    )
    spin_resolved = dmrg_config.spin_resolved()
    _energy, soe1 = run_dmrg_subprocess(
        mf,
        ncas=dmrg_config.ncas,
        nelecas=dmrg_config.nelecas,
        symm_type=dmrg_config.symm_type,
        max_bond_dim=dmrg_config.max_bond_dim,
        verbosity=dmrg_config.verbosity,
        unrestricted=dmrg_config.unrestricted,
        reorder=dmrg_config.reorder,
        spin_resolved=spin_resolved,
    )
    _maybe_plot_soe(output_config, active_orbs, virt_orbs, soe1)

    screened = _screen_by_entropy(
        entropy_config,
        active_orbs,
        virt_orbs,
        soe1,
        spin_resolved=spin_resolved,
    )
    if _screening_empty(screened):
        logger.info(
            "No orbitals passed the entropy screening step. Consider adjusting the "
            "entropy_fraction or using 'global' ranking for a more inclusive selection."
        )
        return np.array([], dtype=int)

    if output_config.save_orb:
        _save_entropy_orbitals(mol, mf, screened, output_config)

    base_path = (
        str(Path(output_config.saving_folder) / output_config.saving_prefix)
        if output_config.save_plots
        else None
    )
    logger.info(
        "Ranking orbitals for AO projection based on %s contribution.",
        projection_config.ranking,
    )
    passes = _ao_projection_passes(screened, projection_config.ranking)
    active_indices = _collect_active_indices(mf, projection_config, passes, base_path)

    if output_config.save_orb:
        _save_active_orbitals(mol, mf, active_indices, output_config)
    if output_config.save_plots:
        _plot_active(mol, mf, active_indices, projection_config, output_config)

    return active_indices
