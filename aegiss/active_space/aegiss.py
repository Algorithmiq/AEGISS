"""Full AEGISS active space selection procedure."""

from __future__ import annotations

from typing import Any, Literal

import numpy as np

from aegiss.active_space.ao_projection import select_by_ao_projection
from aegiss.active_space.entropy_screening import select_by_entropy
from aegiss.utils.dump_info import save_orbitals, save_orbitals_from_entropies
from aegiss.utils.orbital_indices import get_subspaces_from_cas
from aegiss.wavefunction.dmrg import run_dmrg


def select_aegiss(
    mol: Any,
    mf: Any,
    dmrg_ncas: int,
    dmrg_nelecas: int,
    init_bond_dim: int = 10,
    symm_type: Literal["SZ", "SU2"] = "SZ",
    entropy_ranking: Literal["subset", "global"] = "subset",
    entropy_fraction: float = 0.1,
    entropy_fraction_type: Literal["r", "a"] = "r",
    projection_basis: str | None = "minao",
    projection_ao_labels: list[str] | None = None,
    projection_contribution: float = 0.5,
    projection_contribution_type: Literal["a", "r"] = "a",
    projection_top_n: int | None = None,
    projection_ranking: Literal["subset", "global"] = "subset",
    saving_folder: str = "./aegiss/examples/orbs/",
    saving_prefix: str = "benzene",
    verbosity: int = 0,
    *,
    save_orb: bool = True,
) -> np.ndarray:
    """Run the full AEGISS active space selection pipeline.

    Runs DMRG on the specified CAS, then filters the resulting orbitals
    through entropy screening and AO projection.  An orbital is included
    in the final active space when it has been selected by *both* the criterion.
    Therefore, only highly entangled orbitals with significant contribution on the specified AOs are kept.

    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecular geometry and basis set.
    mf : pyscf.scf.hf.SCF
        Converged mean-field object.
    dmrg_ncas : int
        Number of active orbitals for the DMRG calculation.
    dmrg_nelecas : int
        Number of active electrons for the DMRG calculation.
    init_bond_dim : int, default 10
        Initial DMRG bond dimension; grows automatically during the sweep schedule.
    symm_type : {'SZ', 'SU2'}, default 'SZ'
        Spin symmetry used by block2.  ``'SZ'`` is required for entropy
        calculations; ``'SU2'`` is faster but needs MPS conversion for entropies.
    entropy_ranking : {'subset', 'global'}, default 'subset'
        How to rank orbitals for entropy screening.  ``'subset'`` applies the
        threshold independently to occupied and virtual subspaces; ``'global'``
        ranks all active orbitals together.
    entropy_fraction : float, default 0.1
        Entropy threshold value (see *entropy_fraction_type*).
    entropy_fraction_type : {'r', 'a'}, default 'r'
        ``'r'``: keep orbitals whose S(1) >= ``entropy_fraction x max(S(1))``;
        ``'a'``: keep orbitals whose S(1) ≥ ``entropy_fraction`` (absolute).
    projection_basis : str or None, default 'minao'
        Basis set used for AO projection.  ``None`` uses the original
        molecular basis.
    projection_ao_labels : list of str, default ['C 2pz']
        AO labels to project onto.  Format: ``"Element AtomicOrbital"``,
        e.g. ``"C 2pz"``, ``"Fe 3d"``.
    projection_contribution : float, default 0.5
        Projection weight threshold (see *projection_contribution_type*).
    projection_contribution_type : {'a', 'r'}, default 'a'
        ``'a'``: absolute threshold; ``'r'``: relative to the maximum weight.
    projection_top_n : int or None, default None
        If set, keep the top-N orbitals per AO label instead of using a
        contribution threshold.
    projection_ranking : {'subset', 'global'}, default 'subset'
        ``'subset'``: run AO projection independently on the occupied and
        virtual entropy-screened subspaces; ``'global'``: run it on the
        combined set.
    saving_folder : str, default './aegiss/examples/orbs/'
        Directory where cube files are written when *save_orb* is ``True``.
    saving_prefix : str, default 'benzene'
        Prefix used for output cube file names.
    verbosity : int, default 0
        block2 output verbosity level.
    save_orb : bool, default True
        Write cube files for the entropy-screened and final active orbitals.

    Returns:
    -------
    active_indices : np.ndarray of int
        Sorted array of orbital indices forming the selected active space.
    """
    if entropy_ranking not in ("subset", "global"):
        raise ValueError(
            f"Invalid entropy_ranking {entropy_ranking!r}. Must be 'subset' or 'global'."
        )
    if projection_ranking not in ("subset", "global"):
        raise ValueError(
            f"Invalid projection_ranking {projection_ranking!r}. Must be 'subset' or 'global'."
        )

    if projection_ao_labels is None:
        projection_ao_labels = ["C 2pz"]

    active_orbs, virt_orbs = get_subspaces_from_cas(
        mf, ncas=dmrg_ncas, nelecas=dmrg_nelecas
    )
    _energy, soe1, _soe2, _minfo = run_dmrg(
        mf,
        ncas=dmrg_ncas,
        nelecas=dmrg_nelecas,
        symm_type=symm_type,
        init_bond_dim=init_bond_dim,
        verbosity=verbosity,
        get_qmi=True,
    )

    screened_occ, screened_virt = select_by_entropy(
        occ_orbs=active_orbs,
        virt_orbs=virt_orbs,
        soe1=soe1,
        ranking=entropy_ranking,
        fraction=entropy_fraction,
        fraction_type=entropy_fraction_type,
    )

    if save_orb:
        save_orbitals_from_entropies(
            mol,
            mf.mo_coeff,
            screened_occ,
            filename_prefix=saving_prefix + "_entropy",
            folder=saving_folder,
        )
        save_orbitals_from_entropies(
            mol,
            mf.mo_coeff,
            screened_virt,
            filename_prefix=saving_prefix + "_entropy",
            folder=saving_folder,
        )

    final_space: list[int] = []
    if projection_ranking == "subset":
        print("\nRanking orbitals for AO projection based on subset contribution.")
        for sub in (screened_occ, screened_virt):
            final_space.extend(
                select_by_ao_projection(
                    mf,
                    ao_labels=projection_ao_labels,
                    contribution=projection_contribution,
                    projected_basis=projection_basis,
                    contribution_type=projection_contribution_type,
                    top_n=projection_top_n,
                    restricted_space=sub,
                )
            )
    else:
        print("\nRanking orbitals for AO projection based on global contribution.")
        final_space.extend(
            select_by_ao_projection(
                mf,
                ao_labels=projection_ao_labels,
                contribution=projection_contribution,
                projected_basis=projection_basis,
                contribution_type=projection_contribution_type,
                top_n=projection_top_n,
                restricted_space=(screened_occ, screened_virt),
            )
        )

    active_indices = np.asarray(sorted(set(final_space)))

    if save_orb:
        save_orbitals(
            mol,
            mf.mo_coeff,
            active_indices,
            filename_prefix=saving_prefix + "_AEGISS",
            folder=saving_folder,
        )

    return active_indices
