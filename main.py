"""AEGISS pipeline: higher-order function composition."""

from __future__ import annotations

import os

from aegiss.active_space.ao_projection import select_by_ao_projection
from aegiss.active_space.entropy_screening import select_by_entropy
from aegiss.initial_guess import build_mol, run_scf
from aegiss.utils.banner import print_banner
from aegiss.utils.dump_info import save_orbitals, save_orbitals_from_entropies
from aegiss.utils.orbital_indices import get_subspaces_from_cas
from aegiss.wavefunction.dmrg import run_dmrg

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["SCRATCH_DIR"] = "./aegiss/examples/tmp"


if __name__ == "__main__":
    print_banner()
    atom = "./aegiss/examples/data/benzene.xyz"
    mol = build_mol(atom, basis="ccpvdz", symmetry=False, verbose=2)
    mf = run_scf(mol)
    dmrg_ncas, dmrg_nelecas = 20, 20

    active_orbs, virt_orbs = get_subspaces_from_cas(
        mf, ncas=dmrg_ncas, nelecas=dmrg_nelecas
    )
    _energy, soe1, _soe2, minfo = run_dmrg(
        mf,
        ncas=dmrg_ncas,
        nelecas=dmrg_nelecas,
        symm_type="SZ",
        init_bond_dim=10,
        verbosity=0,
        get_qmi=True,
    )

    screened_occ, screened_virt = select_by_entropy(
        occ_orbs=active_orbs,
        virt_orbs=virt_orbs,
        soe1=soe1,
        ranking="relative",
        fraction=0.1,
        fraction_type="r",
    )

    save_orbitals_from_entropies(
        mol,
        mf.mo_coeff,
        screened_occ,
        filename_prefix="benzene_entropy",
        folder="./aegiss/examples/orbs/",
    )

    save_orbitals_from_entropies(
        mol,
        mf.mo_coeff,
        screened_virt,
        filename_prefix="benzene_entropy",
        folder="./aegiss/examples/orbs/",
    )

    final_space = select_by_ao_projection(
        mf,
        ao_labels=["C 2pz"],
        contribution=0.5,
        projected_basis="minao",
        contribution_type="a",
        restricted_space=(screened_occ, screened_virt),
    )

    save_orbitals(
        mol,
        mf.mo_coeff,
        final_space,
        filename_prefix="benzene_aegiss",
        folder="./aegiss/examples/orbs/",
    )
