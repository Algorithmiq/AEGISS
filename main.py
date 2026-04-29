"""AEGISS pipeline: higher-order function composition."""

from __future__ import annotations

import os

from aegiss.active_space.aegiss import select_aegiss
from aegiss.initial_guess import build_mol, run_scf
from aegiss.utils.banner import print_banner

os.environ["OMP_NUM_THREADS"] = "4"
os.environ["SCRATCH_DIR"] = "./aegiss/examples/tmp"


if __name__ == "__main__":
    print_banner()
    atom = "./aegiss/examples/data/benzene.xyz"
    mol = build_mol(atom, basis="ccpvdz", symmetry=False, verbose=2)
    mf = run_scf(mol)
    dmrg_ncas, dmrg_nelecas = 20, 20
    active_indices = select_aegiss(
        mol,
        mf,
        dmrg_ncas,
        dmrg_nelecas,
        init_bond_dim=10,
        verbosity=0,
        entropy_fraction=0.1,
        entropy_fraction_type="r",
        entropy_ranking="subset",
        projection_ao_labels=["C 2pz"],
        projection_contribution=0.5,
        projection_contribution_type="a",
        projection_basis="minao",
        projection_top_n=None,
        projection_ranking="global",
        saving_folder="./aegiss/examples/orbs/",
        saving_prefix="benzene_aegiss",
    )
