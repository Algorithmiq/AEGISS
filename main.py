"""AEGISS pipeline: higher-order function composition."""

from __future__ import annotations

import os

from aegiss.active_space.aegiss import select_aegiss
from aegiss.config_dataclasses import (
    DMRGConfig,
    EntropyConfig,
    OutputConfig,
    ProjectionConfig,
)
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

    dmrg_config = DMRGConfig(
        ncas=dmrg_ncas,
        nelecas=dmrg_nelecas,
        init_bond_dim=10,
        symm_type="SZ",
        verbosity=0,
    )

    entropy_config = EntropyConfig(
        fraction=0.1,
        fraction_type="r",
        ranking="subset",
    )
    projection_config = ProjectionConfig(
        ao_labels=["C 2pz", "C 2px", "C 2py"],
        contribution=0.5,
        contribution_type="a",
        basis="minao",
        top_n=None,
        ranking="global",
    )
    output_config = OutputConfig(
        saving_folder="./aegiss/examples/orbs/",
        saving_prefix="benzene_aegiss",
        save_orb=False,
        save_plots=False,
    )
    active_indices = select_aegiss(
        mol,
        mf,
        dmrg_config=dmrg_config,
        entropy_config=entropy_config,
        projection_config=projection_config,
        output_config=output_config,
    )
