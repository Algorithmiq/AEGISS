"""Runnable example: AEGISS active space selection for the benzene pi system.

Builds benzene at the RHF/cc-pVDZ level, runs a DMRG calculation in the frontier
CAS(30, 30), then narrows that space with the two AEGISS criteria applied in
sequence: single-orbital entanglement entropy first, followed by projection onto
the carbon 2p_z manifold. Only orbitals selected by *both* criteria survive.

Run it from the repository root:

    python main.py

The tutorial notebooks under ``docs/tutorials/`` walk through the same workflow
step by step, with plots and commentary. Expect a few minutes of runtime -- the
DMRG step dominates.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from aegiss.active_space.aegiss import select_aegiss
from aegiss.config_dataclasses import (
    DMRGConfig,
    EntropyConfig,
    OutputConfig,
    ProjectionConfig,
)
from aegiss.initial_guess import build_mol, run_scf
from aegiss.utils.banner import print_banner
from aegiss.utils.logging import configure_logging

HERE = Path(__file__).resolve().parent
GEOMETRY = HERE / "docs" / "tutorials" / "data" / "benzene.xyz"
SCRATCH = HERE / "tmp" / "example_scratch"  # git-ignored
OUTPUT = HERE / "aegiss_output" / "benzene"  # git-ignored

log = logging.getLogger(__name__)


def main() -> None:
    """Run the full AEGISS pipeline on benzene and report the selected space."""
    configure_logging(level=logging.INFO)  # logging.DEBUG for per-orbital detail
    os.environ.setdefault("OMP_NUM_THREADS", "4")
    os.environ["SCRATCH_DIR"] = str(SCRATCH)
    print_banner()

    mol = build_mol(str(GEOMETRY), basis="ccpvdz", symmetry=False, verbose=2)
    mf = run_scf(mol)

    dmrg_config = DMRGConfig(
        ncas=30,
        nelecas=30,
        max_bond_dim=200,
        symm_type="SZ",
        verbosity=0,
    )
    entropy_config = EntropyConfig(
        fraction=0.1,
        fraction_type="r",
        ranking="global",
    )
    projection_config = ProjectionConfig(
        ao_labels=["C 2pz"],
        contribution=0.5,
        contribution_type="a",
        basis="minao",
        top_n=None,
        ranking="global",
    )
    output_config = OutputConfig(
        saving_folder=str(OUTPUT),
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
    log.info("Selected active space: %s", active_indices)


if __name__ == "__main__":
    main()
