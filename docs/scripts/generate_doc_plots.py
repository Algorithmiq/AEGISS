"""Generate the example diagnostic plots shown in the documentation.

Runs a small real calculation on benzene (RHF + a modest DMRG in a frontier CAS,
plus an AO projection) and renders the four utility plots as PNGs into
``docs/assets/images/``. Re-run this to refresh the images:

    python docs/scripts/generate_doc_plots.py

Requires the full AEGISS stack including ``block2`` (for the DMRG entropies).
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import matplotlib as mpl

mpl.use("Agg")

REPO = Path(__file__).resolve().parents[2]
IMG_DIR = REPO / "docs" / "assets" / "images"
SCRATCH = REPO / "tmp" / "_doc_plots_scratch"
SCRATCH.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ["SCRATCH_DIR"] = str(SCRATCH)

from aegiss.active_space.ao_projection import select_by_label  # noqa: E402
from aegiss.initial_guess import build_mol, run_scf  # noqa: E402
from aegiss.utils.plotting import (  # noqa: E402
    plot_ao_contribution,
    plot_pie_contribution,
    soe_heatmap,
    th_plots,
)
from aegiss.wavefunction.dmrg_subprocess import run_dmrg_subprocess  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(message)s")
log = logging.getLogger("gen_doc_plots")


def main() -> None:
    """Run the benzene calculation and write the four example PNGs."""
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    atom = str(REPO / "docs" / "tutorials" / "data" / "benzene.xyz")
    mol = build_mol(atom, basis="ccpvdz", symmetry=False, verbose=0)
    mf = run_scf(mol)

    # --- Frontier CAS for the DMRG entropy plots -------------------------------
    ncas, nelecas = 30, 30
    ncore = (mol.nelectron - nelecas) // 2
    n_occ_active = nelecas // 2
    occ_first_virt = ncore + n_occ_active  # global index of the first virtual

    log.info("Running DMRG on benzene CAS(%d, %d)...", nelecas, ncas)
    _energy, soe1 = run_dmrg_subprocess(
        mf, ncas=ncas, nelecas=nelecas, max_bond_dim=200, symm_type="SZ"
    )
    if soe1 is None:
        msg = "DMRG returned no single-orbital entropies; cannot render the plots."
        raise RuntimeError(msg)
    so1_entropies = [(ncore + i, float(s)) for i, s in enumerate(soe1)]

    # --- SOE heatmap -----------------------------------------------------------
    soe_heatmap(
        so1_entropies,
        n_occ=n_occ_active,
        folder=str(IMG_DIR),
        filename="example_soe_heatmap",
        fmt="png",
    )

    # --- Threshold (TH) plot ---------------------------------------------------
    th_plots(
        so1_entropies,
        folder=str(IMG_DIR),
        occ_index=occ_first_virt,
        filename="example_th_plot",
        fmt="png",
    )

    # --- AO contribution heatmap (real projection onto several AO manifolds) ---
    frontier = list(range(ncore, ncore + ncas))
    space = [(i, 0.0) for i in frontier]
    heatmap_labels = ["C 2pz", "C 2s", "H 1s"]
    ao_contributions = {}
    for j, label in enumerate(heatmap_labels):
        _, weights = select_by_label(
            label=label,
            projected_basis="minao",
            mf=mf,
            full_mo_coeff=mf.mo_coeff,
            space=space,
            contribution=0.0,
            contribution_type="a",
        )
        ao_contributions[j] = weights
    plot_ao_contribution(
        ao_contributions=ao_contributions,
        folder=str(IMG_DIR),
        filename="example_ao_contribution",
        ao_labels=dict(enumerate(heatmap_labels)),
        mo_labels=frontier,
        fmt="png",
    )

    # --- Pie composition per MO (valence sigma framework: genuine mixtures) -----
    pie_labels = ["C 2s", "C 2px", "C 2py", "C 2pz", "H 1s"]
    plot_pie_contribution(
        mol,
        mf,
        ao_labels=pie_labels,
        labels_names=pie_labels,
        orbs=list(range(7, 13)),
        folder=str(IMG_DIR),
        filename="example_pie_composition",
        fmt="png",
    )

    log.info("Done. PNGs written to %s", IMG_DIR)


if __name__ == "__main__":
    main()
