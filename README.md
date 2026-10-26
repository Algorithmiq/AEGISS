# AEGISS

**Atomic Orbital and Entropy based Guided Inference for Space Selection**

AEGISS automates active space selection for multireference quantum chemistry calculations.
It combines two complementary screening strategies:

- **Entropy screening** — ranks orbitals by their single-orbital entanglement entropy
  obtained from a DMRG wavefunction. Orbitals with high entanglement are strongly
  correlated and belong in the active space.
- **AO projection** — projects molecular orbitals onto a user-defined set of atomic
  orbitals (e.g. the C 2p_z shell of a conjugated system). Orbitals with large
  projection weights are chemically relevant to the target fragment.

The two criteria can be used independently or combined. In `select_aegiss` they are
applied sequentially — entropy screening first narrows the candidate set, then AO
projection refines it further — so an orbital must satisfy *both* criteria to appear
in the final active space. Both functions can also be called individually for custom
workflows.

---

## Installation

AEGISS requires Python ≥ 3.12 and uses [uv](https://github.com/astral-sh/uv) for
dependency management.

```bash
git clone https://github.com/Algorithmiq/AEGISS.git
cd AEGISS
uv sync --extra viz          # add --all-groups for the test, docs and notebook tools
source .venv/bin/activate
```

`block2` is fetched from its own package index declared in `pyproject.toml` — `uv sync`
handles it automatically. The `viz` extra is only needed for
[MolTagger](docs/moltagger.md), the interactive tool for picking atoms and generating
`ao_labels`.

See [docs/installation.md](docs/installation.md) for the environment variables that
control threads, memory and scratch space, and for logging setup.

---

## Quick start — benzene π system

```python notest
from aegiss.active_space.aegiss import select_aegiss
from aegiss.config_dataclasses import DMRGConfig, EntropyConfig, OutputConfig, ProjectionConfig
from aegiss.initial_guess import build_mol, run_scf

mol = build_mol("docs/tutorials/data/benzene.xyz", basis="ccpvdz", symmetry=False)
mf = run_scf(mol, method="RHF")

active_indices = select_aegiss(
    mol,
    mf,
    dmrg_config=DMRGConfig(ncas=30, nelecas=30, max_bond_dim=200, symm_type="SZ"),
    entropy_config=EntropyConfig(fraction=0.1, fraction_type="r", ranking="global"),
    projection_config=ProjectionConfig(
        ao_labels=["C 2pz"], contribution=0.5, contribution_type="a", ranking="global"
    ),
    output_config=OutputConfig(saving_prefix="benzene_aegiss", save_orb=False),
)
```

`select_aegiss` runs the full pipeline — DMRG in a CAS(30, 30) window around the Fermi
level → entropy screening → projection onto the carbon 2p_z orbitals — and returns the
0-based indices of the selected orbitals. [`main.py`](main.py) is the same example as a
runnable script; expect a few minutes of runtime, dominated by the DMRG step.

AEGISS can also start from localized or natural orbitals (see `localize` in the
[localization API](docs/api/localization.md)). Keep in mind that the DMRG window is
taken by orbital index, which is a Fermi-level window only for canonical orbitals.

---

## Documentation

The documentation sources live in [`docs/`](docs/index.md) and are built with MkDocs
(see [docs/README.md](docs/README.md)):

- [Methodology](docs/index.md) — the six steps of the AEGISS workflow.
- [Tutorials](docs/tutorials/index.md) — Jupyter notebooks covering SCF and
  localization, DMRG and entropies, AO projection, the full pipeline, and UHF references.
- [Orbital index base](docs/orbital_indexing.md) — printing and passing orbital indices
  0- or 1-based (`aegiss config set-base`, `AEGISS_ORB_BASE`).
- [Unrestricted (UHF) references](docs/uhf.md) — spin-summed vs. spin-resolved entropies.
- [MolTagger](docs/moltagger.md) — interactive atom tagging.
- [API reference](docs/api/index.md) — generated from the docstrings.

---

## Development

```bash
just build --all-groups --all-extras   # or: uv sync --all-groups --all-extras
uv run prek install
just test -m "not slow"
uv run mkdocs serve                    # docs preview at http://127.0.0.1:8000
```

`just` with no arguments lists every recipe.

Code style is enforced by **ruff** and **ty**; both run as git hooks (via
[prek](https://github.com/j178/prek)) and in CI on every pull request. See [CONTRIBUTING.md](CONTRIBUTING.md) for the full workflow.

---

## Contributing

Contributions are welcome. Please read [CONTRIBUTING.md](CONTRIBUTING.md) for the
development setup, branch and commit conventions, and testing requirements, and note
that all contributions require agreement to the
[Contributor License Agreement](CLA.md). All interactions in this repository are
covered by our [Code of Conduct](CODE_OF_CONDUCT.md).

Bugs and feature requests go through the
[issue templates](https://github.com/Algorithmiq/AEGISS/issues/new/choose). Please do
not open public issues for security problems; see [SECURITY.md](SECURITY.md).

---

## Citation

If you use AEGISS in your work, please cite the paper:

```bibtex
@misc{tarocco2025aegissatomicorbital,
      title={AEGISS -- Atomic orbital and Entropy-based Guided Inference for Space Selection -- A novel semi-automated active space selection workflow for quantum chemistry and quantum computing applications},
      author={Fabio Tarocco and Pi A. B. Haase and Fabijan Pavošević and Vijay Krishna and Leonardo Guidoni and Stefan Knecht and Martina Stella},
      year={2025},
      eprint={2508.10671},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2508.10671},
}
```

Machine-readable metadata for both the software and the paper is in
[CITATION.cff](CITATION.cff) — GitHub's "Cite this repository" widget reads it
directly.
