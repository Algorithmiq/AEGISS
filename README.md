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

## Table of contents

- [Installation](#installation)
  - [MolTagger: Interactive Tagging Tool](#moltagger-interactive-tagging-tool)
- [Environment variables](#environment-variables)
- [Logging](#logging)
- [Tutorial — benzene π system](#tutorial--benzene-π-system)
- [API reference](#api-reference)
  - [select_aegiss](#select_aegiss)
  - [Configuration dataclasses](#configuration-dataclasses)
  - [build_mol](#build_mol)
  - [run_scf](#run_scf)
  - [run_dmrg](#run_dmrg)
  - [select_by_entropy](#select_by_entropy)
  - [select_by_ao_projection](#select_by_ao_projection)
  - [localize](#localize)
  - [save_orbitals](#save_orbitals)
- [Development](#development)

---

## Installation

AEGISS requires Python ≥ 3.12 and uses [uv](https://github.com/astral-sh/uv) for
dependency management.

```bash
git clone <repo-url>
cd AEGISS
uv venv
uv sync
source .venv/bin/activate
```

`block2` is fetched from its own package index and is declared in `pyproject.toml`
— `uv sync` handles it automatically.

### `MolTagger`: Interactive Tagging Tool

An interactive 3D molecular viewer for selecting atoms and generating
PySCF-compatible `ao_labels` strings ready to paste into `ProjectionConfig`.
MolTagger requires the `viz` optional extra:

```bash
uv sync --extra viz
```

See the [MolTagger README](aegiss/utils/mol_tagger/README.md) for full installation and usage.

### `Extras` and `Groups`
```bash
uv sync --all-extras --all-groups
```
You'll have:
- `--all-extras` picks up `viz`
- `--all-groups` picks up `test`, `docs`,`dev`, and `interactive`.
---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OMP_NUM_THREADS` | `4` | OpenMP threads used by block2 |
| `SCRATCH_DIR` | `./tmp` | Directory for DMRG scratch files (MPS/MPO) |
| `MEM` | `4` | Memory budget in **GiB** for block2's internal stack (`stack_mem = MEM × 2³⁰` bytes). Set to roughly the free RAM available to the process. |
| `PYSCF_MAX_MEMORY` | `4000` | PySCF memory budget in **MB** (`mol.max_memory`). Controls how much RAM PySCF may use for integral storage and intermediates. |

Set them before running any calculation:

```python
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["SCRATCH_DIR"] = "/scratch/my_job"
os.environ["MEM"] = "32"            # 32 GiB block2 stack
os.environ["PYSCF_MAX_MEMORY"] = "16000"  # 16 GB for PySCF
```

---

## Logging

AEGISS uses Python's standard `logging` module. The library never configures
handlers or formatters — it only emits records through named loggers under the
`aegiss.*` hierarchy. Callers control where and how output appears.

**Enable output in a script:**

```python
import logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
```

Use `logging.DEBUG` to also see per-orbital detail (contribution weights,
individual S(1) values):

```python
logging.basicConfig(level=logging.DEBUG, format="%(message)s")
```

**Silence the library selectively (e.g. in a notebook):**

```python
logging.getLogger("aegiss").setLevel(logging.WARNING)
```

**structlog users** — configure structlog's stdlib integration before importing
AEGISS and the library will emit through it automatically:

```python
import structlog
structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
```

---

## Tutorial — benzene π system

This walkthrough reproduces the `main.py` example and explains each step.

### 1. Build the molecule

```python
from aegiss.initial_guess import build_mol, run_scf

mol = build_mol(
    "aegiss/examples/data/benzene.xyz",
    basis="ccpvdz",
    symmetry=False,
    verbose=2,
)
```

`build_mol` wraps `pyscf.gto.Mole`. Any PySCF `Mole` attribute can be passed as a
keyword argument (`charge`, `spin`, `verbose`, …).

### 2. Run a mean-field calculation

```python
mf = run_scf(mol, method="RHF")
```

### 3. Configure and run the full pipeline

```python
from aegiss.active_space.aegiss import select_aegiss
from aegiss.config_dataclasses import DMRGConfig, EntropyConfig, ProjectionConfig, OutputConfig

dmrg_config = DMRGConfig(ncas=20, nelecas=20, init_bond_dim=10, symm_type="SZ")

entropy_config = EntropyConfig(
    fraction=0.1,
    fraction_type="r",
    ranking="global",
)

projection_config = ProjectionConfig(
    ao_labels=["C 2pz"],
    contribution=0.5,
    contribution_type="a",
)

output_config = OutputConfig(
    saving_folder="./orbs/",
    saving_prefix="benzene_aegiss",
    save_orb=True,
    save_plots=False,
)

active_indices = select_aegiss(
    mol, mf,
    dmrg_config=dmrg_config,
    entropy_config=entropy_config,
    projection_config=projection_config,
    output_config=output_config,
)
```

`select_aegiss` runs the full pipeline — DMRG → entropy screening → AO projection
— and optionally writes orbital cube files and diagnostic plots.

Here `ranking="global"` means occupied and virtual orbitals are ranked together by
entropy before the 10 % threshold is applied, so only the most entangled orbitals
across the full CAS survive. Narrowing `ao_labels` to `["C 2pz"]` targets the
benzene π manifold exclusively.

For custom workflows that call individual steps directly, see `run_dmrg`,
`select_by_entropy`, and `select_by_ao_projection` in the API reference below.

Further step-by-step tutorials covering SCF & localization, DMRG, AO projection, and
the full pipeline are available as Jupyter notebooks in
[aegiss/examples/tutorials/](aegiss/examples/tutorials/README.md).

---

## API reference

### `select_aegiss`

```python
from aegiss.active_space.aegiss import select_aegiss

active_indices = select_aegiss(
    mol, mf,
    dmrg_config=dmrg_config,
    entropy_config=entropy_config,
    projection_config=projection_config,
    output_config=output_config,
)
```

High-level entry point that runs the full pipeline in a single call: DMRG on the
specified CAS → entropy screening → AO projection. An orbital is kept when it passes
*both* criteria — AO projection is restricted to the entropy-screened candidate set.
If entropy screening returns no orbitals, the function returns an empty array early.

| Parameter | Type | Description |
|---|---|---|
| `mol` | `pyscf.gto.Mole` | Molecular geometry and basis set |
| `mf` | SCF object | Converged mean-field |
| `dmrg_config` | `DMRGConfig` | DMRG calculation settings |
| `entropy_config` | `EntropyConfig` | Entropy screening settings |
| `projection_config` | `ProjectionConfig` | AO projection settings |
| `output_config` | `OutputConfig` | Output and saving settings |

Returns `np.ndarray` of sorted orbital indices. See
[Configuration dataclasses](#configuration-dataclasses) for field-level documentation.

---

### Configuration dataclasses

All four dataclasses are frozen (immutable) and validate their inputs on
construction. They are exported from the top-level package:

```python
from aegiss.config_dataclasses import DMRGConfig, EntropyConfig, ProjectionConfig, OutputConfig
```

#### `DMRGConfig`

```python
DMRGConfig(ncas, nelecas, init_bond_dim=10, symm_type="SZ", verbosity=0)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `ncas` | `int` | — | Number of active orbitals |
| `nelecas` | `int` or `tuple[int, int]` | — | Number of active electrons. An integer total count or an explicit `(nalpha, nbeta)` pair — same convention as PySCF's CASCI. Odd integers are treated as high-spin doublets. |
| `init_bond_dim` | `int` | `10` | Initial DMRG bond dimension; grows automatically during the sweep schedule |
| `symm_type` | `"SZ"` or `"SU2"` | `"SZ"` | Spin symmetry: `"SZ"` is required for entropy calculations; `"SU2"` is faster |
| `verbosity` | `int` | `0` | block2 output verbosity |

Open-shell example: `DMRGConfig(ncas=10, nelecas=(5, 4))` sets up a doublet CAS with 5α and 4β electrons.

#### `EntropyConfig`

```python
EntropyConfig(fraction=0.1, fraction_type="r", ranking="subset")
```

| Field | Type | Default | Description |
|---|---|---|---|
| `fraction` | `float` | `0.1` | Entropy threshold value (see `fraction_type`) |
| `fraction_type` | `"r"` or `"a"` | `"r"` | `"r"`: keep S(1) ≥ `fraction × max(S(1))`; `"a"`: absolute threshold |
| `ranking` | `"subset"` or `"global"` | `"subset"` | `"subset"`: threshold occ and virt independently; `"global"`: rank all orbitals together |

#### `ProjectionConfig`

```python
ProjectionConfig(ao_labels, basis="minao", contribution=0.5, contribution_type="a", top_n=None, ranking="subset")
```

| Field | Type | Default | Description |
|---|---|---|---|
| `ao_labels` | `list[str \| list[str]]` | — | AO labels to project onto, e.g. `["C 2pz"]`. Each entry is a single label string or a list of strings treated as one group. See [AO label format](#ao-label-format) |
| `basis` | `str` or `None` | `"minao"` | Basis for projection; `None` uses the original molecular basis |
| `contribution` | `float` or `list[float]` | `0.5` | Projection weight threshold. A single float applies to all labels; a list assigns per-label thresholds |
| `contribution_type` | `"a"` or `"r"` | `"a"` | `"a"`: absolute threshold; `"r"`: relative to the maximum weight |
| `top_n` | `int` or `None` | `None` | Keep top-N orbitals per label instead of using a threshold |
| `ranking` | `"subset"` or `"global"` | `"subset"` | `"subset"`: project occ and virt subspaces separately; `"global"`: project the combined set |

#### `OutputConfig`

```python
OutputConfig(saving_folder="./aegiss_output/", saving_prefix="aegiss_selection", save_orb=True, save_plots=False)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `saving_folder` | `str` | `"./aegiss_output/"` | Directory for output files |
| `saving_prefix` | `str` | `"aegiss_selection"` | Base prefix for output file names |
| `save_orb` | `bool` | `True` | Write cube files for entropy-screened and final active orbitals |
| `save_plots` | `bool` | `False` | Generate and save diagnostic plots: SOE heatmap, threshold analysis, entanglement diagram, and mutual-information heatmap |

`saving_prefix` is expanded with selection-specific suffixes:
`{saving_prefix}_entropy_{idx}.cube` and `{saving_prefix}_AEGISS_{idx}.cube`.

---

### `build_mol`

```python
from aegiss.initial_guess import build_mol

mol = build_mol(atom, **kwargs)
```

| Parameter | Type | Description |
|---|---|---|
| `atom` | `str` or `list[str]` | Geometry in PySCF format or path to an `.xyz` file |
| `**kwargs` | — | Any `pyscf.gto.Mole` attribute (`basis`, `charge`, `spin`, `symmetry`, `verbose`, …) |

Returns a built `pyscf.gto.Mole` object.

---

### `run_scf`

```python
from aegiss.initial_guess import run_scf

mf = run_scf(mol, method="RHF", xc="b3lyp", solvent_info=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mol` | `pyscf.gto.Mole` | — | Molecular geometry and basis |
| `method` | `str` | `"RHF"` | Level of theory: `"RHF"`, `"UHF"`, `"RKS"`, `"UKS"` |
| `xc` | `str` | `"b3lyp"` | XC functional (KS methods only) |
| `solvent_info` | `tuple[str, float]` or `None` | `None` | Solvent model and dielectric constant, e.g. `("PCM", 78.4)` or `("DDCOSMO", 78.4)` |
| `**kwargs` | — | — | Any PySCF SCF attribute forwarded verbatim (`max_cycle`, `conv_tol`, …) |

Returns the converged PySCF mean-field object.

---

### `run_dmrg`

```python
from aegiss.wavefunction.dmrg import run_dmrg

energy, soe1, soe2, minfo = run_dmrg(
    mf, ncas, nelecas,
    init_bond_dim=10,
    symm_type="SU2",
    verbosity=0,
    get_qmi=False,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mf` | SCF object | — | Converged mean-field |
| `ncas` | `int` | — | Number of active orbitals |
| `nelecas` | `int` or `tuple[int, int]` | — | Number of active electrons. An integer total count or an explicit `(nalpha, nbeta)` pair. Odd integers are treated as high-spin doublets. |
| `init_bond_dim` | `int` | `10` | Initial DMRG bond dimension; the schedule grows to a larger final dimension automatically |
| `symm_type` | `str` | `"SU2"` | Spin symmetry: `"SU2"` (faster) or `"SZ"` (required for entropy calculation without MPS conversion) |
| `verbosity` | `int` | `0` | block2 output verbosity |
| `get_qmi` | `bool` | `False` | If `True`, compute and return single- and two-orbital entropies and the quantum mutual information matrix |

Returns `(energy, soe1, soe2, minfo)`. The last three are `None` when `get_qmi=False`.

`OMP_NUM_THREADS`, `SCRATCH_DIR`, and `MEM` environment variables are read at call time (see [Environment variables](#environment-variables) for details).

---

### `select_by_entropy`

```python
from aegiss.active_space.entropy_screening import select_by_entropy

selected_occ, selected_virt = select_by_entropy(
    occ_orbs, virt_orbs, soe1,
    fraction=0.1,
    fraction_type="r",
    ranking="global",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `occ_orbs` | `np.ndarray` | — | Indices of occupied candidate orbitals |
| `virt_orbs` | `np.ndarray` | — | Indices of virtual candidate orbitals |
| `soe1` | `np.ndarray` or `None` | — | Single-orbital entropies (length = `len(occ_orbs) + len(virt_orbs)`). Pass `None` to skip and return empty lists. |
| `fraction` | `float` | `0.1` | Threshold value (see `fraction_type`) |
| `fraction_type` | `"r"` or `"a"` | `"r"` | `"r"`: keep orbitals whose entropy is ≥ `fraction × max(S(1))`; `"a"`: keep orbitals whose entropy is ≥ `fraction` |
| `ranking` | `"global"` or `"subset"` | `"global"` | `"global"`: rank all orbitals together before thresholding; `"subset"`: apply the threshold independently to the occupied and virtual subspaces |

Returns `(selected_occ, selected_virt)` — each a `list[tuple[int, float]]` of
(orbital index, entropy value).

---

### `select_by_ao_projection`

```python
from aegiss.active_space.ao_projection import select_by_ao_projection

active_indices = select_by_ao_projection(
    mf,
    ao_labels,
    projected_basis="minao",
    contribution=None,
    contribution_type=None,
    top_n=None,
    restricted_space=None,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mf` | SCF object | — | Converged mean-field |
| `ao_labels` | `list[str]` or `list[list[str]]` | — | AO labels to project onto. Format: `"Element AtomicOrbital"` (e.g. `"C 2pz"`, `"Fe 3d"`, `"O.1 1s"` for the second oxygen). Each entry in the outer list is processed independently; the results are unioned. |
| `projected_basis` | `str` or `None` | `"minao"` | Basis used for projection. A minimal basis focuses the selection on chemically relevant contributions. `None` uses the original molecular basis. |
| `contribution` | `float` or `list[float]` or `None` | `None` | Projection weight threshold. A single float applies to all labels; a list assigns per-label thresholds. Mutually exclusive with `top_n`. |
| `contribution_type` | `"r"` or `"a"` or `None` | `None` | Required when `contribution` is set. `"r"`: threshold relative to the maximum weight; `"a"`: absolute threshold. |
| `top_n` | `int` or `list[int]` or `None` | `None` | Alternative to `contribution`: keep the top-N orbitals per label. A list assigns per-label counts. |
| `restricted_space` | `list[tuple[int, float]]` or `tuple[list, list]` or `None` | `None` | Restrict candidates to a pre-filtered set. Pass a flat list of `(index, score)` pairs, or a `(occ_list, virt_list)` tuple as returned by `select_by_entropy`. `None` considers all molecular orbitals. |

Returns `np.ndarray` of sorted orbital indices.

**AO label format**

Labels follow PySCF's `search_ao_label` convention:

```
"Element AtomicOrbital"          # all atoms of this element
"Element.N AtomicOrbital"        # N-th atom of this element (1-indexed)
["Fe 3dxz", "Fe 3dyz", "Fe 3dz^2"]  # multiple orbitals at once
```

---

### `localize`

```python
from aegiss.localization.orbital_localization import localize

new_mo_coeff = localize(method, *args, **kwargs)
```

Dispatches to a localization method by name.

| `method` | Aliases | Description |
|---|---|---|
| `"boys"` | — | Foster-Boys localization (minimises orbital spread) |
| `"edmiston"` | `"er"` | Edmiston-Ruedenberg localization (maximises self-repulsion) |
| `"pm"` | `"pipek"` | Pipek-Mezey localization (maximises partial charges) |
| `"nos"` | — | Natural orbitals from a correlated wavefunction |
| `"fnos"` | — | Frozen natural orbitals |

Common keyword arguments for `"boys"`, `"edmiston"`, `"pm"`:

| Keyword | Type | Default | Description |
|---|---|---|---|
| `mol` | `pyscf.gto.Mole` | — | First positional argument |
| `mf` | SCF object | — | Second positional argument |
| `n_occ` | `int` or `None` | `None` | Number of occupied orbitals (inferred from `mf` if omitted) |
| `on_virtual` | `bool` | `False` | Also localize the virtual space |

Additional keyword for `"pm"`:

| Keyword | Type | Default | Description |
|---|---|---|---|
| `pop_method` | `str` | `"becke"` | Population analysis scheme for the virtual block |

Arguments for `"nos"` and `"fnos"`:

| Keyword | Type | Default | Description |
|---|---|---|---|
| `mf` | SCF object | — | First positional argument |
| `correlated_method` | `"mp2"`, `"cisd"`, `"ccsd"` | — | Method used to build the correlated density matrix |
| `split` *(nos only)* | `bool` | `False` | Diagonalise occupied and virtual blocks of the 1-RDM separately |
| `active` *(fnos only)* | `bool` or `list[int]` | `True` | `True`: use all occupied orbitals; list of indices: freeze those orbitals |

```python
# Examples
localize("boys", mol, mf, on_virtual=True)
localize("pm", mol, mf, n_occ=10)
localize("nos", mf, correlated_method="mp2", split=True)
localize("fnos", mf, correlated_method="ccsd", active=[0, 1, 2])
```

---

### `save_orbitals`

```python
from aegiss.utils.dump_info import save_orbitals, save_orbitals_from_entropies
```

**`save_orbitals(mol, mo_coeff, candidate_indices, filename_prefix="mo", folder="./")`**

Writes one `.cube` file per orbital index in `candidate_indices`.

**`save_orbitals_from_entropies(mol, mo_coeff, candidate_indices, filename_prefix="mo", folder="./")`**

Same, but accepts `list[tuple[int, float]]` (the format returned by `select_by_entropy`).

Output file names follow the pattern `{filename_prefix}_{index}.cube`.

---

## Development

```bash
uv sync                       # install all dependency groups
pre-commit install            # install git hooks
pytest                        # run test suite
pytest --cov=aegiss           # run with coverage
```

Code style is enforced by **ruff** (linting + formatting) and **mypy** (strict type
checking). Both run automatically as pre-commit hooks and in CI on every pull request.

### Building the documentation

Install the `docs` dependency group, then use the MkDocs CLI:

```bash
uv sync --group docs          # install MkDocs and plugins
uv run mkdocs serve                  # live-reload preview at http://127.0.0.1:8000
mkdocs build                  # write static site to site/
```

See [docs/README.md](docs/README.md) for a full description of the docs layout and
how to add new pages.
