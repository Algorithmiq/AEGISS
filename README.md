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
  - [1. Build the molecule](#1-build-the-molecule)
  - [2. Run a mean-field calculation](#2-run-a-mean-field-calculation)
  - [3. Configure and run the full pipeline](#3-configure-and-run-the-full-pipeline)
  - [4. Localize orbitals before the pipeline (optional)](#4-localize-orbitals-before-the-pipeline-optional)
- [API reference](#api-reference)
  - [select_aegiss](#select_aegiss)
  - [Configuration dataclasses](#configuration-dataclasses)
  - [build_mol](#build_mol)
  - [run_scf](#run_scf)
  - [run_dmrg_subprocess](#run_dmrg_subprocess)
  - [select_by_entropy](#select_by_entropy)
  - [select_by_ao_projection](#select_by_ao_projection)
  - [plot_ao_contribution](#plot_ao_contribution)
  - [mo_ao_contributions](#mo_ao_contributions)
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

AEGISS ships a `configure_logging` helper that sets up structlog over stdlib,
writing timestamped plain-text output to **stdout**:

```python
import logging
from aegiss.utils.logging import configure_logging

configure_logging(level=logging.INFO)
# configure_logging(level=logging.DEBUG)  # alternative: For additional execution and selection details.
```

`configure_logging` must be called before any AEGISS import that triggers logging.
It silences the noisy `matplotlib` and `h5py` loggers automatically.

**Redirect stdout to control the destination:**

```bash
# local — save to file
python main.py > log.out

# SLURM — output lands in slurm-<jobid>.out automatically
#SBATCH -o slurm-%j.out
```

**Silence AEGISS selectively (e.g. in a notebook):**

```python
logging.getLogger("aegiss").setLevel(logging.WARNING)
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

dmrg_config = DMRGConfig(ncas=20, nelecas=20, max_bond_dim=500, symm_type="SZ")

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

For custom workflows that call individual steps directly, see `run_dmrg_subprocess`,
`select_by_entropy`, and `select_by_ao_projection` in the API reference below.

### 4. Localize orbitals before the pipeline (optional)

Orbital localization can improve DMRG convergence by making the orbitals more
spatially compact. Pass the localized coefficients back into `mf` before calling
`select_aegiss`.

```python
from aegiss.localization.orbital_localization import localize

# Boys localization of occupied orbitals only
mf.mo_coeff = localize("boys", mol, mf)

# Pipek-Mezey including virtuals
mf.mo_coeff = localize("pm", mol, mf, on_virtual=True)

# Intrinsic Bond Orbitals — good for transition metals and mixed bonding
mf.mo_coeff = localize("ibos", mol, mf)

# Intrinsic Atomic Orbitals — atom-centred, useful for population analysis
mf.mo_coeff = localize("iaos", mol, mf)

# Natural orbitals from an MP2 density matrix
mf.mo_coeff = localize("nos", mf, "mp2")
```

#### Freezing core orbitals (FNOs only)

For heavier atoms it is common to exclude chemically inert core orbitals from the
correlated calculation. This is supported by `"fnos"` via the `frozen` keyword
(NOs always use the full orbital space):

```python
import numpy as np

# Freeze the two lowest-energy (core) occupied orbitals
mf.mo_coeff = localize("fnos", mf, "mp2", frozen=[0, 1])
```

The frozen orbitals are excluded from the MP2/CISD/CCSD density matrix but are
kept in the output coefficient matrix, so `mf.mo_coeff` retains its original
shape and the rest of the pipeline is unaffected.

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
DMRGConfig(ncas, nelecas, max_bond_dim=500, symm_type="SZ", verbosity=0)
```

| Field | Type | Default | Description |
|---|---|---|---|
| `ncas` | `int` | — | Number of active orbitals |
| `nelecas` | `int` or `tuple[int, int]` | — | Number of active electrons. An integer total count or an explicit `(nalpha, nbeta)` pair — same convention as PySCF's CASCI. Odd integers are treated as high-spin doublets. |
| `max_bond_dim` | `int` | `500` | Target DMRG bond dimension. The schedule starts at `max_bond_dim // 2` and ramps to `max_bond_dim`. Must be a positive multiple of 4. |
| `symm_type` | `"SZ"` or `"SU2"` | `"SZ"` | Spin symmetry: `"SZ"` is required for entropy calculations; `"SU2"` is faster |
| `verbosity` | `int` | `0` | block2main output verbosity |

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

mf = run_scf(mol, method="RHF", xc="b3lyp", solvent_info=None, relativistic=None, **kwargs)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mol` | `pyscf.gto.Mole` | — | Molecular geometry and basis |
| `method` | `str` | `"RHF"` | Level of theory: `"RHF"`, `"UHF"`, `"RKS"`, `"UKS"` |
| `xc` | `str` | `"b3lyp"` | XC functional (KS methods only) |
| `solvent_info` | `tuple[str, float]` or `None` | `None` | Solvent model and dielectric constant, e.g. `("PCM", 78.4)` or `("DDCOSMO", 78.4)` |
| `relativistic` | `'x2c'` or `None` | `None` | Scalar relativistic correction. `'x2c'` applies the exact two-component (X2C) transformation via `mf.x2c()`. Recommended for transition metals and heavier elements. |
| `**kwargs` | — | — | Any PySCF SCF attribute forwarded verbatim (`max_cycle`, `conv_tol`, …) |

Returns the converged PySCF mean-field object.

---

### `run_dmrg_subprocess`

```python
from aegiss.wavefunction.dmrg_subprocess import run_dmrg_subprocess

energy, soe1 = run_dmrg_subprocess(
    mf, ncas, nelecas,
    max_bond_dim=500,
    symm_type="SZ",
    verbosity=0,
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mf` | SCF object | — | Converged mean-field |
| `ncas` | `int` | — | Number of active orbitals |
| `nelecas` | `int` or `tuple[int, int]` | — | Number of active electrons. An integer total count or an explicit `(nalpha, nbeta)` pair. Odd integers are treated as high-spin doublets. |
| `max_bond_dim` | `int` | `500` | Target DMRG bond dimension. The schedule starts at `max_bond_dim // 2` and ramps to `max_bond_dim`. Must be a positive multiple of 4. |
| `symm_type` | `str` | `"SZ"` | Spin symmetry: `"SZ"` or `"SU2"` |
| `verbosity` | `int` | `0` | block2main output verbosity |
| `get_qmi` | `bool` | `True` | If `True`, compute and return single-orbital entropies |

Returns `(energy, soe1)`. `soe1` is `None` when `get_qmi=False`.

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
| `restricted_space` | `list[int]` or `list[tuple[int, float]]` or `tuple[list, list]` or `None` | `None` | Restrict candidates to a pre-filtered set. Pass a plain list of orbital indices (e.g. `list(range(34, 131))`), a list of `(index, score)` pairs, or a `(occ_list, virt_list)` tuple as returned by `select_by_entropy`. `None` considers all molecular orbitals. |

Returns `np.ndarray` of sorted orbital indices.

**AO label format**

Labels follow PySCF's `search_ao_label` convention:

```
"Element AtomicOrbital"          # all atoms of this element
"Element.N AtomicOrbital"        # N-th atom of this element (1-indexed)
["Fe 3dxz", "Fe 3dyz", "Fe 3dz^2"]  # multiple orbitals at once
```

---

### `plot_ao_contribution`

```python
from aegiss.utils.plotting import plot_ao_contribution
```

Renders the per-orbital AO projection weights as a heatmap — one row per AO
group, columns are MOs, colour is the (normalised) projection weight. Wide
orbital ranges are paginated into stacked panels (`mos_per_plot`).
`select_by_ao_projection(..., save_path=...)` calls this internally for the
restricted case; call it directly for finer control or to combine spin channels.

```python
# Restricted: one row per AO group
plot_ao_contribution(alpha_contributions, folder="./", filename="ao")

# Unrestricted: alpha and beta interleaved (alpha above beta) within each
# group, with a thin white line separating consecutive groups
plot_ao_contribution(
    alpha_contributions,
    beta_contributions,
    folder="./",
    filename="ao",
    normalization="global",
)
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `ao_contributions` | `dict[int, list[tuple[int, float]]]` | — | Per-group `(orbital_index, weight)` lists covering the search space. For unrestricted calculations, the alpha-spin contributions. |
| `ao_contributions_beta` | `dict[int, list[tuple[int, float]]]` or `None` | `None` | Optional beta-spin contributions sharing the same group keys. When given, each group is drawn as an interleaved alpha-above-beta row pair. |
| `folder` | `str` | `"./"` | Output directory. |
| `filename` | `str` | `"ao_contribution"` | Base name; `_heatmap` and the format suffix are appended automatically. |
| `ao_labels` | `dict[int, str \| list[str]]` or `None` | `None` | Row labels per group (a list is joined with newlines). `None` falls back to letter identifiers A, B, C, … |
| `mo_labels` | `list[int]` or `None` | `None` | x-tick labels (orbital indices). Defaults to sequential integers. |
| `mos_per_plot` | `int` | `200` | Maximum MOs per panel; wider ranges paginate into stacked panels. |
| `fmt` | `str` or `None` | `"pdf"` | Output format (`"pdf"`, `"png"`, …). `None` lets matplotlib infer from the filename. |
| `normalization` | `"global"` or `"relative"` | `"global"` | `"global"`: divide the whole matrix by a single maximum, preserving magnitude differences between AO groups. `"relative"`: divide each group by its own maximum (jointly over alpha and beta), emphasising within-group patterns and spin asymmetry instead. |

> **Note** Combining alpha and beta adds *rows*, not columns, so it never worsens
> the horizontal crowding of wide orbital ranges — the figure just grows taller.

---

### `mo_ao_contributions`

```python
from aegiss.utils.mo_composition import mo_ao_contributions

composition = mo_ao_contributions(mf, [4, 5, 6], top_n=10)
```

The drill-down companion to `plot_ao_contribution`: when an MO shows only a
faint weight in the heatmap, this reports **which AOs that orbital is actually
built from**. For each requested MO the AO basis functions are ranked by the
magnitude of their expansion coefficient `mo_coeff[μ, i]`, and the `top_n`
largest are printed — signed, so the orbital phase is preserved — as a table
with one column per orbital, grouped four orbitals per row:

```
Top-4 AO contributions per MO (signed mo_coeff entries):
MO 1                   |  MO 2                   |  MO 3                   |  MO 4
-----------------------+-------------------------+-------------------------+-----------------------
0 O 2s        +0.4429  |  0 O 2py       +0.4900  |  0 O 2pz       +0.5455  |  0 O 2px       +0.6312
0 O 3s        +0.3757  |  1 H 1s        +0.3280  |  0 O 3pz       +0.3653  |  0 O 3px       +0.4957
1 H 1s        +0.1936  |  2 H 1s        -0.3280  |  0 O 3s        -0.3498  |  2 H 2px       +0.0311
2 H 1s        +0.1936  |  0 O 3py       +0.2219  |  2 H 1s        +0.2064  |  1 H 2px       +0.0311
```

For an unrestricted `mf` (`mo_coeff` shape `(2, n_ao, n_mo)`), the alpha and
beta channels are printed as adjacent columns per orbital.

Set `group_by_element=True` to collapse the atom index and merge every AO that
shares an element symbol and basis label (e.g. all `C 2pz` across the molecule)
into a single group, weighted by the **sum of `|coefficient|`** over its
members and ranked by that weight:

```
Top-4 AO contributions per MO (summed |mo_coeff| per element+label):
MO 4              |  MO 5              |  MO 6              |  MO 7
------------------+--------------------+--------------------+------------------
H 1s     +1.2781  |  H 1s     +1.3234  |  H 1s     +0.8674  |  O 2px    +1.3405
O 2py    +0.6054  |  O 2py    +0.6483  |  O 2s     +0.7885  |  O 2s     +0.2387
O 2pz    +0.6054  |  O 2pz    +0.6483  |  O 2pz    +0.6881  |  H 1s     +0.1161
O 2s     +0.0000  |  O 2s     +0.0000  |  O 2py    +0.6881  |  O 2py    +0.1043
```

Pass `clusters={cluster_id: [atom_idx, ...]}` to instead collapse by **cluster
and basis label**. Each atom is mapped to the cluster containing it (atoms in no
cluster fall into a catch-all group keyed `U`), and `|coefficient|` is summed
over the atoms of a cluster sharing a basis label. Each group label reads
`(<cluster id>) <elements> <ao_label>`, listing the cluster's distinct element
symbols (e.g. `(0) C 2pz`, or `(0) C,O 2pz` for a mixed cluster, `(U) H 1s` for
unassigned atoms). The membership log (printed first) lists each cluster's atoms
with indices as `<symbol>(<ids>)`, collapsing same-element atoms and preserving
any PySCF atom tag. `clusters` and `group_by_element` are mutually exclusive.

```
# clusters={0: [0], 1: [1, 2]} on a water molecule
Cluster membership:
  cluster 0: O(0)
  cluster 1: H(1,2)
Top-4 AO contributions per MO (summed |mo_coeff| per cluster+label):
MO 4               |  ...
-------------------+-----
(0) O 2px +0.9876  |  ...
(1) H 1s  +0.4521  |  ...
(0) O 2s  +0.1203  |  ...
(0) O 2pz +0.0000  |  ...
```

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mf` | SCF object | — | Converged mean-field exposing `mo_coeff` and `mol`. Restricted or unrestricted. |
| `mo_indices` | `int` or `Sequence[int]` | — | Orbital index, or indices, to decompose. |
| `top_n` | `int` | `10` | Number of largest AO contributions (or groups) kept per orbital. |
| `group_by_element` | `bool` | `False` | Sum `|coefficient|` over all atoms of the same element sharing a basis label, ranking the groups instead of individual AOs. |
| `clusters` | `Mapping[int, Sequence[int]]` | `None` | Map of cluster id to its atom indices. Sum `|coefficient|` over each cluster's atoms sharing a basis label. Mutually exclusive with `group_by_element`. |

Returns `dict[int, dict[str, list[tuple[int, str, float]]]]` — each MO maps to a
`{spin: entries}` dict (`spin` is `"alpha"`, plus `"beta"` when unrestricted),
where each entry is `(key, label, weight)` ordered by descending weight.
Ungrouped, `key` is the AO index and `weight` is the signed coefficient;
grouped, `key` is the number of AOs in the group and `weight` is the summed
`|coefficient|`. The table is always logged.

> Raw coefficients (not Mulliken populations) are used deliberately: they answer
> "which AOs is this orbital built from" directly and need no overlap matrix.
> They are *not* a normalised partition — in a non-orthogonal AO basis the
> values do not sum to 1 — but they give the same ranking of dominant AOs.

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
| `"ibos"` | `"ibo"` | Intrinsic Bond Orbital localization (maximises IAO population) |
| `"iaos"` | `"iao"` | Intrinsic Atomic Orbital construction (Löwdin-orthogonalized) |
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

Additional keywords for `"ibos"` / `"iaos"`:

| Keyword | Type | Default | Description |
|---|---|---|---|
| `minao` | `str` | `"minao"` | Reference minimal basis for IAO construction |
| `exponent` | `int` | `4` | Exponent in the IBO cost function (`"ibos"` only) |

Both methods transform only the occupied block; the virtual block is unchanged.

Arguments for `"nos"` and `"fnos"`:

| Keyword | Type | Default | Description |
|---|---|---|---|
| `mf` | SCF object | — | First positional argument |
| `correlated_method` | `"mp2"`, `"cisd"`, `"ccsd"` | — | Method used to build the correlated density matrix |
| `split` *(nos only)* | `bool` | `False` | Diagonalise occupied and virtual blocks of the 1-RDM separately |
| `frozen` *(fnos only)* | `list[int]` or `None` | `None` | Indices of occupied orbitals to freeze (0-based). Excluded from the correlated calculation but kept in the output. |
| `active` *(fnos only)* | `True` or `list[int]` | `True` | `True`: use all occupied orbitals; list of indices: the active occupied set (the complement is frozen). Ignored when `frozen` is provided. |

```python
# Examples
localize("boys", mol, mf, on_virtual=True)
localize("pm", mol, mf, n_occ=10)
localize("ibos", mol, mf)                                         # IBO localization
localize("iaos", mol, mf, minao="minao")                          # IAO construction
localize("nos", mf, correlated_method="mp2", split=True)          # full space, split blocks
localize("fnos", mf, correlated_method="ccsd", frozen=[0, 1, 2]) # freeze core orbitals
localize("fnos", mf, correlated_method="ccsd", active=[3, 4, 5]) # equivalent: keep 3,4,5 active
```

---

### `save_orbitals`

```python
from aegiss.utils.dump import save_orbitals, save_orbitals_from_entropies
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
