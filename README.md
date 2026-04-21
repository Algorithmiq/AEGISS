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
- [Environment variables](#environment-variables)
- [Tutorial — benzene π system](#tutorial--benzene-π-system)
- [API reference](#api-reference)
  - [select_aegiss](#select_aegiss)
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

---

## Environment variables

| Variable | Default | Description |
|---|---|---|
| `OMP_NUM_THREADS` | `4` | OpenMP threads used by block2 |
| `SCRATCH_DIR` | `./tmp` | Directory for DMRG scratch files (MPS/MPO) |

Set them before running any calculation:

```python
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["SCRATCH_DIR"] = "/scratch/my_job"
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

### 3. Run DMRG on a preliminary active space

Before screening, we need single-orbital entropies. We run DMRG on a generous
active space that over-selects orbitals.

```python
from aegiss.wavefunction.dmrg import run_dmrg
from aegiss.utils.orbital_indices import get_subspaces_from_cas

ncas, nelecas = 20, 20
active_orbs, virt_orbs = get_subspaces_from_cas(mf, ncas=ncas, nelecas=nelecas)

energy, soe1, _soe2, minfo = run_dmrg(
    mf,
    ncas=ncas,
    nelecas=nelecas,
    symm_type="SZ",
    init_bond_dim=10,
    get_qmi=True,
)
```

`soe1` is a 1-D array of single-orbital entropies for all `ncas` orbitals.

### 4. Screen by entropy

```python
from aegiss.active_space.entropy_screening import select_by_entropy

screened_occ, screened_virt = select_by_entropy(
    occ_orbs=active_orbs,
    virt_orbs=virt_orbs,
    soe1=soe1,
    ranking="subset",     # rank occ and virt separately
    fraction=0.1,         # keep orbitals with S(1) ≥ 10 % of the maximum
    fraction_type="r",    # 'r' = relative threshold
)
```

Both return values are `list[tuple[int, float]]` — orbital index paired with its
(possibly normalised) entropy.

### 5. Refine with AO projection

```python
from aegiss.active_space.ao_projection import select_by_ao_projection

final_space = select_by_ao_projection(
    mf,
    ao_labels=["C 2pz"],          # project onto carbon 2pz orbitals
    contribution=0.5,              # keep orbitals with weight ≥ 0.5
    contribution_type="a",         # 'a' = absolute threshold
    projected_basis="minao",       # project in a minimal basis
    restricted_space=(screened_occ, screened_virt),  # only consider entropy-screened orbitals
)
# final_space : np.ndarray of orbital indices
```

### 6. Save orbitals for visualisation

```python
from aegiss.utils.dump_info import save_orbitals, save_orbitals_from_entropies

# Save entropy-screened orbitals (colour-coded by entropy)
save_orbitals_from_entropies(
    mol, mf.mo_coeff, screened_occ,
    filename_prefix="benzene_entropy_occ",
    folder="./orbs/",
)

# Save the final AEGISS active space
save_orbitals(
    mol, mf.mo_coeff, final_space,
    filename_prefix="benzene_aegiss",
    folder="./orbs/",
)
```

Cube files can be opened in VESTA, VMD, or Avogadro.

---

## API reference

### `select_aegiss`

```python
from aegiss.active_space.aegiss import select_aegiss

active_indices = select_aegiss(
    mol, mf,
    dmrg_ncas, dmrg_nelecas,
    init_bond_dim=10,
    symm_type="SZ",
    entropy_ranking="subset",
    entropy_fraction=0.1,
    entropy_fraction_type="r",
    projection_basis="minao",
    projection_ao_labels=["C 2pz"],
    projection_contribution=0.5,
    projection_contribution_type="a",
    projection_top_n=None,
    projection_ranking="subset",
    saving_folder="./orbs/",
    saving_prefix="molecule",
    verbosity=0,
    save_orb=True,
)
```

High-level entry point that runs the full pipeline in a single call: DMRG on the
specified CAS → entropy screening → AO projection. An orbital is kept when it passes
*both* criteria — AO projection is restricted to the entropy-screened candidate set.
If entropy screening returns no orbitals, the function returns an empty array early.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `mol` | `pyscf.gto.Mole` | — | Molecular geometry and basis set |
| `mf` | SCF object | — | Converged mean-field |
| `dmrg_ncas` | `int` | — | Number of active orbitals for the DMRG calculation |
| `dmrg_nelecas` | `int` | — | Number of active electrons for the DMRG calculation |
| `init_bond_dim` | `int` | `10` | Initial DMRG bond dimension; grows automatically |
| `symm_type` | `str` | `"SZ"` | Spin symmetry: `"SZ"` (required for entropies) or `"SU2"` (faster) |
| `entropy_ranking` | `str` | `"subset"` | `"subset"`: threshold per subspace; `"global"`: all orbitals ranked together |
| `entropy_fraction` | `float` | `0.1` | Entropy threshold value (see `entropy_fraction_type`) |
| `entropy_fraction_type` | `str` | `"r"` | `"r"`: keep S(1) ≥ fraction × max S(1); `"a"`: absolute threshold |
| `projection_basis` | `str` or `None` | `"minao"` | Basis for AO projection; `None` uses the original molecular basis |
| `projection_ao_labels` | `list[str]` | `["C 2pz"]` | AO labels to project onto — same format as `select_by_ao_projection` |
| `projection_contribution` | `float` | `0.5` | Projection weight threshold |
| `projection_contribution_type` | `str` | `"a"` | `"a"`: absolute; `"r"`: relative to maximum weight |
| `projection_top_n` | `int` or `None` | `None` | Keep top-N orbitals per label instead of thresholding |
| `projection_ranking` | `str` | `"subset"` | `"subset"`: project occ and virt subspaces separately; `"global"`: project combined set |
| `saving_folder` | `str` | `"./aegiss/examples/orbs/"` | Directory for output cube files |
| `saving_prefix` | `str` | `"benzene"` | Base prefix for cube file names; AEGISS appends selection-specific suffixes such as `_entropy` and `_AEGISS` before the orbital-specific filename part |
| `verbosity` | `int` | `0` | block2 output verbosity |
| `save_orb` | `bool` | `True` | Write cube files for screened and final orbitals, grouped by the suffixed prefixes derived from `saving_prefix` |

Returns `np.ndarray` of sorted orbital indices.

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
| `nelecas` | `int` | — | Number of active electrons |
| `init_bond_dim` | `int` | `10` | Initial DMRG bond dimension; the schedule grows to a larger final dimension automatically |
| `symm_type` | `str` | `"SU2"` | Spin symmetry: `"SU2"` (faster) or `"SZ"` (required for entropy calculation without MPS conversion) |
| `verbosity` | `int` | `0` | block2 output verbosity |
| `get_qmi` | `bool` | `False` | If `True`, compute and return single- and two-orbital entropies and the quantum mutual information matrix |

Returns `(energy, soe1, soe2, minfo)`. The last three are `None` when `get_qmi=False`.

`OMP_NUM_THREADS` and `SCRATCH_DIR` environment variables are read at call time.

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
