# Tutorials

The tutorials walk through the full AEGISS workflow using benzene as a running example.
Each notebook is self-contained and builds on the previous one.

| Notebook | Topic |
|---|---|
| [01 — SCF & Localization](01_scf_and_localization.ipynb) | Building the molecule, running HF/DFT, solvation models (PCM, DD-COSMO), scalar relativistic corrections (X2C), orbital localization (Boys, PM, ER, NOs, FNOs) |
| [02 — DMRG & Entropy](02_dmrg.ipynb) | `DMRGConfig`, running DMRG, inspecting single-orbital entropies, entropy screening |
| [03 — AO Projection](03_ao_projection.ipynb) | `ProjectionConfig`, AO-projection selection, thresholds and top-N ranking |
| [04 — Full Pipeline](04_aegiss.ipynb) | End-to-end AEGISS pipeline with all four config objects |

---

## Algorithm overview

### Inputs

- Molecular structure: $\{\mathcal{R}\}$
- Mean-field method and basis set
- Correlated wavefunction method: $\mathcal{W}$ (DMRG via block2)
- AO groups: $\mathcal{D}$ — PySCF `ao_labels` strings, e.g. `"Fe 3d"`, `"C 2pz"`
- Entropy threshold: $\tau_E$ (default: $0.1 \times S(1)_{\max}$, configurable)
- AO cutoff: $\epsilon_D$ or rank threshold $n_D$

### Output

Final active orbital space: $\{\psi_r\}$

---

## Step 1 — Atom labeling and AO group definition

Partition the molecule into chemically meaningful clusters $\mathcal{C}_i$ (e.g. metal
centre, ligand π system, σ framework). For each cluster define an AO group
$D_i \in \mathcal{D}$ as a set of PySCF `ao_labels` strings.

!!! tip "MolTagger"
    `aegiss.utils.mol_tagger.MolTagger` provides an interactive 3D viewer for selecting
    atoms and auto-generating `ao_labels` strings ready to paste into `ProjectionConfig`.

---

## Step 2 — Mean-field calculation

Perform [HF](https://pyscf.org/user/scf.html) or [KS-DFT](https://pyscf.org/user/dft.html)
to obtain the MO coefficient matrix $[\mathbf{C}]_{\mu p}$, then select orbitals belonging
to the active region $\mathbf{A}$ defined by the CAS window (`ncas`, `nelecas`).

**Optional solvation** — for molecules in solution, pass `solvent_info=(model, dielectric_constant)`
to `run_scf`. Two implicit-solvent families are supported
([PySCF solvation docs](https://pyscf.org/user/solvent.html)):

| Model | Key | Notes |
|---|---|---|
| [IEF-PCM](https://pyscf.org/user/solvent.html#pcm-model) | `"IEF-PCM"` | ISO standard; general purpose |
| [C-PCM](https://pyscf.org/user/solvent.html#pcm-model) | `"C-PCM"` | Conductor-like limit; slightly cheaper |
| [DD-COSMO](https://pyscf.org/user/solvent.html#ddcosmo) | `"DDCOSMO"` | Linear-scaling; good for large systems |

```python
mf_pcm = run_scf(mol, method="RHF", solvent_info=("IEF-PCM", 78.4))  # water
```

The solvation wrapper is fully composable with the `relativistic` parameter and
propagates into the DMRG integral step transparently via `CASCI.get_h1eff`.

---

**Optional scalar relativistic correction** — for systems containing transition metals
or heavier elements, pass `relativistic='x2c'` to `run_scf` to apply the
[exact two-component (X2C) transformation](https://pyscf.org/user/scf.html#scalar-relativistic-correction).
The relativistic one-electron Hamiltonian propagates automatically into the DMRG integral
step via `CASCI.get_h1eff`.

```python
mf = run_scf(mol, method="RHF", relativistic="x2c")
```

**Optional orbital localization** — canonical MOs can be transformed before DMRG to
improve physical interpretability of entropy screening:

| Key | Method | Criterion |
|---|---|---|
| `"boys"` | Foster–Boys | Minimum orbital spread |
| `"pm"` | Pipek–Mezey | Maximum atomic charge; preserves σ/π separation |
| `"er"` | Edmiston–Ruedenberg | Maximum self-repulsion energy |
| `"nos"` | Natural Orbitals | Diagonalise correlated 1-RDM |
| `"fnos"` | Frozen Natural Orbitals | As NOs but with a frozen core |

---

## Step 3a — Correlated wavefunction in active region

Compute the DMRG wavefunction in $\mathbf{A}$ and construct the one-orbital reduced
density matrix $\rho^{p}$ by tracing over all other sites.

---

## Step 3b — Entropy-based reduction

For each orbital $p$, compute the single-orbital entropy:

$$
S(1)_p = -\sum_{\alpha=1}^{4} \omega_{\alpha,p} \ln \omega_{\alpha,p}
$$

Select entropy-active orbitals:

$$
\{\psi^{\mathbf{A}}_{E}\} = \{\psi_p^{\mathbf{A}} \mid S(1)_p \geq \tau_E\}
$$

| `fraction_type` | Threshold $\tau_E$ |
|---|---|
| `'r'` (relative, default) | $\text{fraction} \times S(1)_{\max}$ |
| `'a'` (absolute) | $\text{fraction}$ directly |

The threshold is applied **independently** to occupied and virtual subspaces
(`ranking='subset'`, default) or to all CAS orbitals jointly (`ranking='global'`).

---

## Step 4 — AO projection for each group $D \in \mathcal{D}$

Compute AO weights by projecting entropy-filtered MOs onto a minimal basis
(AO groups are specified as [`ao_labels`](https://pyscf.org/user/gto.html) strings,
e.g. `"Fe 3d"`, `"C 2pz"`):

$$
w_p^D = \sum_{\eta \in D} \left|[\mathbf{O}_E^D]_{\eta p}\right|
$$

Two selection modes, controlled by `ProjectionConfig`:

- **Threshold** — keep orbitals where $w_p^D \geq \epsilon_D$ (absolute or relative)
- **Top-N** — keep the $n_D$ orbitals with the largest weight regardless of magnitude

---

## Step 5 — Active space construction

Merge selected sets across all AO groups:

$$
\{\psi_r\} = \bigcup_{D \in \mathcal{D}} \{\psi_r^D\}
$$
