# AEGISS: Entropy- and AO-Projected Active-Space Selection

## Tutorial notebooks

| Notebook | Topic |
|---|---|
| [01_scf_and_localization.ipynb](01_scf_and_localization.ipynb) | Building the molecule, running HF/DFT, orbital localization (Boys, PM, ER, NOs, FNOs) |
| [02_dmrg.ipynb](02_dmrg.ipynb) | `DMRGConfig`, running DMRG, inspecting single-orbital entropies, entropy screening |
| [03_ao_projection.ipynb](03_ao_projection.ipynb) | `ProjectionConfig`, AO-projection selection, thresholds and top-N ranking |
| [04_aegiss.ipynb](04_aegiss.ipynb) | End-to-end AEGISS pipeline with all four config objects |

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
- Final active orbital space: $\{\psi_r\}$

---

## 1. Atom Labeling and AO Group Definition

- Partition the molecule into chemically meaningful clusters $\mathcal{C}_i$
  (e.g. metal centre, ligand π system, σ framework)
- For each cluster define an AO group $D_i \in \mathcal{D}$ as a set of PySCF `ao_labels`
  strings (e.g. `"Ru 4d"`, `"C 2pz"`, `["Ce 4f", "Ce 5d"]`)
- The union of all groups spans the chemical character space used for projection
- The active region $\mathbf{A}$ is defined by the CAS window (`ncas`, `nelecas`) and
  determines which orbitals enter the DMRG step

> **MolTagger** (`aegiss.utils.mol_tagger`) provides an interactive 3D viewer for
> selecting atoms and auto-generating `ao_labels` strings ready to paste into
> `ProjectionConfig`.

---

## 2. Mean-Field Calculation

- Perform HF or KS-DFT calculation to obtain the MO coefficient matrix:
  $$
  [\mathbf{C}]_{\mu p}
  $$
- Select orbitals belonging to the active region $\mathbf{A}$:
  $$
  \{\psi_p^{\mathbf{A}}\}
  $$
- Construct:
  $$
  [\mathbf{C}^{\mathbf{A}}]_{\mu p}
  $$

### Optional: orbital localization

Before running DMRG, canonical MOs can be transformed into localized orbitals to improve
the physical interpretability of the entropy screening.  Available methods:

| Key | Method | Criterion |
|---|---|---|
| `"boys"` | Foster–Boys | Minimum orbital spread |
| `"pm"` | Pipek–Mezey | Maximum atomic charge; preserves σ/π separation |
| `"er"` | Edmiston–Ruedenberg | Maximum self-repulsion energy |
| `"nos"` | Natural Orbitals | Diagonalise correlated 1-RDM |
| `"fnos"` | Frozen Natural Orbitals | As NOs but with a frozen core |

---

## 3a. Correlated Wavefunction in Active Region

- Compute the DMRG wavefunction in $\mathbf{A}$:
  $$
  \ket{\Psi_{\mathcal{W}}}
  =
  \ket{\Phi_{\text{occ}}}
  \wedge
  \ket{\psi^{\mathbf{A}}_{\mathcal{W}}}
  \wedge
  \ket{\Phi_{\text{virt}}}
  $$
  where $\ket{\Phi_{\text{occ}}}$ and $\ket{\Phi_{\text{virt}}}$ are the frozen core
  and virtual determinants, and $\ket{\psi^{\mathbf{A}}_{\mathcal{W}}}$ is the DMRG
  MPS restricted to $\mathbf{A}$.

- Construct the one-orbital reduced density matrix by tracing over all other sites:
  $$
  \rho^{p}_{ii'}
  =
  \sum_{\substack{i,i'\in\{0,+,-,2\}\\\mathbf{n}\neq n^p}}
  \braket{\psi_{\mathcal{W}}^{\mathbf{A}} | \mathbf{n}}
  \ket{i}\bra{i'}
  \braket{\mathbf{n} | \psi_{\mathcal{W}}^{\mathbf{A}}}
  $$

---

## 3b. Entropy-Based Reduction

For each orbital $p \in \{\psi_p^{\mathbf{A}}\}$:

1. Extract one-orbital RDM $\rho^p$
2. Diagonalise $\rho^p$ to obtain eigenvalues $\{\omega_{\alpha,p}\}$
3. Compute the single-orbital entropy:
   $$
   S(1)_p = -\sum_{\alpha=1}^{4} \omega_{\alpha,p} \ln \omega_{\alpha,p}
   $$

Determine $S(1)_{\max} = \max_p S(1)_p$ and select entropy-active orbitals:

$$
\{\psi^{\mathbf{A}}_{E}\}
=
\{\psi_p^{\mathbf{A}} \mid S(1)_p \geq \tau_E\}
$$

with threshold $\tau_E$ controlled by `EntropyConfig`:

| `fraction_type` | Threshold $\tau_E$ |
|---|---|
| `'r'` (relative, default) | $\text{fraction} \times S(1)_{\max}$ |
| `'a'` (absolute) | $\text{fraction}$ (directly) |

The threshold is applied **independently** to the occupied and virtual subspaces
(`ranking='subset'`, default) or to all CAS orbitals jointly (`ranking='global'`).

Yields the filtered coefficient matrix:
$$
[\mathbf{C}^{\mathbf{A}}_E]_{\mu p}, \quad p \in \{\psi^{\mathbf{A}}_E\}
$$

---

## 4. AO Projection for Each Group $D \in \mathcal{D}$

- Construct a minimal AO basis (MinAO, default `minao`) on a copy of the molecule
- Compute the cross-overlap matrix between MinAO functions ($\nu$) and original
  molecular AOs ($\mu$):
  $$
  [\mathbf{S}]_{\nu\mu} = \langle \nu^{\text{MinAO}} | \mu^{\text{mol}} \rangle
  $$

For each AO group $D$:

1. Extract the rows of $\mathbf{S}$ corresponding to the AOs in $D$ (via
   `mol.search_ao_label`):
   $$
   [\mathbf{S}^D]_{\eta\mu}, \quad \eta \in D
   $$

2. Compute projected MO coefficients:
   $$
   [\mathbf{O}_E^D]_{\eta p}
   =
   \sum_{\mu}
   [\mathbf{S}^D]_{\eta\mu}
   \,[\mathbf{C}^{\mathbf{A}}_E]_{\mu p}
   $$

3. Compute AO weights (L1 norm over the group):
   $$
   w_p^D = \sum_{\eta \in D} \left|[\mathbf{O}_E^D]_{\eta p}\right|
   $$

4. Selection — two modes, controlled by `ProjectionConfig`:

   - **Threshold** (`contribution` + `contribution_type`):
     $$
     \{\psi_r^D\} = \{\psi_p^{\mathbf{A}}_E \mid w_p^D \geq \epsilon_D\}
     $$
     where $\epsilon_D$ is absolute (`contribution_type='a'`) or relative to
     $\max_p w_p^D$ (`contribution_type='r'`).

   - **Top-N ranking** (`top_n`): select the $n_D$ orbitals with the largest $w_p^D$,
     regardless of their absolute value.

   As with entropy screening, AO projection can run independently per subspace
   (`ranking='subset'`, default) or on the combined entropy-screened set
   (`ranking='global'`).

---

## 5. Active Space Construction

Merge the selected sets across all AO groups:
$$
\{\psi_r\}
=
\bigcup_{D \in \mathcal{D}}
\{\psi_r^D\}
$$

---

## Return

Final active space:
$$
\{\psi_r\}
$$
