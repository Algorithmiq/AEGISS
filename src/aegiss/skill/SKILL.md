---
name: aegiss
description: AEGISS active-space selection (DMRG entropy screening + AO projection). Use when choosing an active space or CAS for a molecule, adding AEGISS to a pipeline, or comparing AEGISS with AVAS or AutoCAS.
---

# AEGISS active-space selection

AEGISS keeps an orbital in the active space when it is both strongly entangled (single-orbital
entropy from an approximate DMRG run) and built from the atomic orbitals that matter for the
process (AVAS-like projection). Method: [the AEGISS procedure](https://algorithmiq.github.io/AEGISS/latest/).

Pick the branch:

- Choosing an active space for a molecule: follow the **Workflow**.
- The user already has entropies or AEGISS output: start at step 5.
- Building AEGISS into a program or pipeline: read [integration.md](integration.md).
- Explaining AEGISS or comparing it with other methods: read [overview.md](overview.md).

Every number the workflow needs (window size, bond dimensions, thresholds, typical
`ao_labels`, the hand-off limit) is in [parameters.md](parameters.md). Read it before step 1.

## Workflow

Climb the cost ladder: each step is cheaper than the next and its check gates the next. A check
that does not pass sends you back, never forward.

### 0. Environment

- `python -c "import aegiss; print(aegiss.__version__)"` and `command -v block2main`
  (it may also live in the interpreter's venv `bin`, where AEGISS finds it even if the shell does not).
- Compare `aegiss.__version__` with `metadata.aegiss-version` in this file's frontmatter. If
  they differ, tell the user to run `aegiss skill install --force` (with the same `--agent` /
  `--user` options as before).
- `MEM`, `SCRATCH_DIR` and `OMP_NUM_THREADS` must suit the machine:
  [environment variables](https://algorithmiq.github.io/AEGISS/latest/installation/#environment-variables).

Done when: `aegiss` imports, `block2main` is on `PATH` or in the venv's `bin`, and the versions match or the user knows.

### 1. Frame the chemistry

Establish with the user: the process of interest, charge, spin, reference (RHF, or UHF for open
shells), basis, the expected CAS(n, m) range and candidate `ao_labels`
([parameters.md](parameters.md#typical-ao-labels)), grouped by cluster
([parameters.md](parameters.md#atom-labelling)). Ask for what the geometry does not tell you.
To label clusters of atoms or find the molecular plane, suggest
[MolTagger](https://algorithmiq.github.io/AEGISS/latest/moltagger/). If the case is one the
authors have not analysed (bond breaking, lone pairs, main-group radicals), say so: the
selection is exploratory.

Done when: the expected CAS range and the draft labels are written down in the conversation.

### 2. Mean field

```python notest
from aegiss.initial_guess import build_mol, run_scf

mol = build_mol("molecule.xyz", basis="cc-pvdz", charge=0, spin=0, symmetry=False)
mf = run_scf(mol, method="RHF")
```

Canonical orbitals are the default; natural orbitals only in the cases
[parameters.md](parameters.md#reference-and-orbitals) lists, and then step 4 also checks that
the window contains the orbitals of interest. Localized orbitals are not an option: the DMRG
window needs energy-ordered orbitals
([why](https://algorithmiq.github.io/AEGISS/latest/#3-single-orbital-entropy-screening)).

Done when: `mf.converged`; for UHF, `mf.spin_square()` is within the tolerance in
[parameters.md](parameters.md#reference-and-orbitals); the orbital type is decided.

### 3. Projection probe

Seconds, no DMRG: project every MO onto the draft labels and inspect what comes out.

```python
from aegiss.active_space import select_by_ao_projection
from aegiss.initial_guess import build_mol, run_scf
from aegiss.utils.mo_composition import mo_ao_contributions

# ethylene in the yz plane: its pi orbitals are built from C 2px
mol = build_mol(
    "C 0 0 0.667; C 0 0 -0.667; H 0 0.923 1.238; H 0 -0.923 1.238;"
    " H 0 0.923 -1.238; H 0 -0.923 -1.238",
    basis="sto-3g",
    symmetry=False,
)
mf = run_scf(mol, method="RHF")
candidates = select_by_ao_projection(mf, ao_labels=["C 2px"], contribution=0.5)
mo_ao_contributions(mf, list(candidates), top_n=3, group_by_element=True)
assert list(candidates) == [7, 8]
```

Done when: the candidates include orbitals of the expected character on both sides of the Fermi
level, near the frontier. Extra higher-lying virtuals are expected (the entropy step removes
them). Revise the labels (or the geometry orientation, for `px`/`py`/`pz` labels) and repeat only
when the expected character is missing or the wrong AOs dominate the candidates: DMRG cannot fix
wrong labels.

### 4. Cheap DMRG

Window size and bond dimension: [parameters.md](parameters.md#dmrg-window) and
[parameters.md](parameters.md#bond-dimension). If the cheap run itself reaches the
[hand-off limit](parameters.md#hand-off-limit) (a window of 50 or more orbitals starts at the
verification bond dimension), hand it off as in step 6.

```python notest
import numpy as np

from aegiss.utils import get_subspaces_from_cas
from aegiss.wavefunction import run_dmrg_subprocess

# Upper-case names are values you take from parameters.md (DMRG window, bond dimension,
# entropy screening); this file deliberately gives none. NELECAS places the window around the
# Fermi level: a centred window has NELECAS == NCAS for a closed shell.
occ, virt = get_subspaces_from_cas(mf, ncas=NCAS, nelecas=NELECAS)
_, soe1 = run_dmrg_subprocess(
    mf, ncas=NCAS, nelecas=NELECAS, max_bond_dim=CHEAP_BOND_DIM, symm_type=SYMM_TYPE,
    reorder=REORDER,
)
np.save("soe1.npy", soe1)  # step 5 retunes on this file, never on a new DMRG run

# window positions run from the lowest occupied to the highest virtual orbital
tau = FRACTION * soe1.max()  # fraction_type="r", global ranking
hot = np.flatnonzero(soe1 > tau)
margin = min(hot.min(), NCAS - 1 - hot.max())
```

For `fraction_type="a"` use `tau = FRACTION`; with entropy `ranking="subset"` take the set above
τ_E per half (occupied, virtual), each with its own maximum.

For UHF, decide spin-summed or spin-resolved first
([UHF references](https://algorithmiq.github.io/AEGISS/latest/uhf/)); with
`spin_resolved=True`, `soe1` has shape `(2, ncas)` and the margin uses
`np.flatnonzero((soe1 > tau).any(axis=0))`. Step 6 must then say so with `spin_soe="resolved"`
(`"summed"` otherwise, never `"auto"`).

Then rerun at the verification bond dimension
([parameters.md](parameters.md#bond-dimension)) and compare the sets above τ_E. If that rerun
reaches the [hand-off limit](parameters.md#hand-off-limit), do not run it: tell the user the
selection rests on an unverified cheap run, and put the verification in the step 6 script.

Done when: the entropies are saved, `margin` is at least the minimum in
[parameters.md](parameters.md#dmrg-window), and the set above τ_E is unchanged at the
verification bond dimension (or the verification is handed off). Otherwise widen the window
or raise the bond dimension and rerun this step.

### 5. Screen and project

```python notest
from aegiss.active_space import select_by_ao_projection, select_by_entropy

screened = select_by_entropy(
    occ,
    virt,
    soe1,
    fraction=FRACTION,
    fraction_type=FRACTION_TYPE,
    ranking=ENTROPY_RANKING,
)
active = select_by_ao_projection(
    mf,
    ao_labels=AO_LABELS,
    projected_basis=PROJECTION_BASIS,
    contribution=CONTRIBUTION,
    contribution_type=CONTRIBUTION_TYPE,
    top_n=TOP_N,  # None when thresholding with CONTRIBUTION
    restricted_space=screened,
)
n_active_electrons = round(mf.mo_occ[active].sum())  # UHF: mf.mo_occ[0][active["alpha"]], per channel
```

Pass every threshold explicitly: the functions and the config dataclasses of step 6 do not
share defaults. To reproduce `ProjectionConfig(ranking="subset")` (needed for `top_n` with an
occupied/virtual balance), project each half separately (`restricted_space=screened[0]`, then
`screened[1]`) and merge the results; this does not depend on the entropy `ranking`. For
spin-resolved entropies use `select_by_spin_entropy`, whose result is an
`{"alpha": ..., "beta": ...}` mapping, and build each half per channel
(`{ch: (win[0], []) for ch, win in screened.items()}`, then `([], win[1])`), as `select_aegiss`
does. Tune `fraction`, `contribution` or `top_n` on the saved `soe1`
([parameters.md](parameters.md#entropy-screening),
[parameters.md](parameters.md#ao-projection)).

Done when: the space passes the [final checks](parameters.md#final-checks), except the CASSCF
occupations (list them in the report or the step 6 script), and survives the
[stability check](parameters.md#stability-check).

### 6. Production run or hand-off

If the production run reaches the [hand-off limit](parameters.md#hand-off-limit), write the
script below with the tuned values, a comment per parameter saying how it was chosen, and the
expected resources (reference runtimes are under the hand-off limit), then stop. Otherwise run
it. Carry every option chosen in steps 4 and 5 into the script (the placeholders below), with
`SPIN_SOE` explicitly `"summed"` or `"resolved"`, never `"auto"`.

```python notest
from aegiss.active_space import select_aegiss
from aegiss.config_dataclasses import DMRGConfig, EntropyConfig, OutputConfig, ProjectionConfig

active = select_aegiss(
    mol,
    mf,
    dmrg_config=DMRGConfig(
        ncas=NCAS,
        nelecas=NELECAS,
        max_bond_dim=PRODUCTION_BOND_DIM,
        symm_type=SYMM_TYPE,
        reorder=REORDER,
        spin_soe=SPIN_SOE,
    ),
    entropy_config=EntropyConfig(
        fraction=FRACTION, fraction_type=FRACTION_TYPE, ranking=ENTROPY_RANKING
    ),
    projection_config=ProjectionConfig(
        ao_labels=AO_LABELS,
        contribution=CONTRIBUTION,
        contribution_type=CONTRIBUTION_TYPE,
        top_n=TOP_N,
        basis=PROJECTION_BASIS,
        ranking=PROJECTION_RANKING,
    ),
    output_config=OutputConfig(saving_folder="./aegiss_output/", save_plots=True),
)
```

Done when: the production selection is in hand, or the user has the script and knows how to run
it.

### 7. Report

- CAS(n, m): `m = len(active)`, `n = round(mf.mo_occ[active].sum())` (per spin channel for UHF).
- The orbitals in the user's index base: `to_display_all(active)` from
  `aegiss.utils.orb_display`, naming the base
  ([orbital index base](https://algorithmiq.github.io/AEGISS/latest/orbital_indexing/)).
  Indices the user gives you go through `from_display_all` first.
- Where orbitals and plots were saved.
- Caveats: the window margin, the threshold stability, the orbital type, the CASSCF occupations
  if checked.
- Work that uses the result should cite AEGISS:
  [citing](https://algorithmiq.github.io/AEGISS/latest/citing/).

Done when: the user has every item above.
