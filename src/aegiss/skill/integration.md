# Integrating AEGISS

Read this when AEGISS is one step of a larger program: a workflow engine, a quantum-computing
stack, a library function.

## Entry points

| Call | Runs DMRG | Returns |
|---|---|---|
| `aegiss.active_space.select_aegiss(mol, mf, dmrg_config, entropy_config, projection_config, output_config)` | yes, on every call | active orbital indices |
| `aegiss.active_space.select_by_ao_projection(mf, ao_labels, ..., restricted_space=None)` | no | active orbital indices |
| `aegiss.wavefunction.run_dmrg_subprocess(mf, ncas, nelecas, ...)` | yes | `(energy, soe1)` |
| `aegiss.active_space.select_by_entropy(occ, virt, soe1, ...)` | no | `(occ, virt)` lists of `(index, entropy)`, a valid `restricted_space` |

To try several thresholds, run DMRG once and screen its `soe1` repeatedly instead of calling
`select_aegiss` again. Signatures and options:
[API reference](https://algorithmiq.github.io/AEGISS/latest/api/). Pass every threshold
explicitly: the config dataclasses and the selection functions do not share defaults, and
the defaults are being changed (see the "Any run" rows of
[parameters.md](parameters.md#entropy-screening) and
[parameters.md](parameters.md#ao-projection)).

## Return values

- Restricted reference: a sorted `np.ndarray` of 0-based MO indices.
- Unrestricted reference: `{"alpha": np.ndarray, "beta": np.ndarray}`. The channels are not
  reconciled: index *k* is a different MO in each. Combining them is the caller's decision
  ([UHF references](https://algorithmiq.github.io/AEGISS/latest/uhf/)).
- No orbital passes the entropy screening: an empty flat `np.ndarray`, for an unrestricted
  reference too. Check the length before indexing or unpacking.

## Orbital indices

The API is 0-based. Convert only at the user boundary, with `to_display_all` (output) and
`from_display_all` (input) from `aegiss.utils.orb_display`, which follow `AEGISS_ORB_BASE` or
`aegiss config set-base`
([orbital index base](https://algorithmiq.github.io/AEGISS/latest/orbital_indexing/)).
PySCF's `sort_mo` assumes 1-based indices unless given `base=0`.

## DMRG runs as a subprocess

- `block2main` (installed with the pinned `block2` wheel) must be on `PATH` or in the `bin` of the
  venv that runs AEGISS; nothing imports `block2` in Python.
- Each call works in its own `SCRATCH_DIR/dmrg_subprocess_<ns timestamp>` folder and leaves it
  behind: long-running or parallel pipelines should give each job its own `SCRATCH_DIR` and
  clean it up.
- `MEM` and `OMP_NUM_THREADS` size the run
  ([environment variables](https://algorithmiq.github.io/AEGISS/latest/installation/#environment-variables)).
- Results depend on the `block2` version, which AEGISS pins.

## From selection to CASSCF

```python
from pyscf import mcscf

from aegiss.active_space import select_by_ao_projection
from aegiss.initial_guess import build_mol, run_scf

mol = build_mol(
    "C 0 0 0.667; C 0 0 -0.667; H 0 0.923 1.238; H 0 -0.923 1.238;"
    " H 0 0.923 -1.238; H 0 -0.923 -1.238",
    basis="sto-3g",
    symmetry=False,
)
mf = run_scf(mol, method="RHF")
# select_aegiss returns the same type; the projection alone keeps this example fast
active = select_by_ao_projection(mf, ao_labels=["C 2px"], contribution=0.5)

ncas = len(active)
nelecas = round(mf.mo_occ[active].sum())
mc = mcscf.CASSCF(mf, ncas, nelecas)
mc.kernel(mc.sort_mo(active, base=0))
assert (ncas, nelecas, mc.converged) == (2, 2, True)
```
