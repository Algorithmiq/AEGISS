# AEGISS as a Snakemake workflow: design

Status: approved design, not implemented. Date: 2026-10-30.

## Goal

Run AEGISS as a workflow of cached, resumable, file-based steps, so that users can

- reuse expensive results (SCF, localization, DMRG) across parameter sweeps and across studies;
- run batches over many molecules and settings;
- run on SLURM clusters, where a single DMRG can exceed the 24 h walltime;
- trace and audit every result back to its inputs;
- call each step on its own, from Python or the command line, without the workflow engine.

`select_aegiss` keeps its signature and behaviour; notebooks and scripts that use it do not change.

## Decisions

| Topic | Decision |
|---|---|
| Where the workflow lives | In this repository, shipped to users as the optional `workflow` extra |
| Engine | Snakemake 9 with `snakemake-executor-plugin-slurm`; local executor for testing |
| Target environment | SLURM clusters: shared filesystem, no internet on compute nodes, 24 h walltime |
| Controller | `snakemake` runs on the login node (allowed there); `--retries` resubmits timed-out jobs |
| Software environment | A uv venv pre-built on the shared filesystem; no per-rule `conda:` or containers |
| Workflow inputs | Built-in SCF (and optional localization), or a user-supplied PySCF chkfile |
| Internal orbital format | PySCF chkfile |
| TREXIO | Deferred until PySCF's TREXIO support stabilizes; the design reserves an import and an export step |
| Batch description | Python: a `Study` of `Run`s built from the config dataclasses, written to a manifest |
| Launching | `aegiss run` / `Study.run()` wrap Snakemake; the Snakefile is public and documented for direct use |
| Provenance | Rerun level (settings and versions) and audit level (checksum chain), no query database |
| Feature coverage | Full matrix in v1: RHF/UHF (and their RKS/UKS counterparts), spin-summed and spin-resolved entropies, SU2/SZ, `global`/`subset` ranking, optional localization. ROHF is untested in AEGISS and out of scope |
| Deliverables | Selection, active-space Hamiltonian (FCIDUMP; TREXIO later), per-study summary table |
| Split UHF selections | No Hamiltonian in v1; a reconciliation step is future work |
| Result storage | Content-addressed store: each step's output folder is named by a hash of its inputs and settings |
| Restarts | SCF through PySCF chkfiles; DMRG through `block2main` keywords only, never the `block2` Python API |

Rejected alternatives: Snakemake paths with the settings encoded as wildcards (the layout breaks
whenever a setting is added and reuse stops at identical paths); a Snakefile calling the private
helpers directly (couples a public artifact to internals, steps not usable without Snakemake);
submit-and-exit via `--immediate-submit` (unreliable with Snakemake 8+ and the SLURM plugin); an
AiiDA-style queryable provenance database (heavier than the requirements); a `pyblock2` driver for
DMRG restarts (AEGISS never imports `block2`, without exception).

## Architecture

```text
src/aegiss/
  active_space/, wavefunction/, ...   # in-memory core: numpy and PySCF objects, no files
  steps/                              # public, one file-in/file-out function per step
  workflow/                           # Study/Run, manifest, hashing, store, Snakefile,
                                      #   SLURM profile, `aegiss run`, trace/verify
```

- **In-memory core.** Mostly existing code, made public. `run_dmrg_subprocess` splits into
  `write_dmrg_inputs`, `run_block2main` (resumable) and `read_dmrg_outputs`. The helpers of
  `select_aegiss` (`_screen_by_entropy`, `_ao_projection_passes`, `_collect_active_indices`)
  become public, and `select_aegiss` is rebuilt from them. `run_dmrg_subprocess` keeps working as
  the composition of the three DMRG pieces.
- **`aegiss.steps`.** Thin adapters: load the input files, call the core, write the outputs and
  `provenance.json`. No dependency on Snakemake or SLURM; callable from tests and as
  `python -m aegiss.steps <step>`.
- **`aegiss.workflow`.** The only module that imports Snakemake. Ships the Snakefile and a SLURM
  profile as package data.

### Steps

Steps are cut where caching or resources matter, not one per function.

| Step | Output folder | Resources | Why it is a separate step |
|---|---|---|---|
| `scf` | `store/scf/<h>/` | SLURM, moderate, resumable | expensive; shared by many runs |
| `import_chk` | `store/scf/<h>/` | local | user-supplied mean field |
| `localize` (optional) | `store/orb/<h>/` | SLURM, moderate | expensive; has its own settings |
| `dmrg` | `store/dmrg/<h>/` | SLURM, heavy, resumable | the expensive step; must survive the walltime |
| `select` | `store/sel/<h>/` | grouped | cheap; what parameter sweeps vary |
| `hamiltonian` | `store/ham/<h>/` | grouped | skipped for split UHF selections |
| `render` (on request) | `store/render/<h>/` | grouped | cube files and plots |
| `summary` | `studies/<study>/`, `results/<study>/` | local | human-readable view of the store |

- Entropy screening, AO projection and merging form one `select` step: they take seconds, and
  separate jobs would only add queue waits. `screened.json` is still written for provenance.
- The single-orbital entropies are computed in `select`, not in `dmrg`: the DMRG output (energy,
  density matrices) does not depend on the screening settings.
- Writing the DMRG inputs is part of the `dmrg` step, so its hash covers the exact `FCIDUMP` that
  produced the result.
- `select`, `hamiltonian` and `render` share a Snakemake `group:`, so a sweep over the screening
  settings costs one SLURM job, not one per value.
- Reserved for later: `import_trexio` (TREXIO to chkfile, before `scf`'s consumers) and
  `export_trexio` (alongside `hamiltonian`), and a UHF reconciliation step before `hamiltonian`.

## Hashing

The Python `Study` computes every hash and writes it into the manifest; Snakemake only follows it.

```text
key    = sha256(canonical_json({step, step_version, settings, inputs, pins}))
folder = store/<step>/<key[:16]>/
```

- `step_version` is an integer per step, bumped by hand when the step's output changes meaning.
- `settings` is the resolved settings dataclass with defaults filled in.
- `inputs` maps each input name to its parent's full key.
- `pins` holds the versions that change the numbers: `block2` for `dmrg`. The `pyscf` and
  `aegiss` versions are recorded in provenance but not hashed, so a library upgrade does not
  invalidate the store; a release that changes a step's numbers bumps that `step_version`.
- Canonical JSON: sorted keys, floats in their shortest round-trip form (`repr`), list order kept
  (the order of `ao_labels` matches `contribution`), orbital indices 0-based.
- Not hashed: `OMP_NUM_THREADS`, `MEM`, `SCRATCH_DIR`. They change speed, not results.
- `import_chk`: the key is the SHA-256 of the file's bytes.

## File formats

Every JSON file carries a `schema` field (`aegiss.<kind>/<major>`) and, for step outputs, the
full `key` of the step that wrote it, so a file copied out of the store still says which setup
produced it. The schemas are public API and follow the deprecation policy. Orbital indices in files are always 0-based and say so
(`"index_base": 0`); `AEGISS_ORB_BASE` only affects printed output and `summary.tsv`.

### Manifest

```json
{
  "schema": "aegiss.manifest/1",
  "study": "benzene_sweep",
  "store": "/shared/aegiss-store",
  "nodes": {
    "a1b2...": {"step": "scf", "step_version": 1, "settings": {}, "inputs": {}},
    "c3d4...": {"step": "dmrg", "step_version": 1, "settings": {}, "inputs": {"orbitals": "a1b2..."}}
  },
  "runs": [{"name": "benzene/frac0.1", "targets": {"select": "e5f6...", "hamiltonian": "..."}}]
}
```

The Snakefile has one rule per step type with a `{h}` wildcard; each rule's inputs are looked up in
`nodes[h]`.

### Step folders

| Folder | Files |
|---|---|
| `scf/<h>/` | `scf.chk`: PySCF chkfile (`mol`, `scf/{mo_coeff, mo_occ, mo_energy, e_tot}`) plus an `aegiss/` group with `{method: RHF \| UHF \| RKS \| UKS, xc, solvent, relativistic, converged}`. `load_mean_field()` rebuilds the matching SCF object, solvent and X2C wrappers included, without rerunning it. Working state, not declared: `work/progress.chk` and `work/diis.h5` (see SCF restart). |
| `orb/<h>/` | `orbitals.chk`: same layout with the localized `mo_coeff`; `mo_energy` optional. |
| `dmrg/<h>/` | Declared outputs: `pdm1.npy`, `npc1.npy`, `result.json` (`{energy, converged, sweeps_done, attempts, resumed_from_sweep, window: {occ, virt}}`). Working state, not declared: `FCIDUMP`, `dmrg.conf`, `restart/`, `state.json`. |
| `sel/<h>/` | `soe1.npy`, `screened.json` (`{index, entropy}` pairs per window and per spin), `active_space.json`. |
| `render/<h>/` | Cube files and plots named `<run>_<key[:8]>_<kind>...` (for example `benzene-frac0.1_e5f6a7b8_AEGISS_17.cube`), because these are the files users copy out of the store and open elsewhere. |
| `ham/<h>/` | `FCIDUMP` and `ham.json` (`{norb, nelec, ms2, e_core}`); for a split UHF selection only `ham.json` with `"status": "skipped", "reason": "unrestricted_split"` (Snakemake needs every declared output to exist). |

Every step folder also holds `provenance.json` and, while it is being computed, `.claim/`.
Machine-readable files keep fixed names, which the Snakefile declares as outputs; their identity
comes from the folder and from the `key` inside them.

### `active_space.json`

```json
{"schema": "aegiss.active_space/1", "key": "<sha256>", "index_base": 0, "reference": "restricted",
 "status": "ok", "indices": [17, 18, 20], "occupied": [17, 18], "virtual": [20]}
```

- `reference` is `restricted` (RHF, RKS) or `unrestricted` (UHF, UKS).
- Unrestricted: `"split": true` with `"alpha"` and `"beta"` objects, each holding `indices`,
  `occupied` and `virtual`.
- No orbital passes the screening: `"status": "empty_screening"` and empty lists, matching the
  current behaviour of `select_aegiss`.

### Study outputs

- `studies/<study>/summary.tsv`: one row per run with name, molecule, key settings, reference,
  active-space size, occupied/virtual split, DMRG energy, `converged`, wall time, and store keys.
- `studies/<study>/`: a copy of the manifest and a study record (command line, versions, user,
  start and end times).
- `results/<study>/<run>/`: one folder per run, which is the grouping of everything a selection
  setup produced. It holds links named after the steps (`scf/`, `orb/`, `dmrg/`, `sel/`, `ham/`,
  `render/`; absent steps have no link) and a `run.json` with the run's name, the resolved
  settings of every step and every step's key. Deleting it never deletes store data.

## SCF restart

A large SCF can also exceed the walltime, so the `scf` step resumes like `dmrg`, following
PySCF's chkfile restart ([example](https://github.com/pyscf/pyscf/blob/master/examples/scf/14-restart.py)):

- The step sets `mf.chkfile` to `store/scf/<h>/work/progress.chk`, which PySCF updates during the
  iterations, and `mf.diis_file` to `work/diis.h5`. Neither is a declared output, so both survive
  a failed attempt.
- On a retry, if `work/progress.chk` exists, the step sets `mf.init_guess = "chk"` with that
  chkfile and restores the DIIS history, so the iterations continue rather than restart.
- Resuming does not change the step's key. `provenance.json` records `resumed: true` and the
  attempts.
- Caveat for UHF/UKS: continuing from a partly converged guess can reach a different solution
  from an uninterrupted run. The resumed flag makes this visible; users who need a specific
  solution give an explicit initial guess.
- An explicit initial guess (a user-supplied chkfile) is a separate setting and is hashed, because
  it can change which solution the SCF reaches.

## DMRG restart protocol

### Locations

| What | Where |
|---|---|
| `block2` scratch (`prefix`) | node-local disk: `$TMPDIR`, falling back to `SCRATCH_DIR` |
| MPS backups (`restart_dir`) | `store/dmrg/<h>/restart/`, two rotating slots |
| Progress record | `store/dmrg/<h>/state.json` |

`restart/` and `state.json` are not declared outputs, because Snakemake deletes the declared
outputs of a failed job.

### Rule

```text
rule dmrg:
    output:    store/dmrg/{h}/result.json, pdm1.npy, npc1.npy, provenance.json
    threads:   from the node's settings
    resources: runtime (e.g. 23h50), mem_mb, slurm_partition: from the node's settings
    retries:   max_attempts from the node's settings (e.g. 6)
```

### One attempt

1. Read `state.json`.
   - Missing: fresh start with the full sweep schedule.
   - `converged` but no density matrices: run `block2main` for the 1-particle density matrix only
     (`restart_onepdm`).
   - Otherwise: copy the last complete MPS backup into scratch, then write a `dmrg.conf` with
     `fullrestart` and a schedule trimmed to the remaining sweeps (bond-dimension steps and the
     two-site to one-site switch shifted by `sweeps_done`). The loaded MPS is used as it is: no
     bond-dimension change before the first resumed sweep.
2. Run `block2main` and follow its output. After each completed sweep, update `state.json`
   atomically (write a temporary file, then rename) with `sweeps_done`, the energy, the bond
   dimension and the complete backup slot.
3. On convergence or at the end of the schedule, write the declared outputs. Reaching the end of
   the schedule without converging is not a failure: `result.json` says `"converged": false` and
   `summary.tsv` flags it.

### Guards

- An attempt that completes no sweep fails with a clear message (one sweep exceeds the walltime:
  lower `max_bond_dim` or add threads) instead of using up the retries.
- A job killed while writing a backup leaves the other slot intact; the worst case loses one sweep.
- Real errors (bad input, a `block2` crash) fail within minutes; `--keep-going` lets the rest of
  the study continue.
- The local executor uses the same protocol: interrupt, rerun, resume.

### Relation to block2's restart procedure

The block2 tutorial [Restarting DMRG](https://block2.readthedocs.io/en/latest/tutorial/restarting-dmrg.html)
describes the clean restart from a backup: copy the `restart_dir` backup into scratch, load the
MPS, keep the original schedule and pass the index of the next sweep (`sweep_start`). Because the
backup is taken at the end of a sweep, block2 determines the sweep direction itself. The MPS must
not be adjusted after loading.

The tutorial uses the `pyblock2` Python API. **AEGISS never imports `block2` or `pyblock2`
(`AGENTS.md`), with no exception for restarts**: every step of the procedure is translated to
`block2main` input keywords, and the experiment below uses `block2main` only. The trimmed schedule
is the `block2main` translation of `sweep_start`, unless the experiment finds a direct keyword.

Only end-of-sweep backups are resumed. Restarting from the middle of a sweep (the tutorial's
other case) needs the scratch folder of the killed job, which lives on node-local disk and is lost.

### Experiment before implementation

The `block2main` documentation does not settle these points. A throwaway experiment, driving
`block2main` only, must answer them before the step is built:

1. Whether `fullrestart` reads an MPS copied from `restart_dir`, or needs a particular tag or
   layout (`mps_tags`).
2. Whether `block2main` accepts a starting sweep index (the `sweep_start` of the tutorial); if not,
   how sweep counting and the schedule behave on restart, so the trimmed schedule matches the
   original one. Also that a one-site MPS resumes in one-site mode, and that the loaded MPS is not
   adjusted before the first sweep.
3. That orbital reordering (`reorder`) gives the same site order on every restart.
4. That sweep completion is detectable from `block2main`'s output, and that a backup interrupted
   mid-write does not corrupt the other slot.
5. That an interrupted-and-resumed run matches an uninterrupted one within a set tolerance.

Fallbacks, all through `block2main`:

- If the schedule cannot be resumed exactly (point 2): resume coarsely by loading the last MPS
  and running a short fixed schedule at the final bond dimension until convergence. The result
  records `resume_mode: "coarse"`.
- If no backup can be loaded (point 1): there is no mid-run resume. The `dmrg` step then fails
  at the walltime with guidance (more threads, a lower `max_bond_dim`, a longer partition), and
  the restart protocol is dropped from v1.

## Provenance

### `provenance.json`

A declared output of every step, written last and atomically; its presence marks the step as
complete, also for readers without Snakemake.

```json
{
  "schema": "aegiss.provenance/1",
  "node": {"step": "sel", "step_version": 1, "key": "<sha256>", "settings": {}, "pins": {}},
  "inputs": {
    "dmrg": {"key": "<sha256>", "path": "../../dmrg/c3d4...", "files": {"pdm1.npy": "<sha256>"}}
  },
  "outputs": {"active_space.json": {"sha256": "<sha256>", "bytes": 412}},
  "environment": {"aegiss": "...", "pyscf": "...", "block2": "...", "numpy": "...",
                  "python": "...", "omp_num_threads": 32, "mem": "...", "host": "..."},
  "execution": {"executor": "slurm", "started": "...", "finished": "...", "wall_s": 81234,
                "attempts": [{"slurm_job_id": "4711", "node": "...", "start": "...",
                              "end": "...", "sweeps_done": 9}]}
}
```

- A step folder can serve several studies, so it does not name a study; studies keep their own
  record under `studies/<study>/`.
- Every declared output is hashed, and so are the DMRG `FCIDUMP` and `dmrg.conf`. MPS backups
  and `state.json` are working state and are not.

### Integrity

- A step hashes its inputs and compares them with the checksums its parents recorded. A mismatch
  is an error, never a silent recompute.
- `aegiss trace <path>` prints the chain from a result back to its geometry or chkfile.
- `aegiss verify <path | study>` rehashes every file in the chain and reports differences.

### Claims

Snakemake locks its working directory, not the store. Before computing, a step creates
`store/<step>/<h>/.claim/` with `mkdir` (atomic, also on NFS), holding
`{study, slurm_job_id, host, started}`.

- A held claim makes the second job fail quickly, naming the study and job that hold it; a rerun
  later picks up the finished result.
- A claim whose job SLURM no longer knows (`squeue`/`sacct`) is stale and is taken over.
  `aegiss store unlock <h>` overrides by hand.

## Testing

| Layer | Tests |
|---|---|
| Refactor safety net | Before `select_aegiss` changes, freeze its output over the full matrix (RHF/UHF x spin-summed/spin-resolved x global/subset) on H2O and N2 in STO-3G, with DMRG mocked by fixed density matrices. The tests must pass unchanged after the refactor. |
| Steps match `select_aegiss` | The same matrix through the file-based steps gives identical selections. |
| Hashing | Fixed expected keys for reference settings, so a key change is deliberate; key order, float formatting and default filling. |
| File formats | Chkfile round-trips (RHF, UHF, localized, and solvent and X2C wrappers); save/load round-trips for every schema; the empty-screening and split-UHF cases; render file names. |
| SCF restart | An SCF stopped after a few iterations (small `max_cycle`) and resumed reaches the same energy as an uninterrupted run; the step's key does not change; provenance records the resume. |
| DMRG restart | A fake `block2main` on `PATH` that simulates sweeps, writes backups and can be killed at any point, mid-backup included: resume, schedule trimming, the no-progress guard, density-matrix-only runs, claims. |
| DMRG restart, real | One `@pytest.mark.slow` test, through `block2main` only: a small CAS killed after k sweeps and resumed with the procedure above, compared with an uninterrupted run and with PySCF FCI. |
| Integrity | `verify` and `trace` on a tampered store and a broken chain; two processes racing for one claim. |
| Workflow | The shipped Snakefile with the local executor on a tiny study and the fake `block2main`: store layout, `summary.tsv`, `results/` run folders and `run.json`, and no DMRG rerun when only `fraction` changes (Snakemake dry-run job list). |
| SLURM | Not in CI. A documented manual checklist: submission, a forced timeout and resume, a concurrent claim. |

New `just` recipes for any new CI lane (for example `test-workflow`); `just build --all-extras`
installs the `workflow` extra. `just sbom sbom-check` must pass with Snakemake's dependencies.

## Documentation

- `docs/workflow.md`: concepts (store, steps, study, manifest), a local run, the SLURM profile,
  restart behaviour, provenance, `trace` and `verify`.
- `docs/workflow_formats.md`: every file format and JSON schema.
- API reference pages for `aegiss.steps` and `aegiss.workflow`; `mkdocs.yml` navigation.
- `docs/installation.md`: the `workflow` extra, `$TMPDIR` and scratch on SLURM.
- A short section in `README.md`.
- `examples/workflow_study.py`: a tiny study, run in CI by `just example`.
- Code blocks that need SLURM are marked `notest`.

## Delivery

0. Experiment (no PR, `block2main` only): the restart questions; findings in `local/`, plan
   adjusted if a fallback is needed.
1. `refactor(active_space): ♻️ expose the AEGISS pipeline steps`: safety-net tests, the DMRG split,
   public helpers, `select_aegiss` rebuilt from them. No behaviour change.
2. `feat(steps): ✨ add file formats, hashing and provenance for pipeline steps`
3. `feat(steps): ✨ add file-based scf, localize, select and hamiltonian steps`: the `scf` step
   includes the SCF restart.
4. `feat(steps): ✨ add a resumable DMRG step`: after 0 and 2; can run alongside 3.
5. `feat(workflow): ✨ run AEGISS studies with Snakemake on SLURM`: `Study`/`Run`, manifest,
   Snakefile, profile, `aegiss run`, summary, `trace`/`verify`, the `workflow` extra, docs.

No breaking changes and no deprecations: `select_aegiss` and `run_dmrg_subprocess` keep their
signatures; everything else is new API.

## Out of scope for v1

- ROHF references (untested in AEGISS).
- TREXIO import and export.
- Reconciling UHF alpha/beta selections into one spatial active space.
- Store cleanup (deleting results no study references).
- A queryable index across runs.
- SLURM in CI.
