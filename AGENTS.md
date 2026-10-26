# AEGISS agent guide

A Python library for automated active-space selection. AEGISS narrows a candidate orbital space
with two criteria: single-orbital entanglement entropy from a DMRG run (`block2`) and projection
of the molecular orbitals onto a user-chosen atomic-orbital manifold (`pyscf`).
`aegiss.active_space.aegiss` drives the workflow; the configuration lives in the dataclasses of
`aegiss.config_dataclasses`.

## Rules

- Type hints in the code, Google-style docstrings without type hints in the signature line. The
  API reference under `docs/api/` is generated from the docstrings by `mkdocstrings`, so they
  must be accurate.
- Use snake case for variables and functions, camel case for classes.
- Comment invariants, contracts and non-obvious choices only. Never narrate the code.
- Orbital indices are 0-based internally. User-facing indices honour `AEGISS_ORB_BASE`; convert
  at the boundary, never in the middle of the algorithm.
- Tests use small molecules and minimal basis sets. Mock `block2`/DMRG unless the test is about
  DMRG; mark anything that runs for a minute or more with `@pytest.mark.slow`.
- **Run `prek run --all-files` and the relevant tests before pushing, and fix every
  finding.** `lint.yml` runs the same hooks over the PR's commit range, so a skipped lint is a
  red PR.
- Changes to the API or to user-facing behaviour should be reflected in `docs/` (and in the
  `mkdocs.yml` nav when you add a page) and, if relevant, in `README.md`.
- Python blocks in `docs/` and `README.md` run as tests (`just doctest`); mark a block that cannot
  run on its own `{.python notest}` (`python notest` in `README.md`) and one that continues the
  previous block `{.python continuation}`. `examples/` scripts run in CI (`just example`).

## Git and PRs

- Branch off `main` as `<type>/<short-description>`; direct commits to `main` are blocked.
- Commits and PR titles: `<type>(<optional scope>): <gitmoji> <description>`. Types: `feat`
  (minor), `fix` (patch), `docs`, `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`;
  `!` for breaking changes.
- Agent commits carry `Assisted-by: <harness>:<model>` and no `Co-authored-by`.
- Follow `.github/PULL_REQUEST_TEMPLATE.md`; prefix agent-authored descriptions, issues and
  comments with `:robot: _AI text below_ :robot:`.
- PR labels are applied by `.github/labeler.yml` and issue labels by
  `.github/workflows/issue-labeler.yml`; the release notes are grouped by label
  (`.github/release.yml`).

## Layout

Python package `src/aegiss/`: `active_space/` (the AEGISS driver, entropy screening, AO
projection), `config_dataclasses/`, `initial_guess/` (SCF), `localization/` (localized, natural
and frozen natural orbitals), `wavefunction/` (DMRG via `block2main`), `utils/` (I/O, plotting,
environment, MolTagger GUI). Tests in `tests/`, one `test_<module>.py` per module. Docs pages
and tutorial notebooks in `docs/`, site config in `mkdocs.yml`.

`scripts/` is local developer scratch: it is gitignored and unlinted. Never edit, move or commit
it. The same goes for `local/` (generated cubes, plots, FCIDUMPs and test runs). `tools/` holds
tracked repository helpers used by the hooks and CI. `docs/scripts/` is tracked and linted.

## Commands

We use `uv` for environment management and [`just`](https://just.systems) as the command
runner; `just` with no arguments lists the recipes.

```bash
just build --all-groups --all-extras
just test -m "not slow"
just lint
just fmt
```

CI workflows call `just` recipes instead of running commands themselves, so a red CI lane can
be reproduced locally with the same recipe; `tools/check-workflow-commands.py` (a prek hook)
enforces it. A new command goes into the `justfile` first.

## Notes

- `block2` is pinned to an exact release from PyPI: DMRG results depend on its version, so bump
  it deliberately.
- DMRG runs `block2main` as a subprocess (`wavefunction/dmrg_subprocess.py`), not through the
  `block2` Python API. Its working files go under `SCRATCH_DIR` (default `./tmp`, gitignored).
- Runtime environment variables (`OMP_NUM_THREADS`, `SCRATCH_DIR`, `MEM`, `PYSCF_MAX_MEMORY`,
  `AEGISS_ORB_BASE`) are read in `utils/env.py` and documented in `docs/installation.md`.
- MolTagger (`utils/mol_tagger/`) needs the `viz` extra and goes through `qtpy`; never import a
  Qt binding directly.
- Ruff skips the notebooks in `docs/tutorials/`, and they are not executed when the docs are
  built, so check them by hand when you change an API they use.
