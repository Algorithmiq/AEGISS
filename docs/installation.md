# Installation

AEGISS requires Python ≥ 3.12 and uses [uv](https://github.com/astral-sh/uv) for
dependency management.

```bash
git clone https://github.com/Algorithmiq/AEGISS.git
cd AEGISS
uv venv
uv sync
source .venv/bin/activate
```

## block2 (DMRG)

The DMRG step needs [block2](https://github.com/block-hczhai/block2-preview), which is
**not** a dependency of AEGISS and is not installed with it. AEGISS runs block2's
`block2main` executable as a subprocess, so install block2 into the same environment:

```bash
uv pip install block2==0.5.4   # or: pip install block2==0.5.4
```

0.5.4 is the release AEGISS is validated with; DMRG results can change between block2
versions. block2 publishes wheels for Linux (x86_64, aarch64) and macOS (arm64) up to
CPython 3.14, and `block2main` needs an MPI runtime (e.g. OpenMPI) on the system.

`uv sync` removes packages it does not manage, so reinstall block2 after syncing, or
sync with `uv sync --inexact`. `uv run` leaves it in place.

Without block2, everything except the DMRG step works: SCF, localization, AO
projection, MolTagger and plotting. `run_dmrg_subprocess` and `select_aegiss` raise a
`RuntimeError` naming the missing `block2main`.

## MolTagger: interactive atom tagging

An interactive 3D molecular viewer for selecting atoms and generating PySCF-compatible
`ao_labels` strings ready to paste into `ProjectionConfig`. Requires the `viz` extra:

```bash
uv sync --extra viz
```

## Environment variables

Set these before running any calculation to control memory and scratch space:

| Variable | Default | Description |
|---|---|---|
| `OMP_NUM_THREADS` | `4` | OpenMP threads used by block2 |
| `SCRATCH_DIR` | `./tmp` | Directory for DMRG scratch files (MPS/MPO) |
| `MEM` | `4` | Memory budget in **GiB** for block2's internal stack |
| `PYSCF_MAX_MEMORY` | `4000` | PySCF memory budget in **MB** |
| `AEGISS_ORB_BASE` | `0` | Base (`0` or `1`) for displaying and interpreting orbital indices — see [Orbital index base](orbital_indexing.md) |

```python
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["SCRATCH_DIR"] = "/scratch/my_job"
os.environ["MEM"] = "32"
os.environ["PYSCF_MAX_MEMORY"] = "16000"
```

## Logging

AEGISS uses [structlog](https://www.structlog.org/) over Python's stdlib logging.
Call `configure_logging` once at the top of your script, **before** any AEGISS
import that triggers logging:

```python
import logging
from aegiss.utils.logging import configure_logging

configure_logging(level=logging.INFO)
# configure_logging(level=logging.DEBUG)  # for per-orbital detail
```

Output goes to **stdout** as plain timestamped text. Redirect it to control the
destination:

```bash
python main.py > log.out         # local — save to file
#SBATCH -o slurm-%j.out          # SLURM — output in slurm-<jobid>.out
```

`configure_logging` silences the noisy `matplotlib` and `h5py` loggers automatically.

**Silence AEGISS selectively (e.g. in a notebook):**

```{.python continuation}
logging.getLogger("aegiss").setLevel(logging.WARNING)
```

## Development

```bash
uv sync --all-groups --all-extras   # install every dependency group (dev, test, docs, …) + extras
uv pip install block2==0.5.4        # only for DMRG runs and the local DMRG integration tests
prek install                        # install git hooks
pytest                              # run test suite
pytest --cov=aegiss                 # run with coverage report
```

A bare `uv sync` installs only the default `dev` group (ruff, ty, prek, pytest); the
`test`, `docs`, and `interactive` groups are opt-in, hence `--all-groups` above (or, more
narrowly, `uv sync --group test` to just run the suite).

Code style is enforced by **ruff** (linting + formatting) and **ty** (type checking).
Both run automatically as git hooks (via [prek](https://github.com/j178/prek)) and in CI.

To build and preview the documentation locally:

```bash
uv sync --group docs
mkdocs serve
```
