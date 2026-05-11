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

`block2` is fetched from its own package index declared in `pyproject.toml` —
`uv sync` handles it automatically.

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

```python
import os
os.environ["OMP_NUM_THREADS"] = "8"
os.environ["SCRATCH_DIR"] = "/scratch/my_job"
os.environ["MEM"] = "32"
os.environ["PYSCF_MAX_MEMORY"] = "16000"
```

## Logging

AEGISS uses Python's standard `logging` module. The library never configures handlers
or formatters — it only emits records through named loggers under the `aegiss.*`
hierarchy.

=== "Script"
    ```python
    import logging
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ```

=== "Debug (per-orbital detail)"
    ```python
    import logging
    logging.basicConfig(level=logging.DEBUG, format="%(message)s")
    ```

=== "Silence in notebooks"
    ```python
    logging.getLogger("aegiss").setLevel(logging.WARNING)
    ```

=== "structlog"
    ```python
    import structlog
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(logging.INFO))
    # import aegiss after configuring structlog
    ```

## Development

```bash
uv sync                   # install all dependency groups
pre-commit install        # install git hooks
pytest                    # run test suite
pytest --cov=aegiss       # run with coverage report
```

Code style is enforced by **ruff** (linting + formatting) and **mypy** (strict type
checking). Both run automatically as pre-commit hooks and in CI.

To build and preview the documentation locally:

```bash
uv sync --group docs
mkdocs serve
```
