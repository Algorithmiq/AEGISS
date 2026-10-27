# Contributing to AEGISS

Thanks for your interest in contributing to `AEGISS`!

First time contributing to an open source project? Use
[this guide](https://opensource.guide/how-to-contribute/) to learn more.

## How to contribute

You can contribute in various ways to all activities we'd like to get help with.
Some examples of ways to contribute:

- Report bugs
- Suggest new features
- Testing and providing feedback
- Improving documentation
- Contributing code

For bugs and feature requests, please use the
[issue templates](https://github.com/Algorithmiq/AEGISS/issues/new/choose). Before
opening an issue, check the [documentation](https://algorithmiq.github.io/AEGISS/)
and the existing issues.

## Contributor License Agreement (CLA)

We require an Individual CLA for all external contributions.

In plain language: by signing the CLA, you confirm that you have the right to
submit your contribution and you allow the project maintainers to use and
redistribute it under the project license.

The Individual CLA text is in the [CLA.md file](CLA.md).

If you're contributing on behalf of your employer, contact
[cla@algorithmiq.fi](mailto:cla@algorithmiq.fi) to arrange a Corporate CLA.

## Code of Conduct

Please read and follow the [Code of Conduct](CODE_OF_CONDUCT.md), which applies for
all interactions in this repository, including issues and PRs.

## Development setup

AEGISS requires Python ≥ 3.12 and uses [uv](https://github.com/astral-sh/uv) for
dependency management.

```bash
git clone https://github.com/Algorithmiq/AEGISS.git
cd AEGISS
uv sync --all-groups --all-extras   # dev, test, docs, interactive groups + viz extra
source .venv/bin/activate
prek install                        # install the git hooks
```

A bare `uv sync` installs only the default `dev` group (ruff, ty, prek, pytest); the
`test`, `docs`, and `interactive` groups are opt-in.

block2 is not a dependency, so none of the above installs it. You need it only to run
DMRG (the tutorials, `main.py`, the DMRG integration tests); the rest of the test suite
mocks it. Install it into the same environment:

```bash
uv pip install block2==0.5.4
```

`uv sync` removes it again, so reinstall it after syncing or use `uv sync --inexact`.

See [docs/installation.md](docs/installation.md) for the environment variables that
control memory and scratch space (`OMP_NUM_THREADS`, `SCRATCH_DIR`, `MEM`,
`PYSCF_MAX_MEMORY`, `AEGISS_ORB_BASE`).

## Making changes

1. **Branch off `main`.** Direct commits to `main` are blocked by a git hook.
   Branch names may only contain letters, digits, `_`, `-`, and `/` — we use
   `<type>/<short-description>`, e.g. `feat/spin-entropy-screening`.
2. **Write [conventional commit](https://www.conventionalcommits.org) messages** with a
   [gitmoji](https://gitmoji.dev) — `<type>(<optional scope>): <gitmoji> <description>`,
   e.g. `fix(projection): 🐛 normalise AO weights`. Types: `feat`, `fix`, `docs`,
   `style`, `refactor`, `perf`, `test`, `build`, `ci`, `chore`; add `!` for breaking
   changes. PR titles follow the same format.
3. **Add or update tests** covering your change (see below).
4. **Update the docs** — docstrings for API changes, and the relevant page under
   [docs/](docs/) for user-facing behaviour.
5. **Open a pull request** against `main` and fill in the PR template.

### Code style

Style is enforced by **ruff** (linting + formatting) and **ty** (type checking), both
configured in [pyproject.toml](pyproject.toml). They run as git hooks, managed by
[prek](https://github.com/j178/prek) from [prek.toml](prek.toml), and in CI on every
pull request:

```bash
prek run --all-files
```

Public functions and classes need type annotations and **Google-style** docstrings —
the API reference is generated from them by `mkdocstrings`.

Every Python file starts with the Apache-2.0 licence header from
[.github/license-header.txt](.github/license-header.txt). The `insert-license` hook adds
it to new files, so commit the file again after the hook has run.

### Tests

```bash
just test                           # full suite
just coverage                       # with branch coverage
just test -m "not slow"             # skip tests marked slow (>= 1 minute)
```

The CI workflows run the same [`just`](https://just.systems) recipes; `just` with no
arguments lists them.

Tests live in [tests/](tests/), one `test_<module>.py` per source module. Mark tests
that take a minute or more with `@pytest.mark.slow`. Quantum-chemistry calculations in
tests should use small systems and minimal basis sets so the suite stays fast; prefer
mocking `block2`/DMRG calls over running them where the test isn't about DMRG itself.

### Documentation

```bash
uv sync --group docs
mkdocs serve                        # live-reload preview at http://127.0.0.1:8000
mkdocs build                        # write the static site to site/
```

See [docs/README.md](docs/README.md) for the docs layout and how to add a page. If you
add a page or a tutorial notebook, add it to the `nav:` tree in
[mkdocs.yml](mkdocs.yml) as well.

## Reporting a security issue

Please do **not** open a public issue for security problems. See
[SECURITY.md](SECURITY.md) for how to report them privately.
