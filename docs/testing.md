# Testing

```bash
just test                  # the whole suite
just test -m "not slow"    # skip the tests that take a minute or more
just coverage              # branch coverage, written to coverage.xml
just doctest               # the Python blocks of README.md and docs/
just example               # the scripts in examples/
```

- Tests live in `tests/`, one `test_<module>.py` per source module; the
  `name-tests-test` hook enforces the naming.
- Quantum-chemistry tests use small molecules and minimal basis sets. Mock `block2`/DMRG
  unless the test is about DMRG, and mark anything that runs for a minute or more with
  `@pytest.mark.slow`.
- Warnings are errors (`filterwarnings = ["error"]` in `pyproject.toml`). Fix the
  cause; when a warning comes from a dependency, add a targeted `ignore` entry with a
  comment saying why.
- CI runs the suite on Linux x86_64, Linux aarch64 and macOS arm64 with Python 3.12 and
  3.14, the platforms `block2` publishes wheels for.
