set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set positional-arguments := true

# List the recipes.
default:
    @just --list

# Install the package; extra arguments go to `uv sync` (e.g. `just build --all-groups --all-extras`).
build *ARGS:
    uv sync "$@"

# Format the code and apply the safe lint fixes.
fmt:
    uv run --no-sync ruff format .
    uv run --no-sync ruff check --fix .

# Run every hook on the whole tree, as the lint workflow does on a PR's commits.
lint:
    uv run --no-sync prek run --all-files

# Type-check with ty.
typecheck:
    uv run --no-sync ty check

# Run the test suite; extra arguments go to pytest (e.g. `just test -m "not slow"`).
test *ARGS:
    uv run --no-sync pytest "$@"

# CI leg: the test suite with a JUnit report named after LABEL.
test-ci LABEL:
    uv run --no-sync pytest -r aR --durations=20 --durations-min=5.0 -o log_cli=false \
        --junit-xml=pytest-{{ LABEL }}.xml -o junit_family=legacy

# The test suite with branch coverage, written to coverage.xml.
coverage:
    uv run --no-sync pytest -o log_cli=false --cov --cov-branch --cov-report=xml:coverage.xml \
        --cov-report=term-missing
