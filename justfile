set shell := ["bash", "-eu", "-o", "pipefail", "-c"]
set positional-arguments := true

git_cliff := "git-cliff==2.14.2"

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

# Run the Python code blocks of README.md and docs/ in a scratch directory (they write files).
doctest:
    #!/usr/bin/env bash
    set -euo pipefail
    root="$(pwd)"
    cd "$(mktemp -d)"
    uv run --project "$root" --no-sync pytest --markdown-docs --markdown-docs-syntax=superfences \
        -p no:cacheprovider --rootdir "$root" -c "$root/pyproject.toml" -o log_cli=false \
        --ignore="$root/docs/scripts" --ignore="$root/docs/tutorials" \
        "$root/README.md" "$root/docs"

# Run the examples in a scratch directory (they write files).
example:
    #!/usr/bin/env bash
    set -euo pipefail
    root="$(pwd)"
    cd "$(mktemp -d)"
    for script in "$root"/examples/*.py; do
        uv run --project "$root" --no-sync python "$script"
    done

# Build the documentation into site/.
docs:
    uv run --no-sync mkdocs build --strict

# Build the site and push it to the gh-pages branch as VERSION (`dev` for main, `X.Y` for a release) with ALIASES.
docs-deploy VERSION *ALIASES:
    #!/usr/bin/env bash
    set -euo pipefail
    version="$1"
    shift
    title="$version"
    [[ "$version" == dev ]] && title="dev (unreleased)"
    uv run --no-sync mike deploy --push --update-aliases --title "$title" "$version" "$@"
    # The site root opens the newest release, or the unreleased docs until there is one
    default=dev
    if uv run --no-sync mike list latest > /dev/null 2>&1; then default=latest; fi
    uv run --no-sync mike set-default --push "$default"

# Serve the documentation with live reload.
serve-docs:
    uv run --no-sync mkdocs serve

# Check the links of the built site, external ones included. Needs `lychee` on PATH.
check-doc-links: docs
    lychee --config .lychee.toml 'site/**/*.html'

# The test suite with branch coverage, written to coverage.xml.
coverage:
    uv run --no-sync pytest -o log_cli=false --cov --cov-branch --cov-report=xml:coverage.xml \
        --cov-report=term-missing

# Preview the changelog entry of the next release, from the PR titles merged since the last one.
changelog:
    uvx --from {{ git_cliff }} git-cliff --unreleased

# Release notes for the checked-out tag TAG: its changelog entry, or the changes so far for a release candidate.
release-notes TAG:
    #!/usr/bin/env bash
    set -euo pipefail
    # cliff.toml only counts final tags as versions, so a candidate's changes are still unreleased
    if [[ "$1" == *rc* ]]; then
        uvx --from {{ git_cliff }} git-cliff --unreleased --tag "$1" --strip header
    else
        uvx --from {{ git_cliff }} git-cliff --current --strip header
    fi

# Build the sdist and wheel into dist/.
dist:
    rm -rf dist
    uv build --out-dir dist

# Write CycloneDX SBOMs of the locked runtime dependencies, without and with the viz extra, to sbom/.
sbom:
    #!/usr/bin/env bash
    set -euo pipefail
    rm -rf sbom
    mkdir sbom
    scratch="$(mktemp -d)"
    trap 'rm -rf "$scratch"' EXIT
    for flavour in core viz; do
        extra=()
        [[ "$flavour" == viz ]] && extra=(--extra viz)
        # AEGISS itself is the SBOM's root component (from pyproject.toml), not a dependency
        UV_PROJECT_ENVIRONMENT="$scratch/$flavour" uv sync --frozen --no-default-groups \
            --no-install-project ${extra[@]+"${extra[@]}"}
        # A PYTHONPATH (the Nix dev shell sets one) would add its packages to the scanned environment
        env -u PYTHONPATH uvx --from cyclonedx-bom==7.5.0 cyclonedx-py environment "$scratch/$flavour" \
            --pyproject pyproject.toml --mc-type library --output-reproducible \
            --output-file "sbom/aegiss-$flavour.cdx.json"
    done

# Check the SBOMs from `just sbom` against the license policy in .grant.yaml. Needs anchore/grant on PATH.
sbom-check:
    grant check --config .grant.yaml --disable-file-search sbom/*.cdx.json

# Move a pinned astral tool (ruff or ty) and its prek hook to VERSION together. GNU sed.
bump-tool TOOL VERSION:
    sed -i -E 's/"{{ TOOL }}==[0-9.]+"/"{{ TOOL }}=={{ VERSION }}"/' pyproject.toml
    sed -i -E '/astral-sh\/{{ TOOL }}-pre-commit"/{n; s/rev = "v[0-9.]+"/rev = "v{{ VERSION }}"/}' prek.toml
    uv lock --upgrade-package {{ TOOL }}
