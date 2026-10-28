# Contributing

Bug reports, feature requests, documentation fixes and code are all welcome. Open
issues with the [issue templates](https://github.com/Algorithmiq/AEGISS/issues/new/choose)
and read the [contributing guide](https://github.com/Algorithmiq/AEGISS/blob/main/CONTRIBUTING.md),
the [Code of Conduct](https://github.com/Algorithmiq/AEGISS/blob/main/CODE_OF_CONDUCT.md)
and the [Contributor License Agreement](https://github.com/Algorithmiq/AEGISS/blob/main/CLA.md)
before opening a pull request.

## Set up

AEGISS uses [uv](https://docs.astral.sh/uv/) for the environment and
[just](https://just.systems) as the command runner; `just` with no arguments lists the
recipes.

```bash
git clone https://github.com/Algorithmiq/AEGISS.git
cd AEGISS
just build --all-groups --all-extras   # every dependency group and the viz extra
uv run prek install                    # git hooks: ruff, ty and the repository checks
```

In VS Code or GitHub Codespaces, the devcontainer in `.devcontainer/` (Ubuntu with uv,
just and OpenMPI) runs the two commands above when it starts. The MolTagger GUI needs a
display, so it does not run inside the container.

On Nix, `nix develop` (or [direnv](https://direnv.net) with the tracked `.envrc`) gives a
shell with uv, just, lychee and OpenMPI; run the same commands inside it.

## Day to day

| Recipe | What it does |
| --- | --- |
| `just fmt` | format with ruff and apply the safe lint fixes |
| `just lint` | run every git hook on the whole tree, as CI does on a PR |
| `just typecheck` | type-check with ty |
| `just test` | run the test suite (`just test -m "not slow"` skips the slow tests) |
| `just coverage` | the test suite with branch coverage |
| `just docs` / `just serve-docs` | build the site strictly / preview it with live reload |
| `just doctest` | run the Python blocks of `README.md` and `docs/` |
| `just example` | run the scripts in `examples/` |
| `just sbom` | write CycloneDX SBOMs of the runtime dependencies to `sbom/` |
| `just sbom-check` | check them against the license policy in `.grant.yaml` (needs [grant](https://oss.anchore.com/docs/installation/grant/)) |

The CI workflows call the same recipes, so a red CI job can be reproduced locally by
running the recipe it ran.

## Pull requests

- Branch off `main` as `<type>/<short-description>` and target `main`, bug fixes
  included; commits to `main` and to `release/*` are blocked. Only bots open PRs against
  release branches.
- PRs are squash-merged: the PR title and description become the commit on `main`, and
  the title becomes the [changelog](https://github.com/Algorithmiq/AEGISS/blob/main/CHANGELOG.md)
  entry, so write it for users. The format is
  `<type>(<optional scope>): <gitmoji> <description>`, e.g.
  `fix(projection): 🐛 normalise AO weights`, with `!` after the type for a breaking
  change (`feat!: 💥 remove the old solver API`). The PR checks workflow enforces it.

    | Type | Use for | In the changelog |
    | --- | --- | --- |
    | `feat` | new functionality | yes |
    | `fix` | bug fixes | yes |
    | `deprecate` | marking something as deprecated | yes, with the migration notes |
    | `perf` | performance improvements | yes |
    | `docs` | documentation only | yes |
    | `refactor`, `test`, `build`, `ci`, `chore`, `style` | changes users don't notice | no |

- Breaking changes and deprecations need a **Migration** section in the PR description
  saying what users have to change, ideally with a before/after example. The PR checks
  workflow requires it, and it is copied into the changelog.
- A bug fix that should also reach the current release gets the `backport` label, before
  or after merging. Features wait for the next minor release.
- Fill in the PR template. Changes to the API or to user-facing behaviour come with
  tests and with the matching documentation update.
- A new or changed runtime dependency must pass the license policy in `.grant.yaml`,
  which the SBOM workflow checks on every PR that touches the dependencies. Record an
  accepted exception in `ignore-packages` together with the reason it passes.
- AEGISS is released under the Apache License 2.0. Every Python file starts with the
  licence header in `.github/license-header.txt`; the `insert-license` git hook adds it
  to new files.

## Deprecations

Public functionality is not removed or changed without warning:

1. **Deprecate first.** The old behaviour keeps working but emits a `DeprecationWarning`
   naming the replacement and the earliest version that removes it; the docstring says
   the same. The test suite turns warnings into errors, so tests of the deprecated API
   expect the warning with `pytest.warns`.
2. **Wait at least one minor release.** Something deprecated in 0.5.0 can go in 0.6.0
   at the earliest; from 1.0 on, removals only happen in a major release.
3. **Remove.** The removal is a breaking change (`!`) with migration notes.

Names starting with `_` and anything documented as experimental can change without a
deprecation.

## Releases

AEGISS follows [semantic versioning](https://semver.org). Patch releases only fix bugs;
before 1.0, a minor release may break existing code, and the changelog lists every
breaking change first, with migration notes. Only the newest release is supported.

All work lands on `main`. Each minor release is stabilised on a `release/X.Y` branch that
only receives backported bug fixes, and a release bot does the tagging and the
bookkeeping:

1. **Start.** When `main` is ready, a maintainer checks that no deprecation is due for
   removal and runs **Actions → Release → Run workflow** on `main` (minor, or major for
   1.0 and later). The bot creates `release/X.Y`, tags the first release candidate
   `vX.Y.0rc1` and opens the release PR `chore: release vX.Y.0`.
2. **Stabilise.** Every backported fix merged into `release/X.Y` before `X.Y.0` produces
   the next release candidate (`pip install --pre aegiss==X.Y.0rcN`) and updates the
   release PR.
3. **Release.** A maintainer reviews the changelog in the release PR and merges it. The
   bot tags `vX.Y.Z`, and the tag runs `deploy.yml`: it builds the sdist and wheel,
   checks their version against the tag, creates a draft GitHub release with the files
   and CycloneDX SBOMs of the runtime dependencies attached, publishes to PyPI through
   trusted publishing with build-provenance attestations, and publishes the GitHub
   release, which makes it immutable. The bot then copies `CHANGELOG.md` back to `main`.
   If the release's docs version is missing from the site, run
   `gh workflow run docs.yml --ref vX.Y.Z`.
4. **Patch.** After `X.Y.0`, the first backport opens the release PR for `X.Y.1`; merge
   it when the fixes should ship. Patch releases have no release candidates.

`just changelog` previews the next release's entry. `CHANGELOG.md` is generated by
[git-cliff](https://git-cliff.org) from the PR titles and migration notes; fix its
wording in the release PR, never by hand on `main`. It starts at the `v0.0.0` tag, which
marks where this process was adopted.

The version comes from the git tag (hatch-vcs). Untagged commits on `main` carry the next
minor version (`0.3.0.devN` while `release/0.2` is open), those on `release/X.Y` the next
patch or release candidate.

The documentation site has one version per minor release, the newest of them under
`latest`, and `dev` for `main`. `docs.yml` updates `dev` on every push to `main` and the
release's version when it is tagged.

A published release never changes: a broken one is yanked on PyPI and replaced by a new
patch release. Check a release and its files with `gh release verify vX.Y.Z` and
`gh attestation verify <file> --repo Algorithmiq/AEGISS`.

## Dependency updates

Dependabot opens weekly grouped updates for the Python dependencies and the GitHub
Actions; ruff and ty are bumped monthly by `update-tools.yml`, together with their
prek hooks. `block2` is never bumped automatically, since DMRG results depend on its
version.
