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

The CI workflows call the same recipes, so a red CI job can be reproduced locally by
running the recipe it ran.

## Pull requests

- Branch off `main` as `<type>/<short-description>`; commits to `main` are blocked.
- Commit messages and PR titles follow
  `<type>(<optional scope>): <gitmoji> <description>`, e.g.
  `fix(projection): 🐛 normalise AO weights`.
- Fill in the PR template. Changes to the API or to user-facing behaviour come with
  tests and with the matching documentation update.
