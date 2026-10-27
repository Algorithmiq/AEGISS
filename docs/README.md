# Docs

This directory contains the source for the AEGISS documentation site, built with
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and served at
<https://algorithmiq.github.io/AEGISS/>.

## Generating the docs locally

**1. Install the `docs` dependency group**

```bash
just build --group docs
```

This pulls in `mkdocs-material`, `mkdocs-jupyter`, and `mkdocstrings[python]`.

**2. Preview with live reload**

```bash
just serve-docs
```

Open <http://127.0.0.1:8000> in your browser. The site rebuilds automatically
whenever you save a file.

**3. Build a static copy**

```bash
just docs
```

Output lands in `site/` (git-ignored). The build is `--strict`, with link and anchor
validation, exactly as in CI. `just doctest` runs the Python blocks of the pages and
`just check-doc-links` checks the external links of the built site (needs
[lychee](https://lychee.cli.rs)). See the [Documenting](documenting.md) page for the
conventions.

## Project layout

```
mkdocs.yml            # site configuration and nav tree
docs/
  index.md            # home page
  installation.md     # installation, environment variables and logging
  orbital_indexing.md # 0- vs 1-based orbital indices
  uhf.md              # unrestricted (UHF) references
  moltagger.md        # MolTagger interactive viewer
  contributing.md     # developer guide: set up, recipes, pull requests
  developer.md        # developer guide: architecture and extending
  testing.md          # developer guide: tests
  documenting.md      # developer guide: docstrings and doc pages
  tutorials/
    index.md          # tutorials overview
    01_scf_and_localization.ipynb
    02_dmrg.ipynb
    03_ao_projection.ipynb
    04_aegiss.ipynb
    05_unrestricted.ipynb
    data/             # XYZ geometries used by the notebooks
  api/
    index.md          # API overview
    active_space.md
    config.md
    initial_guess.md
    wavefunction.md
    localization.md
    utils.md
  assets/images/      # screenshots and example plots
  javascripts/
    mathjax.js        # MathJax configuration
  scripts/
    generate_doc_plots.py  # regenerates the example plots in assets/images/
```

## Regenerating the example plots

The diagnostic plots shown on the [Utilities API page](api/utils.md) are checked into
`assets/images/`. To refresh them from a real benzene calculation (RHF/cc-pVDZ, DMRG in
the frontier CAS(30,30), and an AO projection onto `minao`):

```bash
python docs/scripts/generate_doc_plots.py
```

This needs block2 installed in the same environment (see
[Installation](installation.md#block2-dmrg)), takes a few minutes, and writes
its DMRG scratch to `tmp/_doc_plots_scratch/` (git-ignored). Files under
`docs/scripts/` are excluded from the built site via `exclude_docs` in `mkdocs.yml`.

## Adding a new page

1. Create a Markdown file (or Jupyter notebook) under `docs/`.
2. Add an entry to the `nav:` section in `mkdocs.yml`.
3. Run `mkdocs serve` to verify it renders correctly.

## API reference pages

API pages use `mkdocstrings` directives that pull docstrings directly from the
source code, so keeping docstrings up to date is all that is needed to keep the
reference accurate. Example directive:

```markdown
::: aegiss.active_space.aegiss.select_aegiss
```

Docstrings follow the **Google** style (configured in `mkdocs.yml`).

## Publishing

`.github/workflows/docs.yml` builds the site, runs the doc code blocks and the
examples, and checks the links on every pull request. On `main` it also deploys the
site to GitHub Pages (Settings → Pages → Source: GitHub Actions).
