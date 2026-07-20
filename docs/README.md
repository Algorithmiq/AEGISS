# Docs

This directory contains the source for the AEGISS documentation site, built with
[MkDocs Material](https://squidfunk.github.io/mkdocs-material/) and served at
<https://algorithmiq.github.io/AEGISS/>.

## Generating the docs locally

**1. Install the `docs` dependency group**

```bash
uv sync --group docs
```

This pulls in `mkdocs-material`, `mkdocs-jupyter`, and `mkdocstrings[python]`.

**2. Preview with live reload**

```bash
mkdocs serve
```

Open <http://127.0.0.1:8000> in your browser. The site rebuilds automatically
whenever you save a file.

**3. Build a static copy**

```bash
mkdocs build
```

Output lands in `site/` (git-ignored). Use this to verify the final HTML before
publishing.

## Project layout

```
mkdocs.yml            # site configuration and nav tree
docs/
  index.md            # home page
  installation.md     # installation and environment variables
  moltagger.md        # MolTagger interactive viewer
  tutorials/
    index.md          # tutorials overview
    01_scf_and_localization.ipynb
    02_dmrg.ipynb
    03_ao_projection.ipynb
    04_aegiss.ipynb
  api/
    index.md          # API overview
    active_space.md
    config.md
    initial_guess.md
    wavefunction.md
    localization.md
    utils.md
  javascripts/
    mathjax.js        # MathJax configuration
```

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

The site is deployed automatically by CI on every push to `main`. To trigger a
manual deploy from your local machine:

```bash
mkdocs gh-deploy
```

This builds the site and force-pushes it to the `gh-pages` branch.
