# Documenting

- Google-style docstrings on every public function and class. The API reference pages
  under `docs/api/` pull them in with `mkdocstrings` directives such as
  `::: aegiss.active_space.aegiss.select_aegiss`, so a new public symbol needs a
  directive on the matching page.
- Python code blocks in `docs/` and `README.md` run as tests (`just doctest`). A block
  that cannot run on its own (it needs a DMRG run or a file the reader supplies) is marked
  `notest`; a block that continues the previous one is marked `continuation`.
  Use the brace form, which Material for MkDocs renders:

````markdown
```{.python notest}
active = select_aegiss(mol, mf, dmrg_config, entropy_config, projection_config, output_config)
```
````

  In `README.md`, write `python notest` instead, so GitHub still highlights the block.
- `just docs` builds with `--strict` and link and anchor validation. The docs workflow
  also checks the external links with lychee and deploys the site to GitHub Pages from
  `main`.
- The tutorial notebooks in `docs/tutorials/` are rendered but not executed when the
  site is built; check them by hand when you change an API they use.
- Links inside the notebooks are relative (`02_dmrg.ipynb`, `../uhf.md`), so they work in
  Jupyter and on GitHub. mkdocs-jupyter does not rewrite them, so they are broken on the
  site and `.lychee.toml` skips them until the site is public and they can become
  absolute site URLs (#91).
- The example plots on the Utilities page come from `docs/scripts/generate_doc_plots.py`
  (see `docs/README.md`).
- Changes to the public API or to developer workflows update the docs, `README.md` and
  `AGENTS.md` in the same PR.
