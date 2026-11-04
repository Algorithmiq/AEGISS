# Changelog

All notable changes to this project are listed here.

## 0.2.0 (2026-11-04)

### Breaking changes

- **entropy:** Rank single-orbital entropies over the whole window by default ([#112](https://github.com/Algorithmiq/AEGISS/pull/112))

  Code that builds `EntropyConfig` without `ranking` now thresholds over
  the whole window.
  To keep the previous selection, pass `ranking="subset"`:

  ```python
  # before (implicit subset)
  EntropyConfig(fraction=0.1, fraction_type="r")
  # after, same selection as before
  EntropyConfig(fraction=0.1, fraction_type="r", ranking="subset")
  ```

- **projection:** Reject AO labels that match no projection AO ([#113](https://github.com/Algorithmiq/AEGISS/pull/113))

  A label that matches no AO used to give an empty selection and now
  raises `ValueError`.
  Fix the label using the shells listed in the message. For a 4d/5d metal,
  either give the
  molecule an ECP or project onto the molecular basis:

  ```python notest
  # before: silently empty when Ru has no ECP
  ProjectionConfig(ao_labels=["Ru 4d"])
  # after: give Ru an ECP when building the molecule (as in the paper) ...
  mol = gto.M(atom=..., basis={"Ru": "cc-pvdz-pp", "default": "def2-svp"}, ecp={"Ru": "cc-pvdz-pp"})
  # ... or project onto the molecular basis
  ProjectionConfig(ao_labels=["Ru 4d"], basis=None)
  ```

- **projection:** Default select_by_ao_projection to a contribution of 0.5 ([#114](https://github.com/Algorithmiq/AEGISS/pull/114))

  Calls to `select_by_ao_projection` without `contribution` (and without
  `top_n`) now keep
  only orbitals with a weight of at least 0.5. To keep every orbital, as
  before, pass
  `contribution=0.0`:

  ```python notest
  # before (implicit 0.0: every orbital kept)
  select_by_ao_projection(mf, ao_labels=["C 2pz"])
  # after, same result as before
  select_by_ao_projection(mf, ao_labels=["C 2pz"], contribution=0.0)
  ```

### Fixes

- **projection:** Warn when per-label thresholds do not match the labels ([#116](https://github.com/Algorithmiq/AEGISS/pull/116))

### Documentation

- **parameters:** How to choose the AEGISS parameters ([#117](https://github.com/Algorithmiq/AEGISS/pull/117))
- **uhf:** Define the UHF active space as two independent lists ([#115](https://github.com/Algorithmiq/AEGISS/pull/115))

## 0.1.0 (2026-10-28)

### Features

- Additional localization procedure ([#46](https://github.com/Algorithmiq/AEGISS/pull/46))
- Added format ([#54](https://github.com/Algorithmiq/AEGISS/pull/54))
- Added resolution space without SOE ([#55](https://github.com/Algorithmiq/AEGISS/pull/55))
- Added beta components to `plot_ao_contributions` ([#58](https://github.com/Algorithmiq/AEGISS/pull/58))
- Add `mo_composition` ([#59](https://github.com/Algorithmiq/AEGISS/pull/59))
- Add `clusters` grouping to mo_ao_contributions ([#61](https://github.com/Algorithmiq/AEGISS/pull/61))
- Simple `chemcore` core identification ([#57](https://github.com/Algorithmiq/AEGISS/pull/57))
- Started UHF integration ([#62](https://github.com/Algorithmiq/AEGISS/pull/62))
- Orbital index mutable setup ([#65](https://github.com/Algorithmiq/AEGISS/pull/65))

### Fixes

- Improved plateau visibility
- Expose AO contributions and update outdated tutorials ([#71](https://github.com/Algorithmiq/AEGISS/pull/71))
- Use mulliken as default pop_method for Pipek-Mezey ([#72](https://github.com/Algorithmiq/AEGISS/pull/72))
- IAO shape, Pipek-Mezey occupied pop_method and plot output names ([#76](https://github.com/Algorithmiq/AEGISS/pull/76))
- **scf:** Log the mean-field energy after the SCF ([#86](https://github.com/Algorithmiq/AEGISS/pull/86))

### Documentation

- Refining `mkdocs` ([#64](https://github.com/Algorithmiq/AEGISS/pull/64))
- **localization:** Explain why the DMRG window is not a frontier window for localized orbitals ([#75](https://github.com/Algorithmiq/AEGISS/pull/75))
- Fix docstrings that no longer match the code ([#79](https://github.com/Algorithmiq/AEGISS/pull/79))
- **plotting:** Explain how to use the MI plots with an external mutual-information matrix ([#81](https://github.com/Algorithmiq/AEGISS/pull/81))
- Trim the README to an entry point and fix stale docs pages ([#77](https://github.com/Algorithmiq/AEGISS/pull/77))
- **tutorials:** Use a CAS(18,18) window in 03/04 and fix stale tutorial text ([#78](https://github.com/Algorithmiq/AEGISS/pull/78))
- Deploy the site, run the doc code blocks and add a developer guide ([#88](https://github.com/Algorithmiq/AEGISS/pull/88))
- **uhf:** Validate the entropies against PySCF FCI instead of pyblock2 ([#93](https://github.com/Algorithmiq/AEGISS/pull/93))
- **site:** Show the Apache licence on the docs site ([#101](https://github.com/Algorithmiq/AEGISS/pull/101))
- **citing:** Cite AEGISS and the work it builds on ([#102](https://github.com/Algorithmiq/AEGISS/pull/102))
- **contributing:** Document the Apache-2.0 licence header ([#104](https://github.com/Algorithmiq/AEGISS/pull/104))
