# Utilities

Helper functions for orbital export, plotting, MO composition, entropy data, and
runtime configuration.

```python
from aegiss.utils import (
    dump_qmi,
    get_orbital_spaces,
    get_subspaces_from_cas,
    plot_ao_contribution,
    plot_pie_contribution,
    print_banner,
    save_orbitals,
    soe_heatmap,
    th_plots,
)
from aegiss.utils.dump import save_orbitals_from_entropies
from aegiss.utils.mo_composition import mo_ao_contributions
```

## Orbital spaces

::: aegiss.utils.orbital_indices.get_orbital_spaces
    options:
      heading_level: 3

::: aegiss.utils.orbital_indices.get_subspaces_from_cas
    options:
      heading_level: 3

::: aegiss.utils.orbital_indices.valence_occupied_orbitals
    options:
      heading_level: 3

::: aegiss.utils.orbital_indices.resolve_nelecas
    options:
      heading_level: 3

::: aegiss.utils.orbital_indices.combine_candidate_entropies
    options:
      heading_level: 3

---

## Orbital index base

Display and input conversion for the configured orbital index base; see
[Orbital index base](../orbital_indexing.md).

::: aegiss.utils.orb_display.orbital_base
    options:
      heading_level: 3

::: aegiss.utils.orb_display.to_display
    options:
      heading_level: 3

::: aegiss.utils.orb_display.to_display_all
    options:
      heading_level: 3

::: aegiss.utils.orb_display.from_display
    options:
      heading_level: 3

::: aegiss.utils.orb_display.from_display_all
    options:
      heading_level: 3

::: aegiss.utils.orb_display.reset_cache
    options:
      heading_level: 3

---

## Orbital export

::: aegiss.utils.dump.save_orbitals
    options:
      heading_level: 3

::: aegiss.utils.dump.save_orbitals_from_entropies
    options:
      heading_level: 3

---

## Entropy and mutual information

::: aegiss.utils.dump.dump_qmi
    options:
      heading_level: 3

---

## MO composition

A text/table companion to the AO-composition plots ([`plot_ao_contribution`](#plotting)
and [`plot_pie_contribution`](#plotting) below): it prints and returns the top-N atomic
orbital contributions to each MO.

::: aegiss.utils.mo_composition.mo_ao_contributions
    options:
      heading_level: 3

---

## Plotting

The examples below are produced by
[`docs/scripts/generate_doc_plots.py`](https://github.com/Algorithmiq/AEGISS/blob/main/docs/scripts/generate_doc_plots.py)
from a real benzene calculation (RHF/cc-pVDZ, a DMRG in the frontier CAS(30,30),
and an AO projection onto the `minao` reference). Re-run that script to refresh them.

::: aegiss.utils.plotting.soe_heatmap
    options:
      heading_level: 3

**Example** — single-orbital entropies $S(1)_i$ across the benzene frontier CAS(30,30).
The top row shows each orbital's entropy; the lower rows mark which survive the threshold
$\tau_E$ and which are left empty. The entanglement concentrates on the π-type frontier
orbitals (around 19–22), with most of the 30 CAS orbitals nearly inactive.

![SOE heatmap example](../assets/images/example_soe_heatmap.png)

::: aegiss.utils.plotting.th_plots
    options:
      heading_level: 3

**Example** — the number of retained orbitals as the entropy threshold is swept from
0 to $S(1)_{\max}$. Two plateaus emerge — a broad CAS(6,6) at intermediate thresholds
and a stable CAS(4,4) at higher ones — the latter being the benzene π active space.

![TH plot example](../assets/images/example_th_plot.png)

::: aegiss.utils.plotting.plot_ao_contribution
    options:
      heading_level: 3

**Example** — projection weight of each frontier MO onto three AO manifolds
(`C 2pz`, `C 2s`, `H 1s`). The π orbitals (19–22) light up in the `C 2pz` row, while
the σ framework spreads over `C 2s`/`H 1s`.

![AO contribution heatmap example](../assets/images/example_ao_contribution_heatmap.png)

::: aegiss.utils.plotting.plot_pie_contribution
    options:
      heading_level: 3

**Example** — per-orbital composition of the benzene valence σ orbitals, showing how
each mixes `C 2s`, `C 2px`, `C 2py` and `H 1s` character.

![Pie composition example](../assets/images/example_pie_composition_alpha.png)

::: aegiss.utils.plotting.mi_plot
    options:
      heading_level: 3

::: aegiss.utils.plotting.entanglement_diagrams
    options:
      heading_level: 3

---

## Runtime configuration

Helpers that read runtime settings from environment variables, configure logging,
and print the startup banner.

::: aegiss.utils.env.mem
    options:
      heading_level: 3

::: aegiss.utils.env.omp_num_threads
    options:
      heading_level: 3

::: aegiss.utils.env.scratch_dir
    options:
      heading_level: 3

::: aegiss.utils.env.pyscf_max_memory
    options:
      heading_level: 3

::: aegiss.utils.logging.configure_logging
    options:
      heading_level: 3

::: aegiss.utils.banner.print_banner
    options:
      heading_level: 3

---

## MolTagger

See the [MolTagger](../moltagger.md) page for the full reference and usage guide.
