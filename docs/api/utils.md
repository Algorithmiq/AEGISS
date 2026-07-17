# Utilities

Helper functions for orbital export, plotting, MO composition, and entropy data.

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

::: aegiss.utils.mo_composition.mo_ao_contributions
    options:
      heading_level: 3

---

## Plotting

::: aegiss.utils.plotting.soe_heatmap
    options:
      heading_level: 3

::: aegiss.utils.plotting.th_plots
    options:
      heading_level: 3

::: aegiss.utils.plotting.plot_ao_contribution
    options:
      heading_level: 3

::: aegiss.utils.plotting.plot_pie_contribution
    options:
      heading_level: 3

---

## MolTagger

See the [MolTagger](../moltagger.md) page for the full reference and usage guide.
