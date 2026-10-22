# Localization

Orbital localization methods. `localize` is a dispatcher that accepts a method name
as its first argument; each specific function can also be called directly.

```python
from aegiss.localization import localize
```

!!! warning "Localized orbitals and the DMRG window"
    The DMRG active region is a window of `ncas` orbitals taken **by index** around the
    occupied/virtual boundary. For canonical orbitals that is a Fermi-level window,
    because their index order follows the orbital energies. Localization rotates the
    occupied and virtual blocks separately: the occupied/virtual partition survives, but
    localized orbitals have no orbital energies and their order within each block is
    arbitrary, so the same index window no longer selects the frontier orbitals.
    Natural orbitals are ordered by occupation number, a different but meaningful
    criterion. Localized orbitals can still be used, but check that the window contains
    the orbitals you care about. See [Methodology](../index.md#3-single-orbital-entropy-screening).

::: aegiss.localization.orbital_localization.localize

---

::: aegiss.localization.orbital_localization.localize_boys

---

::: aegiss.localization.orbital_localization.localize_pipek_mezey

---

::: aegiss.localization.orbital_localization.localize_edmiston

---

::: aegiss.localization.orbital_localization.localize_nos

---

::: aegiss.localization.nos.compute_nos

---

::: aegiss.localization.fnos.compute_fnos
