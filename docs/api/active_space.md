# Active Space

Functions for selecting active space orbitals. The high-level entry point
`select_aegiss` runs the full pipeline; the lower-level screening functions can be
called independently for custom workflows.

::: aegiss.active_space.aegiss.select_aegiss

---

::: aegiss.active_space.entropy_screening.select_by_entropy

---

::: aegiss.active_space.entropy_screening.select_by_spin_entropy

---

::: aegiss.active_space.ao_projection.select_by_ao_projection

---

## AO label format

`ao_labels` follow PySCF's `search_ao_label` convention:

```
"C 2pz"                             # this shell on every (untagged) carbon atom
["Fe 3dxz", "Fe 3dyz", "Fe 3dz^2"]  # several orbitals in one group
"C.1 2pz"                           # only carbons labelled C.1 in the geometry
```

The `Element.N` form matches atoms whose label in the geometry is `Element.N`, such as
the tags written by [MolTagger](../moltagger.md). `N` is the tag you chose, not the
position of the atom in the molecule. Tagged atoms are no longer matched by the plain
element label: with a geometry containing `C.1` and `C` atoms, `"C 2pz"` selects only
the untagged carbons.
