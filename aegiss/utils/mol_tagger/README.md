# MolTagger

Interactive 3D molecular viewer for selecting atoms and generating
PySCF-compatible `ao_labels` strings ready to paste into `ProjectionConfig`.

## Requirements

```bash
pip install aegiss[viz]
```

The `viz` extra pulls in `ase`, `pyvista`, `pyvistaqt`, `qtpy`, and `PyQt5`
as the Qt backend.

---

## Quick start

```python
from aegiss.utils.mol_tagger import MolTagger

tagger = MolTagger("my_molecule.xyz")
tagger.show()          # opens the viewer; blocks until the window is closed
print(tagger.export_tags())  # {atom_index: "Element.N", ...}
```

---

## UI overview

```
┌─────────────────────────────────────────────┬──────────────┐
│                                             │  Atoms       │
│                                             │  Tag Selected│
│           3D viewport                       │  Atom Info   │
│        (PyVistaQt / VTK)                    │  AO Labels   │
│                                             │  Export      │
└─────────────────────────────────────────────┴──────────────┘
```

### Selecting atoms

| Action | Effect |
|---|---|
| **Right-click** atom in 3D view | Toggle selection (click, not drag) |
| **Click row** in Atoms panel | Toggle selection from the list |
| **Select All / Clear** buttons | Bulk select / deselect |

Selected atoms are highlighted with a translucent red halo in the 3D view and
highlighted in the atom list simultaneously. Both views stay in sync.

### Tagging atoms

1. Select one or more atoms.
2. Type a whole-number label in the **Tag Selected** text box (e.g. `1`).
3. Press **Apply Tag** or `Return`.

Each tagged atom is stored as `Element.N` (e.g. `C.1`, `Ru.2`). Tags appear
next to the atom coordinates in the Atoms list and are written to the exported
XYZ file. **Remove Tags** clears tags from the current selection.

### Atom Info panel

Shows the minimal-basis shell structure for each selected atom:

```
#4 Ru (Z=44)
  Minimal basis : 1s, 2s, 2p, 3s, 3p, 3d, 4s, 4d
  Valence       : 3d, 4s, 4d
```

---

## AO Labels panel

Automatically suggests PySCF-compatible `ao_labels` strings based on the
selected atoms and the chosen **character mode**.

### Output format

The text box displays a list ready to pass directly to `mol.search_ao_label()`:

```python
["Ru 4d", "N 2pz", "C 2pz"]
```

Press **Copy** to send this string to the system clipboard.

### Character modes

| Mode | Intended use | Logic |
|---|---|---|
| **π / Auto** | π systems, MLCT/LMCT states | Metals → d/f shell. Organic atoms → `np{axis}` where `{axis}` is the plane-normal direction detected by SVD planarity analysis. Falls back to all `np` if atoms are non-planar. |
| **d-shell** | Pure metal-centred active spaces | Returns only d/f-shell labels for transition metals. Warns if ligand atoms are also selected. |
| **σ framework** | Sigma bonding / lone pair character | Organic atoms → `ns` + the two in-plane `np` components (complement of the π normal axis). |
| **Custom** | Manual entry | Unlocks the text box for free-form input. |

### Planarity detection (π / σ modes)

The normal to the molecular plane is found by SVD of the centred position
matrix. The planarity score is `s₂ / s₁` from the singular values — near 0
means flat, ≥ 0.25 means non-planar. The pi-normal axis (x, y, or z) is
mapped directly to the PySCF orbital component label.

| Score | Verdict | Labels emitted |
|---|---|---|
| < 0.25 | Planar | `C 2pz` (or `px`/`py` as appropriate) |
| ≥ 0.25 | Non-planar | All `C 2p` (warning shown in orange) |

### Transition metal classification

Atoms are classified as **metals** (d/f shell) or **main-group** (organic)
based on atomic number:

| Range | Shell |
|---|---|
| Sc–Zn (Z 21–30) | 3d |
| Y–Cd (Z 39–48) | 4d |
| La (Z 57), Hf–Hg (Z 72–80) | 5d |
| Ce–Lu (Z 58–71) | 4f **+ 5d** |
| Ac (Z 89), Rf–Cn (Z 104–112) | 6d |
| Th–Lr (Z 90–103) | 5f **+ 6d** |

Lanthanides and actinides produce **two labels** (both the f and the d shell)
because CASSCF active spaces for f-element complexes routinely require both.

### Example: Ru–bpy complex

Select **Ru only** → `["Ru 4d"]`

Select **Ru + bpy C/N atoms** in π / Auto mode:
```python
["Ru 4d", "C 2pz", "N 2pz"]   # assuming z is the ring-plane normal
```

Select **Ce** in d-shell mode:
```python
["Ce 4f", "Ce 5d"]
```

---

## Tagged XYZ format

**Save Tagged XYZ** writes a standard XYZ file where tagged atoms use the
`Element.N` label instead of the bare element symbol:

```
12

C.1     -1.212345   0.000000   0.000000
C.1      1.212345   0.000000   0.000000
N.2     -2.345678   1.000000   0.000000
Ru.3     0.000000   0.000000   0.000000
H        1.500000   1.000000   0.000000
...
```

The file can be reloaded into MolTagger; tags are automatically recovered from
the `Element.N` labels.

---

## Module layout

| File | Responsibility |
|---|---|
| `__init__.py` | Public `MolTagger` class; Qt app lifecycle wrapper |
| `_window.py` | `MolViewWindow` — Qt main window, all panels and event handling |
| `_renderer.py` | `MolRenderer` — PyVista scene management (atoms, bonds, halos) |
| `_ao_suggest.py` | Chemistry logic: d/f-shell classification, SVD planarity, label generation |
| `_atoms.py` | Bond detection, minimal-basis table, XYZ I/O |
| `_constants.py` | Display constants and Qt dark stylesheet |
