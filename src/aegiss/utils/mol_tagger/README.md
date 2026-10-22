# MolTagger

Interactive 3D molecular viewer for selecting atoms and generating
PySCF-compatible `ao_labels` strings ready to paste into `ProjectionConfig`.

## Requirements

```bash
pip install aegiss[viz]
```

The `viz` extra pulls in `ase`, `pyvista`, `pyvistaqt`, `qtpy`, and `PySide6`
as the Qt backend.

---

## Quick start

```python
from aegiss.utils.mol_tagger import MolTagger

tagger = MolTagger("my_molecule.xyz")
tagger.show()          # opens the viewer; blocks until the window is closed
print(tagger.export_tags())  # {atom_index: "Element.N", ...}
```

or, from the repository root:

```bash
python mol_tagger.py my_molecule.xyz
```

The full guide (UI, tagging, AO-label suggestions, camera controls and the tagged XYZ
format) is in [docs/moltagger.md](../../../../docs/moltagger.md).
