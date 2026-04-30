"""Interactive molecular tagger powered by PyVista."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import ase.atoms
    import pyvista as pv

logger = logging.getLogger(__name__)


class MolTagger:
    """Interactive 3D molecular viewer with atom-tagging capabilities.

    Loads a molecule from an XYZ file and renders it in a PyVistaQt window.
    Atoms can be tagged interactively for downstream use (e.g. active-space
    selection or annotation of AEGISS outputs).

    Attributes:
        filepath: Path to the source XYZ file.
        atoms: ASE Atoms object loaded from *filepath*.
        tags: Mapping from atom index to user-assigned label.
    """

    def __init__(self, filepath: str | Path) -> None:
        """Load a molecule from an XYZ file.

        Args:
            filepath: Path to the XYZ file to visualize.

        Raises:
            ImportError: If ``ase`` is not installed.
            FileNotFoundError: If *filepath* does not exist.
        """
        try:
            import ase.io
        except ImportError as exc:
            raise ImportError(
                "MolTagger requires the 'viz' extras: "
                "pip install aegiss[viz]"
            ) from exc

        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(self.filepath)

        self.atoms: ase.atoms.Atoms = ase.io.read(str(self.filepath))
        self.tags: dict[int, str] = {}
        logger.info("Loaded %d atoms from %s", len(self.atoms), self.filepath)

    def tag(self, index: int, label: str) -> None:
        """Assign a text label to an atom.

        Args:
            index: Zero-based atom index.
            label: Arbitrary string label to attach.

        Raises:
            IndexError: If *index* is out of range for the loaded molecule.
        """
        if index < 0 or index >= len(self.atoms):
            raise IndexError(f"Atom index {index} out of range (0–{len(self.atoms) - 1})")
        self.tags[index] = label
        logger.debug("Tagged atom %d as %r", index, label)

    def _build_mesh(self) -> pv.PolyData:
        """Convert the ASE Atoms object to a PyVista point cloud.

        Returns:
            PyVista PolyData with atom positions and element labels as point data.
        """
        import numpy as np
        import pyvista as pv

        positions = self.atoms.get_positions()
        mesh = pv.PolyData(positions)
        mesh["element"] = self.atoms.get_chemical_symbols()
        mesh["index"] = np.arange(len(self.atoms))
        return mesh

    def show(self) -> None:
        """Open the interactive PyVistaQt viewer.

        Raises:
            ImportError: If ``pyvistaqt`` is not installed.
        """
        try:
            import pyvistaqt
        except ImportError as exc:
            raise ImportError(
                "MolTagger requires the 'viz' extras: "
                "pip install aegiss[viz]"
            ) from exc

        mesh = self._build_mesh()
        plotter = pyvistaqt.BackgroundPlotter(title=f"MolTagger — {self.filepath.name}")
        plotter.add_mesh(
            mesh,
            render_points_as_spheres=True,
            point_size=20,
        )
        plotter.add_point_labels(
            mesh,
            "element",
            point_size=20,
            font_size=12,
            always_visible=True,
        )
        logger.info("Viewer opened for %s", self.filepath.name)
        plotter.app.exec_()

    def export_tags(self) -> dict[int, Any]:
        """Return the current tag mapping.

        Returns:
            Dictionary mapping atom indices to their assigned labels.
        """
        return dict(self.tags)
