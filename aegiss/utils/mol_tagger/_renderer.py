"""PyVista 3D renderer for MolTagger."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pyvista as pv
from ase.data import covalent_radii
from ase.data.colors import jmol_colors

from aegiss.utils.mol_tagger._constants import (
    ATOM_RADIUS_SCALE,
    BOND_RADIUS,
    COMPASS_VIEWPORT,
    HIGHLIGHT_COLOR,
    HIGHLIGHT_OPACITY,
    HIGHLIGHT_SCALE,
    SPHERE_RESOLUTION,
    TUBE_RESOLUTION,
)

if TYPE_CHECKING:
    import ase.atoms
    from pyvistaqt import QtInteractor


class MolRenderer:
    """Manages all PyVista drawing for a single molecule.

    Holds the plotter reference and atom actor handles so that individual
    atoms can be redrawn (e.g. on selection change) without re-rendering
    the entire scene.

    Attributes:
        plotter: The shared PyVistaQt interactor.
        pos: Atom positions array, shape ``(n, 3)``.
        nums: Atomic numbers array, shape ``(n,)``.
        bonds: List of ``(i, j)`` bonded-atom index pairs.
    """

    def __init__(
        self,
        plotter: QtInteractor,
        atoms: ase.atoms.Atoms,
        bonds: list[tuple[int, int]],
    ) -> None:
        """Initialise the renderer.

        Args:
            plotter: Shared QtInteractor owned by the main window.
            atoms: ASE Atoms object for the loaded molecule.
            bonds: Pre-computed bond list from :func:`detect_bonds`.
        """
        self.plotter = plotter
        self.pos: np.ndarray = atoms.get_positions()
        self.nums: np.ndarray = atoms.get_atomic_numbers()
        self.bonds = bonds
        self._atom_actors: dict[int, Any] = {}
        self._halo_actors: dict[int, Any] = {}

    def render_all(self, selected: set[int]) -> None:
        """Clear the scene and redraw all bonds and atoms from scratch.

        Args:
            selected: Set of currently selected atom indices.
        """
        self.plotter.clear()
        self._atom_actors.clear()
        self._halo_actors.clear()
        self._draw_bonds()
        for i in range(len(self.pos)):
            self.draw_atom(i, selected=i in selected)
        self.plotter.reset_camera()
        self.plotter.add_axes(
            interactive=False,
            line_width=2,
            color="white",
            xlabel="x",
            ylabel="y",
            zlabel="z",
            viewport=COMPASS_VIEWPORT,
        )

    def draw_atom(self, i: int, *, selected: bool = False) -> None:
        """(Re)draw a single atom, replacing any existing actors.

        Args:
            i: Zero-based atom index.
            selected: Whether to render a selection halo around the atom.
        """
        if i in self._atom_actors:
            self.plotter.remove_actor(self._atom_actors[i])
        if i in self._halo_actors:
            self.plotter.remove_actor(self._halo_actors[i])
            del self._halo_actors[i]

        Z = int(self.nums[i])
        r = max(covalent_radii[Z] * ATOM_RADIUS_SCALE, 0.15)
        sphere = pv.Sphere(
            radius=r,
            center=self.pos[i],
            theta_resolution=SPHERE_RESOLUTION,
            phi_resolution=SPHERE_RESOLUTION,
        )
        self._atom_actors[i] = self.plotter.add_mesh(
            sphere,
            color=jmol_colors[Z],
            smooth_shading=True,
            pbr=True,
            metallic=0.15,
            roughness=0.4,
            pickable=True,
        )

        if selected:
            halo = pv.Sphere(
                radius=r * HIGHLIGHT_SCALE,
                center=self.pos[i],
                theta_resolution=SPHERE_RESOLUTION,
                phi_resolution=SPHERE_RESOLUTION,
            )
            self._halo_actors[i] = self.plotter.add_mesh(
                halo,
                color=HIGHLIGHT_COLOR,
                opacity=HIGHLIGHT_OPACITY,
                smooth_shading=True,
                pickable=False,
            )

    def _draw_bonds(self) -> None:
        """Draw all bonds as half-bond-coloured cylinders."""
        for i, j in self.bonds:
            mid = (self.pos[i] + self.pos[j]) / 2.0
            for atom_idx, start, end in [
                (i, self.pos[i], mid),
                (j, mid, self.pos[j]),
            ]:
                direction = end - start
                length = float(np.linalg.norm(direction))
                if length < 1e-6:
                    continue
                cyl = pv.Cylinder(
                    center=(start + end) / 2.0,
                    direction=direction,
                    radius=BOND_RADIUS,
                    height=length,
                    resolution=TUBE_RESOLUTION,
                )
                self.plotter.add_mesh(
                    cyl,
                    color=jmol_colors[int(self.nums[atom_idx])],
                    smooth_shading=True,
                    pbr=True,
                    metallic=0.1,
                    roughness=0.5,
                    pickable=False,
                )
