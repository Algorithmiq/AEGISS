"""Qt main window for MolTagger."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from pyvistaqt import QtInteractor
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from aegiss.utils.mol_tagger._atoms import detect_bonds, get_basis, load_xyz
from aegiss.utils.mol_tagger._constants import DARK_STYLE
from aegiss.utils.mol_tagger._renderer import MolRenderer

logger = logging.getLogger(__name__)


class MolViewWindow(QMainWindow):
    """Desktop molecular viewer with interactive atom selection and tagging.

    Attributes:
        xyz_path: Path to the loaded XYZ file.
        tags: Mapping from atom index to comma-separated tag string.
        selected: Set of currently selected atom indices.
    """

    def __init__(self, xyz_path: str) -> None:
        """Load a molecule and build the viewer window.

        Args:
            xyz_path: Path to the XYZ file to visualise.
        """
        super().__init__()
        self.xyz_path = xyz_path
        self.atoms, self.tags = load_xyz(xyz_path)
        self.bonds = detect_bonds(self.atoms)
        self.n = len(self.atoms)
        self.syms: list[str] = self.atoms.get_chemical_symbols()
        self.nums: np.ndarray = self.atoms.get_atomic_numbers()
        self.pos: np.ndarray = self.atoms.get_positions()
        self.selected: set[int] = set()

        self.setWindowTitle(f"MolTagger — {Path(xyz_path).name}")
        self.resize(1100, 700)
        self.setStyleSheet(DARK_STYLE)
        self._build_ui()
        self._renderer = MolRenderer(self.plotter, self.atoms, self.bonds)
        self._renderer.render_all(self.selected)
        logger.info("MolViewWindow opened for %s (%d atoms)", Path(xyz_path).name, self.n)

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        """Assemble the full window layout."""
        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        root.addWidget(self._make_header())

        splitter = QSplitter(Qt.Horizontal)
        root.addWidget(splitter)

        self.plotter = QtInteractor(splitter)
        self.plotter.set_background("#0f0f19")
        self.plotter.enable_point_picking(
            callback=self._on_pick,
            use_picker=True,
            show_message=False,
            show_point=False,
            picker="point",
            tolerance=0.02,
        )
        splitter.addWidget(self.plotter.interactor)
        splitter.addWidget(self._make_sidebar())
        splitter.setSizes([800, 290])

    def _make_header(self) -> QWidget:
        """Build the top header bar."""
        header = QWidget()
        header.setFixedHeight(44)
        header.setStyleSheet("background: #111; border-bottom: 1px solid #333;")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 0, 16, 0)
        title = QLabel("MolTagger")
        title.setObjectName("header")
        hl.addWidget(title)
        info = QLabel(
            f"{Path(self.xyz_path).name}  —  {self.n} atoms, {len(self.bonds)} bonds"
        )
        info.setStyleSheet("color: #777; font-size: 12px; font-family: monospace;")
        hl.addStretch()
        hl.addWidget(info)
        return header

    def _make_sidebar(self) -> QWidget:
        """Build the right-hand control panel."""
        sidebar = QWidget()
        sidebar.setFixedWidth(290)
        sidebar.setStyleSheet("background: #111;")
        sl = QVBoxLayout(sidebar)
        sl.setContentsMargins(10, 10, 10, 10)
        sl.setSpacing(6)
        sl.addWidget(self._make_atom_panel())
        sl.addWidget(self._make_tag_panel())
        sl.addWidget(self._make_orbital_panel())
        sl.addWidget(self._make_export_panel())
        sl.addStretch()
        return sidebar

    def _make_atom_panel(self) -> QGroupBox:
        """Build the atom list panel."""
        grp = QGroupBox("Atoms")
        layout = QVBoxLayout(grp)
        btn_row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_clear = QPushButton("Clear")
        btn_all.clicked.connect(self._select_all)
        btn_clear.clicked.connect(self._clear_selection)
        btn_row.addWidget(btn_all)
        btn_row.addWidget(btn_clear)
        layout.addLayout(btn_row)
        self.atom_list = QListWidget()
        self.atom_list.setSelectionMode(QAbstractItemView.MultiSelection)
        for i in range(self.n):
            self.atom_list.addItem(QListWidgetItem(self._atom_label(i)))
        self.atom_list.itemSelectionChanged.connect(self._on_list_selection)
        layout.addWidget(self.atom_list)
        return grp

    def _make_tag_panel(self) -> QGroupBox:
        """Build the tag-input panel."""
        grp = QGroupBox("Tag Selected")
        layout = QVBoxLayout(grp)
        self.tag_input = QLineEdit()
        self.tag_input.setPlaceholderText("Numeric label, e.g. 1, 2, 3 …")
        self.tag_input.returnPressed.connect(self._apply_tag)
        layout.addWidget(self.tag_input)
        btn_row = QHBoxLayout()
        btn_apply = QPushButton("Apply Tag")
        btn_apply.clicked.connect(self._apply_tag)
        btn_remove = QPushButton("Remove Tags")
        btn_remove.clicked.connect(self._remove_tags)
        btn_row.addWidget(btn_apply)
        btn_row.addWidget(btn_remove)
        layout.addLayout(btn_row)
        return grp

    def _make_orbital_panel(self) -> QGroupBox:
        """Build the orbital-info read-only panel."""
        grp = QGroupBox("Orbital Info")
        layout = QVBoxLayout(grp)
        self.orbital_text = QTextEdit()
        self.orbital_text.setReadOnly(True)
        self.orbital_text.setMinimumHeight(100)
        self.orbital_text.setHtml('<span style="color:#666;">Select atoms to see orbitals</span>')
        layout.addWidget(self.orbital_text)
        return grp

    def _make_export_panel(self) -> QGroupBox:
        """Build the save/export panel."""
        grp = QGroupBox("Export")
        layout = QVBoxLayout(grp)
        btn_save = QPushButton("Save Tagged XYZ")
        btn_save.clicked.connect(self._save_xyz)
        layout.addWidget(btn_save)
        self.status_label = QLabel("")
        self.status_label.setObjectName("status")
        layout.addWidget(self.status_label)
        return grp

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def _on_pick(self, point: Any) -> None:
        """Toggle selection of the atom nearest to the picked 3D point."""
        if point is None:
            return
        picked = np.array(point)
        dists = np.linalg.norm(self.pos - picked, axis=1)
        nearest = int(np.argmin(dists))
        threshold = max(float(self.nums[nearest]) * 0.30 * 2.5, 0.5)
        if float(dists[nearest]) > threshold:
            return
        if nearest in self.selected:
            self.selected.discard(nearest)
        else:
            self.selected.add(nearest)
        self._sync_list_from_selected()
        self._renderer.draw_atom(nearest, selected=nearest in self.selected)
        self._update_orbital_info()
        self.plotter.render()

    def _on_list_selection(self) -> None:
        """Sync 3D selection state when the atom list changes."""
        new_sel = {idx.row() for idx in self.atom_list.selectedIndexes()}
        changed = self.selected.symmetric_difference(new_sel)
        self.selected = new_sel
        for i in changed:
            self._renderer.draw_atom(i, selected=i in self.selected)
        self._update_orbital_info()
        self.plotter.render()

    def _sync_list_from_selected(self) -> None:
        """Update list widget selection to match ``self.selected``."""
        self.atom_list.blockSignals(b=True)
        for i in range(self.n):
            self.atom_list.item(i).setSelected(i in self.selected)
        self.atom_list.blockSignals(b=False)

    def _select_all(self) -> None:
        """Select every atom in the molecule."""
        self.selected = set(range(self.n))
        self._sync_list_from_selected()
        for i in range(self.n):
            self._renderer.draw_atom(i, selected=True)
        self._update_orbital_info()
        self.plotter.render()

    def _clear_selection(self) -> None:
        """Deselect all atoms."""
        old = set(self.selected)
        self.selected.clear()
        self._sync_list_from_selected()
        for i in old:
            self._renderer.draw_atom(i, selected=False)
        self._update_orbital_info()
        self.plotter.render()

    # ------------------------------------------------------------------
    # Tagging
    # ------------------------------------------------------------------

    def _apply_tag(self) -> None:
        """Apply a numeric label to all selected atoms as ``Element.X``."""
        raw = self.tag_input.text().strip()
        if not raw or not self.selected:
            return
        if not raw.isdigit():
            self.status_label.setText("Label must be a whole number.")
            return
        for i in self.selected:
            self.tags[i] = f"{self.syms[i]}.{raw}"
        self.tag_input.clear()
        self._refresh_list_labels()
        self._update_orbital_info()
        self.status_label.setText(f"Tagged {len(self.selected)} atom(s) with label {raw}")

    def _remove_tags(self) -> None:
        """Remove all tags from the selected atoms."""
        if not self.selected:
            return
        for i in self.selected:
            self.tags.pop(i, None)
        self._refresh_list_labels()
        self._update_orbital_info()
        self.status_label.setText(f"Cleared tags on {len(self.selected)} atom(s)")

    def _refresh_list_labels(self) -> None:
        """Redraw atom list labels to reflect current tag state."""
        for i in range(self.n):
            self.atom_list.item(i).setText(self._atom_label(i))

    def _atom_label(self, i: int) -> str:
        """Format the display label for atom *i*."""
        x, y, z = self.pos[i]
        tag = f"{self.tags[i]}" if i in self.tags else ""
        if tag != "":
            tag = f"{tag}"
            return f"{i:>3d} {tag:>2s} {x:8.3f} {y:8.3f} {z:8.3f}"
        return f"{i:>3d} {self.syms[i]:>2s} {x:8.3f} {y:8.3f} {z:8.3f}"

    # ------------------------------------------------------------------
    # Orbital info panel
    # ------------------------------------------------------------------

    def _update_orbital_info(self) -> None:
        """Refresh the orbital info text for the current selection."""
        if not self.selected:
            self.orbital_text.setHtml(
                '<span style="color:#666;">Select atoms to see orbitals</span>'
            )
            return
        lines: list[str] = []
        for i in sorted(self.selected):
            Z = int(self.nums[i])
            basis = get_basis(Z)
            hdr = f"#{i} {self.syms[i]} (Z={Z})"
            if i in self.tags:
                hdr += f"  {self.tags[i]}"
            lines.append(f'<b style="color:#ffdd57;">{hdr}</b>')
            lines.append(f'  Minimal basis : {", ".join(basis["all"])}')
            lines.append(
                f'  <span style="color:#77ddff;">Valence       : '
                f'{", ".join(basis["valence"])}</span>'
            )
            lines.append("")
        self.orbital_text.setHtml("<pre>" + "\n".join(lines) + "</pre>")

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _save_xyz(self) -> None:
        """Open a save dialog and write a tagged XYZ file."""
        default = Path(self.xyz_path).stem + "_tagged.xyz"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Tagged XYZ", default, "XYZ files (*.xyz);;All files (*)"
        )
        if not path:
            return
        lines: list[str] = [f"{self.n}\n", "\n"]
        for i in range(self.n):
            x, y, z = self.pos[i]
            label = self.tags.get(i, self.syms[i])
            lines.append(f"{label:<6s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")
        Path(path).write_text("".join(lines))
        self.status_label.setText(f"Saved {Path(path).name}")
        logger.info("Saved tagged XYZ to %s", path)
