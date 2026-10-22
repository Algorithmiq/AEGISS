"""Qt main window for MolTagger."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
from pyvistaqt import QtInteractor
from qtpy.QtCore import QEvent, QObject, Qt, QTimer
from qtpy.QtGui import QKeySequence
from qtpy.QtWidgets import (
    QAbstractItemView,
    QApplication,
    QComboBox,
    QFileDialog,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QPushButton,
    QShortcut,
    QSplitter,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from vtkmodules.vtkRenderingCore import vtkPropPicker

from aegiss.utils.mol_tagger._ao_suggest import suggest_ao_labels
from aegiss.utils.mol_tagger._atoms import (
    align_selection_to_xy,
    detect_bonds,
    get_basis,
    load_xyz,
)
from aegiss.utils.mol_tagger._constants import DARK_STYLE, DRAG_THRESHOLD_SQ
from aegiss.utils.mol_tagger._renderer import MolRenderer

logger = logging.getLogger(__name__)


class _RightClickPickFilter(QObject):
    """Qt event filter: right-click (not drag) on the 3D view picks an atom."""

    def __init__(self, window: MolViewWindow) -> None:
        super().__init__(window)
        self._win = window
        self._press: tuple[int, int] | None = None

    def eventFilter(self, _obj: Any, event: Any) -> bool:  # noqa: N802
        t = event.type()
        if t == QEvent.MouseButtonPress and event.button() == Qt.RightButton:
            self._press = (int(event.x()), int(event.y()))
        elif (
            t == QEvent.MouseButtonRelease
            and event.button() == Qt.RightButton
            and self._press is not None
        ):
            dx = int(event.x()) - self._press[0]
            dy = int(event.y()) - self._press[1]
            self._press = None
            if dx * dx + dy * dy <= DRAG_THRESHOLD_SQ:
                self._win._pick_at_qt(int(event.x()), int(event.y()))
        return False  # let Qt and VTK handle the event normally


class _ViewportResizeFilter(QObject):
    """Qt event filter: reposition the view-snap overlay when the viewport is resized."""

    def __init__(self, window: MolViewWindow) -> None:
        super().__init__(window)
        self._win = window

    def eventFilter(self, _obj: Any, event: Any) -> bool:  # noqa: N802
        if event.type() == QEvent.Resize:
            self._win._position_view_overlay()
        return False


class MolViewWindow(QMainWindow):
    """Desktop molecular viewer with interactive atom selection and tagging.

    Attributes:
        xyz_path: Path to the loaded XYZ file.
        tags: Mapping from atom index to its ``Element.N`` tag (e.g. ``"C.1"``).
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
        self._pos_history: list[np.ndarray] = []

        self.setWindowTitle(f"AEGISS > MolTagger — {Path(xyz_path).name}")
        self.resize(1100, 700)
        self.setStyleSheet(DARK_STYLE)
        self._build_ui()
        self._renderer = MolRenderer(self.plotter, self.atoms, self.bonds)
        self._renderer.render_all(self.selected)
        logger.info(
            "MolViewWindow opened for %s (%d atoms)", Path(xyz_path).name, self.n
        )

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
        self._pick_filter = _RightClickPickFilter(self)
        self.plotter.interactor.installEventFilter(self._pick_filter)
        splitter.addWidget(self.plotter.interactor)
        splitter.addWidget(self._make_sidebar())
        splitter.setSizes([800, 290])

        QShortcut(QKeySequence("Ctrl+Z"), self).activated.connect(self._undo_geometry)

        self._view_overlay = self._make_view_overlay()
        self._resize_filter = _ViewportResizeFilter(self)
        self.plotter.interactor.installEventFilter(self._resize_filter)
        QTimer.singleShot(0, self._position_view_overlay)

    def _make_view_overlay(self) -> QWidget:
        """Create the semi-transparent 2x3 camera-snap button grid."""
        overlay = QWidget(self.plotter.interactor)
        overlay.setAttribute(Qt.WA_TranslucentBackground)
        overlay.setStyleSheet("""
            QPushButton {
                background: rgba(15, 15, 40, 155);
                color: #9999cc;
                border: 1px solid rgba(70, 70, 140, 110);
                border-radius: 3px;
                font-size: 9px;
                font-family: monospace;
                padding: 0px 3px;
            }
            QPushButton:hover {
                background: rgba(45, 45, 100, 210);
                color: #ffffff;
                border-color: rgba(120, 120, 200, 180);
            }
            QPushButton:pressed {
                background: rgba(70, 70, 150, 230);
            }
""")
        grid = QGridLayout(overlay)
        grid.setContentsMargins(2, 2, 2, 2)
        grid.setSpacing(2)
        views = [
            ("XY", self.plotter.view_xy, False),
            ("XZ", self.plotter.view_xz, False),
            ("YZ", self.plotter.view_yz, False),
            ("XY*", self.plotter.view_xy, True),
            ("XZ*", self.plotter.view_xz, True),
            ("YZ*", self.plotter.view_yz, True),
        ]
        for k, (label, fn, neg) in enumerate(views):
            row, col = divmod(k, 3)
            btn = QPushButton(label, overlay)
            btn.setFixedSize(36, 18)
            btn.clicked.connect(lambda _checked=False, f=fn, n=neg: f(negative=n))
            grid.addWidget(btn, row, col)
        overlay.adjustSize()
        overlay.show()
        return overlay

    def _position_view_overlay(self) -> None:
        """Place the view-snap overlay in the bottom-right corner of the viewport."""
        w = self.plotter.interactor.width()
        h = self.plotter.interactor.height()
        if w <= 0 or h <= 0:
            return
        ow = self._view_overlay.width()
        oh = self._view_overlay.height()
        self._view_overlay.move(w - ow - 6, h - oh - 6)

    def showEvent(self, event: Any) -> None:  # noqa: N802
        """Position the view overlay once the window has its real dimensions."""
        super().showEvent(event)
        self._position_view_overlay()

    def _make_header(self) -> QWidget:
        """Build the top header bar."""
        header = QWidget()
        header.setFixedHeight(44)
        header.setStyleSheet("background: #111; border-bottom: 1px solid #333;")
        hl = QHBoxLayout(header)
        hl.setContentsMargins(16, 0, 16, 0)
        title = QLabel("AEGISS > MolTagger")
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
        sl.addWidget(self._make_ao_label_panel())
        sl.addWidget(self._make_geometry_panel())
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
        grp = QGroupBox("Atom Info")
        layout = QVBoxLayout(grp)
        self.orbital_text = QTextEdit()
        self.orbital_text.setReadOnly(True)
        self.orbital_text.setMinimumHeight(100)
        self.orbital_text.setHtml(
            '<span style="color:#666;">Select atoms to see info</span>'
        )
        layout.addWidget(self.orbital_text)
        return grp

    def _make_ao_label_panel(self) -> QGroupBox:
        """Build the AO label suggestion panel."""
        grp = QGroupBox("AO Labels")
        layout = QVBoxLayout(grp)

        self._char_map: list[str] = ["auto", "d-shell", "sigma", "custom"]
        self.ao_char_combo = QComboBox()
        self.ao_char_combo.addItems(["π / Auto", "d-shell", "σ framework", "Custom"])  # noqa: RUF001
        layout.addWidget(self.ao_char_combo)

        self.ao_label_text = QTextEdit()
        self.ao_label_text.setReadOnly(True)
        self.ao_label_text.setFixedHeight(52)
        self.ao_label_text.setPlaceholderText("select atoms to see suggestions")
        layout.addWidget(self.ao_label_text)

        btn_row = QHBoxLayout()
        btn_copy = QPushButton("Copy")
        btn_copy.setFixedWidth(60)
        btn_copy.clicked.connect(self._copy_ao_labels)
        btn_row.addWidget(btn_copy)
        self.ao_info_label = QLabel("")
        self.ao_info_label.setStyleSheet(
            "color: #777; font-size: 10px; font-family: monospace;"
        )
        self.ao_info_label.setWordWrap(True)
        btn_row.addWidget(self.ao_info_label, 1)
        layout.addLayout(btn_row)

        self.ao_warn_label = QLabel("")
        self.ao_warn_label.setStyleSheet("color: #ffaa44; font-size: 10px;")
        self.ao_warn_label.setWordWrap(True)
        layout.addWidget(self.ao_warn_label)

        self.ao_char_combo.currentIndexChanged.connect(self._update_ao_suggestions)
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

    def _pick_at_qt(self, qx: int, qy: int) -> None:
        """Translate Qt pixel coordinates to a 3D pick and toggle atom selection.

        Qt origin is top-left; VTK origin is bottom-left, so y must be flipped.
        On HiDPI screens the physical pixel count is scaled by devicePixelRatio.
        """
        widget = self.plotter.interactor
        dpr_getter = getattr(widget, "devicePixelRatioF", widget.devicePixelRatio)
        dpr = float(dpr_getter())
        vtk_x = round(qx * dpr)
        vtk_y = round((widget.height() - qy - 1) * dpr)
        picker = vtkPropPicker()
        if picker.Pick(vtk_x, vtk_y, 0, self.plotter.renderer):
            self._on_pick(picker.GetPickPosition())
        else:
            logger.debug(
                "Right-click at Qt(%d,%d) / VTK(%d,%d): no actor hit",
                qx,
                qy,
                vtk_x,
                vtk_y,
            )

    def _on_pick(self, point: Any) -> None:
        """Toggle selection of the atom nearest to the picked 3D point."""
        if point is None:
            return
        picked = np.array(point)
        dists = np.linalg.norm(self.pos - picked, axis=1)
        nearest = int(np.argmin(dists))
        threshold = 0.75
        if float(dists[nearest]) > threshold:
            return
        if nearest in self.selected:
            self.selected.discard(nearest)
        else:
            self.selected.add(nearest)
        self._sync_list_from_selected()
        self._renderer.draw_atom(nearest, selected=nearest in self.selected)
        self._update_orbital_info()
        self._update_ao_suggestions()
        self.plotter.render()

    def _on_list_selection(self) -> None:
        """Sync 3D selection state when the atom list changes."""
        new_sel = {idx.row() for idx in self.atom_list.selectedIndexes()}
        changed = self.selected.symmetric_difference(new_sel)
        self.selected = new_sel
        for i in changed:
            self._renderer.draw_atom(i, selected=i in self.selected)
        self._update_orbital_info()
        self._update_ao_suggestions()
        self.plotter.render()

    def _sync_list_from_selected(self) -> None:
        """Update list widget selection to match ``self.selected``."""
        self.atom_list.blockSignals(True)  # noqa: FBT003
        for i in range(self.n):
            self.atom_list.item(i).setSelected(i in self.selected)
        self.atom_list.blockSignals(False)  # noqa: FBT003

    def _select_all(self) -> None:
        """Select every atom in the molecule."""
        self.selected = set(range(self.n))
        self._sync_list_from_selected()
        for i in range(self.n):
            self._renderer.draw_atom(i, selected=True)
        self._update_orbital_info()
        self._update_ao_suggestions()
        self.plotter.render()

    def _clear_selection(self) -> None:
        """Deselect all atoms."""
        old = set(self.selected)
        self.selected.clear()
        self._sync_list_from_selected()
        for i in old:
            self._renderer.draw_atom(i, selected=False)
        self._update_orbital_info()
        self._update_ao_suggestions()
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
        self.status_label.setText(
            f"Tagged {len(self.selected)} atom(s) with label {raw}"
        )
        self._update_ao_suggestions()

    def _remove_tags(self) -> None:
        """Remove all tags from the selected atoms."""
        if not self.selected:
            return
        for i in self.selected:
            self.tags.pop(i, None)
        self._refresh_list_labels()
        self._update_orbital_info()
        self.status_label.setText(f"Cleared tags on {len(self.selected)} atom(s)")
        self._update_ao_suggestions()

    def _refresh_list_labels(self) -> None:
        """Redraw atom list labels to reflect current tag state."""
        for i in range(self.n):
            item = self.atom_list.item(i)
            if item is not None:
                item.setText(self._atom_label(i))

    def _atom_label(self, i: int) -> str:
        """Format the display label for atom *i*."""
        x, y, z = self.pos[i]
        tag = f"{self.tags[i]}" if i in self.tags else ""
        if tag != "":
            tag = f"{tag}"
            return f"#{i:>1d} {tag:>2s} {x:8.3f} {y:8.3f} {z:8.3f}"
        return f"#{i:>1d} {self.syms[i]:>2s} {x:8.3f} {y:8.3f} {z:8.3f}"

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
            lines.append(f"  Minimal basis : {', '.join(basis['all'])}")
            lines.append(
                f'  <span style="color:#77ddff;">Valence       : '
                f"{', '.join(basis['valence'])}</span>"
            )
            lines.append("")
        self.orbital_text.setHtml("<pre>" + "\n".join(lines) + "</pre>")

    # ------------------------------------------------------------------
    # AO label suggestions
    # ------------------------------------------------------------------

    def _update_ao_suggestions(self) -> None:
        """Refresh the AO label panel for the current selection and character mode."""
        character = self._char_map[self.ao_char_combo.currentIndex()]

        if character == "custom":
            self.ao_label_text.setReadOnly(False)
            self.ao_warn_label.setText("")
            self.ao_info_label.setText("enter manually")
            return

        self.ao_label_text.setReadOnly(True)
        labels, warning, info = suggest_ao_labels(
            self.nums, self.syms, self.pos, self.selected, character, tags=self.tags
        )
        display = '["' + '", "'.join(labels) + '"]' if labels else "[]"
        self.ao_label_text.setPlainText(display)
        self.ao_info_label.setText(info)
        self.ao_warn_label.setText(f"⚠ {warning}" if warning else "")

    def _copy_ao_labels(self) -> None:
        """Copy the suggested AO labels to the system clipboard."""
        text = self.ao_label_text.toPlainText().strip()
        if text and text != "[]":
            QApplication.clipboard().setText(text)
            self.ao_info_label.setText("copied!")

    # ------------------------------------------------------------------
    # Geometry
    # ------------------------------------------------------------------

    def _make_geometry_panel(self) -> QGroupBox:
        """Build the geometry-transformation panel."""
        grp = QGroupBox("Geometry")
        layout = QVBoxLayout(grp)
        btn_align = QPushButton("Align Selection Plane → XY")
        btn_align.setToolTip(
            "Rotate the entire molecule so the best-fit plane of\n"
            "the selected atoms lies in the XY plane (normal → Z).\n"
            "Requires ≥ 3 selected atoms.  Ctrl+Z to undo."
        )
        btn_align.clicked.connect(self._align_to_plane)
        layout.addWidget(btn_align)
        self.geom_status = QLabel("")
        self.geom_status.setObjectName("status")
        self.geom_status.setWordWrap(True)
        layout.addWidget(self.geom_status)
        return grp

    def _align_to_plane(self) -> None:
        """Rotate the molecule so the best-fit plane of the selection is the XY plane."""
        if len(self.selected) < 3:
            self.geom_status.setText("Select ≥ 3 atoms to define a plane.")
            return
        indices = sorted(self.selected)
        new_pos, ok = align_selection_to_xy(self.pos, indices)
        if not ok:
            self.geom_status.setText(
                "Selected atoms are collinear; cannot define a plane."
            )
            return
        self._pos_history.append(self.pos.copy())
        self.pos = new_pos
        self._renderer.pos = new_pos
        self._renderer.render_all(self.selected)
        self._refresh_list_labels()
        n = len(self._pos_history)
        self.geom_status.setText(f"Aligned. Ctrl+Z to undo ({n} step(s) in history).")
        self._update_ao_suggestions()
        logger.info("Aligned %d selected atoms to XY plane", len(indices))

    def _undo_geometry(self) -> None:
        """Restore the previous geometry from the undo history."""
        if not self._pos_history:
            self.geom_status.setText("Nothing to undo.")
            return
        self.pos = self._pos_history.pop()
        self._renderer.pos = self.pos
        self._renderer.render_all(self.selected)
        self._refresh_list_labels()
        n = len(self._pos_history)
        self.geom_status.setText(
            f"Undone. {n} step(s) remaining." if n else "Geometry restored to original."
        )
        self._update_ao_suggestions()
        logger.info("Undo geometry alignment: %d steps remaining", n)

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def _save_xyz(self) -> None:
        """Open a save dialog and write a tagged XYZ file.

        Keep original header line.
        For the comment line, if there was one, keep it,
        otherwise write a blank line to preserve XYZ format.
        """
        default = Path(self.xyz_path).stem + "_tagged.xyz"
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Tagged XYZ", default, "XYZ files (*.xyz);;All files (*)"
        )
        if not path:
            return

        lines: list[str] = []
        with Path.open(Path(self.xyz_path), "r") as f:
            original_lines = f.readlines()
            if len(original_lines) >= 2:
                lines.extend(original_lines[:2])
            else:
                lines.append(f"{self.n}\n")
                lines.append("\n")

        for i in range(self.n):
            x, y, z = self.pos[i]
            label = self.tags.get(i, self.syms[i])
            lines.append(f"{label:<6s}  {x:12.6f}  {y:12.6f}  {z:12.6f}\n")
        Path(path).write_text("".join(lines))
        self.status_label.setText(f"Saved {Path(path).name}")
        logger.info("Saved tagged XYZ to %s", path)
