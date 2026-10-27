# Copyright 2026 Algorithmiq
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Display constants and Qt stylesheet for MolTagger."""

from __future__ import annotations

ATOM_RADIUS_SCALE: float = 0.30
BOND_RADIUS: float = 0.08
SPHERE_RESOLUTION: int = 20
TUBE_RESOLUTION: int = 12

HIGHLIGHT_COLOR: tuple[float, float, float] = (1.0, 0.25, 0.25)
HIGHLIGHT_OPACITY: float = 0.35
HIGHLIGHT_SCALE: float = 1.8

DRAG_THRESHOLD_SQ: int = (
    25  # 5-pixel radius; clicks within this distance trigger atom pick
)

# VTK normalised viewport (x0, y0, x1, y1) for the orientation-axes compass.
# y0=0.13 raises the compass so the view-snap buttons fit below it.
# _window.py uses y0 to calculate the Qt y-position of the buttons overlay.
COMPASS_VIEWPORT: tuple[float, float, float, float] = (0.0, 0.13, 0.20, 0.37)

DARK_STYLE: str = """
QMainWindow, QWidget {
    background-color: #1a1a2e;
    color: #e0e0e0;
    font-family: 'Segoe UI', 'Cantarell', sans-serif;
    font-size: 13px;
}
QGroupBox {
    border: 1px solid #333;
    border-radius: 4px;
    margin-top: 8px;
    padding-top: 14px;
    font-weight: bold;
    color: #aaa;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}
QListWidget {
    background: #12121e;
    border: 1px solid #333;
    border-radius: 3px;
    color: #ddd;
    font-family: 'Fira Code', 'Consolas', monospace;
    font-size: 12px;
}
QListWidget::item:selected {
    background: #2a1a3e;
    color: #ff6666;
}
QListWidget::item:hover {
    background: #222238;
}
QPushButton {
    background: #2a2a3e;
    border: 1px solid #444;
    border-radius: 4px;
    padding: 5px 12px;
    color: #ddd;
    font-size: 12px;
}
QPushButton:hover {
    background: #3a3a5e;
    border-color: #666;
}
QPushButton:pressed {
    background: #1a1a2e;
}
QLineEdit {
    background: #12121e;
    border: 1px solid #444;
    border-radius: 3px;
    padding: 4px 8px;
    color: #eee;
    font-size: 12px;
}
QTextEdit {
    background: #12121e;
    border: 1px solid #333;
    border-radius: 3px;
    color: #88ff88;
    font-family: 'Fira Code', 'Consolas', monospace;
    font-size: 11px;
}
QComboBox {
    background: #12121e;
    border: 1px solid #444;
    border-radius: 3px;
    padding: 3px 8px;
    color: #eee;
    font-size: 12px;
}
QComboBox::drop-down {
    border: none;
    width: 18px;
}
QComboBox QAbstractItemView {
    background: #1a1a2e;
    color: #eee;
    selection-background-color: #2a1a3e;
    border: 1px solid #444;
}
QLabel#header {
    font-size: 18px;
    font-weight: bold;
    color: #ccc;
    letter-spacing: 2px;
}
QLabel#status {
    color: #88ff88;
    font-size: 11px;
    font-family: monospace;
}
QSplitter::handle {
    background: #333;
    width: 2px;
}
"""
