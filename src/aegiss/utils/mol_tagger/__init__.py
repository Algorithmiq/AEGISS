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

"""MolTagger: interactive 3D molecular viewer and atom tagger."""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from aegiss.utils.mol_tagger._window import MolViewWindow

logger = logging.getLogger(__name__)


class MolTagger:
    """Public interface for the interactive molecular tagger.

    Wraps the Qt application lifecycle and :class:`MolViewWindow` so callers
    only need to know about ``filepath``, ``show()``, and ``export_tags()``.

    Attributes:
        filepath (Path): Resolved path to the loaded XYZ file.
    """

    def __init__(self, filepath: str | Path) -> None:
        """Load a molecule from an XYZ file.

        Args:
            filepath (str or Path): Path to the XYZ file to visualise.

        Raises:
            FileNotFoundError: If *filepath* does not exist.
        """
        self.filepath = Path(filepath).resolve()
        if not self.filepath.exists():
            raise FileNotFoundError(self.filepath)

        self._window: MolViewWindow | None = None

    def show(self) -> None:
        """Open the interactive PyVistaQt viewer (blocks until closed).

        Raises:
            ImportError: If the ``viz`` extras are not installed.
        """
        try:
            from qtpy.QtWidgets import QApplication  # noqa: PLC0415

            from aegiss.utils.mol_tagger._window import (  # noqa: PLC0415
                MolViewWindow as _MolViewWindow,
            )
        except ImportError as exc:
            raise ImportError(
                "MolTagger requires the 'viz' extras: pip install aegiss[viz]"
            ) from exc

        app = QApplication.instance() or QApplication(sys.argv)
        self._window = _MolViewWindow(str(self.filepath))
        self._window.show()
        getattr(app, "exec", app.exec_)()

    def export_tags(self) -> dict[int, str]:
        """Return the current atom-tag mapping.

        Returns:
            dict[int, str]: Dictionary mapping atom indices to their assigned tag strings.
                Empty if :meth:`show` has not been called yet.
        """
        if self._window is None:
            return {}
        return dict(self._window.tags)

    def __repr__(self) -> str:
        """Return a concise string representation."""
        return f"MolTagger({self.filepath.name!r})"


__all__ = ["MolTagger"]
