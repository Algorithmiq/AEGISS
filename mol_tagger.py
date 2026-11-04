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

"""CLI entry point for MolTagger: python mol_tagger.py <file.xyz>."""

from __future__ import annotations

from cyclopts import App
from cyclopts.types import ExistingFile  # noqa: TC002 (cyclopts reads the annotation at runtime)

from aegiss.utils.mol_tagger import MolTagger

app = App(
    name="mol_tagger.py",
    help="MolTagger — interactive 3D molecular viewer and atom tagger.",
)


@app.default
def main(filepath: ExistingFile, /) -> None:
    """Launch the interactive viewer.

    Args:
        filepath: Path to an XYZ file.
    """
    tagger = MolTagger(filepath)
    tagger.show()


if __name__ == "__main__":
    app()
