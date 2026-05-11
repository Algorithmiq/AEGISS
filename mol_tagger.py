"""CLI entry point for MolTagger: python mol_tagger.py <file.xyz>."""

from __future__ import annotations

import argparse
from pathlib import Path

from aegiss.utils.mol_tagger import MolTagger


def main() -> None:
    """Parse CLI arguments and launch the interactive viewer."""
    parser = argparse.ArgumentParser(
        description="MolTagger — interactive 3D molecular viewer and atom tagger.",
    )
    parser.add_argument("filepath", type=Path, help="Path to an XYZ file.")
    args = parser.parse_args()

    tagger = MolTagger(args.filepath)
    tagger.show()


if __name__ == "__main__":
    main()
