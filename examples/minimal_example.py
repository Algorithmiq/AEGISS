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

"""Select the benzene π orbitals by projecting the MOs onto the carbon 2pz AOs.

This is the AO-projection half of AEGISS on its own: an RHF reference, then the
orbitals whose weight on ``C 2pz`` exceeds a threshold. It needs no DMRG run, so it
finishes in seconds. The full workflow, with the entropy screening, is in the
tutorials (``docs/tutorials/``).

Run it with ``just example`` or ``uv run python examples/minimal_example.py``.
"""

from __future__ import annotations

from pathlib import Path

from aegiss.active_space import select_by_ao_projection
from aegiss.initial_guess import build_mol, run_scf
from aegiss.utils.orb_display import to_display_all

GEOMETRY = (
    Path(__file__).resolve().parents[1] / "docs" / "tutorials" / "data" / "benzene.xyz"
)


def main() -> None:
    """Run RHF on benzene and select its π space by C 2pz projection."""
    mol = build_mol(str(GEOMETRY), basis="sto-3g", symmetry=False)
    mf = run_scf(mol, method="RHF")

    # an MO enters the active space when at least half of it lies on C 2pz
    selected = select_by_ao_projection(mf, ao_labels=["C 2pz"], contribution=0.5)
    # only an unrestricted reference returns the {"alpha": ..., "beta": ...} split
    if isinstance(selected, dict):
        raise TypeError("expected one index array for a restricted reference")

    print(f"RHF energy: {mf.e_tot:.6f} Eh")
    print(f"pi orbitals ({len(selected)}): {to_display_all(selected)}")


if __name__ == "__main__":
    main()
