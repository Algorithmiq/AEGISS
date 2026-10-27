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

"""Utilities: DMRG helpers, orbital index partitioning, cube file export, plotting, banner."""

from aegiss.utils.banner import print_banner
from aegiss.utils.orbital_indices import get_orbital_spaces, get_subspaces_from_cas

__all__ = [
    "dump_qmi",
    "get_orbital_spaces",
    "get_subspaces_from_cas",
    "plot_ao_contribution",
    "plot_pie_contribution",
    "print_banner",
    "save_orbitals",
    "soe_heatmap",
    "th_plots",
]

_DUMP_NAMES = frozenset({"dump_qmi", "save_orbitals"})
_PLOTTING_NAMES = frozenset(
    {"plot_ao_contribution", "plot_pie_contribution", "soe_heatmap", "th_plots"}
)


def __getattr__(name: str) -> object:
    if name in _DUMP_NAMES:
        from aegiss.utils.dump import dump_qmi, save_orbitals  # noqa: PLC0415

        globals().update(dump_qmi=dump_qmi, save_orbitals=save_orbitals)
        return globals()[name]
    if name in _PLOTTING_NAMES:
        from aegiss.utils.plotting import (  # noqa: PLC0415
            plot_ao_contribution,
            plot_pie_contribution,
            soe_heatmap,
            th_plots,
        )

        globals().update(
            plot_ao_contribution=plot_ao_contribution,
            plot_pie_contribution=plot_pie_contribution,
            soe_heatmap=soe_heatmap,
            th_plots=th_plots,
        )
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
