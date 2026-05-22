"""Utilities: DMRG helpers, orbital index partitioning, cube file export."""

from aegiss.utils.banner import print_banner
from aegiss.utils.dmrg_utils import compute_qmi
from aegiss.utils.orbital_indices import get_orbital_spaces, get_subspaces_from_cas

__all__ = [
    "compute_qmi",
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
