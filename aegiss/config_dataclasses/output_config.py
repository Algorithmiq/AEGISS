"""Output configuration dataclass for AEGISS."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OutputConfig:
    """Configuration dataclass for AEGISS output options.

    Parameters:
    -----------
    saving_folder : str, default './aegiss/examples/orbs/'
        Directory where cube files are written when *save_orb* is ``True``.
    saving_prefix : str, default 'benzene'
        Prefix used for output cube file names.
    save_plots : bool, default False
        Whether to generate and save diagnostic plots (SOE heatmap and TH plot).
    save_orb : bool, default True
        Write cube files for the entropy-screened and final active orbitals.
    """

    saving_folder: str = "./aegiss_output/"
    saving_prefix: str = "aegiss_selection"
    save_orb: bool = True
    save_plots: bool = False
