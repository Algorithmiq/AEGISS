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

"""Output configuration dataclass for AEGISS."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class OutputConfig:
    """Configuration dataclass for AEGISS output options.

    Args:
        saving_folder (str, default './aegiss_output/'):
            Directory where cube files are written when *save_orb* is ``True``.
        saving_prefix (str, default 'aegiss_selection'):
            Prefix used for output cube file names.
        save_plots (bool, default False):
            Whether to generate and save diagnostic plots: the SOE heatmap and
            TH plot (spin-summed entropies only), the AO-contribution heatmaps of
            the projection step, and the AO pie chart of the final active
            orbitals (restricted references only).
        save_orb (bool, default True):
            Write cube files for the entropy-screened and final active orbitals.
    """

    saving_folder: str = "./aegiss_output/"
    saving_prefix: str = "aegiss_selection"
    save_orb: bool = True
    save_plots: bool = False

    def __post_init__(self) -> None:
        """Validate the configuration parameters.

        Raises:
            ValueError: If any of the parameters are out of their expected ranges or have invalid values.
        """
        if not self.saving_folder:
            raise ValueError(
                "OutputConfig: `saving_folder` must be a non-empty string."
            )
        if not self.saving_prefix:
            raise ValueError(
                "OutputConfig: `saving_prefix` must be a non-empty string."
            )
