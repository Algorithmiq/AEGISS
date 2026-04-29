"""Output configuration dataclass for AEGISS."""

from typing import Any, Literal
from dataclasses import dataclass


@dataclass(frozen=True)
class OutputConfig:
    """Configuration dataclass for AEGISS output options."""

    saving_folder: str = "./aegiss_output/"
    saving_prefix: str = "aegiss_selection"
    save_orb: bool = True
    save_plots: bool = False

    def __post_init__(self) -> None:
        if self.saving_folder and not isinstance(self.saving_folder, str):
            raise ValueError(
                f"Saving folder must be a string, got {type(self.saving_folder).__name__}."
            )
        if self.saving_prefix and not isinstance(self.saving_prefix, str):
            raise ValueError(
                f"Saving prefix must be a string, got {type(self.saving_prefix).__name__}."
            )
        if not isinstance(self.save_orb, bool):
            raise ValueError(
                f"save_orb must be a boolean, got {type(self.save_orb).__name__}."
            )
        if not isinstance(self.save_plots, bool):
            raise ValueError(
                f"save_plots must be a boolean, got {type(self.save_plots).__name__}."
            )
