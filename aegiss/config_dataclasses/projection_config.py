"""Atomic orbital projection-based screening configuration dataclass for AEGISS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ProjectionConfig:
    """Configuration dataclass for orbital selection based on atomic orbital projection."""

    projection_basis: str | None = "minao"
    projection_ao_labels: list[str] | None = None
    projection_contribution: float = 0.5
    projection_contribution_type: Literal["a", "r"] = "a"
    projection_top_n: int | None = None
    projection_ranking: Literal["subset", "global"] = "subset"

    def __post_init__(self) -> None:
        """Validate the configuration parameters.

        Raises:
        ------
        ValueError
            If any of the parameters are out of their expected ranges or have invalid values.
        """
        if self.projection_contribution_type not in ("a", "r"):
            raise ValueError(
                f"Invalid projection_contribution_type \
                {self.projection_contribution_type!r}. Must be 'a' or 'r'."
            )
        if self.projection_ranking not in ("subset", "global"):
            raise ValueError(
                f"Invalid projection_ranking {self.projection_ranking!r}. Must be 'subset' or 'global'."
            )
        if (
            not (0 <= self.projection_contribution <= 1)
            and self.projection_contribution_type == "r"
        ):
            raise ValueError(
                f"Projection contribution must be in the range [0, 1]\
                when projection_contribution_type is 'r', got {self.projection_contribution}."
            )
