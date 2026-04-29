from __future__ import annotations

from typing import Any, Literal
from dataclasses import dataclass


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
        if self.projection_contribution_type not in ("a", "r"):
            raise ValueError(
                f"Invalid projection_contribution_type {self.projection_contribution_type!r}. Must be 'a' or 'r'."
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
                f"Projection contribution must be in the range [0, 1] when projection_contribution_type is 'r', got {self.projection_contribution}."
            )
