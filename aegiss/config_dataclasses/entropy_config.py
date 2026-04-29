"""AEGISS entropy-based screening configuration dataclass."""

from __future__ import annotations

from typing import Literal
from dataclasses import dataclass


@dataclass(frozen=True)
class EntropyConfig:
    """Configuration dataclass for entropy-based active space screening."""

    entropy_fraction: float = 0.1
    entropy_fraction_type: Literal["r", "a"] = "r"
    entropy_ranking: Literal["subset", "global"] = "subset"

    def __post_init__(self) -> None:
        if not (0 <= self.entropy_fraction <= 1):
            raise ValueError(
                f"Entropy fraction must be in the range [0, 1], got {self.entropy_fraction}."
            )
        if self.entropy_fraction_type not in ("r", "a"):
            raise ValueError(
                f"Invalid entropy_fraction_type {self.entropy_fraction_type!r}. Must be 'r' or 'a'."
            )
        if self.entropy_ranking not in ("subset", "global"):
            raise ValueError(
                f"Invalid entropy_ranking {self.entropy_ranking!r}. Must be 'subset' or 'global'."
            )
