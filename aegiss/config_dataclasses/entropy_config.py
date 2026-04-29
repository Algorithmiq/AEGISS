"""AEGISS entropy-based screening configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EntropyConfig:
    """Configuration dataclass for entropy-based active space screening.

    Parameters:
    ----------
    entropy_fraction : float, default 0.1
        Entropy threshold value (see *entropy_fraction_type*).
    entropy_fraction_type : {'r', 'a'}, default 'r'
        ``'r'``: keep orbitals whose S(1) >= ``entropy_fraction x max(S(1))``;
        ``'a'``: keep orbitals whose S(1) ≥ ``entropy_fraction`` (absolute).
    entropy_ranking : {'subset', 'global'}, default 'subset'
        How to rank orbitals for entropy screening.  ``'subset'`` applies the
        threshold independently to occupied and virtual subspaces; ``'global'``
        ranks all active orbitals together.
    """

    entropy_fraction: float = 0.1
    entropy_fraction_type: Literal["r", "a"] = "r"
    entropy_ranking: Literal["subset", "global"] = "subset"

    def __post_init__(self) -> None:
        """Validate the configuration parameters.

        Raises:
        ------
        ValueError
            If any of the parameters are out of their expected ranges or have invalid values.
        """
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
