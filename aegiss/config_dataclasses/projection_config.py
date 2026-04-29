"""Atomic orbital projection-based screening configuration dataclass for AEGISS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ProjectionConfig:
    """Configuration dataclass for orbital selection based on atomic orbital projection.

    Parameters:
    -----------
    ao_labels : list of str
        AO labels to project onto.  Format: ``"Element AtomicOrbital"``,
        e.g. ``"C 2pz"``, ``"Fe 3d"``.
    basis : str or None, default 'minao'
        Basis set used for AO projection.  ``None`` uses the original
        molecular basis.
    contribution : float, default 0.5
        Projection weight threshold (see *contribution_type*).
    contribution_type : {'a', 'r'}, default 'a'
        ``'a'``: absolute threshold; ``'r'``: relative to the maximum weight.
    top_n : int or None, default None
        If set, keep the top-N orbitals per AO label instead of using a
        contribution threshold.
    ranking : {'subset', 'global'}, default 'subset'
        ``'subset'``: run AO projection independently on the occupied and
        virtual entropy-screened subspaces; ``'global'``: run it on the
        combined set.
    """

    ao_labels: list[str]
    basis: str | None = "minao"
    contribution: float = 0.5
    contribution_type: Literal["a", "r"] = "a"
    top_n: int | None = None
    ranking: Literal["subset", "global"] = "subset"

    def __post_init__(self) -> None:
        """Validate the configuration parameters.

        Raises:
        ------
        ValueError
            If any of the parameters are out of their expected ranges or have invalid values.
        """
        if not self.ao_labels or not all(
            isinstance(label, str) for label in self.ao_labels
        ):
            raise ValueError(
                "ProjectionConfig: `ao_labels` must be a non-empty list of strings."
            )
        if self.contribution_type not in ("a", "r"):
            raise ValueError(
                f"ProjectionConfig: Invalid `contribution_type` \
                {self.contribution_type!r}. Must be 'a' or 'r'."
            )
        if self.ranking not in ("subset", "global"):
            raise ValueError(
                f"ProjectionConfig: Invalid `ranking` {self.ranking!r}. Must be 'subset' or 'global'."
            )
        if not (0 <= self.contribution <= 1) and self.contribution_type == "r":
            raise ValueError(
                f"ProjectionConfig: `contribution` must be in the range [0, 1]\
                when `contribution_type` is 'r', got {self.contribution}."
            )
