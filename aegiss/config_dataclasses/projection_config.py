"""Atomic orbital projection-based screening configuration dataclass for AEGISS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class ProjectionConfig:
    """Configuration dataclass for orbital selection based on atomic orbital projection.

    Parameters:
    ----------
    ao_labels : list[str | list[str]]
        AO labels to project onto. Each entry is either a single label string
        (e.g. ``"Fe 3d"``) or a list of strings treated as one group
        (e.g. ``["Fe 3d", "Fe 4d"]``).
    basis : str or None, default 'minao'
        Basis set used for AO projection.  ``None`` uses the original
        molecular basis.
    contribution : float or list[float], default 0.5
        Projection weight threshold (see *contribution_type*). A list specifies
        per-label thresholds and must be non-empty; if its length does not match
        *ao_labels*, the first value is broadcast to all labels.
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

    ao_labels: list[str | list[str]]
    basis: str | None = "minao"
    contribution: float | list[float] = 0.5
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
            isinstance(label, str)
            or (
                isinstance(label, list)
                and label
                and all(isinstance(s, str) for s in label)
            )
            for label in self.ao_labels
        ):
            raise ValueError(
                "ProjectionConfig: `ao_labels` must be a non-empty list where each "
                "entry is a non-empty string or a non-empty list of strings."
            )
        if self.contribution_type not in ("a", "r"):
            raise ValueError(
                "ProjectionConfig: Invalid `contribution_type` "
                f"{self.contribution_type!r}. Must be 'a' or 'r'."
            )
        if self.ranking not in ("subset", "global"):
            raise ValueError(
                f"ProjectionConfig: Invalid `ranking` {self.ranking!r}. Must be 'subset' or 'global'."
            )
        contribution_values = (
            self.contribution
            if isinstance(self.contribution, list)
            else [self.contribution]
        )
        if not contribution_values:
            raise ValueError(
                "ProjectionConfig: `contribution` must be a non-empty list when provided as a list."
            )
        for val in contribution_values:
            if self.contribution_type == "r" and not (0 <= val <= 1):
                raise ValueError(
                    "ProjectionConfig: `contribution` must be in the range [0, 1] "
                    f"when `contribution_type` is 'r', got {val}."
                )
            if self.contribution_type == "a" and val < 0:
                raise ValueError(
                    "ProjectionConfig: `contribution` must be non-negative when `contribution_type` is 'a', "
                    f"got {val}."
                )
        if self.top_n is not None and self.top_n <= 0:
            raise ValueError(
                f"ProjectionConfig: `top_n` must be a positive integer or None, got {self.top_n}."
            )
