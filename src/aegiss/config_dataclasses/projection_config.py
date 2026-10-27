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

"""Atomic orbital projection-based screening configuration dataclass for AEGISS."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


def _helper_contributions(
    contribution: float | list[float], contribution_type: Literal["a", "r"]
) -> None:
    """Helper function to validate contribution values based on their type.

    Args:
        contribution (float or list[float]): The contribution threshold(s) to validate.
        contribution_type ({'a', 'r'}): The type of contribution threshold: 'a' for absolute, 'r' for relative.

    Raises:
        ValueError: If any contribution value is out of the expected range based on its type.
    """
    contribution_values = (
        contribution if isinstance(contribution, list) else [contribution]
    )
    if not contribution_values:
        raise ValueError(
            "ProjectionConfig: `contribution` must be a non-empty list when provided as a list."
        )
    for val in contribution_values:
        if contribution_type == "r" and not (0 <= val <= 1):
            raise ValueError(
                "ProjectionConfig: `contribution` must be in the range [0, 1] "
                f"when `contribution_type` is 'r', got {val}."
            )
        if contribution_type == "a" and val < 0:
            raise ValueError(
                "ProjectionConfig: `contribution` must be non-negative when `contribution_type` is 'a', "
                f"got {val}."
            )


def _helper_top_n(top_n: int | list[int]) -> None:
    """Helper function to validate the `top_n` parameter.

    Args:
        top_n (int or list[int]):
            The top-N orbital selection parameter to validate.

    Raises:
        ValueError:
            If `top_n` is not a positive integer, or if it's a list that is empty or contains non-positive integers.
    """
    if isinstance(top_n, list):
        if not top_n:
            raise ValueError(
                "ProjectionConfig: `top_n` must be a non-empty list when provided as a list."
            )
        for n in top_n:
            if not isinstance(n, int) or n <= 0:
                raise ValueError(
                    f"ProjectionConfig: Each entry in `top_n` must be a positive integer, got {n}."
                )
    elif not isinstance(top_n, int) or top_n <= 0:
        raise ValueError(
            f"ProjectionConfig: `top_n` must be a positive integer or None, got {top_n}."
        )


def _valid_ao_label(label: object) -> bool:
    """Helper function to validate AO labels.

    Args:
        label (str or list[str]): The AO label to validate, which can be a single string or a list of strings.

    Returns:
        bool: True if the label is valid (non-empty string or non-empty list of non-empty strings), False otherwise.
    """
    if isinstance(label, str):
        return bool(label)

    if isinstance(label, list):
        return bool(label) and all(isinstance(s, str) and bool(s) for s in label)

    return False


@dataclass(frozen=True)
class ProjectionConfig:
    """Configuration dataclass for orbital selection based on atomic orbital projection.

    Args:
        ao_labels (list[str | list[str]]):
            AO labels to project onto. Each entry is either a single label string
            (e.g. ``"Fe 3d"``) or a list of strings treated as one group
            (e.g. ``["Fe 3d", "Fe 4d"]``).
        basis (str or None, default 'minao'):
            Basis set used for AO projection.  ``None`` uses the original
            molecular basis.
        contribution (float or list[float], default 0.5):
            Projection weight threshold (see *contribution_type*). A list specifies
            per-label thresholds and must be non-empty; if its length does not match
            *ao_labels*, the first value is broadcast to all labels.
        contribution_type ({'a', 'r'}, default 'a'):
            ``'a'``: absolute threshold;
            ``'r'``: relative to the maximum weight.
        top_n (int | list[int] |  None, default None):
            If set, keep the top-N orbitals per AO label instead of using a
            contribution threshold. If a list is provided, it must be non-empty and either
            match the length of *ao_labels* or be broadcastable to it.
        ranking ({'subset', 'global'}, default 'subset'):
            ``'subset'``: run AO projection independently on the occupied and virtual entropy-screened subspaces;
            ``'global'``: run it on the combined set.
    """

    ao_labels: list[str | list[str]]
    basis: str | None = "minao"
    contribution: float | list[float] = 0.5
    contribution_type: Literal["a", "r"] = "a"
    top_n: int | list[int] | None = None
    ranking: Literal["subset", "global"] = "subset"

    def __post_init__(self) -> None:
        """Validate the configuration parameters.

        Raises:
            ValueError: If any of the parameters are out of their expected ranges or have invalid values.
        """
        if not isinstance(self.ao_labels, list):
            raise ValueError(  # noqa: TRY004 -- documented as ValueError
                "ProjectionConfig: `ao_labels` must be a non-empty list."
            )
        if len(self.ao_labels) == 0:
            raise ValueError("ProjectionConfig: `ao_labels` must be a non-empty list.")

        if not (
            isinstance(self.ao_labels, list)
            and self.ao_labels
            and all(_valid_ao_label(label) for label in self.ao_labels)
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

        _helper_contributions(self.contribution, self.contribution_type)

        if self.top_n is not None:
            _helper_top_n(top_n=self.top_n)
