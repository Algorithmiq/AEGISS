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

"""AEGISS entropy-based screening configuration dataclass."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class EntropyConfig:
    """Configuration dataclass for entropy-based active space screening.

    Args:
        fraction (float, default 0.1):
            Entropy threshold value (see *fraction_type*).
        fraction_type ({'r', 'a'}, default 'r'):
            ``'r'``: keep orbitals whose S(1) >= ``fraction x max(S(1))``;
            ``'a'``: keep orbitals whose S(1) ≥ ``fraction`` (absolute).
        ranking ({'subset', 'global'}, default 'subset'):
            How to rank orbitals for entropy screening.
            ``'subset'`` applies the threshold independently to occupied and virtual subspaces;
            ``'global'`` ranks all active orbitals together.
    """

    fraction: float = 0.1
    fraction_type: Literal["r", "a"] = "r"
    ranking: Literal["subset", "global"] = "subset"

    def __post_init__(self) -> None:
        """Validate the configuration parameters.

        Raises:
            ValueError: If any of the parameters are out of their expected ranges or have invalid values.
        """
        if not isinstance(self.fraction, (int, float)):
            raise ValueError(  # noqa: TRY004 -- documented as ValueError
                f"EntropyConfig: `fraction` must be a number, got {type(self.fraction).__name__}."
            )

        if self.fraction < 0 and self.fraction_type == "a":
            raise ValueError(
                f"EntropyConfig: `fraction` must be non-negative when `fraction_type` is 'a', got {self.fraction}."
            )
        if self.fraction_type not in ("r", "a"):
            raise ValueError(
                f"EntropyConfig: Invalid `fraction_type` {self.fraction_type!r}. Must be 'r' or 'a'."
            )
        if not (0 <= self.fraction <= 1) and self.fraction_type == "r":
            raise ValueError(
                f"EntropyConfig: `fraction` must be in the range [0, 1], got {self.fraction}."
            )
        if self.ranking not in ("subset", "global"):
            raise ValueError(
                f"EntropyConfig: Invalid `ranking` {self.ranking!r}. Must be 'subset' or 'global'."
            )
