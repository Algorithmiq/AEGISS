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

"""Tests for aegiss.utils.dump."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from aegiss.utils.dump import dump_qmi

if TYPE_CHECKING:
    from pathlib import Path


class TestDumpQmi:
    def test_round_trip(self, tmp_path: Path) -> None:
        minfo = np.array([[0.0, 0.2], [0.2, 0.0]])
        soe1 = np.array([0.4, 0.5])
        soe2 = np.array([[0.4, 0.7], [0.7, 0.5]])
        path = tmp_path / "qmi.npz"
        dump_qmi(minfo, soe1, soe2, path=path)
        with np.load(path) as data:
            np.testing.assert_array_equal(data["minfo"], minfo)
            np.testing.assert_array_equal(data["soe1"], soe1)
            np.testing.assert_array_equal(data["soe2"], soe2)

    def test_soe2_is_optional(self, tmp_path: Path) -> None:
        path = tmp_path / "qmi.npz"
        dump_qmi(np.zeros((2, 2)), np.ones(2), path=path)
        with np.load(path) as data:
            assert set(data.files) == {"minfo", "soe1"}

    def test_numpy_appends_npz_extension(self, tmp_path: Path) -> None:
        dump_qmi(np.zeros((2, 2)), np.ones(2), path=tmp_path / "qmi")
        assert (tmp_path / "qmi.npz").exists()
