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

"""Tests for aegiss.utils.env — environment variable reader functions."""

from __future__ import annotations

import pytest

from aegiss.utils.env import mem, omp_num_threads, pyscf_max_memory, scratch_dir

_DEFAULT_MEM = 4 * (1 << 30)


class TestMem:
    def test_unset_returns_default(self, monkeypatch) -> None:
        monkeypatch.delenv("MEM", raising=False)
        assert mem() == _DEFAULT_MEM

    def test_valid_value_returns_bytes(self, monkeypatch) -> None:
        monkeypatch.setenv("MEM", "16")
        assert mem() == 16 * (1 << 30)

    def test_valid_value_with_whitespace(self, monkeypatch) -> None:
        monkeypatch.setenv("MEM", "  8  ")
        assert mem() == 8 * (1 << 30)

    def test_zero_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("MEM", "0")
        assert mem() == _DEFAULT_MEM

    def test_zero_with_whitespace_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("MEM", "  0  ")
        assert mem() == _DEFAULT_MEM

    def test_negative_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("MEM", "-16")
        assert mem() == _DEFAULT_MEM

    def test_negative_with_whitespace_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("MEM", "  -8  ")
        assert mem() == _DEFAULT_MEM

    def test_non_integer_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("MEM", "not_an_int")
        assert mem() == _DEFAULT_MEM

    def test_non_integer_with_whitespace_falls_back_to_default(
        self, monkeypatch
    ) -> None:
        monkeypatch.setenv("MEM", "  not_an_int  ")
        assert mem() == _DEFAULT_MEM

    def test_empty_string_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("MEM", "")
        assert mem() == _DEFAULT_MEM

    def test_whitespace_only_falls_back_to_default(self, monkeypatch) -> None:
        monkeypatch.setenv("MEM", "   ")
        assert mem() == _DEFAULT_MEM


class TestOmpNumThreads:
    def test_default(self, monkeypatch) -> None:
        monkeypatch.delenv("OMP_NUM_THREADS", raising=False)
        assert omp_num_threads() == 4

    def test_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("OMP_NUM_THREADS", "8")
        assert omp_num_threads() == 8

    def test_returns_int(self, monkeypatch) -> None:
        monkeypatch.setenv("OMP_NUM_THREADS", "2")
        assert isinstance(omp_num_threads(), int)

    def test_invalid_value(self, monkeypatch) -> None:
        monkeypatch.setenv("OMP_NUM_THREADS", "not_an_int")
        with pytest.raises(
            ValueError, match="Invalid value for OMP_NUM_THREADS: 'not_an_int'"
        ):
            omp_num_threads()

    def test_empty_value(self, monkeypatch) -> None:
        monkeypatch.setenv("OMP_NUM_THREADS", "")
        assert omp_num_threads() == 4

    def test_whitespace_value(self, monkeypatch) -> None:
        monkeypatch.setenv("OMP_NUM_THREADS", "   ")
        assert omp_num_threads() == 4


class TestScratchDir:
    def test_default(self, monkeypatch) -> None:
        monkeypatch.delenv("SCRATCH_DIR", raising=False)
        assert scratch_dir() == "./tmp"

    def test_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("SCRATCH_DIR", "/custom/scratch")
        assert scratch_dir() == "/custom/scratch"

    def test_empty_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("SCRATCH_DIR", "")
        assert scratch_dir() == "./tmp"

    def test_whitespace_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("SCRATCH_DIR", "   ")
        assert scratch_dir() == "./tmp"


class TestPyscfMaxMemory:
    def test_default(self, monkeypatch) -> None:
        monkeypatch.delenv("PYSCF_MAX_MEMORY", raising=False)
        assert pyscf_max_memory() == 4000

    def test_env_var(self, monkeypatch) -> None:
        monkeypatch.setenv("PYSCF_MAX_MEMORY", "16000")
        assert pyscf_max_memory() == 16000

    def test_returns_int(self, monkeypatch) -> None:
        monkeypatch.setenv("PYSCF_MAX_MEMORY", "2000")
        assert isinstance(pyscf_max_memory(), int)

    def test_invalid_value(self, monkeypatch) -> None:
        monkeypatch.setenv("PYSCF_MAX_MEMORY", "not_an_int")
        with pytest.raises(
            ValueError, match="Invalid value for PYSCF_MAX_MEMORY: 'not_an_int'"
        ):
            pyscf_max_memory()

    def test_empty_value(self, monkeypatch) -> None:
        monkeypatch.setenv("PYSCF_MAX_MEMORY", "")
        assert pyscf_max_memory() == 4000

    def test_whitespace_value(self, monkeypatch) -> None:
        monkeypatch.setenv("PYSCF_MAX_MEMORY", "   ")
        assert pyscf_max_memory() == 4000
