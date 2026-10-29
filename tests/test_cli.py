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

"""Tests for the ``aegiss skill`` CLI."""

from __future__ import annotations

import pytest

from aegiss import cli
from aegiss.utils import skill_bundle


@pytest.fixture
def source(tmp_path, monkeypatch):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "SKILL.md").write_text(
        "---\nname: aegiss\ndescription: Test skill.\n---\n\nBody\n", encoding="utf-8"
    )
    monkeypatch.setattr(skill_bundle, "bundled_skill_dir", lambda: bundle)
    return bundle


def test_skill_path_prints_the_bundled_dir(source, capsys):
    assert cli.main(["skill", "path"]) == 0
    assert capsys.readouterr().out.strip() == str(source)


@pytest.mark.usefixtures("source")
def test_install_defaults_to_the_project_agents_dir(tmp_path, monkeypatch, capsys):
    monkeypatch.chdir(tmp_path)
    assert cli.main(["skill", "install"]) == 0
    assert (tmp_path / ".agents" / "skills" / "aegiss" / "SKILL.md").is_file()
    assert "Installed" in capsys.readouterr().out


@pytest.mark.usefixtures("source")
def test_install_for_claude_in_the_home_dir(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path))
    assert cli.main(["skill", "install", "--agent", "claude", "--user"]) == 0
    assert (tmp_path / ".claude" / "skills" / "aegiss" / "SKILL.md").is_file()


@pytest.mark.usefixtures("source")
def test_install_into_an_explicit_dir(tmp_path):
    assert cli.main(["skill", "install", str(tmp_path / "custom")]) == 0
    assert (tmp_path / "custom" / "aegiss" / "SKILL.md").is_file()


@pytest.mark.usefixtures("source")
def test_install_needs_force_to_replace(tmp_path, capsys):
    dest = str(tmp_path / "custom")
    assert cli.main(["skill", "install", dest]) == 0
    assert cli.main(["skill", "install", dest]) == 1
    assert "--force" in capsys.readouterr().err
    assert cli.main(["skill", "install", dest, "--force"]) == 0
