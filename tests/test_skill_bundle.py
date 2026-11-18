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

"""Tests for aegiss.utils.skill_bundle."""

from __future__ import annotations

import pytest

from aegiss.utils.skill_bundle import (
    DOCS_URL,
    docs_version,
    install_skill,
    render,
    skills_root,
    stamp_version,
)

SKILL_MD = (
    "---\nname: aegiss\ndescription: Test skill.\n---\n\n"
    f"# Body\n\nSee [UHF]({DOCS_URL}latest/uhf/).\n"
)


@pytest.fixture
def source(tmp_path):
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "SKILL.md").write_text(SKILL_MD, encoding="utf-8")
    (bundle / "parameters.md").write_text(
        f"[Rules]({DOCS_URL}latest/parameters/)\n", encoding="utf-8"
    )
    (bundle / "notes.txt").write_text("not part of the skill\n", encoding="utf-8")
    return bundle


@pytest.mark.parametrize(
    ("version", "expected"),
    [
        ("0.1.0", "0.1"),
        ("1.12.3", "1.12"),
        ("0.2.0.dev3", "dev"),
        ("0.2.0.dev1+g9b5f644.d19800101", "dev"),
        ("0.2.0rc1", "dev"),
        ("0.1.1+local", "dev"),
    ],
)
def test_docs_version(version, expected):
    assert docs_version(version) == expected


def test_render_rewrites_only_latest_links():
    text = f"{DOCS_URL}latest/uhf/ and {DOCS_URL}0.1/"
    assert render(text, "0.3.0") == f"{DOCS_URL}0.3/uhf/ and {DOCS_URL}0.1/"


def test_stamp_version_extends_the_frontmatter():
    front, body = stamp_version(SKILL_MD, "0.3.0").split("\n---\n", 1)
    assert front.endswith('metadata:\n  aegiss-version: "0.3.0"')
    assert body == SKILL_MD.split("\n---\n", 1)[1]


def test_stamp_version_requires_frontmatter():
    with pytest.raises(ValueError, match="frontmatter"):
        stamp_version("# no frontmatter\n", "0.3.0")


def test_install_copies_renders_and_stamps(tmp_path, source):
    target = install_skill(tmp_path / "skills", version="0.3.0", source=source)
    assert target == tmp_path / "skills" / "aegiss"
    assert sorted(p.name for p in target.iterdir()) == ["SKILL.md", "parameters.md"]
    skill_md = (target / "SKILL.md").read_text(encoding="utf-8")
    assert f"{DOCS_URL}0.3/uhf/" in skill_md
    assert 'aegiss-version: "0.3.0"' in skill_md
    assert (target / "parameters.md").read_text(encoding="utf-8") == (
        f"[Rules]({DOCS_URL}0.3/parameters/)\n"
    )


def test_install_refuses_to_overwrite(tmp_path, source):
    target = install_skill(tmp_path, version="0.3.0", source=source)
    with pytest.raises(FileExistsError, match="force") as excinfo:
        install_skill(tmp_path, version="0.3.0", source=source)
    assert excinfo.value.filename == str(target)


def test_install_force_replaces_the_old_copy(tmp_path, source):
    target = install_skill(tmp_path, version="0.3.0", source=source)
    (target / "stale.md").write_text("old\n", encoding="utf-8")
    install_skill(tmp_path, version="0.4.0", source=source, force=True)
    assert not (target / "stale.md").exists()
    assert 'aegiss-version: "0.4.0"' in (target / "SKILL.md").read_text(
        encoding="utf-8"
    )


def test_install_force_replaces_a_symlink_and_keeps_its_target(tmp_path, source):
    # `npx skills add` links .claude/skills/aegiss to a copy under .agents/skills/
    shared = tmp_path / "shared"
    shared.mkdir()
    (shared / "SKILL.md").write_text("old\n", encoding="utf-8")
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "aegiss").symlink_to(shared, target_is_directory=True)
    with pytest.raises(FileExistsError, match="force"):
        install_skill(skills, version="0.3.0", source=source)
    target = install_skill(skills, version="0.3.0", source=source, force=True)
    assert not target.is_symlink()
    assert 'aegiss-version: "0.3.0"' in (target / "SKILL.md").read_text(
        encoding="utf-8"
    )
    assert (shared / "SKILL.md").read_text(encoding="utf-8") == "old\n"


def test_install_force_replaces_a_dangling_symlink(tmp_path, source):
    skills = tmp_path / "skills"
    skills.mkdir()
    (skills / "aegiss").symlink_to(tmp_path / "missing", target_is_directory=True)
    with pytest.raises(FileExistsError, match="force"):
        install_skill(skills, version="0.3.0", source=source)
    target = install_skill(skills, version="0.3.0", source=source, force=True)
    assert not target.is_symlink()
    assert (target / "SKILL.md").is_file()


def test_install_requires_skill_md(tmp_path, source):
    (source / "SKILL.md").unlink()
    with pytest.raises(FileNotFoundError, match=r"SKILL\.md"):
        install_skill(tmp_path / "skills", version="0.3.0", source=source)


def test_skills_root(tmp_path, monkeypatch):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    assert skills_root("agents", cwd=tmp_path) == tmp_path / ".agents" / "skills"
    assert skills_root("claude", cwd=tmp_path) == tmp_path / ".claude" / "skills"
    assert skills_root("agents", user=True) == tmp_path / "home" / ".agents" / "skills"
