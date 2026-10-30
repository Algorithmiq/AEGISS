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

"""Tests for the agent skill shipped in aegiss/skill/."""

from __future__ import annotations

import re
import shutil
import subprocess
import zipfile
from pathlib import Path

import pytest

from aegiss.utils.skill_bundle import DOCS_URL

ROOT = Path(__file__).resolve().parents[1]
SKILL_DIR = ROOT / "src" / "aegiss" / "skill"
SKILL_FILES = ("SKILL.md", "parameters.md", "integration.md", "overview.md")
SITE_LINK = re.compile(re.escape(DOCS_URL) + r"([^\s)>\]]*)")
RELATIVE_LINK = re.compile(r"\]\(([\w-]+\.md)(?:#[\w-]*)?\)")


def _docs_source(page: str) -> Path | None:
    """Return the docs source file that renders as *page* on the site."""
    docs = ROOT / "docs"
    if not page:
        candidates = [docs / "index.md"]
    else:
        candidates = [
            docs / f"{page}.md",
            docs / page / "index.md",
            docs / f"{page}.ipynb",
        ]
    return next((path for path in candidates if path.is_file()), None)


def _site_links() -> list[tuple[str, str]]:
    return [
        (path.name, match[1])
        for path in sorted(SKILL_DIR.glob("*.md"))
        for match in SITE_LINK.finditer(path.read_text(encoding="utf-8"))
    ]


def test_skill_files_present():
    assert sorted(p.name for p in SKILL_DIR.glob("*.md")) == sorted(SKILL_FILES)


@pytest.mark.parametrize(("file", "path"), _site_links())
def test_site_links_point_at_a_latest_docs_page(file, path):
    assert path.startswith("latest/"), (
        f"{file}: write docs links as latest/, install pins them"
    )
    page = path.removeprefix("latest/").split("#", 1)[0].strip("/")
    assert _docs_source(page) is not None, f"{file}: no docs page renders as {path}"


def test_relative_links_resolve():
    for path in SKILL_DIR.glob("*.md"):
        for match in RELATIVE_LINK.finditer(path.read_text(encoding="utf-8")):
            assert (SKILL_DIR / match[1]).is_file(), (
                f"{path.name}: {match[1]} is missing"
            )


def test_frontmatter():
    text = (SKILL_DIR / "SKILL.md").read_text(encoding="utf-8")
    assert text.startswith("---\n")
    front = text[4 : text.index("\n---\n", 3)].splitlines()
    fields = dict(line.split(": ", 1) for line in front)
    assert fields["name"] == "aegiss"
    assert 0 < len(fields["description"]) <= 1024
    # the installer appends metadata.aegiss-version; a source copy would duplicate the key
    assert "metadata" not in fields


@pytest.mark.skipif(shutil.which("uv") is None, reason="needs uv to build the wheel")
def test_wheel_ships_the_skill(tmp_path):
    uv = shutil.which("uv")
    subprocess.run(  # noqa: S603 (fixed arguments)
        [uv, "build", "--wheel", "--out-dir", str(tmp_path), str(ROOT)],
        check=True,
        capture_output=True,
    )
    (wheel,) = tmp_path.glob("aegiss-*.whl")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
    assert {f"aegiss/skill/{name}" for name in SKILL_FILES} <= names
