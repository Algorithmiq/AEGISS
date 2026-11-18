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

"""Install the AEGISS agent skill shipped with the package.

The skill sources in ``aegiss/skill/`` link to the ``latest`` docs. Installing
rewrites those links to the docs of the installed version and records the version
in the ``SKILL.md`` frontmatter, so an agent can tell when skill and package drift
apart.
"""

from __future__ import annotations

import errno
import re
import shutil
from importlib import resources
from pathlib import Path
from typing import TYPE_CHECKING, Literal

from aegiss._version import __version__

if TYPE_CHECKING:
    from importlib.resources.abc import Traversable

SKILL_NAME = "aegiss"
DOCS_URL = "https://algorithmiq.github.io/AEGISS/"
# Agent families with their own skills folder: pi, Codex and other agents read the
# cross-agent `.agents/skills`; Claude Code reads only its own folder.
Agent = Literal["agents", "claude"]
# Project-relative skills folder per agent family.
AGENT_DIRS: dict[Agent, Path] = {
    "agents": Path(".agents") / "skills",
    "claude": Path(".claude") / "skills",
}
_RELEASE = re.compile(r"(\d+)\.(\d+)\.\d+")


def bundled_skill_dir() -> Traversable:
    """Return the directory of the skill sources shipped with the package."""
    return resources.files("aegiss") / "skill"


def docs_version(version: str) -> str:
    """Return the docs-site version that documents a package version.

    The site has one version per minor release plus ``dev`` for ``main``, and none
    for release candidates.

    Args:
        version: PEP 440 package version, e.g. ``"0.2.0"`` or ``"0.3.0.dev4+gabc"``.

    Returns:
        ``"X.Y"`` for a final release, ``"dev"`` for anything else.
    """
    match = _RELEASE.fullmatch(version)
    return f"{match[1]}.{match[2]}" if match else "dev"


def render(text: str, version: str) -> str:
    """Point the ``latest`` docs links of a skill file at the docs of a version.

    Args:
        text: Content of a skill file.
        version: Package version the links should document.

    Returns:
        The content with every ``latest/`` docs link rewritten.
    """
    return text.replace(f"{DOCS_URL}latest/", f"{DOCS_URL}{docs_version(version)}/")


def stamp_version(skill_md: str, version: str) -> str:
    """Record the package version as ``metadata.aegiss-version`` in ``SKILL.md``.

    The source ``SKILL.md`` must not declare ``metadata`` itself.

    Args:
        skill_md: Content of ``SKILL.md``, starting with its ``---`` frontmatter.
        version: Package version to record.

    Returns:
        The content with the version appended to the frontmatter.

    Raises:
        ValueError: If the content does not start with a frontmatter block.
    """
    if not skill_md.startswith("---\n") or "\n---\n" not in skill_md[3:]:
        raise ValueError("SKILL.md must start with a --- frontmatter block")
    end = skill_md.index("\n---\n", 3)
    return f'{skill_md[:end]}\nmetadata:\n  aegiss-version: "{version}"{skill_md[end:]}'


def skills_root(
    agent: Agent = "agents", *, user: bool = False, cwd: Path | None = None
) -> Path:
    """Return the skills folder an agent family reads.

    Args:
        agent: Agent family, a key of ``AGENT_DIRS``.
        user: Use the folder in the home directory instead of the project one.
        cwd: Project directory; the current directory when ``None``.

    Returns:
        The skills folder (the skill goes in its ``aegiss`` subfolder).
    """
    base = Path.home() if user else (cwd or Path.cwd())
    return base / AGENT_DIRS[agent]


def install_skill(
    skills_dir: Path,
    *,
    version: str = __version__,
    force: bool = False,
    source: Traversable | None = None,
) -> Path:
    """Copy the skill into ``skills_dir/aegiss``.

    Only the ``.md`` files of the source are copied; their docs links are rewritten
    for *version*, which is also stamped into ``SKILL.md``.

    Args:
        skills_dir: Skills folder of the agent.
        version: Package version the installed skill documents.
        force: Replace an installed copy.
        source: Skill sources; the copy shipped with the package when ``None``.

    Returns:
        The folder the skill was installed in.

    Raises:
        FileNotFoundError: If the source has no ``SKILL.md``.
        FileExistsError: If the skill is already installed and *force* is not set.
    """
    source = source or bundled_skill_dir()
    if not (source / "SKILL.md").is_file():
        raise FileNotFoundError(f"no SKILL.md in {source}")
    target = skills_dir / SKILL_NAME
    if target.exists() or target.is_symlink():
        if not force:
            raise FileExistsError(
                errno.EEXIST,
                "skill already installed; pass force=True to replace it",
                str(target),
            )
        # `npx skills add` installs a symlink; remove the link, never what it points to
        if target.is_symlink() or target.is_file():
            target.unlink()
        else:
            shutil.rmtree(target)
    target.mkdir(parents=True)
    for entry in source.iterdir():
        if not entry.name.endswith(".md"):
            continue
        text = render(entry.read_text(encoding="utf-8"), version)
        if entry.name == "SKILL.md":
            text = stamp_version(text, version)
        (target / entry.name).write_text(text, encoding="utf-8")
    return target
