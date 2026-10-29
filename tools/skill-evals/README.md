# Agent skill evaluation

Fixed scenarios that show whether the AEGISS agent skill (`src/aegiss/skill/`) changes what a
coding agent does. Each scenario runs twice in a fresh agent session: once without the skill
(baseline) and once with it. Not part of CI: it needs an agent and, for some scenarios, DMRG.

Rerun the scenarios whenever `src/aegiss/skill/` or the public API changes, and record the runs
in `results.md`.

## Running a scenario

Every run starts from an empty scratch directory with AEGISS installed from this checkout, so
the agent sees neither this repository nor its `AGENTS.md`:

```bash
repo="$(git rev-parse --show-toplevel)"
scratch="$(mktemp -d)"
cp "$repo"/tools/skill-evals/data/* "$repo"/docs/tutorials/data/benzene.xyz "$scratch"/
cd "$scratch"
uv venv --python 3.12
uv pip install "$repo"
source .venv/bin/activate
aegiss skill install          # with-skill runs only (pi, Codex, ...)
# aegiss skill install --agent claude   # instead, for Claude Code
```

Then start a fresh session of the agent in `$scratch` (same agent and model for the baseline
and the with-skill run), paste the scenario prompt from `scenarios.md` verbatim, and answer the
agent's questions with the facts listed under the scenario. When the agent stops, grade the run
against the rubric and add one row to `results.md`; keep the transcript outside the repository.

## Grading

Each rubric item passes or not; an item that does not apply is `n/a`. The skill passes when
every applicable item passes in every scenario with the skill, and in at least one scenario an item that
does not pass in the baseline passes with the skill.
