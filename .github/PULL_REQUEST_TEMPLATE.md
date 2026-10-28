<!-- The PR title becomes the changelog entry, so write it for users:
     `<type>(<optional scope>): <gitmoji> <description>`, `!` after the type for a breaking change.
     Bug fixes that should also reach the current release get the `backport` label.
     Use "Fixes #<issue>" to auto-close the related issue when this PR is merged. -->

## Summary

<!-- Provide a self-contained description of what this PR does and why.
     Explain the problem being solved and the approach taken. -->

## Changes

<!-- Bullet-point list of the main changes. -->

-

<!-- Keep this section above the Checklist: the changelog takes everything up to the next heading. -->
## Migration
<!-- Required for breaking changes (!) and deprecations. Delete otherwise.
     What do users need to change? Show a before/after example if possible. -->

## Checklist

- [ ] Tests added or updated to cover the changes
- [ ] Documentation updated (docstrings, `docs/`, `mkdocs.yml` nav) if needed
- [ ] `prek run --all-files` passes (ruff, ruff-format, ty)
- [ ] Breaking change or deprecation: the Migration section says what users must change

## AI/LLM disclosure

- [ ] I did not use LLM tooling, or used it only privately for ideation
- [ ] I used the following tool to help write this PR description:
- [ ] I used the following tool to generate or modify code:

<!-- Any code generated or substantially modified by an LLM must be noted inline too. -->

> [!IMPORTANT]
> By opening this PR I confirm that I have read [CONTRIBUTING.md](https://github.com/Algorithmiq/AEGISS/blob/main/CONTRIBUTING.md) and I agree to the terms of the [Contributor License Agreement](https://github.com/Algorithmiq/AEGISS/blob/main/CLA.md).

> [!WARNING]
> If you're contributing on behalf of your employer, contact [cla@algorithmiq.fi](mailto:cla@algorithmiq.fi) to arrange a Corporate CLA.
