# API Reference

AEGISS exposes a small, focused public API organized around five concerns:

| Module | What it does |
|---|---|
| [Active Space](active_space.md) | High-level pipeline and individual screening functions |
| [Configuration](config.md) | Frozen dataclasses for all calculation settings |
| [Initial Guess](initial_guess.md) | Molecule construction and mean-field calculation |
| [Wavefunction](wavefunction.md) | DMRG and correlated wavefunction methods |
| [Localization](localization.md) | Orbital localization methods |
| [Utilities](utils.md) | Plotting, orbital export, MO composition, and helper functions |

## Top-level imports

The four configuration dataclasses are available directly from the package root:

```python
from aegiss import DMRGConfig, EntropyConfig, ProjectionConfig, OutputConfig
```

All other symbols should be imported from their respective submodules.
