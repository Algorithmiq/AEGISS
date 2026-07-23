# Orbital index base

AEGISS **stores and computes every orbital index 0-based** — the PySCF
convention. Many quantum-chemistry users, however, prefer to *read* and *type*
orbital indices **1-based** (the Fortran convention used by ORCA, Molpro,
Molcas, …). AEGISS lets you choose that base for **display and input** without
ever changing what happens internally.

The base affects only:

- how indices are **printed** — logs, plot labels, cube-file names; and
- how indices you **pass in** are interpreted before being converted back to
  the internal 0-based representation.

It never affects stored values, array slicing, or the indices **returned** by
the API — those are always 0-based, so results round-trip safely regardless of
the setting.

## Choosing the base

Set it once after installation with the `aegiss` CLI:

```bash
aegiss config set-base 1      # print/read orbital indices 1-based (ORCA/Molpro style)
aegiss config set-base 0      # back to the 0-based default
aegiss config get-base        # show the currently resolved base
aegiss config path            # show the config-file location
```

The choice is persisted to `${XDG_CONFIG_HOME:-~/.config}/aegiss/config.toml`.

### Resolution precedence

The base is resolved on first use (then cached for the process) in this order:

| Priority | Source | Notes |
|---|---|---|
| 1 (highest) | `AEGISS_ORB_BASE` environment variable | Per-run override; wins over the file |
| 2 | `orbital_base` in the config file | Written by `aegiss config set-base` |
| 3 | Default | `0` |

```bash
# Force 0-based for a single run, even if config.toml says orbital_base = 1:
AEGISS_ORB_BASE=0 python your_script.py
```

!!! note "Set it before the pipeline runs"
    `orbital_base()` is resolved lazily on the first index that is printed or
    interpreted, then cached. Set `AEGISS_ORB_BASE` (or the shell prefix) *before*
    calling any AEGISS selection function. If you change it mid-process — e.g. in a
    notebook — call `aegiss.utils.orb_display.reset_cache()` afterwards.

## The input/output contract

!!! important "Inputs use the configured base, but return values are always 0-based"
    This asymmetry is deliberate — it keeps returned indices safe to slice
    `mo_coeff[:, idx]`, save to disk, and feed back into the pipeline regardless
    of the base.

With `AEGISS_ORB_BASE=1`:

```python
from aegiss.active_space.ao_projection import select_by_ao_projection

# You pass 1-based indices …
selected = select_by_ao_projection(
    mf, ao_labels=["C 2pz"], restricted_space=[35, 36, 37]
)
# … but `selected` comes back 0-based:
#   array([34, 35, 36])
```

**Passing indices.** Give them in your configured base. Affected parameters:

| Function | Parameter |
|---|---|
| `select_by_ao_projection` | `restricted_space` (a plain `list[int]` only) |
| `mo_ao_contributions` | `mo_indices` |
| `localize_fnos` | `active`, `frozen` |

An index below the base — e.g. `0` when the base is `1` — raises `ValueError`
rather than silently producing a negative index:

```pycon
>>> select_by_ao_projection(mf, ao_labels=["C 2pz"], restricted_space=[0, 1])
ValueError: Orbital indices [0] are invalid for base 1 (expected >= 1).
```

!!! warning "Don't re-feed returned indices as base-1 input"
    Because results are 0-based, passing a returned array straight back in as a
    plain index list while in base 1 would shift it **down a second time**
    (`[34, 35, 36]` → `[33, 34, 35]`). Returned indices are the canonical 0-based
    form meant for slicing and saving — not for re-typing as user input. To
    display them in your base, wrap them:

    ```python
    from aegiss.utils.orb_display import to_display_all
    to_display_all(selected)   # [35, 36, 37] under base 1
    ```

### What stays 0-based

Only a **plain list of integers** is treated as base-dependent user input.
Values produced by AEGISS internals are always 0-based and are never re-shifted:

- `(index, entropy)` pair-lists and `(occ_list, virt_list)` tuples from
  `select_by_entropy` (as passed through `restricted_space`);
- every value **returned** by the public API (`select_by_ao_projection`,
  `select_aegiss`, `mo_ao_contributions` dict keys, `get_subspaces_from_cas`, …).

This is why a full `select_aegiss` run is unaffected internally: it hands the
pair/tuple forms between stages, so no double-shift can occur even under
`AEGISS_ORB_BASE=1` — the only visible change is the base shown in the logs.

## Programmatic access

The resolver and converters live in `aegiss.utils.orb_display`:

```python
from aegiss.utils.orb_display import (
    orbital_base,     # -> 0 or 1, the resolved base
    to_display,       # 0-based int      -> display base
    to_display_all,   # 0-based iterable -> display base (list)
    from_display,     # display-base int      -> 0-based (raises below base)
    from_display_all, # display-base iterable -> 0-based (raises below base)
    reset_cache,      # clear the cached base after changing env/config
)
```
