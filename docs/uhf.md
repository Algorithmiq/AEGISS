# Unrestricted (UHF) references

AEGISS supports **unrestricted (UHF)** mean-fields end to end. Because the alpha
and beta molecular orbitals of a UHF reference are *different* sets of functions,
the pipeline runs **per spin channel** from the DMRG single-orbital entropies all
the way to the final orbital indices, and returns the alpha and beta selections
**separately** — they are never reconciled into one spatial index set.

This page describes how the three regimes (restricted, UHF spin-summed, UHF
spin-resolved) differ, how to configure them, and how the spin-summed and
spin-resolved single-orbital entropies (SOE) compare.

## Workflow at a glance

The same four stages run in every regime; what is *shared* versus *split* between
the spin channels is the only thing that changes.

| Stage | RHF | UHF — summed | UHF — resolved |
|---|---|---|---|
| DMRG SOE (`run_dmrg_subprocess`) | one 1-D list | one 1-D list (spin-summed) | one `(2, ncas)` array — α and β rows |
| Entropy screen | `select_by_entropy` → 1 window | `select_by_entropy` → 1 window | `select_by_spin_entropy` → **2** windows |
| MO coefficients | one `(nao, nmo)` matrix | two blocks `mo_coeff[0/1]` | two blocks `mo_coeff[0/1]` |
| AO contributions | one pass | two passes, **shared** window | two passes, **own** window each |
| Result | spatial `ndarray` | `{"alpha", "beta"}` | `{"alpha", "beta"}` |

!!! note "The only difference between summed and resolved is the search space"
    From the MO-coefficient split onward, UHF-summed and UHF-resolved are
    identical: two coefficient blocks → two AO-contribution lists → a split
    `alpha`/`beta` selection. They differ *only* in the entropy screening step —
    resolved gives each channel its **own** window, summed **shares** the single
    spin-summed window across both channels.

A **restricted** `mf` (2-D `mo_coeff`) always takes the RHF column regardless of
configuration: there is only one coefficient set to project.

## Configuration

Three `DMRGConfig` fields control the UHF behaviour:

```python
from aegiss.config_dataclasses import DMRGConfig

dmrg_config = DMRGConfig(
    ncas=20,
    nelecas=(11, 9),      # (nalpha, nbeta) for an open-shell target
    symm_type="SZ",       # required for UHF and for spin-resolved SOE
    unrestricted=None,    # None → autodetect isinstance(mf, scf.uhf.UHF)
    spin_soe="auto",      # "auto" | "summed" | "resolved"
)
```

- **`unrestricted`** — whether to emit an unrestricted (`IUHF=1`) FCIDUMP with
  separate `aa`/`bb`/`ab` integral blocks. `None` (default) autodetects from the
  mean-field type; pass `True`/`False` to override. UHF requires `symm_type="SZ"`.
- **`symm_type`** — must be `"SZ"` for UHF (the spin-adapted `"SU2"` path cannot
  represent an unrestricted state), and also for spin-resolved SOE (the
  per-spin 1-PDM is only available without spin adaptation).
- **`spin_soe`** — selects spin-summed or spin-resolved single-orbital
  entropies. The resolution rule (`DMRGConfig.spin_resolved()`):

    | `spin_soe` | Result |
    |---|---|
    | `"resolved"` | spin-resolved (requires `symm_type="SZ"`) |
    | `"summed"` | spin-summed |
    | `"auto"` | resolved **iff** `nalpha != nbeta` *and* `symm_type="SZ"`, else summed |

!!! warning "Reference spin must match the active spin"
    For a UHF reference a single active-orbital window only exists when the
    frozen cores of the two spin channels coincide, i.e. the reference spin
    polarisation equals the requested active spin:
    $N_\alpha^{\text{ref}} - N_\beta^{\text{ref}} = n_\alpha - n_\beta$.
    `get_subspaces_from_cas` raises a `ValueError` otherwise.

## Spin-summed vs. spin-resolved SOE

The single-orbital entropy quantifies how entangled an orbital is with the rest
of the active space. What "an orbital" means differs between the two flavours.

### Spin-summed (4-state spatial orbital)

Each **spatial** orbital is treated as a four-state subsystem — empty,
alpha-only, beta-only, doubly occupied — so its entropy is

$$
S_i = -\sum_{\mu \in \{d,\, \alpha,\, \beta,\, 0\}} \omega_\mu \ln \omega_\mu,
\qquad
\omega = \bigl\{\, d_i,\; n_i^\alpha - d_i,\; n_i^\beta - d_i,\; 1 - n_i + d_i \,\bigr\},
$$

where $n_i = n_i^\alpha + n_i^\beta$ is the spatial occupation and $d_i$ the
double occupation (recovered from the 1-PDM and 1-NPC diagonals). This yields a
single ranking and, downstream, a single spatial active-space window.

### Spin-resolved (2-state spin-orbital)

Each **spin-orbital** is a two-state mode (occupied / empty), so its entropy is
the binary entropy of the spin-orbital occupation $n_i^\sigma = \text{PDM}_\sigma[i,i]$:

$$
H(n_i^\sigma) = -\,n_i^\sigma \ln n_i^\sigma - (1 - n_i^\sigma)\ln(1 - n_i^\sigma),
\qquad \sigma \in \{\alpha, \beta\}.
$$

The alpha and beta channels are then thresholded **independently**, so the two
active windows can contain different orbitals — precisely the open-shell
situation where a singly-occupied orbital (SOMO) is entangled in one spin
channel and frozen in the other.

!!! note "Relationship between the two"
    By subadditivity the spin-summed spatial entropy is bounded by the sum of
    the per-spin entropies, $S_i \le S_i^\alpha + S_i^\beta$, with equality only
    when the two spin modes of the orbital are uncorrelated. Spin-summed hides
    the per-channel asymmetry that spin-resolved exposes.

### Which to use

- **Spin-summed** — closed-shell-like or weakly polarised targets, or whenever a
  single spatial active space is the goal. This is the `"auto"` choice when
  $n_\alpha = n_\beta$.
- **Spin-resolved** — strongly spin-polarised / open-shell targets where the
  alpha and beta correlation patterns genuinely differ. This is the `"auto"`
  choice when $n_\alpha \neq n_\beta$ (and `symm_type="SZ"`).

## AO projection on a UHF reference

`select_by_ao_projection` detects a UHF reference from a 3-D `mf.mo_coeff` of
shape `(2, nao, nmo)` and screens each spin block separately. The AO overlaps
($\langle \text{minao}_\mu | \text{AO}_\nu \rangle$ via `intor_cross`) are
**spin-independent**, so only the MO coefficients differ between channels.

```python
from aegiss.active_space import select_by_ao_projection

result = select_by_ao_projection(
    mf,                       # UHF: mo_coeff.ndim == 3
    ao_labels=["C 2pz"],
    contribution=0.5,
    contribution_type="a",
)
# result == {"alpha": np.ndarray, "beta": np.ndarray}
```

The `restricted_space` argument decides whether the channels share a window:

- a plain list / `(occ, virt)` tuple → **shared** by both channels (spin-summed);
- a `{"alpha": ..., "beta": ...}` dict (as produced by `select_by_spin_entropy`)
  → each channel is restricted to its **own** window (spin-resolved).

Passing a spin-resolved dict together with a *restricted* `mf` raises a
`ValueError`.

## Full pipeline

`select_aegiss` wires everything together. Its return type reflects the
reference: a single `np.ndarray` for a restricted mean-field, or a split
`{"alpha": ..., "beta": ...}` mapping for a UHF one.

```python
from aegiss.initial_guess import build_mol, run_scf
from aegiss.active_space import select_aegiss
from aegiss.config_dataclasses import (
    DMRGConfig, EntropyConfig, ProjectionConfig, OutputConfig,
)

mol = build_mol("open_shell.xyz", basis="ccpvdz")
mf  = run_scf(mol, method="UHF")

active = select_aegiss(
    mol, mf,
    dmrg_config=DMRGConfig(ncas=20, nelecas=(11, 9), symm_type="SZ", spin_soe="resolved"),
    entropy_config=EntropyConfig(fraction=0.1, fraction_type="r", ranking="global"),
    projection_config=ProjectionConfig(ao_labels=["C 2pz"], contribution=0.5),
    output_config=OutputConfig(saving_folder="./orbs/"),
)
# active == {"alpha": np.ndarray, "beta": np.ndarray}
```

!!! warning "The alpha and beta selections are not reconciled"
    Index $k$ refers to a *different* MO in each spin channel, so there is no
    common spatial active space to merge them into — the split is kept
    deliberately. When output is enabled, cube files and any per-channel plots
    are written with `_alpha` / `_beta` tags; the AO pie-contribution plot (which
    expects a single flat orbital list) is skipped for the split selection.
