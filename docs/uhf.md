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
    nelecas=(11, 9),      # (nalpha, nbeta); nalpha - nbeta must equal mol.spin
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
active windows can contain different orbitals — precisely the spin-polarised
situation where an orbital carries a *fractional* occupation (and hence
correlation) that differs between the two channels, e.g. $n_i^\alpha \approx 0.9$
while $n_i^\beta \approx 0.1$. (A cleanly singly-occupied orbital,
$n_i^\alpha = 1,\ n_i^\beta = 0$, is a product state with $H = 0$ in *both*
channels — and $S_i = 0$ spatially too — so it is not, and should not be,
flagged by either measure.)

!!! note "What spin-resolved SOE keeps and drops"
    $H(n_i^\alpha)$ and $H(n_i^\beta)$ are the two **marginals** of the joint
    two-mode (spatial-orbital) state whose entropy is the spin-summed $S_i$. By
    subadditivity $S_i \le S_i^\alpha + S_i^\beta$, with equality only when the
    two spin modes of the orbital are uncorrelated. So spin-resolved SOE
    **exposes** the per-channel asymmetry that spin-summed averages over, but at
    the cost of **discarding the intra-orbital $\alpha$–$\beta$ correlation**
    (the double-occupancy fluctuation $d_i \neq n_i^\alpha n_i^\beta$). For
    static correlation localised *within a single spatial orbital*, the
    spin-summed $S_i$ is the more faithful single number.

### Which to use

- **Spin-summed** — closed-shell-like or weakly polarised targets, or whenever a
  single spatial active space is the goal. This is the `"auto"` choice when
  $n_\alpha = n_\beta$.
- **Spin-resolved** — strongly spin-polarised / open-shell targets where the
  alpha and beta correlation patterns genuinely differ. This is the `"auto"`
  choice when $n_\alpha \neq n_\beta$ (and `symm_type="SZ"`).

### Degenerate targets

The entropies are computed in the **fixed reference $S_z$ sector** — the `SZ`
DMRG targets `spin = nalpha - nbeta`. This is well-defined and reproducible, but
for a **spin-degenerate** open-shell target (a term with $S > 0$ has an
$M_s = -S, \dots, +S$ manifold) the entropies are those of the specific $M_s$
component that `nalpha - nbeta` selects:

- This affects the **spin-summed** path too — it is the same `SZ` run — so it is
  not introduced by spin resolution.
- At the extremal $M_s$, the majority-spin closed / singly-occupied orbitals have
  integer occupation, so $H(n^\sigma) = 0$ there; the per-channel **split**
  reflects that sector's attribution (one channel can be frozen).
- What is $M_s$-robust is the *set of spatial orbitals* carrying genuine
  fractional-occupation correlation — i.e. the $\alpha \cup \beta$ **union**.
  Since a UHF active space is ultimately a set of spatial indices, that union is
  the physically meaningful object even though the split is kept for
  transparency.

A **spatially** degenerate target (a bare atom — e.g. the $^3P$ ground state of
oxygen is also orbitally three-fold degenerate) lets DMRG pick an arbitrary
component of the spatial manifold; this is a general active-space-selection
caveat, unrelated to spin resolution. Molecular targets with a non-degenerate
reference avoid it.

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

The `contribution` threshold follows the same pattern: a float or per-label list
is **shared** by both channels, while a `{"alpha": ..., "beta": ...}` dict applies
a **different** threshold to each spin channel (each value may itself be a float or
per-label list). This is orthogonal to `restricted_space` — you can share the
window but cut the channels at different weights:

```python
result = select_by_ao_projection(
    mf,                       # UHF: mo_coeff.ndim == 3
    ao_labels=["C 2pz"],
    contribution={"alpha": 0.2, "beta": 0.5},   # stricter cut on beta
    contribution_type="r",
)
```

As with a spin-resolved `restricted_space`, passing a `contribution` dict together
with a *restricted* `mf` raises a `ValueError`.

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

mol = build_mol("open_shell.xyz", basis="ccpvdz", spin=2)  # triplet: matches nelecas=(11, 9)
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

## Experimental: validating the entropies against block2

Both AEGISS entropy reconstructions are exact *only* because the reduced density
matrices are diagonal under particle-number symmetry:

- `_compute_soe` builds the 4-state **spatial** $S_i$ from the 1-PDM + 1-NPC
  diagonals;
- `_compute_spin_soe` builds the 2-state **per-spin** $H(n_i^\sigma)$ from the
  1-PDM diagonal.

block2 can measure both *directly* from an MPS with
`get_orbital_entropies(ket, orb_type=1)` (see the block2 tutorial *"Orbital
Entropy and Mutual Information"*): in **`SZ`** mode each site is a spatial orbital
(so it returns $S_i$), and in **`SGF`** mode each site is a spin-orbital (so it
returns $H(n_i^\sigma)$). Reproducing both from AEGISS's functions validates the
spin-summed and spin-resolved paths at once.

The reference must be **non-degenerate**, otherwise the `SZ` and `SGF` runs are
free to converge to different members of a spin manifold (see
[Degenerate targets](#degenerate-targets)). A stretched-H₂ **broken-symmetry
singlet** fits: $S = 0$ is non-degenerate, yet the UHF orbitals are genuinely
spin-polarised ($\alpha$ and $\beta$ localise on opposite atoms). It is reached
by following the RHF→UHF instability — no arbitrary density seeding.

!!! warning "Experimental — not part of AEGISS"

```python
import numpy as np
from pyscf import gto, scf
from pyblock2._pyscf.ao2mo import integrals as itg
from pyblock2.driver.core import DMRGDriver, SymmetryTypes

from aegiss.wavefunction.dmrg_subprocess import _compute_soe, _compute_spin_soe

# Stretched H2/STO-3G: a closed-shell singlet (non-degenerate) with a broken-
# symmetry UHF solution, reached by following the RHF->UHF instability. Because
# the state is non-degenerate, the SZ and SGF runs converge to the SAME state.
mol = gto.M(atom="H 0 0 0; H 0 0 2.5", basis="sto-3g", spin=0, verbose=0)
mf = scf.UHF(mol).run()
for _ in range(2):
    mo = mf.stability()[0]
    if np.allclose(mo, mf.mo_coeff):
        break
    mf = mf.run(mf.make_rdm1(mo, mf.mo_occ))

n = mf.mo_coeff[0].shape[1]
nsite, nelec, spin, ecore, h1e, g2e, sym = itg.get_uhf_integrals(mf, ncas=n, g2e_symm=1)
sweep = dict(n_sweeps=30, bond_dims=[100] * 30, noises=[1e-4] * 4 + [0.0],
             thrds=[1e-12] * 30, iprint=0)

# --- SZ: sites are spatial orbitals -> block2 gives the 4-state spatial S(1) ---
sz = DMRGDriver(scratch="./sz", symm_type=SymmetryTypes.SZ, n_threads=1)
sz.initialize_system(n_sites=nsite, n_elec=nelec, spin=spin, orb_sym=sym)
kz = sz.get_random_mps(tag="SZ", bond_dim=100)
sz.dmrg(sz.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0), kz, **sweep)

s_spatial = np.asarray(sz.get_orbital_entropies(kz, orb_type=1))   # block2 spatial S(1)
pdm1 = np.asarray(sz.get_1pdm(kz))                                # (2, n, n)
pdm2 = np.asarray(sz.get_2pdm(kz))                                # (3, n, n, n, n): aa, ab, bb
na, nb = np.diagonal(pdm1[0]), np.diagonal(pdm1[1])
dbl = np.array([pdm2[1][i, i, i, i] for i in range(n)])           # <n_ia n_ib> (double occ.)
npc1 = np.zeros_like(pdm1)
npc1[0] = np.diag(na + nb + 2 * dbl)                              # 1-NPC diagonal <N_i^2>
assert np.allclose(_compute_soe(pdm1, npc1), s_spatial, atol=1e-10)   # spin-summed path

# --- SGF: sites are spin-orbitals -> block2 gives the per-spin S(1) ---
# Passing the UHF (ha, hb) / (aa, ab, bb) tuples straight to get_qc_mpo lets
# block2 interleave them into its abab order (site 2p = alpha, 2p+1 = beta).
gf = DMRGDriver(scratch="./sgf", symm_type=SymmetryTypes.SGF, n_threads=1)
gf.initialize_system(n_sites=2 * n, n_elec=nelec, orb_sym=[0] * 2 * n)
kg = gf.get_random_mps(tag="SGF", bond_dim=100)
gf.dmrg(gf.get_qc_mpo(h1e=h1e, g2e=g2e, ecore=ecore, iprint=0), kg, **sweep)
soe_so = np.asarray(gf.get_orbital_entropies(kg, orb_type=1))     # per spin-orbital (abab)

Ha, Hb = _compute_spin_soe(pdm1)                                  # spin-resolved path
assert np.allclose(soe_so[0::2], Ha, atol=1e-10)                 # alpha channel
assert np.allclose(soe_so[1::2], Hb, atol=1e-10)                 # beta channel

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
print("SZ  spatial S(1) [block2]     :", s_spatial)
print("    _compute_soe (spin-summed):", _compute_soe(pdm1, npc1))
print("SGF per-spin S(1) [block2] a/b:", soe_so[0::2], soe_so[1::2])
print("    _compute_spin_soe   a/b   :", Ha, Hb)
print("subadditivity |Ha-Hb| <= S_spatial <= Ha+Hb:",
      np.abs(Ha - Hb), "<=", s_spatial, "<=", Ha + Hb)
```

Output:

```text
SZ  spatial S(1) [block2]     : [0.7439 0.7439]
    _compute_soe (spin-summed): [0.7439 0.7439]
SGF per-spin S(1) [block2] a/b: [0.6930 0.6930] [0.6930 0.6930]
    _compute_spin_soe   a/b   : [0.6930 0.6930] [0.6930 0.6930]
subadditivity |Ha-Hb| <= S_spatial <= Ha+Hb: [0.0000 0.0000] <= [0.7439 0.7439] <= [1.3860 1.3860]
```

Both AEGISS functions reproduce block2's direct measurements to $\sim\!10^{-15}$,
and the two runs share one energy (they are the same state). The numbers also
make the subadditivity note concrete: the spatial entropy ($0.7439$) strictly
**exceeds** each per-spin entropy ($0.6930$) — that gap is the intra-orbital
$\alpha$–$\beta$ correlation the spin-resolved measure marginalises out — while
staying below their sum ($1.3860$). Here $H(n^\alpha) = H(n^\beta)$ because a
singlet has $n_i^\alpha = n_i^\beta$; the genuinely *asymmetric* per-channel case
needs a spin-polarised ($n_\alpha \neq n_\beta$) target, which is the
[degenerate](#degenerate-targets) regime.
