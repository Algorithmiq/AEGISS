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

- **Spin-summed** — closed-shell-like or weakly polarised targets, or whenever
  both channels should be screened on one shared entropy window. This is the
  `"auto"` choice when $n_\alpha = n_\beta$.
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
- The alpha and beta selections are therefore specific to that $M_s$ sector.
  They are not merged into a union: index $k$ is a different MO in each
  channel, so the two lists stay independent (see
  [The UHF active space](#the-uhf-active-space)).

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

```{.python notest}
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

```{.python notest}
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

```{.python notest}
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

## The UHF active space

For a UHF reference the active space is **two independent lists** of MO
indices, one per spin channel: `active["alpha"]` indexes the columns of
`mf.mo_coeff[0]` and `active["beta"]` those of `mf.mo_coeff[1]`. The two lists
are never merged into a union or an intersection, since the same index denotes
different orbitals in the two channels.

The lists must satisfy one condition: **the same number of alpha and beta
orbitals**. An unrestricted CAS has a single number of active orbitals,
`ncas`, shared by both channels. AEGISS selects each channel independently and
does not enforce this, so check the two lengths and adjust the selection
before building the CAS.

The active electrons of each channel are the occupied orbitals in its list; the
remaining occupied orbitals of that channel are its core. With equal lengths,
the two lists define an unrestricted CAS in PySCF (indices are 0-based, hence
`base=0`):

```{.python notest}
from pyscf import mcscf

alpha, beta = active["alpha"], active["beta"]
assert len(alpha) == len(beta), "an unrestricted CAS needs as many alpha as beta orbitals"

nelecas = (int(mf.mo_occ[0][alpha].sum()), int(mf.mo_occ[1][beta].sum()))
mc = mcscf.UCASCI(mf, len(alpha), nelecas)
mo = mc.sort_mo((list(alpha), list(beta)), base=0)
mc.kernel(mo)
print(mc.fcisolver.spin_square(mc.ci, mc.ncas, mc.nelecas))  # (<S^2>, 2S+1)
```

Two consequences of different alpha and beta spaces:

- The CAS wavefunction is in general not an eigenfunction of $S^2$; check
  $\langle S^2 \rangle$ of the solution, as in the last line above.
- The CAS needs unrestricted solvers: `UCASCI` / `UCASSCF` in PySCF, or an
  unrestricted (`IUHF=1`) FCIDUMP for DMRG. PySCF's NEVPT2 does not accept an
  unrestricted CAS.

## Validating the entropies against an exact reference

Both AEGISS entropy reconstructions are exact *only* because the reduced density
matrices are diagonal under particle-number symmetry:

- `_compute_soe` builds the 4-state **spatial** $S_i$ from the 1-PDM + 1-NPC
  diagonals;
- `_compute_spin_soe` builds the 2-state **per-spin** $H(n_i^\sigma)$ from the
  1-PDM diagonal.

Both can be checked against the definition directly. For a system small enough for
full CI, the state is a list of determinants with weights $|c_{\alpha\beta}|^2$, and
the single-orbital reduced density matrix of orbital $i$ is diagonal in its four
local states (empty, $\alpha$, $\beta$, double). Its eigenvalues are therefore just
the summed weights of the determinants in which orbital $i$ is in each state, and
summing over one spin gives the per-spin occupations. The check below computes the
entropies this way from the CI coefficients, with no density-matrix code, and
compares them with what AEGISS's functions return from PySCF's RDMs.

The reference must be **non-degenerate** (see
[Degenerate targets](#degenerate-targets)). A stretched-H₂ **broken-symmetry
singlet** fits: $S = 0$ is non-degenerate, yet the UHF orbitals are genuinely
spin-polarised ($\alpha$ and $\beta$ localise on opposite atoms). It is reached
by following the RHF→UHF instability — no arbitrary density seeding. With two
orbitals and two electrons, PySCF's FCI is exact, so there is nothing to converge.

```python
import numpy as np
from pyscf import ao2mo, fci, gto, scf
from pyscf.fci import cistring

from aegiss.wavefunction.dmrg_subprocess import _compute_soe, _compute_spin_soe

# Stretched H2/STO-3G: a closed-shell singlet with a broken-symmetry UHF
# solution, reached by following the RHF->UHF instability.
mol = gto.M(atom="H 0 0 0; H 0 0 2.5", basis="sto-3g", spin=0, verbose=0)
mf = scf.UHF(mol).run()
for _ in range(2):
    mo = mf.stability()[0]
    if np.allclose(mo, mf.mo_coeff):
        break
    mf = mf.run(mf.make_rdm1(mo, mf.mo_occ))

# FCI in the UHF orbitals (alpha and beta orbitals differ, as in AEGISS's UHF DMRG)
ca, cb = mf.mo_coeff
n, nelec = ca.shape[1], mol.nelec
h1 = (ca.T @ mf.get_hcore() @ ca, cb.T @ mf.get_hcore() @ cb)
eri = (ao2mo.full(mol, ca), ao2mo.general(mol, (ca, ca, cb, cb)), ao2mo.full(mol, cb))
solver = fci.direct_uhf.FCISolver(mol)
e_fci, civec = solver.kernel(h1, eri, n, nelec, ecore=mol.energy_nuc())

# --- AEGISS: entropies from the RDMs, as the DMRG path computes them ---
(dm1a, dm1b), (_, dm2ab, _) = solver.make_rdm12s(civec, n, nelec)
pdm1 = np.array([dm1a, dm1b])                                     # (2, n, n)
dbl = np.array([dm2ab[i, i, i, i] for i in range(n)])             # <n_ia n_ib> (double occ.)
npc1 = np.zeros_like(pdm1)
npc1[0] = np.diag(dm1a.diagonal() + dm1b.diagonal() + 2 * dbl)    # 1-NPC diagonal <N_i^2>
s_spatial = _compute_soe(pdm1, npc1)                              # spin-summed path
Ha, Hb = _compute_spin_soe(pdm1)                                  # spin-resolved path

# --- Reference: the definition, straight from the CI coefficients ---
def entropy(p):
    return -sum(x * np.log(x) for x in p if x > 1e-14)

alpha = cistring.make_strings(range(n), nelec[0])                 # occupation bit strings
beta = cistring.make_strings(range(n), nelec[1])
weight = np.abs(civec) ** 2                                       # weight[alpha, beta]
s_ref, h_ref = np.zeros(n), np.zeros((2, n))
for i in range(n):
    p = np.zeros((2, 2))                                          # p[occ_alpha, occ_beta] of orbital i
    for ia, a in enumerate(alpha):
        for ib, b in enumerate(beta):
            p[(a >> i) & 1, (b >> i) & 1] += weight[ia, ib]
    s_ref[i] = entropy(p.ravel())                                 # 4-state spatial S(1)
    h_ref[:, i] = entropy(p.sum(axis=1)), entropy(p.sum(axis=0))  # per-spin S(1)

assert np.allclose(s_spatial, s_ref, atol=1e-10)
assert np.allclose([Ha, Hb], h_ref, atol=1e-10)

np.set_printoptions(precision=4, suppress=True, floatmode="fixed")
print("spatial S(1)  reference     :", s_ref)
print("              _compute_soe  :", s_spatial)
print("per-spin a/b  reference     :", h_ref[0], h_ref[1])
print("        _compute_spin_soe   :", Ha, Hb)
print("subadditivity |Ha-Hb| <= S_spatial <= Ha+Hb:",
      np.abs(Ha - Hb), "<=", s_spatial, "<=", Ha + Hb)
```

Output:

```text
spatial S(1)  reference     : [0.7439 0.7439]
              _compute_soe  : [0.7439 0.7439]
per-spin a/b  reference     : [0.6930 0.6930] [0.6930 0.6930]
        _compute_spin_soe   : [0.6930 0.6930] [0.6930 0.6930]
subadditivity |Ha-Hb| <= S_spatial <= Ha+Hb: [0.0000 0.0000] <= [0.7439 0.7439] <= [1.3860 1.3860]
```

Both AEGISS functions reproduce the definition to $\sim\!10^{-15}$. The numbers also
make the subadditivity note concrete: the spatial entropy ($0.7439$) strictly
**exceeds** each per-spin entropy ($0.6930$) — that gap is the intra-orbital
$\alpha$–$\beta$ correlation the spin-resolved measure marginalises out — while
staying below their sum ($1.3860$). Here $H(n^\alpha) = H(n^\beta)$ because a
singlet has $n_i^\alpha = n_i^\beta$; the genuinely *asymmetric* per-channel case
needs a spin-polarised ($n_\alpha \neq n_\beta$) target, which is the
[degenerate](#degenerate-targets) regime.
