# AO Contribution Norms: L1 vs L2

This document explains the two norm choices available for measuring how much a
molecular orbital (MO) is "composed of" a target set of atomic orbitals (AOs),
and gives guidance on which to choose for a given use case.

---

## Background

`_compute_contribution` projects each entropy-screened MO onto a set of target
AOs defined by a label such as `"Fe 3d"` or `"C 2pz"`.  The projection is done
via cross-overlap integrals between a secondary (typically minimal) basis and
the original MO coefficients:

```
s22[μ, i] = ⟨φ_μ | ψ_i⟩
```

where `φ_μ` runs over all AOs in the secondary basis that match the given label,
and `ψ_i` is the i-th MO expanded in the primary basis.

The **raw weight** `w_i` summarises how much MO `i` overlaps with the target AO
subspace.  The two norms differ in how they aggregate the individual overlaps
`s22[:, i]`.

---

## Definitions

### L1 norm (current implementation)

```
w_i^L1 = Σ_μ |⟨φ_μ | ψ_i⟩|
```

Sums the *absolute values* of the individual overlaps.

### L2 norm (squared)

```
w_i^L2 = Σ_μ |⟨φ_μ | ψ_i⟩|² = ‖s22[:, i]‖²
```

Sums the *squares* of the overlaps.  Taking the squared L2 norm (rather than
the L2 norm itself) is conventional because it equals the exact squared
projection onto the AO subspace when `{φ_μ}` are orthonormal.

---

## Physical Interpretation

| Property | L1 | L2 (squared) |
|---|---|---|
| Bounded in [0, 1] for orthonormal `{φ_μ}` | No — can be > 1 | Yes |
| Equals squared projection onto AO subspace | No | Yes |
| Probabilistic meaning ("fraction of MO in subspace") | No | Yes |
| Sensitive to number of matched AOs | Strongly (linear) | Mildly (quadratic dampening) |
| Sensitive to sign pattern of overlaps | No (absolute values) | No (squares) |

With a minimal basis (`minao`) the projection AOs are approximately orthonormal
within each atom, so L2² is close to the true squared projection and stays in
`[0, 1]`.  The L1 weight has no such bound: for a label matching N AOs each
with overlap 0.4, L1 gives `0.4N` while L2 gives `0.16N`.  For the five Fe 3d
functions this is a factor of 5 difference.

---

## Worked Example

Consider two MOs and the label `"Fe 3d"` (five projection AOs):

| MO | overlaps `s22[:, i]` | L1 | L2² |
|---|---|---|---|
| A — delocalised over all 3d | `[0.4, 0.4, 0.4, 0.4, 0.4]` | 2.0 | 0.80 |
| B — concentrated on one 3d | `[0.9, 0.05, 0.05, 0.05, 0.05]` | 1.1 | 0.82 |
| C — strong on two, zero elsewhere | `[0.6, 0.6, 0, 0, 0]` | 1.2 | 0.72 |

With L1, MO A looks far more "Fe 3d" than MO B or C — but its individual
overlap with each 3d function is only 0.4, while MO B has a single dominant
0.9 coefficient.  With L2², MOs A and B are nearly equivalent (both ~0.8),
which better reflects that both MOs spend roughly 80 % of their weight in the
Fe 3d subspace.

---

## Pros and Cons

### L1

**Pros**
- Simple and cheap to compute.
- Rewards MOs that have consistent, same-sign overlap with many projection AOs
  (constructive interference patterns).
- Equivalent to L2 when only one AO matches the label (single `s` or `p_z`
  orbital in a symmetric environment).

**Cons**
- Not bounded: weights can exceed 1 and grow with the number of matched AOs,
  making thresholds (`contribution`) non-transferable between labels or systems.
- No probabilistic meaning: it is not the fraction of the MO in the AO subspace.
- Sensitive to the size of the projection basis set: adding more diffuse
  functions matching the label inflates the weight even if the physical content
  is unchanged.

### L2 (squared)

**Pros**
- Directly equals the squared projection norm onto the AO subspace (for
  orthonormal `{φ_μ}`), giving a value in `[0, 1]` with clear meaning: "the
  fraction of MO i that lies in the Fe 3d subspace is `w_i^L2`."
- Transferable thresholds: a threshold of 0.5 means the same thing regardless
  of whether the label matches 1, 5, or 10 AOs.
- Consistent with Mulliken/Löwdin projected populations used in the broader
  electronic-structure literature.

**Cons**
- Mildly penalises MOs that are uniformly delocalized over many AOs: each
  individual overlap is small, and squaring makes them smaller still before
  summing.  This is usually the *correct* behaviour (a fully delocalised MO
  genuinely has less Fe 3d character than a localised one), but it may miss
  cases where many weak overlaps collectively signal relevant AO character.
- Slightly more expensive (squaring before summing, though negligible in
  practice).

---

## When to Use Each

### Use L2 (recommended for most cases)

- The target label matches **multiple AOs** (e.g. `"Fe 3d"`, `"N 2p"`, a
  whole shell).
- You need **physically interpretable and transferable thresholds**: setting
  `contribution=0.3` means "select MOs with at least 30 % Fe 3d character."
- You are comparing selections across **different molecules or basis sets**.
- The intent is to identify MOs whose *dominant* character corresponds to the
  target process or transition (e.g., Fe→L charge transfer).

### Use L1

- The target label matches exactly **one AO** (e.g. `"C.1 2pz"` on a
  symmetric molecule), in which case L1 and L2 (un-squared) are equivalent up
  to the square.
- You specifically want to reward **constructive interference** across multiple
  AOs — for instance, when looking for MOs that are in-phase combinations of
  many ligand p functions.
- You are doing a quick heuristic screening and absolute scaling does not matter
  (you will use relative thresholds or `top_n` regardless).

---

## Subtlety: Non-Orthogonal Projection AOs

Both L1 and L2 as defined above assume the projection AOs `{φ_μ}` are
approximately orthogonal.  If they are not (e.g., when using a large secondary
basis), the true squared projection is

```
w_i^exact = c^T (S_proj)^{-1} c,    c_μ = ⟨φ_μ | ψ_i⟩
```

where `S_proj` is the overlap matrix *among the projection AOs*.  Using
`minao` as the secondary basis keeps this correction small, because minimal
basis AOs on different atoms have small mutual overlaps.  If you switch to a
larger secondary basis (e.g., `def2-svp`), the uncorrected L2² can exceed 1
for the same physical MO, in which case the correction above becomes important.

---

## Summary Recommendation

For AEGISS's intended use case — **ranking entropy-screened MOs by their
character in a target AO subspace relevant to a specific mechanism or
transition** — **L2 (squared)** is the more rigorous and physically
meaningful choice.  It gives a bounded, probabilistic weight that is
directly comparable across labels and systems, making the `contribution`
threshold an interpretable and transferable parameter.
