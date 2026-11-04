# AEGISS in brief

Read this when the user asks what AEGISS is, how it compares with other active-space methods,
or whether to use it. Stay within what this file and the linked pages say; for finer points,
point to the paper ([arXiv:2508.10671](https://arxiv.org/abs/2508.10671)).

## What it does

AEGISS selects active spaces for multireference methods (CASSCF, CASPT2, NEVPT2) and hybrid
classical–quantum calculations. An orbital enters the space only when it is both strongly
correlated, judged by its single-orbital entropy from an approximate DMRG run, and chemically
relevant, judged by its projection onto atomic orbitals the user chooses. The six steps:
[the AEGISS procedure](https://algorithmiq.github.io/AEGISS/latest/).

## Relation to other methods

- **AVAS** selects orbitals by projection onto chosen atomic orbitals alone, and rotates the
  orbitals onto that subspace before the CAS step. AEGISS uses the same kind of projection, but
  only on the orbitals the entropy screening marked as correlated, and on unrotated orbitals:
  for benzene the two give different CASCI energies but the same CASSCF energy.
- AVAS builds one projector from all labels, so separate correlation channels (a metal and its
  ligand) end up in mixed orbitals; AEGISS selects per label group (the paper's Trans-Cl case).
- **AutoCAS** selects orbitals from DMRG entanglement measures alone, so it cannot tell which
  correlated orbitals matter for the process, and its cost grows quickly with the system.
  AEGISS screens on the single-orbital entropy in the same spirit, then keeps only the
  entropic orbitals built from the requested atomic orbitals.
- AEGISS is deliberately conservative: it may keep an orbital the user later drops by hand,
  and its results should be inspected.

## When to use it

Summarize [when to use AEGISS](https://algorithmiq.github.io/AEGISS/latest/#when-to-use-aegiss)
for the user's system.

## Citing

Work that uses AEGISS should cite the AEGISS paper and the methods it builds on:
[citing AEGISS](https://algorithmiq.github.io/AEGISS/latest/citing/).
