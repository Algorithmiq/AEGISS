# Scenarios

Answers to the agent's questions come only from the "Facts" of a scenario; anything else gets
"I don't know, use your judgement". Expected outcomes come from the AEGISS authors (interview
record, question numbers in brackets).

## Rubric

| ID | Item |
|---|---|
| O1 | The final active space is one of the acceptable ones (S3: the hand-off script would produce one) |
| P1 | A projection-only probe ran before any DMRG run |
| P2 | The margin of the high-entropy orbitals to the window edge was checked before trusting the entropies |
| P3 | Nothing above the hand-off limit ran; above it, a `select_aegiss` script with explained parameters was handed over |
| P4 | For UHF: the spin-summed or spin-resolved choice is stated and justified |
| C1 | All code the agent ran or handed over runs |
| C2 | Orbital indices are reported in the user's index base, and the base is named |
| B1 | Restricted and unrestricted return types are handled (or the unrestricted case is refused with a reason) |
| B2 | `block2main`, `SCRATCH_DIR` and the leftover scratch folders are addressed |
| E1 | The claims about AEGISS, AVAS and AutoCAS are accurate |
| E2 | The answer mentions that work using AEGISS should cite it |

## S1 — Benzene π (RHF)

Prompt:

> I have benzene in `benzene.xyz`. Use AEGISS to choose an active space for a CASSCF study of its π system.

Facts: cc-pVDZ; neutral singlet; 1-based orbital indices; a 16-core workstation with 64 GB.
Rubric: O1, P1, P2, P3, C1, C2.
Acceptable active spaces [Q20]: CAS(6,6) of the six π orbitals.

## S2 — N₂ dissociation

Prompt:

> Using AEGISS, choose an active space for a CASSCF study of N₂ dissociation. The stretched geometry is in `n2_2.0A.xyz`.

Facts: cc-pVDZ; neutral singlet; 0-based orbital indices; a 16-core workstation with 64 GB.
Rubric: O1, P1, P2, P3, C1, C2.
Acceptable active spaces [Q21]: filled in from the interview record (Task 6).

## S5 — Pipeline integration

Prompt:

> I have a PySCF workflow that builds a molecule and a converged mean-field object `mf`. Add a function `aegiss_casscf(mol, mf, ao_labels)` that selects the active space with AEGISS and returns a PySCF CASSCF object ready to run. It should also work when `mf` is UHF, or tell me clearly why not.

Facts: the workflow runs on a cluster node; jobs run in parallel.
Rubric: C1, C2, B1, B2.

## S6 — Evaluation

Prompt:

> How does AEGISS differ from AVAS? Would you use it to choose the active space for the Diels–Alder reaction of butadiene and ethylene?

Facts: none.
Rubric: E1, E2. What a good answer says [Q24]: filled in from the interview record (Task 6).
