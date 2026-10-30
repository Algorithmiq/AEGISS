# Scenarios

Answers to the agent's questions come only from the "Facts" of a scenario; anything else gets
"I don't know, use your judgement". Expected outcomes come from the AEGISS authors (interview
record, question numbers in brackets) and the preprint (arXiv:2508.10671, sections in
brackets).

## Rubric

| ID | Item |
|---|---|
| O1 | The final active space is one of the acceptable ones (when the production run is handed off, the handed-off script would produce one) |
| P1 | A projection-only probe ran before any DMRG run |
| P2 | The margin of the high-entropy orbitals to the window edge was checked before trusting the entropies |
| P3 | Nothing that reaches the hand-off limit ran; for a handed-off production run, a `select_aegiss` script with explained parameters; for a handed-off cheap run, a script that saves the entropies |
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
Acceptable active spaces [Q20; preprint §4.1]: CAS(6,6), the three bonding and three antibonding π orbitals. The CAS(4,4) entropy plateau misses a bonding/antibonding pair and does not pass. Reference settings: CAS(30,30) window, `max_bond_dim` 500, relative τ_E 0.2, `"C 2pz"` with contribution 0.5 (the molecule lies in the xy plane).

## S2 — DOBNA (π system with heteroatoms)

Prompt:

> Use AEGISS to choose an active space for a state-averaged CASSCF study of the lowest singlet and triplet excited states of DOBNA. The geometry is in `dobna.xyz`.

Facts: 6-31G(d); neutral singlet; canonical RHF orbitals; 0-based orbital indices; a 16-core workstation with 64 GB.
Rubric: O1, P1, P2, P3, C1, C2.
Acceptable active spaces [preprint §4.3, Table 5, SI E.2]: from canonical RHF, CAS(18,18) with relative τ_E 0.1 or CAS(10,10) with τ_E 0.9, both with `["C 2px", "B 2px"]` (the molecule lies in the yz plane; the preprint's SI E.2 writes 2pz, Table 5 writes 2px) and contribution 0.5. If the agent also labels O 2px (parameters.md lists heteroatom labels), grade O1 on the C/B π orbitals inside its selection and note the O-dominated extras as over-inclusion, not a miss. If the agent moves to MP2 natural orbitals, CAS(18,18) or CAS(10,10) also pass. Reference window: CAS(40,40), `max_bond_dim` 500. A good run notices the Rydberg-like virtual orbitals among the first ten virtuals, and hands the user a script for the production DMRG (window ≥ 30 and `max_bond_dim` 500).

## S3 — Ferrocene (3d transition metal)

Prompt:

> Use AEGISS to choose an active space for a CASSCF/NEVPT2 study of the low-lying d–d excited states of ferrocene. The geometry is in `ferrocene_avas.xyz`.

Facts: cc-pVTZ-DK with scalar-relativistic X2C (`run_scf(..., relativistic="x2c")`); neutral singlet; canonical RHF orbitals; 1-based orbital indices; a 16-core workstation with 64 GB.
Rubric: O1, P1, P2, P3, C1, C2.
Acceptable active spaces [preprint §4.2, Table 3, SI E.1]: CAS(10e,7o): five occupied orbitals (three pure Fe 3d, two Fe 3dxz/3dyz mixed with Cp 2pz) and two virtual Fe 3d orbitals, from the single label group `"Fe 3d"` with contribution 0.5 (the same space AVAS finds). Reference window: CAS(40,40), `max_bond_dim` 500, relative τ_E 0.1, about 26 orbitals after the entropy step. The production DMRG is above the hand-off limit (window ≥ 30 and `max_bond_dim` 500): a passing run hands the user a script for it. The basis follows the preprint's Table 3 (its SI E.1 says cc-pVDZ-DK); the orbital count after the entropy step is approximate.

## S4 — Benzene radical cation (UHF)

Prompt:

> Use AEGISS with a UHF reference to choose an active space for the benzene radical cation. The neutral geometry is in `benzene.xyz`.

Facts: charge +1, doublet (`spin=1`); cc-pVDZ; 0-based orbital indices; a 16-core workstation with 64 GB.
Rubric: O1, P1, P2, P3, P4, C1, C2.
Acceptable active spaces [Q23]: pending: the authors are asked for the acceptable active spaces per spin channel and for the spin-summed or spin-resolved choice. Until then grade every item except O1, and record O1 as `n/a (pending)`. The skill flags main-group radicals as exploratory; a passing run tells the user so.

## S5 — Pipeline integration

Prompt:

> I have a PySCF workflow that builds a molecule and a converged mean-field object `mf`. Add a function `aegiss_casscf(mol, mf, ao_labels)` that selects the active space with AEGISS and returns a PySCF CASSCF object ready to run. It should also work when `mf` is UHF, or tell me clearly why not.

Facts: the workflow runs on a cluster node; jobs run in parallel; 1-based orbital indices.
Rubric: C1, C2, B1, B2.

## S6 — Evaluation

Prompt:

> How does AEGISS differ from AVAS? Would you use it to choose the active space for the Diels–Alder reaction of butadiene and ethylene?

Facts: none.
Rubric: E1, E2.
What a good answer says [interview A18; preprint §4.1, §4.4.1, §5]:
- AVAS selects by projection onto the chosen atomic orbitals alone and rotates the orbitals onto that subspace before the CAS step; it builds one projector from all labels, so separate correlation channels (metal and ligand) end up mixed.
- AEGISS first screens orbitals by single-orbital entropy from an approximate DMRG run, then projects only the correlated ones, per label group, on the unrotated orbitals.
- AutoCAS uses the entropy alone and cannot tell which correlated orbitals matter for the process.
- For the Diels–Alder reaction: AEGISS has not been tested along reaction paths; it can be applied geometry by geometry, but the selection must be checked for consistency along the path, and the user should be told this.
- A claim that AEGISS was validated on reactions, or that it needs no DMRG run, does not pass E1.
