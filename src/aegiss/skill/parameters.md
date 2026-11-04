Rules from the AEGISS authors ([preprint](https://arxiv.org/abs/2508.10671) and an interview). Where
the authors have no tested recipe the text says so: a selection made there is exploratory and the
user must be told. Pass every threshold explicitly, and do not carry a threshold from one system to
another: thresholds are not transferable.

## Reference and orbitals

| Situation | Rule | Why |
|---|---|---|
| Closed-shell molecule | Start from canonical RHF orbitals. | The DMRG window is an index window around the Fermi level, so it needs energy-ordered orbitals. |
| Open-shell molecule | Use UHF. ROHF is untested in AEGISS; use UHF for open shells. | AEGISS offers only UHF for open shells. |
| UHF reference | Check ⟨S²⟩ before the DMRG: the deviation from S(S+1) must be at most 10% of S(S+1). Beyond that, flag it to the user instead of continuing silently. | Spin contamination of the UHF orbitals carries into the entropies. |
| Spin-summed or spin-resolved entropies | Follow [the guidance on UHF references](https://algorithmiq.github.io/AEGISS/latest/uhf/#which-to-use). Spin-resolved needs `symm_type="SZ"`. | The criteria are on that page. |
| Many low-lying virtuals of little importance (Rydberg-like or diffuse functions in a large basis) | Start from natural orbitals (NOs). | Canonical virtuals with little correlation sit just above the LUMO and fill the window; NOs carry almost no occupation there and move them down the list. |
| Canonical window would approach about 100 orbitals | Use NOs together with `reorder="fiedler"`, with a smaller window. | NOs concentrate the correlation in fewer orbitals, so a smaller window covers the same physics and the DMRG is cheaper. |
| Canonical RHF is known beforehand to be a poor reference | Use NOs from a correlated method (the paper uses MP2). | They adapt better to the correlation than canonical RHF orbitals. |
| Localized orbitals | Do not use them as input. | They have no orbital energies, so the index window is no longer a Fermi-level window, and there is no workaround yet. |
| Cheap steps in a smaller basis | Do it only if the basis is much smaller; the saving is otherwise small. Redo the final selection in the production basis. | The DMRG dominates the cost and its cost is set by the window size. The setup (labels, thresholds) carries over, but orbital indices change with the basis, so the final selection does not. |

## DMRG window

| Situation | Rule | Why |
|---|---|---|
| Starting `ncas` | Start with 3m, where m is the expected number of active orbitals. | General guideline of the authors. |
| Organic π systems | Expect to need 3–5m. | Range found for the systems of the paper. |
| Transition-metal complexes | Expect to need 6–8m. | The d manifold is spread in energy and mixes with ligand orbitals. |
| Bond breaking | No tested factor; start with 3m and treat the result as exploratory. | The authors have no example. |
| Margin to the edge | Keep at least 3 orbitals between the outermost orbital above τ_E and each edge of the window. | A crowded edge means correlated orbitals may lie outside the window. |
| Margin smaller than 3 | Widen the window by about 5 orbitals on the crowded side and rerun the DMRG. | The entropy ranking is only meaningful for orbitals well inside the window. |
| Closed-shell RHF | `symm_type="SU2"`. | Cheaper at equal accuracy. |
| UHF, spin-resolved entropies, or any open-shell case with per-spin data | `symm_type="SZ"` (the library default). | UHF and spin-resolved entropies require it. |
| Large window, or poor convergence at a given bond dimension | Use `reorder="fiedler"`. | Authors' guideline: reordering shortens the interaction range along the DMRG chain, which helps large windows converge. |

## Bond dimension

| Run | `max_bond_dim` | Why |
|---|---|---|
| Cheap first look | 200. Trust the set of orbitals above τ_E only if a rerun at 500 gives the same set; If 500 is not converged, check at `800`/`1000` (based on the DMRG expected cost); With a `ncas >= 50` the suggestion is to run the exploratory run at `500` and check with `800`/`1000` | The ranking can be unreliable at low bond dimension; an unchanged set at 500 shows it has converged.
| Production | 500, a multiple of 4. The library ramps from `max_bond_dim`/2; the paper ramps from 100 to 500 (Trans-Cl: fixed 500). | Value used by every run of the paper. |
| Checking convergence | Rerun at a higher `max_bond_dim` and stop when the set above τ_E is unchanged. | The library runs the whole ramp in one DMRG call and computes the entropies only from the final wavefunction, so there are no entropies partway through the ramp. |

## Entropy screening

| Situation | Rule | Why |
|---|---|---|
| Starting point | `fraction=0.1`, `fraction_type="r"` (τ_E = S(1)max/10). The paper uses 0.2 for benzene. | The AutoCAS criterion, used by the paper. |
| Flat entropies (every orbital above the 0.1 threshold) | Raise to `fraction=0.9`, relative. | In the paper's DOBNA case with MP2 natural orbitals no orbital fell below 0.1; 0.9 isolated the strongly entropic subset. |
| Comparing entropies across runs | `fraction_type="a"`. | An absolute threshold is the same number in every run; a relative one moves with S(1)max. |
| S(1)max itself is small | `fraction_type="a"`. | A relative cut then keeps orbitals that are only weakly correlated. |
| Occupied and virtual entropies differ strongly in scale | `ranking="subset"`. | A global cut would empty one side. |
| Any run | Pass `fraction`, `fraction_type` and `ranking` explicitly. | The defaults differ between the config classes and the selection functions, and are being aligned (open PRs [#112](https://github.com/Algorithmiq/AEGISS/pull/112) and [#114](https://github.com/Algorithmiq/AEGISS/pull/114)). |

## Atom labelling

| Situation | Rule | Why |
|---|---|---|
| Localized process (d–d, π→π*, bond dissociation) | Use 2–3 clusters and put labels only on the active cluster. In ferrocene only `"Fe 3d"` is labelled; the Cp orbitals enter through the entropy step. Select the cluster's atoms in [MolTagger](https://algorithmiq.github.io/AEGISS/latest/moltagger/) to get atom-specific `ao_labels`. | The entropy step brings in the correlated partners; the labels only say which process matters. |
| Delocalized or charge-transfer process | Put donor and acceptor in separate clusters (Trans-Cl: Ru and the ligand). | Each channel then gets its own threshold or `top_n`. |
| Extended ligand | One cluster per ring or fragment (TLD-1433). | Each ring or fragment then gets its own threshold or `top_n` (TLD-1433 uses 4 clusters). |
| Cannot anticipate where the relevant orbitals are localised | Skip clustering: the labels apply to every atom of that element (the AVAS-like limit). | Without clusters the labels are not tied to a region of the molecule. |
| Refining the labels | Go from coarse to fine and refine only when the selection is unbalanced. Relabelling does not need a new DMRG run, only the AO projection step. | The DMRG is the expensive part and does not depend on the labels. |
| Choosing the molecular plane for `2pz` labels | Orient the molecule so the π orbitals are along z, or use the matching label (`2px` if the molecule lies in the yz plane, as DOBNA does). Use [MolTagger](https://algorithmiq.github.io/AEGISS/latest/moltagger/) to check. [MolTagger](https://algorithmiq.github.io/AEGISS/latest/moltagger/) can be used to identify the plane in which the selected atoms are placed. | A label on the wrong p orbital selects nothing useful. |
| Bad clustering | Expect a less compact space, not a lost orbital. | Clustering never removes correlated orbitals from the entropy-screened pool. |

## AO projection

| Situation | Rule | Why |
|---|---|---|
| Clean, well-separated target manifold (benzene π, ferrocene Fe 3d, DOBNA π) | `contribution=0.5` with `contribution_type="a"`. | The weights separate cleanly, so a threshold picks the manifold. |
| Strongly mixed orbitals (metal–ligand hybridization, Rydberg virtuals) | `top_n` instead of `contribution`. Trans-Cl: `top_n=1` per Ru 4d component and (4,4) occupied/virtual on the bipyridine π (space II: 2 per component and (6,6)); TLD-1433 uses `top_n` per cluster. | A threshold takes either everything or nothing there. |
| Balance between occupied and virtual orbitals | `top_n` with `ranking="subset"` (`ProjectionConfig` only). | This gives the (n_occ, n_virt) count. |
| Reading the weights | Do not treat a `contribution` value as a probability. | The weight sums the absolute AO–MO overlaps over the label's AOs (an L1 norm of non-orthogonal overlaps), so it is not bounded by 1 (open issue, [PR #10](https://github.com/Algorithmiq/AEGISS/pull/10)). |
| Distinct correlation channels (metal and ligand) | One group per channel, each with its own threshold or `top_n`. | Each channel then gets its own selection. |
| One manifold described by several labels (`["C 2pz", "N 2pz"]` for one π system) | A flat list or a single group. | One threshold fits one manifold. |
| 4d/5d metal without a pseudopotential | Do not run with `basis="minao"`. | Without a pseudopotential `minao` mis-numbers the 4d/5d labels and the selection comes out empty; open [PR #113](https://github.com/Algorithmiq/AEGISS/pull/113) makes it raise. |
| Label outside the minimal valence shell (a second d shell, Rydberg or diffuse functions) | `ProjectionConfig(basis=None)` (`projected_basis=None` in `select_by_ao_projection`). | The minimal basis has no such functions. |
| Any run | Pass `contribution`, `contribution_type`, `top_n` and, in `ProjectionConfig`, `ranking` explicitly. | The defaults differ between the config classes and the selection functions (open PRs [#112](https://github.com/Algorithmiq/AEGISS/pull/112) and [#114](https://github.com/Algorithmiq/AEGISS/pull/114)). |

## Typical AO labels

| System | `ao_labels` | Notes |
|---|---|---|
| Conjugated π | `"C 2pz"`, plus the heteroatoms (`"N 2pz"`, `"B 2pz"`, `"O 2pz"`) | `2pz` assumes the molecule lies in the xy plane; the paper writes `C/B 2px` for DOBNA, which lies in the yz plane. |
| 3d metal | `"Fe 3d"` | |
| 4d metal | `"Ru 4d"`, or the five d components separately with `top_n=1` each | Needs a pseudopotential, see above. |
| Ligand of a metal | Its own group, e.g. `["C 2pz", "N 2pz"]` on the ligand atoms | The paper writes this as `C/N 2pz`. |
| Bond breaking (σ/σ*) | No tested recipe | Not analysed by the authors; the selection is exploratory. |
| Lone pairs | No tested recipe | Not analysed by the authors; the selection is exploratory. |
| Main-group radical | No tested recipe | Not analysed by the authors; the selection is exploratory. |
| 4s, or a second d shell (3d′) on a metal | No tested recipe | Not analysed by the authors; the selection is exploratory. |

## Stability check

Rerun the `fraction_type=r` selection with `fraction` 0.08, 0.10 and 0.12. The final active space should be identical;
the entropy-screened pool may shift by an orbital or two, because the AO projection absorbs that. If
the final space changes, an orbital sits right at the threshold: inspect it (cube file, AO composition)
and look for a plateau in the threshold plot, AutoCAS style. The paper does not test this check.

## Hand-off limit

You may run DMRG yourself (as an agent) unless both `ncas` ≥ 30 and `max_bond_dim` ≥ 500; then hand
the user a script. A cheap, low-bond-dimension run is fine on any window. For reference, block2
defaults on one node:

| System | CAS | `max_bond_dim` | Threads | DMRG time |
|---|---|---|---|---|
| Ferrocene | (40,40) | 300 | 32 (LEONARDO) | 654 s (about 11 min) |
| DOBNA | (40,40) | 300 | 32 (LEONARDO) | 291 s (about 5 min) |

## Final checks

Check the final active space against this list; each line gives the check and what is a red flag.

- Electron count: an odd count for a closed shell, or a count that does not match the chemistry.
- Occupied/virtual pairs: an occupied orbital whose correlating partner is missing, or an almost all-virtual space.
- Degenerate partners: one orbital of a degenerate pair selected without the other (`top_n` can split a pair; use it carefully).
- Frontier orbitals: HOMO or LUMO missing.
- Rydberg-like or diffuse virtuals: a virtual with a diffuse shape, or a low total weight on the minimal-basis AOs. Remove it by hand (the paper does so for TLD-1433).
- Window edge: high entropy near the edge of the window; widen the window.
- Occupations after CASSCF: natural occupation numbers above 1.98 or below 0.02 (generic thresholds).
- UHF: an unequal number of alpha and beta orbitals.
- Symmetry: optionally compare irreducible representations (the paper uses them as a post-selection criterion).

## Common mistakes

- Choosing the wrong plane for the `2pz` labels. [MolTagger](https://algorithmiq.github.io/AEGISS/latest/moltagger/) helps.
- Labelling a 4d/5d metal without a pseudopotential.
- Using too small a window.
- Carrying thresholds from one system to another: they are not transferable.
- Index base confusion: inputs follow `AEGISS_ORB_BASE`, but the returned indices are always 0-based; see [Orbital index base](https://algorithmiq.github.io/AEGISS/latest/orbital_indexing/).
- Accepting the output without looking at the orbitals.
