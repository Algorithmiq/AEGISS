# AEGISS

**Atomic Orbital and Entropy based Guided Inference for Space Selection**

AEGISS automates active space selection for multireference quantum chemistry calculations.
It combines two complementary screening strategies:

- **Entropy screening** — ranks orbitals by their single-orbital entanglement entropy
  obtained from a DMRG wavefunction. Orbitals with high entanglement are strongly
  correlated and belong in the active space.
- **AO projection** — projects molecular orbitals onto a user-defined set of atomic
  orbitals (e.g. the C 2p_z shell of a conjugated system). Orbitals with large
  projection weights are chemically relevant to the target fragment.

The two criteria can be used independently or combined. In `select_aegiss` they are
applied sequentially — entropy screening first narrows the candidate set, then AO
projection refines it further — so an orbital must satisfy *both* criteria to appear
in the final active space.

## Quick start

```python
from aegiss.initial_guess import build_mol, run_scf
from aegiss.active_space import select_aegiss
from aegiss.config_dataclasses import DMRGConfig, EntropyConfig, ProjectionConfig, OutputConfig

mol = build_mol("molecule.xyz", basis="ccpvdz")
mf  = run_scf(mol, method="RHF")

active_indices = select_aegiss(
    mol, mf,
    dmrg_config=DMRGConfig(ncas=20, nelecas=20, symm_type="SZ"),
    entropy_config=EntropyConfig(fraction=0.1, fraction_type="r", ranking="global"),
    projection_config=ProjectionConfig(ao_labels=["C 2pz"], contribution=0.5),
    output_config=OutputConfig(saving_folder="./orbs/"),
)
```

## Where to go next

| | |
|---|---|
| [Installation](installation.md) | Dependencies, environment variables, logging |
| [Tutorials](tutorials/index.md) | Step-by-step notebooks from SCF to the full pipeline |
| [MolTagger](moltagger.md) | Interactive 3D viewer for atom tagging and AO label generation |
| [API Reference](api/index.md) | Complete function and class documentation |
