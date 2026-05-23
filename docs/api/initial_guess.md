# Initial Guess

Functions for constructing the molecular system and running the mean-field calculation.
Wraps PySCF's [SCF](https://pyscf.org/user/scf.html) and [DFT](https://pyscf.org/user/dft.html)
modules with optional [solvation](https://pyscf.org/user/solvent.html) and
[scalar relativistic (X2C)](https://pyscf.org/user/scf.html#scalar-relativistic-correction)
corrections.

```python
from aegiss.initial_guess import build_mol, run_scf
```

::: aegiss.initial_guess.scf.build_mol

---

::: aegiss.initial_guess.scf.run_scf
