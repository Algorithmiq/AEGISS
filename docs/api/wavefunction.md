# Wavefunction

Correlated wavefunction methods. `run_dmrg` is the primary method used in the AEGISS
pipeline; the others (`run_mp2`, `run_cisd`, `run_ccsd`) are thin wrappers around
PySCF that produce correlated density matrices for natural orbital localization.

```python
from aegiss.wavefunction import run_dmrg, run_mp2, run_cisd, run_ccsd
```

::: aegiss.wavefunction.dmrg.run_dmrg

---

::: aegiss.wavefunction.methods.run_mp2

---

::: aegiss.wavefunction.methods.run_cisd

---

::: aegiss.wavefunction.methods.run_ccsd
