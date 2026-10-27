# Citing AEGISS

If you use AEGISS in your work, please cite the AEGISS paper:

> F. Tarocco, P. A. B. Haase, F. Pavošević, V. Krishna, L. Guidoni, S. Knecht and
> M. Stella, *AEGISS -- Atomic orbital and Entropy-based Guided Inference for Space
> Selection -- A novel semi-automated active space selection workflow for quantum
> chemistry and quantum computing applications*, arXiv:2508.10671 (2025).
> [doi:10.48550/arXiv.2508.10671](https://doi.org/10.48550/arXiv.2508.10671)

```bibtex
@misc{tarocco2025aegissatomicorbital,
      title={AEGISS -- Atomic orbital and Entropy-based Guided Inference for Space Selection -- A novel semi-automated active space selection workflow for quantum chemistry and quantum computing applications},
      author={Fabio Tarocco and Pi A. B. Haase and Fabijan Pavošević and Vijay Krishna and Leonardo Guidoni and Stefan Knecht and Martina Stella},
      year={2025},
      eprint={2508.10671},
      archivePrefix={arXiv},
      primaryClass={physics.chem-ph},
      url={https://arxiv.org/abs/2508.10671},
}
```

## Methods AEGISS builds on

AEGISS combines a DMRG calculation, run with block2, with an AVAS-like projection onto
atomic orbitals. Please also cite the papers behind them.

**DMRG**

- S. R. White, *Density matrix formulation for quantum renormalization groups*,
  Phys. Rev. Lett. **69**, 2863 (1992).
  [doi:10.1103/PhysRevLett.69.2863](https://doi.org/10.1103/PhysRevLett.69.2863)
- S. R. White, *Density-matrix algorithms for quantum renormalization groups*,
  Phys. Rev. B **48**, 10345 (1993).
  [doi:10.1103/PhysRevB.48.10345](https://doi.org/10.1103/PhysRevB.48.10345)
- G. K.-L. Chan and M. Head-Gordon, *Highly correlated calculations with a polynomial
  cost algorithm: A study of the density matrix renormalization group*,
  J. Chem. Phys. **116**, 4462 (2002).
  [doi:10.1063/1.1449459](https://doi.org/10.1063/1.1449459)

**block2**, the DMRG program AEGISS runs (`block2main`)

- H. Zhai, H. R. Larsson, S. Lee, Z.-H. Cui, T. Zhu, C. Sun, L. Peng, R. Peng, K. Liao,
  J. Tölle, J. Yang, S. Li and G. K.-L. Chan, *Block2: A comprehensive open source
  framework to develop and apply state-of-the-art DMRG algorithms in electronic
  structure and beyond*, J. Chem. Phys. **159**, 234801 (2023).
  [doi:10.1063/5.0180424](https://doi.org/10.1063/5.0180424)
- H. Zhai and G. K.-L. Chan, *Low communication high performance ab initio density
  matrix renormalization group algorithms*, J. Chem. Phys. **154**, 224116 (2021).
  [doi:10.1063/5.0050902](https://doi.org/10.1063/5.0050902)

**AVAS**

- E. R. Sayfutyarova, Q. Sun, G. K.-L. Chan and G. Knizia, *Automated construction of
  molecular active spaces from atomic valence orbitals*, J. Chem. Theory Comput. **13**,
  4063 (2017). [doi:10.1021/acs.jctc.7b00128](https://doi.org/10.1021/acs.jctc.7b00128)

```bibtex
@article{white1992dmrg,
  title={Density matrix formulation for quantum renormalization groups},
  author={White, Steven R.},
  journal={Phys. Rev. Lett.},
  volume={69},
  pages={2863--2866},
  year={1992},
  doi={10.1103/PhysRevLett.69.2863},
}

@article{white1993dmrg,
  title={Density-matrix algorithms for quantum renormalization groups},
  author={White, Steven R.},
  journal={Phys. Rev. B},
  volume={48},
  pages={10345--10356},
  year={1993},
  doi={10.1103/PhysRevB.48.10345},
}

@article{chan2002dmrg,
  title={Highly correlated calculations with a polynomial cost algorithm: A study of the density matrix renormalization group},
  author={Chan, Garnet Kin-Lic and Head-Gordon, Martin},
  journal={J. Chem. Phys.},
  volume={116},
  pages={4462--4476},
  year={2002},
  doi={10.1063/1.1449459},
}

@article{zhai2023block2,
  title={Block2: A comprehensive open source framework to develop and apply state-of-the-art DMRG algorithms in electronic structure and beyond},
  author={Zhai, Huanchen and Larsson, Henrik R. and Lee, Seunghoon and Cui, Zhi-Hao and Zhu, Tianyu and Sun, Chong and Peng, Linqing and Peng, Ruojing and Liao, Ke and T{\"o}lle, Johannes and Yang, Junjie and Li, Shuoxue and Chan, Garnet Kin-Lic},
  journal={J. Chem. Phys.},
  volume={159},
  pages={234801},
  year={2023},
  doi={10.1063/5.0180424},
}

@article{zhai2021block2,
  title={Low communication high performance ab initio density matrix renormalization group algorithms},
  author={Zhai, Huanchen and Chan, Garnet Kin-Lic},
  journal={J. Chem. Phys.},
  volume={154},
  pages={224116},
  year={2021},
  doi={10.1063/5.0050902},
}

@article{sayfutyarova2017avas,
  title={Automated construction of molecular active spaces from atomic valence orbitals},
  author={Sayfutyarova, Elvira R. and Sun, Qiming and Chan, Garnet Kin-Lic and Knizia, Gerald},
  journal={J. Chem. Theory Comput.},
  volume={13},
  pages={4063--4078},
  year={2017},
  doi={10.1021/acs.jctc.7b00128},
}
```

Machine-readable metadata for AEGISS and these papers is in
[CITATION.cff](https://github.com/Algorithmiq/AEGISS/blob/main/CITATION.cff).
