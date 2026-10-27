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

AEGISS ranks orbitals by their single-orbital entropy from a DMRG calculation, run with
block2, and combines that with an AVAS-like projection onto atomic orbitals. It is built
on PySCF, and MolTagger uses ASE. Please also cite the papers behind them.

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

**Single-orbital entropy**, the entanglement measure AEGISS screens on

- Ö. Legeza and J. Sólyom, *Optimizing the density-matrix renormalization group method
  using quantum information entropy*, Phys. Rev. B **68**, 195116 (2003).
  [doi:10.1103/PhysRevB.68.195116](https://doi.org/10.1103/PhysRevB.68.195116)
- J. Rissler, R. M. Noack and S. R. White, *Measuring orbital interaction using quantum
  information theory*, Chem. Phys. **323**, 519 (2006).
  [doi:10.1016/j.chemphys.2005.10.018](https://doi.org/10.1016/j.chemphys.2005.10.018)
- K. Boguslawski, P. Tecmer, Ö. Legeza and M. Reiher, *Entanglement measures for
  single- and multireference correlation effects*, J. Phys. Chem. Lett. **3**, 3129
  (2012). [doi:10.1021/jz301319v](https://doi.org/10.1021/jz301319v)
- K. Boguslawski and P. Tecmer, *Orbital entanglement in quantum chemistry*,
  Int. J. Quantum Chem. **115**, 1289 (2015).
  [doi:10.1002/qua.24832](https://doi.org/10.1002/qua.24832)

**PySCF**, for the mean field, the integrals, the localization and the AO projection

- Q. Sun *et al.*, *PySCF: the Python-based simulations of chemistry framework*,
  WIREs Comput. Mol. Sci. **8**, e1340 (2018).
  [doi:10.1002/wcms.1340](https://doi.org/10.1002/wcms.1340)
- Q. Sun *et al.*, *Recent developments in the PySCF program package*,
  J. Chem. Phys. **153**, 024109 (2020).
  [doi:10.1063/5.0006074](https://doi.org/10.1063/5.0006074)

**ASE**, used by [MolTagger](moltagger.md)

- A. H. Larsen *et al.*, *The atomic simulation environment—a Python library for working
  with atoms*, J. Phys.: Condens. Matter **29**, 273002 (2017).
  [doi:10.1088/1361-648X/aa680e](https://doi.org/10.1088/1361-648X/aa680e)

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

@article{legeza2003entropy,
  title={Optimizing the density-matrix renormalization group method using quantum information entropy},
  author={Legeza, Örs and Sólyom, J.},
  journal={Phys. Rev. B},
  volume={68},
  pages={195116},
  year={2003},
  doi={10.1103/PhysRevB.68.195116},
}

@article{rissler2006orbital,
  title={Measuring orbital interaction using quantum information theory},
  author={Rissler, Jörg and Noack, Reinhard M. and White, Steven R.},
  journal={Chem. Phys.},
  volume={323},
  pages={519--531},
  year={2006},
  doi={10.1016/j.chemphys.2005.10.018},
}

@article{boguslawski2012entanglement,
  title={Entanglement Measures for Single- and Multireference Correlation Effects},
  author={Boguslawski, Katharina and Tecmer, Paweł and Legeza, Örs and Reiher, Markus},
  journal={J. Phys. Chem. Lett.},
  volume={3},
  pages={3129--3135},
  year={2012},
  doi={10.1021/jz301319v},
}

@article{boguslawski2015orbital,
  title={Orbital entanglement in quantum chemistry},
  author={Boguslawski, Katharina and Tecmer, Paweł},
  journal={Int. J. Quantum Chem.},
  volume={115},
  pages={1289--1295},
  year={2015},
  doi={10.1002/qua.24832},
}

@article{sun2018pyscf,
  title={PySCF: the Python-based simulations of chemistry framework},
  author={Sun, Qiming and Berkelbach, Timothy C. and Blunt, Nick S. and Booth, George H. and Guo, Sheng and Li, Zhendong and Liu, Junzi and McClain, James D. and Sayfutyarova, Elvira R. and Sharma, Sandeep and Wouters, Sebastian and Chan, Garnet Kin-Lic},
  journal={WIREs Comput. Mol. Sci.},
  volume={8},
  pages={e1340},
  year={2018},
  doi={10.1002/wcms.1340},
}

@article{sun2020pyscf,
  title={Recent developments in the PySCF program package},
  author={Sun, Qiming and Zhang, Xing and Banerjee, Samragni and Bao, Peng and Barbry, Marc and Blunt, Nick S. and Bogdanov, Nikolay A. and Booth, George H. and Chen, Jia and Cui, Zhi-Hao and Eriksen, Janus J. and Gao, Yang and Guo, Sheng and Hermann, Jan and Hermes, Matthew R. and Koh, Kevin and Koval, Peter and Lehtola, Susi and Li, Zhendong and Liu, Junzi and Mardirossian, Narbe and McClain, James D. and Motta, Mario and Mussard, Bastien and Pham, Hung Q. and Pulkin, Artem and Purwanto, Wirawan and Robinson, Paul J. and Ronca, Enrico and Sayfutyarova, Elvira R. and Scheurer, Maximilian and Schurkus, Henry F. and Smith, James E. T. and Sun, Chong and Sun, Shi-Ning and Upadhyay, Shiv and Wagner, Lucas K. and Wang, Xiao and White, Alec and Whitfield, James Daniel and Williamson, Mark J. and Wouters, Sebastian and Yang, Jun and Yu, Jason M. and Zhu, Tianyu and Berkelbach, Timothy C. and Sharma, Sandeep and Sokolov, Alexander Yu. and Chan, Garnet Kin-Lic},
  journal={J. Chem. Phys.},
  volume={153},
  pages={024109},
  year={2020},
  doi={10.1063/5.0006074},
}

@article{larsen2017ase,
  title={The atomic simulation environment---a Python library for working with atoms},
  author={Hjorth Larsen, Ask and Jørgen Mortensen, Jens and Blomqvist, Jakob and Castelli, Ivano E and Christensen, Rune and Dułak, Marcin and Friis, Jesper and Groves, Michael N and Hammer, Bjørk and Hargus, Cory and Hermes, Eric D and Jennings, Paul C and Bjerre Jensen, Peter and Kermode, James and Kitchin, John R and Leonhard Kolsbjerg, Esben and Kubal, Joseph and Kaasbjerg, Kristen and Lysgaard, Steen and Bergmann Maronsson, Jón and Maxson, Tristan and Olsen, Thomas and Pastewka, Lars and Peterson, Andrew and Rostgaard, Carsten and Schiøtz, Jakob and Schütt, Ole and Strange, Mikkel and Thygesen, Kristian S and Vegge, Tejs and Vilhelmsen, Lasse and Walter, Michael and Zeng, Zhenhua and Jacobsen, Karsten W},
  journal={J. Phys.: Condens. Matter},
  volume={29},
  pages={273002},
  year={2017},
  doi={10.1088/1361-648X/aa680e},
}
```

Machine-readable metadata for AEGISS and these papers is in
[CITATION.cff](https://github.com/Algorithmiq/AEGISS/blob/main/CITATION.cff).
