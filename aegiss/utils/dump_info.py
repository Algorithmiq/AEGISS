from pyscf.tools import cubegen
import numpy as np
import os

def save_orbitals(mol, 
                  mo_coeff:np.ndarray,
                  candidate_indices: np.ndarray,
                  filename_prefix: str = "mo",
                  folder: str  = "./") -> None:
    """Save molecular orbitals as cube files for visualization.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object containing the geometry and basis set information.
    mo_coeff : np.ndarray
        Molecular orbital coefficients in the AO basis. Shape should be (n_ao, n_mo).
    filename_prefix : str
        Prefix for the output cube files. Each orbital will be saved as "{filename_prefix}_orbital_{i}.cube".
    """
    
    # Ensure the output folder exists
    os.makedirs(folder, exist_ok=True)
    
    for i in candidate_indices:
        filename = f"{folder}{filename_prefix}_{i}.cube"
        cubegen.orbital(mol, filename, mo_coeff[:, i])
        
def save_orbitals_from_entropies(mol, 
                  mo_coeff:np.ndarray,
                  candidate_indices: list[tuple[int, float]],
                  filename_prefix: str = "mo",
                  folder: str  = "./") -> None:
    """Save molecular orbitals as cube files for visualization.
    
    Parameters
    ----------
    mol : pyscf.gto.Mole
        Molecule object containing the geometry and basis set information.
    mo_coeff : np.ndarray
        Molecular orbital coefficients in the AO basis. Shape should be (n_ao, n_mo).
    filename_prefix : str
        Prefix for the output cube files. Each orbital will be saved as "{filename_prefix}_orbital_{i}.cube".
    """
    
    os.makedirs(folder, exist_ok=True)
    
    for i, _  in candidate_indices:
        filename = f"{folder}{filename_prefix}_{i}.cube"
        cubegen.orbital(mol, filename, mo_coeff[:, i])