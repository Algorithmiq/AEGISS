
from pathlib import Path
import numpy as np
def dump_qmi(minfo: np.ndarray, 
             soe1: np.ndarray, 
             soe2: np.ndarray | None = None,
             path: Path | str = "./qmi_data.npz") -> None:
    """Dump quantum mutual information (QMI), Single and Double Orbital entropy (SOE1, SOE2) data to disk for debugging and analysis.
    
    Parameters
    ----------
    minfo : np.ndarray
        Mutual information matrix.
    soe1 : np.ndarray
        Single orbital entropy values.
    soe2 : np.ndarray | None
        Double orbital entropy values, or None if not computed.
    base_path : Path or str
        Base path for dumping the data. Will create separate files for minfo, soe1, and soe2.
    """
    filename = Path(path)
    np.savez(filename, minfo=minfo, soe1=soe1, soe2=soe2)
    print(f"QMI data dumped to {filename}")
    