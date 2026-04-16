"""Utilities for dumping intermediate DMRG quantities to disk."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def dump_qmi(
    minfo: np.ndarray,
    soe1: np.ndarray,
    soe2: np.ndarray | None = None,
    path: Path | str = "./qmi_data.npz",
) -> None:
    """Dump QMI, SOE1, and optional SOE2 data to disk.

    Parameters
    ----------
    minfo : np.ndarray
        Mutual information matrix.
    soe1 : np.ndarray
        Single orbital entropy values.
    soe2 : np.ndarray or None
        Double orbital entropy values, or ``None`` if not computed.
    path : Path or str
        Output file path (without extension; ``.npz`` is appended by numpy).
    """
    filename = Path(path)
    data: dict[str, np.ndarray] = {"minfo": minfo, "soe1": soe1}
    if soe2 is not None:
        data["soe2"] = soe2
    np.savez(filename, **data)  # type: ignore[arg-type]
    print(f"QMI data dumped to {filename}")
