# Copyright(c)2026, CNRS, UNIVERSITE COTE D'AZUR, OBSERVATOIRE COTE D'AZUR (OCA), Institut de Recherche pour le Développement (IRD)
# Authors :  Yan SEYFFERT and Agnès FIENGA
# License BSD-3-Clause

# src/tempus/__init__.py
# this file defines the tempus package

from importlib.metadata import PackageNotFoundError, version as _pkg_version

try:
    __version__ = _pkg_version("tempus")
except PackageNotFoundError:      # running from a source tree without an install
    __version__ = "0.0.0+unknown"

from .ephem import CalcephEphem
from .model import TimeScalar, TimeGrid, jd_grid, jd_split, jd_parts_to_et, Delta_TCX, jd_tt_to_tdb, jd_tt_to_tcb
from .integrate import get_integrator

__all__ = [
    "__version__",
    "CalcephEphem",
    "TimeScalar",
    "TimeGrid",
    "jd_grid",
    "jd_split",
    "jd_parts_to_et",
    "Delta_TCX",
    "jd_tt_to_tdb",
    "jd_tt_to_tcb",
    "get_integrator",
]
