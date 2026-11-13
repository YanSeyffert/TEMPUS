# scripts/

Data-download and data-preparation utilities. These are **not** usage examples — see `examples/` for those. Both download scripts write into this repository's `data/` folder wherever they are run from, and skip files that already exist, so they are safe to re-run. If they report certificate errors on an older Linux system, see [../docs/INSTALL_OLDER_LINUX.md](../docs/INSTALL_OLDER_LINUX.md).

| Script | Purpose |
|---|---|
| `download_data.sh` | Downloads the public ephemeris/kernel files used by `examples/*.txt` into `data/ephem/` and `data/kernels/` (all groups, or only those named, e.g. `./download_data.sh inpop de440`). Group `inpop_asc` adds INPOP19a's own TCG−TCB Chebyshev file to `data/time_ephem/` for the validation notebooks. |
| `fetch_klioner_asc.py` | Downloads Klioner's TCX−TCB time ephemerides (TU Dresden) into `data/time_ephem/`: the eleven forward files plus the inverse files the notebooks use. Mars is saved as `TCM` (Klioner's site calls it `TCMar`). |
| `header_supplement_file_generation.py` | Generated `data/ephem/header_mba.19a` from INPOP-style `GM_...` constant lines. |
| `header_supplement_from_twocol.py` | Generates DE-header-style GM supplement files (e.g. `header_kbo.440`) from two-column `index  GM` tables: `python header_supplement_from_twocol.py <table> [output name]`. |

The two header scripts are kept for provenance: their input tables are **not** included, and the headers they produced already live in `data/ephem/` and `data/kernels/`. They only need to be rerun when a new ephemeris release ships with GM values not present in its binary header. The headers are consumed at run time via the `--asteroid-gm-header` CLI option.
See `notebooks/00_tempus_background.ipynb` §2.3 for the file-format details and which header files are official downloads vs. locally generated.
