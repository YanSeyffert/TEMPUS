# Installing TEMPUS on older Linux systems

On older servers (for example CentOS 7, with system Python 3.6, gcc 4.8, wget 1.14 and an outdated certificate store), the installation in the README fails at a few points. The sequence below worked on such a machine; run it from the project root.

## Installation

```bash
python3 --version                      # older than 3.8? look for a newer one:
ls /usr/bin/python3*                   # e.g. python3.10
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
source /opt/rh/devtoolset-11/enable    # gcc 7 or newer (see: ls /opt/rh/, module avail gcc)
pip install cmake                      # CMake 3.7 or newer
export PATH="$(python -c 'import cmake; print(cmake.CMAKE_BIN_DIR)'):$PATH"
pip install --prefer-binary -e .
tempus --version
```

- `--prefer-binary` makes pip take the newest versions that still have prebuilt packages for older systems, instead of building newer ones from source.
- The devtoolset and `PATH` settings are only needed for this installation; afterwards `source .venv/bin/activate` is enough.

Errors this sequence avoids:

| Error during `pip install -e .` | Cause |
|---|---|
| `No matching distribution found for setuptools>=68` | Python older than 3.8 |
| `Failed building wheel for pillow` | the newest Pillow has no prebuilt package for this system (`--prefer-binary`) |
| `gcc: error: unrecognized command line option '-fcode-hoisting'` | system gcc older than the one Python was built with |
| `cmake ... returned non-zero exit status 1` while building calcephpy | system CMake older than 3.7 |
| `No module named 'cmake'` while building calcephpy | the pip-installed CMake is not first on `PATH` |

## Downloading the data

If `scripts/download_data.sh` or `scripts/fetch_klioner_asc.py` report certificate errors (`cannot verify ... certificate`, `CERTIFICATE_VERIFY_FAILED`), the system's certificate store is too old for some of the servers. Use the up-to-date bundle from the `certifi` package for the downloads:

```bash
pip install certifi
echo "ca_certificate = $(python -m certifi)" > ~/tempus-wgetrc
WGETRC=~/tempus-wgetrc bash scripts/download_data.sh
SSL_CERT_FILE=$(python -m certifi) python scripts/fetch_klioner_asc.py
```

Both scripts skip files that are already present, so they can simply be run again after a failure.
