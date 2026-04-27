## Python dependencies

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

## Build the C++ fitter

`TrackFit.py` imports the `trackfit_cpp` extension from `cppfit/`.
The normal developer install path is:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e ./cppfit
```

That editable install compiles `cppfit/trackfit_cpp.cpp` into a local
platform-specific extension module, so each developer should build it on
their own machine after pulling new C++ changes.

If you are working in an offline or restricted environment, reuse the
already-installed build dependencies from the venv:

```bash
pip install --no-build-isolation -e ./cppfit
```

Fallback if editable install is not available:

```bash
cd cppfit
python setup.py build_ext --inplace
```

## System dependencies (RHEL 9)

```bash
sudo dnf install -y \
  python3.13-devel \
  libpcap-devel \
  gcc-c++ make
```
