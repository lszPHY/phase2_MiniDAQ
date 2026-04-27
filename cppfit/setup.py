from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

# Optional OpenMP: enable if your compiler supports it.
# If you don't want OpenMP yet, set USE_OPENMP = False.
USE_OPENMP = False

# Do not use -ffast-math here. The fitter intentionally relies on
# std::isfinite/NaN/inf checks to reject invalid candidate fits.
compile_args = ["-O3", "-DNDEBUG", "-fno-math-errno", "-funroll-loops"]
link_args = []

# This gives best speed on *this machine*.
# If you plan to run on different CPUs, you may want to remove -march=native.
compile_args += ["-march=native"]

if USE_OPENMP:
    compile_args += ["-fopenmp"]
    link_args += ["-fopenmp"]

ext_modules = [
    Pybind11Extension(
        "trackfit_cpp",                 # import trackfit_cpp
        ["trackfit_cpp.cpp"],
        cxx_std=17,
        extra_compile_args=compile_args,
        extra_link_args=link_args,
    ),
]

setup(
    name="trackfit_cpp",
    version="0.1.0",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
)
