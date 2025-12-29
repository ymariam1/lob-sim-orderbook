"""
Setup script for building the lob_sim Python extension.

Usage:
    pip install .          # Install the package
    pip install -e .       # Install in development/editable mode
    python setup.py build  # Just build without installing
"""

import os
import sys
import subprocess
from pathlib import Path

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext


class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=""):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        
        # Required for auto-detection of auxiliary "native" libs
        if not extdir.endswith(os.path.sep):
            extdir += os.path.sep

        cfg = "Debug" if self.debug else "Release"
        
        # Get pybind11 cmake dir
        import pybind11
        pybind11_cmake_dir = pybind11.get_cmake_dir()
        
        # Get Python include and library directories
        import sysconfig
        python_include_dir = sysconfig.get_path('include')
        python_lib_dir = sysconfig.get_config_var('LIBDIR')
        
        # Try to find Python library
        python_lib = None
        if python_lib_dir:
            import glob
            python_version = f"{sys.version_info.major}.{sys.version_info.minor}"
            # Common library naming patterns
            lib_patterns = [
                f"{python_lib_dir}/libpython{python_version}*.so",
                f"{python_lib_dir}/libpython{python_version}*.a",
                f"{python_lib_dir}/libpython{sys.version_info.major}{sys.version_info.minor}*.so",
            ]
            for pattern in lib_patterns:
                matches = glob.glob(pattern)
                if matches:
                    python_lib = matches[0]
                    break

        cmake_args = [
            f"-DCMAKE_LIBRARY_OUTPUT_DIRECTORY={extdir}",
            f"-DPYTHON_EXECUTABLE={sys.executable}",
            f"-DCMAKE_BUILD_TYPE={cfg}",
            f"-Dpybind11_DIR={pybind11_cmake_dir}",
        ]
        
        # Explicitly set Python paths if we found them (helps CMake find Python)
        if python_include_dir and os.path.exists(python_include_dir):
            cmake_args.append(f"-DPython3_INCLUDE_DIR={python_include_dir}")
            if self.verbose:
                print(f"Setting Python3_INCLUDE_DIR={python_include_dir}")
        if python_lib_dir and os.path.exists(python_lib_dir):
            cmake_args.append(f"-DPython3_LIBRARY_DIR={python_lib_dir}")
        if python_lib and os.path.exists(python_lib):
            cmake_args.append(f"-DPython3_LIBRARY={python_lib}")

        build_args = ["--config", cfg]

        # Set CMAKE_BUILD_PARALLEL_LEVEL to control the parallel build level
        if "CMAKE_BUILD_PARALLEL_LEVEL" not in os.environ:
            # Use all available cores
            import multiprocessing
            build_args += ["-j", str(multiprocessing.cpu_count())]

        build_temp = Path(self.build_temp) / ext.name
        build_temp.mkdir(parents=True, exist_ok=True)

        # Print CMake args for debugging
        if self.verbose:
            print(f"CMake args: {' '.join(cmake_args)}")

        subprocess.run(
            ["cmake", ext.sourcedir] + cmake_args,
            cwd=build_temp,
            check=True,
        )
        subprocess.run(
            ["cmake", "--build", "."] + build_args,
            cwd=build_temp,
            check=True,
        )


setup(
    name="lob_sim",
    version="0.1.0",
    description="High-frequency Limit Order Book Simulator with Python bindings",
    ext_modules=[CMakeExtension("lob_sim", sourcedir="src/cpp")],
    cmdclass={"build_ext": CMakeBuild},
    zip_safe=False,
    python_requires=">=3.12",
)
