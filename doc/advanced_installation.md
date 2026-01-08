# Advanced Installation Guide

To achieve maximum performance with `vbcsr`, it is highly recommended to link against an optimized BLAS library (like Intel MKL or OpenBLAS) and ensure OpenMP is correctly configured.

## Specifying BLAS Vendor

You can control which BLAS library CMake searches for using the `BLA_VENDOR` environment variable or CMake option.

### Linking against Intel MKL (Recommended)

```bash
# Using environment variable (recommended for pip)
export CMAKE_ARGS="-DBLA_VENDOR=Intel10_64lp -DMKL_INTERFACE=lp64 -DMKL_THREADING=sequential"
pip install .
```

### Linking against OpenBLAS

```bash
export CMAKE_ARGS="-DBLA_VENDOR=OpenBLAS"
pip install .
```

## Helping CMake Find Libraries

If CMake cannot find your libraries (MKL, OpenMP), you can provide hints via `CMAKE_PREFIX_PATH`. This is particularly useful when using Conda environments.

```bash
# Example for Conda users
export CMAKE_PREFIX_PATH=$CONDA_PREFIX
pip install .
```

## OpenMP Support

This library requires OpenMP for parallel execution.

- **Linux**: Usually available by default (GCC/GOMP).
- **macOS**: You may need to install `libomp`:
  ```bash
  brew install libomp
  export CMAKE_ARGS="-DOpenMP_ROOT=$(brew --prefix libomp)"
  pip install .
  ```
