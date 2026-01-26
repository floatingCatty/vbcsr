#!/bin/bash

# Ensure MPI is found
export MPI_HOME=$PREFIX
export OMPI_CC=$CC
export OMPI_CXX=$CXX

# Build and install
$PYTHON -m pip install . --no-deps -vv
