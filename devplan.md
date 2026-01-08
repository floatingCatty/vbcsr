# Dev Plan

1. The python binding. This datastructure need to support the python interface usage. The dist_graph, the block_csr and the block_vector need to support the python interface usage individually. I hope python side can use them as they are naturally python classes. The dist_vector and dist_multivector need to support the tranformation from and to numpy array format.

2. pip installing, pyproject.toml need to be added to support the c code compiling directing through pip install. It should allow the user to switch between the internal blas kernals.