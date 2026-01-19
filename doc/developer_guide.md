# VBCSR Developer Guide

This guide documents the internal design decisions and conventions of the VBCSR library. It is intended for developers maintaining or extending the core functionality.

## Distributed Graph (`DistGraph`)

The `DistGraph` class manages the distributed adjacency structure of the block sparse matrix. A key design decision involves the ordering of ghost indices.

### Ghost Index Convention: "Sort by Owner"

In a distributed graph, local indices are assigned as follows:
1.  **Owned Indices**: $0$ to $N_{owned}-1$. These correspond to global indices owned by the local rank, sorted by global ID.
2.  **Ghost Indices**: $N_{owned}$ to $N_{total}-1$. These correspond to global indices owned by remote ranks.

**CRITICAL**: Ghost indices are sorted by **Owner Rank** first, and then by Global ID.

#### Rationale
This ordering is chosen to optimize the `DistVector::sync_ghosts` operation, which is the communication backbone of Sparse Matrix-Vector Multiplication (SpMV).
-   `MPI_Alltoallv` delivers data in rank order (data from Rank 0, then Rank 1, etc.).
-   By sorting ghost indices by owner rank, the incoming data buffer from MPI exactly matches the memory layout of the ghost elements in `DistVector`.
-   This enables a **zero-copy receive**, avoiding a costly unpacking step in the inner loop of iterative solvers.

#### Implications for Algorithms
Because ghosts are sorted by owner, **local indices do not necessarily correspond to monotonically increasing global indices**.
-   Example: Rank 0 has ghosts from Rank 2 (GID 50) and Rank 1 (GID 100).
-   Ghost order: Rank 1's ghosts (GID 100), then Rank 2's ghosts (GID 50).
-   Local Indices: $L_{owned}, \dots, L_{ghost1} \rightarrow 100, L_{ghost2} \rightarrow 50$.
-   Here, $L_{ghost1} < L_{ghost2}$ but $Global(L_{ghost1}) > Global(L_{ghost2})$.

**Developers must NOT assume that iterating over local column indices yields sorted global column indices.**
-   Operations like `axpby` (matrix addition) or `spmm` (matrix multiplication) that merge structures must handle unsorted global indices explicitly (e.g., using a "collect, sort, unique" approach instead of a linear merge).

## Block CSR (`BlockSpMat`)

`BlockSpMat` builds upon `DistGraph` to store matrix values.

-   **Storage**: Values are stored in `BlockArena` (a memory pool) to ensure cache locality and reduce allocation overhead.
-   **Handles**: `blk_handles` store offsets/pointers into the arena.
-   **Thread Safety**: `BlockSpMat` is designed for OpenMP threading. Temporary buffers in operations like `axpby` or `spmm` should be thread-local to avoid contention and allocation overhead.
