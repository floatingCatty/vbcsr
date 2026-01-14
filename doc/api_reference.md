# VBCSR API Reference

This document provides detailed documentation for the classes and methods in the `vbcsr` package.

## Table of Contents
1. [VBCSR (Matrix)](#vbcsr-matrix)
2. [DistVector](#distvector)
3. [DistMultiVector](#distmultivector)

---

## VBCSR (Matrix)

The `VBCSR` class represents a distributed block-sparse matrix. It inherits from `scipy.sparse.linalg.LinearOperator`.

### Constructor
```python
VBCSR(graph: Any, dtype: type = np.float64)
```
Initializes the matrix from a `DistGraph`. The user should avoid touching this logic. Typically, you should use the factory methods `create_serial` or `create_distributed` instead.

### Factory Methods

#### `create_serial`
```python
@classmethod
create_serial(cls, comm: Any, global_blocks: int, block_sizes: List[int], adjacency: List[List[int]], dtype: type = np.float64) -> 'VBCSR'
```
Creates a matrix using serial graph construction. Rank 0 defines the entire structure, which is then distributed.
- **comm**: MPI communicator.
- **global_blocks**: Total number of blocks.
- **block_sizes**: List of block sizes (length `global_blocks`).
- **adjacency**: Adjacency list (list of lists of neighbors).

#### `create_distributed`
```python
@classmethod
create_distributed(cls, comm: Any, owned_indices: List[int], block_sizes: List[int], adjacency: List[List[int]], dtype: type = np.float64) -> 'VBCSR'
```
Creates a matrix using distributed graph construction. Each rank defines only its owned blocks.
- **owned_indices**: Global indices of blocks owned by this rank.
- **block_sizes**: Sizes of owned blocks.
- **adjacency**: Adjacency list for owned blocks.

#### `create_random`
```python
@classmethod
create_random(cls, comm: Any, global_blocks: int, block_size_min: int, block_size_max: int, density: float = 0.01, dtype: type = np.float64, seed: int = 42) -> 'VBCSR'
```
Creates a random connected matrix for benchmarking purposes.

### Methods

#### `add_block`
```python
add_block(self, g_row: int, g_col: int, data: np.ndarray, mode: AssemblyMode = AssemblyMode.ADD) -> None
```
Adds or inserts a dense block into the matrix.
- **g_row**, **g_col**: Global block indices.
- **data**: 2D numpy array.

#### `assemble`
```python
assemble(self) -> None
```
Finalizes matrix assembly. Must be called after adding blocks and before multiplication.

#### `extract_submatrix`
```python
extract_submatrix(self, global_indices: List[int]) -> 'VBCSR'
```
Extracts a submatrix corresponding to the specified global block indices.
- **global_indices**: List of global block indices to extract.
- **Returns**: A new `VBCSR` matrix containing the submatrix.

#### `insert_submatrix`
```python
insert_submatrix(self, sub_mat: 'VBCSR', global_indices: List[int]) -> None
```
Inserts a submatrix back into the original matrix at the specified global block indices.
- **sub_mat**: The submatrix to insert (must be a `VBCSR` object).
- **global_indices**: List of global block indices corresponding to the submatrix rows/cols.

#### `to_dense`
```python
to_dense(self) -> np.ndarray
```
Converts the locally owned part of the matrix to a dense 2D NumPy array.
- **Returns**: 2D numpy array of shape `(owned_rows, all_local_cols)`.

#### `from_dense`
```python
from_dense(self, dense_matrix: np.ndarray) -> None
```
Fills the matrix with values from a dense 2D NumPy array. The elements outside the sparsity pattern will be ignored.
- **dense_matrix**: 2D numpy array containing the values.

#### `mult`
```python
mult(self, x: Union[DistVector, DistMultiVector, np.ndarray], y: Optional[Union[DistVector, DistMultiVector]] = None) -> Union[DistVector, DistMultiVector]
```
Performs matrix multiplication $y = A \times x$.
- **x**: Input vector/multivector or numpy array (local part).
- **y**: Optional output vector.

#### `spmm`
```python
spmm(self, B: 'VBCSR', threshold: float = 0.0, transA: bool = False, transB: bool = False) -> 'VBCSR'
```
Sparse Matrix-Matrix Multiplication: $C = op(A) \times op(B)$.
- **B**: The matrix to multiply with.
- **threshold**: Threshold for dropping small blocks.
- **transA**, **transB**: If True, use transpose/conjugate transpose.

#### `@` Operator
The `@` operator is supported for both matrix-vector and matrix-matrix multiplication:
```python
y = A @ x  # Matrix-Vector
C = A @ B  # SpMM (no filtering)
```
> [!NOTE]
> The `@` operator performs standard SpMM without filtering (`threshold=0.0`). If you need to filter small blocks, use the `spmm` method directly.

#### `create_vector`
```python
create_vector(self) -> DistVector
```
Creates a `DistVector` compatible with this matrix.

#### `create_multivector`
```python
create_multivector(self, k: int) -> DistMultiVector
```
Creates a `DistMultiVector` with `k` columns compatible with this matrix.

#### `scale`
```python
scale(self, alpha: Union[float, complex, int]) -> None
```
Scales the matrix in-place by `alpha`.

#### `shift`
```python
shift(self, alpha: Union[float, complex, int]) -> None
```
Adds `alpha` to the diagonal elements (scalar shift).

#### `add_diagonal`
```python
add_diagonal(self, v: Union[DistVector, np.ndarray]) -> None
```
Adds a vector `v` to the diagonal elements ($A_{ii} += v_i$).

---

## DistVector

Represents a distributed 1D vector.

### Methods

#### `to_numpy` / `from_numpy`
```python
to_numpy(self) -> np.ndarray
from_numpy(self, arr: np.ndarray) -> None
```
Convert between the **locally owned** part of the vector and a NumPy array.

#### `set_constant`
```python
set_constant(self, val: Union[float, complex, int]) -> None
```
Sets all local elements to `val`.

#### `scale`
```python
scale(self, alpha: Union[float, complex, int]) -> None
```
Scales the vector in-place.

#### `axpy`
```python
axpy(self, alpha: Union[float, complex, int], x: 'DistVector') -> None
```
Computes $y = \alpha x + y$ in-place.

#### `dot`
```python
dot(self, other: 'DistVector') -> Union[float, complex]
```
Computes the global dot product $\sum \bar{x}_i y_i$.

#### `pointwise_mult`
```python
pointwise_mult(self, other: 'DistVector') -> None
```
Element-wise multiplication $y_i = y_i * x_i$.

---

## DistMultiVector

Represents a distributed collection of vectors (2D, column-major).

### Methods

#### `to_numpy` / `from_numpy`
Convert between locally owned rows and a 2D NumPy array.

#### `num_vectors`
Returns the number of columns.

#### `bdot`
```python
bdot(self, other: 'DistMultiVector') -> list
```
Computes batch dot products (one for each column).

#### `pointwise_mult`
```python
pointwise_mult(self, other: Union['DistMultiVector', DistVector]) -> None
```
Element-wise multiplication. If `other` is a `DistVector`, it is broadcast across all columns.
