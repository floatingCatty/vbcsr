import numpy as np
import vbcsr
from vbcsr import VBCSR, MPI, HAS_MPI
import scipy.sparse.linalg

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() if HAS_MPI and comm else 0
    size = comm.Get_size() if HAS_MPI and comm else 1
    
    if rank == 0:
        print("=== Example 3: SciPy Solver Integration ===")

    # Create a simple diagonal matrix for solving Ax=b
    # A = 2*I
    
    global_blocks = 2
    block_sizes = [2, 2]
    adjacency = [[0], [1]] # Diagonal only
    
    mat = VBCSR.create_serial(global_blocks, block_sizes, adjacency, comm=comm)
    mat.add_block(0, 0, np.eye(2) * 2.0)
    mat.add_block(1, 1, np.eye(2) * 2.0)
    mat.assemble()
    
    # RHS b = [2, 4, 6, 8]
    # Expected x = [1, 2, 3, 4]
    b = np.array([2.0, 4.0, 6.0, 8.0])
    
    # Use SciPy CG solver
    # VBCSR is a LinearOperator, so we can pass it directly
    
    # Note: For distributed runs, SciPy solvers run on each rank independently 
    # if we just pass the local operator. This example is SERIAL.
    # For distributed solvers, one needs to be careful about dot products.
    # But VBCSR works fine as a standard LinearOperator in serial.
    
    if size == 1:
        x, info = scipy.sparse.linalg.cg(mat, b, rtol=1e-5)
        if info == 0:
            print(f"Solver converged. Solution: {x}")
        else:
            print("Solver failed to converge")
    else:
        print("SciPy solver example is designed for serial execution.")

if __name__ == "__main__":
    main()
