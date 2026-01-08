import numpy as np
import vbcsr
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    if size < 2:
        print("This example requires at least 2 MPI ranks.")
        return

    if rank == 0:
        print("=== Example 2: Distributed Usage ===")

    # 1. Distributed Construction
    # Rank 0 owns Block 0 (size 2)
    # Rank 1 owns Block 1 (size 2)
    
    if rank == 0:
        owned_indices = [0]
        my_block_sizes = [2]
        my_adj = [[0, 1]] # Block 0 connects to 0 and 1
    else:
        owned_indices = [1]
        my_block_sizes = [2]
        my_adj = [[0, 1]] # Block 1 connects to 0 and 1
        
    mat = vbcsr.VBCSR.create_distributed(comm, owned_indices, my_block_sizes, my_adj)
    
    # 2. Add blocks
    # Each rank adds blocks it knows about. 
    # Usually, you add blocks corresponding to your owned rows, but you can add any block.
    if rank == 0:
        mat.add_block(0, 0, np.eye(2))
        mat.add_block(0, 1, np.full((2,2), 0.5))
    else:
        mat.add_block(1, 1, np.eye(2))
        mat.add_block(1, 0, np.full((2,2), 0.5))
        
    mat.assemble()
    
    # 3. Create distributed vector
    v = mat.create_vector()
    # Global vector: [1,1, 2,2]
    if rank == 0:
        v.from_numpy(np.array([1.0, 1.0]))
    else:
        v.from_numpy(np.array([2.0, 2.0]))
        
    # 4. Multiply
    res = mat.mult(v)
    print(f"Rank {rank} Result: {res.to_numpy()}")
    
    # Expected:
    # Rank 0 (Rows 0-1): I*[1,1] + 0.5*[2,2] = [1,1] + [1,1] = [2,2]
    # Rank 1 (Rows 2-3): 0.5*[1,1] + I*[2,2] = [0.5,0.5] + [2,2] = [2.5, 2.5]

if __name__ == "__main__":
    main()
