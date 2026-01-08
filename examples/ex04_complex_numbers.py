import numpy as np
import vbcsr
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    if comm.Get_size() > 1:
        return

    print("=== Example 4: Complex Numbers Support ===")

    # Create Complex Matrix
    global_blocks = 1
    block_sizes = [2]
    adjacency = [[0]]
    
    # Specify dtype=np.complex128
    mat = vbcsr.VBCSR.create_serial(comm, global_blocks, block_sizes, adjacency, dtype=np.complex128)
    
    # Add complex block
    # [ 1+1j, 0    ]
    # [ 0,    1-1j ]
    b00 = np.array([[1+1j, 0], [0, 1-1j]], dtype=np.complex128)
    mat.add_block(0, 0, b00)
    mat.assemble()
    
    # Complex Vector
    v_np = np.array([1.0, 1.0j], dtype=np.complex128)
    
    # Multiply
    res = mat.mult(v_np)
    print(f"Result: {res.to_numpy()}")
    
    # Expected:
    # Row 0: (1+1j)*1 + 0 = 1+1j
    # Row 1: 0 + (1-1j)*1j = 1+1j
    
    # Operators with complex scalars
    mat.scale(2.0j) # Scale by 2j
    res2 = mat.mult(v_np)
    print(f"Result after scale(2j): {res2.to_numpy()}")
    # Expected: (1+1j)*2j = -2+2j

if __name__ == "__main__":
    main()
