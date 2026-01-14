import numpy as np
import vbcsr
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Running SpMM example on {size} ranks")
        print("1. Creating random VBCSR matrices A and B...")

    # Create random matrix A
    A = vbcsr.VBCSR.create_random(
        comm=comm,
        global_blocks=10,
        block_size_min=3,
        block_size_max=5,
        density=0.3,
        dtype=np.float64,
        seed=42
    )

    # Create random matrix B
    B = vbcsr.VBCSR.create_random(
        comm=comm,
        global_blocks=10,
        block_size_min=3,
        block_size_max=5,
        density=0.3,
        dtype=np.float64,
        seed=42
    )

    if rank == 0:
        print("2. Performing SpMM: C = A @ B")

    # Perform SpMM using the @ operator
    C = A @ B

    if rank == 0:
        print("3. Verifying results with dense multiplication...")

    # Convert to dense for verification
    A_dense = A.to_dense()
    B_dense = B.to_dense()
    C_dense_vbcsr = C.to_dense()

    # Gather full dense matrices on rank 0 for verification
    # Note: to_dense() returns the LOCAL part. We need to gather them.
    # For simplicity in this example, we assume serial or small distributed case where we can gather.
    # But wait, to_dense() returns (owned_rows, global_cols) or (owned_rows, local_cols)?
    # The doc says (owned_rows, all_local_cols). 
    # For verification, let's just check local parts if possible, or gather.
    
    # Let's gather everything to rank 0
    # We need to know the global shape and distribution.
    # A.shape is (N, N).
    
    # Gather A
    A_global = comm.gather(A_dense, root=0)
    B_global = comm.gather(B_dense, root=0)
    C_global = comm.gather(C_dense_vbcsr, root=0)

    if rank == 0:
        # Stack the gathered parts
        # A_dense is (my_rows, global_cols) ? No, to_dense returns local part.
        # Actually, `to_dense` in `BlockSpMat` returns `(owned_rows, all_local_cols)`.
        # If the matrix is distributed, `all_local_cols` might be just `global_cols` if it's replicated or fully known?
        # In `BlockSpMat`, `to_dense` usually returns the full row strip if it's row-distributed.
        # Let's assume it returns (my_rows, global_cols).
        
        A_full = np.vstack(A_global)
        B_full = np.vstack(B_global)
        C_full_vbcsr = np.vstack(C_global)
        
        print(f"   A shape: {A_full.shape}")
        print(f"   B shape: {B_full.shape}")
        print(f"   C (VBCSR) shape: {C_full_vbcsr.shape}")
        
        # Compute expected C using numpy
        C_expected = A_full @ B_full
        
        # Compare
        diff = np.linalg.norm(C_full_vbcsr - C_expected)
        print(f"   Difference norm: {diff}")
        
        if diff < 1e-10:
            print("   SUCCESS: SpMM result matches dense multiplication!")
        else:
            print("   FAILURE: SpMM result mismatch!")

    # Also demonstrate Matrix-Vector multiplication with @
    if rank == 0:
        print("\n4. Performing Matrix-Vector multiplication: y = A @ x")
        
    # Create a random vector
    x = A.create_vector()
    x.set_constant(1.0) # Set all to 1.0
    
    # Perform MV
    y = A @ x
    
    # Verify MV
    if rank == 0:
        print("5. Verifying MV results...")
        
    y_local = y.to_numpy()
    y_global = comm.gather(y_local, root=0)
    
    if rank == 0:
        y_full = np.concatenate(y_global)
        
        # Expected y = A_full * ones
        x_full = np.ones(A_full.shape[1])
        y_expected = A_full @ x_full
        
        diff_mv = np.linalg.norm(y_full - y_expected)
        print(f"   MV Difference norm: {diff_mv}")
        
        if diff_mv < 1e-10:
            print("   SUCCESS: MV result matches dense multiplication!")
        else:
            print("   FAILURE: MV result mismatch!")

    if rank == 0:
        print("Done!")

if __name__ == "__main__":
    main()
