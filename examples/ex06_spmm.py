import numpy as np
import vbcsr
from vbcsr import VBCSR, MPI, HAS_MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() if HAS_MPI and comm else 0
    size = comm.Get_size() if HAS_MPI and comm else 1

    if rank == 0:
        print(f"Running SpMM example on {size} ranks")
        print("1. Creating random VBCSR matrices A and B...")

    # Create random matrix A
    A = VBCSR.create_random(
        comm=comm,
        global_blocks=10,
        block_size_min=3,
        block_size_max=5,
        density=0.3,
        dtype=np.float64,
        seed=42
    )

    # Create random matrix B
    B = VBCSR.create_random(
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
    if HAS_MPI and comm and size > 1:
        A_global = comm.gather(A_dense, root=0)
        B_global = comm.gather(B_dense, root=0)
        C_global = comm.gather(C_dense_vbcsr, root=0)
    else:
        A_global = [A_dense]
        B_global = [B_dense]
        C_global = [C_dense_vbcsr]

    if rank == 0:
        # Stack the gathered parts
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
    if HAS_MPI and comm and size > 1:
        y_global = comm.gather(y_local, root=0)
    else:
        y_global = [y_local]
    
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
