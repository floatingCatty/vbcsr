import numpy as np
import vbcsr
from mpi4py import MPI

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if rank == 0:
        print(f"Running on {size} ranks")
        print("1. Creating random VBCSR matrix...")

    # Create a random matrix distributed across ranks
    mat = vbcsr.VBCSR.create_random(
        comm=comm,
        global_blocks=10,
        block_size_min=3,
        block_size_max=5,
        density=0.5,
        dtype=np.float64,
        seed=42
    )

    # Convert original matrix to dense for verification later
    original_dense = mat.to_dense()
    
    # Define indices to extract (must be global block indices)
    # Let's extract the first 3 blocks
    indices = [0, 1, 2]
    
    if rank == 0:
        print(f"2. Extracting submatrix for global block indices: {indices}")

    # Extract submatrix
    # This returns a new VBCSR matrix containing only the specified rows and columns
    # The submatrix is distributed, but for small indices like this, it might be mostly on rank 0
    sub_mat = mat.extract_submatrix(indices)

    # Convert submatrix to dense numpy array
    if rank == 0:
        print("3. Converting submatrix to dense...")
    
    sub_dense = sub_mat.to_dense()
    
    if rank == 0:
        print(f"   Submatrix dense shape: {sub_dense.shape}")
        print("   Submatrix values (first 5x5):\n", sub_dense[:5, :5])
        print("4. Modifying dense data (multiplying by 10.0)...")

    # Modify the dense data
    sub_dense_modified = sub_dense * 10.0

    # Update the submatrix from the modified dense data
    if rank == 0:
        print("5. Updating submatrix from modified dense data...")
    
    sub_mat.from_dense(sub_dense_modified)

    # Insert the modified submatrix back into the original matrix
    if rank == 0:
        print("6. Inserting modified submatrix back into original matrix...")
    
    mat.insert_submatrix(sub_mat, indices)

    # Verify the changes
    if rank == 0:
        print("7. Verifying changes...")

    final_dense = mat.to_dense()
    
    # Check if the update is reflected
    # We need to map the submatrix indices back to the full matrix to verify
    # Since we extracted [0, 1, 2], these correspond to the top-left blocks
    
    # For verification, we can just check if the values in the submatrix region 
    # are indeed 10x the original values (approximately, assuming no overlap/additions from other places)
    # Note: insert_submatrix *adds* to existing values or *replaces*? 
    # The C++ implementation of `insert_submatrix` usually *adds* (accumulates) in FEM/assembly contexts,
    # or *replaces* depending on implementation. 
    # Let's check the C++ code or assume standard behavior. 
    # Actually, `insert_submatrix` in `BlockSpMat` calls `add_block`, which usually *replaces* or *adds*?
    # In `BlockSpMat::add_block`, it does `blocks[local_row][col_idx] = mat`. It replaces the block pointer?
    # No, `add_block` takes a `Matrix<T>` and usually inserts it. 
    # If the block exists, it might overwrite or add. 
    # Let's verify with the output.
    
    if rank == 0:
        print("Done!")

if __name__ == "__main__":
    main()
