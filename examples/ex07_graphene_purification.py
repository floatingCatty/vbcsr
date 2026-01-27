"""
Example 07: Graphene Density Matrix via McWeeny Purification

This example demonstrates:
1. Generating a tight-binding Hamiltonian for a graphene lattice.
2. Filling the Hamiltonian into a distributed VBCSR matrix.
3. Computing the density matrix using the McWeeny purification algorithm.
4. Comparing the results with exact diagonalization.

Run with:
    mpirun -np 4 python ex07_graphene_purification.py
"""

import numpy as np
import scipy.sparse as sp
import vbcsr
from vbcsr import VBCSR, MPI, HAS_MPI
import time

def create_graphene_hamiltonian(nx, ny, t=-2.7):
    """
    Create a tight-binding Hamiltonian for a graphene lattice (honeycomb).
    Each unit cell has 2 atoms (A and B).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() if hasattr(comm, 'Get_rank') else 0
    size = comm.Get_size() if hasattr(comm, 'Get_size') else 1
    
    n_atoms = 2 * nx * ny
    # We'll use 1x1 blocks for simplicity in this example, 
    # but VBCSR can handle any block size.
    
    # Global indices for atoms
    # (ix, iy, sub) -> ix * ny * 2 + iy * 2 + sub
    def get_idx(ix, iy, sub):
        return (ix % nx) * ny * 2 + (iy % ny) * 2 + sub

    # Each rank owns a portion of the atoms (rows)
    atoms_per_rank = (n_atoms + size - 1) // size
    start_atom = rank * atoms_per_rank
    end_atom = min((rank + 1) * atoms_per_rank, n_atoms)
    
    owned_indices = list(range(start_atom, end_atom))
    block_sizes = [1] * len(owned_indices)
    
    # Adjacency: each atom connects to 3 neighbors
    adj = []
    for i in owned_indices:
        # Decode i
        ix = i // (ny * 2)
        iy = (i % (ny * 2)) // 2
        sub = i % 2
        
        neighbors = []
        if sub == 0: # A atom
            neighbors.append(get_idx(ix, iy, 1)) # B in same cell
            neighbors.append(get_idx(ix - 1, iy, 1)) # B in cell to the left
            neighbors.append(get_idx(ix, iy - 1, 1)) # B in cell below
        else: # B atom
            neighbors.append(get_idx(ix, iy, 0)) # A in same cell
            neighbors.append(get_idx(ix + 1, iy, 0)) # A in cell to the right
            neighbors.append(get_idx(ix, iy + 1, 0)) # A in cell above
        
        # Add self-connection for diagonal
        neighbors.append(i)
        adj.append(list(set(neighbors)))

    # Create VBCSR matrix
    H = VBCSR.create_distributed(owned_indices, block_sizes, adj, comm=comm)
    
    # Fill Hamiltonian
    for idx, i in enumerate(owned_indices):
        ix = i // (ny * 2)
        iy = (i % (ny * 2)) // 2
        sub = i % 2
        
        # On-site energy (diagonal)
        H.add_block(i, i, np.zeros((1, 1))) 
        
        # Hopping
        if sub == 0:
            H.add_block(i, get_idx(ix, iy, 1), np.array([[t]]))
            H.add_block(i, get_idx(ix - 1, iy, 1), np.array([[t]]))
            H.add_block(i, get_idx(ix, iy - 1, 1), np.array([[t]]))
        else:
            H.add_block(i, get_idx(ix, iy, 0), np.array([[t]]))
            H.add_block(i, get_idx(ix + 1, iy, 0), np.array([[t]]))
            H.add_block(i, get_idx(ix, iy + 1, 0), np.array([[t]]))
            
    H.assemble()
    return H

def mcweeny_purification(H, max_iter=20, threshold=1e-6):
    """
    Compute the density matrix P using McWeeny purification:
    P_{n+1} = 3P_n^2 - 2P_n^3
    """
    comm = H.comm
    rank = comm.Get_rank() if hasattr(comm, 'Get_rank') else 0
    
    # 1. Scale Hamiltonian to [0, 1]
    # For graphene, eigenvalues are in [-3|t|, 3|t|]
    # Let's use a safe bound.
    t_val = 2.7
    E_min, E_max = -3.1 * t_val, 3.1 * t_val
    
    # P = (E_max * I - H) / (E_max - E_min)
    # This maps [E_min, E_max] to [1, 0]
    # Occupied states (near E_min) map to ~1, unoccupied to ~0.
    
    P = H.copy()
    P.scale(-1.0 / (E_max - E_min))
    P.shift(E_max / (E_max - E_min))
    
    if rank == 0:
        print(f"Starting McWeeny purification (max_iter={max_iter}, threshold={threshold})...")
    
    for i in range(max_iter):
        t0 = time.time()
        
        # P2 = P * P
        P2 = P.spmm(P, threshold=threshold)
        
        # P3 = P2 * P
        P3 = P2.spmm(P, threshold=threshold)
        
        # P = 3P^2 - 2P^3
        P = 3.0 * P2 - 2.0 * P3
        P.filter_blocks(threshold)
        
        t1 = time.time()

        density = P.get_block_density()
        
        # Check idempotency: trace(P^2 - P) or similar
        # For simplicity, just print progress
        if rank == 0:
            print(f"  Iteration {i+1:2d}: time = {t1-t0:.4f}s, density = {density:.4f}")
            
    return P

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() if hasattr(comm, 'Get_rank') else 0
    
    # Graphene lattice dimensions
    nx, ny = 20, 20
    
    if rank == 0:
        print(f"--- Graphene Purification Example ({2*nx*ny} atoms) ---")
        if not HAS_MPI:
            print("Running in SERIAL mode (mpi4py not found).")
    
    # 1. Create Hamiltonian
    H = create_graphene_hamiltonian(nx, ny)
    
    # 2. Compute Density Matrix via Purification
    P_purified = mcweeny_purification(H, max_iter=12, threshold=1e-5)
    
    # 3. Compare with Exact Diagonalization (on Rank 0)
    # Convert local parts to dense for comparison
    # Note: This is only feasible for small systems.
    
    # Gather full matrix on Rank 0
    H_dense = H.to_scipy().toarray()
    # In a real distributed run, to_scipy() only returns the local part.
    # We need to gather it.
    
    # For this example, let's just use the local part comparison if serial,
    # or gather if small.
    
    # Actually, VBCSR.to_scipy() returns the local portion.
    # Let's gather the full matrix on rank 0 for verification.
    def gather_matrix(mat):
        local_sp = mat.to_scipy(format='csr')
        if HAS_MPI and comm is not None:
            all_mats = comm.gather(local_sp, root=0)
            if rank == 0:
                return sp.vstack(all_mats).toarray()
            return None
        else:
            return local_sp.toarray()

    H_full = gather_matrix(H)
    P_full_purified = gather_matrix(P_purified)
    
    if rank == 0:
        print("\nVerifying results...")
        
        # Exact Diagonalization
        evals, evecs = np.linalg.eigh(H_full)
        
        # Fermi level (half-filling for graphene)
        n_occ = len(evals) // 2
        P_exact = evecs[:, :n_occ] @ evecs[:, :n_occ].T.conj()
        
        # Error metrics
        diff = P_full_purified - P_exact
        err_frob = np.linalg.norm(diff)
        err_max = np.max(np.abs(diff))
        
        print(f"Frobenius Norm Error: {err_frob:.2e}")
        print(f"Max Absolute Error:   {err_max:.2e}")
        
        if err_max < 1e-2:
            print("\nSUCCESS: Purification results match diagonalization!")
        else:
            print("\nWARNING: Large error detected. Check scaling or iterations.")

if __name__ == "__main__":
    main()
