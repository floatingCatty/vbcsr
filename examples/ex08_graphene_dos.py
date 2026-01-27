"""
Example 08: Graphene Density of States (DOS) via Kernel Polynomial Method (KPM)

This example demonstrates:
1. Generating a graphene tight-binding Hamiltonian.
2. Scaling the Hamiltonian to the range [-1, 1].
3. Using the Kernel Polynomial Method (KPM) with stochastic trace estimation
   to compute the Density of States (DOS).
4. Saving the DOS data for plotting.

Run with:
    python ex08_graphene_dos.py
"""

import numpy as np
import vbcsr
from vbcsr import VBCSR, MPI, HAS_MPI
import time

def create_graphene_hamiltonian(nx, ny, t=-2.7):
    """Create a graphene Hamiltonian (honeycomb lattice)."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() if hasattr(comm, 'Get_rank') else 0
    size = comm.Get_size() if hasattr(comm, 'Get_size') else 1
    
    n_atoms = 2 * nx * ny
    atoms_per_rank = (n_atoms + size - 1) // size
    start_atom = rank * atoms_per_rank
    end_atom = min((rank + 1) * atoms_per_rank, n_atoms)
    
    owned_indices = list(range(start_atom, end_atom))
    block_sizes = [1] * len(owned_indices)
    
    def get_idx(ix, iy, sub):
        return (ix % nx) * ny * 2 + (iy % ny) * 2 + sub

    adj = []
    for i in owned_indices:
        ix = i // (ny * 2)
        iy = (i % (ny * 2)) // 2
        sub = i % 2
        neighbors = []
        if sub == 0: # A atom
            neighbors.extend([get_idx(ix, iy, 1), get_idx(ix - 1, iy, 1), get_idx(ix, iy - 1, 1)])
        else: # B atom
            neighbors.extend([get_idx(ix, iy, 0), get_idx(ix + 1, iy, 0), get_idx(ix, iy + 1, 0)])
        neighbors.append(i)
        adj.append(list(set(neighbors)))

    H = VBCSR.create_distributed(owned_indices, block_sizes, adj, comm=comm)
    for i in owned_indices:
        ix = i // (ny * 2)
        iy = (i % (ny * 2)) // 2
        sub = i % 2
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

def compute_kpm_dos(H, num_moments=200, num_random_vecs=10):
    """
    Compute DOS using Kernel Polynomial Method.
    """
    comm = H.comm
    rank = comm.Get_rank() if hasattr(comm, 'Get_rank') else 0
    n_total = H.shape[0]
    
    # 1. Scale H to [-1, 1]
    t_val = 2.7
    E_bound = 3.1 * t_val
    H_scaled = H.copy()
    H_scaled.scale(1.0 / E_bound)
    
    if rank == 0:
        print(f"Starting KPM DOS calculation (moments={num_moments}, random_vecs={num_random_vecs})...")
        if num_random_vecs > 1:
            print("Using DistMultiVector for batched recurrence.")
    
    start_time = time.time()
    
    if num_random_vecs > 1:
        # Batched version using DistMultiVector
        v0 = H.create_multivector(num_random_vecs)
        local_rows = v0.local_rows
        # Random normal vectors
        phi = np.random.normal(0, 1, (local_rows, num_random_vecs))
        v0.from_numpy(phi)
        
        v1 = H_scaled @ v0
        
        # mu_n = sum_r <r|T_n|r> / (num_r * N)
        # bdot returns a list of dots for each vector
        dots0 = v0.bdot(v0)
        dots1 = v0.bdot(v1)
        
        moments = np.zeros(num_moments)
        moments[0] = np.sum([d.real for d in dots0])
        moments[1] = np.sum([d.real for d in dots1])
        
        v_prev = v0
        v_curr = v1
        
        for n in range(2, num_moments):
            v_next = 2.0 * (H_scaled @ v_curr) - v_prev
            dots_n = v0.bdot(v_next)
            moments[n] = np.sum([d.real for d in dots_n])
            v_prev = v_curr
            v_curr = v_next
    else:
        # Single vector version
        v0 = H.create_vector()
        local_size = v0.local_size
        phi = np.random.normal(0, 1, local_size)
        v0.from_numpy(phi)
        
        v1 = H_scaled @ v0
        
        moments = np.zeros(num_moments)
        moments[0] = v0.dot(v0).real
        moments[1] = v0.dot(v1).real
        
        v_prev = v0
        v_curr = v1
        
        for n in range(2, num_moments):
            v_next = 2.0 * (H_scaled @ v_curr) - v_prev
            moments[n] = v0.dot(v_next).real
            v_prev = v_curr
            v_curr = v_next

    # Average moments and normalize
    moments /= (num_random_vecs * n_total)
    
    if rank == 0:
        print(f"  KPM recurrence finished in {time.time()-start_time:.2f}s")

    # Jackson Kernel for damping
    n = np.arange(num_moments)
    jackson = ((num_moments - n + 1) * np.cos(np.pi * n / (num_moments + 1)) + 
               np.sin(np.pi * n / (num_moments + 1)) / np.tan(np.pi / (num_moments + 1))) / (num_moments + 1)
    moments *= jackson
    
    # Reconstruct DOS on a grid
    energies = np.linspace(-1, 1, 400)
    dos = np.zeros_like(energies)
    for i, e in enumerate(energies):
        cheb_vals = np.cos(np.arange(num_moments) * np.arccos(e))
        dos[i] = (moments[0] + 2.0 * np.sum(moments[1:] * cheb_vals[1:])) / (np.pi * np.sqrt(1.0 - e**2))
        
    return energies * E_bound, dos / E_bound

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() if hasattr(comm, 'Get_rank') else 0
    
    nx, ny = 30, 30
    if rank == 0:
        print(f"--- Graphene DOS Example ({2*nx*ny} atoms) ---")
    
    H = create_graphene_hamiltonian(nx, ny)
    
    start_time = time.time()
    energies, dos = compute_kpm_dos(H, num_moments=100, num_random_vecs=10)
    
    if rank == 0:
        print(f"\nCalculation finished in {time.time()-start_time:.2f}s")
        
        # Save data
        np.savetxt("graphene_dos.txt", np.column_stack((energies, dos)), header="Energy(eV) DOS(states/eV)")
        print("DOS data saved to 'graphene_dos.txt'.")
        
        # Try to plot if matplotlib is available
        try:
            import matplotlib.pyplot as plt
            plt.figure(figsize=(8, 5))
            plt.plot(energies, dos, lw=2, color='blue')
            plt.fill_between(energies, dos, alpha=0.3, color='blue')
            plt.xlabel("Energy (eV)")
            plt.ylabel("DOS (states/eV)")
            plt.title(f"Graphene DOS (KPM, {2*nx*ny} atoms)")
            plt.grid(True, alpha=0.3)
            plt.savefig("graphene_dos.png", dpi=150)
            print("Plot saved to 'graphene_dos.png'.")
        except ImportError:
            print("Matplotlib not found. Skipping plot generation.")

if __name__ == "__main__":
    main()
