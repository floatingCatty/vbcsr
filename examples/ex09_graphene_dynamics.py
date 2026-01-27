"""
Example 09: Quantum Wavepacket Dynamics on Graphene

This example demonstrates:
1. Generating a graphene tight-binding Hamiltonian.
2. Initializing a Gaussian wavepacket on the lattice.
3. Propagating the wavepacket in time using Chebyshev expansion of the propagator e^{-iHt}.
4. Visualizing the spreading of the wavepacket over time.

Run with:
    python ex09_graphene_dynamics.py
"""

import numpy as np
import vbcsr
from vbcsr import VBCSR, MPI, HAS_MPI
import time
from scipy.special import jn # Bessel functions for Chebyshev coefficients

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

    H = VBCSR.create_distributed(owned_indices, block_sizes, adj, dtype=np.complex128, comm=comm)
    for i in owned_indices:
        ix = i // (ny * 2)
        iy = (i % (ny * 2)) // 2
        sub = i % 2
        if sub == 0:
            H.add_block(i, get_idx(ix, iy, 1), np.array([[t]], dtype=np.complex128))
            H.add_block(i, get_idx(ix - 1, iy, 1), np.array([[t]], dtype=np.complex128))
            H.add_block(i, get_idx(ix, iy - 1, 1), np.array([[t]], dtype=np.complex128))
        else:
            H.add_block(i, get_idx(ix, iy, 0), np.array([[t]], dtype=np.complex128))
            H.add_block(i, get_idx(ix + 1, iy, 0), np.array([[t]], dtype=np.complex128))
            H.add_block(i, get_idx(ix, iy + 1, 0), np.array([[t]], dtype=np.complex128))
    H.assemble()
    return H

def initialize_wavepacket(nx, ny, x0, y0, sigma=2.0, k0=[0.0, 0.0]):
    """Initialize a Gaussian wavepacket centered at (x0, y0)."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() if hasattr(comm, 'Get_rank') else 0
    size = comm.Get_size() if hasattr(comm, 'Get_size') else 1
    
    n_atoms = 2 * nx * ny
    atoms_per_rank = (n_atoms + size - 1) // size
    start_atom = rank * atoms_per_rank
    end_atom = min((rank + 1) * atoms_per_rank, n_atoms)
    
    psi_local = np.zeros(end_atom - start_atom, dtype=np.complex128)
    
    # Graphene lattice vectors (approximate for visualization)
    # a1 = (3/2, sqrt(3)/2), a2 = (3/2, -sqrt(3)/2)
    # For simplicity, use a square grid for coordinates
    for i in range(start_atom, end_atom):
        ix = i // (ny * 2)
        iy = (i % (ny * 2)) // 2
        sub = i % 2
        
        # Coordinates
        x = ix * 1.5 + (0.5 if sub == 1 else 0.0)
        y = iy * np.sqrt(3) + (np.sqrt(3)/2 if sub == 1 else 0.0)
        
        dist2 = (x - x0)**2 + (y - y0)**2
        phase = np.exp(1j * (k0[0] * x + k0[1] * y))
        psi_local[i - start_atom] = np.exp(-dist2 / (2 * sigma**2)) * phase
        
    # Normalize
    local_norm2 = np.sum(np.abs(psi_local)**2)
    global_norm2 = comm.allreduce(local_norm2) if HAS_MPI and comm else local_norm2
    psi_local /= np.sqrt(global_norm2)
    
    return psi_local

def propagate_chebyshev(H, psi, dt, num_moments=50):
    """Propagate psi by dt using Chebyshev expansion of e^{-iH*dt}."""
    # Scale H to [-1, 1]
    t_val = 2.7
    E_max = 3.1 * t_val
    H_scaled = H.copy()
    H_scaled.scale(1.0 / E_max)
    
    # Coefficients: c_n = (2-delta_n0) * (-i)^n * J_n(E_max * dt)
    # We ignore the global phase e^{-i*E_center*t} since E_center = 0 for graphene
    alpha = E_max * dt
    coeffs = [(2 if n > 0 else 1) * ((-1j)**n) * jn(n, alpha) for n in range(num_moments)]
    
    v0 = H.create_vector()
    v0.from_numpy(psi)
    
    # Recurrence
    # v_n+1 = 2H_scaled * v_n - v_n-1
    res = v0 * coeffs[0]
    
    v1 = H_scaled @ v0
    res += v1 * coeffs[1]
    
    v_prev = v0
    v_curr = v1
    for n in range(2, num_moments):
        v_next = 2.0 * (H_scaled @ v_curr) - v_prev
        res += v_next * coeffs[n]
        v_prev = v_curr
        v_curr = v_next
        
    return res.to_numpy()

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() if hasattr(comm, 'Get_rank') else 0
    
    nx, ny = 40, 40
    if rank == 0:
        print(f"--- Graphene Wavepacket Dynamics ({2*nx*ny} atoms) ---")
    
    H = create_graphene_hamiltonian(nx, ny)
    
    # Initial wavepacket at center
    x0, y0 = nx * 0.75, ny * np.sqrt(3) / 2
    psi = initialize_wavepacket(nx, ny, x0, y0, sigma=3.0, k0=[1.0, 0.5])
    
    dt = 0.5 # Time step in units of 1/|t|
    num_steps = 10
    
    densities = []
    
    for step in range(num_steps + 1):
        if rank == 0:
            print(f"Step {step}/{num_steps}...")
        
        # Compute local density
        rho_local = np.abs(psi)**2
        
        # Gather full density on rank 0 for plotting
        if HAS_MPI and comm:
            rho_full = comm.gather(rho_local, root=0)
            if rank == 0:
                rho_full = np.concatenate(rho_full)
        else:
            rho_full = rho_local
            
        if rank == 0:
            densities.append(rho_full)
            
        if step < num_steps:
            psi = propagate_chebyshev(H, psi, dt)

    if rank == 0:
        print("\nDynamics finished. Generating plots...")
        try:
            import matplotlib.pyplot as plt
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Plot snapshots at start, middle, end
            indices = [0, num_steps // 2, num_steps]
            titles = ["t = 0", f"t = {num_steps//2 * dt}", f"t = {num_steps * dt}"]
            
            for ax, idx, title in zip(axes, indices, titles):
                rho = densities[idx]
                # Reshape for plotting (approximate)
                rho_grid = rho.reshape(nx, ny, 2).sum(axis=2)
                im = ax.imshow(rho_grid.T, origin='lower', cmap='viridis', extent=[0, nx, 0, ny])
                ax.set_title(title)
                ax.set_xlabel("x")
                ax.set_ylabel("y")
                plt.colorbar(im, ax=ax)
            
            plt.tight_layout()
            plt.savefig("graphene_dynamics.png", dpi=150)
            print("Plot saved to 'graphene_dynamics.png'.")
        except ImportError:
            print("Matplotlib not found. Skipping plot generation.")
            # Save data to file
            np.save("graphene_densities.npy", np.array(densities))
            print("Densities saved to 'graphene_densities.npy'.")

if __name__ == "__main__":
    main()
