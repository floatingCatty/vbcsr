"""
Example 10: Mixed-Element Structural Mechanics (Variable Block Sizes)

This example demonstrates:
1. Assembling a stiffness matrix for a system with variable degrees of freedom (DOFs) per node.
   - Truss nodes: 3 DOFs (u_x, u_y, u_z)
   - Beam nodes: 6 DOFs (u_x, u_y, u_z, theta_x, theta_y, theta_z)
2. Using VBCSR to handle the variable block sizes (3x3, 6x6, 3x6, 6x3).
3. Solving the resulting linear system Ku = f using SciPy's iterative solvers.

Run with:
    python ex10_variable_pde.py
"""

import numpy as np
import vbcsr
from vbcsr import VBCSR, MPI, HAS_MPI
from scipy.sparse.linalg import cg
import time

def assemble_mixed_stiffness(nx, ny, nz):
    """
    Assemble a stiffness matrix for a 3D grid of nodes.
    Bottom half: Truss nodes (3 DOFs).
    Top half: Beam nodes (6 DOFs).
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() if hasattr(comm, 'Get_rank') else 0
    size = comm.Get_size() if hasattr(comm, 'Get_size') else 1
    
    n_nodes = nx * ny * nz
    nodes_per_rank = (n_nodes + size - 1) // size
    start_node = rank * nodes_per_rank
    end_node = min((rank + 1) * nodes_per_rank, n_nodes)
    
    owned_indices = list(range(start_node, end_node))
    
    def get_dof_count(node_idx):
        # Bottom half (z < nz/2) are truss nodes, top half are beam nodes
        iz = node_idx // (nx * ny)
        return 3 if iz < nz // 2 else 6

    block_sizes = [get_dof_count(i) for i in owned_indices]
    
    # Define connectivity (6-neighbor grid)
    adj = []
    for i in owned_indices:
        ix = (i % (nx * ny)) % nx
        iy = (i % (nx * ny)) // nx
        iz = i // (nx * ny)
        
        neighbors = [i]
        if ix > 0: neighbors.append(i - 1)
        if ix < nx - 1: neighbors.append(i + 1)
        if iy > 0: neighbors.append(i - nx)
        if iy < ny - 1: neighbors.append(i + nx)
        if iz > 0: neighbors.append(i - nx * ny)
        if iz < nz - 1: neighbors.append(i + nx * ny)
        adj.append(neighbors)

    K = VBCSR.create_distributed(owned_indices, block_sizes, adj, comm=comm)
    
    # Fill with "stiffness" blocks
    # For a real FEM, these would be element stiffness matrices.
    # Here we use synthetic SPD blocks for demonstration.
    for i in owned_indices:
        ix = (i % (nx * ny)) % nx
        iy = (i % (nx * ny)) // nx
        iz = i // (nx * ny)
        
        di = get_dof_count(i)
        
        # Self-stiffness (diagonal block)
        # Must be SPD
        diag = np.eye(di) * 100.0
        K.add_block(i, i, diag)
        
        # Off-diagonal blocks to neighbors
        for ni in [i-1, i+1, i-nx, i+nx, i-nx*ny, i+nx*ny]:
            if ni < 0 or ni >= n_nodes: continue
            
            # Check if ni is a valid neighbor (grid boundaries)
            nix = (ni % (nx * ny)) % nx
            niy = (ni % (nx * ny)) // nx
            niz = ni // (nx * ny)
            if abs(ix-nix) + abs(iy-niy) + abs(iz-niz) > 1: continue
            
            dn = get_dof_count(ni)
            # Random block of size (di x dn)
            # To keep K symmetric, we should ideally add K_ij and K_ji consistently.
            # VBCSR.add_block handles remote blocks, so we can just add the upper triangle
            # or add both and let it sum.
            block = np.random.rand(di, dn) * -1.0
            K.add_block(i, ni, block)
            
    K.assemble()
    return K

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank() if hasattr(comm, 'Get_rank') else 0
    
    nx, ny, nz = 10, 10, 10
    if rank == 0:
        print(f"--- Mixed-Element PDE Example ({nx*ny*nz} nodes) ---")
        print(f"Nodes 0-{nx*ny*(nz//2)-1}: 3 DOFs (Truss)")
        print(f"Nodes {nx*ny*(nz//2)}-{nx*ny*nz-1}: 6 DOFs (Beam)")
    
    start_time = time.time()
    K = assemble_mixed_stiffness(nx, ny, nz)
    if rank == 0:
        print(f"Matrix assembly finished in {time.time()-start_time:.2f}s")
        print(f"Global shape: {K.shape}")
    
    # Create RHS vector f
    f = K.create_vector()
    f.set_constant(1.0)
    
    # Solve Ku = f using Conjugate Gradient
    # VBCSR implements the LinearOperator interface, so it works with scipy.sparse.linalg
    if rank == 0:
        print("\nSolving Ku = f using Conjugate Gradient...")
    
    u = K.create_vector()
    u.set_constant(0.0)
    
    # Note: CG expects numpy arrays for x and b if we use the standard scipy CG.
    # However, we can wrap our DistVector to look like a numpy array or use a custom solver.
    # For this example, let's use a simple power iteration or a manual CG to stay distributed.
    
    def distributed_cg(A, b, x0, max_iter=100, tol=1e-6):
        r = b - A @ x0
        p = r.duplicate()
        rsold = r.dot(r).real
        
        for i in range(max_iter):
            Ap = A @ p
            alpha = rsold / p.dot(Ap).real
            x0.axpy(alpha, p)
            r.axpy(-alpha, Ap)
            rsnew = r.dot(r).real
            if np.sqrt(rsnew) < tol:
                if rank == 0: print(f"  Converged in {i+1} iterations.")
                break
            p.axpby(1.0, r, rsnew / rsold)
            rsold = rsnew
        return x0

    start_solve = time.time()
    u = distributed_cg(K, f, u)
    
    if rank == 0:
        print(f"Solve finished in {time.time()-start_solve:.2f}s")
        
        # Verify result
        res = f - K @ u
        norm_res = np.sqrt(res.dot(res).real)
        print(f"Final residual norm: {norm_res:.2e}")
        
        if norm_res < 1e-5:
            print("\nSUCCESS: Variable-block PDE system solved correctly!")

if __name__ == "__main__":
    main()
