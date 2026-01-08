import unittest
import numpy as np
import vbcsr
from mpi4py import MPI
import sys

class TestVBCSR(unittest.TestCase):
    def setUp(self):
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        
        # Define a simple 4x4 system with 2 blocks of size 2
        self.global_blocks = 2
        self.block_sizes = [2, 2]
        self.adj = [[0, 1], [0, 1]] # Dense block structure
        
        if self.size == 1:
            self.mat = vbcsr.VBCSR.create_serial(self.comm, self.global_blocks, self.block_sizes, self.adj)
        else:
            # Distributed: Rank 0 owns block 0, Rank 1 owns block 1 (assuming np=2)
            if self.rank == 0:
                owned = [0]
                sizes = [2]
                adj = [[0, 1]]
            else:
                owned = [1]
                sizes = [2]
                adj = [[0, 1]]
            self.mat = vbcsr.VBCSR.create_distributed(self.comm, owned, sizes, adj)

        # Fill matrix
        # Block 0,0: I
        # Block 1,1: I
        # Block 0,1: 0.5
        # Block 1,0: 0.5
        
        # In distributed mode, each rank adds its own blocks (or any blocks)
        # Let's have everyone add everything they know about to be safe/simple, 
        # or strictly local.
        # Serial: Add all.
        # Distributed: Rank 0 adds row 0 blocks, Rank 1 adds row 1 blocks.
        
        if self.size == 1:
            self.mat.add_block(0, 0, np.eye(2))
            self.mat.add_block(1, 1, np.eye(2))
            self.mat.add_block(0, 1, np.full((2,2), 0.5))
            self.mat.add_block(1, 0, np.full((2,2), 0.5))
        else:
            if self.rank == 0:
                self.mat.add_block(0, 0, np.eye(2))
                self.mat.add_block(0, 1, np.full((2,2), 0.5))
            elif self.rank == 1:
                self.mat.add_block(1, 1, np.eye(2))
                self.mat.add_block(1, 0, np.full((2,2), 0.5))
                
        self.mat.assemble()

    def test_vector_ops(self):
        v = self.mat.create_vector()
        v.set_constant(1.0)
        
        # Scale
        v.scale(2.0)
        arr = v.to_numpy()
        self.assertTrue(np.allclose(arr, 2.0))
        
        # Add
        v2 = v + 1.0
        arr2 = v2.to_numpy()
        self.assertTrue(np.allclose(arr2, 3.0))
        
        # In-place add
        v += v2
        arr = v.to_numpy()
        self.assertTrue(np.allclose(arr, 5.0)) # 2 + 3
        
        # Dot
        dot_val = v.dot(v)
        # Global vector size is 4. Each element is 5.0.
        # Dot = 4 * 5*5 = 100
        self.assertAlmostEqual(dot_val, 100.0)

    def test_matrix_mult(self):
        v = self.mat.create_vector()
        # Global vector [1, 1, 1, 1]
        v.set_constant(1.0)
        
        res = self.mat.mult(v)
        res_np = res.to_numpy()
        
        # Matrix is:
        # [I, 0.5]
        # [0.5, I]
        # Row 0: [1,0, 0.5, 0.5] * [1,1,1,1] = 1 + 0.5 + 0.5 = 2.0
        # Row 1: [0,1, 0.5, 0.5] * [1,1,1,1] = 1 + 0.5 + 0.5 = 2.0
        # All rows should be 2.0
        
        self.assertTrue(np.allclose(res_np, 2.0))

    def test_matrix_ops(self):
        # Scale
        mat2 = self.mat * 2.0
        v = self.mat.create_vector()
        v.set_constant(1.0)
        res = mat2.mult(v)
        # Expected 4.0
        self.assertTrue(np.allclose(res.to_numpy(), 4.0))
        
        # Add
        mat3 = self.mat + self.mat
        res = mat3.mult(v)
        # Expected 4.0
        self.assertTrue(np.allclose(res.to_numpy(), 4.0))
        
        # Shift
        mat_shifted = self.mat.duplicate()
        mat_shifted.shift(1.0) # Add 1.0 to diagonal
        # Diagonal becomes 2.0. Off-diagonal 0.5.
        # Row sum: 2.0 + 0.5 + 0.5 = 3.0
        res = mat_shifted.mult(v)
        self.assertTrue(np.allclose(res.to_numpy(), 3.0))
        
        # Add Diagonal
        diag = self.mat.create_vector()
        diag.set_constant(0.5)
        mat_diag = self.mat.duplicate()
        mat_diag.add_diagonal(diag)
        # Diagonal becomes 1.5.
        # Row sum: 1.5 + 0.5 + 0.5 = 2.5
        res = mat_diag.mult(v)
        self.assertTrue(np.allclose(res.to_numpy(), 2.5))

    def test_multivector(self):
        mv = self.mat.create_multivector(2)
        mv.set_constant(1.0)
        
        res = self.mat.mult(mv)
        res_np = res.to_numpy()
        
        # Should be 2.0 everywhere
        self.assertTrue(np.allclose(res_np, 2.0))
        
        # Bdot
        dots = mv.bdot(mv)
        # Each column is [1,1,1,1]. Dot is 4.
        self.assertTrue(np.allclose(dots, 4.0))

    def test_scipy_interface(self):
        # Test _matvec
        v_np = np.ones(self.mat.create_vector().local_size)
        res_np = self.mat._matvec(v_np)
        self.assertTrue(np.allclose(res_np, 2.0))
        
        # Test _matmat
        mv_np = np.ones((self.mat.create_vector().local_size, 2))
        res_np = self.mat._matmat(mv_np)
        self.assertTrue(np.allclose(res_np, 2.0))

if __name__ == '__main__':
    # Filter args for unittest
    unittest.main(argv=[sys.argv[0]])
