#include "../block_csr.hpp"
#include "../dist_vector.hpp"
#include <iostream>
#include <vector>
#include <cassert>

using namespace vbcsr;

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    if (size < 2) {
        if (rank == 0) std::cout << "This test requires at least 2 ranks." << std::endl;
        MPI_Finalize();
        return 0;
    }

    // Scenario:
    // Rank 0 only.
    // Graph A: Owned {0, 1}. Block sizes {1, 1}. Total size 2.
    // Graph B: Owned {0}. Block size {2}. Total size 2.
    
    if (rank == 0) {
        // Graph A
        std::vector<int> owned_A = {0, 1};
        std::vector<int> sizes_A = {1, 1};
        std::vector<std::vector<int>> adj(2);
        
        DistGraph graph_A(MPI_COMM_WORLD);
        // We need to be careful with construct_distributed requiring collective calls.
        // If we run with np 2, rank 1 must participate.
    }
    
    // To make it easy, let's have Rank 1 be empty or symmetric.
    // Rank 0: A={0,1} sz={1,1}. B={0} sz={2}.
    // Rank 1: A={} sz={}. B={} sz={}.
    
    std::vector<int> owned_A, sizes_A;
    std::vector<int> owned_B, sizes_B;
    
    if (rank == 0) {
        owned_A = {0, 1};
        sizes_A = {1, 1};
        owned_B = {0};
        sizes_B = {2};
    }
    
    // Adj needs to match block count
    std::vector<std::vector<int>> adj_A(owned_A.size());
    std::vector<std::vector<int>> adj_B(owned_B.size());
    
    DistGraph graph_A(MPI_COMM_WORLD);
    graph_A.construct_distributed(owned_A, sizes_A, adj_A);
    
    DistGraph graph_B(MPI_COMM_WORLD);
    graph_B.construct_distributed(owned_B, sizes_B, adj_B);
    
    // Vector x bound to Graph A
    DistVector<double> x(&graph_A);
    if (rank == 0) {
        x[0] = 10.0; // Block 0
        x[1] = 20.0; // Block 1
    }
    
    // Matrix M bound to Graph B
    // Expects Block 0 of size 2.
    BlockSpMat<double, NaiveKernel<double>> M(&graph_B);
    if (rank == 0) {
        int gid = 0;
        double val[4] = {1.0, 0.0, 0.0, 1.0}; // Identity 2x2
        M.add_block(gid, gid, val, 2, 2, AssemblyMode::INSERT);
    }
    M.assemble();
    
    DistVector<double> y(&graph_B);
    
    try {
        // This should throw or fail, but will succeed silently
        M.mult(x, y);
    } catch (const std::exception& e) {
        std::cout << "Rank " << rank << " caught exception: " << e.what() << std::endl;
    }
    
    if (rank == 0) {
        // y should be [10.0, 20.0] because M is identity 2x2 and x is [10, 20].
        // But semantically, x was 2 blocks of size 1. M treated it as 1 block of size 2.
        // If the user intended x to be compatible with A, using it with M (Graph B) is a logic error.
        // The library should catch this mismatch.
        // Currently it does not.
        
        std::cout << "Rank 0: Execution finished." << std::endl;
        std::cout << "FAILURE: mult succeeded despite graph mismatch (2 blocks vs 1 block)." << std::endl;
    }
    
    MPI_Finalize();
    return 0;
}
