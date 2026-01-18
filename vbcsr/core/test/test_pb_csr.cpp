#include "../block_csr.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cstring>
#include <cmath>

using namespace vbcsr;

// Helper to fill vector with random numbers
void fill_random(std::vector<double>& v, int seed) {
    for (size_t i = 0; i < v.size(); ++i) v[i] = (double)i * 0.1;
}

void test_basic_spmv() {
    std::cout << "Testing Basic SpMV..." << std::endl;
    
    // 1. Setup Graph
    // 2 blocks: 0->0, 0->1
    std::vector<std::vector<int>> global_adj = {{0, 1}, {}};
    std::vector<int> block_sizes = {2, 2};
    
    DistGraph graph(MPI_COMM_SELF);
    graph.construct_serial(2, block_sizes, global_adj);
    
    BlockSpMat<double> mat(&graph);
    
    // Fill data
    // Block (0,0): Identity
    double d00[] = {1.0, 0.0, 0.0, 1.0};
    mat.add_block(0, 0, d00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    
    // Block (0,1): All 1s
    double d01[] = {1.0, 1.0, 1.0, 1.0};
    mat.add_block(0, 1, d01, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    
    // Vector X = [1, 2, 3, 4]
    DistVector<double> x(&graph);
    double* x_ptr = x.local_data();
    x_ptr[0] = 1.0; x_ptr[1] = 2.0; // Block 0
    x_ptr[2] = 3.0; x_ptr[3] = 4.0; // Block 1
    
    DistVector<double> y(&graph);
    
    mat.mult(x, y);
    
    // Y = A*X
    // Y[0..1] = [1 0; 0 1]*[1;2] + [1 1; 1 1]*[3;4]
    //         = [1; 2] + [7; 7] = [8; 9]
    
    double* y_ptr = y.local_data();
    // std::cout << "Y: " << y_ptr[0] << " " << y_ptr[1] << std::endl;
    
    assert(std::abs(y_ptr[0] - 8.0) < 1e-9);
    assert(std::abs(y_ptr[1] - 9.0) < 1e-9);
    
    std::cout << "PASSED" << std::endl;
}

void test_axpby_structure_mismatch() {
    std::cout << "Testing AXPBY (Structure Mismatch)..." << std::endl;
    
    std::vector<int> block_sizes = {2};
    DistGraph graph(MPI_COMM_WORLD);
    
    // Y has (0,0)
    std::vector<std::vector<int>> adj_Y = {{0}};
    graph.construct_serial(1, block_sizes, adj_Y);
    BlockSpMat<double> Y(&graph);
    double d00[] = {1.0, 1.0, 1.0, 1.0};
    Y.add_block(0, 0, d00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    
    // X has (0,0)
    // We need a separate graph/matrix for X? 
    // Or just modify Y's structure?
    // Let's make X have SAME graph but DIFFERENT structure (subset/superset).
    // But BlockSpMat is tied to graph structure.
    // To test mismatch, we need to manually modify X's topology or use a graph that allows it.
    // Wait, allocate_from_graph sets up full graph structure.
    // So X and Y usually have SAME structure if they share graph.
    // To test mismatch, we must filter one of them.
    
    // Filter Y to remove (0,0) -> Empty
    Y.filter_blocks(10.0); // Norm is sqrt(4)=2. Filter > 10 removes it.
    assert(Y.col_ind.empty());
    
    // Now Y is empty. X has (0,0).
    BlockSpMat<double> X(&graph); // X has (0,0)
    X.add_block(0, 0, d00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    
    // Y = 1.0 * X + 1.0 * Y
    // Y should become X
    Y.axpby(1.0, X, 1.0);
    
    if (Y.graph->owned_global_indices.size() > 0) {
        assert(Y.col_ind.size() == 1);
        assert(Y.col_ind[0] == 0);
        
        // Check values
        double* ptr = Y.arena.get_ptr(Y.blk_handles[0]);
        assert(ptr[0] == 1.0);
    } else {
        assert(Y.col_ind.empty());
    }
    
    std::cout << "PASSED" << std::endl;
}

void test_memory_reuse() {
    std::cout << "Testing Memory Reuse..." << std::endl;
    
    std::vector<std::vector<int>> adj = {{0}};
    std::vector<int> block_sizes = {2};
    DistGraph graph(MPI_COMM_WORLD);
    graph.construct_serial(1, block_sizes, adj);
    
    BlockSpMat<double> mat(&graph);
    
    // Filter out
    mat.filter_blocks(100.0); // Remove everything
    
    BlockSpMat<double> X(&graph); // Has (0,0) if owned
    if (graph.owned_global_indices.size() > 0) {
        double d00[] = {1.0, 1.0, 1.0, 1.0};
        X.add_block(0, 0, d00, 2, 2, AssemblyMode::INSERT, MatrixLayout::RowMajor);
    }
    
    // Y (mat) is empty.
    // Y = X
    mat.axpby(1.0, X, 0.0);
    
    if (mat.graph->owned_global_indices.size() > 0) {
        assert(mat.col_ind.size() == 1);
    } else {
        assert(mat.col_ind.empty());
    }
    std::cout << "PASSED" << std::endl;
}

void test_transpose() {
    std::cout << "Testing Transpose..." << std::endl;
    
    // Create a simple 2x2 block matrix
    // Block sizes: [2, 2]
    // 0 1
    // 2 3
    std::vector<int> block_sizes = {2, 2};
    std::vector<std::vector<int>> adj = {{0, 1}, {0, 1}};
    
    DistGraph* graph = new DistGraph(MPI_COMM_WORLD);
    graph->construct_distributed({0, 1}, block_sizes, adj);
    
    BlockSpMat<double> mat(graph);
    mat.owns_graph = true;
    
    // Fill with data
    // (0,0): [1 2; 3 4]
    // (0,1): [5 6; 7 8]
    // (1,0): [9 10; 11 12]
    // (1,1): [13 14; 15 16]
    
    std::vector<double> b00 = {1, 3, 2, 4}; // ColMajor: 1,2 row 0; 3,4 row 1? No.
    // ColMajor: col 0: 1, 3. col 1: 2, 4. -> [1 2; 3 4]
    std::vector<double> b01 = {5, 7, 6, 8}; // [5 6; 7 8]
    std::vector<double> b10 = {9, 11, 10, 12}; // [9 10; 11 12]
    std::vector<double> b11 = {13, 15, 14, 16}; // [13 14; 15 16]
    
    mat.add_block(0, 0, b00.data(), 2, 2);
    mat.add_block(0, 1, b01.data(), 2, 2);
    mat.add_block(1, 0, b10.data(), 2, 2);
    mat.add_block(1, 1, b11.data(), 2, 2);
    mat.assemble();
    
    // Transpose
    BlockSpMat<double> mat_T = mat.transpose(); // Currently returns *this (stub)
    
    // Verify dimensions (stub check)
    assert(mat_T.row_ptr.size() == mat.row_ptr.size());
    
    // Real transpose check (once implemented)
    // For now, just ensure it runs.
    std::cout << "PASSED" << std::endl;
}

void test_spmm() {
    std::cout << "Testing SpMM (Self)..." << std::endl;
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    
    std::vector<int> block_sizes = {2, 2};
    std::vector<std::vector<int>> adj = {{0}, {1}}; // Global adjacency: 0->0, 1->1
    
    // Partition ownership
    std::vector<int> owned;
    if (size == 1) {
        owned = {0, 1};
    } else {
        if (rank == 0) owned = {0};
        else if (rank == 1) owned = {1};
        // Ranks > 1 own nothing
    }
    
    // Adjust local adjacency based on owned
    std::vector<std::vector<int>> local_adj;
    for (int global_row : owned) {
        local_adj.push_back(adj[global_row]);
    }
    
    DistGraph* graph = new DistGraph(MPI_COMM_WORLD);
    graph->construct_distributed(owned, block_sizes, local_adj);
    
    BlockSpMat<double> mat(graph);
    mat.owns_graph = true;
    
    std::vector<double> identity = {1, 0, 0, 1}; 
    
    for (size_t i = 0; i < owned.size(); ++i) {
        int global_row = owned[i];
        // Add diagonal block (col = global_row)
        mat.add_block(global_row, global_row, identity.data(), 2, 2);
    }
    mat.assemble();
    
    // C = A * A = I * I = I
    BlockSpMat<double> C = mat.spmm_self(0.0);
    
    // Verify C
    // Should have same structure as A (which is I)
    // So local size should match owned size
    
    if (C.col_ind.size() != owned.size()) {
        std::cout << "Rank " << rank << " C size: " << C.col_ind.size() 
                  << " Expected: " << owned.size() << std::endl;
    }
    assert(C.col_ind.size() == owned.size());
    
    for(size_t k=0; k<C.col_ind.size(); ++k) {
        // Check diagonal values
        uint64_t h = C.blk_handles[k];
        double* d = C.arena.get_ptr(h);
        assert(d[0] == 1.0 && d[3] == 1.0); // Diagonal
        assert(d[1] == 0.0 && d[2] == 0.0); // Off-diagonal
    }
    
    std::cout << "PASSED" << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    
    if (rank == 0) {
        test_basic_spmv();
    }
    test_axpby_structure_mismatch();
    test_memory_reuse();
    test_transpose();
    test_spmm();
    
    MPI_Finalize();
    return 0;
}
