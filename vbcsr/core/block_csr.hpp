#ifndef VBCSR_BLOCK_CSR_HPP
#define VBCSR_BLOCK_CSR_HPP

#include "dist_graph.hpp"
#include "dist_vector.hpp"
#include "dist_multivector.hpp"
#include "kernels.hpp"
#include <xmmintrin.h>
#include <vector>
#include <omp.h>
#include <fstream>
#include <iomanip>
#include <complex>
#include <type_traits>
#include <set>
#include <map>
#include <cstring>

namespace vbcsr {

enum class AssemblyMode {
    INSERT,
    ADD
};

enum class MatrixLayout {
    RowMajor,
    ColMajor
};

// Helper for Matrix Market output
template<typename T>
struct MMWriter {
    static void write(std::ostream& os, const T& v) {
        os << v;
    }
    static bool is_complex() { return false; }
};

template<typename T>
struct MMWriter<std::complex<T>> {
    static void write(std::ostream& os, const std::complex<T>& v) {
        os << v.real() << " " << v.imag();
    }
    static bool is_complex() { return true; }
};

struct BlockMeta {
    int col;
    double norm;
};

struct BlockID {
    int row;
    int col;
    bool operator<(const BlockID& other) const {
        if (row != other.row) return row < other.row;
        return col < other.col;
    }
};

template <typename T, typename Kernel = DefaultKernel<T>>
class BlockSpMat {
public:
    DistGraph* graph;
    using KernelType = Kernel;
    
    // CSR structure (local indices)
    std::vector<int> row_ptr;
    std::vector<int> col_ind;
    
    // Values (flattened blocks)
    std::vector<T> val;
    
    // Pointers to block starts in val
    std::vector<size_t> blk_ptr;
    


    bool owns_graph = false;

    // Helper for squared norm
    static double get_sq_norm(const T& v) {
        if constexpr (std::is_same<T, std::complex<double>>::value || std::is_same<T, std::complex<float>>::value) {
            return std::norm(v);
        } else {
            return v * v;
        }
    }

    // Helper to compute Frobenius norms of local blocks
    std::vector<double> compute_block_norms() const {
        int nnz = col_ind.size();
        std::vector<double> norms(nnz);
        
        #pragma omp parallel for
        for (int i = 0; i < nnz; ++i) {
            double sum = 0.0;
            size_t start = blk_ptr[i];
            size_t end = blk_ptr[i+1];
            for (size_t k = start; k < end; ++k) {
                sum += get_sq_norm(val[k]);
            }
            norms[i] = std::sqrt(sum);
        }
        return norms;
    }

public:
    BlockSpMat(DistGraph* g) : graph(g) {
        allocate_from_graph();
    }

    ~BlockSpMat() {
        if (owns_graph && graph) {
            delete graph;
        }
    }

    // Move constructor
    BlockSpMat(BlockSpMat&& other) noexcept : 
        graph(other.graph), 
        row_ptr(std::move(other.row_ptr)),
        col_ind(std::move(other.col_ind)),
        val(std::move(other.val)),
        blk_ptr(std::move(other.blk_ptr)),
        remote_blocks(std::move(other.remote_blocks)),
        owns_graph(other.owns_graph) 
    {
        other.graph = nullptr;
        other.owns_graph = false;
    }

    // Move assignment
    BlockSpMat& operator=(BlockSpMat&& other) noexcept {
        if (this != &other) {
            if (owns_graph && graph) delete graph;
            graph = other.graph;
            row_ptr = std::move(other.row_ptr);
            col_ind = std::move(other.col_ind);
            val = std::move(other.val);
            blk_ptr = std::move(other.blk_ptr);
            remote_blocks = std::move(other.remote_blocks);
            owns_graph = other.owns_graph;
            other.graph = nullptr;
            other.owns_graph = false;
        }
        return *this;
    }

    // Disable copy (use duplicate() instead)
    BlockSpMat(const BlockSpMat&) = delete;
    BlockSpMat& operator=(const BlockSpMat&) = delete;

    // Create a deep copy of the matrix
    BlockSpMat<T, Kernel> duplicate(bool independent_graph = true) const {
        DistGraph* new_graph = graph;
        bool new_owns_graph = false;
        if (independent_graph && graph) {
            new_graph = graph->duplicate();
            new_owns_graph = true;
        }
        BlockSpMat<T, Kernel> new_mat(new_graph);
        new_mat.owns_graph = new_owns_graph;
        new_mat.copy_from(*this);
        return new_mat;
    }

    void allocate_from_graph() {
        // 1. Get CSR structure
        graph->get_matrix_structure(row_ptr, col_ind);
        
        // 2. Cache dimensions - REMOVED, use graph->block_sizes directly
        // row_dims = graph->block_sizes (prefix)
        // col_dims = graph->block_sizes (all)
        
        // 2.1 Compute offsets - REMOVED, use graph->block_offsets directly
        // y_offsets = graph->block_offsets (prefix)
        // x_offsets = graph->block_offsets (all)
        
        // 3. Calculate value size and block pointers
        int nnz = col_ind.size();
        blk_ptr.resize(nnz + 1);
        blk_ptr[0] = 0;
        
        int n_owned = graph->owned_global_indices.size();
        for (int i = 0; i < n_owned; ++i) {

            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            for (int k = start; k < end; ++k) {
                int col = col_ind[k];
                int r_dim = graph->block_sizes[i];
                int c_dim = graph->block_sizes[col];
                blk_ptr[k+1] = blk_ptr[k] + r_dim * c_dim;
            }
        }
        
        val.resize(blk_ptr[nnz], T(0));
    }

    // Buffer for remote assembly
    struct PendingBlock {
        int g_row;
        int g_col;
        int rows;
        int cols;
        AssemblyMode mode;
        std::vector<T> data;
    };
    // Map: Owner -> (GlobalRow, GlobalCol) -> PendingBlock
    std::map<int, std::map<std::pair<int, int>, PendingBlock>> remote_blocks;

    // Add a block (local or remote)
    // Input data layout is specified by `layout`
    void add_block(int global_row, int global_col, const T* data, int rows, int cols, AssemblyMode mode = AssemblyMode::ADD, MatrixLayout layout = MatrixLayout::ColMajor) {
        int owner = graph->find_owner(global_row);
        
        if (owner == graph->rank) {
            // Local: Try to update immediately
            if (graph->global_to_local.find(global_row) != graph->global_to_local.end()) {
                int local_row = graph->global_to_local.at(global_row);
                
                int local_col = -1;
                if (graph->global_to_local.count(global_col)) {
                    local_col = graph->global_to_local.at(global_col);
                }
                
                if (local_col != -1) {
                    // update_local_block handles layout
                    if (update_local_block(local_row, local_col, data, rows, cols, mode, layout)) {
                        return; // Success
                    }
                }
            }
            throw std::runtime_error("Block (row=" + std::to_string(global_row) + ", col=" + std::to_string(global_col) + ") not found in local graph");
        } 
        
        // Remote
        auto& blocks_map = remote_blocks[owner];
        std::pair<int, int> key = {global_row, global_col};
        auto it = blocks_map.find(key);
        
        if (it != blocks_map.end()) {
            // Block exists in buffer
            PendingBlock& pb = it->second;
            
            // Check dims
            if (pb.rows != rows || pb.cols != cols) {
                throw std::runtime_error("Dimension mismatch in add_block accumulation");
            }
            
            // We store pending blocks in ColMajor (canonical format for transport)
            
            if (mode == AssemblyMode::INSERT) {
                // Overwrite
                pb.mode = AssemblyMode::INSERT;
                if (layout == MatrixLayout::ColMajor) {
                    std::memcpy(pb.data.data(), data, rows * cols * sizeof(T));
                } else {
                    // Transpose RowMajor -> ColMajor
                    for (int i = 0; i < rows; ++i) {
                        for (int j = 0; j < cols; ++j) {
                            pb.data[j * rows + i] = data[i * cols + j];
                        }
                    }
                }
            } else {
                // ADD
                // Accumulate
                if (layout == MatrixLayout::ColMajor) {
                    for (size_t i = 0; i < pb.data.size(); ++i) {
                        pb.data[i] += data[i];
                    }
                } else {
                    // Transpose add
                    for (int i = 0; i < rows; ++i) {
                        for (int j = 0; j < cols; ++j) {
                            pb.data[j * rows + i] += data[i * cols + j];
                        }
                    }
                }
            }
        } else {
            // New block
            PendingBlock pb;
            pb.g_row = global_row;
            pb.g_col = global_col;
            pb.rows = rows;
            pb.cols = cols;
            pb.mode = mode;
            pb.data.resize(rows * cols);
            
            if (layout == MatrixLayout::ColMajor) {
                std::memcpy(pb.data.data(), data, rows * cols * sizeof(T));
            } else {
                // Transpose RowMajor -> ColMajor
                for (int i = 0; i < rows; ++i) {
                    for (int j = 0; j < cols; ++j) {
                        pb.data[j * rows + i] = data[i * cols + j];
                    }
                }
            }
            blocks_map[key] = std::move(pb);
        }
    }

    // Finalize assembly by exchanging remote blocks
    void assemble() {
        int size = graph->size;
        
        // 1. Pack buffers
        // Protocol: g_row(int), g_col(int), rows(int), cols(int), mode(int), data(T...)
        // Data is always ColMajor in the protocol
        std::vector<std::vector<char>> send_bufs(size);
        
        for (auto& kv : remote_blocks) {
            int target = kv.first;
            auto& blocks_map = kv.second;
            for (auto& inner_kv : blocks_map) {
                auto& blk = inner_kv.second;
                size_t current_size = send_bufs[target].size();
                size_t data_bytes = blk.data.size() * sizeof(T);
                size_t packet_size = 4 * sizeof(int) + sizeof(int) + data_bytes;
                
                send_bufs[target].resize(current_size + packet_size);
                char* ptr = send_bufs[target].data() + current_size;
                
                std::memcpy(ptr, &blk.g_row, sizeof(int)); ptr += sizeof(int);
                std::memcpy(ptr, &blk.g_col, sizeof(int)); ptr += sizeof(int);
                std::memcpy(ptr, &blk.rows, sizeof(int)); ptr += sizeof(int);
                std::memcpy(ptr, &blk.cols, sizeof(int)); ptr += sizeof(int);
                int mode_int = static_cast<int>(blk.mode);
                std::memcpy(ptr, &mode_int, sizeof(int)); ptr += sizeof(int);
                std::memcpy(ptr, blk.data.data(), data_bytes);
            }
        }
        
        // 2. Exchange
        std::vector<int> send_counts(size);
        for(int i=0; i<size; ++i) send_counts[i] = send_bufs[i].size();
        
        std::vector<int> recv_counts(size);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, graph->comm);
        
        std::vector<int> sdispls(size + 1, 0), rdispls(size + 1, 0);
        for(int i=0; i<size; ++i) {
            sdispls[i+1] = sdispls[i] + send_counts[i];
            rdispls[i+1] = rdispls[i] + recv_counts[i];
        }
        
        std::vector<char> send_blob(sdispls[size]);
        for(int i=0; i<size; ++i) {
            std::memcpy(send_blob.data() + sdispls[i], send_bufs[i].data(), send_counts[i]);
        }
        
        std::vector<char> recv_blob(rdispls[size]);
        MPI_Alltoallv(send_blob.data(), send_counts.data(), sdispls.data(), MPI_BYTE,
                      recv_blob.data(), recv_counts.data(), rdispls.data(), MPI_BYTE, graph->comm);
                      
        // 3. Process received
        for(int i=0; i<size; ++i) {
            char* ptr = recv_blob.data() + rdispls[i];
            char* end = recv_blob.data() + rdispls[i+1];
            
            while(ptr < end) {
                int g_row, g_col, rows, cols, mode_int;
                std::memcpy(&g_row, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&g_col, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&rows, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&cols, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&mode_int, ptr, sizeof(int)); ptr += sizeof(int);
                AssemblyMode mode = static_cast<AssemblyMode>(mode_int);
                
                if (graph->global_to_local.find(g_row) == graph->global_to_local.end()) {
                     throw std::runtime_error("Received block for non-owned row");
                }
                int l_row = graph->global_to_local[g_row];
                
                if (graph->global_to_local.find(g_col) == graph->global_to_local.end()) {
                     throw std::runtime_error("Received block for unknown col");
                }
                int l_col = graph->global_to_local[g_col];
                
                size_t data_bytes = rows * cols * sizeof(T);
                
                // Update local block
                // Received data is ColMajor (from protocol)
                if (!update_local_block(l_row, l_col, (const T*)ptr, rows, cols, mode, MatrixLayout::ColMajor)) {
                    throw std::runtime_error("Received block not in graph");
                }
                
                ptr += data_bytes;
            }
        }
        
        remote_blocks.clear();
    }

    // Helper to update local block
    // Input data layout is specified by `layout`
    // Internal storage is ColMajor
    bool update_local_block(int local_row, int local_col, const T* data, int rows, int cols, AssemblyMode mode, MatrixLayout layout = MatrixLayout::ColMajor) {
        int start = row_ptr[local_row];
        int end = row_ptr[local_row+1];
        
        // optimizable
        for (int k = start; k < end; ++k) {
            if (col_ind[k] == local_col) {
                size_t offset = blk_ptr[k];
                // size_t size = blk_ptr[k+1] - offset; // This is total elements
                T* target = val.data() + offset;
                
                // Check dims
                
                int r_dim = graph->block_sizes[local_row];
                int c_dim = graph->block_sizes[local_col];

                if (r_dim != rows || c_dim != cols) {
                    std::stringstream ss;
                    ss << "\n Dimension mismatch in update_local_block (DEBUG): "
                       << "Row: " << local_row << " (Expected: " << r_dim << ", Got: " << rows << ") \n"
                       << "Col: " << local_col << " (Expected: " << c_dim << ", Got: " << cols << ") \n";
                    throw std::runtime_error(ss.str());
                }

                if (mode == AssemblyMode::INSERT) {
                    if (layout == MatrixLayout::ColMajor) {
                        // Direct copy: Input (ColMajor) -> Internal (ColMajor)
                        std::memcpy(target, data, r_dim * c_dim * sizeof(T));
                    } else {
                        // Transpose copy: Input (RowMajor) -> Internal (ColMajor)
                        // Internal[j*r_dim + i] = Input[i*c_dim + j]
                        for (int i = 0; i < r_dim; ++i) {
                            for (int j = 0; j < c_dim; ++j) {
                                target[j * r_dim + i] = data[i * c_dim + j];
                            }
                        }
                    }
                } else {
                    // ADD
                    if (layout == MatrixLayout::ColMajor) {
                        // Direct add
                         for (int i = 0; i < r_dim * c_dim; ++i) {
                            target[i] += data[i];
                        }
                    } else {
                        // Transpose copy and add: Internal[j*r_dim + i] += Input[i*c_dim + j]
                        for (int i = 0; i < r_dim; ++i) {
                            for (int j = 0; j < c_dim; ++j) {
                                target[j * r_dim + i] += data[i * c_dim + j];
                            }
                        }
                    }
                }
                return true;
            }
        }
        return false;
    }

    // Legacy setter (defaults to INSERT)
    void set_local_block(int local_row, int local_col, const T* data) {
        if (!update_local_block(local_row, local_col, data, AssemblyMode::INSERT)) {
            throw std::runtime_error("Block not found in graph structure");
        }
    }

    // Matrix-Vector Multiplication
    void mult(DistVector<T>& x, DistVector<T>& y) {
        mult_optimized(x, y);
    }
    
    // Refined mult with offsets
    void mult_optimized(DistVector<T>& x, DistVector<T>& y) {
        x.bind_to_graph(graph);
        y.bind_to_graph(graph);
        x.sync_ghosts();
        
        int n_rows = row_ptr.size() - 1;
        
        // Use precomputed offsets
        // y_offsets and x_offsets are members

        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            int r_dim = graph->block_sizes[i];
            T* y_val = y.local_data() + graph->block_offsets[i];
            
            // Initialize y_val to 0? Or assume y is zeroed?
            // Usually mult implies y = A*x. So overwrite.
            std::memset(y_val, 0, r_dim * sizeof(T));
            
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            for (int k = start; k < end; ++k) {

                // remove to recover org Kernel switch
                if (k + 1 < end) {
                    _mm_prefetch((const char*)(val.data() + blk_ptr[k+1]), _MM_HINT_T0);
                    int next_col = col_ind[k+1];
                    _mm_prefetch((const char*)(x.data.data() + graph->block_offsets[next_col]), _MM_HINT_T0);
                }
                // remove to recover org Kernel switch

                int col = col_ind[k];
                int c_dim = graph->block_sizes[col];
                const T* block_val = val.data() + blk_ptr[k];
                const T* x_val = x.data.data() + graph->block_offsets[col]; // x.data includes ghosts
                
                // y_block += A_block * x_block
                SmartKernel<T>::gemv(r_dim, c_dim, T(1), block_val, r_dim, x_val, 1, T(1), y_val, 1);
                // Kernel::gemv(r_dim, c_dim, T(1), block_val, r_dim, x_val, 1, T(1), y_val, 1);
            }
        }
    }

    // Matrix-Matrix Multiplication (Dense RHS)
    void mult_dense(DistMultiVector<T>& X, DistMultiVector<T>& Y) {
        X.bind_to_graph(graph);
        Y.bind_to_graph(graph);
        X.sync_ghosts();
        
        int n_rows = row_ptr.size() - 1;
        int num_vecs = X.num_vectors;
        
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            int r_dim = graph->block_sizes[i];
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            T* y_ptr = &Y(graph->block_offsets[i], 0);
            int ldc = Y.local_rows + Y.ghost_rows;
            
            bool first = true;
            for (int k = start; k < end; ++k) {
                if (k + 1 < end) {
                    _mm_prefetch((const char*)(val.data() + blk_ptr[k+1]), _MM_HINT_T0);
                    int next_col = col_ind[k+1];
                    _mm_prefetch((const char*)(&X(graph->block_offsets[next_col], 0)), _MM_HINT_T0);
                }
                int col = col_ind[k];
                int c_dim = graph->block_sizes[col];
                const T* block_val = val.data() + blk_ptr[k];
                const T* x_ptr = &X(graph->block_offsets[col], 0);
                int ldb = X.local_rows + X.ghost_rows;
                
                T beta = first ? T(0) : T(1);
                SmartKernel<T>::gemm(r_dim, num_vecs, c_dim, T(1), block_val, r_dim, x_ptr, ldb, beta, y_ptr, ldc);
                // Kernel::gemm(r_dim, num_vecs, c_dim, T(1), block_val, r_dim, x_ptr, ldb, beta, y_ptr, ldc);
                first = false;
            }
            
            if (first) {
                for (int v = 0; v < num_vecs; ++v) {
                    for (int r = 0; r < r_dim; ++r) {
                        y_ptr[v * ldc + r] = T(0);
                    }
                }
            }
        }
    }

    // Adjoint Matrix-Vector Multiplication: y = A^dagger * x
    void mult_adjoint(DistVector<T>& x, DistVector<T>& y) {
        x.bind_to_graph(graph);
        y.bind_to_graph(graph);
        
        // Initialize y to 0 (including ghosts for accumulation)
        std::fill(y.data.begin(), y.data.end(), T(0));
        
        int n_rows = row_ptr.size() - 1;
        
        // We can't easily parallelize over rows because multiple rows contribute to the same y_j.
        // We need a thread-local y or atomic adds.
        // For simplicity and correctness, let's use a critical section or atomic if T supports it.
        // Or parallelize over rows but use a temporary per-thread y.
        
        #pragma omp parallel
        {
            std::vector<T> y_local(y.data.size(), T(0));
            
            #pragma omp for
            for (int i = 0; i < n_rows; ++i) {
                int r_dim = graph->block_sizes[i];
                const T* x_val = x.local_data() + graph->block_offsets[i];
                
                int start = row_ptr[i];
                int end = row_ptr[i+1];
                
                for (int k = start; k < end; ++k) {
                    int col = col_ind[k];
                    int c_dim = graph->block_sizes[col];
                    const T* block_val = val.data() + blk_ptr[k];
                    T* y_target = y_local.data() + graph->block_offsets[col];
                    
                    if (k + 1 < end) {
                        _mm_prefetch((const char*)(val.data() + blk_ptr[k+1]), _MM_HINT_T0);
                        int next_col = col_ind[k+1];
                        _mm_prefetch((const char*)(y_local.data() + graph->block_offsets[next_col]), _MM_HINT_T0);
                    }
                    
                    // y_target += A_block^dagger * x_block
                    SmartKernel<T>::gemv_trans(r_dim, c_dim, T(1), block_val, r_dim, x_val, 1, T(1), y_target, 1);
                    // Kernel::gemv(r_dim, c_dim, T(1), block_val, r_dim, x_val, 1, T(1), y_target, 1, CblasConjTrans);
                }
            }
            
            #pragma omp critical
            {
                for (size_t i = 0; i < y.data.size(); ++i) y.data[i] += y_local[i];
            }
        }
        
        y.reduce_ghosts();
    }

    // Adjoint Matrix-Matrix Multiplication: Y = A^dagger * X
    void mult_dense_adjoint(DistMultiVector<T>& X, DistMultiVector<T>& Y) {
        X.bind_to_graph(graph);
        Y.bind_to_graph(graph);
        
        std::fill(Y.data.begin(), Y.data.end(), T(0));
        
        int n_rows = row_ptr.size() - 1;
        int num_vecs = X.num_vectors;
        
        #pragma omp parallel
        {
            std::vector<T> Y_local(Y.data.size(), T(0));
            int ldc_local = Y.local_rows + Y.ghost_rows;
            
            #pragma omp for
            for (int i = 0; i < n_rows; ++i) {
                int r_dim = graph->block_sizes[i];
                const T* x_ptr = &X(graph->block_offsets[i], 0);
                int ldb = X.local_rows + X.ghost_rows;
                
                int start = row_ptr[i];
                int end = row_ptr[i+1];
                
                for (int k = start; k < end; ++k) {
                    int col = col_ind[k];
                    int c_dim = graph->block_sizes[col];
                    const T* block_val = val.data() + blk_ptr[k];
                    T* y_ptr = &Y_local[col]; // This is wrong for column-major
                    // Correct pointer:
                    T* y_target = &Y_local[graph->block_offsets[col]]; 
                    
                    // Y_target += A_block^dagger * X_block
                    // A_block is r_dim x c_dim. A_block^dagger is c_dim x r_dim.
                    // X_block is r_dim x num_vecs.
                    // Y_target is c_dim x num_vecs.
                    
                    if (k + 1 < end) {
                        _mm_prefetch((const char*)(val.data() + blk_ptr[k+1]), _MM_HINT_T0);
                        int next_col = col_ind[k+1];
                        _mm_prefetch((const char*)(Y_local.data() + graph->block_offsets[next_col]), _MM_HINT_T0);
                    }
                    
                    SmartKernel<T>::gemm_trans(c_dim, num_vecs, r_dim, T(1), block_val, r_dim, x_ptr, ldb, T(1), y_target, ldc_local);
                    // Kernel::gemm(c_dim, num_vecs, r_dim, T(1), block_val, r_dim, x_ptr, ldb, T(1), y_target, ldc_local, CblasConjTrans, CblasNoTrans);
                }
            }
            
            #pragma omp critical
            {
                for (size_t i = 0; i < Y.data.size(); ++i) Y.data[i] += Y_local[i];
            }
        }
        
        Y.reduce_ghosts();
    }

    // Utilities
    void scale(T alpha) {
        #pragma omp parallel for
        for (size_t i = 0; i < val.size(); ++i) {
            val[i] *= alpha;
        }
    }

    void copy_from(const BlockSpMat<T, Kernel>& other) {
        // If graphs are different, we should at least check compatibility
        if (graph != other.graph) {
            // Check if owned indices and block sizes match
            if (graph->owned_global_indices != other.graph->owned_global_indices ||
                graph->block_sizes != other.graph->block_sizes) {
                 throw std::runtime_error("Incompatible graph structure in copy_from");
            }
        }

        if (val.size() != other.val.size()) {
            throw std::runtime_error("Matrix value size mismatch in copy_from");
        }
        std::memcpy(val.data(), other.val.data(), val.size() * sizeof(T));
    }

    void axpy(T alpha, const BlockSpMat<T, Kernel>& other) {
        if (val.size() != other.val.size()) {
            throw std::runtime_error("Matrix structure mismatch in axpy");
        }
        #pragma omp parallel for
        for (size_t i = 0; i < val.size(); ++i) {
            val[i] += alpha * other.val[i];
        }
    }

    // Add alpha to diagonal elements
    void shift(T alpha) {
        int n_rows = row_ptr.size() - 1;
        
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            // Find diagonal block (col == local_col corresponding to local_row i)
            // Local row i corresponds to global row G = graph->get_global_index(i)
            
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            int global_row = graph->get_global_index(i);
            
            for (int k = start; k < end; ++k) {
                int local_col = col_ind[k];
                // Check if this local_col maps to global_row
                
                if (graph->get_global_index(local_col) == global_row) {
                    // Found diagonal block
                    size_t offset = blk_ptr[k];
                    int r_dim = graph->block_sizes[i];
                    int c_dim = graph->block_sizes[local_col];
                    
                    // Add alpha to diagonal of the block
                    // Block is ColMajor or RowMajor? Internal is ColMajor.
                    // Diagonal elements are at [j*r_dim + j] for j=0..min(r,c)
                    
                    int min_dim = std::min(r_dim, c_dim);
                    for (int j = 0; j < min_dim; ++j) {
                        val[offset + j * r_dim + j] += alpha;
                    }
                    break; 
                }
            }
        }
    }

    // Add vector elements to diagonal: H_ii += v_i
    void add_diagonal(const DistVector<T>& diag) {
        int n_rows = row_ptr.size() - 1;
        // diag must have at least local_size elements
        if (diag.size() < n_rows) {
            throw std::runtime_error("Vector size too small for add_diagonal");
        }

        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            int global_row = graph->get_global_index(i);
            T v_val = diag[i]; // Local index i corresponds to owned part of diag
            
            for (int k = start; k < end; ++k) {
                int local_col = col_ind[k];
                if (graph->get_global_index(local_col) == global_row) {
                    // Found diagonal block
                    size_t offset = blk_ptr[k];
                    int r_dim = graph->block_sizes[i];
                    int c_dim = graph->block_sizes[local_col];
                    
                    int min_dim = std::min(r_dim, c_dim);
                    for (int j = 0; j < min_dim; ++j) {
                        val[offset + j * r_dim + j] += v_val;
                    }
                    break; 
                }
            }
        }
    }

    // Compute C = [H, R] where R is diagonal (stored as DistVector)
    // C_ij = H_ij * (R_j - R_i)
    // Result C has same structure as H (this).
    void commutator_diagonal(const DistVector<T>& diag, BlockSpMat<T, Kernel>& result) {
        // Ensure result has same structure
        if (result.val.size() != val.size()) {
            result.allocate_from_graph(); // Or throw
        }
        
        const std::vector<T>& R = diag.data; // Includes ghosts
        
        int n_rows = row_ptr.size() - 1;
        
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            T R_i = R[i]; // Local row i corresponds to R[i] (owned part)
            
            for (int k = start; k < end; ++k) {
                int col = col_ind[k];
                T R_j = R[col]; // col is local index (owned or ghost), maps directly to R vector
                
                T diff = R_j - R_i;
                
                size_t offset = blk_ptr[k];
                int r_dim = graph->block_sizes[i];
                int c_dim = graph->block_sizes[col];
                int block_size = r_dim * c_dim;
                
                const T* H_ptr = val.data() + offset;
                T* C_ptr = result.val.data() + offset;
                
                for (int b = 0; b < block_size; ++b) {
                    C_ptr[b] = H_ptr[b] * diff;
                }
            }
        }
    }

    // Map: Global Row of B -> List of (Global Col of B, Norm)

    BlockSpMat spmm(const BlockSpMat& B, double threshold, bool transA = false, bool transB = false) const {
        if (transA) {
            BlockSpMat A_T = this->transpose();
            return A_T.spmm(B, threshold, false, transB);
        }
        if (transB) {
            BlockSpMat B_T = B.transpose();
            return this->spmm(B_T, threshold, transA, false);
        }
        
        // 1. Metadata Exchange
        GhostMetadata meta = exchange_ghost_metadata(B);
        
        // 2. Symbolic Phase (Structure Prediction)
        SymbolicResult sym = symbolic_multiply_filtered(B, meta, threshold);
        
        // 3. Data Exchange (Fetch Ghost Blocks)
        auto [ghost_data_map, ghost_sizes] = B.fetch_ghost_blocks(sym.required_blocks);
        
        // Reorganize ghost data for fast row access
        std::map<int, std::vector<GhostBlockRef>> ghost_rows;
        for (const auto& [bid, data] : ghost_data_map) {
            int c_dim = ghost_sizes.at(bid.col);
            ghost_rows[bid.row].push_back({bid.col, data.data(), c_dim});
        }
        
        // 4. Construct Result Matrix Structure
        std::vector<std::vector<int>> adj(graph->owned_global_indices.size());
        int n_rows = row_ptr.size() - 1;
        for(int i=0; i<n_rows; ++i) {
            int start = sym.c_row_ptr[i];
            int end = sym.c_row_ptr[i+1];
            for(int k=start; k<end; ++k) {
                adj[i].push_back(sym.c_col_ind[k]);
            }
        }
        
        DistGraph* c_graph = new DistGraph(graph->comm);
        c_graph->construct_distributed(graph->owned_global_indices, graph->block_sizes, adj);
        
        BlockSpMat C(c_graph);
        C.owns_graph = true;
        std::fill(C.val.begin(), C.val.end(), T(0));
        
        // 5. Numeric Phase (GEMM)
        numeric_multiply(B, ghost_rows, C);
        
        return C;
    }

    BlockSpMat spmm_self(double threshold, bool transA = false) {
        return spmm(*this, threshold, transA, false);
    }

    BlockSpMat add(const BlockSpMat& B, double alpha = 1.0, double beta = 1.0) {
        if (graph != B.graph) {
             throw std::runtime_error("General addition with different graphs not yet implemented");
        }
        BlockSpMat C = this->duplicate();
        C.scale(alpha);
        C.axpy(beta, B);
        return C;
    }

    BlockSpMat transpose() const {
        // 1. Count blocks to send to each rank
        int size = graph->size;
        int rank = graph->rank;
        
        std::vector<std::vector<int>> send_reqs(size); // row, col, r_dim, c_dim
        std::vector<std::vector<T>> send_data(size);
        
        int n_rows = row_ptr.size() - 1;
        for (int i = 0; i < n_rows; ++i) {
            int g_row = graph->owned_global_indices[i];
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            for (int k = start; k < end; ++k) {
                int g_col = graph->get_global_index(col_ind[k]);
                int owner = graph->find_owner(g_col);
                
                // We are sending A_ij. Target is C_ji.
                // Target row is g_col. Target col is g_row.
                // Owner of g_col needs to receive this.
                
                int r_dim = graph->block_sizes[i];
                int c_dim = graph->block_sizes[col_ind[k]]; // local col index for size
                
                // Pack metadata: target_row (g_col), target_col (g_row), r_dim (of A_ij -> c_dim of C_ji), c_dim (of A_ij -> r_dim of C_ji)
                // Wait, A_ij is r_dim x c_dim.
                // C_ji is c_dim x r_dim.
                // We send A_ij data. Receiver will transpose it.
                
                send_reqs[owner].push_back(g_col);
                send_reqs[owner].push_back(g_row);
                send_reqs[owner].push_back(c_dim); // New r_dim
                send_reqs[owner].push_back(r_dim); // New c_dim
                
                size_t offset = blk_ptr[k];
                size_t count = r_dim * c_dim;
                const T* ptr = val.data() + offset;
                send_data[owner].insert(send_data[owner].end(), ptr, ptr + count);
            }
        }
        
        // 2. Exchange counts
        std::vector<int> send_counts(size);
        std::vector<int> send_data_counts(size);
        for(int i=0; i<size; ++i) {
            send_counts[i] = send_reqs[i].size();
            send_data_counts[i] = send_data[i].size();
        }
        
        std::vector<int> recv_counts(size);
        std::vector<int> recv_data_counts(size);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, graph->comm);
        MPI_Alltoall(send_data_counts.data(), 1, MPI_INT, recv_data_counts.data(), 1, MPI_INT, graph->comm);
        
        // 3. Exchange data
        std::vector<int> sdispls(size + 1, 0), rdispls(size + 1, 0);
        std::vector<int> sdispls_data(size + 1, 0), rdispls_data(size + 1, 0);
        
        for(int i=0; i<size; ++i) {
            sdispls[i+1] = sdispls[i] + send_counts[i];
            rdispls[i+1] = rdispls[i] + recv_counts[i];
            sdispls_data[i+1] = sdispls_data[i] + send_data_counts[i];
            rdispls_data[i+1] = rdispls_data[i] + recv_data_counts[i];
        }
        
        std::vector<int> send_buf(sdispls[size]);
        for(int i=0; i<size; ++i) {
            if (send_counts[i] > 0)
                std::memcpy(send_buf.data() + sdispls[i], send_reqs[i].data(), send_counts[i] * sizeof(int));
        }
        
        std::vector<int> recv_buf(rdispls[size]);
        MPI_Alltoallv(send_buf.data(), send_counts.data(), sdispls.data(), MPI_INT,
                      recv_buf.data(), recv_counts.data(), rdispls.data(), MPI_INT, graph->comm);
                      
        std::vector<T> send_val(sdispls_data[size]);
        for(int i=0; i<size; ++i) {
             if (send_data_counts[i] > 0)
                std::memcpy(send_val.data() + sdispls_data[i], send_data[i].data(), send_data_counts[i] * sizeof(T));
        }
        
        // Convert counts to bytes for MPI_BYTE
        for(int i=0; i<size; ++i) {
            send_data_counts[i] *= sizeof(T);
            recv_data_counts[i] *= sizeof(T);
            sdispls_data[i] *= sizeof(T);
            rdispls_data[i] *= sizeof(T);
        }
        sdispls_data[size] *= sizeof(T);
        rdispls_data[size] *= sizeof(T);
        
        std::vector<T> recv_val(rdispls_data[size]);
        MPI_Alltoallv(send_val.data(), send_data_counts.data(), sdispls_data.data(), MPI_BYTE,
                      recv_val.data(), recv_data_counts.data(), rdispls_data.data(), MPI_BYTE, graph->comm);
                      
        // 4. Construct C
        // Build adjacency for C
        std::vector<std::vector<int>> my_adj(graph->owned_global_indices.size());
        
        int* ptr = recv_buf.data();
        for(int i=0; i<size; ++i) {
            int count = recv_counts[i];
            int* end = ptr + count;
            while(ptr < end) {
                int g_row = *ptr++; // C's row (owned by me)
                int g_col = *ptr++; // C's col
                int r_dim = *ptr++;
                int c_dim = *ptr++;
                
                // Add to adjacency
                if (graph->global_to_local.count(g_row)) {
                    int l_row = graph->global_to_local.at(g_row);
                    if (l_row < (int)my_adj.size()) {
                        my_adj[l_row].push_back(g_col);
                    }
                }
            }
        }
        
        // Unique adjacency
        for(auto& neighbors : my_adj) {
            std::sort(neighbors.begin(), neighbors.end());
            neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
        }
        
        // Create new graph
        DistGraph* graph_C = new DistGraph(graph->comm);
        graph_C->construct_distributed(graph->owned_global_indices, graph->block_sizes, my_adj);
        
        BlockSpMat C(graph_C);
        C.owns_graph = true;
        
        // 5. Unpack and Insert
        ptr = recv_buf.data();
        T* val_ptr = recv_val.data();
        
        for(int i=0; i<size; ++i) {
            int count = recv_counts[i];
            int* end = ptr + count;
            
            while(ptr < end) {
                int g_row = *ptr++; // This is C's row (was A's col)
                int g_col = *ptr++; // This is C's col (was A's row)
                int r_dim = *ptr++; // C's r_dim
                int c_dim = *ptr++; // C's c_dim
                
                int n_elem = r_dim * c_dim;
                
                // Transpose the block data
                // A_ij was r_dim_A x c_dim_A (ColMajor)
                // We received it as is.
                // We want C_ji which is c_dim_A x r_dim_A (ColMajor).
                // C_ji(r, c) = A_ij(c, r).
                // A_ij flat index for (c, r) is c * r_dim_A + r? No, ColMajor: r + c * r_dim_A.
                // C_ji flat index for (r, c) is r + c * r_dim_C.
                // r_dim_C = c_dim_A.
                // So C_ji(r, c) = A_ij(c, r).
                
                std::vector<T> block(n_elem);
                // Transpose kernel
                // A_data is r_dim_A x c_dim_A (ColMajor).
                // We want C_data = A_data^T.
                // r_dim_A = c_dim (of C). c_dim_A = r_dim (of C).
                
                int rows_A = c_dim; 
                int cols_A = r_dim;
                
                for(int c=0; c<cols_A; ++c) {
                    for(int r=0; r<rows_A; ++r) {
                        // A(r,c) -> C(c,r)
                        // A index: r + c * rows_A
                        // C index: c + r * cols_A (which is c + r * r_dim)
                        
                        T val = val_ptr[r + c * rows_A];
                        val = ConjHelper<T>::apply(val);
                        block[c + r * r_dim] = val;
                    }
                }
                
                C.add_block(g_row, g_col, block.data(), r_dim, c_dim, AssemblyMode::ADD);
                
                val_ptr += n_elem;
            }
        }
        C.assemble();
        return C;
    }

    void save_matrix_market(const std::string& filename) {
        if (graph->size > 1) {
            throw std::runtime_error("save_matrix_market only supported for serial execution (MPI size = 1)");
        }

        std::ofstream file(filename);
        if (!file.is_open()) {
            throw std::runtime_error("Could not open file for writing: " + filename);
        }

        file << "%%MatrixMarket matrix coordinate " << (MMWriter<T>::is_complex() ? "complex" : "real") << " general\n";

        // Calculate total NNZ and dimensions
        size_t element_nnz = 0;
        int total_rows = 0;
        int total_cols = 0;
        
        int n_owned = row_ptr.size() - 1;
        
        // Calculate total rows
        for (int i = 0; i < n_owned; ++i) {
            total_rows += graph->block_sizes[i];
        }
        
        // Calculate total cols (assuming serial, so owned == all local cols)
        for (size_t i = 0; i < graph->block_sizes.size(); ++i) {
             total_cols += graph->block_sizes[i];
        }
        
        // Count NNZ
        for (int i = 0; i < n_owned; ++i) {
            int r_dim = graph->block_sizes[i];
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            for (int k = start; k < end; ++k) {
                int col = col_ind[k];
                int c_dim = graph->block_sizes[col];
                element_nnz += r_dim * c_dim;
            }
        }
        
        file << total_rows << " " << total_cols << " " << element_nnz << "\n";
        
        // Write data
        file << std::scientific << std::setprecision(16);
        
        for (int i = 0; i < n_owned; ++i) {
            int r_dim = graph->block_sizes[i];
            int row_start_idx = graph->block_offsets[i] + 1; // 1-based
            
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            for (int k = start; k < end; ++k) {
                int col = col_ind[k];
                int c_dim = graph->block_sizes[col];
                int col_start_idx = graph->block_offsets[col] + 1; // 1-based
                
                size_t offset = blk_ptr[k];
                const T* block_data = val.data() + offset;
                
                // Block is stored in ColMajor
                for (int c = 0; c < c_dim; ++c) {
                    for (int r = 0; r < r_dim; ++r) {
                        T value = block_data[c * r_dim + r];
                        
                        // Write (row, col, val)
                        file << (row_start_idx + r) << " " << (col_start_idx + c) << " ";
                        MMWriter<T>::write(file, value);
                        file << "\n";
                    }
                }
            }
        }
    }

private:
    struct GhostBlockRef { int col; const T* data; int c_dim; };
    using GhostBlockData = std::map<BlockID, std::vector<T>>;
    using GhostSizes = std::map<int, int>;

    using GhostMetadata = std::map<int, std::vector<BlockMeta>>;

    struct SymbolicResult {
        std::vector<int> c_row_ptr;
        std::vector<int> c_col_ind;
        std::vector<BlockID> required_blocks;
    };

    // SpMM Phase 1: Metadata Exchange
    GhostMetadata exchange_ghost_metadata(const BlockSpMat& B) const {
        GhostMetadata metadata;
        int size = graph->size;
        int rank = graph->rank;

        // 1. Identify needed ghost rows
        std::set<int> needed_rows;
        int n_rows = row_ptr.size() - 1;
        for (int i = 0; i < n_rows; ++i) {
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            for (int k = start; k < end; ++k) {
                int global_col = graph->get_global_index(col_ind[k]); // This is global row index in B
                if (graph->find_owner(global_col) != rank) {
                    needed_rows.insert(global_col);
                }
            }
        }

        // 2. Prepare requests
        std::vector<std::vector<int>> send_reqs(size);
        for (int global_row : needed_rows) {
            int owner = graph->find_owner(global_row);
            send_reqs[owner].push_back(global_row);
        }

        // 3. Exchange requests
        std::vector<int> send_counts(size);
        for(int i=0; i<size; ++i) send_counts[i] = send_reqs[i].size();
        std::vector<int> recv_counts(size);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, graph->comm);

        std::vector<int> sdispls(size + 1, 0), rdispls(size + 1, 0);
        for(int i=0; i<size; ++i) {
            sdispls[i+1] = sdispls[i] + send_counts[i];
            rdispls[i+1] = rdispls[i] + recv_counts[i];
        }

        std::vector<int> send_buf(sdispls[size]);
        for(int i=0; i<size; ++i) {
            std::memcpy(send_buf.data() + sdispls[i], send_reqs[i].data(), send_counts[i] * sizeof(int));
        }

        std::vector<int> recv_buf(rdispls[size]);
        MPI_Alltoallv(send_buf.data(), send_counts.data(), sdispls.data(), MPI_INT,
                      recv_buf.data(), recv_counts.data(), rdispls.data(), MPI_INT, graph->comm);

        // 4. Process requests and prepare replies
        std::vector<double> B_norms = B.compute_block_norms();
        std::vector<std::vector<char>> send_replies(size);

        for(int i=0; i<size; ++i) {
            int* ptr = recv_buf.data() + rdispls[i];
            int* end = recv_buf.data() + rdispls[i+1];
            while(ptr < end) {
                int global_row = *ptr++;
                if (graph->global_to_local.find(global_row) == graph->global_to_local.end()) {
                    continue; 
                }
                int local_row = graph->global_to_local.at(global_row);
                
                int start = B.row_ptr[local_row];
                int end_row = B.row_ptr[local_row+1];
                int n_blocks = end_row - start;
                
                size_t packet_size = 2 * sizeof(int) + n_blocks * (sizeof(int) + sizeof(double));
                size_t current_size = send_replies[i].size();
                send_replies[i].resize(current_size + packet_size);
                char* buf = send_replies[i].data() + current_size;
                
                std::memcpy(buf, &global_row, sizeof(int)); buf += sizeof(int);
                std::memcpy(buf, &n_blocks, sizeof(int)); buf += sizeof(int);
                
                for(int k=start; k<end_row; ++k) {
                    int col = B.graph->get_global_index(B.col_ind[k]);
                    double norm = B_norms[k];
                    std::memcpy(buf, &col, sizeof(int)); buf += sizeof(int);
                    std::memcpy(buf, &norm, sizeof(double)); buf += sizeof(double);
                }
            }
        }

        // 5. Exchange replies
        std::vector<int> send_bytes(size);
        for(int i=0; i<size; ++i) send_bytes[i] = send_replies[i].size();
        std::vector<int> recv_bytes(size);
        MPI_Alltoall(send_bytes.data(), 1, MPI_INT, recv_bytes.data(), 1, MPI_INT, graph->comm);
        
        std::vector<int> sdispls_bytes(size + 1, 0), rdispls_bytes(size + 1, 0);
        for(int i=0; i<size; ++i) {
            sdispls_bytes[i+1] = sdispls_bytes[i] + send_bytes[i];
            rdispls_bytes[i+1] = rdispls_bytes[i] + recv_bytes[i];
        }
        
        std::vector<char> send_blob(sdispls_bytes[size]);
        for(int i=0; i<size; ++i) {
             std::memcpy(send_blob.data() + sdispls_bytes[i], send_replies[i].data(), send_bytes[i]);
        }
        
        std::vector<char> recv_blob(rdispls_bytes[size]);
        MPI_Alltoallv(send_blob.data(), send_bytes.data(), sdispls_bytes.data(), MPI_BYTE,
                      recv_blob.data(), recv_bytes.data(), rdispls_bytes.data(), MPI_BYTE, graph->comm);

        // 6. Unpack replies
        for(int i=0; i<size; ++i) {
            char* ptr = recv_blob.data() + rdispls_bytes[i];
            char* end = recv_blob.data() + rdispls_bytes[i+1];
            while(ptr < end) {
                int global_row, n_blocks;
                std::memcpy(&global_row, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&n_blocks, ptr, sizeof(int)); ptr += sizeof(int);
                
                auto& list = metadata[global_row];
                list.reserve(n_blocks);
                
                for(int k=0; k<n_blocks; ++k) {
                    BlockMeta meta;
                    std::memcpy(&meta.col, ptr, sizeof(int)); ptr += sizeof(int);
                    std::memcpy(&meta.norm, ptr, sizeof(double)); ptr += sizeof(double);
                    list.push_back(meta);
                }
            }
        }
        
        return metadata;
    }

    // SpMM Phase 2: Symbolic Multiplication
    SymbolicResult symbolic_multiply_filtered(const BlockSpMat& B, const GhostMetadata& meta, double threshold) const {
        SymbolicResult res;
        int n_rows = row_ptr.size() - 1;
        res.c_row_ptr.resize(n_rows + 1);
        res.c_row_ptr[0] = 0;
        
        std::vector<double> A_norms = compute_block_norms();
        std::vector<double> B_local_norms = B.compute_block_norms();
        
        std::set<BlockID> required_set;
        
        for (int i = 0; i < n_rows; ++i) {
            std::map<int, double> c_row_est;
            
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            
            for (int k = start; k < end; ++k) {
                int global_col_A = graph->get_global_index(col_ind[k]);
                double norm_A = A_norms[k];
                
                if (graph->find_owner(global_col_A) == graph->rank) {
                    int local_row_B = graph->global_to_local.at(global_col_A);
                    int start_B = B.row_ptr[local_row_B];
                    int end_B = B.row_ptr[local_row_B+1];
                    for (int j = start_B; j < end_B; ++j) {
                        int global_col_B = B.graph->get_global_index(B.col_ind[j]);
                        double norm_B = B_local_norms[j];
                        c_row_est[global_col_B] += norm_A * norm_B;
                    }
                } else {
                    if (meta.count(global_col_A)) {
                        const auto& list = meta.at(global_col_A);
                        for (const auto& m : list) {
                            c_row_est[m.col] += norm_A * m.norm;
                        }
                    }
                }
            }
            
            for (auto const& [col, est] : c_row_est) {
                if (est > threshold) {
                    res.c_col_ind.push_back(col);
                }
            }
            res.c_row_ptr[i+1] = res.c_col_ind.size();
        }
        
        for (int i = 0; i < n_rows; ++i) {
            int c_start = res.c_row_ptr[i];
            int c_end = res.c_row_ptr[i+1];
            if (c_start == c_end) continue;
            
            std::set<int> active_cols;
            for(int idx=c_start; idx<c_end; ++idx) active_cols.insert(res.c_col_ind[idx]);
            
            int start = row_ptr[i];
            int end = row_ptr[i+1];
            for (int k = start; k < end; ++k) {
                int global_col_A = graph->get_global_index(col_ind[k]);
                
                if (graph->find_owner(global_col_A) != graph->rank) {
                    if (meta.count(global_col_A)) {
                        const auto& list = meta.at(global_col_A);
                        for (const auto& m : list) {
                            if (active_cols.count(m.col)) {
                                required_set.insert({global_col_A, m.col});
                            }
                        }
                    }
                }
            }
        }
        
        res.required_blocks.assign(required_set.begin(), required_set.end());
        return res;
    }

    // SpMM Phase 3: Fetch Ghost Blocks
    std::pair<GhostBlockData, GhostSizes> fetch_ghost_blocks(const std::vector<BlockID>& required_blocks) const {
        GhostBlockData ghost_data;
        GhostSizes ghost_sizes;
        int size = graph->size;
        int rank = graph->rank;
        
        // 1. Prepare requests
        std::vector<std::vector<int>> send_reqs(size);
        for (const auto& bid : required_blocks) {
            int owner = graph->find_owner(bid.row);
            send_reqs[owner].push_back(bid.row);
            send_reqs[owner].push_back(bid.col);
        }
        
        // 2. Exchange requests
        std::vector<int> send_counts(size);
        for(int i=0; i<size; ++i) send_counts[i] = send_reqs[i].size();
        std::vector<int> recv_counts(size);
        MPI_Alltoall(send_counts.data(), 1, MPI_INT, recv_counts.data(), 1, MPI_INT, graph->comm);
        
        std::vector<int> sdispls(size + 1, 0), rdispls(size + 1, 0);
        for(int i=0; i<size; ++i) {
            sdispls[i+1] = sdispls[i] + send_counts[i];
            rdispls[i+1] = rdispls[i] + recv_counts[i];
        }
        
        std::vector<int> send_buf(sdispls[size]);
        for(int i=0; i<size; ++i) {
            std::memcpy(send_buf.data() + sdispls[i], send_reqs[i].data(), send_counts[i] * sizeof(int));
        }
        
        std::vector<int> recv_buf(rdispls[size]);
        MPI_Alltoallv(send_buf.data(), send_counts.data(), sdispls.data(), MPI_INT,
                      recv_buf.data(), recv_counts.data(), rdispls.data(), MPI_INT, graph->comm);
                      
        // 3. Process requests and pack data
        std::vector<std::vector<char>> send_replies(size);
        for(int i=0; i<size; ++i) {
            int* ptr = recv_buf.data() + rdispls[i];
            int* end = recv_buf.data() + rdispls[i+1];
            while(ptr < end) {
                int g_row = *ptr++;
                int g_col = *ptr++;
                
                if (graph->global_to_local.count(g_row)) {
                    int l_row = graph->global_to_local.at(g_row);
                    int start = row_ptr[l_row];
                    int end_row = row_ptr[l_row+1];
                    for(int k=start; k<end_row; ++k) {
                        if (graph->get_global_index(col_ind[k]) == g_col) {
                                    size_t offset = blk_ptr[k];
                                    size_t next_offset = blk_ptr[k+1];
                                    size_t count = next_offset - offset;
                                    
                                    int r_dim = graph->block_sizes[l_row];
                                    int c_dim = graph->block_sizes[col_ind[k]];
                                    
                                    size_t current_size = send_replies[i].size();
                                    size_t packet_size = 2 * sizeof(int) + 2 * sizeof(int) + count * sizeof(T);
                                    send_replies[i].resize(current_size + packet_size);
                                    char* buf = send_replies[i].data() + current_size;
                                    
                                    std::memcpy(buf, &g_row, sizeof(int)); buf += sizeof(int);
                                    std::memcpy(buf, &g_col, sizeof(int)); buf += sizeof(int);
                                    std::memcpy(buf, &r_dim, sizeof(int)); buf += sizeof(int);
                                    std::memcpy(buf, &c_dim, sizeof(int)); buf += sizeof(int);
                                    std::memcpy(buf, val.data() + offset, count * sizeof(T));
                                    break;
                                }
                    }
                }
            }
        }
        
        // 4. Exchange replies
        std::vector<int> send_bytes(size);
        for(int i=0; i<size; ++i) send_bytes[i] = send_replies[i].size();
        std::vector<int> recv_bytes(size);
        MPI_Alltoall(send_bytes.data(), 1, MPI_INT, recv_bytes.data(), 1, MPI_INT, graph->comm);
        
        std::vector<int> sdispls_bytes(size + 1, 0), rdispls_bytes(size + 1, 0);
        for(int i=0; i<size; ++i) {
            sdispls_bytes[i+1] = sdispls_bytes[i] + send_bytes[i];
            rdispls_bytes[i+1] = rdispls_bytes[i] + recv_bytes[i];
        }
        
        std::vector<char> send_blob(sdispls_bytes[size]);
        for(int i=0; i<size; ++i) {
             std::memcpy(send_blob.data() + sdispls_bytes[i], send_replies[i].data(), send_bytes[i]);
        }
        
        std::vector<char> recv_blob(rdispls_bytes[size]);
        MPI_Alltoallv(send_blob.data(), send_bytes.data(), sdispls_bytes.data(), MPI_BYTE,
                      recv_blob.data(), recv_bytes.data(), rdispls_bytes.data(), MPI_BYTE, graph->comm);
                      
        // 5. Unpack
        for(int i=0; i<size; ++i) {
            char* ptr = recv_blob.data() + rdispls_bytes[i];
            char* end = recv_blob.data() + rdispls_bytes[i+1];
            while(ptr < end) {
                int g_row, g_col, r_dim, c_dim;
                std::memcpy(&g_row, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&g_col, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&r_dim, ptr, sizeof(int)); ptr += sizeof(int);
                std::memcpy(&c_dim, ptr, sizeof(int)); ptr += sizeof(int);
                
                size_t count = r_dim * c_dim;
                std::vector<T> data(count);
                std::memcpy(data.data(), ptr, count * sizeof(T)); ptr += count * sizeof(T);
                
                ghost_data[{g_row, g_col}] = std::move(data);
                ghost_sizes[g_col] = c_dim;
            }
        }
        
        return {ghost_data, ghost_sizes};
    }

    // SpMM Phase 4: Numerical Multiplication
    void numeric_multiply(const BlockSpMat& B, 
                          const std::map<int, std::vector<GhostBlockRef>>& ghost_rows,
                          BlockSpMat& C) const {
        int n_rows = row_ptr.size() - 1;
        #pragma omp parallel for
        for (int i = 0; i < n_rows; ++i) {
            // Build map for C row i: GlobalCol -> Offset
            std::map<int, size_t> c_map;
            int c_start = C.row_ptr[i];
            int c_end = C.row_ptr[i+1];
            for(int k=c_start; k<c_end; ++k) {
                int l_col = C.col_ind[k];
                int g_col = C.graph->get_global_index(l_col);
                c_map[g_col] = C.blk_ptr[k];
            }
            
            int a_start = row_ptr[i];
            int a_end = row_ptr[i+1];
            int r_dim = graph->block_sizes[i];
            
            for (int k = a_start; k < a_end; ++k) {
                int l_col_A = col_ind[k];
                int g_col_A = graph->get_global_index(l_col_A);
                const T* a_val = val.data() + blk_ptr[k];
                int inner_dim = graph->block_sizes[l_col_A];
                
                // Iterate B row g_col_A
                if (graph->find_owner(g_col_A) == graph->rank) {
                    // Local B
                    int l_row_B = graph->global_to_local.at(g_col_A);
                    int b_start = B.row_ptr[l_row_B];
                    int b_end = B.row_ptr[l_row_B+1];
                    for(int j=b_start; j<b_end; ++j) {
                        int l_col_B = B.col_ind[j];
                        int g_col_B = B.graph->get_global_index(l_col_B);
                        const T* b_val = B.val.data() + B.blk_ptr[j];
                        int c_dim = B.graph->block_sizes[l_col_B];
                        
                        if (c_map.count(g_col_B)) {
                            T* c_val = C.val.data() + c_map[g_col_B];
                            SmartKernel<T>::gemm(r_dim, c_dim, inner_dim, T(1), a_val, r_dim, b_val, inner_dim, T(1), c_val, r_dim);
                        }
                    }
                } else {
                    // Ghost B
                    if (ghost_rows.count(g_col_A)) {
                        for (const auto& block : ghost_rows.at(g_col_A)) {
                            int g_col_B = block.col;
                            const T* b_val = block.data;
                            int c_dim = block.c_dim;
                            
                            if (c_map.count(g_col_B)) {
                                T* c_val = C.val.data() + c_map[g_col_B];
                                SmartKernel<T>::gemm(r_dim, c_dim, inner_dim, T(1), a_val, r_dim, b_val, inner_dim, T(1), c_val, r_dim);
                            }
                        }
                    }
                }
            }
        }
    }
};

} // namespace vbcsr

#endif
