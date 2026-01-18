#include "../block_memory_pool.hpp"
#include <iostream>
#include <cassert>
#include <vector>
#include <cstring>

using namespace vbcsr;

void test_basic_allocation() {
    std::cout << "Testing Basic Allocation..." << std::endl;
    BlockArena<double> arena;
    
    size_t size = 100;
    uint64_t handle = arena.allocate(size);
    double* ptr = arena.get_ptr(handle);
    
    for (size_t i = 0; i < size; ++i) {
        ptr[i] = (double)i;
    }
    
    for (size_t i = 0; i < size; ++i) {
        assert(ptr[i] == (double)i);
    }
    std::cout << "PASSED" << std::endl;
}

void test_multi_page() {
    std::cout << "Testing Multi-Page Allocation..." << std::endl;
    BlockArena<double> arena;
    
    // Default page size is 1M elements
    size_t page_size = BlockArena<double>::DEFAULT_PAGE_SIZE;
    
    // Allocate 1st page full
    uint64_t h1 = arena.allocate(page_size);
    
    // Allocate 2nd page
    uint64_t h2 = arena.allocate(100);
    
    double* p1 = arena.get_ptr(h1);
    double* p2 = arena.get_ptr(h2);
    
    // Check pointers are far apart (at least page_size * sizeof(double))
    // Note: They might not be contiguous in virtual memory, but they should be distinct.
    assert(p1 != p2);
    
    // Check handles
    // h1 should be page 0, offset 0 -> 0x0000000000000000
    // h2 should be page 1, offset 0 -> 0x0000000100000000
    
    assert((h1 >> 32) == 0);
    assert((h2 >> 32) == 1);
    
    std::cout << "PASSED" << std::endl;
}

void test_reuse() {
    std::cout << "Testing Memory Reuse..." << std::endl;
    BlockArena<double> arena;
    
    size_t size = 10;
    uint64_t h1 = arena.allocate(size);
    double* p1 = arena.get_ptr(h1);
    
    arena.free(h1, size);
    
    uint64_t h2 = arena.allocate(size);
    double* p2 = arena.get_ptr(h2);
    
    // Should reuse the same handle and pointer
    assert(h1 == h2);
    assert(p1 == p2);
    
    std::cout << "PASSED" << std::endl;
}

void test_massive_filter() {
    std::cout << "Testing Massive Filter..." << std::endl;
    BlockArena<double> arena;
    
    int n_blocks = 10000;
    size_t block_size = 16;
    std::vector<uint64_t> handles;
    
    for (int i = 0; i < n_blocks; ++i) {
        handles.push_back(arena.allocate(block_size));
    }
    
    // Free even blocks
    for (int i = 0; i < n_blocks; i += 2) {
        arena.free(handles[i], block_size);
    }
    
    // Reallocate
    for (int i = 0; i < n_blocks; i += 2) {
        uint64_t h = arena.allocate(block_size);
        // Should be one of the freed handles
        // We can't guarantee order (stack vs queue), but it should be valid
        double* p = arena.get_ptr(h);
        p[0] = 1.0; // Write to verify validity
    }
    
    std::cout << "PASSED" << std::endl;
}

int main() {
    test_basic_allocation();
    test_multi_page();
    test_reuse();
    test_massive_filter();
    return 0;
}
