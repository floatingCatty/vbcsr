import os
import subprocess
import sys

tests = [
    "benchmark_axpby.cpp",
    "benchmark_dist.cpp",
    "benchmark_spmm_complex.cpp",
    "test_asymmetric_filter.cpp",
    "test_axpby.cpp",
    "test_axpby_diff_graph.cpp",
    "test_backend_extensions.cpp",
    "test_block_arena.cpp",
    "test_block_csr.cpp",
    "test_block_csr_export.cpp",
    "test_complex_dist_vector.cpp",
    "test_density.cpp",
    "test_dist_graph.cpp",
    "test_extract_batched.cpp",
    "test_extract_batched_extended.cpp",
    "test_extract_batched_robust.cpp",
    "test_hermitian_product.cpp",
    "test_mult_graph_mismatch.cpp",
    "test_multi_sparsity.cpp",
    "test_pb_csr.cpp",
    "test_robustness.cpp",
    "test_spmm.cpp",
    "test_subgraph.cpp"
]

include_path = "/home/zhanghao/softwares/vbcsr/vbcsr/core"
cwd = "/home/zhanghao/softwares/vbcsr/vbcsr/core/test"

failed_tests = []
passed_tests = []

print(f"Running {len(tests)} tests...")

for test_file in tests:
    exe_name = test_file.replace(".cpp", "")
    print(f"--------------------------------------------------")
    print(f"Testing {exe_name}...")
    
    # Compile
    compile_cmd = [
        "mpicxx", "-std=c++17", "-fopenmp", "-O3", 
        f"-I{include_path}", 
        test_file, "-o", exe_name,
        "-lopenblas", "-lgtest", "-lgtest_main"
    ]
    
    try:
        subprocess.check_output(compile_cmd, cwd=cwd, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        print(f"COMPILATION FAILED: {test_file}")
        print(e.output.decode())
        failed_tests.append(test_file)
        continue
        
    # Run
    run_cmd = ["mpirun", "-np", "4", f"./{exe_name}"]
    try:
        output = subprocess.check_output(run_cmd, cwd=cwd, stderr=subprocess.STDOUT)
        print("PASSED")
        passed_tests.append(test_file)
    except subprocess.CalledProcessError as e:
        print(f"EXECUTION FAILED: {test_file}")
        print(e.output.decode())
        failed_tests.append(test_file)

print("==================================================")
print(f"Summary: {len(passed_tests)}/{len(tests)} passed.")
if failed_tests:
    print("Failed tests:")
    for t in failed_tests:
        print(f"  - {t}")
    sys.exit(1)
else:
    print("All tests passed!")
    sys.exit(0)
