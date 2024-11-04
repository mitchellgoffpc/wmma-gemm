import os
import sys
os.environ['PYCUDA_DISABLE_CACHE'] = '1'

import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit
from pycuda.compiler import SourceModule, compile
from pathlib import Path

sys.path.append(str(Path(__file__).parent))
from params import get_params

MATRIX_DIM = 4096
WARP_SIZE = 32

def to_gpu(data):
    data_gpu = cuda.mem_alloc(data.nbytes)
    cuda.memcpy_htod(data_gpu, data.reshape(-1))
    return data, data_gpu

def benchmark(func, n=100):
    times = []
    for _ in range(n):
        start = cuda.Event()
        end = cuda.Event()
        start.record()
        func()
        end.record()
        end.synchronize()
        times.append(start.time_till(end))
    print("STATS: min=%.3fms, max=%.3fms, avg=%.3fms" % (np.min(times), np.max(times), np.mean(times)))
    return np.mean(times)

def main(kernel_path):
    params = get_params(kernel_path.stem, {'M_GLOBAL': MATRIX_DIM, 'N_GLOBAL': MATRIX_DIM, 'K_GLOBAL': MATRIX_DIM, 'WARP_SIZE': WARP_SIZE})
    with open(kernel_path) as f:
        cuda_code = f.read()
        cuda_code = cuda_code.replace('{', '{{').replace('}', '}}').replace('{{{{', '{').replace('}}}}', '}')

    cuda_code = cuda_code.format(**params)
    # ptx_code = compile(cuda_code, "nvcc", options=None, keep=False, no_extern_c=True, arch=None, code=None, cache_dir=None, include_dirs=[], target="ptx")
    # print(ptx_code.decode())
    # exit()

    mod = SourceModule(cuda_code, options=['-Xptxas="-v"'], arch="sm_86", no_extern_c=True)
    kernel = mod.get_function("wmma_gemm")
    kernel.set_attribute(cuda.function_attribute.MAX_DYNAMIC_SHARED_SIZE_BYTES, params['SHMEM_SIZE'])

    alpha = np.array(1.2, dtype=np.float16)
    beta = np.array(0.3, dtype=np.float16)
    a, a_gpu = to_gpu(np.random.normal(0, 0.1, size=(MATRIX_DIM, MATRIX_DIM)).astype(np.float16))
    b, b_gpu = to_gpu(np.random.normal(0, 0.1, size=(MATRIX_DIM, MATRIX_DIM)).astype(np.float16))
    c, c_gpu = to_gpu(np.random.normal(0, 1, size=(MATRIX_DIM, MATRIX_DIM)).astype(np.float16))
    d, d_gpu = to_gpu(np.zeros((MATRIX_DIM, MATRIX_DIM), dtype=np.float16))

    mean_time = benchmark(lambda: kernel(a_gpu, b_gpu, c_gpu, d_gpu, alpha, beta, block=(params['BLOCK_SIZE'], 1, 1), grid=(params['GRID_SIZE'], 1), shared=params['SHMEM_SIZE']), n=1000)
    gflops = 2 * MATRIX_DIM**3 / (mean_time/1000) / 1e9
    print(f"GFLOPS: {gflops:.2f}")

    cuda.memcpy_dtoh(d.reshape(-1), d_gpu)
    d_ref = alpha.astype(np.float32) * (a.astype(np.float32) @ b.astype(np.float32)) + beta.astype(np.float32) * c.astype(np.float32)
    np.testing.assert_allclose(d, d_ref, atol=6e-2)
    print("Success")


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python test.py <kernel-path>")
        sys.exit(1)

    if not Path(sys.argv[1]).exists():
        print(f"Error: Kernel file {sys.argv[1]!r} not found.")
        sys.exit(1)

    main(Path(sys.argv[1]))
