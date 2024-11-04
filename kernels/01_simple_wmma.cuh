#include <mma.h>
#include <cuda_fp16.h>

constexpr int M_GLOBAL = {{M_GLOBAL}};
constexpr int N_GLOBAL = {{N_GLOBAL}};
constexpr int K_GLOBAL = {{K_GLOBAL}};

constexpr int M_TILE = {{M_TILE}};
constexpr int N_TILE = {{N_TILE}};
constexpr int K_TILE = {{K_TILE}};

using namespace nvcuda;

extern "C" __global__ void wmma_gemm(const half *A, const half *B, const half *C, half *D, half alpha, half beta) {
    int warpsPerBlock = blockDim.x / warpSize;
    int warpsPerRow = N_GLOBAL /  N_TILE;
    int warpIdx = blockIdx.x * warpsPerBlock + (threadIdx.x / warpSize);
    int warpM = (warpIdx / warpsPerRow) *  M_TILE;
    int warpN = (warpIdx % warpsPerRow) *  N_TILE;

    wmma::fragment<wmma::matrix_a,  M_TILE,  N_TILE,  K_TILE, half, wmma::row_major> a;
    wmma::fragment<wmma::matrix_b,  M_TILE,  N_TILE,  K_TILE, half, wmma::row_major> b;
    wmma::fragment<wmma::accumulator,  M_TILE,  N_TILE,  K_TILE, half> c;
    wmma::load_matrix_sync(c, &C[warpM * N_GLOBAL + warpN], N_GLOBAL, wmma::mem_row_major);
    for (int t = 0; t < c.num_elements; t++) {
        c.x[t] *= beta / alpha;
    }

    for (int i = 0; i < K_GLOBAL; i +=  K_TILE) {
        wmma::load_matrix_sync(a, &A[warpM * K_GLOBAL + i], K_GLOBAL);
        wmma::load_matrix_sync(b, &B[i * N_GLOBAL + warpN], N_GLOBAL);
        wmma::mma_sync(c, a, b, c);
    }

    for (int t = 0; t < c.num_elements; t++) {
        c.x[t] *= alpha;
    }
    wmma::store_matrix_sync(&D[warpM * N_GLOBAL + warpN], c, N_GLOBAL, wmma::mem_row_major);
}