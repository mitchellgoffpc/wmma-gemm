#include <mma.h>
#include <cuda_fp16.h>

constexpr int N_GLOBAL = {{N_GLOBAL}};
constexpr int K_GLOBAL = {{K_GLOBAL}};
constexpr int M_BLOCK = {{M_BLOCK}};
constexpr int N_BLOCK = {{N_BLOCK}};
constexpr int M_TILE = {{M_TILE}};
constexpr int N_TILE = {{N_TILE}};
constexpr int K_TILE = {{K_TILE}};
constexpr int K_CHUNK = {{K_CHUNK}};

constexpr int N_TILES = N_GLOBAL / N_TILE;
constexpr int TILE_ROWS_PER_BLOCK = {{TILE_ROWS_PER_BLOCK}};
constexpr int TILE_COLS_PER_BLOCK = {{TILE_COLS_PER_BLOCK}};
constexpr int TILE_ROWS_PER_WARP = {{TILE_ROWS_PER_WARP}};
constexpr int TILE_COLS_PER_WARP = {{TILE_COLS_PER_WARP}};
constexpr int WARP_ROWS_PER_BLOCK = TILE_ROWS_PER_BLOCK / TILE_ROWS_PER_WARP;
constexpr int WARP_COLS_PER_BLOCK = TILE_COLS_PER_BLOCK / TILE_COLS_PER_WARP;
constexpr int WARPS_PER_BLOCK = WARP_ROWS_PER_BLOCK * WARP_COLS_PER_BLOCK;

constexpr int WARP_SIZE = 32;
constexpr int A_SHMEM_STRIDE = K_CHUNK + {{CHUNK_PADDING}};
constexpr int B_SHMEM_STRIDE = N_BLOCK + {{CHUNK_PADDING}};
constexpr int A_COPY_ROWS_PER_WARP = M_BLOCK / WARPS_PER_BLOCK;
constexpr int B_COPY_ROWS_PER_WARP = K_CHUNK / WARPS_PER_BLOCK;

using namespace nvcuda;


extern "C" __global__ void wmma_gemm(const half *A, const half *B, const half *C, half *D, half alpha, half beta) {
    extern __shared__ half shmem[][A_SHMEM_STRIDE];
    half (*A_shmem)[A_SHMEM_STRIDE] = (half (*)[A_SHMEM_STRIDE])(&shmem[0][0]);
    half (*B_shmem)[B_SHMEM_STRIDE] = (half (*)[B_SHMEM_STRIDE])(&shmem[M_BLOCK][0]);

    beta /= alpha;

    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;

    // Compute row and column offsets for blocks and warps
    unsigned int block_pos = blockIdx.x;
    const unsigned int block_tile_i = ((block_pos * TILE_COLS_PER_BLOCK) / N_TILES) * TILE_ROWS_PER_BLOCK;
    const unsigned int block_tile_j = (block_pos * TILE_COLS_PER_BLOCK) % N_TILES;
    const unsigned int warp_tile_i = block_tile_i + (warpId / WARP_COLS_PER_BLOCK) * TILE_ROWS_PER_WARP;
    const unsigned int warp_tile_j = block_tile_j + (warpId % WARP_COLS_PER_BLOCK) * TILE_COLS_PER_WARP;
    const half *A_global = &A[block_tile_i * M_TILE * K_GLOBAL + warpId * A_COPY_ROWS_PER_WARP * K_GLOBAL];
    const half *B_global = &B[block_tile_j * N_TILE + warpId * B_COPY_ROWS_PER_WARP * K_GLOBAL];

    // Load the C matrix tiles into fragments from global memory
    wmma::fragment<wmma::matrix_a, M_TILE, N_TILE, K_TILE, half, wmma::row_major> a[TILE_ROWS_PER_WARP];
    wmma::fragment<wmma::matrix_b, M_TILE, N_TILE, K_TILE, half, wmma::row_major> b[TILE_COLS_PER_WARP];
    wmma::fragment<wmma::accumulator, M_TILE, N_TILE, K_TILE, half> c[TILE_ROWS_PER_WARP][TILE_COLS_PER_WARP];

    for (int i = 0; i < TILE_ROWS_PER_WARP; i++) {
        for (int j = 0; j < TILE_COLS_PER_WARP; j++) {
            wmma::load_matrix_sync(c[i][j], &C[(warp_tile_i + i) * M_TILE * N_GLOBAL + (warp_tile_j + j) * N_TILE], N_GLOBAL, wmma::mem_row_major);
            for (int t = 0; t < c[i][j].num_elements; t++) {
                c[i][j].x[t] *= beta;
            }
        }
    }

    // Iterate through the global K dimension one chunk at a time
    for (int tile_k = 0; tile_k < K_GLOBAL; tile_k += K_CHUNK) {
        // Copy slices of the A and B matrices to shared memory
        for (int i = 0; i < A_COPY_ROWS_PER_WARP; i++) {
            for (int j = 0; j < K_CHUNK; j += WARP_SIZE) {
                A_shmem[warpId * A_COPY_ROWS_PER_WARP + i][j + laneId] = A_global[i * K_GLOBAL + j + laneId + tile_k];
            }
        }
        for (int i = 0; i < B_COPY_ROWS_PER_WARP; i++) {
            for (int j = 0; j < N_BLOCK; j += WARP_SIZE) {
                B_shmem[warpId * B_COPY_ROWS_PER_WARP + i][j + laneId] = B_global[i * N_GLOBAL + j + laneId + tile_k * N_GLOBAL];
            }
        }
        __syncthreads();

        // Compute a grid of C matrix tiles in each warp
        for (int k_step = 0; k_step < K_CHUNK; k_step += K_TILE) {
            for (int i = 0; i < TILE_ROWS_PER_WARP; i++) {
                wmma::load_matrix_sync(a[i], &A_shmem[(warpId / WARP_COLS_PER_BLOCK) * TILE_ROWS_PER_WARP * M_TILE + i * M_TILE][k_step], A_SHMEM_STRIDE);
            }
            for (int j = 0; j < TILE_COLS_PER_WARP; j++) {
                wmma::load_matrix_sync(b[j], &B_shmem[k_step][(warpId % WARP_COLS_PER_BLOCK) * TILE_COLS_PER_WARP * N_TILE + j * N_TILE], B_SHMEM_STRIDE);
            }
            for (int i = 0; i < TILE_ROWS_PER_WARP; i++) {
                for (int j = 0; j < TILE_COLS_PER_WARP; j++) {
                    wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                }
            }
        }
        __syncthreads();
    }

    // Scale and store the D matrix tiles to global memory
    for (int i = 0; i < TILE_ROWS_PER_WARP; i++) {
        for (int j = 0; j < TILE_COLS_PER_WARP; j++) {
            for (int t = 0; t < c[i][j].num_elements; t++) {
                c[i][j].x[t] *= alpha;
            }
            wmma::store_matrix_sync(&D[(warp_tile_i + i) * M_TILE * N_GLOBAL + (warp_tile_j + j) * N_TILE], c[i][j], N_GLOBAL, wmma::mem_row_major);
        }
    }
}