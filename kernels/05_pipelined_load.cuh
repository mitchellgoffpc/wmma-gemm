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
constexpr int A_ROWS_PER_COPY = (sizeof(int4) * WARP_SIZE) / (sizeof(half) * K_CHUNK);
constexpr int B_ROWS_PER_COPY = (sizeof(int4) * WARP_SIZE) / (sizeof(half) * N_BLOCK);
constexpr int A_COPIES_PER_WARP = M_BLOCK / (WARPS_PER_BLOCK * A_ROWS_PER_COPY);
constexpr int B_COPIES_PER_WARP = K_CHUNK / (WARPS_PER_BLOCK * B_ROWS_PER_COPY);

using namespace nvcuda;


extern "C" __global__ void wmma_gemm(const half *A, const half *B, const half *C, half *D, half alpha, half beta) {
    extern __shared__ half shmem[][A_SHMEM_STRIDE];
    half (*A_shmem)[A_SHMEM_STRIDE] = (half (*)[A_SHMEM_STRIDE])(&shmem[0][0]);
    half (*B_shmem)[B_SHMEM_STRIDE] = (half (*)[B_SHMEM_STRIDE])(&shmem[M_BLOCK * 2][0]);

    beta /= alpha;

    const unsigned int warpId = threadIdx.x / WARP_SIZE;
    const unsigned int laneId = threadIdx.x % WARP_SIZE;

    // Compute row and column offsets for blocks and warps
    unsigned int block_pos = blockIdx.x;
    const unsigned int block_tile_i = ((block_pos * TILE_COLS_PER_BLOCK) / N_TILES) * TILE_ROWS_PER_BLOCK;
    const unsigned int block_tile_j = (block_pos * TILE_COLS_PER_BLOCK) % N_TILES;
    const unsigned int warp_tile_i = block_tile_i + (warpId / WARP_COLS_PER_BLOCK) * TILE_ROWS_PER_WARP;
    const unsigned int warp_tile_j = block_tile_j + (warpId % WARP_COLS_PER_BLOCK) * TILE_COLS_PER_WARP;
    const unsigned int A_lane_row = laneId / (WARP_SIZE / A_ROWS_PER_COPY);
    const unsigned int B_lane_row = laneId / (WARP_SIZE / B_ROWS_PER_COPY);
    const unsigned int A_lane_col = laneId % (WARP_SIZE / A_ROWS_PER_COPY) * (sizeof(int4) / sizeof(half));
    const unsigned int B_lane_col = laneId % (WARP_SIZE / B_ROWS_PER_COPY) * (sizeof(int4) / sizeof(half));
    const half *A_global = &A[block_tile_i * M_TILE * K_GLOBAL + warpId * A_ROWS_PER_COPY * A_COPIES_PER_WARP * K_GLOBAL];
    const half *B_global = &B[block_tile_j * N_TILE + warpId * B_ROWS_PER_COPY * B_COPIES_PER_WARP * K_GLOBAL];

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

    // Copy slices of the A and B matrices to shared memory for the first iteration
    for (int i = 0; i < A_COPIES_PER_WARP; i++) {
        *((int4 *)&A_shmem[warpId * A_ROWS_PER_COPY * A_COPIES_PER_WARP + i * A_ROWS_PER_COPY + A_lane_row][A_lane_col]) =
            *((int4 *)&A_global[(i * A_ROWS_PER_COPY + A_lane_row) * K_GLOBAL + A_lane_col]);
    }
    for (int i = 0; i < B_COPIES_PER_WARP; i++) {
        *((int4 *)&B_shmem[warpId * B_ROWS_PER_COPY * B_COPIES_PER_WARP + i * B_ROWS_PER_COPY + B_lane_row][B_lane_col]) =
            *((int4 *)&B_global[(i * B_ROWS_PER_COPY + B_lane_row) * N_GLOBAL + B_lane_col]);
    }
    __syncthreads();

    // Iterate through the global K dimension one chunk at a time
    #pragma unroll
    for (int tile_k = 0; tile_k < K_GLOBAL / K_CHUNK; tile_k++) {
        half (*A_shmem_current)[A_SHMEM_STRIDE] = (half (*)[A_SHMEM_STRIDE])(&A_shmem[(tile_k % 2) * M_BLOCK][0]);
        half (*B_shmem_current)[B_SHMEM_STRIDE] = (half (*)[B_SHMEM_STRIDE])(&B_shmem[(tile_k % 2) * K_CHUNK][0]);
        half (*A_shmem_next)[A_SHMEM_STRIDE] = (half (*)[A_SHMEM_STRIDE])(&A_shmem[((tile_k + 1) % 2) * M_BLOCK][0]);
        half (*B_shmem_next)[B_SHMEM_STRIDE] = (half (*)[B_SHMEM_STRIDE])(&B_shmem[((tile_k + 1) % 2) * K_CHUNK][0]);

        // Copy slices of the A and B matrices to shared memory using cp.async
        if (tile_k < K_GLOBAL / K_CHUNK - 1) {
            for (int i = 0; i < A_COPIES_PER_WARP; i++) {
                const half* src = &A_global[(i * A_ROWS_PER_COPY + A_lane_row) * K_GLOBAL + A_lane_col + (tile_k + 1) * K_CHUNK];
                half* dst = &A_shmem_next[warpId * A_ROWS_PER_COPY * A_COPIES_PER_WARP + i * A_ROWS_PER_COPY + A_lane_row][A_lane_col];
                size_t dst_shared = __cvta_generic_to_shared(dst);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "l"(dst_shared), "l"(src));
            }
            for (int i = 0; i < B_COPIES_PER_WARP; i++) {
                const half* src = &B_global[(i * B_ROWS_PER_COPY + B_lane_row) * N_GLOBAL + B_lane_col + (tile_k + 1) * K_CHUNK * N_GLOBAL];
                half* dst = &B_shmem_next[warpId * B_ROWS_PER_COPY * B_COPIES_PER_WARP + i * B_ROWS_PER_COPY + B_lane_row][B_lane_col];
                size_t dst_shared = __cvta_generic_to_shared(dst);
                asm volatile("cp.async.ca.shared.global [%0], [%1], 16;" :: "l"(dst_shared), "l"(src));
            }
        }

        // Compute a grid of C matrix tiles in each warp
        for (int k_step = 0; k_step < K_CHUNK; k_step += K_TILE) {
            for (int i = 0; i < TILE_ROWS_PER_WARP; i++) {
                wmma::load_matrix_sync(a[i], &A_shmem_current[(warpId / WARP_COLS_PER_BLOCK) * TILE_ROWS_PER_WARP * M_TILE + i * M_TILE][k_step], A_SHMEM_STRIDE);
            }
            for (int j = 0; j < TILE_COLS_PER_WARP; j++) {
                wmma::load_matrix_sync(b[j], &B_shmem_current[k_step][(warpId % WARP_COLS_PER_BLOCK) * TILE_COLS_PER_WARP * N_TILE + j * N_TILE], B_SHMEM_STRIDE);
            }
            for (int i = 0; i < TILE_ROWS_PER_WARP; i++) {
                for (int j = 0; j < TILE_COLS_PER_WARP; j++) {
                    wmma::mma_sync(c[i][j], a[i], b[j], c[i][j]);
                }
            }
        }

        if (tile_k < K_GLOBAL / K_CHUNK - 1) {
            asm volatile("cp.async.wait_all;");
            __syncthreads();
        }
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