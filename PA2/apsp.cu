// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"

/*
when modify block size, check:
1. modify b
2. modify max_steps
3. modify share memory size in kernel function
4. modify share memory size when calling kernel function
*/

namespace {

__global__ void kernel_1(int n, int p, int *graph) {
    // coordinate: (j, i), but is the ith row, the jth col, index: i * n + j
    __shared__ int s[64][64];
    
    // in the block or share memory, row: i_s, col: j_s
    // 0 <= i_s, j_s < b * 2, deal with 4 elements
    const auto i_s = threadIdx.y << 1;
    const auto j_s = threadIdx.x << 1;

    // the p-th diagonal block
    // upper left: (i_s, j_s)
    const auto i = p * 64 + i_s;
    const auto j = p * 64 + j_s;
    const auto index = i * n + j;

    const bool in_bound_ul = i < n && j < n;
    const bool in_bound_ur = i < n && j + 1 < n;
    const bool in_bound_ll = i + 1 < n && j < n;
    const bool in_bound_lr = i + 1 < n && j + 1 < n;

    // copy from global memory to share memory
    // load to registers: old_dist_*, 2 * 2 blocks
    int old_dist_ul;
    if (in_bound_ul) {
        old_dist_ul = graph[index];
        s[i_s][j_s] = old_dist_ul;
    } else {
        old_dist_ul = 66666666;
        s[i_s][j_s] = old_dist_ul;
    }

    int old_dist_ur;
    if (in_bound_ur) {
        old_dist_ur = graph[index + 1];
        s[i_s][j_s + 1] = old_dist_ur;
    } else {
        old_dist_ur = 66666666;
        s[i_s][j_s + 1] = old_dist_ur;
    }

    int old_dist_ll;
    if (in_bound_ll) {
        old_dist_ll = graph[index + n];
        s[i_s + 1][j_s] = old_dist_ll;
    } else {
        old_dist_ll = 66666666;
        s[i_s + 1][j_s] = old_dist_ll;
    }

    int old_dist_lr;
    if (in_bound_lr) {
        old_dist_lr = graph[index + n + 1];
        s[i_s + 1][j_s + 1] = old_dist_lr;
    } else {
        old_dist_lr = 66666666;
        s[i_s + 1][j_s + 1] = old_dist_lr;
    }
    __syncthreads();

    int new_dist_ul, new_dist_ur, new_dist_ll, new_dist_lr;
    #pragma unroll(64)
    for (int k_s = 0; k_s < 64; ++k_s) {
        int tmp1 = s[i_s][k_s];
        int tmp2 = s[i_s + 1][k_s];
        int tmp3 = s[k_s][j_s];
        int tmp4 = s[k_s][j_s + 1];

        new_dist_ul = tmp1 + tmp3;
        old_dist_ul = min(old_dist_ul, new_dist_ul);

        new_dist_ur = tmp1 + tmp4;
        old_dist_ur = min(old_dist_ur, new_dist_ur);

        new_dist_ll = tmp2 + tmp3;
        old_dist_ll = min(old_dist_ll, new_dist_ll);

        new_dist_lr = tmp2 + tmp4;
        old_dist_lr = min(old_dist_lr, new_dist_lr);
    }

    if (in_bound_ul) {
        graph[index] = old_dist_ul;
    }
    if (in_bound_ur) {
        graph[index + 1] = old_dist_ur;
    }
    if (in_bound_ll) {
        graph[index + n] = old_dist_ll;
    }
    if (in_bound_lr) {
        graph[index + n + 1] = old_dist_lr;
    }
}

__global__ void kernel_2(int n, int p, int *graph) {
    // coordinate: (j, i), but is the ith row, the jth col, index: i * n + j
    __shared__ int s_cross[64][64];  // cross block
    __shared__ int s_center[64][64]; // center block

    // in the block or share memory, row: i_s, col: j_s
    // 0 <= i_s, j_s < b * 2
    const auto i_s = threadIdx.y << 1;
    const auto j_s = threadIdx.x << 1;

    // blockIdx: (x, y) 0 <= x < max_steps, 0 <= y < 2
    // blockIdx.y == 0: row
    // blockIdx.y == 1: col

    // the p-th diagonal block
    // upper left: (i_s, j_s)
    auto ip = p * 64 + i_s;
    auto jp = p * 64 + j_s;
    const auto indexp = ip * n + jp;

    // copy from global memory to share memory
    if (ip < n && jp < n) {
        s_center[i_s][j_s] = graph[indexp];
    } else {
        s_center[i_s][j_s] = 66666666;
    }

    if (ip < n && jp + 1 < n) {
        s_center[i_s][j_s + 1] = graph[indexp + 1];
    } else {
        s_center[i_s][j_s + 1] = 66666666;
    }

    if (ip + 1 < n && jp < n) {
        s_center[i_s + 1][j_s] = graph[indexp + n];
    } else {
        s_center[i_s + 1][j_s] = 66666666;
    }

    if (ip + 1 < n && jp + 1 < n) {
        s_center[i_s + 1][j_s + 1] = graph[indexp + n + 1];
    } else {
        s_center[i_s + 1][j_s + 1] = 66666666;
    }

    auto i = ip, j = jp;

    if (blockIdx.y == 0) {
        i = blockIdx.x * 64 + i_s;
    } else {
        j = blockIdx.x * 64 + j_s;
    }

    const auto index = i * n + j;

    const bool in_bound_ul = i < n && j < n;
    const bool in_bound_ur = i < n && j + 1 < n;
    const bool in_bound_ll = i + 1 < n && j < n;
    const bool in_bound_lr = i + 1 < n && j + 1 < n;

    // copy from global memory to share memory
    int old_dist_ul;
    if (in_bound_ul) {
        old_dist_ul = graph[index];
        s_cross[i_s][j_s] = old_dist_ul;
    } else {
        old_dist_ul = 66666666;
        s_cross[i_s][j_s] = old_dist_ul;
    }

    int old_dist_ur;
    if (in_bound_ur) {
        old_dist_ur = graph[index + 1];
        s_cross[i_s][j_s + 1] = old_dist_ur;
    } else {
        old_dist_ur = 66666666;
        s_cross[i_s][j_s + 1] = old_dist_ur;
    }

    int old_dist_ll;
    if (in_bound_ll) {
        old_dist_ll = graph[index + n];
        s_cross[i_s + 1][j_s] = old_dist_ll;
    } else {
        old_dist_ll = 66666666;
        s_cross[i_s + 1][j_s] = old_dist_ll;
    }

    int old_dist_lr;
    if (in_bound_lr) {
        old_dist_lr = graph[index + n + 1];
        s_cross[i_s + 1][j_s + 1] = old_dist_lr;
    } else {
        old_dist_lr = 66666666;
        s_cross[i_s + 1][j_s + 1] = old_dist_lr;
    }
    __syncthreads();

    int new_dist_ul, new_dist_ur, new_dist_ll, new_dist_lr;
    if (blockIdx.y == 0) { // row
        #pragma unroll(64)
        for (int k_s = 0; k_s < 64; ++k_s) {
            int tmp1 = s_cross[i_s][k_s];
            int tmp2 = s_cross[i_s + 1][k_s];
            int tmp3 = s_center[k_s][j_s];
            int tmp4 = s_center[k_s][j_s + 1];

            new_dist_ul = tmp1 + tmp3;
            old_dist_ul = min(old_dist_ul, new_dist_ul);

            new_dist_ur = tmp1 + tmp4;
            old_dist_ur = min(old_dist_ur, new_dist_ur);

            new_dist_ll = tmp2 + tmp3;
            old_dist_ll = min(old_dist_ll, new_dist_ll);

            new_dist_lr = tmp2 + tmp4;
            old_dist_lr = min(old_dist_lr, new_dist_lr);
        }
    } else { // col
        #pragma unroll(64)
        for (int k_s = 0; k_s < 64; ++k_s) {
            int tmp1 = s_center[i_s][k_s];
            int tmp2 = s_center[i_s + 1][k_s];
            int tmp3 = s_cross[k_s][j_s];
            int tmp4 = s_cross[k_s][j_s + 1];

            new_dist_ul = tmp1 + tmp3;
            old_dist_ul = min(old_dist_ul, new_dist_ul);

            new_dist_ur = tmp1 + tmp4;
            old_dist_ur = min(old_dist_ur, new_dist_ur);

            new_dist_ll = tmp2 + tmp3;
            old_dist_ll = min(old_dist_ll, new_dist_ll);

            new_dist_lr = tmp2 + tmp4;
            old_dist_lr = min(old_dist_lr, new_dist_lr);
        }
    }
    if (in_bound_ul) {
        graph[index] = old_dist_ul;
    }
    if (in_bound_ur) {
        graph[index + 1] = old_dist_ur;
    }
    if (in_bound_ll) {
        graph[index + n] = old_dist_ll;
    }
    if (in_bound_lr) {
        graph[index + n + 1] = old_dist_lr;
    }
}

__global__ void kernel_3(int n, int p, int *graph) {
    // coordinate: (j, i), but is the ith row, the jth col, index: i * n + j
    __shared__ int s_cross_row[64][64];  // cross row block
    __shared__ int s_cross_col[64][64];  // cross col block

    // in the block or share memory, row: i_s, col: j_s
    // 0 <= i_s, j_s < b * 2
    const auto i_s = threadIdx.y << 1;
    const auto j_s = threadIdx.x << 1;

    // the p-th diagonal block
    // upper left: (i_s, j_s)
    const auto ip = p * 64 + i_s;
    const auto jp = p * 64 + j_s;

    // in the whole graph(n * n), row: i, col: j, index: i * n + j
    // 0 <= i, j < max_steps * b, may exceed n
    const auto i = blockIdx.y * 64 + i_s;
    const auto j = blockIdx.x * 64 + j_s;
    const auto index = i * n + j;

    // (ip, j) and (ip, jp) are in the same row
    if (ip < n && j < n) {
        s_cross_row[i_s][j_s] = graph[ip * n + j];
    } else {
        s_cross_row[i_s][j_s] = 66666666;
    }

    if (ip < n && j + 1 < n) {
        s_cross_row[i_s][j_s + 1] = graph[ip * n + j + 1];
    } else {
        s_cross_row[i_s][j_s + 1] = 66666666;
    }

    if (ip + 1 < n && j < n) {
        s_cross_row[i_s + 1][j_s] = graph[ip * n + j + n];
    } else {
        s_cross_row[i_s + 1][j_s] = 66666666;
    }

    if (ip + 1 < n && j + 1 < n) {
        s_cross_row[i_s + 1][j_s + 1] = graph[ip * n + j + n + 1];
    } else {
        s_cross_row[i_s + 1][j_s + 1] = 66666666;
    }

    // (i, jp) and (ip, jp) are in the same col
    if (i < n && jp < n) {
        s_cross_col[i_s][j_s] = graph[i * n + jp];
    } else {
        s_cross_col[i_s][j_s] = 66666666;
    }

    if (i < n && jp + 1 < n) {
        s_cross_col[i_s][j_s + 1] = graph[i * n + jp + 1];
    } else {
        s_cross_col[i_s][j_s + 1] = 66666666;
    }

    if (i + 1 < n && jp < n) {
        s_cross_col[i_s + 1][j_s] = graph[i * n + jp + n];
    } else {
        s_cross_col[i_s + 1][j_s] = 66666666;
    }

    if (i + 1 < n && jp + 1 < n) {
        s_cross_col[i_s + 1][j_s + 1] = graph[i * n + jp + n + 1];
    } else {
        s_cross_col[i_s + 1][j_s + 1] = 66666666;
    }
    __syncthreads();

    const bool in_bound_ul = i < n && j < n;
    const bool in_bound_ur = i < n && j + 1 < n;
    const bool in_bound_ll = i + 1 < n && j < n;
    const bool in_bound_lr = i + 1 < n && j + 1 < n;

    int old_dist_ul, old_dist_ur, old_dist_ll, old_dist_lr;
    if (in_bound_ul) {
        old_dist_ul = graph[index];
    }

    if (in_bound_ur) {
        old_dist_ur = graph[index + 1];
    }

    if (in_bound_ll) {
        old_dist_ll = graph[index + n];
    }

    if (in_bound_lr) {
        old_dist_lr = graph[index + n + 1];
    }

    int new_dist_ul, new_dist_ur, new_dist_ll, new_dist_lr;
    #pragma unroll(64)
    for (int k_s = 0; k_s < 64; ++k_s) {
        int tmp1 = s_cross_col[i_s][k_s];
        int tmp2 = s_cross_col[i_s + 1][k_s];
        int tmp3 = s_cross_row[k_s][j_s];
        int tmp4 = s_cross_row[k_s][j_s + 1];

        new_dist_ul = tmp1 + tmp3;
        old_dist_ul = min(old_dist_ul, new_dist_ul);

        new_dist_ur = tmp1 + tmp4;
        old_dist_ur = min(old_dist_ur, new_dist_ur);

        new_dist_ll = tmp2 + tmp3;
        old_dist_ll = min(old_dist_ll, new_dist_ll);

        new_dist_lr = tmp2 + tmp4;
        old_dist_lr = min(old_dist_lr, new_dist_lr);
    }

    if (in_bound_ul) {
        graph[index] = old_dist_ul;
    }

    if (in_bound_ur) {
        graph[index + 1] = old_dist_ur;
    }

    if (in_bound_ll) {
        graph[index + n] = old_dist_ll;
    }

    if (in_bound_lr) {
        graph[index + n + 1] = old_dist_lr;
    }
}

}

void apsp(int n, /* device */ int *graph) {
    // max shared_memory: 48KB, max number of int: 12000
    // blk_size b: 32 * 32
    // max_steps: (n - 1) / 64 + 1
    int max_steps = (n - 1) / 64 + 1;
    dim3 thr(32, 32);
    dim3 blk1(1, 1);
    dim3 blk2(max_steps, 2);
    dim3 blk3(max_steps, max_steps);

    for (int p = 0; p < max_steps; ++p){
        kernel_1<<<blk1, thr>>>(n, p, graph);
        kernel_2<<<blk2, thr>>>(n, p, graph);
        kernel_3<<<blk3, thr>>>(n, p, graph);
    }

}

