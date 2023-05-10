// PLEASE MODIFY THIS FILE TO IMPLEMENT YOUR SOLUTION

// Brute Force APSP Implementation:

#include "apsp.h"
#include "cuda_utils.h"
#include <cassert>
#include <stdio.h>

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
    const int b = 32;
    __shared__ int s[b][b];

    const auto i_s = threadIdx.y;
    const auto j_s = threadIdx.x;

    // the p-th diagonal block
    const auto i = p * b + i_s;
    const auto j = p * b + j_s;
    const auto index = i * n + j;

    const bool in_bound = i < n && j < n;

    // copy from global memory to share memory
    int old_dist;
    if (in_bound) {
        old_dist = graph[index];
        s[i_s][j_s] = old_dist;
    } else {
        old_dist = 66666666;
        s[i_s][j_s] = old_dist;
    }
    __syncthreads();

    int new_dist;
    for (int k_s = 0; k_s < b; ++k_s) {
        new_dist = s[i_s][k_s] + s[k_s][j_s];
        old_dist = min(old_dist, new_dist);
        s[i_s][j_s] = old_dist;
    }

    if (in_bound) {
        graph[index] = s[i_s][j_s];
    }
}

__global__ void kernel_2(int n, int p, int *graph) {
    const int b = 32;
    // coordinate: (j, i), but is the ith row, the jth col, index: i * n + j
    __shared__ int s_cross[b][b];  // cross block
    __shared__ int s_center[b][b]; // center block

    // share memory index
    const auto i_s = threadIdx.y;
    const auto j_s = threadIdx.x;

    // blockIdx: (x, y) 0 <= x < max_steps, 0 <= y < 2
    // blockIdx.y == 0: row
    // blockIdx.y == 1: col
    // the p-th diagonal block
    auto ip = p * b + i_s;
    auto jp = p * b + j_s;
    const auto indexp = ip * n + jp;

    // copy from global memory to share memory
    if (ip < n && jp < n) {
        s_center[i_s][j_s] = graph[indexp];
    } else {
        s_center[i_s][j_s] = 66666666;
    }
    __syncthreads();

    auto i = ip, j = jp;
    if (blockIdx.y == 0) {
        i = blockIdx.x * b + i_s;
    } else {
        j = blockIdx.x * b + j_s;
    }
    const bool in_bound = i < n && j < n;
    const auto index = i * n + j;
    int old_dist;

    // copy from global memory to share memory
    if (in_bound) {
        old_dist = graph[index];
        s_cross[i_s][j_s] = old_dist;
    } else {
        old_dist = 66666666;
        s_cross[i_s][j_s] = old_dist;
    }
    __syncthreads();

    int new_dist;
    if (blockIdx.y == 0) { // row
        for (int k_s = 0; k_s < b; ++k_s) {
            new_dist = s_cross[i_s][k_s] + s_center[k_s][j_s];
            old_dist = min(old_dist, new_dist);
            s_cross[i_s][j_s] = old_dist;
        }
    } else { // col
        for (int k_s = 0; k_s < b; ++k_s) {
            new_dist = s_center[i_s][k_s] + s_cross[k_s][j_s];
            old_dist = min(old_dist, new_dist);
            s_cross[i_s][j_s] = old_dist;
        }
    }
    if (in_bound) {
        graph[index] = s_cross[i_s][j_s];
    }
}

__global__ void kernel_3(int n, int p, int *graph) {
    const int b = 32;
    // coordinate: (j, i), but is the ith row, the jth col, index: i * n + j
    __shared__ int s_cross_row[b][b];  // cross row block
    __shared__ int s_cross_col[b][b];  // cross col block

    // in the block or share memory, row: i_s, col: j_s
    // 0 <= i_s, j_s < b
    const auto i_s = threadIdx.y;
    const auto j_s = threadIdx.x;

    // center block
    const auto ip = p * b + i_s;
    const auto jp = p * b + j_s;

    // in the whole graph(n * n), row: i, col: j, index: i * n + j
    // 0 <= i, j < max_steps * b, may exceed n
    const auto i = blockIdx.y * b + i_s;
    const auto j = blockIdx.x * b + j_s;
    const auto index = i * n + j;

    // (ip, j) and (ip, jp) are in the same row
    if (ip < n && j < n) {
        s_cross_row[i_s][j_s] = graph[ip * n + j];
    } else {
        s_cross_row[i_s][j_s] = 66666666;
    }
    // (i, jp) and (ip, jp) are in the same col
    if (i < n && jp < n) {
        s_cross_col[i_s][j_s] = graph[i * n + jp];
    } else {
        s_cross_col[i_s][j_s] = 66666666;
    }
    __syncthreads();

    int old_dist = graph[index];
    int new_dist;
    for (int k_s = 0; k_s < b; ++k_s) {
        new_dist = s_cross_col[i_s][k_s] + s_cross_row[k_s][j_s];
        old_dist = min(old_dist, new_dist);
    }
    if (i < n && j < n) {
        graph[index] = old_dist;
    }
}

}

void apsp(int n, /* device */ int *graph) {
    // max shared_memory: 48KB, max number of int: 12000
    // blk_size b: 32 * 32
    // max_steps: (n - 1) / 32 + 1
    constexpr int b = 32;
    int max_steps = (n - 1) / b + 1;
    dim3 thr(b, b);
    dim3 blk1(1, 1);
    dim3 blk2(max_steps, 2);
    dim3 blk3(max_steps, max_steps);

    for (int p = 0; p < max_steps; ++p){
        kernel_1<<<blk1, thr>>>(n, p, graph);
        // // test:
        // CHK_CUDA_ERR(cudaDeviceSynchronize());
        // // Check for kernel launch errors
        // cudaError_t error = cudaGetLastError();
        // if (error != cudaSuccess) {
        //     printf("Kernel_1 launch failed with error: %s\n", cudaGetErrorString(error));
        //     exit(1);
        // }

        kernel_2<<<blk2, thr>>>(n, p, graph);

        // kernel_2_row<<<blk2_1, thr>>>(n, p, graph);
        // // test:
        // CHK_CUDA_ERR(cudaDeviceSynchronize());
        // // Check for kernel launch errors
        // error = cudaGetLastError();
        // if (error != cudaSuccess) {
        //     printf("Kernel_2_row launch failed with error: %s\n", cudaGetErrorString(error));
        //     exit(1);
        // }

        // kernel_2_col<<<blk2_2, thr>>>(n, p, graph);
        // // test:
        // CHK_CUDA_ERR(cudaDeviceSynchronize());
        // // Check for kernel launch errors
        // error = cudaGetLastError();
        // if (error != cudaSuccess) {
        //     printf("Kernel_2_col launch failed with error: %s\n", cudaGetErrorString(error));
        //     exit(1);
        // }

        kernel_3<<<blk3, thr>>>(n, p, graph);
        // // test:
        // CHK_CUDA_ERR(cudaDeviceSynchronize());
        // // Check for kernel launch errors
        // error = cudaGetLastError();
        // if (error != cudaSuccess) {
        //     printf("Kernel_3 launch failed with error: %s\n", cudaGetErrorString(error));
        //     exit(1);
        // }
    }

}

