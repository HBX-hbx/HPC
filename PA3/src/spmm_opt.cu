#include "spmm_opt.h"
#include <algorithm>
#include <metis.h>

#define NUM_ELE_PER_THREAD 2  // # of elements per thread

__global__ void spmm_kernel_placeholder(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int INFEATURE)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= num_v) return;
    int begin = ptr[tid], end = ptr[tid + 1];
    for (int j = 0; j < INFEATURE; ++j)
    {
        float result = 0.0f;
        for (int i = begin; i < end; ++i)
        {
            result += vin[idx[i] * INFEATURE + j] * val[i];
        }
        vout[tid * INFEATURE + j] = result;
    }
}

__global__ void spmm_kernel_opt_arxiv_32(int *ptr, int *idx, float *val, float *B, float *C, int M, int K)
{
    const int TILE_SIZE = 128;
    // Declare a shared memory array to store the row-related information of A
    __shared__ float shared_A_val[TILE_SIZE];
    __shared__ int   shared_A_idx[TILE_SIZE];

    int row_idx = blockIdx.x;
    const int j = threadIdx.x;

    const int begin = ptr[row_idx], end = ptr[row_idx + 1];

    float result = 0.0f;
    for (int tile_begin = begin; tile_begin < end; tile_begin += TILE_SIZE) {
        // Load the tile of non-zero elements into shared memory
        int tile_end = min(tile_begin + TILE_SIZE, end);
        int tile_size = tile_end - tile_begin;
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            shared_A_idx[i] = idx[tile_begin + i] * K;
            shared_A_val[i] = val[tile_begin + i];
        }
        __syncwarp();
        // Perform the computation using the tile
        for (int i = 0; i < tile_size; ++i) {
            result += B[shared_A_idx[i] + j] * shared_A_val[i];
        }
    }
    C[row_idx * K + j] = result;
}

__global__ void spmm_kernel_opt_am_32(int *ptr, int *idx, float *val, float *B, float *C, int M, int K)
{
    const int TILE_SIZE = 512;
    // Declare a shared memory array to store the row-related information of A
    __shared__ float shared_A_val[TILE_SIZE];
    __shared__ int   shared_A_idx[TILE_SIZE];

    int row_idx = blockIdx.x;
    const int j = threadIdx.x;

    const int begin = ptr[row_idx], end = ptr[row_idx + 1];

    float result = 0.0f;
    for (int tile_begin = begin; tile_begin < end; tile_begin += TILE_SIZE) {
        // Load the tile of non-zero elements into shared memory
        int tile_end = min(tile_begin + TILE_SIZE, end);
        int tile_size = tile_end - tile_begin;
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            shared_A_idx[i] = idx[tile_begin + i] * K;
            shared_A_val[i] = val[tile_begin + i];
        }
        __syncwarp();
        // Perform the computation using the tile
        for (int i = 0; i < tile_size; ++i) {
            result += B[shared_A_idx[i] + j] * shared_A_val[i];
        }
    }
    C[row_idx * K + j] = result;
}

__global__ void spmm_kernel_opt_32(int *ptr, int *idx, float *val, float *B, float *C, int M, int K)
{
    const int TILE_SIZE = 32;
    // Declare a shared memory array to store the row-related information of A
    __shared__ float shared_A_val[TILE_SIZE];
    __shared__ int   shared_A_idx[TILE_SIZE];

    int row_idx = blockIdx.x;
    const int j = threadIdx.x;

    // for (int row_idx = begin_row_idx; row_idx < end_row_idx; ++row_idx) {
        const int begin = ptr[row_idx], end = ptr[row_idx + 1];
        // printf("[tid: %d] row_idx: %d, j: %d begin: %d, end: %d\n", tid, row_idx, j, begin, end);

        float result = 0.0f;
        for (int tile_begin = begin; tile_begin < end; tile_begin += TILE_SIZE) {
            // Load the tile of non-zero elements into shared memory
            int tile_end = min(tile_begin + TILE_SIZE, end);
            int tile_size = tile_end - tile_begin;
            for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
                shared_A_idx[i] = idx[tile_begin + i] * K;
                shared_A_val[i] = val[tile_begin + i];
            }
            __syncwarp();
            // Perform the computation using the tile
            for (int i = 0; i < tile_size; ++i) {
                result += B[shared_A_idx[i] + j] * shared_A_val[i];
            }
        }
        C[row_idx * K + j] = result;
        // C[permutation[row_idx] * K + j] = result;
    // }

}

__global__ void spmm_kernel_opt_arxiv_256(int *ptr, int *idx, float *val, float *B, float *C, int M, int K)
{
    const int TILE_SIZE = 256;
    // Declare a shared memory array to store the row-related information of A
    __shared__ float shared_A_val[TILE_SIZE];
    __shared__ int   shared_A_idx[TILE_SIZE];

    const int row_idx = blockIdx.x;
    const int j       = blockIdx.y * blockDim.x + threadIdx.x;

    const int begin = ptr[row_idx], end = ptr[row_idx + 1];

    float result_1 = 0.0f;
    for (int tile_begin = begin; tile_begin < end; tile_begin += TILE_SIZE) {
        // Load the tile of non-zero elements into shared memory
        int tile_end = min(tile_begin + TILE_SIZE, end);
        int tile_size = tile_end - tile_begin;
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            shared_A_idx[i] = idx[tile_begin + i] * K;
            shared_A_val[i] = val[tile_begin + i];
        }
        __syncwarp();
        // Perform the computation using the tile
        for (int i = 0; i < tile_size; ++i) {
            float tmp_A_val = shared_A_val[i];
            int   tmp_A_idx = shared_A_idx[i];
            result_1 += B[tmp_A_idx + j] * tmp_A_val;
        }
    }
    C[row_idx * K + j]        = result_1;
}

__global__ void spmm_kernel_opt_am_256(int *ptr, int *idx, float *val, float *B, float *C, int M, int K)
{
    const int TILE_SIZE = 256;
    // Declare a shared memory array to store the row-related information of A
    __shared__ float shared_A_val[TILE_SIZE];
    __shared__ int   shared_A_idx[TILE_SIZE];

    const int row_idx = blockIdx.x;
    const int j       = ((blockIdx.y * blockDim.x) << 1) + threadIdx.x;

    const int begin = ptr[row_idx], end = ptr[row_idx + 1];

    float result_1 = 0.0f;
    float result_2 = 0.0f;

    for (int tile_begin = begin; tile_begin < end; tile_begin += TILE_SIZE) {
        // Load the tile of non-zero elements into shared memory
        int tile_end = min(tile_begin + TILE_SIZE, end);
        int tile_size = tile_end - tile_begin;
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            shared_A_idx[i] = idx[tile_begin + i] * K;
            shared_A_val[i] = val[tile_begin + i];
        }
        __syncwarp();
        // Perform the computation using the tile
        for (int i = 0; i < tile_size; ++i) {
            float tmp_A_val = shared_A_val[i];
            int   tmp_A_idx = shared_A_idx[i];
            result_1 += B[tmp_A_idx + j] * tmp_A_val;
            result_2 += B[tmp_A_idx + j + 32] * tmp_A_val;
        }
    }
    C[row_idx * K + j]        = result_1;
    C[row_idx * K + j + 32]   = result_2;
}

__global__ void spmm_kernel_opt_ddi_256(int *ptr, int *idx, float *val, float *B, float *C, int M, int K)
{
    const int TILE_SIZE = 256;
    // Declare a shared memory array to store the row-related information of A
    __shared__ float shared_A_val[TILE_SIZE];
    __shared__ int   shared_A_idx[TILE_SIZE];

    const int row_idx = blockIdx.x;
    const int j       = ((blockIdx.y * blockDim.x) << 1) + threadIdx.x;

    const int begin = ptr[row_idx], end = ptr[row_idx + 1];

    float result_1 = 0.0f;
    float result_2 = 0.0f;

    for (int tile_begin = begin; tile_begin < end; tile_begin += TILE_SIZE) {
        // Load the tile of non-zero elements into shared memory
        int tile_end = min(tile_begin + TILE_SIZE, end);
        int tile_size = tile_end - tile_begin;
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            shared_A_idx[i] = idx[tile_begin + i] << 8;
            shared_A_val[i] = val[tile_begin + i];
        }
        __syncwarp();
        // Perform the computation using the tile
        for (int i = 0; i < tile_size; ++i) {
            float tmp_A_val = shared_A_val[i];
            int   tmp_A_idx = shared_A_idx[i] + j;
            result_1 += B[tmp_A_idx] * tmp_A_val;
            result_2 += B[tmp_A_idx + 32] * tmp_A_val;
        }
    }
    // TODO: change when NUM_ELE_PER_THREAD change (NUM)
    int tmp_C_idx = (row_idx << 8) + j;
    C[tmp_C_idx]        = result_1;
    C[tmp_C_idx + 32]   = result_2;
}

__global__ void spmm_kernel_opt_wikikg2_256(int *ptr, int *idx, float *val, float *B, float *C, int M, int K)
{
    const int TILE_SIZE = 32;
    // Declare a shared memory array to store the row-related information of A
    __shared__ float shared_A_val[TILE_SIZE];
    __shared__ int   shared_A_idx[TILE_SIZE];

    const int row_idx = blockIdx.x;
    const int j       = ((blockIdx.y * blockDim.x) << 2) + threadIdx.x;

    const int begin = ptr[row_idx], end = ptr[row_idx + 1];

    float result_1 = 0.0f;
    float result_2 = 0.0f;
    float result_3 = 0.0f;
    float result_4 = 0.0f;

    for (int tile_begin = begin; tile_begin < end; tile_begin += TILE_SIZE) {
        // Load the tile of non-zero elements into shared memory
        int tile_end = min(tile_begin + TILE_SIZE, end);
        int tile_size = tile_end - tile_begin;
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            shared_A_idx[i] = idx[tile_begin + i] << 8;
            shared_A_val[i] = val[tile_begin + i];
        }
        __syncwarp();
        // Perform the computation using the tile
        for (int i = 0; i < tile_size; ++i) {
            float tmp_A_val = shared_A_val[i];
            int   tmp_A_idx = shared_A_idx[i] + j;
            result_1 += B[tmp_A_idx] * tmp_A_val;
            result_2 += B[tmp_A_idx + 32] * tmp_A_val;
            result_3 += B[tmp_A_idx + 64] * tmp_A_val;
            result_4 += B[tmp_A_idx + 96] * tmp_A_val;
        }
    }
    // TODO: change when NUM_ELE_PER_THREAD change (NUM)
    int tmp_C_idx = (row_idx << 8) + j;
    C[tmp_C_idx]        = result_1;
    C[tmp_C_idx + 32]   = result_2;
    C[tmp_C_idx + 64]   = result_3;
    C[tmp_C_idx + 96]   = result_4;
}

__global__ void spmm_kernel_opt_256(int *ptr, int *idx, float *val, float *B, float *C, int M, int K)
{
    const int TILE_SIZE = 32;
    // Declare a shared memory array to store the row-related information of A
    __shared__ float shared_A_val[TILE_SIZE];
    __shared__ int   shared_A_idx[TILE_SIZE];

    const int row_idx = blockIdx.x;
    // TODO: change when NUM_ELE_PER_THREAD change (1)
    const int j       = ((blockIdx.y * blockDim.x) << 1) + threadIdx.x;

    const int begin = ptr[row_idx], end = ptr[row_idx + 1];
    // printf("row_idx: %d, j: %d\n", row_idx, j);
    // TODO: change when NUM_ELE_PER_THREAD change (NUM)
    float result_1 = 0.0f;
    float result_2 = 0.0f;
    // float result_3 = 0.0f;
    // float result_4 = 0.0f;
    // float result_5 = 0.0f;
    // float result_6 = 0.0f;
    // float result_7 = 0.0f;
    // float result_8 = 0.0f;
    for (int tile_begin = begin; tile_begin < end; tile_begin += TILE_SIZE) {
        // Load the tile of non-zero elements into shared memory
        int tile_end = min(tile_begin + TILE_SIZE, end);
        int tile_size = tile_end - tile_begin;
        for (int i = threadIdx.x; i < tile_size; i += blockDim.x) {
            shared_A_idx[i] = idx[tile_begin + i] << 8;
            shared_A_val[i] = val[tile_begin + i];
        }
        __syncwarp();
        // Perform the computation using the tile
        for (int i = 0; i < tile_size; ++i) {
            // TODO: change when NUM_ELE_PER_THREAD change (NUM)
            float tmp_A_val = shared_A_val[i];
            int   tmp_A_idx = shared_A_idx[i] + j;
            result_1 += B[tmp_A_idx] * tmp_A_val;
            result_2 += B[tmp_A_idx + 32] * tmp_A_val;
            // result_3 += B[tmp_A_idx + 64] * tmp_A_val;
            // result_4 += B[tmp_A_idx + 96] * tmp_A_val;
            // result_5 += B[tmp_A_idx + 128] * tmp_A_val;
            // result_6 += B[tmp_A_idx + 160] * tmp_A_val;
            // result_7 += B[tmp_A_idx + 192] * tmp_A_val;
            // result_8 += B[tmp_A_idx + 224] * tmp_A_val;
        }
    }
    // TODO: change when NUM_ELE_PER_THREAD change (NUM)
    int tmp_C_idx = (row_idx << 8) + j;
    C[tmp_C_idx]        = result_1;
    C[tmp_C_idx + 32]   = result_2;
    // C[tmp_C_idx + 64]   = result_3;
    // C[tmp_C_idx + 96]   = result_4;
    // C[tmp_C_idx + 128]  = result_5;
    // C[tmp_C_idx + 160]  = result_6;
    // C[tmp_C_idx + 192]  = result_7;
    // C[tmp_C_idx + 224]  = result_8;
}

void preprocessSparseMatrix(int* ptr, int* idx, float* val, int numRows, int* newPtr, int* newIdx, float* newVal) {
    idx_t numVertices = static_cast<idx_t>(numRows);
    idx_t numEdges = static_cast<idx_t>(ptr[numRows]);

    // Allocate memory for METIS inputs and outputs
    idx_t* xadj = new idx_t[numVertices + 1];
    idx_t* adjncy = new idx_t[numEdges];
    idx_t* adjwgt = nullptr; // edge weights (optional)
    // idx_t* vwgt = nullptr; // vertex weights (optional)
    idx_t* options = nullptr; // METIS options (optional)
    idx_t* perm = new idx_t[numVertices];
    idx_t* iperm = nullptr; // inverse permutation (optional)

    // Convert CSR format to METIS input format
    for (int i = 0; i < numRows; ++i) {
        xadj[i] = static_cast<idx_t>(ptr[i]);
        for (int j = ptr[i]; j < ptr[i + 1]; ++j) {
            adjncy[j] = static_cast<idx_t>(idx[j]);
        }
    }
    xadj[numRows] = static_cast<idx_t>(ptr[numRows]);

    // Call METIS_NodeND for graph partitioning
    METIS_NodeND(&numVertices, xadj, adjncy, adjwgt, options, perm, iperm);

    // Apply permutation to reorder the sparse matrix
    for (int i = 0; i < numRows; ++i) {
        newPtr[i] = ptr[perm[i]];
        for (int j = ptr[perm[i]]; j < ptr[perm[i] + 1]; ++j) {
            newIdx[j] = idx[j];
            newVal[j] = val[j];
        }
    }
    newPtr[numRows] = ptr[numRows];

    // Clean up memory
    delete[] xadj;
    delete[] adjncy;
    delete[] perm;
}

void SpMMOpt::get_begin_and_end() {
    int M = num_v;
    int *begin_row = new int[M], *end_row = new int[M], *ptr = new int[M + 1];
    checkCudaErrors(cudaMemcpy(ptr, d_ptr, sizeof(int) * (M + 1), cudaMemcpyDeviceToHost));

    // get the maximun number of nnzs per row
    int MAX_NUM_NNZ_PER_ROW = -1;
    for (int i = 0; i < M; ++i) {
        int num_nnz = ptr[i + 1] - ptr[i]; // the nnzs in row i
        if (num_nnz > MAX_NUM_NNZ_PER_ROW) {
            MAX_NUM_NNZ_PER_ROW = num_nnz;
        }
    }

    grid.x = 0;
    int curr_row_idx = 0;
    while(curr_row_idx < M) {
        begin_row[grid.x] = curr_row_idx;
        int start_row_idx = curr_row_idx;
        curr_row_idx++;
        while (curr_row_idx < M && (ptr[curr_row_idx] - ptr[start_row_idx]) < MAX_NUM_NNZ_PER_ROW) {
            curr_row_idx++;
        }
        end_row[grid.x] = curr_row_idx;
        grid.x++;
    }

    // for(int i = 0; i < grid.x; ++i) {
    //     printf("[%d, %d)\n", begin_row[i], end_row[i]);
    // }

    checkCudaErrors(cudaMalloc2((void**)&d_begin_row, M * sizeof(int)));
    checkCudaErrors(cudaMalloc2((void**)&d_end_row, M * sizeof(int)));
    checkCudaErrors(cudaMemcpy(d_begin_row, begin_row, M * sizeof(int), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_end_row, end_row, M * sizeof(int), cudaMemcpyHostToDevice));

    delete []begin_row;
    delete []end_row;
    delete []ptr;

    // int NUM_THREADS_PER_WARP = 32;
    // thread_cnt = M * NUM_THREADS_PER_WARP;
    // int *begin_col = new int[thread_cnt], *end_col = new int[thread_cnt], *ptr = new int[M + 1];
    // checkCudaErrors(cudaMemcpy(ptr, d_ptr, sizeof(int) * (M + 1), cudaMemcpyDeviceToHost));

    // for (int row_idx = 0; row_idx < M; ++row_idx) {
    //     int nnzs_row = ptr[row_idx + 1] - ptr[row_idx]; // number of nnzs in the ith row
    //     int warp_load = (nnzs_row + NUM_THREADS_PER_WARP - 1) / NUM_THREADS_PER_WARP; // maximum number of nnzs allocated to a thread
    //     int num_threads_with_col = (nnzs_row + warp_load - 1) / warp_load;
    //     // the thread with col
    //     for (int tid_in_warp = 0; tid_in_warp < num_threads_with_col; ++tid_in_warp) {
    //         int tid = row_idx * NUM_THREADS_PER_WARP + tid_in_warp;
    //         begin_col[tid] = tid_in_warp * warp_load + ptr[row_idx];
    //         end_col[tid]   = min(ptr[row_idx + 1], (tid_in_warp + 1) * warp_load + ptr[row_idx]);
    //         // cnt_nnzs_per_thread[tid] = min(warp_load, nnzs_row - tid_in_warp * warp_load);
    //     }
    //     // the thread without col
    //     for (int tid_in_warp = num_threads_with_col; tid_in_warp < NUM_THREADS_PER_WARP; ++tid_in_warp) {
    //         int tid = row_idx * NUM_THREADS_PER_WARP + tid_in_warp;
    //         begin_col[tid] = ptr[row_idx + 1];
    //         end_col[tid]   = ptr[row_idx + 1];
    //         // cnt_nnzs_per_thread[tid] = 0;
    //     }
    // }

    // printf("thread cnt: %d\n", thread_cnt);
    // for(int i = 0; i < M; ++i) {
    //     for (int j = 0; j < NUM_THREADS_PER_WARP; ++j) {
    //         int tid = i * NUM_THREADS_PER_WARP + j;
    //         printf("[%d, %d) ", begin_col[tid], end_col[tid]);
    //     }
    //     printf("\n");
    // }

    // checkCudaErrors(cudaMalloc2((void**)&d_begin_col, thread_cnt * sizeof(int)));
    // checkCudaErrors(cudaMalloc2((void**)&d_end_col, thread_cnt * sizeof(int)));
    // checkCudaErrors(cudaMemcpy(d_begin_col, begin_col, thread_cnt * sizeof(int), cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy(d_end_col, end_col, thread_cnt * sizeof(int), cudaMemcpyHostToDevice));

    // delete []begin_col;
    // delete []end_col;
    // delete []ptr;
}

// Function to perform row permutation on CSR matrix A
void rowPermutation(int* d_ptr, int* d_idx, float* d_val, int *d_permutation, int M, int num_e) {
    int* rowWorkload = new int[M];
    int* permutation = new int[M];

    // copy A from device to host
    int *ptr   = new int[M + 1];
    int *idx   = new int[num_e];
    float *val = new float[num_e];
    checkCudaErrors(cudaMemcpy(ptr, d_ptr, sizeof(int) * (M + 1), cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(idx, d_idx, sizeof(int) * num_e, cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaMemcpy(val, d_val, sizeof(float) * num_e, cudaMemcpyDeviceToHost));

    // Calculate workload for each row
    for (int i = 1; i <= M; ++i) {
        rowWorkload[i - 1] = ptr[i] - ptr[i - 1];
        permutation[i - 1] = i - 1;
    }
    // printf("permutation: \n");
    // for (int i = 0; i < M; ++i) {
    //     printf("%d ", permutation[i]);
    // }
    // printf("\n");
    // printf("ptr: \n");
    // for (int i = 0; i <= M; ++i) {
    //     printf("%d ", ptr[i]);
    // }
    // printf("\n");
    // printf("idx: \n");
    // for (int i = 0; i < num_e; ++i) {
    //     printf("%d ", idx[i]);
    // }
    // printf("\n");
    // printf("val: \n");
    // for (int i = 0; i < num_e; ++i) {
    //     printf("%.2f ", val[i]);
    // }
    // printf("\n");

    std::sort(permutation, permutation + M,
              [&rowWorkload](int a, int b) { return rowWorkload[a] > rowWorkload[b]; });

    int* permutedPtr = new int[M + 1];
    int* permutedIdx = new int[num_e];
    float* permutedVal = new float[num_e];

    // Apply permutation to CSR matrix A
    permutedPtr[0] = 0;
    for (int i = 0; i < M; ++i) {
        permutedPtr[i + 1] = permutedPtr[i] + (ptr[permutation[i] + 1] - ptr[permutation[i]]);
        std::copy_n(&idx[ptr[permutation[i]]], ptr[permutation[i] + 1] - ptr[permutation[i]],
                    &permutedIdx[permutedPtr[i]]);
        std::copy_n(&val[ptr[permutation[i]]], ptr[permutation[i] + 1] - ptr[permutation[i]],
                    &permutedVal[permutedPtr[i]]);
    }
    // printf("permutation: \n");
    // for (int i = 0; i < M; ++i) {
    //     printf("%d ", permutation[i]);
    // }
    // printf("\n");
    // printf("permutedPtr: \n");
    // for (int i = 0; i <= M; ++i) {
    //     printf("%d ", permutedPtr[i]);
    // }
    // printf("\n");
    // printf("permutedIdx: \n");
    // for (int i = 0; i < num_e; ++i) {
    //     printf("%d ", permutedIdx[i]);
    // }
    // printf("\n");
    // printf("permutedVal: \n");
    // for (int i = 0; i < num_e; ++i) {
    //     printf("%.2f ", permutedVal[i]);
    // }
    // printf("\n");

    checkCudaErrors(cudaMemcpy(d_ptr, permutedPtr, sizeof(int) * (M + 1), cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_idx, permutedIdx, sizeof(int) * num_e, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_val, permutedVal, sizeof(float) * num_e, cudaMemcpyHostToDevice));
    checkCudaErrors(cudaMemcpy(d_permutation, permutation, sizeof(int) * M, cudaMemcpyHostToDevice));

    // Apply inverse permutation to obtain matrix C
    // for (int i = 0; i < numRows; ++i) {
    //     std::copy_n(&permutedC[i * numRows], numRows, &C[permutation[i] * numRows]);
    // }

    // Clean up dynamically allocated memory
    delete[] rowWorkload;
    delete[] permutation;
    delete[] permutedPtr;
    delete[] permutedIdx;
    delete[] permutedVal;
    delete []ptr;
    delete []idx;
    delete []val;
}

void SpMMOpt::preprocess(float *vin, float *vout)
{
    // TODO: your code
    // K:    feat_in
    // M:    num_v
    // nnz:  num_e
    // vin:  B (dense matrix)
    // vout: C (dense matrix)
    // ptr / idx / val : A (sparse matrix)
    // get_begin_and_end();
    // checkCudaErrors(cudaMemset(vout, 0, num_v * feat_in * sizeof(float)));
    // int M = num_v;
    int BLOCK_SIZE = 32;
    // get_begin_and_end();
    grid.x = num_v;

    if (feat_in == 32) {
        grid.y = 1; // 1 thread block per row
    } else if (feat_in == 256) {
        if (num_v == 169343) { // arxiv
            grid.y = (256 / 1) / BLOCK_SIZE;
        } else if (num_v == 2500604) { // wikikg2
            grid.y = (256 / 4) / BLOCK_SIZE;
        } else {
            grid.y = (256 / NUM_ELE_PER_THREAD) / BLOCK_SIZE; // 4 thread blks per row
        }
        
    }
    block.x = BLOCK_SIZE;

    // int *ptr   = new int[M + 1];
    // int *idx   = new int[num_e];
    // float *val = new float[num_e];
    // checkCudaErrors(cudaMemcpy(ptr, d_ptr, sizeof(int) * (M + 1), cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(idx, d_idx, sizeof(int) * num_e, cudaMemcpyDeviceToHost));
    // checkCudaErrors(cudaMemcpy(val, d_val, sizeof(float) * num_e, cudaMemcpyDeviceToHost));

    // int *new_ptr   = new int[M + 1];
    // int *new_idx   = new int[num_e];
    // float *new_val = new float[num_e];

    // checkCudaErrors(cudaMalloc2((void**)&d_permutation, sizeof(int) * num_v));
    // rowPermutation(d_ptr, d_idx, d_val, d_permutation, num_v, num_e);
    // preprocessSparseMatrix(ptr, idx, val, M, new_ptr, new_idx, new_val);

    // checkCudaErrors(cudaMemcpy(d_ptr, new_ptr, sizeof(int) * (M + 1), cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy(d_idx, new_idx, sizeof(int) * num_e, cudaMemcpyHostToDevice));
    // checkCudaErrors(cudaMemcpy(d_val, new_val, sizeof(float) * num_e, cudaMemcpyHostToDevice));

    // delete []ptr;
    // delete []idx;
    // delete []val;
    // delete []new_ptr;
    // delete []new_idx;
    // delete []new_val;
}

void SpMMOpt::run(float *vin, float *vout)
{
    // TODO: your code
    if (num_v == 169343) { // arxiv
        if (feat_in == 32) {
            spmm_kernel_opt_arxiv_32<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
        } else if(feat_in == 256) {
            spmm_kernel_opt_arxiv_256<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
        }
    } else if (num_v == 881680) { // am
        if (feat_in == 32) {
            spmm_kernel_opt_am_32<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
        } else if(feat_in == 256) {
            spmm_kernel_opt_am_256<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
        }
    } else if (num_v == 4267) { // ddi, optimize when feat_in == 256
        if (feat_in == 32) {
            spmm_kernel_opt_32<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
        } else if(feat_in == 256) {
            spmm_kernel_opt_ddi_256<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
        }
    } else if (num_v == 2500604) { // wikikg2
        if (feat_in == 32) {
            spmm_kernel_opt_32<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
        } else if(feat_in == 256) {
            spmm_kernel_opt_wikikg2_256<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
        }
    } else {
        if (feat_in == 32) {
            spmm_kernel_opt_32<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
        } else if(feat_in == 256) {
            spmm_kernel_opt_256<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
        }
    }
    
    // spmm_kernel_placeholder<<<grid, block>>>(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}