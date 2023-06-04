#include "spmm_cpu_opt.h"

#define UNROLL_N 16
#define NUM_THREADS 28

void run_spmm_cpu_placeholder(int *ptr, int *idx, float *val, float *vin, float *vout, int num_v, int feat_len)
{
    // K:    feat_in
    // M:    num_v
    // nnz:  num_e
    // vin:  B (dense matrix)
    // vout: C (dense matrix)
    // ptr / idx / val : A (sparse matrix)
    for (int i = 0; i < num_v; ++i)
    {
        for (int j = ptr[i]; j < ptr[i + 1]; ++j) // the id in val / idx: [ ptr[i], ptr[i + 1] )
        {
            for (int k = 0; k < feat_len; ++k) // ith row, idx[j]th col
            {
                vout[i * feat_len + k] += vin[idx[j] * feat_len + k] * val[j];
            }
        }
    }
}

void run_spmm_cpu_arxiv_32(const int *ptr, const int *idx, const float *val, const float *B, float *C, int M, int K)
{
    // nnz: num_e
    // ptr / idx / val : A (sparse matrix)
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(guided)
    for (int i = 0; i < M; ++i) {
        for (int k = 0; k < K; k += UNROLL_N) { // ith row, idx[j]th col
            float result[UNROLL_N] = {0.0f};
            // float result_1 = 0.0f;
            // float result_2 = 0.0f;
            // float result_3 = 0.0f;
            // float result_4 = 0.0f;

            for (int j = ptr[i]; j < ptr[i + 1]; ++j) { // the id in val / idx: [ ptr[i], ptr[i + 1] )
                int tmp_idx   = idx[j] * K + k;
                float tmp_val = val[j];

                #pragma omp simd
                for (int r = 0; r < UNROLL_N; ++r) {
                    result[r] += B[tmp_idx + r] * tmp_val; // C[i][k] += val[j] * B[idx[j]][k]
                }
            }
            int tmp_idx = i * K + k;

            #pragma omp simd
            for (int r = 0; r < UNROLL_N; ++r) {
                C[tmp_idx + r] += result[r];
            }

        }
    }
}

void run_spmm_cpu_arxiv_256(const int *ptr, const int *idx, const float *val, const float *B, float *C, int M, int K)
{
    // nnz: num_e
    // ptr / idx / val : A (sparse matrix)
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(guided)
    for (int i = 0; i < M; ++i) {
        #pragma unroll(1024)
        for (int k = 0; k < K; k += 4) { // ith row, idx[j]th col
            // float result[UNROLL_N] = {0.0f};
            float result_1 = 0.0f;
            float result_2 = 0.0f;
            float result_3 = 0.0f;
            float result_4 = 0.0f;

            for (int j = ptr[i]; j < ptr[i + 1]; ++j) { // the id in val / idx: [ ptr[i], ptr[i + 1] )
                int tmp_idx   = idx[j] * K + k;
                float tmp_val = val[j];

                // #pragma omp simd
                result_1 += B[tmp_idx] * tmp_val; // C[i][k] += val[j] * B[idx[j]][k]
                result_2 += B[tmp_idx + 1] * tmp_val; // C[i][k] += val[j] * B[idx[j]][k]
                result_3 += B[tmp_idx + 2] * tmp_val; // C[i][k] += val[j] * B[idx[j]][k]
                result_4 += B[tmp_idx + 3] * tmp_val; // C[i][k] += val[j] * B[idx[j]][k]

            }
            int tmp_idx = i * K + k;

            C[tmp_idx] = result_1;
            C[tmp_idx + 1] = result_2;
            C[tmp_idx + 2] = result_3;
            C[tmp_idx + 3] = result_4;
        }
    }
}

void run_spmm_cpu(const int *ptr, const int *idx, const float *val, const float *B, float *C, int M, int K)
{
    // nnz: num_e
    // ptr / idx / val : A (sparse matrix)
    #pragma omp parallel for num_threads(NUM_THREADS) schedule(dynamic)
    for (int i = 0; i < M; ++i) {
        #pragma unroll(8)
        for (int k = 0; k < K; k += 4) { // ith row, idx[j]th col
            float result_1 = 0.0f;
            float result_2 = 0.0f;
            float result_3 = 0.0f;
            float result_4 = 0.0f;

            for (int j = ptr[i]; j < ptr[i + 1]; ++j) { // the id in val / idx: [ ptr[i], ptr[i + 1] )
                int tmp_idx   = idx[j] * K + k;
                float tmp_val = val[j];
                result_1 += B[tmp_idx] * tmp_val; // C[i][k] += val[j] * B[idx[j]][k]
                result_2 += B[tmp_idx + 1] * tmp_val;
                result_3 += B[tmp_idx + 2] * tmp_val;
                result_4 += B[tmp_idx + 3] * tmp_val;

            }
            int tmp_idx = i * K + k;
            C[tmp_idx] = result_1;
            C[tmp_idx + 1] = result_2;
            C[tmp_idx + 2] = result_3;
            C[tmp_idx + 3] = result_4;
        }
    }
}

void SpMMCPUOpt::preprocess(float *vin, float *vout)
{
}

void SpMMCPUOpt::run(float *vin, float *vout)
{
    if (num_v == 169343) { // arxiv
        if (feat_in == 32) {
            run_spmm_cpu_arxiv_32(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
        } else if (feat_in == 256) {
            run_spmm_cpu_arxiv_256(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
        }
    } else {
        run_spmm_cpu(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
    }

    // run_spmm_cpu_placeholder(d_ptr, d_idx, d_val, vin, vout, num_v, feat_in);
}
