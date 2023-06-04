#ifndef SpMM_OPT_H
#define SpMM_OPT_H
#include "spmm_base.h"
#include <vector>

class SpMMOpt : public SpMM
{
public:
    SpMMOpt(int *dev_out_ptr, int *dev_out_idx, int out_num_v, int out_num_e, int out_feat_in) : SpMM(dev_out_ptr, dev_out_idx, out_num_v, out_num_e, out_feat_in) {}
    SpMMOpt(CSR *g, int out_feat_in) : SpMM(g, out_feat_in) {}
    ~SpMMOpt() {
        if (target) checkCudaErrors(cudaFree(target));
        if (ptr_scheduled) checkCudaErrors(cudaFree(ptr_scheduled));
    }
     
    virtual void preprocess(float *vin, float *vout);

    virtual void run(float *vin, float *vout);

    void get_begin_and_end();

    void edgesort();

    void neighbor_grouping(int neighbor_num);

private:
    int thread_cnt;
    int *d_permutation;
    int *d_cnt_nnzs_per_thread; // the number of nnzs per thread
    int *d_begin_row, *d_end_row; // for each warp i, deal with the row in [begin_row[i], end_row[i])
    // non used
    int num_target;
    int *target, *ptr_scheduled;
};
#endif
