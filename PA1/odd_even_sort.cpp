#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

// send the min val to the pred process, recv the max val from the pred process, and check
void check_pred(float* data, float* curr_data, float* recv_data, int rank, size_t pred_block_len, size_t block_len, int nprocs_not_oor) {
  MPI_Request req[2];
  float recv_val = -1; // receive the min value from the pred process

  if (rank > 0 && rank < nprocs_not_oor) {
    MPI_Isend(&data[0], 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(&recv_val, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD,  &req[1]);
    MPI_Waitall(2, req, MPI_STATUS_IGNORE);

    if (recv_val - data[0] > 1e-15) {
      // reallocate data with the pred process
      MPI_Isend(data, block_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &req[0]);
      MPI_Irecv(recv_data, pred_block_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &req[1]);
      memcpy(curr_data, data, block_len * sizeof(float)); // copy from data to curr_data
      MPI_Waitall(2, req, MPI_STATUS_IGNORE);
      // get the last [block_len] floats from <curr_data, recv_data>, store in data
      int i = block_len - 1, j = pred_block_len - 1;
      for(int k = block_len - 1; k >= 0; --k) {
        // assert i and j will not be out of bound (block_len <= pred_block_len)
        if (curr_data[i] <= recv_data[j]) {
          data[k] = recv_data[j--];
        } else {
          data[k] = curr_data[i--];
        }
      }
    }
  }
}

// send the max val to the succ process, recv the min val from the succ process, and check
void check_succ(float* data, float* curr_data, float* recv_data, int rank, size_t succ_block_len, size_t block_len, int nprocs_not_oor) {
  MPI_Request req[2];
  float recv_val = -1; // receive the max value from the pred process

  if (rank < nprocs_not_oor - 1) {
    MPI_Isend(&data[block_len - 1], 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(&recv_val, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &req[1]);
    MPI_Waitall(2, req, MPI_STATUS_IGNORE);

    if (data[block_len - 1] - recv_val > 1e-15) {
      // reallocate data with the succ process
      MPI_Isend(data, block_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &req[0]);
      MPI_Irecv(recv_data, succ_block_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &req[1]);
      memcpy(curr_data, data, block_len * sizeof(float)); // copy from data to curr_data
      MPI_Waitall(2, req, MPI_STATUS_IGNORE);
      // get the first [block_len] floats from <curr_data, recv_data>, store in data
      size_t i = 0, j = 0, k = 0;
      while(k < block_len) {
        // block_len >= succ_block_len, consider out of bound !!!
        if (j >= succ_block_len) break;
        if (curr_data[i] <= recv_data[j]) {
          data[k++] = curr_data[i++];
        } else {
          data[k++] = recv_data[j++];
        }
      }
      while(k < block_len) {
        data[k++] = curr_data[i++];
      }
    }
  }
}

void Worker::sort() {
  /** Your code ... */
  // you can use variables in class Worker: 
  // n, nprocs, rank, block_len, data, last_rank, out_of_range

  // step 1: sort inside the process
  std::sort(data, data + block_len);
  if (out_of_range || (nprocs == 1)) return;

  bool is_even_proc = !(rank & 1); // whether the current process is the even idx

  size_t block_size = ceiling(n, nprocs);
  size_t pred_block_len = 0, succ_block_len = 0;

  // ex: n = 99,  nprocs = 28, block_size = 4,  0~23: 4,  24: 3,  25~27: 0
  // nprocs_not_oor: 25
  int nprocs_not_oor = ceiling(n, block_size); // number of process that not out of bound

  if (rank > 0 && rank < nprocs_not_oor) {
    pred_block_len = block_size;
  } else if (rank == nprocs_not_oor) {
    pred_block_len = n - block_size * (rank - 1);
  }
  if (rank < nprocs_not_oor - 2) {
    succ_block_len = block_size;
  } else if (rank == nprocs_not_oor - 2) {
    succ_block_len = n - block_size * (rank + 1);
  }

  float *recv_data = new float[block_size];
  float *curr_data = new float[block_len]; // copy of data

  for (int i = 0; i < nprocs_not_oor; i += 2) {
    if (is_even_proc) { // even process
      check_succ(data, curr_data, recv_data, rank, succ_block_len, block_len, nprocs_not_oor); // step 2: even_part
      check_pred(data, curr_data, recv_data, rank, pred_block_len, block_len, nprocs_not_oor); // step 3: odd part
    } else { // odd process
      check_pred(data, curr_data, recv_data, rank, pred_block_len, block_len, nprocs_not_oor); // step 2: even_part
      check_succ(data, curr_data, recv_data, rank, succ_block_len, block_len, nprocs_not_oor); // step 3: odd part
    }
  }
  delete []recv_data;
  delete []curr_data;
}
