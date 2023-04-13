#include <algorithm>
#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <mpi.h>

#include "worker.h"

// send the min val to the pred process, recv the max val from the pred process, and check
bool check_pred(float* data, float* curr_data, float* recv_data, int rank, size_t pred_block_len, size_t block_len, int nprocs_not_oor) {
  MPI_Request req[2];
  float recv_val = -1; // receive the value from the pred process

  if (rank > 0 && rank < nprocs_not_oor) {
    // printf("process %d sending data to process %d\n", rank, rank - 1);
    // printf("process %d recving data from process %d\n", rank, rank - 1);
    MPI_Isend(&data[0], 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(&recv_val, 1, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD,  &req[1]);
    MPI_Waitall(2, req, MPI_STATUS_IGNORE);

    if (recv_val - data[0] > 1e-15) {
      // reallocate data with the pred process
      // TODO: 
      MPI_Isend(data, block_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &req[0]);
      MPI_Irecv(recv_data, pred_block_len, MPI_FLOAT, rank - 1, 0, MPI_COMM_WORLD, &req[1]);
      for (size_t i = 0; i < block_len; ++i) {
        curr_data[i] = data[i]; // copy from data to curr_data
      }
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
      return true;
    }
  }
  return false;
}

// send the max val to the succ process, recv the min val from the succ process, and check
bool check_succ(float* data, float* curr_data, float* recv_data, int rank, size_t succ_block_len, size_t block_len, int nprocs_not_oor) {
  MPI_Request req[2];
  float recv_val = -1; // receive the value from the pred process

  if (rank < nprocs_not_oor - 1) {
    // printf("process %d sending data to process %d\n", rank, rank + 1);
    // printf("process %d recving data from process %d\n", rank, rank + 1);
    MPI_Isend(&data[block_len - 1], 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &req[0]);
    MPI_Irecv(&recv_val, 1, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &req[1]);
    MPI_Waitall(2, req, MPI_STATUS_IGNORE);

    if (data[block_len - 1] - recv_val > 1e-15) {
      // reallocate data with the succ process
      // TODO: 
      MPI_Isend(data, block_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &req[0]);
      MPI_Irecv(recv_data, succ_block_len, MPI_FLOAT, rank + 1, 0, MPI_COMM_WORLD, &req[1]);
      for (size_t i = 0; i < block_len; ++i) {
        curr_data[i] = data[i]; // copy from data to curr_data
      }
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
      return true;
    }
  }
  return false;
}

void Worker::sort() {
  /** Your code ... */
  // you can use variables in class Worker: 
  // n, nprocs, rank, block_len, data, last_rank, out_of_range

  // step 1: sort inside the process
  std::sort(data, data + block_len);
  // printf("< --------------- after sort ---------------- >\n");
  // for (size_t i = 0; i < block_len; ++i) {
  //   printf("%.2lf ", data[i]);
  // }
  // printf("\n");

  bool is_even_proc = !(rank & 1); // whether the current process is the even idx
  bool has_exchange = true; // whether the current even-odd step has data exchange
  bool* has_exchange_arr = new bool[nprocs];

  size_t block_size = ceiling(n, nprocs);
  size_t pred_block_len = 0, succ_block_len = 0;

  // ex: n = 99,  nprocs = 28, block_size = 4,  0~23: 4,  24: 3,  25~27: 0
  // ex: n = 100, nprocs = 11, block_size = 10, 0~9: 10,  10: 0
  // nprocs_not_oor: 25 / 10
  int nprocs_not_oor = ceiling(n, block_size);
  // if (rank >= nprocs_not_oor) return;

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
  // printf("rank %d, nprocs_not_oor: %d, pred_bsz: %ld, succ_bsz: %ld\n", rank, nprocs_not_oor, pred_block_len, succ_block_len);

  float *recv_data = new float[block_size];
  float *curr_data = new float[block_len]; // copy of data

  if (nprocs_not_oor > 1) {
    while(has_exchange) {
      if (is_even_proc) { // even process
        has_exchange  = check_succ(data, curr_data, recv_data, rank, succ_block_len, block_len, nprocs_not_oor); // step 2: even_part
        // printf("< --------------- after step even ---------------- >\n");
        // for (size_t i = 0; i < block_len; ++i) {
        //   printf("%.2lf ", data[i]);
        // }
        // printf("\n");
        MPI_Barrier(MPI_COMM_WORLD);
        has_exchange |= check_pred(data, curr_data, recv_data, rank, pred_block_len, block_len, nprocs_not_oor); // step 3: odd part
      } else { // odd process
        has_exchange  = check_pred(data, curr_data, recv_data, rank, pred_block_len, block_len, nprocs_not_oor); // step 2: even_part
        // printf("< --------------- after step even ---------------- >\n");
        // for (size_t i = 0; i < block_len; ++i) {
        //   printf("%.2lf ", data[i]);
        // }
        // printf("\n");
        MPI_Barrier(MPI_COMM_WORLD);
        has_exchange |= check_succ(data, curr_data, recv_data, rank, succ_block_len, block_len, nprocs_not_oor); // step 3: odd part
      }
      // check whether should stop
      MPI_Barrier(MPI_COMM_WORLD);
      // printf("< --------------- after step odd ---------------- >\n");
      // for (size_t i = 0; i < block_len; ++i) {
      //   printf("%.2lf ", data[i]);
      // }
      // printf("\n");
      // 进程 0 收集各进程的运算结果
      MPI_Gather(
        &has_exchange, 1, MPI_LOGICAL,      // send_buf_p, send_count, send_type
        has_exchange_arr, 1, MPI_LOGICAL, // recv_buf_p, recv_count, recv_type
        0, MPI_COMM_WORLD             // src_process, comm
      );
      for (int i = 0; i < nprocs_not_oor; ++i) {
        has_exchange |= has_exchange_arr[i];
      }
      MPI_Bcast(&has_exchange, 1, MPI_LOGICAL, 0, MPI_COMM_WORLD);
      MPI_Barrier(MPI_COMM_WORLD);
    }
  }

  delete []has_exchange_arr;
  delete []recv_data;
  delete []curr_data;
}
