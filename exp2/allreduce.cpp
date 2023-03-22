#include <chrono>
#include <iostream>
#include <mpi.h>
#include <time.h>
#include <cstring>
#include <cmath>
#include <algorithm>

#define EPS 1e-8

namespace ch = std::chrono;

void Ring_Allreduce(void* sendbuf, void* recvbuf, int n, MPI_Comm comm, int comm_sz, int my_rank)
{
    float* _sendbuf = (float*) sendbuf;
    int p = comm_sz; // number of process
    int i = my_rank; // the i-th process [0, p)
    int block_size = (n + p - 1) / p; // number of block: p, and the block size
    float* _recvbuf = new float[block_size];
    // if (i == 0) {
    //     printf("p: %d, i: %d, block_size: %d\n", p, i, block_size);
    // }
    // stage 1: reduce-scatter
    for (int k = 0; k < p - 1; ++k) { // k: the k-th step
        int send_block_idx = (i - k + p) % p;
        int recv_block_idx = (i - k + p - 1) % p;
        int send_num = std::min(n - send_block_idx * block_size, block_size);
        int recv_num = std::min(n - recv_block_idx * block_size, block_size);
        // if (i == 0) {
        //     printf("%d-th step, send_block_idx: %d, recv_block_idx: %d, send_num: %d, recv_num: %d\n", k, send_block_idx, recv_block_idx, send_num, recv_num);
        // }
        MPI_Request req[2];
        // send block (i - k) to the i + 1 process
        MPI_Isend(
            _sendbuf + send_block_idx * block_size,
            send_num, MPI_FLOAT,
            (i + 1) % p, 0, comm, &req[0]
        );
        // recv block (i - k - 1) from i - 1 process
        MPI_Irecv(
            _recvbuf, recv_num, MPI_FLOAT, (i - 1 + p) % p, 0, comm, &req[1]
        );
        MPI_Waitall(2, req, MPI_STATUS_IGNORE);
        // add the recv buf to send buf
        for (int j = 0; j < recv_num; ++j) {
            _sendbuf[recv_block_idx * block_size + j] += _recvbuf[j];
        }
    }
    // after stage 1, block (i + 1) % p stores the sum
    // stage 2: all-gather
    for (int k = 0; k < p - 1; ++k) {
        int send_block_idx = (i + 1 - k + p) % p;
        int recv_block_idx = (i - k + p) % p;
        int send_num = std::min(n - send_block_idx * block_size, block_size);
        int recv_num = std::min(n - recv_block_idx * block_size, block_size);
        // if (i == 0) {
        //     printf("%d-th step, send_block_idx: %d, recv_block_idx: %d, send_num: %d, recv_num: %d\n", k, send_block_idx, recv_block_idx, send_num, recv_num);
        // }
        MPI_Request req[2];
        // send block (i + 1 - k) to i + 1 process
        MPI_Isend(
            _sendbuf + send_block_idx * block_size,
            send_num, MPI_FLOAT,
            (i + 1) % p, 0, comm, &req[0]
        );
        // recv block (i - k) from the i - 1 process
        MPI_Irecv(
            _recvbuf, recv_num, MPI_FLOAT, (i - 1 + p) % p, 0, comm, &req[1]
        );
        MPI_Waitall(2, req, MPI_STATUS_IGNORE);
        // bcast the recv buf to send buf
        for (int j = 0; j < recv_num; ++j) {
            _sendbuf[recv_block_idx * block_size + j] = _recvbuf[j];
        }
    }
    // copy from sendbuf to recvbuf
    memcpy(recvbuf, sendbuf, n * sizeof(float));

    delete[] _recvbuf;
}


// reduce + bcast
void Naive_Allreduce(void* sendbuf, void* recvbuf, int n, MPI_Comm comm, int comm_sz, int my_rank)
{
    MPI_Reduce(sendbuf, recvbuf, n, MPI_FLOAT, MPI_SUM, 0, comm);
    MPI_Bcast(recvbuf, n, MPI_FLOAT, 0, comm);
}

int main(int argc, char *argv[])
{
    int ITER = atoi(argv[1]);
    int n = atoi(argv[2]);
    float* mpi_sendbuf = new float[n];
    float* mpi_recvbuf = new float[n];
    float* naive_sendbuf = new float[n];
    float* naive_recvbuf = new float[n];
    float* ring_sendbuf = new float[n];
    float* ring_recvbuf = new float[n];

    const char* num_nodes_str = std::getenv("SLURM_NNODES");
    if (num_nodes_str == nullptr) {
        std::cerr << "SLURM_NNODES not set" << std::endl;
        return 1;
    }

    int N = std::atoi(num_nodes_str);

    MPI_Init(nullptr, nullptr);
    int comm_sz;
    int my_rank;
    MPI_Comm_size(MPI_COMM_WORLD, &comm_sz);
    MPI_Comm_rank(MPI_COMM_WORLD, &my_rank);
    
    srand(time(NULL) + my_rank);
    for (int i = 0; i < n; ++i)
        mpi_sendbuf[i] = static_cast <float> (rand()) / static_cast <float> (RAND_MAX);
    memcpy(naive_sendbuf, mpi_sendbuf, n * sizeof(float));
    memcpy(ring_sendbuf, mpi_sendbuf, n * sizeof(float));

    //warmup and check
    MPI_Allreduce(mpi_sendbuf, mpi_recvbuf, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    Naive_Allreduce(naive_sendbuf, naive_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
    Ring_Allreduce(ring_sendbuf, ring_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
    bool correct = true;
    for (int i = 0; i < n; ++i)
        if (abs(mpi_recvbuf[i] - ring_recvbuf[i]) > EPS)
        {
            correct = false;
            break;
        }

    if (correct)
    {
        auto beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            MPI_Allreduce(mpi_sendbuf, mpi_recvbuf, n, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
        auto end = ch::high_resolution_clock::now();
        double mpi_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms

        beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            Naive_Allreduce(naive_sendbuf, naive_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
        end = ch::high_resolution_clock::now();
        double naive_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms

        beg = ch::high_resolution_clock::now();
        for (int iter = 0; iter < ITER; ++iter)
            Ring_Allreduce(ring_sendbuf, ring_recvbuf, n, MPI_COMM_WORLD, comm_sz, my_rank);
        end = ch::high_resolution_clock::now();
        double ring_dur = ch::duration_cast<ch::duration<double>>(end - beg).count() * 1000; //ms
        
        if (my_rank == 0)
        {
            std::cout << "< ------- n: " << n << ", p: " << comm_sz << ", N: " << N << " ------- >\n";
            std::cout << "Correct." << std::endl;
            std::cout << "MPI_Allreduce:   " << mpi_dur << " ms." << std::endl;
            std::cout << "Naive_Allreduce: " << naive_dur << " ms." << std::endl;
            std::cout << "Ring_Allreduce:  " << ring_dur << " ms." << std::endl;
        }
    }
    else
        if (my_rank == 0)
            std::cout << "Wrong!" << std::endl;

    delete[] mpi_sendbuf;
    delete[] mpi_recvbuf;
    delete[] naive_sendbuf;
    delete[] naive_recvbuf;
    delete[] ring_sendbuf;
    delete[] ring_recvbuf;
    MPI_Finalize();
    return 0;
}
