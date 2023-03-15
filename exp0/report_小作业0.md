<center><font size=6>小作业零：pow_a</font></center>

<center><font size=4>计04 何秉翔 2020010944</font></center>

### 1. 源代码

#### 1.1 `openmp_pow.cpp`

```c++
void pow_a(int *a, int *b, int n, int m) {
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        int x = 1;
        for (int j = 0; j < m; j++)
            x *= a[i];
        b[i] = x;
    }
}
```

#### 1.2 `mpi_pow.cpp`

```c++
void pow_a(int *a, int *b, int n, int m, int comm_sz /* 总进程数 */) {
    int local_n = n / comm_sz;
    for (int i = 0; i < local_n; ++i) {
        int x = 1;
        for (int j = 0; j < m; ++j) {
            x *= a[i];
        }
        b[i] = x;
    }
}
```

### 2. `openmp` 版本

$n=112000, \ m = 100000$：

| 线程数 |  运行时间   | 相对单线程加速比 |
| :----: | :---------: | :--------------: |
|   1    | 14024602 us |        1         |
|   7    | 2021189 us  |       6.94       |
|   14   | 1011582 us  |      13.86       |
|   28   |  510002 us  |      27.50       |

### 3. `MPI` 版本

$n=112000, \ m = 100000$：($N \times P$ 表示 $N$ 台机器，每台机器 $P$ 个进程)

| $N \times P$ |  运行时间   | 相对单进程加速比 |
| :----------: | :---------: | :--------------: |
| $1 \times 1$ | 14010015 us |        1         |
| $1 \times 7$ | 2029290 us  |       6.90       |
| $1\times 14$ | 1021839 us  |      13.71       |
| $1\times 28$ |  504872 us  |      27.75       |
| $2\times 28$ |  394619 us  |      35.50       |

