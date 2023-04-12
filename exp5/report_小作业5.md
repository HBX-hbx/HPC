# 小作业五：自动向量化与基于 `intrinsic` 的手动向量化

<center><font size=4>计04 何秉翔 2020010944</font></center>

### 1. 运行时间

|                  | `baseline` | `auto simd` | `intrinsic` |
| :--------------: | :--------: | :---------: | :---------: |
| `a + b` 运行时间 | $4744\ us$ |  $546\ us$  |  $528\ us$  |

### 2. 实现代码

由于只需考虑 $8$ 整除 $n$ 的情况，于是只需将 $n$ 分成 $\frac n 8$ 段，每段分别：

+ 取出待加的段中的 $8$ 个元素，构成一个 $256$ 位的向量寄存器
+ 调用 `_mm256_add_ps` 进行向量加
+ 将得到的结果存入结果数组的对应位置

代码如下：

```c++
void a_plus_b_intrinsic(float* a, float* b, float* c, int n) {
    // Assume that 8 | n
    for (int i = 0; i < n; i += 8) {
        __m256 tmp_a = _mm256_load_ps(a + i);
        __m256 tmp_b = _mm256_load_ps(b + i);
        __m256 add_res = _mm256_add_ps(tmp_a, tmp_b);
        _mm256_store_ps(c + i, add_res);
    }
}
```

