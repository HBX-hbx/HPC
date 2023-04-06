#include "aplusb.h"
#include <x86intrin.h>

void a_plus_b_intrinsic(float* a, float* b, float* c, int n) {
    // Assume that 8 | n
    for (int i = 0; i < n; i += 8) {
        __m256 tmp_a = _mm256_load_ps(a + i);
        __m256 tmp_b = _mm256_load_ps(b + i);
        __m256 add_res = _mm256_add_ps(tmp_a, tmp_b);
        _mm256_store_ps(c + i, add_res);
    }
}