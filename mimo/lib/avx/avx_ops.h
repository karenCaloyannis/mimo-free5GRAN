/*
 * Copyright 2023-2024 Telecom Paris

  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
 */

#ifndef USRP_MIMO_AVX_OPS_H
#define USRP_MIMO_AVX_OPS_H

#include <x86intrin.h>
#include <immintrin.h>
#include <iostream>
#include "../variables/variables.h"

#if defined(__AVX2__)
/** Multiplies complex<float> together
 *
 * */
__m256 inline multiply_complex_float(__m256 vec1, __m256 vec2) {
    __m256 conj_vec = {1, -1, 1, -1, 1, -1, 1, -1};
    return _mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(_mm256_mul_ps(vec1, vec2), conj_vec),
                                            _mm256_mul_ps(vec1, _mm256_permute_ps(vec2, 0b10110001))),0b11011000);
}
#endif
#if defined(__AVX512__)
__m512 inline multiply_complex_float(__m512 vec1, __m512 vec2) {
    __m512 temp1 = _mm512_mul_ps(vec1, vec2);
    __m512 temp2 = _mm512_mul_ps(vec1, _mm512_permute_ps(vec2, 0b10110001));

    temp1 = _mm512_sub_ps(temp1, _mm512_permute_ps(temp1, 0b10110001));
    temp2 = _mm512_add_ps(temp2, _mm512_permute_ps(temp2, 0b10110001));
    return _mm512_permute_ps(_mm512_shuffle_ps(temp1, temp2, 0b10001000), 0b11011000);
}
#endif

#if defined(__AVX2__)
/** Multiplies conj(complex<float>) in vec1 by complex<float> in vec2
 *
 */
__m256 inline conj_multiply_complex_float(__m256 vec1, __m256 vec2) {
    __m256 conj_vec = {1, -1, 1, -1, 1, -1, 1, -1};
    return _mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(vec1, vec2),
                                            _mm256_mul_ps(_mm256_mul_ps(vec1, _mm256_permute_ps(vec2, 0b10110001)),
                                                          conj_vec)), 0b11011000);
}
#endif

#if defined(__AVX512F__)
__m512 inline conj_multiply_complex_float(__m512 vec1, __m512 vec2) {
    __m512 temp1 = _mm512_mul_ps(vec1, vec2);
    __m512 temp2 = _mm512_mul_ps(vec1, _mm512_permute_ps(vec2,
                                                         0b10110001));
    temp1 = _mm512_add_ps(temp1, _mm512_permute_ps(temp1, 0b10110001));
    temp2 = _mm512_sub_ps(temp2, _mm512_permute_ps(temp2, 0b10110001));
    return _mm512_permute_ps(_mm512_shuffle_ps(temp1, temp2, 0b10001000), 0b11011000);
}
#endif

#if defined(__AVX2__)
/** Given a vector of complex<float> vec, computes the norm of each element and stores them in a
 *  new vector where each element is copied twice :
 *   result = norm(c1) | norm(c1) | norm(c2) | norm(c2) | norm(c3) | norm(c3) | norm(c4) | norm(c4) |
 *
 * @param vec
 * @return
 */
__m256 inline compute_norm_m256(__m256 vec) {
    __m256 temp_vec = _mm256_mul_ps(vec, vec);
    return _mm256_permute_ps(_mm256_hadd_ps(temp_vec, temp_vec), 0b11011000);
}
#endif

#if defined(__AVX512F__)
__m512 inline compute_norm_m512(__m512 vec) {
    __m512 temp_vec = _mm512_mul_ps(vec, vec);
    return _mm512_add_ps(temp_vec, _mm512_permute_ps(temp_vec, 0b10110001));
}
#endif


#endif //USRP_MIMO_AVX_OPS_H
