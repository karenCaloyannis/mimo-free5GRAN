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

#include "vblast.h"

#define TIME_MEASURE 0

using namespace std;

#if defined(__AVX2__)
void vblast_zf_3_layers_avx(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                            std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                            int num_re_pdsch_,
                            std::complex<float> * equalized_symbols_,
                            int nb_rx_ports_) {

    int i, j, l;
    __m256 vec1, vec2, vec3, vec4, vec5;

    __m256 null_vec = _mm256_set1_ps(0);
    __m256 conj_vec = _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1);
    __m256 neg = _mm256_set1_ps(-1);

    __m256 dot_prods[(int) MAX_TX_PORTS * (1 + MAX_TX_PORTS)/2][2];

    __m256 hermitian_matrix[MAX_TX_PORTS][MAX_TX_PORTS];
    __m256 temp_equalized_symbols[MAX_TX_PORTS];

    for(int re = 0; re < num_re_pdsch_; re += 4) {
        /********************************** Compute hermitian matrix ***************************************/
        dot_prods[0][0] = _mm256_set1_ps(0); //00
        dot_prods[1][0] = _mm256_set1_ps(0); //01
        dot_prods[1][1] = _mm256_set1_ps(0); //01
        dot_prods[2][0] = _mm256_set1_ps(0); //02
        dot_prods[2][1] = _mm256_set1_ps(0); //02
        dot_prods[3][0] = _mm256_set1_ps(0); //11
        dot_prods[4][0] = _mm256_set1_ps(0); //12
        dot_prods[4][1] = _mm256_set1_ps(0); //12
        dot_prods[5][0] = _mm256_set1_ps(0); //22

        for (i = 0; i < nb_rx_ports_; i++) {
            /// 0,0 diag coef
            vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[i][0][re]);
            dot_prods[0][0] = _mm256_add_ps(dot_prods[0][0], _mm256_mul_ps(vec1, vec1));

            /// 0,1 coef
            vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[i][0][re]);
            vec2 = _mm256_loadu_ps((float *) &channel_coefficients_[i][1][re]);
            vec3 = _mm256_mul_ps(vec1, vec2);
            vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj_vec), 0b10110001), vec2);
            dot_prods[1][0] = _mm256_add_ps(dot_prods[1][0], vec3);
            dot_prods[1][1] = _mm256_add_ps(dot_prods[1][1], vec4);

            /// 0,2 coef
            vec2 = _mm256_loadu_ps((float *) &channel_coefficients_[i][2][re]);
            vec3 = _mm256_mul_ps(vec1, vec2);
            vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj_vec), 0b10110001), vec2);
            dot_prods[2][0] = _mm256_add_ps(dot_prods[2][0], vec3);
            dot_prods[2][1] = _mm256_add_ps(dot_prods[2][1], vec4);

            /// 1,1 diag coef
            vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[i][1][re]);
            dot_prods[3][0] = _mm256_add_ps(dot_prods[3][0], _mm256_mul_ps(vec1, vec1));

            /// 1,2 coef
            vec2 = _mm256_loadu_ps((float *) &channel_coefficients_[i][2][re]);
            vec3 = _mm256_mul_ps(vec1, vec2);
            vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj_vec), 0b10110001), vec2);
            dot_prods[4][0] = _mm256_add_ps(dot_prods[4][0], vec3);
            dot_prods[4][1] = _mm256_add_ps(dot_prods[4][1], vec4);

            /// 2,2 coef
            vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[i][2][re]);
            dot_prods[5][0] = _mm256_add_ps(dot_prods[5][0], _mm256_mul_ps(vec1, vec1));
        }
        hermitian_matrix[0][0] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[0][0], null_vec), 0b11011000);
        hermitian_matrix[0][1] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[1][0], dot_prods[1][1]), 0b11011000);
        hermitian_matrix[0][2] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[2][0], dot_prods[2][1]), 0b11011000);
        hermitian_matrix[1][1] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[3][0], null_vec), 0b11011000);
        hermitian_matrix[1][2] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[4][0], dot_prods[4][1]), 0b11011000);
        hermitian_matrix[2][2] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[5][0], null_vec), 0b11011000);

        /*************************************************************************************************************/

        /********************************** Multiply received PDSCH symbols by H^h ***********************************/
        for(i = 0; i < 3; i++) {
            dot_prods[0][0] = _mm256_set1_ps(0);
            dot_prods[0][1] = _mm256_set1_ps(0);
            for (j = 0; j < nb_rx_ports_; j++) {
                vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[j][i][re]);
                vec2 = _mm256_loadu_ps((float *) &pdsch_samples_[j][re]);
                vec3 = _mm256_mul_ps(vec1, vec2);
                vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj_vec), 0b10110001), vec2);
                dot_prods[0][0] = _mm256_add_ps(dot_prods[0][0], vec3);
                dot_prods[0][1] = _mm256_add_ps(dot_prods[0][1], vec4);
            }
            temp_equalized_symbols[i] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[0][0], dot_prods[0][1]), 0b11011000);
        }

        /*************************************************************************************************************/

        /************************************************* RDR decomp ************************************************/
        /// Compute h01 first
        hermitian_matrix[0][0] = _mm256_permute_ps(_mm256_hadd_ps(hermitian_matrix[0][0], hermitian_matrix[0][0]), 0b11011000); /// Compute the vector containing real part of h00 copied twice : re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3
        hermitian_matrix[0][1] = _mm256_div_ps(hermitian_matrix[0][1], hermitian_matrix[0][0]); /// Divide by real part of h00

        /// Compute h11
        vec1 = _mm256_mul_ps(hermitian_matrix[0][1], hermitian_matrix[0][1]);
        vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(vec1, null_vec), 0b11011000), hermitian_matrix[0][0]); /// Compute squared_norm(h01) * h00.real() with 0 in between the squared norms : norm0 | 0 | norm1 | 0 | ...
        hermitian_matrix[1][1] = _mm256_sub_ps(hermitian_matrix[1][1], vec2); /// Substract to h11 : h11 - squared_norm(h01) * h00.real();

        /// Compute h12
        /// Compute conj(h01) * h02
        vec1 = _mm256_mul_ps(hermitian_matrix[0][2], hermitian_matrix[0][1]); /// real part coefs : re0 * re1 | im0 * im1
        vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(hermitian_matrix[0][1], conj_vec), 0b10110001), hermitian_matrix[0][2]); /// imag part coefs -im0 * re 1 | re 0 * im 1
        /// Substract to h12.
        hermitian_matrix[1][2] = _mm256_sub_ps(hermitian_matrix[1][2], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));
        /// Divide by real part of h11.
        /// KEEP VECTOR UNMODIFIED : (real part of h11)
        hermitian_matrix[1][1] = _mm256_permute_ps(_mm256_hadd_ps(hermitian_matrix[1][1], hermitian_matrix[1][1]), 0b11011000); ///  this gives the vector re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3 (real part of h11)
        hermitian_matrix[1][2] = _mm256_div_ps(hermitian_matrix[1][2], hermitian_matrix[1][1]);

        /// Update h02
        hermitian_matrix[0][2] = _mm256_div_ps(hermitian_matrix[0][2], hermitian_matrix[0][0]);

        /// Third row
        /// Compute h22
        hermitian_matrix[2][2] = _mm256_sub_ps(hermitian_matrix[2][2],
                                               _mm256_mul_ps(_mm256_permute_ps(
                                                             _mm256_hadd_ps(_mm256_mul_ps(hermitian_matrix[0][2], hermitian_matrix[0][2]),
                                                                            null_vec), 0b11011000),
                                                             hermitian_matrix[0][0]
                                                             )
                                               ); /// Square norm of h02 multiplied by real part of h00
        hermitian_matrix[2][2] = _mm256_sub_ps(hermitian_matrix[2][2],
                                               _mm256_mul_ps(_mm256_permute_ps(
                                                                     _mm256_hadd_ps(
                                                                            _mm256_mul_ps(hermitian_matrix[1][2], hermitian_matrix[1][2]),
                                                                            null_vec),
                                                                     0b11011000),
                                                             hermitian_matrix[1][1]
                                                             )
                                               ); /// Square norm of h12 multiplied by real part of h11

        hermitian_matrix[2][2] = _mm256_permute_ps(_mm256_hadd_ps(hermitian_matrix[2][2], hermitian_matrix[2][2]), 0b11011000); ///  this gives the vector re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3 (real part of h11)

        /********************************* Compute in-place inverse **************************************************/
        hermitian_matrix[0][1] = _mm256_mul_ps(hermitian_matrix[0][1], neg);
        hermitian_matrix[0][2] = _mm256_mul_ps(hermitian_matrix[0][2], neg);
        hermitian_matrix[1][2] = _mm256_mul_ps(hermitian_matrix[1][2], neg);

        /// Compute coefficients in the correct order
        /// hermitian_matrix_[re][0][2] += hermitian_matrix_[re][0][1] * hermitian_matrix_[re][1][2];
        /// Multiply h01 by h12, then add to h02
        vec1 = _mm256_mul_ps(hermitian_matrix[0][1], hermitian_matrix[1][2]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj_vec); /// negate the second coef
        /// Permute h12 vector then multiply by h01 vector
        vec2 = _mm256_mul_ps(hermitian_matrix[0][1], _mm256_permute_ps(hermitian_matrix[1][2], 0b10110001));
        /// add to h02
        hermitian_matrix[0][2] = _mm256_add_ps(hermitian_matrix[0][2], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /// Copy transconj inverse R^(-1)^H in lower part of the array
        hermitian_matrix[1][0] = _mm256_mul_ps(hermitian_matrix[0][1], conj_vec);
        hermitian_matrix[2][0] = _mm256_mul_ps(hermitian_matrix[0][2], conj_vec);
        hermitian_matrix[2][1] = _mm256_mul_ps(hermitian_matrix[1][2], conj_vec);

        /******************************* Multiply preprocessed layers with inverse matrices **************************/

        /********************* *(equalized_symbols_ + 2) += hermitian_matrix_[re][2][0] * *(equalized_symbols_)
          + hermitian_matrix_[re][2][1] * *(equalized_symbols_ + 1); ****************/
        /// hermitian_matrix_[re][2][0] * *(equalized_symbols_)
        vec1 = _mm256_mul_ps(hermitian_matrix[2][0], temp_equalized_symbols[0]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj_vec); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec2 = _mm256_mul_ps(temp_equalized_symbols[0], _mm256_permute_ps(hermitian_matrix[2][0], 0b10110001));
        /// add the result to vec1
        temp_equalized_symbols[2] = _mm256_add_ps(temp_equalized_symbols[2], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /// hermitian_matrix_[re][2][1] * *(equalized_symbols_ + 1)
        vec1 = _mm256_mul_ps(hermitian_matrix[2][1], temp_equalized_symbols[1]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj_vec); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec2 = _mm256_mul_ps(temp_equalized_symbols[1], _mm256_permute_ps(hermitian_matrix[2][1], 0b10110001));
        /// add the result to vec1
        temp_equalized_symbols[2] = _mm256_add_ps(temp_equalized_symbols[2], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /******************* *(equalized_symbols_ + 1) += hermitian_matrix_[re][1][0] * *(equalized_symbols_); ********/
        /// hermitian_matrix_[re][1][0] * *(equalized_symbols_)
        vec1 = _mm256_mul_ps(hermitian_matrix[1][0], temp_equalized_symbols[0]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj_vec); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec2 = _mm256_mul_ps(temp_equalized_symbols[0], _mm256_permute_ps(hermitian_matrix[1][0], 0b10110001));
        /// add the result to vec1
        temp_equalized_symbols[1] = _mm256_add_ps(temp_equalized_symbols[1], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /********************* *(equalized_symbols_)     /= hermitian_matrix_[re][0][0].real(); ***********************/
        temp_equalized_symbols[0] = _mm256_div_ps(temp_equalized_symbols[0], hermitian_matrix[0][0]);
        /********************* *(equalized_symbols_ + 1) /= hermitian_matrix_[re][1][1].real(); ************************/
        temp_equalized_symbols[1] = _mm256_div_ps(temp_equalized_symbols[1], hermitian_matrix[1][1]);
        /********************* *(equalized_symbols_ + 2) /= hermitian_matrix_[re][2][2].real(); ************************/
        temp_equalized_symbols[2] = _mm256_div_ps(temp_equalized_symbols[2], hermitian_matrix[2][2]);

        /*************************** *(equalized_symbols_)     += hermitian_matrix_[re][0][1] * *(equalized_symbols_ + 1)
                + hermitian_matrix_[re][0][2] * *(equalized_symbols_ + 2)
                + hermitian_matrix_[re][0][3] * *(equalized_symbols_ + 3); *********************************************/
        /// hermitian_matrix_[re][0][1] * *(equalized_symbols_ + 1)
        vec1 = _mm256_mul_ps(hermitian_matrix[0][1], temp_equalized_symbols[1]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj_vec); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec2 = _mm256_mul_ps(temp_equalized_symbols[1], _mm256_permute_ps(hermitian_matrix[0][1], 0b10110001));
        /// add the result to vec1
        temp_equalized_symbols[0] = _mm256_add_ps(temp_equalized_symbols[0], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /// hermitian_matrix_[re][0][2] * *(equalized_symbols_ + 2)
        vec1 = _mm256_mul_ps(hermitian_matrix[0][2], temp_equalized_symbols[2]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj_vec); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec2 = _mm256_mul_ps(temp_equalized_symbols[2], _mm256_permute_ps(hermitian_matrix[0][2], 0b10110001));
        /// add the result to vec1
        temp_equalized_symbols[0] = _mm256_add_ps(temp_equalized_symbols[0], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /************************ *(equalized_symbols_ + 1) += hermitian_matrix_[re][1][2] * *(equalized_symbols_ + 2)
                + hermitian_matrix_[re][1][3] * *(equalized_symbols_ + 3); ********************************************/
        /// hermitian_matrix_[re][1][2] * *(equalized_symbols_ + 2)
        vec1 = _mm256_mul_ps(hermitian_matrix[1][2], temp_equalized_symbols[2]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj_vec); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec2 = _mm256_mul_ps(temp_equalized_symbols[2], _mm256_permute_ps(hermitian_matrix[1][2], 0b10110001));
        /// add the result to vec1
        temp_equalized_symbols[1] = _mm256_add_ps(temp_equalized_symbols[1], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /// copy computed coefficients into the final buffer
        for(i = 0; i < 4; i++) {
            for(j = 0; j < 3; j++) {
                (equalized_symbols_ + i * 3 + j)->real(temp_equalized_symbols[j][2 * i]);
                (equalized_symbols_ + i * 3 + j)->imag(temp_equalized_symbols[j][2 * i + 1]);
            }
        }
        equalized_symbols_ += 12;
    }
}
#endif

#if defined(__AVX2__)
void vblast_zf_4_layers_avx(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                         std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                         int num_re_pdsch_,
                         std::complex<float> * equalized_symbols_,
                         int nb_rx_ports_) {

    int i, j, l;
    __m256 vec1, vec2, vec3, vec4, vec5;

    __m256 null_vec = _mm256_set1_ps(0);
    __m256 conj = _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1);
    __m256 neg = _mm256_set1_ps(-1);

    __m256 dot_prods[(int) MAX_TX_PORTS * (1 + MAX_TX_PORTS)/2][2];

    __m256 hermitian_matrix[MAX_TX_PORTS][MAX_TX_PORTS];
    __m256 temp_equalized_symbols[MAX_TX_PORTS];

        __m256i masks[4] = {
           _mm256_setr_epi32(-1, -1, 0, 0, 0, 0, 0, 0),
           _mm256_setr_epi32(0, 0, -1, -1, 0, 0, 0, 0),
           _mm256_setr_epi32(0, 0, 0, 0, -1, -1, 0, 0),
           _mm256_setr_epi32(0, 0, 0, 0, 0, 0, -1, -1),
    };

    for(int re = 0; re < num_re_pdsch_; re += 4) {
        /********************************** Compute hermitian matrix ***************************************/
        dot_prods[0][0] = _mm256_set1_ps(0); //00
        dot_prods[1][0] = _mm256_set1_ps(0); //01
        dot_prods[1][1] = _mm256_set1_ps(0); //01
        dot_prods[2][0] = _mm256_set1_ps(0); //02
        dot_prods[2][1] = _mm256_set1_ps(0); //02
        dot_prods[3][0] = _mm256_set1_ps(0); //03
        dot_prods[3][1] = _mm256_set1_ps(0); //03
        dot_prods[4][0] = _mm256_set1_ps(0); //11
        dot_prods[5][0] = _mm256_set1_ps(0); //12
        dot_prods[5][1] = _mm256_set1_ps(0); //12
        dot_prods[6][0] = _mm256_set1_ps(0); //13
        dot_prods[6][1] = _mm256_set1_ps(0); //13
        dot_prods[7][0] = _mm256_set1_ps(0); //22
        dot_prods[8][0] = _mm256_set1_ps(0); //23
        dot_prods[8][1] = _mm256_set1_ps(0); //23
        dot_prods[9][0] = _mm256_set1_ps(0); //33

        for (i = 0; i < nb_rx_ports_; i++) {
            /// 0,0 diag coef
            vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[i][0][re]);
            dot_prods[0][0] = _mm256_add_ps(dot_prods[0][0], _mm256_mul_ps(vec1, vec1));

            /// 0,1 coef
            vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[i][0][re]);
            vec2 = _mm256_loadu_ps((float *) &channel_coefficients_[i][1][re]);
            vec3 = _mm256_mul_ps(vec1, vec2);
            vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec2);
            dot_prods[1][0] = _mm256_add_ps(dot_prods[1][0], vec3);
            dot_prods[1][1] = _mm256_add_ps(dot_prods[1][1], vec4);

            /// 0,2 coef
            vec2 = _mm256_loadu_ps((float *) &channel_coefficients_[i][2][re]);
            vec3 = _mm256_mul_ps(vec1, vec2);
            vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec2);
            dot_prods[2][0] = _mm256_add_ps(dot_prods[2][0], vec3);
            dot_prods[2][1] = _mm256_add_ps(dot_prods[2][1], vec4);

            /// 0,3 coef
            vec2 = _mm256_loadu_ps((float *) &channel_coefficients_[i][3][re]);
            vec3 = _mm256_mul_ps(vec1, vec2);
            vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec2);
            dot_prods[3][0] = _mm256_add_ps(dot_prods[3][0], vec3);
            dot_prods[3][1] = _mm256_add_ps(dot_prods[3][1], vec4);

            /// 1,1 diag coef
            vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[i][1][re]);
            dot_prods[4][0] = _mm256_add_ps(dot_prods[4][0], _mm256_mul_ps(vec1, vec1));

            /// 1,2 coef
            vec2 = _mm256_loadu_ps((float *) &channel_coefficients_[i][2][re]);
            vec3 = _mm256_mul_ps(vec1, vec2);
            vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec2);
            dot_prods[5][0] = _mm256_add_ps(dot_prods[5][0], vec3);
            dot_prods[5][1] = _mm256_add_ps(dot_prods[5][1], vec4);

            /// 1,3 coef
            vec2 = _mm256_loadu_ps((float *) &channel_coefficients_[i][3][re]);
            vec3 = _mm256_mul_ps(vec1, vec2);
            vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec2);
            dot_prods[6][0] = _mm256_add_ps(dot_prods[6][0], vec3);
            dot_prods[6][1] = _mm256_add_ps(dot_prods[6][1], vec4);

            /// 2,2 coef
            vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[i][2][re]);
            dot_prods[7][0] = _mm256_add_ps(dot_prods[7][0], _mm256_mul_ps(vec1, vec1));

            /// 2,3 coef
            vec2 = _mm256_loadu_ps((float *) &channel_coefficients_[i][3][re]);
            vec3 = _mm256_mul_ps(vec1, vec2);
            vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec2);
            dot_prods[8][0] = _mm256_add_ps(dot_prods[8][0], vec3);
            dot_prods[8][1] = _mm256_add_ps(dot_prods[8][1], vec4);

            /// 3,3 coef
            vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[i][3][re]);
            dot_prods[9][0] = _mm256_add_ps(dot_prods[9][0], _mm256_mul_ps(vec1, vec1));
        }

        hermitian_matrix[0][0] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[0][0], null_vec), 0b11011000);
        //hermitian_matrix[0][0] = _mm256_blend_ps(_mm256_add_ps(dot_prods[0][0], _mm256_permute_ps(dot_prods[0][0], 0b10110001)),
        //                                         null_vec, 0b10101010);
        hermitian_matrix[0][1] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[1][0], dot_prods[1][1]), 0b11011000);
        //hermitian_matrix[0][1] = _mm256_blend_ps(_mm256_add_ps(dot_prods[1][0], _mm256_permute_ps(dot_prods[1][0], 0b10110001)),
        //                _mm256_add_ps(dot_prods[1][1], _mm256_permute_ps(dot_prods[1][1], 0b10110001)),
        //                0b10101010);
        hermitian_matrix[0][2] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[2][0], dot_prods[2][1]), 0b11011000);
        //hermitian_matrix[0][2] = _mm256_blend_ps(_mm256_add_ps(dot_prods[2][0], _mm256_permute_ps(dot_prods[2][0], 0b10110001)),
        //        _mm256_add_ps(dot_prods[2][1], _mm256_permute_ps(dot_prods[2][1], 0b10110001)),
        //        0b10101010);
        hermitian_matrix[0][3] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[3][0], dot_prods[3][1]), 0b11011000);
        //hermitian_matrix[0][3] = _mm256_blend_ps(_mm256_add_ps(dot_prods[3][0], _mm256_permute_ps(dot_prods[3][0], 0b10110001)),
        //_mm256_add_ps(dot_prods[3][1], _mm256_permute_ps(dot_prods[3][1], 0b10110001)),
        //0b10101010);

        hermitian_matrix[1][1] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[4][0], null_vec), 0b11011000);
        //hermitian_matrix[1][1] = _mm256_blend_ps(_mm256_add_ps(dot_prods[4][0], _mm256_permute_ps(dot_prods[4][0], 0b10110001)),
        //                                 null_vec, 0b10101010);
        hermitian_matrix[1][2] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[5][0], dot_prods[5][1]), 0b11011000);
        hermitian_matrix[1][3] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[6][0], dot_prods[6][1]), 0b11011000);

        hermitian_matrix[2][2] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[7][0], null_vec), 0b11011000);
        //hermitian_matrix[2][2] = _mm256_blend_ps(_mm256_add_ps(dot_prods[7][0], _mm256_permute_ps(dot_prods[7][0], 0b10110001)),
        //                         null_vec, 0b10101010);
        hermitian_matrix[2][3] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[8][0], dot_prods[8][1]), 0b11011000);

        hermitian_matrix[3][3] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[9][0], null_vec), 0b11011000);
        //hermitian_matrix[3][3] = _mm256_blend_ps(_mm256_add_ps(dot_prods[9][0], _mm256_permute_ps(dot_prods[9][0], 0b10110001)),
        //                                 null_vec, 0b10101010);

        /*************************************************************************************************************/

        /********************************** Multiply received PDSCH symbols by H^h ***********************************/
        for(i = 0; i < 4; i++) {
            dot_prods[0][0] = _mm256_set1_ps(0);
            dot_prods[0][1] = _mm256_set1_ps(0);
            for (j = 0; j < nb_rx_ports_; j++) {
                vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[j][i][re]);
                vec2 = _mm256_loadu_ps((float *) &pdsch_samples_[j][re]);
                vec3 = _mm256_mul_ps(vec1, vec2);
                vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec2);
                dot_prods[0][0] = _mm256_add_ps(dot_prods[0][0], vec3);
                dot_prods[0][1] = _mm256_add_ps(dot_prods[0][1], vec4);
            }
            temp_equalized_symbols[i] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[0][0], dot_prods[0][1]), 0b11011000);
        }

        /*************************************************************************************************************/

        /************************************************* RDR decomp ************************************************/
        /// Compute h01 first
        hermitian_matrix[0][0] = _mm256_permute_ps(_mm256_hadd_ps(hermitian_matrix[0][0], hermitian_matrix[0][0]), 0b11011000); /// Compute the vector containing real part of h00 copied twice : re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3
        hermitian_matrix[0][1] = _mm256_div_ps(hermitian_matrix[0][1], hermitian_matrix[0][0]); /// Divide by real part of h00

        /// Compute h11
        vec1 = _mm256_mul_ps(hermitian_matrix[0][1], hermitian_matrix[0][1]);
        vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(vec1, null_vec), 0b11011000), hermitian_matrix[0][0]); /// Compute squared_norm(h01) * h00.real() with 0 in between the squared norms : norm0 | 0 | norm1 | 0 | ...
        hermitian_matrix[1][1] = _mm256_sub_ps(hermitian_matrix[1][1], vec2); /// Substract to h11 : h11 - squared_norm(h01) * h00.real();

        /// Compute h12
        /// Compute conj(h01) * h02
        vec1 = _mm256_mul_ps(hermitian_matrix[0][2], hermitian_matrix[0][1]); /// real part coefs : re0 * re1 | im0 * im1
        vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(hermitian_matrix[0][1], conj), 0b10110001), hermitian_matrix[0][2]); /// imag part coefs -im0 * re 1 | re 0 * im 1
        /// Substract to h12.
        hermitian_matrix[1][2] = _mm256_sub_ps(hermitian_matrix[1][2], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));
        /// Divide by real part of h11.
        /// KEEP VECTOR UNMODIFIED : (real part of h11)
        hermitian_matrix[1][1] = _mm256_permute_ps(_mm256_hadd_ps(hermitian_matrix[1][1], hermitian_matrix[1][1]), 0b11011000); ///  this gives the vector re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3 (real part of h11)
        hermitian_matrix[1][2] = _mm256_div_ps(hermitian_matrix[1][2], hermitian_matrix[1][1]);

        /// Compute h13
        /// Compute conj(h01) * h03
        vec1 = _mm256_mul_ps(hermitian_matrix[0][3], hermitian_matrix[0][1]); /// real part coefs : re0 * re1 | im0 * im1
        vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(hermitian_matrix[0][1], conj), 0b10110001), hermitian_matrix[0][3]); /// imag part coefs -im0 * re 1 | re 0 * im 1
        /// Substract to h13.
        hermitian_matrix[1][3] = _mm256_sub_ps(hermitian_matrix[1][3], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /// Update h02
        hermitian_matrix[0][2] = _mm256_div_ps(hermitian_matrix[0][2], hermitian_matrix[0][0]);

        /// Third row
        /// Compute h22
        hermitian_matrix[2][2] = _mm256_sub_ps(hermitian_matrix[2][2],
                                               _mm256_mul_ps(_mm256_permute_ps(
                                                             _mm256_hadd_ps(_mm256_mul_ps(hermitian_matrix[0][2], hermitian_matrix[0][2]),
                                                                            null_vec), 0b11011000),
                                                             hermitian_matrix[0][0]
                                                             )
                                               ); /// Square norm of h02 multiplied by real part of h00
        hermitian_matrix[2][2] = _mm256_sub_ps(hermitian_matrix[2][2],
                                               _mm256_mul_ps(_mm256_permute_ps(
                                                                     _mm256_hadd_ps(
                                                                            _mm256_mul_ps(hermitian_matrix[1][2], hermitian_matrix[1][2]),
                                                                            null_vec),
                                                                     0b11011000),
                                                             hermitian_matrix[1][1]
                                                             )
                                               ); /// Square norm of h12 multiplied by real part of h11
        /// Compute h23
        /// Compute conj(h02) * h03
        vec1 = _mm256_mul_ps(hermitian_matrix[0][2], hermitian_matrix[0][3]);
        vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(hermitian_matrix[0][2], conj), 0b10110001), hermitian_matrix[0][3]);
        hermitian_matrix[2][3] = _mm256_sub_ps(hermitian_matrix[2][3], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));
        /// Compute conj(h12) * h13
        vec1 = _mm256_mul_ps(hermitian_matrix[1][2], hermitian_matrix[1][3]);
        vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(hermitian_matrix[1][2], conj), 0b10110001), hermitian_matrix[1][3]);
        hermitian_matrix[2][3] = _mm256_sub_ps(hermitian_matrix[2][3], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));
        /// Divide by real part of h22
        hermitian_matrix[2][2] = _mm256_permute_ps(_mm256_hadd_ps(hermitian_matrix[2][2], hermitian_matrix[2][2]), 0b11011000); /// KEEP UNMODIFIED (contains real part of h22 : re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3
        hermitian_matrix[2][3] = _mm256_div_ps(hermitian_matrix[2][3], hermitian_matrix[2][2]);

        /// Update h03 and h13
        hermitian_matrix[0][3] = _mm256_div_ps(hermitian_matrix[0][3], hermitian_matrix[0][0]);
        /// Divide by real part of h11.
        hermitian_matrix[1][3] = _mm256_div_ps(hermitian_matrix[1][3], hermitian_matrix[1][1]); ///  _mm256_hadd_ps(vec4, vec4) gives the vector re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3 (real part of h11)

        /// Fourth row
        hermitian_matrix[3][3] = _mm256_sub_ps(hermitian_matrix[3][3], _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(hermitian_matrix[0][3], hermitian_matrix[0][3]), null_vec), 0b11011000), hermitian_matrix[0][0])); /// Square norm of h03 multiplied by real part of h00
        hermitian_matrix[3][3] = _mm256_sub_ps(hermitian_matrix[3][3], _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(hermitian_matrix[1][3], hermitian_matrix[1][3]), null_vec), 0b11011000), hermitian_matrix[1][1])); /// Square norm of h13 multiplied by real part of h11
        hermitian_matrix[3][3] = _mm256_sub_ps(hermitian_matrix[3][3], _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(hermitian_matrix[2][3], hermitian_matrix[2][3]), null_vec), 0b11011000), hermitian_matrix[2][2])); /// Square norm of h23 multiplied by real part of h22

        /********************************* Compute in-place inverse **************************************************/
        hermitian_matrix[0][1] = _mm256_mul_ps(hermitian_matrix[0][1], neg);
        hermitian_matrix[0][2] = _mm256_mul_ps(hermitian_matrix[0][2], neg);
        hermitian_matrix[0][3] = _mm256_mul_ps(hermitian_matrix[0][3], neg);
        hermitian_matrix[1][2] = _mm256_mul_ps(hermitian_matrix[1][2], neg);
        hermitian_matrix[1][3] = _mm256_mul_ps(hermitian_matrix[1][3], neg);
        hermitian_matrix[2][3] = _mm256_mul_ps(hermitian_matrix[2][3], neg);

        /// Compute coefficients in the correct order
        /// hermitian_matrix_[re][0][2] += hermitian_matrix_[re][0][1] * hermitian_matrix_[re][1][2];
        /// Multiply h01 by h12, then add to h02
        vec1 = _mm256_mul_ps(hermitian_matrix[0][1], hermitian_matrix[1][2]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj); /// negate the second coef
        /// Permute h12 vector then multiply by h01 vector
        vec2 = _mm256_mul_ps(hermitian_matrix[0][1], _mm256_permute_ps(hermitian_matrix[1][2], 0b10110001));
        /// add to h02
        hermitian_matrix[0][2] = _mm256_add_ps(hermitian_matrix[0][2], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /// Compute hermitian_matrix_[re][0][3] += hermitian_matrix_[re][0][1] * hermitian_matrix_[re][1][3] + hermitian_matrix_[re][0][2] * hermitian_matrix_[re][2][3];
        /// Multiply h01 by h12, then add to h02
        vec1 = _mm256_mul_ps(hermitian_matrix[0][1], hermitian_matrix[1][3]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj); /// negate the second coef
        /// Permute h12 vector then multiply by h01 vector
        vec2 = _mm256_mul_ps(hermitian_matrix[0][1], _mm256_permute_ps(hermitian_matrix[1][3], 0b10110001));
        /// add to h02
        hermitian_matrix[0][3] = _mm256_add_ps(hermitian_matrix[0][3], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));
        /// Multiply h02 by h23, then add to h02
        vec1 = _mm256_mul_ps(hermitian_matrix[0][2], hermitian_matrix[2][3]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj); /// negate the second coef
        /// Permute h12 vector then multiply by h01 vector
        vec2 = _mm256_mul_ps(hermitian_matrix[0][2], _mm256_permute_ps(hermitian_matrix[2][3], 0b10110001));
        /// add to h02
        hermitian_matrix[0][3] = _mm256_add_ps(hermitian_matrix[0][3], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /// Compute hermitian_matrix_[re][1][3] += hermitian_matrix_[re][1][2] * hermitian_matrix_[re][2][3];
        /// Multiply h01 by h12, then add to h02
        vec1 = _mm256_mul_ps(hermitian_matrix[1][2], hermitian_matrix[2][3]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj); /// negate the second coef
        /// Permute h12 vector then multiply by h01 vector
        vec2 = _mm256_mul_ps(hermitian_matrix[1][2], _mm256_permute_ps(hermitian_matrix[2][3], 0b10110001));
        /// add to h02
        hermitian_matrix[1][3] = _mm256_add_ps(hermitian_matrix[1][3], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /// Copy transconj inverse R^(-1)^H in lower part of the array
        hermitian_matrix[1][0] = _mm256_mul_ps(hermitian_matrix[0][1], conj);
        hermitian_matrix[2][0] = _mm256_mul_ps(hermitian_matrix[0][2], conj);
        hermitian_matrix[2][1] = _mm256_mul_ps(hermitian_matrix[1][2], conj);
        hermitian_matrix[3][0] = _mm256_mul_ps(hermitian_matrix[0][3], conj);
        hermitian_matrix[3][1] = _mm256_mul_ps(hermitian_matrix[1][3], conj);
        hermitian_matrix[3][2] = _mm256_mul_ps(hermitian_matrix[2][3], conj);

        /******************************* Multiply preprocessed layers with inverse matrices **************************/

        /***************(equalized_symbols_ + 3) += hermitian_matrix_[re][3][0] * *(equalized_symbols_)
                + hermitian_matrix_[re][3][1] * *(equalized_symbols_ + 1)
                + hermitian_matrix_[re][3][2] * *(equalized_symbols_ + 2); *******************************************/

        /// hermitian_matrix_[re][3][0] * *(equalized_symbols_)
        vec1 = _mm256_mul_ps(hermitian_matrix[3][0], temp_equalized_symbols[0]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec2 = _mm256_mul_ps(temp_equalized_symbols[0], _mm256_permute_ps(hermitian_matrix[3][0], 0b10110001));
        /// add the result to vec1
        temp_equalized_symbols[3] = _mm256_add_ps(temp_equalized_symbols[3], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /// hermitian_matrix_[re][3][1] * *(equalized_symbols_ + 1)
        vec1 = _mm256_mul_ps(hermitian_matrix[3][1], temp_equalized_symbols[1]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec2 = _mm256_mul_ps(temp_equalized_symbols[1], _mm256_permute_ps(hermitian_matrix[3][1], 0b10110001));
        /// add the result to vec1
        temp_equalized_symbols[3] = _mm256_add_ps(temp_equalized_symbols[3], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /// hermitian_matrix_[re][3][2] * *(equalized_symbols_ + 2)
        vec1 = _mm256_mul_ps(hermitian_matrix[3][2], temp_equalized_symbols[2]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec2 = _mm256_mul_ps(temp_equalized_symbols[2], _mm256_permute_ps(hermitian_matrix[3][2], 0b10110001));
        /// add the result to vec1
        temp_equalized_symbols[3] = _mm256_add_ps(temp_equalized_symbols[3], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /********************* *(equalized_symbols_ + 2) += hermitian_matrix_[re][2][0] * *(equalized_symbols_)
          + hermitian_matrix_[re][2][1] * *(equalized_symbols_ + 1); ****************/
        /// hermitian_matrix_[re][2][0] * *(equalized_symbols_)
        vec1 = _mm256_mul_ps(hermitian_matrix[2][0], temp_equalized_symbols[0]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec2 = _mm256_mul_ps(temp_equalized_symbols[0], _mm256_permute_ps(hermitian_matrix[2][0], 0b10110001));
        /// add the result to vec1
        temp_equalized_symbols[2] = _mm256_add_ps(temp_equalized_symbols[2], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /// hermitian_matrix_[re][2][1] * *(equalized_symbols_ + 1)
        vec1 = _mm256_mul_ps(hermitian_matrix[2][1], temp_equalized_symbols[1]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec2 = _mm256_mul_ps(temp_equalized_symbols[1], _mm256_permute_ps(hermitian_matrix[2][1], 0b10110001));
        /// add the result to vec1
        temp_equalized_symbols[2] = _mm256_add_ps(temp_equalized_symbols[2], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /******************* *(equalized_symbols_ + 1) += hermitian_matrix_[re][1][0] * *(equalized_symbols_); ********/
        /// hermitian_matrix_[re][1][0] * *(equalized_symbols_)
        vec1 = _mm256_mul_ps(hermitian_matrix[1][0], temp_equalized_symbols[0]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec2 = _mm256_mul_ps(temp_equalized_symbols[0], _mm256_permute_ps(hermitian_matrix[1][0], 0b10110001));
        /// add the result to vec1
        temp_equalized_symbols[1] = _mm256_add_ps(temp_equalized_symbols[1], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /********************* *(equalized_symbols_)     /= hermitian_matrix_[re][0][0].real(); ***********************/
        temp_equalized_symbols[0] = _mm256_div_ps(temp_equalized_symbols[0], hermitian_matrix[0][0]);
        /********************* *(equalized_symbols_ + 1) /= hermitian_matrix_[re][1][1].real(); ************************/
        temp_equalized_symbols[1] = _mm256_div_ps(temp_equalized_symbols[1], hermitian_matrix[1][1]);
        /********************* *(equalized_symbols_ + 2) /= hermitian_matrix_[re][2][2].real(); ************************/
        temp_equalized_symbols[2] = _mm256_div_ps(temp_equalized_symbols[2], hermitian_matrix[2][2]);
        /********************* *(equalized_symbols_ + 3) /= hermitian_matrix_[re][3][3].real(); ************************/
        temp_equalized_symbols[3] = _mm256_div_ps(temp_equalized_symbols[3], _mm256_permute_ps(_mm256_hadd_ps(hermitian_matrix[3][3], hermitian_matrix[3][3]), 0b11011000));

        /*************************** *(equalized_symbols_)     += hermitian_matrix_[re][0][1] * *(equalized_symbols_ + 1)
                + hermitian_matrix_[re][0][2] * *(equalized_symbols_ + 2)
                + hermitian_matrix_[re][0][3] * *(equalized_symbols_ + 3); *********************************************/
        /// hermitian_matrix_[re][0][1] * *(equalized_symbols_ + 1)
        vec1 = _mm256_mul_ps(hermitian_matrix[0][1], temp_equalized_symbols[1]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec2 = _mm256_mul_ps(temp_equalized_symbols[1], _mm256_permute_ps(hermitian_matrix[0][1], 0b10110001));
        /// add the result to vec1
        temp_equalized_symbols[0] = _mm256_add_ps(temp_equalized_symbols[0], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /// hermitian_matrix_[re][0][2] * *(equalized_symbols_ + 2)
        vec1 = _mm256_mul_ps(hermitian_matrix[0][2], temp_equalized_symbols[2]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec2 = _mm256_mul_ps(temp_equalized_symbols[2], _mm256_permute_ps(hermitian_matrix[0][2], 0b10110001));
        /// add the result to vec1
        temp_equalized_symbols[0] = _mm256_add_ps(temp_equalized_symbols[0], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /// hermitian_matrix_[re][0][3] * *(equalized_symbols_ + 3)
        vec1 = _mm256_mul_ps(hermitian_matrix[0][3], temp_equalized_symbols[3]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec2 = _mm256_mul_ps(temp_equalized_symbols[3], _mm256_permute_ps(hermitian_matrix[0][3], 0b10110001));
        /// add the result to vec1
        temp_equalized_symbols[0] = _mm256_add_ps(temp_equalized_symbols[0], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /************************ *(equalized_symbols_ + 1) += hermitian_matrix_[re][1][2] * *(equalized_symbols_ + 2)
                + hermitian_matrix_[re][1][3] * *(equalized_symbols_ + 3); ********************************************/
        /// hermitian_matrix_[re][1][2] * *(equalized_symbols_ + 2)
        vec1 = _mm256_mul_ps(hermitian_matrix[1][2], temp_equalized_symbols[2]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec2 = _mm256_mul_ps(temp_equalized_symbols[2], _mm256_permute_ps(hermitian_matrix[1][2], 0b10110001));
        /// add the result to vec1
        temp_equalized_symbols[1] = _mm256_add_ps(temp_equalized_symbols[1], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /// hermitian_matrix_[re][1][3] * *(equalized_symbols_ + 3)
        vec1 = _mm256_mul_ps(hermitian_matrix[1][3], temp_equalized_symbols[3]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec2 = _mm256_mul_ps(temp_equalized_symbols[3], _mm256_permute_ps(hermitian_matrix[1][3], 0b10110001));
        /// add the result to vec1
        temp_equalized_symbols[1] = _mm256_add_ps(temp_equalized_symbols[1], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /************************ *(equalized_symbols_ + 2) += hermitian_matrix_[re][2][3] * *(equalized_symbols_ + 3); *******/
        /// hermitian_matrix_[re][2][3] * *(equalized_symbols_ + 3)
        vec1 = _mm256_mul_ps(hermitian_matrix[2][3], temp_equalized_symbols[3]); /// real part coefficients re0 * re1 | im0 * im1
        vec1 = _mm256_mul_ps(vec1, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec2 = _mm256_mul_ps(temp_equalized_symbols[3], _mm256_permute_ps(hermitian_matrix[2][3], 0b10110001));
        /// add the result to vec1
        temp_equalized_symbols[2] = _mm256_add_ps(temp_equalized_symbols[2], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /// copy computed coefficients into the final buffer
        /*
        cout << "temp equalized symbols : " << endl;
        for(i = 0; i < 8; i+=2) {
            cout << temp_equalized_symbols[0][i] << ", " << temp_equalized_symbols[0][i + 1] << endl;
        } */
        //for(j = 0; j < 4; j++) { /// RE
            //for(i = 0; i < 4; i++) { /// layer
                //_mm256_maskstore_ps((float *) &equalized_symbols_[j * 4 + i - j], masks[j], temp_equalized_symbols[i]);
                //cout << "j * 4 + i : " << j * 4 + i << endl;
                //(equalized_symbols_ + j * 4 + i)->real(temp_equalized_symbols[i][2 * j]);
                //(equalized_symbols_ + j * 4 + i)->imag(temp_equalized_symbols[i][2 * j + 1]);
            //}
        //}
        for(i = 0; i < 4; i++) {
            _mm256_maskstore_ps((float *) &equalized_symbols_[0 + i], masks[0], temp_equalized_symbols[i]);
            _mm256_maskstore_ps((float *) &equalized_symbols_[4 + i - 1], masks[1], temp_equalized_symbols[i]);
            _mm256_maskstore_ps((float *) &equalized_symbols_[8 + i - 2], masks[2], temp_equalized_symbols[i]);
            _mm256_maskstore_ps((float *) &equalized_symbols_[12 + i - 3], masks[3], temp_equalized_symbols[i]);
        }

        /*
        cout << "eq symbols : " << endl;
        for(i = 0; i < 16; i++) {
            cout << equalized_symbols_[i] << endl;
        } */

        equalized_symbols_ += 16;
    }
}
#endif

#if defined(__AVX2__)
void multiply_by_transconj_avx(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                           std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                           int num_re_pdsch_,
                           std::vector<std::complex<float>> equalized_symbols_[MAX_TX_PORTS],
                           int nb_tx_dmrs_ports_,
                           int nb_rx_ports_) {

    int i, j, l;
    __m256 vec1, vec2, vec3, vec4, res, dot_prod_re, dot_prod_im; //dot_prod;
    __m256 null_vec = _mm256_set1_ps(0);
    __m256 conj = _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1);
    for(int re = 0; re < num_re_pdsch_; re += 4) {
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            //dot_prod = _mm256_set1_ps(0);
            dot_prod_re = _mm256_set1_ps(0);
            dot_prod_im = _mm256_set1_ps(0);
            for (j = 0; j < nb_rx_ports_; j++) {
                vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[j][i][re]);
                vec2 = _mm256_loadu_ps((float *) &pdsch_samples_[j][re]);
                vec3 = _mm256_mul_ps(vec1, vec2);
                vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec2);
                dot_prod_re = _mm256_add_ps(dot_prod_re, vec3);
                dot_prod_im = _mm256_add_ps(dot_prod_im, vec4);
                //res = _mm256_permute_ps(_mm256_hadd_ps(vec3, vec4), 0b11011000);
                //dot_prod = _mm256_add_ps(dot_prod, res);
            }
            _mm256_storeu_ps((float *) &equalized_symbols_[i][re], _mm256_permute_ps(_mm256_hadd_ps(dot_prod_re, dot_prod_im), 0b11011000));//dot_prod);
            //_mm256_storeu_ps((float *) &equalized_symbols_[i][re], _mm256_blend_ps(_mm256_add_ps(dot_prod_re, _mm256_permute_ps(dot_prod_re, 0b10110001)),
            //                                                                       _mm256_add_ps(dot_prod_im, _mm256_permute_ps(dot_prod_im, 0b10110001)),
            //                                                                       0b10101010));
        }

    }
}
#endif

void multiply_by_transconj(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                           std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                           int num_re_pdsch_,
                           std::complex<float> * equalized_symbols_,
                           int nb_tx_dmrs_ports_,
                           int nb_rx_ports_) {

    int i, j;
    for(int re = 0; re < num_re_pdsch_; re++) {
        /// Multiply received signal y by H^H, then multiply by the inverse
        for(i = 0; i < 4; i++) {
            *(equalized_symbols_ + i) = conj(channel_coefficients_[0][i][re]) * pdsch_samples_[0][re];
        }
        for(j = 1; j < nb_rx_ports_; j++) {
            for(i = 0; i < 4; i++) {
                *(equalized_symbols_ + i) += conj(channel_coefficients_[j][i][re]) * pdsch_samples_[j][re];
            }
        }
        equalized_symbols_ += 4;
    }
}

#if defined(__AVX2__)
void compute_hermitian_matrix_avx(std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                              std::vector<std::complex<float>> hermitian_matrix[MAX_TX_PORTS][MAX_TX_PORTS],
                              int num_re_pdsch_,
                              int nb_tx_dmrs_ports_,
                              int nb_rx_ports_) {
    int i, j;
    __m256 vec1, vec2, vec3, vec4, res; //dot_prod_re, dot_prod_im; //dot_prod,
    __m256 null_vec = _mm256_set1_ps(0);
    __m256 conj = _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1);

    __m256 dot_prods[(int) MAX_TX_PORTS * (1 + MAX_TX_PORTS)/2][2];

    //__m256 vec5, vec6;

    for (int re = 0; re < num_re_pdsch_; re += 4) {

        dot_prods[0][0] = _mm256_set1_ps(0); //00
        dot_prods[1][0] = _mm256_set1_ps(0); //01
        dot_prods[1][1] = _mm256_set1_ps(0); //01
        dot_prods[2][0] = _mm256_set1_ps(0); //02
        dot_prods[2][1] = _mm256_set1_ps(0); //02
        dot_prods[3][0] = _mm256_set1_ps(0); //03
        dot_prods[3][1] = _mm256_set1_ps(0); //03
        dot_prods[4][0] = _mm256_set1_ps(0); //11
        dot_prods[5][0] = _mm256_set1_ps(0); //12
        dot_prods[5][1] = _mm256_set1_ps(0); //12
        dot_prods[6][0] = _mm256_set1_ps(0); //13
        dot_prods[6][1] = _mm256_set1_ps(0); //13
        dot_prods[7][0] = _mm256_set1_ps(0); //22
        dot_prods[8][0] = _mm256_set1_ps(0); //23
        dot_prods[8][1] = _mm256_set1_ps(0); //23
        dot_prods[9][0] = _mm256_set1_ps(0); //33

        for (i = 0; i < nb_rx_ports_; i++) {
            /// 0,0 diag coef
            vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[i][0][re]);
            dot_prods[0][0] = _mm256_add_ps(dot_prods[0][0], _mm256_mul_ps(vec1, vec1));

            /// 0,1 coef
            vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[i][0][re]);
            vec2 = _mm256_loadu_ps((float *) &channel_coefficients_[i][1][re]);
            vec3 = _mm256_mul_ps(vec1, vec2);
            vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec2);
            dot_prods[1][0] = _mm256_add_ps(dot_prods[1][0], vec3);
            dot_prods[1][1] = _mm256_add_ps(dot_prods[1][1], vec4);

            /// 0,2 coef
            vec2 = _mm256_loadu_ps((float *) &channel_coefficients_[i][2][re]);
            vec3 = _mm256_mul_ps(vec1, vec2);
            vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec2);
            dot_prods[2][0] = _mm256_add_ps(dot_prods[2][0], vec3);
            dot_prods[2][1] = _mm256_add_ps(dot_prods[2][1], vec4);

            /// 0,3 coef
            vec2 = _mm256_loadu_ps((float *) &channel_coefficients_[i][3][re]);
            vec3 = _mm256_mul_ps(vec1, vec2);
            vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec2);
            dot_prods[3][0] = _mm256_add_ps(dot_prods[3][0], vec3);
            dot_prods[3][1] = _mm256_add_ps(dot_prods[3][1], vec4);

            /// 1,1 diag coef
            vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[i][1][re]);
            dot_prods[4][0] = _mm256_add_ps(dot_prods[4][0], _mm256_mul_ps(vec1, vec1));

            /// 1,2 coef
            vec2 = _mm256_loadu_ps((float *) &channel_coefficients_[i][2][re]);
            vec3 = _mm256_mul_ps(vec1, vec2);
            vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec2);
            dot_prods[5][0] = _mm256_add_ps(dot_prods[5][0], vec3);
            dot_prods[5][1] = _mm256_add_ps(dot_prods[5][1], vec4);

            /// 1,3 coef
            vec2 = _mm256_loadu_ps((float *) &channel_coefficients_[i][3][re]);
            vec3 = _mm256_mul_ps(vec1, vec2);
            vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec2);
            dot_prods[6][0] = _mm256_add_ps(dot_prods[6][0], vec3);
            dot_prods[6][1] = _mm256_add_ps(dot_prods[6][1], vec4);

            /// 2,2 coef
            vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[i][2][re]);
            dot_prods[7][0] = _mm256_add_ps(dot_prods[7][0], _mm256_mul_ps(vec1, vec1));

            /// 2,3 coef
            vec2 = _mm256_loadu_ps((float *) &channel_coefficients_[i][3][re]);
            vec3 = _mm256_mul_ps(vec1, vec2);
            vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec2);
            dot_prods[8][0] = _mm256_add_ps(dot_prods[8][0], vec3);
            dot_prods[8][1] = _mm256_add_ps(dot_prods[8][1], vec4);

            /// 3,3 coef
            vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[i][3][re]);
            dot_prods[9][0] = _mm256_add_ps(dot_prods[9][0], _mm256_mul_ps(vec1, vec1));
        }
        //vec5 = _mm256_add_ps(dot_prods[0][0], _mm256_permute_ps(dot_prods[0][0], 0b10110001));
        _mm256_storeu_ps((float *) &hermitian_matrix[0][0][re],
                         _mm256_permute_ps(_mm256_hadd_ps(dot_prods[0][0], null_vec), 0b11011000));
                         //_mm256_blend_ps(vec5, null_vec, 0b10101010));
        //vec5 = _mm256_add_ps(dot_prods[1][0], _mm256_permute_ps(dot_prods[1][0], 0b10110001));
        //vec6 = _mm256_add_ps(dot_prods[1][1], _mm256_permute_ps(dot_prods[1][1], 0b10110001));
        _mm256_storeu_ps((float *) &hermitian_matrix[0][1][re],
                         _mm256_permute_ps(_mm256_hadd_ps(dot_prods[1][0], dot_prods[1][1]), 0b11011000));
                         //_mm256_blend_ps(vec5,
                         //                vec6,
                         //                0b10101010));
        //vec5 = _mm256_add_ps(dot_prods[2][0], _mm256_permute_ps(dot_prods[2][0], 0b10110001));
        //vec6 = _mm256_add_ps(dot_prods[2][1], _mm256_permute_ps(dot_prods[2][1], 0b10110001));
        _mm256_storeu_ps((float *) &hermitian_matrix[0][2][re],
                         _mm256_permute_ps(_mm256_hadd_ps(dot_prods[2][0], dot_prods[2][1]), 0b11011000));
                             //_mm256_blend_ps(vec5,
                             //            vec6,
                             //            0b10101010));
        //vec5 = _mm256_add_ps(dot_prods[3][0], _mm256_permute_ps(dot_prods[3][0], 0b10110001));
        //vec6 = _mm256_add_ps(dot_prods[3][1], _mm256_permute_ps(dot_prods[3][1], 0b10110001));
        _mm256_storeu_ps((float *) &hermitian_matrix[0][3][re],
                         _mm256_permute_ps(_mm256_hadd_ps(dot_prods[3][0], dot_prods[3][1]), 0b11011000));
                         //  _mm256_blend_ps(vec5,
                         //                vec6,
                         //                0b10101010));
        //vec5 = _mm256_add_ps(dot_prods[4][0], _mm256_permute_ps(dot_prods[4][0], 0b10110001));
        _mm256_storeu_ps((float *) &hermitian_matrix[1][1][re],
                         _mm256_permute_ps(_mm256_hadd_ps(dot_prods[4][0], null_vec), 0b11011000));
                         //_mm256_blend_ps(vec5, null_vec, 0b10101010));
        //vec5 = _mm256_add_ps(dot_prods[5][0], _mm256_permute_ps(dot_prods[5][0], 0b10110001));
        //vec6 = _mm256_add_ps(dot_prods[5][1], _mm256_permute_ps(dot_prods[5][1], 0b10110001));
        _mm256_storeu_ps((float *) &hermitian_matrix[1][2][re],
                         _mm256_permute_ps(_mm256_hadd_ps(dot_prods[5][0], dot_prods[5][1]), 0b11011000));
                         // _mm256_blend_ps(vec5,
                         //                 vec6,
                         //                0b10101010));
        ///vec5 = _mm256_add_ps(dot_prods[6][0], _mm256_permute_ps(dot_prods[6][0], 0b10110001));
        //vec6 = _mm256_add_ps(dot_prods[6][1], _mm256_permute_ps(dot_prods[6][1], 0b10110001));
        _mm256_storeu_ps((float *) &hermitian_matrix[1][3][re],
                         _mm256_permute_ps(_mm256_hadd_ps(dot_prods[6][0], dot_prods[6][1]), 0b11011000));
                         //_mm256_blend_ps(vec5,
                         //                vec6,
                         //                0b10101010));
        _mm256_storeu_ps((float *) &hermitian_matrix[2][2][re],
                         _mm256_permute_ps(_mm256_hadd_ps(dot_prods[7][0], null_vec), 0b11011000));
                         //_mm256_blend_ps(_mm256_add_ps(dot_prods[7][0], _mm256_permute_ps(dot_prods[7][0], 0b10110001)), null_vec, 0b10101010));
        //vec5 = _mm256_add_ps(dot_prods[8][0], _mm256_permute_ps(dot_prods[8][0], 0b10110001));
        //vec6 = _mm256_add_ps(dot_prods[8][1], _mm256_permute_ps(dot_prods[8][1], 0b10110001));
        _mm256_storeu_ps((float *) &hermitian_matrix[2][3][re],
                         _mm256_permute_ps(_mm256_hadd_ps(dot_prods[8][0], dot_prods[8][1]), 0b11011000));
                         //                          _mm256_blend_ps(vec5,
                         //                vec6,
                         //                0b10101010));
        _mm256_storeu_ps((float *) &hermitian_matrix[3][3][re],
                         _mm256_permute_ps(_mm256_hadd_ps(dot_prods[9][0], null_vec), 0b11011000));
                         //_mm256_blend_ps(_mm256_add_ps(dot_prods[9][0], _mm256_permute_ps(dot_prods[9][0], 0b10110001)), null_vec, 0b10101010));
    }
}
#endif

void compute_hermitian_matrix(std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                                  std::complex<float> hermitian_matrix[][MAX_TX_PORTS][MAX_TX_PORTS],
                                  int num_re_pdsch_,
                                  int nb_tx_dmrs_ports_,
                                  int nb_rx_ports_) {

    int i, j, l;
    for(int re = 0; re < num_re_pdsch_; re ++) {

        hermitian_matrix[re][0][0].real(channel_coefficients_[0][0][re].real() * channel_coefficients_[0][0][re].real() +
                                    channel_coefficients_[0][0][re].imag() * channel_coefficients_[0][0][re].imag());
        hermitian_matrix[re][0][1] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][1][re];
        hermitian_matrix[re][0][2] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][2][re];
        hermitian_matrix[re][0][3] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][3][re];

        /// Second line from diag coef 1,1
        hermitian_matrix[re][1][1].real(channel_coefficients_[0][1][re].real() * channel_coefficients_[0][1][re].real() +
                                    channel_coefficients_[0][1][re].imag() * channel_coefficients_[0][1][re].imag());
        hermitian_matrix[re][1][2] = conj(channel_coefficients_[0][1][re]) * channel_coefficients_[0][2][re];
        hermitian_matrix[re][1][3] = conj(channel_coefficients_[0][1][re]) * channel_coefficients_[0][3][re];

        /// Third line from diag coef 2,2
        hermitian_matrix[re][2][2].real(channel_coefficients_[0][2][re].real() * channel_coefficients_[0][2][re].real() +
                                    channel_coefficients_[0][2][re].imag() * channel_coefficients_[0][2][re].imag());
        hermitian_matrix[re][2][3] = conj(channel_coefficients_[0][2][re]) * channel_coefficients_[0][3][re];

        /// Fourth line from diag coef 3,3
        hermitian_matrix[re][3][3].real(channel_coefficients_[0][3][re].real() * channel_coefficients_[0][3][re].real() +
                                    channel_coefficients_[0][3][re].imag() * channel_coefficients_[0][3][re].imag());

        /// Compute hermitian matrix
        for(i = 1; i < nb_rx_ports_; i++) {
            hermitian_matrix[re][0][0] += channel_coefficients_[i][0][re].real() * channel_coefficients_[i][0][re].real() +
                                      channel_coefficients_[i][0][re].imag() * channel_coefficients_[i][0][re].imag();
            hermitian_matrix[re][0][1] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][1][re];
            hermitian_matrix[re][0][2] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][2][re];
            hermitian_matrix[re][0][3] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][3][re];
            hermitian_matrix[re][1][1] += channel_coefficients_[i][1][re].real() * channel_coefficients_[i][1][re].real() +
                                      channel_coefficients_[i][1][re].imag() * channel_coefficients_[i][1][re].imag();
            hermitian_matrix[re][1][2] += conj(channel_coefficients_[i][1][re]) * channel_coefficients_[i][2][re];
            hermitian_matrix[re][1][3] += conj(channel_coefficients_[i][1][re]) * channel_coefficients_[i][3][re];
            hermitian_matrix[re][2][2] += channel_coefficients_[i][2][re].real() * channel_coefficients_[i][2][re].real() +
                                      channel_coefficients_[i][2][re].imag() * channel_coefficients_[i][2][re].imag();
            hermitian_matrix[re][2][3] += conj(channel_coefficients_[i][2][re]) * channel_coefficients_[i][3][re];
            hermitian_matrix[re][3][3] += channel_coefficients_[i][3][re].real() * channel_coefficients_[i][3][re].real() +
                                      channel_coefficients_[i][3][re].imag() * channel_coefficients_[i][3][re].imag();
        }
    }
}

#if defined(__AVX2__)
void vblast_compute_rdr_decomp_avx_v2(std::vector<std::complex<float>> hermitian_matrix_[MAX_TX_PORTS][MAX_TX_PORTS],
                                    int num_re_pdsch_,
                                    int nb_tx_dmrs_ports_) {

    /*
     *  array[0] = h00
     *  array[1] = h01
     *  array[2] = h02
     *  array[3] = h03
     *  array[4] = h11
     *  array[5] = h12
     *  array[6] = h13
     *  array[7] = h22
     *  array[8] = h23
     *  array[9] = h33
     */
    __m256 m256_array[10];
    __m256 vec1, vec2;
    __m256 null_vec = _mm256_set1_ps(0);
    __m256 conj = _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1);

    for(int re = 0; re < num_re_pdsch_; re+=4) {
        /************************************************* RDR decomp ************************************************/
        /// Compute h01 first
        m256_array[0] = _mm256_loadu_ps((float *) &hermitian_matrix_[0][0][re]); /// Load h00
        m256_array[0] = _mm256_permute_ps(_mm256_hadd_ps(m256_array[0], m256_array[0]), 0b11011000); /// Compute the vector containing real part of h00 copied twice : re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3
        m256_array[1] = _mm256_loadu_ps((float *) &hermitian_matrix_[0][1][re]); /// Load h01
        m256_array[1] = _mm256_div_ps(m256_array[1], m256_array[0]); /// Divide by real part of h00
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][1][re], m256_array[1]); /// Store updated h01

        /// Compute h11
        m256_array[4] = _mm256_loadu_ps((float *) &hermitian_matrix_[1][1][re]);
        vec1 = _mm256_mul_ps(m256_array[1], m256_array[1]);
        vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(vec1, null_vec), 0b11011000), m256_array[0]); /// Compute squared_norm(h01) * h00.real() with 0 in between the squared norms : norm0 | 0 | norm1 | 0 | ...
        m256_array[4] = _mm256_sub_ps(m256_array[4], vec2); /// Substract to h11 : h11 - squared_norm(h01) * h00.real();
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][1][re], m256_array[4]); /// Store the result

        /// Compute h12
        m256_array[2] = _mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]); /// Load h02, h01 is already loaded in vec3
        /// Compute conj(h01) * h02
        vec1 = _mm256_mul_ps(m256_array[2], m256_array[1]); /// real part coefs : re0 * re1 | im0 * im1
        vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(m256_array[1], conj), 0b10110001), m256_array[2]); /// imag part coefs -im0 * re 1 | re 0 * im 1
        m256_array[5] = _mm256_loadu_ps((float *) &hermitian_matrix_[1][2][re]);
        /// Substract to h12.
        m256_array[5] = _mm256_sub_ps(m256_array[5], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));
        /// Divide by real part of h11.
        /// KEEP VECTOR UNMODIFIED : (real part of h11)
        m256_array[4] = _mm256_permute_ps(_mm256_hadd_ps(m256_array[4], m256_array[4]), 0b11011000); ///  this gives the vector re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3 (real part of h11)
        m256_array[5] = _mm256_div_ps(m256_array[5], m256_array[4]);
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][2][re], m256_array[5]); /// Store the result into h12

        /// Compute h13
        m256_array[3] = _mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]); /// Load h03, h01 is already loaded in vec3
        /// Compute conj(h01) * h03
        vec1 = _mm256_mul_ps(m256_array[3], m256_array[1]); /// real part coefs : re0 * re1 | im0 * im1
        vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(m256_array[1], conj), 0b10110001), m256_array[3]); /// imag part coefs -im0 * re 1 | re 0 * im 1
        m256_array[6] = _mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]);
        /// Substract to h13.
        m256_array[6] = _mm256_sub_ps(m256_array[6], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));

        /// Update h02
        m256_array[2] = _mm256_div_ps(m256_array[2], m256_array[0]);
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][2][re], m256_array[2]);

        /// Third row
        /// Compute h22
        m256_array[7] = _mm256_loadu_ps((float *) &hermitian_matrix_[2][2][re]);
        m256_array[7] = _mm256_sub_ps(m256_array[7], _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(m256_array[2], m256_array[2]), null_vec), 0b11011000), m256_array[0])); /// Square norm of h02 multiplied by real part of h00
        m256_array[7] = _mm256_sub_ps(m256_array[7], _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(m256_array[5], m256_array[5]), null_vec), 0b11011000), m256_array[4])); /// Square norm of h12 multiplied by real part of h11
        _mm256_storeu_ps((float *) &hermitian_matrix_[2][2][re], m256_array[7]); /// Store the result
        /// Compute h23
        /// Compute conj(h02) * h03
        vec1 = _mm256_mul_ps(m256_array[2], m256_array[3]);
        vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(m256_array[2], conj), 0b10110001), m256_array[3]);
        m256_array[8] = _mm256_loadu_ps((float *) &hermitian_matrix_[2][3][re]);
        m256_array[8] = _mm256_sub_ps(m256_array[8], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));
        /// Compute conj(h12) * h13
        vec1 = _mm256_mul_ps(m256_array[5], m256_array[6]);
        vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(m256_array[5], conj), 0b10110001), m256_array[6]);
        m256_array[8] = _mm256_sub_ps(m256_array[8], _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000));
        /// Divide by real part of h22
        m256_array[7] = _mm256_permute_ps(_mm256_hadd_ps(m256_array[7], m256_array[7]), 0b11011000); /// KEEP UNMODIFIED (contains real part of h22 : re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3
        m256_array[8] = _mm256_div_ps(m256_array[8], m256_array[7]);
        _mm256_storeu_ps((float *) &hermitian_matrix_[2][3][re], m256_array[8]);

        /// Update h03 and h13
        m256_array[3] = _mm256_div_ps(m256_array[3], m256_array[0]);
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][3][re], m256_array[3]);
        /// Divide by real part of h11.
        m256_array[6] = _mm256_div_ps(m256_array[6], m256_array[4]); ///  _mm256_hadd_ps(vec4, vec4) gives the vector re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3 (real part of h11)
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][3][re], m256_array[6]); /// Store the result into h12

        /// Fourth row
        m256_array[9] = _mm256_loadu_ps((float *) &hermitian_matrix_[3][3][re]);
        m256_array[9] = _mm256_sub_ps(m256_array[9], _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(m256_array[3], m256_array[3]), null_vec), 0b11011000), m256_array[0])); /// Square norm of h03 multiplied by real part of h00
        m256_array[9] = _mm256_sub_ps(m256_array[9], _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(m256_array[6], m256_array[6]), null_vec), 0b11011000), m256_array[4])); /// Square norm of h13 multiplied by real part of h11
        m256_array[9] = _mm256_sub_ps(m256_array[9], _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(m256_array[8], m256_array[8]), null_vec), 0b11011000), m256_array[7])); /// Square norm of h23 multiplied by real part of h22
        _mm256_storeu_ps((float *) &hermitian_matrix_[3][3][re], m256_array[9]); /// Store to h33

    }
}
#endif

#if defined(__AVX2__)
void vblast_compute_rdr_decomp_avx(std::vector<std::complex<float>> hermitian_matrix_[MAX_TX_PORTS][MAX_TX_PORTS],
                                   int num_re_pdsch_,
                                   int nb_tx_dmrs_ports_) {

    __m256 vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8;
    __m256 null_vec = _mm256_set1_ps(0);
    __m256 conj = _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1);
    __m256 neg = _mm256_set1_ps(-1);

    for(int re = 0; re < num_re_pdsch_; re+=4) {
        /************************************************* RDR decomp ***********************************************
        /// Compute h01 first
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][0][re]); /// Load h00
        vec2 = _mm256_permute_ps(_mm256_hadd_ps(vec1, vec1), 0b11011000); /// Compute the vector containing real part of h00 copied twice : re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3
        //vec2 = _mm256_blend_ps(vec1, _mm256_permute_ps(vec1, 0b10110001), 0b10101010);
        /// KEEP VECTOR UNMODIFIED (real part of h00)
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][1][re]); /// Load h01
        vec3 = _mm256_div_ps(vec3, vec2); /// Divide by real part of h00
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][1][re], vec3); /// Store updated h01

        /// Compute h11
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][1][re]);
        vec4 = _mm256_mul_ps(vec3, vec3);
        vec5 = _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(vec4, null_vec), 0b11011000), vec2); /// Compute squared_norm(h01) * h00.real() with 0 in between the squared norms : norm0 | 0 | norm1 | 0 | ...
        //vec5 = _mm256_mul_ps(_mm256_blend_ps(_mm256_add_ps(vec4, _mm256_permute_ps(vec4, 0b10110001)), null_vec, 0b10101010), vec2);
        vec4 = _mm256_sub_ps(vec1, vec5); /// Substract to h11 : h11 - squared_norm(h01) * h00.real();
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][1][re], vec4); /// Store the result

        /// Compute h12
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]); /// Load h02, h01 is already loaded in vec3
        /// Compute conj(h01) * h02
        vec5 = _mm256_mul_ps(vec1, vec3); /// real part coefs : re0 * re1 | im0 * im1
        vec6 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec3, conj), 0b10110001), vec1); /// imag part coefs -im0 * re 1 | re 0 * im 1
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][2][re]);
        /// Substract to h12.
        vec1 = _mm256_sub_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec5, vec6), 0b11011000));
        /// Divide by real part of h11.
        vec7 = _mm256_permute_ps(_mm256_hadd_ps(vec4, vec4), 0b11011000); ///  this gives the vector re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3 (real part of h11)
        //vec7 = _mm256_blend_ps(vec4, _mm256_permute_ps(vec4, 0b10110001), 0b10101010);
        /// KEEP VECTOR UNMODIFIED : (real part of h11)
        vec1 = _mm256_div_ps(vec1, vec7);
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][2][re], vec1); /// Store the result into h12

        /// Compute h13
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]); /// Load h03, h01 is already loaded in vec3
        /// Compute conj(h01) * h03
        vec5 = _mm256_mul_ps(vec1, vec3); /// real part coefs : re0 * re1 | im0 * im1
        vec6 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec3, conj), 0b10110001), vec1); /// imag part coefs -im0 * re 1 | re 0 * im 1
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]);
        /// Substract to h13.
        vec1 = _mm256_sub_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec5, vec6), 0b11011000));
        /// Divide by real part of h11.
        vec1 = _mm256_div_ps(vec1, vec7); ///  _mm256_hadd_ps(vec4, vec4) gives the vector re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3 (real part of h11)
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][3][re], vec1); /// Store the result into h12

        /// Update h02 and h03
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][2][re], _mm256_div_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]), vec2));
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][3][re], _mm256_div_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]), vec2));

        /// Third row
        /// Compute h22
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]); /// Load h02
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][2][re]); /// Load h22. KEEP VECTOR UNMODIFIED until computing vec8 (contains h22)
        vec3 = _mm256_sub_ps(vec3, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(vec1, vec1), null_vec), 0b11011000), vec2)); /// Square norm of h02 multiplied by real part of h00
        //vec3 = _mm256_sub_ps(vec3, _mm256_mul_ps(_mm256_blend_ps(_mm256_add_ps(vec1, _mm256_permute_ps(vec1, 0b10110001)), null_vec, 0b10101010), vec2));
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][2][re]); /// Load h12
        vec3 = _mm256_sub_ps(vec3, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(vec1, vec1), null_vec), 0b11011000), vec7)); /// Square norm of h12 multiplied by real part of h11
        //vec3 = _mm256_sub_ps(vec3, _mm256_mul_ps(_mm256_blend_ps(_mm256_add_ps(vec1, _mm256_permute_ps(vec1, 0b10110001)), null_vec, 0b10101010), vec7));
        _mm256_storeu_ps((float *) &hermitian_matrix_[2][2][re], vec3); /// Store the result
        /// Compute h23
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]);
        vec4 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]);
        /// Compute conj(h02) * h03
        vec5 = _mm256_mul_ps(vec1, vec4);
        vec6 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec4);
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][3][re]);
        vec1 = _mm256_sub_ps(vec1, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(vec5, vec6), 0b11011000), vec2)); /// Multiply by real part of h00
        //vec1 = _mm256_sub_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec5, vec6), 0b11011000)); /// Multiply by real part of h00
        //vec1 = _mm256_sub_ps(vec1, _mm256_mul_ps(_mm256_blend_ps(_mm256_add_ps(vec5, _mm256_permute_ps(vec5, 0b10110001)), _mm256_add_ps(vec6, _mm256_permute_ps(vec6, 0b10110001)), 0b10101010), vec2));
        _mm256_storeu_ps((float *) &hermitian_matrix_[2][3][re], vec1);
        /// Compute conj(h12) * h13
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][2][re]);
        vec4 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]);
        vec5 = _mm256_mul_ps(vec1, vec4);
        vec6 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec4);
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][3][re]);
        vec1 = _mm256_sub_ps(vec1, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(vec5, vec6), 0b11011000), vec7)); /// Multiply by real part of h11
        //vec1 = _mm256_sub_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec5, vec6), 0b11011000));
        //vec1 = _mm256_sub_ps(vec1, _mm256_mul_ps(_mm256_blend_ps(_mm256_add_ps(vec5, _mm256_permute_ps(vec5, 0b10110001)), _mm256_add_ps(vec6, _mm256_permute_ps(vec6, 0b10110001)), 0b10101010), vec7));
        /// Divide by real part of h22
        vec8 = _mm256_permute_ps(_mm256_hadd_ps(vec3, vec3), 0b11011000); /// KEEP UNMODIFIED (contains real part of h22 : re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3
        //vec8 = _mm256_blend_ps(vec3, _mm256_permute_ps(vec3, 0b10110001), 0b10101010);
        vec1 = _mm256_div_ps(vec1, vec8);
        _mm256_storeu_ps((float *) &hermitian_matrix_[2][3][re], vec1);

         ///  _mm256_hadd_ps(vec4, vec4) gives the vector re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3 (real part of h11)
        //_mm256_storeu_ps((float *) &hermitian_matrix_[1][3][re], _mm256_div_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]), vec7)); /// Store the result into h12
        //_mm256_storeu_ps((float *) &hermitian_matrix_[0][3][re], _mm256_div_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]), vec2));

        /// Fourth row
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]); /// Load h03
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[3][3][re]); /// Load h33
        vec3 = _mm256_sub_ps(vec3, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(vec1, vec1), null_vec), 0b11011000), vec2)); /// Square norm of h03 multiplied by real part of h00
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]); /// Load h13
        vec3 = _mm256_sub_ps(vec3, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(vec1, vec1), null_vec), 0b11011000), vec7)); /// Square norm of h13 multiplied by real part of h11
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][3][re]); /// Load h23
        vec3 = _mm256_sub_ps(vec3, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(vec1, vec1), null_vec), 0b11011000), vec8)); /// Square norm of h23 multiplied by real part of h22
        _mm256_storeu_ps((float *) &hermitian_matrix_[3][3][re], vec3); /// Store to h33 */

        /************************************************* RDR decomp ************************************************/
        /// Compute h01 first
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][0][re]); /// Load h00
        vec2 = _mm256_permute_ps(_mm256_hadd_ps(vec1, vec1), 0b11011000); /// Compute the vector containing real part of h00 copied twice : re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3
        /// KEEP VECTOR UNMODIFIED (real part of h00)
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][1][re]); /// Load h01
        vec3 = _mm256_div_ps(vec3, vec2); /// Divide by real part of h00
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][1][re], vec3); /// Store updated h01

        /// Compute h11
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][1][re]);
        vec4 = _mm256_mul_ps(vec3, vec3);
        vec5 = _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(vec4, null_vec), 0b11011000), vec2); /// Compute squared_norm(h01) * h00.real() with 0 in between the squared norms : norm0 | 0 | norm1 | 0 | ...
        vec4 = _mm256_sub_ps(vec1, vec5); /// Substract to h11 : h11 - squared_norm(h01) * h00.real();
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][1][re], vec4); /// Store the result

        /// Compute h12
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]); /// Load h02, h01 is already loaded in vec3
        /// Compute conj(h01) * h02
        vec5 = _mm256_mul_ps(vec1, vec3); /// real part coefs : re0 * re1 | im0 * im1
        vec6 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec3, conj), 0b10110001), vec1); /// imag part coefs -im0 * re 1 | re 0 * im 1
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][2][re]);
        /// Substract to h12.
        vec1 = _mm256_sub_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec5, vec6), 0b11011000));
        /// Divide by real part of h11.
        vec7 = _mm256_permute_ps(_mm256_hadd_ps(vec4, vec4), 0b11011000); ///  this gives the vector re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3 (real part of h11)
        /// KEEP VECTOR UNMODIFIED : (real part of h11)
        vec1 = _mm256_div_ps(vec1, vec7);
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][2][re], vec1); /// Store the result into h12

        /// Compute h13
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]); /// Load h03, h01 is already loaded in vec3
        /// Compute conj(h01) * h03
        vec5 = _mm256_mul_ps(vec1, vec3); /// real part coefs : re0 * re1 | im0 * im1
        vec6 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec3, conj), 0b10110001), vec1); /// imag part coefs -im0 * re 1 | re 0 * im 1
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]);
        /// Substract to h13.
        vec1 = _mm256_sub_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec5, vec6), 0b11011000));
        /// Divide by real part of h11.
        vec1 = _mm256_div_ps(vec1, vec7); ///  _mm256_hadd_ps(vec4, vec4) gives the vector re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3 (real part of h11)
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][3][re], vec1); /// Store the result into h12

        /// Update h02 and h03
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][2][re], _mm256_div_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]), vec2));
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][3][re], _mm256_div_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]), vec2));

        /// Third row
        /// Compute h22
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]); /// Load h02
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][2][re]); /// Load h22. KEEP VECTOR UNMODIFIED until computing vec8 (contains h22)
        vec3 = _mm256_sub_ps(vec3, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(vec1, vec1), null_vec), 0b11011000), vec2)); /// Square norm of h02 multiplied by real part of h00
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][2][re]); /// Load h12
        vec3 = _mm256_sub_ps(vec3, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(vec1, vec1), null_vec), 0b11011000), vec7)); /// Square norm of h12 multiplied by real part of h11
        _mm256_storeu_ps((float *) &hermitian_matrix_[2][2][re], vec3); /// Store the result
        /// Compute h23
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]);
        vec4 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]);
        /// Compute conj(h02) * h03
        vec5 = _mm256_mul_ps(vec1, vec4);
        vec6 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec4);
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][3][re]);
        vec1 = _mm256_sub_ps(vec1, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(vec5, vec6), 0b11011000), vec2)); /// Multiply by real part of h00
        _mm256_storeu_ps((float *) &hermitian_matrix_[2][3][re], vec1);
        /// Compute conj(h12) * h13
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][2][re]);
        vec4 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]);
        vec5 = _mm256_mul_ps(vec1, vec4);
        vec6 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec4);
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][3][re]);
        vec1 = _mm256_sub_ps(vec1, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(vec5, vec6), 0b11011000), vec7)); /// Multiply by real part of h11
        /// Divide by real part of h22
        vec8 = _mm256_permute_ps(_mm256_hadd_ps(vec3, vec3), 0b11011000); /// KEEP UNMODIFIED (contains real part of h22 : re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3
        vec1 = _mm256_div_ps(vec1, vec8);
        _mm256_storeu_ps((float *) &hermitian_matrix_[2][3][re], vec1);

        /// Fourth row
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]); /// Load h03
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[3][3][re]); /// Load h33
        vec3 = _mm256_sub_ps(vec3, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(vec1, vec1), null_vec), 0b11011000), vec2)); /// Square norm of h03 multiplied by real part of h00
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]); /// Load h13
        vec3 = _mm256_sub_ps(vec3, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(vec1, vec1), null_vec), 0b11011000), vec7)); /// Square norm of h13 multiplied by real part of h11
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][3][re]); /// Load h23
        vec3 = _mm256_sub_ps(vec3, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(vec1, vec1), null_vec), 0b11011000), vec8)); /// Square norm of h23
        _mm256_storeu_ps((float *) &hermitian_matrix_[3][3][re], vec3);
    }
}
#endif

#if defined(__AVX2__)
void vblast_compute_inverse_avx(std::vector<std::complex<float>> hermitian_matrix_[MAX_TX_PORTS][MAX_TX_PORTS],
                                int num_re_pdsch_,
                                int nb_tx_dmrs_ports_) {

    __m256 vec1, vec2, vec3, vec4, vec5;
    __m256 conj = _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1);
    __m256 neg = _mm256_set1_ps(-1);

    for(int re = 0; re < num_re_pdsch_; re+=4) {
        /********************************* Compute in-place inverse **************************************************/
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][1][re],
                         _mm256_mul_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[0][1][re]), neg));
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][2][re],
                         _mm256_mul_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]), neg));
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][3][re],
                         _mm256_mul_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]), neg));
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][2][re],
                         _mm256_mul_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[1][2][re]), neg));
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][3][re],
                         _mm256_mul_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]), neg));
        _mm256_storeu_ps((float *) &hermitian_matrix_[2][3][re],
                         _mm256_mul_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[2][3][re]), neg));

        /// Compute coefficients in the correct order
        /// hermitian_matrix_[re][0][2] += hermitian_matrix_[re][0][1] * hermitian_matrix_[re][1][2];
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]);
        vec2 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][1][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][2][re]);
        /// Multiply h01 by h12, then add to h02
        vec4 = _mm256_mul_ps(vec2, vec3); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the second coef
        /// Permute h12 vector then multiply by h01 vector
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3, 0b10110001));
        /// add to h02
        vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][2][re], vec1);

        /// Compute hermitian_matrix_[re][0][3] += hermitian_matrix_[re][0][1] * hermitian_matrix_[re][1][3] + hermitian_matrix_[re][0][2] * hermitian_matrix_[re][2][3];
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]);
        //vec2 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][1][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]);
        /// Multiply h01 by h12, then add to h02
        vec4 = _mm256_mul_ps(vec2, vec3); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the second coef
        /// Permute h12 vector then multiply by h01 vector
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3, 0b10110001));
        /// add to h02
        vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));
        vec2 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][3][re]);
        /// Multiply h02 by h23, then add to h02
        vec4 = _mm256_mul_ps(vec2, vec3); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the second coef
        /// Permute h12 vector then multiply by h01 vector
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3, 0b10110001));
        /// add to h02
        vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][3][re], vec1);

        /// Compute hermitian_matrix_[re][1][3] += hermitian_matrix_[re][1][2] * hermitian_matrix_[re][2][3];
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]);
        vec2 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][2][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][3][re]);
        /// Multiply h01 by h12, then add to h02
        vec4 = _mm256_mul_ps(vec2, vec3); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the second coef
        /// Permute h12 vector then multiply by h01 vector
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3, 0b10110001));
        /// add to h02
        vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][3][re], vec1);

        /// Copy transconj inverse R^(-1)^H in lower part of the array
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][1][re]);
        vec1 = _mm256_mul_ps(vec1, conj);
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][0][re], vec1);

        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]);
        vec1 = _mm256_mul_ps(vec1, conj);
        _mm256_storeu_ps((float *) &hermitian_matrix_[2][0][re], vec1);

        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][2][re]);
        vec1 = _mm256_mul_ps(vec1, conj);
        _mm256_storeu_ps((float *) &hermitian_matrix_[2][1][re], vec1);

        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]);
        vec1 = _mm256_mul_ps(vec1, conj);
        _mm256_storeu_ps((float *) &hermitian_matrix_[3][0][re], vec1);

        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]);
        vec1 = _mm256_mul_ps(vec1, conj);
        _mm256_storeu_ps((float *) &hermitian_matrix_[3][1][re], vec1);

        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][3][re]);
        vec1 = _mm256_mul_ps(vec1, conj);
        _mm256_storeu_ps((float *) &hermitian_matrix_[3][2][re], vec1);
    }
}
#endif

#if defined(__AVX2__)
void vblast_zf_4_layers_avx(std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                            std::vector<std::complex<float>> hermitian_matrix_[MAX_TX_PORTS][MAX_TX_PORTS],
                            int num_re_pdsch_,
                            std::vector<std::complex<float>> temp_equalized_symbols_[MAX_TX_PORTS],
                            std::complex<float> * equalized_symbols_,
                            int nb_rx_ports_) {

    int i, j;
    int l;
    __m256 vec1, vec2, vec3, vec4, vec5, vec6, vec7, vec8, vec9, vec10;
    __m256 null_vec = _mm256_set1_ps(0);
    __m256 conj = _mm256_setr_ps(1, -1, 1, -1, 1, -1, 1, -1);
    __m256 neg = _mm256_set1_ps(-1);

    for(int re = 0; re < num_re_pdsch_; re+=4) {
        /************************************************* RDR decomp ***********************************************
        /// Compute h01 first
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][0][re]); /// Load h00
        vec2 = _mm256_permute_ps(_mm256_hadd_ps(vec1, vec1), 0b11011000); /// Compute the vector containing real part of h00 copied twice : re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3
                                           /// KEEP VECTOR UNMODIFIED (real part of h00)
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][1][re]); /// Load h01
        vec3 = _mm256_div_ps(vec3, vec2); /// Divide by real part of h00
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][1][re], vec3); /// Store updated h01

        /// Compute h11
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][1][re]);
        vec4 = _mm256_mul_ps(vec3, vec3);
        vec5 = _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(vec4, null_vec), 0b11011000), vec2); /// Compute squared_norm(h01) * h00.real() with 0 in between the squared norms : norm0 | 0 | norm1 | 0 | ...
        vec4 = _mm256_sub_ps(vec1, vec5); /// Substract to h11 : h11 - squared_norm(h01) * h00.real();
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][1][re], vec4); /// Store the result

        /// Compute h12
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]); /// Load h02, h01 is already loaded in vec3
        /// Compute conj(h01) * h02
        vec5 = _mm256_mul_ps(vec1, vec3); /// real part coefs : re0 * re1 | im0 * im1
        vec6 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec3, conj), 0b10110001), vec1); /// imag part coefs -im0 * re 1 | re 0 * im 1
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][2][re]);
        /// Substract to h12.
        vec1 = _mm256_sub_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec5, vec6), 0b11011000));
        /// Divide by real part of h11.
        vec7 = _mm256_permute_ps(_mm256_hadd_ps(vec4, vec4), 0b11011000); ///  this gives the vector re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3 (real part of h11)
                                           /// KEEP VECTOR UNMODIFIED : (real part of h11)
        vec1 = _mm256_div_ps(vec1, vec7);
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][2][re], vec1); /// Store the result into h12

        /// Compute h13
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]); /// Load h03, h01 is already loaded in vec3
        /// Compute conj(h01) * h03
        vec5 = _mm256_mul_ps(vec1, vec3); /// real part coefs : re0 * re1 | im0 * im1
        vec6 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec3, conj), 0b10110001), vec1); /// imag part coefs -im0 * re 1 | re 0 * im 1
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]);
        /// Substract to h13.
        vec1 = _mm256_sub_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec5, vec6), 0b11011000));
        /// Divide by real part of h11.
        vec1 = _mm256_div_ps(vec1, vec7); ///  _mm256_hadd_ps(vec4, vec4) gives the vector re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3 (real part of h11)
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][3][re], vec1); /// Store the result into h12

        /// Update h02 and h03
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][2][re], _mm256_div_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]), vec2));
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][3][re], _mm256_div_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]), vec2));

        /// Third row
        /// Compute h22
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]); /// Load h02
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][2][re]); /// Load h22. KEEP VECTOR UNMODIFIED until computing vec8 (contains h22)
        vec3 = _mm256_sub_ps(vec3, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(vec1, vec1), null_vec), 0b11011000), vec2)); /// Square norm of h02 multiplied by real part of h00
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][2][re]); /// Load h12
        vec3 = _mm256_sub_ps(vec3, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(vec1, vec1), null_vec), 0b11011000), vec7)); /// Square norm of h12 multiplied by real part of h11
        _mm256_storeu_ps((float *) &hermitian_matrix_[2][2][re], vec3); /// Store the result
        /// Compute h23
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]);
        vec4 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]);
        /// Compute conj(h02) * h03
        vec5 = _mm256_mul_ps(vec1, vec4);
        vec6 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec4);
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][3][re]);
        vec1 = _mm256_sub_ps(vec1, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(vec5, vec6), 0b11011000), vec2)); /// Multiply by real part of h00
        _mm256_storeu_ps((float *) &hermitian_matrix_[2][3][re], vec1);
        /// Compute conj(h12) * h13
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][2][re]);
        vec4 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]);
        vec5 = _mm256_mul_ps(vec1, vec4);
        vec6 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj), 0b10110001), vec4);
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][3][re]);
        vec1 = _mm256_sub_ps(vec1, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(vec5, vec6), 0b11011000), vec7)); /// Multiply by real part of h11
        /// Divide by real part of h22
        vec8 = _mm256_permute_ps(_mm256_hadd_ps(vec3, vec3), 0b11011000); /// KEEP UNMODIFIED (contains real part of h22 : re0 | re0 | re1 | re1 | re2 | re2 | re3 | re3
        vec1 = _mm256_div_ps(vec1, vec8);
        _mm256_storeu_ps((float *) &hermitian_matrix_[2][3][re], vec1);

        /// Fourth row
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]); /// Load h03
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[3][3][re]); /// Load h33
        vec3 = _mm256_sub_ps(vec3, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(vec1, vec1), null_vec), 0b11011000), vec2)); /// Square norm of h03 multiplied by real part of h00
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]); /// Load h13
        vec3 = _mm256_sub_ps(vec3, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(vec1, vec1), null_vec), 0b11011000), vec7)); /// Square norm of h13 multiplied by real part of h11
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][3][re]); /// Load h23
        vec3 = _mm256_sub_ps(vec3, _mm256_mul_ps(_mm256_permute_ps(_mm256_hadd_ps(_mm256_mul_ps(vec1, vec1), null_vec), 0b11011000), vec8)); /// Square norm of h23 multiplied by real part of h22
        _mm256_storeu_ps((float *) &hermitian_matrix_[3][3][re], vec3);

        /********************************* Compute in-place inverse *************************************************
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][1][re],
                         _mm256_mul_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[0][1][re]), neg));
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][2][re],
                         _mm256_mul_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]), neg));
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][3][re],
                         _mm256_mul_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]), neg));
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][2][re],
                         _mm256_mul_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[1][2][re]), neg));
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][3][re],
                         _mm256_mul_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]), neg));
        _mm256_storeu_ps((float *) &hermitian_matrix_[2][3][re],
                         _mm256_mul_ps(_mm256_loadu_ps((float *) &hermitian_matrix_[2][3][re]), neg));

        /// Compute coefficients in the correct order
        /// hermitian_matrix_[re][0][2] += hermitian_matrix_[re][0][1] * hermitian_matrix_[re][1][2];
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]);
        vec2 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][1][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][2][re]);
        /// Multiply h01 by h12, then add to h02
        vec4 = _mm256_mul_ps(vec2, vec3); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the second coef
        /// Permute h12 vector then multiply by h01 vector
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3 , 0b10110001));
        /// add to h02
        vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][2][re], vec1);

        /// Compute hermitian_matrix_[re][0][3] += hermitian_matrix_[re][0][1] * hermitian_matrix_[re][1][3] + hermitian_matrix_[re][0][2] * hermitian_matrix_[re][2][3];
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]);
        vec2 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][1][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]);
        /// Multiply h01 by h12, then add to h02
        vec4 = _mm256_mul_ps(vec2, vec3); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the second coef
        /// Permute h12 vector then multiply by h01 vector
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3 , 0b10110001));
        /// add to h02
        vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));
        vec2 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][3][re]);
        /// Multiply h02 by h23, then add to h02
        vec4 = _mm256_mul_ps(vec2, vec3); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the second coef
        /// Permute h12 vector then multiply by h01 vector
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3 , 0b10110001));
        /// add to h02
        vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));
        _mm256_storeu_ps((float *) &hermitian_matrix_[0][3][re], vec1);

        /// Compute hermitian_matrix_[re][1][3] += hermitian_matrix_[re][1][2] * hermitian_matrix_[re][2][3];
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]);
        vec2 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][2][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][3][re]);
        /// Multiply h01 by h12, then add to h02
        vec4 = _mm256_mul_ps(vec2, vec3); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the second coef
        /// Permute h12 vector then multiply by h01 vector
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3, 0b10110001));
        /// add to h02
        vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][3][re], vec1);

        /// Copy transconj inverse R^(-1)^H in lower part of the array
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][1][re]);
        vec1 = _mm256_mul_ps(vec1, conj);
        _mm256_storeu_ps((float *) &hermitian_matrix_[1][0][re], vec1);

        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]);
        vec1 = _mm256_mul_ps(vec1, conj);
        _mm256_storeu_ps((float *) &hermitian_matrix_[2][0][re], vec1);

        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][2][re]);
        vec1 = _mm256_mul_ps(vec1, conj);
        _mm256_storeu_ps((float *) &hermitian_matrix_[2][1][re], vec1);

        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]);
        vec1 = _mm256_mul_ps(vec1, conj);
        _mm256_storeu_ps((float *) &hermitian_matrix_[3][0][re], vec1);

        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]);
        vec1 = _mm256_mul_ps(vec1, conj);
        _mm256_storeu_ps((float *) &hermitian_matrix_[3][1][re], vec1);

        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][3][re]);
        vec1 = _mm256_mul_ps(vec1, conj);
        _mm256_storeu_ps((float *) &hermitian_matrix_[3][2][re], vec1);

        /******************************* Multiply preprocessed layers with inverse matrices **************************/

        /***************(equalized_symbols_ + 3) += hermitian_matrix_[re][3][0] * *(equalized_symbols_)
                + hermitian_matrix_[re][3][1] * *(equalized_symbols_ + 1)
                + hermitian_matrix_[re][3][2] * *(equalized_symbols_ + 2); *******************************************/
        vec1 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[3][re]);
        vec9 = _mm256_set1_ps(0);
        vec10 = _mm256_set1_ps(0);

        /// hermitian_matrix_[re][3][0] * *(equalized_symbols_)
        vec2 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[0][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[3][0][re]);
        vec4 = _mm256_mul_ps(vec3, vec2); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3, 0b10110001));
        /// Add to vec9 (real part of dot prod) and vec10 (imag part of dot prod)
        vec9 = _mm256_add_ps(vec9, vec4);
        vec10 = _mm256_add_ps(vec10, vec5);
        // add the result to vec1
        //vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));

        /// hermitian_matrix_[re][3][1] * *(equalized_symbols_ + 1)
        vec2 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[1][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[3][1][re]);
        vec4 = _mm256_mul_ps(vec3, vec2); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3, 0b10110001));
        /// Add to vec9 (real part of dot prod) and vec10 (imag part of dot prod)
        vec9 = _mm256_add_ps(vec9, vec4);
        vec10 = _mm256_add_ps(vec10, vec5);
        // add the result to vec1
        //vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));

        /// hermitian_matrix_[re][3][2] * *(equalized_symbols_ + 2)
        vec2 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[2][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[3][2][re]);
        vec4 = _mm256_mul_ps(vec3, vec2); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3, 0b10110001));
        /// Add to vec9 (real part of dot prod) and vec10 (imag part of dot prod)
        vec9 = _mm256_add_ps(vec9, vec4);
        vec10 = _mm256_add_ps(vec10, vec5);
        // add the result to vec1
        //vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));

        /// Add final result to vec1
        vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec9, vec10), 0b11011000));
        _mm256_storeu_ps((float *) &temp_equalized_symbols_[3][re], vec1);

        /********************* *(equalized_symbols_ + 2) += hermitian_matrix_[re][2][0] * *(equalized_symbols_)
          + hermitian_matrix_[re][2][1] * *(equalized_symbols_ + 1); ****************/
        vec1 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[2][re]);
        vec9 = _mm256_set1_ps(0);
        vec10 = _mm256_set1_ps(0);

        /// hermitian_matrix_[re][2][0] * *(equalized_symbols_)
        vec2 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[0][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][0][re]);
        vec4 = _mm256_mul_ps(vec3, vec2); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3 , 0b10110001));
        /// Add to vec9 (real part of dot prod) and vec10 (imag part of dot prod)
        vec9 = _mm256_add_ps(vec9, vec4);
        vec10 = _mm256_add_ps(vec10, vec5);
        // add the result to vec1
        //vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));

        /// hermitian_matrix_[re][2][1] * *(equalized_symbols_ + 1)
        vec2 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[1][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][1][re]);
        vec4 = _mm256_mul_ps(vec3, vec2); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3 , 0b10110001));
        /// Add to vec9 (real part of dot prod) and vec10 (imag part of dot prod)
        vec9 = _mm256_add_ps(vec9, vec4);
        vec10 = _mm256_add_ps(vec10, vec5);
        // add the result to vec1
        //vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));

        /// Add final result to vec1
        vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec9, vec10), 0b11011000));
        _mm256_storeu_ps((float *) &temp_equalized_symbols_[2][re], vec1);

        /******************* *(equalized_symbols_ + 1) += hermitian_matrix_[re][1][0] * *(equalized_symbols_); ********/
        vec1 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[1][re]);

        /// hermitian_matrix_[re][1][0] * *(equalized_symbols_)
        vec2 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[0][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][0][re]);
        vec4 = _mm256_mul_ps(vec3, vec2); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3 , 0b10110001));
        /// add the result to vec1
        vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));
        _mm256_storeu_ps((float *) &temp_equalized_symbols_[1][re], vec1);

        /********************* *(equalized_symbols_)     /= hermitian_matrix_[re][0][0].real(); ***********************/
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][0][re]);
        vec2 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[0][re]);
        _mm256_storeu_ps((float *) &temp_equalized_symbols_[0][re], _mm256_div_ps(vec2, _mm256_permute_ps(_mm256_hadd_ps(vec1, vec1), 0b11011000)));
        /********************* *(equalized_symbols_ + 1) /= hermitian_matrix_[re][1][1].real(); ***********************/
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][1][re]);
        vec2 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[1][re]);
        _mm256_storeu_ps((float *) &temp_equalized_symbols_[1][re], _mm256_div_ps(vec2, _mm256_permute_ps(_mm256_hadd_ps(vec1, vec1), 0b11011000)));
        /********************* *(equalized_symbols_ + 2) /= hermitian_matrix_[re][2][2].real(); ***********************/
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][2][re]);
        vec2 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[2][re]);
        _mm256_storeu_ps((float *) &temp_equalized_symbols_[2][re], _mm256_div_ps(vec2, _mm256_permute_ps(_mm256_hadd_ps(vec1, vec1), 0b11011000)));
        /********************* *(equalized_symbols_ + 3) /= hermitian_matrix_[re][3][3].real(); ***********************/
        vec1 = _mm256_loadu_ps((float *) &hermitian_matrix_[3][3][re]);
        vec2 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[3][re]);
        _mm256_storeu_ps((float *) &temp_equalized_symbols_[3][re], _mm256_div_ps(vec2, _mm256_permute_ps(_mm256_hadd_ps(vec1, vec1), 0b11011000)));

        /*************************** *(equalized_symbols_)     += hermitian_matrix_[re][0][1] * *(equalized_symbols_ + 1)
                + hermitian_matrix_[re][0][2] * *(equalized_symbols_ + 2)
                + hermitian_matrix_[re][0][3] * *(equalized_symbols_ + 3); ********************************************/

        vec1 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[0][re]);
        vec9 = _mm256_set1_ps(0);
        vec10 = _mm256_set1_ps(0);

        /// hermitian_matrix_[re][0][1] * *(equalized_symbols_ + 1)
        vec2 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[1][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][1][re]);
        vec4 = _mm256_mul_ps(vec3, vec2); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3, 0b10110001));
        /// Add to vec9 (real part of dot prod) and vec10 (imag part of dot prod)
        vec9 = _mm256_add_ps(vec9, vec4);
        vec10 = _mm256_add_ps(vec10, vec5);
        // add the result to vec1
        //vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));

        /// hermitian_matrix_[re][0][2] * *(equalized_symbols_ + 2)
        vec2 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[2][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][2][re]);
        vec4 = _mm256_mul_ps(vec3, vec2); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3, 0b10110001));
        /// Add to vec9 (real part of dot prod) and vec10 (imag part of dot prod)
        vec9 = _mm256_add_ps(vec9, vec4);
        vec10 = _mm256_add_ps(vec10, vec5);
        // add the result to vec1
        //vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));

        /// hermitian_matrix_[re][0][3] * *(equalized_symbols_ + 3)
        vec2 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[3][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[0][3][re]);
        vec4 = _mm256_mul_ps(vec3, vec2); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3, 0b10110001));
        /// Add to vec9 (real part of dot prod) and vec10 (imag part of dot prod)
        vec9 = _mm256_add_ps(vec9, vec4);
        vec10 = _mm256_add_ps(vec10, vec5);
        // add the result to vec1
        //vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));

        /// Add final result to vec1
        vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec9, vec10), 0b11011000));
        _mm256_storeu_ps((float *) &temp_equalized_symbols_[0][re], vec1);

        /************************ *(equalized_symbols_ + 1) += hermitian_matrix_[re][1][2] * *(equalized_symbols_ + 2)
                + hermitian_matrix_[re][1][3] * *(equalized_symbols_ + 3); *******************************************/
        vec1 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[1][re]);
        vec9 = _mm256_set1_ps(0);
        vec10 = _mm256_set1_ps(0);

        /// hermitian_matrix_[re][1][2] * *(equalized_symbols_ + 2)
        vec2 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[2][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][2][re]);
        vec4 = _mm256_mul_ps(vec3, vec2); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3 , 0b10110001));
        vec9 = _mm256_add_ps(vec9, vec4);
        vec10 = _mm256_add_ps(vec10, vec5);
        // add the result to vec1
        //vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));

        /// hermitian_matrix_[re][1][3] * *(equalized_symbols_ + 3)
        vec2 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[3][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[1][3][re]);
        vec4 = _mm256_mul_ps(vec3, vec2); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3, 0b10110001));
        vec9 = _mm256_add_ps(vec9, vec4);
        vec10 = _mm256_add_ps(vec10, vec5);
        // add the result to vec1
        //vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));

        /// Add final result to vec1
        vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec9, vec10), 0b11011000));
        _mm256_storeu_ps((float *) &temp_equalized_symbols_[1][re], vec1);

        /************************ *(equalized_symbols_ + 2) += hermitian_matrix_[re][2][3] * *(equalized_symbols_ + 3); ******/
        vec1 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[2][re]);

        /// hermitian_matrix_[re][2][3] * *(equalized_symbols_ + 3)
        vec2 = _mm256_loadu_ps((float *) &temp_equalized_symbols_[3][re]);
        vec3 = _mm256_loadu_ps((float *) &hermitian_matrix_[2][3][re]);
        vec4 = _mm256_mul_ps(vec3, vec2); /// real part coefficients re0 * re1 | im0 * im1
        vec4 = _mm256_mul_ps(vec4, conj); /// negate the imaginary part
        /// permute imag and real part of second vector then multiply by vec2
        vec5 = _mm256_mul_ps(vec2, _mm256_permute_ps(vec3, 0b10110001));
        /// add the result to vec1
        vec1 = _mm256_add_ps(vec1, _mm256_permute_ps(_mm256_hadd_ps(vec4, vec5), 0b11011000));
        _mm256_storeu_ps((float *) &temp_equalized_symbols_[2][re], vec1);

        /**
        /// copy computed coefficients into the final buffer
        for (i = 0; i < 4; i++) {
            for (j = 0; j < 4; j++) {
                *(equalized_symbols_ + j * 4 + i) = temp_equalized_symbols_[i][re + j];
            }
        }
        equalized_symbols_ += 16; */
//#endif
    }
}
#endif

void vblast_copy_to_equalized_symbols(int num_re_pdsch_,
                                      std::vector<std::complex<float>> temp_equalized_symbols_[MAX_TX_PORTS],
                                      std::complex<float> * equalized_symbols_,
                                      int nb_tx_dmrs_ports_) {

    int i;
    for(int re = 0; re < num_re_pdsch_; re++) {
        /// copy computed coefficients into the final buffer
        for (i = 0; i < 4; i++) {
            *(equalized_symbols_) = temp_equalized_symbols_[i][re];
            equalized_symbols_++;
        }
    }
}

void vblast_zf_4_layers(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                        std::complex<float> hermitian_matrix_[][MAX_TX_PORTS][MAX_TX_PORTS],
                        int num_re_pdsch_,
                        std::complex<float> * equalized_symbols_,
                        int nb_rx_ports_) {
    complex<float> temp_equalized_symbols[4];
    int i, j;
    int l;

#if TIME_MEASURE == 1
    #if defined(CLOCK_TYPE_CHRONO)
    std::chrono::steady_clock::time_point t1, t2;
#elif defined(CLOCK_TYPE_GETTIME)
    struct timespec t1, t2;
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
    struct timespec t1, t2;
#else
    uint64_t t1, t2;
    unsigned cycles_low1, cycles_low2, cycles_high1, cycles_high2;
#endif
#endif
    for(int re = 0; re < num_re_pdsch_; re++) {
#if TIME_MEASURE == 1
        BOOST_LOG_TRIVIAL(trace) << " ------------------------- " << endl;
        BOOST_LOG_TRIVIAL(trace) << "RE number : " << re << endl;
#if defined(CLOCK_TYPE_CHRONO)
        t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#endif
#endif

#if TIME_MEASURE == 1
        #if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
        << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
        t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
        << (t2 - t1)/TSC_FREQ * 1e9 << endl;

        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#endif
#endif

        ldl_decomp_test(hermitian_matrix_[re], // row major
                        4);

#if TIME_MEASURE == 1
        #if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();

                BOOST_LOG_TRIVIAL(trace) << "Time to compute LDL [ns] : " <<
                     std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

                t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute LDL [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute LDL [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute LDL [ns] : "
                                 << (t2 - t1)/TSC_FREQ * 1e9 << endl;

        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#endif
#endif

        /// Compute in-place inverse
        hermitian_matrix_[re][0][1]  = -hermitian_matrix_[re][0][1];
        hermitian_matrix_[re][0][2]  = -hermitian_matrix_[re][0][2];
        hermitian_matrix_[re][0][3]  = -hermitian_matrix_[re][0][3];
        hermitian_matrix_[re][1][2]  = -hermitian_matrix_[re][1][2];
        hermitian_matrix_[re][1][3]  = -hermitian_matrix_[re][1][3];
        hermitian_matrix_[re][2][3]  = -hermitian_matrix_[re][2][3];

        /// Compute coefficients in the correct order
        hermitian_matrix_[re][0][2] += hermitian_matrix_[re][0][1] * hermitian_matrix_[re][1][2];
        hermitian_matrix_[re][0][3] += hermitian_matrix_[re][0][1] * hermitian_matrix_[re][1][3] + hermitian_matrix_[re][0][2] * hermitian_matrix_[re][2][3];
        hermitian_matrix_[re][1][3] += hermitian_matrix_[re][1][2] * hermitian_matrix_[re][2][3];

        /// Copy inverse R^(-1)^H in lower part of the array
        hermitian_matrix_[re][1][0] = conj(hermitian_matrix_[re][0][1]);
        hermitian_matrix_[re][2][0] = conj(hermitian_matrix_[re][0][2]);
        hermitian_matrix_[re][2][1] = conj(hermitian_matrix_[re][1][2]);
        hermitian_matrix_[re][3][0] = conj(hermitian_matrix_[re][0][3]);
        hermitian_matrix_[re][3][1] = conj(hermitian_matrix_[re][1][3]);
        hermitian_matrix_[re][3][2] = conj(hermitian_matrix_[re][2][3]);

#if TIME_MEASURE == 1
        #if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(trace) << "Time to compute the two inverse matrices [ns] : " <<
             std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
        t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute the two inverse matrices [ns] : "
                    << (t2.tv_nsec - t1.tv_nsec) << endl; ;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute the two inverse matrices [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl; ;
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute the two inverse matrices [ns] : "
                                 << (t2 - t1)/TSC_FREQ * 1e9 << endl;

        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#endif
#endif

        *(equalized_symbols_ + 3) += hermitian_matrix_[re][3][0] * *(equalized_symbols_) + hermitian_matrix_[re][3][1] * *(equalized_symbols_ + 1) + hermitian_matrix_[re][3][2] * *(equalized_symbols_ + 2);
        *(equalized_symbols_ + 2) += hermitian_matrix_[re][2][0] * *(equalized_symbols_) + hermitian_matrix_[re][2][1] * *(equalized_symbols_ + 1);
        *(equalized_symbols_ + 1) += hermitian_matrix_[re][1][0] * *(equalized_symbols_);

        *(equalized_symbols_)     /= hermitian_matrix_[re][0][0].real();
        *(equalized_symbols_ + 1) /= hermitian_matrix_[re][1][1].real();
        *(equalized_symbols_ + 2) /= hermitian_matrix_[re][2][2].real();
        *(equalized_symbols_ + 3) /= hermitian_matrix_[re][3][3].real();

        *(equalized_symbols_)     += hermitian_matrix_[re][0][1] * *(equalized_symbols_ + 1) + hermitian_matrix_[re][0][2] * *(equalized_symbols_ + 2) + hermitian_matrix_[re][0][3] * *(equalized_symbols_ + 3);
        *(equalized_symbols_ + 1) += hermitian_matrix_[re][1][2] * *(equalized_symbols_ + 2) + hermitian_matrix_[re][1][3] * *(equalized_symbols_ + 3);
        *(equalized_symbols_ + 2) += hermitian_matrix_[re][2][3] * *(equalized_symbols_ + 3);

        equalized_symbols_ += 4;
//#endif

#if TIME_MEASURE == 1
        #if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();
                BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : " <<
                     std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : "
                                 << (t2 - t1)/TSC_FREQ * 1e9 << endl;
#endif
#endif

    }
}

void vblast_zf_4_layers_float(const vector<vector<complex<float>>> &pdsch_samples_,
                              vector<complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                              int num_re_pdsch_,
                              complex<float> *equalized_symbols_,
                              int nb_rx_ports_) {

    complex<float> hermitian_matrix[4][4];
    complex<float> channel_matrix[4][4];
    complex<float> temp_equalized_symbols[4];
    float real_matrix[4][4];
    float real_matrix_copy[4][4];
    float u_matrix[4][4];
    float v_matrix[4][4];
    float imag_matrix[4][4];
    float imag_matrix_copy[4][4];
    int i, j;

#if defined(DEBUG_VBLAST_4_LAYERS_FLOAT)
        /// Debug matrices
        complex<float> debug_u_matrix[4][4];
        float debug_matrix1[4][4];
        float debug_matrix2[4][4];
        float debug_matrix3[4][4];
        float debug_matrix4[4][4];
        float debug_matrix5[4][4];
        float debug_matrix6[4][4];

        float u_matrix_copy[4][4];

        complex<float> debug_matrix6_complex[4][4];
        float debug_u_matrix_float[4][4];

        float debug_rt_matrix[4][4], debug_r_matrix[4][4], debug_diag_matrix[4][4];
        memset(debug_rt_matrix, 0, 16 * sizeof(float));
        memset(debug_r_matrix, 0, 16 * sizeof(float));
        memset(debug_diag_matrix, 0, 16 * sizeof(float));

        /// To verify that inverse multiplied by original hermitian matrix is equal to identity
        float debug_rt_matrix_u[4][4], debug_r_matrix_u[4][4], debug_diag_matrix_u[4][4];
        memset(debug_rt_matrix_u, 0, 16 * sizeof(float));
        memset(debug_r_matrix_u, 0, 16 * sizeof(float));
        memset(debug_diag_matrix_u, 0, 16 * sizeof(float));
#endif

#if TIME_MEASURE == 1
    #if defined(CLOCK_TYPE_CHRONO)
        std::chrono::steady_clock::time_point t1, t2;
    #elif defined(CLOCK_TYPE_GETTIME)
        struct timespec t1, t2;
    #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        struct timespec t1, t2;
    #else
    uint64_t t1, t2;
        unsigned cycles_low1, cycles_low2, cycles_high1, cycles_high2;
    #endif
#endif
    for(int re = 0; re<num_re_pdsch_; re++) {
#if TIME_MEASURE == 1
BOOST_LOG_TRIVIAL(trace) << " ------------------------- " << endl;
        BOOST_LOG_TRIVIAL(trace) << "RE number : " << re << endl;
    #if defined(CLOCK_TYPE_CHRONO)
        t1 = std::chrono::steady_clock::now();
    #elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
    #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t1);
    #else
        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
    #endif
#endif
    /// First line
    hermitian_matrix[0][0].real(channel_coefficients_[0][0][re].real() * channel_coefficients_[0][0][re].real() +
    channel_coefficients_[0][0][re].imag() * channel_coefficients_[0][0][re].imag());
    hermitian_matrix[0][1] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][1][re];
    hermitian_matrix[0][2] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][2][re];
    hermitian_matrix[0][3] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][3][re];

    /// Second line from diag coef 1,1
    hermitian_matrix[1][1].real(channel_coefficients_[0][1][re].real() * channel_coefficients_[0][1][re].real() +
    channel_coefficients_[0][1][re].imag() * channel_coefficients_[0][1][re].imag());
    hermitian_matrix[1][2] = conj(channel_coefficients_[0][1][re]) * channel_coefficients_[0][2][re];
    hermitian_matrix[1][3] = conj(channel_coefficients_[0][1][re]) * channel_coefficients_[0][3][re];

    /// Third line from diag coef 2,2
    hermitian_matrix[2][2].real(channel_coefficients_[0][2][re].real() * channel_coefficients_[0][2][re].real() +
    channel_coefficients_[0][2][re].imag() * channel_coefficients_[0][2][re].imag());
    hermitian_matrix[2][3] = conj(channel_coefficients_[0][2][re]) * channel_coefficients_[0][3][re];

    /// Fourth line from diag coef 3,3
    hermitian_matrix[3][3].real(channel_coefficients_[0][3][re].real() * channel_coefficients_[0][3][re].real() +
    channel_coefficients_[0][3][re].imag() * channel_coefficients_[0][3][re].imag());

    /// Compute hermitian matrix
    for(i = 1; i < nb_rx_ports_; i++) {
        hermitian_matrix[0][0] += channel_coefficients_[i][0][re].real() * channel_coefficients_[i][0][re].real() +
        channel_coefficients_[i][0][re].imag() * channel_coefficients_[i][0][re].imag();
        hermitian_matrix[0][1] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][1][re];
        hermitian_matrix[0][2] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][2][re];
        hermitian_matrix[0][3] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][3][re];
    //}
    //for(i = 1; i < nb_rx_ports_; i++) {
        hermitian_matrix[1][1] += channel_coefficients_[i][1][re].real() * channel_coefficients_[i][1][re].real() +
                                  channel_coefficients_[i][1][re].imag() * channel_coefficients_[i][1][re].imag();
        hermitian_matrix[1][2] += conj(channel_coefficients_[i][1][re]) * channel_coefficients_[i][2][re];
        hermitian_matrix[1][3] += conj(channel_coefficients_[i][1][re]) * channel_coefficients_[i][3][re];
    //}
    //for(i = 1; i < nb_rx_ports_; i++) {
        hermitian_matrix[2][2] += channel_coefficients_[i][2][re].real() * channel_coefficients_[i][2][re].real() +
                                  channel_coefficients_[i][2][re].imag() * channel_coefficients_[i][2][re].imag();
        hermitian_matrix[2][3] += conj(channel_coefficients_[i][2][re]) * channel_coefficients_[i][3][re];
    //}
    //for(i = 1; i < nb_rx_ports_; i++) {
        hermitian_matrix[3][3] += channel_coefficients_[i][3][re].real() * channel_coefficients_[i][3][re].real() +
                                  channel_coefficients_[i][3][re].imag() * channel_coefficients_[i][3][re].imag();
    }

#if defined(DEBUG_VBLAST_4_LAYERS_FLOAT)
// For debug
            for(i = 0; i < 4; i++) {
                for(j = i+1; j < 4; j++) {
                    hermitian_matrix[j][i] = conj(hermitian_matrix[i][j]);
                }
            }
#endif

#if TIME_MEASURE == 1
    #if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
        << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
        t1 = std::chrono::steady_clock::now();
    #elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
    #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_MONOTONIC, &t1);
    #else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
        << (t2 - t1)/TSC_FREQ * 1e9 << endl;

        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
    #endif
#endif
    /// Separate Real and Imag part into X and Y matrix. Store only diag and upper elements of the Real matrix
    /// Initialize U matrix before modifying X matrix
    for(i = 0; i< 4; i++) {
        for(j = i; j< 4; j++) {
        real_matrix[i][j] = hermitian_matrix[i][j].real();

        imag_matrix[i][j] = hermitian_matrix[i][j].

        imag();

        }
    }
    for(i = 0; i< 4; i++) {
        for(j = i + 1; j< 4; j++) {
        imag_matrix[j][i] = -imag_matrix[i][j];
        real_matrix[j][i] = real_matrix[i][j];
        }
    }
    for(i = 0; i< 4; i++) {
        for(j = 0; j< 4; j++) {
        real_matrix_copy[i][j] = real_matrix[i][j];
        imag_matrix_copy[i][j] = imag_matrix[i][j];
        }
    }

#if defined(DEBUG_VBLAST_4_LAYERS_FLOAT)
cout << "##########################" << endl;
        cout << "Hermitian matrix : " << endl;
        for(i = 0; i < 4; i++) {
            for(j = 0; j < 4; j++) {
                cout << hermitian_matrix[i][j] << " ";
            }
            cout << endl;
        }
        cout << "Real matrix copy : " << endl;
        for(i = 0; i < 4; i++) {
            for(j = 0; j < 4; j++) {
                cout << real_matrix_copy[i][j] << " ";
            }
            cout << endl;
        }
        cout << "Imag matrix copy : " << endl;
        for(i = 0; i < 4; i++) {
            for(j = 0; j < 4; j++) {
                cout << imag_matrix_copy[i][j] << " ";
            }
            cout << endl;
        }
#endif

#if TIME_MEASURE == 1
    #if defined(CLOCK_TYPE_CHRONO)
    t2 = std::chrono::steady_clock::now();

        BOOST_LOG_TRIVIAL(trace) << "Time to separate real and imag parts [ns] : " <<
             std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

        t1 = std::chrono::steady_clock::now();
    #elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to separate real and imag parts [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
    #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to separate real and imag parts [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_MONOTONIC, &t1);
    #else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to separate real and imag parts [ns] : "
                                 << (t2 - t1)/TSC_FREQ * 1e9 << endl;

        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
    #endif
#endif
    /// Compute RDR decomp of real_matrix
    ldl_decomp_test(real_matrix,4);

    /// Compute Inverse of real_matrix
    real_matrix[0][1] = -real_matrix[0][1];
    real_matrix[0][2] = -real_matrix[0][2];
    real_matrix[0][3] = -real_matrix[0][3];
    real_matrix[1][2] = -real_matrix[1][2];
    real_matrix[1][3] = -real_matrix[1][3];
    real_matrix[2][3] = -real_matrix[2][3];

    /// Compute coefficients in the correct order
    real_matrix[0][2] += real_matrix[0][1] * real_matrix[1][2];
    real_matrix[0][3] += real_matrix[0][1] * real_matrix[1][3] + real_matrix[0][2] * real_matrix[2][3];
    real_matrix[1][3] += real_matrix[1][2] * real_matrix[2][3];

#if TIME_MEASURE == 1
    #if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(trace) << "Time to compute RDR and inverse of real matrix [ns] : " <<
             std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
        t1 = std::chrono::steady_clock::now();
    #elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute inverse of real matrix [ns] : "
                    << (t2.tv_nsec - t1.tv_nsec) << endl; ;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
    #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute inverse of real matrix [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl; ;
        clock_gettime(CLOCK_MONOTONIC, &t1);
    #else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute inverse of real matrix [ns] : "
                                 << (t2 - t1)/TSC_FREQ * 1e9 << endl;

        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
    #endif
#endif
#if defined(DEBUG_VBLAST_4_LAYERS_FLOAT)
/**************************************** DEBUG : COMPUTE NAIVE U *********************************************************************/
            for(i = 0; i < 4; i++) {
                debug_diag_matrix[i][i] = 1/real_matrix[i][i];
            }
            cout << "Debug diag matrix : " << endl;
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    cout << debug_diag_matrix[i][j] << " ";
                }
                cout << endl;
            }
            for(i = 0; i < 4; i++) {
                for(j = i + 1; j < 4; j++) {
                    debug_rt_matrix[j][i] = real_matrix[i][j];
                }
                debug_rt_matrix[i][i] = 1;
            }
            cout << "Debug Rt matrix " << endl;
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    cout << debug_rt_matrix[i][j] << " ";
                }
                cout << endl;
            }
            for(i = 0; i < 4; i++) {
                for(j = i + 1; j < 4; j++) {
                    debug_r_matrix[i][j] = real_matrix[i][j];
                }
                debug_r_matrix[i][i] = 1;
            }
            cout << "Debug r matrix : " << endl;
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    cout << debug_r_matrix[i][j] << " ";
                }
                cout << endl;
            }

            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    debug_matrix1[i][j] = imag_matrix_copy[i][j];
                }
            }
            cout << "debug matrix 1 (copy of imag matrix) :" << endl;
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    cout << debug_matrix1[i][j] << " ";
                }
                cout << endl;
            }

            memset(debug_matrix2, 0, 16 * sizeof(float));
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    for(int k = 0; k < 4; k++) {
                        debug_matrix2[i][j] += debug_rt_matrix[i][k] * debug_matrix1[k][j];
                    }
                }
            }
            memset(debug_matrix3, 0, 16 * sizeof(float));
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    for(int k = 0; k< 4; k++) {
                        debug_matrix3[i][j] += debug_diag_matrix[i][k] * debug_matrix2[k][j];
                    }
                }
            }
            memset(debug_matrix4, 0, 16 * sizeof(float));
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    for(int k =0; k < 4; k++) {
                        debug_matrix4[i][j] += debug_r_matrix[i][k] * debug_matrix3[k][j];
                    }
                }
            }
            memset(debug_matrix5, 0, 16 * sizeof(float));
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    for(int k =0; k < 4; k++) {
                        debug_matrix5[i][j] += imag_matrix_copy[i][k] * debug_matrix4[k][j];
                    }
                }
            }
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    debug_matrix5[i][j] += real_matrix_copy[i][j];
                }
            }

            cout << "Naive U : " << endl;
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    cout << debug_matrix5[i][j] << " ";
                }
                cout << endl;
            }
/*********************************************************************************************************************/
#endif
    /// Compute D^(-1)(Rt)^(-1)Y
    for(j = 0; j< 4; j++) {
        ///  (Rt)^(-1)Y 4th row
        imag_matrix[3][j] += real_matrix[0][3] * imag_matrix[0][j] +
                             real_matrix[1][3] * imag_matrix[1][j] +
                             real_matrix[2][3] * imag_matrix[2][j];
        ///  (Rt)^(-1)Y 3rd row
        imag_matrix[2][j] += real_matrix[0][2] * imag_matrix[0][j] +
                             real_matrix[1][2] * imag_matrix[1][j];
        ///  (Rt)^(-1)Y 2nd row
        imag_matrix[1][j] += real_matrix[0][1] * imag_matrix[0][j];
    }
    for(j = 0; j< 4; j++) {
        imag_matrix[3][j] /= real_matrix[3][3];
        imag_matrix[2][j] /= real_matrix[2][2];
        imag_matrix[1][j] /= real_matrix[1][1];
        ///  (Rt)^(-1)Y 1st row
        imag_matrix[0][j] /= real_matrix[0][0];
    }

    /// Compute R^(-1)D^(-1)(Rt)^(-1)Y
    for(j = 0; j < 4; j++) {
        ///  1st row
        imag_matrix[0][j] += real_matrix[0][1] * imag_matrix[1][j] + real_matrix[0][2] * imag_matrix[2][j] + real_matrix[0][3] * imag_matrix[3][j];
        ///  2nd row
        imag_matrix[1][j] += real_matrix[1][2] * imag_matrix[2][j] + real_matrix[1][3] * imag_matrix[3][j];
        /// 3rd row
        imag_matrix[2][j] += real_matrix[2][3] * imag_matrix[3][j];
        /// 4th row remains unchanged
    }

    /// Compute YR^(-1)D^(-1)(Rt)^(-1)Y. Compute only upper triangular part because the matrix is symmetric.
    u_matrix[0][0] = imag_matrix_copy[0][0] * imag_matrix[0][0];
    u_matrix[0][1] = imag_matrix_copy[0][0] * imag_matrix[0][1];
    u_matrix[0][2] = imag_matrix_copy[0][0] * imag_matrix[0][2];
    u_matrix[0][3] = imag_matrix_copy[0][0] * imag_matrix[0][3];

    u_matrix[1][1] = imag_matrix_copy[1][0] * imag_matrix[0][1];
    u_matrix[1][2] = imag_matrix_copy[1][0] * imag_matrix[0][2];
    u_matrix[1][3] = imag_matrix_copy[1][0] * imag_matrix[0][3];

    u_matrix[2][2] = imag_matrix_copy[2][0] * imag_matrix[0][2];
    u_matrix[2][3] = imag_matrix_copy[2][0] * imag_matrix[0][3];

    u_matrix[3][3] = imag_matrix_copy[3][0] * imag_matrix[0][3];
    for(i = 1; i < 4; i++) {
        u_matrix[0][0] += imag_matrix_copy[0][i] * imag_matrix[i][0];
        u_matrix[0][1] += imag_matrix_copy[0][i] * imag_matrix[i][1];
        u_matrix[0][2] += imag_matrix_copy[0][i] * imag_matrix[i][2];
        u_matrix[0][3] += imag_matrix_copy[0][i] * imag_matrix[i][3];

        u_matrix[1][1] += imag_matrix_copy[1][i] * imag_matrix[i][1];
        u_matrix[1][2] += imag_matrix_copy[1][i] * imag_matrix[i][2];
        u_matrix[1][3] += imag_matrix_copy[1][i] * imag_matrix[i][3];

        u_matrix[2][2] += imag_matrix_copy[2][i] * imag_matrix[i][2];
        u_matrix[2][3] += imag_matrix_copy[2][i] * imag_matrix[i][3];

        u_matrix[3][3] += imag_matrix_copy[3][i] * imag_matrix[i][3];
    }

    /// Add X matrix elements
    for(i = 0; i < 4; i++) {
        for(j = i; j < 4; j++) {
            u_matrix[i][j] += real_matrix_copy[i][j];
        }
    }

    /// Copy upper elements into lower elements (U is symmetric) to compute YU during the next step.
    for(i = 0; i < 4; i++) {
        for(j = i + 1; j < 4; j++) {
            u_matrix[j][i] = u_matrix[i][j];
        }
    }

    /// Compute inverse of U matrix
    ldl_decomp_test(u_matrix, 4);

    /// Compute Inverse of real_matrix
    u_matrix[0][1] = -u_matrix[0][1];
    u_matrix[0][2] = -u_matrix[0][2];
    u_matrix[0][3] = -u_matrix[0][3];
    u_matrix[1][2] = -u_matrix[1][2];
    u_matrix[1][3] = -u_matrix[1][3];
    u_matrix[2][3] = -u_matrix[2][3];

    /// Compute coefficients in the correct order
    u_matrix[0][2] += u_matrix[0][1] * u_matrix[1][2];
    u_matrix[0][3] += u_matrix[0][1] * u_matrix[1][3] + u_matrix[0][2] * u_matrix[2][3];
    u_matrix[1][3] += u_matrix[1][2] * u_matrix[2][3];
#if defined(DEBUG_VBLAST_4_LAYERS_FLOAT)
/**************************************** DEBUG : GET RDR decomp of inverted U *********************************************************************/
            for(i = 0; i < 4; i++) {
                debug_diag_matrix[i][i] = 1/u_matrix[i][i];
                debug_diag_matrix_u[i][i] = 1/u_matrix[i][i];
            }
            cout << "Debug diag matrix : " << endl;
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    cout << debug_diag_matrix[i][j] << " ";
                }
                cout << endl;
            }
            for(i = 0; i < 4; i++) {
                for(j = i + 1; j < 4; j++) {
                    debug_rt_matrix[j][i] = u_matrix[i][j];
                    debug_rt_matrix_u[j][i] = u_matrix[i][j];
                }
                debug_rt_matrix[i][i] = 1;
                debug_rt_matrix_u[i][i] = 1;
            }
            cout << "Debug Rt matrix " << endl;
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    cout << debug_rt_matrix[i][j] << " ";
                }
                cout << endl;
            }
            for(i = 0; i < 4; i++) {
                for(j = i + 1; j < 4; j++) {
                    debug_r_matrix[i][j] = u_matrix[i][j];
                    debug_r_matrix_u[i][j] = u_matrix[i][j];
                }
                debug_r_matrix[i][i] = 1;
                debug_r_matrix_u[i][i] = 1;
            }
            cout << "Debug r matrix : " << endl;
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    cout << debug_r_matrix[i][j] << " ";
                }
                cout << endl;
            }
/***********************************************************************************************************************/
#endif
#if TIME_MEASURE == 1
    #if defined(CLOCK_TYPE_CHRONO)
            t2 = std::chrono::steady_clock::now();

            BOOST_LOG_TRIVIAL(trace) << "Time to compute U matrix [ns] : " <<
                 std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

            t1 = std::chrono::steady_clock::now();
    #elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute U matrix [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
        #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute U matrix [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_MONOTONIC, &t1);
    #else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute U matrix [ns] : "
                                 << (t2 - t1)/TSC_FREQ * 1e9 << endl;

        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
    #endif
#endif

/// Compute YU and store the result in V matrix. Compute all matrix elements.
#if defined(DEBUG_VBLAST_4_LAYERS_FLOAT)
/**************************************** DEBUG : Compute YU *********************************************************************/
            // Naive matrix product between Y and U
            memset(debug_matrix1, 0, 16 * sizeof(float));
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    for(int k = 0; k < 4; k++) {
                        debug_matrix1[i][j] += debug_diag_matrix[i][k] * debug_rt_matrix[k][j];
                    }
                }
            }
            memset(debug_matrix2, 0, 16 * sizeof(float));
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    for(int k = 0; k < 4; k++) {
                        debug_matrix2[i][j] += debug_r_matrix[i][k] * debug_matrix1[k][j];
                    }
                }
            }
            memset(debug_matrix3, 0, 16 * sizeof(float));
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    for(int k = 0; k < 4; k++) {
                        debug_matrix3[i][j] += imag_matrix_copy[i][k] * debug_matrix2[k][j];
                    }
                }
            }

            cout << "Naive YU : " << endl;
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    cout << debug_matrix3[i][j] << " ";
                }
                cout << endl;
            }
/**************************************** DEBUG : Compute -X^(-1)YU *********************************************************************/
            /// Get back RDR decomp from X^(-1)
            for(i = 0; i < 4; i++) {
                debug_diag_matrix[i][i] = 1/real_matrix[i][i];
            }
            cout << "Debug diag matrix : " << endl;
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    cout << debug_diag_matrix[i][j] << " ";
                }
                cout << endl;
            }
            for(i = 0; i < 4; i++) {
                for(j = i + 1; j < 4; j++) {
                    debug_rt_matrix[j][i] = real_matrix[i][j];
                }
                debug_rt_matrix[i][i] = 1;
            }
            cout << "Debug Rt matrix " << endl;
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    cout << debug_rt_matrix[i][j] << " ";
                }
                cout << endl;
            }
            for(i = 0; i < 4; i++) {
                for(j = i + 1; j < 4; j++) {
                    debug_r_matrix[i][j] = real_matrix[i][j];
                }
                debug_r_matrix[i][i] = 1;
            }
            cout << "Debug r matrix : " << endl;
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    cout << debug_r_matrix[i][j] << " ";
                }
                cout << endl;
            }

            /// compute R^(-T)YU
            memset(debug_matrix4, 0, 16 * sizeof(float));
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    for(int k = 0; k < 4; k++) {
                        debug_matrix4[i][j] += debug_rt_matrix[i][k] * debug_matrix3[k][j];
                    }
                }
            }
            /// compute D^(-1)R^(-T)YU
            memset(debug_matrix5, 0, 16 * sizeof(float));
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    for(int k = 0; k < 4; k++) {
                        debug_matrix5[i][j] += debug_diag_matrix[i][k] * debug_matrix4[k][j];
                    }
                }
            }
            /// compute R^(-1)D^(-1)R^(-T)YU
            memset(debug_matrix6, 0, 16 * sizeof(float));
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    for(int k = 0; k < 4; k++) {
                        debug_matrix6[i][j] += debug_r_matrix[i][k] * debug_matrix5[k][j];
                    }
                    debug_matrix6[i][j] = -debug_matrix6[i][j];
                }
            }

            cout << "Naive V : " << endl;
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    cout << debug_matrix6[i][j] << " ";
                }
                cout << endl;
            }
#endif
/************************************** Product beginning on the right ***********************************/
    /// Modify u_matrix so that lower part contains D^(-1)R^(-T)
    for(i = 0; i < 4; i++) {
        for(j = i + 1; j < 4; j++) {
            u_matrix[j][i] = u_matrix[i][j]/u_matrix[j][j];
        }
        u_matrix[i][i] = 1/u_matrix[i][i];
    }

    /// R^(-1)D^(-1)R^(-T). In-place & symmetric
    for(j = 0; j < 4; j++) {
        for(i = j; i < 4; i++) {
            for(int k = i + 1;k < 4; k++) {
                u_matrix[i][j] += u_matrix[j][k] * u_matrix[k][i];
            }
        }
    }
    /// Copy lower elements into upper elements
    for(i = 0; i < 4; i++) {
        for(j = i + 1; j < 4; j++) {
            u_matrix[i][j] = u_matrix[j][i];
        }
    }
    /// Compute YU and load it into V
    for(i = 0; i < 4; i++) {
        for(j = 0; j < 4; j++) {
            v_matrix[i][j] = imag_matrix_copy[i][0] * u_matrix[0][j];
        }
    }
    for(i = 0; i < 4; i++) {
        for(j = 0; j < 4; j++) {
            for(int k = 1; k < 4; k++) {
                v_matrix[i][j] += imag_matrix_copy[i][k] * u_matrix[k][j];
            }
        }
    }
#if defined(DEBUG_VBLAST_4_LAYERS_FLOAT)
    cout << "YU optimized : " << endl;
                for(i = 0; i < 4; i++) {
                    for(j = 0; j < 4; j++) {
                        cout << v_matrix[i][j] << " ";
                    }
                    cout << endl;
                }
#endif
/************************************** Product beginning on the left ***********************************/


/**********************************************************************************************/

    /// Compute -X^(-1)YU
    /// Compute D^(-1)(Rt)^(-1)YU
    v_matrix[3][3] = 0; // Set last diag element to zero directly
    for(j = 1; j < 4; j++) {
        /// 4th row
        v_matrix[3][j] += real_matrix[0][3] * v_matrix[0][j] + real_matrix[1][3] * v_matrix[1][j] + real_matrix[2][3] * v_matrix[2][j];
        v_matrix[3][j] /= real_matrix[3][3];
        }
    for(j = 1; j < 4; j++) {
        ///  3rd row
        v_matrix[2][j] += real_matrix[0][2] * v_matrix[0][j] + real_matrix[1][2] * v_matrix[1][j];
        v_matrix[2][j] /= real_matrix[2][2];
        /// 2nd row
        v_matrix[1][j] += real_matrix[0][1] * v_matrix[0][j];
        v_matrix[1][j] /= real_matrix[1][1];
        /// 1st row
        v_matrix[0][j] /= real_matrix[0][0];
    }

    /// Compute R^(-1)D^(-1)(Rt)^(-1)YU
    v_matrix[0][0] = 0;
    for(j = 1; j < 4; j++) {
        ///  1st row
        v_matrix[0][j] += real_matrix[0][1] * v_matrix[1][j] + real_matrix[0][2] * v_matrix[2][j] + real_matrix[0][3] * v_matrix[3][j];
    }
    v_matrix[1][1] = 0;
    for(j = 2; j < 4; j++) {
        ///  2nd row
        v_matrix[1][j] += real_matrix[1][2] * v_matrix[2][j] + real_matrix[1][3] * v_matrix[3][j];
    }
    /// 3rd row. element 2,3 remains unchanged because v_matrix[3][3] is equal to zero.
    v_matrix[2][2] = 0;
    //v_matrix[2][3]; += real_matrix[2][3] * v_matrix[3][3];
    /// 4th row remains unchanged

    /// Copy upper elements into lower elements (V is antisymmetric with diagonal elements equal to 0 and multiplied by -1)
    for(i = 0; i < 4; i++) {
        for(j = i + 1; j < 4; j++) {
            v_matrix[j][i] = v_matrix[i][j];
        }
    }
    /// Multiply upper elements by -1
    for(i = 0; i < 4; i++) {
        for(j = i + 1; j < 4; j++) {
            v_matrix[i][j] = - v_matrix[i][j];
        }
    }

#if defined(DEBUG_VBLAST_4_LAYERS_FLOAT)
// print v matrix
            cout << "V matrix : " << endl;
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    cout << v_matrix[i][j] << " ";
                }
                cout << endl;
            }
#endif

#if TIME_MEASURE == 1
#if defined(CLOCK_TYPE_CHRONO)
t2 = std::chrono::steady_clock::now();

                BOOST_LOG_TRIVIAL(trace) << "Time to compute V matrix [ns] : " <<
                     std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

                t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute V matrix [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute V matrix [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute V matrix [ns] : "
        << (t2 - t1)/TSC_FREQ * 1e9 << endl;

        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#endif
#endif

#if defined(DEBUG_VBLAST_4_LAYERS_FLOAT)
// Compute (HhH)^(-1)
            memset(debug_matrix2, 0, 16 * sizeof(float));
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    for(int k = 0; k < 4; k++) {
                        debug_matrix2[i][j] += debug_diag_matrix_u[i][k] * debug_rt_matrix_u[k][j];
                    }
                }
            }
            memset(debug_u_matrix_float, 0, 16 * sizeof(float));
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    for(int k = 0; k < 4; k++) {
                        debug_u_matrix_float[i][j] += debug_r_matrix_u[i][k] * debug_matrix2[k][j];
                    }
                }
            }

            /// Verify that U_matrix multiplied by U gives identity matrix
            memset(debug_matrix1, 0, 16 * sizeof(float));
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    for(int k = 0; k< 4; k++) {
                        debug_matrix1[i][j] += debug_u_matrix_float[i][k] * u_matrix_copy[k][j];
                    }
                }
            }
            cout << "U^(-1) * U :" << endl;
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    cout << debug_matrix1[i][j] << " ";
                }
                cout << endl;
            }

            /// Add jV to debug matrix
            memset(debug_u_matrix, 0, 16 * sizeof(complex<float>));
            for(int i = 0; i < 4; i++) {
                for(int j = 0; j < 4; j++) {
                    debug_u_matrix[i][j] = complex<float>(debug_u_matrix_float[i][j], v_matrix[i][j]);
                }
            }

            // Print H^(-1) matrix
            cout << " (H^h H)^(-1) : " << endl;
            for(int i = 0; i < 4; i++) {
                for(int j = 0; j < 4; j++) {
                    cout << debug_u_matrix[i][j] << " ";
                }
                cout << endl;
            }

            /// Multiply debug matrix by hermitian matrix. Verify the result is equal to the identity matrix.
            memset(debug_matrix6_complex, 0, 16 * sizeof(complex<float>));
            for(i = 0; i < 4; i++) {
                for(j = 0; j < 4; j++) {
                    for(int k = 0; k < 4; k++) {
                        debug_matrix6_complex[i][j] += debug_u_matrix[i][k] * hermitian_matrix[k][j];
                    }
                }
            }

            cout << "Inverse by Hermitian product : " << endl;
            for(int i = 0; i < 4; i++) {
                for(int j = 0; j < 4; j++) {
                    cout << debug_matrix6_complex[i][j] << " ";
                }
                cout << endl;
            }
#endif
    /// Multiply received signal y by H^H, then multiply by the inverse (U + jV)
    for(i = 0; i < 4; i++) {
        *(temp_equalized_symbols + i) = conj(channel_coefficients_[0][i][re])
                                        * pdsch_samples_[0][re];
    }
    for(j = 1; j < nb_rx_ports_; j++) {
        for(i = 0; i < 4; i++) {
            *(temp_equalized_symbols + i) +=
            conj(channel_coefficients_[j][i][re])
            * pdsch_samples_[j][re];
        }
    }

    *(equalized_symbols_)     = u_matrix[0][0] * *(temp_equalized_symbols);
    *(equalized_symbols_ + 1) = u_matrix[1][0] * *(temp_equalized_symbols);
    *(equalized_symbols_ + 2) = u_matrix[2][0] * *(temp_equalized_symbols);
    *(equalized_symbols_ + 3) = u_matrix[3][0] * *(temp_equalized_symbols);
    for(i = 1; i < 4; i++) {
        *(equalized_symbols_)     += u_matrix[0][i] * *(temp_equalized_symbols + i);
        *(equalized_symbols_ + 1) += u_matrix[1][i] * *(temp_equalized_symbols + i);
        *(equalized_symbols_ + 2) += u_matrix[2][i] * *(temp_equalized_symbols + i);
        *(equalized_symbols_ + 3) += u_matrix[3][i] * *(temp_equalized_symbols + i);
    }

    for(i = 0; i < 4; i++) {
        for(j = 0; j < 4; j++) {
            (equalized_symbols_ + i)->real((equalized_symbols_ + i)->real() -
                                            v_matrix[i][j] * (temp_equalized_symbols + j)->imag());
            (equalized_symbols_ + i)->imag((equalized_symbols_ + i)->imag() +
                                            v_matrix[i][j] * (temp_equalized_symbols + j)->real());
        }
    }

    equalized_symbols_ += 4;
#if TIME_MEASURE == 1
#if defined(CLOCK_TYPE_CHRONO)
t2 = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : " <<
             std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
#elif defined(CLOCK_TYPE_GETTIME)
clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : "
                         << (t2.tv_nsec - t1.tv_nsec) << endl;
clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
clock_gettime(CLOCK_MONOTONIC, &t2);
BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : "
                         << (t2.tv_nsec - t1.tv_nsec) << endl;
clock_gettime(CLOCK_MONOTONIC, &t1);
#else
asm volatile("RDTSCP\n\t"
             "mov %%edx, %0\n\t"
             "mov %%eax, %1\n\t"
             "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
        "%rax", "%rbx", "%rcx", "%rdx");

t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : "
                         << (t2 - t1)/TSC_FREQ * 1e9 << endl;
#endif
#endif
    }
}

#if defined(__AVX2__)
void call_vblast_zf_avx_functions(int num_layers,
                              const vector<vector<complex<float>>> &pdsch_samples_,
                              vector<complex<float>>
                              channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                              int num_re_pdsch_,
                              complex<float> *equalized_symbols_,
                              int nb_rx_ports_) {

    if(num_layers == 2) {
        vblast_zf_2_layers_avx(pdsch_samples_,
                               channel_coefficients_,
                               num_re_pdsch_,
                               equalized_symbols_,
                               nb_rx_ports_);
    } else if (num_layers == 3) {
        vblast_zf_3_layers_avx(pdsch_samples_,
                               channel_coefficients_,
                               num_re_pdsch_,
                               equalized_symbols_,
                               nb_rx_ports_);

    } else if(num_layers == 4) {
        vblast_zf_4_layers_avx(pdsch_samples_,
                               channel_coefficients_,
                               num_re_pdsch_,
                               equalized_symbols_,
                               nb_rx_ports_);
    }

}
#endif

void call_vblast_zf_functions(int num_layers,
                              const vector<vector<complex<float>>> &pdsch_samples_,
                              vector<complex<float>>
                              channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                              int num_re_pdsch_,
                              complex<float> *equalized_symbols_,
                              int nb_rx_ports_) {

    if(num_layers == 2) {
        vblast_zf_2_layers(pdsch_samples_,
                           channel_coefficients_,
                           num_re_pdsch_,
                           equalized_symbols_,
                           nb_rx_ports_);
    } else if (num_layers == 3) {
        vblast_zf_3_layers(pdsch_samples_,
                channel_coefficients_,
                num_re_pdsch_,
                equalized_symbols_,
                nb_rx_ports_);
    } else if(num_layers == 4) {
        vblast_zf_4_layers(pdsch_samples_,
                channel_coefficients_,
                num_re_pdsch_,
                equalized_symbols_,
                nb_rx_ports_);
    }
}

void vblast_zf(const vector<vector<complex<float>>> &pdsch_samples_,
               vector<complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
               int num_re_pdsch_,
               complex<float> * equalized_symbols_,
               int nb_tx_dmrs_ports_,
               int nb_rx_ports_) {

    /// Hardcode (H^h . H)^(-1)
    switch(nb_tx_dmrs_ports_) {
        case 2: {

            /// Compute the inverse of the channel matrix directly
            if(nb_rx_ports_ == 2) {

#if TIME_MEASURE == 1
                std::chrono::steady_clock::time_point  t1, t2;
#endif
                /**
                complex<float> *h00 = channel_coefficients_[0][0].data(); /// h_rx_tx
                complex<float> *h01 = channel_coefficients_[0][1].data();
                complex<float> *h10 = channel_coefficients_[1][0].data();
                complex<float> *h11 = channel_coefficients_[1][1].data(); */

                complex<float> det = 0;

                for (int re = 0; re < num_re_pdsch_; re++) {

#if TIME_MEASURE == 1
                    t1 = std::chrono::steady_clock::now();
#endif
                    /**
                    det = h00[re] * h11[re] - h10[re] * h01[re];

                    *(equalized_symbols_) = h11[re] * pdsch_samples_[0][re] - h01[re] *  pdsch_samples_[1][re];
                    *(equalized_symbols_ + 1) = -h10[re] * pdsch_samples_[0][re] + h00[re] * pdsch_samples_[1][re]; */


                    det = channel_coefficients_[0][0][re] * channel_coefficients_[1][1][re] - channel_coefficients_[1][0][re] * channel_coefficients_[0][1][re];

                    *(equalized_symbols_) = channel_coefficients_[1][1][re] * pdsch_samples_[0][re] - channel_coefficients_[0][1][re] *  pdsch_samples_[1][re];
                    *(equalized_symbols_ + 1) = -channel_coefficients_[1][0][re] * pdsch_samples_[0][re] + channel_coefficients_[0][0][re] * pdsch_samples_[1][re];

                    *(equalized_symbols_) *= conj(det)/(1.0f * abs(det) * abs(det));
                    *(equalized_symbols_ + 1) *= conj(det)/(1.0f * abs(det) * abs(det));

                    //*(equalized_symbols_) /= det;
                    //*(equalized_symbols_ + 1) /= det;

                    /**
                    *(equalized_symbols_) *= conj(channel_coefficients_[0][0][re] * channel_coefficients_[1][1][re] - channel_coefficients_[1][0][re] * channel_coefficients_[0][1][re])/
                            (1.0f * abs(channel_coefficients_[0][0][re] * channel_coefficients_[1][1][re] - channel_coefficients_[1][0][re] * channel_coefficients_[0][1][re]) *
                            abs(channel_coefficients_[0][0][re] * channel_coefficients_[1][1][re] - channel_coefficients_[1][0][re] * channel_coefficients_[0][1][re]));
                    *(equalized_symbols_ + 1) *= conj(channel_coefficients_[0][0][re] * channel_coefficients_[1][1][re] - channel_coefficients_[1][0][re] * channel_coefficients_[0][1][re])/
                            (1.0f * abs(channel_coefficients_[0][0][re] * channel_coefficients_[1][1][re] - channel_coefficients_[1][0][re] * channel_coefficients_[0][1][re]) *
                            abs(channel_coefficients_[0][0][re] * channel_coefficients_[1][1][re] - channel_coefficients_[1][0][re] * channel_coefficients_[0][1][re])); */

                    //*(equalized_symbols_) /= channel_coefficients_[0][0][re] * channel_coefficients_[1][1][re] - channel_coefficients_[1][0][re] * channel_coefficients_[0][1][re];
                    //*(equalized_symbols_ + 1) /= channel_coefficients_[0][0][re] * channel_coefficients_[1][1][re] - channel_coefficients_[1][0][re] * channel_coefficients_[0][1][re];

                    equalized_symbols_ += 2;

                    /**
                    det = h00[re] * h11[re] - h10[re] * h01[re];

                    equalized_symbols_[2 * re] = h11[re] * pdsch_samples_[0][re] - h01[re] *  pdsch_samples_[1][re];
                    equalized_symbols_[2 * re + 1] = -h10[re] * pdsch_samples_[0][re] + h00[re] * pdsch_samples_[1][re];

                    equalized_symbols_[2 * re] *= conj(det)/(1.0f * abs(det) * abs(det));
                    equalized_symbols_[2 * re + 1] *= conj(det)/(1.0f * abs(det) * abs(det)); */

#if TIME_MEASURE == 1
                    t2 = std::chrono::steady_clock::now();
                    BOOST_LOG_TRIVIAL(trace) << "inversion on 1 RE : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
#endif
                }
            } else {
                complex<float> temp_coef;/// h01 and h10 from H^h . H matrix.
                float temp_matrix_float[2]; /// h00 and h11 from from H^h . H matrix.
                float det = 0;
                complex<float> *h00 = channel_coefficients_[0][0].data(); /// h_rx_tx
                complex<float> *h01 = channel_coefficients_[0][1].data();
                complex<float> *h10 = channel_coefficients_[1][0].data();
                complex<float> *h11 = channel_coefficients_[1][1].data();
                int symbol = 0;
                int sc = 0;

                for (int re = 0; re < num_re_pdsch_; re++) {


                    temp_matrix_float[0] = h00[re].real() * h00[re].real() + h00[re].imag() * h00[re].imag()
                                           + h10[re].real() * h10[re].real() +
                                           h10[re].imag() * h10[re].imag(); /// Coef 0,0
                    temp_matrix_float[1] = h01[re].real() * h01[re].real() + h01[re].imag() * h01[re].imag()
                                           + h11[re].real() * h11[re].real() +
                                           h11[re].imag() * h11[re].imag(); /// Coef 1, 1
                    temp_coef = conj(h00[re]) * h01[re] + conj(h10[re]) * h11[re]; /// Coef 0,1, and conj of Coef 1, 0

                    det = temp_matrix_float[0] * temp_matrix_float[1] -
                          pow(abs(temp_coef), 2);

                    *(equalized_symbols_) = temp_matrix_float[1] * (conj(h00[re]) * pdsch_samples_[0][re] +
                                                                         conj(h10[re]) * pdsch_samples_[1][re])
                                                 - temp_coef * (conj(h01[re]) * pdsch_samples_[0][re] +
                                                                conj(h11[re]) * pdsch_samples_[1][re]);

                    *(equalized_symbols_ + 1) =
                            -conj(temp_coef) * (conj(h00[re]) * pdsch_samples_[0][re] +
                                                conj(h10[re]) * pdsch_samples_[1][re])
                            + temp_matrix_float[0] * (conj(h01[re]) * pdsch_samples_[0][re] +
                                                      conj(h11[re]) * pdsch_samples_[1][re]);

                    /**
                    temp_matrix_float[0] = channel_coefficients_[0][0][re].real() * channel_coefficients_[0][0][re].real() +
                                           channel_coefficients_[0][0][re].imag() * channel_coefficients_[0][0][re].imag()
                                           + channel_coefficients_[1][0][re].real() * channel_coefficients_[1][0][re].real() +
                                             channel_coefficients_[1][0][re].imag() * channel_coefficients_[1][0][re].imag(); /// Coef 0,0
                    temp_matrix_float[1] = channel_coefficients_[0][1][re].real() * channel_coefficients_[0][1][re].real() +
                                           channel_coefficients_[0][1][re].imag() * channel_coefficients_[0][1][re].imag()
                                           + channel_coefficients_[1][1][re].real() * channel_coefficients_[1][1][re].real() +
                                           channel_coefficients_[1][1][re].imag() * channel_coefficients_[1][1][re].imag(); /// Coef 1, 1
                    temp_coef = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][1][re] + conj(channel_coefficients_[1][0][re]) * channel_coefficients_[1][1][re]; /// Coef 0,1, and conj of Coef 1, 0

                    det = temp_matrix_float[0] * temp_matrix_float[1] -
                          pow(abs(temp_coef), 2);

                    *(equalized_symbols_) = temp_matrix_float[1] * (conj(channel_coefficients_[0][0][re]) * pdsch_samples_[0][re] +
                                                                    conj(channel_coefficients_[1][0][re]) * pdsch_samples_[1][re])
                                            - temp_coef * (conj(channel_coefficients_[0][1][re]) * pdsch_samples_[0][re] +
                                                           conj(channel_coefficients_[1][1][re]) * pdsch_samples_[1][re]);

                    *(equalized_symbols_ + 1) =
                            -conj(temp_coef) * (conj(channel_coefficients_[0][0][re]) * pdsch_samples_[0][re] +
                                                conj(channel_coefficients_[1][0][re]) * pdsch_samples_[1][re])
                            + temp_matrix_float[0] * (conj(channel_coefficients_[0][1][re]) * pdsch_samples_[0][re] +
                                                      conj(channel_coefficients_[1][1][re]) * pdsch_samples_[1][re]); */

                    *(equalized_symbols_) /= det;
                    *(equalized_symbols_ + 1) /= det;
                    equalized_symbols_ += 2;
                }
            }
            break;
        }

        case 3 :
        {

#if TIME_MEASURE == 1
            std::chrono::steady_clock::time_point  t1, t2;
#endif

            /// Work on the channel matrix
            //if(nb_rx_ports_ == 3) {



            //} else { /// Compute H^H * H to work on a square matrix
                complex<float> temp_coefs[3]; /// 3 upper elements of H^H.H
                float temp_matrix_float[3]; /// diagonal elements of H^H.H
                float det = 0;
                /**
                complex<float> *h00 = channel_coefficients_[0][0].data(); /// h_rx_tx
                complex<float> *h01 = channel_coefficients_[0][1].data();
                complex<float> *h02 = channel_coefficients_[0][2].data();

                complex<float> *h10 = channel_coefficients_[1][0].data();
                complex<float> *h11 = channel_coefficients_[1][1].data();
                complex<float> *h12 = channel_coefficients_[1][2].data();

                complex<float> *h20 = channel_coefficients_[2][0].data();
                complex<float> *h21 = channel_coefficients_[2][1].data();
                complex<float> *h22 = channel_coefficients_[2][2].data(); */

                vector<complex<float>> temp_mrc(3);
                vector<complex<float>> temp_inv(6);

                for (int re = 0; re < num_re_pdsch_; re++) {

                    /**
                    cout << "h00 : " << h00[re] << endl;
                    cout << "h01 : " << h01[re] << endl;
                    cout << "h02 : " << h02[re] << endl;
                    cout << "h10 : " << h10[re] << endl;
                    cout << "h11 : " << h11[re] << endl;
                    cout << "h12 : " << h12[re] << endl;
                    cout << "h20 : " << h20[re] << endl;
                    cout << "h21 : " << h21[re] << endl;
                    cout << "h22 : " << h22[re] << endl;

                    */

#if TIME_MEASURE == 1
       t1 = std::chrono::steady_clock::now();
#endif

                    //temp_matrix_float[0] = abs(h00[re]) + abs(h10[re]) + abs(h20[re]); /// h00
                    //temp_matrix_float[1] = abs(h01[re]) + abs(h11[re]) + abs(h21[re]); /// h11
                    //temp_matrix_float[2] = abs(h02[re]) + abs(h12[re]) + abs(h22[re]); /// h22

                    temp_matrix_float[0] = abs(channel_coefficients_[0][0][re]) + abs(channel_coefficients_[1][0][re]) + abs(channel_coefficients_[2][0][re]); /// h00
                    temp_matrix_float[1] = abs(channel_coefficients_[0][1][re]) + abs(channel_coefficients_[1][1][re]) + abs(channel_coefficients_[2][1][re]); /// h11
                    temp_matrix_float[2] = abs(channel_coefficients_[0][2][re]) + abs(channel_coefficients_[1][2][re]) + abs(channel_coefficients_[2][2][re]); /// h22

                    /**
                    temp_coefs[0] = conj(h00[re]) * h01[re] + conj(h10[re]) * h11[re] + conj(h20[re]) * h21[re]; /// h01
                    temp_coefs[1] = conj(h00[re]) * h02[re] + conj(h10[re]) * h12[re] + conj(h20[re]) * h22[re]; /// h02
                    temp_coefs[2] = conj(h01[re]) * h02[re] + conj(h11[re]) * h12[re] + conj(h21[re]) * h22[re]; /// h12
                    */

                    temp_coefs[0] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][1][re] +
                                    conj(channel_coefficients_[1][0][re]) * channel_coefficients_[1][1][re] +
                                    conj(channel_coefficients_[2][0][re]) * channel_coefficients_[2][1][re]; /// h01
                    temp_coefs[1] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][2][re] +
                                    conj(channel_coefficients_[1][0][re]) * channel_coefficients_[1][2][re] +
                                    conj(channel_coefficients_[2][0][re]) * channel_coefficients_[2][2][re]; /// h02
                    temp_coefs[2] = conj(channel_coefficients_[0][1][re]) * channel_coefficients_[0][2][re] +
                                    conj(channel_coefficients_[1][1][re]) * channel_coefficients_[1][2][re] +
                                    conj(channel_coefficients_[2][1][re]) * channel_coefficients_[2][2][re]; /// h12

#if TIME_MEASURE == 1
        t2 = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian matrix : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
#endif

                    /**
                    cout << "temp_matrix_float[0] : " << temp_matrix_float[0] << endl;
                    cout << "temp_matrix_float[1] : " << temp_matrix_float[1] << endl;
                    cout << "temp_matrix_float[2] : " << temp_matrix_float[2] << endl;
                    cout << "temp_coefs[0] : " << temp_coefs[0] << endl;
                    cout << "temp_coefs[1] : " << temp_coefs[1] << endl;
                    cout << "temp_coefs[2] : " << temp_coefs[2] << endl; */

#if TIME_MEASURE == 1
       t1 = std::chrono::steady_clock::now();
#endif

                    det = temp_matrix_float[0] * temp_matrix_float[1] * temp_matrix_float[2] +
                          2 * (temp_coefs[0] * temp_coefs[2] * conj(temp_coefs[1])).real() +
                          - pow(abs(temp_coefs[1]), 2) * temp_matrix_float[1] -
                          pow(abs(temp_coefs[0]), 2) * temp_matrix_float[2] -
                          pow(abs(temp_coefs[2]), 2) * temp_matrix_float[0];
#if TIME_MEASURE == 1
                    t2 = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(trace) << "Time to compute determinant : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
#endif
                    /**
                    temp_mrc[0] = conj(h00[re]) * pdsch_samples_[0][re] + conj(h10[re]) * pdsch_samples_[1][re] +
                                  conj(h20[re]) * pdsch_samples_[2][re];
                    temp_mrc[1] = conj(h01[re]) * pdsch_samples_[0][re] + conj(h11[re]) * pdsch_samples_[1][re] +
                                  conj(h21[re]) * pdsch_samples_[2][re];
                    temp_mrc[2] = conj(h02[re]) * pdsch_samples_[0][re] + conj(h12[re]) * pdsch_samples_[1][re] +
                                  conj(h22[re]) * pdsch_samples_[2][re]; */

#if TIME_MEASURE == 1
                    t1 = std::chrono::steady_clock::now();
#endif

                    temp_mrc[0] = conj(channel_coefficients_[0][0][re]) * pdsch_samples_[0][re] + conj(channel_coefficients_[1][0][re]) * pdsch_samples_[1][re] +
                                  conj(channel_coefficients_[2][0][re]) * pdsch_samples_[2][re];
                    temp_mrc[1] = conj(channel_coefficients_[0][1][re]) * pdsch_samples_[0][re] + conj(channel_coefficients_[1][1][re]) * pdsch_samples_[1][re] +
                                  conj(channel_coefficients_[2][1][re]) * pdsch_samples_[2][re];
                    temp_mrc[2] = conj(channel_coefficients_[0][2][re]) * pdsch_samples_[0][re] + conj(channel_coefficients_[1][2][re]) * pdsch_samples_[1][re] +
                                  conj(channel_coefficients_[2][2][re]) * pdsch_samples_[2][re];

#if TIME_MEASURE == 1
                    t2 = std::chrono::steady_clock::now();
                    BOOST_LOG_TRIVIAL(trace) << "Time to multiple the received signal by H^H: " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
                    t1 = std::chrono::steady_clock::now();
#endif

                    /// Diagonal elements of inverse matrix
                    temp_inv[0] = temp_matrix_float[1] * temp_matrix_float[2] -
                                  abs(temp_coefs[2]) * abs(temp_coefs[2]); /// 00
                    temp_inv[1] = temp_matrix_float[0] * temp_matrix_float[2] -
                                  abs(temp_coefs[1]) * abs(temp_coefs[1]); /// 11
                    temp_inv[2] = temp_matrix_float[0] * temp_matrix_float[1] -
                                  abs(temp_coefs[0]) * abs(temp_coefs[0]); /// 22

                    temp_inv[3] = -(temp_coefs[0] * temp_matrix_float[2] - temp_coefs[1] * conj(temp_coefs[2])); /// 01
                    temp_inv[4] = temp_coefs[0] * temp_coefs[2] - temp_coefs[1] * temp_matrix_float[1]; /// 02
                    temp_inv[5] = -(temp_matrix_float[0] * temp_coefs[2] - temp_coefs[1] * conj(temp_coefs[0])); /// 12

#if TIME_MEASURE == 1
                    t2 = std::chrono::steady_clock::now();
                    BOOST_LOG_TRIVIAL(trace) << "Time to compute the inverse matrix : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
                    t1 = std::chrono::steady_clock::now();
#endif

                    *(equalized_symbols_) = temp_inv[0].real() * temp_mrc[0] +
                                                 temp_inv[3] * temp_mrc[1] +
                                                 temp_inv[4] * temp_mrc[2];

                    *(equalized_symbols_ + 1) = conj(temp_inv[3]) * temp_mrc[0] +
                                                     temp_inv[1].real() * temp_mrc[1] +
                                                     temp_inv[5] * temp_mrc[2];

                    *(equalized_symbols_ + 2) = conj(temp_inv[4]) * temp_mrc[0] +
                                                     conj(temp_inv[5]) * temp_mrc[1] +
                                                     temp_inv[2].real() * temp_mrc[2];

                    *(equalized_symbols_) /= det;
                    *(equalized_symbols_ + 1) /= det;
                    *(equalized_symbols_ + 2) /= det;

                    equalized_symbols_ += 3;

#if TIME_MEASURE == 1
                    t2 = std::chrono::steady_clock::now();
                    BOOST_LOG_TRIVIAL(trace) << "Time equalize the symbols : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
#endif

                    //cout << "det : " << det << endl;
                    /**
                    cout << "equalized symbol 0 : " << equalized_symbols_[re * 3] << endl;
                    cout << "equalized symbol 1 : " << equalized_symbols_[re * 3 + 1] << endl;
                    cout << "equalized symbol 2 : " << equalized_symbols_[re * 3 + 2] << endl;

                    cout << "00 : " << (temp_inv[0] * temp_matrix_float[0] + temp_inv[3] * conj(temp_coefs[0]) + temp_inv[4] * conj(temp_coefs[1]))/det << endl;
                    cout << "01 : " << (temp_inv[0] * temp_coefs[0]        + temp_inv[3] * temp_matrix_float[1]   + temp_inv[4] * conj(temp_coefs[2]))/det << endl;
                    cout << "02 : " << (temp_inv[0] * temp_coefs[1]        + temp_inv[3] * temp_coefs[2]          + temp_inv[4] * temp_matrix_float[2])/det << endl;

                    cout << "10 : " << (conj(temp_inv[3]) * temp_matrix_float[0] + temp_inv[1] * conj(temp_coefs[0]) + temp_inv[5] * conj(temp_coefs[1]))/det << endl;
                    cout << "11 : " << (conj(temp_inv[3]) * temp_coefs[0]        + temp_inv[1] * temp_matrix_float[1]   + temp_inv[5] * conj(temp_coefs[2]))/det << endl;
                    cout << "12 : " << (conj(temp_inv[3]) * temp_coefs[1]        + temp_inv[1] * temp_coefs[2]          + temp_inv[5] * temp_matrix_float[2])/det << endl;

                    cout << "20 : " << (conj(temp_inv[4]) * temp_matrix_float[0] + conj(temp_inv[5]) * conj(temp_coefs[0]) + temp_inv[2] * conj(temp_coefs[1]))/det << endl;
                    cout << "21 : " << (conj(temp_inv[4]) * temp_coefs[0]        + conj(temp_inv[5]) * temp_matrix_float[1]   + temp_inv[2] * conj(temp_coefs[2]))/det << endl;
                    cout << "22 : " << (conj(temp_inv[4]) * temp_coefs[1]        + conj(temp_inv[5]) * temp_coefs[2]          + temp_inv[2] * temp_matrix_float[2])/det << endl; */
                }
            //}

            break;
        }

        default : /// 4 TX for now
        {
            //float _Complex hermitian_matrix[4][4];
            //float _Complex channel_matrix[4][4];
            //float _Complex temp_equalized_symbols[4];
            complex<float> hermitian_matrix[4][4];
            complex<float> channel_matrix[4][4];
            complex<float> temp_equalized_symbols[4];

#if TIME_MEASURE == 1
            std::chrono::steady_clock::time_point t1, t2;
#endif
            for(int re = 0; re < num_re_pdsch_; re++) {

#if TIME_MEASURE == 1
                BOOST_LOG_TRIVIAL(trace) << " ------------------------- " << endl;
                BOOST_LOG_TRIVIAL(trace) << "RE number : " << re << endl;
                t1 = std::chrono::steady_clock::now();
#endif

                /// Load channel coefs into transposed channel matrix
                /**
                for(int i = 0 ; i < nb_rx_ports_; i++) {
                    for(int j = 0; j < 4; j++) {
                        channel_matrix[j * nb_rx_ports_ + i] = channel_coefficients_[i][j][re].real() + I * channel_coefficients_[i][j][re].imag();
                    }
                } */

                /// Column major channel matrix
                //for(int i = 0 ; i < nb_rx_ports_; i++) {
                //    for(int j = 0; j < 4; j++) {
                //        channel_matrix[j * nb_rx_ports_ + i] = channel_coefficients_[i][j][re];
                //    }
                //}

                /// First line
                //temp = channel_coefficients_[0][0][re].real();
                //temp1 = channel_coefficients_[0][0][re].imag();
                hermitian_matrix[0][0].real(channel_coefficients_[0][0][re].real() * channel_coefficients_[0][0][re].real() +
                                            channel_coefficients_[0][0][re].imag() * channel_coefficients_[0][0][re].imag());
//temp * temp + temp1 * temp1; //conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][0][re];
                //hermitian_matrix[0] = cabsf(channel_matrix[0]) * cabsf(channel_matrix[0]);
                //hermitian_matrix[0].real(pow(abs(channel_matrix[0]), 2));

                /**
                for(int i = 1; i < nb_rx_ports_; i++) {
                    /// First line
                    temp = channel_coefficients_[i][0][re].real();
                    temp1 = channel_coefficients_[i][0][re].imag();
                    hermitian_matrix[0] += temp * temp + temp1 * temp1; //conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][0][re];
                } */

                hermitian_matrix[0][1] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][1][re]; // complex<float>(  channel_coefficients_[0][0][re].real() * channel_coefficients_[0][1][re].real() +
                                      //                 channel_coefficients_[0][0][re].imag() * channel_coefficients_[0][1][re].imag(),
                                      //               - channel_coefficients_[0][0][re].imag() * channel_coefficients_[0][1][re].real() +
                                      //                 channel_coefficients_[0][0][re].real() * channel_coefficients_[0][1][re].real());//
                //hermitian_matrix[1] = conjf(channel_matrix[0]) * channel_matrix[4];
                //hermitian_matrix[1] = conj(channel_matrix[0]) * channel_matrix[4];

                /**
                for(int i = 1; i < nb_rx_ports_; i++) {
                    /// First line
                    hermitian_matrix[1] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][1][re];
                } */
                hermitian_matrix[0][2] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][2][re]; //complex<float>(  channel_coefficients_[0][0][re].real() * channel_coefficients_[0][2][re].real() +
                                                        //channel_coefficients_[0][0][re].imag() * channel_coefficients_[0][2][re].imag(),
                                                        //- channel_coefficients_[0][0][re].imag() * channel_coefficients_[0][2][re].real() +
                                                        //channel_coefficients_[0][0][re].real() * channel_coefficients_[0][2][re].real());//
                //hermitian_matrix[2] = conjf(channel_matrix[0]) * channel_matrix[8];
                //hermitian_matrix[2] = conj(channel_matrix[0]) * channel_matrix[8];
                /**
                for(int i = 1; i < nb_rx_ports_; i++) {
                    hermitian_matrix[2] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][2][re];
                } */
                hermitian_matrix[0][3] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][3][re]; //complex<float>(  channel_coefficients_[0][0][re].real() * channel_coefficients_[0][3][re].real() +
                                      //                 channel_coefficients_[0][0][re].imag() * channel_coefficients_[0][3][re].imag(),
                                      //                 - channel_coefficients_[0][0][re].imag() * channel_coefficients_[0][3][re].real() +
                                      //                 channel_coefficients_[0][0][re].real() * channel_coefficients_[0][3][re].real()); //
                //hermitian_matrix[3] = conjf(channel_matrix[0]) * channel_matrix[12];
                //hermitian_matrix[3] = conj(channel_matrix[0]) * channel_matrix[12];

                /**
                for(int i = 1; i < nb_rx_ports_; i++) {
                    hermitian_matrix[3] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][3][re];
                } */

                /// Second line from diag coef 1,1
                //temp = channel_coefficients_[0][1][re].real();
                //temp1 = channel_coefficients_[0][1][re].imag();
                hermitian_matrix[1][1].real(channel_coefficients_[0][1][re].real() * channel_coefficients_[0][1][re].real() +
                                      channel_coefficients_[0][1][re].imag() * channel_coefficients_[0][1][re].imag());
                                      //temp * temp + temp1 * temp1; //conj(channel_coefficients_[0][1][re]) * channel_coefficients_[0][1][re];
                //hermitian_matrix[5] = cabsf(channel_matrix[4]) * cabsf(channel_matrix[4]);
                //hermitian_matrix[5].real(pow(abs(channel_matrix[4]), 2));

                /**
                for(int i = 1; i < nb_rx_ports_; i++) {
                    /// Second line from diag coef 1,1
                    temp = channel_coefficients_[i][1][re].real();
                    temp1 = channel_coefficients_[i][1][re].imag();
                    hermitian_matrix[5] += temp * temp + temp1 * temp1; //conj(channel_coefficients_[i][1][re]) * channel_coefficients_[i][1][re];
                } */
                hermitian_matrix[1][2] = conj(channel_coefficients_[0][1][re]) * channel_coefficients_[0][2][re]; // complex<float>(  channel_coefficients_[0][1][re].real() * channel_coefficients_[0][2][re].real() +
                                      //                 channel_coefficients_[0][1][re].imag() * channel_coefficients_[0][2][re].imag(),
                                      //                 - channel_coefficients_[0][1][re].imag() * channel_coefficients_[0][2][re].real() +
                                      //                 channel_coefficients_[0][1][re].real() * channel_coefficients_[0][2][re].real());//
                //hermitian_matrix[6] = conjf(channel_matrix[4]) * channel_matrix[8];
                //hermitian_matrix[6] = conj(channel_matrix[4]) * channel_matrix[8];

                /**
                for(int i = 1; i < nb_rx_ports_; i++) {
                    hermitian_matrix[6] += conj(channel_coefficients_[i][1][re]) * channel_coefficients_[i][2][re];
                } */

                hermitian_matrix[1][3] = conj(channel_coefficients_[0][1][re]) * channel_coefficients_[0][3][re]; //complex<float>(  channel_coefficients_[0][1][re].real() * channel_coefficients_[0][3][re].real() +
                                      //                 channel_coefficients_[0][1][re].imag() * channel_coefficients_[0][3][re].imag(),
                                      //                 - channel_coefficients_[0][1][re].imag() * channel_coefficients_[0][3][re].real() +
                                      //                 channel_coefficients_[0][1][re].real() * channel_coefficients_[0][3][re].real()); //
                //hermitian_matrix[7] = conjf(channel_matrix[4]) * channel_matrix[12];
                //hermitian_matrix[7] = conj(channel_matrix[4]) * channel_matrix[12];

                /**
                for(int i = 1; i < nb_rx_ports_; i++) {
                    hermitian_matrix[7] += conj(channel_coefficients_[i][1][re]) * channel_coefficients_[i][3][re];
                } */

                /// Third line from diag coef 2,2
                //temp = channel_coefficients_[0][2][re].real();
                //temp1 = channel_coefficients_[0][2][re].imag();
                hermitian_matrix[2][2].real(channel_coefficients_[0][2][re].real() * channel_coefficients_[0][2][re].real() +
                                            channel_coefficients_[0][2][re].imag() * channel_coefficients_[0][2][re].imag());
                                            //temp * temp + temp1 * temp1; //conj(channel_coefficients_[0][2][re]) * channel_coefficients_[0][2][re];
                //hermitian_matrix[10] = cabsf(channel_matrix[8]) * cabsf(channel_matrix[8]);
                //hermitian_matrix[10].real(pow(abs(channel_matrix[8]), 2));

                /**
                for(int i = 1; i < nb_rx_ports_; i++) {
                    temp = channel_coefficients_[i][2][re].real();
                    temp1 = channel_coefficients_[i][2][re].imag();
                    hermitian_matrix[10] += temp * temp + temp1 * temp1; //conj(channel_coefficients_[i][2][re]) * channel_coefficients_[i][2][re];
                } */
                hermitian_matrix[2][3] = conj(channel_coefficients_[0][2][re]) * channel_coefficients_[0][3][re]; //complex<float>(  channel_coefficients_[0][2][re].real() * channel_coefficients_[0][3][re].real() +
                                       //                 channel_coefficients_[0][2][re].imag() * channel_coefficients_[0][3][re].imag(),
                                       //                 - channel_coefficients_[0][2][re].imag() * channel_coefficients_[0][3][re].real() +
                                       //                 channel_coefficients_[0][2][re].real() * channel_coefficients_[0][3][re].real()); // conj(channel_coefficients_[0][2][re]) * channel_coefficients_[0][3][re];
                //hermitian_matrix[11] = conjf(channel_matrix[8]) * channel_matrix[12];
                //hermitian_matrix[11] = conj(channel_matrix[8]) * channel_matrix[12];

                /**
                for(int i = 1; i < nb_rx_ports_; i++) {
                    hermitian_matrix[11] += conj(channel_coefficients_[i][2][re]) * channel_coefficients_[i][3][re];
                } */

                /// Fourth line from diag coef 3,3
                //temp = channel_coefficients_[0][3][re].real();
                //temp1 = channel_coefficients_[0][3][re].imag();
                hermitian_matrix[3][3].real(channel_coefficients_[0][3][re].real() * channel_coefficients_[0][3][re].real() +
                                            channel_coefficients_[0][3][re].imag() * channel_coefficients_[0][3][re].imag()); //temp * temp + temp1 * temp1;//conj(channel_coefficients_[0][3][re]) * channel_coefficients_[0][3][re];
                //hermitian_matrix[15] = cabsf(channel_matrix[12]) * cabsf(channel_matrix[12]);
                //hermitian_matrix[15].real(pow(abs(channel_matrix[12]), 2));

                /**
                /// Compute hermitian matrix
                for(int i = 1; i < nb_rx_ports_; i++) {
                    temp = channel_coefficients_[i][3][re].real();
                    temp1 = channel_coefficients_[i][3][re].imag();
                    hermitian_matrix[15] += temp * temp + temp1 * temp1; //conj(channel_coefficients_[i][3][re]) * channel_coefficients_[i][3][re];
                } */

                /// Compute hermitian matrix
                for(int i = 1; i < nb_rx_ports_; i++) {
                    /// First line
                    //temp = channel_coefficients_[i][0][re].real();
                    //temp1 = channel_coefficients_[i][0][re].imag();
                    hermitian_matrix[0][0] += channel_coefficients_[i][0][re].real() * channel_coefficients_[i][0][re].real() +
                    channel_coefficients_[i][0][re].imag() * channel_coefficients_[i][0][re].imag(); // temp * temp + temp1 * temp1; //conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][0][re];
                    hermitian_matrix[1][0] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][1][re];//  complex<float>(  channel_coefficients_[i][0][re].real() * channel_coefficients_[i][1][re].real() +
                    //channel_coefficients_[i][0][re].imag() * channel_coefficients_[i][1][re].imag(),
                    //- channel_coefficients_[i][0][re].imag() * channel_coefficients_[i][1][re].real() +
                    //channel_coefficients_[i][0][re].real() * channel_coefficients_[i][1][re].real()); //
                    hermitian_matrix[2][0] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][2][re]; //complex<float>(  channel_coefficients_[i][0][re].real() * channel_coefficients_[i][2][re].real() +
                    //                channel_coefficients_[i][0][re].imag() * channel_coefficients_[i][2][re].imag(),
                    //                - channel_coefficients_[i][0][re].imag() * channel_coefficients_[i][2][re].real() +
                    //                channel_coefficients_[i][0][re].real() * channel_coefficients_[i][2][re].real()); //
                    hermitian_matrix[3][0] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][3][re]; //complex<float>(  channel_coefficients_[i][0][re].real() * channel_coefficients_[i][3][re].real() +
                    //channel_coefficients_[i][0][re].imag() * channel_coefficients_[i][3][re].imag(),
                    //- channel_coefficients_[i][0][re].imag() * channel_coefficients_[i][3][re].real() +
                    //channel_coefficients_[i][0][re].real() * channel_coefficients_[i][3][re].real()); //

                    /// Second line from diag coef 1,1
                    //temp = channel_coefficients_[i][1][re].real();
                    //temp1 = channel_coefficients_[i][1][re].imag();
                    hermitian_matrix[1][1] += channel_coefficients_[i][1][re].real() * channel_coefficients_[i][1][re].real() +
                    channel_coefficients_[i][1][re].imag() * channel_coefficients_[i][1][re].imag(); // temp * temp + temp1 * temp1; //conj(channel_coefficients_[i][1][re]) * channel_coefficients_[i][1][re];
                    hermitian_matrix[1][2] += conj(channel_coefficients_[i][1][re]) * channel_coefficients_[i][2][re]; //complex<float>(  channel_coefficients_[i][1][re].real() * channel_coefficients_[i][2][re].real() +
                    //                 channel_coefficients_[i][1][re].imag() * channel_coefficients_[i][2][re].imag(),
                    //                 - channel_coefficients_[i][1][re].imag() * channel_coefficients_[i][2][re].real() +
                    //                 channel_coefficients_[i][1][re].real() * channel_coefficients_[i][2][re].real()); //conj(channel_coefficients_[i][1][re]) * channel_coefficients_[i][2][re];
                    hermitian_matrix[1][3] += conj(channel_coefficients_[i][1][re]) * channel_coefficients_[i][3][re]; // complex<float>(  channel_coefficients_[i][1][re].real() * channel_coefficients_[i][3][re].real() +
                    //                  channel_coefficients_[i][1][re].imag() * channel_coefficients_[i][3][re].imag(),
                    //                  - channel_coefficients_[i][1][re].imag() * channel_coefficients_[i][3][re].real() +
                    //                  channel_coefficients_[i][1][re].real() * channel_coefficients_[i][3][re].real()); // conj(channel_coefficients_[i][1][re]) * channel_coefficients_[i][3][re];
                    //hermitian_matrix[5] += pow(abs(channel_matrix[4 + i]), 2);
                    //hermitian_matrix[6] += conj(channel_matrix[4 + i]) * channel_matrix[8 + i];
                    //hermitian_matrix[7] += conj(channel_matrix[4 + i]) * channel_matrix[12 + i];

                    /**
                    hermitian_matrix[5] += cabsf(channel_matrix[4 + i]) * cabsf(channel_matrix[4 + i]); // temp * temp + temp1 * temp1; //conj(channel_coefficients_[i][1][re]) * channel_coefficients_[i][1][re];
                    hermitian_matrix[6] += conjf(channel_matrix[4 + i]) * channel_matrix[8 + i];
                    hermitian_matrix[7] += conjf(channel_matrix[4 + i]) * channel_matrix[12 + i]; */

                    /// Third line from diag coef 2,2
                    //temp = channel_coefficients_[i][2][re].real();
                    //temp1 = channel_coefficients_[i][2][re].imag();
                    hermitian_matrix[2][2] += channel_coefficients_[i][2][re].real() * channel_coefficients_[i][2][re].real() +
                    channel_coefficients_[i][2][re].imag() * channel_coefficients_[i][2][re].imag(); // temp * temp + temp1 * temp1; //conj(channel_coefficients_[i][2][re]) * channel_coefficients_[i][2][re];
                    hermitian_matrix[2][3] += conj(channel_coefficients_[i][2][re]) * channel_coefficients_[i][3][re]; //complex<float>(  channel_coefficients_[i][2][re].real() * channel_coefficients_[i][3][re].real() +
                    //channel_coefficients_[i][2][re].imag() * channel_coefficients_[i][3][re].imag(),
                    //- channel_coefficients_[i][2][re].imag() * channel_coefficients_[i][3][re].real() +
                    //channel_coefficients_[i][2][re].real() * channel_coefficients_[i][3][re].real()); // conj(channel_coefficients_[i][2][re]) * channel_coefficients_[i][3][re];
                    //hermitian_matrix[10] += pow(abs(channel_matrix[8 + i]), 2);
                    //hermitian_matrix[11] += conj(channel_matrix[8 + i]) * channel_matrix[12 + i];

                    /**
                    hermitian_matrix[10] += cabsf(channel_matrix[8 + i]) * cabsf(channel_matrix[8 + i]);
                    hermitian_matrix[11] += conjf(channel_matrix[8 + i]) * channel_matrix[12 + i]; */

                    /// Fourth line from diag coef 3,3
                    //temp = channel_coefficients_[i][3][re].real();
                    //temp1 = channel_coefficients_[i][3][re].imag();
                    hermitian_matrix[3][3] += channel_coefficients_[i][3][re].real() * channel_coefficients_[i][3][re].real() +
                    channel_coefficients_[i][3][re].imag() * channel_coefficients_[i][3][re].imag(); // temp * temp + temp1 * temp1; //conj(channel_coefficients_[i][3][re]) * channel_coefficients_[i][3][re];
                    //hermitian_matrix[15] += cabsf(channel_matrix[12 + i]) * cabsf(channel_matrix[12 + i]);
                    //hermitian_matrix[15] += pow(abs(channel_matrix[12 + i]), 2);
                }

#if TIME_MEASURE == 1
                t2 = std::chrono::steady_clock::now();
                BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian : " <<
                     std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

                //t1 = std::chrono::steady_clock::now();
#endif
                /// Load conjugate transpose, row major
                //for(int i = 0; i < nb_rx_ports_; i++) {
                //    for(int j = 0; j < 4; j++) {
                //        conjugate_transpose[j * nb_rx_ports_ + i] = conj(channel_coefficients_[i][j][re]);
                //    }
                //}

#if TIME_MEASURE == 1
                //t2 = std::chrono::steady_clock::now();

                //BOOST_LOG_TRIVIAL(trace) << "Time to Load conjugate transpose : " <<
                //     std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

                t1 = std::chrono::steady_clock::now();
#endif
                /// Perform LDL decomposition
                /**
                ldl_decomp_harcoded(hermitian_matrix, // row major
                                    r_matrix,
                                    diag_matrix,
                                    4); */

                ldl_decomp_test(hermitian_matrix, // row major
                                4);

#if TIME_MEASURE == 1
                t2 = std::chrono::steady_clock::now();

                BOOST_LOG_TRIVIAL(trace) << "Time to compute LDL : " <<
                     std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

                t1 = std::chrono::steady_clock::now();
#endif
                /// Compute the inverse in-place
                /**
                for(int i = 1; i < 9; i ++) {
                    if ((i == 4) or (i == 7)) {
                        continue;
                    }
                    r_matrix[i] = -r_matrix[i];
                } */

                /**
                for(int i = 0; i < 4; i++) {
                    for(int j = i + 1; j < 4; j++) {
                        hermitian_matrix[i * 4 + j] = -hermitian_matrix[i * 4 + j];
                    }
                } */

                hermitian_matrix[0][1]  = -hermitian_matrix[0][1];
                hermitian_matrix[0][2]  = -hermitian_matrix[0][2];
                hermitian_matrix[0][3]  = -hermitian_matrix[0][3];
                hermitian_matrix[1][2]  = -hermitian_matrix[1][2];
                hermitian_matrix[1][3]  = -hermitian_matrix[1][3];
                hermitian_matrix[2][3] = -hermitian_matrix[2][3];

                /// Compute coefficients in the correct order
                hermitian_matrix[0][2] += hermitian_matrix[0][1] * hermitian_matrix[1][2];
                hermitian_matrix[0][3] += hermitian_matrix[0][1] * hermitian_matrix[1][3] + hermitian_matrix[0][2] * hermitian_matrix[2][3];
                hermitian_matrix[1][3] += hermitian_matrix[1][2] * hermitian_matrix[2][3];

                /// Copy inverse R^(-1) in lower part of the array
                hermitian_matrix[1][0] = hermitian_matrix[0][1];
                hermitian_matrix[2][0] = hermitian_matrix[0][2];
                hermitian_matrix[2][1] = hermitian_matrix[1][2];
                hermitian_matrix[3][0] = hermitian_matrix[0][3];
                hermitian_matrix[3][1] = hermitian_matrix[1][3];
                hermitian_matrix[3][2] = hermitian_matrix[2][3];

                /// Multiply by D^-1
                hermitian_matrix[0][1] = conj(hermitian_matrix[0][1]);
                hermitian_matrix[0][2] = conj(hermitian_matrix[0][2]);
                hermitian_matrix[0][3] = conj(hermitian_matrix[0][3]);
                hermitian_matrix[1][2] = conj(hermitian_matrix[1][2]);
                hermitian_matrix[1][3] = conj(hermitian_matrix[1][3]);
                hermitian_matrix[2][3] = conj(hermitian_matrix[2][3]);

#if TIME_MEASURE == 1
                t2 = std::chrono::steady_clock::now();

                BOOST_LOG_TRIVIAL(trace) << "Time to compute the two inverse matrices : " <<
                     std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

                t1 = std::chrono::steady_clock::now();
#endif
                /// Multiply received signal y by H^H, then multiply by the inverse
                //std::fill(temp_symbols.begin(), temp_symbols.end(), 0);
                //for(int i = 0; i < 4; i++) {
                //    temp_symbols[i] = conjugate_transpose[i * nb_rx_ports_] * pdsch_samples_[0][re];
                //}
                for(int i = 0; i < 4; i++) {
                    *(equalized_symbols_ + i) = conj(channel_coefficients_[0][i][re]) * pdsch_samples_[0][re];
                    //*(equalized_symbols_ + i) = conj(channel_matrix[i * 4]) * pdsch_samples_[0][re];
                    //temp_equalized_symbols[i] = conj(channel_matrix[i * 4]) * pdsch_samples_[0][re];
                }
                for(int j = 1; j < nb_rx_ports_; j++) {
                    for(int i = 0; i < 4; i++) {
                        //temp_symbols[i] += conjugate_transpose[i * nb_rx_ports_ + j] * pdsch_samples_[j][re];
                        //*(equalized_symbols_ + i) += conjugate_transpose[i * nb_rx_ports_ + j] * pdsch_samples_[j][re];
                        *(equalized_symbols_ + i) += conj(channel_coefficients_[j][i][re]) * pdsch_samples_[j][re];
                        //*(equalized_symbols_ + i) += conj(channel_matrix[i * nb_rx_ports_ + j]) * pdsch_samples_[j][re];
                        //temp_equalized_symbols[i] += conj(channel_matrix[i * nb_rx_ports_ + j]) * pdsch_samples_[j][re];
                    }
                }

                /**
                for(int i = 0; i < 16; i += 4) {
                    *(temp_equalized_symbols + i) = conj(channel_matrix[i]) * (pdsch_samples_[0][re].real() + I * pdsch_samples_[0][re].imag());
                }
                for(int i = 0; i < 4; i++) {
                    for(int j = 1; j < nb_rx_ports_; j++) {
                        //temp_symbols[i] += conjugate_transpose[i * nb_rx_ports_ + j] * pdsch_samples_[j][re];
                        //*(equalized_symbols_ + i) += conjugate_transpose[i * nb_rx_ports_ + j] * pdsch_samples_[j][re];
                        *(temp_equalized_symbols + i) += conj(channel_matrix[i * 4 + j]) * (pdsch_samples_[j][re].real() + I * pdsch_samples_[j][re].imag());
                    }
                } */

                /// Multiply by D^(-1) R^(*)^(-1)
                /**
                temp_symbols_2[3] = r_matrix[3] * temp_symbols[0] + r_matrix[6] * temp_symbols[1] + r_matrix[8] * temp_symbols[2] + r_matrix[9] * temp_symbols[3];
                temp_symbols_2[2] = r_matrix[2] * temp_symbols[0] + r_matrix[5] * temp_symbols[1] + r_matrix[7] * temp_symbols[2];
                temp_symbols_2[1] = r_matrix[1] * temp_symbols[0] + r_matrix[4] * temp_symbols[1];
                temp_symbols_2[0] = r_matrix[0] * temp_symbols[0]; */

                /**
                *(equalized_symbols_ + 3) = r_matrix[3] * *(equalized_symbols_) + r_matrix[6] * *(equalized_symbols_ + 1) + r_matrix[8] * *(equalized_symbols_ + 2) + r_matrix[9] * *(equalized_symbols_ + 3);
                *(equalized_symbols_ + 2) = r_matrix[2] * *(equalized_symbols_) + r_matrix[5] * *(equalized_symbols_ + 1) + r_matrix[7] * *(equalized_symbols_ + 2);
                *(equalized_symbols_ + 1) = r_matrix[1] * *(equalized_symbols_) + r_matrix[4] * *(equalized_symbols_ + 1);
                *(equalized_symbols_)     = r_matrix[0] * *(equalized_symbols_);

                *(equalized_symbols_)     /= diag_matrix[0];
                *(equalized_symbols_ + 1) /= diag_matrix[1];
                *(equalized_symbols_ + 2) /= diag_matrix[2];
                *(equalized_symbols_ + 3) /= diag_matrix[3]; */


                *(equalized_symbols_ + 3) += hermitian_matrix[0][3] * *(equalized_symbols_) + hermitian_matrix[1][3] * *(equalized_symbols_ + 1) + hermitian_matrix[2][3] * *(equalized_symbols_ + 2);
                *(equalized_symbols_ + 2) += hermitian_matrix[0][2] * *(equalized_symbols_) + hermitian_matrix[1][2] * *(equalized_symbols_ + 1);
                *(equalized_symbols_ + 1) += hermitian_matrix[0][1] * *(equalized_symbols_);
                *(equalized_symbols_)     /= hermitian_matrix[0][0].real();
                *(equalized_symbols_ + 1) /= hermitian_matrix[1][1].real();
                *(equalized_symbols_ + 2) /= hermitian_matrix[2][2].real();
                *(equalized_symbols_ + 3) /= hermitian_matrix[3][3].real();

                /**
                temp_equalized_symbols[3] += hermitian_matrix[3] * temp_equalized_symbols[0] +
                                             hermitian_matrix[7] * temp_equalized_symbols[1] +
                                             hermitian_matrix[11] * temp_equalized_symbols[2];

                temp_equalized_symbols[2] += hermitian_matrix[2] * *(equalized_symbols_) +
                                             hermitian_matrix[6] * *(equalized_symbols_ + 1);

                temp_equalized_symbols[1] += hermitian_matrix[1] * *(equalized_symbols_);

                temp_equalized_symbols[0] /= hermitian_matrix[0].real();
                temp_equalized_symbols[1] /= hermitian_matrix[5].real();
                temp_equalized_symbols[2] /= hermitian_matrix[10].real();
                temp_equalized_symbols[3] /= hermitian_matrix[15].real(); */

                /// Multiply by R^(-1)
                /**
                *(equalized_symbols_) = r_inverse[0] * temp_symbols_2[0] + r_inverse[1] * temp_symbols_2[1] + r_inverse[2] * temp_symbols_2[2] + r_inverse[3] * temp_symbols_2[3];
                *(equalized_symbols_ + 1) = r_inverse[4] * temp_symbols_2[1] + r_inverse[5] * temp_symbols_2[2] + r_inverse[6] * temp_symbols_2[3];
                *(equalized_symbols_ + 2) = r_inverse[7] * temp_symbols_2[2] + r_inverse[8] * temp_symbols_2[3];
                *(equalized_symbols_ + 3) = r_inverse[9] * temp_symbols_2[3]; */

                /**
                //*(equalized_symbols_)     = r_inverse[0] * *(equalized_symbols_)     + r_inverse[1] * *(equalized_symbols_ + 1) + r_inverse[2] * *(equalized_symbols_ + 2) + r_inverse[3] * *(equalized_symbols_ + 3);
                 *(equalized_symbols_)     = *(equalized_symbols_) + r_inverse[1] * *(equalized_symbols_ + 1) + r_inverse[2] * *(equalized_symbols_ + 2) + r_inverse[3] * *(equalized_symbols_ + 3); // r_inverse[0] = 1
                //*(equalized_symbols_ + 1) = r_inverse[4] * *(equalized_symbols_ + 1) + r_inverse[5] * *(equalized_symbols_ + 2) + r_inverse[6] * *(equalized_symbols_ + 3);
                *(equalized_symbols_ + 1) = *(equalized_symbols_ + 1) + r_inverse[5] * *(equalized_symbols_ + 2) + r_inverse[6] * *(equalized_symbols_ + 3); // r_inverse[4] = 1
                 //*(equalized_symbols_ + 2) = r_inverse[7] * *(equalized_symbols_ + 2) + r_inverse[8] * *(equalized_symbols_ + 3);
                 *(equalized_symbols_ + 2)  += + r_inverse[8] * *(equalized_symbols_ + 3); // r_inverse[7] = 1;
                //*(equalized_symbols_ + 3) = r_inverse[9] * *(equalized_symbols_ + 3); // No need to compute because r_inverse[9] = 1 */


                *(equalized_symbols_)     += hermitian_matrix[1][0]  * *(equalized_symbols_ + 1) + hermitian_matrix[2][0] * *(equalized_symbols_ + 2) + hermitian_matrix[3][0] * *(equalized_symbols_ + 3);
                *(equalized_symbols_ + 1) += hermitian_matrix[2][1]  * *(equalized_symbols_ + 2) + hermitian_matrix[3][1] * *(equalized_symbols_ + 3);
                *(equalized_symbols_ + 2) += hermitian_matrix[3][2] * *(equalized_symbols_ + 3);

                /**
                temp_equalized_symbols[0] += hermitian_matrix[4]  * temp_equalized_symbols[1] + hermitian_matrix[8] * temp_equalized_symbols[2] + hermitian_matrix[12] * temp_equalized_symbols[3];
                temp_equalized_symbols[1] += hermitian_matrix[9]  * temp_equalized_symbols[2] + hermitian_matrix[13] * temp_equalized_symbols[3];
                temp_equalized_symbols[2] += hermitian_matrix[14] * temp_equalized_symbols[3]; */

                /**
                (equalized_symbols_)->real(crealf(*(temp_equalized_symbols)));
                (equalized_symbols_)->imag(cimagf(*(temp_equalized_symbols)));
                (equalized_symbols_ + 1)->real(crealf(*(temp_equalized_symbols + 1)));
                (equalized_symbols_ + 1)->imag(cimagf(*(temp_equalized_symbols + 1)));
                (equalized_symbols_ + 2)->real(crealf(*(temp_equalized_symbols + 2)));
                (equalized_symbols_ + 2)->imag(cimagf(*(temp_equalized_symbols + 2)));
                (equalized_symbols_ + 3)->real(crealf(*(temp_equalized_symbols + 3)));
                (equalized_symbols_ + 3)->imag(cimagf(*(temp_equalized_symbols + 3))); */

                //*(equalized_symbols_) = temp_equalized_symbols[0];
                //*(equalized_symbols_ + 1) = temp_equalized_symbols[1];
                //*(equalized_symbols_ + 2) = temp_equalized_symbols[2];
                //*(equalized_symbols_ + 3) = temp_equalized_symbols[3];

                equalized_symbols_ += 4;

                //std::fill(hermitian_matrix.begin(), hermitian_matrix.end(), 0);

#if TIME_MEASURE == 1
                t2 = std::chrono::steady_clock::now();
                BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols : " <<
                     std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
                BOOST_LOG_TRIVIAL(trace) << " ------------------------- " << endl;
#endif
            }
            break;
        }
    }
}

#if defined(__AVX2__)
void vblast_zf_2_layers_avx(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                        std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                        int num_re_pdsch_,
                        std::complex<float> *equalized_symbols_,
                        int nb_rx_ports_) {

    /// Compute the inverse of the channel matrix directly
    /*
    if(nb_rx_ports_ == 2) {
        __m256 det_vec;
        __m256 channel_matrix_vec[2][2];
        __m256 pdsch_samples_vec[2];
        __m256 temp_equalized_symbol;
        __m256 det_vec_norm;
        __m256 conj_vec = {1, -1, 1, -1, 1, -1, 1, -1};

        __m256 vec1, vec2;

        int i, j;

        for (int re = 0; re < num_re_pdsch_; re += 4) {

            for(i = 0; i < 2; i++) {
                for(j = 0; j < 2; j++) {
                    channel_matrix_vec[i][j] = _mm256_loadu_ps((float *) &channel_coefficients_[i][j][re]);
                }
            }

            /// Compute determinant
            vec1 = _mm256_mul_ps(channel_matrix_vec[0][0], channel_matrix_vec[1][1]);
            vec1 = _mm256_mul_ps(vec1, conj_vec);
            vec2 = _mm256_mul_ps(channel_matrix_vec[0][0], _mm256_permute_ps(channel_matrix_vec[1][1], 0b10110001));

            det_vec = _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000);

            vec1 = _mm256_mul_ps(channel_matrix_vec[0][1], channel_matrix_vec[1][0]);
            vec1 = _mm256_mul_ps(vec1, conj_vec);
            vec2 = _mm256_mul_ps(channel_matrix_vec[0][1], _mm256_permute_ps(channel_matrix_vec[1][0], 0b10110001));

            vec1 = _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000);
            det_vec = _mm256_sub_ps(det_vec, vec1);

            /// Squared norm of determinant
            det_vec_norm = _mm256_mul_ps(det_vec, det_vec);
            det_vec_norm = _mm256_permute_ps(_mm256_hadd_ps(det_vec_norm, det_vec_norm), 0b11011000);

            /// Multiply received samples by inverse
            pdsch_samples_vec[0] = _mm256_loadu_ps((float *) &pdsch_samples_[0][re]);
            pdsch_samples_vec[1] = _mm256_loadu_ps((float *) &pdsch_samples_[1][re]);

            vec1 = _mm256_mul_ps(channel_matrix_vec[1][1], pdsch_samples_vec[0]);
            vec1 = _mm256_mul_ps(vec1, conj_vec);
            vec2 = _mm256_mul_ps(channel_matrix_vec[1][1], _mm256_permute_ps(pdsch_samples_vec[0], 0b10110001));

            temp_equalized_symbol = _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000);

            vec1 = _mm256_mul_ps(channel_matrix_vec[0][1], pdsch_samples_vec[1]);
            vec1 = _mm256_mul_ps(vec1, conj_vec);
            vec2 = _mm256_mul_ps(channel_matrix_vec[0][1], _mm256_permute_ps(pdsch_samples_vec[1], 0b10110001));

            vec1 = _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000);
            temp_equalized_symbol = _mm256_sub_ps(temp_equalized_symbol, vec1);

            vec1 = _mm256_mul_ps(temp_equalized_symbol, det_vec);
            vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(det_vec, conj_vec), 0b10110001), temp_equalized_symbol);
            temp_equalized_symbol = _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000);

            temp_equalized_symbol = _mm256_div_ps(temp_equalized_symbol, det_vec_norm);

            for(i = 0; i < 8; i += 2) {
                (equalized_symbols_ + i)->real(temp_equalized_symbol[i]);
                (equalized_symbols_ + i)->imag(temp_equalized_symbol[i + 1]);
            }

            vec1 = _mm256_mul_ps(channel_matrix_vec[0][0], pdsch_samples_vec[1]);
            vec1 = _mm256_mul_ps(vec1, conj_vec);
            vec2 = _mm256_mul_ps(channel_matrix_vec[0][0], _mm256_permute_ps(pdsch_samples_vec[1], 0b10110001));

            temp_equalized_symbol = _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000);

            vec1 = _mm256_mul_ps(channel_matrix_vec[1][0], pdsch_samples_vec[0]);
            vec1 = _mm256_mul_ps(vec1, conj_vec);
            vec2 = _mm256_mul_ps(channel_matrix_vec[1][0], _mm256_permute_ps(pdsch_samples_vec[0], 0b10110001));

            vec1 = _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000);
            temp_equalized_symbol = _mm256_sub_ps(temp_equalized_symbol, vec1);

            vec1 = _mm256_mul_ps(temp_equalized_symbol, det_vec);
            vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(det_vec, conj_vec), 0b10110001), temp_equalized_symbol);
            temp_equalized_symbol = _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000);

            temp_equalized_symbol = _mm256_div_ps(temp_equalized_symbol, det_vec_norm);

            for(i = 0; i < 8; i += 2) {
                (equalized_symbols_ + i + 1)->real(temp_equalized_symbol[i]);
                (equalized_symbols_ + i + 1)->imag(temp_equalized_symbol[i + 1]);
            }

            equalized_symbols_ += 8;
        }

    } else { */
        __m256 hermitian_matrix[2][2];
        int i;
        __m256 det;
        __m256 temp_equalized_symbols[2];
        __m256 equalized_symbol;
        __m256 vec1, vec2, vec3, vec4;
        __m256 dot_prods[3][2];
        __m256 conj_vec = {1, -1, 1, -1, 1, -1, 1, -1};
        __m256 null_vec = _mm256_set1_ps(0);

        complex<float> hermitian_matrix_debug[2][2];
        complex<float> temp_equalized_symbols_debug[2];
        float det_debug;
        complex<float> equalized_symbols_debug[2];

        for (int re = 0; re < num_re_pdsch_; re+= 4) {

            dot_prods[0][0] = _mm256_set1_ps(0); //00
            dot_prods[1][0] = _mm256_set1_ps(0); //01
            dot_prods[1][1] = _mm256_set1_ps(0);
            dot_prods[2][0] = _mm256_set1_ps(0); //11

            for (i = 0; i < nb_rx_ports_; i++) {
                /// 0,0 diag coef
                vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[i][0][re]);
                dot_prods[0][0] = _mm256_add_ps(dot_prods[0][0], _mm256_mul_ps(vec1, vec1));

                /// 0, 1 coef
                vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[i][0][re]);
                vec2 = _mm256_loadu_ps((float *) &channel_coefficients_[i][1][re]);
                vec3 = _mm256_mul_ps(vec1, vec2);
                vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj_vec), 0b10110001), vec2);
                dot_prods[1][0] = _mm256_add_ps(dot_prods[1][0], vec3);
                dot_prods[1][1] = _mm256_add_ps(dot_prods[1][1], vec4);

                /// 1,1 diag coef
                vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[i][1][re]);
                dot_prods[2][0] = _mm256_add_ps(dot_prods[2][0], _mm256_mul_ps(vec1, vec1));
            }

            hermitian_matrix[0][0] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[0][0], null_vec), 0b11011000);
            hermitian_matrix[0][1] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[1][0], dot_prods[1][1]), 0b11011000);
            hermitian_matrix[1][1] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[2][0], null_vec), 0b11011000);

            vec1 = _mm256_mul_ps(hermitian_matrix[0][0], hermitian_matrix[1][1]);
            vec1 = _mm256_hadd_ps(vec1, vec1);
            vec1 = _mm256_permute_ps(vec1, 0b11011000);
            vec2 = _mm256_mul_ps(hermitian_matrix[0][1], hermitian_matrix[0][1]);
            vec2 = _mm256_hadd_ps(vec2, vec2);
            vec2 = _mm256_permute_ps(vec2, 0b11011000);
            det = _mm256_sub_ps(vec1,
                                vec2);
            temp_equalized_symbols[1] = _mm256_permute_ps(_mm256_hadd_ps(vec3, vec4), 0b11011000);

            dot_prods[0][0] = _mm256_set1_ps(0); //00
            dot_prods[0][1] = _mm256_set1_ps(0); //00
            dot_prods[1][0] = _mm256_set1_ps(0); //01
            dot_prods[1][1] = _mm256_set1_ps(0); //01
            for(i = 0; i < nb_rx_ports_; i++) {
                vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[i][0][re]);
                vec2 = _mm256_loadu_ps((float *) &pdsch_samples_[i][re]);
                vec3 = _mm256_mul_ps(vec1, vec2);
                vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj_vec), 0b10110001), vec2);
                dot_prods[0][0] = _mm256_add_ps(dot_prods[0][0], vec3);
                dot_prods[0][1] = _mm256_add_ps(dot_prods[0][1], vec4);

                vec1 = _mm256_loadu_ps((float *) &channel_coefficients_[i][1][re]);
                vec2 = _mm256_loadu_ps((float *) &pdsch_samples_[i][re]);
                vec3 = _mm256_mul_ps(vec1, vec2);
                vec4 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(vec1, conj_vec), 0b10110001), vec2);
                dot_prods[1][0] = _mm256_add_ps(dot_prods[1][0], vec3);
                dot_prods[1][1] = _mm256_add_ps(dot_prods[1][1], vec4);
            }
            temp_equalized_symbols[0] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[0][0], dot_prods[0][1]), 0b11011000);
            temp_equalized_symbols[1] = _mm256_permute_ps(_mm256_hadd_ps(dot_prods[1][0], dot_prods[1][1]), 0b11011000);

            hermitian_matrix[0][0] = _mm256_permute_ps(_mm256_hadd_ps(hermitian_matrix[0][0], hermitian_matrix[0][0]), 0b11011000);
            hermitian_matrix[1][1] = _mm256_permute_ps(_mm256_hadd_ps(hermitian_matrix[1][1], hermitian_matrix[1][1]), 0b11011000);

            equalized_symbol = _mm256_mul_ps(hermitian_matrix[1][1], temp_equalized_symbols[0]);
            vec1 = _mm256_mul_ps(hermitian_matrix[0][1], temp_equalized_symbols[1]);
            vec1 = _mm256_mul_ps(vec1, conj_vec);
            vec2 = _mm256_mul_ps(hermitian_matrix[0][1], _mm256_permute_ps(temp_equalized_symbols[1], 0b10110001));
            vec1 = _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000);
            equalized_symbol = _mm256_sub_ps(equalized_symbol, vec1);
            equalized_symbol = _mm256_div_ps(equalized_symbol, det);

            for(i = 0; i < 8; i += 2) {
                (equalized_symbols_ + i)->real(equalized_symbol[i]);
                (equalized_symbols_ + i)->imag(equalized_symbol[i + 1]);
            }

            hermitian_matrix[1][0] = _mm256_mul_ps(hermitian_matrix[0][1], conj_vec);
            equalized_symbol = _mm256_mul_ps(hermitian_matrix[0][0], temp_equalized_symbols[1]);
            vec1 = _mm256_mul_ps(hermitian_matrix[1][0], temp_equalized_symbols[0]);
            vec1 = _mm256_mul_ps(vec1, conj_vec);
            vec2 = _mm256_mul_ps(hermitian_matrix[1][0], _mm256_permute_ps(temp_equalized_symbols[0], 0b10110001));
            vec1 = _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000);
            equalized_symbol = _mm256_sub_ps(equalized_symbol, vec1);
            equalized_symbol = _mm256_div_ps(equalized_symbol, det);

            for(i = 0; i < 8; i += 2) {
                (equalized_symbols_ + i + 1)->real(equalized_symbol[i]);
                (equalized_symbols_ + i + 1)->imag(equalized_symbol[i + 1]);
            }

            equalized_symbols_ += 8;
        }
    //}
}
#endif

void vblast_zf_2_layers(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                        std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                        int num_re_pdsch_,
                        std::complex<float> *equalized_symbols_,
                        int nb_rx_ports_) {

    /// Compute the inverse of the channel matrix directly
    if(nb_rx_ports_ == 2) {
        complex<float> det;
        for (int re = 0; re < num_re_pdsch_; re++) {
            det = channel_coefficients_[0][0][re] * channel_coefficients_[1][1][re] - channel_coefficients_[1][0][re] * channel_coefficients_[0][1][re];
            *(equalized_symbols_) = channel_coefficients_[1][1][re] * pdsch_samples_[0][re] - channel_coefficients_[0][1][re] *  pdsch_samples_[1][re];
            *(equalized_symbols_ + 1) = -channel_coefficients_[1][0][re] * pdsch_samples_[0][re] + channel_coefficients_[0][0][re] * pdsch_samples_[1][re];
            *(equalized_symbols_) /= det;
            *(equalized_symbols_ + 1) /= det;
            equalized_symbols_ += 2;
        }
    } else {
        complex<float> hermitian_matrix[2][2];
        int i;
        float det;
        complex<float> temp_equalized_symbols[2];

        for (int re = 0; re < num_re_pdsch_; re++) {

            hermitian_matrix[0][0] = channel_coefficients_[0][0][re].real() * channel_coefficients_[0][0][re].real() +
                                     channel_coefficients_[0][0][re].imag() * channel_coefficients_[0][0][re].imag();
            hermitian_matrix[0][1] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][1][re];
            hermitian_matrix[1][1] = channel_coefficients_[0][1][re].real() * channel_coefficients_[0][1][re].real() +
                                     channel_coefficients_[0][1][re].imag() * channel_coefficients_[0][1][re].imag();

            for(i = 1; i < nb_rx_ports_; i++) {
                hermitian_matrix[0][0] += channel_coefficients_[i][0][re].real() * channel_coefficients_[i][0][re].real() +
                         channel_coefficients_[i][0][re].imag() * channel_coefficients_[i][0][re].imag();
                hermitian_matrix[0][1] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][1][re];
                hermitian_matrix[1][1] += channel_coefficients_[i][1][re].real() * channel_coefficients_[i][1][re].real() +
                                         channel_coefficients_[i][1][re].imag() * channel_coefficients_[i][1][re].imag();
            }
            hermitian_matrix[1][0] = conj(hermitian_matrix[0][1]);

            det = hermitian_matrix[0][0].real() * hermitian_matrix[1][1].real() - (hermitian_matrix[0][1].real() * hermitian_matrix[0][1].real() +
                                                                                   hermitian_matrix[0][1].imag() * hermitian_matrix[0][1].imag());

            temp_equalized_symbols[0] = conj(channel_coefficients_[0][0][re]) * pdsch_samples_[0][re];
            temp_equalized_symbols[1] = conj(channel_coefficients_[0][1][re]) * pdsch_samples_[0][re];
            for(i = 1; i < nb_rx_ports_; i++) {
                temp_equalized_symbols[0] += conj(channel_coefficients_[i][0][re]) * pdsch_samples_[i][re];
                temp_equalized_symbols[1] += conj(channel_coefficients_[i][1][re]) * pdsch_samples_[i][re];
            }

            *(equalized_symbols_) = hermitian_matrix[1][1] * temp_equalized_symbols[0] - hermitian_matrix[0][1] * temp_equalized_symbols[1];
            *(equalized_symbols_) /= det;
            *(equalized_symbols_ + 1) = -hermitian_matrix[1][0] * temp_equalized_symbols[0] + hermitian_matrix[0][0] * temp_equalized_symbols[1];
            *(equalized_symbols_ + 1) /= det;
            equalized_symbols_ += 2;
        }

     /*
        complex < float > temp_coef;/// h01 and h10 from H^h . H matrix.
        float temp_matrix_float[2]; /// h00 and h11 from from H^h . H matrix.
        float det = 0;
        complex < float > *h00 = channel_coefficients_[0][0].data(); /// h_rx_tx
        complex < float > *h01 = channel_coefficients_[0][1].data();
        complex < float > *h10 = channel_coefficients_[1][0].data();
        complex < float > *h11 = channel_coefficients_[1][1].data();
        int symbol = 0;
        int sc = 0;

        for (int re = 0; re < num_re_pdsch_; re++) {

            temp_matrix_float[0] = h00[re].real() * h00[re].real() + h00[re].imag() * h00[re].imag() +
                                   h10[re].real() * h10[re].real() + h10[re].imag() * h10[re].imag(); /// Coef 0,0
            temp_matrix_float[1] = h01[re].real() * h01[re].real() + h01[re].imag() * h01[re].imag() +
                                   h11[re].real() * h11[re].real() + h11[re].imag() * h11[re].imag(); /// Coef 1, 1
            temp_coef = conj(h00[re]) * h01[re] + conj(h10[re]) * h11[re]; /// Coef 0,1, and conj of Coef 1, 0

            det = temp_matrix_float[0] * temp_matrix_float[1] -
                  (temp_coef.real() * temp_coef.real() + temp_coef.imag() * temp_coef.imag());

            *(equalized_symbols_) = temp_matrix_float[1] * (conj(h00[re]) * pdsch_samples_[0][re] +
                                                            conj(h10[re]) * pdsch_samples_[1][re])
                                    - temp_coef * (conj(h01[re]) * pdsch_samples_[0][re] +
                                                   conj(h11[re]) * pdsch_samples_[1][re]);

            *(equalized_symbols_ + 1) =
                    -conj(temp_coef) * (conj(h00[re]) * pdsch_samples_[0][re] +
                                        conj(h10[re]) * pdsch_samples_[1][re])
                    + temp_matrix_float[0] * (conj(h01[re]) * pdsch_samples_[0][re] +
                                              conj(h11[re]) * pdsch_samples_[1][re]);

            *(equalized_symbols_) /= det;
            *(equalized_symbols_ + 1) /= det;
            equalized_symbols_ += 2;
        } */
    }
}

void vblast_zf_3_layers(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                        std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                        int num_re_pdsch_,
                        std::complex<float> *equalized_symbols_,
                        int nb_rx_ports_) {

    /************** Do not use closed form inversion because determinant is close to 0 *****************/

        // Compute H^H * H to work on a square matrix
        //complex <float> temp_coefs[3]; /// 3 upper elements of H^H.H
        //float temp_matrix_float[3]; /// diagonal elements of H^H.H

        /*
        complex<float> hermitian_matrix[3][3];
        complex<float> inverse_matrix[3][3];
        float det = 0;
        vector<complex<float>> temp_equalized_symbols(3);
        vector<complex<float>> temp_inv(6);

        int i;

        for (int re = 0; re < num_re_pdsch_; re++) {

            hermitian_matrix[0][0] = channel_coefficients_[0][0][re].real() * channel_coefficients_[0][0][re].real() +
                                     channel_coefficients_[0][0][re].imag() * channel_coefficients_[0][0][re].imag();
            hermitian_matrix[0][1] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][1][re];
            hermitian_matrix[0][2] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][2][re];
            hermitian_matrix[1][1] = channel_coefficients_[0][1][re].real() * channel_coefficients_[0][1][re].real() +
                                     channel_coefficients_[0][1][re].imag() * channel_coefficients_[0][1][re].imag();
            hermitian_matrix[1][2] = conj(channel_coefficients_[0][1][re]) * channel_coefficients_[0][2][re];
            hermitian_matrix[2][2] = channel_coefficients_[0][2][re].real() * channel_coefficients_[0][2][re].real() +
                                     channel_coefficients_[0][2][re].imag() * channel_coefficients_[0][2][re].imag();

            for(i = 1; i < nb_rx_ports_; i++) {
                hermitian_matrix[0][0] += channel_coefficients_[i][0][re].real() * channel_coefficients_[i][0][re].real() +
                         channel_coefficients_[i][0][re].imag() * channel_coefficients_[i][0][re].imag();
                hermitian_matrix[0][1] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][1][re];
                hermitian_matrix[0][2] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][2][re];
                hermitian_matrix[1][1] += channel_coefficients_[i][1][re].real() * channel_coefficients_[i][1][re].real() +
                                         channel_coefficients_[i][1][re].imag() * channel_coefficients_[i][1][re].imag();
                hermitian_matrix[1][2] += conj(channel_coefficients_[i][1][re]) * channel_coefficients_[i][2][re];
                hermitian_matrix[2][2] += channel_coefficients_[i][2][re].real() * channel_coefficients_[i][2][re].real() +
                         channel_coefficients_[i][2][re].imag() * channel_coefficients_[i][2][re].imag();
            }
            hermitian_matrix[1][0] = conj(hermitian_matrix[0][1]);
            hermitian_matrix[2][0] = conj(hermitian_matrix[0][2]);
            hermitian_matrix[2][1] = conj(hermitian_matrix[1][2]);

            det = hermitian_matrix[0][0].real() * hermitian_matrix[1][1].real() * hermitian_matrix[2][2].real() +
                  2 * (hermitian_matrix[0][1] * hermitian_matrix[1][2] * hermitian_matrix[2][0]).real() -
                  hermitian_matrix[1][1].real() * (hermitian_matrix[0][2].real() * hermitian_matrix[0][2].real() + hermitian_matrix[0][2].imag() * hermitian_matrix[0][2].imag()) -
                  hermitian_matrix[2][2].real() * (hermitian_matrix[0][1].real() * hermitian_matrix[0][1].real() + hermitian_matrix[0][1].imag() * hermitian_matrix[0][1].imag()) -
                  hermitian_matrix[0][0].real() * (hermitian_matrix[1][2].real() * hermitian_matrix[1][2].real() + hermitian_matrix[1][2].imag() * hermitian_matrix[1][2].imag());

            temp_equalized_symbols[0] = conj(channel_coefficients_[0][0][re]) * pdsch_samples_[0][re];
            temp_equalized_symbols[1] = conj(channel_coefficients_[0][1][re]) * pdsch_samples_[0][re];
            temp_equalized_symbols[2] = conj(channel_coefficients_[0][2][re]) * pdsch_samples_[0][re];
            for(i = 1; i < nb_rx_ports_; i++) {
                temp_equalized_symbols[0] += conj(channel_coefficients_[i][0][re]) * pdsch_samples_[i][re];
                temp_equalized_symbols[1] += conj(channel_coefficients_[i][1][re]) * pdsch_samples_[i][re];
                temp_equalized_symbols[2] += conj(channel_coefficients_[i][2][re]) * pdsch_samples_[i][re];
            }

            /// Invert matrix
            inverse_matrix[0][0] = hermitian_matrix[1][1].real() * hermitian_matrix[2][2].real() - (hermitian_matrix[1][2].real() * hermitian_matrix[1][2].real() + hermitian_matrix[1][2].imag() * hermitian_matrix[1][2].imag());
            inverse_matrix[0][1] = -(hermitian_matrix[0][1] * hermitian_matrix[2][2].real() - hermitian_matrix[0][2] * hermitian_matrix[2][1]);
            inverse_matrix[0][2] = -(hermitian_matrix[0][1] * hermitian_matrix[1][2] - hermitian_matrix[0][2] * hermitian_matrix[1][1].real());
            inverse_matrix[1][0] = conj(inverse_matrix[0][1]);
            inverse_matrix[1][1] = hermitian_matrix[0][0].real() * hermitian_matrix[2][2].real() - (hermitian_matrix[0][2].real() * hermitian_matrix[0][2].real() + hermitian_matrix[0][2].imag() * hermitian_matrix[0][2].imag());
            inverse_matrix[1][2] = -(hermitian_matrix[0][0].real() * hermitian_matrix[1][2] - hermitian_matrix[0][2] * hermitian_matrix[1][0]);
            inverse_matrix[2][0] = conj(inverse_matrix[0][2]);
            inverse_matrix[2][1] = conj(inverse_matrix[1][2]);
            inverse_matrix[2][2] = hermitian_matrix[0][0].real() * hermitian_matrix[1][1].real() - (hermitian_matrix[0][1].real() * hermitian_matrix[0][1].real() + hermitian_matrix[0][1].imag() * hermitian_matrix[0][1].imag());

            *(equalized_symbols_) = temp_equalized_symbols[0] * inverse_matrix[0][0].real() +
                                    temp_equalized_symbols[1] * inverse_matrix[0][1] +
                                    temp_equalized_symbols[2] * inverse_matrix[0][2];
            *(equalized_symbols_) /= det;
            *(equalized_symbols_ + 1) = temp_equalized_symbols[0] * inverse_matrix[1][0] +
                                        temp_equalized_symbols[1] * inverse_matrix[1][1].real() +
                                        temp_equalized_symbols[2] * inverse_matrix[1][2];
            *(equalized_symbols_ + 1) /= det;
            *(equalized_symbols_ + 2) = temp_equalized_symbols[0] * inverse_matrix[2][0] +
                                        temp_equalized_symbols[1] * inverse_matrix[2][1] +
                                        temp_equalized_symbols[2] * inverse_matrix[2][2].real();
            *(equalized_symbols_ + 2) /= det;

            equalized_symbols_ += 3;

            /**
            temp_matrix_float[0] = channel_coefficients_[0][0][re].real() * channel_coefficients_[0][0][re].real() +
                                   channel_coefficients_[0][0][re].imag() * channel_coefficients_[0][0][re].imag() +
                                   channel_coefficients_[1][0][re].real() * channel_coefficients_[1][0][re].real() +
                                   channel_coefficients_[1][0][re].imag() * channel_coefficients_[1][0][re].imag() +
                                   channel_coefficients_[2][0][re].real() * channel_coefficients_[2][0][re].real() +
                                   channel_coefficients_[2][0][re].imag() * channel_coefficients_[2][0][re].imag(); /// h00
            temp_matrix_float[1] = channel_coefficients_[0][1][re].real() * channel_coefficients_[0][1][re].real() +
                                   channel_coefficients_[0][1][re].imag() * channel_coefficients_[0][1][re].imag() +
                                   channel_coefficients_[1][1][re].real() * channel_coefficients_[1][1][re].real() +
                                   channel_coefficients_[1][1][re].imag() * channel_coefficients_[1][1][re].imag() +
                                   channel_coefficients_[2][1][re].real() * channel_coefficients_[2][1][re].real() +
                                   channel_coefficients_[2][1][re].imag() * channel_coefficients_[2][1][re].imag(); /// h11
            temp_matrix_float[2] = channel_coefficients_[0][2][re].real() * channel_coefficients_[0][2][re].real() +
                                   channel_coefficients_[0][2][re].imag() * channel_coefficients_[0][2][re].imag() +
                                   channel_coefficients_[1][2][re].real() * channel_coefficients_[1][2][re].real() +
                                   channel_coefficients_[1][2][re].imag() * channel_coefficients_[1][2][re].imag() +
                                   channel_coefficients_[2][2][re].real() * channel_coefficients_[2][2][re].real() +
                                   channel_coefficients_[2][2][re].imag() * channel_coefficients_[2][2][re].imag(); /// h22

            temp_coefs[0] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][1][re] +
                            conj(channel_coefficients_[1][0][re]) * channel_coefficients_[1][1][re] +
                            conj(channel_coefficients_[2][0][re]) * channel_coefficients_[2][1][re]; /// h01
            temp_coefs[1] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][2][re] +
                            conj(channel_coefficients_[1][0][re]) * channel_coefficients_[1][2][re] +
                            conj(channel_coefficients_[2][0][re]) * channel_coefficients_[2][2][re]; /// h02
            temp_coefs[2] = conj(channel_coefficients_[0][1][re]) * channel_coefficients_[0][2][re] +
                            conj(channel_coefficients_[1][1][re]) * channel_coefficients_[1][2][re] +
                            conj(channel_coefficients_[2][1][re]) * channel_coefficients_[2][2][re]; /// h12

            det = temp_matrix_float[0] * temp_matrix_float[1] * temp_matrix_float[2] +
                  2 * (temp_coefs[0] * temp_coefs[2] * conj(temp_coefs[1])).real() +
                  -(temp_coefs[1].real() * temp_coefs[1].real() + temp_coefs[1].imag() * temp_coefs[1].imag())* temp_matrix_float[1] -
                   (temp_coefs[0].real() * temp_coefs[0].real() + temp_coefs[0].imag() * temp_coefs[0].imag()) * temp_matrix_float[2] -
                   (temp_coefs[2].real() * temp_coefs[2].real() + temp_coefs[2].imag() * temp_coefs[2].imag()) * temp_matrix_float[0];

            temp_mrc[0] = conj(channel_coefficients_[0][0][re]) * pdsch_samples_[0][re] +
                          conj(channel_coefficients_[1][0][re]) * pdsch_samples_[1][re] +
                          conj(channel_coefficients_[2][0][re]) * pdsch_samples_[2][re];
            temp_mrc[1] = conj(channel_coefficients_[0][1][re]) * pdsch_samples_[0][re] +
                          conj(channel_coefficients_[1][1][re]) * pdsch_samples_[1][re] +
                          conj(channel_coefficients_[2][1][re]) * pdsch_samples_[2][re];
            temp_mrc[2] = conj(channel_coefficients_[0][2][re]) * pdsch_samples_[0][re] +
                          conj(channel_coefficients_[1][2][re]) * pdsch_samples_[1][re] +
                          conj(channel_coefficients_[2][2][re]) * pdsch_samples_[2][re];

            /// Diagonal elements of inverse matrix
            temp_inv[0] = temp_matrix_float[1] * temp_matrix_float[2] -
                          abs(temp_coefs[2]) * abs(temp_coefs[2]); /// 00
            temp_inv[1] = temp_matrix_float[0] * temp_matrix_float[2] -
                          abs(temp_coefs[1]) * abs(temp_coefs[1]); /// 11
            temp_inv[2] = temp_matrix_float[0] * temp_matrix_float[1] -
                          abs(temp_coefs[0]) * abs(temp_coefs[0]); /// 22

            temp_inv[3] = -(temp_coefs[0] * temp_matrix_float[2] - temp_coefs[1] * conj(temp_coefs[2])); /// 01
            temp_inv[4] = temp_coefs[0] * temp_coefs[2] - temp_coefs[1] * temp_matrix_float[1]; /// 02
            temp_inv[5] = -(temp_matrix_float[0] * temp_coefs[2] - temp_coefs[1] * conj(temp_coefs[0])); /// 12

            *(equalized_symbols_) = temp_inv[0].real() * temp_mrc[0] +
                        temp_inv[3] * temp_mrc[1] +
                        temp_inv[4] * temp_mrc[2];
            *(equalized_symbols_ + 1) = conj(temp_inv[3]) * temp_mrc[0] +
                                        temp_inv[1].real() * temp_mrc[1] +
                                        temp_inv[5] * temp_mrc[2];
            *(equalized_symbols_ + 2) = conj(temp_inv[4]) * temp_mrc[0] +
                                        conj(temp_inv[5]) * temp_mrc[1] +
                                        temp_inv[2].real() * temp_mrc[2];
            *(equalized_symbols_) /= det;
            *(equalized_symbols_ + 1) /= det;
            *(equalized_symbols_ + 2) /= det;
            equalized_symbols_ += 3;

        } */

    complex<float> hermitian_matrix[MAX_TX_PORTS][MAX_TX_PORTS];
    complex<float> temp_equalized_symbols[3];
    int i, j;
    int l;

#if TIME_MEASURE == 1
    #if defined(CLOCK_TYPE_CHRONO)
    std::chrono::steady_clock::time_point t1, t2;
#elif defined(CLOCK_TYPE_GETTIME)
    struct timespec t1, t2;
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
    struct timespec t1, t2;
#else
    uint64_t t1, t2;
    unsigned cycles_low1, cycles_low2, cycles_high1, cycles_high2;
#endif
#endif
    for(int re = 0; re < num_re_pdsch_; re++) {
#if TIME_MEASURE == 1
        BOOST_LOG_TRIVIAL(trace) << " ------------------------- " << endl;
        BOOST_LOG_TRIVIAL(trace) << "RE number : " << re << endl;
#if defined(CLOCK_TYPE_CHRONO)
        t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#endif
#endif

        /// First line
        hermitian_matrix[0][0].real(channel_coefficients_[0][0][re].real() * channel_coefficients_[0][0][re].real() +
                                    channel_coefficients_[0][0][re].imag() * channel_coefficients_[0][0][re].imag());
        hermitian_matrix[0][1] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][1][re];
        hermitian_matrix[0][2] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][2][re];

        /// Second line from diag coef 1,1
        hermitian_matrix[1][1].real(channel_coefficients_[0][1][re].real() * channel_coefficients_[0][1][re].real() +
                                    channel_coefficients_[0][1][re].imag() * channel_coefficients_[0][1][re].imag());
        hermitian_matrix[1][2] = conj(channel_coefficients_[0][1][re]) * channel_coefficients_[0][2][re];

        /// Third line from diag coef 2,2
        hermitian_matrix[2][2].real(channel_coefficients_[0][2][re].real() * channel_coefficients_[0][2][re].real() +
                                    channel_coefficients_[0][2][re].imag() * channel_coefficients_[0][2][re].imag());

        /// Compute hermitian matrix
        for(i = 1; i < nb_rx_ports_; i++) {
            hermitian_matrix[0][0] += channel_coefficients_[i][0][re].real() * channel_coefficients_[i][0][re].real() +
                                      channel_coefficients_[i][0][re].imag() * channel_coefficients_[i][0][re].imag();
            hermitian_matrix[0][1] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][1][re];
            hermitian_matrix[0][2] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][2][re];
            hermitian_matrix[1][1] += channel_coefficients_[i][1][re].real() * channel_coefficients_[i][1][re].real() +
                                      channel_coefficients_[i][1][re].imag() * channel_coefficients_[i][1][re].imag();
            hermitian_matrix[1][2] += conj(channel_coefficients_[i][1][re]) * channel_coefficients_[i][2][re];
            hermitian_matrix[2][2] += channel_coefficients_[i][2][re].real() * channel_coefficients_[i][2][re].real() +
                                      channel_coefficients_[i][2][re].imag() * channel_coefficients_[i][2][re].imag();
        }

#if TIME_MEASURE == 1
        #if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
        << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
        t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
        << (t2 - t1)/TSC_FREQ * 1e9 << endl;

        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#endif
#endif

        ldl_decomp_test(hermitian_matrix, // row major
                        3);

#if TIME_MEASURE == 1
        #if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();

                BOOST_LOG_TRIVIAL(trace) << "Time to compute LDL [ns] : " <<
                     std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

                t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute LDL [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute LDL [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute LDL [ns] : "
                                 << (t2 - t1)/TSC_FREQ * 1e9 << endl;

        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#endif
#endif

        /// Compute in-place inverse
        hermitian_matrix[0][1]  = -hermitian_matrix[0][1];
        hermitian_matrix[0][2]  = -hermitian_matrix[0][2];
        hermitian_matrix[1][2]  = -hermitian_matrix[1][2];

        /// Compute coefficients in the correct order
        hermitian_matrix[0][2] += hermitian_matrix[0][1] * hermitian_matrix[1][2];

        /// Copy inverse R^(-1)^H in lower part of the array
        hermitian_matrix[1][0] = conj(hermitian_matrix[0][1]);
        hermitian_matrix[2][0] = conj(hermitian_matrix[0][2]);
        hermitian_matrix[2][1] = conj(hermitian_matrix[1][2]);

#if TIME_MEASURE == 1
        #if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(trace) << "Time to compute the two inverse matrices [ns] : " <<
             std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
        t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute the two inverse matrices [ns] : "
                    << (t2.tv_nsec - t1.tv_nsec) << endl; ;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute the two inverse matrices [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl; ;
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute the two inverse matrices [ns] : "
                                 << (t2 - t1)/TSC_FREQ * 1e9 << endl;

        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#endif
#endif

        /// Multiply received signal y by H^H, then multiply by the inverse
        for(i = 0; i < 3; i++) {
            *(equalized_symbols_ + i) = conj(channel_coefficients_[0][i][re]) * pdsch_samples_[0][re];
        }
        for(j = 1; j < nb_rx_ports_; j++) {
            for(i = 0; i < 3; i++) {
                *(equalized_symbols_ + i) += conj(channel_coefficients_[j][i][re]) * pdsch_samples_[j][re];
            }
        }

        *(equalized_symbols_ + 2) += hermitian_matrix[2][0] * *(equalized_symbols_) +
                                     hermitian_matrix[2][1] * *(equalized_symbols_ + 1);
        *(equalized_symbols_ + 1) += hermitian_matrix[1][0] * *(equalized_symbols_);
        *(equalized_symbols_)     /= hermitian_matrix[0][0].real();
        *(equalized_symbols_ + 1) /= hermitian_matrix[1][1].real();
        *(equalized_symbols_ + 2) /= hermitian_matrix[2][2].real();

        *(equalized_symbols_)     += hermitian_matrix[0][1] * *(equalized_symbols_ + 1) +
                                     hermitian_matrix[0][2] * *(equalized_symbols_ + 2);
        *(equalized_symbols_ + 1) += hermitian_matrix[1][2] * *(equalized_symbols_ + 2);

        equalized_symbols_ += 3;

#if TIME_MEASURE == 1
        #if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();
                BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : " <<
                     std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : "
                                 << (t2 - t1)/TSC_FREQ * 1e9 << endl;
#endif
#endif
        }

}

void vblast_4_layers_block_wise_inversion(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                                          std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                                          int num_re_pdsch_,
                                          std::complex<float> *equalized_symbols_,
                                          int nb_rx_ports_) {

    int i, j, k;
    complex<float> hermitian_matrix[4][4];
    //complex<float> copy_hermitian_matrix[4][4];
    complex<float> temp_equalized_symbols[4];
    //complex<float> test_inverse[4][4];
    //complex<float> test_inverse_2x2[2][2];
    //memset(test_inverse_2x2, 0, 4 * sizeof(complex<float>));

    //complex<float> transpose_conjugate_matrix[MAX_RX_PORTS][4];
    //complex<float> temp_received_pdsch_samples[MAX_RX_PORTS];

    /**
    cout << "hermitian matrix init" << endl;
    for(j = 0; j < 4; j++) {
        for(k = 0; k < 4; k++) {
            cout << hermitian_matrix[j][k];
        }
        cout << endl;
    } */

    complex<float> d_inv[2][2], a_inv[2][2];
    memset(d_inv, 0, 4 * sizeof(complex<float>));
    memset(a_inv, 0, 4 * sizeof(complex<float>));
    complex<float> dot_prod_b_d_inv[2][2], dot_prod_c_a_inv[2][2];
    complex<float> inverse_hermitian_matrix[4][4];
    float det;
    //complex<float> det;

    for(int re = 0; re < num_re_pdsch_; re++) {

        /**
        for(i = 0; i < 4; i++) {
            for(j = 0; j < 4; j++) {
                hermitian_matrix[i][j] = channel_coefficients_[i][j][re];
            }
        } */

        /**
        for(i = 0; i < nb_rx_ports_; i++) {
            for(j = 0; j < 4; j++) {
                transpose_conjugate_matrix[j][i] = conj(channel_coefficients_[i][j][re]);
            }
            temp_received_pdsch_samples[i] = pdsch_samples_[i][re];
        } */

        /// First line
        hermitian_matrix[0][0].real(channel_coefficients_[0][0][re].real() * channel_coefficients_[0][0][re].real() +
                                    channel_coefficients_[0][0][re].imag() * channel_coefficients_[0][0][re].imag());
        hermitian_matrix[0][1] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][1][re];
        hermitian_matrix[0][2] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][2][re];
        hermitian_matrix[0][3] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][3][re];

        /// Second line from diag coef 1,1
        hermitian_matrix[1][1].real(channel_coefficients_[0][1][re].real() * channel_coefficients_[0][1][re].real() +
                                    channel_coefficients_[0][1][re].imag() * channel_coefficients_[0][1][re].imag());
        hermitian_matrix[1][2] = conj(channel_coefficients_[0][1][re]) * channel_coefficients_[0][2][re];
        hermitian_matrix[1][3] = conj(channel_coefficients_[0][1][re]) * channel_coefficients_[0][3][re];

        /// Third line from diag coef 2,2
        hermitian_matrix[2][2].real(channel_coefficients_[0][2][re].real() * channel_coefficients_[0][2][re].real() +
                                    channel_coefficients_[0][2][re].imag() * channel_coefficients_[0][2][re].imag());
        hermitian_matrix[2][3] = conj(channel_coefficients_[0][2][re]) * channel_coefficients_[0][3][re];

        /// Fourth line from diag coef 3,3
        hermitian_matrix[3][3].real(channel_coefficients_[0][3][re].real() * channel_coefficients_[0][3][re].real() +
                                    channel_coefficients_[0][3][re].imag() * channel_coefficients_[0][3][re].imag());

        /// Compute hermitian matrix
        for(i = 1; i < nb_rx_ports_; i++) {
            hermitian_matrix[0][0] += channel_coefficients_[i][0][re].real() * channel_coefficients_[i][0][re].real() +
                                      channel_coefficients_[i][0][re].imag() * channel_coefficients_[i][0][re].imag();
            hermitian_matrix[0][1] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][1][re];
            hermitian_matrix[0][2] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][2][re];
            hermitian_matrix[0][3] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][3][re];
            hermitian_matrix[1][1] += channel_coefficients_[i][1][re].real() * channel_coefficients_[i][1][re].real() +
                                      channel_coefficients_[i][1][re].imag() * channel_coefficients_[i][1][re].imag();
            hermitian_matrix[1][2] += conj(channel_coefficients_[i][1][re]) * channel_coefficients_[i][2][re];
            hermitian_matrix[1][3] += conj(channel_coefficients_[i][1][re]) * channel_coefficients_[i][3][re];
            hermitian_matrix[2][2] += channel_coefficients_[i][2][re].real() * channel_coefficients_[i][2][re].real() +
                                      channel_coefficients_[i][2][re].imag() * channel_coefficients_[i][2][re].imag();
            hermitian_matrix[2][3] += conj(channel_coefficients_[i][2][re]) * channel_coefficients_[i][3][re];
            hermitian_matrix[3][3] += channel_coefficients_[i][3][re].real() * channel_coefficients_[i][3][re].real() +
                                      channel_coefficients_[i][3][re].imag() * channel_coefficients_[i][3][re].imag();
        }

        hermitian_matrix[1][0] = conj(hermitian_matrix[0][1]);
        hermitian_matrix[2][0] = conj(hermitian_matrix[0][2]);
        hermitian_matrix[2][1] = conj(hermitian_matrix[1][2]);
        hermitian_matrix[3][0] = conj(hermitian_matrix[0][3]);
        hermitian_matrix[3][1] = conj(hermitian_matrix[1][3]);
        hermitian_matrix[3][2] = conj(hermitian_matrix[2][3]);

        /**
        memcpy(copy_hermitian_matrix, hermitian_matrix, 16 * sizeof(complex<float>));

        cout << "hermitian matrix : " << endl;
        for(i = 0; i < 4; i++) {
            for(j = 0; j < 4; j++) {
                cout << hermitian_matrix[i][j] << " ";
            }
            cout << endl;
        } */

        /// Compute D^(-1)
        det = hermitian_matrix[2][2].real() * hermitian_matrix[3][3].real() - (hermitian_matrix[2][3].real() * hermitian_matrix[2][3].real() + hermitian_matrix[2][3].imag() * hermitian_matrix[2][3].imag());
        //det = hermitian_matrix[2][2] * hermitian_matrix[3][3] - hermitian_matrix[2][3] * hermitian_matrix[3][2];
        d_inv[0][0] = hermitian_matrix[3][3].real() / det;
        d_inv[0][1] = -hermitian_matrix[2][3] / det;
        d_inv[1][0] = conj(d_inv[0][1]);
        d_inv[1][1] = hermitian_matrix[2][2].real() / det;

        /**
        memset(test_inverse_2x2, 0, 4 * sizeof(complex<float>));
        for(i = 0; i < 2; i++) {
            for(j = 0; j < 2; j++) {
                for(k = 0; k < 2; k++) {
                    test_inverse_2x2[i][j] += d_inv[i][k] * hermitian_matrix[k + 2][j + 2];
                }
            }
        }
        cout << "verif D^-1" << endl;
        for(i = 0; i < 2; i++) {
            for(j = 0; j < 2; j++) {
                cout << test_inverse_2x2[i][j];
            }
            cout << endl;
        } */

        /// Compute A^(-1)
        det = hermitian_matrix[0][0].real() * hermitian_matrix[1][1].real() - (hermitian_matrix[0][1].real() * hermitian_matrix[0][1].real() + hermitian_matrix[0][1].imag() * hermitian_matrix[0][1].imag());
        //det = hermitian_matrix[0][0] * hermitian_matrix[1][1] - hermitian_matrix[0][1] * hermitian_matrix[1][0];
        a_inv[0][0] = hermitian_matrix[1][1].real() / det;
        a_inv[0][1] = -hermitian_matrix[0][1] / det;
        a_inv[1][0] = conj(a_inv[0][1]);
        a_inv[1][1] = hermitian_matrix[0][0].real() / det;

        /**
        memset(test_inverse_2x2, 0, 4 * sizeof(complex<float>));
        for(i = 0; i < 2; i++) {
            for(j = 0; j < 2; j++) {
                for(k = 0; k < 2; k++) {
                    test_inverse_2x2[i][j] += a_inv[i][k] * hermitian_matrix[k][j];
                }
            }
        }
        cout << "verif A^-1" << endl;
        for(i = 0; i < 2; i++) {
            for(j = 0; j < 2; j++) {
                cout << test_inverse_2x2[i][j];
            }
            cout << endl;
        } */

        /// Compute B*D^(-1)
        dot_prod_b_d_inv[0][0] = hermitian_matrix[0][2] * d_inv[0][0].real() + hermitian_matrix[0][3] * d_inv[1][0];
        dot_prod_b_d_inv[0][1] = hermitian_matrix[0][2] * d_inv[0][1] + hermitian_matrix[0][3] * d_inv[1][1].real();
        dot_prod_b_d_inv[1][0] = hermitian_matrix[1][2] * d_inv[0][0].real() + hermitian_matrix[1][3] * d_inv[1][0];
        dot_prod_b_d_inv[1][1] = hermitian_matrix[1][2] * d_inv[0][1] + hermitian_matrix[1][3] * d_inv[1][1].real();

        /**
        memset(test_inverse_2x2, 0, 4 * sizeof(complex<float>));
        for(i = 0; i < 2; i++) {
            for(j = 0; j < 2; j++) {
                for(k = 0; k < 2; k++) {
                    test_inverse_2x2[i][j] += dot_prod_b_d_inv[i][k] * hermitian_matrix[k + 2][j + 2];
                }
            }
        }
        cout << "verif BD^-1 * D" << endl;
        for(i = 0; i < 2; i++) {
            for(j = 0; j < 2; j++) {
                cout << test_inverse_2x2[i][j];
            }
            cout << endl;
        } */

        /**
        dot_prod_b_d_inv[0][0] = hermitian_matrix[0][2] * d_inv[0][0] + hermitian_matrix[0][3] * d_inv[1][0];
        dot_prod_b_d_inv[0][1] = hermitian_matrix[0][2] * d_inv[0][1] + hermitian_matrix[0][3] * d_inv[1][1];
        dot_prod_b_d_inv[1][0] = hermitian_matrix[1][2] * d_inv[0][0] + hermitian_matrix[1][3] * d_inv[1][0];
        dot_prod_b_d_inv[1][1] = hermitian_matrix[1][2] * d_inv[0][1] + hermitian_matrix[1][3] * d_inv[1][1]; */

        /// Compute C*A^(-1)
        dot_prod_c_a_inv[0][0] = hermitian_matrix[2][0] * a_inv[0][0].real() + hermitian_matrix[2][1] * a_inv[1][0];
        dot_prod_c_a_inv[0][1] = hermitian_matrix[2][0] * a_inv[0][1] + hermitian_matrix[2][1] * a_inv[1][1].real();
        dot_prod_c_a_inv[1][0] = hermitian_matrix[3][0] * a_inv[0][0].real() + hermitian_matrix[3][1] * a_inv[1][0];
        dot_prod_c_a_inv[1][1] = hermitian_matrix[3][0] * a_inv[0][1] + hermitian_matrix[3][1] * a_inv[1][1].real();

        /**
        memset(test_inverse_2x2, 0, 4 * sizeof(complex<float>));
        for(i = 0; i < 2; i++) {
            for(j = 0; j < 2; j++) {
                for(k = 0; k < 2; k++) {
                    test_inverse_2x2[i][j] += dot_prod_c_a_inv[i][k] * hermitian_matrix[k][j];
                }
            }
        }
        cout << "verif CA^-1 * A" << endl;
        for(i = 0; i < 2; i++) {
            for(j = 0; j < 2; j++) {
                cout << test_inverse_2x2[i][j];
            }
            cout << endl;
        } */

        /**
        dot_prod_c_a_inv[0][0] = hermitian_matrix[2][0] * a_inv[0][0] + hermitian_matrix[2][1] * a_inv[1][0];
        dot_prod_c_a_inv[0][1] = hermitian_matrix[2][0] * a_inv[0][1] + hermitian_matrix[2][1] * a_inv[1][1];
        dot_prod_c_a_inv[1][0] = hermitian_matrix[3][0] * a_inv[0][0] + hermitian_matrix[3][1] * a_inv[1][0];
        dot_prod_c_a_inv[1][1] = hermitian_matrix[3][0] * a_inv[0][1] + hermitian_matrix[3][1] * a_inv[1][1]; */

        /// Compute (A - BD^(-1)C)
        hermitian_matrix[0][0] -= (dot_prod_b_d_inv[0][0] * hermitian_matrix[2][0] + dot_prod_b_d_inv[0][1] * hermitian_matrix[3][0]).real();
        hermitian_matrix[0][1] -= (dot_prod_b_d_inv[0][0] * hermitian_matrix[2][1] + dot_prod_b_d_inv[0][1] * hermitian_matrix[3][1]);
        //hermitian_matrix[1][0] -= (dot_prod_b_d_inv[1][0] * hermitian_matrix[2][0] + dot_prod_b_d_inv[1][1] * hermitian_matrix[3][0]);
        hermitian_matrix[1][1] -= (dot_prod_b_d_inv[1][0] * hermitian_matrix[2][1] + dot_prod_b_d_inv[1][1] * hermitian_matrix[3][1]).real();

        /// Compute (D - CA^(-1)B)
        hermitian_matrix[2][2] -= (dot_prod_c_a_inv[0][0] * hermitian_matrix[0][2] + dot_prod_c_a_inv[0][1] * hermitian_matrix[1][2]).real();
        hermitian_matrix[2][3] -= (dot_prod_c_a_inv[0][0] * hermitian_matrix[0][3] + dot_prod_c_a_inv[0][1] * hermitian_matrix[1][3]);
        //hermitian_matrix[3][2] -= (dot_prod_c_a_inv[1][0] * hermitian_matrix[0][2] + dot_prod_c_a_inv[1][1] * hermitian_matrix[1][2]);
        hermitian_matrix[3][3] -= (dot_prod_c_a_inv[1][0] * hermitian_matrix[0][3] + dot_prod_c_a_inv[1][1] * hermitian_matrix[1][3]).real();

        /// Compute the inverse hermitian matrix
        /// Upper left block (A - BD^(-1)C)^(-1)
        det = hermitian_matrix[0][0].real() * hermitian_matrix[1][1].real() - (hermitian_matrix[0][1].real() * hermitian_matrix[0][1].real() + hermitian_matrix[0][1].imag() * hermitian_matrix[0][1].imag());
        inverse_hermitian_matrix[0][0] = hermitian_matrix[1][1].real() / det;
        inverse_hermitian_matrix[0][1] = -hermitian_matrix[0][1] / det;
        inverse_hermitian_matrix[1][0] = conj(inverse_hermitian_matrix[0][1]);
        inverse_hermitian_matrix[1][1] = hermitian_matrix[0][0].real() / det;

        /**
        memset(test_inverse_2x2, 0, 4 * sizeof(complex<float>));
        for(i = 0; i < 2; i++) {
            for(j = 0; j < 2; j++) {
                for(k = 0; k < 2; k++) {
                    test_inverse_2x2[i][j] += inverse_hermitian_matrix[i][k] * hermitian_matrix[k][j];
                }
            }
        }
        cout << "verif (A - BD^(-1)C)^(-1) * (A - BD^(-1)C)" << endl;
        for(i = 0; i < 2; i++) {
            for(j = 0; j < 2; j++) {
                cout << test_inverse_2x2[i][j];
            }
            cout << endl;
        } */

        /// Upper right block -(A - BD^(-1)C)^(-1)BD^(-1)
        inverse_hermitian_matrix[0][2] = -inverse_hermitian_matrix[0][0] * dot_prod_b_d_inv[0][0] - inverse_hermitian_matrix[0][1] * dot_prod_b_d_inv[1][0];
        inverse_hermitian_matrix[0][3] = -inverse_hermitian_matrix[0][0] * dot_prod_b_d_inv[0][1] - inverse_hermitian_matrix[0][1] * dot_prod_b_d_inv[1][1];
        inverse_hermitian_matrix[1][2] = -inverse_hermitian_matrix[1][0] * dot_prod_b_d_inv[0][0] - inverse_hermitian_matrix[1][1] * dot_prod_b_d_inv[1][0];
        inverse_hermitian_matrix[1][3] = -inverse_hermitian_matrix[1][0] * dot_prod_b_d_inv[0][1] - inverse_hermitian_matrix[1][1] * dot_prod_b_d_inv[1][1];

        /// Lower right block (D - CA^(-1)B)^(-1)
        det = hermitian_matrix[2][2].real() * hermitian_matrix[3][3].real() - (hermitian_matrix[2][3].real() * hermitian_matrix[2][3].real() + hermitian_matrix[2][3].imag() * hermitian_matrix[2][3].imag());
        inverse_hermitian_matrix[2][2] = hermitian_matrix[3][3].real() / det;
        inverse_hermitian_matrix[2][3] = -hermitian_matrix[2][3] / det;
        inverse_hermitian_matrix[3][2] = conj(inverse_hermitian_matrix[2][3]);
        inverse_hermitian_matrix[3][3] = hermitian_matrix[2][2].real() / det;

        /**
        memset(test_inverse_2x2, 0, 4 * sizeof(complex<float>));
        for(i = 0; i < 2; i++) {
            for(j = 0; j < 2; j++) {
                for(k = 0; k < 2; k++) {
                    test_inverse_2x2[i][j] += inverse_hermitian_matrix[i + 2][k + 2] * hermitian_matrix[k+ 2][j + 2];
                }
            }
        }
        cout << "verif (D - CA^(-1)B)^(-1) * (D - CA^(-1)B)" << endl;
        for(i = 0; i < 2; i++) {
            for(j = 0; j < 2; j++) {
                cout << test_inverse_2x2[i][j];
            }
            cout << endl;
        } */

        /// Lower left block -(D - CA^(-1)B)^(-1)CA^(-1)
        /**
        inverse_hermitian_matrix[2][0] = -inverse_hermitian_matrix[2][2] * dot_prod_c_a_inv[0][0] - inverse_hermitian_matrix[2][3] * dot_prod_c_a_inv[1][0];
        inverse_hermitian_matrix[2][1] = -inverse_hermitian_matrix[2][2] * dot_prod_c_a_inv[0][1] - inverse_hermitian_matrix[2][3] * dot_prod_c_a_inv[1][1];
        inverse_hermitian_matrix[3][0] = -inverse_hermitian_matrix[3][2] * dot_prod_c_a_inv[0][0] - inverse_hermitian_matrix[3][3] * dot_prod_c_a_inv[1][0];
        inverse_hermitian_matrix[3][1] = -inverse_hermitian_matrix[3][2] * dot_prod_c_a_inv[0][1] - inverse_hermitian_matrix[3][3] * dot_prod_c_a_inv[1][1]; */
        inverse_hermitian_matrix[2][0] = conj(inverse_hermitian_matrix[0][2]);
        inverse_hermitian_matrix[2][1] = conj(inverse_hermitian_matrix[1][2]);
        inverse_hermitian_matrix[3][0] = conj(inverse_hermitian_matrix[0][3]);
        inverse_hermitian_matrix[3][1] = conj(inverse_hermitian_matrix[1][3]);

        /**
        cout << "inverse matrix : " << endl;
        for(i = 0; i< 4; i++) {
            for(j = 0; j < 4; j++) {
                cout << inverse_hermitian_matrix[i][j] << " ";
            }
            cout << endl;
        }

        memset(test_inverse, 0, 16 * sizeof(complex<float>));
        for(i = 0; i < 4; i++) {
            for(j = 0; j < 4; j++) {
                for(k = 0; k < 4; k++) {
                    test_inverse[i][j] += inverse_hermitian_matrix[i][k] * copy_hermitian_matrix[k][j];
                }
            }
        }
        cout << "inverse by hermitian matrix : " << endl;
        for(i = 0; i< 4; i++) {
            for(j = 0; j < 4; j++) {
                cout << test_inverse[i][j] << " ";
            }
            cout << endl;
        } */

        /// Equalize the symbols
        /// Multiply received signal y by H^H, then multiply by the inverse
        /// Multiply received signal y by H^H, then multiply by the inverse
        for(i = 0; i < 4; i++) {
            *(temp_equalized_symbols + i) = conj(channel_coefficients_[0][i][re]) * pdsch_samples_[0][re];
        }
        for(i = 1; i < nb_rx_ports_; i++) {
            for(j = 0; j < 4; j++) {
                *(temp_equalized_symbols + j) += conj(channel_coefficients_[i][j][re]) * pdsch_samples_[i][re];
            }
        }

        *(equalized_symbols_) = inverse_hermitian_matrix[0][0].real() * *(temp_equalized_symbols);
        *(equalized_symbols_ + 1) = inverse_hermitian_matrix[1][0] * *(temp_equalized_symbols);
        *(equalized_symbols_ + 2) = inverse_hermitian_matrix[2][0] * *(temp_equalized_symbols);
        *(equalized_symbols_ + 3) = inverse_hermitian_matrix[3][0] * *(temp_equalized_symbols);

        for(i = 0; i < 4; i++) {
            for(j = 1; j < 4; j++) {
                *(equalized_symbols_ + i) += inverse_hermitian_matrix[i][j] * *(temp_equalized_symbols + j);
            }
        }

        equalized_symbols_ += 4;
    }
}

void vblast_zf_4_layers(complex<float> pdsch_samples_[][MAX_RX_PORTS],
                        complex<float> channel_coefficients_[][MAX_RX_PORTS][MAX_TX_PORTS],
                        int num_re_pdsch_,
                        std::complex<float> * equalized_symbols_,
                        int nb_rx_ports_) {

    //cout << "test" << endl;

    complex<float> hermitian_matrix[4][4];
    float diag_elements[4];
    complex<float> r_matrix[4][4];
    complex<float> r_transconj_matrix[4][4];

    //complex<float> channel_matrix[4][4];
    complex<float> temp_equalized_symbols[4];
    int i, j;
    int l;

    for(i = 0; i < 4; i++) {
        r_transconj_matrix[i][i] = 1;
        for(j = i + 1; j < 4; j++) {
            r_transconj_matrix[i][j] = 0;
        }
    }

    for(i = 0; i < 4; i++) {
        r_matrix[i][i] = 1;
        for(j = i + 1; j < 4; j++) {
            r_matrix[j][i] = 0;
        }
    }

#if defined(VBLAST_AVX2)
    __m256 vec1;
    __m256 vec2;
    __m256 vec3;
    __m256 vec4;
    __m256 vec5;
    __m256 dot_product_sum_r;
    __m256 dot_product_sum_i;
    __m256 neg = _mm256_setr_ps(1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0);
    __m128 vlow, vlow1, vhigh, high64;

#endif

#if TIME_MEASURE == 1
#if defined(CLOCK_TYPE_CHRONO)
    std::chrono::steady_clock::time_point t1, t2;
#elif defined(CLOCK_TYPE_GETTIME)
    struct timespec t1, t2;
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
    struct timespec t1, t2;
#else
    uint64_t t1, t2;
    unsigned cycles_low1, cycles_low2, cycles_high1, cycles_high2;
#endif
#endif
    for(int re = 0; re < num_re_pdsch_; re++) {//, pdsch_samples_++, equalized_symbols_++, channel_coefficients_++) {
#if TIME_MEASURE == 1
        BOOST_LOG_TRIVIAL(trace) << " ------------------------- " << endl;
        BOOST_LOG_TRIVIAL(trace) << "RE number : " << re << endl;
#if defined(CLOCK_TYPE_CHRONO)
        t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#endif
#endif

#if defined(VBLAST_AVX2)

        /// Don't use any mask if number of RX ports is a multiple of 4
        if(nb_rx_ports_ % 4 == 0) {

            /// Compute the mask of the vector containing the 4 norms
            //__m256i mask = _mm256_setr_epi32(-1, -1, -1, -1, 3, 3, 3, 3);

            /// Compute diagonal elements
            for(i = 0; i < 4; i ++) {
                for (j = 0; j < nb_rx_ports_; j += 4) {
                    /// Load the mask into result vector
                    //vec2 = _mm256_maskload_ps(res_array, mask);

                    /// Load each channel coefficient in vector 1
                    vec1 = _mm256_loadu_ps((float *) channel_coefficients_[re][i]); /// Channel coefficient needs to be transposed to [re][MAX_TX_PORTS][MAX_RX_PORTS]

                    /// Multiply the 2 vectors and store the result in vector 2
                    vec1 = _mm256_mul_ps(vec1, vec1);

                    /// Add each element horizontally
                    vlow = _mm256_castps256_ps128(vec1);
                    vhigh = _mm256_extractf128_ps(vec1, 1);
                    vlow = _mm_add_ps(vlow, vhigh);
                    high64 = _mm_unpackhi_ps(vlow, vlow);
                    vlow1 = _mm_add_ps(vlow, high64);
                    high64 = _mm_permute_ps(high64, 177);
                    hermitian_matrix[i][i] = _mm_cvtss_f32(_mm_add_ps(vlow1, high64));
                }
            }

        /// Compute other elements
        /// https://www.google.com/search?channel=fs&client=ubuntu-sn&q=avx+reorder+elements#fpstate=ive&vld=cid:4a75d74f,vid:AT5nuQQO96o
        for(i = 0; i < 4; i++) {
            for(j = i + 1; j < 4; j++) {
                dot_product_sum_r = _mm256_set1_ps(0.0);
                dot_product_sum_i = _mm256_set1_ps(0.0);
                for(l = 0; l < nb_rx_ports_; l+=4) {
                    /// Load each channel coefficient in vector 1 and vector 2
                    vec1 = _mm256_loadu_ps((float *) channel_coefficients_[re][i]);
                    /// Negate imag part of vec1
                    vec1 = _mm256_mul_ps(vec1, neg);

                    vec2 = _mm256_loadu_ps((float *) channel_coefficients_[re][j]);

                    /**
                    cout << "vec2" << endl;
                    for(int k = 0; k < 8; k++) {
                        cout << vec2[k] << endl;
                    } */

                    /// Multiply vec1 by vec2
                    vec3 = _mm256_mul_ps(vec1, vec2); /// Real part

                    /// Negate imag part of vec2
                    vec2 = _mm256_mul_ps(vec2, neg);

                    /**
                    cout << "negated vec2" << endl;
                    for(int k = 0; k < 8; k++) {
                        cout << vec2[k] << endl;
                    } */

                    /// Permute Real and imag part in vec4
                    vec4 = _mm256_permute_ps(vec2, 0b10110001);

                    /**
                    cout << "Permuted vec (vec4)" << endl;
                    for(int k = 0; k < 8; k++) {
                        cout << vec4[k] << endl;
                    } */

                    /// Multiply vec1 by modified vec2
                    vec5 = _mm256_mul_ps(vec1, vec4); /// Imag part

                    /**
                    cout << "vec5" << endl;
                    for(int k = 0; k < 8; k++) {
                        cout << vec5[k] << endl;
                    }

                    cout << "dot product sum r" << endl;
                    for(int k = 0; k < 8; k++) {
                        cout << dot_product_sum_r[k] << endl;
                    }

                    cout << "dot product sum i " << endl;
                    for(int k = 0; k < 8; k++) {
                        cout << dot_product_sum_i[k] << endl;
                    }

                    cout << "vec3" << endl;
                    for(int k = 0; k < 8; k++) {
                        cout << vec3[k] << endl;
                    } */

                    /// Add elements together
                    dot_product_sum_r = _mm256_add_ps(dot_product_sum_r, vec3);
                    dot_product_sum_i = _mm256_add_ps(dot_product_sum_i, vec5);
                }

                /// Add all elements in final dot_product_sum_r vector
                dot_product_sum_r = _mm256_hadd_ps(dot_product_sum_r, dot_product_sum_r);
                dot_product_sum_r = _mm256_hadd_ps(dot_product_sum_r, dot_product_sum_r);
                dot_product_sum_r = _mm256_add_ps(dot_product_sum_r, _mm256_permute2f128_ps(dot_product_sum_r, dot_product_sum_r, 1));

                /// Add all elements in final dot_product_sum_i vector
                dot_product_sum_i = _mm256_hadd_ps(dot_product_sum_i, dot_product_sum_i);
                dot_product_sum_i = _mm256_hadd_ps(dot_product_sum_i, dot_product_sum_i);
                dot_product_sum_i = _mm256_add_ps(dot_product_sum_i, _mm256_permute2f128_ps(dot_product_sum_i, dot_product_sum_i, 1));

                /**
                cout << "hermitian matrix : " << i << " " << j << endl;
                for(l = 0; l < 4; l++) {
                    cout << dot_product_sum_r[0] << endl;
                    cout << dot_product_sum_i[0] << endl;
                }*/

                hermitian_matrix[i][j] = complex<float>(_mm256_cvtss_f32(dot_product_sum_r), _mm256_cvtss_f32(dot_product_sum_i));
            }
        }

        } else {
            /// Set the mask vector depending on the number of receive antennas.
           // __m256i mask = _mm256_setr_epi32(-1, -1, -1, -1, -1, 3, 3, 3);
        }

#else
        /// First line
        hermitian_matrix[0][0].real(channel_coefficients_[re][0][0].real() * channel_coefficients_[re][0][0].real() +
                                    channel_coefficients_[re][0][0].imag() * channel_coefficients_[re][0][0].imag());
        hermitian_matrix[0][1] = conj(channel_coefficients_[re][0][0]) * channel_coefficients_[re][0][1];
        hermitian_matrix[0][2] = conj(channel_coefficients_[re][0][0]) * channel_coefficients_[re][0][2];
        hermitian_matrix[0][3] = conj(channel_coefficients_[re][0][0]) * channel_coefficients_[re][0][3];

        /// Second line from diag coef 1,1
        hermitian_matrix[1][1].real(channel_coefficients_[re][0][1].real() * channel_coefficients_[re][0][1].real() +
                                            channel_coefficients_[re][0][1].imag() * channel_coefficients_[re][0][1].imag());
        hermitian_matrix[1][2] = conj(channel_coefficients_[re][0][1]) * channel_coefficients_[re][0][2];
        hermitian_matrix[1][3] = conj(channel_coefficients_[re][0][1]) * channel_coefficients_[re][0][3];

        /// Third line from diag coef 2,2
        hermitian_matrix[2][2].real(channel_coefficients_[re][0][2].real() * channel_coefficients_[re][0][2].real() +
                                            channel_coefficients_[re][0][2].imag() * channel_coefficients_[re][0][2].imag());
        hermitian_matrix[2][3] = conj(channel_coefficients_[re][0][2]) * channel_coefficients_[re][0][3];

        /// Fourth line from diag coef 3,3
        hermitian_matrix[3][3].real(channel_coefficients_[re][0][3].real() * channel_coefficients_[re][0][3].real() +
                                            channel_coefficients_[re][0][3].imag() * channel_coefficients_[re][0][3].imag());

        /// Compute hermitian matrix
        for(i = 1; i < nb_rx_ports_; i++) {
            hermitian_matrix[0][0] += channel_coefficients_[re][i][0].real() * channel_coefficients_[re][i][0].real() +
                    channel_coefficients_[re][i][0].imag() * channel_coefficients_[re][i][0].imag();
            hermitian_matrix[0][1] += conj(channel_coefficients_[re][i][0]) * channel_coefficients_[re][i][1];
            hermitian_matrix[0][2] += conj(channel_coefficients_[re][i][0]) * channel_coefficients_[re][i][2];
            hermitian_matrix[0][3] += conj(channel_coefficients_[re][i][0]) * channel_coefficients_[re][i][3];
            hermitian_matrix[1][1] += channel_coefficients_[re][i][1].real() * channel_coefficients_[re][i][1].real() +
                    channel_coefficients_[re][i][1].imag() * channel_coefficients_[re][i][1].imag();
            hermitian_matrix[1][2] += conj(channel_coefficients_[re][i][1]) * channel_coefficients_[re][i][2];
            hermitian_matrix[1][3] += conj(channel_coefficients_[re][i][1]) * channel_coefficients_[re][i][3];
            hermitian_matrix[2][2] += channel_coefficients_[re][i][2].real() * channel_coefficients_[re][i][2].real() +
                    channel_coefficients_[re][i][2].imag() * channel_coefficients_[re][i][2].imag();
            hermitian_matrix[2][3] += conj(channel_coefficients_[re][i][2]) * channel_coefficients_[re][i][3];
            hermitian_matrix[3][3] += channel_coefficients_[re][i][3].real() * channel_coefficients_[re][i][3].real() +
                    channel_coefficients_[re][i][3].imag() * channel_coefficients_[re][i][3].imag();
        }
#endif


#if TIME_MEASURE == 1
#if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
        << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
        t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
                                 << (t2 - t1)/TSC_FREQ * 1e9 << endl;

        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#endif
#endif

        ldl_decomp_test(hermitian_matrix, // row major
                        4);

#if TIME_MEASURE == 1
#if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();

                BOOST_LOG_TRIVIAL(trace) << "Time to compute LDL [ns] : " <<
                     std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

                t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute LDL [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute LDL [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute LDL [ns] : "
                                 << (t2 - t1)/TSC_FREQ * 1e9 << endl;

        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#endif
#endif

        diag_elements[0] = 1/hermitian_matrix[0][0].real();
        diag_elements[1] = 1/hermitian_matrix[1][1].real();
        diag_elements[2] = 1/hermitian_matrix[2][2].real();
        diag_elements[3] = 1/hermitian_matrix[3][3].real();

        /// Compute in-place inverse
        r_matrix[0][1]  = -hermitian_matrix[0][1];
        r_matrix[0][2]  = -hermitian_matrix[0][2];
        r_matrix[0][3]  = -hermitian_matrix[0][3];
        r_matrix[1][2]  = -hermitian_matrix[1][2];
        r_matrix[1][3]  = -hermitian_matrix[1][3];
        r_matrix[2][3]  = -hermitian_matrix[2][3];

        /// Compute coefficients in the correct order
        r_matrix[0][2] += r_matrix[0][1] * r_matrix[1][2];
        r_matrix[0][3] += r_matrix[0][1] * r_matrix[1][3] + r_matrix[0][2] * r_matrix[2][3];
        r_matrix[1][3] += r_matrix[1][2] * r_matrix[2][3];

        /// Copy inverse R^(-T) in lower part of the array
        r_transconj_matrix[1][0] = conj(r_matrix[0][1]);
        r_transconj_matrix[2][0] = conj(r_matrix[0][2]);
        r_transconj_matrix[2][1] = conj(r_matrix[1][2]);
        r_transconj_matrix[3][0] = conj(r_matrix[0][3]);
        r_transconj_matrix[3][1] = conj(r_matrix[1][3]);
        r_transconj_matrix[3][2] = conj(r_matrix[2][3]);

#if TIME_MEASURE == 1
#if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(trace) << "Time to compute the two inverse matrices [ns] : " <<
             std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
        t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute the two inverse matrices [ns] : "
                    << (t2.tv_nsec - t1.tv_nsec) << endl; ;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute the two inverse matrices [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl; ;
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute the two inverse matrices [ns] : "
                                 << (t2 - t1)/TSC_FREQ * 1e9 << endl;

        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#endif
#endif

/**
        /// Multiply received signal y by H^H, then multiply by the inverse
        for(i = 0; i < 4; i++) {
            vec1 = _mm256_loadu_ps((float *) channel_coefficients_[re][i]);
            vec1 = _mm256_mul_ps(vec1, neg); /// negate imag part
            vec2 = _mm256_loadu_ps((float *) pdsch_samples_[re]);
            /// Multiply vec1 by vec2
            vec3 = _mm256_mul_ps(vec1, vec2); /// Real part
            /// Negate imag part of vec2
            vec2 = _mm256_mul_ps(vec2, neg);
            /// Permute Real and imag part in vec4
            vec4 = _mm256_permute_ps(vec2, 0b10110001);
            /// Multiply vec1 by modified vec2
            vec5 = _mm256_mul_ps(vec1, vec4); /// Imag part
            /// Add elements together
            dot_product_sum_r = _mm256_add_ps(dot_product_sum_r, vec3);
            dot_product_sum_i = _mm256_add_ps(dot_product_sum_i, vec5);

            /// Add all elements in final dot_product_sum_r vector
            dot_product_sum_r = _mm256_hadd_ps(dot_product_sum_r, dot_product_sum_r);
            dot_product_sum_r = _mm256_hadd_ps(dot_product_sum_r, dot_product_sum_r);
            dot_product_sum_r = _mm256_add_ps(dot_product_sum_r, _mm256_permute2f128_ps(dot_product_sum_r, dot_product_sum_r, 1));

            /// Add all elements in final dot_product_sum_i vector
            dot_product_sum_i = _mm256_hadd_ps(dot_product_sum_i, dot_product_sum_i);
            dot_product_sum_i = _mm256_hadd_ps(dot_product_sum_i, dot_product_sum_i);
            dot_product_sum_i = _mm256_add_ps(dot_product_sum_i, _mm256_permute2f128_ps(dot_product_sum_i, dot_product_sum_i, 1));

            *(equalized_symbols_ + i) = complex<float>(dot_product_sum_r[0], dot_product_sum_i[0]);
        }
*/
        /**
        for(i = 0; i < 4; i++) {
            *(equalized_symbols_ + i) = conj(channel_coefficients_[re][0][0]) * pdsch_samples_[re][0];
            //*(temp_equalized_symbols + i) = conj(channel_coefficients_[0][i][re]) * pdsch_samples_[0][re];
        }

        for(i = 0; i < 4; i++) {
            for(j = 1; j < nb_rx_ports_; j++) {
                *(equalized_symbols_ + i) += conj(channel_coefficients_[re][i][j]) * pdsch_samples_[re][j];
                //*(temp_equalized_symbols + i) += conj(channel_coefficients_[j][i][re]) * pdsch_samples_[j][re];
            }
        } */

        /// Multiply by R^(-T), then by D^(-1) and then R^(-1)
        /**
        for(i = 3; i > 0; i--) {
            vec1 = _mm256_loadu_ps((float *) r_transconj_matrix[i]);
            vec1 = _mm256_mul_ps(vec1, neg); /// negate imag part
            vec2 = _mm256_loadu_ps((float *) equalized_symbols_ + i);
            /// Multiply vec1 by vec2
            vec3 = _mm256_mul_ps(vec1, vec2); /// Real part
            /// Negate imag part of vec2
            vec2 = _mm256_mul_ps(vec2, neg);
            /// Permute Real and imag part in vec4
            vec4 = _mm256_permute_ps(vec2, 0b10110001);
            /// Multiply vec1 by modified vec2
            vec5 = _mm256_mul_ps(vec1, vec4); /// Imag part
            /// Add elements together
            dot_product_sum_r = _mm256_add_ps(dot_product_sum_r, vec3);
            dot_product_sum_i = _mm256_add_ps(dot_product_sum_i, vec5);

            /// Add all elements in final dot_product_sum_r vector
            dot_product_sum_r = _mm256_hadd_ps(dot_product_sum_r, dot_product_sum_r);
            dot_product_sum_r = _mm256_hadd_ps(dot_product_sum_r, dot_product_sum_r);
            dot_product_sum_r = _mm256_add_ps(dot_product_sum_r, _mm256_permute2f128_ps(dot_product_sum_r, dot_product_sum_r, 1));

            /// Add all elements in final dot_product_sum_i vector
            dot_product_sum_i = _mm256_hadd_ps(dot_product_sum_i, dot_product_sum_i);
            dot_product_sum_i = _mm256_hadd_ps(dot_product_sum_i, dot_product_sum_i);
            dot_product_sum_i = _mm256_add_ps(dot_product_sum_i, _mm256_permute2f128_ps(dot_product_sum_i, dot_product_sum_i, 1));

            *(equalized_symbols_ + i) = complex<float>(dot_product_sum_r[0], dot_product_sum_i[0]);
        } */

        /**
        *(equalized_symbols_ + 3) += hermitian_matrix[3][0] * *(equalized_symbols_) + hermitian_matrix[3][1] * *(equalized_symbols_ + 1) + hermitian_matrix[3][2] * *(equalized_symbols_ + 2);
        *(equalized_symbols_ + 2) += hermitian_matrix[2][0] * *(equalized_symbols_ + 0) + hermitian_matrix[2][1] * *(equalized_symbols_ + 1);
        *(equalized_symbols_ + 1) += hermitian_matrix[1][0] * *(equalized_symbols_);

        vec1 = _mm256_loadu_ps((float *) equalized_symbols_);
        vec2 = _mm256_loadu_ps(diag_elements);
        vec3 = _mm256_div_ps(vec1, vec2);
        equalized_symbols_[0] = vec3[0];
        equalized_symbols_[1] = vec3[1];
        equalized_symbols_[2] = vec3[2];
        equalized_symbols_[3] = vec3[3]; */

        /**
        for(i = 0; i < 3; i++) {
            vec1 = _mm256_loadu_ps((float *) r_matrix[i]);
            vec1 = _mm256_mul_ps(vec1, neg); /// negate imag part
            vec2 = _mm256_loadu_ps((float *) equalized_symbols_ + i);
            /// Multiply vec1 by vec2
            vec3 = _mm256_mul_ps(vec1, vec2); /// Real part
            /// Negate imag part of vec2
            vec2 = _mm256_mul_ps(vec2, neg);
            /// Permute Real and imag part in vec4
            vec4 = _mm256_permute_ps(vec2, 0b10110001);
            /// Multiply vec1 by modified vec2
            vec5 = _mm256_mul_ps(vec1, vec4); /// Imag part
            /// Add elements together
            dot_product_sum_r = _mm256_add_ps(dot_product_sum_r, vec3);
            dot_product_sum_i = _mm256_add_ps(dot_product_sum_i, vec5);

            /// Add all elements in final dot_product_sum_r vector
            dot_product_sum_r = _mm256_hadd_ps(dot_product_sum_r, dot_product_sum_r);
            dot_product_sum_r = _mm256_hadd_ps(dot_product_sum_r, dot_product_sum_r);
            dot_product_sum_r = _mm256_add_ps(dot_product_sum_r, _mm256_permute2f128_ps(dot_product_sum_r, dot_product_sum_r, 1));

            /// Add all elements in final dot_product_sum_i vector
            dot_product_sum_i = _mm256_hadd_ps(dot_product_sum_i, dot_product_sum_i);
            dot_product_sum_i = _mm256_hadd_ps(dot_product_sum_i, dot_product_sum_i);
            dot_product_sum_i = _mm256_add_ps(dot_product_sum_i, _mm256_permute2f128_ps(dot_product_sum_i, dot_product_sum_i, 1));

            *(equalized_symbols_ + i) = complex<float>(dot_product_sum_r[0], dot_product_sum_i[0]);
        } */

        /**
        *(equalized_symbols_ + 3) += hermitian_matrix[3][0] * *(equalized_symbols_) + hermitian_matrix[3][1] * *(equalized_symbols_ + 1) + hermitian_matrix[3][2] * *(equalized_symbols_ + 2);
        *(equalized_symbols_ + 3) /= hermitian_matrix[3][3].real();
        *(equalized_symbols_ + 2) += hermitian_matrix[2][0] * *(equalized_symbols_ + 0) + hermitian_matrix[2][1] * *(equalized_symbols_ + 1);
        *(equalized_symbols_ + 2) /= hermitian_matrix[2][2].real();
        *(equalized_symbols_ + 1) += hermitian_matrix[1][0] * *(equalized_symbols_);
        *(equalized_symbols_ + 1) /= hermitian_matrix[1][1].real();
        *(equalized_symbols_) /= hermitian_matrix[0][0].real(); */

        *(equalized_symbols_) += hermitian_matrix[0][1]  * *(equalized_symbols_ + 1) + hermitian_matrix[0][2] * *(equalized_symbols_ + 2) + hermitian_matrix[0][3] * *(equalized_symbols_ + 3);
        *(equalized_symbols_ + 1) += hermitian_matrix[1][2]  * *(equalized_symbols_ + 2) + hermitian_matrix[1][3] * *(equalized_symbols_ + 3);
        *(equalized_symbols_ + 2) += hermitian_matrix[2][3] * *(equalized_symbols_ + 3);

        equalized_symbols_ += 4;

#if TIME_MEASURE == 1
#if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();
                BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : " <<
                     std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : "
                                 << (t2 - t1)/TSC_FREQ * 1e9 << endl;
#endif
#endif
    }
}

void vblast_zf_4_layers(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                        std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                        int num_re_pdsch_,
                        std::complex<float> * equalized_symbols_,
                        int nb_rx_ports_) {

    complex<float> hermitian_matrix[4][4];
    complex<float> temp_equalized_symbols[4];
    int i, j;
    int l;

#if TIME_MEASURE == 1
    #if defined(CLOCK_TYPE_CHRONO)
    std::chrono::steady_clock::time_point t1, t2;
#elif defined(CLOCK_TYPE_GETTIME)
    struct timespec t1, t2;
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
    struct timespec t1, t2;
#else
    uint64_t t1, t2;
    unsigned cycles_low1, cycles_low2, cycles_high1, cycles_high2;
#endif
#endif
    for(int re = 0; re < num_re_pdsch_; re++) {
#if TIME_MEASURE == 1
        BOOST_LOG_TRIVIAL(trace) << " ------------------------- " << endl;
        BOOST_LOG_TRIVIAL(trace) << "RE number : " << re << endl;
#if defined(CLOCK_TYPE_CHRONO)
        t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#endif
#endif

        /// First line
        hermitian_matrix[0][0].real(channel_coefficients_[0][0][re].real() * channel_coefficients_[0][0][re].real() +
                                    channel_coefficients_[0][0][re].imag() * channel_coefficients_[0][0][re].imag());
        hermitian_matrix[0][1] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][1][re];
        hermitian_matrix[0][2] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][2][re];
        hermitian_matrix[0][3] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[0][3][re];

        /// Second line from diag coef 1,1
        hermitian_matrix[1][1].real(channel_coefficients_[0][1][re].real() * channel_coefficients_[0][1][re].real() +
                                    channel_coefficients_[0][1][re].imag() * channel_coefficients_[0][1][re].imag());
        hermitian_matrix[1][2] = conj(channel_coefficients_[0][1][re]) * channel_coefficients_[0][2][re];
        hermitian_matrix[1][3] = conj(channel_coefficients_[0][1][re]) * channel_coefficients_[0][3][re];

        /// Third line from diag coef 2,2
        hermitian_matrix[2][2].real(channel_coefficients_[0][2][re].real() * channel_coefficients_[0][2][re].real() +
                                    channel_coefficients_[0][2][re].imag() * channel_coefficients_[0][2][re].imag());
        hermitian_matrix[2][3] = conj(channel_coefficients_[0][2][re]) * channel_coefficients_[0][3][re];

        /// Fourth line from diag coef 3,3
        hermitian_matrix[3][3].real(channel_coefficients_[0][3][re].real() * channel_coefficients_[0][3][re].real() +
                                    channel_coefficients_[0][3][re].imag() * channel_coefficients_[0][3][re].imag());

        /// Compute hermitian matrix
        for(i = 1; i < nb_rx_ports_; i++) {
            hermitian_matrix[0][0] += channel_coefficients_[i][0][re].real() * channel_coefficients_[i][0][re].real() +
                                      channel_coefficients_[i][0][re].imag() * channel_coefficients_[i][0][re].imag();
            hermitian_matrix[0][1] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][1][re];
            hermitian_matrix[0][2] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][2][re];
            hermitian_matrix[0][3] += conj(channel_coefficients_[i][0][re]) * channel_coefficients_[i][3][re];
            hermitian_matrix[1][1] += channel_coefficients_[i][1][re].real() * channel_coefficients_[i][1][re].real() +
                                      channel_coefficients_[i][1][re].imag() * channel_coefficients_[i][1][re].imag();
            hermitian_matrix[1][2] += conj(channel_coefficients_[i][1][re]) * channel_coefficients_[i][2][re];
            hermitian_matrix[1][3] += conj(channel_coefficients_[i][1][re]) * channel_coefficients_[i][3][re];
            hermitian_matrix[2][2] += channel_coefficients_[i][2][re].real() * channel_coefficients_[i][2][re].real() +
                                      channel_coefficients_[i][2][re].imag() * channel_coefficients_[i][2][re].imag();
            hermitian_matrix[2][3] += conj(channel_coefficients_[i][2][re]) * channel_coefficients_[i][3][re];
            hermitian_matrix[3][3] += channel_coefficients_[i][3][re].real() * channel_coefficients_[i][3][re].real() +
                                      channel_coefficients_[i][3][re].imag() * channel_coefficients_[i][3][re].imag();
        }

#if TIME_MEASURE == 1
        #if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
        << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
        t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian [ns] : "
        << (t2 - t1)/TSC_FREQ * 1e9 << endl;

        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#endif
#endif

        ldl_decomp_test(hermitian_matrix, // row major
                        4);

#if TIME_MEASURE == 1
        #if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();

                BOOST_LOG_TRIVIAL(trace) << "Time to compute LDL [ns] : " <<
                     std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

                t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute LDL [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute LDL [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute LDL [ns] : "
                                 << (t2 - t1)/TSC_FREQ * 1e9 << endl;

        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#endif
#endif

        /// Compute in-place inverse
        hermitian_matrix[0][1]  = -hermitian_matrix[0][1];
        hermitian_matrix[0][2]  = -hermitian_matrix[0][2];
        hermitian_matrix[0][3]  = -hermitian_matrix[0][3];
        hermitian_matrix[1][2]  = -hermitian_matrix[1][2];
        hermitian_matrix[1][3]  = -hermitian_matrix[1][3];
        hermitian_matrix[2][3]  = -hermitian_matrix[2][3];

        /// Compute coefficients in the correct order
        hermitian_matrix[0][2] += hermitian_matrix[0][1] * hermitian_matrix[1][2];
        hermitian_matrix[0][3] += hermitian_matrix[0][1] * hermitian_matrix[1][3] +
                                  hermitian_matrix[0][2] * hermitian_matrix[2][3];
        hermitian_matrix[1][3] += hermitian_matrix[1][2] * hermitian_matrix[2][3];

        /// Copy inverse R^(-1)^H in lower part of the array
        hermitian_matrix[1][0] = conj(hermitian_matrix[0][1]);
        hermitian_matrix[2][0] = conj(hermitian_matrix[0][2]);
        hermitian_matrix[2][1] = conj(hermitian_matrix[1][2]);
        hermitian_matrix[3][0] = conj(hermitian_matrix[0][3]);
        hermitian_matrix[3][1] = conj(hermitian_matrix[1][3]);
        hermitian_matrix[3][2] = conj(hermitian_matrix[2][3]);

#if TIME_MEASURE == 1
        #if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(trace) << "Time to compute the two inverse matrices [ns] : " <<
             std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
        t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute the two inverse matrices [ns] : "
                    << (t2.tv_nsec - t1.tv_nsec) << endl; ;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute the two inverse matrices [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl; ;
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to compute the two inverse matrices [ns] : "
                                 << (t2 - t1)/TSC_FREQ * 1e9 << endl;

        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#endif
#endif

        /// Multiply received signal y by H^H, then multiply by the inverse
        for(i = 0; i < 4; i++) {
            *(equalized_symbols_ + i) = conj(channel_coefficients_[0][i][re]) * pdsch_samples_[0][re];
        }
        for(j = 1; j < nb_rx_ports_; j++) {
            for(i = 0; i < 4; i++) {
                *(equalized_symbols_ + i) += conj(channel_coefficients_[j][i][re]) * pdsch_samples_[j][re];
            }
        }

        *(equalized_symbols_ + 3) += hermitian_matrix[3][0] * *(equalized_symbols_) +
                                     hermitian_matrix[3][1] * *(equalized_symbols_ + 1) +
                                     hermitian_matrix[3][2] * *(equalized_symbols_ + 2);
        *(equalized_symbols_ + 2) += hermitian_matrix[2][0] * *(equalized_symbols_) +
                                     hermitian_matrix[2][1] * *(equalized_symbols_ + 1);
        *(equalized_symbols_ + 1) += hermitian_matrix[1][0] * *(equalized_symbols_);
        *(equalized_symbols_)     /= hermitian_matrix[0][0].real();
        *(equalized_symbols_ + 1) /= hermitian_matrix[1][1].real();
        *(equalized_symbols_ + 2) /= hermitian_matrix[2][2].real();
        *(equalized_symbols_ + 3) /= hermitian_matrix[3][3].real();

        *(equalized_symbols_)     += hermitian_matrix[0][1] * *(equalized_symbols_ + 1) +
                                     hermitian_matrix[0][2] * *(equalized_symbols_ + 2) +
                                     hermitian_matrix[0][3] * *(equalized_symbols_ + 3);
        *(equalized_symbols_ + 1) += hermitian_matrix[1][2] * *(equalized_symbols_ + 2) +
                                     hermitian_matrix[1][3] * *(equalized_symbols_ + 3);
        *(equalized_symbols_ + 2) += hermitian_matrix[2][3] * *(equalized_symbols_ + 3);

        equalized_symbols_ += 4;

#if TIME_MEASURE == 1
        #if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();
                BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : " <<
                     std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
        clock_gettime(CLOCK_MONOTONIC, &t2);
        BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : "
                                 << (t2.tv_nsec - t1.tv_nsec) << endl;
        clock_gettime(CLOCK_MONOTONIC, &t1);
#else
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
        BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols [ns] : "
                                 << (t2 - t1)/TSC_FREQ * 1e9 << endl;
#endif
#endif

    }
}

void vblast_zf(const vector<vector<complex<float>>> &pdsch_samples_,
               complex<float> * channel_matrix_, // realigned
               complex<float> * hermitian_matrix_, // realigned
               int num_re_pdsch_,
               complex<float> * equalized_symbols_,
               int nb_tx_dmrs_ports_,
               int nb_rx_ports_) {

    /// Hardcode (H^h . H)^(-1)
    switch(nb_tx_dmrs_ports_) {
        case 2: {

            /// Compute the inverse of the channel matrix directly
            if (nb_rx_ports_ == 2) {

#if TIME_MEASURE == 1
                std::chrono::steady_clock::time_point  t1, t2;
#endif
                /**
                complex<float> *h00 = channel_coefficients_[0][0].data(); /// h_rx_tx
                complex<float> *h01 = channel_coefficients_[0][1].data();
                complex<float> *h10 = channel_coefficients_[1][0].data();
                complex<float> *h11 = channel_coefficients_[1][1].data(); */

                complex<float> det = 0;

                for (int re = 0; re < num_re_pdsch_; re++) {

#if TIME_MEASURE == 1
                    t1 = std::chrono::steady_clock::now();
#endif
                    /**
                    det = h00[re] * h11[re] - h10[re] * h01[re];

                    *(equalized_symbols_) = h11[re] * pdsch_samples_[0][re] - h01[re] *  pdsch_samples_[1][re];
                    *(equalized_symbols_ + 1) = -h10[re] * pdsch_samples_[0][re] + h00[re] * pdsch_samples_[1][re]; */

                    det = channel_matrix_[0] * channel_matrix_[3] - channel_matrix_[1] * channel_matrix_[2];

                    *(equalized_symbols_) =
                            channel_matrix_[3] * pdsch_samples_[0][re] - channel_matrix_[2] * pdsch_samples_[1][re];
                    *(equalized_symbols_ + 1) =
                            -channel_matrix_[1] * pdsch_samples_[0][re] + channel_matrix_[0] * pdsch_samples_[1][re];

                    *(equalized_symbols_) *= conj(det) / (1.0f * abs(det) * abs(det));
                    *(equalized_symbols_ + 1) *= conj(det) / (1.0f * abs(det) * abs(det));

                    //*(equalized_symbols_) /= det;
                    //*(equalized_symbols_ + 1) /= det;

                    /**
                    *(equalized_symbols_) *= conj(channel_coefficients_[0][0][re] * channel_coefficients_[1][1][re] - channel_coefficients_[1][0][re] * channel_coefficients_[0][1][re])/
                            (1.0f * abs(channel_coefficients_[0][0][re] * channel_coefficients_[1][1][re] - channel_coefficients_[1][0][re] * channel_coefficients_[0][1][re]) *
                            abs(channel_coefficients_[0][0][re] * channel_coefficients_[1][1][re] - channel_coefficients_[1][0][re] * channel_coefficients_[0][1][re]));
                    *(equalized_symbols_ + 1) *= conj(channel_coefficients_[0][0][re] * channel_coefficients_[1][1][re] - channel_coefficients_[1][0][re] * channel_coefficients_[0][1][re])/
                            (1.0f * abs(channel_coefficients_[0][0][re] * channel_coefficients_[1][1][re] - channel_coefficients_[1][0][re] * channel_coefficients_[0][1][re]) *
                            abs(channel_coefficients_[0][0][re] * channel_coefficients_[1][1][re] - channel_coefficients_[1][0][re] * channel_coefficients_[0][1][re])); */

                    //*(equalized_symbols_) /= channel_coefficients_[0][0][re] * channel_coefficients_[1][1][re] - channel_coefficients_[1][0][re] * channel_coefficients_[0][1][re];
                    //*(equalized_symbols_ + 1) /= channel_coefficients_[0][0][re] * channel_coefficients_[1][1][re] - channel_coefficients_[1][0][re] * channel_coefficients_[0][1][re];

                    equalized_symbols_ += 2;
                    /**
                    det = h00[re] * h11[re] - h10[re] * h01[re];

                    equalized_symbols_[2 * re] = h11[re] * pdsch_samples_[0][re] - h01[re] *  pdsch_samples_[1][re];
                    equalized_symbols_[2 * re + 1] = -h10[re] * pdsch_samples_[0][re] + h00[re] * pdsch_samples_[1][re];

                    equalized_symbols_[2 * re] *= conj(det)/(1.0f * abs(det) * abs(det));
                    equalized_symbols_[2 * re + 1] *= conj(det)/(1.0f * abs(det) * abs(det)); */

                    channel_matrix_ += nb_tx_dmrs_ports_ * nb_rx_ports_;
#if TIME_MEASURE == 1
                    t2 = std::chrono::steady_clock::now();
                    BOOST_LOG_TRIVIAL(trace) << "inversion on 1 RE : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
#endif
                }
            } else {
            }
            break;
        }

        case 3 :
        {

            /// Work on the channel matrix
            if(nb_rx_ports_ == 3) {
                complex<float> temp_coefs[3]; /// 3 upper elements of H^H.H
                float temp_matrix_float[3]; /// diagonal elements of H^H.H
                float det = 0;
                vector<complex<float>> temp_mrc(3);
                vector<complex<float>> temp_inv(6);

                for (int re = 0; re < num_re_pdsch_; re++) {

                    /**
                    cout << "h00 : " << h00[re] << endl;
                    cout << "h01 : " << h01[re] << endl;
                    cout << "h02 : " << h02[re] << endl;
                    cout << "h10 : " << h10[re] << endl;
                    cout << "h11 : " << h11[re] << endl;
                    cout << "h12 : " << h12[re] << endl;
                    cout << "h20 : " << h20[re] << endl;
                    cout << "h21 : " << h21[re] << endl;
                    cout << "h22 : " << h22[re] << endl;

                    */

                    temp_matrix_float[0] = abs(channel_matrix_[0]) + abs(channel_matrix_[1]) + abs(channel_matrix_[2]); /// h00
                    temp_matrix_float[1] = abs(channel_matrix_[3]) + abs(channel_matrix_[4]) + abs(channel_matrix_[5]); /// h11
                    temp_matrix_float[2] = abs(channel_matrix_[6]) + abs(channel_matrix_[7]) + abs(channel_matrix_[8]); /// h22

                    temp_coefs[0] = conj(channel_matrix_[0]) * channel_matrix_[3] + conj(channel_matrix_[1]) * channel_matrix_[4] + conj(channel_matrix_[6]) * channel_matrix_[5]; /// h01
                    temp_coefs[1] = conj(channel_matrix_[0]) * channel_matrix_[6] + conj(channel_matrix_[1]) * channel_matrix_[7] + conj(channel_matrix_[2]) * channel_matrix_[8]; /// h02
                    temp_coefs[2] = conj(channel_matrix_[3]) * channel_matrix_[6] + conj(channel_matrix_[4]) * channel_matrix_[7] + conj(channel_matrix_[5]) * channel_matrix_[8]; /// h12

                    /**
                    cout << "temp_matrix_float[0] : " << temp_matrix_float[0] << endl;
                    cout << "temp_matrix_float[1] : " << temp_matrix_float[1] << endl;
                    cout << "temp_matrix_float[2] : " << temp_matrix_float[2] << endl;
                    cout << "temp_coefs[0] : " << temp_coefs[0] << endl;
                    cout << "temp_coefs[1] : " << temp_coefs[1] << endl;
                    cout << "temp_coefs[2] : " << temp_coefs[2] << endl; */

                    det = temp_matrix_float[0] * temp_matrix_float[1] * temp_matrix_float[2] +
                          (temp_coefs[0] * temp_coefs[2] * conj(temp_coefs[1])).real() +
                          (temp_coefs[0] * temp_coefs[2] * conj(temp_coefs[1])).real()
                          - pow(abs(temp_coefs[1]), 2) * temp_matrix_float[1] -
                          pow(abs(temp_coefs[0]), 2) * temp_matrix_float[2] -
                          pow(abs(temp_coefs[2]), 2) * temp_matrix_float[0];

                    temp_mrc[0] = conj(channel_matrix_[0]) * pdsch_samples_[0][re] + conj(channel_matrix_[1]) * pdsch_samples_[1][re] +
                                  conj(channel_matrix_[2]) * pdsch_samples_[2][re];
                    temp_mrc[1] = conj(channel_matrix_[3]) * pdsch_samples_[0][re] + conj(channel_matrix_[4]) * pdsch_samples_[1][re] +
                                  conj(channel_matrix_[5]) * pdsch_samples_[2][re];
                    temp_mrc[2] = conj(channel_matrix_[6]) * pdsch_samples_[0][re] + conj(channel_matrix_[7]) * pdsch_samples_[1][re] +
                                  conj(channel_matrix_[8]) * pdsch_samples_[2][re];

                    /// Diagonal elements of inverse matrix
                    temp_inv[0] = temp_matrix_float[1] * temp_matrix_float[2] -
                                  abs(temp_coefs[2]) * abs(temp_coefs[2]); /// 00
                    temp_inv[1] = temp_matrix_float[0] * temp_matrix_float[2] -
                                  abs(temp_coefs[1]) * abs(temp_coefs[1]); /// 11
                    temp_inv[2] = temp_matrix_float[0] * temp_matrix_float[1] -
                                  abs(temp_coefs[0]) * abs(temp_coefs[0]); /// 22

                    temp_inv[3] = -(temp_coefs[0] * temp_matrix_float[2] - temp_coefs[1] * conj(temp_coefs[2])); /// 01
                    temp_inv[4] = temp_coefs[0] * temp_coefs[2] - temp_coefs[1] * temp_matrix_float[1]; /// 02
                    temp_inv[5] = -(temp_matrix_float[0] * temp_coefs[2] - temp_coefs[1] * conj(temp_coefs[0])); /// 12

                    equalized_symbols_[re * 3] = temp_inv[0].real() * temp_mrc[0] +
                                                 temp_inv[3] * temp_mrc[1] +
                                                 temp_inv[4] * temp_mrc[2];

                    equalized_symbols_[re * 3 + 1] = conj(temp_inv[3]) * temp_mrc[0] +
                                                     temp_inv[1].real() * temp_mrc[1] +
                                                     temp_inv[5] * temp_mrc[2];

                    equalized_symbols_[re * 3 + 2] = conj(temp_inv[4]) * temp_mrc[0] +
                                                     conj(temp_inv[5]) * temp_mrc[1] +
                                                     temp_inv[2].real() * temp_mrc[2];

                    equalized_symbols_[re * 3] /= det;
                    equalized_symbols_[re * 3 + 1] /= det;
                    equalized_symbols_[re * 3 + 2] /= det;

                    //cout << "det : " << det << endl;
                    /**
                    cout << "equalized symbol 0 : " << equalized_symbols_[re * 3] << endl;
                    cout << "equalized symbol 1 : " << equalized_symbols_[re * 3 + 1] << endl;
                    cout << "equalized symbol 2 : " << equalized_symbols_[re * 3 + 2] << endl;

                    cout << "00 : " << (temp_inv[0] * temp_matrix_float[0] + temp_inv[3] * conj(temp_coefs[0]) + temp_inv[4] * conj(temp_coefs[1]))/det << endl;
                    cout << "01 : " << (temp_inv[0] * temp_coefs[0]        + temp_inv[3] * temp_matrix_float[1]   + temp_inv[4] * conj(temp_coefs[2]))/det << endl;
                    cout << "02 : " << (temp_inv[0] * temp_coefs[1]        + temp_inv[3] * temp_coefs[2]          + temp_inv[4] * temp_matrix_float[2])/det << endl;

                    cout << "10 : " << (conj(temp_inv[3]) * temp_matrix_float[0] + temp_inv[1] * conj(temp_coefs[0]) + temp_inv[5] * conj(temp_coefs[1]))/det << endl;
                    cout << "11 : " << (conj(temp_inv[3]) * temp_coefs[0]        + temp_inv[1] * temp_matrix_float[1]   + temp_inv[5] * conj(temp_coefs[2]))/det << endl;
                    cout << "12 : " << (conj(temp_inv[3]) * temp_coefs[1]        + temp_inv[1] * temp_coefs[2]          + temp_inv[5] * temp_matrix_float[2])/det << endl;

                    cout << "20 : " << (conj(temp_inv[4]) * temp_matrix_float[0] + conj(temp_inv[5]) * conj(temp_coefs[0]) + temp_inv[2] * conj(temp_coefs[1]))/det << endl;
                    cout << "21 : " << (conj(temp_inv[4]) * temp_coefs[0]        + conj(temp_inv[5]) * temp_matrix_float[1]   + temp_inv[2] * conj(temp_coefs[2]))/det << endl;
                    cout << "22 : " << (conj(temp_inv[4]) * temp_coefs[1]        + conj(temp_inv[5]) * temp_coefs[2]          + temp_inv[2] * temp_matrix_float[2])/det << endl; */

                    channel_matrix_ += nb_tx_dmrs_ports_ * nb_rx_ports_;

                }
            } else {

            }

            break;
        }
        default : /// 4 TX and 4 RX for now
        {
            complex<float> hermitian_matrix[16]; /// Store row major
            //hermitian_matrix.reserve(16);
            //vector<complex<float>> conjugate_transpose(4 * nb_rx_ports_);
            //conjugate_transpose.reserve(nb_tx_dmrs_ports_ * nb_rx_ports_);
            complex<float> r_matrix[10];
            //r_matrix.reserve(10);
            complex<float> r_inverse[10];
            //r_inverse.reserve(10);
            complex<float> diag_matrix[4];
            //diag_matrix.reserve(4);
            complex<float> temp_symbols[4];
            //temp_symbols.reserve(4);
            complex<float> temp_symbols_2[4];
            //temp_symbols_2.reserve(4);
            int iter = 0;

#if TIME_MEASURE == 1
            std::chrono::steady_clock::time_point t1, t2;
#endif
            for(int re = 0; re < num_re_pdsch_; re++) {

#if TIME_MEASURE == 1
                t1 = std::chrono::steady_clock::now();
#endif

                /**
                /// First line
                hermitian_matrix[0] = hermitian_matrix_[0] * channel_matrix_[0];
                hermitian_matrix[1] = hermitian_matrix_[0] * channel_matrix_[4];
                hermitian_matrix[2] = hermitian_matrix_[0] * channel_matrix_[8];
                hermitian_matrix[3] = hermitian_matrix_[0] * channel_matrix_[12];

                /// Second line from diag coef 1,1
                hermitian_matrix[5] = hermitian_matrix_[4] * channel_matrix_[4];
                hermitian_matrix[6] = hermitian_matrix_[4] * channel_matrix_[8];
                hermitian_matrix[7] = hermitian_matrix_[4] * channel_matrix_[12 + 0];

                /// Third line from diag coef 2,2
                hermitian_matrix[10] = hermitian_matrix_[8] * channel_matrix_[8];
                hermitian_matrix[11] = hermitian_matrix_[8] * channel_matrix_[12];

                /// Fourth line from diag coef 3,3
                hermitian_matrix[15] = hermitian_matrix_[12] * channel_matrix_[12]; */

                /// First line
                hermitian_matrix[0] = pow(abs(channel_matrix_[0]), 2);
                hermitian_matrix[1] = conj(channel_matrix_[0]) * channel_matrix_[4];
                hermitian_matrix[2] = conj(channel_matrix_[0]) * channel_matrix_[8];
                hermitian_matrix[3] = conj(hermitian_matrix_[0])  * channel_matrix_[12];

                /// Second line from diag coef 1,1
                hermitian_matrix[5] = pow(abs(channel_matrix_[4]), 2);
                hermitian_matrix[6] = conj(channel_matrix_[4]) * channel_matrix_[8];
                hermitian_matrix[7] = conj(channel_matrix_[4]) * channel_matrix_[12 + 0];

                /// Third line from diag coef 2,2
                hermitian_matrix[10] = pow(abs(channel_matrix_[8]), 2);
                hermitian_matrix[11] = conj(channel_matrix_[8]) * channel_matrix_[12];

                /// Fourth line from diag coef 3,3
                hermitian_matrix[15] = pow(abs(channel_matrix_[12]), 2);

                /// Compute hermitian matrix
                for(iter = 1; iter < 4; iter++) {
                    /// First line
                    hermitian_matrix[0] += pow(abs(channel_matrix_[iter]), 2);
                //}
                //for(iter = 1; iter < 4; iter++) {
                    hermitian_matrix[1] += conj(channel_matrix_[iter]) * channel_matrix_[4 + iter];
                //}
                //for(iter = 1; iter < 4; iter++) {
                    hermitian_matrix[2] += conj(channel_matrix_[iter]) * channel_matrix_[8 + iter];
                //}
                //for(iter = 1; iter < 4; iter++) {
                    hermitian_matrix[3] += conj(channel_matrix_[iter]) * channel_matrix_[12 + iter];
                //}
                //for(iter = 1; iter < 4; iter++) {
                    /// Second line from diag coef 1,1
                    hermitian_matrix[5] += pow(abs(channel_matrix_[4 + iter]), 2);
                //}
                //for(iter = 1; iter < 4; iter++) {
                    hermitian_matrix[6] += conj(channel_matrix_[4 + iter]) * channel_matrix_[8 + iter];
                //}
                //for(iter = 1; iter < 4; iter++) {
                    hermitian_matrix[7] += conj(channel_matrix_[4 + iter]) * channel_matrix_[12 + iter];
                //}
                //for(iter = 1; iter < 4; iter++) {
                    /// Third line from diag coef 2,2
                    hermitian_matrix[10] += pow(abs(channel_matrix_[8 + iter]), 2);
                //}
                //for(iter = 1; iter < 4; iter++) {
                    hermitian_matrix[11] += conj(channel_matrix_[8 + iter]) * channel_matrix_[12 + iter];
                //}
                //for(iter = 1; iter < 4; iter++) {
                    /// Fourth line from diag coef 3,3
                    hermitian_matrix[15] += pow(abs(channel_matrix_[12 + iter]), 2);
                }

#if TIME_MEASURE == 1
                t2 = std::chrono::steady_clock::now();
                BOOST_LOG_TRIVIAL(trace) << "Time to compute hermitian : " <<
                     std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

                t1 = std::chrono::steady_clock::now();
#endif
                /// Load conjugate transpose, row major
                //for(int i = 0; i < nb_rx_ports_; i++) {
                //    for(int j = 0; j < 4; j++) {
                //        conjugate_transpose[j * nb_rx_ports_ + i] = conj(channel_coefficients_[i][j][re]);
                //    }
                //}

#if TIME_MEASURE == 1
                t2 = std::chrono::steady_clock::now();

                BOOST_LOG_TRIVIAL(trace) << "Time to Load conjugate transpose : " <<
                     std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

                t1 = std::chrono::steady_clock::now();
#endif
                /// Perform LDL decomposition
                /**
                ldl_decomp(hermitian_matrix.data(), // row major
                           r_matrix.data(),
                           diag_matrix.data(),
                           4);
                */

                /// Perform LDL decomposition
                /**
                ldl_decomp_harcoded(hermitian_matrix, // row major
                                    r_matrix,
                                    diag_matrix,
                                    4); */

                ldl_decomp_inplace(hermitian_matrix,
                                   4);

#if TIME_MEASURE  == 1
                t2 = std::chrono::steady_clock::now();

                BOOST_LOG_TRIVIAL(trace) << "Time to compute LDL : " <<
                std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
                t1 = std::chrono::steady_clock::now();

#endif
                /// Compute the inverse in-place
                /**
                for(int i = 1; i < 9; i ++) {
                    if ((i == 4) or (i == 7)) {
                        continue;
                    }
                    r_matrix[i] = -r_matrix[i];
                } */

                for(int i = 0; i < 4; i++) {
                    for(int j = i + 1; j < 4; j++) {
                        hermitian_matrix[i * 4 + j] = -hermitian_matrix[i * 4 + j];
                    }
                }

                /// Compute coefficients in the correct order
                /**
                r_matrix[2] += r_matrix[1] * r_matrix[5];
                r_matrix[3] += r_matrix[1] * r_matrix[6] + r_matrix[2] * r_matrix[8];
                r_matrix[6] += r_matrix[5] * r_matrix[8]; */

                hermitian_matrix[2] += hermitian_matrix[1] * hermitian_matrix[6];
                hermitian_matrix[3] += hermitian_matrix[1] * hermitian_matrix[7] + hermitian_matrix[2] * hermitian_matrix[11];
                hermitian_matrix[7] += hermitian_matrix[6] * hermitian_matrix[11];

                /// Copy into other array
                //memcpy(r_inverse, r_matrix, 10 * sizeof(complex<float>));

                /// Multiply by D^-1
                /**
                r_matrix[0] /= diag_matrix[0];
                r_matrix[1] /= diag_matrix[1];
                r_matrix[2] /= diag_matrix[2];
                r_matrix[3] /= diag_matrix[3];
                r_matrix[4] /= diag_matrix[1];
                r_matrix[5] /= diag_matrix[2];
                r_matrix[6] /= diag_matrix[3];
                r_matrix[7] /= diag_matrix[2];
                r_matrix[8] /= diag_matrix[3];
                r_matrix[9] /= diag_matrix[3]; */

                /// Compute the conjugate
                /**
                for(int i = 0; i < 10; i++) {
                    r_matrix[i] = conj(r_matrix[i]);
                } */

                for(int i = 0; i < 4; i++) {
                    for(int j = i + 1; j < 4; j++) {
                        hermitian_matrix[i * 4 + j].imag(-hermitian_matrix[i * 4 + j].imag());
                    }
                }

#if TIME_MEASURE == 1
                t2 = std::chrono::steady_clock::now();

                BOOST_LOG_TRIVIAL(trace) << "Time to compute the two inverse matrices : " <<
                std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
                t1 = std::chrono::steady_clock::now();

#endif

                /// Multiply received signal y by H^H, then multiply by the inverse
                //std::fill(temp_symbols.begin(), temp_symbols.end(), 0);
                //for(int i = 0; i < 4; i++) {
                //    temp_symbols[i] = conjugate_transpose[i * nb_rx_ports_] * pdsch_samples_[0][re];
                //}
                for(int i = 0; i < 4; i++) {
                    *(equalized_symbols_ + i) = conj(channel_matrix_[i]) * pdsch_samples_[0][re]; //hermitian_matrix_[i * 4] * pdsch_samples_[0][re]; //
                }

                for(int i = nb_rx_ports_; i < 4 * nb_rx_ports_; i += nb_rx_ports_) {
                    for(int j = 0; j < nb_rx_ports_; j++) {
                        *(equalized_symbols_ + i) += conj(channel_matrix_[i + j]) * pdsch_samples_[j][re]; //hermitian_matrix_[i * 4 + j] * pdsch_samples_[j][re]; //
                    }
                }

                /// Multiply by D^(-1) R^(*)^(-1)
                /**
                temp_symbols_2[3] = r_matrix[3] * temp_symbols[0] + r_matrix[6] * temp_symbols[1] + r_matrix[8] * temp_symbols[2] + r_matrix[9] * temp_symbols[3];
                temp_symbols_2[2] = r_matrix[2] * temp_symbols[0] + r_matrix[5] * temp_symbols[1] + r_matrix[7] * temp_symbols[2];
                temp_symbols_2[1] = r_matrix[1] * temp_symbols[0] + r_matrix[4] * temp_symbols[1];
                temp_symbols_2[0] = r_matrix[0] * temp_symbols[0]; */

                /**
                *(equalized_symbols_ + 3) = r_matrix[3] * *(equalized_symbols_) + r_matrix[6] * *(equalized_symbols_ + 1) + r_matrix[8] * *(equalized_symbols_ + 2) + r_matrix[9] * *(equalized_symbols_ + 3);
                *(equalized_symbols_ + 2) = r_matrix[2] * *(equalized_symbols_) + r_matrix[5] * *(equalized_symbols_ + 1) + r_matrix[7] * *(equalized_symbols_ + 2);
                *(equalized_symbols_ + 1) = r_matrix[1] * *(equalized_symbols_) + r_matrix[4] * *(equalized_symbols_ + 1);
                *(equalized_symbols_)     = r_matrix[0] * *(equalized_symbols_);

                *(equalized_symbols_)     /= diag_matrix[0];
                *(equalized_symbols_ + 1) /= diag_matrix[1];
                *(equalized_symbols_ + 2) /= diag_matrix[2];
                *(equalized_symbols_ + 3) /= diag_matrix[3]; */

                *(equalized_symbols_ + 3) += hermitian_matrix[3] * *(equalized_symbols_) +
                                             hermitian_matrix[7] * *(equalized_symbols_ + 1) +
                                             hermitian_matrix[11] * *(equalized_symbols_ + 2);
                *(equalized_symbols_ + 2) += hermitian_matrix[2] * *(equalized_symbols_) +
                                             hermitian_matrix[6] * *(equalized_symbols_ + 1);
                *(equalized_symbols_ + 1) += hermitian_matrix[1] * *(equalized_symbols_);

                *(equalized_symbols_)     /= hermitian_matrix[0];
                *(equalized_symbols_ + 1) /= hermitian_matrix[5];
                *(equalized_symbols_ + 2) /= hermitian_matrix[10];
                *(equalized_symbols_ + 3) /= hermitian_matrix[15];

                /// Multiply by R^(-1)
                /**
                 *(equalized_symbols_) = r_inverse[0] * temp_symbols_2[0] + r_inverse[1] * temp_symbols_2[1] + r_inverse[2] * temp_symbols_2[2] + r_inverse[3] * temp_symbols_2[3];
                 *(equalized_symbols_ + 1) = r_inverse[4] * temp_symbols_2[1] + r_inverse[5] * temp_symbols_2[2] + r_inverse[6] * temp_symbols_2[3];
                 *(equalized_symbols_ + 2) = r_inverse[7] * temp_symbols_2[2] + r_inverse[8] * temp_symbols_2[3];
                 *(equalized_symbols_ + 3) = r_inverse[9] * temp_symbols_2[3]; */

                /**
                //*(equalized_symbols_)     = r_inverse[0] * *(equalized_symbols_)     + r_inverse[1] * *(equalized_symbols_ + 1) + r_inverse[2] * *(equalized_symbols_ + 2) + r_inverse[3] * *(equalized_symbols_ + 3);
                 *(equalized_symbols_)     = *(equalized_symbols_) + r_inverse[1] * *(equalized_symbols_ + 1) + r_inverse[2] * *(equalized_symbols_ + 2) + r_inverse[3] * *(equalized_symbols_ + 3); // r_inverse[0] = 1
                //*(equalized_symbols_ + 1) = r_inverse[4] * *(equalized_symbols_ + 1) + r_inverse[5] * *(equalized_symbols_ + 2) + r_inverse[6] * *(equalized_symbols_ + 3);
                *(equalized_symbols_ + 1) = *(equalized_symbols_ + 1) + r_inverse[5] * *(equalized_symbols_ + 2) + r_inverse[6] * *(equalized_symbols_ + 3); // r_inverse[4] = 1
                 //*(equalized_symbols_ + 2) = r_inverse[7] * *(equalized_symbols_ + 2) + r_inverse[8] * *(equalized_symbols_ + 3);
                 *(equalized_symbols_ + 2)  += + r_inverse[8] * *(equalized_symbols_ + 3); // r_inverse[7] = 1;
                //*(equalized_symbols_ + 3) = r_inverse[9] * *(equalized_symbols_ + 3); // No need to compute because r_inverse[9] = 1 */

                *(equalized_symbols_)     += hermitian_matrix[4]   * *(equalized_symbols_ + 1) + hermitian_matrix[8] * *(equalized_symbols_ + 2) + hermitian_matrix[12] * *(equalized_symbols_ + 3);
                *(equalized_symbols_ + 1) += hermitian_matrix[9]   * *(equalized_symbols_ + 2) + hermitian_matrix[13] * *(equalized_symbols_ + 3);
                *(equalized_symbols_ + 2) +=  hermitian_matrix[14] * *(equalized_symbols_ + 3);

                equalized_symbols_ += 4;

                //std::fill(hermitian_matrix.begin(), hermitian_matrix.end(), 0);

                channel_matrix_ += 4 * nb_rx_ports_;
                hermitian_matrix_ += 4 * nb_rx_ports_;

#if TIME_MEASURE == 1
                t2 = std::chrono::steady_clock::now();

                BOOST_LOG_TRIVIAL(trace) << "Time to equalize the symbols : " <<
                     std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
#endif
            }
            break;
        }
    }
}

void ldl_decomp(complex<float> * hermitian_matrix_, /// Row major
                complex<float> * output_r_, /// Row major upper triangular matrix
                complex<float> * output_diag_, /// Diagonal coefficients
                int size_) {

    // i = 0.
    output_diag_[0] = hermitian_matrix_[0];

    output_r_[0] = 1;
    for(int i = 0; i < size_; i++) {
        output_r_[i] = hermitian_matrix_[i]/output_diag_[0];
    }

    int cum_sum_count[size_];
    for(int i = 0; i < size_; i++) {
        cum_sum_count[i] = i*(2 * size_ - i + 1)/2;
    }

    // Recurse on remaining coefs
    int count = 0;
    for(int i = 1; i < size_; i++) {
        count = cum_sum_count[i];
        output_diag_[i] = hermitian_matrix_[i];
        for(int j = 0; j < i; j++) {
             output_diag_[i] -= (output_r_[cum_sum_count[j] + i].real() * output_r_[cum_sum_count[j] + i].real() +
                                 output_r_[cum_sum_count[j] + i].imag() * output_r_[cum_sum_count[j] + i].imag())
                                         * output_diag_[j];
        }
        output_r_[count] = 1;
        //output_r_[i*(2 * size_ - i + 1)/2] = 1; // diag elements of R = 1
        for(int j = i+1; j < size_; j++) {
            //output_r_[i*(2 * size_ - i + 1)/2 + j] = hermitian_matrix_[i * size_ + j];
            output_r_[count + j] = hermitian_matrix_[i * size_ + j];
            for(int k = 0; k < i; k++) {
                output_r_[count + j] -= conj(output_r_[cum_sum_count[k] + i]) * output_r_[cum_sum_count[k] + j];
            }
            output_r_[count + j] /= output_diag_[i];
        }
    }
}

void ldl_decomp_harcoded(complex<float> * hermitian_matrix_, /// Row major
                         complex<float> * output_r_, /// Row major upper triangular matrix
                         complex<float> * output_diag_, /// Diagonal coefficients
                         int size_) {

    if(size_ == 4) {

        /// First row
        output_diag_[0] = hermitian_matrix_[0];
        output_r_[0] = 1;
        output_r_[1] = hermitian_matrix_[1]/output_diag_[0];
        output_r_[2] = hermitian_matrix_[2]/output_diag_[0];
        output_r_[3] = hermitian_matrix_[3]/output_diag_[0];

        /// Second row
        output_diag_[1] = hermitian_matrix_[5] - (output_r_[1].real() * output_r_[1].real() + output_r_[1].imag() * output_r_[1].imag()) * output_diag_[0];
        output_r_[4] = 1;
        output_r_[5] = (hermitian_matrix_[6] - conj(output_r_[1]) * output_r_[2] * output_diag_[0])/output_diag_[1];
        output_r_[6] = (hermitian_matrix_[7] - conj(output_r_[1]) * output_r_[3] * output_diag_[0])/output_diag_[1];

        /// Third row
        output_diag_[2] = hermitian_matrix_[10] - (output_r_[2].real() * output_r_[2].real() + output_r_[2].imag() * output_r_[2].imag()) * output_diag_[0]
                                                - (output_r_[5].real() * output_r_[5].real() + output_r_[5].imag() * output_r_[5].imag()) * output_diag_[1];
        output_r_[7] = 1;
        output_r_[8] = (hermitian_matrix_[11] - conj(output_r_[2]) * output_r_[3] * output_diag_[0] - conj(output_r_[5]) * output_r_[6] * output_diag_[1])/output_diag_[2];

        /// Fourth row
        output_diag_[3] = hermitian_matrix_[15] - (output_r_[3].real() * output_r_[3].real() + output_r_[3].imag() * output_r_[3].imag()) * output_diag_[0]
                                                - (output_r_[6].real() * output_r_[6].real() + output_r_[6].imag() * output_r_[6].imag()) * output_diag_[1]
                                                - (output_r_[8].real() * output_r_[8].real() + output_r_[8].imag() * output_r_[8].imag()) * output_diag_[2];
        output_r_[9] = 1;

    }
}

void ldl_decomp_inplace(complex<float> * hermitian_matrix_, /// Row major
                         int size_) {

    if(size_ == 4) {
        /// First row
        hermitian_matrix_[1] = hermitian_matrix_[1]/hermitian_matrix_[0].real();
        hermitian_matrix_[2] = hermitian_matrix_[2]/hermitian_matrix_[0].real();
        hermitian_matrix_[3] = hermitian_matrix_[3]/hermitian_matrix_[0].real();

        /// Second row
        hermitian_matrix_[5] = hermitian_matrix_[5].real() - (hermitian_matrix_[1].real() * hermitian_matrix_[1].real() + hermitian_matrix_[1].imag() * hermitian_matrix_[1].imag()) * hermitian_matrix_[0].real();
        hermitian_matrix_[6] = (hermitian_matrix_[6] - conj(hermitian_matrix_[1]) * hermitian_matrix_[2] * hermitian_matrix_[0].real())/hermitian_matrix_[5].real();
        hermitian_matrix_[7] = (hermitian_matrix_[7] - conj(hermitian_matrix_[1]) * hermitian_matrix_[3] * hermitian_matrix_[0].real())/hermitian_matrix_[5].real();

        /// Third row
        hermitian_matrix_[10] = hermitian_matrix_[10].real() - (hermitian_matrix_[2].real() * hermitian_matrix_[2].real() + hermitian_matrix_[2].imag() * hermitian_matrix_[2].imag()) * hermitian_matrix_[0].real()
                                - (hermitian_matrix_[6].real() * hermitian_matrix_[6].real() + hermitian_matrix_[6].imag() * hermitian_matrix_[6].imag()) * hermitian_matrix_[5].real();
        hermitian_matrix_[11] = (hermitian_matrix_[11] - conj(hermitian_matrix_[2]) * hermitian_matrix_[3] * hermitian_matrix_[0].real() - conj(hermitian_matrix_[6]) * hermitian_matrix_[7] * hermitian_matrix_[5].real())/hermitian_matrix_[10].real();

        /// Fourth row
        hermitian_matrix_[15] = hermitian_matrix_[15].real() - (hermitian_matrix_[3].real() * hermitian_matrix_[3].real() + hermitian_matrix_[3].imag() * hermitian_matrix_[3].imag()) * hermitian_matrix_[0].real()
                                - (hermitian_matrix_[7].real() * hermitian_matrix_[7].real() + hermitian_matrix_[7].imag() * hermitian_matrix_[7].imag()) * hermitian_matrix_[5].real()
                                - (hermitian_matrix_[11].real() * hermitian_matrix_[11].real() + hermitian_matrix_[11].imag() * hermitian_matrix_[11].imag()) * hermitian_matrix_[10].real();
    }

}

void vblast_mf(const vector<vector<complex<float>>> &pdsch_samples_,
               const vector<complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
               const int &num_re_pdsch_,
               complex<float> * equalized_symbols_,
               int &nb_tx_dmrs_ports_,
               int &nb_rx_ports_) {

    complex<float> temp_received_symbols[nb_tx_dmrs_ports_]; /// non equalized transmitted symbols after MF
    //int symbol = 0;
    //int sc = 0;
    float norms[nb_tx_dmrs_ports_]; /// column norms of the channel matrix

    for(int re = 0; re < num_re_pdsch_; re++) {
        std::fill(temp_received_symbols, temp_received_symbols + nb_tx_dmrs_ports_, 0);
        std::fill(norms, norms + nb_tx_dmrs_ports_, 0);
        for(int tx = 0; tx < nb_tx_dmrs_ports_; tx++) {

            //symbol = pdsch_positions_[2* re] - pdsch_start_symbol_;
            //sc = pdsch_positions_[2 * re + 1];

            /// Multiply received signal by the corresponding H^H column
            temp_received_symbols[tx] = conj(channel_coefficients_[0][tx][re]) * pdsch_samples_[0][re];
            for(int rx = 1; rx < nb_rx_ports_; rx++) {
                temp_received_symbols[tx] += conj(channel_coefficients_[rx][tx][re]) * pdsch_samples_[rx][re];
                /// Normalize by the squared column norm of the corresponding layer
                norms[tx] += pow(abs(channel_coefficients_[rx][tx][re]), 2);
            }
            temp_received_symbols[tx] /= norms[tx];

            /// Add coefficients to the equalized grid
            equalized_symbols_[re * nb_tx_dmrs_ports_ + tx] = temp_received_symbols[tx];
        }
    }
}

void mimo_vblast_decoder_qr_decomp(const complex<float> * channel_coefficients_,
                                   const vector<vector<complex<float>>> &pdsch_samples,
                                   const int * pdsch_positions_,
                                   int num_re_pdsch_,
                                   int pdsch_length_,
                                   int fft_size_,
                                   complex<float> * equalized_symbols_,
                                   int &nb_tx_dmrs_ports_,
                                   int &nb_rx_ports_,
                                   const int &pdsch_start_symbol_,
                                   complex<float> * constellation_symbols,
                                   int * detected_symbols_,
                                   const int &constellation_type_) {

    complex<float> q_h_symbols[nb_tx_dmrs_ports_];
    complex<float> q_matrix[nb_rx_ports_ * nb_tx_dmrs_ports_];
    complex<float> r_matrix[nb_tx_dmrs_ports_ * nb_tx_dmrs_ports_];
    complex<float> temp_received_symbols[nb_rx_ports_];
    float temp_squared_norms[nb_tx_dmrs_ports_];
    int detection_reversed_order[nb_tx_dmrs_ports_];
    int temp_detected_symbols_iteration[nb_tx_dmrs_ports_];

    //complex<float> test_channel_matrix[nb_rx_ports_ * nb_tx_dmrs_ports_];
    //complex<float> test_channel_matrix2[nb_tx_dmrs_ports_ * nb_tx_dmrs_ports_];

    for(int re = 0; re < num_re_pdsch_; re++) {

        /// Load channel coefficients in Q matrix
        for(int col = 0; col < nb_tx_dmrs_ports_; col++) {
            for(int row = 0; row < nb_rx_ports_; row++) {
                q_matrix[col * nb_rx_ports_ + row] = channel_coefficients_[row * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ // receiver
                                                                           + col * pdsch_length_ * fft_size_ // transmitter
                                                                           + (pdsch_positions_[2*re] - pdsch_start_symbol_) * fft_size_ // symbol
                                                                           + pdsch_positions_[2*re + 1] // subcarrier
                ];
                /**
                test_channel_matrix[col * nb_rx_ports_ + row] = channel_coefficients_[row * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ // receiver
                                                                + col * pdsch_length_ * fft_size_ // transmitter
                                                                + (pdsch_positions_[2*re] - pdsch_start_symbol_) * fft_size_ // symbol
                                                                + pdsch_positions_[2*re + 1] // subcarrier
                ];

                cout << "TX " << col << " RX" << row << test_channel_matrix[col * nb_rx_ports_ + row] << endl ; */
            }
        }

        /// Reset R to zero
        std::fill(r_matrix, r_matrix + nb_tx_dmrs_ports_ * nb_tx_dmrs_ports_, 0);

        /// Load squared norms
        //cout << "squared_norms" << endl;
        std::fill(temp_squared_norms, temp_squared_norms + nb_tx_dmrs_ports_, 0); /// reset values
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            for(int receiver = 0; receiver < nb_rx_ports_; receiver++) {
                temp_squared_norms[transmitter] += pow(abs(q_matrix[transmitter * nb_rx_ports_ + receiver]), 2); //q_matrix[transmitter * nb_rx_ports_ + receiver].real() * q_matrix[transmitter * nb_rx_ports_ + receiver].real()
                //+ q_matrix[transmitter * nb_rx_ports_ + receiver].imag() * q_matrix[transmitter * nb_rx_ports_ + receiver].imag();
            }
            //cout << temp_squared_norms[transmitter];
        }

        /**
        cout << "squared norms before QR decomp" << endl;
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
                cout << temp_squared_norms[transmitter] << endl;
        } */

        /// compute QR decomposition
        compute_qr_decomp(q_matrix,
                          r_matrix,
                          temp_squared_norms,
                          detection_reversed_order,
                          nb_tx_dmrs_ports_,
                          nb_tx_dmrs_ports_,
                          nb_rx_ports_,
                          nb_tx_dmrs_ports_);

        /**
        cout << "q_h * H" << endl;
        std::fill(test_channel_matrix2, test_channel_matrix2 + nb_rx_ports_ * nb_tx_dmrs_ports_, 0);
        for(int col = 0; col < nb_tx_dmrs_ports_; col++) {
            for(int row = 0; row < nb_tx_dmrs_ports_; row++) {
                for(int k = 0; k < nb_rx_ports_; k++) {
                    test_channel_matrix2[col * nb_rx_ports_ + row] +=
                            conj(q_matrix[detection_reversed_order[row] * nb_rx_ports_ + k]) *
                            test_channel_matrix[col * nb_rx_ports_ + k];
                }
            }
        }

        for(int col = 0; col < nb_tx_dmrs_ports_; col++) {
            for(int row = 0; row < nb_rx_ports_; row++) {
                cout << test_channel_matrix2[col * nb_rx_ports_ + row] << endl;
            }
        }

        cout << "detection order : " << endl;
        for(int k = 0; k < nb_tx_dmrs_ports_; k++) {
            cout << detection_reversed_order[k] << endl;
        }


        cout << "R matrix after QR decomp : " << endl;
        for(int col = 0; col < nb_tx_dmrs_ports_; col++) {
            for(int row = 0; row < nb_tx_dmrs_ports_; row++) {
                cout << r_matrix[col * nb_tx_dmrs_ports_ + row] << endl;
            }
        } */

        /// Suppress contribution of Q matrix in the received signal
        //t1 = std::chrono::steady_clock::now();
        //cout << "pdsch samples : " << endl;
        //for (int receiver = 0; receiver < nb_rx_ports_; receiver++) {
        //    temp_received_symbols[receiver] = ;
        //cout << pdsch_samples[receiver][re] << endl;
        //counter++;
        //cout << counter << endl;
        //}
        //t2 = std::chrono::steady_clock::now();
        //cout << " load received symbols : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

        std::fill(q_h_symbols, q_h_symbols + nb_tx_dmrs_ports_, 0);
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            for(int receiver = 0; receiver < nb_rx_ports_; receiver++) {
                q_h_symbols[transmitter] += conj(q_matrix[detection_reversed_order[transmitter] * nb_rx_ports_ // column
                                                          + receiver // row
                                                 ]) * pdsch_samples[receiver][re];
            }
        }

        /**
        cout << "q_h_symbols before SIC : " << endl;
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            cout << q_h_symbols[transmitter] << endl;
        } */

        /// SIC detector
        /// Decode the first symbol
        /// remove contribution of r(i,i)
        q_h_symbols[nb_tx_dmrs_ports_ - 1] /= r_matrix[detection_reversed_order[nb_tx_dmrs_ports_ - 1] * nb_tx_dmrs_ports_ + detection_reversed_order[nb_tx_dmrs_ports_ - 1]].real();

        /// Slicing
        ml_detector_complex[constellation_type_](q_h_symbols[nb_tx_dmrs_ports_ - 1], temp_detected_symbols_iteration[detection_reversed_order[nb_tx_dmrs_ports_ - 1]]);

        /// Remaining iterations
        for(int i = nb_tx_dmrs_ports_ - 2; i > - 1; i--) {

            /// remove contribution of previously detected symbols
            for(int k =  i + 1; k < nb_tx_dmrs_ports_; k++) {
                q_h_symbols[i] -= r_matrix[detection_reversed_order[k] * nb_tx_dmrs_ports_
                                           + detection_reversed_order[i]] * constellation_symbols[temp_detected_symbols_iteration[detection_reversed_order[k]]];
                //cout << "rij : " << r_matrix[detection_reversed_order[k] * nb_tx_dmrs_ports_
                //                 + detection_reversed_order[i]] << endl;
            }

            /// Equalization / remove contribution of r(i,i)
            q_h_symbols[i] /= r_matrix[detection_reversed_order[i] * nb_tx_dmrs_ports_ + detection_reversed_order[i]].real();
            //cout << "current tx antenna : " << detection_reversed_order[i] << endl;
            //cout << "rii" << r_matrix[detection_reversed_order[i] * nb_tx_dmrs_ports_ + detection_reversed_order[i]] << endl;

            /// Slicing
            ml_detector_complex[constellation_type_](q_h_symbols[i],
                                                     temp_detected_symbols_iteration[detection_reversed_order[i]]);
        }

        /// Add temp_detected symbols to detected_symbols buffer
        for(int i = 0; i < nb_tx_dmrs_ports_; i++) {
            equalized_symbols_[re * nb_tx_dmrs_ports_ + detection_reversed_order[i]] = q_h_symbols[i];
            detected_symbols_[re * nb_tx_dmrs_ports_ + i] = temp_detected_symbols_iteration[i];
            //cout << "symbol : " << detection_reversed_order[i] << endl;
            //cout << "q_h_symbols " << q_h_symbols[i] << endl;
            //cout << "equalized symbol : " << equalized_symbols_[re * nb_tx_dmrs_ports_ + detection_reversed_order[i]] << endl;
            //cout << "detected symbol : " << temp_detected_symbols_iteration[detection_reversed_order[i]] << endl;
            //cout << "re + i : " << re * nb_tx_dmrs_ports_ + detection_reversed_order[i] << endl;
        }
    }
}

void mimo_vblast_decoder_qr_decomp(const complex<float> * channel_coefficients_,
                                   const vector<vector<complex<float>>> &pdsch_samples,
                                   int num_re_pdsch_,
                                   complex<float> * equalized_symbols_,
                                   int &nb_tx_dmrs_ports_,
                                   int &nb_rx_ports_,
                                   complex<float> * constellation_symbols,
                                   int * detected_symbols_,
                                   const int &constellation_type_) {

    std::chrono::steady_clock::time_point  t1{}, t2{};

    t1 = std::chrono::steady_clock::now();
    complex<float> q_h_symbols[nb_tx_dmrs_ports_];
    complex<float> q_matrix[nb_rx_ports_ * nb_tx_dmrs_ports_];
    complex<float> r_matrix[nb_tx_dmrs_ports_ * nb_tx_dmrs_ports_];
    complex<float> temp_received_symbols[nb_rx_ports_];
    float temp_squared_norms[nb_tx_dmrs_ports_];
    int detection_reversed_order[nb_tx_dmrs_ports_];
    int temp_detected_symbols_iteration[nb_tx_dmrs_ports_];
    t2 = std::chrono::steady_clock::now();

    complex<float> test_channel_matrix[nb_rx_ports_ * nb_tx_dmrs_ports_];
    complex<float> test_channel_matrix2[nb_tx_dmrs_ports_ * nb_tx_dmrs_ports_];

    for(int re = 0; re < num_re_pdsch_; re++) {

        cout << "init. QR decomp local arrays : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

        t1 = std::chrono::steady_clock::now();
        /// Load channel coefficients in Q matrix
        for(int col = 0; col < nb_tx_dmrs_ports_; col++) {
            for(int row = 0; row < nb_rx_ports_; row++) {
                q_matrix[col * nb_rx_ports_ + row] = channel_coefficients_[row * nb_tx_dmrs_ports_ * num_re_pdsch_ // receiver
                                                                           + col * num_re_pdsch_ // transmitter
                                                                           + re];
                /**
                test_channel_matrix[col * nb_rx_ports_ + row] = channel_coefficients_[row * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ // receiver
                                                                + col * pdsch_length_ * fft_size_ // transmitter
                                                                + (pdsch_positions_[2*re] - pdsch_start_symbol_) * fft_size_ // symbol
                                                                + pdsch_positions_[2*re + 1] // subcarrier
                ];

                cout << "TX " << col << " RX" << row << test_channel_matrix[col * nb_rx_ports_ + row] << endl ; */
            }
        }
        t2 = std::chrono::steady_clock::now();
        cout << "initialize Q and R matrices : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

        /// Reset R to zero
        std::fill(r_matrix, r_matrix + nb_tx_dmrs_ports_ * nb_tx_dmrs_ports_, 0);

        /// Load squared norms
        //cout << "squared_norms" << endl;
        t1 = std::chrono::steady_clock::now();
        std::fill(temp_squared_norms, temp_squared_norms + nb_tx_dmrs_ports_, 0); /// reset values
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            for(int receiver = 0; receiver < nb_rx_ports_; receiver++) {
                temp_squared_norms[transmitter] += pow(abs(q_matrix[transmitter * nb_rx_ports_ + receiver]), 2); //q_matrix[transmitter * nb_rx_ports_ + receiver].real() * q_matrix[transmitter * nb_rx_ports_ + receiver].real()
                //+ q_matrix[transmitter * nb_rx_ports_ + receiver].imag() * q_matrix[transmitter * nb_rx_ports_ + receiver].imag();
            }
            //cout << temp_squared_norms[transmitter];
        }

        /**
        cout << "squared norms before QR decomp" << endl;
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
                cout << temp_squared_norms[transmitter] << endl;
        } */

        t2 = std::chrono::steady_clock::now();
        cout << "Load squared norms : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

        t1 = std::chrono::steady_clock::now();
        /// compute QR decomposition
        compute_qr_decomp(q_matrix,
                          r_matrix,
                          temp_squared_norms,
                          detection_reversed_order,
                          nb_tx_dmrs_ports_,
                          nb_tx_dmrs_ports_,
                          nb_rx_ports_,
                          nb_tx_dmrs_ports_);
        t2 = std::chrono::steady_clock::now();
        cout << "Compute QR decomp : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

        /**
        cout << "q_h * H" << endl;
        std::fill(test_channel_matrix2, test_channel_matrix2 + nb_rx_ports_ * nb_tx_dmrs_ports_, 0);
        for(int col = 0; col < nb_tx_dmrs_ports_; col++) {
            for(int row = 0; row < nb_tx_dmrs_ports_; row++) {
                for(int k = 0; k < nb_rx_ports_; k++) {
                    test_channel_matrix2[col * nb_rx_ports_ + row] +=
                            conj(q_matrix[detection_reversed_order[row] * nb_rx_ports_ + k]) *
                            test_channel_matrix[col * nb_rx_ports_ + k];
                }
            }
        }

        for(int col = 0; col < nb_tx_dmrs_ports_; col++) {
            for(int row = 0; row < nb_rx_ports_; row++) {
                cout << test_channel_matrix2[col * nb_rx_ports_ + row] << endl;
            }
        }

        cout << "detection order : " << endl;
        for(int k = 0; k < nb_tx_dmrs_ports_; k++) {
            cout << detection_reversed_order[k] << endl;
        }


        cout << "R matrix after QR decomp : " << endl;
        for(int col = 0; col < nb_tx_dmrs_ports_; col++) {
            for(int row = 0; row < nb_tx_dmrs_ports_; row++) {
                cout << r_matrix[col * nb_tx_dmrs_ports_ + row] << endl;
            }
        } */

        /// Suppress contribution of Q matrix in the received signal
        //t1 = std::chrono::steady_clock::now();
        //cout << "pdsch samples : " << endl;
        //for (int receiver = 0; receiver < nb_rx_ports_; receiver++) {
        //    temp_received_symbols[receiver] = ;
        //cout << pdsch_samples[receiver][re] << endl;
        //counter++;
        //cout << counter << endl;
        //}
        //t2 = std::chrono::steady_clock::now();
        //cout << " load received symbols : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

        t1 = std::chrono::steady_clock::now();
        std::fill(q_h_symbols, q_h_symbols + nb_tx_dmrs_ports_, 0);
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            for(int receiver = 0; receiver < nb_rx_ports_; receiver++) {
                q_h_symbols[transmitter] += conj(q_matrix[detection_reversed_order[transmitter] * nb_rx_ports_ // column
                                                          + receiver // row
                                                 ]) * pdsch_samples[receiver][re];
            }
        }

        /**
        cout << "q_h_symbols before SIC : " << endl;
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            cout << q_h_symbols[transmitter] << endl;
        } */

        t2 = std::chrono::steady_clock::now();
        cout << "multiply qH by y : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

        /// SIC detector
        /// Decode the first symbol
        /// remove contribution of r(i,i)
        t1 = std::chrono::steady_clock::now();
        q_h_symbols[nb_tx_dmrs_ports_ - 1] /= r_matrix[detection_reversed_order[nb_tx_dmrs_ports_ - 1] * nb_tx_dmrs_ports_ + detection_reversed_order[nb_tx_dmrs_ports_ - 1]].real();

        /// Slicing
        ml_detector_complex[constellation_type_](q_h_symbols[nb_tx_dmrs_ports_ - 1], temp_detected_symbols_iteration[detection_reversed_order[nb_tx_dmrs_ports_ - 1]]);

        /// Remaining iterations
        for(int i = nb_tx_dmrs_ports_ - 2; i > - 1; i--) {

            /// remove contribution of previously detected symbols
            for(int k =  i + 1; k < nb_tx_dmrs_ports_; k++) {
                q_h_symbols[i] -= r_matrix[detection_reversed_order[k] * nb_tx_dmrs_ports_
                                           + detection_reversed_order[i]] * constellation_symbols[temp_detected_symbols_iteration[detection_reversed_order[k]]];
                //cout << "rij : " << r_matrix[detection_reversed_order[k] * nb_tx_dmrs_ports_
                //                 + detection_reversed_order[i]] << endl;
            }

            /// Equalization / remove contribution of r(i,i)
            q_h_symbols[i] /= r_matrix[detection_reversed_order[i] * nb_tx_dmrs_ports_ + detection_reversed_order[i]].real();
            //cout << "current tx antenna : " << detection_reversed_order[i] << endl;
            //cout << "rii" << r_matrix[detection_reversed_order[i] * nb_tx_dmrs_ports_ + detection_reversed_order[i]] << endl;

            /// Slicing
            ml_detector_complex[constellation_type_](q_h_symbols[i],
                                                     temp_detected_symbols_iteration[detection_reversed_order[i]]);
        }

        /// Add temp_detected symbols to detected_symbols buffer
        for(int i = 0; i < nb_tx_dmrs_ports_; i++) {
            equalized_symbols_[re * nb_tx_dmrs_ports_ + detection_reversed_order[i]] = q_h_symbols[i];
            detected_symbols_[re * nb_tx_dmrs_ports_ + i] = temp_detected_symbols_iteration[i];
            //cout << "symbol : " << detection_reversed_order[i] << endl;
            //cout << "q_h_symbols " << q_h_symbols[i] << endl;
            //cout << "equalized symbol : " << equalized_symbols_[re * nb_tx_dmrs_ports_ + detection_reversed_order[i]] << endl;
            //cout << "detected symbol : " << temp_detected_symbols_iteration[detection_reversed_order[i]] << endl;
            //cout << "re + i : " << re * nb_tx_dmrs_ports_ + detection_reversed_order[i] << endl;
        }
        t2 = std::chrono::steady_clock::now();
        cout << "SIC detector : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << "\n" << endl;
    }
}

void mimo_vblast_decoder_load_channel_coefficients_in_q(const vector<complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                                                        complex<float> q_matrix[][MAX_TX_PORTS][MAX_RX_PORTS],
                                                        int num_re_pdsch_,
                                                        int nb_tx_dmrs_ports_,
                                                        int nb_rx_ports_) {
    int i, j;
    for(int re = 0; re < num_re_pdsch_; re++) {
        /// Load channel coefficients in Q matrix
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 0; j < nb_rx_ports_; j++) {
                q_matrix[re][i][j] = channel_coefficients_[j][i][re];
            }
        }
    }
}

void mimo_vblast_decoder_compute_qr_decomp(std::complex<float> r_matrix[][MAX_TX_PORTS][MAX_TX_PORTS],
                                           std::complex<float> q_matrix[][MAX_TX_PORTS][MAX_RX_PORTS],
                                           //std::vector<std::complex<float>> channel_coefficients[MAX_TX_PORTS][MAX_RX_PORTS],
                                           int detection_reversed_orders[][MAX_TX_PORTS],
                                           const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                           int num_re_pdsch_,
                                           int nb_tx_dmrs_ports_,
                                           int nb_rx_ports_) {

    int i, j;
    float temp_squared_norms[MAX_TX_PORTS];
    for(int re = 0; re < num_re_pdsch_; re++) {
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            temp_squared_norms[i] = q_matrix[re][i][0].real() * q_matrix[re][i][0].real() +
                                              q_matrix[re][i][0].imag() * q_matrix[re][i][0].imag();
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(int j = 1; j < nb_rx_ports_; j++) {
                temp_squared_norms[i] += q_matrix[re][i][j].real() * q_matrix[re][i][j].real() +
                                                   q_matrix[re][i][j].imag() * q_matrix[re][i][j].imag();
            }
        }

        /*
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            temp_squared_norms[i] = channel_coefficients[i][0][re].real() * channel_coefficients[i][0][re].real() +
                                              channel_coefficients[i][0][re].imag() * channel_coefficients[i][0][re].imag();
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(int j = 1; j < nb_rx_ports_; j++) {
                temp_squared_norms[i] += channel_coefficients[i][j][re].real() * channel_coefficients[i][j][re].real() +
                                                   channel_coefficients[i][j][re].imag() * channel_coefficients[i][j][re].imag();
            }
        } */

        compute_qr_decomp(q_matrix[re],
                         r_matrix[re],
                         temp_squared_norms,
                         detection_reversed_orders[re],
                         nb_rx_ports_,
                         nb_tx_dmrs_ports_);
    }
}

void mimo_vblast_decoder_compute_qr_decomp(complex<float> r_matrix[][MAX_TX_PORTS][MAX_TX_PORTS],
                                           vector<complex<float>> channel_coefficients[MAX_TX_PORTS][MAX_RX_PORTS],
                                           int detection_reversed_orders[][MAX_TX_PORTS],
                                           const vector<vector<complex<float>>> &pdsch_samples,
                                           int num_re_pdsch_,
                                           int nb_tx_dmrs_ports_,
                                           int nb_rx_ports_) {

    int i, j;
    float temp_squared_norms[MAX_TX_PORTS];
    for(int re = 0; re < num_re_pdsch_; re++) {
        /*
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            temp_squared_norms[i] = q_matrix[re][i][0].real() * q_matrix[re][i][0].real() +
                                              q_matrix[re][i][0].imag() * q_matrix[re][i][0].imag();
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(int j = 1; j < nb_rx_ports_; j++) {
                temp_squared_norms[i] += q_matrix[re][i][j].real() * q_matrix[re][i][j].real() +
                                                   q_matrix[re][i][j].imag() * q_matrix[re][i][j].imag();
            }
        } */

        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            temp_squared_norms[i] = channel_coefficients[i][0][re].real() * channel_coefficients[i][0][re].real() +
                                              channel_coefficients[i][0][re].imag() * channel_coefficients[i][0][re].imag();
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(int j = 1; j < nb_rx_ports_; j++) {
                temp_squared_norms[i] += channel_coefficients[i][j][re].real() * channel_coefficients[i][j][re].real() +
                                                   channel_coefficients[i][j][re].imag() * channel_coefficients[i][j][re].imag();
            }
        }

        int computed[MAX_TX_PORTS];
        memset(&computed, 0, MAX_TX_PORTS * sizeof(int));
        int argmin = 0;
        bool first = 1;
        float temp_squared_norm = 0;

        for(int i = 0; i < nb_tx_dmrs_ports_; i++) {

            /// Find the column with lowest norm
            if(i < nb_tx_dmrs_ports_ - 1) { /// Stop if the last column has been computed
                for(int j = 0; j < nb_tx_dmrs_ports_; j++) {
                    if(not computed[j]) {
                        if (first) {
                            argmin = j;
                            temp_squared_norm = temp_squared_norms[j];
                            first = 0;
                            continue;
                        } else {
                            if (temp_squared_norms[j] < temp_squared_norm) {
                                argmin = j;
                                temp_squared_norm = temp_squared_norms[j];
                            }
                        }
                    }
                }

                computed[argmin] = 1; /// indicate that the current column of Q must not be modified in the next iterations
                first = 1;

                } else {
                    for(int j = 0; j < nb_tx_dmrs_ports_; j++) {
                        if(not computed[j]) {
                            argmin = j;
                        }
                    }
                }

            detection_reversed_orders[re][i] = argmin;

            /// Compute diagonal r(argmin, argmin) coefficient
            r_matrix[re][argmin][argmin] = sqrt(temp_squared_norms[argmin]);

            /// Normalize column q_argmin
            for(int j = 0; j < nb_tx_dmrs_ports_; j++) {
                channel_coefficients[argmin][j][re] /= r_matrix[re][argmin][argmin].real();
            }

            /// Project other columns vectors onto q_argmin and orthogonalize.
            if(i < nb_tx_dmrs_ports_ - 1) { /// Stop if the last column has been computed
                for(int j = 0; j < nb_tx_dmrs_ports_; j++) {
                    if(not computed[j]) {
                        //r_matrix[i][count_r_col] = conj(q_matrix[argmin][0]) * q_matrix[j][0];
                        r_matrix[re][argmin][j] = conj(channel_coefficients[argmin][0][re]) * channel_coefficients[j][0][re];
                        for(int k = 1; k < nb_rx_ports_; k++) {
                            //r_matrix[i][count_r_col] += conj(q_matrix[argmin][k]) * q_matrix[j][k];
                            r_matrix[re][argmin][j] += conj(channel_coefficients[argmin][k][re]) * channel_coefficients[j][k][re];
                        }
                        for(int k = 0; k < nb_rx_ports_; k++) {
                            //q_matrix[j][k] = q_matrix[j][k] - r_matrix[i][count_r_col] * q_matrix[argmin][k];
                            channel_coefficients[j][k][re] = channel_coefficients[j][k][re] - r_matrix[re][argmin][j] * channel_coefficients[argmin][k][re];
                        }

                        /*
                        squared_norms_[j] -= (r_matrix[i][count_r_col].real() * r_matrix[i][count_r_col].real() +
                                              r_matrix[i][count_r_col].imag() * r_matrix[i][count_r_col].imag());
                        count_r_col++; */

                        temp_squared_norms[j] -= (r_matrix[re][argmin][j].real() * r_matrix[re][argmin][j].real() +
                                              r_matrix[re][argmin][j].imag() * r_matrix[re][argmin][j].imag());

                    }
                }
            }
        }
    }
}

void mimo_vblast_decoder_qr_decomp_multiply_by_q_matrix(const vector<vector<complex<float>>> &pdsch_samples,
                                                        std::complex<float> q_matrix[][MAX_RX_PORTS][MAX_TX_PORTS],
                                                        int num_re_pdsch_,
                                                        complex<float> * equalized_symbols_,
                                                        int nb_tx_dmrs_ports_,
                                                        int nb_rx_ports_) {

    int i, j;
    for(int re = 0; re < num_re_pdsch_; re++) {
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            *(equalized_symbols_  + i) = conj(q_matrix[re][i][0]) * pdsch_samples[0][re];
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 1; j < nb_rx_ports_; j++) {
                *(equalized_symbols_  + i) += conj(q_matrix[re][i][j]) * pdsch_samples[j][re];
            }
        }
        equalized_symbols_ += 4;
    }
}

void mimo_vblast_decoder_qr_decomp_multiply_by_q_matrix(const vector<vector<complex<float>>> &pdsch_samples,
                                                        vector<complex<float>> pdsch_channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                                        int num_re_pdsch_,
                                                        complex<float> * equalized_symbols_,
                                                        int nb_tx_dmrs_ports_,
                                                        int nb_rx_ports_) {

    int i, j;
    for(int re = 0; re < num_re_pdsch_; re++) {
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            *(equalized_symbols_  + i) = conj(pdsch_channel_coefficients_[i][0][re]) * pdsch_samples[0][re];
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 1; j < nb_rx_ports_; j++) {
                *(equalized_symbols_  + i) += conj(pdsch_channel_coefficients_[i][j][re]) * pdsch_samples[j][re];
            }
        }
        equalized_symbols_ += 4;
    }
}

void mimo_vblast_decoder_sic_detection(complex<float> r_matrix[][MAX_TX_PORTS][MAX_TX_PORTS],
                                       int detection_reversed_order[][MAX_TX_PORTS],
                                       complex<float> * equalized_symbols_,
                                       int * detected_symbols_,
                                       complex<float> * constellation_symbols,
                                       int constellation_type_,
                                       int num_re_pdsch_,
                                       int nb_tx_dmrs_ports_) {

    int i, k;
    for(int re = 0; re < num_re_pdsch_; re++) {
        *(equalized_symbols_ + detection_reversed_order[re][nb_tx_dmrs_ports_ - 1]) /=
                 r_matrix[re][detection_reversed_order[re][nb_tx_dmrs_ports_ - 1]][detection_reversed_order[re][nb_tx_dmrs_ports_ - 1]].real();

        /// Slicing
        ml_detector_complex[constellation_type_](*(equalized_symbols_ + detection_reversed_order[re][nb_tx_dmrs_ports_ - 1]),
                                                 *(detected_symbols_  + detection_reversed_order[re][nb_tx_dmrs_ports_ - 1]));

        /// Remaining iterations
        for(i = nb_tx_dmrs_ports_ - 2; i > - 1; i--) {

            /// remove contribution of previously detected symbols
            for(k =  i + 1; k < nb_tx_dmrs_ports_; k++) {
                *(equalized_symbols_  + detection_reversed_order[re][i])  -=
                        r_matrix[re][detection_reversed_order[re][i]][detection_reversed_order[re][k]] * constellation_symbols[*(detected_symbols_ + detection_reversed_order[re][k])];
            }

            /// Equalization / remove contribution of r(i,i)
            *(equalized_symbols_  + detection_reversed_order[re][i]) /= r_matrix[re][detection_reversed_order[re][i]][detection_reversed_order[re][i]].real();

            /// Slicing
            ml_detector_complex[constellation_type_](*(equalized_symbols_ + detection_reversed_order[re][i]),
                                                     *(detected_symbols_  + detection_reversed_order[re][i]));

        }
        equalized_symbols_ += nb_tx_dmrs_ports_;
        detected_symbols_ += nb_tx_dmrs_ports_;
    }
}

#if defined(__AVX512__)
void mimo_vblast_sqrd_avx512(vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                           const vector<vector<complex<float>>> &pdsch_samples,
                           int num_re_pdsch_,
                           complex<float> * equalized_symbols_,
                           int nb_tx_dmrs_ports_,
                           int nb_rx_ports_,
                           complex<float> * constellation_symbols,
                           int * detected_symbols_,
                           int constellation_type_) {

    __m512 conj_vec = {1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1};

    __m512 r_matrix[MAX_TX_PORTS][MAX_TX_PORTS];
    __m512 q_matrix_transposed[MAX_TX_PORTS][MAX_RX_PORTS];
    __m512 squared_norms[MAX_TX_PORTS];
    __m512 temp_equalized_symbols[MAX_TX_PORTS];
    __m256i temp_detected_symbol_indexes[MAX_TX_PORTS];
    __m512 temp_detected_symbols[MAX_TX_PORTS];

    __m512 dot_prod_re, dot_prod_im, vec1, vec2;

    int detection_reversed_order[MAX_TX_PORTS];
    int i, j;

    int current_symbol;

    for(int re = 0; re < num_re_pdsch_; re+= 8) {

        /// Load channel coefs in q_matrix and compute the squared norms
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            /// Compute the square norm of the column
            q_matrix_transposed[i][0] = _mm512_loadu_ps((float *) &channel_coefficients_[i][0][re]);
            vec1 = _mm512_mul_ps(q_matrix_transposed[i][0], q_matrix_transposed[i][0]);
            squared_norms[i] = _mm512_add_ps(vec1, _mm512_permute_ps(vec1, 0b10110001));
            for(j = 1; j < nb_rx_ports_; j++) {
                q_matrix_transposed[i][j] = _mm512_loadu_ps((float *) &channel_coefficients_[i][j][re]);
                vec1 = _mm512_mul_ps(q_matrix_transposed[i][j], q_matrix_transposed[i][j]);
                squared_norms[i] = _mm512_add_ps(squared_norms[i], _mm512_add_ps(vec1, _mm512_permute_ps(vec1, 0b10110001)));
            }
        }

        /// Compute QR decomp
        compute_qr_decomp(q_matrix_transposed,
                          r_matrix,
                          squared_norms,
                          detection_reversed_order,
                          nb_tx_dmrs_ports_,
                          nb_rx_ports_);

        /// Multiply received vector by transconj(Q)
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            dot_prod_re = _mm512_set1_ps(0);
            dot_prod_im = _mm512_set1_ps(0);
            for (j = 0; j < nb_rx_ports_; j++) {
                vec2 = _mm512_loadu_ps((float *) &pdsch_samples[j][re]);
                dot_prod_re = _mm512_add_ps(dot_prod_re, _mm512_mul_ps(q_matrix_transposed[i][j], vec2));
                dot_prod_im = _mm512_add_ps(dot_prod_im, _mm512_mul_ps(_mm512_permute_ps(_mm512_mul_ps(q_matrix_transposed[i][j], conj_vec), 0b10110001), vec2));
            }
            dot_prod_re = _mm512_add_ps(dot_prod_re, _mm512_permute_ps(dot_prod_re, 0b10110001));
            dot_prod_im = _mm512_add_ps(dot_prod_im, _mm512_permute_ps(dot_prod_im, 0b10110001));

            temp_equalized_symbols[i] = _mm512_permute_ps(_mm512_shuffle_ps(dot_prod_re, dot_prod_im, 0b10001000), 0b11011000);
        }

        /// Perform SIC
        for(i = nb_tx_dmrs_ports_ - 1; i > -1; i--) {
            current_symbol = detection_reversed_order[i];

            /// Suppress interference
            for(j = i + 1; j < nb_tx_dmrs_ports_; j++) {
                vec1 = _mm512_mul_ps(r_matrix[current_symbol][detection_reversed_order[j]], temp_detected_symbols[j]);
                vec2 = _mm512_mul_ps(r_matrix[current_symbol][detection_reversed_order[j]], _mm512_permute_ps(temp_detected_symbols[j], 0b10110001));
                vec1 = _mm512_sub_ps(vec1, _mm512_permute_ps(vec1, 0b10110001));
                vec2 = _mm512_add_ps(vec2, _mm512_permute_ps(vec2, 0b10110001));
                /*
                temp_equalized_symbols[current_symbol] = _mm512_sub_ps(temp_equalized_symbols[current_symbol],
                                                                       multiply_complex_float(r_matrix[current_symbol][detection_reversed_order[j]],
                                                                                              temp_detected_symbols[j])); */
                temp_equalized_symbols[current_symbol] = _mm512_sub_ps(temp_equalized_symbols[current_symbol],
                                                                       _mm512_permute_ps(_mm512_shuffle_ps(vec1, vec2, 0b10001000), 0b11011000));
            }

            /// Divide by diagonal coef in R
            temp_equalized_symbols[current_symbol] = _mm512_div_ps(temp_equalized_symbols[current_symbol], r_matrix[current_symbol][current_symbol]);

            /// Slicing
            ml_detector_mm512(temp_equalized_symbols[current_symbol],
                              temp_detected_symbol_indexes[i],
                              temp_detected_symbols[i],
                              constellation_type_);

            /// Store in final buffer
            for(j = 0; j < 8; j++) {
                (equalized_symbols_ + j * nb_tx_dmrs_ports_ + current_symbol)->real(temp_equalized_symbols[current_symbol][2 * j]);
                (equalized_symbols_ + j * nb_tx_dmrs_ports_ + current_symbol)->imag(temp_equalized_symbols[current_symbol][2 * j + 1]);
                *(detected_symbols_ + j * nb_tx_dmrs_ports_ + current_symbol) = temp_detected_symbol_indexes[i][j];
            }
        }

        equalized_symbols_ += 8 * nb_tx_dmrs_ports_;
        detected_symbols_  += 8 * nb_tx_dmrs_ports_;
    }
}


void mimo_vblast_sqrd_avx512_no_reordering(vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                           const vector<vector<complex<float>>> &pdsch_samples,
                                           int num_re_pdsch_,
                                           complex<float> * equalized_symbols_,
                                           int nb_tx_dmrs_ports_,
                                           int nb_rx_ports_,
                                           complex<float> * constellation_symbols,
                                           int * detected_symbols_,
                                           int constellation_type_) {

    __m512 conj_vec = {1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1, 1, -1};

    __m512 r_matrix[MAX_TX_PORTS][MAX_TX_PORTS];
    __m512 q_matrix_transposed[MAX_TX_PORTS][MAX_RX_PORTS];
    __m512 squared_norms[MAX_TX_PORTS];
    __m512 temp_equalized_symbols[MAX_TX_PORTS];
    __m256i temp_detected_symbol_indexes[MAX_TX_PORTS];
    __m512 temp_detected_symbols[MAX_TX_PORTS];

    __m512 dot_prod_re, dot_prod_im, vec1, vec2;

    int i, j;

    for(int re = 0; re < num_re_pdsch_; re+= 8) {

        /// Load channel coefs in q_matrix and compute the squared norms
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            /// Compute the square norm of the column
            q_matrix_transposed[i][0] = _mm512_loadu_ps((float *) &channel_coefficients_[i][0][re]);
            vec1 = _mm512_mul_ps(q_matrix_transposed[i][0], q_matrix_transposed[i][0]);
            squared_norms[i] = _mm512_add_ps(vec1, _mm512_permute_ps(vec1, 0b10110001));
            for(j = 1; j < nb_rx_ports_; j++) {
                q_matrix_transposed[i][j] = _mm512_loadu_ps((float *) &channel_coefficients_[i][j][re]);
                vec1 = _mm512_mul_ps(q_matrix_transposed[i][j], q_matrix_transposed[i][j]);
                squared_norms[i] = _mm512_add_ps(squared_norms[i], _mm512_add_ps(vec1, _mm512_permute_ps(vec1, 0b10110001)));
            }
        }

        /// Compute QR decomp
        compute_qr_decomp(q_matrix_transposed,
                          r_matrix,
                          squared_norms,
                          nb_tx_dmrs_ports_,
                          nb_rx_ports_);

        /// Multiply received vector by transconj(Q)
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            dot_prod_re = _mm512_set1_ps(0);
            dot_prod_im = _mm512_set1_ps(0);
            for (j = 0; j < nb_rx_ports_; j++) {
                vec2 = _mm512_loadu_ps((float *) &pdsch_samples[j][re]);
                dot_prod_re = _mm512_add_ps(dot_prod_re, _mm512_mul_ps(q_matrix_transposed[i][j], vec2));
                dot_prod_im = _mm512_add_ps(dot_prod_im, _mm512_mul_ps(_mm512_permute_ps(_mm512_mul_ps(q_matrix_transposed[i][j], conj_vec), 0b10110001), vec2));
            }
            dot_prod_re = _mm512_add_ps(dot_prod_re, _mm512_permute_ps(dot_prod_re, 0b10110001));
            dot_prod_im = _mm512_add_ps(dot_prod_im, _mm512_permute_ps(dot_prod_im, 0b10110001));

            temp_equalized_symbols[i] = _mm512_permute_ps(_mm512_shuffle_ps(dot_prod_re, dot_prod_im, 0b10001000), 0b11011000);
        }

        /// Perform SIC
        for(i = nb_tx_dmrs_ports_ - 1; i > -1; i--) {

            /// Suppress interference
            for(j = i + 1; j < nb_tx_dmrs_ports_; j++) {
                vec1 = _mm512_mul_ps(r_matrix[i][j], temp_detected_symbols[j]);
                vec2 = _mm512_mul_ps(r_matrix[i][j], _mm512_permute_ps(temp_detected_symbols[j], 0b10110001));
                vec1 = _mm512_sub_ps(vec1, _mm512_permute_ps(vec1, 0b10110001));
                vec2 = _mm512_add_ps(vec2, _mm512_permute_ps(vec2, 0b10110001));
                temp_equalized_symbols[i] = _mm512_sub_ps(temp_equalized_symbols[i],
                                                                       _mm512_permute_ps(_mm512_shuffle_ps(vec1, vec2, 0b10001000), 0b11011000));
            }

            /// Divide by diagonal coef in R
            temp_equalized_symbols[i] = _mm512_div_ps(temp_equalized_symbols[i], r_matrix[i][i]);

            /// Slicing
            ml_detector_mm512(temp_equalized_symbols[i],
                              temp_detected_symbol_indexes[i],
                              temp_detected_symbols[i],
                              constellation_type_);

            /// Store in final buffer
            for(j = 0; j < 8; j++) {
                (equalized_symbols_ + j * 4 + i)->real(temp_equalized_symbols[i][2 * j]);
                (equalized_symbols_ + j * 4 + i)->imag(temp_equalized_symbols[i][2 * j + 1]);
                *(detected_symbols_ + j * 4 + i) = temp_detected_symbol_indexes[i][j];
            }
        }

        equalized_symbols_ += 32;
        detected_symbols_  += 32;
    }
}

#endif

#if defined(__AVX2__)
void mimo_vblast_sqrd_avx2(vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                           const vector<vector<complex<float>>> &pdsch_samples,
                           int num_re_pdsch_,
                           complex<float> * equalized_symbols_,
                           int nb_tx_dmrs_ports_,
                           int nb_rx_ports_,
                           complex<float> * constellation_symbols,
                           int * detected_symbols_,
                           int constellation_type_) {

    __m256 conj_vec = {1, -1, 1, -1, 1, -1, 1, -1};

    //complex<float> r_matrix_debug[MAX_TX_PORTS][MAX_TX_PORTS];
    //complex<float> temp_equalized_symbols_debug[MAX_TX_PORTS];
    //float squared_norms_debug[MAX_TX_PORTS];

    __m256 r_matrix[MAX_TX_PORTS][MAX_TX_PORTS];
    __m256 q_matrix_transposed[MAX_TX_PORTS][MAX_RX_PORTS];
    __m256 squared_norms[MAX_TX_PORTS];
    __m256 temp_equalized_symbols[MAX_TX_PORTS];
    __m128i temp_detected_symbol_indexes[MAX_TX_PORTS];
    __m256 temp_detected_symbols[MAX_TX_PORTS];

    __m256 dot_prod_re, dot_prod_im, vec1, vec2;

    int detection_reversed_order[MAX_TX_PORTS];
    //int detection_reversed_order_debug[MAX_TX_PORTS];
    int i, j;

    int current_symbol;

    for(int re = 0; re < num_re_pdsch_; re+= 4) {

        /// Load channel coefs in q_matrix and compute the squared norms
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            /// Compute the square norm of the column
            q_matrix_transposed[i][0] = _mm256_loadu_ps((float *) &channel_coefficients_[i][0][re]);
            squared_norms[i] = compute_norm_m256(q_matrix_transposed[i][0]);
            for(j = 1; j < nb_rx_ports_; j++) {
                q_matrix_transposed[i][j] = _mm256_loadu_ps((float *) &channel_coefficients_[i][j][re]);
                squared_norms[i] = _mm256_add_ps(squared_norms[i], compute_norm_m256(q_matrix_transposed[i][j]));
            }
        }

        /*
         for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            squared_norms_debug[i] = channel_coefficients_[i][0][re].real() * channel_coefficients_[i][0][re].real() +
                                              channel_coefficients_[i][0][re].imag() * channel_coefficients_[i][0][re].imag();
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 1; j < nb_rx_ports_; j++) {
                squared_norms_debug[i] += channel_coefficients_[i][j][re].real() * channel_coefficients_[i][j][re].real() +
                                                   channel_coefficients_[i][j][re].imag() * channel_coefficients_[i][j][re].imag();
            }
        } */

        /// Compute QR decomp
        compute_qr_decomp(q_matrix_transposed,
                          r_matrix,
                          squared_norms,
                          detection_reversed_order,
                          nb_tx_dmrs_ports_,
                          nb_rx_ports_);

        /*
        cout << "detection reversed order : " << endl;
        for(int k = 0; k < 4; k++) {
            cout << detection_reversed_order[k] << ", ";
        } cout << endl;

        cout << "R matrix : " << endl;
        for(int k = 0; k < 8; k += 2) {
            cout << "RE " << k << ":" << endl;
            for(int l = 0; l < 4; l++) {
                for(int m = 0; m < 4; m++) {
                    cout << "(" <<  r_matrix[m][l][k] << ",";
                    cout << r_matrix[m][l][k + 1] << ") ";
                }
                cout << endl;
            }
        }

        cout << "Q matrix : " << endl;
        for(int k = 0; k < 8; k += 2) {
            cout << "RE " << k << ":" << endl;
            for(int l = 0; l < 4; l++) {
                for(int m = 0; m < 4; m++) {
                    cout << "(" << q_matrix_transposed[l][m][k] << ",";
                    cout << q_matrix_transposed[l][m][k + 1] << ") ";
                }
                cout << endl;
            }
        } */

        /*
        memset(r_matrix_debug, 0, 16 * sizeof(complex<float>));
                    compute_qr_decomp(channel_coefficients_,
                              r_matrix_debug,
                              squared_norms_debug,
                              detection_reversed_order_debug,
                              nb_tx_dmrs_ports_,
                              nb_rx_ports_,
                              re);
        cout << "detection reversed order debug : " << endl;
        for(int k = 0; k < 4; k++) {
            cout << detection_reversed_order_debug[k] << ", ";
        } cout << endl;

        cout << "R matrix debug : " << endl;
        for(int l = 0; l < 4; l++) {
            for(int m = 0; m < 4; m++) {
                cout << "(" << r_matrix_debug[m][l] << ") ";
            }
            cout << endl;
        }

        cout << "Q matrix debug : " << endl;
        for(int l = 0; l < 4; l++) {
            for(int m = 0; m < 4; m++) {
                cout << "(" << channel_coefficients_[l][m][re] << ") ";
            }
            cout << endl;
        } */

        /// Multiply received vector by transconj(Q)
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            dot_prod_re = _mm256_set1_ps(0);
            dot_prod_im = _mm256_set1_ps(0);
            for (j = 0; j < nb_rx_ports_; j++) {
                vec2 = _mm256_loadu_ps((float *) &pdsch_samples[j][re]);
                dot_prod_re = _mm256_add_ps(dot_prod_re, _mm256_mul_ps(q_matrix_transposed[i][j], vec2));
                dot_prod_im = _mm256_add_ps(dot_prod_im, _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(q_matrix_transposed[i][j], conj_vec), 0b10110001), vec2));
            }
            temp_equalized_symbols[i] = _mm256_permute_ps(_mm256_hadd_ps(dot_prod_re, dot_prod_im), 0b11011000);
        }

        /*
        cout << "Equalized symbols RE 0 : " << endl;
        for(i = 0; i < 4; i++) {
            cout << "(" << temp_equalized_symbols[i][0] << ", " << temp_equalized_symbols[i][1] << ") " << endl;
        }

        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            *(temp_equalized_symbols_debug  + i) = conj(channel_coefficients_[i][0][re]) * pdsch_samples[0][re];
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 1; j < nb_rx_ports_; j++) {
                *(temp_equalized_symbols_debug + i) += conj(channel_coefficients_[i][j][re]) * pdsch_samples[j][re];
            }
        }

        cout << "Equalized symbols debug : " << endl;
        for(i = 0; i < 4; i++) {
            cout << temp_equalized_symbols_debug[i] << endl;
        } */

        /// Perform SIC
        for(i = nb_tx_dmrs_ports_ - 1; i > -1; i--) {
            current_symbol = detection_reversed_order[i];

            /// Suppress interference
            for(j = i + 1; j < nb_tx_dmrs_ports_; j++) {
                temp_equalized_symbols[current_symbol] = _mm256_sub_ps(temp_equalized_symbols[current_symbol],
                                                                       multiply_complex_float(r_matrix[current_symbol][detection_reversed_order[j]],
                                                                                              temp_detected_symbols[j]));
            }

            /// Divide by diagonal coef in R
            temp_equalized_symbols[current_symbol] = _mm256_div_ps(temp_equalized_symbols[current_symbol], r_matrix[current_symbol][current_symbol]);

            /// Slicing
            ml_detector_mm256(temp_equalized_symbols[current_symbol],
                              temp_detected_symbol_indexes[i],
                              temp_detected_symbols[i],
                              constellation_type_);

            /// Store in final buffer
            for(j = 0; j < 4; j++) {
                (equalized_symbols_ + j * nb_tx_dmrs_ports_ + current_symbol)->real(temp_equalized_symbols[current_symbol][2 * j]);
                (equalized_symbols_ + j * nb_tx_dmrs_ports_ + current_symbol)->imag(temp_equalized_symbols[current_symbol][2 * j + 1]);
                *(detected_symbols_ + j * nb_tx_dmrs_ports_ + current_symbol) = temp_detected_symbol_indexes[i][j];
            }
        }

        equalized_symbols_ += 4 * nb_tx_dmrs_ports_;
        detected_symbols_  += 4 * nb_tx_dmrs_ports_;
    }
}
#endif
/*************************************** Harcoded SQRD for 2 and 4 layers ************************************/
#if defined(__AVX2__)
void mimo_vblast_sqrd_avx2_2_layers(vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                    const vector<vector<complex<float>>> &pdsch_samples,
                                    int num_re_pdsch_,
                                    complex<float> * equalized_symbols_,
                                    int nb_tx_dmrs_ports_,
                                    int nb_rx_ports_,
                                    complex<float> * constellation_symbols,
                                    int * detected_symbols_,
                                    int constellation_type_) {

    __m256 conj_vec = {1, -1, 1, -1, 1, -1, 1, -1};

    __m256 r_matrix[MAX_TX_PORTS][MAX_TX_PORTS];
    __m256 q_matrix_transposed[MAX_TX_PORTS][MAX_RX_PORTS];
    __m256 squared_norms[2];
    __m256 temp_equalized_symbols[MAX_TX_PORTS];
    __m128i temp_detected_symbol_indexes[MAX_TX_PORTS];
    __m256 temp_detected_symbols[MAX_TX_PORTS];

    __m256 dot_prod_re, dot_prod_im, vec1, vec2;

    int detection_reversed_order[MAX_TX_PORTS];
    int i, j;

    int current_symbol;

    for(int re = 0; re < num_re_pdsch_; re+= 4) {

        /// Load channel coefs in q_matrix and compute the squared norms
        /// Compute the square norm of the column
        q_matrix_transposed[0][0] = _mm256_loadu_ps((float *) &channel_coefficients_[0][0][re]);
        squared_norms[0] = compute_norm_m256(q_matrix_transposed[0][0]);
        q_matrix_transposed[1][0] = _mm256_loadu_ps((float *) &channel_coefficients_[1][0][re]);
        squared_norms[1] = compute_norm_m256(q_matrix_transposed[1][0]);
        for(j = 1; j < nb_rx_ports_; j++) {
            q_matrix_transposed[0][j] = _mm256_loadu_ps((float *) &channel_coefficients_[0][j][re]);
            squared_norms[0] = _mm256_add_ps(squared_norms[0], compute_norm_m256(q_matrix_transposed[0][j]));
            q_matrix_transposed[1][j] = _mm256_loadu_ps((float *) &channel_coefficients_[1][j][re]);
            squared_norms[1] = _mm256_add_ps(squared_norms[1], compute_norm_m256(q_matrix_transposed[1][j]));
        }

        /// Compute QR decomp
        compute_qr_decomp_2_layers(q_matrix_transposed,
                          r_matrix,
                          squared_norms,
                          detection_reversed_order,
                          nb_rx_ports_);

        /// Multiply received vector by transconj(Q)
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            dot_prod_re = _mm256_set1_ps(0);
            dot_prod_im = _mm256_set1_ps(0);
            for (j = 0; j < nb_rx_ports_; j++) {
                vec2 = _mm256_loadu_ps((float *) &pdsch_samples[j][re]);
                dot_prod_re = _mm256_add_ps(dot_prod_re, _mm256_mul_ps(q_matrix_transposed[i][j], vec2));
                dot_prod_im = _mm256_add_ps(dot_prod_im, _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(q_matrix_transposed[i][j], conj_vec), 0b10110001), vec2));
            }
            temp_equalized_symbols[i] = _mm256_permute_ps(_mm256_hadd_ps(dot_prod_re, dot_prod_im), 0b11011000);
        }

        /// Perform SIC
        for(i = nb_tx_dmrs_ports_ - 1; i > -1; i--) {
            current_symbol = detection_reversed_order[i];

            /// Suppress interference
            for(j = i + 1; j < nb_tx_dmrs_ports_; j++) {
                temp_equalized_symbols[current_symbol] = _mm256_sub_ps(temp_equalized_symbols[current_symbol],
                                                                       multiply_complex_float(r_matrix[current_symbol][detection_reversed_order[j]],
                                                                                              temp_detected_symbols[j]));
            }

            /// Divide by diagonal coef in R
            temp_equalized_symbols[current_symbol] = _mm256_div_ps(temp_equalized_symbols[current_symbol], r_matrix[current_symbol][current_symbol]);

            /// Slicing
            ml_detector_mm256(temp_equalized_symbols[current_symbol],
                              temp_detected_symbol_indexes[i],
                              temp_detected_symbols[i],
                              constellation_type_);

            /// Store in final buffer
            for(j = 0; j < 4; j++) {
                (equalized_symbols_ + j * 2 + current_symbol)->real(temp_equalized_symbols[current_symbol][2 * j]);
                (equalized_symbols_ + j * 2 + current_symbol)->imag(temp_equalized_symbols[current_symbol][2 * j + 1]);
                *(detected_symbols_ + j * 2 + current_symbol) = temp_detected_symbol_indexes[i][j];
            }
        }

        /* /// TODO : to be debugged
        if(detection_reversed_order[1] == 0) {
            /// Divide by diagonal coef in R
            temp_equalized_symbols[0] = _mm256_div_ps(temp_equalized_symbols[0], r_matrix[0][0]);

            /// Slicing
            ml_detector_mm256(temp_equalized_symbols[0],
                              temp_detected_symbol_indexes[0],
                              temp_detected_symbols[0],
                              constellation_type_);

                        /// Store in final buffer
            for(j = 0; j < 4; j++) {
                (equalized_symbols_ + j * nb_tx_dmrs_ports_ + 0)->real(temp_equalized_symbols[0][2 * j]);
                (equalized_symbols_ + j * nb_tx_dmrs_ports_ + 0)->imag(temp_equalized_symbols[0][2 * j + 1]);
                *(detected_symbols_ + j * nb_tx_dmrs_ports_ + 0) = temp_detected_symbol_indexes[0][j];
            }

            temp_equalized_symbols[1] = _mm256_sub_ps(temp_equalized_symbols[1],
                                                                       multiply_complex_float(r_matrix[1][0],
                                                                                              temp_detected_symbols[0]));
            /// Divide by diagonal coef in R
            temp_equalized_symbols[1] = _mm256_div_ps(temp_equalized_symbols[1], r_matrix[1][1]);

            /// Slicing
            ml_detector_mm256(temp_equalized_symbols[1],
                              temp_detected_symbol_indexes[1],
                              temp_detected_symbols[1],
                              constellation_type_);

            /// Store in final buffer
            for(j = 0; j < 4; j++) {
                (equalized_symbols_ + j * nb_tx_dmrs_ports_ + 1)->real(temp_equalized_symbols[1][2 * j]);
                (equalized_symbols_ + j * nb_tx_dmrs_ports_ + 1)->imag(temp_equalized_symbols[1][2 * j + 1]);
                *(detected_symbols_ + j * nb_tx_dmrs_ports_ + 1) = temp_detected_symbol_indexes[1][j];
            }
        } else {
            /// Divide by diagonal coef in R
            temp_equalized_symbols[1] = _mm256_div_ps(temp_equalized_symbols[1], r_matrix[1][1]);

            /// Slicing
            ml_detector_mm256(temp_equalized_symbols[1],
                              temp_detected_symbol_indexes[1],
                              temp_detected_symbols[1],
                              constellation_type_);

                        /// Store in final buffer
            for(j = 0; j < 4; j++) {
                (equalized_symbols_ + j * nb_tx_dmrs_ports_ + 1)->real(temp_equalized_symbols[1][2 * j]);
                (equalized_symbols_ + j * nb_tx_dmrs_ports_ + 1)->imag(temp_equalized_symbols[1][2 * j + 1]);
                *(detected_symbols_ + j * nb_tx_dmrs_ports_ + 1) = temp_detected_symbol_indexes[1][j];
            }

            temp_equalized_symbols[0] = _mm256_sub_ps(temp_equalized_symbols[0],
                                                                       multiply_complex_float(r_matrix[0][1],
                                                                                              temp_detected_symbols[1]));
            /// Divide by diagonal coef in R
            temp_equalized_symbols[0] = _mm256_div_ps(temp_equalized_symbols[0], r_matrix[0][0]);

            /// Slicing
            ml_detector_mm256(temp_equalized_symbols[0],
                              temp_detected_symbol_indexes[0],
                              temp_detected_symbols[0],
                              constellation_type_);

            /// Store in final buffer
            for(j = 0; j < 4; j++) {
                (equalized_symbols_ + j * nb_tx_dmrs_ports_ + 0)->real(temp_equalized_symbols[0][2 * j]);
                (equalized_symbols_ + j * nb_tx_dmrs_ports_ + 0)->imag(temp_equalized_symbols[0][2 * j + 1]);
                *(detected_symbols_ + j * nb_tx_dmrs_ports_ + 0) = temp_detected_symbol_indexes[0][j];
            }
        } */

        equalized_symbols_ += 8;
        detected_symbols_  += 8;
    }

}
#endif

void call_vblast_qrd_no_reordering(vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                    const vector<vector<complex<float>>> &pdsch_samples,
                                    int num_re_pdsch_,
                                    complex<float> * equalized_symbols_,
                                    int nb_tx_dmrs_ports_,
                                    int nb_rx_ports_,
                                    complex<float> * constellation_symbols,
                                    int * detected_symbols_,
                                    int constellation_type_) {
    if(nb_tx_dmrs_ports_ == 2) {
        mimo_vblast_decoder_qr_decomp_no_reordering_2_layers(channel_coefficients_,
                                                             pdsch_samples,
                                                             num_re_pdsch_,
                                                             equalized_symbols_,
                                                             nb_tx_dmrs_ports_,
                                                             nb_rx_ports_,
                                                             constellation_symbols,
                                                             detected_symbols_,
                                                             constellation_type_);

    } else {
        mimo_vblast_decoder_qr_decomp_no_reordering_modified(channel_coefficients_,
                                                             pdsch_samples,
                                                             num_re_pdsch_,
                                                             equalized_symbols_,
                                                             nb_tx_dmrs_ports_,
                                                             nb_rx_ports_,
                                                             constellation_symbols,
                                                             detected_symbols_,
                                                             constellation_type_);

    }

}

void call_vblast_sqrd_functions(vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                    const vector<vector<complex<float>>> &pdsch_samples,
                                    int num_re_pdsch_,
                                    complex<float> * equalized_symbols_,
                                    int nb_tx_dmrs_ports_,
                                    int nb_rx_ports_,
                                    complex<float> * constellation_symbols,
                                    int * detected_symbols_,
                                    int constellation_type_) {

    if(nb_tx_dmrs_ports_ == 2) {
        mimo_vblast_decoder_qr_decomp_modified_2_layers(channel_coefficients_,
                                                        pdsch_samples,
                                                        num_re_pdsch_,
                                                        equalized_symbols_,
                                                        nb_tx_dmrs_ports_,
                                                        nb_rx_ports_,
                                                        constellation_symbols,
                                                        detected_symbols_,
                                                        constellation_type_);
    } else {
        mimo_vblast_decoder_qr_decomp_modified(channel_coefficients_,
                                                pdsch_samples,
                                                num_re_pdsch_,
                                                equalized_symbols_,
                                                nb_tx_dmrs_ports_,
                                                nb_rx_ports_,
                                                constellation_symbols,
                                                detected_symbols_,
                                                constellation_type_);
    }
}

#if defined(__AVX2__)
void call_vblast_sqrd_functions_avx2(vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                    const vector<vector<complex<float>>> &pdsch_samples,
                                    int num_re_pdsch_,
                                    complex<float> * equalized_symbols_,
                                    int nb_tx_dmrs_ports_,
                                    int nb_rx_ports_,
                                    complex<float> * constellation_symbols,
                                    int * detected_symbols_,
                                    int constellation_type_) {

    if(nb_tx_dmrs_ports_ == 2) {
        mimo_vblast_sqrd_avx2_2_layers(channel_coefficients_,
                                       pdsch_samples,
                                        num_re_pdsch_,
                                        equalized_symbols_,
                                        nb_tx_dmrs_ports_,
                                        nb_rx_ports_,
                                        constellation_symbols,
                                        detected_symbols_,
                                        constellation_type_);
    } else {
        mimo_vblast_sqrd_avx2(channel_coefficients_,
                              pdsch_samples,
                              num_re_pdsch_,
                              equalized_symbols_,
                              nb_tx_dmrs_ports_,
                              nb_rx_ports_,
                              constellation_symbols,
                              detected_symbols_,
                              constellation_type_);
    }
}
#endif
#if defined(__AVX2__)
void call_vblast_qrd_col_norm_functions_avx2(vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                             const vector<vector<complex<float>>> &pdsch_samples,
                                             int num_re_pdsch_,
                                            complex<float> * equalized_symbols_,
                                            int nb_tx_dmrs_ports_,
                                            int nb_rx_ports_,
                                            complex<float> * constellation_symbols,
                                            int * detected_symbols_,
                                            int constellation_type_) {

    if(nb_tx_dmrs_ports_ == 2) {
        mimo_vblast_qrd_col_norm_avx2_2_layers(channel_coefficients_,
                                               pdsch_samples,
                                               num_re_pdsch_,
                                               equalized_symbols_,
                                               nb_tx_dmrs_ports_,
                                               nb_rx_ports_,
                                               constellation_symbols,
                                               detected_symbols_,
                                               constellation_type_);
    } else {
        mimo_vblast_qrd_col_norm_avx2(channel_coefficients_,
                                      pdsch_samples,
                                      num_re_pdsch_,
                                      equalized_symbols_,
                                      nb_tx_dmrs_ports_,
                                      nb_rx_ports_,
                                      constellation_symbols,
                                      detected_symbols_,
                                      constellation_type_);
    }
}
#endif
void call_vblast_qrd_col_norm_functions(vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                        const vector<vector<complex<float>>> &pdsch_samples,
                                        int num_re_pdsch_,
                                        complex<float> * equalized_symbols_,
                                        int nb_tx_dmrs_ports_,
                                        int nb_rx_ports_,
                                        complex<float> * constellation_symbols,
                                        int * detected_symbols_,
                                        int constellation_type_) {

    if(nb_tx_dmrs_ports_ == 2) {
        mimo_vblast_qrd_col_norm_modified_2_layers(channel_coefficients_,
                                                    pdsch_samples,
                                                    num_re_pdsch_,
                                                    equalized_symbols_,
                                                    nb_tx_dmrs_ports_,
                                                    nb_rx_ports_,
                                                    constellation_symbols,
                                                    detected_symbols_,
                                                    constellation_type_);
    } else {
        mimo_vblast_qrd_col_norm_modified(channel_coefficients_,
                                            pdsch_samples,
                                            num_re_pdsch_,
                                            equalized_symbols_,
                                            nb_tx_dmrs_ports_,
                                            nb_rx_ports_,
                                            constellation_symbols,
                                            detected_symbols_,
                                            constellation_type_);
    }
}

/*************************************************************************************************************/
void mimo_vblast_decoder_qr_decomp_modified(complex<float> channel_coefficients_[][MAX_TX_PORTS][MAX_RX_PORTS],
                                            const vector<vector<complex<float>>> &pdsch_samples,
                                            int num_re_pdsch_,
                                            complex<float> * equalized_symbols_,
                                            int nb_tx_dmrs_ports_,
                                            int nb_rx_ports_,
                                            complex<float> * constellation_symbols,
                                            int * detected_symbols_,
                                            int constellation_type_) {

    complex<float> r_matrix[MAX_TX_PORTS][MAX_TX_PORTS];
    float temp_squared_norms[MAX_TX_PORTS];
    int detection_reversed_order[MAX_TX_PORTS];

    int i, j;
    int current_symbol;

    for(int re = 0; re < num_re_pdsch_; re++) {
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            temp_squared_norms[i] = channel_coefficients_[re][i][0].real() * channel_coefficients_[re][i][0].real() +
                                              channel_coefficients_[re][i][0].imag() * channel_coefficients_[re][i][0].imag();
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 1; j < nb_rx_ports_; j++) {
                temp_squared_norms[i] += channel_coefficients_[re][i][j].real() * channel_coefficients_[re][i][j].real() +
                                                   channel_coefficients_[re][i][j].imag() * channel_coefficients_[re][i][j].imag();
            }
        }

        compute_qr_decomp(channel_coefficients_[re],
                          r_matrix,
                          temp_squared_norms,
                          detection_reversed_order,
                          nb_tx_dmrs_ports_,
                          nb_rx_ports_);

        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            *(equalized_symbols_  + i) = conj(channel_coefficients_[re][i][0]) * pdsch_samples[0][re];
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 1; j < nb_rx_ports_; j++) {
                *(equalized_symbols_  + i) += conj(channel_coefficients_[re][i][j]) * pdsch_samples[j][re];
            }
        }

         current_symbol = detection_reversed_order[nb_tx_dmrs_ports_ - 1];
         *(equalized_symbols_  + detection_reversed_order[nb_tx_dmrs_ports_ - 1]) /=
                 r_matrix[current_symbol][current_symbol].real();
        /// Slicing
        *(detected_symbols_  + current_symbol) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + current_symbol));
        /// Remaining iterations
        for(i = nb_tx_dmrs_ports_ - 2; i > - 1; i--) {
            current_symbol = detection_reversed_order[i];

            /// remove contribution of previously detected symbols
            for(j =  i + 1; j < nb_tx_dmrs_ports_; j++) {
                *(equalized_symbols_ + current_symbol)  -= r_matrix[current_symbol][detection_reversed_order[j]] * constellation_symbols[*(detected_symbols_ + detection_reversed_order[j])];

            }

            /// Equalization / remove contribution of r(i,i)
            *(equalized_symbols_  + detection_reversed_order[i]) /= r_matrix[current_symbol][current_symbol].real();

            /// Slicing
            *(detected_symbols_  + current_symbol) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + current_symbol));
        }
        equalized_symbols_ += nb_tx_dmrs_ports_;
        detected_symbols_ += nb_tx_dmrs_ports_;

    }
}

/*************************************************************************************************************/

void mimo_vblast_decoder_qr_decomp_modified(vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                            const vector<vector<complex<float>>> &pdsch_samples,
                                            int num_re_pdsch_,
                                            complex<float> * equalized_symbols_,
                                            int nb_tx_dmrs_ports_,
                                            int nb_rx_ports_,
                                            complex<float> * constellation_symbols,
                                            int * detected_symbols_,
                                            int constellation_type_) {

    complex<float> r_matrix[MAX_TX_PORTS][MAX_TX_PORTS];
    float temp_squared_norms[MAX_TX_PORTS];
    int detection_reversed_order[MAX_TX_PORTS];

    int i, j;
    int current_symbol;

    for(int re = 0; re < num_re_pdsch_; re++) {
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            temp_squared_norms[i] = channel_coefficients_[i][0][re].real() * channel_coefficients_[i][0][re].real() +
                                              channel_coefficients_[i][0][re].imag() * channel_coefficients_[i][0][re].imag();
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 1; j < nb_rx_ports_; j++) {
                temp_squared_norms[i] += channel_coefficients_[i][j][re].real() * channel_coefficients_[i][j][re].real() +
                                                   channel_coefficients_[i][j][re].imag() * channel_coefficients_[i][j][re].imag();
            }
        }

        compute_qr_decomp(channel_coefficients_,
                          r_matrix,
                          temp_squared_norms,
                          detection_reversed_order,
                          nb_tx_dmrs_ports_,
                          nb_rx_ports_,
                          re);

        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            *(equalized_symbols_  + i) = conj(channel_coefficients_[i][0][re]) * pdsch_samples[0][re];
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 1; j < nb_rx_ports_; j++) {
                *(equalized_symbols_  + i) += conj(channel_coefficients_[i][j][re]) * pdsch_samples[j][re];
            }
        }

         current_symbol = detection_reversed_order[nb_tx_dmrs_ports_ - 1];
         *(equalized_symbols_  + detection_reversed_order[nb_tx_dmrs_ports_ - 1]) /=
                 r_matrix[current_symbol][current_symbol].real();
        /// Slicing
        *(detected_symbols_  + current_symbol) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + current_symbol));
        /// Remaining iterations
        for(i = nb_tx_dmrs_ports_ - 2; i > - 1; i--) {
            current_symbol = detection_reversed_order[i];

            /// remove contribution of previously detected symbols
            for(j =  i + 1; j < nb_tx_dmrs_ports_; j++) {
                *(equalized_symbols_ + current_symbol)  -= r_matrix[current_symbol][detection_reversed_order[j]] * constellation_symbols[*(detected_symbols_ + detection_reversed_order[j])];

            }

            /// Equalization / remove contribution of r(i,i)
            *(equalized_symbols_  + detection_reversed_order[i]) /= r_matrix[current_symbol][current_symbol].real();

            /// Slicing
            *(detected_symbols_  + current_symbol) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + current_symbol));
        }
        equalized_symbols_ += nb_tx_dmrs_ports_;
        detected_symbols_ += nb_tx_dmrs_ports_;
    }
}

void mimo_vblast_decoder_qr_decomp_modified_2_layers(vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                                     const vector<vector<complex<float>>> &pdsch_samples,
                                                     int num_re_pdsch_,
                                                     complex<float> * equalized_symbols_,
                                                     int nb_tx_dmrs_ports_,
                                                     int nb_rx_ports_,
                                                     complex<float> * constellation_symbols,
                                                     int * detected_symbols_,
                                                     int constellation_type_) {

    complex<float> r_matrix[2][2];
    float temp_squared_norms[2];
    int detection_reversed_order[2];
    int i, j;

    for(int re = 0; re < num_re_pdsch_; re++) {
        temp_squared_norms[0] = channel_coefficients_[0][0][re].real() * channel_coefficients_[0][0][re].real() +
                                          channel_coefficients_[0][0][re].imag() * channel_coefficients_[0][0][re].imag();
        temp_squared_norms[1] = channel_coefficients_[1][0][re].real() * channel_coefficients_[1][0][re].real() +
                                          channel_coefficients_[1][0][re].imag() * channel_coefficients_[1][0][re].imag();
        for(j = 1; j < nb_rx_ports_; j++) {
            temp_squared_norms[0] += channel_coefficients_[0][j][re].real() * channel_coefficients_[0][j][re].real() +
                                               channel_coefficients_[0][j][re].imag() * channel_coefficients_[0][j][re].imag();
            temp_squared_norms[1] += channel_coefficients_[1][j][re].real() * channel_coefficients_[1][j][re].real() +
                                   channel_coefficients_[1][j][re].imag() * channel_coefficients_[1][j][re].imag();
        }

        compute_qr_decomp_2_layers(channel_coefficients_,
          r_matrix,
          temp_squared_norms,
          detection_reversed_order,
          nb_tx_dmrs_ports_,
          nb_rx_ports_,
          re);

        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            *(equalized_symbols_  + i) = conj(channel_coefficients_[i][0][re]) * pdsch_samples[0][re];
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 1; j < nb_rx_ports_; j++) {
                *(equalized_symbols_  + i) += conj(channel_coefficients_[i][j][re]) * pdsch_samples[j][re];
            }
        }

        if(detection_reversed_order[0] == 0) {
             *(equalized_symbols_  + 1) /= r_matrix[1][1].real();
            /// Slicing
            *(detected_symbols_  + 1) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + 1));

            *(equalized_symbols_ + 0)  -= r_matrix[0][1] * constellation_symbols[*(detected_symbols_ + 1)];

            /// Equalization / remove contribution of r(i,i)
            *(equalized_symbols_  + 0) /= r_matrix[0][0].real();

            /// Slicing
            *(detected_symbols_  + 0) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + 0));

        } else {
            *(equalized_symbols_  + 0) /= r_matrix[0][0].real();
            /// Slicing
            *(detected_symbols_  + 0) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + 0));

            *(equalized_symbols_ + 1)  -= r_matrix[1][0] * constellation_symbols[*(detected_symbols_ + 0)];

            /// Equalization / remove contribution of r(i,i)
            *(equalized_symbols_  + 1) /= r_matrix[1][1].real();

            /// Slicing
            *(detected_symbols_  + 1) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + 1));
        }

        equalized_symbols_ += 2;
        detected_symbols_ += 2;
    }
}

void mimo_vblast_decoder_qr_decomp(const vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                   const vector<vector<complex<float>>> &pdsch_samples,
                                   int num_re_pdsch_,
                                   complex<float> * equalized_symbols_,
                                   int nb_tx_dmrs_ports_,
                                   int nb_rx_ports_,
                                   complex<float> * constellation_symbols,
                                   int * detected_symbols_,
                                   int constellation_type_) {

    complex<float> q_matrix[MAX_TX_PORTS][MAX_RX_PORTS];
    complex<float> r_matrix[MAX_TX_PORTS][MAX_TX_PORTS];

    float temp_squared_norms[MAX_TX_PORTS];
    int detection_reversed_order[MAX_TX_PORTS];
    int i, j, k;

    int current_symbol;

    //complex<float> test_channel_matrix[nb_rx_ports_ * nb_tx_dmrs_ports_];
    //complex<float> test_channel_matrix2[nb_tx_dmrs_ports_ * nb_tx_dmrs_ports_];
    //complex<float> test_channel_matrix2[MAX_RX_PORTS][MAX_TX_PORTS];


    for(int re = 0; re < num_re_pdsch_; re++) {

        /// Load channel coefficients in Q matrix
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 0; j < nb_rx_ports_; j++) {
                q_matrix[i][j] = channel_coefficients_[i][j][re];
            }
        }

        /// Load squared norms
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            temp_squared_norms[i] = q_matrix[i][0].real() * q_matrix[i][0].real() +
                                              q_matrix[i][0].imag() * q_matrix[i][0].imag();
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 1; j < nb_rx_ports_; j++) {
                temp_squared_norms[i] += q_matrix[i][j].real() * q_matrix[i][j].real() +
                                         q_matrix[i][j].imag() * q_matrix[i][j].imag();
            }
        }

#if TIME_MEASURE == 1
        t2 = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(trace) << "Load squared norms : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

        t1 = std::chrono::steady_clock::now();
#endif
        /// compute QR decomposition
        compute_qr_decomp(q_matrix,
                          r_matrix,
                          temp_squared_norms,
                          detection_reversed_order,
                          nb_rx_ports_,
                          nb_tx_dmrs_ports_);

        /**
        compute_qr_decomp(q_matrix,
                          r_matrix,
                          temp_squared_norms,
                          detection_reversed_order,
                          nb_tx_dmrs_ports_,
                          //nb_tx_dmrs_ports_,
                          nb_rx_ports_);
                          //nb_tx_dmrs_ports_); */
#if TIME_MEASURE == 1
        t2 = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(trace) << "Compute QR decomp : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
#endif

        /*
        cout << "Q_h * H" << endl;
        memset(&test_channel_matrix2[0], 0, nb_rx_ports_ * nb_tx_dmrs_ports_ * sizeof(complex<float>));
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 0; j < nb_tx_dmrs_ports_; j++) {
                for(int k = 0; k < nb_rx_ports_; k++) {
                    /**
                    test_channel_matrix2[col * nb_rx_ports_ + row] +=
                            conj(q_matrix[detection_reversed_order[row] * nb_rx_ports_ + k]) *
                            test_channel_matrix[col * nb_rx_ports_ + k];
                            test_channel_matrix2[i][j] += conj(q_matrix[i][k]) * channel_coefficients_[k][j][re];

                }
            }
        }

        for(int row = 0; row < nb_rx_ports_; row++) {
            for(int col = 0; col < nb_tx_dmrs_ports_; col++) {
                //cout << test_channel_matrix2[col * nb_rx_ports_ + row] << endl;
                cout << test_channel_matrix2[row][col] << " ";
            }
            cout << endl;
        }

        cout << "detection reversed order : " << endl;
        for(int k = 0; k < nb_tx_dmrs_ports_; k++) {
            cout << detection_reversed_order[k] << endl;
        }

        cout << "R matrix after QR decomp : " << endl;
        for(int row = 0; row < nb_tx_dmrs_ports_; row++) {
            for(int col = 0; col < nb_tx_dmrs_ports_; col++) {
                //cout << r_matrix[col * nb_tx_dmrs_ports_ + row] << endl;
                cout << r_matrix[row][col] << " ";
            }
            cout << endl;
        } */

        /// Suppress contribution of Q matrix in the received signal
        //t1 = std::chrono::steady_clock::now();
        //cout << "pdsch samples : " << endl;
        //for (int receiver = 0; receiver < nb_rx_ports_; receiver++) {
        //    temp_received_symbols[receiver] = ;
        //cout << pdsch_samples[receiver][re] << endl;
        //counter++;
        //cout << counter << endl;
        //}
        //t2 = std::chrono::steady_clock::now();
        //cout << " load received symbols : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

#if TIME_MEASURE == 1
        t1 = std::chrono::steady_clock::now();
#endif

        //std::fill(q_h_symbols, q_h_symbols + nb_tx_dmrs_ports_, 0);
        /**
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            q_h_symbols[transmitter] = conj(q_matrix[detection_reversed_order[transmitter] * nb_rx_ports_]) * pdsch_samples[0][re];
        }
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            for(int receiver = 1; receiver < nb_rx_ports_; receiver++) {
                q_h_symbols[transmitter] += conj(q_matrix[detection_reversed_order[transmitter] * nb_rx_ports_ // column
                                                          + receiver // row
                                                 ]) * pdsch_samples[receiver][re];
            }
        } */

        /**
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            *(equalized_symbols_  + transmitter) = conj(q_matrix[transmitter * nb_rx_ports_]) * pdsch_samples[0][re];
        }
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            for(int receiver = 1; receiver < nb_rx_ports_; receiver++) {
                *(equalized_symbols_  + transmitter) += conj(q_matrix[transmitter * nb_rx_ports_ // column
                                                          + receiver // row
                                                 ]) * pdsch_samples[receiver][re];
            }
        } */
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            *(equalized_symbols_  + i) = conj(q_matrix[i][0]) * pdsch_samples[0][re];
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 1; j < nb_rx_ports_; j++) {
                *(equalized_symbols_  + i) += conj(q_matrix[i][j]) * pdsch_samples[j][re];
            }
        }

        /**
        cout << "q_h_symbols before SIC : " << endl;
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            cout << q_h_symbols[transmitter] << endl;
        } */

#if TIME_MEASURE == 1
        t2 = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(trace) << "multiply qH by y : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;
#endif

        /// SIC detector
        /// Decode the first symbol
        /// remove contribution of r(i,i)
#if TIME_MEASURE == 1
        t1 = std::chrono::steady_clock::now();
#endif
        //q_h_symbols[nb_tx_dmrs_ports_ - 1] /= r_matrix[detection_reversed_order[nb_tx_dmrs_ports_ - 1] * nb_tx_dmrs_ports_ + detection_reversed_order[nb_tx_dmrs_ports_ - 1]].real();

         //*(equalized_symbols_  + detection_reversed_order[nb_tx_dmrs_ports_ - 1]) /= r_matrix[detection_reversed_order[nb_tx_dmrs_ports_ - 1] * nb_tx_dmrs_ports_ + detection_reversed_order[nb_tx_dmrs_ports_ - 1]].real();

         //*(equalized_symbols_  + detection_reversed_order[nb_tx_dmrs_ports_ - 1]) /=
         //        r_matrix[nb_tx_dmrs_ports_ - 1][nb_tx_dmrs_ports_ - 1].real();

         current_symbol = detection_reversed_order[nb_tx_dmrs_ports_ - 1];

         *(equalized_symbols_  + detection_reversed_order[nb_tx_dmrs_ports_ - 1]) /=
                 r_matrix[current_symbol][current_symbol].real();

        /// Slicing
        //ml_detector_complex[constellation_type_](q_h_symbols[nb_tx_dmrs_ports_ - 1], temp_detected_symbols_iteration[detection_reversed_order[nb_tx_dmrs_ports_ - 1]]);

        *(detected_symbols_  + current_symbol) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + current_symbol));

        /// Remaining iterations
        for(i = nb_tx_dmrs_ports_ - 2; i > - 1; i--) {
            current_symbol = detection_reversed_order[i];

            /// remove contribution of previously detected symbols
            for(k =  i + 1; k < nb_tx_dmrs_ports_; k++) {
                //q_h_symbols[i] -= r_matrix[detection_reversed_order[k] * nb_tx_dmrs_ports_
                //                           + detection_reversed_order[i]] * constellation_symbols[temp_detected_symbols_iteration[detection_reversed_order[k]]];
                //cout << "rij : " << r_matrix[detection_reversed_order[k] * nb_tx_dmrs_ports_
                //                 + detection_reversed_order[i]] << endl;

                //*(equalized_symbols_  + detection_reversed_order[i])  -= r_matrix[detection_reversed_order[k] * nb_tx_dmrs_ports_
                //                                                                  + detection_reversed_order[i]] * constellation_symbols[*(detected_symbols_ + detection_reversed_order[k])];
                *(equalized_symbols_ + current_symbol)  -= r_matrix[current_symbol][detection_reversed_order[k]] * constellation_symbols[*(detected_symbols_ + detection_reversed_order[k])];
                //*(equalized_symbols_  + detection_reversed_order[i])  -= r_matrix[i][k] * constellation_symbols[*(detected_symbols_ + detection_reversed_order[k])];
            }

            /// Equalization / remove contribution of r(i,i)
            //q_h_symbols[i] /= r_matrix[detection_reversed_order[i] * nb_tx_dmrs_ports_ + detection_reversed_order[i]].real();
            //cout << "current tx antenna : " << detection_reversed_order[i] << endl;
            //cout << "rii" << r_matrix[detection_reversed_order[i] * nb_tx_dmrs_ports_ + detection_reversed_order[i]] << endl;

            //*(equalized_symbols_  + detection_reversed_order[i]) /= r_matrix[detection_reversed_order[i] * nb_tx_dmrs_ports_ + detection_reversed_order[i]].real();
            //*(equalized_symbols_  + detection_reversed_order[i]) /= r_matrix[detection_reversed_order[i]][detection_reversed_order[i]].real();
            *(equalized_symbols_  + detection_reversed_order[i]) /= r_matrix[current_symbol][current_symbol].real();

            /// Slicing
            //ml_detector_complex[constellation_type_](q_h_symbols[i],
            //                                         temp_detected_symbols_iteration[detection_reversed_order[i]]);
            *(detected_symbols_  + current_symbol) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + current_symbol));
        }

        /// Add temp_detected symbols to detected_symbols buffer
        /**
        for(int i = 0; i < nb_tx_dmrs_ports_; i++) {
            equalized_symbols_[detection_reversed_order[i]] = q_h_symbols[i];
            detected_symbols_[i] = temp_detected_symbols_iteration[i];
            //cout << "symbol : " << detection_reversed_order[i] << endl;
            //cout << "q_h_symbols " << q_h_symbols[i] << endl;
            //cout << "equalized symbol : " << equalized_symbols_[re * nb_tx_dmrs_ports_ + detection_reversed_order[i]] << endl;
            //cout << "detected symbol : " << temp_detected_symbols_iteration[detection_reversed_order[i]] << endl;
            //cout << "re + i : " << re * nb_tx_dmrs_ports_ + detection_reversed_order[i] << endl;
        } */

        equalized_symbols_ += nb_tx_dmrs_ports_;
        detected_symbols_ += nb_tx_dmrs_ports_;

#if TIME_MEASURE == 1
        t2 = std::chrono::steady_clock::now();
        BOOST_LOG_TRIVIAL(trace) << "SIC detector : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << "\n" << endl;
#endif
    }
}

void mimo_vblast_decoder_qr_decomp_2_layers(const vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                           const vector<vector<complex<float>>> &pdsch_samples,
                                           int num_re_pdsch_,
                                           complex<float> * equalized_symbols_,
                                           int nb_tx_dmrs_ports_,
                                           int nb_rx_ports_,
                                           complex<float> * constellation_symbols,
                                           int * detected_symbols_,
                                           int constellation_type_) {

    complex<float> q_matrix[2][MAX_RX_PORTS];
    complex<float> r_matrix[2][2];

    float temp_squared_norms[2];
    int detection_reversed_order[2];
    int i, j, k;

    int current_symbol;

    for(int re = 0; re < num_re_pdsch_; re++) {

        /// Load channel coefficients in Q matrix
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 0; j < nb_rx_ports_; j++) {
                q_matrix[i][j] = channel_coefficients_[i][j][re];
            }
        }

        /// Load squared norms
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            temp_squared_norms[i] = q_matrix[i][0].real() * q_matrix[i][0].real() +
                                              q_matrix[i][0].imag() * q_matrix[i][0].imag();
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 1; j < nb_rx_ports_; j++) {
                temp_squared_norms[i] += q_matrix[i][j].real() * q_matrix[i][j].real() +
                                         q_matrix[i][j].imag() * q_matrix[i][j].imag();
            }
        }

        /// compute QR decomposition
        compute_qr_decomp_2_layers(q_matrix,
                                   r_matrix,
                                   temp_squared_norms,
                                   detection_reversed_order,
                                   nb_rx_ports_,
                                   nb_tx_dmrs_ports_);

        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            *(equalized_symbols_  + i) = conj(q_matrix[i][0]) * pdsch_samples[0][re];
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 1; j < nb_rx_ports_; j++) {
                *(equalized_symbols_  + i) += conj(q_matrix[i][j]) * pdsch_samples[j][re];
            }
        }

        if(detection_reversed_order[0] == 0) {
             /// SIC detector
             *(equalized_symbols_  + 1) /=
                     r_matrix[1][1].real();

            /// Slicing
            *(detected_symbols_  + 1) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + 1));

            *(equalized_symbols_ + 0)  -= r_matrix[0][1] * constellation_symbols[*(detected_symbols_ + 1)];

            /// Equalization / remove contribution of r(i,i)
            *(equalized_symbols_ + 0) /= r_matrix[0][0].real();

            /// Slicing
            *(detected_symbols_  + 0) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + 0));

        } else {
             /// SIC detector
             *(equalized_symbols_  + 0) /=
                     r_matrix[0][0].real();

            /// Slicing
            *(detected_symbols_ + 0) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + 0));

            *(equalized_symbols_ + 1)  -= r_matrix[1][0] * constellation_symbols[*(detected_symbols_ + 0)];

            /// Equalization / remove contribution of r(i,i)
            *(equalized_symbols_ + 1) /= r_matrix[1][1].real();

            /// Slicing
            *(detected_symbols_  + 1) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + 1));
        }

        equalized_symbols_ += 2;
        detected_symbols_ += 2;
    }
}

void mimo_vblast_decoder_qr_decomp_no_reordering_modified(vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                                          const vector<vector<complex<float>>> &pdsch_samples,
                                                          int num_re_pdsch_,
                                                          complex<float> * equalized_symbols_,
                                                          int nb_tx_dmrs_ports_,
                                                          int nb_rx_ports_,
                                                          complex<float> * constellation_symbols,
                                                          int * detected_symbols_,
                                                          int constellation_type_) {

    complex<float> r_matrix[MAX_TX_PORTS][MAX_TX_PORTS];
    float temp_squared_norms[MAX_TX_PORTS];

    for(int re = 0; re < num_re_pdsch_; re++) {

        /// Load squared norms
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            temp_squared_norms[transmitter] = channel_coefficients_[transmitter][0][re].real() * channel_coefficients_[transmitter][0][re].real()
                                              + channel_coefficients_[transmitter][0][re].imag() * channel_coefficients_[transmitter][0][re].imag();
        }
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            for(int receiver = 1; receiver < nb_rx_ports_; receiver++) {
                temp_squared_norms[transmitter] += channel_coefficients_[transmitter][receiver][re].real() * channel_coefficients_[transmitter][receiver][re].real() +
                                                   channel_coefficients_[transmitter][receiver][re].imag() * channel_coefficients_[transmitter][receiver][re].imag();
            }
        }

        /// compute QR decomposition
        compute_qr_decomp(channel_coefficients_,
                          r_matrix,
                          temp_squared_norms,
                          nb_tx_dmrs_ports_,
                          nb_rx_ports_,
                          re);

        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            *(equalized_symbols_  + transmitter) = conj(channel_coefficients_[transmitter][0][re]) * pdsch_samples[0][re];
        }
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            for(int receiver = 1; receiver < nb_rx_ports_; receiver++) {
                *(equalized_symbols_  + transmitter) += conj(channel_coefficients_[transmitter][receiver][re]) * pdsch_samples[receiver][re];
            }
        }

        /// SIC detector
        /// Decode the first symbol
        /// remove contribution of r(i,i)
        *(equalized_symbols_  + nb_tx_dmrs_ports_ - 1) /= r_matrix[nb_tx_dmrs_ports_ - 1][nb_tx_dmrs_ports_ - 1].real();
        /// Slicing
        *(detected_symbols_  + nb_tx_dmrs_ports_ - 1) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + nb_tx_dmrs_ports_ - 1));

        /// Remaining iterations
        for(int i = nb_tx_dmrs_ports_ - 2; i > - 1; i--) {
            /// remove contribution of previously detected symbols
            for(int k =  i + 1; k < nb_tx_dmrs_ports_; k++) {
                *(equalized_symbols_  + i)  -= r_matrix[i][k] * constellation_symbols[*(detected_symbols_ + k)];
            }
            /// Equalization / remove contribution of r(i,i)
            *(equalized_symbols_  + i) /= r_matrix[i][i].real();

            /// Slicing
            *(detected_symbols_  + i) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + i));
        }

        equalized_symbols_ += nb_tx_dmrs_ports_;
        detected_symbols_ += nb_tx_dmrs_ports_;
    }

}

void mimo_vblast_decoder_qr_decomp_no_reordering_2_layers(vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                                          const vector<vector<complex<float>>> &pdsch_samples,
                                                          int num_re_pdsch_,
                                                          complex<float> * equalized_symbols_,
                                                          int nb_tx_dmrs_ports_,
                                                          int nb_rx_ports_,
                                                          complex<float> * constellation_symbols,
                                                          int * detected_symbols_,
                                                          int constellation_type_) {

    complex<float> r_matrix[2][2];
    float temp_squared_norms[2];

    for(int re = 0; re < num_re_pdsch_; re++) {

        /// Load squared norms
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            temp_squared_norms[transmitter] = channel_coefficients_[transmitter][0][re].real() * channel_coefficients_[transmitter][0][re].real()
                                              + channel_coefficients_[transmitter][0][re].imag() * channel_coefficients_[transmitter][0][re].imag();
        }
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            for(int receiver = 1; receiver < nb_rx_ports_; receiver++) {
                temp_squared_norms[transmitter] += channel_coefficients_[transmitter][receiver][re].real() * channel_coefficients_[transmitter][receiver][re].real() +
                                                   channel_coefficients_[transmitter][receiver][re].imag() * channel_coefficients_[transmitter][receiver][re].imag();
            }
        }

        /// compute QR decomposition
        compute_qr_decomp_2_layers(channel_coefficients_,
                                  r_matrix,
                                  temp_squared_norms,
                                  nb_rx_ports_,
                                  re);

        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            *(equalized_symbols_  + transmitter) = conj(channel_coefficients_[transmitter][0][re]) * pdsch_samples[0][re];
        }
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            for(int receiver = 1; receiver < nb_rx_ports_; receiver++) {
                *(equalized_symbols_  + transmitter) += conj(channel_coefficients_[transmitter][receiver][re]) * pdsch_samples[receiver][re];
            }
        }

        /// SIC detector
        /// Decode the first symbol
        /// remove contribution of r(i,i)
        equalized_symbols_[1] /= r_matrix[1][1].real();
        /// Slicing
        detected_symbols_[1] = ml_detector_inline[constellation_type_](equalized_symbols_[1]);

        equalized_symbols_[0]  -= r_matrix[0][1] * constellation_symbols[detected_symbols_[1]];

        equalized_symbols_[0] /= r_matrix[0][0].real();
        /// Slicing
        detected_symbols_[0] = ml_detector_inline[constellation_type_](equalized_symbols_[0]);

        equalized_symbols_ += 2;
        detected_symbols_ += 2;
    }
}

void mimo_vblast_decoder_qr_decomp_no_reordering(vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                                 const vector<vector<complex<float>>> &pdsch_samples,
                                                 int num_re_pdsch_,
                                                 complex<float> * equalized_symbols_,
                                                 int nb_tx_dmrs_ports_,
                                                 int nb_rx_ports_,
                                                 complex<float> * constellation_symbols,
                                                 int * detected_symbols_,
                                                 int constellation_type_) {

    complex<float> q_matrix[MAX_RX_PORTS][MAX_TX_PORTS]; /// Store transposed Q matrix
    complex<float> r_matrix[MAX_TX_PORTS][MAX_TX_PORTS];
    float temp_squared_norms[MAX_TX_PORTS];

    for(int re = 0; re < num_re_pdsch_; re++) {

        /// Load channel coefficients in Q matrix
        for(int row = 0; row < nb_rx_ports_; row++) {
            for(int col = 0; col < nb_tx_dmrs_ports_; col++) {
                q_matrix[col][row] = channel_coefficients_[col][row][re];
            }
        }

        /// Load squared norms
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            temp_squared_norms[transmitter] = q_matrix[transmitter][0].real() * q_matrix[transmitter][0].real()
                                              + q_matrix[transmitter][0].imag() * q_matrix[transmitter][0].imag();
            //temp_squared_norms[transmitter] = q_matrix[transmitter * nb_rx_ports_].real() * q_matrix[transmitter * nb_rx_ports_].real()
            //+ q_matrix[transmitter * nb_rx_ports_].imag() * q_matrix[transmitter * nb_rx_ports_].imag();
        }
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            for(int receiver = 1; receiver < nb_rx_ports_; receiver++) {
                temp_squared_norms[transmitter] += q_matrix[transmitter][receiver].real() * q_matrix[transmitter][receiver].real() +
                                                   q_matrix[transmitter][receiver].imag() * q_matrix[transmitter][receiver].imag();
            }
        }

        /// compute QR decomposition
        compute_qr_decomp(q_matrix,
                          r_matrix,
                          temp_squared_norms,
                          nb_tx_dmrs_ports_,
                          nb_rx_ports_);

        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            *(equalized_symbols_  + transmitter) = conj(q_matrix[transmitter][0]) * pdsch_samples[0][re];
        }
        for(int transmitter = 0; transmitter < nb_tx_dmrs_ports_; transmitter++) {
            for(int receiver = 1; receiver < nb_rx_ports_; receiver++) {
                *(equalized_symbols_  + transmitter) += conj(q_matrix[transmitter][receiver]) * pdsch_samples[receiver][re];
            }
        }

        /// SIC detector
        /// Decode the first symbol
        /// remove contribution of r(i,i)
        *(equalized_symbols_  + nb_tx_dmrs_ports_ - 1) /= r_matrix[nb_tx_dmrs_ports_ - 1][nb_tx_dmrs_ports_ - 1].real();
        /// Slicing
        *(detected_symbols_  + nb_tx_dmrs_ports_ - 1) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + nb_tx_dmrs_ports_ - 1));

        /// Remaining iterations
        for(int i = nb_tx_dmrs_ports_ - 2; i > - 1; i--) {
            /// remove contribution of previously detected symbols
            for(int k =  i + 1; k < nb_tx_dmrs_ports_; k++) {
                *(equalized_symbols_  + i)  -= r_matrix[i][k] * constellation_symbols[*(detected_symbols_ + k)];
            }
            /// Equalization / remove contribution of r(i,i)
            *(equalized_symbols_  + i) /= r_matrix[i][i].real();

            /// Slicing
            *(detected_symbols_  + i) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + i));
        }

        equalized_symbols_ += nb_tx_dmrs_ports_;
        detected_symbols_ += nb_tx_dmrs_ports_;
    }
}

/** QR decomp decoder with column norm reordering before QR decomposition.
 *  Reordering is performed on each RE.
 *
 * @param channel_coefficients_
 * @param pdsch_samples
 * @param squared_norms
 * @param num_re_pdsch_
 * @param equalized_symbols_
 * @param nb_tx_dmrs_ports_
 * @param nb_rx_ports_
 * @param constellation_symbols
 * @param detected_symbols_
 * @param constellation_type_
 */
void mimo_vblast_qrd_col_norm(const vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                               const vector<vector<complex<float>>> &pdsch_samples,
                               int num_re_pdsch_,
                               complex<float> * equalized_symbols_,
                               int nb_tx_dmrs_ports_,
                               int nb_rx_ports_,
                               complex<float> * constellation_symbols,
                               int * detected_symbols_,
                               int constellation_type_) {

    complex<float> q_matrix[MAX_TX_PORTS][MAX_RX_PORTS];
    complex<float> r_matrix[MAX_TX_PORTS][MAX_TX_PORTS];
    vector<float> temp_squared_norms(MAX_TX_PORTS);
    vector<float> temp_reordered_squared_norms(MAX_TX_PORTS);
    vector<int> detection_order(nb_tx_dmrs_ports_);
    int i, j;

    for(int re = 0; re < num_re_pdsch_; re++) {

        /// Compute the column norms
        std::fill(temp_squared_norms.begin(), temp_squared_norms.end(), 0);
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 0; j < nb_rx_ports_; j++) {
                temp_squared_norms[i] += channel_coefficients_[i][j][re].real() * channel_coefficients_[i][j][re].real() +
                                         channel_coefficients_[i][j][re].imag() * channel_coefficients_[i][j][re].imag();
            }
        }

        /// Detection reodering based on column norms
        iota(detection_order.begin(), detection_order.begin() + nb_tx_dmrs_ports_, 0);
        std::sort(detection_order.begin(), detection_order.end(), [&temp_squared_norms](int i1, int i2)
        {
            return temp_squared_norms[i1] < temp_squared_norms[i2];
        });
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            temp_reordered_squared_norms[i] = temp_squared_norms[detection_order[i]];
        }

        /// Load the reordered channel coefficients into the Q matrix
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 0; j < nb_rx_ports_; j++) {
                q_matrix[i][j] = channel_coefficients_[detection_order[i]][j][re];
            }
        }

        /// compute QR decomposition
        compute_qr_decomp(q_matrix,
                          r_matrix,
                          temp_reordered_squared_norms.data(),
                          nb_tx_dmrs_ports_,
                          nb_rx_ports_);

        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            *(equalized_symbols_  + detection_order[i]) = conj(q_matrix[i][0]) * pdsch_samples[0][re];
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 1; j < nb_rx_ports_; j++) {
                *(equalized_symbols_  + detection_order[i]) += conj(q_matrix[i][j]) * pdsch_samples[j][re];
            }
        }

        /// SIC detector
        /// Decode the first symbol
        *(equalized_symbols_  + detection_order[nb_tx_dmrs_ports_ - 1]) /= r_matrix[nb_tx_dmrs_ports_ - 1][nb_tx_dmrs_ports_ - 1].real();

        /// Slicing
        *(detected_symbols_  + detection_order[nb_tx_dmrs_ports_ - 1]) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + detection_order[nb_tx_dmrs_ports_ - 1]));

        /// Remaining iterations
        for(i = nb_tx_dmrs_ports_ - 2; i > - 1; i--) {

            /// remove contribution of previously detected symbols
            for(j =  i + 1; j < nb_tx_dmrs_ports_; j++) {
                *(equalized_symbols_  + detection_order[i]) -= r_matrix[i][j] * constellation_symbols[*(detected_symbols_ + detection_order[j])];
            }

            /// Equalization / remove contribution of r(i,i)
            *(equalized_symbols_  + detection_order[i]) /= r_matrix[i][i].real();

            /// Slicing
            *(detected_symbols_  + detection_order[i]) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + detection_order[i]));
        }

        equalized_symbols_ += nb_tx_dmrs_ports_;
        detected_symbols_ += nb_tx_dmrs_ports_;
    }
}

/** QR decomp decoder with column norm reordering before QR decomposition.
 *  Reordering is performed on each RE.
 *
 * @param channel_coefficients_
 * @param pdsch_samples
 * @param squared_norms
 * @param num_re_pdsch_
 * @param equalized_symbols_
 * @param nb_tx_dmrs_ports_
 * @param nb_rx_ports_
 * @param constellation_symbols
 * @param detected_symbols_
 * @param constellation_type_
 */
void mimo_vblast_qrd_col_norm_2_layers(const vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                       const vector<vector<complex<float>>> &pdsch_samples,
                                       int num_re_pdsch_,
                                       complex<float> * equalized_symbols_,
                                       int nb_tx_dmrs_ports_,
                                       int nb_rx_ports_,
                                       complex<float> * constellation_symbols,
                                       int * detected_symbols_,
                                       int constellation_type_) {

    complex<float> q_matrix[MAX_TX_PORTS][MAX_RX_PORTS];
    complex<float> r_matrix[MAX_TX_PORTS][MAX_TX_PORTS];
    vector<float> temp_squared_norms(MAX_TX_PORTS);
    vector<int> detection_order(nb_tx_dmrs_ports_);
    int i, j;
    float tmp = 0;

    for(int re = 0; re < num_re_pdsch_; re++) {

        /// Compute the column norms
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
                temp_squared_norms[i] = channel_coefficients_[i][0][re].real() * channel_coefficients_[i][0][re].real() +
                             channel_coefficients_[i][0][re].imag() * channel_coefficients_[i][0][re].imag();
            for(j = 1; j < nb_rx_ports_; j++) {
                temp_squared_norms[i] += channel_coefficients_[i][j][re].real() * channel_coefficients_[i][j][re].real() +
                                         channel_coefficients_[i][j][re].imag() * channel_coefficients_[i][j][re].imag();
            }
        }

        /// Detection reodering based on column norms
        if(temp_squared_norms[0] < temp_squared_norms[1]) {
            detection_order[0] = 0;
            detection_order[1] = 1;

            /// Load the reordered channel coefficients into the Q matrix
            for(j = 0; j < nb_rx_ports_; j++) {
                q_matrix[0][j] = channel_coefficients_[0][j][re];
                q_matrix[1][j] = channel_coefficients_[1][j][re];
            }

            /// compute QR decomposition
            compute_qr_decomp(q_matrix,
                              r_matrix,
                              temp_squared_norms.data(),
                              nb_tx_dmrs_ports_,
                              nb_rx_ports_);

            for(i = 0; i < nb_tx_dmrs_ports_; i++) {
                *(equalized_symbols_  + i) = conj(q_matrix[i][0]) * pdsch_samples[0][re];
            }
            for(i = 0; i < nb_tx_dmrs_ports_; i++) {
                for(j = 1; j < nb_rx_ports_; j++) {
                    *(equalized_symbols_  + i) += conj(q_matrix[i][j]) * pdsch_samples[j][re];
                }
            }

        } else {
            detection_order[0] = 1;
            detection_order[1] = 0;
            tmp = temp_squared_norms[1];
            temp_squared_norms[1] = temp_squared_norms[0];
            temp_squared_norms[0] = tmp;
            /// compute QR decomposition
            compute_qr_decomp(q_matrix,
                              r_matrix,
                              temp_squared_norms.data(),
                              nb_tx_dmrs_ports_,
                              nb_rx_ports_);

            for(j = 0; j < nb_rx_ports_; j++) {
                q_matrix[0][j] = channel_coefficients_[1][j][re];
                q_matrix[1][j] = channel_coefficients_[0][j][re];
            }

            for(i = 0; i < nb_tx_dmrs_ports_; i++) {
                *(equalized_symbols_  + detection_order[i]) = conj(q_matrix[i][0]) * pdsch_samples[0][re];
            }
            for(i = 0; i < nb_tx_dmrs_ports_; i++) {
                for(j = 1; j < nb_rx_ports_; j++) {
                    *(equalized_symbols_  + detection_order[i]) += conj(q_matrix[i][j]) * pdsch_samples[j][re];
                }
            }
        }

        /// SIC detector
        /// Decode the first symbol
        *(equalized_symbols_  + detection_order[nb_tx_dmrs_ports_ - 1]) /= r_matrix[nb_tx_dmrs_ports_ - 1][nb_tx_dmrs_ports_ - 1].real();

        /// Slicing
        *(detected_symbols_  + detection_order[nb_tx_dmrs_ports_ - 1]) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + detection_order[nb_tx_dmrs_ports_ - 1]));

        /// Remaining iterations
        for(i = nb_tx_dmrs_ports_ - 2; i > - 1; i--) {

            /// remove contribution of previously detected symbols
            for(j =  i + 1; j < nb_tx_dmrs_ports_; j++) {
                *(equalized_symbols_  + detection_order[i]) -= r_matrix[i][j] * constellation_symbols[*(detected_symbols_ + detection_order[j])];
            }

            /// Equalization / remove contribution of r(i,i)
            *(equalized_symbols_  + detection_order[i]) /= r_matrix[i][i].real();

            /// Slicing
            *(detected_symbols_  + detection_order[i]) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + detection_order[i]));
        }
        equalized_symbols_ += 2;
        detected_symbols_ += 2;
    }
}


/** QR decomp decoder with column norm reordering before QR decomposition.
 *  Reordering is performed on each RE.
 *
 * @param channel_coefficients_
 * @param pdsch_samples
 * @param squared_norms
 * @param num_re_pdsch_
 * @param equalized_symbols_
 * @param nb_tx_dmrs_ports_
 * @param nb_rx_ports_
 * @param constellation_symbols
 * @param detected_symbols_
 * @param constellation_type_
 */
void mimo_vblast_qrd_col_norm_modified(vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                       const vector<vector<complex<float>>> &pdsch_samples,
                                       int num_re_pdsch_,
                                       complex<float> * equalized_symbols_,
                                       int nb_tx_dmrs_ports_,
                                       int nb_rx_ports_,
                                       complex<float> * constellation_symbols,
                                       int * detected_symbols_,
                                       int constellation_type_) {

    complex<float> r_matrix[MAX_TX_PORTS][MAX_TX_PORTS];
    vector<float> temp_squared_norms(MAX_TX_PORTS);
    vector<int> detection_order(nb_tx_dmrs_ports_);

    int i, j;

    std::vector<int> computed(MAX_TX_PORTS);
    bool first;

    float tmp_squared_norm = 0;
    int argmin = 0;

    //complex<float> test_channel_matrix2[MAX_TX_PORTS][MAX_TX_PORTS];

    for(int re = 0; re < num_re_pdsch_; re++) {
        /// Compute the column norms
        std::fill(temp_squared_norms.begin(), temp_squared_norms.end(), 0);
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            temp_squared_norms[i] = channel_coefficients_[i][0][re].real() * channel_coefficients_[i][0][re].real() +
                                           channel_coefficients_[i][0][re].imag() * channel_coefficients_[i][0][re].imag();
            for(j = 1; j < nb_rx_ports_; j++) {
                temp_squared_norms[i] += channel_coefficients_[i][j][re].real() * channel_coefficients_[i][j][re].real() +
                                           channel_coefficients_[i][j][re].imag() * channel_coefficients_[i][j][re].imag();
            }
        }

        /// Compute detection order based on column norm
        iota(detection_order.begin(), detection_order.begin() + nb_tx_dmrs_ports_, 0);
        std::sort(detection_order.begin(), detection_order.begin() + nb_tx_dmrs_ports_, [&temp_squared_norms](int i1, int i2)
        {
            return temp_squared_norms[i1] < temp_squared_norms[i2];
        });

        /**
        std::fill(computed.begin(), computed.begin() + nb_tx_dmrs_ports_, 0);
        for (int i = 0; i < nb_tx_dmrs_ports_; i++) {
            for (int j = 0; j < nb_tx_dmrs_ports_; j++) {
                if (not computed[j]) {
                    if (first) {
                        argmin = j;
                        tmp_squared_norm = temp_squared_norms[j];
                        first = 0;
                        continue;
                    } else {
                        if (temp_squared_norms[j] < tmp_squared_norm) {
                            argmin = j;
                            tmp_squared_norm = temp_squared_norms[j];
                        }
                    }
                }
            }
            first = true;
            computed[argmin] = 1;
            detection_order[i] = argmin;
        } */

        /// compute QR decomposition
        compute_qr_decomp_col_norm_reordering(channel_coefficients_,
                                              r_matrix,
                                              temp_squared_norms.data(),
                                              detection_order.data(),
                                              nb_tx_dmrs_ports_,
                                              nb_rx_ports_,
                                              re);

        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            *(equalized_symbols_  + i) = conj(channel_coefficients_[i][0][re]) * pdsch_samples[0][re];
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 1; j < nb_rx_ports_; j++) {
                *(equalized_symbols_  + i) += conj(channel_coefficients_[i][j][re]) * pdsch_samples[j][re];
            }
        }

         argmin = detection_order[nb_tx_dmrs_ports_ - 1];
         *(equalized_symbols_  + argmin) /=
                 r_matrix[argmin][argmin].real();
        /// Slicing
        *(detected_symbols_  + argmin) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + argmin));
        /// Remaining iterations
        for(i = nb_tx_dmrs_ports_ - 2; i > - 1; i--) {
            argmin = detection_order[i];

            /// remove contribution of previously detected symbols
            for(j =  i + 1; j < nb_tx_dmrs_ports_; j++) {
                *(equalized_symbols_ + argmin)  -= r_matrix[argmin][detection_order[j]] * constellation_symbols[*(detected_symbols_ + detection_order[j])];
            }

            /// Equalization / remove contribution of r(i,i)
            *(equalized_symbols_  + argmin) /= r_matrix[argmin][argmin].real();

            /// Slicing
            *(detected_symbols_  + argmin) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + argmin));
        }
        equalized_symbols_ += nb_tx_dmrs_ports_;
        detected_symbols_ += nb_tx_dmrs_ports_;
    }
}

void mimo_vblast_qrd_col_norm_modified_2_layers(vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                               const vector<vector<complex<float>>> &pdsch_samples,
                                               int num_re_pdsch_,
                                               complex<float> * equalized_symbols_,
                                               int nb_tx_dmrs_ports_,
                                               int nb_rx_ports_,
                                               complex<float> * constellation_symbols,
                                               int * detected_symbols_,
                                               int constellation_type_) {

    complex<float> r_matrix[2][2];
    vector<float> temp_squared_norms(2);
    vector<int> detection_order(2);

    int i, j;

    std::vector<int> computed(MAX_TX_PORTS);
    bool first;

    int argmin = 0;

    for(int re = 0; re < num_re_pdsch_; re++) {
        /// Compute the column norms
            temp_squared_norms[0] = channel_coefficients_[0][0][re].real() * channel_coefficients_[0][0][re].real() +
                                           channel_coefficients_[0][0][re].imag() * channel_coefficients_[0][0][re].imag();
            temp_squared_norms[1] = channel_coefficients_[1][0][re].real() * channel_coefficients_[1][0][re].real() +
                               channel_coefficients_[1][0][re].imag() * channel_coefficients_[1][0][re].imag();
            for(j = 1; j < nb_rx_ports_; j++) {
                temp_squared_norms[0] += channel_coefficients_[0][j][re].real() * channel_coefficients_[0][j][re].real() +
                                           channel_coefficients_[0][j][re].imag() * channel_coefficients_[0][j][re].imag();
                temp_squared_norms[1] += channel_coefficients_[1][j][re].real() * channel_coefficients_[1][j][re].real() +
                           channel_coefficients_[1][j][re].imag() * channel_coefficients_[1][j][re].imag();
            }

        /// Compute detection order based on column norm
        compute_qr_decomp_2_layers(channel_coefficients_,
                                   r_matrix,
                                   temp_squared_norms.data(),
                                   detection_order.data(),
                                   nb_tx_dmrs_ports_,
                                   nb_rx_ports_,
                                   re);

        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            *(equalized_symbols_  + i) = conj(channel_coefficients_[i][0][re]) * pdsch_samples[0][re];
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 1; j < nb_rx_ports_; j++) {
                *(equalized_symbols_  + i) += conj(channel_coefficients_[i][j][re]) * pdsch_samples[j][re];
            }
        }

        if(detection_order[0] == 0) {
            *(equalized_symbols_  + 1) /= r_matrix[1][1].real();
            /// Slicing
            *(detected_symbols_  + 1) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + 1));

            *(equalized_symbols_ + 0)  -= r_matrix[0][1] * constellation_symbols[*(detected_symbols_ + 1)];

            /// Equalization / remove contribution of r(i,i)
            *(equalized_symbols_  + 0) /= r_matrix[0][0].real();

            /// Slicing
            *(detected_symbols_  + 0) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + 0));
        } else {
            *(equalized_symbols_  + 0) /= r_matrix[0][0].real();
            /// Slicing
            *(detected_symbols_  + 0) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + 0));

            *(equalized_symbols_ + 1)  -= r_matrix[1][0] * constellation_symbols[*(detected_symbols_ + 0)];

            /// Equalization / remove contribution of r(i,i)
            *(equalized_symbols_  + 1) /= r_matrix[1][1].real();

            /// Slicing
            *(detected_symbols_  + 1) = ml_detector_inline[constellation_type_](*(equalized_symbols_ + 1));
        }
        equalized_symbols_ += 2;
        detected_symbols_ += 2;
    }
}

#if defined(__AVX2__)
void mimo_vblast_qrd_col_norm_avx2(vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                   const vector<vector<complex<float>>> &pdsch_samples,
                                   int num_re_pdsch_,
                                   complex<float> * equalized_symbols_,
                                   int nb_tx_dmrs_ports_,
                                   int nb_rx_ports_,
                                   complex<float> * constellation_symbols,
                                   int * detected_symbols_,
                                   int constellation_type_) {

    __m256 conj_vec = {1, -1, 1, -1, 1, -1, 1, -1};

    /*
    complex<float> q_matrix_debug[MAX_TX_PORTS][MAX_RX_PORTS];
    complex<float> r_matrix_debug[MAX_TX_PORTS][MAX_TX_PORTS];
    complex<float> equalized_symbols_debug[MAX_TX_PORTS];
    int detected_symbols_debug[MAX_TX_PORTS];
    vector<float> temp_squared_norms_debug(MAX_TX_PORTS);
    vector<float> temp_reordered_squared_norms_debug(MAX_TX_PORTS);
    */

    /*
    vector<int> detection_order_debug(nb_tx_dmrs_ports_); */

    __m256 r_matrix[MAX_TX_PORTS][MAX_TX_PORTS];
    __m256 q_matrix_transposed[MAX_TX_PORTS][MAX_RX_PORTS];
    __m256 squared_norms[MAX_TX_PORTS];
    __m256 reordered_squared_norms[MAX_TX_PORTS];
    __m256 temp_equalized_symbols[MAX_TX_PORTS];
    __m128i temp_detected_symbol_indexes[MAX_TX_PORTS];
    __m256 temp_detected_symbols[MAX_TX_PORTS];

    __m256 dot_prod_re, dot_prod_im, vec1, vec2;
    __m128 vec1_128, vec2_128;

    vector<int> detection_reversed_order(MAX_TX_PORTS);
    int i, j;

    float mean_squared_norms[MAX_TX_PORTS]; /// Squared norms of columns in Q averaged over 4 REs.

    for(int re = 0; re < num_re_pdsch_; re+= 4) {

        /// Compute squared norms
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            squared_norms[i] = _mm256_set1_ps(0);
            /// Compute the square norm of the column
            for(j = 0; j < nb_rx_ports_; j++) {
                squared_norms[i] = _mm256_add_ps(squared_norms[i], compute_norm_m256(_mm256_loadu_ps((float *) &channel_coefficients_[i][j][re])));
            }
        }

        /// Compute mean of squared norms on all REs
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            vec1 = _mm256_permute_ps(squared_norms[i], 0b11011000);
            vec1 = _mm256_hadd_ps(vec1, vec1);
            vec1_128 = _mm256_castps256_ps128(vec1);
            vec2_128 = _mm256_extractf128_ps(vec1, 1);
            mean_squared_norms[i] = _mm_cvtss_f32(_mm_add_ps(vec1_128, vec2_128));
        }

        /// Compute detection order
        iota(detection_reversed_order.begin(), detection_reversed_order.begin() + nb_tx_dmrs_ports_, 0);
        std::sort(detection_reversed_order.begin(), detection_reversed_order.begin() + nb_tx_dmrs_ports_, [&mean_squared_norms](int i1, int i2)
        {
            return mean_squared_norms[i1] < mean_squared_norms[i2];
        });
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            reordered_squared_norms[i] = squared_norms[detection_reversed_order[i]];
        }

        /*
        cout << "detection order AVX2 : " << endl;
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            cout << detection_reversed_order[i];
        }
        cout << endl; */

        /// Load coefs in Q matrix
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 0; j < nb_rx_ports_; j++) {
                q_matrix_transposed[i][j] = _mm256_loadu_ps((float *) &channel_coefficients_[detection_reversed_order[i]][j][re]);
            }
        }

        /// Compute QR decomp
        compute_qr_decomp(q_matrix_transposed,
                          r_matrix,
                          reordered_squared_norms,
                          nb_tx_dmrs_ports_,
                          nb_rx_ports_);

        /***************************************
        /// Compute the column norms
        std::fill(temp_squared_norms_debug.begin(), temp_squared_norms_debug.end(), 0);
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 0; j < nb_rx_ports_; j++) {
                temp_squared_norms_debug[i] += channel_coefficients_[i][j][re].real() * channel_coefficients_[i][j][re].real() +
                                         channel_coefficients_[i][j][re].imag() * channel_coefficients_[i][j][re].imag();
            }
        }

        /// Detection reodering based on column norms
        iota(detection_order_debug.begin(), detection_order_debug.begin() + nb_tx_dmrs_ports_, 0);
        std::sort(detection_order_debug.begin(), detection_order_debug.end(), [&temp_squared_norms_debug](int i1, int i2)
        {
            return temp_squared_norms_debug[i1] < temp_squared_norms_debug[i2];
        });
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            temp_reordered_squared_norms_debug[i] = temp_squared_norms_debug[detection_order_debug[i]];
        }

        cout << "detection order : " << endl;
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            cout << detection_order_debug[i];
        }
        cout << endl;

        /// Load the reordered channel coefficients into the Q matrix
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 0; j < nb_rx_ports_; j++) {
                q_matrix_debug[i][j] = channel_coefficients_[detection_order_debug[i]][j][re];
            }
        }

        /// compute QR decomposition
        compute_qr_decomp(q_matrix_debug,
                          r_matrix_debug,
                          temp_reordered_squared_norms_debug.data(),
                          nb_tx_dmrs_ports_,
                          nb_rx_ports_);

        cout << "Q matrix AVX2 : " << endl;
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 0; j < nb_rx_ports_; j++) {
                cout << "(" << q_matrix_transposed[i][j][0] << "," << q_matrix_transposed[i][j][1] << ")";
            }
            cout << endl;
        }

        cout << "Q matrix : " << endl;
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 0; j < nb_rx_ports_; j++) {
                cout << q_matrix_debug[i][j];
            }
            cout << endl;
        }

        cout << "R matrix AVX2 : " << endl;
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 0; j < nb_tx_dmrs_ports_; j++) {
                cout << "(" << r_matrix[i][j][0] << "," << r_matrix[i][j][1] << ")";
            }
            cout << endl;
        }

        cout << "R matrix : " << endl;
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 0; j < nb_tx_dmrs_ports_; j++) {
                cout << r_matrix_debug[i][j];
            }
            cout << endl;
        }

        /****************************************/

        /// Multiply received vector by transconj(Q)
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            dot_prod_re = _mm256_set1_ps(0);
            dot_prod_im = _mm256_set1_ps(0);
            for (j = 0; j < nb_rx_ports_; j++) {
                vec2 = _mm256_loadu_ps((float *) &pdsch_samples[j][re]);
                dot_prod_re = _mm256_add_ps(dot_prod_re, _mm256_mul_ps(q_matrix_transposed[i][j], vec2));
                dot_prod_im = _mm256_add_ps(dot_prod_im, _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(q_matrix_transposed[i][j], conj_vec), 0b10110001), vec2));
            }
            temp_equalized_symbols[i] = _mm256_permute_ps(_mm256_hadd_ps(dot_prod_re, dot_prod_im), 0b11011000);
        }

        /****************************
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            *(equalized_symbols_debug  + detection_order_debug[i]) = conj(q_matrix_debug[i][0]) * pdsch_samples[0][re];
        }
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            for(j = 1; j < nb_rx_ports_; j++) {
                *(equalized_symbols_debug  + detection_order_debug[i]) += conj(q_matrix_debug[i][j]) * pdsch_samples[j][re];
            }
        }

        cout << "preequalized symbols AVX2 : " << endl;
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            cout << "(" << temp_equalized_symbols[i][0] << "," << temp_equalized_symbols[i][1] << ")";
        }
        cout << endl;
        cout << "preequalized symbols : " << endl;
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            cout << equalized_symbols_debug[i];
        }
        cout << endl;
        /*****************************/

        /// Perform SIC
        for(i = nb_tx_dmrs_ports_ - 1; i > -1; i--) {
            /// Suppress interference
            for(j = i + 1; j < nb_tx_dmrs_ports_; j++) {
                temp_equalized_symbols[i] = _mm256_sub_ps(temp_equalized_symbols[i],
                                                                       multiply_complex_float(r_matrix[i][j],
                                                                                              temp_detected_symbols[j]));
            }

            /// Divide by diagonal coef in R
            temp_equalized_symbols[i] = _mm256_div_ps(temp_equalized_symbols[i], r_matrix[i][i]);

            /// Slicing
            ml_detector_mm256(temp_equalized_symbols[i],
                              temp_detected_symbol_indexes[i],
                              temp_detected_symbols[i],
                              constellation_type_);

            /// Store in final buffer
            for(j = 0; j < 4; j++) {
                (equalized_symbols_ + j * nb_tx_dmrs_ports_ + detection_reversed_order[i])->real(temp_equalized_symbols[i][2 * j]);
                (equalized_symbols_ + j * nb_tx_dmrs_ports_ + detection_reversed_order[i])->imag(temp_equalized_symbols[i][2 * j + 1]);
                *(detected_symbols_ + j * nb_tx_dmrs_ports_ + detection_reversed_order[i]) = temp_detected_symbol_indexes[i][j];
            }
        }

        /****************************************
        /// SIC detector
        /// Decode the first symbol
        *(equalized_symbols_debug  + detection_order_debug[nb_tx_dmrs_ports_ - 1]) /= r_matrix_debug[nb_tx_dmrs_ports_ - 1][nb_tx_dmrs_ports_ - 1].real();

        /// Slicing
        *(detected_symbols_debug  + detection_order_debug[nb_tx_dmrs_ports_ - 1]) = ml_detector_inline[constellation_type_](*(equalized_symbols_debug + detection_order_debug[nb_tx_dmrs_ports_ - 1]));

        /// Remaining iterations
        for(i = nb_tx_dmrs_ports_ - 2; i > - 1; i--) {

            /// remove contribution of previously detected symbols
            for(j =  i + 1; j < nb_tx_dmrs_ports_; j++) {
                *(equalized_symbols_debug  + detection_order_debug[i]) -= r_matrix_debug[i][j] * constellation_symbols[*(detected_symbols_debug + detection_order_debug[j])];
            }

            /// Equalization / remove contribution of r(i,i)
            *(equalized_symbols_debug  + detection_order_debug[i]) /= r_matrix_debug[i][i].real();

            /// Slicing
            *(detected_symbols_debug  + detection_order_debug[i]) = ml_detector_inline[constellation_type_](*(equalized_symbols_debug + detection_order_debug[i]));
        }

        cout << "equalized symbols AVX : ";
        for(i = 0; i < 4; i++) {
            cout << *(equalized_symbols_ + i);
        } cout << endl;
        cout << "equalized symbols : ";
        for(i = 0; i < 4; i++) {
            cout << *(equalized_symbols_debug + i);
        } cout << endl;

        /*****************************/

        equalized_symbols_ += 4 * nb_tx_dmrs_ports_;
        detected_symbols_  += 4 * nb_tx_dmrs_ports_;
    }

}
#endif

#if defined(__AVX2__)
void mimo_vblast_qrd_col_norm_avx2_2_layers(vector<complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                            const vector<vector<complex<float>>> &pdsch_samples,
                                            int num_re_pdsch_,
                                            complex<float> * equalized_symbols_,
                                            int nb_tx_dmrs_ports_,
                                            int nb_rx_ports_,
                                            complex<float> * constellation_symbols,
                                            int * detected_symbols_,
                                            int constellation_type_) {

    __m256 conj_vec = {1, -1, 1, -1, 1, -1, 1, -1};

    vector<int> detection_order_debug(nb_tx_dmrs_ports_);

    __m256 r_matrix[MAX_TX_PORTS][MAX_TX_PORTS];
    __m256 q_matrix_transposed[MAX_TX_PORTS][MAX_RX_PORTS];
    __m256 squared_norms[MAX_TX_PORTS];
    __m256 temp_equalized_symbols[MAX_TX_PORTS];
    __m128i temp_detected_symbol_indexes[MAX_TX_PORTS];
    __m256 temp_detected_symbols[MAX_TX_PORTS];

    __m256 dot_prod_re, dot_prod_im, vec1, vec2;
    __m128 vec1_128, vec2_128;

    vector<int> detection_reversed_order(2);
    int i, j;

    float mean_squared_norms[MAX_TX_PORTS]; /// Squared norms of columns in Q averaged over 4 REs.

    int current_symbol = 0;

    for(int re = 0; re < num_re_pdsch_; re+= 4) {

        /// Load channel coefs in q_matrix and compute the squared norms
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            /// Compute the square norm of the column
            q_matrix_transposed[i][0] = _mm256_loadu_ps((float *) &channel_coefficients_[i][0][re]);
            squared_norms[i] = compute_norm_m256(q_matrix_transposed[i][0]);
            for(j = 1; j < nb_rx_ports_; j++) {
                q_matrix_transposed[i][j] = _mm256_loadu_ps((float *) &channel_coefficients_[i][j][re]);
                squared_norms[i] = _mm256_add_ps(squared_norms[i], compute_norm_m256(q_matrix_transposed[i][j]));
            }
        }

        /// Compute QR decomp
        compute_qr_decomp_2_layers(q_matrix_transposed,
                                  r_matrix,
                                  squared_norms,
                                  detection_reversed_order.data(),
                                  nb_rx_ports_);

        /// Multiply received vector by transconj(Q)
        for(i = 0; i < nb_tx_dmrs_ports_; i++) {
            dot_prod_re = _mm256_set1_ps(0);
            dot_prod_im = _mm256_set1_ps(0);
            for (j = 0; j < nb_rx_ports_; j++) {
                vec2 = _mm256_loadu_ps((float *) &pdsch_samples[j][re]);
                dot_prod_re = _mm256_add_ps(dot_prod_re, _mm256_mul_ps(q_matrix_transposed[i][j], vec2));
                dot_prod_im = _mm256_add_ps(dot_prod_im, _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(q_matrix_transposed[i][j], conj_vec), 0b10110001), vec2));
            }
            temp_equalized_symbols[i] = _mm256_permute_ps(_mm256_hadd_ps(dot_prod_re, dot_prod_im), 0b11011000);
        }

        /// Perform SIC
        for(i = nb_tx_dmrs_ports_ - 1; i > -1; i--) {
            current_symbol = detection_reversed_order[i];

            /// Suppress interference
            for(j = i + 1; j < nb_tx_dmrs_ports_; j++) {
                temp_equalized_symbols[current_symbol] = _mm256_sub_ps(temp_equalized_symbols[current_symbol],
                                                                       multiply_complex_float(r_matrix[current_symbol][detection_reversed_order[j]],
                                                                                              temp_detected_symbols[j]));
            }

            /// Divide by diagonal coef in R
            temp_equalized_symbols[current_symbol] = _mm256_div_ps(temp_equalized_symbols[current_symbol], r_matrix[current_symbol][current_symbol]);

            /// Slicing
            ml_detector_mm256(temp_equalized_symbols[current_symbol],
                              temp_detected_symbol_indexes[i],
                              temp_detected_symbols[i],
                              constellation_type_);

            /// Store in final buffer
            for(j = 0; j < 4; j++) {
                (equalized_symbols_ + j * 2 + current_symbol)->real(temp_equalized_symbols[current_symbol][2 * j]);
                (equalized_symbols_ + j * 2 + current_symbol)->imag(temp_equalized_symbols[current_symbol][2 * j + 1]);
                *(detected_symbols_ + j * 2 + current_symbol) = temp_detected_symbol_indexes[i][j];
            }
        }


        equalized_symbols_ += 8;
        detected_symbols_  += 8;
    }
}
#endif