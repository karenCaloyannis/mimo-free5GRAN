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

#ifndef VBLAST_H
#define VBLAST_H

#include <string>
#include <iostream>
#include <complex>
#include <vector>
#include <iostream>
#include <complex>
#include <chrono>
#include <numeric>
#include <omp.h>

#if defined(__AVX2__)
#include <x86intrin.h>
#include <immintrin.h>
#endif

#include <boost/log/core.hpp>
#include <boost/log/utility/setup/file.hpp>
#include <boost/log/trivial.hpp>

#include <complex.h>
#include "time.h"

#include "../../free5gran_utils.h"
#include "../../variables/variables.h"
#include "../channel_estimation/channel_estimation.h"
#include "../../avx/avx_ops.h"

/** Used for debugging purposes only, when ZF decoder is segmented.
 *  Computes the hermitian of the channel matrix using AVX2 optimizations.
 *
 * @param[in] channel_coefficients_ : the channel coefficients
 * @param[in, out] hermitian_matrix : the output hermitian matrices for each RE
 * @param[in] num_re_pdsch_ : Number of PDSCH REs
 * @param[in] nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param[in] nb_rx_ports_ : Number of RX ports
 */
#if defined(__AVX2_)
void compute_hermitian_matrix_avx(std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                                  std::vector<std::complex<float>> hermitian_matrix[MAX_TX_PORTS][MAX_TX_PORTS],
                                  int num_re_pdsch_,
                                  int nb_tx_dmrs_ports_,
                                  int nb_rx_ports_);
#endif

/** Used for debugging purposes only, whe ZF decoder is segmented.
 *  Computes the hermitian of the channel matrix.
 * @param[in] channel_coefficients_ : the channel coefficients
 * @param[in, out] hermitian_matrix : the output hermitian matrices for each RE
 * @param[in] num_re_pdsch_ : Number of PDSCH REs
 * @param[in] nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param[in] nb_rx_ports_ : Number of RX ports
 */
void compute_hermitian_matrix(std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                              std::complex<float> hermitian_matrix[][MAX_RX_PORTS][MAX_TX_PORTS],
                              int num_re_pdsch_,
                              int nb_tx_dmrs_ports_,
                              int nb_rx_ports_);

/** Used for debugging purpose only when ZF decoder is segmented.
 *  Multiplies transconjugate of H by the received PDSCH samples.
 *
 * @param[in] pdsch_samples_ : the PDSCH samples on each receive port
 * @param[in] channel_coefficients_ : channel coefficients on each RE
 * @param[in] num_re_pdsch_ : Number of PDSCH REs
 * @param[in, out] equalized_symbols_ : the output symbols
 * @param[in] nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param[in] nb_rx_ports_ : Number of RX ports
 */
#if defined(__AVX2_)
void multiply_by_transconj_avx(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                           std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                           int num_re_pdsch_,
                           std::vector<std::complex<float>> equalized_symbols_[MAX_TX_PORTS],
                           int nb_tx_dmrs_ports_,
                           int nb_rx_ports_);
#endif

/** Used for debugging purpose only when ZF decoder is segmented.
 *  Multiplies transconjugate of H by the received PDSCH samples.
 *
 * @param[in] pdsch_samples_ : the PDSCH samples on each receive port
 * @param[in] channel_coefficients_ : channel coefficients on each RE
 * @param[in] num_re_pdsch_ : Number of PDSCH REs
 * @param[in, out] equalized_symbols_ : the output symbols
 * @param[in] nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param[in] nb_rx_ports_ : Number of RX ports
 */
void multiply_by_transconj(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                           std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                           int num_re_pdsch_,
                           std::complex<float> * equalized_symbols_,
                           int nb_tx_dmrs_ports_,
                           int nb_rx_ports_);

#if defined(__AVX2__)
/** Used for debugging purpose only when ZF decoder is segmented.
 *  Computes the LDL decomp of the hermitian matrix in row-major order, using AVX2 optimizations.
 *  Performs the decomposition in-place inside hermitian_matrix_
 * @param hermitian_matrix_ : the hermitian matrix
 * @param num_re_pdsch_ : Number of REs to be computed
 * @param nb_tx_dmrs_ports_ : Number of TX ports
 */
void vblast_compute_rdr_decomp_avx(std::vector<std::complex<float>> hermitian_matrix_[MAX_TX_PORTS][MAX_TX_PORTS],
                                   int num_re_pdsch_,
                                   int nb_tx_dmrs_ports_);

/** Used for debugging purpose only when ZF decoder is segmented.
 *  Computes the inverse of the hermitian matrix in row-major order, using AVX2 optimizations.
 *  Performs the inversion in-place inside hermitian_matrix_.
 *  R^(-1) is stored in the upper part of H, D on the diagonal and R^(-h) in the lower part.
 * @param hermitian_matrix_ : the hermitian matrix
 * @param num_re_pdsch_ : Number of REs to be computed
 * @param nb_tx_dmrs_ports_ : Number of TX ports
 */
void vblast_compute_inverse_avx(std::vector<std::complex<float>> hermitian_matrix_[MAX_TX_PORTS][MAX_TX_PORTS],
                                int num_re_pdsch_,
                                int nb_tx_dmrs_ports_);

/** Used for debugging purpose only when ZF decoder is segmented.
 *  Computes the LDL decomp of the hermitian matrix in row-major order, using AVX2 optimizations.
 *  Performs the decomposition in-place inside hermitian_matrix_.
 *
 *  Uses an array of __m256 instead of separate vectors as done in the analog none v2 function
 *
 * @param hermitian_matrix_ : the hermitian matrix
 * @param num_re_pdsch_ : Number of REs to be computed
 * @param nb_tx_dmrs_ports_ : Number of TX ports
 */
void vblast_compute_rdr_decomp_avx_v2(std::vector<std::complex<float>> hermitian_matrix_[MAX_TX_PORTS][MAX_TX_PORTS],
                                      int num_re_pdsch_,
                                      int nb_tx_dmrs_ports_);
#endif

/** Used for debugging purposes only when ZF decoder is segmented.
 *  Copies the realigned equalized symbols into the final buffer.
 *
 * @param num_re_pdsch_ : Number of REs
 * @param temp_equalized_symbols_ : Realigned buffer of equalized symbols used for AVX2 optimizations
 * @param equalized_symbols_ : final buffer
 * @param nb_tx_dmrs_ports_ : Number of TX ports
 */
void vblast_copy_to_equalized_symbols(int num_re_pdsch_,
                                      std::vector<std::complex<float>> temp_equalized_symbols_[MAX_TX_PORTS],
                                      std::complex<float> * equalized_symbols_,
                                      int nb_tx_dmrs_ports_);

#if defined(__AVX2__)
/** Used for debugging purporses only, when ZF decoder is segmented.
 *  Performs the Zero-forcing equalization for 4 layers based on the computed inverses
 *  in hermitian_matrix_, using AVX2 optimizations.
 *
 * @param channel_coefficients_ : the matrix of channel coefficients on each RE
 * @param hermitian_matrix_ : the matrix containing R^(-1) in the upper part, D on the diagonal and R^(-h) in the lower part
 * @param num_re_pdsch_ : Number of REs to be computed
 * @param temp_equalized_symbols_ : Vector of equalized symbols used for AVX2 optimizations
 * @param equalized_symbols_ : final buffer of equalized symbols.
 * @param nb_rx_ports_ : Number of receive ports.
 */
void vblast_zf_4_layers_avx(std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                        std::vector<std::complex<float>> hermitian_matrix_[MAX_TX_PORTS][MAX_TX_PORTS],
                        int num_re_pdsch_,
                        std::vector<std::complex<float>> temp_equalized_symbols_[MAX_TX_PORTS],
                        std::complex<float> * equalized_symbols_,
                        int nb_rx_ports_);
#endif

/*************************** Test function, not currently used *********************************/

/** Performs ZF equalization for 4 REs with pre-computed hermitian matrices.
 *
 * @param pdsch_samples_ : received PDSCH samples on each RX port
 * @param hermitian_matrix_ : the pre-computed hermitian matrices
 * @param num_re_pdsch_ : Number of PDSCH REs
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_rx_ports_ : Number of RX ports
 */
void vblast_zf_4_layers(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                        std::complex<float> hermitian_matrix_[][MAX_TX_PORTS][MAX_TX_PORTS],
                        int num_re_pdsch_,
                        std::complex<float> * equalized_symbols_,
                        int nb_rx_ports_);

/** Performs ZF equalization for 4 layers with realigned buffers for pdsch_samples
 *  and channel_coefficients_
 *
 * @param pdsch_samples_ : received PDSCH samples on each RX port
 * @param channel_coefficients_ : the matrix of channel coefficient on each RE (first dimension corresponds to RE)
 * @param num_re_pdsch_ : the number of REs to be computed
 * @param equalized_symbols_ : finall buffer of equalized symbols
 * @param nb_rx_ports_ : Number of RX ports
 */
void vblast_zf_4_layers(std::complex<float> pdsch_samples_[][MAX_RX_PORTS],
                        std::complex<float> channel_coefficients_[][MAX_RX_PORTS][MAX_TX_PORTS],
                        int num_re_pdsch_,
                        std::complex<float> * equalized_symbols_,
                        int nb_rx_ports_);
/***********************************************************************************************/

/** Calls the Zero-Forcing decoders based on the number of layers given in input.
 *
 * @param num_layers : number of layers
 * @param pdsch_samples_ : received PDSCH samples
 * @param channel_coefficients_ : matrix of channel coefficients
 * @param num_re_pdsch_ : Number of REs to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_rx_ports_ : Number of RX ports
 */
void call_vblast_zf_functions(int num_layers,
                              const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                              std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                              int num_re_pdsch_,
                              std::complex<float> *equalized_symbols_,
                              int nb_rx_ports_);

/** Calls the Zero-Forcing decoders using AVX2 optimizations
 *  based on the number of layers given in input.
 *
 * @param num_layers : number of layers
 * @param pdsch_samples_ : received PDSCH samples
 * @param channel_coefficients_ : matrix of channel coefficients
 * @param num_re_pdsch_ : Number of REs to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_rx_ports_ : Number of RX ports
 */
#if defined(__AVX2__)
void call_vblast_zf_avx_functions(int num_layers,
                                  const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                                  std::vector<std::complex<float>>
                                  channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                                  int num_re_pdsch_,
                                  std::complex<float> *equalized_symbols_,
                                  int nb_rx_ports_);
#endif

/** Performs Zero-forcing equalization for 4 layers.
 *
 * @param pdsch_samples_ : received PDSCH samples
 * @param channel_coefficients_ : matrix of channel coefficients on each RE
 * @param num_re_pdsch_ : Number of REs to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_rx_ports_ : Number of RX ports
 */
void vblast_zf_4_layers(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                        std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                        int num_re_pdsch_,
                        std::complex<float> * equalized_symbols_,
                        int nb_rx_ports_);

/** Zero-forcing decoder for 4 layers using AVX2 optimizations.
 *
 * @param pdsch_samples_ : PDSCH samples received on each port
 * @param channel_coefficients_ : Matrix of channel coefficients on each RE
 * @param num_re_pdsch_ : Number of REs to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_rx_ports_ : Number of receive ports.
 */
#if defined(__AVX2__)
void vblast_zf_4_layers_avx(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                            std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                            int num_re_pdsch_,
                            std::complex<float> * equalized_symbols_,
                            int nb_rx_ports_);
#endif

/** Performs Zero-forcing for 2 layers.
 *
 * @param pdsch_samples_ : received PDSCH samples
 * @param channel_coefficients_ : matrix of channel coefficients
 * @param num_re_pdsch_ : Number of REs to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_rx_ports_ : Number of RX ports
 */
void vblast_zf_2_layers(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                        std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                        int num_re_pdsch_,
                        std::complex<float> *equalized_symbols_,
                        int nb_rx_ports_);

/** Performs Zero-forcing for 2 layers using AVX2 optimizations.
 *
 * @param pdsch_samples_ : received PDSCH samples
 * @param channel_coefficients_ : matrix of channel coefficients
 * @param num_re_pdsch_ : Number of REs to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_rx_ports_ : Number of RX ports
 */
#if defined(__AVX2__)
void vblast_zf_2_layers_avx(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                            std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                            int num_re_pdsch_,
                            std::complex<float> *equalized_symbols_,
                            int nb_rx_ports_);
#endif

/** Performs Zero-forcing for 3 layers.
 *
 * @param pdsch_samples_ : received PDSCH samples
 * @param channel_coefficients_ : matrix of channel coefficients
 * @param num_re_pdsch_ : Number of REs to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_rx_ports_ : Number of RX ports
 */
void vblast_zf_3_layers(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                        std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                        int num_re_pdsch_,
                        std::complex<float> *equalized_symbols_,
                        int nb_rx_ports_);

/** Performs Zero-forcing for 3 layers using AVX2 optimizations.
 *
 * @param pdsch_samples_ : received PDSCH samples
 * @param channel_coefficients_ : matrix of channel coefficients
 * @param num_re_pdsch_ : Number of REs to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_rx_ports_ : Number of RX ports
 */
#if defined(__AVX2__)
void vblast_zf_3_layers_avx(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                            std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                            int num_re_pdsch_,
                            std::complex<float> * equalized_symbols_,
                            int nb_rx_ports_);
#endif

/** Performs Zero-forcing equalization by real-matrix inversions.
 *  Based on : https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=7443678
 * @param pdsch_samples_ : received PDSCH samples on each RX port
 * @param channel_coefficients_ : matrix of channel coefficients on each RE
 * @param num_re_pdsch_ : Number of PDSCH RE to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols.
 * @param nb_rx_ports_ : Number of RX ports.
 */
void vblast_zf_4_layers_float(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                              std::vector<std::complex<float>>
                              channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                              int num_re_pdsch_,
                              std::complex<float> *equalized_symbols_,
                              int nb_rx_ports_);

/** Performs Zero-forcing equalization by block-wise inversion
 *  Based on : http://www.cs.nthu.edu.tw/~jang/book/addenda/matinv/matinv/
 * @param pdsch_samples_ : received PDSCH samples on each RX port
 * @param channel_coefficients_ : matrix of channel coefficients on each RE
 * @param num_re_pdsch_ : Number of PDSCH RE to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols.
 * @param nb_rx_ports_ : Number of RX ports.
 */
void vblast_4_layers_block_wise_inversion(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                                          std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                                          int num_re_pdsch_,
                                          std::complex<float> *equalized_symbols_,
                                          int nb_rx_ports_);

/** Performs the LDL decomposition in row-major order, in-place inside the Hermitian matrix.
 *  Hermitian matrix is given as a 1-D array and indexes have to be computed manually
 *
 * @param hermitian_matrix_ : the input hermitian matrix
 * @param size_ : the size of the matrix
 */
void ldl_decomp_inplace(float _Complex * hermitian_matrix_, /// Row major
                        int size_);

/** Performs LDL decomposition of a hermitian matrix in row major order, on its real part only.
 *  Performs the decomposition in-place insice hermitian_matrix
 * @param hermitian_matrix_  : the input hermitian matrix
 * @param size_ : the size of the matrix
 */
void inline ldl_decomp_real_part(std::complex<float> hermitian_matrix_[MAX_TX_PORTS][MAX_TX_PORTS],
                                 int size_) {

    if(size_ == 4) {
        /// First row
        hermitian_matrix_[0][1].real(hermitian_matrix_[0][1].real() / hermitian_matrix_[0][0].real());
        hermitian_matrix_[0][2].real(hermitian_matrix_[0][2].real() / hermitian_matrix_[0][0].real());
        hermitian_matrix_[0][3].real(hermitian_matrix_[0][3].real() / hermitian_matrix_[0][0].real());

        /// Second row
        hermitian_matrix_[1][1].real(hermitian_matrix_[1][1].real() - (hermitian_matrix_[0][1].real() * hermitian_matrix_[0][1].real()) * hermitian_matrix_[0][0].real());
        hermitian_matrix_[1][2].real((hermitian_matrix_[1][2].real() - hermitian_matrix_[0][1].real() * hermitian_matrix_[0][2].real() * hermitian_matrix_[0][0].real())/hermitian_matrix_[1][1].real());
        hermitian_matrix_[1][3].real((hermitian_matrix_[1][3].real() - hermitian_matrix_[0][1].real() * hermitian_matrix_[0][3].real() * hermitian_matrix_[0][0].real())/hermitian_matrix_[1][1].real());

        /// Third row
        hermitian_matrix_[2][2].real(hermitian_matrix_[2][2].real()
                                  - (hermitian_matrix_[0][2].real() * hermitian_matrix_[0][2].real()) * hermitian_matrix_[0][0].real()
                                  - (hermitian_matrix_[1][2].real() * hermitian_matrix_[1][2].real()) * hermitian_matrix_[1][1].real());
        hermitian_matrix_[2][3].real((hermitian_matrix_[2][3].real() - hermitian_matrix_[0][2].real() * hermitian_matrix_[0][3].real() * hermitian_matrix_[0][0].real()
                                    - hermitian_matrix_[1][2].real() * hermitian_matrix_[1][3].real() * hermitian_matrix_[1][1].real())/hermitian_matrix_[2][2].real());

        /// Fourth row
        hermitian_matrix_[3][3].real(hermitian_matrix_[3][3].real()
                                  - (hermitian_matrix_[0][3].real() * hermitian_matrix_[0][3].real()) * hermitian_matrix_[0][0].real()
                                  - (hermitian_matrix_[1][3].real() * hermitian_matrix_[1][3].real()) * hermitian_matrix_[1][1].real()
                                  - (hermitian_matrix_[2][3].real() * hermitian_matrix_[2][3].real()) * hermitian_matrix_[2][2].real());
    }
}

/** Performs the in-place LDL decomposition in row-major order of a float matrix.
 *
 * @param symmetric_matrix_ : the input matrix
 * @param size_ : the size of the matrix
 */
void inline ldl_decomp_test(float symmetric_matrix_[MAX_TX_PORTS][MAX_TX_PORTS],
                            int size_) {

    if(size_ == 4) {
        /// First row
        symmetric_matrix_[0][1] /= symmetric_matrix_[0][0];
        symmetric_matrix_[0][2] /= symmetric_matrix_[0][0];
        symmetric_matrix_[0][3] /= symmetric_matrix_[0][0];

        /// Second row
        symmetric_matrix_[1][1] = symmetric_matrix_[1][1] - (symmetric_matrix_[0][1] * symmetric_matrix_[0][1]) * symmetric_matrix_[0][0];
        symmetric_matrix_[1][2] = (symmetric_matrix_[1][2] - symmetric_matrix_[0][1] * symmetric_matrix_[0][2] * symmetric_matrix_[0][0])/symmetric_matrix_[1][1];
        symmetric_matrix_[1][3] = (symmetric_matrix_[1][3] - symmetric_matrix_[0][1] * symmetric_matrix_[0][3] * symmetric_matrix_[0][0])/symmetric_matrix_[1][1];

        /// Third row
        symmetric_matrix_[2][2] = symmetric_matrix_[2][2]
                                - (symmetric_matrix_[0][2] * symmetric_matrix_[0][2]) * symmetric_matrix_[0][0]
                                - (symmetric_matrix_[1][2] * symmetric_matrix_[1][2]) * symmetric_matrix_[1][1];
        symmetric_matrix_[2][3] = (symmetric_matrix_[2][3] - symmetric_matrix_[0][2] * symmetric_matrix_[0][3] * symmetric_matrix_[0][0]
                                 - symmetric_matrix_[1][2] * symmetric_matrix_[1][3] * symmetric_matrix_[1][1])/symmetric_matrix_[2][2];

        /// Fourth row
        symmetric_matrix_[3][3] = symmetric_matrix_[3][3]
                                - (symmetric_matrix_[0][3] * symmetric_matrix_[0][3]) * symmetric_matrix_[0][0]
                                - (symmetric_matrix_[1][3] * symmetric_matrix_[1][3]) * symmetric_matrix_[1][1]
                                - (symmetric_matrix_[2][3] * symmetric_matrix_[2][3]) * symmetric_matrix_[2][2];
    }
}

/** Performs the in-place LDL decomposition in row-major order of a hermitian matrix.
 *
 * @param hermitian_matrix_ : the input hermitian matrix
 * @param size_ : the size of the matrix
 */
void inline ldl_decomp_test(std::complex<float> hermitian_matrix_[MAX_TX_PORTS][MAX_TX_PORTS],
                            int size_) {

    if(size_ == 4) {
        /// First row
        hermitian_matrix_[0][1] /= hermitian_matrix_[0][0].real();
        hermitian_matrix_[0][2] /= hermitian_matrix_[0][0].real();
        hermitian_matrix_[0][3] /= hermitian_matrix_[0][0].real();

        /// Second row
        hermitian_matrix_[1][1] = hermitian_matrix_[1][1].real() - (hermitian_matrix_[0][1].real() * hermitian_matrix_[0][1].real()
                                                                    + hermitian_matrix_[0][1].imag() * hermitian_matrix_[0][1].imag()) * hermitian_matrix_[0][0].real();
        hermitian_matrix_[1][2] = (hermitian_matrix_[1][2] - conj(hermitian_matrix_[0][1]) * hermitian_matrix_[0][2] * hermitian_matrix_[0][0].real())/hermitian_matrix_[1][1].real();
        hermitian_matrix_[1][3] = (hermitian_matrix_[1][3] - conj(hermitian_matrix_[0][1]) * hermitian_matrix_[0][3] * hermitian_matrix_[0][0].real())/hermitian_matrix_[1][1].real();

        //hermitian_matrix_[1][2] = (hermitian_matrix_[1][2] - conj(hermitian_matrix_[0][1]) * hermitian_matrix_[0][2])/hermitian_matrix_[1][1].real();
        //hermitian_matrix_[1][3] = (hermitian_matrix_[1][3] - conj(hermitian_matrix_[0][1]) * hermitian_matrix_[0][3])/hermitian_matrix_[1][1].real();
        //hermitian_matrix_[1][3] = (hermitian_matrix_[1][3] - conj(hermitian_matrix_[0][1]) * hermitian_matrix_[0][3]);

        // Update h02 and h03 after computing second row to avoid computing hermitian_matrix[0][i] * hermitian_matrix[0][0].real()
        //hermitian_matrix_[0][2] /= hermitian_matrix_[0][0].real();
        //hermitian_matrix_[0][3] /= hermitian_matrix_[0][0].real();

        /// Third row
        hermitian_matrix_[2][2] = hermitian_matrix_[2][2].real()
                                  - (hermitian_matrix_[0][2].real() * hermitian_matrix_[0][2].real() +
                                     hermitian_matrix_[0][2].imag() * hermitian_matrix_[0][2].imag()) * hermitian_matrix_[0][0].real()
                                  - (hermitian_matrix_[1][2].real() * hermitian_matrix_[1][2].real() +
                                     hermitian_matrix_[1][2].imag() * hermitian_matrix_[1][2].imag()) * hermitian_matrix_[1][1].real();

        hermitian_matrix_[2][3] = (hermitian_matrix_[2][3]
                                   - conj(hermitian_matrix_[0][2]) * hermitian_matrix_[0][3] * hermitian_matrix_[0][0].real()
                                   - conj(hermitian_matrix_[1][2]) * hermitian_matrix_[1][3] * hermitian_matrix_[1][1].real())
                                  /hermitian_matrix_[2][2].real();

        /*
        hermitian_matrix_[2][3] = (hermitian_matrix_[2][3]
                                    - conj(hermitian_matrix_[0][2]) * hermitian_matrix_[0][3]
                                    - conj(hermitian_matrix_[1][2]) * hermitian_matrix_[1][3])
                                    /hermitian_matrix_[2][2].real(); */

        //hermitian_matrix_[1][3] = (hermitian_matrix_[1][3] - conj(hermitian_matrix_[0][1]) * hermitian_matrix_[0][3]);
        //hermitian_matrix_[0][3] /= hermitian_matrix_[0][0].real();

        /// Fourth row
        hermitian_matrix_[3][3] = hermitian_matrix_[3][3].real()
                                  - (hermitian_matrix_[0][3].real() * hermitian_matrix_[0][3].real() +
                                     hermitian_matrix_[0][3].imag() * hermitian_matrix_[0][3].imag()) * hermitian_matrix_[0][0].real()
                                  - (hermitian_matrix_[1][3].real() * hermitian_matrix_[1][3].real() +
                                     hermitian_matrix_[1][3].imag() * hermitian_matrix_[1][3].imag()) * hermitian_matrix_[1][1].real()
                                  - (hermitian_matrix_[2][3].real() * hermitian_matrix_[2][3].real() +
                                     hermitian_matrix_[2][3].imag() * hermitian_matrix_[2][3].imag()) * hermitian_matrix_[2][2].real();
    } else if (size_ == 3) {
        /// First row
        hermitian_matrix_[0][1] /= hermitian_matrix_[0][0].real();
        hermitian_matrix_[0][2] /= hermitian_matrix_[0][0].real();

        /// Second row
        hermitian_matrix_[1][1] = hermitian_matrix_[1][1].real() - (hermitian_matrix_[0][1].real() * hermitian_matrix_[0][1].real()
        + hermitian_matrix_[0][1].imag() * hermitian_matrix_[0][1].imag()) * hermitian_matrix_[0][0].real();
        hermitian_matrix_[1][2] = (hermitian_matrix_[1][2] - conj(hermitian_matrix_[0][1]) * hermitian_matrix_[0][2] * hermitian_matrix_[0][0].real())/hermitian_matrix_[1][1].real();

        /// Third row
        hermitian_matrix_[2][2] = hermitian_matrix_[2][2].real()
        - (hermitian_matrix_[0][2].real() * hermitian_matrix_[0][2].real() +
        hermitian_matrix_[0][2].imag() * hermitian_matrix_[0][2].imag()) * hermitian_matrix_[0][0].real()
        - (hermitian_matrix_[1][2].real() * hermitian_matrix_[1][2].real() +
        hermitian_matrix_[1][2].imag() * hermitian_matrix_[1][2].imag()) * hermitian_matrix_[1][1].real();
    }
}

/** Performs the LDL decomposition in row-major order of a hermitian matrix and place the result
 *  in the upper triangular matrix R and the diagonal matrix D.
 *
 * @param[in] hermitian_matrix_ : the input hermitian matrix
 * @param[in, out] output_r_ : the output upper triangular matrix R
 * @param[in, out] output_diag_ : the diagonal matrix D
 * @param[in] size_ : the size of the matrix
 */
void ldl_decomp(std::complex<float> * hermitian_matrix_, /// Row major
                std::complex<float> * output_r_, /// Row major upper triangular matrix
                std::complex<float> * output_diag_, /// Diagonal coefficients
                int size_);

/** Performs the LDL decomposition in row-major order of a hermitian matrix and place the result
 *  in the upper triangular matrix R and the diagonal matrix D. (Hardcoded version)
 *
 * @param[in] hermitian_matrix_ : the input hermitian matrix
 * @param[in, out] output_r_ : the output upper triangular matrix R
 * @param[in, out] output_diag_ : the diagonal matrix D
 * @param[in] size_ : the size of the matrix
 */
void ldl_decomp_harcoded(std::complex<float> * hermitian_matrix_, /// Row major
                         std::complex<float> * output_r_, /// Row major upper triangular matrix
                         std::complex<float> * output_diag_, /// Diagonal coefficients
                         int size_);

/** Performs the LDL decomposition in row-major order, in-place inside the Hermitian matrix.
 *  Hermitian matrix is given as a 1-D array and indexes have to be computed manually
 *
 * @param hermitian_matrix_ : the input hermitian matrix
 * @param size_ : the size of the matrix
 */
void ldl_decomp_inplace(std::complex<float> * hermitian_matrix_, /// Row major
                        int size_);

/** Deprecated V-BLAST ZF function that does not call functions for each layer separately.
 *
 * @param pdsch_samples_ : received PDSCH samples on each RX port
 * @param channel_coefficients_ : matrix of channel coefficients
 * @param num_re_pdsch_ : Number of PDSCH REs to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 */
void vblast_zf(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
               std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
               int num_re_pdsch_,
               std::complex<float> *equalized_symbols_,
               int nb_tx_dmrs_ports_,
               int nb_rx_ports_);

/** Matched filter receiver.
 *
 * @param pdsch_samples_ : received PDSCH samples on each RX port
 * @param channel_coefficients_ : matrix of channel coefficients
 * @param num_re_pdsch_ : Number of PDSCH REs to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 */
void vblast_mf(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
               const std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
               const int &num_re_pdsch_,
               std::complex<float> *equalized_symbols_,
               int &nb_tx_dmrs_ports_,
               int &nb_rx_ports_);

#if defined(__AVX2__)
/** SQRD equalization using AVX2 optimizations to perform computations on 4 REs at once.
 *
 * @param channel_coefficients_ : matrix of channel coefficients
 * @param pdsch_samples : received PDSCH samples on each RX port
 * @param num_re_pdsch_ : Number of PDSCH REs
 * @param equalized_symbols_ : buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 * @param constellation_symbols : constellation symbols (see variables.cpp)
 * @param detected_symbols_ : final buffer of detected symbols
 * @param constellation_type_ : type of constellation (see variables.cpp)
 */
void mimo_vblast_sqrd_avx2(std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                        const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                        int num_re_pdsch_,
                                        std::complex<float> * equalized_symbols_,
                                        int nb_tx_dmrs_ports_,
                                        int nb_rx_ports_,
                                        std::complex<float> * constellation_symbols,
                                        int * detected_symbols_,
                                        int constellation_type_);
#endif
#if defined(__AVX2__)
/** QRD-CN equalization using AVX2 optimizations to perform computations on 4 REs at once.
 *
 * @param channel_coefficients_ : matrix of channel coefficients
 * @param pdsch_samples : received PDSCH samples on each RX port
 * @param num_re_pdsch_ : Number of PDSCH REs
 * @param equalized_symbols_ : buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 * @param constellation_symbols : constellation symbols (see variables.cpp)
 * @param detected_symbols_ : final buffer of detected symbols
 * @param constellation_type_ : type of constellation (see variables.cpp)
 */
void mimo_vblast_qrd_col_norm_avx2(std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                    const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                    int num_re_pdsch_,
                                    std::complex<float> * equalized_symbols_,
                                    int nb_tx_dmrs_ports_,
                                    int nb_rx_ports_,
                                    std::complex<float> * constellation_symbols,
                                    int * detected_symbols_,
                                    int constellation_type_);
#endif
#if defined(__AVX2__)
/** Performs Sorted QR decomp. on 4 REs simultanesouly using AVX2
 *
 * @param[in, out] q_matrix_transposed : transposed input Q matrix
 * @param[in, out] r_matrix : untransposed R matrix
 * @param[in] squared_norms_ : precomputed squared norms of each column in Q
 * @param[in, out] detection_reversed_order_ : detection order determined during SQRD
 * @param[in] num_tx_ports_ : Number of TX ports
 * @param[in] num_rx_ports_ : Number of RX ports
 */
void static inline compute_qr_decomp(__m256 q_matrix_transposed[MAX_TX_PORTS][MAX_RX_PORTS],
                                     __m256 r_matrix[MAX_TX_PORTS][MAX_TX_PORTS],
                                     __m256 * squared_norms_,
                                     int * detection_reversed_order_,
                                     int num_tx_ports_,
                                     int num_rx_ports_) {

    int computed[MAX_TX_PORTS];
    memset(&computed, 0, MAX_TX_PORTS * sizeof(int));
    int argmin = 0;
    bool first = 1;
    float temp_squared_norm = 0;
    float mean_squared_norms[MAX_TX_PORTS]; /// Squared norms of columns in Q averaged over 4 REs.

    __m256 vec1;
    __m128 vec1_128, vec2_128;

    /// Compute mean of squared norms
    for(int i = 0; i < num_tx_ports_; i++) {
        vec1 = _mm256_permute_ps(squared_norms_[i], 0b11011000);
        vec1 = _mm256_hadd_ps(vec1, vec1);
        vec1_128 = _mm256_castps256_ps128(vec1);
        vec2_128 = _mm256_extractf128_ps(vec1, 1);
        mean_squared_norms[i] = _mm_cvtss_f32(_mm_add_ps(vec1_128, vec2_128));
    }

    for(int i = 0; i < num_tx_ports_; i++) {
        /// Find the column with lowest norm
        if(i <num_tx_ports_ - 1) { /// Stop if the last column has been computed
            for(int j = 0; j < num_tx_ports_; j++) {
                if(not computed[j]) {
                    if (first) {
                        argmin = j;
                        temp_squared_norm = mean_squared_norms[j];
                        first = 0;
                        continue;
                    } else {
                        if (mean_squared_norms[j] < temp_squared_norm) {
                            argmin = j;
                            temp_squared_norm = mean_squared_norms[j];
                        }
                    }
                }
            }

            computed[argmin] = 1; /// indicate that the current column of Q must not be modified in the next iterations
            first = 1;

        } else {
            for(int j = 0; j < num_tx_ports_; j++) {
                if(not computed[j]) {
                    argmin = j;
                }
            }
        }

        detection_reversed_order_[i] = argmin;

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[argmin][argmin] = _mm256_sqrt_ps(squared_norms_[argmin]);

        /// Normalize column q_argmin
        for(int j = 0; j < num_tx_ports_; j++) {
            q_matrix_transposed[argmin][j] = _mm256_div_ps(q_matrix_transposed[argmin][j], r_matrix[argmin][argmin]);
        }

        /// Project other columns vectors onto q_argmin and orthogonalize.
        if(i < num_tx_ports_ - 1) { /// Stop if the last column has been computed
            for(int j = 0; j < num_tx_ports_; j++) {
                if(not computed[j]) {
                    r_matrix[argmin][j] = conj_multiply_complex_float(q_matrix_transposed[argmin][0], q_matrix_transposed[j][0]);
                    for(int k = 1; k < num_rx_ports_; k++) {
                        r_matrix[argmin][j] = _mm256_add_ps(r_matrix[argmin][j], conj_multiply_complex_float(q_matrix_transposed[argmin][k], q_matrix_transposed[j][k]));
                    }

                    for(int k = 0; k < num_rx_ports_; k++) {
                        q_matrix_transposed[j][k] = _mm256_sub_ps(q_matrix_transposed[j][k], multiply_complex_float(r_matrix[argmin][j], q_matrix_transposed[argmin][k]));
                    }

                    /// Update squared norms
                    squared_norms_[j] = _mm256_sub_ps(squared_norms_[j], compute_norm_m256(r_matrix[argmin][j]));

                    /// Update mean of squared norms
                    vec1 = _mm256_permute_ps(squared_norms_[j], 0b11011000);
                    vec1 = _mm256_hadd_ps(vec1, vec1);
                    vec1_128 = _mm256_castps256_ps128(vec1);
                    vec2_128 = _mm256_extractf128_ps(vec1, 1);
                    mean_squared_norms[j] = _mm_cvtss_f32(_mm_add_ps(vec1_128, vec2_128));
                }
            }
        }
    }
}
#endif
#if defined(__AVX2__)
/** Performs QR decomposition of the transposed Q matrix given in input.
 *  Uses AVX2 optimizations to perform computations on 4 REs at once.
 *
 * @param[in, out] q_matrix_transposed : transposed Q matrix
 * @param[in, out] r_matrix : untransposed R matrix
 * @param[in] squared_norms_ : precomputed squared norms of columns in Q matrix
 * @param[in] num_tx_ports_ : Number of TX ports
 * @param[in] num_rx_ports_ : Number of RX ports
 */
void static inline compute_qr_decomp(__m256 q_matrix_transposed[MAX_TX_PORTS][MAX_RX_PORTS],
                                     __m256 r_matrix[MAX_TX_PORTS][MAX_TX_PORTS],
                                     __m256 * squared_norms_,
                                     int num_tx_ports_,
                                     int num_rx_ports_) {

    __m256 vec1;

    for(int i = 0; i < num_tx_ports_; i++) {

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[i][i] = _mm256_sqrt_ps(squared_norms_[i]);

        /// Normalize column q_argmin
        for(int j = 0; j < num_tx_ports_; j++) {
            q_matrix_transposed[i][j] = _mm256_div_ps(q_matrix_transposed[i][j], r_matrix[i][i]);
        }

        /// Project other columns vectors onto q_argmin and orthogonalize.
        for(int j = i + 1; j < num_tx_ports_; j++) {
                r_matrix[i][j] = conj_multiply_complex_float(q_matrix_transposed[i][0], q_matrix_transposed[j][0]);
                for(int k = 1; k < num_rx_ports_; k++) {
                    r_matrix[i][j] = _mm256_add_ps(r_matrix[i][j], conj_multiply_complex_float(q_matrix_transposed[i][k], q_matrix_transposed[j][k]));
                }

                for(int k = 0; k < num_rx_ports_; k++) {
                    q_matrix_transposed[j][k] = _mm256_sub_ps(q_matrix_transposed[j][k], multiply_complex_float(r_matrix[i][j], q_matrix_transposed[i][k]));
                }

                /// Update squared norms
                squared_norms_[j] = _mm256_sub_ps(squared_norms_[j], compute_norm_m256(r_matrix[i][j]));
        }
    }
}
#endif

/// SQRD for 8 REs simultaneously
#if defined(__AVX512__)
/** Performs SQRD equalization on 8 REs at once using AVX512.
 *
 * @param channel_coefficients_ : matrix of channel coefficients
 * @param pdsch_samples : received PDSCH samples on each RX port
 * @param num_re_pdsch_ : Number of PDSCH REs
 * @param equalized_symbols_ : buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 * @param constellation_symbols : constellation symbols (see variables.cpp)
 * @param detected_symbols_ : final buffer of detected symbols
 * @param constellation_type_ : type of constellation (see variables.cpp)
 */
void mimo_vblast_sqrd_avx512(std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                             const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                             int num_re_pdsch_,
                             std::complex<float> * equalized_symbols_,
                             int nb_tx_dmrs_ports_,
                             int nb_rx_ports_,
                             std::complex<float> * constellation_symbols,
                             int * detected_symbols_,
                             int constellation_type_);

/** Performs unsorted QRD equalization on 8 REs at once using AVX512.
 *
 * @param channel_coefficients_ : matrix of channel coefficients
 * @param pdsch_samples : received PDSCH samples on each RX port
 * @param num_re_pdsch_ : Number of PDSCH REs
 * @param equalized_symbols_ : buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 * @param constellation_symbols : constellation symbols (see variables.cpp)
 * @param detected_symbols_ : final buffer of detected symbols
 * @param constellation_type_ : type of constellation (see variables.cpp)
 */
void mimo_vblast_sqrd_avx512_no_reordering(std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                           const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                           int num_re_pdsch_,
                                           std::complex<float> * equalized_symbols_,
                                           int nb_tx_dmrs_ports_,
                                           int nb_rx_ports_,
                                           std::complex<float> * constellation_symbols,
                                           int * detected_symbols_,
                                           int constellation_type_);

/** Performs Sorted QR decomp. on 8 REs simultanesouly using AVX512
 *
 * @param[in, out] q_matrix_transposed : transposed input Q matrix
 * @param[in, out] r_matrix : untransposed R matrix
 * @param[in] squared_norms_ : precomputed squared norms of each column in Q
 * @param[in, out] detection_reversed_order_ : detection order determined during SQRD
 * @param[in] num_tx_ports_ : Number of TX ports
 * @param[in] num_rx_ports_ : Number of RX ports
 */
void static inline compute_qr_decomp(__m512 q_matrix_transposed[MAX_TX_PORTS][MAX_RX_PORTS],
                                     __m512 r_matrix[MAX_TX_PORTS][MAX_TX_PORTS],
                                     __m512 * squared_norms_,
                                     int * detection_reversed_order_,
                                     int num_tx_ports_,
                                     int num_rx_ports_) {

    int computed[MAX_TX_PORTS];
    memset(&computed, 0, MAX_TX_PORTS * sizeof(int));
    int argmin = 0;
    bool first = 1;
    float temp_squared_norm = 0;
    float mean_squared_norms[MAX_TX_PORTS]; /// Squared norms of columns in Q averaged over 8 REs.

    __m512 vec1, vec2;

    /// Compute mean of squared norms
    for(int i = 0; i < num_tx_ports_; i++) {
        mean_squared_norms[i] = _mm512_reduce_add_ps(squared_norms_[i]);
    }

    for(int i = 0; i < num_tx_ports_; i++) {
        /// Find the column with lowest norm
        if(i <num_tx_ports_ - 1) { /// Stop if the last column has been computed
            for(int j = 0; j < num_tx_ports_; j++) {
                if(not computed[j]) {
                    if (first) {
                        argmin = j;
                        temp_squared_norm = mean_squared_norms[j];
                        first = 0;
                        continue;
                    } else {
                        if (mean_squared_norms[j] < temp_squared_norm) {
                            argmin = j;
                            temp_squared_norm = mean_squared_norms[j];
                        }
                    }
                }
            }

            computed[argmin] = 1; /// indicate that the current column of Q must not be modified in the next iterations
            first = 1;

        } else {
            for(int j = 0; j < num_tx_ports_; j++) {
                if(not computed[j]) {
                    argmin = j;
                }
            }
        }

        detection_reversed_order_[i] = argmin;

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[argmin][argmin] = _mm512_sqrt_ps(squared_norms_[argmin]);

        /// Normalize column q_argmin
        for(int j = 0; j < num_tx_ports_; j++) {
            q_matrix_transposed[argmin][j] = _mm512_div_ps(q_matrix_transposed[argmin][j], r_matrix[argmin][argmin]);
        }

        /// Project other columns vectors onto q_argmin and orthogonalize.
        if(i < num_tx_ports_ - 1) { /// Stop if the last column has been computed
            for(int j = 0; j < num_tx_ports_; j++) {
                if(not computed[j]) {

                    vec1 = _mm512_mul_ps(q_matrix_transposed[argmin][0], q_matrix_transposed[j][0]);
                    vec2 = _mm512_mul_ps(q_matrix_transposed[argmin][0], _mm512_permute_ps(q_matrix_transposed[j][0],
                                                                 0b10110001));
                    vec1 = _mm512_add_ps(vec1, _mm512_permute_ps(vec1, 0b10110001));
                    vec2 = _mm512_sub_ps(vec2, _mm512_permute_ps(vec2, 0b10110001));
                    //r_matrix[argmin][j] = conj_multiply_complex_float(q_matrix_transposed[argmin][0], q_matrix_transposed[j][0]);
                    r_matrix[argmin][j] = _mm512_permute_ps(_mm512_shuffle_ps(vec1, vec2, 0b10001000), 0b11011000);
                    for(int k = 1; k < num_rx_ports_; k++) {
                        vec1 = _mm512_mul_ps(q_matrix_transposed[argmin][k], q_matrix_transposed[j][k]);
                        vec2 = _mm512_mul_ps(q_matrix_transposed[argmin][k], _mm512_permute_ps(q_matrix_transposed[j][k],
                                                                                               0b10110001));
                        vec1 = _mm512_add_ps(vec1, _mm512_permute_ps(vec1, 0b10110001));
                        vec2 = _mm512_sub_ps(vec2, _mm512_permute_ps(vec2, 0b10110001));
                        //r_matrix[argmin][j] = _mm512_add_ps(r_matrix[argmin][j], conj_multiply_complex_float(q_matrix_transposed[argmin][k], q_matrix_transposed[j][k]));
                        r_matrix[argmin][j] = _mm512_add_ps(r_matrix[argmin][j], _mm512_permute_ps(_mm512_shuffle_ps(vec1, vec2, 0b10001000), 0b11011000));
                    }

                    for(int k = 0; k < num_rx_ports_; k++) {
                        vec1 = _mm512_mul_ps(r_matrix[argmin][j], q_matrix_transposed[argmin][k]);
                        vec2 = _mm512_mul_ps(r_matrix[argmin][j], _mm512_permute_ps(q_matrix_transposed[argmin][k], 0b10110001));
                        vec1 = _mm512_sub_ps(vec1, _mm512_permute_ps(vec1, 0b10110001));
                        vec2 = _mm512_add_ps(vec2, _mm512_permute_ps(vec2, 0b10110001));
                        //q_matrix_transposed[j][k] = _mm512_sub_ps(q_matrix_transposed[j][k], multiply_complex_float(r_matrix[argmin][j], q_matrix_transposed[argmin][k]));
                        q_matrix_transposed[j][k] = _mm512_sub_ps(q_matrix_transposed[j][k], _mm512_permute_ps(_mm512_shuffle_ps(vec1, vec2, 0b10001000), 0b11011000));
                    }

                    /// Update squared norms
                    vec1 = _mm512_mul_ps(r_matrix[argmin][j], r_matrix[argmin][j]);
                    //squared_norms_[j] = _mm512_sub_ps(squared_norms_[j], compute_norm_m512(r_matrix[argmin][j]));
                    squared_norms_[j] = _mm512_sub_ps(squared_norms_[j], _mm512_add_ps(vec1, _mm512_permute_ps(vec1, 0b10110001)));

                    /// Update mean of squared norms
                    mean_squared_norms[j] = _mm512_reduce_add_ps(squared_norms_[j]);
                }
            }
        }
    }
}

/** Performs unsorted QR decomp. on 8 REs simultanesouly using AVX512
 *
 * @param[in, out] q_matrix_transposed : transposed input Q matrix
 * @param[in, out] r_matrix : untransposed R matrix
 * @param[in] squared_norms_ : precomputed squared norms of each column in Q
 * @param[in] num_tx_ports_ : Number of TX ports
 * @param[in] num_rx_ports_ : Number of RX ports
 */
void static inline compute_qr_decomp(__m512 q_matrix_transposed[MAX_TX_PORTS][MAX_RX_PORTS],
                                     __m512 r_matrix[MAX_TX_PORTS][MAX_TX_PORTS],
                                     __m512 * squared_norms_,
                                     int num_tx_ports_,
                                     int num_rx_ports_) {

    int computed[MAX_TX_PORTS];
    memset(&computed, 0, MAX_TX_PORTS * sizeof(int));
    bool first = 1;
    float temp_squared_norm = 0;
    float mean_squared_norms[MAX_TX_PORTS]; /// Squared norms of columns in Q averaged over 8 REs.

    __m512 vec1, vec2;

    /// Compute mean of squared norms
    for(int i = 0; i < num_tx_ports_; i++) {
        mean_squared_norms[i] = _mm512_reduce_add_ps(squared_norms_[i]);
    }

    for(int i = 0; i < num_tx_ports_; i++) {

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[i][i] = _mm512_sqrt_ps(squared_norms_[i]);

        /// Normalize column q_argmin
        for(int j = 0; j < num_tx_ports_; j++) {
            q_matrix_transposed[i][j] = _mm512_div_ps(q_matrix_transposed[i][j], r_matrix[i][i]);
        }

        /// Project other columns vectors onto q_argmin and orthogonalize.
        if(i < num_tx_ports_ - 1) { /// Stop if the last column has been computed
            for(int j = 0; j < num_tx_ports_; j++) {
                if(not computed[j]) {

                    vec1 = _mm512_mul_ps(q_matrix_transposed[i][0], q_matrix_transposed[j][0]);
                    vec2 = _mm512_mul_ps(q_matrix_transposed[i][0], _mm512_permute_ps(q_matrix_transposed[j][0],
                                                                                           0b10110001));
                    vec1 = _mm512_add_ps(vec1, _mm512_permute_ps(vec1, 0b10110001));
                    vec2 = _mm512_sub_ps(vec2, _mm512_permute_ps(vec2, 0b10110001));
                    //r_matrix[argmin][j] = conj_multiply_complex_float(q_matrix_transposed[argmin][0], q_matrix_transposed[j][0]);
                    r_matrix[i][j] = _mm512_permute_ps(_mm512_shuffle_ps(vec1, vec2, 0b10001000), 0b11011000);
                    for(int k = 1; k < num_rx_ports_; k++) {
                        vec1 = _mm512_mul_ps(q_matrix_transposed[i][k], q_matrix_transposed[j][k]);
                        vec2 = _mm512_mul_ps(q_matrix_transposed[i][k], _mm512_permute_ps(q_matrix_transposed[j][k],
                                                                                               0b10110001));
                        vec1 = _mm512_add_ps(vec1, _mm512_permute_ps(vec1, 0b10110001));
                        vec2 = _mm512_sub_ps(vec2, _mm512_permute_ps(vec2, 0b10110001));
                        //r_matrix[argmin][j] = _mm512_add_ps(r_matrix[argmin][j], conj_multiply_complex_float(q_matrix_transposed[argmin][k], q_matrix_transposed[j][k]));
                        r_matrix[i][j] = _mm512_add_ps(r_matrix[i][j], _mm512_permute_ps(_mm512_shuffle_ps(vec1, vec2, 0b10001000), 0b11011000));
                    }

                    for(int k = 0; k < num_rx_ports_; k++) {
                        vec1 = _mm512_mul_ps(r_matrix[i][j], q_matrix_transposed[i][k]);
                        vec2 = _mm512_mul_ps(r_matrix[i][j], _mm512_permute_ps(q_matrix_transposed[i][k], 0b10110001));
                        vec1 = _mm512_sub_ps(vec1, _mm512_permute_ps(vec1, 0b10110001));
                        vec2 = _mm512_add_ps(vec2, _mm512_permute_ps(vec2, 0b10110001));
                        //q_matrix_transposed[j][k] = _mm512_sub_ps(q_matrix_transposed[j][k], multiply_complex_float(r_matrix[argmin][j], q_matrix_transposed[argmin][k]));
                        q_matrix_transposed[j][k] = _mm512_sub_ps(q_matrix_transposed[j][k], _mm512_permute_ps(_mm512_shuffle_ps(vec1, vec2, 0b10001000), 0b11011000));
                    }

                    /// Update squared norms
                    vec1 = _mm512_mul_ps(r_matrix[i][j], r_matrix[i][j]);
                    //squared_norms_[j] = _mm512_sub_ps(squared_norms_[j], compute_norm_m512(r_matrix[argmin][j]));
                    squared_norms_[j] = _mm512_sub_ps(squared_norms_[j], _mm512_add_ps(vec1, _mm512_permute_ps(vec1, 0b10110001)));

                    /// Update mean of squared norms
                    mean_squared_norms[j] = _mm512_reduce_add_ps(squared_norms_[j]);
                }
            }
        }
    }
}

#endif

/** Performs Sorted QR decomp directly on the buffer containing the channel matrix coefficients
 *
 * @param[in] channel_coefficients : the TRANSPOSED Channel matrix for all REs
 * @param[out} r_matrix : output R matrix
 * @param[in] squared_norms_ : precomputed squared norms of columns in the channel matrix
 * @param[in] detection_reversed_order_ : detection order in reverse order determined during QR decomposition
 * @param[in] num_tx_ports_ : Number of TX ports
 * @param[in] num_rx_ports_ : Number of RX ports
 * @param[in] re : RE index in the channel_coefficients buffer to be computed
 */
void static inline compute_qr_decomp(std::vector<std::complex<float>> channel_coefficients[MAX_TX_PORTS][MAX_RX_PORTS],
                                     std::complex<float> r_matrix[MAX_TX_PORTS][MAX_TX_PORTS],
                                     float * squared_norms_,
                                     int * detection_reversed_order_,
                                     int num_tx_ports_,
                                     int num_rx_ports_,
                                     int re) {
    int computed[MAX_TX_PORTS];
    memset(&computed, 0, MAX_TX_PORTS * sizeof(int));
    int argmin = 0;
    bool first = 1;
    float temp_squared_norm = 0;

    for(int i = 0; i < num_tx_ports_; i++) {

        /// Find the column with lowest norm
        if(i <num_tx_ports_ - 1) { /// Stop if the last column has been computed
            for(int j = 0; j < num_tx_ports_; j++) {
                if(not computed[j]) {
                    if (first) {
                        argmin = j;
                        temp_squared_norm = squared_norms_[j];
                        first = 0;
                        continue;
                    } else {
                        if (squared_norms_[j] < temp_squared_norm) {
                            argmin = j;
                            temp_squared_norm = squared_norms_[j];
                        }
                    }
                }
            }

            computed[argmin] = 1; /// indicate that the current column of Q must not be modified in the next iterations
            first = 1;

        } else {
            for(int j = 0; j < num_tx_ports_; j++) {
                if(not computed[j]) {
                    argmin = j;
                }
            }
        }

        detection_reversed_order_[i] = argmin;

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[argmin][argmin] = sqrt(squared_norms_[argmin]);

        /// Normalize column q_argmin
        for(int j = 0; j < num_tx_ports_; j++) {
            channel_coefficients[argmin][j][re] /= r_matrix[argmin][argmin].real();
        }

        /// Project other columns vectors onto q_argmin and orthogonalize.
        if(i < num_tx_ports_ - 1) { /// Stop if the last column has been computed
            for(int j = 0; j < num_tx_ports_; j++) {
                if(not computed[j]) {
                    r_matrix[argmin][j] = conj(channel_coefficients[argmin][0][re]) * channel_coefficients[j][0][re];
                    for(int k = 1; k < num_rx_ports_; k++) {
                        r_matrix[argmin][j] += conj(channel_coefficients[argmin][k][re]) * channel_coefficients[j][k][re];
                    }
                    for(int k = 0; k < num_rx_ports_; k++) {
                        channel_coefficients[j][k][re] = channel_coefficients[j][k][re] - r_matrix[argmin][j] * channel_coefficients[argmin][k][re];
                    }
                    squared_norms_[j] -= (r_matrix[argmin][j].real() * r_matrix[argmin][j].real() +
                                          r_matrix[argmin][j].imag() * r_matrix[argmin][j].imag());
                }
            }
        }
    }
}

/** SQRD equalization working directly on the transposed channel matrix (No intermediate Q matrix involved).
 *
 * @param channel_coefficients_ : matrix of channel coefficients
 * @param pdsch_samples : received PDSCH samples on each RX port
 * @param num_re_pdsch_ : Number of PDSCH REs
 * @param equalized_symbols_ : buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 * @param constellation_symbols : constellation symbols (see variables.cpp)
 * @param detected_symbols_ : final buffer of detected symbols
 * @param constellation_type_ : type of constellation (see variables.cpp)
 */
void mimo_vblast_decoder_qr_decomp_modified(std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                            const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                            int num_re_pdsch_,
                                            std::complex<float> * equalized_symbols_,
                                            int nb_tx_dmrs_ports_,
                                            int nb_rx_ports_,
                                            std::complex<float> * constellation_symbols,
                                            int * detected_symbols_,
                                            int constellation_type_);


/*********************************** Tes function not used in the code ***********************************************/

/** SQRD equalization working directly on the transposed channel matrix (No intermediate Q matrix involved).
 *  Test for realign channel matrix buffer with RE in the first dimension.
 *
 * @param channel_coefficients_ : matrix of channel coefficients
 * @param pdsch_samples : received PDSCH samples on each RX port
 * @param num_re_pdsch_ : Number of PDSCH REs
 * @param equalized_symbols_ : buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 * @param constellation_symbols : constellation symbols (see variables.cpp)
 * @param detected_symbols_ : final buffer of detected symbols
 * @param constellation_type_ : type of constellation (see variables.cpp)
 */
void mimo_vblast_decoder_qr_decomp_modified(std::complex<float> channel_coefficients_[][MAX_TX_PORTS][MAX_RX_PORTS],
                                            const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                            int num_re_pdsch_,
                                            std::complex<float> * equalized_symbols_,
                                            int nb_tx_dmrs_ports_,
                                            int nb_rx_ports_,
                                            std::complex<float> * constellation_symbols,
                                            int * detected_symbols_,
                                            int constellation_type_);

/********************************************************************************************************************/

/** Unsorted QR decomp. Matrices are provided as 1-D arrays and indexes have to be computed manually.
 *
 * @param q_matrix : transposed Q matrix
 * @param r_matrix : R matrix
 * @param squared_norms_ : precomputed squared norms of columns in Q
 * @param num_cols_q_ : number of columns in Q
 * @param num_rows_q_ : number of rows in Q
 */
void inline compute_qr_decomp(std::complex<float> * q_matrix, /// Load q matrix with channel coefficients before QR decomp
                              std::complex<float> * r_matrix,
                              float * squared_norms_,
                              int num_cols_q_,
                              int num_rows_q_) {

    for(int i = 0; i < num_cols_q_; i++) {
        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[i * num_cols_q_ + i] = sqrt(squared_norms_[i]);
        /// Normalize column q_argmin
        for(int j = 0; j < num_rows_q_; j++) {
            q_matrix[i * num_rows_q_ + j] /= r_matrix[i * num_cols_q_ + i].real();
        }

        /// Project other columns vectors onto q_argmin and orthogonalize.
        if(i < num_cols_q_ - 1) { /// Stop if the last column has been computed
            for(int j = i + 1; j < num_cols_q_; j++) {
                r_matrix[j * num_cols_q_ + i] = conj(q_matrix[i * num_rows_q_]) * q_matrix[j * num_rows_q_];
                for(int k = 1; k < num_rows_q_; k++) {
                    r_matrix[j * num_cols_q_ + i] += conj(q_matrix[i * num_rows_q_ + k]) * q_matrix[j * num_rows_q_ + k];
                }

                for(int k = 0; k < num_rows_q_; k++) {
                    q_matrix[j * num_rows_q_ + k] = q_matrix[j * num_rows_q_ + k] - r_matrix[j * num_cols_q_ + i] * q_matrix[i * num_rows_q_ + k];
                }

                squared_norms_[j] = squared_norms_[j] - (r_matrix[j * num_cols_q_ + i].real() * r_matrix[j * num_cols_q_ + i].real() +
                r_matrix[j * num_cols_q_ + i].imag() * r_matrix[j * num_cols_q_ + i].imag()); //pow(abs(r_matrix[j * num_cols_q_ + i]), 2);
            }
        }
    }
}

/** Unsorted QR decomposition
 *
 * @param[in] q_matrix : transposed Q matrix
 * @param[in] r_matrix : untransposed R matrix
 * @param[in] squared_norms_ : pre computed squared norms of columns in Q
 * @param[in] num_cols_q_ : number of columns in Q
 * @param[in] num_rows_q_ : number of rows in Q
 */
void inline compute_qr_decomp(std::complex<float> q_matrix[MAX_TX_PORTS][MAX_RX_PORTS], /// Transposed matrix/'column major'
                              std::complex<float> r_matrix[MAX_TX_PORTS][MAX_TX_PORTS], /// Untransposed R
                              float * squared_norms_,
                              int num_cols_q_,
                              int num_rows_q_) {

    for(int i = 0; i < num_cols_q_; i++) {
        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[i][i] = sqrt(squared_norms_[i]);
        /// Normalize column q_argmin
        for(int j = 0; j < num_rows_q_; j++) {
            q_matrix[i][j] /= r_matrix[i][i].real();
        }

        /// Project other columns vectors onto q_argmin and orthogonalize.
        if(i < num_cols_q_ - 1) { /// Stop if the last column has been computed
            for(int j = i + 1; j < num_cols_q_; j++) {
                r_matrix[i][j] = conj(q_matrix[i][0]) * q_matrix[j][0];
                for(int k = 1; k < num_rows_q_; k++) {
                    r_matrix[i][j] += conj(q_matrix[i][k]) * q_matrix[j][k];
                }
                for(int k = 0; k < num_rows_q_; k++) {
                    q_matrix[j][k] = q_matrix[j][k] - r_matrix[i][j] * q_matrix[i][k];
                }
                    squared_norms_[j] = squared_norms_[j] - (r_matrix[i][j].real() * r_matrix[i][j].real() +
                    r_matrix[i][j].imag() * r_matrix[i][j].imag());
            }
        }
    }
}

/** Compute sorted SR decomp based on the precomputed reversed detection order given in input (given in ascending order
 *  of squared norms).
 *
 * @param q_matrix : transposed Q matrix for each RE.
 * @param r_matrix : R matrix
 * @param squared_norms_ : square norms of columns in Q
 * @param detection_order_ : detection order in reversed order (first element corresponds to the last detected layer)
 * @param num_cols_q_ : number of oclumns in Q
 * @param num_rows_q_ : number of rows in Q
 * @param re : RE index
 */
void inline compute_qr_decomp_col_norm_reordering(std::vector<std::complex<float>> q_matrix[MAX_TX_PORTS][MAX_RX_PORTS], /// Transposed matrix/'column major'
                                                  std::complex<float> r_matrix[MAX_TX_PORTS][MAX_TX_PORTS], /// Untransposed R
                                                  float * squared_norms_,
                                                  int * detection_order_,
                                                  int num_cols_q_,
                                                  int num_rows_q_,
                                                  int re) {

    int current_col = 0;
    std::vector<int> computed(MAX_TX_PORTS);

    for (int i = 0; i < num_cols_q_; i++) {
        current_col = detection_order_[i];
        computed[current_col] = 1;

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[current_col][current_col] = sqrt(squared_norms_[current_col]);

        /// Normalize column q_argmin
        for (int j = 0; j < num_rows_q_; j++) {
            q_matrix[current_col][j][re] /= r_matrix[current_col][current_col].real();
        }

        /// Project other columns vectors onto q_argmin and orthogonalize.
        if (i < num_cols_q_ - 1) { /// Stop if the last column has been computed
            for (int j = 0; j < num_cols_q_; j++) {
                if (not computed[j]) {
                    r_matrix[current_col][j] = conj(q_matrix[current_col][0][re]) * q_matrix[j][0][re];
                    for (int k = 1; k < num_rows_q_; k++) {
                        r_matrix[current_col][j] += conj(q_matrix[current_col][k][re]) * q_matrix[j][k][re];
                    }
                    for (int k = 0; k < num_rows_q_; k++) {
                        q_matrix[j][k][re] =
                                q_matrix[j][k][re] - r_matrix[current_col][j] * q_matrix[current_col][k][re];
                    }
                    squared_norms_[current_col] =
                            squared_norms_[current_col] -
                            (r_matrix[current_col][j].real() * r_matrix[current_col][j].real() +
                             r_matrix[current_col][j].imag() * r_matrix[current_col][j].imag());
                }
            }
        }
    }
}

/** Compute the unsorted QR decomp directly on the buffer containing the channel matrix coefficeints
 *
 * @param channel_coefficients_ : the TRANSPOSED matrix of channel coefficients on each RE
 * @param r_matrix : R matrix
 * @param squared_norms_ : square norms of columns in the channel matrix
 * @param num_cols_q_ : number of columns in the channel matrix
 * @param num_rows_q_ : number of rows in the channel matrix
 * @param re : RE index
 */
void inline compute_qr_decomp(std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS], /// Transposed matrix/'column major'
                              std::complex<float> r_matrix[MAX_TX_PORTS][MAX_TX_PORTS], /// Untransposed R
                              float * squared_norms_,
                              int num_cols_q_,
                              int num_rows_q_,
                              int re) {

    for(int i = 0; i < num_cols_q_; i++) {
        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[i][i] = sqrt(squared_norms_[i]);
        /// Normalize column q_argmin
        for(int j = 0; j < num_rows_q_; j++) {
            channel_coefficients_[i][j][re] /= r_matrix[i][i].real();
        }

        /// Project other columns vectors onto q_argmin and orthogonalize.
        if(i < num_cols_q_ - 1) { /// Stop if the last column has been computed
        for(int j = i + 1; j < num_cols_q_; j++) {
            r_matrix[i][j] = conj(channel_coefficients_[i][0][re]) * channel_coefficients_[j][0][re];
            for(int k = 1; k < num_rows_q_; k++) {
                r_matrix[i][j] += conj(channel_coefficients_[i][k][re]) * channel_coefficients_[j][k][re];
            }
            for(int k = 0; k < num_rows_q_; k++) {
                channel_coefficients_[j][k][re] = channel_coefficients_[j][k][re] - r_matrix[i][j] * channel_coefficients_[i][k][re];
            }
            squared_norms_[j] = squared_norms_[j] - (r_matrix[i][j].real() * r_matrix[i][j].real() +
                r_matrix[i][j].imag() * r_matrix[i][j].imag());
            }
        }
    }
}

/**************************************** QRD optimizations for 2 layers ***********************************************/
/** SQRD equalization optimization for 2 layers.
 *
 * @param channel_coefficients_ : transposed channel matrix coefficients
 * @param pdsch_samples : received PDSCH samples on each RX port
 * @param num_re_pdsch_ : Numbre of PDSCH to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 * @param constellation_symbols : array of constellation symbols (see variables.cpp)
 * @param detected_symbols_ : final buffer of detected symbols
 * @param constellation_type_ : constellation type (see variables.cpp)
 */
void mimo_vblast_decoder_qr_decomp_2_layers(const std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                            const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                            int num_re_pdsch_,
                                            std::complex<float> * equalized_symbols_,
                                            int nb_tx_dmrs_ports_,
                                            int nb_rx_ports_,
                                            std::complex<float> * constellation_symbols,
                                            int * detected_symbols_,
                                            int constellation_type_);

/** SQRD equalization optimization for 2 layerrs working directly on the buffers of channel coefficients.
 *
 * @param channel_coefficients_ : transposed channel matrix coefficients
 * @param pdsch_samples : received PDSCH samples on each RX port
 * @param num_re_pdsch_ : Numbre of PDSCH to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 * @param constellation_symbols : array of constellation symbols (see variables.cpp)
 * @param detected_symbols_ : final buffer of detected symbols
 * @param constellation_type_ : constellation type (see variables.cpp)
 */
void mimo_vblast_decoder_qr_decomp_modified_2_layers(std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                                     const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                                     int num_re_pdsch_,
                                                     std::complex<float> * equalized_symbols_,
                                                     int nb_tx_dmrs_ports_,
                                                     int nb_rx_ports_,
                                                     std::complex<float> * constellation_symbols,
                                                     int * detected_symbols_,
                                                     int constellation_type_);

/** QRD-CN equalization optimization for 2 layers.
 *
 * @param channel_coefficients_ : transposed channel matrix coefficients
 * @param pdsch_samples : received PDSCH samples on each RX port
 * @param num_re_pdsch_ : Numbre of PDSCH to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 * @param constellation_symbols : array of constellation symbols (see variables.cpp)
 * @param detected_symbols_ : final buffer of detected symbols
 * @param constellation_type_ : constellation type (see variables.cpp)
 */
void mimo_vblast_qrd_col_norm_modified_2_layers(std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                                const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                                int num_re_pdsch_,
                                                std::complex<float> * equalized_symbols_,
                                                int nb_tx_dmrs_ports_,
                                                int nb_rx_ports_,
                                                std::complex<float> * constellation_symbols,
                                                int * detected_symbols_,
                                                int constellation_type_);

/** QRD-CN equalization.
 *  Used only for 2 layers
 *
 * @param channel_coefficients_ : matrix of channel coefficients
 * @param pdsch_samples : received PDSCH samples on each RX port
 * @param num_re_pdsch_ : Number of PDSCH REs
 * @param equalized_symbols_ : buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 * @param constellation_symbols : constellation symbols (see variables.cpp)
 * @param detected_symbols_ : final buffer of detected symbols
 * @param constellation_type_ : type of constellation (see variables.cpp)
 */
void mimo_vblast_qrd_col_norm_2_layers(const std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                       const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                       int num_re_pdsch_,
                                       std::complex<float> * equalized_symbols_,
                                       int nb_tx_dmrs_ports_,
                                       int nb_rx_ports_,
                                       std::complex<float> * constellation_symbols,
                                       int * detected_symbols_,
                                       int constellation_type_);

/** QRD-CN equalization optimization for 2 layers working directly on the buffer of channel coefficients
 *
 * @param channel_coefficients_ : transposed channel matrix coefficients
 * @param pdsch_samples : received PDSCH samples on each RX port
 * @param num_re_pdsch_ : Numbre of PDSCH to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 * @param constellation_symbols : array of constellation symbols (see variables.cpp)
 * @param detected_symbols_ : final buffer of detected symbols
 * @param constellation_type_ : constellation type (see variables.cpp)
 */
void mimo_vblast_decoder_qr_decomp_no_reordering_2_layers(std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                                            const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                                            int num_re_pdsch_,
                                                            std::complex<float> * equalized_symbols_,
                                                            int nb_tx_dmrs_ports_,
                                                            int nb_rx_ports_,
                                                            std::complex<float> * constellation_symbols,
                                                            int * detected_symbols_,
                                                            int constellation_type_);

/** Calls the unsorted QRD equalization algorithms based on the number of layers.
 *
 * @param channel_coefficients_ : transposed channel matrix coefficients
 * @param pdsch_samples : received PDSCH samples on each RX port
 * @param num_re_pdsch_ : Numbre of PDSCH to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 * @param constellation_symbols : array of constellation symbols (see variables.cpp)
 * @param detected_symbols_ : final buffer of detected symbols
 * @param constellation_type_ : constellation type (see variables.cpp)
 */
void call_vblast_qrd_no_reordering(std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                    const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                    int num_re_pdsch_,
                                    std::complex<float> * equalized_symbols_,
                                    int nb_tx_dmrs_ports_,
                                    int nb_rx_ports_,
                                    std::complex<float> * constellation_symbols,
                                    int * detected_symbols_,
                                    int constellation_type_);

/** Calls the SQRD equalization algorithms based on the number of layers.
 *
 * @param channel_coefficients_ : transposed channel matrix coefficients
 * @param pdsch_samples : received PDSCH samples on each RX port
 * @param num_re_pdsch_ : Numbre of PDSCH to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 * @param constellation_symbols : array of constellation symbols (see variables.cpp)
 * @param detected_symbols_ : final buffer of detected symbols
 * @param constellation_type_ : constellation type (see variables.cpp)
 */
void call_vblast_sqrd_functions(std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                int num_re_pdsch_,
                                std::complex<float> * equalized_symbols_,
                                int nb_tx_dmrs_ports_,
                                int nb_rx_ports_,
                                std::complex<float> * constellation_symbols,
                                int * detected_symbols_,
                                int constellation_type_);

/** Calls the SQRD AVX2 equalization algorithms based on the number of layers.
 *
 * @param channel_coefficients_ : transposed channel matrix coefficients
 * @param pdsch_samples : received PDSCH samples on each RX port
 * @param num_re_pdsch_ : Numbre of PDSCH to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 * @param constellation_symbols : array of constellation symbols (see variables.cpp)
 * @param detected_symbols_ : final buffer of detected symbols
 * @param constellation_type_ : constellation type (see variables.cpp)
 */
void call_vblast_sqrd_functions_avx2(std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                    const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                    int num_re_pdsch_,
                                    std::complex<float> * equalized_symbols_,
                                    int nb_tx_dmrs_ports_,
                                    int nb_rx_ports_,
                                    std::complex<float> * constellation_symbols,
                                    int * detected_symbols_,
                                    int constellation_type_);

/** Calls the QRD-CN equalization algorithms based on the number of layers.
 *
 * @param channel_coefficients_ : transposed channel matrix coefficients
 * @param pdsch_samples : received PDSCH samples on each RX port
 * @param num_re_pdsch_ : Numbre of PDSCH to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 * @param constellation_symbols : array of constellation symbols (see variables.cpp)
 * @param detected_symbols_ : final buffer of detected symbols
 * @param constellation_type_ : constellation type (see variables.cpp)
 */
void call_vblast_qrd_col_norm_functions(std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                        const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                        int num_re_pdsch_,
                                        std::complex<float> * equalized_symbols_,
                                        int nb_tx_dmrs_ports_,
                                        int nb_rx_ports_,
                                        std::complex<float> * constellation_symbols,
                                        int * detected_symbols_,
                                        int constellation_type_);

/** Calls the QRD-CN AVX2 equalization algorithms based on the number of layers.
 *
 * @param channel_coefficients_ : transposed channel matrix coefficients
 * @param pdsch_samples : received PDSCH samples on each RX port
 * @param num_re_pdsch_ : Numbre of PDSCH to be computed
 * @param equalized_symbols_ : final buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 * @param constellation_symbols : array of constellation symbols (see variables.cpp)
 * @param detected_symbols_ : final buffer of detected symbols
 * @param constellation_type_ : constellation type (see variables.cpp)
 */
void call_vblast_qrd_col_norm_functions_avx2(std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                             const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                             int num_re_pdsch_,
                                             std::complex<float> * equalized_symbols_,
                                             int nb_tx_dmrs_ports_,
                                             int nb_rx_ports_,
                                             std::complex<float> * constellation_symbols,
                                             int * detected_symbols_,
                                             int constellation_type_);

/** Sorted QR decomp working directly on the channel coefficients buffers.
 *  Simplified for 2 layers.
 *
 * @param[in] channel_coefficients : transposed channel matrix coefficients buffer
 * @param[in] r_matrix : untransposed R matrix
 * @param[in] squared_norms_ : precomputed squared norm of columns in the channel matrix
 * @param[in, out] detection_reversed_order_ : reversed detection order (the last element is the first layer to be decoded)
 * @param[in] num_tx_ports_ : Number of TX DMRS ports
 * @param[in] num_rx_ports_ : Number of RX ports
 * @param[in] re : RE index
 */
void static inline compute_qr_decomp_2_layers(std::vector<std::complex<float>> channel_coefficients[MAX_TX_PORTS][MAX_RX_PORTS],
                                              std::complex<float> r_matrix[2][2],
                                              float * squared_norms_,
                                              int * detection_reversed_order_,
                                              int num_tx_ports_,
                                              int num_rx_ports_,
                                              int re) {

    /// Find the column with lowest norm
    if(squared_norms_[0] < squared_norms_[1]) {
        detection_reversed_order_[0] = 0;
        detection_reversed_order_[1] = 1;

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[0][0] = sqrt(squared_norms_[0]);

        /// Normalize column q_argmin
        for(int j = 0; j < num_tx_ports_; j++) {
            channel_coefficients[0][j][re] /= r_matrix[0][0].real();
        }

        r_matrix[0][1] = conj(channel_coefficients[0][0][re]) * channel_coefficients[1][0][re];
        for(int k = 1; k < num_rx_ports_; k++) {
            r_matrix[0][1] += conj(channel_coefficients[0][k][re]) * channel_coefficients[1][k][re];
        }
        for(int k = 0; k < num_rx_ports_; k++) {
            channel_coefficients[1][k][re] = channel_coefficients[1][k][re] - r_matrix[0][1] * channel_coefficients[0][k][re];
        }
        squared_norms_[1] -= (r_matrix[0][1].real() * r_matrix[0][1].real() +
                              r_matrix[0][1].imag() * r_matrix[0][1].imag());

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[1][1] = sqrt(squared_norms_[1]);

        /// Normalize column q_argmin
        for(int j = 0; j < num_tx_ports_; j++) {
            channel_coefficients[1][j][re] /= r_matrix[1][1].real();
        }

    } else {
        detection_reversed_order_[0] = 1;
        detection_reversed_order_[1] = 0;

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[1][1] = sqrt(squared_norms_[1]);

        /// Normalize column q_argmin
        for(int j = 0; j < num_tx_ports_; j++) {
            channel_coefficients[1][j][re] /= r_matrix[1][1].real();
        }

        r_matrix[1][0] = conj(channel_coefficients[1][0][re]) * channel_coefficients[0][0][re];
        for(int k = 1; k < num_rx_ports_; k++) {
            r_matrix[1][0] += conj(channel_coefficients[1][k][re]) * channel_coefficients[0][k][re];
        }
        for(int k = 0; k < num_rx_ports_; k++) {
            channel_coefficients[0][k][re] = channel_coefficients[0][k][re] - r_matrix[1][0] * channel_coefficients[1][k][re];
        }
        squared_norms_[0] -= (r_matrix[1][0].real() * r_matrix[1][0].real() +
                              r_matrix[1][0].imag() * r_matrix[1][0].imag());

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[0][0] = sqrt(squared_norms_[0]);

        /// Normalize column q_argmin
        for(int j = 0; j < num_tx_ports_; j++) {
            channel_coefficients[0][j][re] /= r_matrix[0][0].real();
        }
    }
}

/** Sorted QR decomp working directly on the channel coefficients buffers.
 *  Simplified for 2 layers.
 *
 * @param[in] channel_coefficients : transposed channel matrix coefficients buffer
 * @param[in] r_matrix : untransposed R matrix
 * @param[in] squared_norms_ : precomputed squared norm of columns in the channel matrix
 * @param[in, out] detection_reversed_order_ : reversed detection order (the last element is the first layer to be decoded)
 * @param[in] num_tx_ports_ : Number of TX DMRS ports
 * @param[in] num_rx_ports_ : Number of RX ports
 * @param[in] re : RE index
 */
void inline compute_qr_decomp_2_layers(std::complex<float> q_matrix[2][MAX_RX_PORTS], /// Transposed Q matrix
                                       std::complex<float> r_matrix[2][2], /// Untransposed R matrix
                                       float squared_norms_[2],
                                       int * detection_reversed_order_,
                                       int num_rx_ports_,
                                       int num_tx_ports_) {

    float temp_squared_norm = 0;

    if(squared_norms_[0] < squared_norms_[1]) {
        detection_reversed_order_[0] = 0;
        detection_reversed_order_[1] = 1;

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[0][0] = sqrt(squared_norms_[0]);

        /// Normalize column q_argmin
        for(int j = 0; j < num_rx_ports_; j++) {
            q_matrix[0][j] /= r_matrix[0][0].real();
        }

        r_matrix[0][1] = conj(q_matrix[0][0]) * q_matrix[1][0];

        for(int k = 1; k < num_rx_ports_; k++) {
            r_matrix[0][1] += conj(q_matrix[0][k]) * q_matrix[1][k];
        }

        for(int k = 0; k < num_rx_ports_; k++) {
            q_matrix[1][k] = q_matrix[1][k] - r_matrix[0][1] * q_matrix[0][k];
        }

        squared_norms_[1] -= (r_matrix[0][1].real() * r_matrix[0][1].real() +
        r_matrix[0][1].imag() * r_matrix[0][1].imag());

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[1][1] = sqrt(squared_norms_[1]);

        /// Normalize column q_argmin
        for(int j = 0; j < num_rx_ports_; j++) {
            q_matrix[1][j] /= r_matrix[1][1].real();
        }
    } else {
        detection_reversed_order_[0] = 1;
        detection_reversed_order_[1] = 0;

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[1][1] = sqrt(squared_norms_[1]);

        /// Normalize column q_argmin
        for(int j = 0; j < num_rx_ports_; j++) {
            q_matrix[1][j] /= r_matrix[1][1].real();
        }

        r_matrix[1][0] = conj(q_matrix[1][0]) * q_matrix[0][0];

        for(int k = 1; k < num_rx_ports_; k++) {
            r_matrix[1][0] += conj(q_matrix[1][k]) * q_matrix[0][k];
        }

        for(int k = 0; k < num_rx_ports_; k++) {
            q_matrix[0][k] = q_matrix[0][k] - r_matrix[1][0] * q_matrix[1][k];
        }

        squared_norms_[0] -= (r_matrix[1][0].real() * r_matrix[1][0].real() +
                              r_matrix[1][0].imag() * r_matrix[1][0].imag());

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[0][0] = sqrt(squared_norms_[0]);

        /// Normalize column q_argmin
        for(int j = 0; j < num_rx_ports_; j++) {
            q_matrix[0][j] /= r_matrix[0][0].real();
        }
    }
}

/** Unsorted QR decomp working directly on the channel coefficients buffers.
 *  Simplified for 2 layers.
 *
 * @param[in] channel_coefficients : transposed channel matrix coefficients buffer
 * @param[in] r_matrix : untransposed R matrix
 * @param[in] squared_norms_ : precomputed squared norm of columns in the channel matrix
 * @param[in, out] detection_reversed_order_ : reversed detection order (the last element is the first layer to be decoded)
 * @param[in] num_tx_ports_ : Number of TX DMRS ports
 * @param[in] num_rx_ports_ : Number of RX ports
 * @param[in] re : RE index
 */
void inline compute_qr_decomp_2_layers(std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS], /// Transposed matrix/'column major'
                                      std::complex<float> r_matrix[2][2], /// Untransposed R
                                      float * squared_norms_,
                                      int num_rows_q_,
                                      int re) {

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[0][0] = sqrt(squared_norms_[0]);
        /// Normalize column q_argmin
        for(int j = 0; j < num_rows_q_; j++) {
            channel_coefficients_[0][j][re] /= r_matrix[0][0].real();
        }
        /// Project other columns vectors onto q_argmin and orthogonalize.
        r_matrix[0][1] = conj(channel_coefficients_[0][0][re]) * channel_coefficients_[1][0][re];
        for(int k = 1; k < num_rows_q_; k++) {
            r_matrix[0][1] += conj(channel_coefficients_[0][k][re]) * channel_coefficients_[1][k][re];
        }
        for(int k = 0; k < num_rows_q_; k++) {
            channel_coefficients_[1][k][re] = channel_coefficients_[1][k][re] - r_matrix[0][1] * channel_coefficients_[0][k][re];
        }
        squared_norms_[1] = squared_norms_[1] - (r_matrix[0][1].real() * r_matrix[0][1].real() +
                                                 r_matrix[0][1].imag() * r_matrix[0][1].imag());

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[1][1] = sqrt(squared_norms_[1]);
        /// Normalize column q_argmin
        for(int j = 0; j < num_rows_q_; j++) {
            channel_coefficients_[1][j][re] /= r_matrix[1][1].real();
        }
}

#if defined(__AVX2__)
/** QRD-CN equalization using AVX2 optimizations to perform computations on 4 REs at once.
 *  USed only for 2 layers.
 *
 * @param channel_coefficients_ : matrix of channel coefficients
 * @param pdsch_samples : received PDSCH samples on each RX port
 * @param num_re_pdsch_ : Number of PDSCH REs
 * @param equalized_symbols_ : buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 * @param constellation_symbols : constellation symbols (see variables.cpp)
 * @param detected_symbols_ : final buffer of detected symbols
 * @param constellation_type_ : type of constellation (see variables.cpp)
 */
void mimo_vblast_qrd_col_norm_avx2_2_layers(std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                            const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                            int num_re_pdsch_,
                                            std::complex<float> * equalized_symbols_,
                                            int nb_tx_dmrs_ports_,
                                            int nb_rx_ports_,
                                            std::complex<float> * constellation_symbols,
                                            int * detected_symbols_,
                                            int constellation_type_);
#endif

#if defined(__AVX2__)
/** SQRD equalization using AVX2 optimizations to perform computations on 4 REs at once.
 *  USed only for 2 layers.
 *
 * @param channel_coefficients_ : matrix of channel coefficients
 * @param pdsch_samples : received PDSCH samples on each RX port
 * @param num_re_pdsch_ : Number of PDSCH REs
 * @param equalized_symbols_ : buffer of equalized symbols
 * @param nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param nb_rx_ports_ : Number of RX ports
 * @param constellation_symbols : constellation symbols (see variables.cpp)
 * @param detected_symbols_ : final buffer of detected symbols
 * @param constellation_type_ : type of constellation (see variables.cpp)
 */
void mimo_vblast_sqrd_avx2_2_layers(std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                    const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                    int num_re_pdsch_,
                                    std::complex<float> * equalized_symbols_,
                                    int nb_tx_dmrs_ports_,
                                    int nb_rx_ports_,
                                    std::complex<float> * constellation_symbols,
                                    int * detected_symbols_,
                                    int constellation_type_);

/** Sorted QR decomp using AVX2 optimizations to compute 4 REs at once.
 *  Used only for 2 layers.
 *
 * @param q_matrix_transposed : transposed Q matrix
 * @param r_matrix : untransposed R matrix
 * @param squared_norms_ : squared norms of columns in Q
 * @param detection_reversed_order_ : reversed detection order (the last element corresponds to the first layer to be decoded)
 * @param num_rx_ports_ : Number of RX ports
 */
void static inline compute_qr_decomp_2_layers(__m256 q_matrix_transposed[MAX_TX_PORTS][MAX_RX_PORTS],
                                              __m256 r_matrix[MAX_TX_PORTS][MAX_TX_PORTS],
                                              __m256 * squared_norms_,
                                              int * detection_reversed_order_,
                                              int num_rx_ports_) {

    float mean_squared_norms[2]; /// Squared norms of columns in Q averaged over 4 REs.

    __m256 vec1;
    __m128 vec1_128, vec2_128;

    /// Compute mean of squared norms
    for(int i = 0; i < 2; i++) {
        vec1 = _mm256_permute_ps(squared_norms_[i], 0b11011000);
        vec1 = _mm256_hadd_ps(vec1, vec1);
        vec1_128 = _mm256_castps256_ps128(vec1);
        vec2_128 = _mm256_extractf128_ps(vec1, 1);
        mean_squared_norms[i] = _mm_cvtss_f32(_mm_add_ps(vec1_128, vec2_128));
    }

    if(mean_squared_norms[0] < mean_squared_norms[1]) {
        detection_reversed_order_[0] = 0;
        detection_reversed_order_[1] = 1;

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[0][0] = _mm256_sqrt_ps(squared_norms_[0]);

        /// Normalize column q_argmin
        for(int j = 0; j < 2; j++) {
            q_matrix_transposed[0][j] = _mm256_div_ps(q_matrix_transposed[0][j], r_matrix[0][0]);
        }

        r_matrix[0][1] = conj_multiply_complex_float(q_matrix_transposed[0][0], q_matrix_transposed[1][0]);
        for(int k = 1; k < num_rx_ports_; k++) {
            r_matrix[0][1] = _mm256_add_ps(r_matrix[0][1],
                                           conj_multiply_complex_float(q_matrix_transposed[0][k],
                                                                       q_matrix_transposed[1][k]));
        }

        for(int k = 0; k < num_rx_ports_; k++) {
            q_matrix_transposed[1][k] = _mm256_sub_ps(q_matrix_transposed[1][k],
                                                      multiply_complex_float(r_matrix[0][1],
                                                                             q_matrix_transposed[0][k]));
        }

        /// Update squared norms
        squared_norms_[1] = _mm256_sub_ps(squared_norms_[1], compute_norm_m256(r_matrix[0][1]));

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[1][1] = _mm256_sqrt_ps(squared_norms_[1]);

        /// Normalize column q_argmin
        for(int j = 0; j < 2; j++) {
            q_matrix_transposed[1][j] = _mm256_div_ps(q_matrix_transposed[1][j], r_matrix[1][1]);
        }

    } else {
        detection_reversed_order_[0] = 1;
        detection_reversed_order_[1] = 0;

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[1][1] = _mm256_sqrt_ps(squared_norms_[1]);

        /// Normalize column q_argmin
        for(int j = 0; j < 2; j++) {
            q_matrix_transposed[1][j] = _mm256_div_ps(q_matrix_transposed[1][j], r_matrix[1][1]);
        }

        r_matrix[1][0] = conj_multiply_complex_float(q_matrix_transposed[1][0],
                                                     q_matrix_transposed[0][0]);
        for(int k = 1; k < num_rx_ports_; k++) {
            r_matrix[1][0] = _mm256_add_ps(r_matrix[1][0],
                                           conj_multiply_complex_float(q_matrix_transposed[1][k],
                                                                       q_matrix_transposed[0][k]));
        }

        for(int k = 0; k < num_rx_ports_; k++) {
            q_matrix_transposed[0][k] = _mm256_sub_ps(q_matrix_transposed[0][k],
                                                      multiply_complex_float(r_matrix[1][0],
                                                                             q_matrix_transposed[1][k]));
        }

        /// Update squared norms
        squared_norms_[0] = _mm256_sub_ps(squared_norms_[0], compute_norm_m256(r_matrix[1][0]));

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[0][0] = _mm256_sqrt_ps(squared_norms_[0]);

        /// Normalize column q_argmin
        for(int j = 0; j < 2; j++) {
            q_matrix_transposed[0][j] = _mm256_div_ps(q_matrix_transposed[0][j], r_matrix[0][0]);
        }
    }
}
#endif
/***************************************************************************************/

/** Sorted QR decomp.
 *
 * @param[in] q_matrix : transposed Q matrix
 * @param[in] r_matrix : R matrix
 * @param[in] squared_norms_ : squared norms of columns in Q
 * @param[in, out] detection_reversed_order_ : reversed detection order (the last elements corresponds to the first layer to be decoded)
 * @param[in] num_rx_ports_ : Number of RX ports
 * @param[in] num_tx_ports_ Number of TX ports
 */
void inline compute_qr_decomp(std::complex<float> q_matrix[MAX_TX_PORTS][MAX_RX_PORTS], /// Transposed Q matrix
                              std::complex<float> r_matrix[MAX_TX_PORTS][MAX_TX_PORTS], /// Untransposed R matrix
                              float squared_norms_[MAX_TX_PORTS],
                              int * detection_reversed_order_,
                              int num_rx_ports_,
                              int num_tx_ports_) {

    int computed[MAX_TX_PORTS];
    memset(&computed, 0, MAX_TX_PORTS * sizeof(int));
    int argmin = 0;
    bool first = 1;
    float temp_squared_norm = 0;

    //int count_r_col = 0;

    for(int i = 0; i < num_tx_ports_; i++) {

        //count_r_col = i + 1;
        /// Find the column with lowest norm
        if(i < num_tx_ports_ - 1) { /// Stop if the last column has been computed
            for(int j = 0; j < num_tx_ports_; j++) {
                if(not computed[j]) {
                    if (first) {
                        argmin = j;
                        temp_squared_norm = squared_norms_[j];
                        first = 0;
                        continue;
                    } else {
                        if (squared_norms_[j] < temp_squared_norm) {
                            argmin = j;
                            temp_squared_norm = squared_norms_[j];
                        }
                    }
                }
            }

        computed[argmin] = 1; /// indicate that the current column of Q must not be modified in the next iterations
        first = 1;

        /*
        /// exchange columns in Q
        memcpy(q_temp, q_matrix[argmin], num_rx_ports_ * sizeof(std::complex<float>));
        memcpy(q_matrix[argmin], q_matrix[i], num_rx_ports_ * sizeof(std::complex<float>));
        memcpy(q_matrix[i], q_temp, num_rx_ports_ * sizeof(std::complex<float>));

        // exhchange columns in R
        memcpy(r_temp, r_matrix[argmin], num_tx_ports_ * sizeof(std::complex<float>));
        memcpy(r_matrix[argmin], r_matrix[i], num_tx_ports_ * sizeof(std::complex<float>));
        memcpy(r_matrix[i], r_temp, num_tx_ports_ * sizeof(std::complex<float>));
        */

        } else {
            for(int j = 0; j < num_tx_ports_; j++) {
                if(not computed[j]) {
                    argmin = j;
                }
            }
        }

        detection_reversed_order_[i] = argmin;

        /// Compute diagonal r(argmin, argmin) coefficient
        //r_matrix[i][i] = sqrt(squared_norms_[argmin]);
        r_matrix[argmin][argmin] = sqrt(squared_norms_[argmin]);

        /// Normalize column q_argmin
        for(int j = 0; j < num_rx_ports_; j++) {
            q_matrix[argmin][j] /= r_matrix[argmin][argmin].real();
            //q_matrix[argmin][j] /= r_matrix[i][i].real();
            //q_matrix[i][j] /= r_matrix[i][i].real();
        }

        /// Project other columns vectors onto q_argmin and orthogonalize.
        if(i < num_tx_ports_ - 1) { /// Stop if the last column has been computed
            for(int j = 0; j < num_tx_ports_; j++) {
                if(not computed[j]) {
                    //r_matrix[i][count_r_col] = conj(q_matrix[argmin][0]) * q_matrix[j][0];
                    r_matrix[argmin][j] = conj(q_matrix[argmin][0]) * q_matrix[j][0];

                    for(int k = 1; k < num_rx_ports_; k++) {
                        //r_matrix[i][count_r_col] += conj(q_matrix[argmin][k]) * q_matrix[j][k];
                        r_matrix[argmin][j] += conj(q_matrix[argmin][k]) * q_matrix[j][k];
                    }

                    for(int k = 0; k < num_rx_ports_; k++) {
                        //q_matrix[j][k] = q_matrix[j][k] - r_matrix[i][count_r_col] * q_matrix[argmin][k];
                        q_matrix[j][k] = q_matrix[j][k] - r_matrix[argmin][j] * q_matrix[argmin][k];
                    }

                    /*
                    squared_norms_[j] -= (r_matrix[i][count_r_col].real() * r_matrix[i][count_r_col].real() +
                                          r_matrix[i][count_r_col].imag() * r_matrix[i][count_r_col].imag());
                    count_r_col++; */

                    squared_norms_[j] -= (r_matrix[argmin][j].real() * r_matrix[argmin][j].real() +
                                          r_matrix[argmin][j].imag() * r_matrix[argmin][j].imag());

                }
            }
        /*
        for(int j = i + 1; j < num_tx_ports_; j++) {
            r_matrix[j][i] = conj(q_matrix[i][0]) * q_matrix[j][0];
            for(int k = 1; k < num_rx_ports_; k++) {
                //r_matrix[i][count_r_col] += conj(q_matrix[argmin][k]) * q_matrix[j][k];
                r_matrix[j][i] += conj(q_matrix[i][k]) * q_matrix[j][k];
            }
            for(int k = 0; k < num_rx_ports_; k++) {
                //q_matrix[j][k] = q_matrix[j][k] - r_matrix[i][count_r_col] * q_matrix[argmin][k];
                q_matrix[j][k] = q_matrix[j][k] - r_matrix[j][i] * q_matrix[argmin][k];
            }

            squared_norms_[j] -= (r_matrix[j][i].real() * r_matrix[j][i].real() +
                                  r_matrix[j][i].imag() * r_matrix[j][i].imag());
            } */
        }
    }
}

/** Performs Sorted QR decomp. Matrix are provided as 1-D arrays so index are computed manually.
 *
 * @param[in] q_matrix : transposed Q matrix
 * @param[in] r_matrix : untransposed R matrix
 * @param[in] squared_norms_ : squared norms of columns in Q
 * @param[in, out] detection_reversed_order_ : reversed detection order (the last
 * @param[in] num_cols_q_ : Number of columns in Q
 * @param[in] num_cols_r_ : Number of rows in R
 * @param[in] num_rows_q_ : Number of rows in Q
 * @param[in] num_rows_r_ : Number of rows in R
 */
void inline compute_qr_decomp(std::complex<float> * q_matrix, /// Load q matrix with channel coefficients before QR decomp
                              std::complex<float> * r_matrix,
                              float * squared_norms_,
                              int * detection_reversed_order_,
                              int num_cols_q_,
                              int num_cols_r_,
                              int num_rows_q_,
                              int num_rows_r_) {

    int computed[MAX_TX_PORTS];
    memset(&computed, 0, num_cols_q_*sizeof(int));

    int argmin = 0;
    bool first = 1;
    float temp_squared_norm = 0;
    //float test = 0;
    //float test2 = 0;

    //std::cout << "norms berfore qr decomp" << std::endl;
    //for(int i = 0; i < num_cols_q_; i++) {
    //    std::cout << squared_norms_[i] << std::endl;
    //}

    /**
    cout << "Q columns : " << endl;
    for (int col = 0; col < num_cols_q_; col++) {
    cout << "col : " << col << endl;
    for(int row = 0; row < num_rows_r_; row++) {
    cout << q_matrix[col * num_rows_q_ + row] << endl;
    }
    } */

    for(int i = 0; i < num_cols_q_; i++) {
        /// Find the column with lowest norm
        if(i < num_cols_q_ - 1) { /// Stop if the last column has been computed
            for(int j = 0; j < num_cols_q_; j++) {
                if(not computed[j]) {
                    if (first) {
                        argmin = j;
                        temp_squared_norm = squared_norms_[j];
                        first = 0;
                        continue;
                    } else {
                        if (squared_norms_[j] < temp_squared_norm) {
                            argmin = j;
                            temp_squared_norm = squared_norms_[j];
                        }
                    }
                }
            }

            computed[argmin] = 1; /// indicate that the current column of Q must not be modified in the next iterations
            first = 1;

        } else {
            for(int j = 0; j < num_cols_q_; j++) {
                if(not computed[j]) {
                    argmin = j;
                }
            }
        }

        detection_reversed_order_[i] = argmin;

        /**
        cout << "iteration : " << i << endl;
        cout << "argmin : " << argmin << endl;
        cout << "q_matrix coefs : " << endl;
        for(int j = 0; j < num_rows_q_; j++) {
            cout << q_matrix[argmin * num_rows_q_ + j] << endl;
        }

        cout << "squared norms : " << endl;
        for(int j = 0; j < num_cols_q_; j++) {
            cout << squared_norms_[j] << endl;
        } */

        /// Compute diagonal r(argmin, argmin) coefficient
        r_matrix[argmin * num_rows_r_ + argmin] = sqrt(squared_norms_[argmin]);
        /// Normalize column q_argmin
        for(int j = 0; j < num_rows_q_; j++) {
            q_matrix[argmin * num_rows_q_ + j].real(q_matrix[argmin * num_rows_q_ + j].real() / r_matrix[argmin * num_rows_r_ + argmin].real());
            q_matrix[argmin * num_rows_q_ + j].imag(q_matrix[argmin * num_rows_q_ + j].imag() / r_matrix[argmin * num_rows_r_ + argmin].real());
        }
        /**
        cout << "r coef : " << r_matrix[argmin * num_rows_r_ + argmin] << endl;

        cout << "q_argmin after normalization : " << endl;
        for(int j = 0; j < num_rows_q_; j++) {
            cout << q_matrix[argmin * num_rows_q_ + j] << endl;
        } */

        //cout << "norm : " << endl;
        /**
        for(int j = 0; j < num_rows_q_; j++) {
            test += q_matrix[argmin * num_rows_q_ + j].real() * q_matrix[argmin * num_rows_q_ + j].real()+
                       q_matrix[argmin * num_rows_q_ + j].imag() * q_matrix[argmin * num_rows_q_ + j].imag();
        }
        cout << "norm : " << test << endl;
        */
        /// Project other columns vectors onto q_argmin and orthogonalize.
        if(i < num_cols_q_ - 1) { /// Stop if the last column has been computed
            for(int j = 0; j < num_cols_q_; j++) {
                //test2 = 0;
                if(not computed[j]) {
                    r_matrix[j * num_rows_r_ + argmin] = conj(q_matrix[argmin * num_rows_q_]) * q_matrix[j * num_rows_q_];
                    for(int k = 1; k < num_rows_q_; k++) {
                        r_matrix[j * num_rows_r_ + argmin] += conj(q_matrix[argmin * num_rows_q_ + k]) * q_matrix[j * num_rows_q_ + k];
                    }
                    //cout << "rij : " << endl;
                    //cout << r_matrix[j * num_rows_r_ + argmin] << endl;

                    /**
                    cout << "q_matrix berfore othogonalization" << endl;
                    for(int k = 0; k < num_rows_q_; k++) {
                        cout << q_matrix[j * num_rows_q_ + k] << endl;
                    } */
                    for(int k = 0; k < num_rows_q_; k++) {
                        q_matrix[j * num_rows_q_ + k] = q_matrix[j * num_rows_q_ + k] - r_matrix[j * num_rows_r_ + argmin] * q_matrix[argmin * num_rows_q_ + k];
                    }

                    /**
                    cout << "manual norm : ";
                    for(int k = 0; k < num_rows_q_; k++) {
                        test2 += pow(abs(q_matrix[j * num_rows_q_ + k]), 2);
                    }
                    cout << test2 << endl; */

                    squared_norms_[j] = squared_norms_[j] - (r_matrix[j * num_rows_r_ + argmin].real() * r_matrix[j * num_rows_r_ + argmin].real() +
                                                             r_matrix[j * num_rows_r_ + argmin].imag() * r_matrix[j * num_rows_r_ + argmin].imag());

                    //cout << "squared norm : " << squared_norms_[j] << endl;
                }
            }
        }
    }

    //std::cout << "norms after qr decomp : " << std::endl;
    //for(int i = 0; i < num_cols_q_; i++) {
    //    std::cout << squared_norms_[i] << std::endl;
    //}
}

/** QRD without reordering equalization.
 *
 * @param[in] channel_coefficients_ : matrix of channel coefficients
 * @param[in] pdsch_samples : received PDSCH samples on each RX port
 * @param[in] num_re_pdsch_ : Number of PDSCH REs
 * @param[in, out] equalized_symbols_ : buffer of equalized symbols
 * @param[in] nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param[in] nb_rx_ports_ : Number of RX ports
 * @param[in] constellation_symbols : constellation symbols (see variables.cpp)
 * @param[in, out] detected_symbols_ : final buffer of detected symbols
 * @param[in] constellation_type_ : type of constellation (see variables.cpp)
 */
void mimo_vblast_decoder_qr_decomp_no_reordering(
        std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_RX_PORTS],
        const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
        int num_re_pdsch_,
        std::complex<float>
        *equalized_symbols_,
        int nb_tx_dmrs_ports_,
        int nb_rx_ports_,
        std::complex<float>
        *constellation_symbols,
        int *detected_symbols_,
        int constellation_type_
);

/** QRD without reordering equalization working directly on the channel matrix coefficients
 *
 * @param[in] channel_coefficients_ : matrix of channel coefficients
 * @param[in] pdsch_samples : received PDSCH samples on each RX port
 * @param[in] num_re_pdsch_ : Number of PDSCH REs
 * @param[in, out] equalized_symbols_ : buffer of equalized symbols
 * @param[in] nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param[in] nb_rx_ports_ : Number of RX ports
 * @param[in] constellation_symbols : constellation symbols (see variables.cpp)
 * @param[in, out] detected_symbols_ : final buffer of detected symbols
 * @param[in] constellation_type_ : type of constellation (see variables.cpp)
 */
void mimo_vblast_decoder_qr_decomp_no_reordering_modified(
        std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_RX_PORTS],
        const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
        int num_re_pdsch_,
        std::complex<float>
        *equalized_symbols_,
        int nb_tx_dmrs_ports_,
        int nb_rx_ports_,
        std::complex<float>
        *constellation_symbols,
        int *detected_symbols_,
        int constellation_type_
);

/** QRD-CN equalization.
 *
 * @param[in] channel_coefficients_ : matrix of channel coefficients
 * @param[in] pdsch_samples : received PDSCH samples on each RX port
 * @param[in] num_re_pdsch_ : Number of PDSCH REs
 * @param[in, out] equalized_symbols_ : buffer of equalized symbols
 * @param[in] nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param[in] nb_rx_ports_ : Number of RX ports
 * @param[in] constellation_symbols : constellation symbols (see variables.cpp)
 * @param[in, out] detected_symbols_ : final buffer of detected symbols
 * @param[in] constellation_type_ : type of constellation (see variables.cpp)
 */
void mimo_vblast_qrd_col_norm(const std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                              const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                              int num_re_pdsch_,
                              std::complex<float> * equalized_symbols_,
                              int nb_tx_dmrs_ports_,
                              int nb_rx_ports_,
                              std::complex<float> * constellation_symbols,
                              int * detected_symbols_,
                              int constellation_type_);

/** QRD-CN equalization working directly on channel coefficients
 *
 * @param[in] channel_coefficients_ : matrix of channel coefficients
 * @param[in] pdsch_samples : received PDSCH samples on each RX port
 * @param[in] num_re_pdsch_ : Number of PDSCH REs
 * @param[in, out] equalized_symbols_ : buffer of equalized symbols
 * @param[in] nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param[in] nb_rx_ports_ : Number of RX ports
 * @param[in] constellation_symbols : constellation symbols (see variables.cpp)
 * @param[in, out] detected_symbols_ : final buffer of detected symbols
 * @param[in] constellation_type_ : type of constellation (see variables.cpp)
 */
void mimo_vblast_qrd_col_norm_modified(std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                      const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                      int num_re_pdsch_,
                                      std::complex<float> * equalized_symbols_,
                                      int nb_tx_dmrs_ports_,
                                      int nb_rx_ports_,
                                      std::complex<float> * constellation_symbols,
                                      int * detected_symbols_,
                                      int constellation_type_);

/******************** Steps of SQRD divided in different functions *********************/
/** Loads the untransposed channel coefficients in Q
 *
 * @param[in] channel_coefficients_ : untransposed channels coefficients
 * @param[in, out ] q_matrix : transposed Q matrix
 * @param[in] num_re_pdsch_ : Number of PDSCH RE to be computed
 * @param[in] nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param[in] nb_rx_ports_ : Number of RX ports
 */
void mimo_vblast_decoder_load_channel_coefficients_in_q(const std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
            std::complex<float> q_matrix[][MAX_TX_PORTS][MAX_RX_PORTS],
            int num_re_pdsch_,
            int nb_tx_dmrs_ports_,
            int nb_rx_ports_);

/** Computes the QR decomposition directly on the channel coefficients
 *
 * @param[in, out ] r_matrix : untransposed R matrix
 * @param[in, out] channel_coefficients : channel coefficients
 * @param[in, out] detection_reversed_orders : precomputed detection reversed orders
 * @param[in] pdsch_samples : received PDSCH samples
 * @param[in] num_re_pdsch_ : Number of REs to be computed
 * @param[in] nb_tx_dmrs_ports_ :  Number of TX DMRS ports
 * @param[in] nb_rx_ports_ : Number of RX ports
 */
void mimo_vblast_decoder_compute_qr_decomp(std::complex<float> r_matrix[][MAX_TX_PORTS][MAX_TX_PORTS],
                                           std::vector<std::complex<float>> channel_coefficients[MAX_TX_PORTS][MAX_RX_PORTS],
                                           int detection_reversed_orders[][MAX_TX_PORTS],
                                           const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                           int num_re_pdsch_,
                                           int nb_tx_dmrs_ports_,
                                           int nb_rx_ports_);

#if defined(__AVX2__)
/** Computes QR decomposition for 4 REs at once using AVX2.
 *
 * @param[in, out] r_matrix : R matrix for each RE
 * @param[in, out] q_matrix : transposed Q matrix for each RE
 * @param[in] channel_coefficients : transposed channel coefficients
 * @param[in, out] detection_reversed_orders : reversed detection orders on each RE (the last element correspond to the first layer to be decoded)
 * @param[in] pdsch_samples : received PDSCH sampels
 * @param[in] num_re_pdsch_ : Number of REs to be computed
 * @param[in] nb_tx_dmrs_ports_ :  Number of TX DMRS ports
 * @param[in] nb_rx_ports_ : Number of RX ports
 */
void mimo_vblast_decoder_compute_qr_decomp(std::complex<float> r_matrix[][MAX_TX_PORTS][MAX_TX_PORTS],
                                            std::complex<float> q_matrix[][MAX_TX_PORTS][MAX_RX_PORTS],
                                            //std::vector<std::complex<float>> channel_coefficients[MAX_TX_PORTS][MAX_RX_PORTS],
                                            int detection_reversed_orders[][MAX_TX_PORTS],
                                            const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                            int num_re_pdsch_,
                                            int nb_tx_dmrs_ports_,
                                            int nb_rx_ports_);
#endif

/** Multiplies the received PDSCH samples by Q^h.
 *
 * @param[in] pdsch_samples : received PDSCH samples
 * @param[in] pdsch_channel_coefficients_ : channel coefficients modified during QR decomposition performed directly on the channel coefficients
 * @param[in] num_re_pdsch_ : Number of REs to be computed
 * @param[in, out] equalized_symbols_ : final buffer of equalized symbols
 * @param[in] nb_tx_dmrs_ports_ : Number of TX ports
 * @param[in] nb_rx_ports_ : Number of RX ports
 */
void mimo_vblast_decoder_qr_decomp_multiply_by_q_matrix(const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                                        std::vector<std::complex<float>> pdsch_channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                                                        int num_re_pdsch_,
                                                        std::complex<float> * equalized_symbols_,
                                                        int nb_tx_dmrs_ports_,
                                                        int nb_rx_ports_);
/** Multiplies the received PDSCH samples by Q^h.
 *
 * @param[in] pdsch_samples : received PDSCH samples
 * @param[in, out] q_matrix : precomputed Q matrix for each RE
 * @param[in] num_re_pdsch_ : Number of REs to be computed
 * @param[in, out] equalized_symbols_ : final buffer of equalized symbols
 * @param[in] nb_tx_dmrs_ports_ : Number of TX ports
 * @param[in] nb_rx_ports_ : Number of RX ports
 */
void mimo_vblast_decoder_qr_decomp_multiply_by_q_matrix(const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                                        std::complex<float> q_matrix[][MAX_RX_PORTS][MAX_TX_PORTS],
                                                        int num_re_pdsch_,
                                                        std::complex<float> * equalized_symbols_,
                                                        int nb_tx_dmrs_ports_,
                                                        int nb_rx_ports_);

/** Performs SIC detection based on the detection order and the R matrix
 *
 * @param[in, out] r_matrix  : precomputed R matrix
 * @param[in] detection_reversed_order : reversed detection order (the last element correponds to the first layer to be decoded)
 * @param[in] equalized_symbols_ : final buffer of equalized symbols
 * @param[in, out] detected_symbols_ : finall buffer of detected symbols
 * @param[in] constellation_symbols : constellation symbols (see variables.cpp)
 * @param[in] constellation_type_ : constellation type (see variables.cpp)
 * @param[in] num_re_pdsch_ : Number of REs to be computed
 * @param[in] nb_tx_dmrs_ports_ : Number of TX DMRS ports
 */
void mimo_vblast_decoder_sic_detection(std::complex<float> r_matrix[][MAX_TX_PORTS][MAX_TX_PORTS],
                                       int detection_reversed_order[][MAX_TX_PORTS],
                                       std::complex<float> * equalized_symbols_,
                                       int * detected_symbols_,
                                       std::complex<float> * constellation_symbols,
                                       int constellation_type_,
                                       int num_re_pdsch_,
                                       int nb_tx_dmrs_ports_);
/***************************************************************************************/

void
mimo_vblast_decoder_qr_decomp(const std::vector<std::complex<float>> channel_coefficients_[MAX_TX_PORTS][MAX_RX_PORTS],
                              const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                              int num_re_pdsch_,
                              std::complex<float> *equalized_symbols_,
                              int nb_tx_dmrs_ports_,
                              int nb_rx_ports_,
                              std::complex<float> *constellation_symbols,
                              int *detected_symbols_,
                              int constellation_type_);

void mimo_vblast_decoder_qr_decomp(const std::complex<float> *channel_coefficients_,
                                   //const float *squared_norms_channel_coefs_,
                                   const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                   //const int *pdsch_positions_,
                                   int num_re_pdsch_,
                                   //int pdsch_length_,
                                   //int fft_size_,
                                   std::complex<float> *equalized_symbols_,
                                   int &nb_tx_dmrs_ports_,
                                   int &nb_rx_ports_,
                                   //const int &pdsch_start_symbol_,
                                   std::complex<float> *constellation_symbols,
                                   int *detected_symbols_,
                                   const int &constellation_type_);

void mimo_vblast_decoder_qr_decomp(const std::complex<float> *channel_coefficients_,
                                   const std::vector<std::vector<std::complex<float>>> &pdsch_samples,
                                   const int *pdsch_positions_,
                                   int num_re_pdsch_,
                                   int pdsch_length_,
                                   int fft_size_,
                                   std::complex<float> *equalized_symbols_,
                                   int &nb_tx_dmrs_ports_,
                                   int &nb_rx_ports_,
                                   const int &pdsch_start_symbol_,
                                   std::complex<float> *constellation_symbols,
                                   int *detected_symbols_,
                                   const int &constellation_type_);

void mimo_vblast_decoder_qr_decomp(const std::vector<std::vector<std::vector<std::complex<float>>>> &received_grids_,
                                    const std::vector<std::vector<std::vector<std::vector<std::complex<float>>>>> &channel_coefficients_,
                                    const std::vector<std::pair<int, int>> &pdsch_positions_,
                                    const std::vector<int> &ordered_tx_antenna_indexes_,
                                    std::vector<int> &temp_ordered_tx_antenna_indexes_,
                                    int &nb_tx_dmrs_ports_,
                                    int &nb_rx_ports_,
                                    int &pdsch_start_,
                                    int &constellation_type_,
                                    std::vector<std::complex<float>> &equalized_symbols_,
                                    std::vector<int> &detected_symbols_,
                                    const std::vector<std::complex<float>> &constellation_symbols_,
                                    std::vector<std::vector<std::complex<float>>> &channel_matrix_,
                                    std::vector<std::vector<std::complex<float>>> &r_matrix_,
                                    std::vector<std::vector<std::complex<float>>> &q_matrix_,
                                    std::vector<std::complex<float>> &q_h_symbols_, /// **size of N_TX**
                                    std::vector<std::complex<float>> &received_symbols_, /// **size of N_TX **
                                    std::vector<float> &column_norms);

#endif //USRP_MIMO_VBLAST_H
