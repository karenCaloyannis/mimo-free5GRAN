/*
    Copyright (C) 2023  Telecom Paris

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.

    This file contains equalization algorithms. synchronize_slots and ml_detectors
    functions are reused from free5GRAN and are modified.
*/


#ifndef CHANNEL_ESTIMATION_H
#define CHANNEL_ESTIMATION_H

#include <string>
#include <complex>
#include <utility>
#include <vector>
#include <iostream>
#include <complex>
#include <chrono>
#include <numeric>
#include <algorithm>

#if defined(__AVX2__)
#include <x86intrin.h>
#include <immintrin.h>
#endif

#include <fftw3.h>
#define _USE_MATH_DEFINES
#include <cmath>

#include "../../free5gran_utils.h"
#include "../../variables/variables.h"

/******************************* AVX functions **********************************************/
#if defined(__AVX2__)
/** Interpolate channel coefficients on the whole PDSCH allocation using AVX2 to perform computations
 *  on 4 REs at once.
 *
 *  WORKS ONLY FOR DMRS CONFIG. TYPE 1
 *
 * @param[in, out] coef_grid : contains all the coefficients for the PDSCH allocation including DMRS
 * @param[in] dmrs_coefs_  : DMRS coefficients
 * @param[in] cdm_group_number : CDM group number of the TX DMRS port
 * @param[in] cdm_group_size : CDM group size of the TX DMRS port
 * @param[in] rx_antenna_port_number_ : RX port index
 * @param[in] dmrs_symbols_ : Position of DMRS on OFDM symbols within the slot
 * @param[in] pdsch_start_symbol_ : First symbol of the PDSCH within the slot
 * @param[in] dmrs_symbols_per_grid_ : Number of DMRS symbols per slot
 * @param[in] dmrs_sequence_size_ : DMRS sequence size in frequency per slot
 * @param[in] fft_size_ : FFT size of the grid
 * @param[in] symbols_per_grid_ : Number of symbols in the grid
 * @param[in] nb_tx : Number of TX DMRS ports
 * @param[in] nb_rx : Number of RX ports
 */
void interpolate_coefs_avx(std::complex<float> * coef_grid, /// TODO: mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                           std::complex<float> * dmrs_coefs_, /// TODO : mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                           int cdm_group_number, /// Number of the DMRS CDM group to be interpolated, 0 or 1
                           int cdm_group_size,
                           int rx_antenna_port_number_,
                           int * dmrs_symbols_,
                           int pdsch_start_symbol_,
                           int dmrs_symbols_per_grid_,
                           int dmrs_sequence_size_,
                           int fft_size_,
                           int symbols_per_grid_,
                           int nb_tx,
                           int nb_rx);

/** Perforsm channel estimation on DMRS using AVX2 to compute 4 REs at once.
 * WORKS ONLY FOR DMRS CONFIG. TYPE 1
 *  TODO : implement double symbol DMRS
 *
 * @param[in] tx_dmrs_ports_ : Array containing the TX DMRS port numbers minus 1000
 * @param[in] cdm_groups_sizes_ : Array containing the sizes of CDM groups for each DMRS port
 * @param[in] dmrs_symbols_ : Array containing the position of DMRS on OFDM symbols within one slot
 * @param[in] received_dmrs_samples : received DMRS samples in each CDM group
 * @param[in] dmrsSequences_ : array containing the DMRS sequence for each DMRS port and each DMRS symbol
 *                         of the slot
 * @param[in, out] estimated_chan_coefs : containing the estimated channel coefficients for each path
 *                                        between the DMRS ports and the current receiver
 * @param[in] double_symbol_ : True if double DMRS symbols is used
 * @param[in] dmrs_sequence_size_ : DMRS sequence size in frequency.
 * @param[in] slot_number_ : the slot number in the whole frame
 * @param[in] num_dmrs_symbols_per_slot_ : Number of DMRS OFDM symbols per slot
 * @param[in] num_tx_ports_ : number of TX DMRS ports
 * @param[in] nb_pdsch_slots_ : number of slots containing PDSCH in the frame
 * @param[in] cdm_groups_ : array containing the CDM groups
 */
void estimate_pilots_avx(const int * tx_dmrs_ports_,
                         const int * cdm_groups_sizes_,
                         const int * dmrs_symbols_,
                         std::vector<std::complex<float>> * received_dmrs_samples,
                         std::complex<float> * dmrsSequences_,
                         std::complex<float> * estimated_chan_coefs, /// TODO: Ajouter l'offset au pointer en dehors de la fonction
                         bool double_symbol_,
                         int dmrs_sequence_size_,
                         int slot_number_,
                         int num_dmrs_symbols_per_slot_,
                         int num_tx_ports_,
                         int nb_pdsch_slots_,
                         const int * cdm_groups_);
#endif

/** Interpolate channel coefficients on the whole PDSCH allocation
 *  on 4 REs at once.
 *
 *  WORKS ONLY FOR DMRS CONFIG. TYPE 1
 *
 * @param[in, out] coef_grid : contains all the coefficients for the PDSCH allocation including DMRS
 * @param[in] dmrs_coefs_  : DMRS coefficients
 * @param[in] cdm_group_number : CDM group number of the TX DMRS port
 * @param[in] cdm_group_size : CDM group size of the TX DMRS port
 * @param[in] rx_antenna_port_number_ : RX port index
 * @param[in] dmrs_symbols_ : Position of DMRS on OFDM symbols within the slot
 * @param[in] pdsch_start_symbol_ : First symbol of the PDSCH within the slot
 * @param[in] dmrs_symbols_per_grid_ : Number of DMRS symbols per slot
 * @param[in] dmrs_sequence_size_ : DMRS sequence size in frequency per slot
 * @param[in] fft_size_ : FFT size of the grid
 * @param[in] symbols_per_grid_ : Number of symbols in the grid
 * @param[in] nb_tx : Number of TX DMRS ports
 * @param[in] nb_rx : Number of RX ports
 */
void interpolate_coefs(std::complex<float> * coef_grid, /// TODO: mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                       std::complex<float> * dmrs_coefs_, // TODO : mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                       int cdm_group_number, /// Number of the DMRS CDM group to be interpolated, 0 or 1
                       int cdm_group_size_,
                       int rx_antenna_port_number_,
                       int * dmrs_symbols_,
                       int pdsch_start_symbol_,
                       int dmrs_symbols_per_grid_,
                       int dmrs_sequence_size_,
                       int fft_size_,
                       int symbols_per_grid_,
                       int nb_tx,
                       int nb_rx);

/** Perforsm channel estimation on DMRS.
 * WORKS ONLY FOR DMRS CONFIG. TYPE 1
 *  TODO : implement double symbol DMRS
 *
 * @param[in] tx_dmrs_ports_ : Array containing the TX DMRS port numbers minus 1000
 * @param[in] cdm_groups_sizes_ : Array containing the sizes of CDM groups for each DMRS port
 * @param[in] dmrs_symbols_ : Array containing the position of DMRS on OFDM symbols within one slot
 * @param[in] received_dmrs_samples : received DMRS samples in each CDM group
 * @param[in] dmrsSequences_ : array containing the DMRS sequence for each DMRS port and each DMRS symbol
 *                         of the slot
 * @param[in, out] estimated_chan_coefs : containing the estimated channel coefficients for each path
 *                                        between the DMRS ports and the current receiver
 * @param[in] double_symbol_ : True if double DMRS symbols is used
 * @param[in] dmrs_sequence_size_ : DMRS sequence size in frequency.
 * @param[in] slot_number_ : the slot number in the whole frame
 * @param[in] num_dmrs_symbols_per_slot_ : Number of DMRS OFDM symbols per slot
 * @param[in] num_tx_ports_ : number of TX DMRS ports
 * @param[in] nb_pdsch_slots_ : number of slots containing PDSCH in the frame
 * @param[in] cdm_groups_ : array containing the CDM groups
 */
void estimate_pilots_cdm_groups_one_rx(const int * tx_dmrs_ports_,
                                       const int * cdm_groups_sizes_,
                                       const int * dmrs_symbols_,
                                       std::vector<std::complex<float>> * received_dmrs_samples,
                                       std::complex<float> * dmrsSequences_,
                                       std::complex<float> * estimated_chan_coefs, /// TODO: Ajouter l'offset au pointer en dehors de la fonction
                                       bool double_symbol_,
                                       int dmrs_sequence_size_,
                                       int slot_number_,
                                       int num_dmrs_symbols_per_slot_,
                                       int num_tx_ports_,
                                       int nb_pdsch_slots_,
                                       const int * cdm_groups_);

/** FROM FREE5GRAN
 *  Synchronizes the received slots, so that the first sample of the PSS is
 *  the first sample of the synchronized signal. We only use the PSS to synchronize.
 *
 * @param[in] received_signal     : received time-domain signal
 * @param[out] synchronized_grid  : synchronized frequency-domain grid
 * @param[out] synchronized_signal : synchronized time-domain signal
 * @param[in] nbSubframes              : number of subframes received
 * @param[in] subframes_to_keep        : number of subframes to keep in the synchronized signal
 */
void synchronize_slots(const std::vector<std::complex<float>> &received_signal,
                       int num_samples_received_signal_,
                       int subframes_to_keep,
                       int fft_size_,
                       int symbols_per_subframe_,
                       int cp_length_pss_,
                       const int * cum_sum_cp_lengths_,
                       const int * cp_lengths_,
                       int scs_,
                       int &n_id_2_,
                       int &pss_start_index_,
                       int &synchronization_index_,
                       int downsampling_factor_,
                       bool known_n_id_2,
                       std::vector<std::vector<std::complex<float>>> &time_signals_pss_,
                       std::vector<std::vector<std::complex<float>>> &time_signals_pss_downsampled_);

/** Extracts the PDSCH channel coefficients in the grid of interpolated channel coefficients
 *
 * @param[in] coef_grid_ : the PDSCH allocation with interpolated coefficients
 * @param[in, out] pdsch_channel_coefficients_ : the coefficients on PDSCH position only
 * @param[in] pdsch_start_ : first OFDM symbol of the PDSCH in the slot
 * @param[in] dmrs_symbols_ : array containing the position of DMRS on OFDM symbols in one slot
 * @param[in] fft_size_ : FFT size of the PDSCH allocation
 * @param[in] pdsch_length_ : PDSCH length parameter (number of PDSCH OFDM symbols in the slot)
 * @param[in] dmrs_config_type_ : DMRS configuration type 1 or type 2
 * @param[in] num_cdm_groups_without_data_ : Number of CDM groups without data parameter
 */
void get_pdsch_channel_coefficients(std::complex<float> * coef_grid_,
                                    std::complex<float> * pdsch_channel_coefficients_,
                                    int pdsch_start_,
                                    int * dmrs_symbols_,
                                    int fft_size_,
                                    int pdsch_length_,
                                    int dmrs_config_type_,
                                    int num_cdm_groups_without_data_);

/** ML detector for QPSK constellation
 *
 * @param[in] received_symbols : input symbols
 * @param[in, out] detected_symbols : output symbol indexes in QPSK constellation (see variables.cpp)
 */
void ml_detector_qpsk(const std::vector<std::complex<float>> &received_symbols,
                      std::vector<int> &detected_symbols);

/** ML detector for BPSK constellation
 *
 * @param[in] received_symbols : input symbols
 * @param[in, out] detected_symbols : output symbol indexes in BPSK constellation (see variables.cpp)
 */
void ml_detector_bpsk(const std::vector<std::complex<float>> &received_symbols,
                      std::vector<int> &detected_symbols);

/** ML detector for QPSK constellation
 *
 * @param[in] equalized_symbol : input symbol
 * @param[in, out] detected_symbols : output symbol index in QPSK constellation (see variables.cpp)
 */
void ml_detector_qpsk(const std::complex<float> &equalized_symbol,
                      int &detected_symbol);

/** ML detector for BPSK constellation
 *
 * @param[in] equalized_symbol : input symbols
 * @param[in, out] detected_symbol : output symbol indexes in BPSK constellation (see variables.cpp)
 */
void ml_detector_bpsk(const std::complex<float> &equalized_symbol,
                      int &detected_symbol);

/** ML detector for QPSK constellation
 *
 * @param[in] received_symbols : input symbols
 * @param[in, out] detected_symbols :output symbol indexes in BPSK constellation (see variables.cpp)
 * @param[in] buffer_size : number of received symbols
 */
void ml_detector_qpsk(std::complex<float> * received_symbols,
                      int * detected_symbols,
                      const int &buffer_size);

/** ML detector for BPSK constellation
 *
 * @param[in] received_symbols : input symbols
 * @param[in, out] detected_symbols :output symbol indexes in BPSK constellation (see variables.cpp)
 * @param[in] buffer_size : number of received symbols
 */
void ml_detector_bpsk(std::complex<float> * received_symbols,
                      int * detected_symbols,
                      const int &buffer_size);

/** ML detector for QPSK constellation
 *
 * @param[in] equalized_symbol_ : input symbol
 * @return symbol index in QPSK constellation (see variables.cpp)
 */
int inline ml_detector_qpsk(const std::complex<float> &equalized_symbol_) {

    /// Check in which Region of decision the symbol is and assign the corresponding symbol index
    /// (check the integer corresponding to each qpsk symbol in variables.cpp)
    if (equalized_symbol_.real() < 0) {
        if(equalized_symbol_.imag() < 0) {
            return 2;
        } else {
            return 1;
        }
    } else {
        if(equalized_symbol_.imag() < 0) {
            return 3;
        } else { /// Real and Imag. part positive
            return 0;
        }
    }
}

/** ML detector for BPSK constellation
 *
 * @param[in] equalized_symbol_ : input symbol
 * @return symbol index in BPSK constellation (see variables.cpp)
 */
int inline ml_detector_bpsk(const std::complex<float> &equalized_symbol_) {
    if(equalized_symbol_.real() < 0) {
        return 1;
    } else {
        return  0;
    }
}

#if defined(__AVX2__)
/** ML detector for symbols stored in __m256 vectors.
 *  Contains the symbols for 4 REs.
 *
 *  See variables.cpp for parameters
 *
 * @param[in] equalized_symbol_ : input symbols on each RE
 * @param[in, out] detected_symbol_indexes_ : output symbol index on each RE
 * @param[in, out] detected_symbol_ : output symbols in each RE
 * @param[in] constellation_type : constellation type
 */
void inline ml_detector_mm256(const __m256 &equalized_symbol_,
                              __m128i &detected_symbol_indexes_,
                              __m256 &detected_symbol_,
                              int constellation_type) {
    __m256 zero_vec = _mm256_set1_ps(0);

    /// Compare real and imag part simultaneously
    __m256 verif = _mm256_cmp_ps(equalized_symbol_, zero_vec, _CMP_LT_OQ); /// less than

    if(constellation_type == 0) { /// QPSK

        for(int i = 0; i < 4; i ++) {
            if(verif[2 * i] == 0xFFFFFFFF) {
                if(verif[2 * i + 1] == 0xFFFFFFFF) {
                    detected_symbol_indexes_[i] = 2;
                    detected_symbol_[2 * i] = qpsk[2].real();
                    detected_symbol_[2 * i + 1] = qpsk[2].imag();
                } else {
                    detected_symbol_indexes_[i] = 1;
                    detected_symbol_[2 * i] = qpsk[1].real();
                    detected_symbol_[2 * i + 1] = qpsk[1].imag();
                }
            } else {
                if(verif[2 * i + 1] == 0xFFFFFFFF) {
                    detected_symbol_indexes_[i] = 3;
                    detected_symbol_[2 * i] = qpsk[3].real();
                    detected_symbol_[2 * i + 1] = qpsk[3].imag();
                } else {
                    detected_symbol_indexes_[i] = 0;
                    detected_symbol_[2 * i] = qpsk[0].real();
                    detected_symbol_[2 * i + 1] = qpsk[0].imag();
                }
            }
        }
    } else { /// BPSK
        for(int i = 0; i < 4; i++) {
            if(verif[2 * i] == 0xFFFFFFFF) {
                detected_symbol_indexes_[i] = 1;
                detected_symbol_[2 * i] = bpsk[1].real();
                detected_symbol_[2 * i + 1] = bpsk[1].imag();
            } else {
                detected_symbol_indexes_[i] = 0;
                detected_symbol_[2 * i] = bpsk[0].real();
                detected_symbol_[2 * i + 1] = bpsk[0].imag();
            }
        }
    }
}
#endif

#if defined(__AVX2__)
/** ML detector for symbols stored in __m512 vectors.
 *  Contains the symbols for 8 REs.
 *
 *  See variables.cpp for parameters
 *
 * @param[in] equalized_symbol_ : input symbols on each RE
 * @param[in, out] detected_symbol_indexes_ : output symbol index on each RE
 * @param[in, out] detected_symbol_ : output symbols in each RE
 * @param[in] constellation_type : constellation type e
 */
void inline ml_detector_mm512(const __m512 &equalized_symbol_,
                              __m256i &detected_symbol_indexes_,
                              __m512 &detected_symbol_,
                              int constellation_type) {

    __m512 zero_vec = _mm512_set1_ps(0);
    __m512 greater_than_zero = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0};
    __m512 lower_than_zero   = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};

    /// Compare real and imag part simultaneously
    __mmask16 mask = _mm512_cmp_ps_mask(equalized_symbol_, zero_vec, _CMP_LT_OQ); /// less than

    __m512 verif = _mm512_mask_blend_ps(mask, greater_than_zero, lower_than_zero);

    if(constellation_type == 0) { /// QPSK
        for(int i = 0; i < 8; i ++) {
            if(verif[2 * i]) {
                if(verif[2 * i + 1]) {
                    detected_symbol_indexes_[i] = 2;
                    detected_symbol_[2 * i] = qpsk[2].real();
                    detected_symbol_[2 * i + 1] = qpsk[2].imag();
                } else {
                    detected_symbol_indexes_[i] = 1;
                    detected_symbol_[2 * i] = qpsk[1].real();
                    detected_symbol_[2 * i + 1] = qpsk[1].imag();
                }
            } else {
                if(verif[2 * i + 1]) {
                    detected_symbol_indexes_[i] = 3;
                    detected_symbol_[2 * i] = qpsk[3].real();
                    detected_symbol_[2 * i + 1] = qpsk[3].imag();
                } else {
                    detected_symbol_indexes_[i] = 0;
                    detected_symbol_[2 * i] = qpsk[0].real();
                    detected_symbol_[2 * i + 1] = qpsk[0].imag();
                }
            }
        }
    } else { /// BPSK
        for(int i = 0; i < 8; i++) {
            if(verif[2 * i]) {
                detected_symbol_indexes_[i] = 1;
                detected_symbol_[2 * i] = bpsk[1].real();
                detected_symbol_[2 * i + 1] = bpsk[1].imag();
            } else {
                detected_symbol_indexes_[i] = 0;
                detected_symbol_[2 * i] = bpsk[0].real();
                detected_symbol_[2 * i + 1] = bpsk[0].imag();
            }
        }
    }
}
#endif

/** Compute the symbol error rate between the input vector and the
 *  known symbol indexes
 *
 * @param[in] detected_symbols : input detected symbol indexes in constellation
 * @param[in] sending_buffer_symbol_indexes_ : known symbol indexes
 * @return symbol error rate
 */
double symbol_error_rate(std::vector<int> &detected_symbols, std::vector<int> &sending_buffer_symbol_indexes_);

/** Compute the symbol error rate between the input vector and the
 *  known symbol indexes
 * @param[in] detected_symbol_indexes_ : input detected symbol indexes in constellation
 * @param[in] sending_buffer_symbol_indexes_ : known symbol indexes
 * @param buffer_size : number of symbols in the buffers
 * @return symbol error rate
 */
double symbol_error_rate(int * detected_symbol_indexes_, int * sending_buffer_symbol_indexes_, int buffer_size);


/********************** DEPRECATED OR TEST FUNCTIONS BELOW. UNUSED ****************************************************/
void compute_sic_order(std::vector<std::complex<float>> pdsch_channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                       std::vector<float> squared_norms_[MAX_RX_PORTS][MAX_TX_PORTS],
                       std::vector<std::vector<float>> &columns_norms,
                       std::vector<std::vector<int>> &sic_orders_,
                       int nb_rx_,
                       int nb_tx_,
                       int num_pdsch_re_,
                       int num_cdm_groups_without_data_);

void compute_sic_order(std::vector<std::complex<float>> interp_channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                       std::vector<float> squared_norms_[MAX_RX_PORTS][MAX_TX_PORTS],
                       std::vector<std::vector<int>> &sic_orders_,
                       int nb_rx_,
                       int nb_tx_,
                       int * dmrs_symbols_,
                       int num_dmrs_symbols_,
                       int fft_size_,
                       int num_symbols_per_grid,
                       int pdsch_start_symbol_,
                       int num_cdm_groups_without_data_);

void estimate_pilots_cdm_groups_one_rx(const int * tx_dmrs_ports_,
                                       const int * cdm_groups_sizes_,
                                       const int * dmrs_symbols_,
                                       std::vector<float> * received_dmrs_samples_real,
                                       std::vector<float> * received_dmrs_samples_imag,
                                       std::vector<float> * dmrs_coefs_real,
                                       std::vector<float> * dmrs_coefs_imag,
                                       std::vector<float> pilot_coefs_real[MAX_RX_PORTS][MAX_TX_PORTS],
                                       std::vector<float> pilot_coefs_imag[MAX_RX_PORTS][MAX_TX_PORTS],
                                       bool double_symbol_,
                                       int dmrs_sequence_size_,
                                       int slot_number_,
                                       int num_dmrs_symbols_per_slot_,
                                       int num_tx_ports_,
                                       int receiver_no_,
                                       const int * cdm_groups_);

/********************************************************** Test complex<float>[][] ***********************************/
void estimate_pilots_cdm_groups_one_rx_test(const int * tx_dmrs_ports_,
                                       const int * cdm_groups_sizes_,
                                       const int * dmrs_symbols_,
                                       bool double_symbol_,
                                       int dmrs_sequence_size_,
                                       int slot_number_,
                                       int num_dmrs_symbols_per_slot_,
                                       int num_tx_ports_,
                                       int nb_pdsch_slots_,
                                       int receiver_no_,
                                       const int * cdm_groups_,
                                       std::vector<std::complex<float>> * received_dmrs_samples,
                                       std::complex<float> * dmrsSequences_,
                                       std::complex<float> estimated_chan_coefs[MAX_TX_PORTS][MAX_DMRS_SYMBOLS][MAX_DMRS_SUBCARRIERS] /// TODO: Ajouter l'offset au pointer en dehors de la fonction
                                       );
/******************************************************************************************************/

void get_pdsch_squared_norms(float * squared_norms_,
                             float * pdsch_squared_norms_,
                             int pdsch_start_,
                             int * dmrs_symbols_,
                             int fft_size_,
                             int pdsch_length_,
                             int dmrs_config_type_,
                             int num_cdm_groups_without_data_);

void get_pdsch_channel_coefficients(std::complex<float> * coef_grid_,
                                    std::complex<float> pdsch_channel_coefficients_[][MAX_RX_PORTS][MAX_TX_PORTS],
                                    int tx_port_index,
                                    int rx_port_index,
                                    int pdsch_start_,
                                    int * dmrs_symbols_,
                                    int fft_size_,
                                    int pdsch_length_,
                                    int dmrs_config_type_,
                                    int num_cdm_groups_without_data_);

void estimate_pilots_cdm_groups_one_rx(const int * tx_dmrs_ports_,
                                       const int * cdm_groups_sizes_,
                                       const int * dmrs_symbols_,
                                       std::vector<std::complex<float>> received_dmrs_samples[MAX_RX_PORTS][MAX_NUM_CDM_GROUPS],
                                       std::vector<std::complex<float>> * dmrs_sequences_,
                                       std::vector<std::complex<float>> * estimated_chan_coefs, /// TODO: Ajouter l'offset au pointer en dehors de la fonction
                                       bool double_symbol_,
                                       int dmrs_sequence_size_,
                                       int num_dmrs_symbols_per_slot_,
                                       int num_tx_ports_,
                                       int receiver_no_,
                                       const int * cdm_groups_);

/** Computes a centered moving mean on the grid of channel coefficients given in argument.
 *  Chose the appropriate size for the frequency and time window sizes. Their value must be and odd number
 *  equal or less than the size of the computed grid.
 *
 * @param[out] averagedCoefs   : output grid containing the averaged channel coefficients.
 * @param[in] nonAveragedCoefs : input grid containing non averaged channel coefficients.
 * @param[in] freq_window_size : frequency window size (odd number, less or equal to the size of the non averaged grid.
 * @param[in] time_window_size : time window size (odd numner, less or equal to the size of the non averaged grid.
 */
void centered_mov_mean(std::vector<std::vector<std::complex<float>>> &averagedCoefs, // Coefficients moyennés
                       const std::vector<std::vector<std::complex<float>>> &nonAveragedCoefs, // Coefficients non moyennés
                       const int &freq_window_size,   // fenêtre en fréquence
                       const int &time_window_size);


/******************** Test interpolation complex<float>[][] **********************/
void interpolate_coefs_test(int dmrs_symbols_per_grid_,
                       int dmrs_sequence_size_,
                       int fft_size_,
                       int symbols_per_grid_,
                       std::complex<float> coef_grid[MAX_SYMBOLS][MAX_SUBCARRIERS], /// TODO: mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                       std::complex<float> dmrs_coefs_[MAX_DMRS_SYMBOLS][MAX_DMRS_SUBCARRIERS], // TODO : mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                       int cdm_group_number, /// Number of the DMRS CDM group to be interpolated, 0 or 1
                       int cdm_group_size,
                       int rx_antenna_port_number_,
                       int * dmrs_symbols_,
                       int pdsch_start_symbol_,
                       int nb_tx,
                       int nb_rx);
/*********************************************************************************/


/************************ Test interpolation 1 *************************************/
void interpolate_coefs_test1(std::complex<float> * coef_grid, /// TODO: mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                             std::complex<float> * dmrs_coefs_, // TODO : mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                             int cdm_group_number, /// Number of the DMRS CDM group to be interpolated, 0 or 1
                             int cdm_group_size,
                             int rx_antenna_port_number_,
                             int * dmrs_symbols_,
                             int pdsch_start_symbol_,
                             int dmrs_symbols_per_grid_,
                             int dmrs_sequence_size_,
                             int fft_size_,
                             int symbols_per_grid_,
                             int nb_tx,
                             int nb_rx);
/*******************************************************************************/

/******************* Test réalignement ******************************************/
void interpolate_coefs(std::complex<float> coef_grid[][MAX_RX_PORTS][MAX_TX_PORTS], /// TODO: mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                       std::complex<float> * dmrs_coefs_, // TODO : mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                       int cdm_group_number, /// Number of the DMRS CDM group to be interpolated, 0 or 1
                       int cdm_group_size,
                       int tx_port_index_,
                       int rx_port_index_,
                       int * dmrs_symbols_,
                       int pdsch_start_symbol_,
                       int dmrs_symbols_per_grid_,
                       int dmrs_sequence_size_,
                       int fft_size_,
                       int symbols_per_grid_,
                       int nb_tx,
                       int nb_rx);
/********************************************************************************/

/**
void interpolate_coefs_test(std::complex<float> * coef_grid, /// TODO: mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                            std::complex<float> * dmrs_coefs_, // TODO : mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                            int cdm_group_number, /// Number of the DMRS CDM group to be interpolated, 0 or 1
                            int cdm_group_size,
                            int rx_antenna_port_number_,
                            int * dmrs_symbols_,
                            int pdsch_start_symbol_,
                            int dmrs_symbols_per_grid_,
                            int dmrs_sequence_size_,
                            int fft_size_,
                            int symbols_per_grid_,
                            int nb_tx,
                            int nb_rx);*/

/** Interpolates the channel coefficients computed from DMRS signals on all the REs of the grid,
 *  in frequency domain first, then in time domain.
 *
 * @param[out] coefGrid    : output grid containing all the channel coefficients (DMRS and interpolated)
 * @param[in] dmrs_coefs   : grid containing the channels coefficients computed from DMRS (averaged or not)
 * @param[out] rsPositions : grid containing the DMRS positions.
 */
void interpolate_coefs(std::vector<std::vector<std::complex<float>>> &coefGrid, /// TODO: mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                        std::complex<float> * dmrs_coefs_, // TODO : mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                        const int &cdm_group_number, /// Number of the DMRS CDM group to be interpolated, 0 or 1
                        const int &rx_antenna_port_number_,
                        int * dmrs_symbols_,
                        const int &pdsch_start_symbol_,
                        const int &dmrs_symbols_per_grid_,
                        const int &dmrs_sequence_size_,
                        const int &fft_size_,
                        const int &symbols_per_grid_,
                        const int &nb_tx,
                        const int &nb_rx);

/************************************ Séparation interpolation **************************************************/
void call_interp_functions(std::complex<float> * coef_grid, /// TODO: mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                           std::complex<float> * dmrs_coefs_, // TODO : mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                           int cdm_group_number, /// Number of the DMRS CDM group to be interpolated, 0 or 1
                           int cdm_group_size,
                           int * dmrs_symbols_,
                           int pdsch_start_symbol_,
                           int dmrs_symbols_per_grid_,
                           int dmrs_sequence_size_,
                           int fft_size_,
                           int symbols_per_grid_);

void inline interp_cdm_group0_2(std::complex<float> * coef_grid, /// TODO: mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                         std::complex<float> * dmrs_coefs_, // TODO : mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                         int * dmrs_symbols_,
                         int pdsch_start_symbol_,
                         int dmrs_symbols_per_grid_,
                         int dmrs_sequence_size_,
                         int fft_size_,
                         int symbols_per_grid_) {

    std::complex<float> *temp_dmrs_coefs;
    std::complex<float> *temp_coef_grid;
    std::complex<float> *lower_dmrs_coef;
    std::complex<float> *upper_dmrs_coef;
    int step_symbols;

    temp_dmrs_coefs = dmrs_coefs_;
    step_symbols = 0;

    /// Optimizations if 2 DMRS ports are mutliplexed by OCC in the CDM group
    /// Fill the grid with DMRS
    for (int symbol = 0; symbol < dmrs_symbols_per_grid_; symbol++) {
        temp_coef_grid = coef_grid + (dmrs_symbols_[symbol] - pdsch_start_symbol_) *
                                     fft_size_; /// Initialize to the first subcarrier of the current symbol
        temp_dmrs_coefs = dmrs_coefs_ + symbol * dmrs_sequence_size_;
        for (int sc = 0; sc < dmrs_sequence_size_; sc++) {
            temp_coef_grid[2 * sc] = temp_dmrs_coefs[sc];
        }
    }
    /// interpolate in frequency domain
    for (int *temp_dmrs_symbols = dmrs_symbols_;
         temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {
        temp_coef_grid = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                     fft_size_; /// Initialize to the first subcarrier of the current symbol
        for (int sc = 0; sc < fft_size_ / 4 - 1; sc++) {
            /// Asign the same value to the subcarrier located in-between the two DMRS
            *(temp_coef_grid + 1) = *(temp_coef_grid);
            /// Interpolate real and imaginary part of the subcarrier located after the second DMRS
            (temp_coef_grid + 3)->real((real(*(temp_coef_grid + 4)) + real(*(temp_coef_grid + 2))) * 0.5);
            (temp_coef_grid + 3)->imag((imag(*(temp_coef_grid + 4)) + imag(*(temp_coef_grid + 2))) * 0.5);
            /// Interpolate norm and renormalize
            *(temp_coef_grid + 3) *=
                    ((abs(*(temp_coef_grid + 4)) + abs(*(temp_coef_grid + 2))) * 0.5) /
                    abs(*(temp_coef_grid + 3));
            temp_coef_grid += 4; /// Jump to the next sc to be interpolated
        }
        /// Assign the same value to the four last DMRS subcarriers
        *(temp_coef_grid + 1) = *(temp_coef_grid);
        *(temp_coef_grid + 3) = *(temp_coef_grid);
    }
    int last_dmrs_symbol = *(dmrs_symbols_ + dmrs_symbols_per_grid_ - 1);
    int *temp_dmrs_symbols = dmrs_symbols_;

    /// interpolate in time domain
    /// Only one DMRS in the PDSCH
    if (dmrs_symbols_per_grid_ == 1) {
        /// Reset pointer
        lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
        temp_coef_grid = coef_grid;
        /// Apply the same coefficients in time domain on each subcarrier
        for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {
            /// No need to apply values on DRMS symbol as we have interpolated in frequency domain
            if (symbol == *(dmrs_symbols_) - pdsch_start_symbol_) {
                temp_coef_grid += fft_size_;
                continue;
            }
            for (int j = 0; j < fft_size_; j++) {
                *(temp_coef_grid) = *(lower_dmrs_coef);
                temp_coef_grid++;
                lower_dmrs_coef++;
            }
        }
    } else {
        /// Reset pointers
        temp_coef_grid = coef_grid;
        step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
        int step = 0;
        lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
        upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
        for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {
            /// If current symbol is a DMRS symbol, do not interpolate on this symbol
            if ((symbol == (*(temp_dmrs_symbols) - pdsch_start_symbol_)) or
                (symbol == (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_))) {
                temp_coef_grid += fft_size_;
                continue;
            }
            /// If current symbol is greater than the upper DMRS symbol,
            /// update lower and upper DMRS coefs
            if ((symbol > (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_)) and (symbol < last_dmrs_symbol)) {
                temp_dmrs_symbols++;
                lower_dmrs_coef = upper_dmrs_coef;
                upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
                step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
            }

            step = (symbol - *(temp_dmrs_symbols) + pdsch_start_symbol_); // / step_symbols;

            for (int sc = 0; sc < fft_size_ / 4 - 1; sc++) {
                /// Interpolate on the first subcarrier
                (temp_coef_grid)->real(lower_dmrs_coef->real() +
                                       (upper_dmrs_coef->real() - lower_dmrs_coef->real()) *
                                       step / step_symbols);
                (temp_coef_grid)->imag(lower_dmrs_coef->imag() +
                                       (upper_dmrs_coef->imag() - lower_dmrs_coef->imag()) *
                                       step / step_symbols);
                /// Interpolate norm and renormalize
                *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) * step +
                                      abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));
                /// Assign the same value to the 2 next subcarriers
                *(temp_coef_grid + 1) = *(temp_coef_grid);
                *(temp_coef_grid + 2) = *(temp_coef_grid);
                /// Interpolate on subcarrier number 3
                /// Interpolate on the first subcarrier
                (temp_coef_grid + 3)->real((lower_dmrs_coef + 3)->real() +
                                           ((upper_dmrs_coef + 3)->real() - (lower_dmrs_coef + 3)->real()) *
                                           step / step_symbols);
                (temp_coef_grid + 3)->imag((lower_dmrs_coef + 3)->imag() +
                                           ((upper_dmrs_coef + 3)->imag() - (lower_dmrs_coef + 3)->imag()) *
                                           step / step_symbols);
                /// Interpolate norm and renormalize
                *(temp_coef_grid + 3) *= ((abs(*(upper_dmrs_coef + 3)) - abs(*(lower_dmrs_coef + 3))) *
                                          step / step_symbols +
                                          abs(*(lower_dmrs_coef + 3))) / abs(*(temp_coef_grid + 3));
                temp_coef_grid += 4;
                lower_dmrs_coef += 4;
                upper_dmrs_coef += 4;
            }

            /// Interpolate on first subcarrier of the 4 last subcarriers
            /// Interpolate on the first subcarrier
            (temp_coef_grid)->real(lower_dmrs_coef->real() +
                                   (upper_dmrs_coef->real() - lower_dmrs_coef->real()) *
                                   step / step_symbols);
            (temp_coef_grid)->imag(lower_dmrs_coef->imag() +
                                   (upper_dmrs_coef->imag() - lower_dmrs_coef->imag()) *
                                   step / step_symbols);
            /// Interpolate norm and renormalize
            *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) * step / step_symbols +
                                  abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));
            /// Assign the value to the last 4 subcarriers
            *(temp_coef_grid + 1) = *(temp_coef_grid);
            *(temp_coef_grid + 2) = *(temp_coef_grid);
            *(temp_coef_grid + 3) = *(temp_coef_grid);
            temp_coef_grid += 4; /// Jump to the first subcarrier of the next sytmbol
            /// Reinitialize the pointers to the two DMRS symbols used for interpolation
            lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
            upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
        }
    }
}

void inline interp_cdm_group1_2(std::complex<float> * coef_grid, /// TODO: mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                         std::complex<float> * dmrs_coefs_, // TODO : mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                         int * dmrs_symbols_,
                         int pdsch_start_symbol_,
                         int dmrs_symbols_per_grid_,
                         int dmrs_sequence_size_,
                         int fft_size_,
                         int symbols_per_grid_) {
    std::complex<float> * temp_dmrs_coefs;
    std::complex<float> * temp_coef_grid;
    std::complex<float> * lower_dmrs_coef;
    std::complex<float> * upper_dmrs_coef;
    int step_symbols;

    temp_dmrs_coefs = dmrs_coefs_;
    step_symbols = 0;

    /// Optimizations if 2 DMRS ports are mutliplexed by OCC in the CDM group
    /// Interpolation of DMRS from CDM group 1
    /// Fill the grid with DMRS
    for (int symbol = 0; symbol < dmrs_symbols_per_grid_; symbol++) {
        temp_coef_grid = coef_grid + (dmrs_symbols_[symbol] - pdsch_start_symbol_) *
                                     fft_size_; /// Initialize to the first subcarrier of the current symbol
        temp_dmrs_coefs = dmrs_coefs_ + symbol * dmrs_sequence_size_;
        for (int sc = 0; sc < dmrs_sequence_size_; sc++) {
            temp_coef_grid[2 * sc + 1] = temp_dmrs_coefs[sc];
        }
    }
    /// interpolate in frequency domain
    for (int *temp_dmrs_symbols = dmrs_symbols_;
         temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {
        temp_coef_grid = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                     fft_size_; /// Initialize to the first RE to be interpolated (on sc number 0)
        /// Assign the same value for the first DMRS subcarriers
        *(temp_coef_grid) = *(temp_coef_grid + 1);
        *(temp_coef_grid + 2) = *(temp_coef_grid + 1);
        temp_coef_grid += 4; /// Jump to the next sc to be interpolated
        for (int sc = 1; sc < fft_size_/4; sc++) {
            /// Assign the same value to the subcarrier in-between the two DMRS subcarriers
            *(temp_coef_grid + 2) = *(temp_coef_grid + 1);
            /// Interpolate the subcarrier below the first DMRS subcarrier/in-between the two OCC
            (temp_coef_grid)->real((real(*(temp_coef_grid + 1)) + real(*(temp_coef_grid - 1))) * 0.5);
            (temp_coef_grid)->imag((imag(*(temp_coef_grid + 1)) + imag(*(temp_coef_grid - 1))) * 0.5);
            /// Interpolate norm and renormalize
            *(temp_coef_grid) *=
                    ((abs(*(temp_coef_grid + 1)) + abs(*(temp_coef_grid - 1))) * 0.5) / abs(*(temp_coef_grid));
            temp_coef_grid += 4; /// Jump to the next sc to be interpolated
        }
    }
    int last_dmrs_symbol = *(dmrs_symbols_ + dmrs_symbols_per_grid_ - 1);
    int *temp_dmrs_symbols = dmrs_symbols_;

    /// interpolate in time domain
    /// Only one DMRS in the PDSCH
    if (dmrs_symbols_per_grid_ == 1) {
        /// Reset pointer
        lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
        temp_coef_grid = coef_grid;
        /// Apply the same coefficients in time domain on each subcarrier
        for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {
            /// Reset pointer to DMRS channel coefficients
            temp_dmrs_coefs = dmrs_coefs_;
            /// No need to apply values on DRMS symbol as we have interpolated in frequency domain
            if (symbol == *(dmrs_symbols_) - pdsch_start_symbol_) {
                temp_coef_grid += fft_size_;
                continue;
            }
            for (int j = 0; j < fft_size_; j++) {
                *(temp_coef_grid) = *(lower_dmrs_coef);
                temp_coef_grid++;
                lower_dmrs_coef++;
            }
        }
    } else {
        /// Reset pointers
        temp_coef_grid = coef_grid;
        step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
        int step = 0;
        lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
        upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
        for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {
            /// If current symbol is a DMRS symbol, do not interpolate on this symbol
            if ((symbol == (*(temp_dmrs_symbols) - pdsch_start_symbol_)) or
                (symbol == (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_))) {
                temp_coef_grid += fft_size_;
                continue;
            }
            /// If current symbol is greater than the upper DMRS symbol,
            /// update lower and upper DMRS coefs
            if ((symbol > (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_)) and (symbol < last_dmrs_symbol)) {
                temp_dmrs_symbols++;
                lower_dmrs_coef = upper_dmrs_coef;
                upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
                step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
            }
            step = (symbol - *(temp_dmrs_symbols) + pdsch_start_symbol_); // / step_symbols;
            /// Interpolate on the first subcarrier
            (temp_coef_grid)->real(lower_dmrs_coef->real() +
                                   (upper_dmrs_coef->real() - lower_dmrs_coef->real()) *
                                   step / step_symbols);
            (temp_coef_grid)->imag(lower_dmrs_coef->imag() +
                                   (upper_dmrs_coef->imag() - lower_dmrs_coef->imag()) *
                                   step / step_symbols);
            /// Interpolate norm and renormalize
            *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                  step / step_symbols +
                                  abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));
            /// Assign the same value to the 3 next subcarriers
            *(temp_coef_grid + 1) = *(temp_coef_grid);
            *(temp_coef_grid + 2) = *(temp_coef_grid);
            *(temp_coef_grid + 3) = *(temp_coef_grid);
            temp_coef_grid  += 4; /// Jump to the next sc to be interpolated
            lower_dmrs_coef += 4;
            upper_dmrs_coef += 4;
            for (int sc = 1; sc < fft_size_/4; sc++) {
                /// Interpolate on the first subcarrier
                (temp_coef_grid)->real(lower_dmrs_coef->real() +
                                       (upper_dmrs_coef->real() - lower_dmrs_coef->real()) *
                                       step / step_symbols);
                (temp_coef_grid)->imag(lower_dmrs_coef->imag() +
                                       (upper_dmrs_coef->imag() - lower_dmrs_coef->imag()) *
                                       step / step_symbols);
                /// Interpolate norm and renormalize
                *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                      step / step_symbols +
                                      abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));

                /// Interpolate on subcarrier number 3
                (temp_coef_grid + 3)->real((lower_dmrs_coef + 3)->real() +
                                           ((upper_dmrs_coef + 3)->real() - (lower_dmrs_coef + 3)->real()) *
                                           step / step_symbols);
                (temp_coef_grid + 3)->imag((lower_dmrs_coef + 3)->imag() +
                                           ((upper_dmrs_coef + 3)->imag() - (lower_dmrs_coef + 3)->imag()) *
                                           step / step_symbols);
                /// Interpolate norm and renormalize
                *(temp_coef_grid + 3) *= ((abs(*(upper_dmrs_coef + 3)) - abs(*(lower_dmrs_coef + 3))) *
                                          step / step_symbols +
                                          abs(*(lower_dmrs_coef + 3))) / abs(*(temp_coef_grid + 3));

                /// Assign the same value to the subcarriers 1 and 2
                *(temp_coef_grid + 1) = *(temp_coef_grid + 3);
                *(temp_coef_grid + 2) = *(temp_coef_grid + 3);
                temp_coef_grid  += 4;
                lower_dmrs_coef += 4;
                upper_dmrs_coef += 4;
            }

            /// Reinitialize the pointers to the two DMRS symbols used for interpolation
            lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
            upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
        }
    }
}

void inline interp_cdm_group0_1(std::complex<float> * coef_grid, /// TODO: mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                         std::complex<float> * dmrs_coefs_, // TODO : mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                         int * dmrs_symbols_,
                         int pdsch_start_symbol_,
                         int dmrs_symbols_per_grid_,
                         int dmrs_sequence_size_,
                         int fft_size_,
                         int symbols_per_grid_) {

    std::complex<float> * temp_dmrs_coefs;
    std::complex<float> * temp_coef_grid;
    std::complex<float> * lower_dmrs_coef;
    std::complex<float> * upper_dmrs_coef;
    int step_symbols;

    temp_dmrs_coefs = dmrs_coefs_;
    step_symbols = 0;

    /// Interpolation of DMRS from CDM group 0
    /// Fill the grid with DMRS
    for (int symbol = 0; symbol < dmrs_symbols_per_grid_; symbol++) {
        temp_coef_grid = coef_grid + (dmrs_symbols_[symbol] - pdsch_start_symbol_) *
                                     fft_size_; /// Initialize to the first subcarrier of the current symbol
        temp_dmrs_coefs = dmrs_coefs_ + symbol * dmrs_sequence_size_;
        for(int sc = 0; sc < dmrs_sequence_size_; sc++) {
            temp_coef_grid[2*sc] = temp_dmrs_coefs[sc];
        }
    }
    /// interpolate in frequency domain
    for (int *temp_dmrs_symbols = dmrs_symbols_;
         temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {
        temp_coef_grid = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                     fft_size_; /// Initialize to the first RE of the current symbol
        temp_dmrs_coefs = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                      fft_size_; /// Initialize to the first RE containing DMRS of the current symbol
        /// Interpolate every odd subcarrier
        for (int sc = 0; sc < fft_size_; sc+=2) {
            /// Interpolate real and imaginary part
            temp_coef_grid[sc].real((temp_dmrs_coefs[sc + 1].real() + temp_dmrs_coefs[sc - 1].real()) * 0.5);
            temp_coef_grid[sc].imag((temp_dmrs_coefs[sc + 1].imag() + temp_dmrs_coefs[sc - 1].imag()) * 0.5);
            /// Interpolate norm and renormalize
            temp_coef_grid[sc] *=
                    ((abs(temp_dmrs_coefs[sc + 1]) + abs(temp_dmrs_coefs[sc - 1])) * 0.5) / abs(temp_coef_grid[sc]);
        }
        /// Extrapolate the last value
        temp_coef_grid[fft_size_ - 1].real((temp_dmrs_coefs[fft_size_ - 2].real() * 3 - temp_dmrs_coefs[fft_size_ - 4].real()) * 0.5);
        temp_coef_grid[fft_size_ - 1].imag((temp_dmrs_coefs[fft_size_ - 2].imag() * 3 - temp_dmrs_coefs[fft_size_ - 4].imag()) * 0.5);
        /// Interpolate norm and renormalize
        temp_coef_grid[fft_size_ - 1] *= (abs(temp_dmrs_coefs[fft_size_ - 2]) * 3 * 0.5 - abs(temp_dmrs_coefs[fft_size_ - 4]) * 0.5) /
                                         abs(temp_coef_grid[fft_size_ - 1]);
    }
    int last_dmrs_symbol = *(dmrs_symbols_ + dmrs_symbols_per_grid_ - 1);
    int * temp_dmrs_symbols = dmrs_symbols_;
    /// interpolate in time domain
    /// Only one DMRS in the PDSCH
    if (dmrs_symbols_per_grid_ == 1) {
        /// Reset pointer
        lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
        /// Apply the same coefficients in time domain on each subcarrier
        for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {
            /// Reset pointer to DMRS channel coefficients
            temp_dmrs_coefs = dmrs_coefs_;
            /// No need to apply values on DRMS symbol as we have interpolated in frequency domain
            if (symbol == *(dmrs_symbols_) - pdsch_start_symbol_) {
                temp_coef_grid += fft_size_;
                continue;
            }
            for (int j = 0; j < fft_size_; j++) {
                *(temp_coef_grid) = *(lower_dmrs_coef);
                temp_coef_grid++;
                lower_dmrs_coef++;
            }
        }
    } else {
        /// Reset pointers
        temp_coef_grid = coef_grid;
        temp_dmrs_symbols = dmrs_symbols_;
        step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
        float step = 0;

        lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
        upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;

        for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {
            /// If current symbol is a DMRS symbol, do not interpolate on this symbol
            if ((symbol == (*(temp_dmrs_symbols) - pdsch_start_symbol_)) or
                (symbol == (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_))) {
                temp_coef_grid += fft_size_;
                continue;
            }

            /// If current symbol is greater than the upper DMRS symbol,
            /// update lower and upper DMRS coefs
            if ((symbol > (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_)) and (symbol < last_dmrs_symbol)) {
                temp_dmrs_symbols++;
                lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
                upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
                step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
            }

            step = (symbol - (*temp_dmrs_symbols) + pdsch_start_symbol_) / step_symbols;

            for (int sc = 0; sc < fft_size_; sc++) {

                (temp_coef_grid)->real(lower_dmrs_coef->real() +
                                       (upper_dmrs_coef->real() - lower_dmrs_coef->real()) *
                                       step);
                (temp_coef_grid)->imag(lower_dmrs_coef->imag() +
                                       (upper_dmrs_coef->imag() - lower_dmrs_coef->imag()) *
                                       step);

                /// Interpolate norm and renormalize
                *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                      step +
                                      abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));

                //output << *(temp_coef_grid) << endl;

                temp_coef_grid++;
                lower_dmrs_coef++;
                upper_dmrs_coef++;
            }

            /// Reinitialize the pointers to the two DMRS symbols used for interpolation
            //lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
            //upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
            lower_dmrs_coef -= fft_size_;
            upper_dmrs_coef -= fft_size_;
        }
    }
}

void inline interp_cdm_group1_1(std::complex<float> * coef_grid, /// add offset to start at first element of the grid to compute
                         std::complex<float> * dmrs_coefs_, /// add offset to start at first element of the grid to compute
                         int * dmrs_symbols_,
                         int pdsch_start_symbol_,
                         int dmrs_symbols_per_grid_,
                         int dmrs_sequence_size_,
                         int fft_size_,
                         int symbols_per_grid_) {
    std::complex<float> * temp_dmrs_coefs;
    std::complex<float> * temp_coef_grid;
    std::complex<float> * lower_dmrs_coef;
    std::complex<float> * upper_dmrs_coef;
    int step_symbols;
    temp_dmrs_coefs = dmrs_coefs_;
    temp_coef_grid = coef_grid;
    step_symbols = 0;

    /// CDM group 1
    /// Fill the grid with DMRS
    for (int symbol = 0; symbol < dmrs_symbols_per_grid_; symbol++) {
        temp_coef_grid = coef_grid + (dmrs_symbols_[symbol] - pdsch_start_symbol_) *
                                     fft_size_; /// Initialize to the first subcarrier of the current symbol
        temp_dmrs_coefs = dmrs_coefs_ + symbol * dmrs_sequence_size_;
        for(int sc = 0; sc < dmrs_sequence_size_; sc++) {
            temp_coef_grid[2*sc + 1] = temp_dmrs_coefs[sc];
        }
    }

    /// interpolate in frequency domain
    for (int *temp_dmrs_symbols = dmrs_symbols_;
         temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {
        temp_coef_grid = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                     fft_size_; /// Initialize to the first RE to be interpolated (on sc number 0)
        temp_dmrs_coefs = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                      fft_size_; /// Initialize to the first RE containing DMRS of the current symbol
        /// Extrapolate first value
        temp_coef_grid[0].real((temp_dmrs_coefs[1].real() * 3 - temp_dmrs_coefs[3].real()) * 0.5);
        temp_coef_grid[0].imag((temp_dmrs_coefs[1].imag() * 3 - temp_dmrs_coefs[3].imag()) * 0.5);
        /// Interpolate norm and renormalize
        temp_coef_grid[0] *= (abs(temp_dmrs_coefs[1]) * 3 * 0.5 - abs(temp_dmrs_coefs[3]) * 0.5) / abs(temp_coef_grid[0]);
        //output << *(temp_coef_grid) << endl;
        /// Add first DMRS RE to the grid
        //*(temp_coef_grid + 1) = *(temp_dmrs_coefs);
        //output << *(temp_coef_grid + 1) << endl;
        //temp_coef_grid += 2; /// Jump to the next sc to be interpolated
        for (int sc = 2; sc < fft_size_; sc+=2) {
            temp_coef_grid[sc].real((temp_dmrs_coefs[sc + 1].real() + temp_dmrs_coefs[sc - 1].real()) * 0.5);
            temp_coef_grid[sc].imag((temp_coef_grid[sc + 1].imag() + temp_dmrs_coefs[sc - 1].imag()) * 0.5);
            /// Interpolate norm and renormalize
            temp_coef_grid[sc] *=
                    ((abs(temp_dmrs_coefs[sc + 1]) + abs(temp_dmrs_coefs[sc - 1])) * 0.5) /
                    abs(temp_dmrs_coefs[sc]);
            //output << *(temp_coef_grid) << endl;
            //output << *(temp_coef_grid + 1) << endl;
            //temp_coef_grid += 2; /// Jump to the next sc to be interpolated
            //temp_dmrs_coefs++;
        }
        //temp_coef_grid++;
        //temp_dmrs_coefs++;
    }
    int last_dmrs_symbol = *(dmrs_symbols_ + dmrs_symbols_per_grid_ - 1);
    int * temp_dmrs_symbols = dmrs_symbols_;

    /// interpolate in time domain
    /// Only one DMRS in the PDSCH
    if (dmrs_symbols_per_grid_ == 1) {
        /// Reset pointer
        lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
        /// Apply the same coefficients in time domain on each subcarrier
        for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {
            /// Reset pointer to DMRS channel coefficients
            temp_dmrs_coefs = dmrs_coefs_;
            /// No need to apply values on DRMS symbol as we have interpolated in frequency domain
            if (symbol == *(dmrs_symbols_) - pdsch_start_symbol_) {
                temp_coef_grid += fft_size_;
                continue;
            }
            for (int j = 0; j < fft_size_; j++) {
                *(temp_coef_grid) = *(lower_dmrs_coef);
                temp_coef_grid++;
                lower_dmrs_coef++;
            }
        }
    } else {
        /// Reset pointers
        temp_coef_grid = coef_grid;
        temp_dmrs_symbols = dmrs_symbols_;

        step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
        float step = 0;

        lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_ + 1;
        upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_ + 1;

        for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {
            /// If current symbol is a DMRS symbol, do not interpolate on this symbol
            if ((symbol == (*(temp_dmrs_symbols) - pdsch_start_symbol_)) or
                (symbol == (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_))) {
                temp_coef_grid += fft_size_;
                continue;
            }

            /// If current symbol is greater than the upper DMRS symbol,
            /// update lower and upper DMRS coefs
            if ((symbol > (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_)) and (symbol < last_dmrs_symbol)) {
                temp_dmrs_symbols++;
                //lower_dmrs_coef = upper_dmrs_coef;
                lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_ + 1;
                upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_ + 1;
                step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
                //upper_dmrs_coef += step_symbols * fft_size_;
            }

            step = 1.0f * (symbol - *(temp_dmrs_symbols) + pdsch_start_symbol_) / step_symbols;

            /// Interpolate on the first subcarrier
            (temp_coef_grid)->real(lower_dmrs_coef->real() +
                                   (upper_dmrs_coef->real() - lower_dmrs_coef->real()) *
                                   step);
            (temp_coef_grid)->imag(lower_dmrs_coef->imag() +
                                   (upper_dmrs_coef->imag() - lower_dmrs_coef->imag()) *
                                   step);

            /// Interpolate norm and renormalize
            *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                  step +
                                  abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));

            /// Assign the same value to the 3 next subcarriers
            *(temp_coef_grid + 1) = *(temp_coef_grid);
            *(temp_coef_grid + 2) = *(temp_coef_grid);
            *(temp_coef_grid + 3) = *(temp_coef_grid);

            temp_coef_grid  += 4; /// Jump to the next sc to be interpolated
            lower_dmrs_coef += 4;
            upper_dmrs_coef += 4;

            for (int sc = 1; sc < fft_size_/4; sc ++) {

                /// Interpolate on the first subcarrier
                (temp_coef_grid)->real(lower_dmrs_coef->real() +
                                       (upper_dmrs_coef->real() - lower_dmrs_coef->real()) *
                                       step);
                (temp_coef_grid)->imag(lower_dmrs_coef->imag() +
                                       (upper_dmrs_coef->imag() - lower_dmrs_coef->imag()) *
                                       step);

                //*(temp_coef_grid) = *(lower_dmrs_coef) + (*(upper_dmrs_coef) - *(lower_dmrs_coef)) * step;

                /// Interpolate norm and renormalize
                *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                      step +
                                      abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));

                /// Interpolate on subcarrier number 3
                (temp_coef_grid + 3)->real((lower_dmrs_coef + 3)->real() +
                                           ((upper_dmrs_coef + 3)->real() - (lower_dmrs_coef + 3)->real()) *
                                           step);
                (temp_coef_grid + 3)->imag((lower_dmrs_coef + 3)->imag() +
                                           ((upper_dmrs_coef + 3)->imag() - (lower_dmrs_coef + 3)->imag()) *
                                           step);

                //*(temp_coef_grid + 3) = *(lower_dmrs_coef + 3) + (*(upper_dmrs_coef + 3) - *(lower_dmrs_coef)) * step;

                /// Interpolate norm and renormalize
                *(temp_coef_grid + 3) *= ((abs(*(upper_dmrs_coef + 3)) - abs(*(lower_dmrs_coef + 3))) *
                                          step +
                                          abs(*(lower_dmrs_coef + 3))) / abs(*(temp_coef_grid + 3));

                /// Assign the same value to the subcarriers 1 and 2
                *(temp_coef_grid + 1) = *(temp_coef_grid + 3);
                *(temp_coef_grid + 2) = *(temp_coef_grid + 3);

                //output << *(temp_coef_grid) << endl;

                temp_coef_grid  += 4;
                lower_dmrs_coef += 4;
                upper_dmrs_coef += 4;
                //cout << sc << endl;
            }

            /// Reinitialize the pointers to the two DMRS symbols used for interpolation
            lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_ + 1;
            upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_ + 1;
        }
    }
}

/***************************************************************************************************************/

#endif // CHANNEL_ESTIMATION_H
