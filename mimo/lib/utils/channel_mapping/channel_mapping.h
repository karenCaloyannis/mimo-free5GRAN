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

    Functions reused from free5GRAN and modified :
    - compute_dmrs_positions
    - compute_dmrs_positions_type1A
    - compute_dmrs_sequences_type1
    - compute_dmrs_sequences
    - compute_pdsch_positions
*/

#ifndef USRP_MIMO_CHANNEL_MAPPING_H
#define USRP_MIMO_CHANNEL_MAPPING_H

#include <string>
#include <complex>
#include <utility>
#include <vector>
#include <iostream>
#include <complex>
#include <chrono>
#include <x86intrin.h>
#include <immintrin.h>
#include "../../free5gran_utils.h"
#include "../../variables/variables.h"

/** @brief Computes the DMRS positions for each antenna port according to their CDM group
     *  and stores them in the positions_tx array.
     *
     *  This function only computes DMRS positions for configuration type 1, PDSCH mapping type A,
     *  signel symbol DMRS.
     *
     * @param l0_                      : OFDM symbol index of the first DMRS sequence
     * @param duration_                : duration of the PDSCH
     * @param dmrsAdditionalPositions_ : number of additionnal DMRS positions
     * @param antenna_ports_           : vector containing antenna port numbers minues '1000'
     * @param fft_size_                : fft size
     * @param nb_slots_                : number of slots in the frame
     */
void compute_dmrs_positions(int dmrsAdditionalPositions_,
                            char pdsch_mapping_type_,
                            int dmrs_config_type_,
                            int &l0_,
                            int &pdsch_start_,
                            int &pdsch_length_,
                            int &num_dmrs_symbols_per_slot_,
                            int n_rb_,
                            int &dmrs_sequence_size_,
                            bool &double_symbol,
                            int *&dmrs_symbols);

/************************ TEST réalignement ************************************************/
void get_pdsch_and_dmrs_samples(const std::vector<std::vector<std::complex<float>>> &synchronized_grid_,
                                int slot_number_,
                                const int * dmrs_symbols_,
                                int dmrs_sequence_size_,
                                int num_dmrs_symbols_per_slot_,
                                int num_cdm_groups_without_data,
                                int pdsch_length_,
                                int fft_size_,
                                int pdsch_start_symbol_,
                                int rx_port_index,
                                std::complex<float> pdsch_samples_[][MAX_RX_PORTS],
                                std::vector<std::complex<float>> * dmrs_samples_);
/******************************************************************************************/

void get_pdsch_and_dmrs_samples(const std::vector<std::vector<std::complex<float>>> &synchronized_grid_,
                                const int &slot_number_,
                                const int * dmrs_symbols_,
                                const int &dmrs_sequence_size_,
                                const int &num_dmrs_symbols_per_slot_,
                                const int &num_cdm_groups_without_data,
                                int dmrs_conf_type, /// type 1 or type 2
                                const int &pdsch_length_,
                                const int &n_rb, /// number of RB
                                const int &sc_rb_start_, /// subcarrier index of the first RB
                                const int &pdsch_start_symbol_,
                                std::vector<std::complex<float>> &pdsch_samples_,
                                std::vector<std::complex<float>> * dmrs_samples_);

void get_pdsch_and_dmrs_samples(const std::vector<std::vector<std::complex<float>>> &synchronized_grid_,
                                std::vector<std::complex<float>> &pdsch_samples_,
                                std::vector<std::complex<float>> dmrs_samples_[MAX_NUM_CDM_GROUPS],
                                int slot_number_,
                                const int *dmrs_symbols_,
                                int num_dmrs_symbols_per_slot_,
                                int num_cdm_groups_without_data,
                                int pdsch_length_,
                                int fft_size_,
                                int sc_rb_start_,
                                int pdsch_start_symbol_,
                                int dmrs_config_type_,
                                int *cdm_groups_,
                                int num_used_cdm_groups_);

/** Added extraction of squared norms of channel coefficients on PDSCH REs only.
 *
 */
void get_pdsch_channel_coefficients(std::complex<float> * coef_grid_,
                                    std::complex<float> * pdsch_channel_coefficients_,
                                    float * squared_norms_,
                                    float * pdsch_squared_norms_,
                                    int pdsch_start_,
                                    int * dmrs_symbols_,
                                    int fft_size_,
                                    int pdsch_length_,
                                    int dmrs_config_type_,
                                    int num_cdm_groups_without_data_);

void get_pdsch_and_dmrs_samples(const std::vector<std::vector<std::complex<float>>> &synchronized_grid_,
                                       int slot_number_,
                                       const int * dmrs_symbols_,
                                       int dmrs_sequence_size_,
                                       int num_dmrs_symbols_per_slot_,
                                       int num_cdm_groups_without_data,
                                       int pdsch_length_,
                                       int fft_size_,
                                       int pdsch_start_symbol_,
                                       std::vector<std::complex<float>> &pdsch_samples_,
                                       std::vector<std::complex<float>> * dmrs_samples_);

void get_pdsch_and_dmrs_samples(const std::vector<std::vector<std::complex<float>>> &received_grid_,
                                const int * dmrs_symbols_,
                                const int &dmrs_sequence_size_,
                                const int &num_dmrs_symbols_per_slot_,
                                const int &num_cdm_groups_without_data,
                                const int &pdsch_length_,
                                const int &fft_size_,
                                std::complex<float> * pdsch_samples_,
                                std::complex<float> * dmrs_samples_);

/** @brief From free5GRAN (libphy.cpp, get_pdsch_dmrs_symbols).
 *  Computes the DMRS symbols and subcarriers for dmrs config type 1, pdsch mapping type A
 *  **within only one slot**
 *
 *  Computes the positions for all possible DMRS ports (currently for 4 DMRS ports
 *  1000, 1001, 1002, 1003)
 *
 *  DMRS double symbol is not currently supported
 *
 *  Corrects the l0, PDSCH start, PDSCH length if necessary
 *
 * @param[in, out] dmrsAdditionalPositions_ : DMRS additionnal positions RRC parameters
 * @param[in, out] l0_ : Position of the first DMRS symbol in the slot
 * @param[in, out] pdsch_start_ : OFDM start symbol of the PDSCH in the slot
 * @param[in, out] pdsch_length_ : Length of the PDSCH in symbols in the slot
 * @param[in, out] num_dmrs_symbols_per_slot_ : Number of DMRS OFDM symbols in one slot
 * @param[in] num_tx_dmrs_ports_ : Number of TX ports
 * @param[in] fft_size_ : FFT size of the PDSCH allocation
 * @param[in] nb_slots_ : Number of slots to be computed in the whole frame starting from slot 0
 * @param[in] dmrs_sequence_size_ : Size of the DMRS sequence
 * @param[in] double_symbol : True if double symbol DMRS is used, False otherwise.
 *                        DOES NOT WORK CURRENTLY
 * @param[out]  dmrs_symbols : array containing the DMRS OFDM symbol indexes in the slot
 * @param[out]  dmrs_subcarriers : Array containing the DMRS subcarriers
 * @param[out] dmrs_grid_positions : Array indicated for each RE in the grid whether it is occupied by a DMRS or not
 *                              DMRS REs are marked by 1, others to 0.
 */
void compute_dmrs_positions_type1A(int &dmrsAdditionalPositions_,
                                   int &l0_,
                                   int &pdsch_start_,
                                   int &pdsch_length_,
                                   int &num_dmrs_symbols_per_slot_,
                                   const int &num_tx_dmrs_ports_,
                                   int fft_size_,
                                   int nb_slots_,
                                   int &dmrs_sequence_size_,
                                   bool double_symbol,
                                   int * &dmrs_symbols,
                                   int * &dmrs_subcarriers,
                                   int * dmrs_grid_positions);

/** Computes the DMRS sequences for each antenna port given in argument, and encodes it with the OCC
 *  According to TS 38.211 section 7.4.1.1.2
 *
 *  implements only DMRS config. type 1
 *
 * @param[in] dmrs_symbols_ : Array containing the pre-computed DMRS symbol indexes
 * @param[out] dmrs_sequences_ : Array containing DMRS sequences for each DMRS port,
 *                               each slot and each DMRS symbol in the frame
 * @param[in] num_dmrs_per_slot_ : Number of DMRS OFDM symbol in a slot
 * @param[in] dmrs_sequence_size_ : Size of the DMRS sequence in frequency
 * @param[in] nb_slots_ : Number of slots to be computed starting from 0
 * @param[in] double_symbol_ : True for double symbol DMRS, false otherwise
 */
void compute_dmrs_sequences_type1(int * dmrs_symbols_,
                                  std::complex<float> * dmrs_sequences_,
                                  const int &num_dmrs_per_slot_,
                                  const int &dmrs_sequence_size_,
                                  const int &nb_slots_,
                                  const bool &double_symbol_);

void compute_dmrs_sequences(int * dmrs_symbols_,
                            std::vector<std::complex<float>> * dmrs_sequences_,
                            int num_dmrs_per_slot_,
                            int dmrs_sequence_size_,
                            int nb_slots_,
                            int double_symbol_,
                            int dmrs_config_type);

void compute_dmrs_sequences(int * dmrs_symbols_,
                            std::vector<float> * dmrs_sequences_real,
                            std::vector<float> * dmrs_sequences_imag,
                            int num_dmrs_per_slot_,
                            int dmrs_sequence_size_,
                            int nb_slots_,
                            int double_symbol_,
                            int dmrs_config_type);

void compute_pdsch_positions(const int &pdsch_start_,
                             const int &pdsch_length_,
                             int * dmrs_symbols_,
                             int * pdsch_symbols_,
                             int * pdsch_sc_,
                             int dmrs_conf_type_,
                             int num_cdm_groups_without_data,
                             int n_rb); /// number of RBs

/** Computes tje PDSCH positions within one slot, i.e.
 *  the indexes of REs not transmitting DMRS for every PDSCH
 *  slot
*
* @param[in] pdsch_start_ : First OFDM symbol of the PDSCH
* @param[in] pdsch_length_ : Length of the PDSCH in OFDM symbols
* @param[in] fft_size_ : FFT Size
* @param[in] dmrs_grid_positions_ : Reference grid containing indications on DMRS positions
* @param[out] pdsch_positions : output array containing PDSCH indexes
*/
void compute_pdsch_positions(const int &pdsch_start_,
                             const int &pdsch_length_,
                             const int &fft_size_,
                             int * dmrs_grid_positions_,
                             int * pdsch_positions);

void compute_cum_sum_samples(int *dmrs_symbols_,
                             int *cum_sum_samples, /// Number of PDSCH samples within each OFDM symbol
                             int num_cdm_groups_without_data,
                             int fft_size_,
                             int pdsch_length_,
                             int num_dmrs_symbols_,
                             int pdsch_start);
#endif //USRP_MIMO_CHANNEL_MAPPING_H
