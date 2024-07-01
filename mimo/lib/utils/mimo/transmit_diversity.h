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
 */

#ifndef ALAMOUTI_H
#define ALAMOUTI_H

#include <string>
#include <complex>
#include <vector>
#include <iostream>
#include <complex>
#include <utility>
#include "../../free5gran_utils.h"
#include "../../variables/variables.h"

/** Performs Alamouti SFBC decoding
 *
 * @param[in] pdsch_samples_ : received PDSCH samples on each RX port
 * @param[in] channel_coefficients_ : channel coefficients on PDSCH RE only, extracted from the interpolated coefficients grid
 * @param[in] num_re_pdsch_ : Number of REs to be computed
 * @param[in, out] equalized_symbols_ : final buffer of equalized symbols
 * @param[in] nb_tx_dmrs_ports_ : Number of TX DMRS ports
 * @param[in] nb_rx_ports_ : Number of RX ports
 */
void mimo_transmit_diversity_decoder(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                                     const std::vector<std::complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                                     int num_re_pdsch_,
                                     std::complex<float> *equalized_symbols_,
                                     int nb_tx_dmrs_ports_,
                                     int nb_rx_ports_);

void mimo_transmit_diversity_decoder(const std::vector<std::vector<std::complex<float >>> * received_grids_,
                                     const std::complex<float> * channel_coefficients_,
                                     const int &slot_number_,
                                     const int * pdsch_positions_,
                                     const int &num_re_pdsch_,
                                     std::complex<float> * equalized_symbols_,
                                     int &nb_tx_dmrs_ports_,
                                     int &nb_rx_ports_,
                                     const int &pdsch_start_symbol_,
                                     const int &pdsch_length_,
                                     const int &fft_size_);

void mimo_transmit_diversity_decoder(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                                     const std::complex<float> * channel_coefficients_,
                                     //const int &slot_number_,
                                     //const int * pdsch_positions_,
                                     const int &num_re_pdsch_,
                                     //const int &pdsch_length_,
                                     //const int fft_size_,
                                     std::complex<float> * equalized_symbols_,
                                     int &nb_tx_dmrs_ports_,
                                     int &nb_rx_ports_);//,
                                     //const int &pdsch_start_symbol_);

void mimo_transmit_diversity_decoder_v2(const std::vector<std::vector<std::complex<float >>> * received_grids_,
                                        const std::vector<std::vector<std::complex<float >>> * channel_coefficients_,
                                        const int &slot_number_,
                                        const int * pdsch_positions_,
                                        const int &num_re_pdsch_,
                                        std::complex<float> * equalized_symbols_,
                                        int &nb_tx_dmrs_ports_,
                                        int &nb_rx_ports_,
                                        const int &pdsch_start_symbol_);

void mimo_transmit_diversity_decoder_v2(const std::vector<std::vector<std::complex<float >>> * received_grids_,
                                        const std::complex<float> * channel_coefficients_,
                                        const int &slot_number_,
                                        const int * pdsch_positions_,
                                        const int &num_re_pdsch_,
                                        const int &pdsch_length_,
                                        const int fft_size_,
                                        std::complex<float> * equalized_symbols_,
                                        int &nb_tx_dmrs_ports_,
                                        int &nb_rx_ports_,
                                        const int &pdsch_start_symbol_);

void mimo_transmit_diversity_decoder_v2(const std::vector<std::vector<std::complex<float>>> &pdsch_samples_,
                                        const std::complex<float> * channel_coefficients_,
                                        const int &slot_number_,
                                        const int * pdsch_positions_,
                                        const int &num_re_pdsch_,
                                        const int &pdsch_length_,
                                        const int fft_size_,
                                        std::complex<float> * equalized_symbols_,
                                        int &nb_tx_dmrs_ports_,
                                        int &nb_rx_ports_,
                                        const int &pdsch_start_symbol_);

#endif //ALAMOUTI_H
