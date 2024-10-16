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

#ifndef USRP_MIMO_MIMO_H
#define USRP_MIMO_MIMO_H

#include <string>
#include <complex>
#include <vector>
#include <iostream>
#include <complex>
#include "../../free5gran_utils.h"
#include "../../variables/variables.h"

/**
 *  Contains Layer mapping and Precoding functions for spatial multiplexing
 *  and diversity. Functions are not optimized.
 *
 */

/** From TS 36.211 Section 6.3.3.3
 *  Used for layer mapping.
 *  Because of the buffer alignments, this function is useless and
 *  the codeword can actually be fed directly to the transmit_diversity_precoding()
 *  function.
 *  Maybe useful for 4 layer mapping when the number of symbols
 *  per layer is not a multiple of 4
 * @param[in] codeword : PDSCH codeword used for layer mapping
 * @param[in, out] layers : buffer containing all the layer (same alignment as codeword)
 * @param[in] num_layers_ : Number of layers
 * @param[in] num_symbols_per_code_word_ : number of symbols in the codeword
 * @param[in] num_symbols_per_layer_ : number of symbols per layer
 */
void transmit_diversity_layer_mapper(std::complex<float> * codeword,
                                     std::complex<float> * layers,
                                     int num_layers_,
                                     int num_symbols_per_code_word_,
                                     int num_symbols_per_layer_);

/** From TS 36.211 Section 6.3.3.3
 *  Performs layer mapping of a PDSCH codeword
 * @param[in] codeword : PDSCH codeword
 * @param[in, out] layers : Final buffer containing the layers
 * @param[in] num_layers_ : Number of layers
 * @param[in] num_symbols_per_code_word_ : Number of symbols per codeword
 * @param[in] num_symbols_per_layer_ : Number of symbols per layer
 */
void transmit_diversity_layer_mapper(std::complex<float> * codeword,
                                     std::vector<std::vector<std::complex<float>>> &layers,
                                     int num_layers_,
                                     int num_symbols_per_code_word_,
                                     int num_symbols_per_layer_);

/** Performs LTE TM 1 transmit diversity precoding
 *  TS 36.211 section 6.3.4.3
 * @param[in] layers : Layers to be precoded (codeword can be fed directly,
 *                 see the transmit_diversity_layer_mapper function)
 * @param[in, out] precoded_layers : output buffer containing the precoded layers
 * @param[in] num_layers : Number of Layers
 * @param[in] num_symbols_per_layer_ : Number of symbols per layer
 * @param[in] num_symbols_per_precoded_layer_ : Number of symbols per precoded layer
 */
void transmit_diversity_precoding(std::complex<float> * layers,
                                  std::complex<float> * precoded_layers,
                                  int num_layers,
                                  int num_symbols_per_layer_,
                                  int num_symbols_per_precoded_layer_) ;

/** Performs LTE TM 1 transmit diversity precoding
 *  TS 36.211 section 6.3.4.3
 * @param[in] layers : Layers to be precoded (codeword can be fed directly,
 *                 see the transmit_diversity_layer_mapper function)
 * @param[in, out] precoded_layers : output buffer containing the precoded layers
 * @param[in] num_layers : Number of Layers
 * @param[in] num_symbols_per_layer_ : Number of symbols per layer
 * @param[in] num_symbols_per_precoded_layer_ : Number of symbols per precoded layer
 */
void transmit_diversity_precoding(std::vector<std::vector<std::complex<float>>> &layers,
                                  std::vector<std::vector<std::complex<float>>> &precoded_layers,
                                  int num_layers,
                                  int num_symbols_per_layer_,
                                  int num_symbols_per_precoded_layer_);

/** Performs NR layer mapping TS 38.211 table 7.3.1.3-1
 *  Maps up to 4 layers if one codeword is given in input
 *  Maps up to 8 layers if two codewords are provided
 *
 * @param[in] codewords : PDSCH codewords to be mapped on layers. First dimension : pointer to the codeword
 *                    2nd dimension : the actual codeword
 * @param[in, out] layers : final buffer containing the layers
 * @param[in] num_layers : Number of layers
 * @param[in] num_codewords_ : Number of codewords
 * @param[in] num_symbols_per_layer_ : Array containing the number of symbols for each layer
 * @param[in] num_symbols_codeword1_ : Number of symbols in codeword 1
 * @param[in] num_symbols_codeword2_ : Number of symbols in codeword 2
 */
void nr_layer_mapper(std::complex<float> ** codewords,
                     std::complex<float> * layers,
                     int num_layers,
                     int num_codewords_,
                     const int * num_symbols_per_layer_,
                     int num_symbols_codeword1_,
                     int num_symbols_codeword2_ = 0);

/** Counts the number of symbols per layer according to the
 *  number of symbols per codeword. To be used before NR layer mapper.
 *
 * @param[in] num_layers_ : Number of layers
 * @param[in] num_symbols_codeword_1 : Number of symbols in codeword 1
 * @param[in] num_symbols_codeword_2 : Number of symbols in codeword 2
 * @param[out] num_symbols_per_layer_ : Array containing the number of symbols for each layer
 */
void nr_num_symbols_per_layer(int num_layers_,
                              int num_symbols_codeword_1,
                              int num_symbols_codeword_2,
                              int * num_symbols_per_layer_);

/** TODO : Do for double symbols, mapping type B and type 2 */
/** Computes the CDM groups of each associated antenna port according to the input vector of antennas.
 *  For the DMRS single symbol case only.
 *
 * @param cdm_groups_       : cdm_groups to be computed (give an empty vector as argument)
 * @param input_antennas_   : input antenna ports
 */
void compute_cdm_groups_dmrs_single_symbol(std::vector<std::vector<int>> &cdm_groups_,
                                           std::vector<int> &input_antennas_,
                                           int dmrs_config_type_);

#endif //USRP_MIMO_MIMO_H
