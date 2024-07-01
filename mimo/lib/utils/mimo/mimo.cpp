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

#include "mimo.h"

using namespace std;

/** Computes the CDM groups of each associated antenna port according to the input vector of antennas.
 *  For the DMRS single symbol case only.
 *
 * @param cdm_groups_       : cdm_groups to be computed (give an empty vector as argument)
 * @param input_antennas_   : input antenna ports
 */
void compute_cdm_groups_dmrs_single_symbol(vector<vector<int>> &cdm_groups_,
                                           vector<int> &dmrs_ports_,
                                           int dmrs_config_type_) {

    /// Make sure that the input antenna ports are sorted in ascending order
    sort(dmrs_ports_.begin(), dmrs_ports_.end());

    if(dmrs_config_type_ == 1) {

        vector<vector<int>> temp_cdm_groups(2); // initialize up to two groups. Delete the group afterwards if needed.

        for(auto antenna_port : dmrs_ports_) {

            // CDM group 1
            if ((antenna_port == 0) or (antenna_port == 1)) {
                temp_cdm_groups[0].push_back(antenna_port);

                // CDM group 2
            } else if ((antenna_port == 2) or (antenna_port == 3)) {
                temp_cdm_groups[1].push_back(antenna_port);
            } else {
                /// TODO throw exception
                // "antenna port out of range for DMRS config type 1"

            }
        }

        /// Append non empty groups
        if(not temp_cdm_groups[0].empty()) {
            cdm_groups_.push_back(temp_cdm_groups[0]);
        }

        if(not temp_cdm_groups[1].empty()) {
            cdm_groups_.push_back(temp_cdm_groups[1]);
        }

    } else {
        vector<vector<int>> temp_cdm_groups(3); // initialize up to three groups. Delete the group afterwards if needed.

        for(auto antenna_port : dmrs_ports_) {

            // CDM group 1
            if ((antenna_port == 0) or (antenna_port == 1)) {
                temp_cdm_groups[0].push_back(antenna_port);

                // CDM group 2
            } else if ((antenna_port == 2) or (antenna_port == 3)) {
                temp_cdm_groups[1].push_back(antenna_port);

            } else if ((antenna_port == 4) or (antenna_port == 5)) {
                temp_cdm_groups[2].push_back(antenna_port);

            } else {
                /// TODO Throw exception
                /// "antenna port out of range for DMRS config type 1"
            }
        }

        /// Append non empty groups
        if(not temp_cdm_groups[0].empty()) {
            cdm_groups_.push_back(temp_cdm_groups[0]);
        }

        if(not temp_cdm_groups[1].empty()) {
            cdm_groups_.push_back(temp_cdm_groups[1]);
        }

        if(not temp_cdm_groups[2].empty()) {
            cdm_groups_.push_back(temp_cdm_groups[2]);
        }
    }
}

void transmit_diversity_layer_mapper(complex<float> * codeword,
                                     complex<float> * layers,
                                     int num_layers_,
                                     int num_symbols_per_code_word_,
                                     int num_symbols_per_layer_) {

    for(int symbol = 0; symbol < num_symbols_per_layer_; symbol++) {
        for(int layer = 0; layer < num_layers_; layer++) {
            layers[num_layers_ * symbol + layer] = codeword[num_layers_ * symbol + layer];
        }
    }
}

void transmit_diversity_layer_mapper(complex<float> * codeword,
                                     vector<vector<complex<float>>> &layers,
                                     int num_layers_,
                                     int num_symbols_per_code_word_,
                                     int num_symbols_per_layer_) {

    if(num_layers_ == 2) {
        for(int symbol = 0; symbol < num_symbols_per_layer_; symbol++) {
            layers[0][symbol] = *(codeword);
            layers[1][symbol] = *(codeword + 1);
            codeword += 2;
        }
    } else if (num_layers_ == 4) {
        for(int symbol = 0; symbol < num_symbols_per_layer_; symbol++) {
            layers[0][symbol] = *(codeword);
            layers[1][symbol] = *(codeword + 1);
            layers[2][symbol] = *(codeword + 2);
            layers[3][symbol] = *(codeword + 3);
            codeword += 4;
        }
    }
}

void nr_num_symbols_per_layer(int num_layers_,
                              int num_symbols_codeword_1,
                              int num_symbols_codeword_2,
                              int * num_symbols_per_layer_) {

    switch (num_layers_) {
        case 1: /// Only 1 layer, and only 1 codeword
            num_symbols_per_layer_[0] = num_symbols_codeword_1;
            break;

        case 5:
            for (int layer = 0; layer < num_layers_; layer++) {
                if (layer < 2) {
                    num_symbols_per_layer_[layer] = num_symbols_codeword_1 / 2;
                } else {
                    num_symbols_per_layer_[layer] = num_symbols_codeword_2 / 3;
                }
            }
            break;

        case 7 :
            for (int layer = 0; layer < num_layers_; layer++) {
                if (layer < 3) {
                    num_symbols_per_layer_[layer] = num_symbols_codeword_1 / 3;
                } else {
                    num_symbols_per_layer_[layer] = num_symbols_codeword_2 / 4;
                }
            }
            break;

        default : /// Fill the layers using round robin on the codewords

            /// 1 codeword case
            if (num_layers_ < 5) {

                /// Resize the layers
                for (int layer = 0; layer < num_layers_; layer++) {
                    num_symbols_per_layer_[layer] = (num_symbols_codeword_1 / num_layers_);
                }

                /// 2 codewords case
                /// The two codewords must have the same size
            } else {

                /// Resize the layers
                for (int layer = 0; layer < num_layers_; layer++) {
                    num_symbols_per_layer_[layer] = (num_symbols_codeword_1 / num_layers_);
                }
                break;
            }
        }
}

/**
 * TODO : throw exception for the following cases :
 *          - number of codewords = 1 and number of layers > 4
 *          - number of codewords != 1 and number of layers < 4
 *          - number of layers different from number of layers in vector &layers
 */
void nr_layer_mapper(complex<float> ** codewords,
                     complex<float> * layers,
                     int num_layers,
                     int num_codewords_,
                     const int * num_symbols_per_layer_,
                     int num_symbols_codeword1_,
                     int num_symbols_codeword2_) {

    switch(num_layers) {

        case 1:
            /// Fill the layer with the codeword
            for(int symbol = 0; symbol < num_symbols_codeword1_; symbol++) {
                layers[0 * num_symbols_per_layer_[0] + symbol] = codewords[0][symbol];
            }
            break ;

        case 5:
            for(int symbol = 0; symbol < num_symbols_codeword1_/2; symbol++) {
                for(int layer = 0; layer < 2; layer++) {
                    layers[layer * num_symbols_per_layer_[layer] + symbol] = codewords[0][2 * symbol + layer];
                }
            }

            for(int symbol = 0; symbol < num_symbols_codeword2_/3; symbol++) {
                for(int layer = 3; layer < num_layers; layer++) {
                    layers[layer * num_symbols_per_layer_[layer] + symbol] = codewords[1][3 * symbol + layer - 2];
                }
            }

            break ;

        case 7 :
            for(int symbol = 0; symbol < num_symbols_codeword1_/3; symbol++) {
                for (int layer = 0; layer < num_layers; layer++) {
                    layers[layer * num_symbols_per_layer_[layer] + symbol] = codewords[0][3 * symbol + layer];
                }
            }

            for(int symbol = 0; symbol < num_symbols_codeword2_/4; symbol++) {
                for(int layer = 3; layer < num_layers; layer++) {
                    layers[layer * num_symbols_per_layer_[layer] + symbol] = codewords[1][4 * symbol + layer - 3];
                }
            }
            break ;

        default : /// Fill the layers using round robin on the codewords
            /// 1 codeword case
            if(num_layers < 5) {

                for(int symbol = 0; symbol < num_symbols_codeword1_/num_layers; symbol++) {
                    for(int layer = 0; layer < num_layers; layer++) {
                        layers[layer * num_symbols_per_layer_[layer] + symbol] = codewords[0][num_layers * symbol + layer];
                    }
                }

                /// 2 codewords case
                /// The two codewords must have the same size
            } else {

                for(int symbol = 0; symbol < num_symbols_codeword1_/num_layers; symbol++) {

                    for(int layer = 0; layer < num_layers/2; layer++) {
                        layers[layer * num_symbols_per_layer_[layer] + symbol] = codewords[0][num_layers/2 * symbol + layer];
                    }

                    for(int layer = num_layers/2; layer < num_layers; layer++) {
                        layers[layer * num_symbols_per_layer_[layer] + symbol] = codewords[1][num_layers/2 * symbol + (layer%num_layers)];
                    }
                }

            }

            break;
    }
}

void transmit_diversity_precoding(complex<float> * layers,
                                  complex<float> * precoded_layers,
                                  int num_layers,
                                  int num_symbols_per_layer_,
                                  int num_symbols_per_precoded_layer_) {

    /// MIMO 2x2
    if(num_layers == 2) { /// Case where rank = 2 and antenna ports used are {0, 1}

        for(int i = 0; i < num_symbols_per_layer_; i++) {
            *(precoded_layers + 0 * num_symbols_per_precoded_layer_ + 2 * i) = *(layers + i * num_layers + 0); //(1/ float (sqrt(2))) * layers[0][i];
            *(precoded_layers + 1 * num_symbols_per_precoded_layer_ + 2 * i) = - conj(*(layers + i * num_layers + 1)); //(1/ float (sqrt(2))) * (- conj(layers[1][i]));
            *(precoded_layers + 0 * num_symbols_per_precoded_layer_ + 2 * i + 1) = *(layers + i * num_layers + 1); // (1/ float (sqrt(2))) * layers[1][i];
            *(precoded_layers + 1 * num_symbols_per_precoded_layer_ + 2 * i + 1) = conj(*(layers + i * num_layers + 0)); //(1/ float (sqrt(2))) * conj(layers[0][i]);
        }

        /// MIMO 4x4
    } else if(num_layers == 4) {
        for(int i = 0; i < num_symbols_per_layer_; i++) {
            *(precoded_layers + 0 * num_symbols_per_precoded_layer_ + 4 * i)     = *(layers + i * num_layers + 0); //(1/ float (sqrt(2))) * layers[0][i];
            *(precoded_layers + 2 * num_symbols_per_precoded_layer_ + 4 * i)     = - conj(*(layers + i * num_layers + 1)); //(1/ float (sqrt(2))) * (- conj(layers[1][i]));
            *(precoded_layers + 0 * num_symbols_per_precoded_layer_ + 4 * i + 1) = *(layers + i * num_layers + 1); //(1/ float (sqrt(2))) * layers[1][i];
            *(precoded_layers + 2 * num_symbols_per_precoded_layer_ + 4 * i + 1) = conj(*(layers + i * num_layers + 0)); //(1/ float (sqrt(2))) *    conj(layers[0][i]);
            *(precoded_layers + 1 * num_symbols_per_precoded_layer_ + 4 * i + 2) = *(layers + i * num_layers + 2); //(1/ float (sqrt(2))) * layers[2][i];
            *(precoded_layers + 3 * num_symbols_per_precoded_layer_ + 4 * i + 2) = - conj(*(layers + i * num_layers + 3)); //(1/ float (sqrt(2))) * (- conj(layers[3][i]));
            *(precoded_layers + 1 * num_symbols_per_precoded_layer_ + 4 * i + 3) = *(layers + i * num_layers + 3); //(1/ float (sqrt(2))) * layers[3][i];
            *(precoded_layers + 3 * num_symbols_per_precoded_layer_ + 4 * i + 3) = conj(*(layers + i * num_layers + 2)); //(1/ float (sqrt(2))) *    conj(layers[2][i]);
        }
    }
}

void transmit_diversity_precoding(vector<vector<complex<float>>> &layers,
                                  vector<vector<complex<float>>> &precoded_layers,
                                  int num_layers,
                                  int num_symbols_per_layer_,
                                  int num_symbols_per_precoded_layer_) {

    /// Comment the scaling factor of 1/sqrt(2) because scaling factor is already applied for IFFT

    /// MIMO 2x2
    if(num_layers == 2) { /// Case where rank = 2 and antenna ports used are {0, 1}
        for(int i = 0; i < num_symbols_per_layer_; i++) {
            precoded_layers[0][2 * i] = layers[0][i]; //(1/ float (sqrt(2))) * layers[0][i];
            precoded_layers[1][2 * i] = - conj(layers[1][i]); //(1/ float (sqrt(2))) * (- conj(layers[1][i]));
            precoded_layers[0][2 * i + 1] = layers[1][i]; // (1/ float (sqrt(2))) * layers[1][i];
            precoded_layers[1][2 * i + 1] = conj(layers[0][i]); //(1/ float (sqrt(2))) * conj(layers[0][i]);
        }

        /// MIMO 4x4
    } else if(num_layers == 4) {

        for(int i = 0; i < num_symbols_per_layer_; i++) {
            precoded_layers[0][4*i]     = layers[0][i]; //(1/ float (sqrt(2))) * layers[0][i];
            precoded_layers[2][4*i]     = - conj(layers[1][i]); //(1/ float (sqrt(2))) * (- conj(layers[1][i]));
            precoded_layers[0][4*i + 1] = layers[1][i]; //(1/ float (sqrt(2))) * layers[1][i];
            precoded_layers[2][4*i + 1] = conj(layers[0][i]); //(1/ float (sqrt(2))) *    conj(layers[0][i]);
            precoded_layers[1][4*i + 2] = layers[2][i]; //(1/ float (sqrt(2))) * layers[2][i];
            precoded_layers[3][4*i + 2] = - conj(layers[3][i]); //(1/ float (sqrt(2))) * (- conj(layers[3][i]));
            precoded_layers[1][4*i + 3] = layers[3][i]; //(1/ float (sqrt(2))) * layers[3][i];
            precoded_layers[3][4*i + 3] = conj(layers[2][i]); //(1/ float (sqrt(2))) *    conj(layers[2][i]);
       }
    }
}