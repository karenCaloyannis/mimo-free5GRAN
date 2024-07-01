/*
   Copyright 2023-2024 Telecom Paris

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

   This file is a modified version of the lib/common_utils/common_utils.cpp
   file from the free5GRAN library.
*/

#include "variables.h"

using namespace std;

sem_t * rx_estimation_semaphores;
sem_t * rx_interpolation_semaphores;
sem_t * wait_estimation_and_interpolation_semaphores;

int SSS_BASE_X0_SEQUENCE[] = {1, 0, 0, 0, 0, 0, 0};
int SSS_BASE_X1_SEQUENCE[] = {1, 0, 0, 0, 0, 0, 0};

/// Table 7.4.1.1.2-1 TS 38.211 : Parameters for PDSCH DM-RS configuration type 1
/// Dimensions : port number
///                 CDM group lambda, Delta, w_f(k') (2 columns), w_t(l') (2 columns)
int ts_38_211_7_4_1_1_2_1[8][6] = {{0, 0, +1, +1, +1, +1},
                                   {0, 0, +1, -1, +1, +1},
                                   {1, 1, +1, +1, +1, +1},
                                   {1, 1, +1, -1, +1, +1},
                                   {0, 0, +1, +1, +1, -1},
                                   {0, 0, +1, -1, +1, -1},
                                   {1, 1, +1, +1, +1, -1},
                                   {1, 1, +1, -1, +1, -1}};

/// Table 7.4.1.1.2-2 TS 38.211 : Parameters for PDSCH DM-RS configuration type 2
/// Dimensions : port number
///                 CDM group lambda, Delta, w_f(k'), w_t(l')
int ts_38_211_7_4_1_1_2_2[12][6] = {{0, 0, +1, +1, +1, +1},
                                    {0, 0, +1, -1, +1, +1},
                                    {1, 2, +1, +1, +1, +1},
                                    {1, 2, +1, -1, +1, +1},
                                    {2, 4, +1, +1, +1, +1},
                                    {2, 4, +1, -1, +1, +1},
                                    {0, 0, +1, +1, +1, -1},
                                    {0, 0, +1, -1, +1, -1},
                                    {1, 2, +1, +1, +1, -1},
                                    {1, 2, +1, -1, +1, -1},
                                    {2, 4, +1, +1, +1, -1},
                                    {2, 4, +1, -1, +1, -1}};

/// Array containing the qpsk symbols
const complex<float> qpsk[4] = { (complex<float>)   1/((float) sqrt(2)) + ((complex<float>) complex<float>(0,1)/((float) sqrt(2))),
                                 (complex<float>) - 1/((float) sqrt(2)) + ((complex<float>) complex<float>(0,1)/((float) sqrt(2))),
                                 (complex<float>) - 1/((float) sqrt(2)) - ((complex<float>) complex<float>(0,1)/((float) sqrt(2))),
                                 (complex<float>)   1/((float) sqrt(2)) - ((complex<float>) complex<float>(0,1)/((float) sqrt(2))) };

/// Array containing the bpsk symbols
const complex<float> bpsk[2] = {1/sqrt(2), -1/sqrt(2)};

/// Mapping between value of 'Antenna ports', 'Number of CDM groups without data', 'DMRS ports' and additionnally the CDM groups
/// Check TS 38.212 section 7.3.1.2.2
/// Table Table 7.3.1.2.2-1: Antenna port(s) (1000 + DMRS port), dmrs-Type=1, maxLength=1
vector<int *> antenna_port_cdm_groups = {
        new int[1] {0}, // DMRS port 0, num cdm groups wo data = 1
        new int[1] {0}, // DMRS port 1, num cdm groups wo data = 1
        new int[2] {0, 0}, // DMRS port 0-1, num cdm groups wo data = 1
        new int[1] {0}, // DMRS port 0, num cdm groups wo data = 2
        new int[1] {0}, // DMRS port 1, num cdm groups wo data = 2
        new int[1] {1}, // DMRS port 2, num cdm groups wo data = 2
        new int[1] {1}, // DMRS port 3, num cdm groups wo data = 2
        new int[2] {0, 0}, // DMRS port 0-1, num cdm groups wo data = 2
        new int[2] {1, 1}, // DMRS port 2-3, num cdm groups wo data = 2
        new int[3] {0, 0, 1}, // DMRS port 0 - 2, num cdm groups wo data = 2
        new int[4] {0, 0, 1, 1}, // DMRS port 0 to 3, num cdm groups wo data = 2
        new int[2] {0, 1}  // DMRS port 0 and 2, num cdm groups wo data = 2
};

vector<int *> antenna_port_cdm_groups_sizes = {
        new int[2] {1, 0},
        new int[2] {1, 0},
        new int[2] {1, 0},
        new int[2] {1, 0},
        new int[2] {1, 0},
        new int[2] {0, 1},
        new int[2] {0, 1},
        new int[2] {2, 0},
        new int[2] {0, 2},
        new int[2] {2, 1},
        new int[2] {2, 2},
        new int[2] {1, 1}
};

vector<int *> antenna_port_used_cdm_groups = {
        new int[1] {0},
        new int[1] {0},
        new int[1] {0},
        new int[1] {0},
        new int[1] {0},
        new int[1] {1},
        new int[1] {1},
        new int[1] {0},
        new int[1] {1},
        new int[2] {0, 1},
        new int[2] {0, 1},
        new int[2] {0, 1}
};

vector<int*> cdm_group_sizes = {
        {new int[1]{1}},
        {new int[1]{1}},
        {new int[2]{2, 2}},
        {new int[1]{1}},
        {new int[1]{1}},
        {new int[1]{1}},
        {new int[1]{1}},
        {new int[2]{2, 2}},
        {new int[2]{2, 2}},
        {new int[3]{2, 2, 1}},
        {new int[4]{2, 2, 2, 2}},
        {new int[2]{1, 1}},
};

vector<int * > antenna_port_dmrs_ports = {
        new int[1] {0},
        new int[1] {1},
        new int[2] {0, 1},
        new int[1] {0},
        new int[1] {1},
        new int[1] {2},
        new int[1] {3},
        new int[2] {0, 1},
        new int[2] {2, 3},
        new int[3] {0, 1, 2},
        new int[4] {0, 1, 2, 3},
        new int[2] {0, 2}
};

int antenna_port_num_cdm_groups[12] = {
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        1,
        2,
        2,
        2
};

std::vector<std::vector<size_t>> antenna_port_value_tx_usrp_ports = {
        {0},
        {1},
        {0, 1},
        {0},
        {1},
        {2},
        {3},
        {0, 1},
        {2, 3},
        {0, 1, 3},
        {0, 1, 2, 3},
        {0, 2}
};
std::vector<std::vector<size_t>> antenna_port_value_rx_usrp_ports = {
        {0},
        {1},
        {0, 1},
        {0},
        {1},
        {2},
        {3},
        {0, 1},
        {2, 3},
        {0, 1, 3},
        {0, 1, 2, 3},
        {0, 2}
};

int antenna_ports_num_cdm_groups_without_data[12] = {
        1,
        1,
        1,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
        2,
};

int antenna_ports_num_dmrs_ports[12] = {
        1,
        1,
        2,
        1,
        1,
        1,
        1,
        2,
        2,
        3,
        4,
        2
};

/// Array containing all the constellations and corresponding symbols
complex<float> * constellations[2] = {

        /// QPSK symbols
        new complex<float>[4] {(complex<float>)   1/((float) sqrt(2)) + (complex<float>(0, 1)/((float) sqrt(2))),
                                   (complex<float>) - 1/((float) sqrt(2)) + (complex<float>(0, 1)/((float) sqrt(2))),
                                   (complex<float>) - 1/((float) sqrt(2)) - (complex<float>(0, 1)/((float) sqrt(2))),
                                   (complex<float>)   1/((float) sqrt(2)) - (complex<float>(0, 1)/((float) sqrt(2)))},

        /// BPSK symbols
        new complex<float>[2] {1/sqrt(2), -1/sqrt(2)}
};

int constellation_sizes[2] = {
        4, 2
};

/// Table 5.2.2.2.1-1 TS 38.214 : Codebooks for 1-layer and 2-layer CSI reporting using antenna ports 3000 to 3001
/// No beamforming
/*
vector<vector<Eigen::MatrixXcf> > ts_38_214_5_2_2_2_1_1 =
        {
                {

                        Eigen::MatrixXcf // Codebook index 0, 1 layer
                        {
                            {(1 / sqrt(2))},
                            {(1 / sqrt(2))}
                        },

                        Eigen::MatrixXcf // Codebook index 0, 2 layers
                        {
                            { 0.5, 0.5},
                            { 0.5, -0.5}
                        }

                },

                {
                        Eigen::MatrixXcf // Codebook index 1, 1 layer
                                {
                                        {complex<float>(1 / sqrt(2))},
                                        {complex<float>(0, 1 / sqrt(2))}
                                },

                        Eigen::MatrixXcf // Codebook index 1, 2 layers
                                {
                                        { 0.5, 0.5},
                                        { complex<float>(0.5, 1), complex<float> (0.5, -1)}
                                }

                },

                {
                        Eigen::MatrixXcf // Codebook index 2, 1 layer. No Codebook index 2 for 2 layers
                        {
                                { (1 / sqrt(2))},
                                { - (1 / sqrt(2))}
                        }
                },


                {
                        Eigen::MatrixXcf {  // Codebook index 2, 1 layer. No Codebook index 2 for 2 layers
                                { (1 / sqrt(2))},
                                { complex<float>(1 / sqrt(2), -1)}
                        }
                },
        };


*/

/// From ShareTechNote : https://www.sharetechnote.com/html/5G/5G_SLIV.html
/// First dimension : SLIV value
/// Second dimension : S value
/// Third dimension : L value
int sliv_s_l_mapping[105][2] = {
        {0, 1},
        {1, 1},
        {2, 1},
        {3, 1},
        {4, 1},
        {5, 1},
        {6, 1},
        {7, 1},
        {8, 1},
        {9, 1},
        {10, 1},
        {11, 1},
        {12, 1},
        {13, 1},
        {0, 2},
        {1, 2},
        {2, 2},
        {3, 2},
        {4, 2},
        {5, 2},
        {6, 2},
        {7, 2},
        {8, 2},
        {9, 2},
        {10, 2},
        {11, 2},
        {12, 2},
        {0, 14},
        {0, 3},
        {1, 3},
        {2, 3},
        {3, 3},
        {4, 3},
        {5, 3},
        {6, 3},
        {7, 3},
        {8, 3},
        {9, 3},
        {10, 3},
        {11, 3},
        {1, 13},
        {0, 13},
        {0, 4},
        {1, 4},
        {2, 4},
        {3, 4},
        {4, 4},
        {5, 4},
        {6, 4},
        {7, 4},
        {8, 4},
        {9, 4},
        {10, 4},
        {2, 12},
        {1, 12},
        {0, 12},
        {0, 5},
        {1, 5},
        {2, 5},
        {3, 5},
        {4, 5},
        {5, 5},
        {6, 5},
        {7, 5},
        {8, 5},
        {9, 5},
        {3, 11},
        {2, 11},
        {1, 11},
        {0, 11},
        {0, 6},
        {1, 6},
        {2, 6},
        {3, 6},
        {4, 6},
        {5, 6},
        {6, 6},
        {7, 6},
        {8, 6},
        {4, 10},
        {3, 10},
        {2, 10},
        {1, 10},
        {0, 10},
        {0, 7},
        {1, 7},
        {2, 7},
        {3, 7},
        {4, 7},
        {5, 7},
        {6, 7},
        {7, 7},
        {5, 9},
        {4, 9},
        {3, 9},
        {2, 9},
        {1, 9},
        {0, 9},
        {0, 8},
        {1, 8},
        {2, 8},
        {3, 8},
        {4, 8},
        {5, 8},
        {6, 8}
};