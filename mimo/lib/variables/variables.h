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

   This file is a modified version of the lib/common_utils/common_utils.h
   file from the free5GRAN library.

 */

#ifndef VARIABLES_H
#define VARIABLES_H

#include <complex>
#include <vector>
#include <vector>
#include <semaphore.h>
#include <boost/log/trivial.hpp>

#define MAX_TX_PORTS 4
#define MAX_RX_PORTS 4
#define MAX_NUM_CDM_GROUPS 3

#define MAX_SYMBOLS 14
#define MAX_DMRS_SYMBOLS 4
#define MAX_SUBCARRIERS 256
#define MAX_DMRS_SUBCARRIERS 128

#define SIZE_PSS_SSS_SIGNAL 127
#define NUM_N_ID_2 3 /// maximum number of N_ID_2

//#define INTERP_NORM

#define PILOT_ESTIMATE_AVX2

/** define CLOCK_TYPE_CHRONO to use std::chrono::steady_clock()
 *  define CLOCK_TYPE_GETTIME to use clock_gettime
 *  define CLOCK_TYPE_ASM to use RDTSC & RDTSCP
 *  define CLOCK_TYPE_CLOCK to use clock()
 */
#define CLOCK_TYPE_ASM
#define TSC_FREQ 3699.850e6

extern sem_t * rx_estimation_semaphores;
extern sem_t * rx_interpolation_semaphores;
extern sem_t * wait_estimation_and_interpolation_semaphores;

extern int stop_signal_called_channel_estimation;

/// Pointers to the ML detector functions
typedef void (*ml_detector_complex_pointer)(const std::complex<float> &, int &);
typedef void (*ml_detector_vectors_pointer)(const std::vector<std::complex<float>> &, std::vector<int>&);
typedef void (*ml_detector_tabs_pointer)(std::complex<float> *, int*, const int&);
typedef int  (*ml_detector_complex_inline_ptr)(const std::complex<float> &);

typedef void (*interp_func_ptr)(std::complex<float> * coef_grid,
                                std::complex<float> * dmrs_coefs_,
                                int cdm_group_number,
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

enum mimo_encoding_type {
    diversity,
    vblast,
    none
};

enum constellation_type {
    qpsk_constellation,
    bpsk_constellation
};

enum dmrs_additionnal_positions_enum {
    pos0, pos1, pos2, pos3
};

enum pdsch_mapping_type_enum {
    typeB = (char) 'b', typeA = (char) 'a'
};

enum dmrs_config_type_enum {
    type1 = 1, type2 = 2
};

enum boost_log_level {
    trace = boost::log::trivial::trace,
    info = boost::log::trivial::info
};

/// Mapping between value of 'Antenna ports', 'Number of CDM groups without data', 'DMRS ports' and additionnally the CDM groups
/// Check TS 38.212 section 7.3.1.2.2
/// Table Table 7.3.1.2.2-1: Antenna port(s) (1000 + DMRS port), dmrs-Type=1, maxLength=1
extern std::vector<int *> antenna_port_cdm_groups;
extern std::vector<int *> antenna_port_cdm_groups_sizes;
extern std::vector<int *> antenna_port_dmrs_ports;
extern int antenna_ports_num_dmrs_ports[12];
extern int antenna_port_num_cdm_groups[12];
extern int antenna_ports_num_cdm_groups_without_data[12];
extern std::vector<int*> cdm_group_sizes;
extern std::vector<std::vector<size_t>> antenna_port_value_tx_usrp_ports;
extern std::vector<std::vector<size_t>> antenna_port_value_rx_usrp_ports;
extern std::vector<int *> antenna_port_used_cdm_groups;

/// Defined in channel_estimation.cpp
extern ml_detector_complex_pointer ml_detector_complex[2];
extern ml_detector_vectors_pointer ml_detector_vectors[2];
extern ml_detector_tabs_pointer ml_detector_tabs[2];
extern ml_detector_complex_inline_ptr ml_detector_inline[2];

/// Array containing all the constellations and corresponding symbols
extern std::complex<float> * constellations[2];
extern int constellation_sizes[2];

/// Array containing the qpsk symbols
extern const std::complex<float> qpsk[4];

/// Array containing the bpsk symbols
extern const std::complex<float> bpsk[2];

/// from free5GRAN. PDSCH time domain allocation A for normal CP
extern int TS_38_214_TABLE_5_1_2_1_1_2[16][2][4];

/// Table 7.4.1.1.2-1 TS 38.211 : Parameters for PDSCH DM-RS configuration type 1
/// Dimensions : port number
///                 CDM group lambda, Delta, w_f(k'), w_t(l')
extern int ts_38_211_7_4_1_1_2_1[8][6];

/// Table 7.4.1.1.2-2 TS 38.211 : Parameters for PDSCH DM-RS configuration type 2
/// Dimensions : port number
///                 CDM group lambda, Delta, w_f(k'), w_t(l')
extern int ts_38_211_7_4_1_1_2_2[12][6];

/// Table 5.2.2.2.1-1 TS 38.214 : Codebooks for 1-layer and 2-layer CSI reporting using antenna ports 3000 to 3001
/// No beamforming
//extern std::vector<std::vector<Eigen::MatrixXcf> > ts_38_214_5_2_2_2_1_1;

/// Variables used for VBLAST QR decomp. decoder. Initialized in the Grid Class.
/**
extern Eigen::MatrixXcf channel_matrix, r_matrix, q_matrix;
extern Eigen::VectorXcf received_symbols, w_ki;
extern Eigen::VectorXcf q_h_symbols;
extern std::vector<std::complex<float>> equalized_symbols;
*/

/// Mapping between SLIV value and S, L allocations of PDSCH
extern int sliv_s_l_mapping[105][2];

extern int SSS_BASE_X0_SEQUENCE[7];
extern int SSS_BASE_X1_SEQUENCE[7];

#endif //VARIABLES_H