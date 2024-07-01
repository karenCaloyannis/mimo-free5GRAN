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
#include <thread>
#include <csignal>
#include <math.h>
#include <future>
#include <chrono>
#include <sys/resource.h>
#include <sys/sysinfo.h>
#include <experimental/random>
#include <boost/log/core.hpp>
#include <boost/log/expressions.hpp>
#include <boost/log/trivial.hpp>
#include <boost/log/utility/setup/common_attributes.hpp>
#include <boost/log/utility/setup/file.hpp>

#include "time.h"
//#include <ctime>

#include "../../lib/usrp/usrp_x300.h"
#include "../../lib/free5gran_utils.h"
#include "../../lib/utils/channel_estimation/channel_estimation.h"
#include "../../lib/utils/channel_mapping/channel_mapping.h"
#include "../../lib/utils/mimo/mimo.h"
#include "../../lib/utils/mimo/transmit_diversity.h"
#include "../../lib/utils/mimo/vblast.h"
#include "../../lib/variables/variables.h"

/** Define :
 *      VBLAST_ZF for ZF, uses AVX2 if built with AVX2
 *      VBLAST_ZF_BLOCK_WISE_INV for ZF with block wise inversion
 *      VBLAST_ZF_FLOAT for ZF working on float values
 *      VBLASt_ZF_SEP for ZF with each separate step timed with RDTSC
 *      VBLAST_QRD_COL_NORM for QRD with column norm redordering prior to QRD
 *      VBLAST_QRD_COL_NORM_AVX2 for QRD with wolumn norm reordering prior to QRD with AVX2
 *      VBLAST_SQRD for SQRD uses AVX2 if built with AVX2
 *                  uses AVX512 if built with AVX512
 *      VBLAST_MF for Matched Filter
 *      VBLAST_QRD for QRD without reordering
 */
#define VBLAST_SQRD

mimo_encoding_type encoding_type = vblast; // Use "diversity" for alamouti encoding, "vblast" or "none" to use only 1TX/1RX

/// Define the number of layers to be used, up to 4 layers.
/// Set to 2 or 4 for alamouti, and 1 for SISO
#define NUM_LAYERS 4

int stop_signal_called_main = 0; /// Set to 1 to stop the main thread
int stop_signal_called_channel_estimation = 0; /// Set to 1 to stop the RX channel estimation threads

using namespace std;

/** Handler of SIGINT signal
 */
void sigintHandler(int sig_num) {
    stop_signal_called_main = 1;
}

int UHD_SAFE_MAIN(int argc, char *argv[]) {

#if defined(CLOCK_TYPE_CHRONO)
    std::chrono::steady_clock::time_point t1{}, t2{}, t3{}, t4{}, t5{}, t6{};
#elif defined(CLOCK_TYPE_GETTIME)
    struct timespec t1, t2, t3, t4;
#elif defined(CLOCK_TYPE_ASM)
    uint64_t t1, t2, t3, t4, t_cumul;
    unsigned cycles_low1, cycles_low2, cycles_low3, cycles_low4,
            cycles_high1, cycles_high2, cycles_high3, cycles_high4;
#elif defined(CLOCK_TYPE_CLOCK)
    clock_t t1, t2, t3, t4;
    cout << "CLOCKS_PER_SEC : " << CLOCKS_PER_SEC << endl;
#elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
    struct timespec t1, t2, t3, t4;
#elif defined(CLOCK_TYPE_RUSAGE)
    struct rusage usage;
    timeval t1_utime, t2_utime, t3_utime, t4_utime;
#endif
    /// Use boost to log to a file like in free5GRAN
    boost::log::add_file_log(boost::log::keywords::file_name = "logfile.log",
                             boost::log::keywords::target_file_name = "logfile.log");
    boost::log::core::get()->set_filter(
            boost::log::trivial::severity >= boost::log::trivial::trace
    );
    boost::log::add_common_attributes();

    /** Set process priority */
    setpriority(PRIO_PROCESS, 0, -20);

    string command1 = "sudo sysctl -w net.core.wmem_max=24862979";
    int res1 = system(command1.c_str());

    /** Set CPUs to performance mode: */
    int res;
    for (int i = 0; i < get_nprocs_conf(); i++)
    {
        string command = "sudo cpufreq-set -c " + to_string(i) + " -r -g performance";
        res = system(command.c_str());
    }

    // Stop main on Ctrl+C
    signal(SIGINT, &sigintHandler);

    /// Grid parameters
    float        bandwidth                 = 15.36e6;                  // bandwidth in Hz
    float        sampling_rate             = bandwidth;               // sampling rate
    unsigned int numerology_index          = 1;                       // numerology index
    int          symbols_per_subframe      = 14*pow(2, numerology_index);     // Number of symbols per subframe.
    int          nb_tx_subframes           = 10;                       // number of subframes to build in the transmitted signal
    int          nb_rx_subframes           = 2 * nb_tx_subframes;     // number of Subframes to keep in the receiving buffer for slot synchronization
    float        scs                       = 15e3 * pow(2, numerology_index); // SCS
    int          fft_size                  = int(bandwidth/scs);      // Number of subcarriers / FFT size
    float        scaling_factor            = 0.02;                    // IFFT scaling factor
    float        center_frequency          = 3.8e9;               // center frequency NR ARFCN : 653333
    int          constellation_type        = 0;                       // Constellation type, use 0 for QPSK or 1 for BPSK
    float        gain_tx                   = 35;                      // max 89.8 dB
    float        gain_rx                   = 30;                      // max 76 dB
    bool         double_symbol             = false;                   // set to True to use double symbol DMRS
    int          l0                        = 2;                       // First symbol of DMRS within a PDSCH allocation
    int          pdsch_length              = 14;                      // PDSCH duration within a slot
    int          dmrs_additional_positions = 3;                       // Number of additional positions
    int          pdsch_start               = 0;                       // First symbol of the PDSCH within a slot
    int pss_position = 0, n_id_2 = 0, n_id_1 = 0;
    int nb_slots = nb_tx_subframes * pow(2, numerology_index);

    boost_log_level level = trace;

    BOOST_LOG_TRIVIAL(info) << "Bandwidth : " << bandwidth << endl;
    BOOST_LOG_TRIVIAL(info) << "Sampline Rate : " << bandwidth << endl;
    BOOST_LOG_TRIVIAL(info) << "numerology index : " << numerology_index << endl;
    BOOST_LOG_TRIVIAL(info) << "Symbols per subframe : " << symbols_per_subframe << endl;
    BOOST_LOG_TRIVIAL(info) << "Number of transmitted subframes : " << nb_tx_subframes <<  endl;
    BOOST_LOG_TRIVIAL(info) << "Number of received subframes : " << nb_rx_subframes << endl;
    BOOST_LOG_TRIVIAL(info) << "SCS : " << scs << endl;
    BOOST_LOG_TRIVIAL(info) << "FFT size : " << fft_size << endl;
    cout << "FFT size : " << fft_size << endl;
    BOOST_LOG_TRIVIAL(info) << "Scaling factor : " << scaling_factor << endl;
    BOOST_LOG_TRIVIAL(info) << "Center frequency : " << center_frequency << endl;
    BOOST_LOG_TRIVIAL(info) << "Constellation type : " << constellation_type << endl;
    BOOST_LOG_TRIVIAL(info) << "Gain TX : " << gain_tx << endl;
    BOOST_LOG_TRIVIAL(info) << "Gain RX : " << gain_rx << endl;
    //cout << "nb slots : " << nb_slots << endl;
    BOOST_LOG_TRIVIAL(trace) << "number of slots : " << nb_slots << endl;
    //cout << "scaling factor : " << scaling_factor << endl;

#if defined(VBLAST_ZF)
#if defined(__AVX2__) and defined(AVX2_PROCESSING)
    cout << "VBLAST decoding type : ZF AVX2" << endl;
    BOOST_LOG_TRIVIAL(info) << "VBLAST decoding type : ZF AVX2" << endl;
#else
    cout << "VBLAST decoding type : ZF" << endl;
    BOOST_LOG_TRIVIAL(info) << "VBLAST decoding type : ZF" << endl;
#endif
#elif defined(VBLAST_ZF_SEP)
    cout << "VBLAST decoding type : ZF SEPARATE STEPS" << endl;
    BOOST_LOG_TRIVIAL(info) << "VBLAST decoding type : ZF SEPARATE STEPS" << endl;
#elif defined(VBLAST_QRD_COL_NORM)
#if defined(__AVX2__) and defined(AVX2_PROCESSING)
    cout << "VBLAST decoding type : QRD column norm reordering" << endl;
    BOOST_LOG_TRIVIAL(info) << "VBLAST decoding type : QRD column norm reordering AVX2" << endl;
#else
    cout << "VBLAST decoding type : QRD column norm reordering" << endl;
    BOOST_LOG_TRIVIAL(info) << "VBLAST decoding type : QRD column norm reordering" << endl;
#endif
#elif defined(VBLAST_SQRD)
    #if defined(__AVX2__) and defined(AVX2_PROCESSING)
        cout << "VBLAST decoding type : SQRD AVX2" << endl;
    BOOST_LOG_TRIVIAL(info) << "VBLAST decoding type : SQRD" << endl;
    #else
    cout << "VBLAST decoding type : SQRD" << endl;
    BOOST_LOG_TRIVIAL(info) << "VBLAST decoding type : SQRD" << endl;
    #endif
#elif defined(VBLAST_MF)
cout << "VBLAST decoding type : MF" << endl;
BOOST_LOG_TRIVIAL(info) << "VBLAST decoding type : MF" << endl;
#elif defined(VBLAST_QRD)
    cout << "VBLAST decoding type : QRD no reordering" << endl;
    BOOST_LOG_TRIVIAL(info) << "VBLAST decoding type : QRD no reordering"  << endl;
#endif
    init_fft_plans(fft_size);
    init_ifft_plans(fft_size);
    init_sync_pss_plans(fft_size, fft_size/SIZE_PSS_SSS_SIGNAL); /// Try a downsampling factor of 4

    /// Compute the CP lengths of each symbols within a subframe
    int cp_lengths[symbols_per_subframe];
    int cum_sum_cp_lengths[symbols_per_subframe];
    compute_cp_lengths( scs/1000, fft_size, 0, symbols_per_subframe, cp_lengths, cum_sum_cp_lengths);

    /// Set the antenna port value
    int antenna_port_value = 0;
#if NUM_LAYERS == 4
    antenna_port_value = 10;
#elif NUM_LAYERS == 2
    antenna_port_value = 2;
#elif NUM_LAYERS == 3
    antenna_port_value = 9;
#elif NUM_LAYERS == 1
    antenna_port_value = 0;
#endif

    // Vector containing the TX antenna ports used by each user. Configure only 1 user using the 2 antenna ports available
    vector<vector<size_t>> ports_tx_usrp = {antenna_port_value_rx_usrp_ports[antenna_port_value]};

    // Vector containing the RX antenna ports used by each user. Configure only 1 user using the 2 antenna ports available
    vector<vector<size_t>> ports_rx_usrp = {antenna_port_value_tx_usrp_ports[antenna_port_value]};

    int num_tx_dmrs_ports = antenna_ports_num_dmrs_ports[antenna_port_value];
    cout << "Number of layers : " << num_tx_dmrs_ports << endl;
    BOOST_LOG_TRIVIAL(trace) << "Number of layers : " << num_tx_dmrs_ports << endl;

    vector<int> ordered_tx_antenna_indexes(num_tx_dmrs_ports);
    for(int i = 0; i < ordered_tx_antenna_indexes.size(); i++) {
        ordered_tx_antenna_indexes[i] = i;
    }
    vector<int> temp_ordered_tx_antenna_indexes(num_tx_dmrs_ports);
    int nb_rx_ports = ports_rx_usrp[0].size();
    cout << "Number of recive ports : " << nb_rx_ports << endl;
    BOOST_LOG_TRIVIAL(trace) << "Number of recive ports : " << nb_rx_ports << endl;

    cout << "MIMO encoding type used : " << encoding_type << endl;
    BOOST_LOG_TRIVIAL(trace) << "MIMO encoding type" << encoding_type << endl;

    /**************** rescale output power **************/
    gain_tx -= 10 * log10(num_tx_dmrs_ports);
    /***************************************************/

    int dmrs_config_type = 1; // Fix the DMRS config type to 1 for the moment.

    /*********** Initialize all possible vectors and variables before tranmission, because it takes too long **********/
    /// Initialize the size of the OFDM grids to be transmitted on each DMRS port
    vector<vector<complex<float>>> grids[num_tx_dmrs_ports]; // OFDM grids to be sent on each TX port
    for(int tx_no = 0; tx_no < num_tx_dmrs_ports; tx_no++) {
        grids[tx_no].resize(symbols_per_subframe * nb_tx_subframes, vector<complex<float>>(fft_size, 0));
    }

    /// Initalize the size of the time domain grids to be sent
    vector<vector<complex<float>>> time_domain_grids(num_tx_dmrs_ports,
                                                     vector<complex<float>>(nb_tx_subframes * (cum_sum_cp_lengths[symbols_per_subframe - 1] + cp_lengths[symbols_per_subframe - 1] + fft_size)));


    //rx_estimation_semaphores = (sem_t *)malloc(nb_rx_ports * sizeof(sem_t)); /// TODO : delete pointer
    //rx_interpolation_semaphores = (sem_t *)malloc(nb_rx_ports * num_tx_dmrs_ports * sizeof(sem_t)); /// TODO :delete pointer
    //wait_estimation_and_interpolation_semaphores = (sem_t *)malloc(nb_rx_ports * sizeof(sem_t)); /// TODO : delete pointer

    /**
    for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
        sem_init(rx_estimation_semaphores + receiver, 0, 0);
        sem_init(wait_estimation_and_interpolation_semaphores + receiver, 0, 0);
        for(int transmitter = 0; transmitter < num_tx_dmrs_ports; transmitter++) {
            sem_init(rx_interpolation_semaphores + receiver * num_tx_dmrs_ports + transmitter, 0, 0);
        }
    } */

    //int slot_number_channel_estimation = 0;

    /// Contains the interpolated channel coefficients for one slot.
#if defined(VBLAST_SQRD) or defined(VBLAST_QRD_COL_NORM) or defined(VBLAST_SQRD_AVX2) or defined(VBLAST_QRD)
    vector<complex<float>> interp_coefficients_slot[MAX_TX_PORTS][MAX_RX_PORTS];
    for(int i = 0; i < MAX_RX_PORTS; i++) {
        for(int j = 0; j < MAX_TX_PORTS; j++) {
            interp_coefficients_slot[i][j] = vector<complex<float>>(pdsch_length * fft_size);
        }
    }

#else
    vector<complex<float>> interp_coefficients_slot[MAX_RX_PORTS][MAX_TX_PORTS];
    for(int i = 0; i < MAX_RX_PORTS; i++) {
        for(int j = 0; j < MAX_TX_PORTS; j++) {
            interp_coefficients_slot[i][j] = vector<complex<float>>(pdsch_length * fft_size);

        }
    }
#endif

    /********* Compute the DMRS positions within one slot ********************/
    int num_dmrs_symbols_per_slot = 0;
    int dmrs_sequence_size = fft_size/2;
    int * dmrs_symbols; // tab of dmrs symbols /// TODO : delete pointer
    int * dmrs_subcarriers; // tab of dmrs subcarriers /// TODO : delete pointer
    int dmrs_grid_positions[14*fft_size];
    memset(dmrs_grid_positions, 0, 14 * fft_size * sizeof(int));
    compute_dmrs_positions_type1A(dmrs_additional_positions,
                                  l0,
                                  pdsch_start,
                                  pdsch_length,
                                  num_dmrs_symbols_per_slot,
                                  num_tx_dmrs_ports,
                                  fft_size,
                                  nb_slots,
                                  dmrs_sequence_size,
                                  double_symbol,
                                  dmrs_symbols,
                                  dmrs_subcarriers,
                                  dmrs_grid_positions); // Use single symbol

    /// Depends on num_dmrs_symbols_per_slot
    vector<complex<float>> pilot_coefficients_slot(nb_rx_ports * num_tx_dmrs_ports * num_dmrs_symbols_per_slot * dmrs_sequence_size);
    float sto_phase_offsets[MAX_RX_PORTS][MAX_TX_PORTS][num_dmrs_symbols_per_slot];

    /**************************** Compute PDSCH positions within one slot **************************/
    /// Number of PDSCH RE within one slot
    int num_re_dmrs_per_slot = num_dmrs_symbols_per_slot * dmrs_sequence_size * antenna_port_num_cdm_groups[antenna_port_value];
    int num_pdsch_re_per_slot = pdsch_length * fft_size - num_re_dmrs_per_slot;
    /// Double the size to store symbols on even indexes and subcarriers on odd indexes.
    vector<int> pdsch_positions(2 * num_pdsch_re_per_slot);
    compute_pdsch_positions(pdsch_start,
                            pdsch_length,
                            fft_size,
                            dmrs_grid_positions,
                            pdsch_positions.data());

    BOOST_LOG_TRIVIAL(info) << "DMRS double symbol : " << double_symbol << endl;
    BOOST_LOG_TRIVIAL(info) << "DMRS start symbol : " << l0 << endl;
    BOOST_LOG_TRIVIAL(info) << "PDSCH length : " << pdsch_length << endl;
    BOOST_LOG_TRIVIAL(info) << "DMRS additionnal positions : " << dmrs_additional_positions << endl;
    BOOST_LOG_TRIVIAL(info) << "PDSCH start symbol : " << pdsch_start << endl;

#if defined(VBLAST_SQRD) or defined(VBLAST_QRD_COL_NORM) or defined(VBLAST_SQRD_AVX2) or defined(VBLAST_QRD)
    vector<complex<float>> pdsch_channel_coefficients[MAX_TX_PORTS][MAX_RX_PORTS];
    for(int transmitter = 0; transmitter < MAX_TX_PORTS; transmitter++) {
        for(int receiver = 0; receiver < MAX_RX_PORTS; receiver++) {
            pdsch_channel_coefficients[transmitter][receiver] =
                    vector<complex<float>> (num_pdsch_re_per_slot);
        }
    }

#else
    vector<complex<float>> pdsch_channel_coefficients[MAX_RX_PORTS][MAX_TX_PORTS];
    vector<float> pdsch_squared_norms[MAX_RX_PORTS][MAX_TX_PORTS];
    for(int receiver = 0; receiver < MAX_RX_PORTS; receiver++) {
        for(int transmitter = 0; transmitter < MAX_TX_PORTS; transmitter++) {
            pdsch_channel_coefficients[receiver][transmitter] =
                    vector<complex<float>> (num_pdsch_re_per_slot);
            pdsch_squared_norms[receiver][transmitter] = vector<float> (num_pdsch_re_per_slot);
        }
    }
#endif

#if defined(VBLAST_ZF_SEP)
    complex<float> hermitian_matrix_copy[num_pdsch_re_per_slot][MAX_TX_PORTS][MAX_TX_PORTS];
#endif

    /************** DMRS sequence generation *******************/
    /// Generate the DMRS sequences for the whole frame according to the antenna port number
    vector<complex<float>> dmrs_sequences(MAX_TX_PORTS * (nb_slots - 1) * num_dmrs_symbols_per_slot * dmrs_sequence_size);

    compute_dmrs_sequences_type1(dmrs_symbols,
                                 dmrs_sequences.data(),
                                 num_dmrs_symbols_per_slot,
                                 dmrs_sequence_size,
                                 nb_slots,
                                 double_symbol);

    /**
    ofstream dmrs_sequences_file("dmrs_sequences.txt");
    for(int transmitter = 0; transmitter < 4; transmitter++) {
        for(int slot = 0; slot < nb_slots - 1; slot++) {
            dmrs_sequences_file << "transmitter : " << transmitter << endl;
            dmrs_sequences_file << "slot number : " << slot << endl;
            for(int symbol = 0; symbol < num_dmrs_symbols_per_slot; symbol++) {
                dmrs_sequences_file << "symbol : " << dmrs_symbols[symbol] << endl;
                for(int sc = 0; sc < dmrs_sequence_size; sc++) {
                    dmrs_sequences_file << dmrs_sequences[
                            transmitter * (nb_slots - 1) * num_dmrs_symbols_per_slot * dmrs_sequence_size +
                            slot * num_dmrs_symbols_per_slot * dmrs_sequence_size +
                            symbol * dmrs_sequence_size
                            + sc] << endl;
                }
            }
        }
    } */

    /// Resize the buffers depending on the encoding type
    int buffer_size = 0;
    if(encoding_type == diversity) {
        buffer_size = num_pdsch_re_per_slot * (nb_slots - 1);
    } else if (encoding_type == vblast) {
        buffer_size = num_pdsch_re_per_slot * (nb_slots - 1) * num_tx_dmrs_ports;
    } else if (encoding_type == none) {
        buffer_size = num_pdsch_re_per_slot * (nb_slots - 1);
    }
    vector<complex<float>> sending_buffer(buffer_size);
    vector<complex<float>> equalized_symbols(buffer_size);
#if defined(VBLAST_ZF_SEP)
    vector<complex<float>> equalized_symbols_copy(buffer_size);
#endif

    vector<int> sending_buffer_symbol_indexes(buffer_size);
    vector<int> detected_symbol_indexes(buffer_size);

    /// Fill the sending buffer with random symbols
    int num_symbols = constellation_sizes[constellation_type]; // number of symbols in the constellation
    for(int symbol = 0; symbol < buffer_size; symbol ++) {
        sending_buffer_symbol_indexes[symbol] = std::experimental::randint(0, num_symbols - 1);
        sending_buffer[symbol] = constellations[constellation_type][sending_buffer_symbol_indexes[symbol]];
    }

    /// Prepare the grids to be sent
    /// Generate PSS sequence and place it on the first symbol of the frame
    /// The PSS occupies only 127 subcarriers
    int pss_sequence[SIZE_PSS_SSS_SIGNAL] {0};
    n_id_2 = 0; // Fix n_id_2 to 0, only 0 is tested here
    generate_pss_sequence(n_id_2, pss_sequence);

    /// Compute time domain PSS for correlation at the receiver
    vector<vector<complex<float>>> time_signal_pss(NUM_N_ID_2,
                                                   vector<complex<float>> (cp_lengths[pss_position] + fft_size));
    vector<vector<complex<float>>> time_signal_pss_downsampled(NUM_N_ID_2,
                                                               vector<complex<float>>((cp_lengths[pss_position] + fft_size) / (fft_size / SIZE_PSS_SSS_SIGNAL))); /// Try a downsampling factor of 4
    compute_time_domain_pss(time_signal_pss,
                            fft_size,
                            cp_lengths[pss_position],
                            false);
    compute_time_domain_pss(time_signal_pss_downsampled,
                            fft_size / (fft_size / SIZE_PSS_SSS_SIGNAL),
                            cp_lengths[pss_position] / (fft_size / SIZE_PSS_SSS_SIGNAL), /// Try a downsampling factor of 4
                            true);

    /// PSS RE mapping
    for(int pss_carrier = 0; pss_carrier < SIZE_PSS_SSS_SIGNAL; pss_carrier++) {
        for(int grid_no = 0; grid_no < num_tx_dmrs_ports; grid_no++) {
            grids[grid_no][0][fft_size/2 - 1 - (SIZE_PSS_SSS_SIGNAL-1)/2 + pss_carrier] = pss_sequence[pss_carrier];
        }
    }

    /************************ add SSS for CFO estimation ******************************************/
    vector<int> sss_sequence(SIZE_PSS_SSS_SIGNAL);
    generateSssSequence(n_id_1, n_id_2, sss_sequence);

    /// SSS RE mapping on symbol 2
    for(int sss_carrier = 0; sss_carrier < SIZE_PSS_SSS_SIGNAL; sss_carrier++) {
        for(int grid_no = 0; grid_no < num_tx_dmrs_ports; grid_no++) {
            grids[grid_no][2][fft_size/2 - 1 - (SIZE_PSS_SSS_SIGNAL-1)/2 + sss_carrier] =
                    sss_sequence[sss_carrier];
        }
    }

    /************************ add dummy bits for CFO estimation on PBCH ***************************/
    vector<complex<float>> simulated_pbch_data(2 * 240
                                               + 2 * 48);

    /// PBCH is QPSK modulated
    for(int i = 0; i < simulated_pbch_data.size(); i++) {
        simulated_pbch_data[i] = constellations[0][std::experimental::randint(0, 3)];
    }

    /// PBCH RE mapping
    for(int grid_no = 0; grid_no < num_tx_dmrs_ports; grid_no++) {
        for(int sc = 0; sc < 240; sc++) {
            grids[grid_no][1][fft_size/2 - 1 - (240-1)/2 + sc] =
                    simulated_pbch_data[sc];
        }
    }
    for(int grid_no = 0; grid_no < num_tx_dmrs_ports; grid_no++) {
        for(int sc = 0; sc < 48; sc++) {
            grids[grid_no][2][fft_size/2 - 1 - (240-1)/2 + sc] =
                    simulated_pbch_data[240 + sc];
        }
    }
    for(int grid_no = 0; grid_no < num_tx_dmrs_ports; grid_no++) {
        for(int sc = 192; sc < 239; sc++) {
            grids[grid_no][2][fft_size/2 - 1 - (240-1)/2 + sc] =
                    simulated_pbch_data[240 + 48 + sc];
        }
    }
    for(int grid_no = 0; grid_no < num_tx_dmrs_ports; grid_no++) {
        for(int sc = 0; sc < 240; sc++) {
            grids[grid_no][3][fft_size/2 - 1 - (240-1)/2 + sc] =
                    simulated_pbch_data[240 +
                                        2 * 48 +
                                        sc];
        }
    }

    /*********************** DMRS RE mapping ************************************/
    int symbol_re_dmrs = 0, sc_re_dmrs = 0;
    int port = 0;

    for(int dmrs_port_index = 0; dmrs_port_index < num_tx_dmrs_ports; dmrs_port_index++) {

        port = antenna_port_dmrs_ports[antenna_port_value][dmrs_port_index];

        for (int slot = 1; slot < nb_slots; slot++) {
            for (int symbol = 0; symbol < num_dmrs_symbols_per_slot; symbol++) {

                /// Symbol index of the DMRS RE relatively to the start of the entire frame
                symbol_re_dmrs = dmrs_symbols[symbol] + slot * 14;

                for (int sc = 0; sc < dmrs_sequence_size; sc++) {

                    /// Subcarrier index of the DMRS RE
                    sc_re_dmrs = dmrs_subcarriers[dmrs_port_index * dmrs_sequence_size + sc];

                    /// Place the sequence in the grid
                    grids[dmrs_port_index][symbol_re_dmrs][sc_re_dmrs] =
                            dmrs_sequences[port * (nb_slots - 1) * num_dmrs_symbols_per_slot * dmrs_sequence_size
                                           + (slot - 1) * num_dmrs_symbols_per_slot * dmrs_sequence_size +
                                           symbol * dmrs_sequence_size + sc];


                }
            }
        }
    }

    /// Prepare PDSCH
    int symbol = 0;
    int sc     = 0, next_sc = 0, buffer_count = 0;

    if(encoding_type == diversity) {
        int num_symbols_per_layer = buffer_size / num_tx_dmrs_ports;
        int num_symbols_per_precoded_layer = 0;
        vector<complex<float>> layers(num_tx_dmrs_ports * buffer_size);

        if(num_tx_dmrs_ports == 2) {
            num_symbols_per_precoded_layer = 2 * num_symbols_per_layer;
        } else if((num_symbols_per_layer % 4 == 0) and (num_tx_dmrs_ports == 4)) {
            num_symbols_per_precoded_layer = 4 * num_symbols_per_layer;
        } else if((num_symbols_per_layer % 4 != 0) and (num_tx_dmrs_ports == 4)) {
            num_symbols_per_precoded_layer = 4 * num_symbols_per_layer - 2;
        }

        vector<complex<float>> precoded_layers(num_tx_dmrs_ports * num_symbols_per_precoded_layer);

        /// Layer mapping and precoding
        transmit_diversity_layer_mapper(sending_buffer.data(),
                                        layers.data(),
                                        num_tx_dmrs_ports,
                                        buffer_size,
                                        num_symbols_per_layer);

        transmit_diversity_precoding(layers.data(),
                                     precoded_layers.data(),
                                     num_tx_dmrs_ports,
                                     num_symbols_per_layer,
                                     num_symbols_per_precoded_layer);

        //ofstream precoded_layers_file("precoded_layers.txt");
        /// RE mapping on PDSCH. Start at slot no. 1
        for(int slot = 1; slot < nb_slots; slot++) {
            for (int i = 0; i < num_pdsch_re_per_slot; i++) {

                /// Get symbol and subcarrier number in position
                symbol = pdsch_positions[2 * i] + slot * 14;
                sc = pdsch_positions[2 * i + 1];

                for (int layer = 0; layer < num_tx_dmrs_ports; layer++) {
                    grids[layer][symbol][sc] = precoded_layers[layer * num_symbols_per_precoded_layer + buffer_count];
                    //precoded_layers_file << precoded_layers[layer * num_symbols_per_precoded_layer + buffer_count] << endl;
                }

                buffer_count++;
            }
        }

    } else if (encoding_type == vblast) {
        int num_symbols_per_layer[num_tx_dmrs_ports];

        nr_num_symbols_per_layer(num_tx_dmrs_ports,
                                 buffer_size,
                                 0,
                                 num_symbols_per_layer);

        complex<float> layers[buffer_size];
        complex<float> * codewords[2] = { sending_buffer.data(), nullptr};

        /// Layer mapping. No precoding applied
        int num_codewords = 1;
        nr_layer_mapper(codewords,
                        layers,
                        nb_rx_ports,
                        num_codewords, // only 1 codeword
                        num_symbols_per_layer,
                        buffer_size);

        for(int slot = 1; slot < nb_slots; slot++) {
            for (int i = 0; i < num_pdsch_re_per_slot; i++) {
                symbol = pdsch_positions[2 * i] + slot * 14;
                sc = pdsch_positions[2*i + 1];

                /// RE mapping on PDSCH. Start at slot no. 1
                for(int layer = 0; layer < num_tx_dmrs_ports; layer++) {
                    grids[layer][symbol][sc] = layers[layer * num_symbols_per_layer[layer] + buffer_count];
                    //grids[layer][symbol][sc] = precoded_layers[layer * num_symbols_per_layer[layer] + buffer_count];
                }

                buffer_count++;

            }
        }

    } else if (encoding_type == none) {

        /// Start at slot no. 1, because there is no PDSCH on slot no. 0
        for(int slot = 1; slot < nb_slots; slot++) {
            for (int i = 0; i < num_pdsch_re_per_slot; i++) {
                symbol = pdsch_positions[2*i] + slot * 14;
                sc     = pdsch_positions[2*i + 1];

                grids[0][symbol][sc] = sending_buffer[buffer_count];
                buffer_count++;
            }
        }
    }

    /// Compute the IFFT to get the time domain grids to be sent on each URSP port
    for(int grid_no = 0; grid_no < num_tx_dmrs_ports; grid_no++) {
        compute_ifft(grids[grid_no],
                     time_domain_grids[grid_no],
                     cp_lengths,
                     cum_sum_cp_lengths, scaling_factor,
                     fft_size, symbols_per_subframe,
                     nb_tx_subframes,
                     scs);
    }

    cout << "grid samples : " << time_domain_grids[0].size() << endl;

    /// initialize pointer to the transmitting USRPB210 object
    string device_args = "type=x300, master_clock_rate=184.32e6, recv_frame_size=8000, send_frame_size=8000, num_recv_frames=256, num_send_frames=256, dboard_clock_rate=50e6";
    shared_ptr<Usrp> usrp_tx_ptr = make_shared<UsrpX300>(device_args,
                                                         "A:0 B:0",
                                                         "A:0 B:0",
                                                         "external",
                                                         bandwidth,
                                                         sampling_rate,
                                                         center_frequency,
                                                         gain_tx,
                                                         gain_rx,
                                                         184.32e6,
                                                         "TX/RX",
                                                         "RX2",
                                                         ports_tx_usrp,
                                                         ports_rx_usrp);

    cout << "Number of sent subframes :" << nb_tx_subframes << endl;

    /// Export the data
    export1dVector("non_encoded.txt", sending_buffer.data(), buffer_size);
    export1dVector("sending_buffer_symbol_indexes.txt", sending_buffer_symbol_indexes.data(), buffer_size);

    /// Export the grids to be plotted
    for(int grid_no = 0; grid_no < num_tx_dmrs_ports; grid_no++) {
        /// Export signal to be sent on TX1 (time domain signal)
        export1dVector("tx" + to_string(grid_no + 1) + ".txt", time_domain_grids[grid_no]);
        exportGrid("tx"+ to_string(grid_no + 1) + "_grid.txt", grids[grid_no]);
    }


    /// Prepare buffers to retrieve the contents of the receiving buffers
    vector<vector<vector<complex<float>>>> all_rx_buffers;
    vector<vector<complex<float>>> receiveBuffers;

    /// Initialize the sending and receiving buffers
    usrp_tx_ptr->init_tx_buffers({time_domain_grids});
    usrp_tx_ptr->init_rx_buffers(2 * time_domain_grids[0].size());

    /// Configure new TX and RX streamers
    usrp_tx_ptr->config_tx_streams();
    usrp_tx_ptr->config_rx_streams();

    /// Start sending and receiving to/from the usrp device
    usrp_tx_ptr->start_sending();
    sleep(2);
    usrp_tx_ptr->start_receiving();

    /// Stop the main thread on SIGINT
    while(!stop_signal_called_main) {
        sleep(5);
        stop_signal_called_main = 1;
    }

    ofstream output_file_estimated_channel_coefs("estimated_channel_coefs_main.txt");
    ofstream output_file_interp_channel_coefs("interp_channel_coefs_main.txt");

    /// Stop the sending and receiving threads when SIGINT called
    cout << "stopping the tx and rx threads" << endl;
    usrp_tx_ptr->stop_tx_rx_threads();

    cout << "threads stopped" << endl;

    if(usrp_tx_ptr->get_nothing_received()) {

        cout << "No signal received. Ending the main thread ..." << endl;

        return EXIT_SUCCESS;
    }

    /// Retrieve the contents of the receiving buffers
    usrp_tx_ptr->get_receiving_buffer(all_rx_buffers);
    receiveBuffers = all_rx_buffers[0]; // Get the data for the only user configured

    /// Export the received signals, then remove CP and compute the FFT
    vector<vector<vector<complex<float>>>> receivedGrids(nb_rx_ports,
                                                         vector<vector<complex<float>>>(symbols_per_subframe*nb_rx_subframes,
                                                                 vector<complex<float>>(fft_size, 0)));

    for(int i = 0; i < receiveBuffers.size(); i++) {

        cout << "\n Exporting received signal on channel "+to_string(i)+"... " << std::endl;
        BOOST_LOG_TRIVIAL(info) << "\n Exporting received signal on channel "+to_string(i)+"... " << endl;
        export1dVector("rx"+to_string(i+1)+".txt", receiveBuffers[i]);
    }

    /// Decode the grids
    vector<complex<float>> synchronized_signals[nb_rx_ports];
    for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
        synchronized_signals[receiver].resize((cum_sum_cp_lengths[symbols_per_subframe - 1]
                                               + fft_size + cp_lengths[symbols_per_subframe - 1]) * nb_tx_subframes);
    }

    int num_samples_per_received_signal = receiveBuffers[0].size();

    vector<vector<complex<float>>> synchronized_grids[nb_rx_ports];

    for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
        synchronized_grids[receiver].resize(symbols_per_subframe * nb_tx_subframes,
                                            vector<complex<float>>(fft_size, 0));
    }

    int pss_sample_index = 0;

    /************************************ Perform multiple simulations ***********************************/
    asm volatile ("CPUID\n\t"
                  "RDTSC\n\t"
                  "mov %%edx, %0\n\t"
                  "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
            "%rax", "%rbx", "%rcx", "%rdx");

    /// Perform synchronization on receive antenna 0
    int synchronization_index = 0;
    synchronize_slots(receiveBuffers[0],
                      num_samples_per_received_signal,
                      nb_tx_subframes,
                      fft_size,
                      symbols_per_subframe,
                      cp_lengths[pss_position],
                      cum_sum_cp_lengths,
                      cp_lengths,
                      scs,
                      n_id_2,
                      pss_sample_index,
                      synchronization_index,
                      fft_size / SIZE_PSS_SSS_SIGNAL,
                      true, /// Consider N_ID_2 is already known
                      time_signal_pss,
                      time_signal_pss_downsampled);

    asm volatile("RDTSCP\n\t"
                 "mov %%edx, %0\n\t"
                 "mov %%eax, %1\n\t"
                 "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
            "%rax", "%rbx", "%rcx", "%rdx");

    t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
    t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

    t_cumul += t2 - t1;

    cout << "Synchronization by cross-correlating known PSS [µs]: " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

    asm volatile ("CPUID\n\t"
                  "RDTSC\n\t"
                  "mov %%edx, %0\n\t"
                  "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
            "%rax", "%rbx", "%rcx", "%rdx");

    /// Compute the FFT for all the receive ports
    for (int grid_no = 0; grid_no < nb_rx_ports; grid_no++) {
        compute_fft(receiveBuffers[grid_no].data() + pss_sample_index,
                    synchronized_grids[grid_no],
                    nb_tx_subframes,
                    fft_size,
                    symbols_per_subframe,
                    scs, cp_lengths,
                    cum_sum_cp_lengths);
    }

    asm volatile("RDTSCP\n\t"
                 "mov %%edx, %0\n\t"
                 "mov %%eax, %1\n\t"
                 "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
            "%rax", "%rbx", "%rcx", "%rdx");

    t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
    t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

    t_cumul += t2 - t1;

    cout << "Computing FFT based on determined PSS start index [µs]: " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

    int subframe_size = cum_sum_cp_lengths[symbols_per_subframe - 1] + fft_size + cp_lengths[symbols_per_subframe - 1];
    for(int grid_no = 0; grid_no < nb_rx_ports; grid_no++) {
        /** Export the synchronized signals in time domain
         *  and the synchronized grids
         */
        std::copy(receiveBuffers[grid_no].begin() + pss_sample_index,
                  receiveBuffers[grid_no].begin() + pss_sample_index + subframe_size * nb_tx_subframes,
                  synchronized_signals[grid_no].begin());

        cout << "\n Exporting synchronized signal on channel " + to_string(grid_no) + "... " << std::endl;
        BOOST_LOG_TRIVIAL(info) << "\n Exporting synchronized signal on channel " + to_string(grid_no) + "... " << endl;
        export1dVector("sync_rx" + to_string(grid_no+1) + ".txt", synchronized_signals[grid_no]);
        exportGrid("sync_rx" + to_string(grid_no+1) + "_grid.txt", synchronized_grids[grid_no]);
    }

    /******************* Frequency offset correction ****************************/

    asm volatile ("CPUID\n\t"
                  "RDTSC\n\t"
                  "mov %%edx, %0\n\t"
                  "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
            "%rax", "%rbx", "%rcx", "%rdx");

    int start_symbol_frequency_offset = 0;
    float frequency_offset = 0;
    start_symbol_frequency_offset = 0; /// Start on the PSS symbol
    for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
        /// Compute an average of the frequency offset over the first 4 symbols
        /// of the first PDSCH allocations, starting from symbol 1
        computeFineFrequencyOffset(
                &synchronized_signals[receiver][cum_sum_cp_lengths[start_symbol_frequency_offset]],
                cp_lengths[start_symbol_frequency_offset] + fft_size,
                fft_size,
                cp_lengths[start_symbol_frequency_offset],
                scs, frequency_offset,
                4);
        cout << "frequency offset : " << frequency_offset << endl;

        /// Apply correction on the whole received frame
        transposeSignal(&synchronized_signals[receiver], frequency_offset, (double) sampling_rate);
    }

    asm volatile("RDTSCP\n\t"
                 "mov %%edx, %0\n\t"
                 "mov %%eax, %1\n\t"
                 "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
            "%rax", "%rbx", "%rcx", "%rdx");

    t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
    t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

    t_cumul += t2 - t1;

    cout << "CFO estimation & correction on each channel [µs]: " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

    /******************** Frequency offset correction end ***********************/

    int tx_dmrs_port = 0, path = 0;

    vector<vector<complex<float>>> pilot_coefficients =
            vector<vector<complex<float>>>(nb_slots - 1 ,
                    vector<complex<float>>(nb_rx_ports * num_tx_dmrs_ports * num_dmrs_symbols_per_slot * dmrs_sequence_size, 0));

    vector<vector<complex<float>>> pilot_coefficients_before_sto =
            vector<vector<complex<float>>>(nb_slots - 1 ,
                    vector<complex<float>>(nb_rx_ports * num_tx_dmrs_ports * num_dmrs_symbols_per_slot * dmrs_sequence_size, 0));

#if defined(VBLAST_SQRD) or defined(VBLAST_QRD_COL_NORM) or defined(VBLAST_SQRD_AVX2) or defined(VBLAST_QRD)
    vector<vector<complex<float>>> interp_coefficients[MAX_TX_PORTS][MAX_RX_PORTS];
    for(int transmitter = 0; transmitter < MAX_TX_PORTS; transmitter++) {
        for(int receiver = 0; receiver < MAX_RX_PORTS; receiver++) {
            interp_coefficients[transmitter][receiver] = vector<vector<complex<float>>>(nb_slots - 1,
                    vector<complex<float>>(pdsch_length * fft_size, 0));
        }
    }
#else
    vector<vector<complex<float>>> interp_coefficients[MAX_RX_PORTS][MAX_TX_PORTS];
    for(int receiver = 0; receiver < MAX_RX_PORTS; receiver++) {
        for(int transmitter = 0; transmitter < MAX_TX_PORTS; transmitter++) {
            interp_coefficients[receiver][transmitter] = vector<vector<complex<float>>>(nb_slots - 1,
                    vector<complex<float>>(pdsch_length * fft_size, 0));
        }
    }
#endif

    /* //Create separate threads for channel estimation on DMRS for each RX antenna
    int nb_paths = num_tx_dmrs_ports * nb_rx_ports;
    thread rx_threads[nb_rx_ports];
    thread interp_threads[nb_paths];
    int sem_value = 0;
     */

    vector<vector<complex<float>>> pdsch_samples(MAX_RX_PORTS,
                                                 vector<complex<float>> (num_pdsch_re_per_slot));

    vector<vector<complex<float>>> dmrs_samples[nb_rx_ports];
    for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
        dmrs_samples[receiver].resize(antenna_ports_num_cdm_groups_without_data[antenna_port_value], vector<complex<float>>(num_dmrs_symbols_per_slot * dmrs_sequence_size));
    }

    /**
    vector<vector<float>> dmrs_samples_real[nb_rx_ports];
    for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
        dmrs_samples_real[receiver].resize(antenna_ports_num_cdm_groups_without_data[antenna_port_value], vector<float>(num_dmrs_symbols_per_slot * dmrs_sequence_size));
    }
    vector<vector<float>> dmrs_samples_imag[nb_rx_ports];
    for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
        dmrs_samples_imag[receiver].resize(antenna_ports_num_cdm_groups_without_data[antenna_port_value], vector<float>(num_dmrs_symbols_per_slot * dmrs_sequence_size));
    } */

    /**
    for (int receiver = 0; receiver < nb_rx_ports; receiver++) {
        *(rx_threads + receiver) = std::thread([
                                                       &arg0 = tab_tx_dmrs_ports,
                                                       arg1 = tab_cdm_group_sizes,
                                                       arg2 = dmrs_symbols,
                                                       arg3 = dmrs_subcarriers,
                                                       arg4 = dmrs_samples[receiver].data(), //synchronized_grids[receiver],
                                                       &arg5 = dmrs_sequences,
                                                       arg6 = pilot_coefficients_slot +
                                                              receiver * num_tx_dmrs_ports * num_dmrs_symbols_per_slot *
                                                              dmrs_sequence_size,
                                                       &arg7 = double_symbol,
                                                       &arg8 = dmrs_sequence_size,
                                                       arg9 = receiver,
                                                       &arg10 = num_dmrs_symbols_per_slot,
                                                       arg11 = num_tx_dmrs_ports,
                                                       &arg12 = nb_slots,
                                                       &arg13 = slot_number_channel_estimation,
                                                       arg14 = interp_coefficients_slot
                                                               + receiver * num_tx_dmrs_ports * pdsch_length * fft_size,
                                                       &arg15 = pdsch_length,
                                                       &arg16 = fft_size,
                                                       &arg17 = tx_dmrs_ports_cdm_groups,
                                                       &arg18 = pdsch_start,
                                                       &arg20 = nb_rx_ports,
                                                       sem_offset = receiver * num_tx_dmrs_ports]() {

            //std::chrono::steady_clock::time_point  t1, t2;

            while(not stop_signal_called_channel_estimation) {

                sem_wait(rx_estimation_semaphores + arg9);

                //cout << "got semaphore for estimation" << endl;

                if (stop_signal_called_channel_estimation) {
                    break;
                }

                //t1 = std::chrono::steady_clock::now();
                estimate_pilots_cdm_groups_one_rx(arg0,
                                                  arg1,
                                                  arg2,
                                                  arg3,
                                                  arg4,
                                                  arg5,
                                                  arg6,
                                                  arg7,
                                                  arg8,
                                                  arg13,
                                                  arg10,
                                                  arg11,
                                                  arg12 - 1,
                                                  arg9);

                //t2 = std::chrono::steady_clock::now();
                //cout << "duration of estimation within thread : " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;

                //sem_post(wait_estimation_and_interpolation_semaphores + arg9);
                //sem_init(rx_interpolation_semaphores + arg9, 0, 4);
                for(int transmitter = 0; transmitter < arg11; transmitter++) {
                    sem_post(rx_interpolation_semaphores + sem_offset + transmitter);
                    //cout << "val estimation : " << arg9 * arg11 + transmitter << endl;
                }
                //cout << "estimation done " << endl;
            }
        });
    } */

    //int count_interpolated_grids[nb_rx_ports];

    /// Create separate threads for interpolation of channel coefficients on each TX-RX path
    /**
    for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
        count_interpolated_grids[receiver] = 0;

        for (int transmitter = 0; transmitter < num_tx_dmrs_ports; transmitter++) {

            //interp_coefficients_grid[receiver * num_tx_dmrs_ports + transmitter].resize(
            //        pdsch_length, vector<complex<float>>(fft_size));

            *(interp_threads + receiver * num_tx_dmrs_ports + transmitter) = thread(
                    [ arg0 = interp_coefficients_slot + receiver * num_tx_dmrs_ports * pdsch_length * fft_size
                             + transmitter * pdsch_length * fft_size,
                            //&arg0 = interp_coefficients_grid[receiver * num_tx_dmrs_ports + transmitter],
                            arg1 = pilot_coefficients_slot
                                   + receiver * num_tx_dmrs_ports * num_dmrs_symbols_per_slot * dmrs_sequence_size
                                   + transmitter * num_dmrs_symbols_per_slot * dmrs_sequence_size,
                            &arg2 = tx_dmrs_ports_cdm_groups[transmitter],
                            arg3 = receiver,
                            arg4 = dmrs_symbols,
                            &arg5 = pdsch_start,
                            &arg6 = num_dmrs_symbols_per_slot,
                            &arg7 = dmrs_sequence_size,
                            &arg8 = fft_size,
                            &arg9 = pdsch_length,
                            arg10 = num_tx_dmrs_ports,
                            &arg11 = nb_rx_ports,
                            &arg12 = count_interpolated_grids[receiver],
                            &arg13 = cdm_group_sizes[antenna_port_value][transmitter],
                            sem_offset = receiver * num_tx_dmrs_ports + transmitter
                    ]() {

                        //std::chrono::steady_clock::time_point  t1, t2;

                        //int sem_value = 0;

                        while(not stop_signal_called_channel_estimation) {

                            //cout << "val : " << arg3 * arg10 + arg14 << endl;
                            sem_wait(rx_interpolation_semaphores + sem_offset);
                            //sem_getvalue(rx_interpolation_semaphores + arg3 * arg10 + arg14, &sem_value);
                            //printf("sem value : %d", sem_value);

                            //cout << "got semaphore for interpolation" << endl;

                            if (stop_signal_called_channel_estimation) {
                                break;
                            }

                            //t1 = std::chrono::steady_clock::now();
                            interpolate_coefs(arg0,
                                              arg1,
                                              arg2,
                                              arg13,
                                              arg3,
                                              arg4,
                                              arg5,
                                              arg6,
                                              arg7,
                                              arg8,
                                              arg9,
                                              arg10,
                                              arg11);
                            //t2 = std::chrono::steady_clock::now();
                            //cout << "duration of interpolation within thread (one thread per path) : "
                            //<< std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;

                            arg12++;
                            if(arg12 == arg10) {
                                sem_post(wait_estimation_and_interpolation_semaphores + arg3);
                                //cout << "posted interp semaphore for receiver : " << arg3 << endl;
                            }
                        }

                    });
        }
    } */

    /// For each slot within the received frame
    for(int slot = 1; slot < nb_slots; slot++) {

        cout << "\n######## SLOT " << slot << " ########" << endl;

#if defined(CLOCK_TYPE_CHRONO)
        t3 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t3);
#elif defined(CLOCK_TYPE_ASM)
        t_cumul = 0;
        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high3), "=r" (cycles_low3)::
                "%rax", "%rbx", "%rcx", "%rdx");
#elif defined(CLOCK_TYPE_CLOCK)
        t3 = clock();
        #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
                clock_gettime(CLOCK_MONOTONIC, &t3);
        #elif defined(CLOCK_TYPE_RUSAGE)
                getrusage(RUSAGE_SELF, &usage);
                t3_utime = usage.ru_utime;
#endif
        /// Perform channel estimation
        /// Estimate the pilots on each DMRS port
#if defined(CLOCK_TYPE_CHRONO)
        t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_ASM)
        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#elif defined(CLOCK_TYPE_CLOCK)
        t1 = clock();
        #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
                clock_gettime(CLOCK_MONOTONIC, &t1);
        #elif defined(CLOCK_TYPE_RUSAGE)
                getrusage(RUSAGE_SELF, &usage);
                t1_utime = usage.ru_utime;
#endif
        /// Extract PDSCH and DMRS samples
        for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
            get_pdsch_and_dmrs_samples(synchronized_grids[receiver],
                                       slot,
                                       dmrs_symbols,
                                       dmrs_sequence_size,
                                       num_dmrs_symbols_per_slot,
                                       antenna_ports_num_cdm_groups_without_data[antenna_port_value],
                                       pdsch_length,
                                       fft_size,
                                       pdsch_start,
                                       pdsch_samples[receiver],
                                       dmrs_samples[receiver].data());
        }

#if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();

                cout << "Getting PDSCH and DMRS samples (steady_clock) [µs] : "
                     << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;
                BOOST_LOG_TRIVIAL(trace) << "Getting PDSCH and DMRS samples (steady_clock) [µs]  : "
                                         << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
                cout << "Getting PDSCH and DMRS samples (clock_gettime) [µs] : "
                     << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
                BOOST_LOG_TRIVIAL(trace) << "Getting PDSCH and DMRS samples (clock_gettime) [µs]  : "
                                         << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
#elif defined(CLOCK_TYPE_ASM)
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

        t_cumul += t2 - t1;

        cout << "Getting PDSCH and DMRS samples (RDTSC/RDTSCP) [µs]: " << (t2 - t1)/TSC_FREQ * 1e6 << endl;
        BOOST_LOG_TRIVIAL(trace) << "Getting PDSCH and DMRS samples (RDTSC/RDTSCP) [µs]: " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

#elif defined(CLOCK_TYPE_CLOCK)
        t2 = clock();
                cout << "Getting PDSCH and DMRS samples (clock()) [µs]: " << (t2 - t1) * 1e6 / CLOCKS_PER_SEC << endl;
                BOOST_LOG_TRIVIAL(trace) << "Getting PDSCH and DMRS samples (clock()) [µs]: " << (t2 - t1) * 1e6 / CLOCKS_PER_SEC << endl;
        #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
                clock_gettime(CLOCK_MONOTONIC, &t2);
                cout << "Getting PDSCH and DMRS samples (clock_gettime) [µs] : "
                     << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
                BOOST_LOG_TRIVIAL(trace) << "Getting PDSCH and DMRS samples (clock_gettime) [µs]  : "
                                         << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
        #elif defined(CLOCK_TYPE_RUSAGE)
                getrusage(RUSAGE_SELF, &usage);
                t2_utime = usage.ru_utime;
                cout << "Getting PDSCH and DMRS samples (clock_gettime) [µs] : "
                     << t2_utime.tv_usec - t1_utime.tv_usec << endl;
                BOOST_LOG_TRIVIAL(trace) << "Getting PDSCH and DMRS samples (clock_gettime) [µs]  : "
                                         << t2_utime.tv_usec - t1_utime.tv_usec << endl;
#endif

#if defined(CLOCK_TYPE_CHRONO)
        t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_ASM)
        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#elif defined(CLOCKS_TYPE_CLOCK)
        t1 = clock();
        #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
                clock_gettime(CLOCK_MONOTONIC, &t1);
        #elif defined(CLOCK_TYPE_RUSAGE)
                getrusage(RUSAGE_SELF, &usage);
                t1_utime = usage.ru_utime;
#endif

#if defined(__AVX2__) and defined(AVX2_PROCESSING)
        //#pragma omp parallel for
        for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
            estimate_pilots_avx(antenna_port_dmrs_ports[antenna_port_value],
                                cdm_group_sizes[antenna_port_value],
                                dmrs_symbols,
                                dmrs_samples[receiver].data(), //synchronized_grids[receiver],
                                dmrs_sequences.data(),
                                pilot_coefficients_slot.data() + receiver * num_tx_dmrs_ports * num_dmrs_symbols_per_slot * dmrs_sequence_size,
                                double_symbol,
                                dmrs_sequence_size,
                                slot,
                                num_dmrs_symbols_per_slot,
                                num_tx_dmrs_ports,
                                nb_slots - 1,
                                antenna_port_cdm_groups[antenna_port_value]);
        }
#else
        //#pragma omp parallel for
        for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
            estimate_pilots_cdm_groups_one_rx(antenna_port_dmrs_ports[antenna_port_value],
                                              cdm_group_sizes[antenna_port_value],
                                              dmrs_symbols,
                                              dmrs_samples[receiver].data(), //synchronized_grids[receiver],
                                              dmrs_sequences.data(),
                                              pilot_coefficients_slot.data() + receiver * num_tx_dmrs_ports * num_dmrs_symbols_per_slot * dmrs_sequence_size,
                                              double_symbol,
                                              dmrs_sequence_size,
                                              slot,
                                              num_dmrs_symbols_per_slot,
                                              num_tx_dmrs_ports,
                                              nb_slots - 1,
                                              receiver,
                                              antenna_port_cdm_groups[antenna_port_value]);
        }
#endif

#if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();

                cout << "\n Channel estimation on pilots (steady_clock) [µs] : "
                     << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;
                BOOST_LOG_TRIVIAL(trace) << "\n Channel estimation on pilots (steady_clock) [µs] : "
                                         << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
                cout << "\nChannel estimation on pilots (clock_gettime) [µs] : " << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
                BOOST_LOG_TRIVIAL(trace) << "\nChannel estimation on pilots (clock_gettime) [µs] : " << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
#elif defined(CLOCK_TYPE_ASM)
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

        t_cumul += t2 - t1;

        cout << "Channel estimation on pilots (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;
        BOOST_LOG_TRIVIAL(trace) << "Channel estimation on pilots (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;
#elif defined(CLOCK_TYPE_CLOCK)
        t2 = clock();
                cout << "Channel estimation on pilots (clock()) [µs] : "
                     << (t2 - t1) * 1e6 / CLOCKS_PER_SEC << endl;
                BOOST_LOG_TRIVIAL(trace) << "Channel estimation on pilots (clock()) [µs] : "
                     << (t2 - t1) * 1e6 / CLOCKS_PER_SEC << endl;
        #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
                clock_gettime(CLOCK_MONOTONIC, &t2);
                cout << "Channel estimation on pilots (clock_gettime) [µs] : " << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
                BOOST_LOG_TRIVIAL(trace) << "Channel estimation on pilots (clock_gettime) [µs] : " << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
        #elif defined(CLOCK_TYPE_RUSAGE)
                getrusage(RUSAGE_SELF, &usage);
                t2_utime = usage.ru_utime;
                cout << "Channel estimation on pilots (clock_gettime) [µs] : " << t2_utime.tv_usec - t1_utime.tv_usec << endl;
                BOOST_LOG_TRIVIAL(trace) << "Channel estimation on pilots (clock_gettime) [µs] : " << t2_utime.tv_usec - t1_utime.tv_usec << endl;567
#endif

#if defined(CLOCK_TYPE_CHRONO)
        t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_ASM)
        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#elif defined(CLOCK_TYPE_CLOCK)
        t1 = clock();
        #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
                clock_gettime(CLOCK_MONOTONIC, &t1);
        #elif defined(CLOCK_TYPE_RUSAGE)
                getrusage(RUSAGE_SELF, &usage);
                t1_utime = usage.ru_utime;
#endif

        /**
        slot_number_channel_estimation = slot; /// Update value for channel estimation
        for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
            sem_post(rx_estimation_semaphores + receiver);
        }

        //cout << "started estimation" << endl;

        /// Wait for interpolation to be done
        for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
            sem_wait(wait_estimation_and_interpolation_semaphores + receiver);
            count_interpolated_grids[receiver] = 0;
        } */

#if defined(VBLAST_SQRD) or defined(VBLAST_QRD_COL_NORM) or defined(VBLAST_SQRD_AVX2) or defined(VBLAST_QRD)
        #if defined(__AVX2__) and defined(AVX2_PROCESSING)
        for(int transmitter = 0; transmitter < num_tx_dmrs_ports; transmitter++) {
            for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
                interpolate_coefs_avx(interp_coefficients_slot[transmitter][receiver].data(),
                                      pilot_coefficients_slot.data()
                                      + receiver * num_tx_dmrs_ports * num_dmrs_symbols_per_slot * dmrs_sequence_size
                                      + transmitter * num_dmrs_symbols_per_slot * dmrs_sequence_size,
                                      antenna_port_cdm_groups[antenna_port_value][transmitter],
                                      cdm_group_sizes[antenna_port_value][transmitter],
                                      receiver,
                                      dmrs_symbols,
                                      pdsch_start,
                                      num_dmrs_symbols_per_slot,
                                      dmrs_sequence_size,
                                      fft_size,
                                      pdsch_length,
                                      num_tx_dmrs_ports,
                                      nb_rx_ports);
            }
        }
        #else
        for(int transmitter = 0; transmitter < num_tx_dmrs_ports; transmitter++) {
            for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
                interpolate_coefs(interp_coefficients_slot[transmitter][receiver].data(),
                                      pilot_coefficients_slot.data()
                                      + receiver * num_tx_dmrs_ports * num_dmrs_symbols_per_slot * dmrs_sequence_size
                                      + transmitter * num_dmrs_symbols_per_slot * dmrs_sequence_size,
                                      antenna_port_cdm_groups[antenna_port_value][transmitter],
                                      cdm_group_sizes[antenna_port_value][transmitter],
                                      receiver,
                                      dmrs_symbols,
                                      pdsch_start,
                                      num_dmrs_symbols_per_slot,
                                      dmrs_sequence_size,
                                      fft_size,
                                      pdsch_length,
                                      num_tx_dmrs_ports,
                                      nb_rx_ports);
            }
        }

        #endif
#else
#if defined(__AVX2__) and defined(AVX2_PROCESSING)
        //#pragma omp parallel for);
        for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
            for(int transmitter = 0; transmitter < num_tx_dmrs_ports; transmitter++) {
                interpolate_coefs_avx(interp_coefficients_slot[receiver][transmitter].data(),
                                      pilot_coefficients_slot.data()
                                      + receiver * num_tx_dmrs_ports * num_dmrs_symbols_per_slot * dmrs_sequence_size
                                      + transmitter * num_dmrs_symbols_per_slot * dmrs_sequence_size,
                                      antenna_port_cdm_groups[antenna_port_value][transmitter],
                                      cdm_group_sizes[antenna_port_value][transmitter],
                                      receiver,
                                      dmrs_symbols,
                                      pdsch_start,
                                      num_dmrs_symbols_per_slot,
                                      dmrs_sequence_size,
                                      fft_size,
                                      pdsch_length,
                                      num_tx_dmrs_ports,
                                      nb_rx_ports);

            }
        }

#else

        for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
            for(int transmitter = 0; transmitter < num_tx_dmrs_ports; transmitter++) {
                interpolate_coefs(interp_coefficients_slot[receiver][transmitter].data(),
                                      pilot_coefficients_slot.data()
                                      + receiver * num_tx_dmrs_ports * num_dmrs_symbols_per_slot * dmrs_sequence_size
                                      + transmitter * num_dmrs_symbols_per_slot * dmrs_sequence_size,
                                      antenna_port_cdm_groups[antenna_port_value][transmitter],
                                      cdm_group_sizes[antenna_port_value][transmitter],
                                      receiver,
                                      dmrs_symbols,
                                      pdsch_start,
                                      num_dmrs_symbols_per_slot,
                                      dmrs_sequence_size,
                                      fft_size,
                                      pdsch_length,
                                      num_tx_dmrs_ports,
                                      nb_rx_ports);
            }
        }
#endif
#endif

#if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();
                cout << "Channel coefficients interpolation (steady_clock) [µs] : "
                << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;
                BOOST_LOG_TRIVIAL(trace) << "Channel coefficients interpolation (steady_clock) [µs] : "
                << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
                cout << "Channel coefficients interpolation (clock_gettime) [µs] : " << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
                BOOST_LOG_TRIVIAL(trace) << "Channel coefficients interpolation (clock_gettime) [µs] : " << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
#elif defined(CLOCK_TYPE_ASM)
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

        t_cumul += t2 - t1;

        cout << "Channel coefficients interpolation (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;
        BOOST_LOG_TRIVIAL(trace) << "Channel coefficients interpolation (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;
#elif defined(CLOCK_TYPE_CLOCK)
        t2 = clock();
                cout << "Channel coefficients interpolation (clock()) [µs] : "
                     << (t2 - t1) * 1e6 / CLOCKS_PER_SEC << endl;
                BOOST_LOG_TRIVIAL(trace) << "Channel coefficients interpolation (clock()) [µs] : "
                     << (t2 - t1) * 1e6 / CLOCKS_PER_SEC << endl;
        #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
                clock_gettime(CLOCK_MONOTONIC, &t2);
                cout << "Channel coefficients interpolation (clock_gettime) [µs] : " << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
                BOOST_LOG_TRIVIAL(trace) << "Channel coefficients interpolation (clock_gettime) [µs] : " << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
        #elif defined(CLOCK_TYPE_RUSAGE)
                getrusage(RUSAGE_SELF, &usage);
                t2_utime = usage.ru_utime;
                cout << "Channel coefficients interpolation (clock_gettime) [µs] : " << t2_utime.tv_usec - t1_utime.tv_usec << endl;
                BOOST_LOG_TRIVIAL(trace) << "Channel coefficients interpolation (clock_gettime) [µs] : " << t2_utime.tv_usec - t1_utime.tv_usec << endl;
#endif

#if defined(CLOCK_TYPE_CHRONO)
        t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_ASM)
        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#elif defined(CLOCK_TYPE_CLOCK)
        t1 = clock();
        #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
                clock_gettime(CLOCK_MONOTONIC, &t1);
        #elif defined(CLOCK_TYPE_RUSAGE)
                getrusage(RUSAGE_SELF, &usage);
                t1_utime = usage.ru_utime;
#endif

        memcpy(pilot_coefficients[slot - 1].data(), pilot_coefficients_slot.data(), nb_rx_ports * num_tx_dmrs_ports * num_dmrs_symbols_per_slot * dmrs_sequence_size * sizeof(complex<float>));

#if defined(VBLAST_SQRD) or defined(VBLAST_QRD_COL_NORM) or defined(VBLAST_SQRD_AVX2) or defined(VBLAST_QRD)
        for(int transmitter = 0; transmitter < num_tx_dmrs_ports; transmitter++) {
                    for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
                        memcpy(interp_coefficients[transmitter][receiver][slot - 1].data(), interp_coefficients_slot[transmitter][receiver].data(), pdsch_length * fft_size * sizeof(complex<float>));
                    }
                }
#else
        for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
            for(int transmitter = 0; transmitter < num_tx_dmrs_ports; transmitter++) {
                memcpy(interp_coefficients[receiver][transmitter][slot - 1].data(), interp_coefficients_slot[receiver][transmitter].data(), pdsch_length * fft_size * sizeof(complex<float>));
            }
        }
#endif

#if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();
                cout << "Export channel coefficients (steady_clock) [µs] : "
                << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;
                BOOST_LOG_TRIVIAL(trace) << "Export channel coefficients (steady_clock) [µs] : "
                                         << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
                cout << "Export channel coefficients (clock_gettime) [µs] : " << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
                BOOST_LOG_TRIVIAL(trace) << "Export channel coefficients (clock_gettime) [µs] : "
                                         << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
#elif defined(CLOCK_TYPE_ASM)
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

        cout << "Export channel coefficients (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;
        BOOST_LOG_TRIVIAL(trace) << "\n Export channel coefficients (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;
#elif defined(CLOCK_TYPE_CLOCK)
        t2 = clock();
                cout << "Export channel coefficients (clock()) [µs] : "
                     << (t2 - t1) * 1e6 / CLOCKS_PER_SEC << endl;
                BOOST_LOG_TRIVIAL(trace) << "Export channel coefficients (clock()) [µs] : "
                     << (t2 - t1) * 1e6 / CLOCKS_PER_SEC << endl;
        #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
                clock_gettime(CLOCK_MONOTONIC, &t2);
                cout << "Export channel coefficients (clock_gettime) [µs] : " << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
                BOOST_LOG_TRIVIAL(trace) << "Export channel coefficients (clock_gettime) [µs] : "
                                         << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
        #elif defined(CLOCK_TYPE_RUSAGE)
                getrusage(RUSAGE_SELF, &usage);
                t2_utime = usage.ru_utime;
                cout << "Export channel coefficients (clock_gettime) [µs] : " << t2_utime.tv_usec - t1_utime.tv_usec << endl;
                BOOST_LOG_TRIVIAL(trace) << "Export channel coefficients (clock_gettime) [µs] : " << t2_utime.tv_usec - t1_utime.tv_usec << endl;
#endif

#if defined(CLOCK_TYPE_CHRONO)
        t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t1);
#elif defined(CLOCK_TYPE_ASM)
        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");
#elif defined(CLOCK_TYPE_CLOCK)
        t1 = clock();
        #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
                clock_gettime(CLOCK_MONOTONIC, &t1);
        #elif defined(CLOCK_TYPE_RUSAGE)
                getrusage(RUSAGE_SELF, &usage);
                t1_utime = usage.ru_utime;
#endif

#if defined (VBLAST_SQRD) or defined(VBLAST_QRD_COL_NORM) or defined(VBLAST_SQRD_AVX2) or defined(VBLAST_QRD)
        for(int transmitter = 0; transmitter < num_tx_dmrs_ports; transmitter++) {
                    for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
                        get_pdsch_channel_coefficients(interp_coefficients_slot[transmitter][receiver].data(),
                                                       pdsch_channel_coefficients[transmitter][receiver].data(),
                                                       pdsch_start,
                                                       dmrs_symbols,
                                                       fft_size,
                                                       pdsch_length,
                                                       dmrs_config_type,
                                                       antenna_ports_num_cdm_groups_without_data[antenna_port_value]);
                    }
                }
#else
        for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
            for(int transmitter = 0; transmitter < num_tx_dmrs_ports; transmitter++) {
                get_pdsch_channel_coefficients(interp_coefficients_slot[receiver][transmitter].data(),
                                               pdsch_channel_coefficients[receiver][transmitter].data(),
                                               pdsch_start,
                                               dmrs_symbols,
                                               fft_size,
                                               pdsch_length,
                                               dmrs_config_type,
                                               antenna_ports_num_cdm_groups_without_data[antenna_port_value]);
            }
        }
#endif

#if defined(CLOCK_TYPE_CHRONO)
        t2 = std::chrono::steady_clock::now();
                cout << "Extracting PDSCH channel coefficients [µs] : "
                << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;
                BOOST_LOG_TRIVIAL(trace) << "Extracting PDSCH channel coefficients [µs] : "
                                         << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
                cout << "Extracting PDSCH channel coefficients [µs] : " << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
                BOOST_LOG_TRIVIAL(trace) << "Extracting PDSCH channel coefficients [µs] : " << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
#elif defined(CLOCK_TYPE_ASM)
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

        t_cumul += t2 - t1;

        cout << "Extracting PDSCH channel coefficients [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;
        BOOST_LOG_TRIVIAL(trace) << "Extracting PDSCH channel coefficients [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;
#elif defined(CLOCK_TYPE_CLOCK)
        t2 = clock();
                cout << "Extracting PDSCH channel coefficients (clock()) [µs] : "
                     << (t2 - t1) * 1e6 / CLOCKS_PER_SEC << endl;
                BOOST_LOG_TRIVIAL(trace) << "Extracting PDSCH channel coefficients (clock()) [µs] : "
                     << (t2 - t1) * 1e6 / CLOCKS_PER_SEC << endl;
        #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
                clock_gettime(CLOCK_MONOTONIC, &t2);
                cout << "Extracting PDSCH channel coefficients [µs] : " << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
                BOOST_LOG_TRIVIAL(trace) << "Extracting PDSCH channel coefficients [µs] : " << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
        #elif defined(CLOCK_TYPE_RUSAGE)
                getrusage(RUSAGE_SELF, &usage);
                t2_utime = usage.ru_utime;
                cout << "Extracting PDSCH channel coefficients [µs] : " << t2_utime.tv_usec - t1_utime.tv_usec << endl;
                BOOST_LOG_TRIVIAL(trace) << "Extracting PDSCH channel coefficients [µs] : " << t2_utime.tv_usec - t1_utime.tv_usec << endl;
#endif

        /// Use the appropriate decoder
        if (encoding_type == diversity) {

#if defined(CLOCK_TYPE_CHRONO)
            t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
#elif defined(CLOCK_TYPE_ASM)
            asm volatile ("CPUID\n\t"
                          "RDTSC\n\t"
                          "mov %%edx, %0\n\t"
                          "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                    "%rax", "%rbx", "%rcx", "%rdx");
#elif defined(CLOCK_TYPE_CLOCK)
            t1 = clock();
        #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
                    clock_gettime(CLOCK_MONOTONIC, &t2);
        #elif defined(CLOCK_TYPE_RUSAGE)
                    getrusage(RUSAGE_SELF, &usage);
                    t1_utime = usage.ru_utime;
#endif
            /// Use alamouti SFBC decoder
            mimo_transmit_diversity_decoder(pdsch_samples,
                                            pdsch_channel_coefficients,
                                            num_pdsch_re_per_slot,
                                            equalized_symbols.data() + (slot - 1) * num_pdsch_re_per_slot,
                                            num_tx_dmrs_ports,
                                            nb_rx_ports);

#if defined(CLOCK_TYPE_CHRONO)
            t2 = std::chrono::steady_clock::now();
                    cout << "Alamouti decoding (steady_clock) [µs] : "
                         << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << endl;
                    BOOST_LOG_TRIVIAL(trace) << "Alamouti decoding (steady_clock) [µs] : "
                                             << chrono::duration_cast<chrono::microseconds>(t2 - t1).count() << endl;
#elif defined(CLOCK_TYPE_GETTIME)
            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
                    cout << "Alamouti decoding (clock_gettime) [ns] : "
                         << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
                    BOOST_LOG_TRIVIAL(trace) << "Alamouti decoding (clock_gettime) [ns] : "
                                             << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
#elif defined(CLOCK_TYPE_ASM)
            asm volatile("RDTSCP\n\t"
                         "mov %%edx, %0\n\t"
                         "mov %%eax, %1\n\t"
                         "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
            t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

            cout << "Alamouti decoding (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;
            BOOST_LOG_TRIVIAL(trace) << "Alamouti decoding (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;
#elif defined(CLOCK_TYPE_CLOCK)
            t2 = clock();
                    cout << "Alamouti decoding (clock()) [µs] : "
                         << (t2 - t1) * 1e6 / CLOCKS_PER_SEC << endl;
                    BOOST_LOG_TRIVIAL(trace) << "Alamouti decoding (clock()) [µs] : "
                         << (t2 - t1) * 1e6 / CLOCKS_PER_SEC << endl;
        #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
                    clock_gettime(CLOCK_MONOTONIC, &t2);
                    cout << "Alamouti decoding (clock_gettime) [ns] : "
                         << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
                    BOOST_LOG_TRIVIAL(trace) << "Alamouti decoding (clock_gettime) [µs] : "
                                             << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
        #elif defined(CLOCK_TYPE_RUSAGE)
                getrusage(RUSAGE_SELF, &usage);
                t2_utime = usage.ru_utime;
                cout << "Alamouti decoding (clock_gettime) [µs] : " << t2_utime.tv_usec - t1_utime.tv_usec << endl;
                BOOST_LOG_TRIVIAL(trace) << "Alamouti decoding (clock_gettime) [µs] : " << t2_utime.tv_usec - t1_utime.tv_usec << endl;
#endif
        } else if (encoding_type == vblast) {

#if defined(CLOCK_TYPE_CHRONO)
            t1 = std::chrono::steady_clock::now();
#elif defined(CLOCK_TYPE_GETTIME)
            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
#elif defined(CLOCK_TYPE_ASM)
            asm volatile ("CPUID\n\t"
                          "RDTSC\n\t"
                          "mov %%edx, %0\n\t"
                          "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                    "%rax", "%rbx", "%rcx", "%rdx");
#elif defined(CLOCK_TYPE_CLOCK)
            t1 = clock();
        #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
                    clock_gettime(CLOCK_MONOTONIC, &t2);
        #elif defined(CLOCK_TYPE_RUSAGE)
                    getrusage(RUSAGE_SELF, &usage);
                    t1_utime = usage.ru_utime;
#endif

#if defined(VBLAST_QRD_COL_NORM)
            #if defined(__AVX2__) and defined(AVX2_PROCESSING)
                    mimo_vblast_qrd_col_norm_avx2(pdsch_channel_coefficients,
                                                          pdsch_samples,
                                                          num_pdsch_re_per_slot,
                                                          equalized_symbols.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                          num_tx_dmrs_ports,
                                                          nb_rx_ports,
                                                          constellations[constellation_type],
                                                          detected_symbol_indexes.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                          constellation_type);
        #else
                    mimo_vblast_qrd_col_norm_modified(pdsch_channel_coefficients,
                                                     pdsch_samples,
                                                     num_pdsch_re_per_slot,
                                                     equalized_symbols.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                     num_tx_dmrs_ports,
                                                     nb_rx_ports,
                                                     constellations[constellation_type],
                                                     detected_symbol_indexes.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                     constellation_type);
        #endif
#elif defined(VBLAST_SQRD) /// SQRD
            #if defined(__AVX2__) and defined(AVX2_PROCESSING)
                                mimo_vblast_sqrd_avx2(pdsch_channel_coefficients,
                                  pdsch_samples,
                                  num_pdsch_re_per_slot,
                                  equalized_symbols.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                  num_tx_dmrs_ports,
                                  nb_rx_ports,
                                  constellations[constellation_type],
                                  detected_symbol_indexes.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                  constellation_type);
        #else
            /* SQRD */
            mimo_vblast_decoder_qr_decomp_modified(pdsch_channel_coefficients,
                                                   pdsch_samples,
                                                   num_pdsch_re_per_slot,
                                                   equalized_symbols.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                   num_tx_dmrs_ports,
                                                   nb_rx_ports,
                                                   constellations[constellation_type],
                                                   detected_symbol_indexes.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                   constellation_type);

            /*
            asm volatile ("CPUID\n\t"
                          "RDTSC\n\t"
                          "mov %%edx, %0\n\t"
                          "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            mimo_vblast_decoder_load_channel_coefficients_in_q(pdsch_channel_coefficients,
            q_matrix_qr_decomp, num_pdsch_re_per_slot, num_tx_dmrs_ports, nb_rx_ports);

            asm volatile("RDTSCP\n\t"
                         "mov %%edx, %0\n\t"
                         "mov %%eax, %1\n\t"
                         "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
            t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
            t_cumul += t2 - t1;

            cout << "VBLAST decoder (RDTSC/RDTSCP) [µs] Load channel coefs in Q : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

            asm volatile ("CPUID\n\t"
                          "RDTSC\n\t"
                          "mov %%edx, %0\n\t"
                          "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            mimo_vblast_decoder_compute_qr_decomp(r_matrix_qr_decomp,
                                                  q_matrix_qr_decomp,
                                                  pdsch_channel_coefficients,
                                                  detection_reversed_orders_qr_decomp,
                                                  pdsch_samples,
                                                  num_pdsch_re_per_slot,
                                                  num_tx_dmrs_ports,
                                                  nb_rx_ports);

            asm volatile("RDTSCP\n\t"
                         "mov %%edx, %0\n\t"
                         "mov %%eax, %1\n\t"
                         "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
            t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
            t_cumul += t2 - t1;

            cout << "VBLAST decoder (RDTSC/RDTSCP) [µs] compute QR : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

            asm volatile ("CPUID\n\t"
                          "RDTSC\n\t"
                          "mov %%edx, %0\n\t"
                          "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            mimo_vblast_decoder_qr_decomp_multiply_by_q_matrix(pdsch_samples,
                                                               //q_matrix_qr_decomp,
                                                               pdsch_channel_coefficients,
                                                               num_pdsch_re_per_slot,
                                                               equalized_symbols.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                               num_tx_dmrs_ports,
                                                               nb_rx_ports);

            asm volatile("RDTSCP\n\t"
                         "mov %%edx, %0\n\t"
                         "mov %%eax, %1\n\t"
                         "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
            t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);
            t_cumul += t2 - t1;

            cout << "VBLAST decoder (RDTSC/RDTSCP) [µs] multiply R by received symbols : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

            asm volatile ("CPUID\n\t"
                          "RDTSC\n\t"
                          "mov %%edx, %0\n\t"
                          "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            mimo_vblast_decoder_sic_detection(r_matrix_qr_decomp,
                                              detection_reversed_orders_qr_decomp,
                                              equalized_symbols.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                              detected_symbol_indexes.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                              constellations[constellation_type],
                                              constellation_type,
                                              num_pdsch_re_per_slot,
                                              num_tx_dmrs_ports); */
        #endif
#elif defined(VBLAST_QRD)
            mimo_vblast_decoder_qr_decomp_no_reordering_modified(pdsch_channel_coefficients,
                                                                pdsch_samples,
                                                                num_pdsch_re_per_slot,
                                                                equalized_symbols.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                                num_tx_dmrs_ports,
                                                                nb_rx_ports,
                                                                constellations[constellation_type],
                                                                detected_symbol_indexes.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                                constellation_type);

#elif defined(VBLAST_MF)/// Matched_filter
            vblast_mf(pdsch_samples,
                      pdsch_channel_coefficients,
                      num_pdsch_re_per_slot,
                      equalized_symbols + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                      num_tx_dmrs_ports,
                      nb_rx_ports);
#elif defined(VBLAST_ZF)
#if defined(__AVX2__) and defined(AVX2_PROCESSING)

            call_vblast_zf_avx_functions(num_tx_dmrs_ports,
                                         pdsch_samples,
                                         pdsch_channel_coefficients,
                                         num_pdsch_re_per_slot,
                                         equalized_symbols.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                         nb_rx_ports);

#else

            call_vblast_zf_functions(num_tx_dmrs_ports,
                                     pdsch_samples,
                                     pdsch_channel_coefficients,
                                     num_pdsch_re_per_slot,
                                     equalized_symbols.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                     nb_rx_ports);
#endif


#elif defined(VBLAST_ZF_BLOCK_WISE_INV)
            vblast_4_layers_block_wise_inversion(pdsch_samples,
                                                 pdsch_channel_coefficients,
                                                 num_pdsch_re_per_slot,
                                                 equalized_symbols + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                 num_tx_dmrs_ports,
                                                 nb_rx_ports);
#elif defined(VBLAST_ZF_FLOAT)
            vblast_zf_4_layers_float(pdsch_samples,
                                     pdsch_channel_coefficients,
                                     num_pdsch_re_per_slot,
                                     equalized_symbols + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                     num_tx_dmrs_ports,
                                     nb_rx_ports);


#elif defined(VBLAST_ZF_SEP)
            compute_hermitian_matrix(pdsch_channel_coefficients,
                                     hermitian_matrix_copy,
                                     num_pdsch_re_per_slot,
                                     num_tx_dmrs_ports,
                                     nb_rx_ports);

             multiply_by_transconj(pdsch_samples,
                      pdsch_channel_coefficients,
                      num_pdsch_re_per_slot,
                      equalized_symbols_copy.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                      num_tx_dmrs_ports,
                      nb_rx_ports);

            vblast_zf_4_layers(pdsch_samples,
                   pdsch_channel_coefficients,
                   hermitian_matrix_copy,
                   num_pdsch_re_per_slot,
                   equalized_symbols_copy.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                   num_tx_dmrs_ports,
                   nb_rx_ports);

#endif

#if defined(CLOCK_TYPE_CHRONO)
            t2 = std::chrono::steady_clock::now();
                    cout << "VBLAST decoder (steady_clock) [µs] : "
                         << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;
                    BOOST_LOG_TRIVIAL(trace) << "VBLAST decoder (steady_clock) [µs] : "
                                             << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count()
                                             << endl;
#elif defined(CLOCK_TYPE_GETTIME)
            clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t2);
                    cout << "VBLAST decoder (clock_gettime) [ns] : "
                         << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
                    BOOST_LOG_TRIVIAL(trace) << "VBLAST decoder (clock_gettime) [ns] : "
                                             << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
#elif defined(CLOCK_TYPE_ASM)
            asm volatile("RDTSCP\n\t"
                         "mov %%edx, %0\n\t"
                         "mov %%eax, %1\n\t"
                         "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
            t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

            t_cumul += t2 - t1;
            cout << "VBLAST decoder (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;
            BOOST_LOG_TRIVIAL(trace) << "VBLAST decoder (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;
#elif defined(CLOCK_TYPE_CLOCK)
            t2 = clock();
                    cout << "VBLAST decoder (clock()) [µs] : " << (t2 - t1) * 1e6 / CLOCKS_PER_SEC << endl;
                    BOOST_LOG_TRIVIAL(trace) << "VBLAST decoder (clock()) [µs] : " << (t2 - t1) * 1e6 / CLOCKS_PER_SEC << endl;
            #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
                        clock_gettime(CLOCK_MONOTONIC, &t2);
                        cout << "VBLAST decoder (clock_gettime) [µs] : "
                             << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
                        BOOST_LOG_TRIVIAL(trace) << "VBLAST decoder (clock_gettime) [µs] : "
                                                 << (t2.tv_nsec - t1.tv_nsec) * 1e-3 << endl;
            #elif defined(CLOCK_TYPE_RUSAGE)
                        getrusage(RUSAGE_SELF, &usage);
                        t2_utime = usage.ru_utime;
                        cout << "VBLAST decoder (clock_gettime) [µs] : " << t2_utime.tv_usec - t1_utime.tv_usec << endl;
                        BOOST_LOG_TRIVIAL(trace) << "VBLAST decoder (clock_gettime) [µs] : " << t2_utime.tv_usec - t1_utime.tv_usec << endl;
#endif

        } else if (encoding_type == none) {
            /// Use the zero-forcing decoder
            for (int re = 0; re < num_pdsch_re_per_slot; re++) {
                equalized_symbols[re] = pdsch_samples[0][re] / pdsch_channel_coefficients[0][0][re];
            }
        }

        /// Detect the symbols only for ZF equalization. Detection is already done
        /// in QRD
#if defined(VBLAST_ZF) or defined(VBLAST_ZF_AVX2)

        /// Detect the symbols and export the symbol indexes
        if(encoding_type == vblast) {
            asm volatile ("CPUID\n\t"
                          "RDTSC\n\t"
                          "mov %%edx, %0\n\t"
                          "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            ml_detector_tabs[constellation_type](
                    equalized_symbols.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                    detected_symbol_indexes.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                    num_pdsch_re_per_slot * num_tx_dmrs_ports);

            asm volatile("RDTSCP\n\t"
                         "mov %%edx, %0\n\t"
                         "mov %%eax, %1\n\t"
                         "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
            t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

            cout << "Ml detection for one slot ZF (RDTSC/RDTSCP) [µs] : "
                                     << (t2 - t1) / TSC_FREQ * 1e6 << endl;
            BOOST_LOG_TRIVIAL(trace) << "Ml detection for one slot ZF (RDTSC/RDTSCP) [µs] : "
                                     << (t2 - t1) / TSC_FREQ * 1e6 << endl;
        } else if (encoding_type == diversity) {
            asm volatile ("CPUID\n\t"
                          "RDTSC\n\t"
                          "mov %%edx, %0\n\t"
                          "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            ml_detector_tabs[constellation_type](
                    equalized_symbols.data() + (slot - 1) * num_pdsch_re_per_slot,
                    detected_symbol_indexes.data() + (slot - 1) * num_pdsch_re_per_slot,
                    num_pdsch_re_per_slot);

            asm volatile("RDTSCP\n\t"
                         "mov %%edx, %0\n\t"
                         "mov %%eax, %1\n\t"
                         "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
            t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

            cout << "Ml detection for one slot Alamouti (RDTSC/RDTSCP) [µs] : "
                                     << (t2 - t1) / TSC_FREQ * 1e6 << endl;
            BOOST_LOG_TRIVIAL(trace) << "Ml detection for one slot Alamouti (RDTSC/RDTSCP) [µs] : "
                                     << (t2 - t1) / TSC_FREQ * 1e6 << endl;
        } else if (encoding_type == none) {
            asm volatile ("CPUID\n\t"
                          "RDTSC\n\t"
                          "mov %%edx, %0\n\t"
                          "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            ml_detector_tabs[constellation_type](
                    equalized_symbols.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                    detected_symbol_indexes.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                    num_pdsch_re_per_slot * num_tx_dmrs_ports);

            asm volatile("RDTSCP\n\t"
                         "mov %%edx, %0\n\t"
                         "mov %%eax, %1\n\t"
                         "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
            t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

            cout << "Ml detection for one slot ZF SISO (RDTSC/RDTSCP) [µs] : "
                                     << (t2 - t1) / TSC_FREQ * 1e6 << endl;
            BOOST_LOG_TRIVIAL(trace) << "Ml detection for one slot ZF SISO (RDTSC/RDTSCP) [µs] : "
                                     << (t2 - t1) / TSC_FREQ * 1e6 << endl;
        }
#endif

#if defined(CLOCK_TYPE_CHRONO)
        t4 = std::chrono::steady_clock::now();
                cout << "time to decode one slot [µs] : " << std::chrono::duration_cast<std::chrono::microseconds>(t4 -t3).count() << endl;
                BOOST_LOG_TRIVIAL(trace) << "time to decode one slot [µs] : " << std::chrono::duration_cast<std::chrono::microseconds>(t4 -t3).count() << endl;
#elif defined(CLOCK_TYPE_GETTIME)
        clock_gettime(CLOCK_THREAD_CPUTIME_ID, &t4);
                cout << "time to decode one slot (clock_gettime) [µs] : " << (t4.tv_nsec - t3.tv_nsec) * 1e-3 << endl;
                BOOST_LOG_TRIVIAL(trace) << "time to decode one slot (clock_gettime) [µs] : " << (t4.tv_nsec - t3.tv_nsec) * 1e-3 << endl;
#elif defined(CLOCK_TYPE_ASM)
        asm volatile("RDTSCP\n\t"
                     "mov %%edx, %0\n\t"
                     "mov %%eax, %1\n\t"
                     "CPUID\n\t": "=r" (cycles_high4), "=r" (cycles_low4)::
                "%rax", "%rbx", "%rcx", "%rdx");

        t3 = (((uint64_t) cycles_high3 << 32) | cycles_low3);
        t4 = (((uint64_t) cycles_high4 << 32) | cycles_low4);

        cout << "t4 - t3 diff (RDTSC/RDTSCP) [µs] : " << (t4 - t3)/TSC_FREQ * 1e6 << endl;
        cout << "time to decode one slot without export (cumulative time) [µs] : " << t_cumul/TSC_FREQ * 1e6 << endl;
        BOOST_LOG_TRIVIAL(trace) << "time to decode one slot (RDTSC/RDTSCP) [µs] : " << t_cumul/TSC_FREQ * 1e6 << endl;
#elif defined(CLOCK_TYPE_CLOCK)
        t4 = clock();
                cout << "time to decode one slot (clock()) [µs] : " << (t4 - t3) * 1e6 / CLOCKS_PER_SEC << endl;
                BOOST_LOG_TRIVIAL(trace) << "time to decode one slot (clock()) [µs] : " << (t4 - t3) * 1e6 / CLOCKS_PER_SEC << endl;
        #elif defined(CLOCK_TYPE_GETTIME_MONOTONIC)
                clock_gettime(CLOCK_MONOTONIC, &t4);
                cout << "time to decode one slot (clock_gettime) [µs] : " << (t4.tv_nsec - t3.tv_nsec) * 1e-3 << endl;
                BOOST_LOG_TRIVIAL(trace) << "time to decode one slot (clock_gettime) [µs] : " << (t4.tv_nsec - t3.tv_nsec) * 1e-3 << endl;
        #elif defined(CLOCK_TYPE_RUSAGE)
                getrusage(RUSAGE_SELF, &usage);
                t4_utime = usage.ru_utime;
                cout << "time to decode one slot (clock_gettime) [µs] : " << t4_utime.tv_usec - t3_utime.tv_usec << endl;
                BOOST_LOG_TRIVIAL(trace) << "time to decode one slot (clock_gettime) [µs] : " << t4_utime.tv_usec - t3_utime.tv_usec << endl;
#endif
    }

    /// Export equalized symbols
    //int temp = equalized_symbols.size();
    export1dVector("equalized_grid.txt", equalized_symbols.data(), buffer_size); //buffer_size);

    /// Detect the symbols and export the symbol indexes
    //ml_detector_tabs[constellation_type](equalized_symbols.data(), detected_symbol_indexes.data(), buffer_size);
    export1dVector("decoded_grid.txt", detected_symbol_indexes.data(), buffer_size);

    /// Compute the symbol error rate (if transmission and reception are done on the same usrp device & main function)
    cout << "symbol error rate : " << symbol_error_rate(detected_symbol_indexes.data(), sending_buffer_symbol_indexes.data(), buffer_size) << endl;
    BOOST_LOG_TRIVIAL(info) << "symbol error rate: " << symbol_error_rate(detected_symbol_indexes.data(), sending_buffer_symbol_indexes.data(), buffer_size) << endl;

#if defined (VBLAST_ZF_SEP)
    ml_detector_tabs[constellation_type](equalized_symbols_copy.data(), detected_symbol_indexes.data(), buffer_size);
    cout << "symbol error rate : " << symbol_error_rate(detected_symbol_indexes.data(), sending_buffer_symbol_indexes.data(), buffer_size) << endl;
#endif
    /// Export pilot coefficients and interpolated coefficients
    /// Export averaged and interpolated coefs
    for (int transmitter = 0; transmitter < num_tx_dmrs_ports; transmitter++) {
        for (int receiver = 0; receiver < nb_rx_ports; receiver++) {
            exportGrid("tx" + to_string(transmitter + 1) + "_rx" + to_string(receiver + 1) + "_pilots.txt",
                       pilot_coefficients, num_dmrs_symbols_per_slot, dmrs_sequence_size,
                       num_tx_dmrs_ports, transmitter, receiver);

            exportGrid("tx" + to_string(transmitter + 1) + "_rx" + to_string(receiver + 1) + "_pilots_before_sto.txt",
                       pilot_coefficients_before_sto, num_dmrs_symbols_per_slot, dmrs_sequence_size,
                       num_tx_dmrs_ports, transmitter, receiver);

            exportGrid("tx" + to_string(transmitter + 1) + "_rx" + to_string(receiver + 1) + "_coefs.txt",
                       interp_coefficients[receiver][transmitter]);
        }
    }

    destroy_fft_plans();
    destroy_ifft_plans();
    fftwf_cleanup(); // clear still reachable memory

    free(dmrs_symbols);
    free(dmrs_subcarriers);
    free(rx_estimation_semaphores);
    free(rx_interpolation_semaphores);
    free(wait_estimation_and_interpolation_semaphores);
    free_sync_pss_plans();

    for(int i = 0; i < antenna_port_cdm_groups.size(); i++) {
        free(antenna_port_cdm_groups[i]);
        free(antenna_port_cdm_groups_sizes[i]);
        free(cdm_group_sizes[i]);
        free(antenna_port_dmrs_ports[i]);
    }

    delete constellations[0];
    delete constellations[1];

    /// stop channel estimation & interpolation threads
    /**
    stop_signal_called_channel_estimation = 1;
    for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
        sem_post(rx_estimation_semaphores + receiver);
        (rx_threads + receiver)->join();

        for(int transmitter = 0; transmitter < num_tx_dmrs_ports; transmitter++) {
            sem_post(rx_interpolation_semaphores + receiver * num_tx_dmrs_ports + transmitter);
            (interp_threads +  receiver * num_tx_dmrs_ports + transmitter)->join();
        }
    } */

    return EXIT_SUCCESS;
}
