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

#include <boost/align/aligned_allocator.hpp>

#include "time.h"
//#include <ctime>

#include "../../../lib/usrp/usrp_x300.h"
#include "../../../lib/free5gran_utils.h"
#include "../../../lib/utils/channel_estimation/channel_estimation.h"
#include "../../../lib/utils/channel_mapping/channel_mapping.h"
#include "../../../lib/utils/mimo/mimo.h"
#include "../../../lib/utils/mimo/transmit_diversity.h"
#include "../../../lib/utils/mimo/vblast.h"
#include "../../../lib/variables/variables.h"

int stop_signal_called_main = 0; /// Set to 1 to stop the main thread

/// Define the encoding type to be used
mimo_encoding_type encoding_type = vblast; // Use "diversity" for alamouti encoding, "vblast" or "none" to use only 1TX/1RX

/// Define the number of layers to be used, up to 4 layers.
/// Set to 2 or 4 for alamouti, and 1 for SISO
#define NUM_LAYERS 4

#define RES_PATH string("./vblast/") /// Path where results will be exported

using namespace std;

/** Handler of SIGINT signal
 */
void sigintHandler(int sig_num) {
    stop_signal_called_main = 1;
}

/// PDSCH equalization for SISO case
void zero_forcing(std::complex<float> * pdsch_samples_,
                  std::complex<float> * pdsch_channel_coefficients_,
                  std::complex<float> * equalized_symbols_,
                  int num_re_pdsch_);

void zero_forcing_avx2(std::complex<float> * pdsch_samples_,
                       std::complex<float> * pdsch_channel_coefficients_,
                       std::complex<float> * equalized_symbols_,
                       int num_re_pdsch_);

int UHD_SAFE_MAIN(int argc, char *argv[]) {

    uint64_t t1, t2, t3, t4;
    vector<uint64_t> t_cumul(4);
    unsigned cycles_low1, cycles_low2, cycles_low3, cycles_low4,
             cycles_high1, cycles_high2, cycles_high3, cycles_high4;

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

    cout << "MIMO encoding type used : " << encoding_type << endl;
    BOOST_LOG_TRIVIAL(trace) << "MIMO encoding type used : " << encoding_type << endl;

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
    float        gain_tx                   = 30;                      // max 89.8 dB
    float        gain_rx                   = 30;                      // max 76 dB
    bool         double_symbol             = false;                   // set to True to use double symbol DMRS
    int          l0                        = 2;                       // First symbol of DMRS within a PDSCH allocation
    int          pdsch_length              = 14;                      // PDSCH duration within a slot
    int          dmrs_additional_positions = 3;                       // Number of additional positions
    int          pdsch_start               = 0;                       // First symbol of the PDSCH within a slot
    int pss_position = 0, n_id_2 = 0, n_id_1 = 0;
    int nb_slots = nb_tx_subframes * pow(2, numerology_index);

    boost_log_level level = trace;

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
    /// Vector containing the TX antenna ports used by each transmit thread. Configure only 1 thread using the 2 antenna ports available
    vector<vector<size_t>> ports_tx_usrp = {antenna_port_value_rx_usrp_ports[antenna_port_value]};

    /// Vector containing the RX antenna ports used by each receive thread. Configure only 1 thread using the 2 antenna ports available
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


    //int slot_number_channel_estimation = 0;

    /// Contains the interpolated channel coefficients for one slot.
    vector<complex<float>>//, boost::alignment::aligned_allocator<complex<float>, 32>>
    interp_coefficients_slot_qrd[MAX_TX_PORTS][MAX_RX_PORTS];
    for(int i = 0; i < MAX_RX_PORTS; i++) {
        for(int j = 0; j < MAX_TX_PORTS; j++) {
            interp_coefficients_slot_qrd[i][j] = vector<complex<float>>//, boost::alignment::aligned_allocator<complex<float>, 32>>
            (pdsch_length * fft_size);
        }
    }
    vector<complex<float>> //, boost::alignment::aligned_allocator<complex<float>, 32>>
    interp_coefficients_slot[MAX_RX_PORTS][MAX_TX_PORTS];
    for(int i = 0; i < MAX_RX_PORTS; i++) {
        for(int j = 0; j < MAX_TX_PORTS; j++) {
            interp_coefficients_slot[i][j] = vector<complex<float>>//, boost::alignment::aligned_allocator<complex<float>, 32>>
            (pdsch_length * fft_size);
        }
    }

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
    vector<complex<float>>//,
            //boost::alignment::aligned_allocator<complex<float>, 32>>
            pilot_coefficients_slot(nb_rx_ports * num_tx_dmrs_ports * num_dmrs_symbols_per_slot * dmrs_sequence_size);

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

    vector<complex<float>> pdsch_channel_coefficients_sqrd[MAX_TX_PORTS][MAX_RX_PORTS];
    vector<complex<float>> pdsch_channel_coefficients_qrd_col_norm[MAX_TX_PORTS][MAX_RX_PORTS];
    vector<complex<float>> pdsch_channel_coefficients_qrd[MAX_TX_PORTS][MAX_RX_PORTS];
    for(int transmitter = 0; transmitter < MAX_TX_PORTS; transmitter++) {
        for(int receiver = 0; receiver < MAX_RX_PORTS; receiver++) {
            pdsch_channel_coefficients_sqrd[transmitter][receiver] =
                    vector<complex<float>> (num_pdsch_re_per_slot);
            pdsch_channel_coefficients_qrd_col_norm[transmitter][receiver] =
                    vector<complex<float>> (num_pdsch_re_per_slot);
            pdsch_channel_coefficients_qrd[transmitter][receiver] =
                    vector<complex<float>> (num_pdsch_re_per_slot);
        }
    }




    vector<complex<float>> pdsch_channel_coefficients[MAX_RX_PORTS][MAX_TX_PORTS];
    for(int receiver = 0; receiver < MAX_RX_PORTS; receiver++) {
        for(int transmitter = 0; transmitter < MAX_TX_PORTS; transmitter++) {
            pdsch_channel_coefficients[receiver][transmitter] =
                    vector<complex<float>> (num_pdsch_re_per_slot);
        }
    }

    vector<vector<int>> sic_orders_slot(num_pdsch_re_per_slot, vector<int>(num_tx_dmrs_ports));

    vector<vector<float>> column_norms_slot(num_pdsch_re_per_slot, vector<float>(num_tx_dmrs_ports));

    /************** DMRS sequence generation *******************/
    /// Generate the DMRS sequences for the whole frame according to the antenna port number
    vector<complex<float>> dmrs_sequences(MAX_TX_PORTS * (nb_slots - 1) * num_dmrs_symbols_per_slot * dmrs_sequence_size);

    compute_dmrs_sequences_type1(dmrs_symbols,
                                 dmrs_sequences.data(),
                                 num_dmrs_symbols_per_slot,
                                 dmrs_sequence_size,
                                 nb_slots,
                                 double_symbol);

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
    vector<vector<complex<float>>>//,boost::alignment::aligned_allocator<complex<float>, 32>>>
            equalized_symbols(4, vector<complex<float>> //, boost::alignment::aligned_allocator<complex<float>, 32>>
                    (buffer_size)); /// Symbols equalized by the MIMO decoder

    vector<int> sending_buffer_symbol_indexes(buffer_size);
    vector<vector<int>> detected_symbol_indexes(4, vector<int>(buffer_size)); /// Symbol indexes detected by the ML decoder

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
            grids[grid_no][0][fft_size/2 - 1 - (SIZE_PSS_SSS_SIGNAL-1)/2 + pss_carrier] =
                    pss_sequence[pss_carrier];
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
        //complex<float> layers[num_tx_dmrs_ports * buffer_size];
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

        /// RE mapping on PDSCH. Start at slot no. 1
        for(int slot = 1; slot < nb_slots; slot++) {
            for (int i = 0; i < num_pdsch_re_per_slot; i++) {

                /// Get symbol and subcarrier number in position
                symbol = pdsch_positions[2 * i] + slot * 14;
                sc = pdsch_positions[2 * i + 1];

                for (int layer = 0; layer < num_tx_dmrs_ports; layer++) {
                    grids[layer][symbol][sc] = precoded_layers[layer * num_symbols_per_precoded_layer + buffer_count];
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

    BOOST_LOG_TRIVIAL(info) << "Bandwidth TX : " << usrp_tx_ptr->get_tx_bandwith(0) << endl;
    BOOST_LOG_TRIVIAL(info) << "Bandwidth RX : " << usrp_tx_ptr->get_tx_bandwith(0) << endl;
    BOOST_LOG_TRIVIAL(info) << "Sampling Rate TX : " << usrp_tx_ptr->get_tx_rate(0) << endl;
    BOOST_LOG_TRIVIAL(info) << "Sampling Rate RX : " << usrp_tx_ptr->get_rx_rate(0) << endl;
    BOOST_LOG_TRIVIAL(info) << "numerology index : " << numerology_index << endl;
    BOOST_LOG_TRIVIAL(info) << "Symbols per subframe : " << symbols_per_subframe << endl;
    BOOST_LOG_TRIVIAL(info) << "Number of transmitted subframes : " << nb_tx_subframes <<  endl;
    BOOST_LOG_TRIVIAL(info) << "Number of received subframes : " << nb_rx_subframes << endl;
    BOOST_LOG_TRIVIAL(info) << "SCS : " << scs << endl;
    BOOST_LOG_TRIVIAL(info) << "FFT size : " << fft_size << endl;
    BOOST_LOG_TRIVIAL(info) << "Scaling factor : " << scaling_factor << endl;
    BOOST_LOG_TRIVIAL(info) << "Center frequency TX : " << usrp_tx_ptr->get_tx_center_frequency(0) << endl;
    BOOST_LOG_TRIVIAL(info) << "Center frequency RX : " << usrp_tx_ptr->get_rx_center_frequency(0) << endl;
    BOOST_LOG_TRIVIAL(info) << "Constellation type : " << constellation_type << endl;
    BOOST_LOG_TRIVIAL(info) << "Gain TX : " << usrp_tx_ptr->get_tx_gain(0) << endl;
    BOOST_LOG_TRIVIAL(info) << "Gain RX : " << usrp_tx_ptr->get_rx_gain(0) << endl;
    BOOST_LOG_TRIVIAL(trace) << "number of slots per frame : " << nb_slots << endl;

    /// Export the data
    export1dVector(RES_PATH+"non_encoded.txt", sending_buffer.data(), buffer_size);
    export1dVector(RES_PATH+"sending_buffer_symbol_indexes.txt", sending_buffer_symbol_indexes.data(), buffer_size);

    /// Export the grids to be plotted
    for(int grid_no = 0; grid_no < num_tx_dmrs_ports; grid_no++) {
        /// Export signal to be sent on TX1 (time domain signal)
        export1dVector(RES_PATH+"tx" + to_string(grid_no + 1) + ".txt", time_domain_grids[grid_no]);
        exportGrid(RES_PATH+"tx"+ to_string(grid_no + 1) + "_grid.txt", grids[grid_no]);
    }

    /// Retrieve the contents of the receiving buffers
    vector<vector<vector<complex<float>>>> all_rx_buffers;
    vector<vector<complex<float>>> receiveBuffers; // Get the data for the only user configured

    /******* Number of repeated simulations *******/
    int num_simulations = 20;
    vector<vector<vector<double>>> times(15,
                                   vector<vector<double>>(num_simulations,
                                                  vector<double>(nb_slots - 1)));
    vector<double> times_pss_sync(num_simulations);
    vector<double> times_fft(num_simulations);
    vector<vector<double>> times_ml_detection(num_simulations,
                                              vector<double>(nb_slots - 1));
    vector<vector<double>> symbol_error_rates(num_simulations,
                                              vector<double>(4));

    /******************** Perform multiple simulations **************************/

    /// Initialize all buffers before decoding
    vector<vector<vector<complex<float>>>> receivedGrids(nb_rx_ports,
                                                         vector<vector<complex<float>>>(symbols_per_subframe*nb_rx_subframes,
                                                                 vector<complex<float>>(fft_size, 0)));
    vector<complex<float>> synchronized_signals[nb_rx_ports];
    vector<vector<complex<float>>> synchronized_grids[nb_rx_ports];
    int pss_sample_index;
    int synchronization_index;
    int tx_dmrs_port, path;
    int subframe_size;
    int start_symbol_frequency_offset = 0;
    float frequency_offset = 0;

    vector<vector<complex<float>>> pilot_coefficients =
            vector<vector<complex<float>>>(nb_slots - 1 ,
                    vector<complex<float>>(nb_rx_ports * num_tx_dmrs_ports * num_dmrs_symbols_per_slot * dmrs_sequence_size, 0));

    vector<vector<complex<float>>> interp_coefficients_qrd[MAX_TX_PORTS][MAX_RX_PORTS];
            for(int transmitter = 0; transmitter < MAX_TX_PORTS; transmitter++) {
                for(int receiver = 0; receiver < MAX_RX_PORTS; receiver++) {
                    interp_coefficients_qrd[transmitter][receiver] = vector<vector<complex<float>>>(nb_slots - 1,
                            vector<complex<float>>(pdsch_length * fft_size, 0));
                }
            }
    vector<vector<complex<float>>> interp_coefficients[MAX_RX_PORTS][MAX_TX_PORTS];
    for(int receiver = 0; receiver < MAX_RX_PORTS; receiver++) {
        for(int transmitter = 0; transmitter < MAX_TX_PORTS; transmitter++) {
            interp_coefficients[receiver][transmitter] = vector<vector<complex<float>>>(nb_slots - 1,
                    vector<complex<float>>(pdsch_length * fft_size, 0));
        }
    }

    vector<vector<complex<float>>> pdsch_samples(MAX_RX_PORTS,
                                                 vector<complex<float>> (num_pdsch_re_per_slot));

    vector<vector<complex<float>>> dmrs_samples[nb_rx_ports];
    for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
        dmrs_samples[receiver].resize(antenna_ports_num_cdm_groups_without_data[antenna_port_value],
                                      vector<complex<float>>(num_dmrs_symbols_per_slot * dmrs_sequence_size));
    }

    usrp_tx_ptr->init_tx_buffers({time_domain_grids});
    usrp_tx_ptr->config_tx_streams();
    usrp_tx_ptr->start_sending();
    sleep(2.5);
    for(int sim = 0; sim < num_simulations; sim++) {
        usrp_tx_ptr->init_rx_buffers(2 * time_domain_grids[0].size());

        /// Reconfigure new TX and RX streamers
        usrp_tx_ptr->config_rx_streams();

        /// Start sending and receiving to/from the usrp device
        usrp_tx_ptr->start_receiving();
        /// Stop the main thread on SIGINT
        while(!stop_signal_called_main) {
            sleep(2.5);
            stop_signal_called_main = 1;
        }
        stop_signal_called_main = 0;
        ofstream output_file_estimated_channel_coefs("estimated_channel_coefs_main.txt");
        ofstream output_file_interp_channel_coefs("interp_channel_coefs_main.txt");
        /// Stop the sending and receiving threads when SIGINT called
        usrp_tx_ptr->stop_rx_threads();
        if(usrp_tx_ptr->get_nothing_received()) {
            cout << "No signal received. Ending the main thread ..." << endl;
            break;
        }
        /// Retrieve the contents of the receiving buffers
        usrp_tx_ptr->get_receiving_buffer(all_rx_buffers);
        receiveBuffers = all_rx_buffers[0]; // Get the data for the only user configured

        for(int i = 0; i < receiveBuffers.size(); i++) {
            cout << "\n Exporting received signal on channel "+to_string(i)+"... " << std::endl;
            BOOST_LOG_TRIVIAL(info) << "\n Exporting received signal on channel "+to_string(i)+"... " << endl;
            export1dVector(RES_PATH+"rx"+to_string(i+1)+"_frame"+to_string(sim)+".txt", receiveBuffers[i]);
        }

        /// Decode the grids
        for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
            synchronized_signals[receiver].resize((cum_sum_cp_lengths[symbols_per_subframe - 1]
                                                   + fft_size + cp_lengths[symbols_per_subframe - 1]) * nb_tx_subframes);
        }

        int num_samples_per_received_signal = receiveBuffers[0].size();

        for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
            synchronized_grids[receiver].resize(symbols_per_subframe * nb_tx_subframes,
                                                vector<complex<float>>(fft_size, 0));
        }

        pss_sample_index = 0;

        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");

        /// Perform synchronization on receive antenna 0
        synchronization_index = 0;
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

        times_pss_sync[sim] = (t2 - t1) / TSC_FREQ * 1e6;

        BOOST_LOG_TRIVIAL(trace) << "PSS synchronization on one grid based on known N_ID_2 [µs]: " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");

        /// Compute FFT for all grids
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

        times_fft[sim] = (t2 - t1) / TSC_FREQ * 1e6;

        BOOST_LOG_TRIVIAL(trace) << "FFT [µs]: " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

        subframe_size = cum_sum_cp_lengths[symbols_per_subframe - 1] + fft_size + cp_lengths[symbols_per_subframe - 1];
        for(int grid_no = 0; grid_no < nb_rx_ports; grid_no++) {
            /** Export the synchronized signals in time domain
             *  and the synchronized grids
             */
            std::copy(receiveBuffers[grid_no].begin() + pss_sample_index,
                      receiveBuffers[grid_no].begin() + pss_sample_index + subframe_size * nb_tx_subframes,
                      synchronized_signals[grid_no].begin());

            cout << "\n Exporting synchronized signal on channel " + to_string(grid_no) + "... " << std::endl;
            BOOST_LOG_TRIVIAL(info) << "\n Exporting synchronized signal on channel " + to_string(grid_no) + "... " << endl;
            export1dVector(RES_PATH+"sync_rx" + to_string(grid_no+1) + "_frame"+to_string(sim)+ ".txt", synchronized_signals[grid_no]);
        }

        /******************* Frequency offset correction *************************
        asm volatile ("CPUID\n\t"
                      "RDTSC\n\t"
                      "mov %%edx, %0\n\t"
                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                "%rax", "%rbx", "%rcx", "%rdx");

        start_symbol_frequency_offset = 0;
        frequency_offset = 0;
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
        cout << "CFO estimation & correction on each channel [µs]: " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

        /******************** Frequency offset correction end ***********************/
        tx_dmrs_port = 0, path = 0;

        /// For each slot within the received frame
        for(int slot = 1; slot < nb_slots; slot++) {

            cout << "Decoding slot " << slot << " on frame " << sim << endl;
            BOOST_LOG_TRIVIAL(trace) << "\n######## SLOT " << slot << " frame " << sim << " ########" << endl;

            for(int i = 0; i < t_cumul.size(); i++) {
                t_cumul[i] = 0;
            }
                asm volatile ("CPUID\n\t"
                              "RDTSC\n\t"
                              "mov %%edx, %0\n\t"
                              "mov %%eax, %1\n\t": "=r" (cycles_high3), "=r" (cycles_low3)::
                        "%rax", "%rbx", "%rcx", "%rdx");

            /// Perform channel estimation
            /// Estimate the pilots on each DMRS port

            asm volatile ("CPUID\n\t"
                                      "RDTSC\n\t"
                                      "mov %%edx, %0\n\t"
                                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                                "%rax", "%rbx", "%rcx", "%rdx");

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

            asm volatile("RDTSCP\n\t"
                                     "mov %%edx, %0\n\t"
                                     "mov %%eax, %1\n\t"
                                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                                "%rax", "%rbx", "%rcx", "%rdx");

                        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
                        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

                        for(int i = 0; i < t_cumul.size(); i++) {
                            t_cumul[i] += t2 - t1;
                        }

                        times[0][sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;

                        BOOST_LOG_TRIVIAL(trace) << "Getting PDSCH and DMRS samples (RDTSC/RDTSCP) [µs]: " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

            asm volatile ("CPUID\n\t"
                                      "RDTSC\n\t"
                                      "mov %%edx, %0\n\t"
                                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                                "%rax", "%rbx", "%rcx", "%rdx");

#if defined(__AVX2__) and defined(AVX2_PROCESSING)

            for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
                estimate_pilots_avx(antenna_port_dmrs_ports[antenna_port_value],
                                    cdm_group_sizes[antenna_port_value],
                                    dmrs_symbols,
                                    dmrs_samples[receiver].data(),
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

            asm volatile("RDTSCP\n\t"
                         "mov %%edx, %0\n\t"
                         "mov %%eax, %1\n\t"
                         "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
            t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

            for(int i = 0; i < t_cumul.size(); i++) {
                t_cumul[i] += t2 - t1;
            }

            times[1][sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;

            BOOST_LOG_TRIVIAL(trace) << "Channel estimation on pilots AVX2  (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

#else
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
                        asm volatile("RDTSCP\n\t"
                                     "mov %%edx, %0\n\t"
                                     "mov %%eax, %1\n\t"
                                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                                "%rax", "%rbx", "%rcx", "%rdx");

                        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
                        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

                        for(int i = 0; i < t_cumul.size(); i++) {
                            t_cumul[i] += t2 - t1;
                        }

                        times[1][sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;

                        BOOST_LOG_TRIVIAL(trace) << "Channel estimation on pilots (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

#endif

            asm volatile ("CPUID\n\t"
                                      "RDTSC\n\t"
                                      "mov %%edx, %0\n\t"
                                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                                "%rax", "%rbx", "%rcx", "%rdx");

#if defined(__AVX2__) and defined(AVX2_PROCESSING)
            for(int transmitter = 0; transmitter < num_tx_dmrs_ports; transmitter++) {
                for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
                    interpolate_coefs_avx(interp_coefficients_slot_qrd[transmitter][receiver].data(),
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

            asm volatile("RDTSCP\n\t"
                         "mov %%edx, %0\n\t"
                         "mov %%eax, %1\n\t"
                         "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
            t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

            for(int i = 1; i < t_cumul.size(); i++) { /// Update for all QR decoders
                t_cumul[i] += t2 - t1;
            }

            times[2][sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;

            BOOST_LOG_TRIVIAL(trace) << "Channel coefficients interpolation SQRD AVX2 (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

            asm volatile ("CPUID\n\t"
                          "RDTSC\n\t"
                          "mov %%edx, %0\n\t"
                          "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                    "%rax", "%rbx", "%rcx", "%rdx");
#else
            for(int transmitter = 0; transmitter < num_tx_dmrs_ports; transmitter++) {
                for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
                    interpolate_coefs(interp_coefficients_slot_qrd[transmitter][receiver].data(),
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

                        asm volatile("RDTSCP\n\t"
                         "mov %%edx, %0\n\t"
                         "mov %%eax, %1\n\t"
                         "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
            t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

            for(int i = 1; i < t_cumul.size(); i++) { /// Update for all QR decoders
                t_cumul[i] += t2 - t1;
            }

            times[2][sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;

            BOOST_LOG_TRIVIAL(trace) << "Channel coefficients interpolation SQRD (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

            asm volatile ("CPUID\n\t"
                          "RDTSC\n\t"
                          "mov %%edx, %0\n\t"
                          "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                    "%rax", "%rbx", "%rcx", "%rdx");

#endif

#if defined(__AVX2__) and defined(AVX2_PROCESSING)
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

            asm volatile("RDTSCP\n\t"
                         "mov %%edx, %0\n\t"
                         "mov %%eax, %1\n\t"
                         "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
            t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

            t_cumul[0] += t2 - t1; /// Update only for ZF

            times[3][sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;

            BOOST_LOG_TRIVIAL(trace) << "Channel coefficients interpolation AVX2 (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

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

            asm volatile("RDTSCP\n\t"
                         "mov %%edx, %0\n\t"
                         "mov %%eax, %1\n\t"
                         "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
            t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

            t_cumul[0] += t2 - t1; /// Update only for ZF

            times[3][sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;

            BOOST_LOG_TRIVIAL(trace) << "Channel coefficients interpolation (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

#endif

            asm volatile ("CPUID\n\t"
                                      "RDTSC\n\t"
                                      "mov %%edx, %0\n\t"
                                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                                "%rax", "%rbx", "%rcx", "%rdx");

            memcpy(pilot_coefficients[slot - 1].data(), pilot_coefficients_slot.data(), nb_rx_ports * num_tx_dmrs_ports * num_dmrs_symbols_per_slot * dmrs_sequence_size * sizeof(complex<float>));

            for(int transmitter = 0; transmitter < num_tx_dmrs_ports; transmitter++) {
                for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
                    memcpy(interp_coefficients[transmitter][receiver][slot - 1].data(), interp_coefficients_slot[transmitter][receiver].data(), pdsch_length * fft_size * sizeof(complex<float>));
                }
            }

            for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
                for(int transmitter = 0; transmitter < num_tx_dmrs_ports; transmitter++) {
                    memcpy(interp_coefficients[receiver][transmitter][slot - 1].data(), interp_coefficients_slot[receiver][transmitter].data(), pdsch_length * fft_size * sizeof(complex<float>));
                }
            }

            asm volatile("RDTSCP\n\t"
                                     "mov %%edx, %0\n\t"
                                     "mov %%eax, %1\n\t"
                                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                                "%rax", "%rbx", "%rcx", "%rdx");

                        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
                        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

            BOOST_LOG_TRIVIAL(trace) << "\n Export channel coefficients (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

            asm volatile ("CPUID\n\t"
                                      "RDTSC\n\t"
                                      "mov %%edx, %0\n\t"
                                      "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                                "%rax", "%rbx", "%rcx", "%rdx");

            for(int transmitter = 0; transmitter < num_tx_dmrs_ports; transmitter++) {
                for(int receiver = 0; receiver < nb_rx_ports; receiver++) {
                    get_pdsch_channel_coefficients(interp_coefficients_slot_qrd[transmitter][receiver].data(),
                                                   pdsch_channel_coefficients_qrd[transmitter][receiver].data(),
                                                   pdsch_start,
                                                   dmrs_symbols,
                                                   fft_size,
                                                   pdsch_length,
                                                   dmrs_config_type,
                                                   antenna_ports_num_cdm_groups_without_data[antenna_port_value]);
                }
            }

            asm volatile("RDTSCP\n\t"
                         "mov %%edx, %0\n\t"
                         "mov %%eax, %1\n\t"
                         "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                    "%rax", "%rbx", "%rcx", "%rdx");

            t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
            t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

            for(int i = 1; i < t_cumul.size(); i++) { /// Update for all QR decoders
                t_cumul[i] += t2 - t1;
            }

            times[4][sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;

            BOOST_LOG_TRIVIAL(trace) << "Extracting PDSCH channel coefficients SQRD [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

            /// Creates copies for SQRD and QRD
            for (int transmitter = 0; transmitter < num_tx_dmrs_ports; transmitter++) {
                for (int receiver = 0; receiver < nb_rx_ports; receiver++) {
                    std::copy(pdsch_channel_coefficients_qrd[transmitter][receiver].begin(),
                              pdsch_channel_coefficients_qrd[transmitter][receiver].end(),
                              pdsch_channel_coefficients_sqrd[transmitter][receiver].begin());
                    std::copy(pdsch_channel_coefficients_qrd[transmitter][receiver].begin(),
                              pdsch_channel_coefficients_qrd[transmitter][receiver].end(),
                              pdsch_channel_coefficients_qrd_col_norm[transmitter][receiver].begin());
                }
            }

            asm volatile ("CPUID\n\t"
                          "RDTSC\n\t"
                          "mov %%edx, %0\n\t"
                          "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                    "%rax", "%rbx", "%rcx", "%rdx");

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

            asm volatile("RDTSCP\n\t"
                                     "mov %%edx, %0\n\t"
                                     "mov %%eax, %1\n\t"
                                     "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                                "%rax", "%rbx", "%rcx", "%rdx");

                        t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
                        t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

                        t_cumul[0] += t2 - t1; /// Update only for ZF

                        times[5][sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;

                        BOOST_LOG_TRIVIAL(trace) << "Extracting PDSCH channel coefficients [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

            /// Use the appropriate decoder
            if (encoding_type == diversity) {

                asm volatile ("CPUID\n\t"
                              "RDTSC\n\t"
                              "mov %%edx, %0\n\t"
                              "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                              "%rax", "%rbx", "%rcx", "%rdx");

                /// Use alamouti SFBC decoder
                mimo_transmit_diversity_decoder(pdsch_samples,
                                                pdsch_channel_coefficients,
                                                num_pdsch_re_per_slot,
                                                equalized_symbols[0].data() + (slot - 1) * num_pdsch_re_per_slot,
                                                num_tx_dmrs_ports,
                                                nb_rx_ports);

                asm volatile("RDTSCP\n\t"
                                         "mov %%edx, %0\n\t"
                                         "mov %%eax, %1\n\t"
                                         "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                                    "%rax", "%rbx", "%rcx", "%rdx");

                            t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
                            t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

                            t_cumul[0] += t2 - t1; /// Update only for 1 decoder

                            times[6][sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;

                            BOOST_LOG_TRIVIAL(trace) << "Alamouti decoding (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;
            }

            else if (encoding_type == vblast) {

#if defined(__AVX2__) and defined(AVX2_PROCESSING)

                asm volatile ("CPUID\n\t"
                              "RDTSC\n\t"
                              "mov %%edx, %0\n\t"
                              "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                        "%rax", "%rbx", "%rcx", "%rdx");

               call_vblast_sqrd_functions_avx2(pdsch_channel_coefficients_sqrd,
                                      pdsch_samples,
                                      num_pdsch_re_per_slot,
                                      equalized_symbols[0].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                      num_tx_dmrs_ports,
                                      nb_rx_ports,
                                      constellations[constellation_type],
                                      detected_symbol_indexes[0].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                      constellation_type);

                asm volatile("RDTSCP\n\t"
                             "mov %%edx, %0\n\t"
                             "mov %%eax, %1\n\t"
                             "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                        "%rax", "%rbx", "%rcx", "%rdx");

                t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
                t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

                t_cumul[1] += t2 - t1;

                times[6][sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;

                BOOST_LOG_TRIVIAL(trace) << "VBLAST SQRD AVX2 (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

                asm volatile ("CPUID\n\t"
                              "RDTSC\n\t"
                              "mov %%edx, %0\n\t"
                              "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                        "%rax", "%rbx", "%rcx", "%rdx");

                call_vblast_qrd_col_norm_functions_avx2(pdsch_channel_coefficients_qrd_col_norm,
                                                          pdsch_samples,
                                                          num_pdsch_re_per_slot,
                                                          equalized_symbols[1].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                          num_tx_dmrs_ports,
                                                          nb_rx_ports,
                                                          constellations[constellation_type],
                                                          detected_symbol_indexes[1].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                          constellation_type);

                asm volatile("RDTSCP\n\t"
                             "mov %%edx, %0\n\t"
                             "mov %%eax, %1\n\t"
                             "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                        "%rax", "%rbx", "%rcx", "%rdx");

                t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
                t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

                t_cumul[2] += t2 - t1;

                times[7][sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;


                BOOST_LOG_TRIVIAL(trace) << "VBLAST QRD col. norm. (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

                asm volatile ("CPUID\n\t"
                              "RDTSC\n\t"
                              "mov %%edx, %0\n\t"
                              "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                        "%rax", "%rbx", "%rcx", "%rdx");

                        call_vblast_zf_avx_functions(num_tx_dmrs_ports,
                                                     pdsch_samples,
                                                     pdsch_channel_coefficients,
                                                     num_pdsch_re_per_slot,
                                                     equalized_symbols[2].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                     nb_rx_ports);

                asm volatile("RDTSCP\n\t"
                             "mov %%edx, %0\n\t"
                             "mov %%eax, %1\n\t"
                             "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                        "%rax", "%rbx", "%rcx", "%rdx");

                t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
                t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

                t_cumul[0] += t2 - t1;

                times[8][sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;

                BOOST_LOG_TRIVIAL(trace) << "VBLAST ZF AVX2 (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

#elif defined(VBLAST_AVX512) and defined(AVX512_MIMO_SQRD)

                asm volatile ("CPUID\n\t"
                              "RDTSC\n\t"
                              "mov %%edx, %0\n\t"
                              "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                        "%rax", "%rbx", "%rcx", "%rdx");

                        mimo_vblast_sqrd_avx512(pdsch_channel_coefficients,
                                                  pdsch_samples,
                                                  num_pdsch_re_per_slot,
                                                  equalized_symbols.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                  num_tx_dmrs_ports,
                                                  nb_rx_ports,
                                                  constellations[constellation_type],
                                                  detected_symbol_indexes.data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                  constellation_type);

                asm volatile("RDTSCP\n\t"
                             "mov %%edx, %0\n\t"
                             "mov %%eax, %1\n\t"
                             "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                             "%rax", "%rbx", "%rcx", "%rdx");

                t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
                t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

                t_cumul += t2 - t1;

                times[5][sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;

                cout << "VBLAST SQRD AVX512 (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;
                BOOST_LOG_TRIVIAL(trace) << "VBLAST SQRD AVX512 (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;
#else
                asm volatile ("CPUID\n\t"
                              "RDTSC\n\t"
                              "mov %%edx, %0\n\t"
                              "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                              "%rax", "%rbx", "%rcx", "%rdx");

                /* SQRD */
                /*
                mimo_vblast_decoder_qr_decomp_modified(pdsch_channel_coefficients_qrd,
                                                       pdsch_samples,
                                                       num_pdsch_re_per_slot,
                                                       equalized_symbols[0].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                       num_tx_dmrs_ports,
                                                       nb_rx_ports,
                                                       constellations[constellation_type],
                                                       detected_symbol_indexes[0].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                       constellation_type); */

                call_vblast_sqrd_functions(pdsch_channel_coefficients_sqrd,
                                           pdsch_samples,
                                           num_pdsch_re_per_slot,
                                           equalized_symbols[0].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                           num_tx_dmrs_ports,
                                           nb_rx_ports,
                                           constellations[constellation_type],
                                           detected_symbol_indexes[0].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                           constellation_type);

                asm volatile("RDTSCP\n\t"
                             "mov %%edx, %0\n\t"
                             "mov %%eax, %1\n\t"
                             "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                             "%rax", "%rbx", "%rcx", "%rdx");

                t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
                t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

                t_cumul[1] += t2 - t1; /// Update only for 1 decoder

                times[6][sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;

                BOOST_LOG_TRIVIAL(trace) << "VBLAST SQRD (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

                asm volatile ("CPUID\n\t"
                              "RDTSC\n\t"
                              "mov %%edx, %0\n\t"
                              "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                        "%rax", "%rbx", "%rcx", "%rdx");

                /**
                mimo_vblast_qrd_col_norm(pdsch_channel_coefficients_qrd,
                                                  pdsch_samples,
                                                  num_pdsch_re_per_slot,
                                                  equalized_symbols[1].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                  num_tx_dmrs_ports,
                                                  nb_rx_ports,
                                                  constellations[constellation_type],
                                                  detected_symbol_indexes[1].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                  constellation_type); */

                call_vblast_qrd_col_norm_functions(pdsch_channel_coefficients_qrd_col_norm,
                                                   pdsch_samples,
                                                   num_pdsch_re_per_slot,
                                                   equalized_symbols[1].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                   num_tx_dmrs_ports,
                                                   nb_rx_ports,
                                                   constellations[constellation_type],
                                                   detected_symbol_indexes[1].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                                   constellation_type);

                asm volatile("RDTSCP\n\t"
                             "mov %%edx, %0\n\t"
                             "mov %%eax, %1\n\t"
                             "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                        "%rax", "%rbx", "%rcx", "%rdx");

                t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
                t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

                t_cumul[2] += t2 - t1; /// Update only for 1 decoder

                times[7][sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;

                BOOST_LOG_TRIVIAL(trace) << "VBLAST QRD col. norm. (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

                asm volatile("RDTSCP\n\t"
                             "mov %%edx, %0\n\t"
                             "mov %%eax, %1\n\t"
                             "CPUID\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                             "%rax", "%rbx", "%rcx", "%rdx");

                call_vblast_zf_functions(num_tx_dmrs_ports,
                                         pdsch_samples,
                                         pdsch_channel_coefficients,
                                         num_pdsch_re_per_slot,
                                         equalized_symbols[2].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                         nb_rx_ports);

                asm volatile("RDTSCP\n\t"
                             "mov %%edx, %0\n\t"
                             "mov %%eax, %1\n\t"
                             "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                             "%rax", "%rbx", "%rcx", "%rdx");

                t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
                t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

                t_cumul[0] += t2 - t1; /// Update only for 1 decoder

                times[8][sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;

                BOOST_LOG_TRIVIAL(trace) << "VBLAST ZF (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

                asm volatile ("CPUID\n\t"
                              "RDTSC\n\t"
                              "mov %%edx, %0\n\t"
                              "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                        "%rax", "%rbx", "%rcx", "%rdx");

                call_vblast_qrd_no_reordering(pdsch_channel_coefficients_qrd,
                                             pdsch_samples,
                                             num_pdsch_re_per_slot,
                                             equalized_symbols[3].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                             num_tx_dmrs_ports,
                                             nb_rx_ports,
                                             constellations[constellation_type],
                                             detected_symbol_indexes[3].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                                             constellation_type);

                asm volatile("RDTSCP\n\t"
                             "mov %%edx, %0\n\t"
                             "mov %%eax, %1\n\t"
                             "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                        "%rax", "%rbx", "%rcx", "%rdx");

                t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
                t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

                t_cumul[3] += t2 - t1; /// Update only for 1 decoder

                times[9][sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;

                BOOST_LOG_TRIVIAL(trace) << "VBLAST QRD no reordering (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;

#endif
            } else if (encoding_type == none) {

                /// Detect the symbols and export the symbol indexes
                asm volatile ("CPUID\n\t"
                              "RDTSC\n\t"
                              "mov %%edx, %0\n\t"
                              "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                        "%rax", "%rbx", "%rcx", "%rdx");

#if defined(__AVX2__) and defined(AVX2_PROCESSING)
                /// Use the zero-forcing decoder
                zero_forcing_avx2(pdsch_samples[0].data(),
                             pdsch_channel_coefficients[0][0].data(),
                             equalized_symbols[0].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                             num_pdsch_re_per_slot);
#else
                /// Use the zero-forcing decoder
                zero_forcing(pdsch_samples[0].data(),
                             pdsch_channel_coefficients[0][0].data(),
                             equalized_symbols[0].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                             num_pdsch_re_per_slot);
#endif

                asm volatile("RDTSCP\n\t"
                             "mov %%edx, %0\n\t"
                             "mov %%eax, %1\n\t"
                             "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                        "%rax", "%rbx", "%rcx", "%rdx");

                t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
                t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

                times[6][sim][slot - 1] = (t2 - t1)/TSC_FREQ * 1e6;
                t_cumul[0] += t2 - t1; /// Update only for ZF decoder

                BOOST_LOG_TRIVIAL(trace) << "ZF equalization (RDTSC/RDTSCP) [µs] : " << (t2 - t1)/TSC_FREQ * 1e6 << endl;
            }

            /// Detect the symbols and export the symbol indexes
            if(encoding_type == vblast) {
                asm volatile ("CPUID\n\t"
                              "RDTSC\n\t"
                              "mov %%edx, %0\n\t"
                              "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                        "%rax", "%rbx", "%rcx", "%rdx");

                ml_detector_tabs[constellation_type](
                        equalized_symbols[2].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                        detected_symbol_indexes[2].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                        num_pdsch_re_per_slot * num_tx_dmrs_ports);

                asm volatile("RDTSCP\n\t"
                             "mov %%edx, %0\n\t"
                             "mov %%eax, %1\n\t"
                             "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                        "%rax", "%rbx", "%rcx", "%rdx");

                t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
                t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

                times_ml_detection[sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;
                t_cumul[0] += t2 - t1; /// Update only for ZF decoder

                BOOST_LOG_TRIVIAL(trace) << "Ml detection for one slot ZF (RDTSC/RDTSCP) [µs] : "
                                         << (t2 - t1) / TSC_FREQ * 1e6 << endl;
            } else if (encoding_type == diversity) {
                asm volatile ("CPUID\n\t"
                              "RDTSC\n\t"
                              "mov %%edx, %0\n\t"
                              "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                        "%rax", "%rbx", "%rcx", "%rdx");

                ml_detector_tabs[constellation_type](
                        equalized_symbols[0].data() + (slot - 1) * num_pdsch_re_per_slot,
                        detected_symbol_indexes[0].data() + (slot - 1) * num_pdsch_re_per_slot,
                        num_pdsch_re_per_slot);

                asm volatile("RDTSCP\n\t"
                             "mov %%edx, %0\n\t"
                             "mov %%eax, %1\n\t"
                             "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                        "%rax", "%rbx", "%rcx", "%rdx");

                t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
                t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

                times_ml_detection[sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;
                t_cumul[0] += t2 - t1; /// Update only for ZF decoder

                BOOST_LOG_TRIVIAL(trace) << "Ml detection for one slot Alamouti (RDTSC/RDTSCP) [µs] : "
                                         << (t2 - t1) / TSC_FREQ * 1e6 << endl;
            } else if (encoding_type == none) {
                asm volatile ("CPUID\n\t"
                              "RDTSC\n\t"
                              "mov %%edx, %0\n\t"
                              "mov %%eax, %1\n\t": "=r" (cycles_high1), "=r" (cycles_low1)::
                        "%rax", "%rbx", "%rcx", "%rdx");

                ml_detector_tabs[constellation_type](
                        equalized_symbols[0].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                        detected_symbol_indexes[0].data() + (slot - 1) * num_pdsch_re_per_slot * num_tx_dmrs_ports,
                        num_pdsch_re_per_slot * num_tx_dmrs_ports);

                asm volatile("RDTSCP\n\t"
                             "mov %%edx, %0\n\t"
                             "mov %%eax, %1\n\t"
                             "CPUID\n\t": "=r" (cycles_high2), "=r" (cycles_low2)::
                        "%rax", "%rbx", "%rcx", "%rdx");

                t1 = (((uint64_t) cycles_high1 << 32) | cycles_low1);
                t2 = (((uint64_t) cycles_high2 << 32) | cycles_low2);

                times_ml_detection[sim][slot - 1] = (t2 - t1) / TSC_FREQ * 1e6;
                t_cumul[0] += t2 - t1; /// Update only for ZF decoder

                BOOST_LOG_TRIVIAL(trace) << "Ml detection for one slot ZF SISO (RDTSC/RDTSCP) [µs] : "
                                         << (t2 - t1) / TSC_FREQ * 1e6 << endl;
            }

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

            if(encoding_type == vblast) {
                for (int i = 0; i < 4; i++) {
                    times[10 + i][sim][slot - 1] = t_cumul[i] / TSC_FREQ * 1e6;
                }

                BOOST_LOG_TRIVIAL(trace) << "time to decode one slot SQRD (RDTSC/RDTSCP) [µs] : "
                                         << t_cumul[1] / TSC_FREQ * 1e6 << endl;

                BOOST_LOG_TRIVIAL(trace) << "time to decode one slot QRD col. norm. (RDTSC/RDTSCP) [µs] : "
                                         << t_cumul[2] / TSC_FREQ * 1e6 << endl;

                BOOST_LOG_TRIVIAL(trace) << "time to decode one slot ZF (RDTSC/RDTSCP) [µs] : "
                                         << t_cumul[0] / TSC_FREQ * 1e6 << endl;

                BOOST_LOG_TRIVIAL(trace) << "time to decode one slot QRD (RDTSC/RDTSCP) [µs] : "
                                         << t_cumul[3] / TSC_FREQ * 1e6 << endl;
            } else if (encoding_type == diversity) {
                times[10][sim][slot - 1] = t_cumul[0] / TSC_FREQ * 1e6;
                BOOST_LOG_TRIVIAL(trace) << "time to decode one slot Alamouti (RDTSC/RDTSCP) [µs] : "
                                         << times[10][sim][slot - 1] << endl;
            } else if (encoding_type == none) {
                times[10][sim][slot - 1] = t_cumul[0] / TSC_FREQ * 1e6;
                BOOST_LOG_TRIVIAL(trace) << "time to decode one slot ZF SISO (RDTSC/RDTSCP) [µs] : "
                                         << times[10][sim][slot - 1] << endl;
            }
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

        /// Compute the symbol error rate (if transmission and reception are done on the same usrp device & main function)
        if(encoding_type == vblast) {
            export1dVector(RES_PATH+"equalized_grid_sqrd_frame"+to_string(sim)+".txt", equalized_symbols[0].data(), buffer_size);
            export1dVector(RES_PATH+"equalized_grid_qrd_col_norm_frame"+to_string(sim)+".txt", equalized_symbols[1].data(), buffer_size);
            export1dVector(RES_PATH+"equalized_grid_zf_frame"+to_string(sim)+".txt", equalized_symbols[2].data(), buffer_size);
            export1dVector(RES_PATH+"equalized_grid_qrd_no_reordering_frame"+to_string(sim)+".txt", equalized_symbols[3].data(), buffer_size);

            export1dVector(RES_PATH+"decoded_grid_sqrd_frame"+to_string(sim)+".txt", detected_symbol_indexes[0].data(), buffer_size);
            export1dVector(RES_PATH+"decoded_grid_qrd_col_norm_frame"+to_string(sim)+".txt", detected_symbol_indexes[1].data(), buffer_size);
            export1dVector(RES_PATH+"decoded_grid_zf_frame"+to_string(sim)+".txt", detected_symbol_indexes[2].data(), buffer_size);
            export1dVector(RES_PATH+"decoded_grid_qrd_no_reordering_frame"+to_string(sim)+".txt", detected_symbol_indexes[3].data(), buffer_size);
            symbol_error_rates[sim][0] = symbol_error_rate(detected_symbol_indexes[0], sending_buffer_symbol_indexes);
            cout << "symbol error rate SQRD : " << symbol_error_rates[sim][0] << endl;
            BOOST_LOG_TRIVIAL(info) << "symbol error rate SQRD : " << symbol_error_rates[sim][0] << endl;
            symbol_error_rates[sim][1] = symbol_error_rate(detected_symbol_indexes[1], sending_buffer_symbol_indexes);
            cout << "symbol error rate QRD col. norm. : " << symbol_error_rates[sim][1] << endl;
            BOOST_LOG_TRIVIAL(info) << "symbol error rate QRD col. norm. : " << symbol_error_rates[sim][1] << endl;
            symbol_error_rates[sim][2] = symbol_error_rate(detected_symbol_indexes[2], sending_buffer_symbol_indexes);
            cout << "symbol error rate ZF : " << symbol_error_rates[sim][2] << endl;
            BOOST_LOG_TRIVIAL(info) << "symbol error rate ZF : " << symbol_error_rates[sim][2] << endl;
            symbol_error_rates[sim][3] = symbol_error_rate(detected_symbol_indexes[3], sending_buffer_symbol_indexes);
            cout << "symbol error rate QRD no reordering : " << symbol_error_rates[sim][3] << endl;
            BOOST_LOG_TRIVIAL(info) << "symbol error rate QRD no reordering : " << symbol_error_rates[sim][3] << endl;

            BOOST_LOG_TRIVIAL(info) << "\n" << endl;
        } else if (encoding_type == diversity) {
            export1dVector(RES_PATH+"equalized_grid_alamouti_frame"+to_string(sim)+".txt", equalized_symbols[0].data(), buffer_size);
            export1dVector(RES_PATH+"decoded_grid_alamouti_frame"+to_string(sim)+".txt", detected_symbol_indexes[0].data(), buffer_size);

            symbol_error_rates[sim][0] = symbol_error_rate(detected_symbol_indexes[0], sending_buffer_symbol_indexes);

            cout << "symbol error rate Alamouti : " << symbol_error_rates[sim][0] << "\n" << endl;
            BOOST_LOG_TRIVIAL(info) << "symbol error rate Alamouti : " << symbol_error_rates[sim][0] << "\n" << endl;
        } else if (encoding_type == none) {
            export1dVector(RES_PATH+"equalized_grid_siso_frame"+to_string(sim)+".txt", equalized_symbols[0].data(), buffer_size);
            export1dVector(RES_PATH+"decoded_grid_siso_frame"+to_string(sim)+".txt", detected_symbol_indexes[0].data(), buffer_size);

            symbol_error_rates[sim][0] = symbol_error_rate(detected_symbol_indexes[0], sending_buffer_symbol_indexes);

            cout << "symbol error rate ZF SISO : " << symbol_error_rates[sim][0] << endl;
            BOOST_LOG_TRIVIAL(info) << "symbol error rate ZF SISO : " << symbol_error_rates[sim][0] << endl;
            BOOST_LOG_TRIVIAL(info) << "\n" << endl;
        }

        /// Export pilot coefficients and interpolated coefficients
        /// Export averaged and interpolated coefs
        for (int transmitter = 0; transmitter < num_tx_dmrs_ports; transmitter++) {
            for (int receiver = 0; receiver < nb_rx_ports; receiver++) {
                exportGrid(RES_PATH+"tx" + to_string(transmitter + 1)
                            + "_rx" + to_string(receiver + 1)
                            + "_frame" + to_string(sim)
                            + "_pilots.txt",
                           pilot_coefficients, num_dmrs_symbols_per_slot, dmrs_sequence_size,
                           num_tx_dmrs_ports, transmitter, receiver);

                exportGrid(RES_PATH+"tx" + to_string(transmitter + 1)
                            + "_rx" + to_string(receiver + 1)
                            + "_frame" + to_string(sim)
                            + "_coefs.txt",
                           interp_coefficients[receiver][transmitter]);
            }
        }
    }

    cout << "############ Mean and Variances ############" << endl;

    // Compute the mean and variance of each execution time
    vector<double> means(times.size());
    vector<double> variance(times.size());
    vector<double> min(times.size());
    vector<double> max(times.size());
    vector<vector<double>> time_values(times.size());
    double mean_sync;
    double min_sync, max_sync;
    double variance_sync;
    double mean_ml_detection;
    double min_ml_detection = 0;
    double max_ml_detection = 0;
    double variance_ml_detection;
    double mean_fft = 0;
    double min_fft = 0;
    double max_fft = 0;
    double variance_fft = 0;

    for(int i = 0; i < times.size(); i++) {
        for (int j = 0; j < num_simulations; j++) {
            for(int k = 0; k < nb_slots - 1; k++) {
                means[i] += times[i][j][k];
                time_values[i].push_back(times[i][j][k]);
            }
        }
        means[i] /= num_simulations * (nb_slots - 1);

        /// Determine min and max
        min[i] = *(std::min_element(time_values[i].begin(), time_values[i].end()));
        max[i] = *(std::max_element(time_values[i].begin(), time_values[i].end()));
    }

    double temp;
    for(int i = 0; i < times.size(); i++) {
        for(int j = 0; j < num_simulations; j++) {
            for(int k = 0; k < nb_slots - 1; k++) {
                temp = (times[i][j][k] - means[i]);
                variance[i] += temp * temp;
            }
        }
        variance[i] /= num_simulations * (nb_slots - 1) - 1;
    }

    for (int j = 0; j < num_simulations; j++) {
        mean_sync += times_pss_sync[j];
    }
    mean_sync /= num_simulations;
    /// determine min and max of PSS sync
    min_sync = *(std::min_element(times_pss_sync.begin(), times_pss_sync.end()));
    max_sync = *(std::max_element(times_pss_sync.begin(), times_pss_sync.end()));

    for(int j = 0; j < num_simulations; j++) {
        temp = times_pss_sync[j] - mean_sync;
        variance_sync += temp * temp;
    }
    variance_sync /= num_simulations - 1;

    for (int i = 0; i < times_ml_detection.size(); i++) {
        for(int j = 0; j < times_ml_detection[0].size(); j++)
        mean_ml_detection += times_ml_detection[i][j];
    }
    mean_ml_detection /= times_ml_detection.size() * times_ml_detection[0].size();
    double temp_min, temp_max;
    for(int i = 0; i < times_ml_detection.size(); i++) {
        for (int j = 0; j < times_ml_detection[0].size(); j++) {
            temp = times_ml_detection[i][j] - mean_ml_detection;
            variance_ml_detection += temp * temp;
        }

        if(i == 0) {
            min_ml_detection = *(std::min_element(times_ml_detection[i].begin(), times_ml_detection[i].end()));
            max_ml_detection = *(std::max_element(times_ml_detection[i].begin(), times_ml_detection[i].end()));
        } else {
            /// determine min and max of ML detection
            temp_min = *(std::min_element(times_ml_detection[i].begin(), times_ml_detection[i].end()));
            temp_max = *(std::max_element(times_ml_detection[i].begin(), times_ml_detection[i].end()));
            if (temp_min < min_ml_detection) {
                min_ml_detection = temp_min;
            }
            if (temp_max > max_ml_detection) {
               max_ml_detection = temp_max;
            }
        }
    }
    variance_ml_detection /= times_ml_detection.size() * times_ml_detection[0].size() - 1;

    for (int i = 0; i < times_fft.size(); i++) {
            mean_fft += times_fft[i];
    }
    mean_fft /= times_fft.size();

    min_fft = *(std::min_element(times_fft.begin(), times_fft.end()));
    max_fft = *(std::max_element(times_fft.begin(), times_fft.end()));

    for(int i = 0; i < times_fft.size(); i++) {
        temp = times_fft[i] - mean_fft;
        variance_fft += temp * temp;
    }
    variance_fft /= times_fft.size() - 1;

#if defined(__AVX2__) and defined(AVX2_PROCESSING)

    cout << "Synchronization AVX2 mean [µs] : " << mean_sync << " / standard deviation [µs] : " << sqrt(variance_sync) << endl;
    cout << " Synchronization AVX2 min / max[µs] : " << min_sync << " / " << max_sync << endl;
    BOOST_LOG_TRIVIAL(trace) << "Synchronization AVX2 mean [µs] : " << mean_sync << " / standard deviation [µs] : " << sqrt(variance_sync) << endl;
    BOOST_LOG_TRIVIAL(trace) << "   Synchronization AVX2 min / max[µs] : " << min_sync << " / " << max_sync << endl;

    cout << "FFT mean [µs] : " << mean_fft << " / standard deviation [µs] : " << sqrt(variance_fft) << endl;
    cout << " FFT min / max[µs] : " << min_fft << " / " << max_fft << endl;
    BOOST_LOG_TRIVIAL(trace) << "FFT mean [µs] : " << mean_fft << " / standard deviation [µs] : " << sqrt(variance_fft) << endl;
    BOOST_LOG_TRIVIAL(trace) << " FFT min / max[µs] : " << min_fft << " / " << max_fft << endl;

    cout << "Getting PDSCH and DMRS samples mean [µs] : " << means[0] << " / standard deviation [µs] : " << sqrt(variance[0]) << endl;
    cout << " Getting PDSCH and DMRS samples min / max[µs] : " << min[0] << " / " << max[0] << endl;
    BOOST_LOG_TRIVIAL(trace) << "Getting PDSCH and DMRS samples mean [µs] : " << means[0] << " / standard deviation [µs] : " << sqrt(variance[0]) << endl;
    BOOST_LOG_TRIVIAL(trace) << " Getting PDSCH and DMRS samples min / max[µs] : " << min[0] << " / " << max[0] << endl;

    cout << "Channel estimation on pilots AVX2 mean [µs] : " << means[1] << " / standard deviation [µs] : " << sqrt(variance[1]) << endl;
    BOOST_LOG_TRIVIAL(trace) << "Channel estimation AVX2 on pilots on pilots mean [µs] : " << means[1] << " / standard deviation [µs] : " << sqrt(variance[1]) << endl;
    cout << " Channel estimation on pilots AVX2 min / max[µs] : " << min[1] << " / " << max[1] << endl;
    BOOST_LOG_TRIVIAL(trace) << " Channel estimation AVX2 on pilots min / max[µs] : " << min[1] << " / " << max[1] << endl;

    cout << "Channel coefficients interpolation SQRD AVX2 mean [µs] : " << means[2] << " / standard deviation [µs] : " << sqrt(variance[2]) << endl;
    BOOST_LOG_TRIVIAL(trace) << "Channel coefficients interpolation SQRD AVX2 mean [µs] : " << means[2] << " / standard deviation [µs] : " << sqrt(variance[3]) << endl;
    cout << " Channel coefficients interpolation SQRD AVX2 min / max[µs] : " << min[2] << " / " << max[2] << endl;
    BOOST_LOG_TRIVIAL(trace) << " Channel coefficients interpolation SQRD AVX2 min / max[µs] : " << min[2] << " / " << max[2] << endl;

    cout << "Channel coefficients interpolation AVX2 mean [µs] : " << means[3] << " / standard deviation [µs] : " << sqrt(variance[3]) << endl;
    BOOST_LOG_TRIVIAL(trace) << "Channel coefficients interpolation AVX2 mean [µs] : " << means[3] << " / standard deviation [µs] : " << sqrt(variance[3]) << endl;
    cout << " Channel coefficients interpolation AVX2 min / max[µs] : " << min[3] << " / " << max[3] << endl;
    BOOST_LOG_TRIVIAL(trace) << " Channel coefficients interpolation AVX2 min / max[µs] : " << min[3] << " / " << max[3] << endl;

    cout << "Extracting PDSCH channel coefficients SQRD mean [µs] : " << means[4] << " / standard deviation [µs] : "
         << sqrt(variance[4]) << endl;
    BOOST_LOG_TRIVIAL(trace) << "Extracting PDSCH channel coefficients SQRD mean [µs] : " << means[4]
                             << " / standard deviation [µs] : " << sqrt(variance[4]) << endl;
    cout << " Extracting PDSCH channel coefficients min / max[µs] : " << min[4] << " / " << max[4] << endl;
    BOOST_LOG_TRIVIAL(trace) << " Extracting PDSCH channel coefficients SQRD min / max[µs] : " << min[4] << " / "
                             << max[4] << endl;

    cout << "Extracting PDSCH channel coefficients mean [µs] : " << means[5] << " / standard deviation [µs] : " << sqrt(variance[5]) << endl;
    BOOST_LOG_TRIVIAL(trace) << "Extracting PDSCH channel coefficients mean [µs] : " << means[5] << " / standard deviation [µs] : " << sqrt(variance[5]) << endl;
    cout << " Extracting PDSCH channel coefficients min / max[µs] : " << min[5] << " / " << max[5] << endl;
    BOOST_LOG_TRIVIAL(trace) << " Extracting PDSCH channel coefficients min / max[µs] : " << min[5] << " / " << max[5] << endl;

    if(encoding_type == vblast) {
        cout << "VBLAST SQRD AVX2 mean [µs] : " << means[6] << " / standard deviation [µs] : " << sqrt(variance[6]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "VBLAST SQRD mean [µs] : " << means[6] << " / standard deviation [µs] : "
                                 << sqrt(variance[6]) << endl;
        cout << " VBLAST SQRD AVX2 min / max[µs] : " << min[6] << " / " << max[6] << endl;
        BOOST_LOG_TRIVIAL(trace) << "VBLAST SQRD AVX2 min / max[µs] : " << min[6] << " / " << max[6] << endl;

        cout << "VBLAST QRD col. norm. AVX2 mean [µs] : " << means[7] << " / standard deviation [µs] : " << sqrt(variance[7])
             << endl;
        BOOST_LOG_TRIVIAL(trace) << "VBLAST QRD col. norm. AVX2 mean [µs] : " << means[7] << " / standard deviation [µs] : "
                                 << sqrt(variance[7]) << endl;
        cout << " VBLAST QRD col. norm. AVX2 min / max[µs] : " << min[7] << " / " << max[7] << endl;
        BOOST_LOG_TRIVIAL(trace) << " VBLAST QRD col. norm. AVX2 min / max[µs] : " << min[7] << " / " << max[7] << endl;

        cout << "VBLAST ZF AVX2 mean [µs] : " << means[8] << " / standard deviation [µs] : " << sqrt(variance[8]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "VBLAST ZF AVX2 mean [µs] : " << means[8] << " / standard deviation [µs] : "
                                 << sqrt(variance[8]) << endl;
        cout << " VBLAST ZF min / max[µs] : " << min[8] << " / " << max[8] << endl;
        BOOST_LOG_TRIVIAL(trace) << " VBLAST ZF AVX2 min / max[µs] : " << min[8] << " / " << max[8] << endl;

        cout << "VBLAST QRD AVX2 no reordering mean [µs] : " << means[9] << " / standard deviation [µs] : "
             << sqrt(variance[9]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "VBLAST QRD AVX2 no reordering mean [µs] : " << means[9]
                                 << " / standard deviation [µs] : " << sqrt(variance[9]) << endl;
        cout << " VBLAST QRD AVX2 no reordering min / max[µs] : " << min[9] << " / " << max[9] << endl;
        BOOST_LOG_TRIVIAL(trace) << " VBLAST QRD AVX2 no reordering min / max[µs] : " << min[9] << " / " << max[9] << endl;

        cout << "ML detection mean [µs] : " << mean_ml_detection << "/ standard deviation [µs] : "
             << sqrt(variance_ml_detection) << endl;
        BOOST_LOG_TRIVIAL(trace) << "ML detection mean [µs] : " << mean_ml_detection << "/ standard deviation [µs] : "
                                 << sqrt(variance_ml_detection) << endl;
        cout << " ML detection mean min / max[µs] : " << min_ml_detection << " / " << max_ml_detection << endl;
        BOOST_LOG_TRIVIAL(trace) << " ML detection mean min / max[µs] : " << min_ml_detection << " / "
                                 << max_ml_detection << endl;

        cout << "Time to decode one slot SQRD mean [µs] : " << means[11] << " / standard deviation [µs] : "
             << sqrt(variance[11]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Time to decode one slot SQRD mean [µs] : " << means[11]
                                 << " / standard deviation [µs] : " << sqrt(variance[11]) << endl;
        cout << " Time to decode one slot min / max[µs] : " << min[11] << " / " << max[11] << endl;
        BOOST_LOG_TRIVIAL(trace) << " Time to decode one slot min / max[µs] : " << min[11] << " / " << max[11] << endl;

        cout << "Time to decode one slot QRD col. norm. mean [µs] : " << means[12] << " / standard deviation [µs] : "
             << sqrt(variance[12]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Time to decode one slot QRD col. norm. mean [µs] : " << means[12]
                                 << " / standard deviation [µs] : " << sqrt(variance[12]) << endl;
        cout << " Time to decode one slot QRD col. norm. min / max[µs] : " << min[12] << " / " << max[12] << endl;
        BOOST_LOG_TRIVIAL(trace) << " Time to decode one slot QRD col. norm. min / max[µs] : " << min[12] << " / "
                                 << max[12] << endl;

        cout << "Time to decode one slot ZF mean [µs] : " << means[10] << " / standard deviation [µs] : "
             << sqrt(variance[10]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Time to decode one ZF slot mean [µs] : " << means[10]
                                 << " / standard deviation [µs] : " << sqrt(variance[10]) << endl;
        cout << " Time to decode one slot ZF min / max[µs] : " << min[10] << " / " << max[10] << endl;
        BOOST_LOG_TRIVIAL(trace) << " Time to decode one slot ZF min / max[µs] : " << min[10] << " / " << max[10]
                                 << endl;

        cout << "Time to decode one slot QRD mean [µs] : " << means[13] << " / standard deviation [µs] : "
             << sqrt(variance[13]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Time to decode one slot QRD mean [µs] : " << means[13]
                                 << " / standard deviation [µs] : " << sqrt(variance[13]) << endl;
        cout << " Time to decode one slot QRD min / max[µs] : " << min[13] << " / " << max[13] << endl;
        BOOST_LOG_TRIVIAL(trace) << " Time to decode one slot QRD min / max[µs] : " << min[13] << " / " << max[13]
                                 << endl;
    } else if (encoding_type == diversity) {
        cout << "Alamouti decoding [µs] : " << means[6] << " / standard deviation [µs] : " << sqrt(variance[6]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Alamouti decoding mean [µs] : " << means[6] << " / standard deviation [µs] : "
                                 << sqrt(variance[6]) << endl;
        cout << " Alamouti decoding min / max[µs] : " << min[6] << " / " << max[6] << endl;
        BOOST_LOG_TRIVIAL(trace) << " Alamouti decoding min / max[µs] : " << min[6] << " / " << max[6] << endl;

        cout << "ML detection mean [µs] : " << mean_ml_detection << "/ standard deviation [µs] : "
             << sqrt(variance_ml_detection) << endl;
        BOOST_LOG_TRIVIAL(trace) << "ML detection mean [µs] : " << mean_ml_detection << "/ standard deviation [µs] : "
                                 << sqrt(variance_ml_detection) << endl;
        cout << " ML detection mean min / max[µs] : " << min_ml_detection << " / " << max_ml_detection << endl;
        BOOST_LOG_TRIVIAL(trace) << " ML detection mean min / max[µs] : " << min_ml_detection << " / "
                                 << max_ml_detection << endl;

        cout << "Time to decode one slot Alamouti mean [µs] : " << means[10] << " / standard deviation [µs] : "
             << sqrt(variance[10]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Time to decode one slot Alamouti mean [µs] : " << means[10]
                                 << " / standard deviation [µs] : " << sqrt(variance[10]) << endl;
        cout << " Time to decode one slot Alamouti min / max[µs] : " << min[10] << " / " << max[10] << endl;
        BOOST_LOG_TRIVIAL(trace) << " Time to decode one slot Alamouti min / max[µs] : " << min[10] << " / " << max[10]
                                 << endl;

    } else if (encoding_type == none) {
        cout << "ZF SISO AVX2 equalization [µs] : " << means[6] << " / standard deviation [µs] : " << sqrt(variance[6]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "ZF SISO AVX2 equalization [µs] : " << means[6] << " / standard deviation [µs] : "
                                 << sqrt(variance[6]) << endl;
        cout << " ZF SISO AVX2 equalization min / max[µs] : " << min[6] << " / " << max[6] << endl;
        BOOST_LOG_TRIVIAL(trace) << " ZF SISO AVX2 equalization min / max[µs] : " << min[6] << " / " << max[6] << endl;

        cout << "ML detection mean [µs] : " << mean_ml_detection << "/ standard deviation [µs] : "
             << sqrt(variance_ml_detection) << endl;
        BOOST_LOG_TRIVIAL(trace) << "ML detection mean [µs] : " << mean_ml_detection << "/ standard deviation [µs] : "
                                 << sqrt(variance_ml_detection) << endl;
        cout << " ML detection mean min / max[µs] : " << min_ml_detection << " / " << max_ml_detection << endl;
        BOOST_LOG_TRIVIAL(trace) << " ML detection mean min / max[µs] : " << min_ml_detection << " / "
                                 << max_ml_detection << endl;

        cout << "Time to decode one slot ZF SISO AVX2 mean [µs] : " << means[10] << " / standard deviation [µs] : "
             << sqrt(variance[10]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Time to decode one slot ZF SISO AVX2 mean [µs] : " << means[10]
                                 << " / standard deviation [µs] : " << sqrt(variance[10]) << endl;
        cout << " Time to decode one slot ZF SISO AVX2 min / max[µs] : " << min[10] << " / " << max[10] << endl;
        BOOST_LOG_TRIVIAL(trace) << " Time to decode one slot ZF SISO AVX2 min / max[µs] : " << min[10] << " / " << max[10]
                                 << endl;
    }
#else

    cout << "Synchronization mean [µs] : " << mean_sync << " / standard deviation [µs] : " << sqrt(variance_sync)
         << endl;
    cout << " Synchronization min / max[µs] : " << min_sync << " / " << max_sync << endl;
    BOOST_LOG_TRIVIAL(trace) << "Synchronization mean [µs] : " << mean_sync << " / standard deviation [µs] : "
                             << sqrt(variance_sync) << endl;
    BOOST_LOG_TRIVIAL(trace) << "   Synchronization min / max[µs] : " << min_sync << " / " << max_sync << endl;

    cout << "FFT mean [µs] : " << mean_fft << " / standard deviation [µs] : " << sqrt(variance_fft) << endl;
    cout << " FFT min / max[µs] : " << min_fft << " / " << max_fft << endl;
    BOOST_LOG_TRIVIAL(trace) << "FFT mean [µs] : " << mean_fft << " / standard deviation [µs] : " << sqrt(variance_fft) << endl;
    BOOST_LOG_TRIVIAL(trace) << " FFT min / max[µs] : " << min_fft << " / " << max_fft << endl;

    cout << "Getting PDSCH and DMRS samples mean [µs] : " << means[0] << " / standard deviation [µs] : "
         << sqrt(variance[0]) << endl;
    cout << " Getting PDSCH and DMRS samples min / max[µs] : " << min[0] << " / " << max[0] << endl;
    BOOST_LOG_TRIVIAL(trace) << "Getting PDSCH and DMRS samples mean [µs] : " << means[0]
                             << " / standard deviation [µs] : " << sqrt(variance[0]) << endl;
    BOOST_LOG_TRIVIAL(trace) << " Getting PDSCH and DMRS samples min / max[µs] : " << min[0] << " / " << max[0]
                             << endl;

    cout << "Channel estimation on pilots mean [µs] : " << means[1] << " / standard deviation [µs] : "
         << sqrt(variance[1]) << endl;
    BOOST_LOG_TRIVIAL(trace) << "Channel estimation on pilots on pilots mean [µs] : " << means[1]
                             << " / standard deviation [µs] : " << sqrt(variance[1]) << endl;
    cout << " Channel estimation on pilots min / max[µs] : " << min[1] << " / " << max[1] << endl;
    BOOST_LOG_TRIVIAL(trace) << " Channel estimation on pilots min / max[µs] : " << min[1] << " / " << max[1]
                             << endl;

    cout << "Channel coefficients interpolation SQRD mean [µs] : " << means[2] << " / standard deviation [µs] : " << sqrt(variance[2]) << endl;
    BOOST_LOG_TRIVIAL(trace) << "Channel coefficients interpolation SQRD mean [µs] : " << means[2] << " / standard deviation [µs] : " << sqrt(variance[3]) << endl;
    cout << " Channel coefficients interpolation SQRD min / max[µs] : " << min[2] << " / " << max[2] << endl;
    BOOST_LOG_TRIVIAL(trace) << " Channel coefficients interpolation SQRD min / max[µs] : " << min[2] << " / " << max[2] << endl;

    cout << "Channel coefficients interpolation mean [µs] : " << means[3] << " / standard deviation [µs] : "
         << sqrt(variance[3]) << endl;
    BOOST_LOG_TRIVIAL(trace) << "Channel coefficients interpolation mean [µs] : " << means[3]
                             << " / standard deviation [µs] : " << sqrt(variance[3]) << endl;
    cout << " Channel coefficients interpolation min / max[µs] : " << min[3] << " / " << max[3] << endl;
    BOOST_LOG_TRIVIAL(trace) << " Channel coefficients interpolation min / max[µs] : " << min[3] << " / " << max[3]
                             << endl;

    cout << "Extracting PDSCH channel coefficients SQRD mean [µs] : " << means[4] << " / standard deviation [µs] : "
         << sqrt(variance[4]) << endl;
    BOOST_LOG_TRIVIAL(trace) << "Extracting PDSCH channel coefficients SQRD mean [µs] : " << means[4]
                             << " / standard deviation [µs] : " << sqrt(variance[4]) << endl;
    cout << " Extracting PDSCH channel coefficients min / max[µs] : " << min[4] << " / " << max[4] << endl;
    BOOST_LOG_TRIVIAL(trace) << " Extracting PDSCH channel coefficients SQRD min / max[µs] : " << min[4] << " / "
                             << max[4] << endl;

    cout << "Extracting PDSCH channel coefficients mean [µs] : " << means[5] << " / standard deviation [µs] : "
         << sqrt(variance[5]) << endl;
    BOOST_LOG_TRIVIAL(trace) << "Extracting PDSCH channel coefficients mean [µs] : " << means[5]
                             << " / standard deviation [µs] : " << sqrt(variance[5]) << endl;
    cout << " Extracting PDSCH channel coefficients min / max[µs] : " << min[5] << " / " << max[5] << endl;
    BOOST_LOG_TRIVIAL(trace) << " Extracting PDSCH channel coefficients min / max[µs] : " << min[5] << " / "
                             << max[5] << endl;

    if(encoding_type == vblast) {
        cout << "VBLAST SQRD mean [µs] : " << means[6] << " / standard deviation [µs] : " << sqrt(variance[6]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "VBLAST SQRD mean [µs] : " << means[6] << " / standard deviation [µs] : "
                                 << sqrt(variance[6]) << endl;
        cout << " VBLAST SQRD min / max[µs] : " << min[6] << " / " << max[6] << endl;
        BOOST_LOG_TRIVIAL(trace) << "VBLAST SQRD min / max[µs] : " << min[6] << " / " << max[6] << endl;

        cout << "VBLAST QRD col. norm. mean [µs] : " << means[7] << " / standard deviation [µs] : " << sqrt(variance[7])
             << endl;
        BOOST_LOG_TRIVIAL(trace) << "VBLAST QRD col. norm. mean [µs] : " << means[7] << " / standard deviation [µs] : "
                                 << sqrt(variance[7]) << endl;
        cout << " VBLAST QRD col. norm. / max[µs] : " << min[7] << " / " << max[7] << endl;
        BOOST_LOG_TRIVIAL(trace) << " VBLAST QRD col. norm. min / max[µs] : " << min[7] << " / " << max[7] << endl;

        cout << "VBLAST ZF mean [µs] : " << means[8] << " / standard deviation [µs] : " << sqrt(variance[8]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "VBLAST ZF mean [µs] : " << means[8] << " / standard deviation [µs] : "
                                 << sqrt(variance[8]) << endl;
        cout << " VBLAST ZF min / max[µs] : " << min[8] << " / " << max[8] << endl;
        BOOST_LOG_TRIVIAL(trace) << " VBLAST ZF min / max[µs] : " << min[8] << " / " << max[8] << endl;

        cout << "VBLAST QRD no reordering mean [µs] : " << means[9] << " / standard deviation [µs] : "
             << sqrt(variance[9]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "VBLAST QRD no reordering mean [µs] : " << means[9]
                                 << " / standard deviation [µs] : " << sqrt(variance[9]) << endl;
        cout << " VBLAST QRD no reordering min / max[µs] : " << min[9] << " / " << max[9] << endl;
        BOOST_LOG_TRIVIAL(trace) << " VBLAST QRD no reordering min / max[µs] : " << min[9] << " / " << max[9] << endl;

        cout << "ML detection mean [µs] : " << mean_ml_detection << "/ standard deviation [µs] : "
             << sqrt(variance_ml_detection) << endl;
        BOOST_LOG_TRIVIAL(trace) << "ML detection mean [µs] : " << mean_ml_detection << "/ standard deviation [µs] : "
                                 << sqrt(variance_ml_detection) << endl;
        cout << " ML detection mean min / max[µs] : " << min_ml_detection << " / " << max_ml_detection << endl;
        BOOST_LOG_TRIVIAL(trace) << " ML detection mean min / max[µs] : " << min_ml_detection << " / "
                                 << max_ml_detection << endl;

        cout << "Time to decode one slot SQRD mean [µs] : " << means[11] << " / standard deviation [µs] : "
             << sqrt(variance[11]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Time to decode one slot SQRD mean [µs] : " << means[11]
                                 << " / standard deviation [µs] : " << sqrt(variance[11]) << endl;
        cout << " Time to decode one slot min / max[µs] : " << min[11] << " / " << max[11] << endl;
        BOOST_LOG_TRIVIAL(trace) << " Time to decode one slot min / max[µs] : " << min[11] << " / " << max[11] << endl;

        cout << "Time to decode one slot QRD col. norm. mean [µs] : " << means[12] << " / standard deviation [µs] : "
             << sqrt(variance[12]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Time to decode one slot QRD col. norm. mean [µs] : " << means[12]
                                 << " / standard deviation [µs] : " << sqrt(variance[12]) << endl;
        cout << " Time to decode one slot QRD col. norm. min / max[µs] : " << min[12] << " / " << max[12] << endl;
        BOOST_LOG_TRIVIAL(trace) << " Time to decode one slot QRD col. norm. min / max[µs] : " << min[12] << " / "
                                 << max[12] << endl;

        cout << "Time to decode one slot ZF mean [µs] : " << means[10] << " / standard deviation [µs] : "
             << sqrt(variance[10]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Time to decode one ZF slot mean [µs] : " << means[10]
                                 << " / standard deviation [µs] : " << sqrt(variance[10]) << endl;
        cout << " Time to decode one slot ZF min / max[µs] : " << min[10] << " / " << max[10] << endl;
        BOOST_LOG_TRIVIAL(trace) << " Time to decode one slot ZF min / max[µs] : " << min[10] << " / " << max[10]
                                 << endl;

        cout << "Time to decode one slot QRD mean [µs] : " << means[13] << " / standard deviation [µs] : "
             << sqrt(variance[13]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Time to decode one slot QRD mean [µs] : " << means[13]
                                 << " / standard deviation [µs] : " << sqrt(variance[13]) << endl;
        cout << " Time to decode one slot QRD min / max[µs] : " << min[13] << " / " << max[13] << endl;
        BOOST_LOG_TRIVIAL(trace) << " Time to decode one slot QRD min / max[µs] : " << min[13] << " / " << max[13]
                                 << endl;
    } else if (encoding_type == diversity) {
        cout << "Alamouti decoding [µs] : " << means[6] << " / standard deviation [µs] : " << sqrt(variance[6]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Alamouti decoding mean [µs] : " << means[6] << " / standard deviation [µs] : "
                                 << sqrt(variance[6]) << endl;
        cout << " Alamouti decoding min / max[µs] : " << min[6] << " / " << max[6] << endl;
        BOOST_LOG_TRIVIAL(trace) << " Alamouti decoding min / max[µs] : " << min[6] << " / " << max[6] << endl;

        cout << "ML detection mean [µs] : " << mean_ml_detection << "/ standard deviation [µs] : "
             << sqrt(variance_ml_detection) << endl;
        BOOST_LOG_TRIVIAL(trace) << "ML detection mean [µs] : " << mean_ml_detection << "/ standard deviation [µs] : "
                                 << sqrt(variance_ml_detection) << endl;
        cout << " ML detection mean min / max[µs] : " << min_ml_detection << " / " << max_ml_detection << endl;
        BOOST_LOG_TRIVIAL(trace) << " ML detection mean min / max[µs] : " << min_ml_detection << " / "
                                 << max_ml_detection << endl;

        cout << "Time to decode one slot Alamouti mean [µs] : " << means[10] << " / standard deviation [µs] : "
             << sqrt(variance[10]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Time to decode one slot Alamouti mean [µs] : " << means[10]
                                 << " / standard deviation [µs] : " << sqrt(variance[10]) << endl;
        cout << " Time to decode one slot Alamouti min / max[µs] : " << min[10] << " / " << max[10] << endl;
        BOOST_LOG_TRIVIAL(trace) << " Time to decode one slot Alamouti min / max[µs] : " << min[10] << " / " << max[10]
                                 << endl;

    } else if (encoding_type == none) {
        cout << "ZF SISO equalization [µs] : " << means[6] << " / standard deviation [µs] : " << sqrt(variance[6]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "ZF SISO equalization [µs] : " << means[6] << " / standard deviation [µs] : "
                                 << sqrt(variance[6]) << endl;
        cout << " ZF SISO equalization min / max[µs] : " << min[6] << " / " << max[6] << endl;
        BOOST_LOG_TRIVIAL(trace) << " ZF SISO equalization min / max[µs] : " << min[6] << " / " << max[6] << endl;

        cout << "ML detection mean [µs] : " << mean_ml_detection << "/ standard deviation [µs] : "
             << sqrt(variance_ml_detection) << endl;
        BOOST_LOG_TRIVIAL(trace) << "ML detection mean [µs] : " << mean_ml_detection << "/ standard deviation [µs] : "
                                 << sqrt(variance_ml_detection) << endl;
        cout << " ML detection mean min / max[µs] : " << min_ml_detection << " / " << max_ml_detection << endl;
        BOOST_LOG_TRIVIAL(trace) << " ML detection mean min / max[µs] : " << min_ml_detection << " / "
                                 << max_ml_detection << endl;

        cout << "Time to decode one slot ZF SISO mean [µs] : " << means[10] << " / standard deviation [µs] : "
             << sqrt(variance[10]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Time to decode one slot ZF SISO mean [µs] : " << means[10]
                                 << " / standard deviation [µs] : " << sqrt(variance[10]) << endl;
        cout << " Time to decode one slot Alamouti min / max[µs] : " << min[10] << " / " << max[10] << endl;
        BOOST_LOG_TRIVIAL(trace) << " Time to decode one slot Alamouti min / max[µs] : " << min[10] << " / " << max[10]
                                 << endl;

    }
#endif

    vector<vector<double>> symbol_error_rates_realigned(symbol_error_rates[0].size(), vector<double> (symbol_error_rates.size()));
    for(int i = 0; i < symbol_error_rates.size(); i++) {
        for (int j = 0; j < symbol_error_rates[0].size(); j++) {
            symbol_error_rates_realigned[j][i] = symbol_error_rates[i][j];
        }
    }

    vector<vector<double>::iterator> max_ser(symbol_error_rates_realigned.size());
    vector<vector<double>::iterator> min_ser(symbol_error_rates_realigned.size());
    for (int i = 0; i < symbol_error_rates_realigned.size(); i++) {
        for (int j = 0; j < symbol_error_rates_realigned[0].size(); j++) {
            max_ser[i] = std::max_element(symbol_error_rates_realigned[i].begin(), symbol_error_rates_realigned[i].end());
            min_ser[i] = std::min_element(symbol_error_rates_realigned[i].begin(), symbol_error_rates_realigned[i].end());
        }
    }

    vector<int> num_non_decoded_frames(symbol_error_rates_realigned.size());
    for(int i = 0; i < symbol_error_rates_realigned.size(); i++) {
        for(int j = 0; j < symbol_error_rates_realigned[0].size(); j++) {
            if (symbol_error_rates_realigned[i][j] > 0.1) {
                num_non_decoded_frames[i] += 1;
            }
        }
    }

    if(encoding_type == vblast) {
        cout << "Max. symbol error rate SQRD = " << *(max_ser[0]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Max. symbol error rate SQRD = " << *(max_ser[0]) << endl;
        cout << "Min. symbol error rate SQRD = " << *(min_ser[0]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Min. symbol error rate SQRD = " << *(min_ser[0]) << endl;

        cout << "Max. symbol error rate QRD col. norm. = " << *(max_ser[1]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Max. symbol error rate QRD col. norm. = " << *(max_ser[1]) << endl;
        cout << "Min. symbol error rate QRD col. norm. = " << *(min_ser[1]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Min. symbol error rate QRD col. norm. = " << *(min_ser[1]) << endl;

        cout << "Max. symbol error rate ZF = " << *(max_ser[2]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Max. symbol error rate ZF = " << *(max_ser[2]) << endl;
        cout << "Min. symbol error rate ZF = " << *(min_ser[2]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Min. symbol error rate ZF = " << *(min_ser[2]) << endl;

        cout << "Max. symbol error rate QRD no reordering = " << *(max_ser[3]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Max. symbol error rate QRD no reordering = " << *(max_ser[3]) << endl;
        cout << "Min. symbol error rate QRD no reordering = " << *(min_ser[3]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Min. symbol error rate QR no reordering = " << *(min_ser[3]) << endl;

        cout << "Number of decoded frames SQRD : " << num_simulations - num_non_decoded_frames[0] << endl;
        BOOST_LOG_TRIVIAL(trace) << "Number of decoded frames SQRD : " << num_simulations - num_non_decoded_frames[0] << endl;
        cout << "Number of decoded frames QRD col. norm. : " << num_simulations - num_non_decoded_frames[1] << endl;
        BOOST_LOG_TRIVIAL(trace) << "Number of decoded frames QRD col. norm. : " << num_simulations - num_non_decoded_frames[1] << endl;
        cout << "Number of decoded frames ZF : " << num_simulations - num_non_decoded_frames[2] << endl;
        BOOST_LOG_TRIVIAL(trace) << "Number of decoded frames ZF : " << num_simulations - num_non_decoded_frames[2] << endl;
        cout << "Number of decoded frames QRD no reordering : " << num_simulations - num_non_decoded_frames[3] << endl;
        BOOST_LOG_TRIVIAL(trace) << "Number of decoded frames QRD no reordering : " << num_simulations - num_non_decoded_frames[3] << endl;

    } else if (encoding_type == diversity) {
        cout << "Max. symbol error rate Alamouti = " << *(max_ser[0]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Max. symbol error rate Alamouti = " << *(max_ser[0]) << endl;
        cout << "Min. symbol error rate Alamouti = " << *(min_ser[0]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Min. symbol error rate Alamouti = " << *(min_ser[0]) << endl;

        cout << "Number of decoded frames Alamouti : " << num_simulations - num_non_decoded_frames[0] << endl;
        BOOST_LOG_TRIVIAL(trace) << "Number of decoded frames Alamouti : " << num_simulations - num_non_decoded_frames[0] << endl;
    } else if (encoding_type == none) {
        cout << "Max. symbol error rate ZF SISO = " << *(max_ser[0]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Max. symbol error rate ZF SISO = " << *(max_ser[0]) << endl;
        cout << "Min. symbol error rate ZF SISO = " << *(min_ser[0]) << endl;
        BOOST_LOG_TRIVIAL(trace) << "Min. symbol error rate ZF SISO = " << *(min_ser[0]) << endl;

        cout << "Number of decoded frames ZF SISO : " << num_simulations - num_non_decoded_frames[0] << endl;
        BOOST_LOG_TRIVIAL(trace) << "Number of decoded frames ZF SISO : " << num_simulations - num_non_decoded_frames[0] << endl;
    }

    /************************************ Perform multiple simulations END ***********************************/

    usrp_tx_ptr->stop_tx_rx_threads();

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

    return EXIT_SUCCESS;
}

/// PDSCH equalization for SISO case
void zero_forcing(std::complex<float> * pdsch_samples_,
                  std::complex<float> * pdsch_channel_coefficients_,
                  std::complex<float> * equalized_symbols_,
                  int num_re_pdsch_) {

    /// Use the zero-forcing decoder
    for (int re = 0; re < num_re_pdsch_; re++) {
    equalized_symbols_[re] = pdsch_samples_[re] / pdsch_channel_coefficients_[re];
    }
}

void zero_forcing_avx2(complex<float> * pdsch_samples_,
                       complex<float> * pdsch_channel_coefficients_,
                       complex<float> * equalized_symbols_,
                       int num_re_pdsch_) {

    __m256 pdsch_samples_vec;
    __m256 pdsch_channel_coefficients_vec;
    __m256 equalized_symbols_vec;
    __m256 vec1, vec2;
    __m256 conj_vec = {1, -1, 1, -1, 1, -1, 1, -1};

    for(int re = 0; re < num_re_pdsch_; re+= 4) {
        pdsch_samples_vec = _mm256_loadu_ps((float *) &pdsch_samples_[re]);
        pdsch_channel_coefficients_vec = _mm256_loadu_ps((float *) &pdsch_channel_coefficients_[re]);
        vec1 = _mm256_mul_ps(pdsch_channel_coefficients_vec, pdsch_samples_vec);
        vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(pdsch_channel_coefficients_vec, conj_vec), 0b10110001), pdsch_samples_vec);
        pdsch_channel_coefficients_vec = compute_norm_m256(pdsch_channel_coefficients_vec);
        equalized_symbols_vec = _mm256_permute_ps(_mm256_hadd_ps(vec1, vec2), 0b11011000);
        equalized_symbols_vec = _mm256_div_ps(equalized_symbols_vec, pdsch_channel_coefficients_vec);
        //_mm256_stream_ps((float *) &equalized_symbols_[re], equalized_symbols_vec);
        _mm256_storeu_ps((float *) &equalized_symbols_[re], equalized_symbols_vec);
    }
}