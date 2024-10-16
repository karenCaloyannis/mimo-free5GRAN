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

    This file contains equalization algorithms. synchronize_slots and ml_detectors
    functions are reused from free5GRAN and are modified.
*/

#include "channel_estimation.h"

using namespace std;

//extern int slot_number_static;

ml_detector_complex_pointer ml_detector_complex[2] = {
        ml_detector_qpsk,
        ml_detector_bpsk
};

ml_detector_vectors_pointer ml_detector_vectors[2] = {
    ml_detector_qpsk,
    ml_detector_bpsk
};

ml_detector_tabs_pointer ml_detector_tabs[2] = {
        ml_detector_qpsk,
        ml_detector_bpsk
};

ml_detector_complex_inline_ptr ml_detector_inline[2] = {
        ml_detector_qpsk,
        ml_detector_bpsk
};


void compute_sic_order(vector<complex<float>> pdsch_channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                       vector<float> squared_norms_[MAX_RX_PORTS][MAX_TX_PORTS],
                       vector<vector<float>> &columns_norms,
                       vector<vector<int>> &sic_orders_,
                       int nb_rx_,
                       int nb_tx_,
                       int num_pdsch_re_,
                       int num_cdm_groups_without_data_) {

    //std::chrono::steady_clock::time_point t1, t2;

    //t1 = std::chrono::steady_clock::now();

    for(int rx = 0; rx < nb_rx_; rx++) {
        for(int tx = 0; tx < nb_tx_; tx++) {

            /// compute the norms of the coefficients for each subcarrier
            for(int re = 0; re < num_pdsch_re_; re++) {
                squared_norms_[rx][tx][re] = pdsch_channel_coefficients_[rx][tx][re].real() * pdsch_channel_coefficients_[rx][tx][re].real() +
                                             pdsch_channel_coefficients_[rx][tx][re].imag() * pdsch_channel_coefficients_[rx][tx][re].imag();
            }

            /**
            for(int re = 0; re < num_pdsch_re_; re++) {
                squared_norms_[rx][tx][re] *= squared_norms_[rx][tx][re];
            } */
        }
    }

    //t2 = std::chrono::steady_clock::now();
    //cout << "Time to compute the squared norms [ns] : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

    //t1 = std::chrono::steady_clock::now();

    /// Compute the column norms for each RE
    for(int tx = 0; tx < nb_tx_; tx++) {
        for(int re = 0; re < num_pdsch_re_; re++) {
            columns_norms[re][tx] = squared_norms_[0][tx][re];
        }
    }
    for(int tx = 0; tx < nb_tx_; tx++) {
        for(int rx = 1; rx < nb_rx_; rx++) {
            for(int re = 0; re < num_pdsch_re_; re++) {
                columns_norms[re][tx] += squared_norms_[rx][tx][re];
            }
        }
    }

    //t2 = std::chrono::steady_clock::now();
    //cout << "Time to compute the column norms [ns] : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

    //t1 = std::chrono::steady_clock::now();

    /// Compute the SIC orders for each RE
    for(int re = 0; re < num_pdsch_re_; re++) {
        iota(sic_orders_[re].begin(), sic_orders_[re].begin() + nb_tx_, 0);
        std::sort(sic_orders_[re].begin(), sic_orders_[re].end(), [&columns_norms, &re](int i1, int i2)
        {
            return columns_norms[re][i1] < columns_norms[re][i2];
        });
    }

    //t2 = std::chrono::steady_clock::now();
    //cout << "Time to compute the SIC orders (std::sort) [ns] : " << std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count() << endl;

}


void compute_sic_order(vector<complex<float>> interp_channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                       vector<float> squared_norms_[MAX_RX_PORTS][MAX_TX_PORTS],
                       vector<vector<int>> &sic_orders_,
                       int nb_rx_,
                       int nb_tx_,
                       int * dmrs_symbols_,
                       int num_dmrs_symbols_,
                       int fft_size_,
                       int num_symbols_per_grid,
                       int pdsch_start_symbol_,
                       int num_cdm_groups_without_data_) {

    float squared_norms[14][MAX_TX_PORTS]; // Store the mean of the columns squared norms on each symbol

    for(int i = 0; i < 14; i++) {
        memset(&squared_norms[i], 0, MAX_TX_PORTS * sizeof(float));
    }

    if(num_dmrs_symbols_ == 1) {
        /// Compute the same order for the whole frame


    } else {
        for(int rx = 0; rx < nb_rx_; rx++) {
            for(int tx = 0; tx < nb_tx_; tx++) {

                /// compute the norms of the coefficients for each subcarrier
                for(int re = 0; re < num_symbols_per_grid * fft_size_; re++) {
                    squared_norms_[rx][tx][re] = abs(interp_channel_coefficients_[rx][tx][re]);
                }

                /**
                /// Compute the mean of squared columns norms for DMRS symbols only
                for(int symbol = 0; symbol < num_dmrs_symbols_; symbol++) {

                    /// Compute the squared norm
                    for(int sc = 0; sc < fft_size_; sc++) {
                        squared_norms_[rx][tx][dmrs_symbols_[symbol] * fft_size_ + sc] *= squared_norms_[rx][tx][dmrs_symbols_[symbol] * fft_size_ + sc];
                    }
                } */

                for(int re = 0; re < num_symbols_per_grid * fft_size_; re++) {
                    squared_norms_[rx][tx][re] *= squared_norms_[rx][tx][re];
                }
            }
        }

        /// Compute the mean of column norms
        for(int symbol = 0; symbol < num_symbols_per_grid; symbol++) {
            for(int rx = 0; rx < nb_rx_; rx++) {
                for (int tx = 0; tx < nb_tx_; tx++) {
                    for (int sc = 0; sc < fft_size_; sc++) {
                        squared_norms[symbol][tx] += squared_norms_[rx][tx][symbol * fft_size_ + sc];
                    }
                }
            }
        }

        for(int symbol = 0; symbol < num_symbols_per_grid; symbol++) {
            for(int tx = 0; tx < nb_tx_; tx++) {
                squared_norms[symbol][tx] /= fft_size_;
            }
        }

        /**
        int * temp_dmrs_symbols = dmrs_symbols_;
        int step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
        float * lower_norm;
        float * upper_norm;
        int last_dmrs_symbol = *(dmrs_symbols_ + num_dmrs_symbols_ - 1);
        float step = 0;

        /// Interpolate between DMRS symbols
        for(int symbol = 0; symbol < num_symbols_per_grid; symbol++) {

            /// If current symbol is a DMRS symbol, do not interpolate on this symbol
            if ((symbol == (*(temp_dmrs_symbols) - pdsch_start_symbol_)) or
                (symbol == (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_))) {
                continue;
            }

            /// If current symbol is greater than the upper DMRS symbol,
            /// update lower and upper DMRS coefs
            if ((symbol > (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_)) and (symbol < last_dmrs_symbol)) {
                temp_dmrs_symbols++;
                step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
            }

            step = 1.0f * (symbol - *(temp_dmrs_symbols) + pdsch_start_symbol_) / step_symbols;

            for(int tx = 0; tx < nb_tx_; tx++) {
                lower_norm = squared_norms[tx] + (*(temp_dmrs_symbols) - pdsch_start_symbol_);
                upper_norm = squared_norms[tx] + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_);
                squared_norms[tx][symbol] *= 2 / (step * step);
                squared_norms[tx][symbol] += (*(upper_norm) + *(lower_norm)) / (step * step);
            }
        } */
    }

        /// Compute the SIC orders for each symbols
        if(num_cdm_groups_without_data_ == 2) { /// Do not sort on DMRS symbol because there is no PDSHC
            for (int symbol = 0; symbol < num_symbols_per_grid; symbol++) {

                if(symbol == *(dmrs_symbols_)) {
                    dmrs_symbols_++;
                    continue;
                }

                iota(sic_orders_[symbol].begin(), sic_orders_[symbol].begin() + nb_tx_, 0);
                std::stable_sort(sic_orders_[symbol].begin(), sic_orders_[symbol].end(), [&squared_norms](int i1, int i2)
                {
                    return squared_norms[i1] > squared_norms[i2];
                });

            }
        } else { /// Sort on DMRS symbols because there is PDSCH
            for (int symbol = 0; symbol < num_symbols_per_grid; symbol++) {

                iota(sic_orders_[symbol].begin(), sic_orders_[symbol].begin() + nb_tx_, 0);
                std::stable_sort(sic_orders_[symbol].begin(), sic_orders_[symbol].end(), [&squared_norms](int i1, int i2)
                {
                    return squared_norms[i1] < squared_norms[i2];
                });
            }
        }
}


/** Synchronizes the received slots, so that the first sample of the PSS is
 *  the first sample of the synchronized signal. We onyl use the PSS to synchronize.
 *
 * @param[in] received_signal     : received time-domain signal
 * @param[out] synchronized_grid  : synchronized frequency-domain grid
 * @param[out] synchronized_signal : synchronized time-domain signal
 * @param[in] nbSubframes              : number of subframes received
 * @param[in] subframes_to_keep        : number of subframes to keep in the synchronized signal
 */
void synchronize_slots(const vector<complex<float>> &received_signal,
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
                       vector<vector<complex<float>>> &time_signals_pss_,
                       vector<vector<complex<float>>> &time_signals_pss_downsampled_) {

    if(not known_n_id_2) {
        synchronization_index_ = 0;
    }

    float peak_value = 0;

    /// Search PSS only within the first half of the received signal to avoid
    /// confusion between several received PSS signals
    int symbolDuration = cp_length_pss_ + fft_size_;
    size_t numSamples = num_samples_received_signal_/2;

    /// Downsample signal for better performance
    //int downsampling_factor_ = fft_size_ / SIZE_PSS_SSS_SIGNAL;
    int  symbolDurationDownsampled =  symbolDuration / downsampling_factor_;

    if(known_n_id_2) { /// Correlation has already been performed on the first received grid. Use the determined N_ID_2 directly
        /// Compute PSS start index in downsampled signal

        vector<complex<float>> pssSignalDownSampled(numSamples / downsampling_factor_);
        for (int i = 0; i < numSamples / downsampling_factor_; i++)
        {
            pssSignalDownSampled[i] = received_signal[(size_t)downsampling_factor_ * i];
        }

        /// Search PSS in downsampled signal
        search_pss(n_id_2_, synchronization_index_, peak_value, cp_length_pss_ / downsampling_factor_,
                   pssSignalDownSampled.begin(), pssSignalDownSampled.size(),
                   fft_size_ / downsampling_factor_,
                   true, /// Consider N_ID_2 has been determined previously
                   time_signals_pss_downsampled_);

        //cout << "FFT size downsampled : " << fft_size_ / downsampling_factor_ << endl;
        //cout << "CP length downsampled : " << cp_length_pss_ / downsampling_factor_ << endl;

        //cout << "synchronization_index downsampled new algorithm : " << synchronization_index_ << endl;

        /// Compute PSS start index in downsampled signal
        pss_start_index_ = downsampling_factor_ * (synchronization_index_ -  symbolDurationDownsampled + 1);

        /// Synchronization on full signal
        //vector<complex<float>> finePssSignal(symbolDuration + (2 * downsamplingFactor + 1));
        //memcpy(finePssSignal.data(), &pss_search_buffer[pss_start_index_  - downsamplingFactor], (symbolDuration + 2 * downsamplingFactor + 1) * sizeof(complex<float>));
        //std::copy(received_signal.begin() + pss_start_index_  - downsamplingFactor,
        //          received_signal.begin() + pss_start_index_  - downsamplingFactor + (symbolDuration + 2 * downsamplingFactor + 1),
        //          finePssSignal.begin());
        /// Work directly on the vector of received samples

        /// Search PSS in fine PSS signal
        /*
        int sync_index_local = synchronization_index_;
        search_pss(n_id_2_,
                   sync_index_local,
                   peak_value, cp_length_pss_,
                   received_signal.begin() + pss_start_index_ - downsamplingFactor,
                   symbolDuration + (2 * downsamplingFactor + 1),
                   fft_size_,
                   true,
                   time_signals_pss_); */
        search_pss(n_id_2_,
                   synchronization_index_,
                   peak_value, cp_length_pss_,
                   received_signal.begin() + pss_start_index_ - downsampling_factor_,
                   symbolDuration + (2 * downsampling_factor_ + 1),
                   fft_size_,
                   true,
                   time_signals_pss_);

        /// Compute PSS start index
        //int pss_start_index_local  = pss_start_index_  + sync_index_local - symbolDuration + 1 - downsamplingFactor;
        pss_start_index_ += synchronization_index_ - symbolDuration + 1 - downsampling_factor_;

        //cout << "pss start index fine 1st grid new algorithm " << pss_start_index_local << endl;

        /// Compute number of samples in one subframe
        int subframe_size = cum_sum_cp_lengths_[symbols_per_subframe_ - 1] + fft_size_ + cp_lengths_[symbols_per_subframe_ - 1];

    } else {
        vector<complex<float>> pssSignalDownSampled(numSamples / downsampling_factor_);
        for (int i = 0; i < numSamples / downsampling_factor_; i++)
        {
            pssSignalDownSampled[i] = received_signal[(size_t)downsampling_factor_ * i];
        }

        /// Search PSS in downsampled signal
        search_pss(n_id_2_, synchronization_index_, peak_value, cp_length_pss_ / downsampling_factor_,
                   pssSignalDownSampled.begin(), pssSignalDownSampled.size(),
                   fft_size_ / downsampling_factor_,
                   false, /// N_ID_2 is unknown
                   time_signals_pss_downsampled_);

        //cout << "FFT size downsampled : " << fft_size_ / downsampling_factor_ << endl;
        //cout << "CP length downsampled : " << cp_length_pss_ / downsampling_factor_ << endl;

        //cout << "synchronization_index downsampled new algorithm : " << synchronization_index_ << endl;

        /// Compute PSS start index in downsampled signal
        pss_start_index_ = downsampling_factor_ * (synchronization_index_ -  symbolDurationDownsampled + 1);

        /// Synchronization on full signal
        //vector<complex<float>> finePssSignal(symbolDuration + (2 * downsamplingFactor + 1));
        //memcpy(finePssSignal.data(), &pss_search_buffer[pss_start_index_  - downsamplingFactor], (symbolDuration + 2 * downsamplingFactor + 1) * sizeof(complex<float>));
        //std::copy(received_signal.begin() + pss_start_index_  - downsamplingFactor,
        //          received_signal.begin() + pss_start_index_  - downsamplingFactor + (symbolDuration + 2 * downsamplingFactor + 1),
        //          finePssSignal.begin());
        /// Work directly on the vector of received samples

        /// Search PSS in fine PSS signal
        /*
        int sync_index_local = synchronization_index_;
        search_pss(n_id_2_,
                   sync_index_local,
                   peak_value, cp_length_pss_,
                   received_signal.begin() + pss_start_index_ - downsamplingFactor,
                   symbolDuration + (2 * downsamplingFactor + 1),
                   fft_size_,
                   true,
                   time_signals_pss_); */
        search_pss(n_id_2_,
                   synchronization_index_,
                   peak_value, cp_length_pss_,
                   received_signal.begin() + pss_start_index_ - downsampling_factor_,
                   symbolDuration + (2 * downsampling_factor_ + 1),
                   fft_size_,
                   true,
                   time_signals_pss_);

        /// Compute PSS start index
        //int pss_start_index_local  = pss_start_index_  + sync_index_local - symbolDuration + 1 - downsamplingFactor;
        pss_start_index_ += synchronization_index_ - symbolDuration + 1 - downsampling_factor_;

        //cout << "pss start index fine 1st grid new algorithm " << pss_start_index_local << endl;

        /// Compute number of samples in one subframe
        int subframe_size = cum_sum_cp_lengths_[symbols_per_subframe_ - 1] + fft_size_ + cp_lengths_[symbols_per_subframe_ - 1];
    }
}

/** Estimate the pilots only for DMRS conf. type 1, single symbol or double symbol
 *
 * @param double_symbol   : 'True' is double symbol DMRS, else 'False'
 * @param cdm_groups      : vector containing the antenna ports grouped by CDM groups. We made sure that antenna ports where given in ascending order when computing the CDM groups.
 * @param received_grids_ : vector containing OFDM grids of the received signal on each receiver port
 * @return
 */
 /**
void estimate_pilots_cdm_groups(const vector<vector<int>> &tx_dmrs_ports_,
                                const int * cdm_groups_sizes_,
                                const int * dmrs_symbols_,
                                const int * dmrs_subcarriers_,
                                vector<vector<complex<float>>> * received_grids_,
                                complex<float> * dmrsSequences_,
                                complex<float> * estimated_chan_coefs,
                                const bool &double_symbol_,
                                const int &dmrs_sequence_size_,
                                const int &slot_number_,
                                const int &num_cdm_groups_,
                                const int &num_dmrs_symbols_per_slot_,
                                const int &nb_tx,
                                const int &nb_rx,
                                const int &nb_pdsch_slots_) {

    int port1 = 0, port2 = 0;
    int port1_index = 0, port2_index = 1;
    int offset = slot_number_ * num_dmrs_symbols_per_slot_;
    int offset_slot = slot_number_ * 14;

    int tx_rx_block_size = num_dmrs_symbols_per_slot_ * dmrs_sequence_size_;
    int tx_offset = tx_rx_block_size * nb_rx;

    complex<float> * temp_pointer_estimated_chan_coefs_port1 = estimated_chan_coefs;
    complex<float> * temp_pointer_estimated_chan_coefs_port2 = estimated_chan_coefs;
    vector<vector<complex<float>>> *temp_received_grids = received_grids_;
    complex<float> * temp_dmrs_sequences_port1 = dmrsSequences_;
    complex<float> * temp_dmrs_sequences_port2 = dmrsSequences_;

    /// Compute the same norm for all DMRS symbols, as they belong to a QPSK constellation
    float dmrs_symbols_norm = pow(abs(*(dmrsSequences_)), 2);

    /// Increment pointer
    dmrsSequences_ += + slot_number_ * num_dmrs_symbols_per_slot_ * dmrs_sequence_size_;

    /// DMRS configuration type 1, double symbol
    if(double_symbol_) {

        /// TODO : implement channel estimation for double symbol

        /// Retrieve the channel coefficients for each CDM group
        for(int group = 0; group < num_cdm_groups_; group++) {

            /// Two antenna ports in the same group multiplexed by OCC
            if(cdm_groups_sizes_[group] > 1) {

            }

                /// TODO : Three antenna ports in the same group multiplexed by OCC


                /// TODO : Four antenna ports in the same group multiplexed by OCC

                /// Only one antenna port in the CDM group, simply compute the channel coefficients according to the DMRS sequence
            else {

            }
        }

    } else { /// DMRS configuration type 1 (every two subcarrier is occupied by DMRS on an OFDM symbol used for DMRS transmission)

        /// Retrieve the channel coefficients for each CDM group
        for(int group = 0; group < num_cdm_groups_; group++) {

            /// Reset values of pointers
            temp_pointer_estimated_chan_coefs_port1 = estimated_chan_coefs;
            temp_pointer_estimated_chan_coefs_port2 = estimated_chan_coefs;
            temp_dmrs_sequences_port1 = dmrsSequences_;
            temp_dmrs_sequences_port2 = dmrsSequences_;

            /// Two antenna ports in same group mutliplexed by OCC
            if(cdm_groups_sizes_[group] > 1) {

                /// Get the two antenna ports of the CDM group
                port1 = tx_dmrs_ports_[group][0];
                port2 = tx_dmrs_ports_[group][1];

                /// increment pointer
                temp_dmrs_sequences_port1 += port1 * nb_pdsch_slots_ * num_dmrs_symbols_per_slot_ * dmrs_sequence_size_;
                temp_dmrs_sequences_port2 += port2 * nb_pdsch_slots_ * num_dmrs_symbols_per_slot_ * dmrs_sequence_size_;
                temp_pointer_estimated_chan_coefs_port1 += port1_index * tx_offset;
                temp_pointer_estimated_chan_coefs_port2 += port2_index * tx_offset;

                /// For each path compute the channel coefficient
                /// The received grids must be given in ascending order of antenna port number
                //#pragma omp parallel for
                for(int receiver = 0; receiver < nb_rx; receiver++) {
                    for(int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {
                        for(int sc = 0; sc < dmrs_sequence_size_/2; sc++) {
                            /// Device received symbols by DMRS symbols from sequence of the first antenna port
                            *(temp_pointer_estimated_chan_coefs_port1) =
                                    received_grids_[receiver][dmrs_symbols_[symbol] + offset_slot][dmrs_subcarriers_[port1 * dmrs_sequence_size_ + sc]]
                                    * conj(*(temp_dmrs_sequences_port1))
                                    / dmrs_symbols_norm
                                    + received_grids_[receiver][dmrs_symbols_[symbol] + offset_slot][dmrs_subcarriers_[port1 * dmrs_sequence_size_ + sc] + 2]
                                    * conj(*(temp_dmrs_sequences_port1 + 1))
                                    / dmrs_symbols_norm;

                            *(temp_pointer_estimated_chan_coefs_port2) =
                                    received_grids_[receiver][*(dmrs_symbols_ + symbol) + offset_slot][dmrs_subcarriers_[port1 * dmrs_sequence_size_ + sc]]
                                    * conj(*(temp_dmrs_sequences_port1))
                                    / dmrs_symbols_norm
                                    - received_grids_[receiver][*(dmrs_symbols_ + symbol) + offset_slot][dmrs_subcarriers_[port1 * dmrs_sequence_size_ + sc] + 2]
                                    * conj(*(temp_dmrs_sequences_port1 + 1))
                                    / dmrs_symbols_norm;

                            /// Apply 1/2 coefficient on each pilot
                            *(temp_pointer_estimated_chan_coefs_port1) /= 2;
                            *(temp_pointer_estimated_chan_coefs_port2) /= 2;

                            /// Because of OCC, the coefficient remains constant over two subcarriers
                            *(temp_pointer_estimated_chan_coefs_port1 + 1) = *(temp_pointer_estimated_chan_coefs_port1);
                            *(temp_pointer_estimated_chan_coefs_port2 + 1) = *(temp_pointer_estimated_chan_coefs_port2);

                            temp_dmrs_sequences_port1 += 2;
                            temp_dmrs_sequences_port2 += 2;
                            temp_pointer_estimated_chan_coefs_port1 += 2;
                            temp_pointer_estimated_chan_coefs_port2 += 2;
                        }
                    }
                }

                port1_index += 2;
                port2_index += 2;

                /// Only one antenna port in the CDM group, simply compute the channel coefficients according to the DMRS sequence
            } else {

                /// Get the antenna port in the CDM group
                port1 = tx_dmrs_ports_[group][0];

                /// increment pointer
                temp_dmrs_sequences_port1 += port1 * nb_pdsch_slots_ * num_dmrs_symbols_per_slot_ * dmrs_sequence_size_;
                temp_pointer_estimated_chan_coefs_port1 += port1_index * tx_offset;

                /// For each path compute the channel coefficient
                /// The received grids must be given in ascending order of antenna port number
                //#pragma omp parallel for
                for(int receiver = 0; receiver < nb_rx; receiver++) {
                    for(int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {
                        for(int sc = 0; sc < dmrs_sequence_size_; sc++) {

                            /// Compute the channel coefficient
                            *(temp_pointer_estimated_chan_coefs_port1) =
                                    received_grids_[receiver][dmrs_symbols_[symbol] + offset_slot][dmrs_subcarriers_[port1 * dmrs_sequence_size_ + sc]]
                                    * conj(*(temp_dmrs_sequences_port1))
                                    / dmrs_symbols_norm;

                            temp_dmrs_sequences_port1++;
                            temp_pointer_estimated_chan_coefs_port1 ++;
                        }
                    }
                }
                port1_index += 1;
                port2_index += 1;
            }
        }
    }
} */

/**
void estimate_pilots_cdm_groups_one_rx(const int * tx_dmrs_ports_,
                                       const int * cdm_groups_sizes_,
                                       const int * dmrs_symbols_,
                                       const int * dmrs_subcarriers_,
                                       const vector<vector<complex<float>>> &received_grid_,
                                       complex<float> * dmrsSequences_,
                                       complex<float> * estimated_chan_coefs, /// TODO: Ajouter l'offset au pointer en dehors de la fonction
                                       const bool &double_symbol_,
                                       const int &dmrs_sequence_size_,
                                       const int &slot_number_,
                                       const int &num_dmrs_symbols_per_slot_,
                                       const int &num_tx_ports_,
                                       const int &nb_pdsch_slots_,
                                       const int &receiver_no_) {

    int current_dmrs_port_no = 0;
    int offset_slot = 0;
    complex<float> *temp_dmrs_sequences;
    complex<float> *temp_estimated_chan_coefs;

    offset_slot = slot_number_ * 14;
    temp_estimated_chan_coefs = estimated_chan_coefs;

    //ofstream output_file = ofstream("dmrs_sequences_decodeur_rx" + to_string(rx_no%4) + "slot_" + to_string(slot_number_) + ".txt");

    //cout << "estimating coefs for receiver " << receiver_no << endl;
    //cout << "estimating coefs for slot " << slot_number_ << endl;

    //ofstream output_file_estimated_channel_coefs("estimated_channel_coefs_rx" + to_string(receiver_no_) + "slot_" + to_string(slot_number_) + ".txt");

    /// DMRS configuration type 1, double symbol
    if (double_symbol_) {

        /// TODO : implement channel estimation for double symbol

    } else { /// Single symbol DMRS

        for (int port_no = 0; port_no < num_tx_ports_; port_no++) {

            /// Reset the value of the pointer to DRMS sequences
            temp_dmrs_sequences = dmrsSequences_;

            current_dmrs_port_no = *(tx_dmrs_ports_ + port_no);

            /// Increment pointer to get the start of the buffer containing sequences for the current DMRS port
            /// And for the current slot
            temp_dmrs_sequences +=
                    current_dmrs_port_no * nb_pdsch_slots_ * num_dmrs_symbols_per_slot_ * dmrs_sequence_size_ +
                    (slot_number_ - 1) * num_dmrs_symbols_per_slot_ * dmrs_sequence_size_;

            if (cdm_groups_sizes_[port_no] > 1) { /// Two DMRS ports in the CDM group. Perform OCC descrambling
                for (int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {

                    /**
                    output_file << "----------------------" << endl;
                    output_file << "Port number : " << current_dmrs_port_no << endl;
                    output_file << "Slot number : " << slot_number_ << endl;
                    output_file << "Symbol number : " << dmrs_symbols_[symbol] << endl;
                    output_file << "______________________" << endl;

                    output_file_estimated_channel_coefs << "----------------------" << endl;
                    output_file_estimated_channel_coefs << "Port number : " << current_dmrs_port_no << endl;
                    output_file_estimated_channel_coefs << "RX port number : " << receiver_no_ << endl;
                    output_file_estimated_channel_coefs << "Slot number : " << slot_number_ << endl;
                    output_file_estimated_channel_coefs << "Symbol number : " << dmrs_symbols_[symbol] << endl;
                    output_file_estimated_channel_coefs << "______________________" << endl;


                    for (int sc = 0; sc < dmrs_sequence_size_ / 2; sc++) {
                        *(temp_estimated_chan_coefs) =
                                (received_grid_[dmrs_symbols_[symbol] + offset_slot][dmrs_subcarriers_[
                                        current_dmrs_port_no * dmrs_sequence_size_ + 2 * sc]]
                                *
                                conj(*(temp_dmrs_sequences))); /// Do not divide by DMRS norm, because it is equal to 1 anyways

                        *(temp_estimated_chan_coefs + 1) =
                                (received_grid_[dmrs_symbols_[symbol] + offset_slot][dmrs_subcarriers_[
                                        current_dmrs_port_no * dmrs_sequence_size_ + 2 * sc + 1]]
                                * conj(*(temp_dmrs_sequences +
                                         1))); /// Do not divide by DMRS norm, because it is equal to 1 anyways

                        //output_file << *(temp_dmrs_sequences) << "\n";
                        //output_file << *(temp_dmrs_sequences + 1) << "\n";

                        /// OCC descrambling, take the mean between the two received DMRS REs.
                        *(temp_estimated_chan_coefs) += *(temp_estimated_chan_coefs + 1);
                        *(temp_estimated_chan_coefs) /= 2;
                        *(temp_estimated_chan_coefs + 1) = *(temp_estimated_chan_coefs);

                        //output_file_estimated_channel_coefs << *(temp_estimated_chan_coefs) << endl;
                        //output_file_estimated_channel_coefs << *(temp_estimated_chan_coefs + 1) << endl;

                        temp_estimated_chan_coefs += 2;
                        temp_dmrs_sequences += 2;
                    }
                }
            } else { /// only one DMRS port in the CDM group
                for (int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {
                    for (int sc = 0; sc < dmrs_sequence_size_; sc++) {
                        *(temp_estimated_chan_coefs) =
                                (received_grid_[dmrs_symbols_[symbol] + offset_slot][dmrs_subcarriers_[
                                        current_dmrs_port_no * dmrs_sequence_size_ + sc]]
                                *
                                conj(*(temp_dmrs_sequences))); /// Do not divide by DMRS norm, because it is equal to 1 anyways
                        temp_estimated_chan_coefs++;
                        temp_dmrs_sequences++;
                    }
                }
            }
        }
    }
} */

void estimate_pilots_cdm_groups_one_rx(const int * tx_dmrs_ports_,
                                       const int * cdm_groups_sizes_,
                                       const int * dmrs_symbols_,
                                       vector<complex<float>> * received_dmrs_samples,
                                       complex<float> * dmrsSequences_,
                                       complex<float> * estimated_chan_coefs, /// TODO: Ajouter l'offset au pointer en dehors de la fonction
                                       bool double_symbol_,
                                       int dmrs_sequence_size_,
                                       int slot_number_,
                                       int num_dmrs_symbols_per_slot_,
                                       int num_tx_ports_,
                                       int nb_pdsch_slots_,
                                       const int * cdm_groups_) {

    int current_dmrs_port_no = 0;
    int offset_slot = 0;
    complex<float> *temp_dmrs_sequences;
    complex<float> *temp_estimated_chan_coefs;
    complex<float> *temp_received_dmrs_samples;

    offset_slot = slot_number_ * 14;
    temp_estimated_chan_coefs = estimated_chan_coefs;

    /// DMRS configuration type 1, double symbol
    if (double_symbol_) {

        /// TODO : implement channel estimation for double symbol

    } else { /// Single symbol DMRS

        for (int port_no = 0; port_no < num_tx_ports_; port_no++) {

            /// Reset the value of the pointer to DRMS sequences
            temp_dmrs_sequences = dmrsSequences_;

            /// Reset pointer to the received dmrs samples
            temp_received_dmrs_samples = received_dmrs_samples[cdm_groups_[port_no]].data();
            current_dmrs_port_no = *(tx_dmrs_ports_ + port_no);

            /// Increment pointer to get the start of the buffer containing sequences for the current DMRS port
            /// And for the current slot
            temp_dmrs_sequences +=
                    current_dmrs_port_no * nb_pdsch_slots_ * num_dmrs_symbols_per_slot_ * dmrs_sequence_size_ +
                    (slot_number_ - 1) * num_dmrs_symbols_per_slot_ * dmrs_sequence_size_;

            if (cdm_groups_sizes_[port_no] > 1) { /// Two DMRS ports in the CDM group. Perform OCC descrambling
                for (int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {
                    for (int sc = 0; sc < dmrs_sequence_size_ / 2; sc ++) { // / 2; sc++) {
                        *(temp_estimated_chan_coefs) =
                                *(temp_received_dmrs_samples) *
                                 conj(*(temp_dmrs_sequences)); /// Do not divide by DMRS norm, because it is equal to 1

                        *(temp_estimated_chan_coefs + 1) =
                                *(temp_received_dmrs_samples + 1)
                                 * conj(*(temp_dmrs_sequences +
                                          1)); /// Do not divide by DMRS norm, because it is equal to 1

                        //output_file << *(temp_dmrs_sequences) << "\n";
                        //output_file << *(temp_dmrs_sequences + 1) << "\n";

                        /// OCC descrambling, take the mean between the two received DMRS REs.
                        *(temp_estimated_chan_coefs) += *(temp_estimated_chan_coefs + 1);
                        *(temp_estimated_chan_coefs) /= 2;
                        *(temp_estimated_chan_coefs + 1) = *(temp_estimated_chan_coefs);

                        //output_file_estimated_channel_coefs << *(temp_estimated_chan_coefs) << endl;
                        //output_file_estimated_channel_coefs << *(temp_estimated_chan_coefs + 1) << endl;

                        //cout << *(temp_estimated_chan_coefs) << endl;
                        //cout << *(temp_estimated_chan_coefs + 1) << endl;

                        temp_estimated_chan_coefs += 2;
                        temp_dmrs_sequences += 2;
                        temp_received_dmrs_samples += 2;
                    }
                }
            } else { /// only one DMRS port in the CDM group.
                for (int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {
                    for (int sc = 0; sc < dmrs_sequence_size_; sc++) {
                        *(temp_estimated_chan_coefs) =
                                *(temp_received_dmrs_samples)
                                 *
                                 conj(*(temp_dmrs_sequences)); /// Do not divide by DMRS norm, because it is equal to 1 anyways

                        temp_estimated_chan_coefs++;
                        temp_dmrs_sequences++;
                        temp_received_dmrs_samples++;
                    }
                }
            }
        }
    }
}

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
                                       vector<complex<float>> * received_dmrs_samples,
                                       complex<float> * dmrsSequences_,
                                   complex<float> estimated_chan_coefs[MAX_TX_PORTS][MAX_DMRS_SYMBOLS][MAX_DMRS_SUBCARRIERS] /// TODO: Ajouter l'offset au pointer en dehors de la fonction
                                       ) {

    int current_dmrs_port_no = 0;
    int offset_slot = 0;
    complex<float> *temp_dmrs_sequences;
    complex<float> *temp_received_dmrs_samples;

    offset_slot = slot_number_ * 14;

    /// DMRS configuration type 1, double symbol
    if (double_symbol_) {

        /// TODO : implement channel estimation for double symbol

    } else { /// Single symbol DMRS

        for (int port_no = 0; port_no < num_tx_ports_; port_no++) {

            /// Reset the value of the pointer to DRMS sequences
            temp_dmrs_sequences = dmrsSequences_;

            /// Reset pointer to the received dmrs samples
            temp_received_dmrs_samples = received_dmrs_samples[cdm_groups_[port_no]].data();
            current_dmrs_port_no = *(tx_dmrs_ports_ + port_no);

            /// Increment pointer to get the start of the buffer containing sequences for the current DMRS port
            /// And for the current slot
            temp_dmrs_sequences +=
                    current_dmrs_port_no * nb_pdsch_slots_ * num_dmrs_symbols_per_slot_ * dmrs_sequence_size_ +
                    (slot_number_ - 1) * num_dmrs_symbols_per_slot_ * dmrs_sequence_size_;

            if (cdm_groups_sizes_[port_no] > 1) { /// Two DMRS ports in the CDM group. Perform OCC descrambling
                for (int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {
                    for (int sc = 0; sc < dmrs_sequence_size_; sc+= 2) { // / 2; sc++) {
                        estimated_chan_coefs[port_no][symbol][sc] =
                                *(temp_received_dmrs_samples) *
                                conj(*(temp_dmrs_sequences)); /// Do not divide by DMRS norm, because it is equal to 1

                        estimated_chan_coefs[port_no][symbol][sc + 1] =
                                *(temp_received_dmrs_samples + 1)
                                * conj(*(temp_dmrs_sequences +
                                         1)); /// Do not divide by DMRS norm, because it is equal to 1

                        /// OCC descrambling, take the mean between the two received DMRS REs.
                        estimated_chan_coefs[port_no][symbol][sc] += estimated_chan_coefs[port_no][symbol][sc + 1];
                        estimated_chan_coefs[port_no][symbol][sc] /= 2;
                        estimated_chan_coefs[port_no][symbol][sc + 1] = estimated_chan_coefs[port_no][symbol][sc];
                        temp_dmrs_sequences += 2;
                        temp_received_dmrs_samples += 2;
                    }
                }
            } else { /// only one DMRS port in the CDM group.
                for (int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {
                    for (int sc = 0; sc < dmrs_sequence_size_; sc++) {
                        estimated_chan_coefs[port_no][symbol][sc] = *(temp_received_dmrs_samples) * conj(*(temp_dmrs_sequences)); /// Do not divide by DMRS norm, because it is equal to 1 anyways
                        temp_dmrs_sequences++;
                        temp_received_dmrs_samples++;
                    }
                }
            }
        }
    }
}
/*********************************************************************************************************************/

#if defined(__AVX2__)
void estimate_pilots_avx(const int * tx_dmrs_ports_,
                         const int * cdm_groups_sizes_,
                         const int * dmrs_symbols_,
                         vector<complex<float>> * received_dmrs_samples,
                         complex<float> * dmrsSequences_,
                         complex<float> * estimated_chan_coefs, /// TODO: Ajouter l'offset au pointer en dehors de la fonction
                         bool double_symbol_,
                         int dmrs_sequence_size_,
                         int slot_number_,
                         int num_dmrs_symbols_per_slot_,
                         int num_tx_ports_,
                         int nb_pdsch_slots_,
                         const int * cdm_groups_) {

    int current_dmrs_port_no = 0;
    int offset_slot = 0;
    complex<float> *temp_dmrs_sequences;
    complex<float> *temp_estimated_chan_coefs;
    complex<float> *temp_received_dmrs_samples;

    offset_slot = slot_number_ * 14;
    temp_estimated_chan_coefs = estimated_chan_coefs;

    __m256 vec1, vec2, vec3;
    __m256 conj_vec = {1, -1, 1, -1, 1, -1, 1, -1};
    __m256 cst_vec = _mm256_set1_ps(0.5);

    /// DMRS configuration type 1, double symbol
    if (double_symbol_) {

        /// TODO : implement channel estimation for double symbol

    } else { /// Single symbol DMRS

        for (int port_no = 0; port_no < num_tx_ports_; port_no++) {

            /// Reset the value of the pointer to DRMS sequences
            temp_dmrs_sequences = dmrsSequences_;

            /// Reset pointer to the received dmrs samples
            temp_received_dmrs_samples = received_dmrs_samples[cdm_groups_[port_no]].data();
            current_dmrs_port_no = *(tx_dmrs_ports_ + port_no);

            /// Increment pointer to get the start of the buffer containing sequences for the current DMRS port
            /// And for the current slot
            temp_dmrs_sequences +=
                    current_dmrs_port_no * nb_pdsch_slots_ * num_dmrs_symbols_per_slot_ * dmrs_sequence_size_ +
                    (slot_number_ - 1) * num_dmrs_symbols_per_slot_ * dmrs_sequence_size_;

            if (cdm_groups_sizes_[port_no] > 1) { /// Two DMRS ports in the CDM group. Perform OCC descrambling
                for (int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {
                    for (int sc = 0; sc < dmrs_sequence_size_; sc += 4) {
                        vec1 = _mm256_loadu_ps((float *) temp_dmrs_sequences);
                        vec2 = _mm256_loadu_ps((float *) temp_received_dmrs_samples);

                        vec3 = _mm256_hadd_ps(
                                        _mm256_mul_ps(vec1, vec2),
                                        _mm256_mul_ps(
                                                _mm256_permute_ps(
                                                        _mm256_mul_ps(vec1, conj_vec), 0b10110001), vec2));

                        _mm256_storeu_ps((float *) temp_estimated_chan_coefs, _mm256_mul_ps(_mm256_hadd_ps(vec3, vec3), cst_vec));
                        //_mm256_stream_ps((float *) temp_estimated_chan_coefs, _mm256_mul_ps(_mm256_hadd_ps(vec3, vec3), cst_vec));

                        temp_dmrs_sequences += 4;
                        temp_received_dmrs_samples += 4;
                        temp_estimated_chan_coefs += 4;
                    }
                }
            } else { /// only one DMRS port in the CDM group.
                for (int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {
                    for (int sc = 0; sc < dmrs_sequence_size_; sc += 4) {
                        vec1 = _mm256_loadu_ps((float *) temp_dmrs_sequences);
                        vec2 = _mm256_loadu_ps((float *) temp_received_dmrs_samples);

                        _mm256_storeu_ps((float *) temp_estimated_chan_coefs,
                                         _mm256_permute_ps(
                                                 _mm256_hadd_ps(
                                                    _mm256_mul_ps(vec1, vec2),
                                                    _mm256_mul_ps(
                                                            _mm256_permute_ps(
                                                                    _mm256_mul_ps(vec1, conj_vec), 0b10110001), vec2)), 0b11011000));

                        temp_dmrs_sequences += 4;
                        temp_received_dmrs_samples += 4;
                        temp_estimated_chan_coefs += 4;
                    }
                }
            }
        }
    }
}
#endif

void estimate_pilots_cdm_groups_one_rx(const int * tx_dmrs_ports_,
                                       const int * cdm_groups_sizes_,
                                       const int * dmrs_symbols_,
                                       vector<complex<float>> received_dmrs_samples[MAX_RX_PORTS][MAX_NUM_CDM_GROUPS],
                                       vector<complex<float>> * dmrs_sequences_,
                                       vector<complex<float>> * estimated_chan_coefs, /// TODO: Ajouter l'offset au pointer en dehors de la fonction
                                       bool double_symbol_,
                                       int dmrs_sequence_size_,
                                       int num_dmrs_symbols_per_slot_,
                                       int num_tx_ports_,
                                       int receiver_no_,
                                       const int * cdm_groups_) {

    int current_dmrs_port_no = 0;
    complex<float> * temp_dmrs_sequence;
    complex<float> * temp_estimated_chan_coefs;
    complex<float> * temp_received_dmrs_samples;

    /// DMRS configuration type 1, double symbol
    if (double_symbol_) {

        /// TODO : implement channel estimation for double symbol

    } else { /// Single symbol DMRS

        for (int port_no = 0; port_no < num_tx_ports_; port_no++) {

            current_dmrs_port_no = *(tx_dmrs_ports_ + port_no);

            /// Reset pointer to the dmrs_seuences
            temp_dmrs_sequence = dmrs_sequences_[current_dmrs_port_no].data();

            /// Reset pointer to the received dmrs samples
            temp_received_dmrs_samples = received_dmrs_samples[receiver_no_][cdm_groups_[port_no]].data();

            temp_estimated_chan_coefs = estimated_chan_coefs[port_no].data();

            if (cdm_groups_sizes_[port_no] > 1) { /// Two DMRS ports in the CDM group. Perform OCC descrambling
                for (int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {

                    /// Increment pointer to get the start of the buffer containing sequences for the current DMRS port
                    /// And for the current slot
                    //generate_pdsch_dmrs_sequence(14, slot_number_, symbol, n_scid, n_id_scid, temp_dmrs_sequence, dmrs_sequence_size_);

                    for (int sc = 0; sc < dmrs_sequence_size_; sc++) {
                        temp_estimated_chan_coefs[sc] =
                                temp_received_dmrs_samples[sc] *
                                conj(temp_dmrs_sequence[sc]); /// Do not divide by DMRS norm, because it is equal to 1
                    }
                    for(int sc = 0; sc < dmrs_sequence_size_; sc += 2) {
                        temp_estimated_chan_coefs[sc] += temp_estimated_chan_coefs[sc + 1];
                        temp_estimated_chan_coefs[sc] /= 2;
                    }
                    for(int sc = 1; sc < dmrs_sequence_size_; sc += 2) {
                        temp_estimated_chan_coefs[sc] = temp_estimated_chan_coefs[sc - 1];
                    }
                    temp_estimated_chan_coefs  += dmrs_sequence_size_;
                    temp_received_dmrs_samples += dmrs_sequence_size_;
                    temp_dmrs_sequence += dmrs_sequence_size_;
                }
            } else { /// only one DMRS port in the CDM group. And CDM group is CDM 0
                for (int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {

                    for (int sc = 0; sc < dmrs_sequence_size_; sc++) {
                        temp_estimated_chan_coefs[sc] =
                                temp_received_dmrs_samples[sc] *
                                conj(temp_dmrs_sequence[sc]); /// Do not divide by DMRS norm, because it is equal to 1
                    }
                    temp_estimated_chan_coefs  += dmrs_sequence_size_;
                    temp_received_dmrs_samples += dmrs_sequence_size_;
                    temp_dmrs_sequence += dmrs_sequence_size_;
                }
            }
        }
    }
}

/** Computes a centered moving mean on the grid of channel coefficients given in argument.
 *  Chose the appropriate size for the frequency and time window sizes. Their value must be and odd number
 *  equal or less than the size of the computed grid.
 *
 * @param[out] averagedCoefs   : output grid containing the averaged channel coefficients.
 * @param[in] nonAveragedCoefs : input grid containing non averaged channel coefficients.
 * @param[in] freq_window_size : frequency window size (odd number, less or equal to the size of the non averaged grid.
 * @param[in] time_window_size : time window size (odd numner, less or equal to the size of the non averaged grid.
 */
void centered_mov_mean(vector<vector<complex<float>>> &averagedCoefs, // Coefficients moyennés
                       const vector<vector<complex<float>>> &nonAveragedCoefs, // Coefficients non moyennés
                       const int &freq_window_size,   // fenêtre en fréquence
                       const int &time_window_size) { // fenêtre en temps

    int lowerCoef;
    int upperCoef;
    int mid_time_window = min(int(averagedCoefs.size()), time_window_size);
    mid_time_window     = floor(mid_time_window/2);
    int mid_freq_window = min(int(averagedCoefs[0].size()), freq_window_size);
    mid_freq_window     = floor(freq_window_size/2);
    float temp_re, temp_im;
    float temp_norm{};

    /// Average in time domain
    for(int subcarrier = 0; subcarrier < nonAveragedCoefs[0].size(); subcarrier ++) {
        for(int t = 0; t < nonAveragedCoefs.size(); t++) {

            /// If the actual time domain position is lesser than or equal to
            /// half of the size of the time window, compute the mean between
            /// the first coefficient at t=0 up to the coefficient à 2*t
            if(t - mid_time_window < 0) {
                lowerCoef = 0;      /// position of the lower DMRS signal
                upperCoef = 2 * t;    /// position of the upper DMRS signal

                /// If the size of the time domain window exceeds the value of the last
                /// position in time domain, compute the mean between the coefficients
                /// starting from t - (upperCoef - t) , up to the last coefficient (at upperCoef)
            } else if (t + mid_time_window > nonAveragedCoefs.size() - 1) {
                upperCoef = nonAveragedCoefs.size() - 1; /// position of the lower DMRS signal
                lowerCoef = t - (upperCoef - t);

                /// Compute the mean on all the coefficients within the time window.
            } else {
                lowerCoef = t - mid_time_window;
                upperCoef = t + mid_time_window;
            }

            /// Sum the value of all coefficients within the window
            for(int w = lowerCoef;  w < upperCoef + 1 ; w ++) {
                temp_re += nonAveragedCoefs[w][subcarrier].real();
                temp_im += nonAveragedCoefs[w][subcarrier].imag();
                //temp_norm += abs(nonAveragedCoefs[w][subcarrier]);
            }

            /// Apply the value to the actual coefficient
            //averagedCoefs[t][subcarrier].real(temp.real());
            //averagedCoefs[t][subcarrier].imag(temp.imag());

            averagedCoefs[t][subcarrier].real(temp_re);
            averagedCoefs[t][subcarrier].imag(temp_im);

            //averagedCoefs[t][subcarrier] = nonAveragedCoefs[t][subcarrier];

            /// Compute the mean between all coefficients within the window
            if(upperCoef - lowerCoef > 0) {
                //averagedCoefs[t][subcarrier].real(averagedCoefs[t][subcarrier].real() / (upperCoef - lowerCoef + 1));
                //averagedCoefs[t][subcarrier].imag(averagedCoefs[t][subcarrier].imag() / (upperCoef - lowerCoef + 1));

                averagedCoefs[t][subcarrier] /= (upperCoef - lowerCoef);

                //temp_norm /= (upperCoef - lowerCoef);
            }

            //averagedCoefs[t][subcarrier] *= temp_norm / abs(averagedCoefs[t][subcarrier]);

            temp_re = 0; // reinitialize temp
            temp_im = 0;
            //temp_norm = 0;
        }
    }

    /// Average in frequency domain
    for(int t = 0; t < nonAveragedCoefs.size(); t++) {
        for(int subcarrier = 0; subcarrier < nonAveragedCoefs[0].size(); subcarrier++) {

            /// If the actual time domain position is lesser than or equal to
            /// half of the size of the time window, compute the mean between
            /// the first coefficient at t=0 up to the coefficient à 2*t
            if(subcarrier - mid_freq_window < 0) {
                lowerCoef = 0;
                upperCoef = 2 * subcarrier;

                /// If the size of the time domain window exceeds the value of the last
                /// position in time domain, compute the mean between the coefficients
                /// starting from f - (upperCoef - f) , up to the last coefficient (at upperCoef)
            } else if (subcarrier + mid_freq_window > nonAveragedCoefs[0].size() - 1) {
                upperCoef = nonAveragedCoefs[0].size() - 1;
                lowerCoef = subcarrier - (upperCoef - subcarrier);

                /// Else compute the mean on all the coefficient within the frequency window
            } else {
                lowerCoef = subcarrier - mid_freq_window;
                upperCoef = subcarrier + mid_freq_window;
            }

            /// Sum all the coefficients within the window
            for(int w = lowerCoef;  w < upperCoef + 1; w ++) {
                temp_re += averagedCoefs[t][w].real();
                temp_im += averagedCoefs[t][w].imag();
                //temp_norm += abs(averagedCoefs[t][w]);
            }

            /// Asign the sum to the actual coefficient
            averagedCoefs[t][subcarrier].real(temp_re);
            averagedCoefs[t][subcarrier].imag(temp_im);

            //averagedCoefs[t][subcarrier] = nonAveragedCoefs[t][subcarrier];

            /// Compute the mean
            if(upperCoef - lowerCoef > 0) {
                //averagedCoefs[t][subcarrier].real(averagedCoefs[t][subcarrier].real() / (upperCoef - lowerCoef + 1));
                //averagedCoefs[t][subcarrier].imag(averagedCoefs[t][subcarrier].imag() / (upperCoef - lowerCoef + 1));

                averagedCoefs[t][subcarrier] /= (upperCoef - lowerCoef);

                //temp_norm /= (upperCoef - lowerCoef);
            }

            //averagedCoefs[t][subcarrier] *= temp_norm / abs(averagedCoefs[t][subcarrier]);

            temp_re = 0; // Reinitialize temp
            temp_im = 0;
            //temp_norm = 0;
        }
    }
}

/** Interpolates the channel coefficients computed from DMRS signals on all the REs of the grid,
     *  in frequency domain first, then in time domain.
     *
     * @param[out] coefGrid    : output grid containing all the channel coefficients (DMRS and interpolated)
     * @param[in] dmrs_coefs   : grid containing the channels coefficients computed from DMRS (averaged or not)
     * @param[out] rsPositions : grid containing the DMRS positions.
     */
void interpolate_coefs(vector<vector<complex<float>>> &coefGrid, /// TODO: mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                       complex<float> * dmrs_coefs_, // TODO : mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                               const int &cdm_group_number, /// Number of the DMRS CDM group to be interpolated, 0 or 1
                               const int &rx_antenna_port_number_,
                               int * dmrs_symbols_,
                               const int &pdsch_start_symbol_,
                               const int &dmrs_symbols_per_grid_,
                               const int &dmrs_sequence_size_,
                               const int &fft_size_,
                               const int &symbols_per_grid_,
                               const int &nb_tx,
                               const int &nb_rx) {

    int lower_dmrs_symbol{};        // to store the position of the lower DMRS in the grid
    int upper_dmrs_symbol{};        // to store the position of the upper DMRS in the grid
    int * temp_dmrs_symbols = dmrs_symbols_;
    complex<float> * temp_dmrs_coefs = dmrs_coefs_;
    complex<float> * lower_dmrs_coef, upper_dmrs_coef;

    /// Interpolation of DMRS from CDM group 0
    if(cdm_group_number == 0) {

        /// Interpolate in frequency domain
        for(int i = 0; i < dmrs_symbols_per_grid_; i++) {

            /// Add the first dmrs to the OFDM grid
            coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][0] = *(temp_dmrs_coefs);

            for(int j = 0; j < dmrs_sequence_size_ - 1; j++) {

                /// Add DMRS coefs in the grid
                coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][2*j + 2] = *(temp_dmrs_coefs + 1);

                /// Interpolate real & im. part of each RE bw the 2 DMRS
                coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][2*j + 1].real(temp_dmrs_coefs->real() + ((temp_dmrs_coefs + 1)->real() - temp_dmrs_coefs->real()) * 0.5);
                coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][2*j + 1].imag(temp_dmrs_coefs->imag() + ((temp_dmrs_coefs + 1)->imag() - temp_dmrs_coefs->imag()) * 0.5);

                /// Interpolate norm
                coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][2*j + 1] *= abs(*(temp_dmrs_coefs + 1)) * 0.5 + abs(*(temp_dmrs_coefs)) * 0.5 / abs(coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][2*j + 1]);

                temp_dmrs_coefs++;
            }

            /// Extrapolate the value on the last subcarrier
            coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][fft_size_ - 1].real((temp_dmrs_coefs)->real() * 3 * 0.5 - (temp_dmrs_coefs - 1)->real() * 0.5);
            coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][fft_size_ - 1].imag((temp_dmrs_coefs)->imag() * 3 * 0.5 - (temp_dmrs_coefs - 1)->imag() * 0.5);

            /// Test norm
            coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][fft_size_ - 1] *= abs(*(temp_dmrs_coefs)) * 3 * 0.5 - abs(*(temp_dmrs_coefs - 1)) * 0.5 / abs(coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][fft_size_ - 1]);

            temp_dmrs_symbols++;
            temp_dmrs_coefs++; /// Increment tot get the first dmrs of the next dmrs symbol
        }

    } else { /// CDM group 1

        /// Interpolate in frequency domain
        for(int i = 0; i < dmrs_symbols_per_grid_; i++) {

            /// Add coefs to the OFDM grid at the DMRs position on subcarrier 1 and 3
            coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][1] = *(temp_dmrs_coefs);
            //coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][3] = *(temp_dmrs_coefs + 1);

            /// Extrapolate the value on subcarrier 0
            coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][0].real(((temp_dmrs_coefs + 1)->real() + temp_dmrs_coefs->real()) * 0.5);
            coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][0].imag(((temp_dmrs_coefs + 1)->imag() + temp_dmrs_coefs->imag()) * 0.5);

            /// Interpolate norm
            coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][0] *= (abs(*(temp_dmrs_coefs + 1)) + abs(*(temp_dmrs_coefs))) * 0.5 / abs(coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][0]);

            /// Loop on the remaining subcarrier starting from subcarrier 1
            for(int j = 1; j < dmrs_sequence_size_; j++) {

                /// Add coefs to the OFDM grid at the DMRs positions
                coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][2*j + 1] = *(temp_dmrs_coefs + 1);

                /// Interpolate real & im. part of each RE bw the 2 DMRS
                coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][2*j].real(((temp_dmrs_coefs + 1)->real() + temp_dmrs_coefs->real()) * 0.5);
                coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][2*j].imag(((temp_dmrs_coefs + 1)->imag() + temp_dmrs_coefs->imag()) * 0.5);

                /// Interpolate norm
                coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][2*j] *= abs(*(temp_dmrs_coefs + 1)) * 0.5 + abs(*(temp_dmrs_coefs)) * 0.5 / abs(coefGrid[*(temp_dmrs_symbols) - pdsch_start_symbol_][2*j]);

                temp_dmrs_coefs++;
            }

            temp_dmrs_symbols++;
            temp_dmrs_coefs++; /// Increment tot get the first dmrs of the next dmrs symbol
        }
    }

    int step_symbols{}; // Number of OFDM symbols separating the lower and upper DMRS
    int dmrs_symbol{};
    int last_dmrs_symbol = *(temp_dmrs_symbols);

    /// Interpolation en temps
    /// Only one DMRS in the PDSCH
    if(dmrs_symbols_per_grid_ == 1) {

        temp_dmrs_symbols = dmrs_symbols_;

        /// Get symbol containing DMRS
        dmrs_symbol = *(temp_dmrs_symbols) - pdsch_start_symbol_;

        /// Apply the same coefficients in time domain on each subcarrier
        for(int symbol = 0; symbol < symbols_per_grid_; symbol++) {

            /// do not loop on dmrs symbol
            if(symbol == dmrs_symbol) {
                continue;
            }

            for (int j = 0; j < fft_size_; j++) {
                coefGrid[symbol][j] = coefGrid[dmrs_symbol][j];
            }
        }

    } else {

            /// Reset pointer
            temp_dmrs_symbols = dmrs_symbols_;

            /// Get lower and upper symbols containing DMRS
            lower_dmrs_symbol = *(temp_dmrs_symbols) - pdsch_start_symbol_;
            upper_dmrs_symbol = *(temp_dmrs_symbols + 1) - pdsch_start_symbol_;

            /// Number of symbols between lower and upper_symbol
            step_symbols = upper_dmrs_symbol - lower_dmrs_symbol;

            for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {

                //output << "Symbol number : " << symbol << endl;

                /// If current symbol is a DMRS symbol, do not interpolate on this symbol
                if ((symbol == (*(temp_dmrs_symbols) - pdsch_start_symbol_)) or
                    (symbol == (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_))) {
                    continue;
                }

                /// If current symbol is greater than the upper DMRS symbol,
                /// update lower and upper DMRS coefs
                if ((symbol > (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_)) and (symbol < last_dmrs_symbol)) {
                    temp_dmrs_symbols++;
                    lower_dmrs_symbol = *(temp_dmrs_symbols) - pdsch_start_symbol_;
                    upper_dmrs_symbol = *(temp_dmrs_symbols + 1) - pdsch_start_symbol_;
                    step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
                }

                for (int sc = 0; sc < fft_size_; sc++) {

                    coefGrid[symbol][sc].real(coefGrid[lower_dmrs_symbol][sc].real() +
                                             (coefGrid[upper_dmrs_symbol][sc].real() - coefGrid[lower_dmrs_symbol][sc].real()) * (symbol - lower_dmrs_symbol) / step_symbols);
                    coefGrid[symbol][sc].imag(coefGrid[lower_dmrs_symbol][sc].imag() +
                                             (coefGrid[upper_dmrs_symbol][sc].imag() - coefGrid[lower_dmrs_symbol][sc].imag()) * (symbol - lower_dmrs_symbol) / step_symbols);

                    /// Interpolate norm and renormalize
                    coefGrid[symbol][sc] *= (abs(coefGrid[upper_dmrs_symbol][sc]) - abs( coefGrid[lower_dmrs_symbol][sc])) / step_symbols * (symbol - lower_dmrs_symbol) + abs(coefGrid[lower_dmrs_symbol][sc]) / abs(coefGrid[symbol][sc]);

                    //output << *(temp_coef_grid) << endl;
                }
            }
    }
}

void compute_squared_norms(complex<float> * coef_grid_,
                   float * squared_norms_,
                   const int &symbols_per_grid_,
                   const int &fft_size_) {

    for(int symbol = 0; symbol < symbols_per_grid_; symbol++) {
        for(int sc = 0; sc < fft_size_; sc++) {
            squared_norms_[symbol * fft_size_ + sc] = coef_grid_[symbol * fft_size_ + sc].real() * coef_grid_[symbol * fft_size_ + sc].real()
                    + coef_grid_[symbol * fft_size_ + sc].imag() * coef_grid_[symbol * fft_size_ + symbol].imag();
        }
    }
}

/*************************************************** Séparation interpolation ****************************************/
void call_interp_functions(complex<float> * coef_grid, /// TODO: mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                           complex<float> * dmrs_coefs_, // TODO : mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                           int cdm_group_number, /// Number of the DMRS CDM group to be interpolated, 0 or 1
                           int cdm_group_size,
                           int * dmrs_symbols_,
                           int pdsch_start_symbol_,
                           int dmrs_symbols_per_grid_,
                           int dmrs_sequence_size_,
                           int fft_size_,
                           int symbols_per_grid_) {
    /// Optimizations if 2 DMRS ports are mutliplexed by OCC in the CDM group
    if(cdm_group_size > 1) {
        /// Interpolation of DMRS from CDM group 0
        if (cdm_group_number == 0) {
            interp_cdm_group0_2(coef_grid,
                                dmrs_coefs_,
                                dmrs_symbols_,
                                pdsch_start_symbol_,
                                dmrs_symbols_per_grid_,
                                dmrs_sequence_size_,
                                fft_size_,
                                symbols_per_grid_);
        } else { /// CDM group 1
            interp_cdm_group1_2(coef_grid,
                                dmrs_coefs_,
                                dmrs_symbols_,
                                pdsch_start_symbol_,
                                dmrs_symbols_per_grid_,
                                dmrs_sequence_size_,
                                fft_size_,
                                symbols_per_grid_);
        }
    } else { /// No OCC used
        /// Interpolation of DMRS from CDM group 0
        if (cdm_group_number == 0) {
            interp_cdm_group0_1(coef_grid,
                                dmrs_coefs_,
                                dmrs_symbols_,
                                pdsch_start_symbol_,
                                dmrs_symbols_per_grid_,
                                dmrs_sequence_size_,
                                fft_size_,
                                symbols_per_grid_);
        } else { /// CDM group 1
            interp_cdm_group1_1(coef_grid,
                                dmrs_coefs_,
                                dmrs_symbols_,
                                pdsch_start_symbol_,
                                dmrs_symbols_per_grid_,
                                dmrs_sequence_size_,
                                fft_size_,
                                symbols_per_grid_);
        }
    }
}

/********************************************************************************************************************/


/**************************************** TEST réalignement ********************************************************/
void interpolate_coefs(complex<float> coef_grid[][MAX_RX_PORTS][MAX_TX_PORTS], /// TODO: mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                       complex<float> * dmrs_coefs_, // TODO : mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
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
                       int nb_rx) {

    complex<float> (* temp_coef_grid)[MAX_RX_PORTS][MAX_TX_PORTS];
    complex<float> * temp_dmrs_coefs;
    int step_symbols;
    step_symbols = 0;

    int last_dmrs_symbol = *(dmrs_symbols_ + dmrs_symbols_per_grid_ - 1);
    int *temp_dmrs_symbols = dmrs_symbols_;

    complex<float> (* lower_dmrs_coef)[MAX_RX_PORTS][MAX_TX_PORTS];
    complex<float> (* upper_dmrs_coef)[MAX_RX_PORTS][MAX_TX_PORTS];

    step_symbols = 0;

    /// Optimizations if 2 DMRS ports are mutliplexed by OCC in the CDM group
    if(cdm_group_size > 1) {
        /// Interpolation of DMRS from CDM group 0
        if (cdm_group_number == 0) {
            /// Fill the grid with DMRS
            for (int symbol = 0; symbol < dmrs_symbols_per_grid_; symbol++) {
                temp_coef_grid = coef_grid + (dmrs_symbols_[symbol] - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first subcarrier of the current symbol
                temp_dmrs_coefs = dmrs_coefs_ + symbol * dmrs_sequence_size_;
                for (int sc = 0; sc < dmrs_sequence_size_; sc++) {
                    temp_coef_grid[2 * sc][rx_port_index_][tx_port_index_] = temp_dmrs_coefs[sc];
                }
            }
            temp_dmrs_coefs = dmrs_coefs_;
            /// interpolate in frequency domain
            for (int *temp_dmrs_symbols = dmrs_symbols_;
                 temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {
                temp_coef_grid = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first subcarrier of the current symbol
                for (int sc = 0; sc < fft_size_ / 4 - 1; sc++) {
                    /// Asign the same value to the subcarrier located in-between the two DMRS
                    *(temp_coef_grid + 1)[rx_port_index_][tx_port_index_] = *(temp_coef_grid)[rx_port_index_][tx_port_index_];
                    /// Interpolate real and imaginary part of the subcarrier located after the second DMRS
                    (temp_coef_grid + 3)[rx_port_index_][tx_port_index_]->real((real(*(temp_coef_grid + 4)[rx_port_index_][tx_port_index_]) + real(*(temp_coef_grid + 2)[rx_port_index_][tx_port_index_])) * 0.5);
                    (temp_coef_grid + 3)[rx_port_index_][tx_port_index_]->imag((imag(*(temp_coef_grid + 4)[rx_port_index_][tx_port_index_]) + imag(*(temp_coef_grid + 2)[rx_port_index_][tx_port_index_])) * 0.5);
                    /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                    *(temp_coef_grid + 3) *=
                            ((abs(*(temp_coef_grid + 4)) + abs(*(temp_coef_grid + 2))) * 0.5) /
                            abs(*(temp_coef_grid + 3));
#endif
                    temp_coef_grid += 4; /// Jump to the next sc to be interpolated
                }
                /// Assign the same value to the four last DMRS subcarriers
                *(temp_coef_grid + 1)[rx_port_index_][tx_port_index_] = *(temp_coef_grid)[rx_port_index_][tx_port_index_];
                *(temp_coef_grid + 3)[rx_port_index_][tx_port_index_] = *(temp_coef_grid)[rx_port_index_][tx_port_index_];
            }
        } else { /// CDM group 1
            /// Fill the grid with DMRS
            for (int symbol = 0; symbol < dmrs_symbols_per_grid_; symbol++) {
                temp_coef_grid = coef_grid + (dmrs_symbols_[symbol] - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first subcarrier of the current symbol
                temp_dmrs_coefs = dmrs_coefs_ + symbol * dmrs_sequence_size_;
                for (int sc = 0; sc < dmrs_sequence_size_; sc++) {
                    temp_coef_grid[2 * sc + 1][rx_port_index_][tx_port_index_] = temp_dmrs_coefs[sc];
                }
            }

            /// interpolate in frequency domain
            for (int *temp_dmrs_symbols = dmrs_symbols_;
                 temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {
                temp_coef_grid = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first RE to be interpolated (on sc number 0)
                /// Assign the same value for the first DMRS subcarriers
                *(temp_coef_grid)[rx_port_index_][tx_port_index_] = *(temp_coef_grid + 1)[rx_port_index_][tx_port_index_];
                *(temp_coef_grid + 2)[rx_port_index_][tx_port_index_] = *(temp_coef_grid + 1)[rx_port_index_][tx_port_index_];
                temp_coef_grid += 4; /// Jump to the next sc to be interpolated
                for (int sc = 1; sc < fft_size_/4; sc++) {
                    /// Assign the same value to the subcarrier in-between the two DMRS subcarriers
                    *(temp_coef_grid + 2)[rx_port_index_][tx_port_index_] = *(temp_coef_grid + 1)[rx_port_index_][tx_port_index_];
                    /// Interpolate the subcarrier below the first DMRS subcarrier/in-between the two OCC
                    (temp_coef_grid)[rx_port_index_][tx_port_index_]->real((real(*(temp_coef_grid + 1)[rx_port_index_][tx_port_index_]) + real(*(temp_coef_grid - 1)[rx_port_index_][tx_port_index_])) * 0.5);
                    (temp_coef_grid)[rx_port_index_][tx_port_index_]->imag((imag(*(temp_coef_grid + 1)[rx_port_index_][tx_port_index_]) + imag(*(temp_coef_grid - 1)[rx_port_index_][tx_port_index_])) * 0.5);
                    /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                    *(temp_coef_grid) *=
                            ((abs(*(temp_coef_grid + 1)) + abs(*(temp_coef_grid - 1))) * 0.5) / abs(*(temp_coef_grid));
#endif
                    temp_coef_grid += 4; /// Jump to the next sc to be interpolated
                }
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
                    *(temp_coef_grid)[rx_port_index_][tx_port_index_] = *(lower_dmrs_coef)[rx_port_index_][tx_port_index_];
                    temp_coef_grid++;
                    lower_dmrs_coef++;
                }
            }
        } else {
            /// Reset pointers
            temp_coef_grid = coef_grid;
            step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
            int step = 0;
            //int lower_dmrs_symbol_offset = (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
            //int upper_dmrs_symbol_offset = (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
            lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
            upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
            if (cdm_group_number == 0) {
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
                        //lower_dmrs_symbol_offset = (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
                        //upper_dmrs_symbol_offset = (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
                        //lower_dmrs_coef = coef_grid + lower_dmrs_symbol_offset;
                        lower_dmrs_coef = upper_dmrs_coef;
                        upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
                        step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
                        //upper_dmrs_coef += step_symbols * fft_size_;
                    }

                    step = (symbol - *(temp_dmrs_symbols) + pdsch_start_symbol_); // / step_symbols;

                    for (int sc = 0; sc < fft_size_/4 - 1; sc++) {
                        /// Interpolate on the first subcarrier
                        (temp_coef_grid)[rx_port_index_][tx_port_index_]->real((lower_dmrs_coef)[rx_port_index_][tx_port_index_]->real() +
                                               ((upper_dmrs_coef)[rx_port_index_][tx_port_index_]->real() - (lower_dmrs_coef)[rx_port_index_][tx_port_index_]->real()) *
                                               step / step_symbols);
                        (temp_coef_grid)[rx_port_index_][tx_port_index_]->imag((lower_dmrs_coef)[rx_port_index_][tx_port_index_]->imag() +
                                               ((upper_dmrs_coef)[rx_port_index_][tx_port_index_]->imag() - (lower_dmrs_coef)[rx_port_index_][tx_port_index_]->imag()) *
                                               step / step_symbols);
                        /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                        *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) * step +
                                              abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));
#endif
                        /// Assign the same value to the 2 next subcarriers
                        *(temp_coef_grid + 1)[rx_port_index_][tx_port_index_] = *(temp_coef_grid)[rx_port_index_][tx_port_index_];
                        *(temp_coef_grid + 2)[rx_port_index_][tx_port_index_] = *(temp_coef_grid)[rx_port_index_][tx_port_index_];
                        /// Interpolate on subcarrier number 3
                        /// Interpolate on the first subcarrier
                        (temp_coef_grid + 3)[rx_port_index_][tx_port_index_]->real((lower_dmrs_coef + 3)[rx_port_index_][tx_port_index_]->real() +
                                                   ((upper_dmrs_coef + 3)[rx_port_index_][tx_port_index_]->real() - (lower_dmrs_coef + 3)[rx_port_index_][tx_port_index_]->real()) *
                                                   step / step_symbols);
                        (temp_coef_grid + 3)[rx_port_index_][tx_port_index_]->imag((lower_dmrs_coef + 3)[rx_port_index_][tx_port_index_]->imag() +
                                                   ((upper_dmrs_coef + 3)[rx_port_index_][tx_port_index_]->imag() - (lower_dmrs_coef + 3)[rx_port_index_][tx_port_index_]->imag()) *
                                                   step / step_symbols);
                        /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                        *(temp_coef_grid + 3) *= ((abs(*(upper_dmrs_coef + 3)) - abs(*(lower_dmrs_coef + 3))) *
                                                  step / step_symbols +
                                                  abs(*(lower_dmrs_coef + 3))) / abs(*(temp_coef_grid + 3));
#endif
                        temp_coef_grid  += 4;
                        lower_dmrs_coef += 4;
                        upper_dmrs_coef += 4;
                    }

                    /// Interpolate on first subcarrier of the 4 last subcarriers
                    /// Interpolate on the first subcarrier
                    (temp_coef_grid)[rx_port_index_][tx_port_index_]->real(lower_dmrs_coef[rx_port_index_][tx_port_index_]->real() +
                                           (upper_dmrs_coef[rx_port_index_][tx_port_index_]->real() - lower_dmrs_coef[rx_port_index_][tx_port_index_]->real()) *
                                           step / step_symbols);
                    (temp_coef_grid)[rx_port_index_][tx_port_index_]->imag(lower_dmrs_coef[rx_port_index_][tx_port_index_]->imag() +
                                           (upper_dmrs_coef[rx_port_index_][tx_port_index_]->imag() - lower_dmrs_coef[rx_port_index_][tx_port_index_]->imag()) *
                                           step / step_symbols);
                    /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                    *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) * step / step_symbols +
                                          abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));
#endif
                    /// Assign the value to the last 4 subcarriers
                    *(temp_coef_grid + 1)[rx_port_index_][tx_port_index_] = *(temp_coef_grid)[rx_port_index_][tx_port_index_];
                    *(temp_coef_grid + 2)[rx_port_index_][tx_port_index_] = *(temp_coef_grid)[rx_port_index_][tx_port_index_];
                    *(temp_coef_grid + 3)[rx_port_index_][tx_port_index_] = *(temp_coef_grid)[rx_port_index_][tx_port_index_];
                    temp_coef_grid  += 4; /// Jump to the first subcarrier of the next sytmbol
                    /// Reinitialize the pointers to the two DMRS symbols used for interpolation
                    lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
                    upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
                    //lower_dmrs_coef -= fft_size_ - 4;
                    //upper_dmrs_coef -= fft_size_ - 4;
                }
            }
                /// CDM group 1
            else {
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
                        //lower_dmrs_symbol_offset = (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
                        //upper_dmrs_symbol_offset = (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
                        //lower_dmrs_coef = coef_grid + lower_dmrs_symbol_offset;
                        lower_dmrs_coef = upper_dmrs_coef;
                        upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
                        step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
                        //upper_dmrs_coef += step_symbols * fft_size_;
                    }
                    step = (symbol - *(temp_dmrs_symbols) + pdsch_start_symbol_); // / step_symbols;
                    /// Interpolate on the first subcarrier
                    (temp_coef_grid)[rx_port_index_][tx_port_index_]->real(lower_dmrs_coef[rx_port_index_][tx_port_index_]->real() +
                                           (upper_dmrs_coef[rx_port_index_][tx_port_index_]->real() - lower_dmrs_coef[rx_port_index_][tx_port_index_]->real()) *
                                           step / step_symbols);
                    (temp_coef_grid)[rx_port_index_][tx_port_index_]->imag(lower_dmrs_coef[rx_port_index_][tx_port_index_]->imag() +
                                           (upper_dmrs_coef[rx_port_index_][tx_port_index_]->imag() - lower_dmrs_coef[rx_port_index_][tx_port_index_]->imag()) *
                                           step / step_symbols);
                    //*(temp_coef_grid) = *(lower_dmrs_coef) + (*(upper_dmrs_coef) - *(lower_dmrs_coef)) * step;

                    /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                    *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                          step / step_symbols +
                                          abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));
#endif
                    /// Assign the same value to the 3 next subcarriers
                    *(temp_coef_grid + 1)[rx_port_index_][tx_port_index_] = *(temp_coef_grid)[rx_port_index_][tx_port_index_];
                    *(temp_coef_grid + 2)[rx_port_index_][tx_port_index_] = *(temp_coef_grid)[rx_port_index_][tx_port_index_];
                    *(temp_coef_grid + 3)[rx_port_index_][tx_port_index_] = *(temp_coef_grid)[rx_port_index_][tx_port_index_];
                    temp_coef_grid  += 4; /// Jump to the next sc to be interpolated
                    lower_dmrs_coef += 4;
                    upper_dmrs_coef += 4;
                    for (int sc = 1; sc < fft_size_/4; sc++) {
                        /// Interpolate on the first subcarrier
                        (temp_coef_grid)[rx_port_index_][tx_port_index_]->real(lower_dmrs_coef[rx_port_index_][tx_port_index_]->real() +
                                               (upper_dmrs_coef[rx_port_index_][tx_port_index_]->real() - lower_dmrs_coef[rx_port_index_][tx_port_index_]->real()) *
                                               step / step_symbols);
                        (temp_coef_grid)[rx_port_index_][tx_port_index_]->imag(lower_dmrs_coef[rx_port_index_][tx_port_index_]->imag() +
                                               (upper_dmrs_coef[rx_port_index_][tx_port_index_]->imag() - lower_dmrs_coef[rx_port_index_][tx_port_index_]->imag()) *
                                               step / step_symbols);
                        /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                        *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                              step / step_symbols +
                                              abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));
#endif

                        /// Interpolate on subcarrier number 3
                        (temp_coef_grid + 3)[rx_port_index_][tx_port_index_]->real((lower_dmrs_coef + 3)[rx_port_index_][tx_port_index_]->real() +
                                                   ((upper_dmrs_coef + 3)[rx_port_index_][tx_port_index_]->real() - (lower_dmrs_coef + 3)[rx_port_index_][tx_port_index_]->real()) *
                                                   step / step_symbols);
                        (temp_coef_grid + 3)[rx_port_index_][tx_port_index_]->imag((lower_dmrs_coef + 3)[rx_port_index_][tx_port_index_]->imag() +
                                                   ((upper_dmrs_coef + 3)[rx_port_index_][tx_port_index_]->imag() - (lower_dmrs_coef + 3)[rx_port_index_][tx_port_index_]->imag()) *
                                                   step / step_symbols);
                        /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                        *(temp_coef_grid + 3) *= ((abs(*(upper_dmrs_coef + 3)) - abs(*(lower_dmrs_coef + 3))) *
                                                  step / step_symbols +
                                                  abs(*(lower_dmrs_coef + 3))) / abs(*(temp_coef_grid + 3));
#endif

                        /// Assign the same value to the subcarriers 1 and 2
                        *(temp_coef_grid + 1)[rx_port_index_][tx_port_index_] = *(temp_coef_grid + 3)[rx_port_index_][tx_port_index_];
                        *(temp_coef_grid + 2)[rx_port_index_][tx_port_index_] = *(temp_coef_grid + 3)[rx_port_index_][tx_port_index_];
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
    } else { /// No OCC used
        /// Interpolation of DMRS from CDM group 0
        if (cdm_group_number == 0) {
            /// Fill the grid with DMRS
            for (int symbol = 0; symbol < dmrs_symbols_per_grid_; symbol++) {
                temp_coef_grid = coef_grid + (dmrs_symbols_[symbol] - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first subcarrier of the current symbol
                temp_dmrs_coefs = dmrs_coefs_ + symbol * dmrs_sequence_size_;
                for(int sc = 0; sc < dmrs_sequence_size_; sc++) {
                    temp_coef_grid[2*sc][rx_port_index_][tx_port_index_] = temp_dmrs_coefs[sc];
                }
            }

            /// interpolate in frequency domain
            for (int *temp_dmrs_symbols = dmrs_symbols_;
                 temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {

                //output << "Symbol number : " << *(temp_dmrs_symbols) << endl;

                temp_coef_grid = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first RE of the current symbol

                temp_dmrs_coefs = dmrs_coefs_ + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                              fft_size_; /// Initialize to the first RE containing DMRS of the current symbol

                /// Interpolate every odd subcarrier
                for (int sc = 0; sc < fft_size_; sc+=2) {

                    /// Interpolate real and imaginary part
                    temp_coef_grid[sc][rx_port_index_][tx_port_index_].real((temp_dmrs_coefs[sc + 1].real() + temp_dmrs_coefs[sc - 1].real()) * 0.5);
                    temp_coef_grid[sc][rx_port_index_][tx_port_index_].imag((temp_dmrs_coefs[sc + 1].imag() + temp_dmrs_coefs[sc - 1].imag()) * 0.5);

                    /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                    temp_coef_grid[sc] *=
                            ((abs(temp_dmrs_coefs[sc + 1]) + abs(temp_dmrs_coefs[sc - 1])) * 0.5) / abs(temp_coef_grid[sc]);
#endif
                }

                /// Extrapolate the last value
                temp_coef_grid[fft_size_ - 1][rx_port_index_][tx_port_index_].real((temp_dmrs_coefs[fft_size_ - 2].real() * 3 - temp_dmrs_coefs[fft_size_ - 4].real()) * 0.5);
                temp_coef_grid[fft_size_ - 1][rx_port_index_][tx_port_index_].imag((temp_dmrs_coefs[fft_size_ - 2].imag() * 3 - temp_dmrs_coefs[fft_size_ - 4].imag()) * 0.5);
                /*
                /// Interpolate norm and renormalize
                temp_coef_grid[fft_size_ - 1] *= (abs(temp_dmrs_coefs[fft_size_ - 2]) * 3 * 0.5 - abs(temp_dmrs_coefs[fft_size_ - 4]) * 0.5) /
                                                 abs(temp_coef_grid[fft_size_ - 1]); */
            }

        } else { /// CDM group 1

            /// Fill the grid with DMRS
            for (int symbol = 0; symbol < dmrs_symbols_per_grid_; symbol++) {

                temp_coef_grid = coef_grid + (dmrs_symbols_[symbol] - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first subcarrier of the current symbol

                temp_dmrs_coefs = dmrs_coefs_ + symbol * dmrs_sequence_size_;

                for(int sc = 0; sc < dmrs_sequence_size_; sc++) {
                    temp_coef_grid[2*sc + 1][rx_port_index_][tx_port_index_] = temp_dmrs_coefs[sc];
                }
            }

            /// interpolate in frequency domain
            for (int *temp_dmrs_symbols = dmrs_symbols_;
                 temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {

                //output << "Symbol number : " << *(temp_dmrs_symbols) << endl;

                temp_coef_grid = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first RE to be interpolated (on sc number 0)

                temp_dmrs_coefs = dmrs_coefs_ + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                              fft_size_; /// Initialize to the first RE containing DMRS of the current symbol

                /// Extrapolate first value
                temp_coef_grid[0][rx_port_index_][tx_port_index_].real((temp_dmrs_coefs[1].real() * 3 - temp_dmrs_coefs[3].real()) * 0.5);
                temp_coef_grid[0][rx_port_index_][tx_port_index_].imag((temp_dmrs_coefs[1].imag() * 3 - temp_dmrs_coefs[3].imag()) * 0.5);
                /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                temp_coef_grid[0] *= (abs(temp_dmrs_coefs[1]) * 3 * 0.5 - abs(temp_dmrs_coefs[3]) * 0.5) / abs(temp_coef_grid[0]);
#endif

                //output << *(temp_coef_grid) << endl;

                /// Add first DMRS RE to the grid
                //*(temp_coef_grid + 1) = *(temp_dmrs_coefs);

                //output << *(temp_coef_grid + 1) << endl;

                //temp_coef_grid += 2; /// Jump to the next sc to be interpolated

                for (int sc = 2; sc < fft_size_; sc+=2) {

                    /// Add next DMRS RE to the grid
                    //*(temp_coef_grid + 1) = *(temp_dmrs_coefs + 1);

                    temp_coef_grid[sc][rx_port_index_][tx_port_index_].real((temp_dmrs_coefs[sc + 1].real() + temp_dmrs_coefs[sc - 1].real()) * 0.5);
                    temp_coef_grid[sc][rx_port_index_][tx_port_index_].imag((temp_dmrs_coefs[sc + 1].imag() + temp_dmrs_coefs[sc - 1].imag()) * 0.5);

                    /*
                    /// Interpolate norm and renormalize
                    temp_coef_grid[sc][rx_port_index_][tx_port_index_] *=
                            ((abs(temp_dmrs_coefs[sc + 1]) + abs(temp_dmrs_coefs[sc - 1])) * 0.5) /
                            abs(temp_dmrs_coefs[sc]); */

                    //output << *(temp_coef_grid) << endl;
                    //output << *(temp_coef_grid + 1) << endl;
                    //temp_coef_grid += 2; /// Jump to the next sc to be interpolated
                    //temp_dmrs_coefs++;
                }
                //temp_coef_grid++;
                //temp_dmrs_coefs++;
            }
        }
        //t2 = std::chrono::steady_clock::now();
        //cout << "Duration of interpolation in frequency domain : " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;

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
                    *(temp_coef_grid)[rx_port_index_][tx_port_index_] = *(lower_dmrs_coef)[rx_port_index_][tx_port_index_];
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

            if(cdm_group_number == 0) {
                lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
                upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;

                for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {

                    //output << "Symbol number : " << symbol << endl;

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

                    step = 1.0f * (symbol - (*temp_dmrs_symbols) + pdsch_start_symbol_) / step_symbols;

                    for (int sc = 0; sc < fft_size_; sc++) {

                        (temp_coef_grid)[rx_port_index_][tx_port_index_]->real(lower_dmrs_coef[rx_port_index_][tx_port_index_]->real() +
                                               (upper_dmrs_coef[rx_port_index_][tx_port_index_]->real() - lower_dmrs_coef[rx_port_index_][tx_port_index_]->real()) *
                                               step);
                        (temp_coef_grid)[rx_port_index_][tx_port_index_]->imag(lower_dmrs_coef[rx_port_index_][tx_port_index_]->imag() +
                                               (upper_dmrs_coef[rx_port_index_][tx_port_index_]->imag() - lower_dmrs_coef[rx_port_index_][tx_port_index_]->imag()) *
                                               step);

                        /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                        *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                              step +
                                              abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));
#endif
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

            } else {
                lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_ + 1;
                upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_ + 1;

                for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {
                    //output << "Symbol number : " << symbol << endl;

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
                    (temp_coef_grid)[rx_port_index_][tx_port_index_]->real(lower_dmrs_coef[rx_port_index_][tx_port_index_]->real() +
                                           (upper_dmrs_coef[rx_port_index_][tx_port_index_]->real() - lower_dmrs_coef[rx_port_index_][tx_port_index_]->real()) *
                                           step);
                    (temp_coef_grid)[rx_port_index_][tx_port_index_]->imag(lower_dmrs_coef[rx_port_index_][tx_port_index_]->imag() +
                                           (upper_dmrs_coef[rx_port_index_][tx_port_index_]->imag() - lower_dmrs_coef[rx_port_index_][tx_port_index_]->imag()) *
                                           step);
                    //*(temp_coef_grid) = *(lower_dmrs_coef) + (*(upper_dmrs_coef) - *(lower_dmrs_coef)) * step;

                    /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                    *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                          step +
                                          abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));
#endif
                    /// Assign the same value to the 3 next subcarriers
                    *(temp_coef_grid + 1)[rx_port_index_][tx_port_index_] = *(temp_coef_grid)[rx_port_index_][tx_port_index_];
                    *(temp_coef_grid + 2)[rx_port_index_][tx_port_index_] = *(temp_coef_grid)[rx_port_index_][tx_port_index_];
                    *(temp_coef_grid + 3)[rx_port_index_][tx_port_index_] = *(temp_coef_grid)[rx_port_index_][tx_port_index_];

                    temp_coef_grid  += 4; /// Jump to the next sc to be interpolated
                    lower_dmrs_coef += 4;
                    upper_dmrs_coef += 4;

                    for (int sc = 1; sc < fft_size_/4; sc ++) {

                        /// Interpolate on the first subcarrier
                        (temp_coef_grid)[rx_port_index_][tx_port_index_]->real(lower_dmrs_coef[rx_port_index_][tx_port_index_]->real() +
                                               (upper_dmrs_coef[rx_port_index_][tx_port_index_]->real() - lower_dmrs_coef[rx_port_index_][tx_port_index_]->real()) *
                                               step);
                        (temp_coef_grid)[rx_port_index_][tx_port_index_]->imag(lower_dmrs_coef[rx_port_index_][tx_port_index_]->imag() +
                                               (upper_dmrs_coef[rx_port_index_][tx_port_index_]->imag() - lower_dmrs_coef[rx_port_index_][tx_port_index_]->imag()) *
                                               step);

                        //*(temp_coef_grid) = *(lower_dmrs_coef) + (*(upper_dmrs_coef) - *(lower_dmrs_coef)) * step;

                        /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                        *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                              step +
                                              abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));
#endif
                        /// Interpolate on subcarrier number 3
                        (temp_coef_grid + 3)[rx_port_index_][tx_port_index_]->real((lower_dmrs_coef + 3)[rx_port_index_][tx_port_index_]->real() +
                                                   ((upper_dmrs_coef + 3)[rx_port_index_][tx_port_index_]->real() - (lower_dmrs_coef + 3)[rx_port_index_][tx_port_index_]->real()) *
                                                   step);
                        (temp_coef_grid + 3)[rx_port_index_][tx_port_index_]->imag((lower_dmrs_coef + 3)[rx_port_index_][tx_port_index_]->imag() +
                                                   ((upper_dmrs_coef + 3)[rx_port_index_][tx_port_index_]->imag() - (lower_dmrs_coef + 3)[rx_port_index_][tx_port_index_]->imag()) *
                                                   step);

                        //*(temp_coef_grid + 3) = *(lower_dmrs_coef + 3) + (*(upper_dmrs_coef + 3) - *(lower_dmrs_coef)) * step;

                        /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                        *(temp_coef_grid + 3) *= ((abs(*(upper_dmrs_coef + 3)) - abs(*(lower_dmrs_coef + 3))) *
                                                  step +
                                                  abs(*(lower_dmrs_coef + 3))) / abs(*(temp_coef_grid + 3));
#endif
                        /// Assign the same value to the subcarriers 1 and 2
                        *(temp_coef_grid + 1)[rx_port_index_][tx_port_index_] = *(temp_coef_grid + 3)[rx_port_index_][tx_port_index_];
                        *(temp_coef_grid + 2)[rx_port_index_][tx_port_index_] = *(temp_coef_grid + 3)[rx_port_index_][tx_port_index_];

                        //output << *(temp_coef_grid) << endl;

                        temp_coef_grid  += 4;
                        lower_dmrs_coef += 4;
                        upper_dmrs_coef += 4;
                        //cout << sc << endl;
                    }

                    /// Reinitialize the pointers to the two DMRS symbols used for interpolation
                    lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_ + 1;
                    upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_ + 1;
                    //lower_dmrs_coef -= fft_size_;
                    //upper_dmrs_coef -= fft_size_;
                }
            }
        }
    }
}
/**************************************** fin TEST réalignement ****************************************************/

/************************************** Test interpolate in time domain only **************************************/
void interpolate_coefs_test1(complex<float> * coef_grid, /// TODO: mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                            complex<float> * dmrs_coefs_, // TODO : mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
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
                            int nb_rx) {

    complex<float> * temp_coef_grid;
    complex<float> * temp_dmrs_coefs;
    int step_symbols;
    step_symbols = 0;

    int last_dmrs_symbol = *(dmrs_symbols_ + dmrs_symbols_per_grid_ - 1);
    int *temp_dmrs_symbols = dmrs_symbols_;

    complex<float> * lower_dmrs_coef;
    complex<float> * upper_dmrs_coef;

    /// Optimizations if 2 DMRS ports are mutliplexed by OCC in the CDM group
    if(cdm_group_size > 1) {

        /// Interpolation of DMRS from CDM group 0
        if (cdm_group_number == 0) {
            /// Interpolate in frequency domain on the DMRS symbol
            if(dmrs_symbols_per_grid_ == 1) {
            } else {
                /// Reset pointers
                temp_coef_grid = coef_grid;
                step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
                int step = 0;
                lower_dmrs_coef = dmrs_coefs_;
                upper_dmrs_coef = dmrs_coefs_ + dmrs_sequence_size_;
                for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {
                    /// If current symbol is a DMRS symbol, do not interpolate on this symbol
                    if ((symbol == (*(temp_dmrs_symbols) - pdsch_start_symbol_)) or
                        (symbol == (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_))) {
                        continue;
                    }
                    /// If current symbol is greater than the upper DMRS symbol,
                    /// update lower and upper DMRS coefs
                    if ((symbol > (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_)) and (symbol < last_dmrs_symbol)) {
                        temp_dmrs_symbols++;
                        lower_dmrs_coef = upper_dmrs_coef;
                        upper_dmrs_coef += dmrs_sequence_size_;
                        step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
                    }

                    step = (symbol - *(temp_dmrs_symbols) + pdsch_start_symbol_); // / step_symbols;

                    for (int sc = 0; sc < fft_size_ / 4 - 1; sc++, lower_dmrs_coef += 2, upper_dmrs_coef += 2, temp_coef_grid += 4) {
                        /// Interpolate in time domain on subcarrier 1
                        /// Interpolate on the first subcarrier
                        (*temp_coef_grid).real(lower_dmrs_coef->real() +
                                                                       (upper_dmrs_coef->real() - lower_dmrs_coef->real()) *
                                                                       step / step_symbols);
                        (*temp_coef_grid).imag(lower_dmrs_coef->imag() +
                                                                       (upper_dmrs_coef->imag() - lower_dmrs_coef->imag()) *
                                                                       step / step_symbols);

                        /// Interpolate norm and renormalize
                        (*temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                                                              step / step_symbols +
                                                                              abs(*(lower_dmrs_coef))) / abs((*temp_coef_grid));

                        /// Assign the same value to the 2 next subcarriers
                        *(temp_coef_grid + 1) = (*temp_coef_grid);
                        *(temp_coef_grid + 2) = (*temp_coef_grid);
                        /// Interpolate subcarrier 3 in frequency domain
                        (*(temp_coef_grid + 3)).real(((*(temp_coef_grid + 4)).real() + (*(temp_coef_grid + 2)).real()) * 0.5);
                        (*(temp_coef_grid + 3)).imag(((*(temp_coef_grid + 4)).imag() + (*(temp_coef_grid + 2)).imag()) * 0.5);
                        /// Interpolate norm and renormalize
                        (*(temp_coef_grid + 3)) *= ((abs((*(temp_coef_grid + 4))) +
                                                                                     abs((*(temp_coef_grid + 2)))) * 0.5) / abs((*(temp_coef_grid + 3)));
                    }

                    /// Interpolate on first subcarrier of the 4 last subcarriers
                    /// Interpolate on the first subcarrier
                    (*temp_coef_grid).real(lower_dmrs_coef->real() +
                                                                           (upper_dmrs_coef->real() - lower_dmrs_coef->real()) *
                                                                           step / step_symbols);
                    (*temp_coef_grid).imag(lower_dmrs_coef->imag() +
                                                                           (upper_dmrs_coef->imag() - lower_dmrs_coef->imag()) *
                                                                           step / step_symbols);
                    /// Interpolate norm and renormalize
                    ((*temp_coef_grid)) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) * step / step_symbols +
                                                                            abs(*(lower_dmrs_coef))) / abs((*temp_coef_grid));
                    /// Assign the value to the last 4 subcarriers
                    *(temp_coef_grid + 1) = *(temp_coef_grid);
                    *(temp_coef_grid + 2) = *(temp_coef_grid);
                    *(temp_coef_grid + 3) = *(temp_coef_grid);
                    temp_coef_grid += 4; /// Jump to the first subcarrier of the next sytmbol
                    /// Reinitialize the pointers to the two DMRS symbols used for interpolation
                    lower_dmrs_coef -= dmrs_sequence_size_ - 1;
                    upper_dmrs_coef -= dmrs_sequence_size_ - 1;
                }
            }
        } else { /// CDM group 1
            /// Interpolate in time domain on DMRS subcarriers only
            if (dmrs_symbols_per_grid_ == 1) {
                /// Interpolate in frequency domain on the DMRS symbol
                /// Interpolate in time domain
            } else {
                /// Reset pointers
                temp_coef_grid = coef_grid;
                step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
                int step = 0;
                lower_dmrs_coef = dmrs_coefs_;
                upper_dmrs_coef = dmrs_coefs_ + dmrs_sequence_size_;
                for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {
                    /// If current symbol is a DMRS symbol, do not interpolate on this symbol
                    if ((symbol == (*(temp_dmrs_symbols) - pdsch_start_symbol_)) or
                        (symbol == (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_))) {
                        continue;
                    }
                    /// If current symbol is greater than the upper DMRS symbol,
                    /// update lower and upper DMRS coefs
                    if ((symbol > (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_)) and (symbol < last_dmrs_symbol)) {
                        temp_dmrs_symbols++;
                        lower_dmrs_coef = upper_dmrs_coef;
                        upper_dmrs_coef += dmrs_sequence_size_ ;
                        step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
                    }
                    step = (symbol - *(temp_dmrs_symbols) + pdsch_start_symbol_); // / step_symbols;
                    /// Interpolate on the first DMRS subcarrier
                    (*(temp_coef_grid + 1)).real(lower_dmrs_coef->real() +
                                                                                 (upper_dmrs_coef->real() - lower_dmrs_coef->real()) *
                                                                                 step / step_symbols);
                    (*(temp_coef_grid + 1)).imag(lower_dmrs_coef->imag() +
                                                                                 (upper_dmrs_coef->imag() - lower_dmrs_coef->imag()) *
                                                                                 step / step_symbols);

                    /// Interpolate norm and renormalize
                    (*(temp_coef_grid + 1)) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                                                                step / step_symbols +
                                                                                abs(*(lower_dmrs_coef))) / abs((*(temp_coef_grid + 1)));
                    /// Assign the same value to the 3 other subcarriers
                    *(temp_coef_grid) = *(temp_coef_grid + 1);
                    *(temp_coef_grid + 2) = *(temp_coef_grid + 1);
                    *(temp_coef_grid + 3) = *(temp_coef_grid + 1);
                    temp_coef_grid += 4; /// Jump to the next sc to be interpolated
                    lower_dmrs_coef += 2;
                    upper_dmrs_coef += 2;
                    for (int sc = 1; sc < fft_size_ / 4; sc++, temp_coef_grid += 4, lower_dmrs_coef += 2, upper_dmrs_coef += 2) {
                        /// Interpolate on subcarrier number 3
                        (*(temp_coef_grid + 3)).real((lower_dmrs_coef + 1)->real() +
                                                                                     ((upper_dmrs_coef + 1)->real() - (lower_dmrs_coef + 1)->real()) *
                                                                                     step / step_symbols);
                        (*(temp_coef_grid + 3)).imag((lower_dmrs_coef + 1)->imag() +
                                                                                     ((upper_dmrs_coef + 1)->imag() - (lower_dmrs_coef + 1)->imag()) *
                                                                                     step / step_symbols);
                        /// Interpolate norm and renormalize
                        (*(temp_coef_grid + 3)) *= ((abs(*(upper_dmrs_coef + 3)) - abs(*(lower_dmrs_coef + 3))) *
                                                                                    step / step_symbols +
                                                                                    abs(*(lower_dmrs_coef + 3))) / abs(*(temp_coef_grid + 3));

                        /// Assign the same value to the subcarriers 1 and 2
                        *(temp_coef_grid + 1) = *(temp_coef_grid + 3);
                        *(temp_coef_grid + 2) = *(temp_coef_grid + 3);

                        /// Interpolate in frequency domain on the first subcarrier
                        (*temp_coef_grid).real(((*(temp_coef_grid + 1)).real() + (*(temp_coef_grid - 1)).real()) *
                                                                               0.5);
                        (*temp_coef_grid).real(((*(temp_coef_grid + 1)).imag() + (*(temp_coef_grid - 1)).imag()) *
                                                                               0.5);
                        /// Interpolate norm and renormalize
                        (*temp_coef_grid) *= (abs(*(temp_coef_grid + 1)) + abs(*(temp_coef_grid - 1))) * 0.5 /
                                                                             abs((*temp_coef_grid));
                    }

                    /// Reinitialize the pointers to the two DMRS symbols used for interpolation
                    lower_dmrs_coef -= dmrs_sequence_size_ - 1;
                    upper_dmrs_coef -= dmrs_sequence_size_ - 1;
                }
            }
        }

    } else { /// No OCC used

        /// Interpolation of DMRS from CDM group 0
        if (cdm_group_number == 0) {
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

                /// Interpolate every odd subcarrier
                for (int sc = 0; sc < fft_size_; sc+=2) {

                    /// Interpolate real and imaginary part
                    temp_coef_grid[sc].real((temp_coef_grid[sc + 1].real() + temp_coef_grid[sc - 1].real()) * 0.5);
                    temp_coef_grid[sc].imag((temp_coef_grid[sc + 1].imag() + temp_coef_grid[sc - 1].imag()) * 0.5);

#if defined(INTERP_NORM)
                    /// Interpolate norm and renormalize
                    temp_coef_grid[sc] *=
                            ((abs(temp_coef_grid[sc + 1]) + abs(temp_coef_grid[sc - 1])) * 0.5)
                            / abs(temp_coef_grid[sc]);
#endif
                }

                /// Extrapolate the last value
                temp_coef_grid[fft_size_ - 1].real((temp_coef_grid[fft_size_ - 2].real() * 3 -
                                                                                    temp_coef_grid[fft_size_ - 4].real()) * 0.5);
                temp_coef_grid[fft_size_ - 1].imag((temp_coef_grid[fft_size_ - 2].imag() * 3 -
                                                                                    temp_coef_grid[fft_size_ - 4].imag()) * 0.5);
                /// Interpolate norm and renormalize
                temp_coef_grid[fft_size_ - 1] *= (abs(temp_coef_grid[fft_size_ - 2]) * 3 * 0.5 -
                                                                                  abs(temp_coef_grid[fft_size_ - 4]) * 0.5) /
                                                                                 abs(temp_coef_grid[fft_size_ - 1]);
            }

        } else { /// CDM group 1

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

                /// Extrapolate first value
                temp_coef_grid[0].real((temp_coef_grid[1].real() * 3 - temp_coef_grid[3].real()) * 0.5);
                temp_coef_grid[0].imag((temp_coef_grid[1].imag() * 3 - temp_coef_grid[3].imag()) * 0.5);

#if defined(INTERP_NORM)
                /// Interpolate norm and renormalize
                temp_coef_grid[0] *= (abs(temp_coef_grid[1]) * 3 * 0.5 -
                                             abs(temp_coef_grid[3]) * 0.5) / abs(temp_coef_grid[0]);
#endif
                for (int sc = 2; sc < fft_size_; sc+=2) {
                    temp_coef_grid[sc].real((temp_coef_grid[sc + 1].real() + temp_coef_grid[sc - 1].real()) * 0.5);
                    temp_coef_grid[sc].imag((temp_coef_grid[sc + 1].imag() + temp_coef_grid[sc - 1].imag()) * 0.5);

#if defined(INTERP_NORM)
                    /// Interpolate norm and renormalize
                    temp_coef_grid[sc] *=
                            ((abs(temp_coef_grid[sc + 1]) + abs(temp_coef_grid[sc - 1])) * 0.5) /
                            abs(temp_coef_grid[sc]);
#endif
                }
            }
        }
        //t2 = std::chrono::steady_clock::now();
        //cout << "Duration of interpolation in frequency domain : " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;

        last_dmrs_symbol = *(dmrs_symbols_ + dmrs_symbols_per_grid_ - 1);
        temp_dmrs_symbols = dmrs_symbols_;

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

                memcpy(temp_coef_grid, lower_dmrs_coef, fft_size_ * sizeof(complex<float>));
                temp_coef_grid += fft_size_;
                lower_dmrs_coef += fft_size_;
                /*
                for (int j = 0; j < fft_size_; j++) {
                    *(temp_coef_grid) = *(lower_dmrs_coef);
                    temp_coef_grid++;
                    lower_dmrs_coef++;
                } */
            }

        } else {

            /// Reset pointers
            temp_coef_grid = coef_grid;
            temp_dmrs_symbols = dmrs_symbols_;

            step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
            float step = 0;

            if(cdm_group_number == 0) {
                lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
                upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;

                for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {

                    //output << "Symbol number : " << symbol << endl;

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

                    step = 1.0f * (symbol - (*temp_dmrs_symbols) + pdsch_start_symbol_) / step_symbols;

                    for (int sc = 0; sc < fft_size_; sc++) {

                        ((*temp_coef_grid)).real((*lower_dmrs_coef).real() +
                                                                                 ((*upper_dmrs_coef).real() - (*lower_dmrs_coef).real()) *
                                                                                 step);
                        ((*temp_coef_grid)).imag((*lower_dmrs_coef).imag() +
                                                                                 ((*upper_dmrs_coef).imag() - (*lower_dmrs_coef).imag()) *
                                                                                 step);
#if defined(INTERP_NORM)
                        /// Interpolate norm and renormalize
                        ((*temp_coef_grid)) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                                                                step +
                                                                                abs(*(lower_dmrs_coef))) / abs(((*temp_coef_grid)));
#endif
                        temp_coef_grid++;
                        lower_dmrs_coef++;
                        upper_dmrs_coef++;
                    }

                    /// Reinitialize the pointers to the two DMRS symbols used for interpolation
                    lower_dmrs_coef -= fft_size_;
                    upper_dmrs_coef -= fft_size_;
                }

            } else {
                lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_ + 1;
                upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_ + 1;

                for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {
                    //output << "Symbol number : " << symbol << endl;

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
                        lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_ + 1;
                        upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_ + 1;
                        step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
                    }

                    step = 1.0f * (symbol - *(temp_dmrs_symbols) + pdsch_start_symbol_) / step_symbols;

                    /// Interpolate on the first subcarrier
                    ((*temp_coef_grid)).real((*lower_dmrs_coef).real() +
                                                                             ((*upper_dmrs_coef).real() - (*lower_dmrs_coef).real()) *
                                                                             step);
                    ((*temp_coef_grid)).imag((*lower_dmrs_coef).imag() +
                                                                             ((*upper_dmrs_coef).imag() - (*lower_dmrs_coef).imag()) *
                                                                             step);
#if defined(INTERP_NORM)
                    /// Interpolate norm and renormalize
                    ((*temp_coef_grid)) *= ((abs((*upper_dmrs_coef)) - abs((*lower_dmrs_coef))) *
                                                                            step +
                                                                            abs((*lower_dmrs_coef))) / abs(((*temp_coef_grid)));
#endif
                    /// Assign the same value to the 3 next subcarriers
                    (*(temp_coef_grid + 1)) = (*(temp_coef_grid));
                    (*(temp_coef_grid + 2)) = (*(temp_coef_grid));
                    (*(temp_coef_grid + 3)) = (*(temp_coef_grid));

                    temp_coef_grid  += 4; /// Jump to the next sc to be interpolated
                    lower_dmrs_coef += 4;
                    upper_dmrs_coef += 4;

                    for (int sc = 1; sc < fft_size_/4; sc ++) {

                        /// Interpolate on the first subcarrier
                        (*temp_coef_grid).real((*lower_dmrs_coef).real() +
                                               ((*upper_dmrs_coef).real() - (*lower_dmrs_coef).real()) *
                                               step);
                        (*temp_coef_grid).imag((*lower_dmrs_coef).imag() +
                                               ((*upper_dmrs_coef).imag() - (*lower_dmrs_coef).imag()) *
                                               step);
#if defined(INTERP_NORM)
                        /// Interpolate norm and renormalize
                        (*temp_coef_grid) *= ((abs((*upper_dmrs_coef)) - abs((*lower_dmrs_coef))) *
                                                                              step +
                                                                              abs((*lower_dmrs_coef))) / abs((*temp_coef_grid));
#endif
                        /// Interpolate on subcarrier number 3
                        (*(temp_coef_grid + 3)).real((*(lower_dmrs_coef + 3)).real() +
                                                                                     ((*(upper_dmrs_coef + 3)).real() - (*(lower_dmrs_coef + 3)).real()) *
                                                                                     step);
                        (*(temp_coef_grid + 3)).imag((*(lower_dmrs_coef + 3)).imag() +
                                                                                     ((*(upper_dmrs_coef + 3)).imag() - (*(lower_dmrs_coef + 3)).imag()) *
                                                                                     step);
#if defined(INTERP_NORM)
                        /// Interpolate norm and renormalize
                        (*(temp_coef_grid + 3)) *= ((abs((*(upper_dmrs_coef + 3))) - abs((*(lower_dmrs_coef + 3)))) *
                                                                                    step +
                                                                                    abs((*(lower_dmrs_coef + 3)))) / abs((*(temp_coef_grid + 3)));
#endif
                        /// Assign the same value to the subcarriers 1 and 2
                        (*(temp_coef_grid + 1)) = (*(temp_coef_grid + 3));
                        (*(temp_coef_grid + 2)) = (*(temp_coef_grid + 3));
                        temp_coef_grid  += 4;
                        lower_dmrs_coef += 4;
                        upper_dmrs_coef += 4;
                    }

                    /// Reinitialize the pointers to the two DMRS symbols used for interpolation
                    lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_ + 1;
                    upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_ + 1;
                }
            }
        }
    }
}
/******************************************************************************************************************/
void interpolate_coefs(complex<float> * coef_grid, /// TODO: mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                       complex<float> * dmrs_coefs_, // TODO : mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
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
                       int nb_rx) {

    complex<float> * temp_dmrs_coefs;
    complex<float> * temp_coef_grid;
    complex<float> * lower_dmrs_coef;
    complex<float> * upper_dmrs_coef;
    int step_symbols;

    step_symbols = 0;

    /// Optimizations if 2 DMRS ports are mutliplexed by OCC in the CDM group
    if(cdm_group_size > 1) {
        /// Interpolation of DMRS from CDM group 0
        if (cdm_group_number == 0) {
            /// Fill the grid with DMRS
            for (int symbol = 0; symbol < dmrs_symbols_per_grid_; symbol++) {
                temp_coef_grid = coef_grid + (dmrs_symbols_[symbol] - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first subcarrier of the current symbol
                temp_dmrs_coefs = dmrs_coefs_ + symbol * dmrs_sequence_size_;
                for (int sc = 0; sc < dmrs_sequence_size_; sc++) {
                    temp_coef_grid[2 * sc] = temp_dmrs_coefs[sc];
                }
            }
            temp_dmrs_coefs = dmrs_coefs_;
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
#if defined(INTERP_NORM)
                    *(temp_coef_grid + 3) *=
                            ((abs(*(temp_coef_grid + 4)) + abs(*(temp_coef_grid + 2))) * 0.5) /
                            abs(*(temp_coef_grid + 3));
#endif
                    temp_coef_grid += 4; /// Jump to the next sc to be interpolated
                }
                /// Assign the same value to the four last DMRS subcarriers
                *(temp_coef_grid + 1) = *(temp_coef_grid);
                *(temp_coef_grid + 3) = *(temp_coef_grid);
            }
        } else { /// CDM group 1
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
#if defined(INTERP_NORM)
                    *(temp_coef_grid) *=
                            ((abs(*(temp_coef_grid + 1)) + abs(*(temp_coef_grid - 1))) * 0.5) / abs(*(temp_coef_grid));
#endif
                    temp_coef_grid += 4; /// Jump to the next sc to be interpolated
                }
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
                memcpy(temp_coef_grid, lower_dmrs_coef, fft_size_ * sizeof(complex<float>));
                temp_coef_grid += fft_size_;
                lower_dmrs_coef += fft_size_;
                /*
                for (int j = 0; j < fft_size_; j++) {
                    *(temp_coef_grid) = *(lower_dmrs_coef);
                    temp_coef_grid++;
                    lower_dmrs_coef++;
                } */
            }
        } else {
            /// Reset pointers
            temp_coef_grid = coef_grid;
            step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
            int step = 0;
            //int lower_dmrs_symbol_offset = (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
            //int upper_dmrs_symbol_offset = (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
            lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
            upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
            if (cdm_group_number == 0) {
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
                        //lower_dmrs_symbol_offset = (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
                        //upper_dmrs_symbol_offset = (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
                        //lower_dmrs_coef = coef_grid + lower_dmrs_symbol_offset;
                        lower_dmrs_coef = upper_dmrs_coef;
                        upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
                        step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
                        //upper_dmrs_coef += step_symbols * fft_size_;
                    }

                    step = (symbol - *(temp_dmrs_symbols) + pdsch_start_symbol_); // / step_symbols;

                    for (int sc = 0; sc < fft_size_/4 - 1; sc++) {
                        /// Interpolate on the first subcarrier
                        (temp_coef_grid)->real(lower_dmrs_coef->real() +
                                               (upper_dmrs_coef->real() - lower_dmrs_coef->real()) *
                                               step / step_symbols);
                        (temp_coef_grid)->imag(lower_dmrs_coef->imag() +
                                               (upper_dmrs_coef->imag() - lower_dmrs_coef->imag()) *
                                               step / step_symbols);
                        /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                        *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) * step +
                                              abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));
#endif
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
#if defined(INTERP_NORM)
                        *(temp_coef_grid + 3) *= ((abs(*(upper_dmrs_coef + 3)) - abs(*(lower_dmrs_coef + 3))) *
                                                  step / step_symbols +
                                                  abs(*(lower_dmrs_coef + 3))) / abs(*(temp_coef_grid + 3));
#endif
                        temp_coef_grid  += 4;
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
#if defined(INTERP_NORM)
                    *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) * step / step_symbols +
                                          abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));
#endif
                    /// Assign the value to the last 4 subcarriers
                    *(temp_coef_grid + 1) = *(temp_coef_grid);
                    *(temp_coef_grid + 2) = *(temp_coef_grid);
                    *(temp_coef_grid + 3) = *(temp_coef_grid);
                    temp_coef_grid  += 4; /// Jump to the first subcarrier of the next sytmbol
                    /// Reinitialize the pointers to the two DMRS symbols used for interpolation
                    lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
                    upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
                    //lower_dmrs_coef -= fft_size_ - 4;
                    //upper_dmrs_coef -= fft_size_ - 4;
                }
            }
                /// CDM group 1
            else {
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
                        //lower_dmrs_symbol_offset = (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
                        //upper_dmrs_symbol_offset = (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
                        //lower_dmrs_coef = coef_grid + lower_dmrs_symbol_offset;
                        lower_dmrs_coef = upper_dmrs_coef;
                        upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
                        step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
                        //upper_dmrs_coef += step_symbols * fft_size_;
                    }
                    step = (symbol - *(temp_dmrs_symbols) + pdsch_start_symbol_); // / step_symbols;
                    /// Interpolate on the first subcarrier
                    (temp_coef_grid)->real(lower_dmrs_coef->real() +
                                           (upper_dmrs_coef->real() - lower_dmrs_coef->real()) *
                                           step / step_symbols);
                    (temp_coef_grid)->imag(lower_dmrs_coef->imag() +
                                           (upper_dmrs_coef->imag() - lower_dmrs_coef->imag()) *
                                           step / step_symbols);
                    //*(temp_coef_grid) = *(lower_dmrs_coef) + (*(upper_dmrs_coef) - *(lower_dmrs_coef)) * step;

                    /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                    *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                          step / step_symbols +
                                          abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));
#endif
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
#if defined(INTERP_NORM)
                        *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                              step / step_symbols +
                                              abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));
#endif

                        /// Interpolate on subcarrier number 3
                        (temp_coef_grid + 3)->real((lower_dmrs_coef + 3)->real() +
                                                   ((upper_dmrs_coef + 3)->real() - (lower_dmrs_coef + 3)->real()) *
                                                   step / step_symbols);
                        (temp_coef_grid + 3)->imag((lower_dmrs_coef + 3)->imag() +
                                                   ((upper_dmrs_coef + 3)->imag() - (lower_dmrs_coef + 3)->imag()) *
                                                   step / step_symbols);
                        /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                        *(temp_coef_grid + 3) *= ((abs(*(upper_dmrs_coef + 3)) - abs(*(lower_dmrs_coef + 3))) *
                                                  step / step_symbols +
                                                  abs(*(lower_dmrs_coef + 3))) / abs(*(temp_coef_grid + 3));
#endif

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
    } else { /// No OCC used
        /// Interpolation of DMRS from CDM group 0
        if (cdm_group_number == 0) {
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
            int symbol = 0;
            for (int *temp_dmrs_symbols = dmrs_symbols_;
                 temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++, symbol++) {

                /// Pointer to coef_grid memory locations containing PDSCH RE to be interpolated
                temp_coef_grid = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first RE of the current symbol

                temp_dmrs_coefs = dmrs_coefs_ + symbol *
                                              dmrs_sequence_size_; /// Initialize to the first RE containing DMRS of the current symbol

                /// Interpolate every odd subcarrier
                temp_coef_grid++; /// Jump to the next sc to be interpolated
                for (int sc = 0; sc < dmrs_sequence_size_ - 1; sc ++) {

                    /// Interpolate real and imaginary part
                    *(temp_coef_grid) = (temp_dmrs_coefs[sc] + temp_dmrs_coefs[sc + 1]) * 0.5f;

                    /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                    *(temp_coef_grid) *=
                            ((abs(temp_dmrs_coefs[sc]) + abs(temp_dmrs_coefs[sc + 1])) * 0.5) / abs(*(temp_coef_grid));
#endif
                    temp_coef_grid += 2;
                }

                /// Extrapolate the last value
                *(temp_coef_grid) = (temp_dmrs_coefs[dmrs_sequence_size_ - 1] * 3.0f - temp_dmrs_coefs[dmrs_sequence_size_ - 2]) * 0.5f;
#if defined(INTERP_NORM)
                /// Interpolate norm and renormalize
                *(temp_coef_grid) *= (abs(temp_dmrs_coefs[dmrs_sequence_size_ - 1]) * 3 * 0.5 - abs(temp_dmrs_coefs[dmrs_sequence_size_ - 2]) * 0.5) /
                                     abs(*(temp_coef_grid));
#endif
            }

        } else { /// CDM group 1

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
            int symbol = 0;
            for (int *temp_dmrs_symbols = dmrs_symbols_;
                 temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++, symbol++) {

                temp_coef_grid = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first RE to be interpolated (on sc number 0)

                temp_dmrs_coefs = dmrs_coefs_ + symbol *
                                              dmrs_sequence_size_; /// Initialize to the first RE containing DMRS of the current symbol

                /// Extrapolate first value
                temp_coef_grid[0] = (temp_dmrs_coefs[0] * 3.0f - temp_dmrs_coefs[1]) * 0.5f;
                /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                temp_coef_grid[0] *= (abs(temp_dmrs_coefs[0]) * 3 * 0.5 - abs(temp_dmrs_coefs[1]) * 0.5) / abs(temp_coef_grid[0]);
#endif
                temp_coef_grid += 2; /// Jump to the next sc to be interpolated
                for (int sc = 0; sc < dmrs_sequence_size_ - 1; sc++) {
                    *(temp_coef_grid) = (temp_dmrs_coefs[sc] + temp_dmrs_coefs[sc + 1]) * 0.5f;

#if defined(INTERP_NORM)
                    /// Interpolate norm and renormalize
                    *(temp_coef_grid) *=
                            ((abs(temp_dmrs_coefs[sc + 1]) + abs(temp_dmrs_coefs[sc])) * 0.5) /
                            abs(*(temp_coef_grid);
#endif
                    temp_coef_grid += 2;
                }
            }
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

                /// No need to apply values on DRMS symbol as we have interpolated in frequency domain
                if (symbol == *(dmrs_symbols_) - pdsch_start_symbol_) {
                    temp_coef_grid += fft_size_;
                    continue;
                }

                memcpy(temp_coef_grid, lower_dmrs_coef, fft_size_ * sizeof(complex<float>));
                //for (int j = 0; j < fft_size_; j++) {
                //    *(temp_coef_grid) = *(lower_dmrs_coef);
                //    temp_coef_grid++;
                //    lower_dmrs_coef++;
                //}
                temp_coef_grid += fft_size_;
                lower_dmrs_coef += fft_size_;
            }

        } else {

            /// Reset pointers
            temp_coef_grid = coef_grid;
            temp_dmrs_symbols = dmrs_symbols_;

            step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
            float step = 0;

            if(cdm_group_number == 0) {
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

                    step = 1.0f * (symbol - (*temp_dmrs_symbols) + pdsch_start_symbol_) / step_symbols;

                    for (int sc = 0; sc < fft_size_; sc++) {
                        temp_coef_grid->real((lower_dmrs_coef)->real() + ((upper_dmrs_coef)->real() - (lower_dmrs_coef)->real()) * step);
                        temp_coef_grid->imag((lower_dmrs_coef)->imag() + ((upper_dmrs_coef)->imag() - (lower_dmrs_coef)->imag()) * step);
                        /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                        *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                              step +
                                              abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));
#endif
                        temp_coef_grid++;
                        lower_dmrs_coef++;
                        upper_dmrs_coef++;
                    }

                    /// Reinitialize the pointers to the two DMRS symbols used for interpolation
                    lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
                    upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
                    //lower_dmrs_coef -= fft_size_;
                    //upper_dmrs_coef -= fft_size_;
                }

            } else {
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
                        lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
                        upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
                        step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
                    }

                    step = 1.0f * (symbol - *(temp_dmrs_symbols) + pdsch_start_symbol_) / step_symbols;

                    /// Interpolate on the first subcarrier
                    *(temp_coef_grid) = *(lower_dmrs_coef) +
                                           (*(upper_dmrs_coef) - *(lower_dmrs_coef)) *
                                           step;
                    /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                    *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                          step +
                                          abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));
#endif
                    temp_coef_grid++; /// Jump to the next sc to be interpolated
                    lower_dmrs_coef++;
                    upper_dmrs_coef++;

                    for (int sc = 1; sc < fft_size_; sc++) {
                        /*
                        *(temp_coef_grid) = *(lower_dmrs_coef) +
                                               (*(upper_dmrs_coef) - *(lower_dmrs_coef)) *
                                               step; */
                        temp_coef_grid->real((lower_dmrs_coef)->real() + ((upper_dmrs_coef)->real() - (lower_dmrs_coef)->real()) * step);
                        temp_coef_grid->imag((lower_dmrs_coef)->imag() + ((upper_dmrs_coef)->imag() - (lower_dmrs_coef)->imag()) * step);

                        /// Interpolate norm and renormalize
#if defined(INTERP_NORM)
                        *(temp_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                              step +
                                              abs(*(lower_dmrs_coef))) / abs(*(temp_coef_grid));
#endif
                        temp_coef_grid  ++;
                        lower_dmrs_coef ++;
                        upper_dmrs_coef ++;
                    }

                    /// Reinitialize the pointers to the two DMRS symbols used for interpolation
                    lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
                    upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
                    //lower_dmrs_coef -= fft_size_;
                    //upper_dmrs_coef -= fft_size_;
                }
            }
        }
    }
}

/**************************************** Test interpolation AVX ******************************************************/
#if defined(__AVX2__)
void interpolate_coefs_avx(complex<float> * coef_grid, /// TODO: mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                           complex<float> * dmrs_coefs_, /// TODO : mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
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
                           int nb_rx) {

    complex<float> *temp_dmrs_coefs;
    complex<float> *temp_dmrs_coefs2;
    complex<float> *temp_coef_grid;
    complex<float> *lower_dmrs_coef;
    complex<float> *upper_dmrs_coef;
    int step_symbols;
    __m256 lower_dmrs_vec;
    __m256 upper_dmrs_vec;
    __m256 interp;
    __m256 step_vec;
    __m256 abs_lower_coefs, abs_upper_coefs, abs_interp;
    __m256 vec1, vec2, vec3;
    __m128 lower_dmrs_vec128, upper_dmrs_vec128,
           permute_lower_dmrs, permute_upper_dmrs,
           vec1_128, vec2_128, vec3_128, vec4_128, abs_interp128, abs128;
    __m128 step_vec128 = _mm_set1_ps(0.5);
    int count_dmrs_symbol = 0;

    /****************************** Frequency domain interpolation **********************************************/
    /// Optimizations if 2 DMRS ports are mutliplexed by OCC in the CDM group
    if (cdm_group_size > 1) {

        /// Interpolation of DMRS from CDM group 0
        if (cdm_group_number == 0) {
            temp_dmrs_coefs = dmrs_coefs_;
            temp_dmrs_coefs2 = temp_dmrs_coefs + 1;
            /// interpolate in frequency domain
            for (int *temp_dmrs_symbols = dmrs_symbols_;
                 temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {
                temp_coef_grid = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first subcarrier of the current symbol
                for (int sc = 0; sc < fft_size_ - 4; sc += 4) {

                    /// Interpolate real and imag part
                    lower_dmrs_vec128 = _mm_loadu_ps((float *) temp_dmrs_coefs);
                    upper_dmrs_vec128 = _mm_loadu_ps((float *) temp_dmrs_coefs2);

                    permute_lower_dmrs = _mm_permute_ps(lower_dmrs_vec128, 0b11011000);
                    permute_upper_dmrs = _mm_permute_ps(upper_dmrs_vec128, 0b11011000);

                    vec1_128 = _mm_hadd_ps(permute_lower_dmrs, permute_upper_dmrs);
                    vec3_128 = _mm_mul_ps(vec1_128, step_vec128);
                    /// Interpolate norm
#if defined(INTERP_NORM)
                    abs128 = _mm_sqrt_ps(_mm_hadd_ps(_mm_mul_ps(lower_dmrs_vec128, lower_dmrs_vec128), _mm_mul_ps(upper_dmrs_vec128, upper_dmrs_vec128)));
                    vec4_128 = _mm_mul_ps(vec3_128, vec3_128);
                    abs_interp128 = _mm_permute_ps(_mm_sqrt_ps(_mm_hadd_ps(vec4_128, vec4_128)), 0b11011000);
                    vec3_128 = _mm_mul_ps(vec3_128, _mm_mul_ps(_mm_permute_ps(_mm_hadd_ps(abs128, abs128), 0b11011000), step_vec128));
                    vec3_128 = _mm_div_ps(vec3_128, abs_interp128);
#endif
                    /// Store oefficients into the grid
                    interp = _mm256_insertf128_ps(interp, _mm_shuffle_ps(lower_dmrs_vec128, vec3_128, 0b01000100), 0);
                    interp = _mm256_insertf128_ps(interp, _mm_shuffle_ps(lower_dmrs_vec128, vec3_128, 0b11101110), 1);
                    _mm256_storeu_ps((float *) temp_coef_grid, interp);

                    temp_coef_grid += 4; /// Jump to the next sc to be interpolated
                    temp_dmrs_coefs += 2;
                    temp_dmrs_coefs2 += 2;
                }
                /// Assign the same value to the four last DMRS subcarriers
                *(temp_coef_grid) = *(temp_dmrs_coefs);
                *(temp_coef_grid + 2) = *(temp_dmrs_coefs + 1);
                *(temp_coef_grid + 1) = *(temp_coef_grid);
                *(temp_coef_grid + 3) = *(temp_coef_grid);

                /// Reset pointers to DMRS sequence
                count_dmrs_symbol++;
                temp_dmrs_coefs = dmrs_coefs_ + count_dmrs_symbol * dmrs_sequence_size_;
                temp_dmrs_coefs2 = temp_dmrs_coefs + 1;
            }
            /**
            /// Fill the grid with DMRS
            for (int symbol = 0; symbol < dmrs_symbols_per_grid_; symbol++) {
                temp_coef_grid = coef_grid + (dmrs_symbols_[symbol] - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first subcarrier of the current symbol
                temp_dmrs_coefs = dmrs_coefs_ + symbol * dmrs_sequence_size_;
                for (int sc = 0; sc < dmrs_sequence_size_; sc++) {
                    temp_coef_grid[2 * sc] = temp_dmrs_coefs[sc];
                }
            }
            temp_dmrs_coefs = dmrs_coefs_;
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
                    //*(temp_coef_grid + 3) *=
                    //        ((abs(*(temp_coef_grid + 4)) + abs(*(temp_coef_grid + 2))) * 0.5) /
                    //        abs(*(temp_coef_grid + 3));
                    temp_coef_grid += 4; /// Jump to the next sc to be interpolated
                }
                /// Assign the same value to the four last DMRS subcarriers
                *(temp_coef_grid + 1) = *(temp_coef_grid);
                *(temp_coef_grid + 3) = *(temp_coef_grid);
            } */

        } else { /// CDM group 1
            /// interpolate in frequency domain
            temp_dmrs_coefs = dmrs_coefs_;
            for (int *temp_dmrs_symbols = dmrs_symbols_;
                 temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {
                temp_coef_grid = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first RE to be interpolated (on sc number 0)
                /// Assign the same value for the first DMRS subcarriers
                *(temp_coef_grid) = *(temp_dmrs_coefs);
                *(temp_coef_grid + 1) = *(temp_dmrs_coefs);
                *(temp_coef_grid + 2) = *(temp_dmrs_coefs);
                *(temp_coef_grid + 3) = *(temp_dmrs_coefs);
                temp_coef_grid += 4; /// Jump to the next sc to be interpolated
                temp_dmrs_coefs += 2;
                temp_dmrs_coefs2 = temp_dmrs_coefs - 1;
                for (int sc = 4; sc < fft_size_; sc += 4) {
                    /// Interpolate real and imag part
                    lower_dmrs_vec128 = _mm_loadu_ps((float *) temp_dmrs_coefs);
                    upper_dmrs_vec128 = _mm_loadu_ps((float *) temp_dmrs_coefs2);

                    permute_lower_dmrs = _mm_permute_ps(lower_dmrs_vec128, 0b11011000);
                    permute_upper_dmrs = _mm_permute_ps(upper_dmrs_vec128, 0b11011000);

                    vec1_128 = _mm_hadd_ps(permute_upper_dmrs, permute_lower_dmrs);
                    vec3_128 = _mm_mul_ps(vec1_128, step_vec128);
                    /// Interpolate norm
#if defined(INTERP_NORM)
                    abs128 = _mm_sqrt_ps(_mm_hadd_ps(_mm_mul_ps(lower_dmrs_vec128, lower_dmrs_vec128), _mm_mul_ps(upper_dmrs_vec128, upper_dmrs_vec128)));
                    vec4_128 = _mm_mul_ps(vec3_128, vec3_128);
                    abs_interp128 = _mm_permute_ps(_mm_sqrt_ps(_mm_hadd_ps(vec4_128, vec4_128)), 0b11011000);
                    vec3_128 = _mm_mul_ps(vec3_128, _mm_mul_ps(_mm_permute_ps(_mm_hadd_ps(abs128, abs128), 0b11011000), step_vec128));
                    vec3_128 = _mm_div_ps(vec3_128, abs_interp128);
#endif
                    /// Store oefficients into the grid
                    interp = _mm256_insertf128_ps(interp, _mm_shuffle_ps(vec3_128, lower_dmrs_vec128, 0b01000100), 0);
                    interp = _mm256_insertf128_ps(interp, _mm_shuffle_ps(vec3_128, lower_dmrs_vec128, 0b11101110), 1);
                    _mm256_storeu_ps((float *) temp_coef_grid, interp);

                    temp_coef_grid += 4; /// Jump to the next sc to be interpolated
                    temp_dmrs_coefs += 2;
                    temp_dmrs_coefs2 += 2;
                }
                /// Reset pointers to DMRS sequence
                count_dmrs_symbol++;
                temp_dmrs_coefs = dmrs_coefs_ + count_dmrs_symbol * dmrs_sequence_size_;
                temp_dmrs_coefs2 = temp_dmrs_coefs - 1;
            }
            /**
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
                    //*(temp_coef_grid) *=
                    //        ((abs(*(temp_coef_grid + 1)) + abs(*(temp_coef_grid - 1))) * 0.5) / abs(*(temp_coef_grid));
                    temp_coef_grid += 4; /// Jump to the next sc to be interpolated
                }
            } */
        }

    } else { /// Interpolation for a port alone in a CDM group
        /// Interpolation of DMRS from CDM group 0
        if (cdm_group_number == 0) {
            temp_dmrs_coefs = dmrs_coefs_;
            temp_dmrs_coefs2 = temp_dmrs_coefs + 1;
            /// interpolate in frequency domain
            for (int *temp_dmrs_symbols = dmrs_symbols_;
                 temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {
                temp_coef_grid = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first subcarrier of the current symbol
                for (int sc = 0; sc < fft_size_ - 4; sc += 4) {

                    /// Interpolate real and imag part
                    lower_dmrs_vec128 = _mm_loadu_ps((float *) temp_dmrs_coefs);
                    upper_dmrs_vec128 = _mm_loadu_ps((float *) temp_dmrs_coefs2);

                    permute_lower_dmrs = _mm_permute_ps(lower_dmrs_vec128, 0b11011000);
                    permute_upper_dmrs = _mm_permute_ps(upper_dmrs_vec128, 0b11011000);

                    vec1_128 = _mm_hadd_ps(permute_lower_dmrs, permute_upper_dmrs);
                    vec3_128 = _mm_mul_ps(vec1_128, step_vec128);
                    /// Interpolate norm
#if defined(INTERP_NORM)
                    abs128 = _mm_sqrt_ps(_mm_hadd_ps(_mm_mul_ps(lower_dmrs_vec128, lower_dmrs_vec128), _mm_mul_ps(upper_dmrs_vec128, upper_dmrs_vec128)));
                    vec4_128 = _mm_mul_ps(vec3_128, vec3_128);
                    abs_interp128 = _mm_permute_ps(_mm_sqrt_ps(_mm_hadd_ps(vec4_128, vec4_128)), 0b11011000);
                    vec3_128 = _mm_mul_ps(vec3_128, _mm_mul_ps(_mm_permute_ps(_mm_hadd_ps(abs128, abs128), 0b11011000), step_vec128));
                    vec3_128 = _mm_div_ps(vec3_128, abs_interp128);
#endif
                    /// Store coefficients into the grid
                    interp = _mm256_insertf128_ps(interp, _mm_shuffle_ps(lower_dmrs_vec128, vec3_128, 0b01000100), 0);
                    interp = _mm256_insertf128_ps(interp, _mm_shuffle_ps(lower_dmrs_vec128, vec3_128, 0b11101110), 1);
                    _mm256_storeu_ps((float *) temp_coef_grid, interp);

                    temp_coef_grid += 4; /// Jump to the next sc to be interpolated
                    temp_dmrs_coefs += 2;
                    temp_dmrs_coefs2 += 2;
                }
                /// Assign the DMRS
                *(temp_coef_grid) = *(temp_dmrs_coefs);
                *(temp_coef_grid + 2) = *(temp_dmrs_coefs + 1);

                /// Interpolate on the last subcarriers
                *(temp_coef_grid + 1) = 0.5f * (*(temp_coef_grid) + *(temp_coef_grid + 2));

#if defined(INTERP_NORM)
                *(temp_coef_grid + 1) *= 0.5f * (abs(*(temp_coef_grid) + abs(*(temp_coef_grid + 2)))) / abs(*(temp_coef_grid + 1));
#endif

                *(temp_coef_grid + 3) = 1.5f * *(temp_coef_grid + 2) - 0.5f * *(temp_coef_grid);

#if defined(INTERP_NORM)
                *(temp_coef_grid + 3) *= (1.5f * abs(*(temp_coef_grid)) - 0.5f * abs(*(temp_coef_grid + 2))) / abs(*(temp_coef_grid + 3));
#endif

                /// Reset pointers to DMRS sequence
                count_dmrs_symbol++;
                temp_dmrs_coefs = dmrs_coefs_ + count_dmrs_symbol * dmrs_sequence_size_;
                temp_dmrs_coefs2 = temp_dmrs_coefs + 1;
            }

        } else { /// CDM group 1
            /// interpolate in frequency domain
            temp_dmrs_coefs = dmrs_coefs_;
            for (int *temp_dmrs_symbols = dmrs_symbols_;
                 temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {
                temp_coef_grid = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first RE to be interpolated (on sc number 0)
                /// Assign DMRS
                *(temp_coef_grid + 1) = *(temp_dmrs_coefs);
                *(temp_coef_grid + 3) = *(temp_dmrs_coefs);

                /// Interpolate on non DMRS subcarriers
                *(temp_coef_grid) = 1.5f * *(temp_coef_grid + 1) - 0.5f * *(temp_coef_grid + 3);

#if defined(INTERP_NORM)
                *(temp_coef_grid) *= (1.5f * abs(*(temp_coef_grid + 1)) - 0.5f * abs(*(temp_coef_grid + 3))) / abs(*(temp_coef_grid + 2));
#endif

                *(temp_coef_grid + 2) = 0.5f * (*(temp_coef_grid + 3) + *(temp_coef_grid + 1));

#if defined(INTERP_NORM)
                *(temp_coef_grid + 2) *= 0.5f * (abs(*(temp_coef_grid + 3)) - abs(*(temp_coef_grid + 1))) / abs(*(temp_coef_grid + 2));
#endif

                temp_coef_grid += 4; /// Jump to the next sc to be interpolated
                temp_dmrs_coefs += 2;
                temp_dmrs_coefs2 = temp_dmrs_coefs - 1;
                for (int sc = 4; sc < fft_size_; sc += 4) {
                    /// Interpolate real and imag part
                    lower_dmrs_vec128 = _mm_loadu_ps((float *) temp_dmrs_coefs);
                    upper_dmrs_vec128 = _mm_loadu_ps((float *) temp_dmrs_coefs2);

                    permute_lower_dmrs = _mm_permute_ps(lower_dmrs_vec128, 0b11011000);
                    permute_upper_dmrs = _mm_permute_ps(upper_dmrs_vec128, 0b11011000);

                    vec1_128 = _mm_hadd_ps(permute_upper_dmrs, permute_lower_dmrs);
                    vec3_128 = _mm_mul_ps(vec1_128, step_vec128);
                    /// Interpolate norm
#if defined(INTERP_NORM)
                    abs128 = _mm_sqrt_ps(_mm_hadd_ps(_mm_mul_ps(lower_dmrs_vec128, lower_dmrs_vec128), _mm_mul_ps(upper_dmrs_vec128, upper_dmrs_vec128)));
                    vec4_128 = _mm_mul_ps(vec3_128, vec3_128);
                    abs_interp128 = _mm_permute_ps(_mm_sqrt_ps(_mm_hadd_ps(vec4_128, vec4_128)), 0b11011000);
                    vec3_128 = _mm_mul_ps(vec3_128, _mm_mul_ps(_mm_permute_ps(_mm_hadd_ps(abs128, abs128), 0b11011000), step_vec128));
                    vec3_128 = _mm_div_ps(vec3_128, abs_interp128);
#endif
                    /// Store coefficients into the grid
                    interp = _mm256_insertf128_ps(interp, _mm_shuffle_ps(vec3_128, lower_dmrs_vec128, 0b01000100), 0);
                    interp = _mm256_insertf128_ps(interp, _mm_shuffle_ps(vec3_128, lower_dmrs_vec128, 0b11101110), 1);
                    _mm256_storeu_ps((float *) temp_coef_grid, interp);

                    temp_coef_grid += 4; /// Jump to the next sc to be interpolated
                    temp_dmrs_coefs += 2;
                    temp_dmrs_coefs2 += 2;
                }
                /// Reset pointers to DMRS sequence
                count_dmrs_symbol++;
                temp_dmrs_coefs = dmrs_coefs_ + count_dmrs_symbol * dmrs_sequence_size_;
                temp_dmrs_coefs2 = temp_dmrs_coefs - 1;
            }
        }
    }

    /****************************** Time domain interpolation **********************************************/
        int last_dmrs_symbol = *(dmrs_symbols_ + dmrs_symbols_per_grid_ - 1);
        int *temp_dmrs_symbols = dmrs_symbols_;

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
                memcpy(temp_coef_grid, lower_dmrs_coef, fft_size_);
                /**
                for (int j = 0; j < fft_size_; j++) {
                    *(temp_coef_grid) = *(lower_dmrs_coef);
                    temp_coef_grid++;
                    lower_dmrs_coef++;
                }*/
            }
        } else {
            /// Reset pointers
            temp_coef_grid = coef_grid;
            step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
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

                step_vec = _mm256_set1_ps((float) (symbol - *(temp_dmrs_symbols) + pdsch_start_symbol_) / step_symbols);

                for (int sc = 0; sc < fft_size_; sc += 4) {
                    /// Load lower and upper coefficints on DMRS symbols
                    lower_dmrs_vec = _mm256_loadu_ps((float *) lower_dmrs_coef);
                    upper_dmrs_vec = _mm256_loadu_ps((float *) upper_dmrs_coef);

                    /// Interpolate real and imag part
                    interp = _mm256_mul_ps(_mm256_sub_ps(upper_dmrs_vec, lower_dmrs_vec), step_vec);
                    interp = _mm256_add_ps(interp, lower_dmrs_vec);

                    /// Interpolate norm
#if defined(INTERP_NORM)
                    lower_dmrs_vec = _mm256_mul_ps(lower_dmrs_vec, lower_dmrs_vec);
                    abs_lower_coefs = _mm256_sqrt_ps(_mm256_permute_ps(_mm256_hadd_ps(lower_dmrs_vec, lower_dmrs_vec), 0b11011000));
                    upper_dmrs_vec = _mm256_mul_ps(upper_dmrs_vec, upper_dmrs_vec);
                    abs_upper_coefs = _mm256_sqrt_ps(_mm256_permute_ps(_mm256_hadd_ps(upper_dmrs_vec, upper_dmrs_vec), 0b11011000));

                    vec3 = _mm256_mul_ps(interp, interp);
                    abs_interp = _mm256_sqrt_ps(_mm256_permute_ps(_mm256_hadd_ps(vec3, vec3), 0b11011000));

                    interp = _mm256_mul_ps(interp, _mm256_add_ps(_mm256_mul_ps(_mm256_sub_ps(abs_upper_coefs, abs_lower_coefs), step_vec), abs_lower_coefs));
                    interp = _mm256_div_ps(interp, abs_interp);
#endif
                    _mm256_storeu_ps((float *) temp_coef_grid, interp); /// Store coefficients into the grid
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
#endif
/**************************************** Fin test interpolation AVX **************************************************/

/**************************************** Test complex<float>[][]******************************************************/
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
                            int nb_rx) {

    int * temp_dmrs_symbols = dmrs_symbols_;

    /// Optimizations if 2 DMRS ports are mutliplexed by OCC in the CDM group
    if(cdm_group_size > 1) {
        /// Interpolation of DMRS from CDM group 0
        if (cdm_group_number == 0) {
            /// Fill the grid with DMRS
            for (int symbol = 0; symbol < dmrs_symbols_per_grid_; symbol++) {
                for (int sc = 0; sc < dmrs_sequence_size_; sc++) {
                    coef_grid[*temp_dmrs_symbols][2 * sc] = dmrs_coefs_[symbol][sc];
                }
                temp_dmrs_symbols++;
            }
            /// interpolate in frequency domain
            for (temp_dmrs_symbols = dmrs_symbols_; temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {
                for (int sc = 0; sc < fft_size_ - 4; sc+=4) {
                    /// Asign the same value to the subcarrier located in-between the two DMRS
                    coef_grid[*temp_dmrs_symbols][sc + 1] = coef_grid[*temp_dmrs_symbols][sc];
                    /// Interpolate real and imaginary part of the subcarrier located after the second DMRS
                    coef_grid[*temp_dmrs_symbols][sc + 3].real((real(coef_grid[*temp_dmrs_symbols][sc + 4]) +
                                                                real(coef_grid[*temp_dmrs_symbols][sc + 2])) * 0.5);
                    coef_grid[*temp_dmrs_symbols][sc + 3].imag((imag(coef_grid[*temp_dmrs_symbols][sc + 4]) +
                                                                imag(coef_grid[*temp_dmrs_symbols][sc + 2])) * 0.5);
                    //*(temp_coef_grid + 3) = (*(temp_coef_grid + 4) + *(temp_coef_grid + 2)) * float(0.5);

                    /// Interpolate norm and renormalize
                    coef_grid[*temp_dmrs_symbols][sc + 3] *=
                            ((abs(coef_grid[*temp_dmrs_symbols][sc + 4]) + abs(coef_grid[*temp_dmrs_symbols][sc + 2])) *
                             0.5) /
                            abs(coef_grid[*temp_dmrs_symbols][sc + 3]);
                }

                /// Assign the same value to the four last DMRS subcarriers
                coef_grid[*temp_dmrs_symbols][fft_size_ - 1] = coef_grid[*temp_dmrs_symbols][fft_size_ - 2];
                coef_grid[*temp_dmrs_symbols][fft_size_ - 3] = coef_grid[*temp_dmrs_symbols][fft_size_ - 2];
            }
        } else { /// CDM group 1

            /// Fill the grid with DMRS
            for (int symbol = 0; symbol < dmrs_symbols_per_grid_; symbol++) {
                for (int sc = 0; sc < dmrs_sequence_size_; sc++) {
                    coef_grid[*(temp_dmrs_symbols)][2 * sc + 1] = dmrs_coefs_[symbol][sc];
                }
                temp_dmrs_symbols++;
            }

            /// interpolate in frequency domain
            for (temp_dmrs_symbols = dmrs_symbols_; temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {
                coef_grid[*temp_dmrs_symbols][0] = coef_grid[*temp_dmrs_symbols][1];
                coef_grid[*temp_dmrs_symbols][2] = coef_grid[*temp_dmrs_symbols][1];

                for (int sc = 1; sc < fft_size_; sc += 4) {
                    /// Assign the same value to the subcarrier in-between the two DMRS subcarriers
                    coef_grid[*temp_dmrs_symbols][sc + 2] = coef_grid[*temp_dmrs_symbols][sc + 1];

                    /// Interpolate the subcarrier below the first DMRS subcarrier/in-between the two OCC
                    coef_grid[*temp_dmrs_symbols][sc].real((real(coef_grid[*temp_dmrs_symbols][sc + 1]) + real(coef_grid[*temp_dmrs_symbols][sc - 1])) * 0.5);
                    coef_grid[*temp_dmrs_symbols][sc].imag((imag(coef_grid[*temp_dmrs_symbols][sc + 1]) + imag(coef_grid[*temp_dmrs_symbols][sc - 1])) * 0.5);

                    /// Interpolate norm and renormalize
                    coef_grid[*temp_dmrs_symbols][sc] *=
                            ((abs(coef_grid[*temp_dmrs_symbols][sc + 1]) + abs(coef_grid[*temp_dmrs_symbols][sc - 1])) * 0.5) /
                            abs(coef_grid[*temp_dmrs_symbols][sc]);
                }
            }
        }
        //t2 = std::chrono::steady_clock::now();
        //cout << "Duration of interpolation in frequency domain : " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;

        //t1 = std::chrono::steady_clock::now();
        int last_dmrs_symbol = *(dmrs_symbols_ + dmrs_symbols_per_grid_ - 1);
        temp_dmrs_symbols = dmrs_symbols_;

        /// interpolate in time domain
        /// Only one DMRS in the PDSCH
        if (dmrs_symbols_per_grid_ == 1) {
            int dmrs_symbol = dmrs_symbols_[0];
            /// Apply the same coefficients in time domain on each subcarrier
            for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {
                /// No need to apply values on DRMS symbol as we have interpolated in frequency domain
                if (symbol == *(dmrs_symbols_) - pdsch_start_symbol_) {
                    continue;
                }
                for (int j = 0; j < fft_size_; j++) {
                    coef_grid[symbol][j] = coef_grid[dmrs_symbol][j];
                }
            }
        } else {
            float step = 0;
            int lower_dmrs_symbol = dmrs_symbols_[0];
            int upper_dmrs_symbol = dmrs_symbols_[1];
            temp_dmrs_symbols = dmrs_symbols_;
            int step_symbols = upper_dmrs_symbol - lower_dmrs_symbol;

            if (cdm_group_number == 0) {
                for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {

                    /// If current symbol is a DMRS symbol, do not interpolate on this symbol
                    if ((symbol == (*(temp_dmrs_symbols) - pdsch_start_symbol_)) or
                        (symbol == (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_))) {
                        continue;
                    }

                    /// If current symbol is greater than the upper DMRS symbol,
                    /// update lower and upper DMRS coefs
                    if ((symbol > (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_)) and (symbol < last_dmrs_symbol)) {
                        temp_dmrs_symbols++;
                        lower_dmrs_symbol = *(temp_dmrs_symbols);
                        upper_dmrs_symbol = *(temp_dmrs_symbols + 1);
                        step_symbols = upper_dmrs_symbol - lower_dmrs_symbol;
                    }

                    step = (symbol - *(temp_dmrs_symbols) + pdsch_start_symbol_) / step_symbols;

                    for (int sc = 0; sc < fft_size_ - 4; sc+=4) {

                        /// Interpolate on the first subcarrier
                        coef_grid[symbol][sc].real(real(coef_grid[lower_dmrs_symbol][sc]) +
                                               (real(coef_grid[upper_dmrs_symbol][sc]) - real(coef_grid[lower_dmrs_symbol][sc])) *
                                               step);
                        coef_grid[symbol][sc].imag(imag(coef_grid[lower_dmrs_symbol][sc]) +
                                                   (imag(coef_grid[upper_dmrs_symbol][sc]) - imag(coef_grid[lower_dmrs_symbol][sc])) *
                                                   step);

                        /// Interpolate norm and renormalize
                        coef_grid[symbol][sc] *= ((abs(coef_grid[upper_dmrs_symbol][sc]) - abs(coef_grid[lower_dmrs_symbol][sc])) * step +
                                                   abs(coef_grid[lower_dmrs_symbol][sc])) / abs(coef_grid[symbol][sc]);

                        /// Assign the same value to the 2 next subcarriers
                        coef_grid[symbol][sc + 1] = coef_grid[symbol][sc];
                        coef_grid[symbol][sc + 2] = coef_grid[symbol][sc];

                        /// Interpolate on subcarrier number 3
                        /// Interpolate on the first subcarrier
                        coef_grid[symbol][sc + 3].real(coef_grid[lower_dmrs_symbol][sc + 3].real() +
                                                      (coef_grid[upper_dmrs_symbol][sc + 3].real() -  coef_grid[lower_dmrs_symbol][sc + 3].real()) *
                                                       step);
                        coef_grid[symbol][sc + 3].imag(coef_grid[lower_dmrs_symbol][sc + 3].imag() +
                                                      (coef_grid[upper_dmrs_symbol][sc + 3].imag() -  coef_grid[lower_dmrs_symbol][sc + 3].imag()) *
                                                       step);

                        /// Interpolate norm and renormalize
                        coef_grid[symbol][sc + 3] *= ((abs(coef_grid[upper_dmrs_symbol][sc + 3])) - abs(coef_grid[lower_dmrs_symbol][sc + 3]) *
                                                       step + abs(coef_grid[lower_dmrs_symbol][sc + 3])) / abs(coef_grid[symbol][sc + 3]);

                        //output << *(temp_coef_grid) << endl;
                    }

                    /// Interpolate on first subcarrier of the 4 last subcarriers
                    /// Interpolate on the first subcarrier
                    coef_grid[symbol][fft_size_ - 4].real(coef_grid[lower_dmrs_symbol][fft_size_ - 4].real() +
                                                         (coef_grid[upper_dmrs_symbol][fft_size_ - 4].real() - coef_grid[lower_dmrs_symbol][fft_size_ - 4].real()) *
                                                          step);
                    coef_grid[symbol][fft_size_ - 4].imag(coef_grid[lower_dmrs_symbol][fft_size_ - 4].imag() +
                                                         (coef_grid[upper_dmrs_symbol][fft_size_ - 4].imag() - coef_grid[lower_dmrs_symbol][fft_size_ - 4].imag()) *
                                                          step);

                    /// Interpolate norm and renormalize
                    coef_grid[symbol][fft_size_ - 4] *= ((abs(coef_grid[upper_dmrs_symbol][fft_size_ - 4]) - abs(coef_grid[lower_dmrs_symbol][fft_size_ - 4])) * step +
                                          abs(coef_grid[lower_dmrs_symbol][fft_size_ - 4])) / abs(coef_grid[symbol][fft_size_ - 4]);

                    /// Assign the value to the last 4 subcarriers
                    coef_grid[symbol][fft_size_ - 3] = coef_grid[symbol][fft_size_ - 4];
                    coef_grid[symbol][fft_size_ - 2] = coef_grid[symbol][fft_size_ - 4];
                    coef_grid[symbol][fft_size_ - 1] = coef_grid[symbol][fft_size_ - 4];
                }
            }
                /// CDM group 1
            else {

                for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {
                    //output << "Symbol number : " << symbol << endl;

                    /// If current symbol is a DMRS symbol, do not interpolate on this symbol
                    if ((symbol == (*(temp_dmrs_symbols) - pdsch_start_symbol_)) or
                        (symbol == (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_))) {
                        continue;
                    }

                    /// If current symbol is greater than the upper DMRS symbol,
                    /// update lower and upper DMRS coefs
                    if ((symbol > (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_)) and (symbol < last_dmrs_symbol)) {
                        temp_dmrs_symbols++;
                        lower_dmrs_symbol = *(temp_dmrs_symbols);
                        upper_dmrs_symbol = *(temp_dmrs_symbols + 1);
                        step_symbols = upper_dmrs_symbol - lower_dmrs_symbol;
                    }

                    step = 1.0f * (symbol - *(temp_dmrs_symbols) + pdsch_start_symbol_) / step_symbols;

                    /// Interpolate on the first subcarrier
                    coef_grid[symbol][0].real(coef_grid[lower_dmrs_symbol][0].real() +
                                             (coef_grid[upper_dmrs_symbol][0].real() - coef_grid[lower_dmrs_symbol][0].real()) *
                                              step);
                    coef_grid[symbol][0].imag(coef_grid[lower_dmrs_symbol][0].imag() +
                                             (coef_grid[upper_dmrs_symbol][0].imag() - coef_grid[lower_dmrs_symbol][0].imag()) *
                                              step);
                    //*(temp_coef_grid) = *(lower_dmrs_coef) + (*(upper_dmrs_coef) - *(lower_dmrs_coef)) * step;

                    /// Interpolate norm and renormalize
                    coef_grid[symbol][0] *= ((abs(coef_grid[upper_dmrs_symbol][0]) - abs(coef_grid[lower_dmrs_symbol][0])) *
                                              step + abs(coef_grid[lower_dmrs_symbol][0])) / abs(coef_grid[symbol][0]);

                    /// Assign the same value to the 3 next subcarriers
                    coef_grid[symbol][1] = coef_grid[symbol][0];
                    coef_grid[symbol][2] = coef_grid[symbol][1];
                    coef_grid[symbol][3] = coef_grid[symbol][2];

                    for (int sc = 4; sc < fft_size_; sc+= 4) {

                        /// Interpolate on the first subcarrier
                        coef_grid[symbol][sc].real(coef_grid[lower_dmrs_symbol][sc].real() +
                                                  (coef_grid[upper_dmrs_symbol][sc].real() - coef_grid[lower_dmrs_symbol][sc].real()) *
                                                   step);
                        coef_grid[symbol][sc].imag(coef_grid[lower_dmrs_symbol][sc].imag() +
                                                   (coef_grid[upper_dmrs_symbol][sc].imag() - coef_grid[lower_dmrs_symbol][sc].imag()) *
                                                   step);

                        /// Interpolate norm and renormalize
                        coef_grid[symbol][sc] *= ((abs(coef_grid[upper_dmrs_symbol][sc]) - abs(coef_grid[lower_dmrs_symbol][sc])) *
                                                   step +
                                                   abs(coef_grid[lower_dmrs_symbol][sc])) / abs(coef_grid[symbol][sc]);

                        /// Interpolate on subcarrier number 3
                        coef_grid[symbol][sc + 3].real(coef_grid[lower_dmrs_symbol][sc + 3].real() +
                                                   (coef_grid[upper_dmrs_symbol][sc + 3].real() - coef_grid[lower_dmrs_symbol][sc + 3].real()) *
                                                   step);
                        coef_grid[symbol][sc + 3].imag(coef_grid[lower_dmrs_symbol][sc + 3].imag() +
                                                   (coef_grid[upper_dmrs_symbol][sc + 3].imag() - coef_grid[lower_dmrs_symbol][sc + 3].imag()) *
                                                   step);

                        /// Interpolate norm and renormalize
                        coef_grid[symbol][sc + 3] *= ((abs(coef_grid[upper_dmrs_symbol][sc + 3]) - abs(coef_grid[lower_dmrs_symbol][sc + 3])) *
                                                  step +
                                                  abs(coef_grid[lower_dmrs_symbol][sc + 3])) / abs(coef_grid[symbol][sc + 3]);

                        /// Assign the same value to the subcarriers 1 and 2
                        coef_grid[symbol][sc + 1] = coef_grid[symbol][sc + 3];
                        coef_grid[symbol][sc + 2] = coef_grid[symbol][sc + 3];

                        //output << *(temp_coef_grid) << endl;
                    }
                }
            }
        }
        //t2 = std::chrono::steady_clock::now();
        //cout << "Duration of interpolation in time domain : " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;
    } else { /// No OCC used
        /// Interpolation of DMRS from CDM group 0
        if (cdm_group_number == 0) {
            /// Fill the grid with DMRS
            for (int symbol = 0; symbol < dmrs_symbols_per_grid_; symbol++) {
                for (int sc = 0; sc < dmrs_sequence_size_; sc++) {
                    coef_grid[*(temp_dmrs_symbols)][2 * sc] = dmrs_coefs_[symbol][sc];
                }
                temp_dmrs_symbols++;
            }
            /// interpolate in frequency domain
            for (temp_dmrs_symbols = dmrs_symbols_; temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {
                /// Interpolate every odd subcarrier
                for (int sc = 0; sc < fft_size_; sc+=2) {
                    /// Interpolate real and imaginary part
                    coef_grid[*temp_dmrs_symbols][sc].real((coef_grid[*temp_dmrs_symbols][sc + 1].real() + coef_grid[*temp_dmrs_symbols][sc - 1].real()) * 0.5);
                    coef_grid[*temp_dmrs_symbols][sc].imag((coef_grid[*temp_dmrs_symbols][sc + 1].imag() + coef_grid[*temp_dmrs_symbols][sc - 1].imag()) * 0.5);

                    /// Interpolate norm and renormalize
                    coef_grid[*temp_dmrs_symbols][sc] *=
                            ((abs(coef_grid[*temp_dmrs_symbols][sc + 1]) + abs(coef_grid[*temp_dmrs_symbols][sc - 1])) * 0.5) / abs(coef_grid[*temp_dmrs_symbols][sc]);
                }
                /// Extrapolate the last value
                coef_grid[*temp_dmrs_symbols][fft_size_ - 1].real((coef_grid[*temp_dmrs_symbols][fft_size_ - 2].real() * 3 - coef_grid[*temp_dmrs_symbols][fft_size_ - 4].real()) * 0.5);
                coef_grid[*temp_dmrs_symbols][fft_size_ - 1].imag((coef_grid[*temp_dmrs_symbols][fft_size_ - 2].imag() * 3 - coef_grid[*temp_dmrs_symbols][fft_size_ - 4].imag()) * 0.5);
                /// Interpolate norm and renormalize
                coef_grid[*temp_dmrs_symbols][fft_size_ - 1] *= (abs(coef_grid[*temp_dmrs_symbols][fft_size_ - 2]) * 3 * 0.5 - abs(coef_grid[*temp_dmrs_symbols][fft_size_ - 4]) * 0.5) /
                                                 abs(coef_grid[*temp_dmrs_symbols][fft_size_ - 1]);
            }
            temp_dmrs_symbols++;

        } else { /// CDM group 1
            /// Fill the grid with DMRS
            for (int symbol = 0; symbol < dmrs_symbols_per_grid_; symbol++) {
                for (int sc = 0; sc < dmrs_sequence_size_; sc++) {
                    coef_grid[*(temp_dmrs_symbols)][2 * sc + 1] = dmrs_coefs_[symbol][sc];
                }
                temp_dmrs_symbols++;
            }

            /// interpolate in frequency domain
            for (temp_dmrs_symbols = dmrs_symbols_;
                 temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {
                /// Extrapolate first value
                coef_grid[*temp_dmrs_symbols][0].real((coef_grid[*temp_dmrs_symbols][1].real() * 3 - coef_grid[*temp_dmrs_symbols][3].real()) * 0.5);
                coef_grid[*temp_dmrs_symbols][0].imag((coef_grid[*temp_dmrs_symbols][1].imag() * 3 - coef_grid[*temp_dmrs_symbols][3].imag()) * 0.5);
                /// Interpolate norm and renormalize
                coef_grid[*temp_dmrs_symbols][0] *= (abs(coef_grid[*temp_dmrs_symbols][1]) * 3 * 0.5 - abs(coef_grid[*temp_dmrs_symbols][3]) * 0.5) / abs(coef_grid[*temp_dmrs_symbols][0]);

                for (int sc = 2; sc < fft_size_; sc+=2) {
                    coef_grid[*temp_dmrs_symbols][sc].real((coef_grid[*temp_dmrs_symbols][sc + 1].real() + coef_grid[*temp_dmrs_symbols][sc - 1].real()) * 0.5);
                    coef_grid[*temp_dmrs_symbols][sc].imag((coef_grid[*temp_dmrs_symbols][sc + 1].imag() + coef_grid[*temp_dmrs_symbols][sc - 1].imag()) * 0.5);

                    /// Interpolate norm and renormalize
                    coef_grid[*temp_dmrs_symbols][sc] *=
                            ((abs(coef_grid[*temp_dmrs_symbols][sc + 1]) + abs(coef_grid[*temp_dmrs_symbols][sc - 1])) * 0.5) /
                            abs(coef_grid[*temp_dmrs_symbols][sc]);
                }
            }
        }
        //t2 = std::chrono::steady_clock::now();
        //cout << "Duration of interpolation in frequency domain : " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;

        int last_dmrs_symbol = *(dmrs_symbols_ + dmrs_symbols_per_grid_ - 1);
        temp_dmrs_symbols = dmrs_symbols_;

        /// interpolate in time domain
        /// Only one DMRS in the PDSCH
        if (dmrs_symbols_per_grid_ == 1) {
            int dmrs_symbol = dmrs_symbols_[0];
            /// Apply the same coefficients in time domain on each subcarrier
            for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {
                /// No need to apply values on DRMS symbol as we have interpolated in frequency domain
                if (symbol == *(dmrs_symbols_) - pdsch_start_symbol_) {
                    continue;
                }
                for (int j = 0; j < fft_size_; j++) {
                    coef_grid[symbol][j] = coef_grid[dmrs_symbol][j];
                }
            }
        } else {

            float step = 0;
            int lower_dmrs_symbol = dmrs_symbols_[0];
            int upper_dmrs_symbol = dmrs_symbols_[1];
            temp_dmrs_symbols = dmrs_symbols_;
            int step_symbols = upper_dmrs_symbol - lower_dmrs_symbol;

            for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {

                /// If current symbol is a DMRS symbol, do not interpolate on this symbol
                if ((symbol == (*(temp_dmrs_symbols) - pdsch_start_symbol_)) or
                    (symbol == (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_))) {
                    continue;
                }

                /// If current symbol is greater than the upper DMRS symbol,
                /// update lower and upper DMRS coefs
                if ((symbol > (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_)) and (symbol < last_dmrs_symbol)) {
                    temp_dmrs_symbols++;
                    lower_dmrs_symbol = *(temp_dmrs_symbols);
                    upper_dmrs_symbol = *(temp_dmrs_symbols + 1);
                    step_symbols = upper_dmrs_symbol - lower_dmrs_symbol;
                }

                step = (symbol - (*temp_dmrs_symbols) + pdsch_start_symbol_) / step_symbols;

                for (int sc = 0; sc < fft_size_; sc++) {

                    coef_grid[symbol][sc].real(coef_grid[lower_dmrs_symbol][sc].real() +
                                              (coef_grid[upper_dmrs_symbol][sc].real() - coef_grid[lower_dmrs_symbol][sc].real()) *
                                               step);
                    coef_grid[symbol][sc].imag(coef_grid[lower_dmrs_symbol][sc].imag() +
                                              (coef_grid[upper_dmrs_symbol][sc].imag() - coef_grid[lower_dmrs_symbol][sc].imag()) *
                                               step);

                    /// Interpolate norm and renormalize
                    coef_grid[symbol][sc] *= ((abs(coef_grid[upper_dmrs_symbol][sc]) - abs(coef_grid[lower_dmrs_symbol][sc])) *
                                               step + abs(coef_grid[lower_dmrs_symbol][sc])) / abs(coef_grid[symbol][sc]);
                }

                /// Reinitialize the pointers to the two DMRS symbols used for interpolation
            }
        }
    }
}

/*********************************************************************************************************/


void interpolate_coefs_test(complex<float> * coef_grid, /// TODO: mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
                            complex<float> * dmrs_coefs_, // TODO : mettre l'offset pour fournir la bonne grille à interpoler en dehors de la fonction
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
                           int nb_rx) {

    //std::chrono::steady_clock::time_point t1{}, t2{};

    complex<float> * temp_dmrs_coefs;
    complex<float> * ptr_coef_grid;
    complex<float> * lower_dmrs_coef;
    complex<float> * upper_dmrs_coef;
    int step_symbols;

    //t1 = std::chrono::steady_clock::now();
    /**
    ofstream output("interp_tx"+to_string(tx_antenna_port_number_)+"_rx"+ to_string(rx_antenna_port_number_)+"slot"+to_string(slot_number_static)+".txt");

    output << "----------------" << endl;
    output << "Port number : " << tx_antenna_port_number_ << endl;
    output << "RX port number : " << rx_antenna_port_number_ << endl;
    output << "Slot number : " << slot_number_static << endl;
    */

    temp_dmrs_coefs = dmrs_coefs_;
    step_symbols = 0;

    /// Optimizations if 2 DMRS ports are mutliplexed by OCC in the CDM group
    if(cdm_group_size > 1) {

        /// Interpolation of DMRS from CDM group 0
        if (cdm_group_number == 0) {

            /// Fill the grid with DMRS
            for (int symbol = 0; symbol < dmrs_symbols_per_grid_; symbol++) {

                temp_dmrs_coefs = dmrs_coefs_ + symbol * dmrs_sequence_size_;
                for(complex<float> * temp_coef_grid = coef_grid + (dmrs_symbols_[symbol] - pdsch_start_symbol_) *
                                                                  fft_size_; temp_coef_grid < coef_grid + (dmrs_symbols_[symbol] - pdsch_start_symbol_) *
                                                                                                          fft_size_ + fft_size_; temp_coef_grid += 2) {
                    *(temp_coef_grid) = *(temp_dmrs_coefs);
                    temp_dmrs_coefs++;
                }
            }

            /// interpolate in frequency domain
            for (int *temp_dmrs_symbols = dmrs_symbols_;
                 temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {

                ptr_coef_grid = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                            fft_size_; /// Initialize to the first subcarrier of the current symbol

 /// Initialize to the first subcarrier of the current symbol

                for (complex<float> * temp_coef_grid =  coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_ + 1;
                                                        temp_coef_grid < coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_ + fft_size_ - 1; temp_coef_grid += 2) {

                    /// Interpolate real and imaginary part of the subcarrier located after the second DMRS
                    temp_coef_grid->real(((temp_coef_grid + 1)->real() + (temp_coef_grid - 1)->real()) * 0.5);
                    temp_coef_grid->imag(((temp_coef_grid + 1)->imag() + (temp_coef_grid - 1)->imag()) * 0.5);
                    //*(temp_coef_grid + 3) = (*(temp_coef_grid + 4) + *(temp_coef_grid + 2)) * float(0.5);

                    /// Interpolate norm and renormalize
                    *(temp_coef_grid) *=
                            ((abs(*(temp_coef_grid + 1)) + abs(*(temp_coef_grid - 1))) * 0.5) /
                            abs(*(temp_coef_grid - 1));
                }

                /// Assign the same value to the last DMRS subcarrier
                *(ptr_coef_grid + fft_size_ - 1) = *(ptr_coef_grid + fft_size_ - 2);

            }

        } else { /// CDM group 1

            /// Fill the grid with DMRS
            for (int symbol = 0; symbol < dmrs_symbols_per_grid_; symbol++) {

                ptr_coef_grid = coef_grid + (dmrs_symbols_[symbol] - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first subcarrier of the current symbol

                temp_dmrs_coefs = dmrs_coefs_ + symbol * dmrs_sequence_size_;

                for(complex<float> * temp_coef_grid = coef_grid + (dmrs_symbols_[symbol] - pdsch_start_symbol_) *
                                                                  fft_size_ + 1; temp_coef_grid < coef_grid + (dmrs_symbols_[symbol] - pdsch_start_symbol_) *
                                                                                                          fft_size_ + fft_size_; temp_coef_grid += 2) {
                    *(temp_coef_grid) = *(temp_dmrs_coefs);
                    temp_dmrs_coefs++;
                }
            }

            /// interpolate in frequency domain
            for (int *temp_dmrs_symbols = dmrs_symbols_;
                 temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {

                for (complex<float> * temp_coef_grid = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                                                   fft_size_ + 2; temp_coef_grid < coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                                                                          fft_size_ + fft_size_; temp_coef_grid += 2) {

                    /// Interpolate the subcarrier below the first DMRS subcarrier/in-between the two OCC
                    //(temp_coef_grid)->real(((temp_dmrs_coefs)->real() + (temp_dmrs_coefs - 1)->real()) * 0.5);
                    //(temp_coef_grid)->imag(((temp_dmrs_coefs)->imag() + (temp_dmrs_coefs - 1)->imag()) * 0.5);

                    (temp_coef_grid)->real(((temp_coef_grid + 1)->real() + (temp_coef_grid - 1)->real()) * 0.5);
                    (temp_coef_grid)->imag(((temp_coef_grid + 1)->imag() + (temp_coef_grid - 1)->imag()) * 0.5);

                    //*(temp_coef_grid) = (*(temp_coef_grid + 1) + *(temp_coef_grid - 1)) * float(0.5);

                    /// Interpolate norm and renormalize
                    *(temp_coef_grid) *=
                            ((abs(*(temp_coef_grid + 1)) + abs(*(temp_coef_grid - 1))) * 0.5) / abs(*(temp_coef_grid));
                }
            }
        }
        //t2 = std::chrono::steady_clock::now();
        //cout << "Duration of interpolation in frequency domain : " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;

        //t1 = std::chrono::steady_clock::now();
        int last_dmrs_symbol = *(dmrs_symbols_ + dmrs_symbols_per_grid_ - 1);
        int *temp_dmrs_symbols = dmrs_symbols_;
        ptr_coef_grid = coef_grid;

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
                    ptr_coef_grid += fft_size_;
                    continue;
                }

                for (int j = 0; j < fft_size_; j++) {
                    *(ptr_coef_grid) = *(lower_dmrs_coef);
                    ptr_coef_grid++;
                    lower_dmrs_coef++;
                }
            }

        } else {

            /// Reset pointers
            ptr_coef_grid = coef_grid;

            lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
            upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;

            step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);

            float step = 0;

            if (cdm_group_number == 0) {
                for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {

                    //output << "Symbol number : " << symbol << endl;

                    /// If current symbol is a DMRS symbol, do not interpolate on this symbol
                    if ((symbol == (*(temp_dmrs_symbols) - pdsch_start_symbol_)) or
                        (symbol == (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_))) {
                        ptr_coef_grid += fft_size_;
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

                    step = 1.0f * (symbol - *(temp_dmrs_symbols) + pdsch_start_symbol_) / step_symbols;

                    for (int sc = 0; sc < fft_size_/4 - 1; sc ++) {

                        /// Interpolate on the first subcarrier
                        (ptr_coef_grid)->real(lower_dmrs_coef->real() +
                                               (upper_dmrs_coef->real() - lower_dmrs_coef->real()) *
                                               step);
                        (ptr_coef_grid)->imag(lower_dmrs_coef->imag() +
                                               (upper_dmrs_coef->imag() - lower_dmrs_coef->imag()) *
                                               step);
                        //*(temp_coef_grid) = *(lower_dmrs_coef) + (*(upper_dmrs_coef) - *(lower_dmrs_coef)) * step;


                        /// Interpolate norm and renormalize
                        *(ptr_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) * step +
                                              abs(*(lower_dmrs_coef))) / abs(*(ptr_coef_grid));

                        /// Assign the same value to the 2 next subcarriers
                        *(ptr_coef_grid + 1) = *(ptr_coef_grid);
                        *(ptr_coef_grid + 2) = *(ptr_coef_grid);

                        /// Interpolate on subcarrier number 3
                        /// Interpolate on the first subcarrier
                        (ptr_coef_grid + 3)->real((lower_dmrs_coef + 3)->real() +
                                                   ((upper_dmrs_coef + 3)->real() - (lower_dmrs_coef + 3)->real()) *
                                                   step);
                        (ptr_coef_grid + 3)->imag((lower_dmrs_coef + 3)->imag() +
                                                   ((upper_dmrs_coef + 3)->imag() - (lower_dmrs_coef + 3)->imag()) *
                                                   step);

                        //*(temp_coef_grid + 3) = *(lower_dmrs_coef + 3) + (*(upper_dmrs_coef + 3) - *(lower_dmrs_coef + 3)) * step;

                        /// Interpolate norm and renormalize
                        *(ptr_coef_grid + 3) *= ((abs(*(upper_dmrs_coef + 3)) - abs(*(lower_dmrs_coef + 3))) *
                                                  step +
                                                  abs(*(lower_dmrs_coef + 3))) / abs(*(ptr_coef_grid + 3));

                        //output << *(temp_coef_grid) << endl;

                        ptr_coef_grid  += 4;
                        lower_dmrs_coef += 4;
                        upper_dmrs_coef += 4;
                    }

                    /// Interpolate on first subcarrier of the 4 last subcarriers
                    /// Interpolate on the first subcarrier
                    (ptr_coef_grid)->real(lower_dmrs_coef->real() +
                                           (upper_dmrs_coef->real() - lower_dmrs_coef->real()) *
                                           step);
                    (ptr_coef_grid)->imag(lower_dmrs_coef->imag() +
                                           (upper_dmrs_coef->imag() - lower_dmrs_coef->imag()) *
                                           step);

                    //*(temp_coef_grid) = *(lower_dmrs_coef) + (*(upper_dmrs_coef) - *(lower_dmrs_coef)) * step;

                    /// Interpolate norm and renormalize
                    *(ptr_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) * step +
                                          abs(*(lower_dmrs_coef))) / abs(*(ptr_coef_grid));

                    /// Assign the value to the last 4 subcarriers
                    *(ptr_coef_grid + 1) = *(ptr_coef_grid);
                    *(ptr_coef_grid + 2) = *(ptr_coef_grid);
                    *(ptr_coef_grid + 3) = *(ptr_coef_grid);

                    ptr_coef_grid += 4; /// Jump to the first subcarrier of the next sytmbol

                    /// Reinitialize the pointers to the two DMRS symbols used for interpolation
                    lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
                    upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
                }
            }
                /// CDM group 1
            else {

                for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {
                    //output << "Symbol number : " << symbol << endl;

                    /// If current symbol is a DMRS symbol, do not interpolate on this symbol
                    if ((symbol == (*(temp_dmrs_symbols) - pdsch_start_symbol_)) or
                        (symbol == (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_))) {
                        ptr_coef_grid += fft_size_;
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

                    step = 1.0f * (symbol - *(temp_dmrs_symbols) + pdsch_start_symbol_) / step_symbols;

                    /// Interpolate on the first subcarrier
                    (ptr_coef_grid)->real(lower_dmrs_coef->real() +
                                           (upper_dmrs_coef->real() - lower_dmrs_coef->real()) *
                                           step);
                    (ptr_coef_grid)->imag(lower_dmrs_coef->imag() +
                                           (upper_dmrs_coef->imag() - lower_dmrs_coef->imag()) *
                                           step);
                    //*(temp_coef_grid) = *(lower_dmrs_coef) + (*(upper_dmrs_coef) - *(lower_dmrs_coef)) * step;

                    /// Interpolate norm and renormalize
                    *(ptr_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                          step +
                                          abs(*(lower_dmrs_coef))) / abs(*(ptr_coef_grid));

                    /// Assign the same value to the 3 next subcarriers
                    *(ptr_coef_grid + 1) = *(ptr_coef_grid);
                    *(ptr_coef_grid + 2) = *(ptr_coef_grid);
                    *(ptr_coef_grid + 3) = *(ptr_coef_grid);

                    ptr_coef_grid  += 4; /// Jump to the next sc to be interpolated
                    lower_dmrs_coef += 4;
                    upper_dmrs_coef += 4;

                    for (int sc = 1; sc < fft_size_/4; sc ++) {

                        /// Interpolate on the first subcarrier
                        (ptr_coef_grid)->real(lower_dmrs_coef->real() +
                                               (upper_dmrs_coef->real() - lower_dmrs_coef->real()) *
                                               step);
                        (ptr_coef_grid)->imag(lower_dmrs_coef->imag() +
                                               (upper_dmrs_coef->imag() - lower_dmrs_coef->imag()) *
                                               step);

                        //*(temp_coef_grid) = *(lower_dmrs_coef) + (*(upper_dmrs_coef) - *(lower_dmrs_coef)) * step;

                        /// Interpolate norm and renormalize
                        *(ptr_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                              step +
                                              abs(*(lower_dmrs_coef))) / abs(*(ptr_coef_grid));

                        /// Interpolate on subcarrier number 3
                        (ptr_coef_grid + 3)->real((lower_dmrs_coef + 3)->real() +
                                                   ((upper_dmrs_coef + 3)->real() - (lower_dmrs_coef + 3)->real()) *
                                                   step);
                        (ptr_coef_grid + 3)->imag((lower_dmrs_coef + 3)->imag() +
                                                   ((upper_dmrs_coef + 3)->imag() - (lower_dmrs_coef + 3)->imag()) *
                                                   step);

                        //*(temp_coef_grid + 3) = *(lower_dmrs_coef + 3) + (*(upper_dmrs_coef + 3) - *(lower_dmrs_coef)) * step;

                        /// Interpolate norm and renormalize
                        *(ptr_coef_grid + 3) *= ((abs(*(upper_dmrs_coef + 3)) - abs(*(lower_dmrs_coef + 3))) *
                                                  step +
                                                  abs(*(lower_dmrs_coef + 3))) / abs(*(ptr_coef_grid + 3));

                        /// Assign the same value to the subcarriers 1 and 2
                        *(ptr_coef_grid + 1) = *(ptr_coef_grid + 3);
                        *(ptr_coef_grid + 2) = *(ptr_coef_grid + 3);

                        //output << *(temp_coef_grid) << endl;

                        ptr_coef_grid  += 4;
                        lower_dmrs_coef += 4;
                        upper_dmrs_coef += 4;
                    }

                    /// Reinitialize the pointers to the two DMRS symbols used for interpolation
                    lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
                    upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
                }
            }
        }

        //t2 = std::chrono::steady_clock::now();
        //cout << "Duration of interpolation in time domain : " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;
    }

        /// Interpolate all values if no OCC is used
    else {
        /// Interpolation of DMRS from CDM group 0
        if (cdm_group_number == 0) {

            /// Fill the grid with DMRS
            for (int symbol = 0; symbol < dmrs_symbols_per_grid_; symbol++) {

                ptr_coef_grid = coef_grid + (dmrs_symbols_[symbol] - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first subcarrier of the current symbol

                temp_dmrs_coefs = dmrs_coefs_ + symbol * dmrs_sequence_size_;
                for(int sc = 0; sc < dmrs_sequence_size_; sc++) {
                    ptr_coef_grid[2*sc] = temp_dmrs_coefs[sc];
                }
            }

            /// interpolate in frequency domain
            for (int *temp_dmrs_symbols = dmrs_symbols_;
                 temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {

                //output << "Symbol number : " << *(temp_dmrs_symbols) << endl;

                ptr_coef_grid = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first RE of the current symbol

                /// Add first DMRS RE to the grid
                //*(temp_coef_grid) = *(temp_dmrs_coefs);
                //output << *(temp_coef_grid) << endl;

                //temp_coef_grid++;

                temp_dmrs_coefs = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                              fft_size_; /// Initialize to the first RE containing DMRS of the current symbol

                /// Interpolate every odd subcarrier
                for (int sc = 0; sc < fft_size_; sc+=2) {

                    /// Interpolate real and imaginary part
                    ptr_coef_grid[sc].real((temp_dmrs_coefs[sc + 1].real() + temp_dmrs_coefs[sc - 1].real()) * 0.5);
                    ptr_coef_grid[sc].imag((temp_dmrs_coefs[sc + 1].imag() + temp_dmrs_coefs[sc - 1].imag()) * 0.5);

                    /// Interpolate norm and renormalize
                    ptr_coef_grid[sc] *=
                            ((abs(temp_dmrs_coefs[sc + 1]) + abs(temp_dmrs_coefs[sc - 1])) * 0.5) / abs(ptr_coef_grid[sc]);

                    //output << *(temp_coef_grid) << endl;
                    //output << *(temp_coef_grid + 1) << endl;
                }

                /// Extrapolate the last value
                ptr_coef_grid[fft_size_ - 1].real(temp_dmrs_coefs[fft_size_ - 2].real() * 3 * 0.5 - temp_dmrs_coefs[fft_size_ - 4].real() * 0.5);
                ptr_coef_grid[fft_size_ - 1].imag(temp_dmrs_coefs[fft_size_ - 2].imag() * 3 * 0.5 - temp_dmrs_coefs[fft_size_ - 4].imag() * 0.5);
                /// Interpolate norm and renormalize
                ptr_coef_grid[fft_size_ - 1] *= (abs(temp_dmrs_coefs[fft_size_ - 2]) * 3 * 0.5 - abs(temp_dmrs_coefs[fft_size_ - 4]) * 0.5) /
                                                 abs(ptr_coef_grid[fft_size_ - 1]);

                //output << *(temp_coef_grid) << endl;
            }

        } else { /// CDM group 1

            /// Fill the grid with DMRS
            for (int symbol = 0; symbol < dmrs_symbols_per_grid_; symbol++) {

                ptr_coef_grid = coef_grid + (dmrs_symbols_[symbol] - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first subcarrier of the current symbol

                temp_dmrs_coefs = dmrs_coefs_ + symbol * dmrs_sequence_size_;

                for(int sc = 0; sc < dmrs_sequence_size_; sc++) {
                    ptr_coef_grid[2*sc + 1] = temp_dmrs_coefs[sc];
                }
            }

            /// interpolate in frequency domain
            for (int *temp_dmrs_symbols = dmrs_symbols_;
                 temp_dmrs_symbols < dmrs_symbols_ + dmrs_symbols_per_grid_; temp_dmrs_symbols++) {

                //output << "Symbol number : " << *(temp_dmrs_symbols) << endl;

                ptr_coef_grid = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                             fft_size_; /// Initialize to the first RE to be interpolated (on sc number 0)

                temp_dmrs_coefs = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) *
                                              fft_size_; /// Initialize to the first RE containing DMRS of the current symbol

                /// Extrapolate first value
                ptr_coef_grid[0].real(temp_dmrs_coefs[1].real() * 3 * 0.5 - temp_dmrs_coefs[3].real() * 0.5);
                ptr_coef_grid[0].imag(temp_dmrs_coefs[1].imag() * 3 * 0.5 - temp_dmrs_coefs[3].imag() * 0.5);
                /// Interpolate norm and renormalize
                ptr_coef_grid[0] *= (abs(temp_dmrs_coefs[1]) * 3 * 0.5 - abs(temp_dmrs_coefs[3]) * 0.5) / abs(ptr_coef_grid[0]);

                //output << *(temp_coef_grid) << endl;

                /// Add first DMRS RE to the grid
                //*(temp_coef_grid + 1) = *(temp_dmrs_coefs);

                //output << *(temp_coef_grid + 1) << endl;

                //temp_coef_grid += 2; /// Jump to the next sc to be interpolated

                for (int sc = 2; sc < fft_size_; sc+=2) {

                    /// Add next DMRS RE to the grid
                    //*(temp_coef_grid + 1) = *(temp_dmrs_coefs + 1);

                    ptr_coef_grid[sc].real((temp_dmrs_coefs[sc + 1].real() + temp_dmrs_coefs[sc - 1].real()) * 0.5);
                    ptr_coef_grid[sc].imag((ptr_coef_grid[sc + 1].imag() + temp_dmrs_coefs[sc - 1].imag()) * 0.5);

                    /// Interpolate norm and renormalize
                    ptr_coef_grid[sc] *=
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
        }
        //t2 = std::chrono::steady_clock::now();
        //cout << "Duration of interpolation in frequency domain : " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;

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
                    ptr_coef_grid += fft_size_;
                    continue;
                }

                for (int j = 0; j < fft_size_; j++) {
                    *(ptr_coef_grid) = *(lower_dmrs_coef);
                    ptr_coef_grid++;
                    lower_dmrs_coef++;
                }
            }

        } else {

            /// Reset pointers
            ptr_coef_grid = coef_grid;
            temp_dmrs_symbols = dmrs_symbols_;

            lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
            upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;

            step_symbols = *(temp_dmrs_symbols + 1) - *(temp_dmrs_symbols);
            float step = 0;

            for (int symbol = 0; symbol < symbols_per_grid_; symbol++) {

                //output << "Symbol number : " << symbol << endl;

                /// If current symbol is a DMRS symbol, do not interpolate on this symbol
                if ((symbol == (*(temp_dmrs_symbols) - pdsch_start_symbol_)) or
                    (symbol == (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_))) {
                    ptr_coef_grid += fft_size_;
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

                step = 1.0f * (symbol - (*temp_dmrs_symbols) + pdsch_start_symbol_) / step_symbols;

                for (int sc = 0; sc < fft_size_; sc++) {

                    (ptr_coef_grid)->real(lower_dmrs_coef->real() +
                                           (upper_dmrs_coef->real() - lower_dmrs_coef->real()) *
                                           step);
                    (ptr_coef_grid)->imag(lower_dmrs_coef->imag() +
                                           (upper_dmrs_coef->imag() - lower_dmrs_coef->imag()) *
                                           step);

                    /// Interpolate norm and renormalize
                    *(ptr_coef_grid) *= ((abs(*(upper_dmrs_coef)) - abs(*(lower_dmrs_coef))) *
                                          step +
                                          abs(*(lower_dmrs_coef))) / abs(*(ptr_coef_grid));

                    //output << *(temp_coef_grid) << endl;

                    ptr_coef_grid++;
                    lower_dmrs_coef++;
                    upper_dmrs_coef++;
                }

                /// Reinitialize the pointers to the two DMRS symbols used for interpolation
                lower_dmrs_coef = coef_grid + (*(temp_dmrs_symbols) - pdsch_start_symbol_) * fft_size_;
                upper_dmrs_coef = coef_grid + (*(temp_dmrs_symbols + 1) - pdsch_start_symbol_) * fft_size_;
            }
        }
    }
}


/**
void get_pdsch_channel_coefficients(complex<float> * coef_grid_,
                                    complex<float> * pdsch_channel_coefficients_,
                                    int pdsch_start_,
                                    int * dmrs_symbols_,
                                    int n_rb_,
                                    int pdsch_length_,
                                    int dmrs_config_type_,
                                    int num_cdm_groups_without_data_) {

    int count_dmrs_symbols = 0;

    if(dmrs_config_type_ == 1) {

        if(num_cdm_groups_without_data_ == 2) { /// CDM group 1 and 0
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {

                if(symbol - pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    continue;
                }

                for(int sc = 0; sc < 12 * n_rb_; sc++) {
                    *(pdsch_channel_coefficients_) = coef_grid_[symbol * 12 * n_rb_ + sc];
                    pdsch_channel_coefficients_++;
                }
            }
        } else if(num_cdm_groups_without_data_ == 1) { /// CDM group 0 only
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {
                if(symbol - pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    for(int sc = 1; sc < 12 * n_rb_; sc+=2) {
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * 12 * n_rb_ + sc];
                        pdsch_channel_coefficients_++;
                    }
                    continue;
                }

                for(int sc = 0; sc < 12 * n_rb_; sc++) {
                    *(pdsch_channel_coefficients_) = coef_grid_[symbol * 12 * n_rb_ + sc];
                    pdsch_channel_coefficients_++;
                }
            }
        }

    } else if (dmrs_config_type_ == 2) {

        if (num_cdm_groups_without_data_ == 3) { /// CDM groups 0, 1, 2
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {

                if(symbol - pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    continue;
                }

                for(int sc = 0; sc < 12 * n_rb_; sc++) {
                    *(pdsch_channel_coefficients_) = coef_grid_[symbol * 12 * n_rb_ + sc];
                    pdsch_channel_coefficients_++;
                }
            }
        } else if (num_cdm_groups_without_data_ == 2) { /// CDM gorups 0, 1
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {

                if(symbol - pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    for(int sc = 0; sc < 12 * n_rb_; sc+=12) {
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * 12 * n_rb_ + 4];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * 12 * n_rb_ + 5];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * 12 * n_rb_ + 10];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * 12 * n_rb_ + 11];
                        pdsch_channel_coefficients_++;
                    }
                    continue;
                }

                for(int sc = 0; sc < 12 * n_rb_; sc++) {
                    *(pdsch_channel_coefficients_) = coef_grid_[symbol * 12 * n_rb_ + sc];
                    pdsch_channel_coefficients_++;
                }
            }
        } else if (num_cdm_groups_without_data_ == 1) { /// CDM group 0
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {

                if(symbol - pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    for(int sc = 0; sc < 12 * n_rb_; sc+=12) {
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * 12 * n_rb_ + 4];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * 12 * n_rb_ + 5];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * 12 * n_rb_ + 10];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * 12 * n_rb_ + 11];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * 12 * n_rb_ + 2];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * 12 * n_rb_ + 3];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * 12 * n_rb_ + 8];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * 12 * n_rb_ + 9];
                        pdsch_channel_coefficients_++;
                    }
                    continue;
                }

                for(int sc = 0; sc < 12 * n_rb_; sc++) {
                    *(pdsch_channel_coefficients_) = coef_grid_[symbol * 12 * n_rb_ + sc];
                    pdsch_channel_coefficients_++;
                }
            }
        }
    }
} */

/** Added extraction of squared norms of channel coefficients on PDSCH REs only.
 *
 */
void get_pdsch_squared_norms(float * squared_norms_,
                            float * pdsch_squared_norms_,
                            int pdsch_start_,
                            int * dmrs_symbols_,
                            int fft_size_,
                            int pdsch_length_,
                            int dmrs_config_type_,
                            int num_cdm_groups_without_data_) {

    int count_dmrs_symbols = 0;

    if(dmrs_config_type_ == 1) {

        if(num_cdm_groups_without_data_ == 2) { /// CDM group 1 and 0
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {

                if(symbol + pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    continue;
                }

                for(int sc = 0; sc < fft_size_; sc++) {
                    *(pdsch_squared_norms_) = squared_norms_[symbol * fft_size_ + sc];
                    pdsch_squared_norms_++;
                }
            }
        } else if(num_cdm_groups_without_data_ == 1) { /// CDM group 0 only
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {
                if(symbol + pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    for(int sc = 1; sc < fft_size_; sc+= 2) {
                        *(pdsch_squared_norms_) = squared_norms_[symbol * fft_size_ + sc];
                        pdsch_squared_norms_++;
                    }
                    continue;
                }

                for(int sc = 0; sc < fft_size_; sc++) {
                    *(pdsch_squared_norms_) = squared_norms_[symbol * fft_size_ + sc];
                    pdsch_squared_norms_++;
                }
            }
        }

    } else if (dmrs_config_type_ == 2) {

        if (num_cdm_groups_without_data_ == 3) { /// CDM groups 0, 1, 2
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {

                if(symbol + pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    continue;
                }

                for(int sc = 0; sc < fft_size_; sc++) {
                    *(pdsch_squared_norms_) = squared_norms_[symbol * fft_size_ + sc];
                    pdsch_squared_norms_++;
                }
            }
        } else if (num_cdm_groups_without_data_ == 2) { /// CDM gorups 0, 1
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {

                if(symbol + pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    for(int sc = 0; sc < fft_size_; sc+= 12) {
                        *(pdsch_squared_norms_) = squared_norms_[symbol * fft_size_ + 4];
                        pdsch_squared_norms_++;
                        *(pdsch_squared_norms_) = squared_norms_[symbol * fft_size_ + 5];
                        pdsch_squared_norms_++;
                        *(pdsch_squared_norms_) = squared_norms_[symbol * fft_size_ + 10];
                        pdsch_squared_norms_++;
                        *(pdsch_squared_norms_) = squared_norms_[symbol * fft_size_ + 11];
                        pdsch_squared_norms_++;
                    }
                    continue;
                }

                for(int sc = 0; sc < fft_size_; sc++) {
                    *(pdsch_squared_norms_) = squared_norms_[symbol * fft_size_ + sc];
                    pdsch_squared_norms_++;
                }
            }
        } else if (num_cdm_groups_without_data_ == 1) { /// CDM group 0
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {

                if(symbol + pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    for(int sc = 0; sc < fft_size_; sc+=12) {
                        *(pdsch_squared_norms_) = squared_norms_[symbol * fft_size_ + 4];
                        pdsch_squared_norms_++;
                        *(pdsch_squared_norms_) = squared_norms_[symbol * fft_size_ + 5];
                        pdsch_squared_norms_++;
                        *(pdsch_squared_norms_) = squared_norms_[symbol * fft_size_ + 10];
                        pdsch_squared_norms_++;
                        *(pdsch_squared_norms_) = squared_norms_[symbol * fft_size_ + 11];
                        pdsch_squared_norms_++;
                        *(pdsch_squared_norms_) = squared_norms_[symbol * fft_size_ + 2];
                        pdsch_squared_norms_++;
                        *(pdsch_squared_norms_) = squared_norms_[symbol * fft_size_ + 3];
                        pdsch_squared_norms_++;
                        *(pdsch_squared_norms_) = squared_norms_[symbol * fft_size_ + 8];
                        pdsch_squared_norms_++;
                        *(pdsch_squared_norms_) = squared_norms_[symbol * fft_size_ + 9];
                        pdsch_squared_norms_++;
                    }
                    continue;
                }

                for(int sc = 0; sc < fft_size_; sc++) {
                    *(pdsch_squared_norms_) = squared_norms_[symbol * fft_size_ + sc];
                    pdsch_squared_norms_++;
                }
            }
        }
    }
}

void get_pdsch_channel_coefficients(complex<float> * coef_grid_,
                                    complex<float> pdsch_channel_coefficients_[][MAX_TX_PORTS][MAX_RX_PORTS],
                                    int tx_port_index,
                                    int rx_port_index,
                                    int pdsch_start_,
                                    int * dmrs_symbols_,
                                    int fft_size_,
                                    int pdsch_length_,
                                    int dmrs_config_type_,
                                    int num_cdm_groups_without_data_) {

    int count_dmrs_symbols = 0;

    if(dmrs_config_type_ == 1) {

        if(num_cdm_groups_without_data_ == 2) { /// CDM group 1 and 0
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {

                if(symbol + pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    continue;
                }

                for(int sc = 0; sc < fft_size_; sc++) {
                    *(pdsch_channel_coefficients_)[tx_port_index][rx_port_index] = coef_grid_[symbol * fft_size_ + sc];
                    pdsch_channel_coefficients_++;
                }
            }
        } else if(num_cdm_groups_without_data_ == 1) { /// CDM group 0 only
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {
                if(symbol + pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    for(int sc = 1; sc < fft_size_; sc+= 2) {
                        *(pdsch_channel_coefficients_)[tx_port_index][rx_port_index] = coef_grid_[symbol * fft_size_ + sc];
                        pdsch_channel_coefficients_++;
                    }
                    continue;
                }

                for(int sc = 0; sc < fft_size_; sc++) {
                    *(pdsch_channel_coefficients_)[tx_port_index][rx_port_index] = coef_grid_[symbol * fft_size_ + sc];
                    pdsch_channel_coefficients_++;
                }
            }
        }

    } else if (dmrs_config_type_ == 2) {

        if (num_cdm_groups_without_data_ == 3) { /// CDM groups 0, 1, 2
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {

                if(symbol + pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    continue;
                }

                for(int sc = 0; sc < fft_size_; sc++) {
                    *(pdsch_channel_coefficients_)[tx_port_index][rx_port_index] = coef_grid_[symbol * fft_size_ + sc];
                    pdsch_channel_coefficients_++;
                }
            }
        } else if (num_cdm_groups_without_data_ == 2) { /// CDM gorups 0, 1
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {

                if(symbol + pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    for(int sc = 0; sc < fft_size_; sc+= 12) {
                        *(pdsch_channel_coefficients_)[tx_port_index][rx_port_index] = coef_grid_[symbol * fft_size_ + 4];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_)[tx_port_index][rx_port_index] = coef_grid_[symbol * fft_size_ + 5];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_)[tx_port_index][rx_port_index] = coef_grid_[symbol * fft_size_ + 10];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_)[tx_port_index][rx_port_index] = coef_grid_[symbol * fft_size_ + 11];
                        pdsch_channel_coefficients_++;
                    }
                    continue;
                }

                for(int sc = 0; sc < fft_size_; sc++) {
                    *(pdsch_channel_coefficients_)[rx_port_index][tx_port_index] = coef_grid_[symbol * fft_size_ + sc];
                    pdsch_channel_coefficients_++;
                }
            }
        } else if (num_cdm_groups_without_data_ == 1) { /// CDM group 0
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {

                if(symbol + pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    for(int sc = 0; sc < fft_size_; sc+=12) {
                        *(pdsch_channel_coefficients_)[tx_port_index][rx_port_index] = coef_grid_[symbol * fft_size_ + 4];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_)[tx_port_index][rx_port_index] = coef_grid_[symbol * fft_size_ + 5];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_)[tx_port_index][rx_port_index] = coef_grid_[symbol * fft_size_ + 10];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_)[tx_port_index][rx_port_index] = coef_grid_[symbol * fft_size_ + 11];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_)[tx_port_index][rx_port_index] = coef_grid_[symbol * fft_size_ + 2];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_)[tx_port_index][rx_port_index] = coef_grid_[symbol * fft_size_ + 3];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_)[tx_port_index][rx_port_index] = coef_grid_[symbol * fft_size_ + 8];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_)[tx_port_index][rx_port_index] = coef_grid_[symbol * fft_size_ + 9];
                        pdsch_channel_coefficients_++;
                    }
                    continue;
                }

                for(int sc = 0; sc < fft_size_; sc++) {
                    *(pdsch_channel_coefficients_)[tx_port_index][rx_port_index] = coef_grid_[symbol * fft_size_ + sc];
                    pdsch_channel_coefficients_++;
                }
            }
        }
    }
}

void get_pdsch_channel_coefficients(complex<float> * coef_grid_,
                                    complex<float> * pdsch_channel_coefficients_,
                                    int pdsch_start_,
                                    int * dmrs_symbols_,
                                    int fft_size_,
                                    int pdsch_length_,
                                    int dmrs_config_type_,
                                    int num_cdm_groups_without_data_) {

    int count_dmrs_symbols = 0;

    if(dmrs_config_type_ == 1) {
        if(num_cdm_groups_without_data_ == 2) { /// CDM group 1 and 0
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {

                if(symbol + pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    continue;
                }

#if defined(__AVX2__) and defined(AVX2_PROCESSING)
                for(int sc = 0; sc < fft_size_; sc+=4) {
                    _mm256_storeu_ps((float *) &pdsch_channel_coefficients_[0], _mm256_loadu_ps((float *) &coef_grid_[symbol * fft_size_ + sc]));
                    pdsch_channel_coefficients_ += 4;
                }
#else
                memcpy(pdsch_channel_coefficients_, &coef_grid_[symbol * fft_size_], fft_size_ * sizeof(complex<float>));
                pdsch_channel_coefficients_ += fft_size_;
#endif
            }
        } else if(num_cdm_groups_without_data_ == 1) { /// CDM group 0 only
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {
                if(symbol + pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    for(int sc = 1; sc < fft_size_; sc+= 2) {
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * fft_size_ + sc];
                        pdsch_channel_coefficients_++;
                    }
                    continue;
                }

#if defined(__AVX2__) and defined(AVX2_PROCESSING)
                for(int sc = 0; sc < fft_size_; sc+=4) {
                    _mm256_storeu_ps((float *) &pdsch_channel_coefficients_[0], _mm256_loadu_ps((float *) &coef_grid_[symbol * fft_size_ + sc]));
                    pdsch_channel_coefficients_ += 4;
                }
#else
                memcpy(pdsch_channel_coefficients_, &coef_grid_[symbol * fft_size_], fft_size_ * sizeof(complex<float>));
                pdsch_channel_coefficients_ += fft_size_;
#endif
            }
        }

    } else if (dmrs_config_type_ == 2) {

        if (num_cdm_groups_without_data_ == 3) { /// CDM groups 0, 1, 2
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {

                if(symbol + pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    continue;
                }

                for(int sc = 0; sc < fft_size_; sc++) {
                    *(pdsch_channel_coefficients_) = coef_grid_[symbol * fft_size_ + sc];
                    pdsch_channel_coefficients_++;
                }
            }
        } else if (num_cdm_groups_without_data_ == 2) { /// CDM gorups 0, 1
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {

                if(symbol + pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    for(int sc = 0; sc < fft_size_; sc+= 12) {
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * fft_size_ + 4];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * fft_size_ + 5];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * fft_size_ + 10];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * fft_size_ + 11];
                        pdsch_channel_coefficients_++;
                    }
                    continue;
                }

                for(int sc = 0; sc < fft_size_; sc++) {
                    *(pdsch_channel_coefficients_) = coef_grid_[symbol * fft_size_ + sc];
                    pdsch_channel_coefficients_++;
                }
            }
        } else if (num_cdm_groups_without_data_ == 1) { /// CDM group 0
            for(int symbol = 0; symbol < pdsch_length_; symbol++) {

                if(symbol + pdsch_start_ == dmrs_symbols_[count_dmrs_symbols]) {
                    count_dmrs_symbols++;
                    for(int sc = 0; sc < fft_size_; sc+=12) {
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * fft_size_ + 4];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * fft_size_ + 5];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * fft_size_ + 10];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * fft_size_ + 11];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * fft_size_ + 2];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * fft_size_ + 3];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * fft_size_ + 8];
                        pdsch_channel_coefficients_++;
                        *(pdsch_channel_coefficients_) = coef_grid_[symbol * fft_size_ + 9];
                        pdsch_channel_coefficients_++;
                    }
                    continue;
                }

                for(int sc = 0; sc < fft_size_; sc++) {
                    *(pdsch_channel_coefficients_) = coef_grid_[symbol * fft_size_ + sc];
                    pdsch_channel_coefficients_++;
                }
            }
        }
    }
}

/** Detects the sent symbols according to the equalized symbol.
 *  For each equalized symbol :
 *      - computes the distance between the sent symbol and each symbol of the consellation
 *      - chooses the closest constellation symbol as the sent symbol.
 *
 * @param[in] estimatedGrid      : grid of equalized symbols.
 * @param[out] decodedGrid       : grid containing the detected symbols.
 * @param[in] constellation_type : constellation used either "qpsk" or "bpsk"
 */
void ml_detector_qpsk(const vector<complex<float>> &received_symbols,
                       vector<int> &detected_symbols) {

    float real_part{}, imaginary_part{};

    /// Check in which Region of decision the symbol is and assign the corresponding symbol index
    /// (check the integer corresponding to each qpsk symbol in variables.cpp)
    for(int i = 0; i < received_symbols.size(); i++) {

        real_part = received_symbols[i].real();
        imaginary_part = received_symbols[i].imag();

        if (real_part < 0) {
            if(imaginary_part < 0) {
                detected_symbols[i] = 2;
            } else {
                detected_symbols[i] = 1;
            }
        } else {
            if(imaginary_part < 0) {
                detected_symbols[i] = 3;
            } else { /// Real and Imag. part positive
                detected_symbols[i] = 0;
            }
        }
    }
}

void ml_detector_bpsk(const vector<complex<float>> &received_symbols,
                      vector<int> &detected_symbols) {

    float real_part{};

    for(int i = 0; i < received_symbols.size(); i++) {
        real_part = received_symbols[i].real();

        if(real_part < 0) {
            detected_symbols[i] = 1;
        } else {
            detected_symbols[i] = 0;
        }
    }

}

/** Detects the sent symbols according to the equalized symbol.
 *  For each equalized symbol :
 *      - computes the distance between the sent symbol and each symbol of the consellation
 *      - chooses the closest constellation symbol as the sent symbol.
 *
 * @param[in] estimatedGrid      : grid of equalized symbols.
 * @param[out] decodedGrid       : grid containing the detected symbols.
 * @param[in] constellation_type : constellation used either "qpsk" or "bpsk"
 */
void ml_detector_qpsk(complex<float> * received_symbols,
                      int * detected_symbols,
                      const int &buffer_size) {

    float real_part{}, imaginary_part{};

    /// Check in which Region of decision the symbol is and assign the corresponding symbol index
    /// (check the integer corresponding to each qpsk symbol in variables.cpp)
    for(int i = 0; i < buffer_size; i++) {

        real_part = received_symbols[i].real();
        imaginary_part = received_symbols[i].imag();

        if (real_part < 0) {
            if(imaginary_part < 0) {
                detected_symbols[i] = 2;
            } else {
                detected_symbols[i] = 1;
            }
        } else {
            if(imaginary_part < 0) {
                detected_symbols[i] = 3;
            } else { /// Real and Imag. part positive
                detected_symbols[i] = 0;
            }
        }
    }
}

void ml_detector_bpsk(complex<float> * received_symbols,
                      int * detected_symbols,
                      const int &buffer_size) {

    float real_part{};

    for(int i = 0; i < buffer_size; i++) {
        real_part = received_symbols[i].real();

        if(real_part < 0) {
            detected_symbols[i] = 1;
        } else {
            detected_symbols[i] = 0;
        }
    }

}

void ml_detector_qpsk(const complex<float> &equalized_symbol_,
                     int &detected_symbol) {

    /// Check in which Region of decision the symbol is and assign the corresponding symbol index
    /// (check the integer corresponding to each qpsk symbol in variables.cpp)
    if (equalized_symbol_.real() < 0) {
        if(equalized_symbol_.imag() < 0) {
            detected_symbol = 2;
        } else {
            detected_symbol = 1;
        }
    } else {
        if(equalized_symbol_.imag() < 0) {
            detected_symbol = 3;
        } else { /// Real and Imag. part positive
            detected_symbol = 0;
        }
    }
}


void ml_detector_bpsk(const complex<float> &equalized_symbol_,
                      int &detected_symbol) {

    float real_part{};

    real_part = equalized_symbol_.real();

    if(real_part < 0) {
        detected_symbol = 1;
    } else {
        detected_symbol = 0;
    }
}


/** Computes the symbol error rate between the decoded grid given in argument
     * and the stored grid internal to this object instance.
     *
     * @param[in] decodedGrid : the decoded Grid to be compared
     * @return Returns the symbol error rate
     */
double symbol_error_rate(vector<int> &detected_symbol_indexes_, vector<int> &sending_buffer_symbol_indexes_) {

    double errorRate = 0;

    for (int i = 0; i < detected_symbol_indexes_.size(); i++) {
        errorRate += detected_symbol_indexes_[i] != sending_buffer_symbol_indexes_[i] ? 1 : 0;
    }

    errorRate = errorRate/(detected_symbol_indexes_.size());

    return errorRate;
}

double symbol_error_rate(int * detected_symbol_indexes_, int * sending_buffer_symbol_indexes_, int buffer_size) {

    double errorRate = 0;

    for (int i = 0; i < buffer_size; i++) {
        errorRate += detected_symbol_indexes_[i] != sending_buffer_symbol_indexes_[i] ? 1 : 0;
    }

    errorRate = errorRate/buffer_size;

    return errorRate;
}