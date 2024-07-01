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

#include "transmit_diversity.h"

using namespace std;

/// TODO : suppress the cases depending on number of TX/RX. Initialize vectors
/// when initializing phy parameters, and decode using for loops, looping on nb of TX and RX ports.
void mimo_transmit_diversity_decoder(const vector<vector<complex<float >>> * received_grids_,
                                     const complex<float> * channel_coefficients_,
                                     const int &slot_number_,
                                     const int * pdsch_positions_,
                                     const int &num_re_pdsch_,
                                     complex<float> * equalized_symbols_,
                                     int &nb_tx_dmrs_ports_,
                                     int &nb_rx_ports_,
                                     const int &pdsch_start_symbol_,
                                     const int &pdsch_length_,
                                     const int &fft_size_) {

    int symbol = 0;
    int sc = 0;

    complex<float> r_coefs[nb_rx_ports_][nb_tx_dmrs_ports_];

    /// Case with 2TX
    if (nb_tx_dmrs_ports_ == 2) {

        complex<float> s0, s1;
        float norm = 0;

        for (int i = 0; i < num_re_pdsch_; i++) {

            /// RE-initialize s0, s1 and the norm
            s0 = 0, s1 = 0, norm = 0;

            symbol  = pdsch_positions_[2*i];
            sc      = pdsch_positions_[2*i + 1];

            for (int sc_no = 0; sc_no < nb_tx_dmrs_ports_; sc_no++) {
                for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                    r_coefs[rx_no][sc_no] = received_grids_[rx_no][symbol + slot_number_ * 14][pdsch_positions_[2*(i + sc_no) + 1]];
                }
            }

            /// Normalization factor to equalize the symbols
            symbol = symbol - pdsch_start_symbol_;
            for (int tx_no = 0; tx_no < nb_tx_dmrs_ports_; tx_no++) {
                for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                    norm += pow(abs(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + tx_no * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]),2);
                }
            }

            /// equalize s0 and s1
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                s0 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 0 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc])
                        * r_coefs[rx_no][0] + channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc] *
                                              conj(r_coefs[rx_no][1]);
                s1 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 0 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc])
                        * r_coefs[rx_no][1] - channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc] *
                                              conj(r_coefs[rx_no][0]);
            }

            s0 /= norm;
            s1 /= norm;

            equalized_symbols_[i + (slot_number_ - 1) * num_re_pdsch_] = s0;
            equalized_symbols_[i + 1 + (slot_number_ - 1) * num_re_pdsch_] = s1;

            /// Increment i to jump to current RE + 2
            i++;
        }


        /// Case with 4TX ports
    } else if (nb_tx_dmrs_ports_ == 4) {

        complex<float> s0, s1, s2, s3;
        float norm[2];
        int sc2 = 0;

        for (int i = 0; i < num_re_pdsch_; i++) {

            /// RE-initialize s0, s1, s2 and s3, and the norms
            s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            norm[0] = 0, norm[1] = 0;

            symbol  = pdsch_positions_[2*i];
            sc      = pdsch_positions_[2*i + 1];
            sc2 = pdsch_positions_[2*(i + 1) + 1];

            for (int sc_no = 0; sc_no < nb_tx_dmrs_ports_; sc_no++) {
                for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                    r_coefs[rx_no][sc_no] = received_grids_[rx_no][symbol + slot_number_ * 14][pdsch_positions_[2*(i + sc_no) + 1]];
                }
            }

            /// Normalization factor to equalize the symbols s0 and s1
            symbol = symbol - pdsch_start_symbol_;
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
               norm[0] += pow(abs(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 0 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]),2) +
                          pow(abs(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 2 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]),2);
            }

            /// Normalization factor to equalize the symbols s2 and s3
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                norm[1] += pow(abs(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc2]),2) +
                           pow(abs(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 3 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc2]),2);
            }

            /// equalize s0 and s1
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                s0 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 0 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]) * r_coefs[rx_no][0] +
                        channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 2 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc] * conj(r_coefs[rx_no][1]);
                s1 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 0 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]) * r_coefs[rx_no][1] -
                        channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 2 * pdsch_length_ * fft_size_ + symbol * fft_size_ +sc] * conj(r_coefs[rx_no][0]);
            }

            /// equalize s2 and s3
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                s2 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]) * r_coefs[rx_no][2] +
                        channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 3 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc] * conj(r_coefs[rx_no][3]);
                s3 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]) * r_coefs[rx_no][3] -
                        channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 3 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc] * conj(r_coefs[rx_no][2]);
            }

            s0 /= norm[0];
            s1 /= norm[0];
            s2 /= norm[1];
            s3 /= norm[1];

            equalized_symbols_[i + (slot_number_ - 1) * num_re_pdsch_] = s0;
            equalized_symbols_[i + 1 + (slot_number_ - 1) * num_re_pdsch_] = s1;
            equalized_symbols_[i + 2 + (slot_number_ - 1) * num_re_pdsch_] = s2;
            equalized_symbols_[i + 3 + (slot_number_ - 1) * num_re_pdsch_] = s3;

            /// Jump to the subcarrier at current subcarrier + 4
            i += 3;
        }

    }
}


void mimo_transmit_diversity_decoder(const vector<complex<float>> pdsch_samples_[MAX_RX_PORTS],
                                     const complex<float> * channel_coefficients_,
                                     const int &num_re_pdsch_,
                                     complex<float> * equalized_symbols_,
                                     int &nb_tx_dmrs_ports_,
                                     int &nb_rx_ports_ ) {

    //int symbol = 0;
    //int sc = 0;

    complex<float> r_coefs[nb_rx_ports_][nb_tx_dmrs_ports_];

    /// Case with 2TX
    if (nb_tx_dmrs_ports_ == 2) {

        complex<float> s0, s1;
        float norm = 0;

        for (int i = 0; i < num_re_pdsch_; i++) {

            /// RE-initialize s0, s1 and the norm
            s0 = 0, s1 = 0, norm = 0;

            //symbol  = pdsch_positions_[2*i];
            //sc      = pdsch_positions_[2*i + 1];

            for (int sc_no = 0; sc_no < nb_tx_dmrs_ports_; sc_no++) {
                for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                    r_coefs[rx_no][sc_no] = pdsch_samples_[rx_no][i + sc_no];
                }
            }

            /// Normalization factor to equalize the symbols
            //symbol = symbol - pdsch_start_symbol_;
            for (int tx_no = 0; tx_no < 2; tx_no++) {
                for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                    //norm += pow(abs(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + tx_no * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]),2);
                    norm += pow(abs(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * num_re_pdsch_ + tx_no * num_re_pdsch_ + i]),2);
                }
            }

            /// equalize s0 and s1
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                /**
                s0 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 0 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc])
                      * r_coefs[rx_no][0] + channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc] *
                                            conj(r_coefs[rx_no][1]);
                s1 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 0 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc])
                      * r_coefs[rx_no][1] - channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc] *
                                            conj(r_coefs[rx_no][0]);
                                            */
                s0 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * num_re_pdsch_ + 0 * num_re_pdsch_ + i])
                      * r_coefs[rx_no][0] + channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * num_re_pdsch_ + 1 * num_re_pdsch_ + i] *
                                            conj(r_coefs[rx_no][1]);
                s1 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * num_re_pdsch_ + 0 * num_re_pdsch_ + i])
                      * r_coefs[rx_no][1] - channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * num_re_pdsch_ + 1 * num_re_pdsch_ + i] *
                                            conj(r_coefs[rx_no][0]);
            }

            s0 /= norm;
            s1 /= norm;

            equalized_symbols_[i] = s0;
            equalized_symbols_[i + 1] = s1;

            /// Increment i to jump to current RE + 2
            i++;
        }


        /// Case with 4TX ports
    } else if (nb_tx_dmrs_ports_ == 4) {

        complex<float> s0, s1, s2, s3;
        float norm[2];
        //int sc2 = 0;

        for (int i = 0; i < num_re_pdsch_; i++) {

            /// RE-initialize s0, s1, s2 and s3, and the norms
            s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            norm[0] = 0, norm[1] = 0;

            //symbol  = pdsch_positions_[2*i];
            //sc      = pdsch_positions_[2*i + 1];
            //sc2 = pdsch_positions_[2*(i + 1) + 1];

            for (int sc_no = 0; sc_no < 4; sc_no++) {
                for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                    r_coefs[rx_no][sc_no] = pdsch_samples_[rx_no][i + sc_no];
                }
            }

            /// Normalization factor to equalize the symbols s0 and s1
            //symbol = symbol - pdsch_start_symbol_;
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                /**
                norm[0] += pow(abs(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 0 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]),2) +
                           pow(abs(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 2 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]),2);
                           */
               norm[0] += pow(abs(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * num_re_pdsch_ + 0 * num_re_pdsch_ + i]),2) +
                          pow(abs(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * num_re_pdsch_ + 2 * num_re_pdsch_ + i]),2);
            }

            /// Normalization factor to equalize the symbols s2 and s3
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                /**
                norm[1] += pow(abs(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc2]),2) +
                           pow(abs(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 3 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc2]),2);
                           */
                norm[1] += pow(abs(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * num_re_pdsch_ + 1 * num_re_pdsch_ + i]),2) +
                           pow(abs(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * num_re_pdsch_ + 3 * num_re_pdsch_ + i]),2);
            }

            /// equalize s0 and s1
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                /**
                s0 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 0 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]) * r_coefs[rx_no][0] +
                              channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 2 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc] * conj(r_coefs[rx_no][1]);
                s1 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 0 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]) * r_coefs[rx_no][1] -
                              channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 2 * pdsch_length_ * fft_size_ + symbol * fft_size_ +sc] * conj(r_coefs[rx_no][0]);
                              */
                s0 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * num_re_pdsch_ + 0 * num_re_pdsch_ + i]) * r_coefs[rx_no][0] +
                      channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * num_re_pdsch_ + 2 * num_re_pdsch_ + i] * conj(r_coefs[rx_no][1]);
                s1 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * num_re_pdsch_ + 0 * num_re_pdsch_ + i]) * r_coefs[rx_no][1] -
                      channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * num_re_pdsch_ + 2 * num_re_pdsch_ + i] * conj(r_coefs[rx_no][0]);
            }

            /// equalize s2 and s3
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                /**
                s2 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]) * r_coefs[rx_no][2] +
                              channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 3 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc] * conj(r_coefs[rx_no][3]);
                s3 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]) * r_coefs[rx_no][3] -
                              channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 3 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc] * conj(r_coefs[rx_no][2]);
                              */
                s2 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * num_re_pdsch_ + 1 * num_re_pdsch_ + i]) * r_coefs[rx_no][2] +
                      channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * num_re_pdsch_ + 3 * num_re_pdsch_ + i] * conj(r_coefs[rx_no][3]);
                s3 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * num_re_pdsch_ + 1 * num_re_pdsch_ + i]) * r_coefs[rx_no][3] -
                      channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * num_re_pdsch_ + 3 * num_re_pdsch_ + i] * conj(r_coefs[rx_no][2]);
            }

            s0 /= norm[0];
            s1 /= norm[0];
            s2 /= norm[1];
            s3 /= norm[1];

            equalized_symbols_[i] = s0;
            equalized_symbols_[i + 1] = s1;
            equalized_symbols_[i + 2] = s2;
            equalized_symbols_[i + 3] = s3;

            /// Jump to the subcarrier at current subcarrier + 4
            i += 3;
        }
    }
}

void mimo_transmit_diversity_decoder(const vector<vector<complex<float>>> &pdsch_samples_,
                                     const vector<complex<float>> channel_coefficients_[MAX_RX_PORTS][MAX_TX_PORTS],
                                     int num_re_pdsch_,
                                     complex<float> * equalized_symbols_,
                                     int nb_tx_dmrs_ports_,
                                     int nb_rx_ports_) {

    complex<float> r_coefs[nb_rx_ports_][nb_tx_dmrs_ports_];

    /// Case with 2TX
    if (nb_tx_dmrs_ports_ == 2) {

        complex<float> s0, s1;
        float norm = 0;

        for (int i = 0; i < num_re_pdsch_; i++) {

            /// RE-initialize s0, s1 and the norm
            s0 = 0, s1 = 0, norm = 0;

            //symbol  = pdsch_positions_[2*i];
            //sc      = pdsch_positions_[2*i + 1];

            for (int sc_no = 0; sc_no < nb_tx_dmrs_ports_; sc_no++) {
                for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                    r_coefs[rx_no][sc_no] = pdsch_samples_[rx_no][i + sc_no];
                }
            }

            /// Normalization factor to equalize the symbols
            //symbol = symbol - pdsch_start_symbol_;
            for (int tx_no = 0; tx_no < 2; tx_no++) {
                for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                    norm += pow(abs(channel_coefficients_[rx_no][tx_no][i]),2);
                }
            }

            /// equalize s0 and s1
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                /**
                s0 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 0 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc])
                      * r_coefs[rx_no][0] + channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc] *
                                            conj(r_coefs[rx_no][1]);
                s1 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 0 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc])
                      * r_coefs[rx_no][1] - channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc] *
                                            conj(r_coefs[rx_no][0]);
                                            */
                s0 += conj(channel_coefficients_[rx_no][0][i])
                      * r_coefs[rx_no][0] + channel_coefficients_[rx_no][1][i] *
                                            conj(r_coefs[rx_no][1]);
                s1 += conj(channel_coefficients_[rx_no][0][i])
                      * r_coefs[rx_no][1] - channel_coefficients_[rx_no][1][i] *
                                            conj(r_coefs[rx_no][0]);
            }

            s0 /= norm;
            s1 /= norm;

            equalized_symbols_[i] = s0;
            equalized_symbols_[i + 1] = s1;

            /// Increment i to jump to current RE + 2
            i++;
        }


        /// Case with 4TX ports
    } else if (nb_tx_dmrs_ports_ == 4) {

        complex<float> s0, s1, s2, s3;
        float norm[2];
        //int sc2 = 0;

        for (int i = 0; i < num_re_pdsch_; i++) {

            /// RE-initialize s0, s1, s2 and s3, and the norms
            s0 = 0, s1 = 0, s2 = 0, s3 = 0;
            norm[0] = 0, norm[1] = 0;

            //symbol  = pdsch_positions_[2*i];
            //sc      = pdsch_positions_[2*i + 1];
            //sc2 = pdsch_positions_[2*(i + 1) + 1];

            for (int sc_no = 0; sc_no < 4; sc_no++) {
                for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                    r_coefs[rx_no][sc_no] = pdsch_samples_[rx_no][i + sc_no];
                }
            }

            /// Normalization factor to equalize the symbols s0 and s1
            //symbol = symbol - pdsch_start_symbol_;
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                /**
                norm[0] += pow(abs(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 0 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]),2) +
                           pow(abs(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 2 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]),2);
                           */
                norm[0] += pow(abs(channel_coefficients_[rx_no][0][i]),2) +
                           pow(abs(channel_coefficients_[rx_no][2][i]),2);
            }

            /// Normalization factor to equalize the symbols s2 and s3
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                /**
                norm[1] += pow(abs(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc2]),2) +
                           pow(abs(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 3 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc2]),2);
                           */
                norm[1] += pow(abs(channel_coefficients_[rx_no][1][i]),2) +
                           pow(abs(channel_coefficients_[rx_no][3][i]),2);
            }

            /// equalize s0 and s1
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                /**
                s0 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 0 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]) * r_coefs[rx_no][0] +
                              channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 2 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc] * conj(r_coefs[rx_no][1]);
                s1 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 0 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]) * r_coefs[rx_no][1] -
                              channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 2 * pdsch_length_ * fft_size_ + symbol * fft_size_ +sc] * conj(r_coefs[rx_no][0]);
                              */
                s0 += conj(channel_coefficients_[rx_no][0][i]) * r_coefs[rx_no][0] +
                      channel_coefficients_[rx_no][2][i] * conj(r_coefs[rx_no][1]);
                s1 += conj(channel_coefficients_[rx_no][0][i]) * r_coefs[rx_no][1] -
                      channel_coefficients_[rx_no][2][i] * conj(r_coefs[rx_no][0]);
            }

            /// equalize s2 and s3
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                /**
                s2 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]) * r_coefs[rx_no][2] +
                              channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 3 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc] * conj(r_coefs[rx_no][3]);
                s3 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc]) * r_coefs[rx_no][3] -
                              channel_coefficients_[rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 3 * pdsch_length_ * fft_size_ + symbol * fft_size_ + sc] * conj(r_coefs[rx_no][2]);
                              */
                s2 += conj(channel_coefficients_[rx_no][1][i]) * r_coefs[rx_no][2] +
                      channel_coefficients_[rx_no][3][i] * conj(r_coefs[rx_no][3]);
                s3 += conj(channel_coefficients_[rx_no][1][i]) * r_coefs[rx_no][3] -
                      channel_coefficients_[rx_no][3][i] * conj(r_coefs[rx_no][2]);
            }

            s0 /= norm[0];
            s1 /= norm[0];
            s2 /= norm[1];
            s3 /= norm[1];

            equalized_symbols_[i] = s0;
            equalized_symbols_[i + 1] = s1;
            equalized_symbols_[i + 2] = s2;
            equalized_symbols_[i + 3] = s3;

            /// Jump to the subcarrier at current subcarrier + 4
            i += 3;
        }
    }
}

void mimo_transmit_diversity_decoder_v2(const vector<vector<complex<float>>> * received_grids_,
                                        const vector<vector<complex<float>>> * channel_coefficients_,
                                        const int &slot_number_,
                                        const int * pdsch_positions_,
                                        const int &num_re_pdsch_,
                                        complex<float> * equalized_symbols_,
                                        int &nb_tx_dmrs_ports_,
                                        int &nb_rx_ports_,
                                        const int &pdsch_start_symbol_) {

    int symbol = 0;
    int sc = 0, next_sc = 0, dmrs_symbol = 0;

    complex<float> r_coefs[nb_rx_ports_][nb_tx_dmrs_ports_];

    /// Case with 2TX and 1RX
    if (nb_tx_dmrs_ports_ == 2) {

        complex<float> s0, s1;
        complex<float> norm_factor_s0 = 0;
        complex<float> norm_factor_s1 = 0;

        for (int i = 0; i < num_re_pdsch_; i++) {

            /// Re-initialize s0 and s1, and the norm
            s0 = 0, s1 = 0;
            norm_factor_s0 = 0, norm_factor_s1 = 0;

            symbol  = pdsch_positions_[2*i];
            sc      = pdsch_positions_[2*i + 1];
            next_sc = pdsch_positions_[2*(i + 1) + 1];

            for (int sc_no = 0; sc_no < nb_tx_dmrs_ports_; sc_no++) {
                for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                    r_coefs[rx_no][sc_no] = received_grids_[rx_no][symbol + slot_number_ * 14][pdsch_positions_[2*(i + sc_no) + 1]];
                }
            }

            /// Normalization factor to equalize the symbols
            dmrs_symbol = symbol - pdsch_start_symbol_;
            for(int tx_no = 0; tx_no < nb_tx_dmrs_ports_; tx_no++) {
                for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                    norm_factor_s0 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + tx_no][dmrs_symbol][next_sc]) *
                                              channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + tx_no][dmrs_symbol][sc];
                }
            }

            norm_factor_s1 = conj(norm_factor_s0);

            /// equalize s0 and s1
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                s0 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 0][dmrs_symbol][next_sc]) * r_coefs[rx_no][0] +
                        channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 1][dmrs_symbol][sc] * conj(r_coefs[rx_no][1]);

                s1 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 0][dmrs_symbol][sc]) * r_coefs[rx_no][1] -
                        channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 1][dmrs_symbol][next_sc] * conj(r_coefs[rx_no][0]);
            }

            s0 *= conj(norm_factor_s0) / (float) pow(abs(norm_factor_s0), 2);
            s1 *= conj(norm_factor_s1) / (float) pow(abs(norm_factor_s1), 2);

            equalized_symbols_[i + (slot_number_ - 1) * num_re_pdsch_] = s0;
            equalized_symbols_[i + 1 + (slot_number_ - 1) * num_re_pdsch_] = s1;

            /// Increment i to jump to current RE + 2
            i++;
        }


        /// Case with 4TX ports
    } else if (nb_tx_dmrs_ports_ == 4) {

        complex<float> s0, s1, s2, s3;
        complex<float> norm_s0_s2[2];  // to normalize s0 then s2
        complex<float> norm_s1_s3[2]; // to normalize s1 then s3

        for (int i = 0; i < num_re_pdsch_; i++) {

            /// Re-initialize s0, s1, s2 and s3, and the norms
            s0 = 0, s1 = 0, s2 = 0, s3 = 0;

            norm_s0_s2[0] = 0, norm_s0_s2[1] = 0;
            norm_s1_s3[0] = 0, norm_s1_s3[1] = 0;

            symbol  = pdsch_positions_[2*i];
            sc      = pdsch_positions_[2*i + 1];
            next_sc = pdsch_positions_[2*(i + 1) + 1];

            for (int sc_no = 0; sc_no < nb_tx_dmrs_ports_; sc_no++) {
                for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                    r_coefs[rx_no][sc_no] = received_grids_[rx_no][symbol + slot_number_ * 14][pdsch_positions_[2*(i + sc_no) + 1]];
                }
            }

            /// Normalization factor to equalize the symbols s0 and s1
            dmrs_symbol = symbol - pdsch_start_symbol_;
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                norm_s0_s2[0] += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 0][dmrs_symbol][next_sc]) *
                                 channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 0][dmrs_symbol][sc] +
                                 conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 2][dmrs_symbol][next_sc]) *
                                 channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 2][dmrs_symbol][sc];
            }

            norm_s1_s3[0] = conj(norm_s0_s2[0]);

            /// equalize s0 and s1
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                s0 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 0][dmrs_symbol][next_sc]) * r_coefs[rx_no][0] +
                      channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 2][dmrs_symbol][sc] * conj(r_coefs[rx_no][1]);

                s1 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 0][dmrs_symbol][sc]) * r_coefs[rx_no][1] -
                      channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 2][dmrs_symbol][next_sc] * conj(r_coefs[rx_no][0]);
            }

            /// Normalize s0 and s1
            s0 *= conj(norm_s0_s2[0]) / (float) pow(abs(norm_s0_s2[0]), 2);
            s1 *= conj(norm_s1_s3[0]) / (float) pow(abs(norm_s1_s3[0]), 2);

            /// Prepare subcarrier indexes for s2 and s3 equalization
            sc      = pdsch_positions_[2*(i + 2) + 1];
            next_sc = pdsch_positions_[2*(i + 3) + 1];

            /// Normalization factor to equalize the symbols s2 and s3
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                norm_s0_s2[1] += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 1][dmrs_symbol][next_sc]) *
                                 channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 1][dmrs_symbol][sc] +
                                 conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 3][dmrs_symbol][next_sc]) *
                                 channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 3][dmrs_symbol][sc];
            }

            norm_s1_s3[1] = conj(norm_s0_s2[1]);

            /// equalize s2 and s3
            for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                s2 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 1][dmrs_symbol][next_sc]) * r_coefs[rx_no][2] +
                      channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 3][dmrs_symbol][sc] * conj(r_coefs[rx_no][3]);

                s3 += conj(channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 1][dmrs_symbol][sc]) * r_coefs[rx_no][3] -
                      channel_coefficients_[rx_no * nb_tx_dmrs_ports_ + 3][dmrs_symbol][next_sc] * conj(r_coefs[rx_no][2]);
            }

            s2 *= conj(norm_s0_s2[1]) / (float) pow(abs(norm_s0_s2[1]), 2);
            s3 *= conj(norm_s1_s3[1]) / (float) pow(abs(norm_s1_s3[1]), 2);

            equalized_symbols_[i + (slot_number_ - 1) * num_re_pdsch_] = s0;
            equalized_symbols_[i + 1 + (slot_number_ - 1) * num_re_pdsch_] = s1;
            equalized_symbols_[i + 2 + (slot_number_ - 1) * num_re_pdsch_] = s2;
            equalized_symbols_[i + 3 + (slot_number_ - 1) * num_re_pdsch_] = s3;

            /// Jump to the subcarrier at current subcarrier + 4
            i += 3;
        }
    }
}

void mimo_transmit_diversity_decoder_v2(const vector<vector<complex<float >>> * received_grids_,
                                        const complex<float> * channel_coefficients_,
                                        const int &slot_number_,
                                        const int * pdsch_positions_,
                                        const int &num_re_pdsch_,
                                        const int &pdsch_length_,
                                        const int fft_size_,
                                        complex<float> * equalized_symbols_,
                                        int &nb_tx_dmrs_ports_,
                                        int &nb_rx_ports_,
                                        const int &pdsch_start_symbol_) {

    int symbol = 0;
    int sc = 0, next_sc = 0, dmrs_symbol = 0;

    complex<float> r_coefs[nb_rx_ports_][nb_tx_dmrs_ports_];

    /// Case with 2TX and 1RX
    if (nb_tx_dmrs_ports_ == 2) {

        complex<float> s0, s1;
        complex<float> norm_factor_s0 = 0;
        complex<float> norm_factor_s1 = 0;

        for (int i = 0; i < num_re_pdsch_; i++) {

            /// Re-initialize s0 and s1, and the norm
            s0 = 0, s1 = 0;
            norm_factor_s0 = 0, norm_factor_s1 = 0;

            symbol  = pdsch_positions_[2*i];
            sc      = pdsch_positions_[2*i + 1];
            next_sc = pdsch_positions_[2*(i + 1) + 1];

            for (int sc_no = 0; sc_no < nb_tx_dmrs_ports_; sc_no++) {
                for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                    r_coefs[rx_no][sc_no] = received_grids_[rx_no][symbol + slot_number_ * 14][*(pdsch_positions_ + 2*(i + sc_no) + 1)];
                }
            }

            /// Normalization factor to equalize the symbols
            dmrs_symbol = symbol - pdsch_start_symbol_;
            for(uint8_t tx_no = 0; tx_no < nb_tx_dmrs_ports_; tx_no++) {
                for (uint8_t rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                    norm_factor_s0 += conj( *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + tx_no * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc)) *
                                      *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + tx_no * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc);
                }
            }

            norm_factor_s1 = conj(norm_factor_s0);

            /// equalize s0 and s1
            for (uint8_t rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                s0 += conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc)) * r_coefs[rx_no][0] +
                        *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc) * conj(r_coefs[rx_no][1]);

                s1 += conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc)) * r_coefs[rx_no][1] -
                        *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc) * conj(r_coefs[rx_no][0]);
            }

            s0 *= conj(norm_factor_s0) / (float) pow(abs(norm_factor_s0), 2);
            s1 *= conj(norm_factor_s1) / (float) pow(abs(norm_factor_s1), 2);

            equalized_symbols_[i + (slot_number_ - 1) * num_re_pdsch_] = s0;
            equalized_symbols_[i + 1 + (slot_number_ - 1) * num_re_pdsch_] = s1;

            /// Increment i to jump to current RE + 2
            i++;
        }


        /// Case with 4TX ports
    } else if (nb_tx_dmrs_ports_ == 4) {

        complex<float> s0, s1, s2, s3;
        complex<float> norm_s0_s2[2];  // to normalize s0 then s2
        complex<float> norm_s1_s3[2]; // to normalize s1 then s3

        for (int i = 0; i < num_re_pdsch_; i++) {

            /// Re-initialize s0, s1, s2 and s3, and the norms
            s0 = 0, s1 = 0, s2 = 0, s3 = 0;

            norm_s0_s2[0] = 0, norm_s0_s2[1] = 0;
            norm_s1_s3[0] = 0, norm_s1_s3[1] = 0;

            symbol  = pdsch_positions_[2*i];
            sc      = pdsch_positions_[2*i + 1];
            next_sc = pdsch_positions_[2*(i + 1) + 1];

            for (int sc_no = 0; sc_no < nb_tx_dmrs_ports_; sc_no++) {
                for (uint8_t rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                    r_coefs[rx_no][sc_no] = received_grids_[rx_no][symbol + slot_number_ * 14][*(pdsch_positions_ + 2*(i + sc_no) + 1)];
                }
            }

            /// Normalization factor to equalize the symbols s0 and s1
            dmrs_symbol = symbol - pdsch_start_symbol_;
            for (uint8_t rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                norm_s0_s2[0] += conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc)) *
                                         *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc) +
                                 conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 2 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc)) *
                                         *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 2 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc);
            }

            norm_s1_s3[0] = conj(norm_s0_s2[0]);

            /// equalize s0 and s1
            for (uint8_t rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                s0 += conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc)) * r_coefs[rx_no][0] +
                        *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 2 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc) * conj(r_coefs[rx_no][1]);

                s1 += conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc)) * r_coefs[rx_no][1] -
                        *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 2 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc) * conj(r_coefs[rx_no][0]);
            }

            /// Normalize s0 and s1
            s0 *= conj(norm_s0_s2[0]) / (float) pow(abs(norm_s0_s2[0]), 2);
            s1 *= conj(norm_s1_s3[0]) / (float) pow(abs(norm_s1_s3[0]), 2);

            /// Prepare subcarrier indexes for s2 and s3 equalization
            sc      = pdsch_positions_[2*(i + 2) + 1];
            next_sc = pdsch_positions_[2*(i + 3) + 1];

            /// Normalization factor to equalize the symbols s2 and s3
            for (uint8_t rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                norm_s0_s2[1] += conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc)) *
                                         *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc) +
                                 conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 3 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc)) *
                                         *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 3 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc);
            }

            norm_s1_s3[1] = conj(norm_s0_s2[1]);

            /// equalize s2 and s3
            for (uint8_t rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                s2 += conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc)) * r_coefs[rx_no][2] +
                        *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 3 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc) * conj(r_coefs[rx_no][3]);

                s3 += conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc)) * r_coefs[rx_no][3] -
                        *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 3 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc) * conj(r_coefs[rx_no][2]);
            }

            s2 *= conj(norm_s0_s2[1]) / (float) pow(abs(norm_s0_s2[1]), 2);
            s3 *= conj(norm_s1_s3[1]) / (float) pow(abs(norm_s1_s3[1]), 2);

            equalized_symbols_[i + (slot_number_ - 1) * num_re_pdsch_] = s0;
            equalized_symbols_[i + 1 + (slot_number_ - 1) * num_re_pdsch_] = s1;
            equalized_symbols_[i + 2 + (slot_number_ - 1) * num_re_pdsch_] = s2;
            equalized_symbols_[i + 3 + (slot_number_ - 1) * num_re_pdsch_] = s3;

            /// Jump to the subcarrier at current subcarrier + 4
            i += 3;
        }
    }
}



void mimo_transmit_diversity_decoder_v2(const vector<vector<complex<float>>> &pdsch_samples_,
                                        const complex<float> * channel_coefficients_,
                                        const int &slot_number_,
                                        const int * pdsch_positions_,
                                        const int &num_re_pdsch_,
                                        const int &pdsch_length_,
                                        const int fft_size_,
                                        complex<float> * equalized_symbols_,
                                        int &nb_tx_dmrs_ports_,
                                        int &nb_rx_ports_,
                                        const int &pdsch_start_symbol_) {

    int symbol = 0;
    int sc = 0, next_sc = 0, dmrs_symbol = 0;
    int count_pdsch_samples = 0;

    complex<float> r_coefs[nb_rx_ports_][nb_tx_dmrs_ports_];

    /// Case with 2TX and 1RX
    if (nb_tx_dmrs_ports_ == 2) {

        complex<float> s0, s1;
        complex<float> norm_factor_s0 = 0;
        complex<float> norm_factor_s1 = 0;

        for (int i = 0; i < num_re_pdsch_; i++) {

            /// Re-initialize s0 and s1, and the norm
            s0 = 0, s1 = 0;
            norm_factor_s0 = 0, norm_factor_s1 = 0;

            symbol  = pdsch_positions_[2*i];
            sc      = pdsch_positions_[2*i + 1];
            next_sc = pdsch_positions_[2*(i + 1) + 1];

            for (int sc_no = 0; sc_no < nb_tx_dmrs_ports_; sc_no++) {
                for (int rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                    r_coefs[rx_no][sc_no] = pdsch_samples_[rx_no][i + sc_no]; //count_pdsch_samples];
                }
                //count_pdsch_samples++;
            }

            /// Normalization factor to equalize the symbols
            dmrs_symbol = symbol - pdsch_start_symbol_;
            for(uint8_t tx_no = 0; tx_no < nb_tx_dmrs_ports_; tx_no++) {
                for (uint8_t rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                    norm_factor_s0 += conj( *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + tx_no * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc)) *
                                      *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + tx_no * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc);
                }
            }

            norm_factor_s1 = conj(norm_factor_s0);

            /// equalize s0 and s1
            for (uint8_t rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                s0 += conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc)) * r_coefs[rx_no][0] +
                      *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc) * conj(r_coefs[rx_no][1]);

                s1 += conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc)) * r_coefs[rx_no][1] -
                      *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc) * conj(r_coefs[rx_no][0]);
            }

            s0 *= conj(norm_factor_s0) / (float) pow(abs(norm_factor_s0), 2);
            s1 *= conj(norm_factor_s1) / (float) pow(abs(norm_factor_s1), 2);

            equalized_symbols_[i + (slot_number_ - 1) * num_re_pdsch_] = s0;
            equalized_symbols_[i + 1 + (slot_number_ - 1) * num_re_pdsch_] = s1;

            /// Increment i to jump to current RE + 2
            i++;
        }


        /// Case with 4TX ports
    } else if (nb_tx_dmrs_ports_ == 4) {

        complex<float> s0, s1, s2, s3;
        complex<float> norm_s0_s2[2];  // to normalize s0 then s2
        complex<float> norm_s1_s3[2]; // to normalize s1 then s3

        for (int i = 0; i < num_re_pdsch_; i++) {

            /// Re-initialize s0, s1, s2 and s3, and the norms
            s0 = 0, s1 = 0, s2 = 0, s3 = 0;

            norm_s0_s2[0] = 0, norm_s0_s2[1] = 0;
            norm_s1_s3[0] = 0, norm_s1_s3[1] = 0;

            symbol  = pdsch_positions_[2*i];
            sc      = pdsch_positions_[2*i + 1];
            next_sc = pdsch_positions_[2*(i + 1) + 1];

            for (int sc_no = 0; sc_no < nb_tx_dmrs_ports_; sc_no++) {
                for (uint8_t rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                    r_coefs[rx_no][sc_no] = pdsch_samples_[rx_no][i + sc_no]; //count_pdsch_samples];
                }
                //count_pdsch_samples++;
            }

            /// Normalization factor to equalize the symbols s0 and s1
            dmrs_symbol = symbol - pdsch_start_symbol_;
            for (uint8_t rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                norm_s0_s2[0] += conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc)) *
                                 *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc) +
                                 conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 2 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc)) *
                                 *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 2 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc);
            }

            norm_s1_s3[0] = conj(norm_s0_s2[0]);

            /// equalize s0 and s1
            for (uint8_t rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                s0 += conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc)) * r_coefs[rx_no][0] +
                      *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 2 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc) * conj(r_coefs[rx_no][1]);

                s1 += conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc)) * r_coefs[rx_no][1] -
                      *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 2 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc) * conj(r_coefs[rx_no][0]);
            }

            /// Normalize s0 and s1
            s0 *= conj(norm_s0_s2[0]) / (float) pow(abs(norm_s0_s2[0]), 2);
            s1 *= conj(norm_s1_s3[0]) / (float) pow(abs(norm_s1_s3[0]), 2);

            /// Prepare subcarrier indexes for s2 and s3 equalization
            sc      = pdsch_positions_[2*(i + 2) + 1];
            next_sc = pdsch_positions_[2*(i + 3) + 1];

            /// Normalization factor to equalize the symbols s2 and s3
            for (uint8_t rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                norm_s0_s2[1] += conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc)) *
                                 *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc) +
                                 conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 3 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc)) *
                                 *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 3 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc);
            }

            norm_s1_s3[1] = conj(norm_s0_s2[1]);

            /// equalize s2 and s3
            for (uint8_t rx_no = 0; rx_no < nb_rx_ports_; rx_no++) {
                s2 += conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc)) * r_coefs[rx_no][2] +
                      *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 3 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc) * conj(r_coefs[rx_no][3]);

                s3 += conj(*(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 1 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + sc)) * r_coefs[rx_no][3] -
                      *(channel_coefficients_ + rx_no * nb_tx_dmrs_ports_ * pdsch_length_ * fft_size_ + 3 * pdsch_length_ * fft_size_ + dmrs_symbol * fft_size_ + next_sc) * conj(r_coefs[rx_no][2]);
            }

            s2 *= conj(norm_s0_s2[1]) / (float) pow(abs(norm_s0_s2[1]), 2);
            s3 *= conj(norm_s1_s3[1]) / (float) pow(abs(norm_s1_s3[1]), 2);

            equalized_symbols_[i + (slot_number_ - 1) * num_re_pdsch_] = s0;
            equalized_symbols_[i + 1 + (slot_number_ - 1) * num_re_pdsch_] = s1;
            equalized_symbols_[i + 2 + (slot_number_ - 1) * num_re_pdsch_] = s2;
            equalized_symbols_[i + 3 + (slot_number_ - 1) * num_re_pdsch_] = s3;

            /// Jump to the subcarrier at current subcarrier + 4
            i += 3;
        }
    }
}