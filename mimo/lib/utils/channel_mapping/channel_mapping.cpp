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

    Functions reused from free5GRAN and modified :
    - compute_dmrs_positions
    - compute_dmrs_positions_type1A
    - compute_dmrs_sequences_type1
    - compute_dmrs_sequences
    - compute_pdsch_positions
*/

#include "channel_mapping.h"

using namespace std;


void compute_cum_sum_samples(int *dmrs_symbols_,
                             int *cum_sum_samples, /// Number of PDSCH samples within each OFDM symbol
                             int num_cdm_groups_without_data,
                             int fft_size_,
                             int pdsch_length_,
                             int num_dmrs_symbols_,
                             int pdsch_start) {

    /// DMRS conf. type 1 only
    if(num_cdm_groups_without_data == 1) { /// PDSCH is located on DMRS symbols as well
        /// Initialize symbol 0
        cum_sum_samples[0] = fft_size_;
        for(int symbol = 1; symbol < pdsch_length_; symbol++) {
            if (symbol == *(dmrs_symbols_) - pdsch_start) {
                cum_sum_samples[symbol] += fft_size_/2 + cum_sum_samples[symbol - 1];
                dmrs_symbols_++;
            } else {
                cum_sum_samples[symbol] += fft_size_ + cum_sum_samples[symbol - 1];
            }
        }
    } else if (num_cdm_groups_without_data == 2) { /// No PDSCH on DMRS symbols
        for(int symbol = 0; symbol < pdsch_length_ - num_dmrs_symbols_; symbol++) {
            cum_sum_samples[symbol] += fft_size_ + cum_sum_samples[symbol - 1];
        }
    }
}

/** TODO : double symbol case to be implemented
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
                                   int * dmrs_grid_positions) {

    dmrs_sequence_size_ = fft_size_/2;

    if ((pdsch_start_ < 0 ) or (pdsch_start_ > l0_) or (pdsch_start_ > 3)) {
        //cout << "invalid PDSCH start. Setting it to 0. " << endl;
        pdsch_start_ = 0;
    } else if (pdsch_length_ - pdsch_start_ > 14) {
        //cout << "duration is too long. Setting new pdsch duration and length" << endl;
        pdsch_start_ = 0;
        pdsch_length_ = 14;
    }

    if ((l0_ < 2) or (l0_ > 3)) {
        //cout << "starting symbol l0 not supported, setting l0 to 2" << endl;
        l0_ = 2;
    }

    if ((pdsch_length_ < 0) or (pdsch_length_ > 14)) {
        //cout << "pdsch duration not supported, setting duration to 14" << endl;
        pdsch_length_ = 14;
    }

    /// Double symbol case
    if(double_symbol) {
        switch (dmrsAdditionalPositions_) {

            case 0 :
                dmrs_symbols = (int *)(malloc(2 * sizeof(int)));
                dmrs_symbols[0] = l0_;
                dmrs_symbols[1] = l0_ + 1;

                num_dmrs_symbols_per_slot_ = 2;
                break;

            default : // Case 1
                if ((pdsch_length_ < 13) and (pdsch_length_ > 3)) {
                    dmrs_symbols = (int *)(malloc(4 * sizeof(int)));
                    dmrs_symbols[0] = l0_;
                    dmrs_symbols[1] = l0_ + 1;
                    dmrs_symbols[2] = 8;
                    dmrs_symbols[4] = 9;

                    num_dmrs_symbols_per_slot_ = 4;

                } else if(pdsch_length_ > 13) {
                    dmrs_symbols = (int *)(malloc(4 * sizeof(int)));
                    dmrs_symbols[0] = l0_;
                    dmrs_symbols[1] = l0_ + 1;
                    dmrs_symbols[2] = 10;
                    dmrs_symbols[3] = 11;

                    num_dmrs_symbols_per_slot_ = 4;

                }

                break;
        }

        /// Single symbol case
    } else {

        /** from free5GRAN */
        switch(dmrsAdditionalPositions_)
        {
            case 0 :
                dmrs_symbols = (int *)(malloc(1 * sizeof(int)));
                dmrs_symbols[0] = l0_;
                num_dmrs_symbols_per_slot_ = 1;

                break;

            case 1 :
                if(pdsch_length_ < 8) {
                    //cout << "pdsch length too short for dmrs-AdditionalPosition = 1. Using only one symbol. " << endl;
                    dmrs_symbols = (int *)(malloc(1 * sizeof(int)));
                    dmrs_symbols[0] = l0_;
                    num_dmrs_symbols_per_slot_ = 1;

                } else {
                    dmrs_symbols = (int *)(malloc(2 * sizeof(int)));
                    dmrs_symbols[0] = l0_;
                    dmrs_symbols[1] = 7;
                    num_dmrs_symbols_per_slot_ = 2;
                }

                break;

            case 2 :
                if ((pdsch_length_ < 10) and (pdsch_length_ > 7)) {
                    //cout << "pdsch length too short for dmrs-AdditionalPosition = 2. Using only two symbols. " << endl;
                    dmrs_symbols = (int *)(malloc(2 * sizeof(int)));
                    dmrs_symbols[0] = l0_;
                    dmrs_symbols[1] = 7;
                    num_dmrs_symbols_per_slot_ = 2;

                } else if(pdsch_length_ < 8) {
                    //cout << "pdsch length too short for dmrs-AdditionalPosition = 2. Using only one symbol. " << endl;
                    dmrs_symbols = (int *)(malloc(1 * sizeof(int)));
                    dmrs_symbols[0] = l0_;
                    num_dmrs_symbols_per_slot_ = 1;

                } else if ((pdsch_length_ > 9) and (pdsch_length_ < 12)) {
                    dmrs_symbols = (int *)(malloc(3 * sizeof(int)));
                    dmrs_symbols[0] = l0_;
                    dmrs_symbols[1] = 6;
                    dmrs_symbols[2] = 9;
                    num_dmrs_symbols_per_slot_ = 3;

                } else {
                    dmrs_symbols = (int *)(malloc(3 * sizeof(int)));
                    dmrs_symbols[0] = l0_;
                    dmrs_symbols[1] = 7;
                    dmrs_symbols[2] = 11;
                    num_dmrs_symbols_per_slot_ = 3;

                }
                break;

            default : /// Default = 3
                if (pdsch_length_ < 8) {
                    //cout << "pdsch length too short for dmrs-AdditionalPosition = 3. Using only one symbol. " << endl;
                    dmrs_symbols = (int *)(malloc(1 * sizeof(int)));
                    dmrs_symbols[0] = l0_;
                    num_dmrs_symbols_per_slot_ = 1;
                } else if ((pdsch_length_ > 7) and (pdsch_length_ < 10)) {
                    //cout << "pdsch length too short for dmrs-AdditionalPosition = 3. Using only two symbols. " << endl;
                    dmrs_symbols = (int *)(malloc(2 * sizeof(int)));
                    dmrs_symbols[0] = l0_;
                    dmrs_symbols[1] = 7;
                    num_dmrs_symbols_per_slot_ = 2;

                } else if ((pdsch_length_ > 9) and (pdsch_length_ < 12)) {
                    //cout << "pdsch length too short for dmrs-AdditionalPosition = 3. Using only three symbols. " << endl;
                    dmrs_symbols = (int *)(malloc(3 * sizeof(int)));
                    dmrs_symbols[0] = l0_;
                    dmrs_symbols[1] = 6;
                    dmrs_symbols[2] = 9;
                    num_dmrs_symbols_per_slot_ = 3;

                } else if (pdsch_length_ > 11) {
                    dmrs_symbols = (int *)(malloc(4 * sizeof(int)));
                    dmrs_symbols[0] = l0_;
                    dmrs_symbols[1] = 5;
                    dmrs_symbols[2] = 8;
                    dmrs_symbols[3] = 11;
                    num_dmrs_symbols_per_slot_ = 4;
                }

                break;
        }
    }

    // Always initialize the DMRS positions for all the antenna ports even if a subset is used. In configuration type 1, mapping type A, at most 4 antenna ports
    // can be used simultaneously.
    dmrs_subcarriers = (int *)(malloc(4 * fft_size_ / 2 * sizeof(int)));

    int delta = 0;

    /// Initialize the positions_tx vector as if all the antennas were used.
    /// The ports are in ascending order, meaning that positions_tx[0] gives the DMRS positions for antenna port 0, positions_tx[1] for antenna port 1 etc.
    for(int antenna_port = 0; antenna_port < 4; antenna_port++) {

        delta = ts_38_211_7_4_1_1_2_1[antenna_port][1]; // Delta number from the norm. Gives the offset of the RE in frequency domain for the antenna port's CDM group.
        for(int sc = 0; sc < fft_size_/2; sc++) {
            dmrs_subcarriers[antenna_port * fft_size_/2 + sc] = 2*sc + delta;
        }
    }

    /// Initialize the dmrsPositions grid with the used antenna ports
    for(int antenna_port = 0; antenna_port < num_tx_dmrs_ports_; antenna_port++) {
        for(int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {

            for(int sc = 0; sc < dmrs_sequence_size_; sc++) {

                /// Indicate that this RE is used for DMRS transmission by this antenna_port number. Add 1 to prevent indication for port 0 to be 0
                dmrs_grid_positions[dmrs_symbols[symbol] * fft_size_ + dmrs_subcarriers[antenna_port * fft_size_/2 +  sc]] = antenna_port + 1;
            }
        }
    }

    //cout << "pdsch_start  : "  << pdsch_start_  << endl;
    //cout << "pdsch_length : "  << pdsch_length_ << endl;
    //cout << "pdsch_l0     : "  << l0_ << endl;

    //cout << "dmrs symbols :" << endl;
    //for(int i = 0; i < num_dmrs_symbols_per_slot_; i++) {
    //    cout << dmrs_symbols[i] << endl;
    //}

}



/** @brief From free5GRAN (libphy.cpp, get_pdsch_dmrs_symbols).
 *  Computes the DMRS symbols and the dmrs sequence size.
 *  **within only one slot**
 *
 *
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
                            int * &dmrs_symbols) {

    if(dmrs_config_type_ == 1) {
        dmrs_sequence_size_ = 6 * n_rb_; /// config type 1
    } else if (dmrs_config_type_ == 2) {
        dmrs_sequence_size_ = 4 * n_rb_; /// config type 2
    }

    if ((pdsch_start_ < 0 ) or (pdsch_start_ > l0_) or (pdsch_start_ > 3)) {
        //cout << "invalid PDSCH start. Setting it to 0. " << endl;
        pdsch_start_ = 0;
    } else if (pdsch_length_ - pdsch_start_ > 14) {
        //cout << "duration is too long. Setting new pdsch duration and length" << endl;
        pdsch_start_ = 0;
        pdsch_length_ = 14;
    }

    if ((pdsch_length_ < 0) or (pdsch_length_ > 14)) {
        //cout << "pdsch duration not supported, setting duration to 14" << endl;
        pdsch_length_ = 14;
    }

    if(pdsch_mapping_type_ == 'a') {
        if ((l0_ < 2) or (l0_ > 3)) {
            //cout << "starting symbol l0 not supported, setting l0 to 2" << endl;
            l0_ = 2;
        }
    } else if (pdsch_mapping_type_ == 'b') {
        if (l0_ != pdsch_start_) {
            //cout << "first DMRS symbol must be at the start of the PDSCH" << endl;
            l0_ = pdsch_start_;
        }
    }


    /// PDSCH mapping type B
    if(pdsch_mapping_type_ == 'b') {

        /// Double symbol case
        if(double_symbol) {
            if(dmrsAdditionalPositions_ == 0) {
                dmrs_symbols = (int * )(malloc(1 * sizeof(int)));
                dmrs_symbols[0] = l0_;
            } else if(dmrsAdditionalPositions_ == 1) {
                if ((pdsch_length_ > 4) and (pdsch_length_ < 8)) {
                    dmrs_symbols = (int * )(malloc(1 * sizeof(int)));
                    dmrs_symbols[0] = l0_;
                } else if ((pdsch_length_ == 8) or (pdsch_length_ == 9)) {
                    dmrs_symbols = (int *)(malloc(2 * sizeof(int)));
                    dmrs_symbols[0] = l0_;
                    dmrs_symbols[1] = 5 + pdsch_start_;
                } else if ((pdsch_length_ == 10) or (pdsch_length_ == 11)) {
                    dmrs_symbols = (int *)(malloc(2 * sizeof(int)));
                    dmrs_symbols[0] = l0_;
                    dmrs_symbols[1] = 7 + pdsch_start_;
                } else if ((pdsch_length_ == 12) or (pdsch_length_ == 13)) {
                    dmrs_symbols = (int *)(malloc(2 * sizeof(int)));
                    dmrs_symbols[0] = l0_;
                    dmrs_symbols[1] = 8 + pdsch_start_;
                }
            }

        /// Single symbol case
        } else {

            /** from free5GRAN */
            switch(dmrsAdditionalPositions_)
            {
                case 0 :
                    dmrs_symbols = (int *)(malloc(1 * sizeof(int)));
                    dmrs_symbols[0] = l0_;
                    num_dmrs_symbols_per_slot_ = 1;

                    break;

                case 1 :
                    if(pdsch_length_ < 5) {
                        //cout << "pdsch length too short for dmrs-AdditionalPosition = 1. Using only one symbol. " << endl;
                        dmrs_symbols = (int *)(malloc(1 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        num_dmrs_symbols_per_slot_ = 1;

                    } else if ((pdsch_length_ > 4) and (pdsch_length_ < 8)) {
                        dmrs_symbols = (int *)(malloc(2 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 4 + pdsch_start_;
                        num_dmrs_symbols_per_slot_ = 2;
                    } else if (pdsch_length_ == 8) {
                        dmrs_symbols = (int *)(malloc(2 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 6 + pdsch_start_;
                        num_dmrs_symbols_per_slot_ = 2;
                    } else if ((pdsch_length_ == 9) or (pdsch_length_ == 10)) {
                        dmrs_symbols = (int *)(malloc(2 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 7 + pdsch_start_;
                        num_dmrs_symbols_per_slot_ = 2;
                    } else if (pdsch_length_ == 11) {
                        dmrs_symbols = (int *)(malloc(2 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 8 + pdsch_start_;
                        num_dmrs_symbols_per_slot_ = 2;
                    } else if (pdsch_length_ > 11) {
                        dmrs_symbols = (int *)(malloc(2 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 9 + pdsch_start_;
                        num_dmrs_symbols_per_slot_ = 2;
                    }

                    break;

                case 2 :
                    if (pdsch_length_ < 5) {
                        dmrs_symbols = (int *)(malloc(1 * sizeof (int)));
                        dmrs_symbols[0] = l0_;
                        num_dmrs_symbols_per_slot_ = 1;
                    } else if ((pdsch_length_ < 8) and (pdsch_length_ > 4)) {
                        dmrs_symbols = (int *)(malloc(2 * sizeof (int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 4 + pdsch_start_;
                        num_dmrs_symbols_per_slot_ = 2;
                    } else if (pdsch_length_ == 8) {
                        dmrs_symbols = (int *) (malloc(3 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 3 + pdsch_start_;
                        dmrs_symbols[2] = 4 + pdsch_start_;
                    } else if ((pdsch_length_ == 9) or (pdsch_length_ == 10)) {
                        dmrs_symbols = (int * ) (malloc(3 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 4 + pdsch_start_;
                        dmrs_symbols[2] = 7 + pdsch_start_;
                    } else if (pdsch_length_ == 11) {
                        dmrs_symbols = (int *) (malloc(3 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 4 + pdsch_start_;
                        dmrs_symbols[2] = 8 + pdsch_start_;
                    } else if (pdsch_length_ > 11) {
                        dmrs_symbols = (int *) (malloc(3 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 5 + pdsch_start_;
                        dmrs_symbols[2] = 9 + pdsch_start_;
                    }
                    break;

                default : /// Default = 3
                    if (pdsch_length_ < 8) {
                        //cout << "pdsch length too short for dmrs-AdditionalPosition = 3. Using only one symbol. " << endl;
                        dmrs_symbols = (int *)(malloc(1 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        num_dmrs_symbols_per_slot_ = 1;
                    } else if ((pdsch_length_ > 4) and (pdsch_length_ < 8)) {
                        //cout << "pdsch length too short for dmrs-AdditionalPosition = 3. Using only two symbols. " << endl;
                        dmrs_symbols = (int *)(malloc(2 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 4 + pdsch_start_;
                        num_dmrs_symbols_per_slot_ = 2;
                    } else if (pdsch_length_ == 8) {
                        //cout << "pdsch length too short for dmrs-AdditionalPosition = 3. Using only three symbols. " << endl;
                        dmrs_symbols = (int *)(malloc(3 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 3 + pdsch_start_;
                        dmrs_symbols[2] = 6 + pdsch_start_;
                        num_dmrs_symbols_per_slot_ = 3;
                    } else if ((pdsch_length_ == 9) or (pdsch_length_ == 10))  {
                        dmrs_symbols = (int *)(malloc(3 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 4 + pdsch_start_;
                        dmrs_symbols[2] = 7 + pdsch_start_;
                        num_dmrs_symbols_per_slot_ = 3;
                    } else if (pdsch_length_ > 10) {
                        dmrs_symbols = (int *)(malloc(4 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 3 + pdsch_start_;
                        dmrs_symbols[2] = 6 + pdsch_start_;
                        dmrs_symbols[3] = 9 + pdsch_start_;
                    }
                    break;
            }
        }

    } else if (pdsch_mapping_type_ == 'a') { /// PDSCH mapping type A
        /// Double symbol case
        if(double_symbol) {
            switch (dmrsAdditionalPositions_) {

                case 0 :
                    dmrs_symbols = (int *)(malloc(2 * sizeof(int)));
                    dmrs_symbols[0] = l0_;
                    dmrs_symbols[1] = l0_ + 1;
                    num_dmrs_symbols_per_slot_ = 2;
                    break;

                default : // Case 1
                    if (pdsch_length_ < 10) {
                        dmrs_symbols = (int *)(malloc(2 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = l0_ + 1;
                        num_dmrs_symbols_per_slot_ = 2;
                    }

                    if ((pdsch_length_ < 13) and (pdsch_length_ > 9)) {
                        dmrs_symbols = (int *)(malloc(4 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = l0_ + 1;
                        dmrs_symbols[2] = 8;
                        dmrs_symbols[4] = 9;
                        num_dmrs_symbols_per_slot_ = 4;
                    } else if(pdsch_length_ > 12) {
                        dmrs_symbols = (int *)(malloc(4 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = l0_ + 1;
                        dmrs_symbols[2] = 10;
                        dmrs_symbols[3] = 11;
                        num_dmrs_symbols_per_slot_ = 4;
                    }
                    break;
            }

            /// Single symbol case
        } else {

            /** from free5GRAN */
            switch(dmrsAdditionalPositions_)
            {
                case 0 :
                    dmrs_symbols = (int *)(malloc(1 * sizeof(int)));
                    dmrs_symbols[0] = l0_;
                    num_dmrs_symbols_per_slot_ = 1;
                    break;
                case 1 :
                    if(pdsch_length_ < 8) {
                        //cout << "pdsch length too short for dmrs-AdditionalPosition = 1. Using only one symbol. " << endl;
                        dmrs_symbols = (int *)(malloc(1 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        num_dmrs_symbols_per_slot_ = 1;
                    } else {
                        dmrs_symbols = (int *)(malloc(2 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 7;
                        num_dmrs_symbols_per_slot_ = 2;
                    }
                    break;

                case 2 :
                    if ((pdsch_length_ < 10) and (pdsch_length_ > 7)) {
                        //cout << "pdsch length too short for dmrs-AdditionalPosition = 2. Using only two symbols. " << endl;
                        dmrs_symbols = (int *)(malloc(2 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 7;
                        num_dmrs_symbols_per_slot_ = 2;
                    } else if(pdsch_length_ < 8) {
                        //cout << "pdsch length too short for dmrs-AdditionalPosition = 2. Using only one symbol. " << endl;
                        dmrs_symbols = (int *)(malloc(1 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        num_dmrs_symbols_per_slot_ = 1;
                    } else if ((pdsch_length_ > 9) and (pdsch_length_ < 12)) {
                        dmrs_symbols = (int *)(malloc(3 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 6;
                        dmrs_symbols[2] = 9;
                        num_dmrs_symbols_per_slot_ = 3;
                    } else {
                        dmrs_symbols = (int *)(malloc(3 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 7;
                        dmrs_symbols[2] = 11;
                        num_dmrs_symbols_per_slot_ = 3;
                    }
                    break;

                default : /// Default = 3
                    if (pdsch_length_ < 8) {
                        //cout << "pdsch length too short for dmrs-AdditionalPosition = 3. Using only one symbol. " << endl;
                        dmrs_symbols = (int *)(malloc(1 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        num_dmrs_symbols_per_slot_ = 1;
                    } else if ((pdsch_length_ > 7) and (pdsch_length_ < 10)) {
                        //cout << "pdsch length too short for dmrs-AdditionalPosition = 3. Using only two symbols. " << endl;
                        dmrs_symbols = (int *)(malloc(2 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 7;
                        num_dmrs_symbols_per_slot_ = 2;
                    } else if ((pdsch_length_ > 9) and (pdsch_length_ < 12)) {
                        //cout << "pdsch length too short for dmrs-AdditionalPosition = 3. Using only three symbols. " << endl;
                        dmrs_symbols = (int *)(malloc(3 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 6;
                        dmrs_symbols[2] = 9;
                        num_dmrs_symbols_per_slot_ = 3;
                    } else if (pdsch_length_ > 11) {
                        dmrs_symbols = (int *)(malloc(4 * sizeof(int)));
                        dmrs_symbols[0] = l0_;
                        dmrs_symbols[1] = 5;
                        dmrs_symbols[2] = 8;
                        dmrs_symbols[3] = 11;
                        num_dmrs_symbols_per_slot_ = 4;
                    }
                    break;
            }
        }
    }
}

/** Computes the PDSCH positions within one slot */
void compute_pdsch_positions(const int &pdsch_start_,
                             const int &pdsch_length_,
                             const int &fft_size_,
                             int * dmrs_grid_positions_,
                             int * pdsch_positions) {

    //ofstream output_file("pdsch_positions.txt");

    /// Begin with loop on symbols, then loop on all subcarriers
    for(int symbol = 0; symbol < 14; symbol++) {

        /// Start PDSCH at l0 within the slot
        if(symbol % 14 < pdsch_start_) {
            continue;
        }

        /// End PDSCH at l0 + length within the slot
        if(symbol % 14 - pdsch_start_ + 1 > pdsch_length_) {
            break;
        }

        for(int sc = 0; sc < fft_size_; sc++) {

            /// Do not place PDSCH on the DMRS REs
            if(dmrs_grid_positions_[symbol * fft_size_ + sc]) {
                continue;
            }

            *(pdsch_positions) = symbol; /// Store symbol on even indexes
            *(pdsch_positions + 1) = sc; /// Store subcarriers on odd indexes

            //output_file << *(pdsch_positions) << endl;
            //output_file << *(pdsch_positions + 1) << endl;

            pdsch_positions += 2;
        }
    }
}

/** Computes the PDSCH positions within one slot */
void compute_pdsch_positions(const int &pdsch_start_,
                             const int &pdsch_length_,
                             int * dmrs_symbols_,
                             int * pdsch_symbols_,
                             int * pdsch_sc_,
                             int dmrs_conf_type_,
                             int num_cdm_groups_without_data,
                             int n_rb) /// number of RBs
{

    //ofstream output_file("pdsch_positions.txt");

    int count_dmrs_symbol = 0;

    /// DMRS configuration type 1 : every even subcarrier is occupied by a DMRS RE on DMRS symbols
    if(dmrs_conf_type_ == 1) {

        if(num_cdm_groups_without_data == 2) { /// CDM groups 0 and 1

            /// Begin with loop on symbols, then loop on all subcarriers
            for(int symbol = pdsch_start_; symbol < pdsch_start_ + pdsch_length_; symbol++) {

                /// Do not place PDSCH on DMRS (num_cdm_groups_wo_data = 2)
                if(symbol == dmrs_symbols_[count_dmrs_symbol]) {
                    count_dmrs_symbol++;
                    continue;
                }

                for(int sc = 0; sc < 12 * n_rb; sc++) {
                    *(pdsch_symbols_) = symbol; /// Store symbol on even indexes
                    *(pdsch_sc_) = sc; /// Store subcarriers on odd indexes

                    //output_file << *(pdsch_positions) << endl;
                    //output_file << *(pdsch_positions + 1) << endl;
                    pdsch_symbols_++;
                    pdsch_sc_++;
                }
            }

        } else { /// CDM group 0 only

            // Begin with loop on symbols, then loop on all subcarriers
            for(int symbol = pdsch_start_; symbol < pdsch_start_ + pdsch_length_; symbol++) {

                /// Place PDSCH DMRS on odd subcarriers only
                if(symbol == dmrs_symbols_[count_dmrs_symbol]) {
                    count_dmrs_symbol++;
                    for(int sc = 1; sc < 12 * n_rb; sc+= 2) {
                        *(pdsch_symbols_) = symbol;
                        *(pdsch_sc_) = sc;

                        //output_file << *(pdsch_positions) << endl;
                        //output_file << *(pdsch_positions + 1) << endl;
                        pdsch_symbols_++;
                        pdsch_sc_++;
                    }
                    continue;
                }

                for(int sc = 0; sc < 12 * n_rb; sc++) {
                    *(pdsch_symbols_) = symbol;
                    *(pdsch_sc_) = sc;

                    //output_file << *(pdsch_positions) << endl;
                    //output_file << *(pdsch_positions + 1) << endl;
                    pdsch_symbols_++;
                    pdsch_sc_++;
                }
            }
        }

    } else { /// Conf type 2

        if(num_cdm_groups_without_data == 3) { /// CDM groups 0, 1 and 2

            // Begin with loop on symbols, then loop on all subcarriers
            for(int symbol = pdsch_start_; symbol < pdsch_start_ + pdsch_length_; symbol++) {

                /// Do not place PDSCH on DMRS (num_cdm_groups_wo_data = 2)
                if(symbol == dmrs_symbols_[count_dmrs_symbol]) {
                    count_dmrs_symbol++;
                    continue;
                }

                for(int sc = 0; sc < 12 * n_rb; sc++) {
                    *(pdsch_symbols_) = symbol; /// Store symbol on even indexes
                    *(pdsch_sc_) = sc; /// Store subcarriers on odd indexes

                    //output_file << *(pdsch_positions) << endl;
                    //output_file << *(pdsch_positions + 1) << endl;

                    pdsch_symbols_++;
                    pdsch_sc_++;
                }
            }

        } else if (num_cdm_groups_without_data == 2) { /// CDM groups 0 and 1

            // Begin with loop on symbols, then loop on all subcarriers
            for(int symbol = pdsch_start_; symbol < pdsch_start_ + pdsch_length_; symbol++) {

                /// Place PDSCH DMRS on subcarriers 4,5 and 10, 11 only
                if(symbol == dmrs_symbols_[count_dmrs_symbol]) {
                    count_dmrs_symbol++;
                    for(int rb = 0; rb < n_rb; rb++) {
                        *(pdsch_symbols_)     = symbol;
                        *(pdsch_symbols_ + 1) = symbol;
                        *(pdsch_symbols_ + 2) = symbol;
                        *(pdsch_symbols_ + 3) = symbol;
                        *(pdsch_sc_)     = 12 * rb + 4;
                        *(pdsch_sc_ + 1) = 12 * rb + 5;
                        *(pdsch_sc_ + 2) = 12 * rb + 10;
                        *(pdsch_sc_ + 3) = 12 * rb + 11;

                        //output_file << *(pdsch_positions) << endl;
                        //output_file << *(pdsch_positions + 1) << endl;
                        pdsch_symbols_ += 4;
                        pdsch_sc_ += 4;
                    }
                    continue;
                }

                for(int sc = 0; sc < 12 * n_rb; sc++) {
                    *(pdsch_symbols_) = symbol;
                    *(pdsch_sc_) = sc;

                    //output_file << *(pdsch_positions) << endl;
                    //output_file << *(pdsch_positions + 1) << endl;

                    pdsch_symbols_++;
                    pdsch_sc_++;
                }
            }

        } else { /// CDM group 0 only. num_cdm_groups_wo_data = 1

            // Begin with loop on symbols, then loop on all subcarriers
            for(int symbol = pdsch_start_; symbol < pdsch_start_ + pdsch_length_; symbol++) {

                /// Place PDSCH DMRS on subcarriers 2 to 5 and 8 to 11 only
                if(symbol == dmrs_symbols_[count_dmrs_symbol]) {
                    count_dmrs_symbol++;
                    for(int rb = 0; rb < n_rb; rb++) {
                        *(pdsch_symbols_)     = symbol;
                        *(pdsch_symbols_ + 1) = symbol;
                        *(pdsch_symbols_ + 2) = symbol;
                        *(pdsch_symbols_ + 3) = symbol;
                        *(pdsch_symbols_ + 4)     = symbol;
                        *(pdsch_symbols_ + 5) = symbol;
                        *(pdsch_symbols_ + 6) = symbol;
                        *(pdsch_symbols_ + 7) = symbol;

                        *(pdsch_sc_)     = 12 * rb + 4;
                        *(pdsch_sc_ + 1) = 12 * rb + 5;
                        *(pdsch_sc_ + 2) = 12 * rb + 10;
                        *(pdsch_sc_ + 3) = 12 * rb + 11;
                        *(pdsch_sc_ + 4) = 12 * rb + 2;
                        *(pdsch_sc_ + 5) = 12 * rb + 3;
                        *(pdsch_sc_ + 6) = 12 * rb + 8;
                        *(pdsch_sc_ + 7) = 12 * rb + 9;

                        //output_file << *(pdsch_positions) << endl;
                        //output_file << *(pdsch_positions + 1) << endl;
                        pdsch_symbols_ += 8;
                        pdsch_sc_ += 8;
                    }
                    continue;
                }

                for(int sc = 0; sc < 12 * n_rb; sc++) {
                    *(pdsch_symbols_) = symbol;
                    *(pdsch_sc_) = sc;

                    //output_file << *(pdsch_positions) << endl;
                    //output_file << *(pdsch_positions + 1) << endl;

                    pdsch_symbols_++;
                    pdsch_sc_++;
                }
            }

        }

    }

}

/** Computes the DMRS sequences for each antenna port given in argument, and encodes it with the OCC
     *  According to TS 38.211 section 7.4.1.1.2
     * @param positions_tx_  : vector contaning the DMRS positions within only one slot, for all the possible antenna ports
     *
     */
void compute_dmrs_sequences_type1(int * dmrs_symbols_,
                                  complex<float> * dmrs_sequences_,
                                  const int &num_dmrs_per_slot_,
                                  const int &dmrs_sequence_size_,
                                  const int &nb_slots_,
                                  const bool &double_symbol_) {

    //std::chrono::steady_clock::time_point t1{}, t2{};

    int nb_symbols_dmrs = num_dmrs_per_slot_ * (nb_slots_ - 1);    // Number of OFDM symbols used for DMRS transmission in the entire frame
    int wf_k_prime = 0, wt_l_prime = 0;
    int offset = 0, current_symbol = 0;

    int slot_index = 0;

    //ofstream output_file("dmrs_sequences.txt");

    /// Double symbol case
    if(double_symbol_) {

        /// Compute the DMRS sequence for each symbol within only slot, and encode it with the corresponding OCC (given the antenna
        /// port number)
        for(int antenna_port = 0; antenna_port < 4; antenna_port++) {
            for (int slot = 0; slot < nb_slots_ - 1; slot++) {

                offset = slot * num_dmrs_per_slot_;

                for (int symbol = 0; symbol < num_dmrs_per_slot_; symbol++) {

                    current_symbol = symbol + offset;

                    /// Generate the corresponding DMRS sequence, without OCC encoding
                    generate_pdsch_dmrs_sequence(14,
                                                 slot + 1,
                                                 dmrs_symbols_[symbol],
                                                 0,
                                                 0, // use a cell id of 0
                                                 &dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_],
                                                 dmrs_sequence_size_);

                    /// Encode the DMRS sequence withe the Orthogonal Cover Code (OCC)
                    for (int sc = 0; sc < dmrs_sequence_size_ / 2; sc++) {

                        /// Using DMRS double symbol, compute the sequences for l' = 0 and l' = 1
                        /// Computing sequences for l' = 0
                        wt_l_prime = ts_38_211_7_4_1_1_2_1[antenna_port][4]; // l' = 0

                        /// Even indexes in the sequence
                        wf_k_prime = ts_38_211_7_4_1_1_2_1[antenna_port][2];
                        dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_* dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc].real(
                                wf_k_prime * wt_l_prime * dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc].real());
                        dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc].imag(
                                wf_k_prime * wt_l_prime * dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc].imag());

                        /// Odd indexes in the sequence
                        wf_k_prime = ts_38_211_7_4_1_1_2_1[antenna_port][3];
                        dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc + 1].real(
                                wf_k_prime * wt_l_prime * dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc + 1].real());
                        dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc + 1].imag(
                                wf_k_prime * wt_l_prime * dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc + 1].imag());

                        /// Computing sequences for l' = 1
                        wt_l_prime = ts_38_211_7_4_1_1_2_1[antenna_port][5]; // l' = 1

                        /// Even indexes in the sequence
                        wf_k_prime = ts_38_211_7_4_1_1_2_1[antenna_port][2];
                        dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc].real(
                                wf_k_prime * wt_l_prime * dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc].real());
                        dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc].imag(
                                wf_k_prime * wt_l_prime * dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc].imag());

                        /// Odd indexes in the sequence
                        wf_k_prime = ts_38_211_7_4_1_1_2_1[antenna_port][3];
                        dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc + 1].real(
                                wf_k_prime * wt_l_prime * dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc + 1].real());
                        dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc + 1].imag(
                                wf_k_prime * wt_l_prime * dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc + 1].imag());
                    }
                }
            }
        }

        /// Single symbol case
    } else {

        /// Compute the DMRS sequence for each symbol and encode it with the corresponding OCC (given the antenna
        /// port number)
        for (int antenna_port = 0; antenna_port < 4; antenna_port++) {

            for (int slot = 0; slot < nb_slots_ - 1; slot++) {

                offset = slot * num_dmrs_per_slot_;

                for (int symbol = 0; symbol < num_dmrs_per_slot_; symbol++) {

                    current_symbol = symbol + offset;

                    //output_file << "----------------------" << endl;
                    //output_file << "Port number : " << antenna_port << endl;
                    //output_file << "Slot number : " << slot + 1 << endl;
                    //output_file << "Symbol number : " << dmrs_symbols_[symbol] << endl;
                    //output_file << "______________________" << endl;

                    /// Generate the corresponding DMRS sequence, without OCC encoding
                    generate_pdsch_dmrs_sequence(14,
                                                 slot + 1,
                                                 dmrs_symbols_[symbol],
                                                 0,
                                                 0, // use a cell id of 0
                                                 &dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_],
                                                 dmrs_sequence_size_);

                    /// Encode the DMRS sequence with the Orthogonal Cover Code (OCC)
                    for (int sc = 0; sc < dmrs_sequence_size_/ 2; sc++) {

                        /// Using DMRS single symbol, wt_l_prime is always equal to 1
                        wt_l_prime = ts_38_211_7_4_1_1_2_1[antenna_port][4];

                        /// Even indexes in the sequence
                        wf_k_prime = ts_38_211_7_4_1_1_2_1[antenna_port][2];
                        dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc].real(
                                wf_k_prime * wt_l_prime * dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc].real());
                        dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc].imag(
                                wf_k_prime * wt_l_prime * dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc].imag());

                        /// Odd indexes in the sequence
                        wf_k_prime = ts_38_211_7_4_1_1_2_1[antenna_port][3];
                        dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc + 1].real(
                                wf_k_prime * wt_l_prime * dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc + 1].real());
                        dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc + 1].imag(
                                wf_k_prime * wt_l_prime * dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc + 1].imag());

                        //output_file << dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc] << endl;
                        //output_file << "sc no : " << 2*sc << endl;
                        //output_file << dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc + 1] << endl;
                        //output_file << "sc no : " << 2*sc + 1 << endl;
                    }
                }
            }
        }
    }
}


/** Computes the DMRS sequences for each slot of the entire frame, for each antenna port given in argument, and encodes it with the OCC
     *  According to TS 38.211 section 7.4.1.1.2
     * @param positions_tx_  : vector contaning the DMRS positions within only one slot, for all the possible antenna ports
     *
     */
void compute_dmrs_sequences(int * dmrs_symbols_,
                            vector<float> * dmrs_sequences_real,
                            vector<float> * dmrs_sequences_imag,
                            int num_dmrs_per_slot_,
                            int dmrs_sequence_size_,
                            int nb_slots_,
                            int double_symbol_,
                            int dmrs_config_type) {

    //std::chrono::steady_clock::time_point t1{}, t2{};

    int nb_symbols_dmrs = num_dmrs_per_slot_ * (nb_slots_ - 1);    // Number of OFDM symbols used for DMRS transmission in the entire frame
    int wf_k_prime = 0, wt_l_prime = 0;
    int offset = 0;

    int slot_index = 0;

    //ofstream output_file("dmrs_sequences.txt");

    vector<complex<float>> output_sequence(dmrs_sequence_size_);

    /// DMRS configuration type 1
    if(dmrs_config_type == 1) {
        /// Double symbol case
        if (double_symbol_) {

            /// Compute the DMRS sequence for each symbol within only slot, and encode it with the corresponding OCC (given the antenna
            /// port number)
            for (int antenna_port = 0; antenna_port < 4; antenna_port++) {
                for (int slot = 0; slot < nb_slots_; slot++) {

                    offset = slot * num_dmrs_per_slot_;

                    for (int symbol = 0; symbol < num_dmrs_per_slot_; symbol+= 2) {

                        /// Generate the corresponding DMRS sequence, without OCC encoding
                        generate_pdsch_dmrs_sequence(14,
                                                     slot,
                                                     dmrs_symbols_[symbol],
                                                     0,
                                                     0, // use a cell id of 0
                                                     output_sequence.data(), //dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_],
                                                     dmrs_sequence_size_);

                        /// Encode the DMRS sequence withe the Orthogonal Cover Code (OCC)
                        for (int sc = 0; sc < dmrs_sequence_size_ / 2; sc++) {

                            /// Using DMRS double symbol, compute the sequences for l' = 0 and l' = 1
                            /// Computing sequences for l' = 0
                            wt_l_prime = ts_38_211_7_4_1_1_2_1[antenna_port][4]; // l' = 0

                            /// Even indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_1[antenna_port][2];
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] = output_sequence[2 * sc].real();
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] = output_sequence[2 * sc].imag();
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] *= wf_k_prime * wt_l_prime;
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] *= wf_k_prime * wt_l_prime;

                            /// Odd indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_1[antenna_port][3];
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] = output_sequence[2 * sc + 1].real();
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] = output_sequence[2 * sc + 1].imag();
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] *= wf_k_prime * wt_l_prime;
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] *= wf_k_prime * wt_l_prime;

                            /// Computing sequences for l' = 1
                            wt_l_prime = ts_38_211_7_4_1_1_2_1[antenna_port][5]; // l' = 1

                            /// Even indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_1[antenna_port][2];

                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] = output_sequence[2 * sc].real();
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] = output_sequence[2 * sc].imag();
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] *= wf_k_prime * wt_l_prime;
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] *= wf_k_prime * wt_l_prime;

                            /// Odd indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_1[antenna_port][3];
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] = output_sequence[2 * sc + 1].real();
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] = output_sequence[2 * sc + 1].imag();
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] *= wf_k_prime * wt_l_prime;
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] *= wf_k_prime * wt_l_prime;
                        }
                    }
                }
            }

            /// Single symbol case
        } else {

            /// Compute the DMRS sequence for each symbol and encode it with the corresponding OCC (given the antenna
            /// port number)
            for (int antenna_port = 0; antenna_port < 4; antenna_port++) {

                for (int slot = 0; slot < nb_slots_; slot++) {

                    offset = slot * num_dmrs_per_slot_;

                    for (int symbol = 0; symbol < num_dmrs_per_slot_; symbol++) {
                        //output_file << "----------------------" << endl;
                        //output_file << "Port number : " << antenna_port << endl;
                        //output_file << "Slot number : " << slot + 1 << endl;
                        //output_file << "Symbol number : " << dmrs_symbols_[symbol] << endl;
                        //output_file << "______________________" << endl;

                        /// Generate the corresponding DMRS sequence, without OCC encoding
                        generate_pdsch_dmrs_sequence(14,
                                                     slot,
                                                     dmrs_symbols_[symbol],
                                                     0,
                                                     0, // use a cell id of 0
                                                     output_sequence.data(), //dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_],
                                                     dmrs_sequence_size_);

                        /// Encode the DMRS sequence with the Orthogonal Cover Code (OCC)
                        for (int sc = 0; sc < dmrs_sequence_size_ / 2; sc++) {

                            /// Using DMRS single symbol, wt_l_prime is always equal to 1
                            wt_l_prime = ts_38_211_7_4_1_1_2_1[antenna_port][4];

                            /// Even indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_1[antenna_port][2];
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] = output_sequence[2 * sc].real();
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] = output_sequence[2 * sc].imag();
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] *= wf_k_prime * wt_l_prime;
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] *= wf_k_prime * wt_l_prime;

                            /// Odd indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_1[antenna_port][3];
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] = output_sequence[2 * sc + 1].real();
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] = output_sequence[2 * sc + 1].imag();
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] *= wf_k_prime * wt_l_prime;
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] *= wf_k_prime * wt_l_prime;

                            //output_file << dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc] << endl;
                            //output_file << "sc no : " << 2*sc << endl;
                            //output_file << dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc + 1] << endl;
                            //output_file << "sc no : " << 2*sc + 1 << endl;
                        }
                    }
                }
            }
        }

        /// DMRS configuration type 2
    } else if (dmrs_config_type == 2) {
        /// Double symbol case
        if (double_symbol_) {

            /// Compute the DMRS sequence for each symbol within only slot, and encode it with the corresponding OCC (given the antenna
            /// port number)
            for (int antenna_port = 0; antenna_port < 4; antenna_port++) {
                for (int slot = 0; slot < nb_slots_; slot++) {

                    offset = slot * num_dmrs_per_slot_;

                    for (int symbol = 0; symbol < num_dmrs_per_slot_; symbol+=2) {
                        /// Generate the corresponding DMRS sequence, without OCC encoding
                        generate_pdsch_dmrs_sequence(14,
                                                     slot,
                                                     dmrs_symbols_[symbol],
                                                     0,
                                                     0, // use a cell id of 0
                                                     output_sequence.data(), //dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_],
                                                     dmrs_sequence_size_);

                        /// Encode the DMRS sequence withe the Orthogonal Cover Code (OCC)
                        for (int sc = 0; sc < dmrs_sequence_size_ / 2; sc++) {

                            /// Using DMRS double symbol, compute the sequences for l' = 0 and l' = 1
                            /// Computing sequences for l' = 0
                            wt_l_prime = ts_38_211_7_4_1_1_2_2[antenna_port][4]; // l' = 0

                            /// Even indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_2[antenna_port][2];
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] = output_sequence[2 * sc].real();
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] = output_sequence[2 * sc].imag();
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] *= wf_k_prime * wt_l_prime;
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] *= wf_k_prime * wt_l_prime;

                            /// Odd indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_2[antenna_port][3];
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] = output_sequence[2 * sc + 1].real();
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] = output_sequence[2 * sc + 1].imag();
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] *= wf_k_prime * wt_l_prime;
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] *= wf_k_prime * wt_l_prime;

                            /// Computing sequences for l' = 1
                            wt_l_prime = ts_38_211_7_4_1_1_2_2[antenna_port][5]; // l' = 1

                            /// Even indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_2[antenna_port][2];
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] = output_sequence[2 * sc].real();
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] = output_sequence[2 * sc].imag();
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] *= wf_k_prime * wt_l_prime;
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] *= wf_k_prime * wt_l_prime;

                            /// Odd indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_2[antenna_port][3];
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] = output_sequence[2 * sc + 1].real();
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] = output_sequence[2 * sc + 1].imag();
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] *= wf_k_prime * wt_l_prime;
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] *= wf_k_prime * wt_l_prime;
                        }
                    }
                }
            }

            /// Single symbol case
        } else {

            /// Compute the DMRS sequence for each symbol and encode it with the corresponding OCC (given the antenna
            /// port number)
            for (int antenna_port = 0; antenna_port < 4; antenna_port++) {

                for (int slot = 0; slot < nb_slots_; slot++) {

                    offset = slot * num_dmrs_per_slot_;

                    for (int symbol = 0; symbol < num_dmrs_per_slot_; symbol++) {

                        //output_file << "----------------------" << endl;
                        //output_file << "Port number : " << antenna_port << endl;
                        //output_file << "Slot number : " << slot + 1 << endl;
                        //output_file << "Symbol number : " << dmrs_symbols_[symbol] << endl;
                        //output_file << "______________________" << endl;

                        /// Generate the corresponding DMRS sequence, without OCC encoding
                        generate_pdsch_dmrs_sequence(14,
                                                     slot,
                                                     dmrs_symbols_[symbol],
                                                     0,
                                                     0, // use a cell id of 0
                                                     output_sequence.data(),//&dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_],
                                                     dmrs_sequence_size_);

                        /// Encode the DMRS sequence with the Orthogonal Cover Code (OCC)
                        for (int sc = 0; sc < dmrs_sequence_size_ / 2; sc++) {

                            /// Using DMRS single symbol, wt_l_prime is always equal to 1
                            wt_l_prime = ts_38_211_7_4_1_1_2_2[antenna_port][4];

                            /// Even indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_2[antenna_port][2];
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] = output_sequence[2 * sc].real();
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] = output_sequence[2 * sc].imag();
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] *= wf_k_prime * wt_l_prime;
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] *= wf_k_prime * wt_l_prime;

                            /// Odd indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_2[antenna_port][3];
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] = output_sequence[2 * sc + 1].real();
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] = output_sequence[2 * sc + 1].imag();
                            dmrs_sequences_real[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] *= wf_k_prime * wt_l_prime;
                            dmrs_sequences_imag[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] *= wf_k_prime * wt_l_prime;

                            //output_file << dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc] << endl;
                            //output_file << "sc no : " << 2*sc << endl;
                            //output_file << dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc + 1] << endl;
                            //output_file << "sc no : " << 2*sc + 1 << endl;
                        }
                    }
                }
            }
        }
    }
}

/** Computes the DMRS sequences for each slot of the entire frame, for each antenna port given in argument, and encodes it with the OCC
     *  According to TS 38.211 section 7.4.1.1.2
     * @param positions_tx_  : vector contaning the DMRS positions within only one slot, for all the possible antenna ports
     *
     */
void compute_dmrs_sequences(int * dmrs_symbols_,
                            vector<complex<float>> * dmrs_sequences_,
                            int num_dmrs_per_slot_,
                            int dmrs_sequence_size_,
                            int nb_slots_,
                            int double_symbol_,
                            int dmrs_config_type) {

    //std::chrono::steady_clock::time_point t1{}, t2{};

    int nb_symbols_dmrs = num_dmrs_per_slot_ * (nb_slots_ - 1);    // Number of OFDM symbols used for DMRS transmission in the entire frame
    int wf_k_prime = 0, wt_l_prime = 0;
    int offset = 0;

    int slot_index = 0;

    //ofstream output_file("dmrs_sequences.txt");

    /// DMRS configuration type 1
    if(dmrs_config_type == 1) {
        /// Double symbol case
        if (double_symbol_) {

            /// Compute the DMRS sequence for each symbol within only slot, and encode it with the corresponding OCC (given the antenna
            /// port number)
            for (int antenna_port = 0; antenna_port < 4; antenna_port++) {
                for (int slot = 0; slot < nb_slots_; slot++) {

                    offset = slot * num_dmrs_per_slot_;

                    for (int symbol = 0; symbol < num_dmrs_per_slot_; symbol+= 2) {

                        /// Generate the corresponding DMRS sequence, without OCC encoding
                        generate_pdsch_dmrs_sequence(14,
                                                     slot,
                                                     dmrs_symbols_[symbol],
                                                     0,
                                                     0, // use a cell id of 0
                                                     &dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_],
                                                     dmrs_sequence_size_);

                        /// Encode the DMRS sequence withe the Orthogonal Cover Code (OCC)
                        for (int sc = 0; sc < dmrs_sequence_size_ / 2; sc++) {

                            /// Using DMRS double symbol, compute the sequences for l' = 0 and l' = 1
                            /// Computing sequences for l' = 0
                            wt_l_prime = ts_38_211_7_4_1_1_2_1[antenna_port][4]; // l' = 0

                            /// Even indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_1[antenna_port][2];
                            dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] *= wf_k_prime * wt_l_prime;

                            /// Odd indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_1[antenna_port][3];
                            dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] *= wf_k_prime * wt_l_prime;

                            /// Computing sequences for l' = 1
                            wt_l_prime = ts_38_211_7_4_1_1_2_1[antenna_port][5]; // l' = 1

                            /// Even indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_1[antenna_port][2];
                            dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][(symbol + 1) * dmrs_sequence_size_ + 2 * sc] *= wf_k_prime * wt_l_prime;

                            /// Odd indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_1[antenna_port][3];
                            dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][(symbol + 1) * dmrs_sequence_size_ + 2 * sc + 1] *= wf_k_prime * wt_l_prime;
                        }
                    }
                }
            }

            /// Single symbol case
        } else {

            /// Compute the DMRS sequence for each symbol and encode it with the corresponding OCC (given the antenna
            /// port number)
            for (int antenna_port = 0; antenna_port < 4; antenna_port++) {

                for (int slot = 0; slot < nb_slots_; slot++) {

                    offset = slot * num_dmrs_per_slot_;

                    for (int symbol = 0; symbol < num_dmrs_per_slot_; symbol++) {
                        //output_file << "----------------------" << endl;
                        //output_file << "Port number : " << antenna_port << endl;
                        //output_file << "Slot number : " << slot + 1 << endl;
                        //output_file << "Symbol number : " << dmrs_symbols_[symbol] << endl;
                        //output_file << "______________________" << endl;

                        /// Generate the corresponding DMRS sequence, without OCC encoding
                        generate_pdsch_dmrs_sequence(14,
                                                     slot,
                                                     dmrs_symbols_[symbol],
                                                     0,
                                                     0, // use a cell id of 0
                                                     &dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_],
                                                     dmrs_sequence_size_);

                        /// Encode the DMRS sequence with the Orthogonal Cover Code (OCC)
                        for (int sc = 0; sc < dmrs_sequence_size_ / 2; sc++) {

                            /// Using DMRS single symbol, wt_l_prime is always equal to 1
                            wt_l_prime = ts_38_211_7_4_1_1_2_1[antenna_port][4];

                            /// Even indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_1[antenna_port][2];
                            dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] *= wf_k_prime * wt_l_prime;

                            /// Odd indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_1[antenna_port][3];
                            dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] *= wf_k_prime * wt_l_prime;

                            //output_file << dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc] << endl;
                            //output_file << "sc no : " << 2*sc << endl;
                            //output_file << dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc + 1] << endl;
                            //output_file << "sc no : " << 2*sc + 1 << endl;
                        }
                    }
                }
            }
        }

    /// DMRS configuration type 2
    } else if (dmrs_config_type == 2) {
        /// Double symbol case
        if (double_symbol_) {

            /// Compute the DMRS sequence for each symbol within only slot, and encode it with the corresponding OCC (given the antenna
            /// port number)
            for (int antenna_port = 0; antenna_port < 4; antenna_port++) {
                for (int slot = 0; slot < nb_slots_; slot++) {

                    offset = slot * num_dmrs_per_slot_;

                    for (int symbol = 0; symbol < num_dmrs_per_slot_; symbol+=2) {
                        /// Generate the corresponding DMRS sequence, without OCC encoding
                        generate_pdsch_dmrs_sequence(14,
                                                     slot,
                                                     dmrs_symbols_[symbol],
                                                     0,
                                                     0, // use a cell id of 0
                                                     &dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_],
                                                     dmrs_sequence_size_);

                        /// Encode the DMRS sequence withe the Orthogonal Cover Code (OCC)
                        for (int sc = 0; sc < dmrs_sequence_size_ / 2; sc++) {

                            /// Using DMRS double symbol, compute the sequences for l' = 0 and l' = 1
                            /// Computing sequences for l' = 0
                            wt_l_prime = ts_38_211_7_4_1_1_2_2[antenna_port][4]; // l' = 0

                            /// Even indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_2[antenna_port][2];
                            dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] *= wf_k_prime * wt_l_prime;

                            /// Odd indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_2[antenna_port][3];
                            dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] *= wf_k_prime * wt_l_prime;

                            /// Computing sequences for l' = 1
                            wt_l_prime = ts_38_211_7_4_1_1_2_2[antenna_port][5]; // l' = 1

                            /// Even indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_2[antenna_port][2];
                            dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][(symbol + 1) * dmrs_sequence_size_ + 2 * sc] *= wf_k_prime * wt_l_prime;

                            /// Odd indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_2[antenna_port][3];
                            dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][(symbol + 1) * dmrs_sequence_size_ + 2 * sc + 1] *= wf_k_prime * wt_l_prime;
                        }
                    }
                }
            }

            /// Single symbol case
        } else {

            /// Compute the DMRS sequence for each symbol and encode it with the corresponding OCC (given the antenna
            /// port number)
            for (int antenna_port = 0; antenna_port < 4; antenna_port++) {

                for (int slot = 0; slot < nb_slots_; slot++) {

                    offset = slot * num_dmrs_per_slot_;

                    for (int symbol = 0; symbol < num_dmrs_per_slot_; symbol++) {

                        //output_file << "----------------------" << endl;
                        //output_file << "Port number : " << antenna_port << endl;
                        //output_file << "Slot number : " << slot + 1 << endl;
                        //output_file << "Symbol number : " << dmrs_symbols_[symbol] << endl;
                        //output_file << "______________________" << endl;

                        /// Generate the corresponding DMRS sequence, without OCC encoding
                        generate_pdsch_dmrs_sequence(14,
                                                     slot,
                                                     dmrs_symbols_[symbol],
                                                     0,
                                                     0, // use a cell id of 0
                                                     &dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_],
                                                     dmrs_sequence_size_);

                        /// Encode the DMRS sequence with the Orthogonal Cover Code (OCC)
                        for (int sc = 0; sc < dmrs_sequence_size_ / 2; sc++) {

                            /// Using DMRS single symbol, wt_l_prime is always equal to 1
                            wt_l_prime = ts_38_211_7_4_1_1_2_2[antenna_port][4];

                            /// Even indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_2[antenna_port][2];
                            dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc] *= wf_k_prime * wt_l_prime;

                            /// Odd indexes in the sequence
                            wf_k_prime = ts_38_211_7_4_1_1_2_2[antenna_port][3];
                            dmrs_sequences_[slot * MAX_TX_PORTS + antenna_port][symbol * dmrs_sequence_size_ + 2 * sc + 1] *= wf_k_prime * wt_l_prime;

                            //output_file << dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc] << endl;
                            //output_file << "sc no : " << 2*sc << endl;
                            //output_file << dmrs_sequences_[antenna_port * (nb_slots_ - 1) * num_dmrs_per_slot_ * dmrs_sequence_size_ + slot * num_dmrs_per_slot_ * dmrs_sequence_size_ + symbol * dmrs_sequence_size_ + 2 * sc + 1] << endl;
                            //output_file << "sc no : " << 2*sc + 1 << endl;
                        }
                    }
                }
            }
        }
    }
}

/************************ TEST réalignement ************************************************/
void get_pdsch_and_dmrs_samples(const vector<vector<complex<float>>> &synchronized_grid_,
                                int slot_number_,
                                const int * dmrs_symbols_,
                                int dmrs_sequence_size_,
                                int num_dmrs_symbols_per_slot_,
                                int num_cdm_groups_without_data,
                                int pdsch_length_,
                                int fft_size_,
                                int pdsch_start_symbol_,
                                int rx_port_index,
                                complex<float> pdsch_samples_[][MAX_RX_PORTS],
                                vector<complex<float>> * dmrs_samples_) {

    int dmrs_symbol_index = 0;
    int count_pdsch_samples = 0;
    int count_dmrs_samples = 0;

    /// For DMRS conf type 1 only. if num cdm groups without data is equal to 2,
    /// All the DMRS symbols are occupied by DMRS in frequency
    if(num_cdm_groups_without_data > 1) {

        for(int symbol = pdsch_start_symbol_; symbol < pdsch_length_ + pdsch_start_symbol_; symbol++) {

            if(symbol == dmrs_symbols_[dmrs_symbol_index]) {
                dmrs_symbol_index++;
                continue; /// Do not take any PDSCH on DMRS symbol
            }

            for(int sc = 0; sc < fft_size_; sc++) {
                pdsch_samples_[count_pdsch_samples][rx_port_index] = synchronized_grid_[symbol + slot_number_ * 14][sc];
                count_pdsch_samples++;
            }
        }

        /// DMRS extraction
        for (int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {
            for (int sc = 0; sc < dmrs_sequence_size_; sc++) {
                dmrs_samples_[0][count_dmrs_samples] = synchronized_grid_[dmrs_symbols_[symbol] + slot_number_ * 14][2*sc]; /// samples for CDM group 0
                dmrs_samples_[1][count_dmrs_samples] = synchronized_grid_[dmrs_symbols_[symbol] + slot_number_ * 14][2*sc + 1]; //// samples for CDM group 1
                count_dmrs_samples++;
                //cout << "DMRS symbol : " << dmrs_symbols_[symbol] + slot_number_ * 14 << endl;
                //cout << "DMRS subcarrier CDM group 0 : " << 2 * sc << endl;
                //cout << "DMRS subcarrier CDM group 1 : " << 2 * sc + 1 << endl;
            }
        }

    } else { /// No co-scheduled DL DMRS. The only group possibly used is CDM group 0 (see TS 38.212 tables).

        /// PDSCH extraction
        for(int symbol = pdsch_start_symbol_; symbol < pdsch_length_ + pdsch_start_symbol_; symbol++) {

            if(symbol == dmrs_symbols_[dmrs_symbol_index]) {
                dmrs_symbol_index++;

                /// Extract PDSCH on odd subcarriers on DMRS symbols
                for(int sc = 1; sc < fft_size_; sc+=2) {
                    pdsch_samples_[count_pdsch_samples][rx_port_index] = synchronized_grid_[symbol + slot_number_ * 14][sc];
                    count_pdsch_samples++;
                }

            } else {
                /// Extract PDSCH on all subcarriers
                for(int sc = 0; sc < fft_size_; sc++) {
                    pdsch_samples_[count_pdsch_samples][rx_port_index] = synchronized_grid_[symbol + slot_number_ * 14][sc];
                    count_pdsch_samples++;
                }
            }
        }

        /// DMRS extraction
        for(int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {
            for(int sc = 0; sc < dmrs_sequence_size_; sc++) {
                dmrs_samples_[0][count_dmrs_samples] = synchronized_grid_[dmrs_symbols_[symbol] + slot_number_ * 14][2*sc]; /// samples for CDM group 0
                count_dmrs_samples++;
            }
        }
    }
}
/*******************************************************************************************/

void get_pdsch_and_dmrs_samples(const vector<vector<complex<float>>> &synchronized_grid_,
                                int slot_number_,
                                const int * dmrs_symbols_,
                                int dmrs_sequence_size_,
                                int num_dmrs_symbols_per_slot_,
                                int num_cdm_groups_without_data,
                                int pdsch_length_,
                                int fft_size_,
                                int pdsch_start_symbol_,
                                vector<complex<float>> &pdsch_samples_,
                                vector<complex<float>> * dmrs_samples_) {

    int dmrs_symbol_index = 0;
    int count_pdsch_samples = 0;
    int count_dmrs_samples = 0;
    int current_symbol;

    /// For DMRS conf type 1 only. if num cdm groups without data is equal to 2,
    /// All the DMRS symbols are occupied by DMRS in frequency
    if(num_cdm_groups_without_data > 1) {

        for(int symbol = pdsch_start_symbol_; symbol < pdsch_length_ + pdsch_start_symbol_; symbol++) {

            if(symbol == dmrs_symbols_[dmrs_symbol_index]) {
                dmrs_symbol_index++;
                continue; /// Do not take any PDSCH on DMRS symbol
            }


#if defined(__AVX2__) and (AVX2_PROCESSING)
            for(int sc = 0; sc < fft_size_; sc+=4) {
                _mm256_storeu_ps((float *) &pdsch_samples_[count_pdsch_samples], _mm256_loadu_ps((float *) &synchronized_grid_[symbol + slot_number_ * 14][sc]));
                count_pdsch_samples += 4;
            }
#else
            memcpy(&pdsch_samples_[count_pdsch_samples], synchronized_grid_[symbol + slot_number_ * 14].data(), fft_size_ * sizeof(complex<float>));
            count_pdsch_samples += fft_size_;
            /*
            for(int sc = 0; sc < fft_size_; sc+=4) {
                pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol + slot_number_ * 14][sc];
                count_pdsch_samples++;
                //cout << "PDSCH symbol : " << symbol + slot_number_ * 14 << endl;
                //cout << "PDSCH sc : " << sc << endl;
                //count_pdsch_samples += 4;
            } */
#endif
        }

            /// DMRS extraction
            for (int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {
                current_symbol = dmrs_symbols_[symbol] + slot_number_ * 14;
                for (int sc = 0; sc < dmrs_sequence_size_; sc++) {
                    dmrs_samples_[0][count_dmrs_samples] = synchronized_grid_[current_symbol][2*sc]; /// samples for CDM group 0
                    dmrs_samples_[1][count_dmrs_samples] = synchronized_grid_[current_symbol][2*sc + 1]; //// samples for CDM group 1

                    count_dmrs_samples++;
                    //cout << "DMRS symbol : " << dmrs_symbols_[symbol] + slot_number_ * 14 << endl;
                    //cout << "DMRS subcarrier CDM group 0 : " << 2 * sc << endl;
                    //cout << "DMRS subcarrier CDM group 1 : " << 2 * sc + 1 << endl;
                }
            }

    } else { /// No co-scheduled DL DMRS. The only group possibly used is CDM group 0 (see TS 38.212 tables).

        /// PDSCH extraction
        for(int symbol = pdsch_start_symbol_; symbol < pdsch_length_ + pdsch_start_symbol_; symbol++) {

            if(symbol == dmrs_symbols_[dmrs_symbol_index]) {
                dmrs_symbol_index++;

                /// Extract PDSCH on odd subcarriers on DMRS symbols
                for(int sc = 1; sc < fft_size_; sc+=2) {
                    pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol + slot_number_ * 14][sc];
                    count_pdsch_samples++;
                }

            } else {

#if defined(__AVX2__) and defined(AVX2_PROCESSING)
                /// Extract PDSCH on all subcarriers
                for(int sc = 0; sc < fft_size_; sc+=4) {
                    _mm256_storeu_ps((float *) &pdsch_samples_[count_pdsch_samples], _mm256_loadu_ps((float *) &synchronized_grid_[symbol + slot_number_ * 14][sc]));
                    count_pdsch_samples += 4;
                }
#else
                /// Extract PDSCH on all subcarriers
                memcpy(&pdsch_samples_[count_pdsch_samples], synchronized_grid_[symbol + slot_number_ * 14].data(), fft_size_ * sizeof(complex<float>));
                count_pdsch_samples += fft_size_;
                /**
                for(int sc = 0; sc < fft_size_; sc++) {
                    pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol + slot_number_ * 14][sc];
                    count_pdsch_samples++;
                } */
#endif
            }
        }

        /// DMRS extraction
        /*
#if defined(__AVX2__) and defined(AVX2_PROCESSING)
        for(int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {
            current_symbol = dmrs_symbols_[symbol] + slot_number_ * 14;
            for(int sc = 0; sc < fft_size_; sc += 4) {
                _mm256_maskstore_ps((float *) &dmrs_samples_[0][count_dmrs_samples], dmrs_mask, _mm256_loadu_ps((float *) &synchronized_grid_[current_symbol][sc]));
                count_dmrs_samples += 2;
            }
        }
#else */
        for(int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {
            current_symbol = dmrs_symbols_[symbol] + slot_number_ * 14;
            for(int sc = 0; sc < dmrs_sequence_size_; sc++) {
                dmrs_samples_[0][count_dmrs_samples] = synchronized_grid_[current_symbol][2*sc]; /// samples for CDM group 0
                count_dmrs_samples++;
            }
        }
//#endif
    }
}

void get_pdsch_and_dmrs_samples(const vector<vector<complex<float>>> &synchronized_grid_,
                                vector<complex<float>> &pdsch_samples_,
                                vector<complex<float>> dmrs_samples_[MAX_NUM_CDM_GROUPS],
                                int slot_number_,
                                const int * dmrs_symbols_,
                                int num_dmrs_symbols_per_slot_,
                                int num_cdm_groups_without_data,
                                int pdsch_length_,
                                int fft_size_,
                                int sc_rb_start_,
                                int pdsch_start_symbol_,
                                int dmrs_config_type_,
                                int * cdm_groups_,
                                int num_used_cdm_groups_) {

    int dmrs_symbol_index = 0;
    int count_pdsch_samples = 0;
    int count_dmrs_samples = 0;

    if (dmrs_config_type_ == 1) {
        /// For DMRS conf type 1 only. if num cdm groups without data is equal to 2,
        /// All the DMRS symbols are occupied by DMRS in frequency
        if (num_cdm_groups_without_data > 1) {

            for (int symbol = pdsch_start_symbol_ + slot_number_  * 14; symbol < pdsch_length_ + pdsch_start_symbol_ + slot_number_ * 14; symbol++) {

                if (symbol % 14 == dmrs_symbols_[dmrs_symbol_index]) {
                    dmrs_symbol_index++;
                    continue; /// Do not take any PDSCH on DMRS symbol
                }

                for (int sc = sc_rb_start_; sc < fft_size_ + sc_rb_start_; sc++) {
                    pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol][sc];
                    count_pdsch_samples++;
                }
            }

            /// DMRS extraction. Extract only the useful DMRS
            for (int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {
                for (int sc = sc_rb_start_; sc < fft_size_ + sc_rb_start_; sc += 2) {
                    for(int cdm_group = 0; cdm_group < num_used_cdm_groups_; cdm_group++) {
                        dmrs_samples_[cdm_groups_[cdm_group]][count_dmrs_samples] = synchronized_grid_[dmrs_symbols_[symbol] +
                                                                                  slot_number_ * 14][sc + cdm_groups_[cdm_group]]; //// samples for CDM group 1
                    }
                    count_dmrs_samples++;
                }
            }

        } else { /// No co-scheduled DL DMRS. The only group possibly used is CDM group 0 (see TS 38.212 tables).

            /// PDSCH extraction
            for (int symbol = pdsch_start_symbol_ + slot_number_ * 14; symbol < pdsch_length_ + pdsch_start_symbol_ + slot_number_ * 14; symbol++) {

                if (symbol % 14 == dmrs_symbols_[dmrs_symbol_index]) {
                    dmrs_symbol_index++;

                    /// Extract PDSCH on odd subcarriers on DMRS symbols
                    for (int sc = sc_rb_start_ + 1; sc < fft_size_ + sc_rb_start_; sc += 2) {
                        pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol][sc];
                        count_pdsch_samples++;
                    }

                } else {
                    /// Extract PDSCH on all subcarriers
                    for (int sc = sc_rb_start_; sc < fft_size_ + sc_rb_start_; sc++) {
                        pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol][sc];
                        count_pdsch_samples++;
                    }
                }
            }

            /// DMRS extraction
            for (int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {
                for (int sc = sc_rb_start_; sc < fft_size_ + sc_rb_start_; sc += 2) {
                    dmrs_samples_[0][count_dmrs_samples] = synchronized_grid_[dmrs_symbols_[symbol] +
                                                                              slot_number_ * 14][sc]; /// samples for CDM group 0
                    count_dmrs_samples++;
                }
            }
        }

        // DMRS config type 2
    } else if (dmrs_config_type_ == 2) {

        /// CDM groups 0, 1 and 2 are used
        if (num_cdm_groups_without_data == 3) {

            for (int symbol = pdsch_start_symbol_ + slot_number_ * 14; symbol < pdsch_length_ + pdsch_start_symbol_ + slot_number_ * 14; symbol++) {

                if (symbol % 14 == dmrs_symbols_[dmrs_symbol_index]) {
                    dmrs_symbol_index++;
                    continue; /// Do not take any PDSCH on DMRS symbol
                }

                for (int sc = sc_rb_start_; sc < fft_size_ + sc_rb_start_; sc++) {
                    pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol][sc];
                    count_pdsch_samples++;
                }
            }

            /// DMRS extraction
            for (int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {
                for (int sc = sc_rb_start_; sc < fft_size_ + sc_rb_start_; sc += 6) {
                    for (int cdm_group = 0; cdm_group < num_used_cdm_groups_; cdm_group++) {
                        dmrs_samples_[cdm_groups_[cdm_group]][count_dmrs_samples] =
                                synchronized_grid_[dmrs_symbols_[symbol] + slot_number_ * 14][sc + 2 * cdm_groups_[cdm_group]];
                        dmrs_samples_[cdm_groups_[cdm_group]][count_dmrs_samples + 1] = synchronized_grid_[dmrs_symbols_[symbol] +
                                                                                  slot_number_ * 14][sc + 1 + 2 * cdm_groups_[cdm_group]];
                    }
                    count_dmrs_samples += 2;
                }
            }

            /// CDM groups 0 and 1 are used
        } else if (num_cdm_groups_without_data == 2) {

            /// PDSCH extraction
            for (int symbol = pdsch_start_symbol_ + slot_number_ * 14; symbol < pdsch_length_ + pdsch_start_symbol_ + slot_number_ * 14; symbol++) {

                if (symbol % 14 == dmrs_symbols_[dmrs_symbol_index]) {
                    dmrs_symbol_index++;

                    /// Extract PDSCH on subcarriers 4, 5 and 10, 11 of a RB
                    for (int sc = sc_rb_start_; sc < fft_size_ + sc_rb_start_; sc += 12) {
                        pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol][sc + 4];
                        count_pdsch_samples++;
                        pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol][sc + 5];
                        count_pdsch_samples++;
                        pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol][sc + 10];
                        count_pdsch_samples++;
                        pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol][sc + 11];
                        count_pdsch_samples++;
                    }

                } else {
                    /// Extract PDSCH on all subcarriers
                    for (int sc = sc_rb_start_; sc < fft_size_ + sc_rb_start_; sc++) {
                        pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol][sc];
                        count_pdsch_samples++;
                    }
                }
            }

            /// DMRS extraction
            for (int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {
                for (int sc = sc_rb_start_; sc < fft_size_ + sc_rb_start_; sc += 6) {
                    for (int cdm_group = 0; cdm_group < num_used_cdm_groups_; cdm_group++) {
                        dmrs_samples_[cdm_groups_[cdm_group]][count_dmrs_samples] =
                                synchronized_grid_[dmrs_symbols_[symbol] + slot_number_ * 14][sc + 2 * cdm_groups_[cdm_group]];
                        dmrs_samples_[cdm_groups_[cdm_group]][count_dmrs_samples + 1] = synchronized_grid_[dmrs_symbols_[symbol] +
                                                                                                           slot_number_ * 14][sc + 1 + 2 * cdm_groups_[cdm_group]];
                    }
                    count_dmrs_samples += 2;
                }
            }
        }

            /// CDM group 0 is used
        else { /// No co-scheduled DL DMRS. The only group possibly used is CDM group 0 (see TS 38.212 tables).

            /// PDSCH extraction
            for (int symbol = pdsch_start_symbol_ + slot_number_ * 14; symbol < pdsch_length_ + pdsch_start_symbol_ + slot_number_ * 14; symbol++) {

                if (symbol % 14 == dmrs_symbols_[dmrs_symbol_index]) {
                    dmrs_symbol_index++;

                    /// Extract PDSCH on subcarriers 2, 3, 4, 5 and 8, 9, 10, 11
                    for (int sc = sc_rb_start_; sc < fft_size_ + sc_rb_start_; sc += 12) {
                        pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol][sc + 2];
                        count_pdsch_samples++;
                        pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol][sc + 3];
                        count_pdsch_samples++;
                        pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol][sc + 4];
                        count_pdsch_samples++;
                        pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol][sc + 5];
                        count_pdsch_samples++;
                        pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol][sc + 8];
                        count_pdsch_samples++;
                        pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol][sc + 9];
                        count_pdsch_samples++;
                        pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol][sc + 10];
                        count_pdsch_samples++;
                        pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol][sc + 11];
                        count_pdsch_samples++;
                    }

                } else {
                    /// Extract PDSCH on all subcarriers
                    for (int sc = sc_rb_start_; sc < fft_size_ + sc_rb_start_; sc++) {
                        pdsch_samples_[count_pdsch_samples] = synchronized_grid_[symbol][sc];
                        count_pdsch_samples++;
                    }
                }
            }

            /// DMRS extraction
            for (int symbol = 0; symbol < num_dmrs_symbols_per_slot_; symbol++) {
                for (int sc = sc_rb_start_; sc < fft_size_ + sc_rb_start_; sc += 6) {
                    dmrs_samples_[0][count_dmrs_samples] = synchronized_grid_[dmrs_symbols_[symbol] +
                                                                              slot_number_ *
                                                                              14][sc]; //// samples for CDM group 0
                    count_dmrs_samples++;
                    dmrs_samples_[0][count_dmrs_samples] = synchronized_grid_[dmrs_symbols_[symbol] +
                                                                              slot_number_ * 14][sc +
                                                                                                 1]; //// samples for CDM group 0
                    count_dmrs_samples++;
                }
            }
        }
    }
}