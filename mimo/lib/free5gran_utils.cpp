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

   This file contains all functions reused or modified from the free5GRAN library.
*/

#include "free5gran_utils.h"

using namespace std;

static fftwf_complex * in_ifft;
static fftwf_complex * out_ifft;
static fftwf_plan p_ifft;

static fftwf_complex * in_fft;
static fftwf_complex * out_fft;
static fftwf_plan p_fft;

static fftwf_complex * pss_in_0_static;
static fftwf_complex * pss_out_0_static;
static fftwf_plan ifft_plan_0_static;

static fftwf_complex * pss_in_0_downsampled;
static fftwf_complex * pss_out_0_downsampled;
static fftwf_plan ifft_plan_0_downsampled;

/** From free5GRAN. Base sequence for DMRS generation */
static int DMRS_BASE_X1_SEQUENCE[] = {1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                                      0, 0, 0, 0, 0, 0, 0, 0, 0};

/** From free5GRAN. Base sequence for PSS generation */
static int PSS_BASE_SEQUENCE[]     = {0, 1, 1, 0, 1, 1, 1};

/** Initialize the static FFT plans used for PSS synchronization
 *
 * @param fft_size_ : FFT size
 * @param downsampling_factor_ : the downsampling factor typically set to fft_size_/PSS_SEQUENCE_SIZE
 */
void init_sync_pss_plans(int fft_size_, int downsampling_factor_) {
    pss_in_0_static    = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * fft_size_);
    pss_out_0_static   = (fftwf_complex*)fftwf_malloc(sizeof(fftwf_complex) * fft_size_);
    ifft_plan_0_static = fftwf_plan_dft_1d(fft_size_, pss_in_0_static, pss_out_0_static,
                                               FFTW_BACKWARD, FFTW_MEASURE);

    int fft_size_downsampled = fft_size_ / downsampling_factor_;
    cout << "fft_size / downsampling_factor " << fft_size_downsampled << endl;

    pss_in_0_downsampled = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * fft_size_/downsampling_factor_);
    pss_out_0_downsampled = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * fft_size_/downsampling_factor_);
    ifft_plan_0_downsampled = fftwf_plan_dft_1d(fft_size_/downsampling_factor_, pss_in_0_downsampled, pss_out_0_downsampled,
                                                           FFTW_BACKWARD, FFTW_MEASURE);

}

/** Free the static FFT plans used for PSS synchronization */
void free_sync_pss_plans() {
    fftwf_destroy_plan(ifft_plan_0_static);
    fftwf_free(pss_in_0_static);
    fftwf_free(pss_out_0_static);

    fftwf_destroy_plan(ifft_plan_0_downsampled);
    fftwf_free(pss_in_0_downsampled);
    fftwf_free(pss_out_0_downsampled);
}

/** Initialize the static FFT plans
 *
 * @param fft_size : FFT size
 */
void init_fft_plans(const int &fft_size) {
    in_fft  = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    out_fft = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    p_fft   = fftwf_plan_dft_1d(fft_size, in_fft, out_fft, FFTW_FORWARD, FFTW_MEASURE);
}

/** Initialize the static IFFT plans
 *
 * @param fft_size : FFT size
 */
void init_ifft_plans(const int &fft_size) {
    in_ifft  = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    out_ifft = (fftwf_complex*) fftwf_malloc(sizeof(fftwf_complex) * fft_size);
    p_ifft   = fftwf_plan_dft_1d(fft_size, in_ifft, out_ifft, FFTW_BACKWARD, FFTW_MEASURE);
}

/** Free the static FFT plans */
void destroy_fft_plans() {
    fftwf_destroy_plan(p_fft);
    fftwf_free(in_fft);
    fftwf_free(out_fft);
}

/** Free the static ifft plans */
void destroy_ifft_plans() {
    fftwf_destroy_plan(p_ifft);
    fftwf_free(in_ifft);
    fftwf_free(out_ifft);
}

/** Exports a one dimension vector
 *
 * @param filename_      : output stream
 * @param signal : signal to be exported
 */
void export1dVector(string filename_, vector<complex<float>> &signal) {

    ofstream s(filename_);

    for (int i = 0; i < signal.size(); i++) {
        s << signal[i] << '\n';
    }
}

/** Exports a one dimension vector of ints
 *
 * @param filename_      : output stream
 * @param signal : vector to be exported
 */
void export1dVector(string filename_, vector<int> &signal) {

    ofstream s(filename_);

    for (int i = 0; i < signal.size(); i++) {
        s << to_string(signal[i])
          << '\n';
    }
}

/** Exports a one dimension array of ints
 *
 * @param filename_      : output stream
 * @param signal : array to be exported
 * @param num_elements : number of elements in the  array
 */
void export1dVector(string filename_, int * signal, int &num_elements) {

    ofstream s(filename_);

    for (int i = 0; i < num_elements; i++) {
        s << to_string(signal[i])
          << '\n';
    }
}

/** Exports a one dimension array of complex<float>
 *
 * @param filename_      : output stream
 * @param signal : array to be exported
 * @param num_elements : number of elements in the array
 */
void export1dVector(string filename_, complex<float> * signal, int &num_elements) {

    ofstream s(filename_);

    for (int i = 0; i < num_elements; i++) {
        s << signal[i] << '\n';
    }
}

/** Used to export the frequency domain grids
 *
 * @param filename : output stream
 * @param signal : grid to be exported
 */
void exportGrid(string filename_, vector<vector<complex<float>>> &signal) {

    ofstream s(filename_);

    for (int i = 0; i < signal.size(); i++) {
        for(int j = 0; j < signal[i].size(); j++) {
            s << signal[i][j] << '\n';
        }
    }
}

/** Export a signal that has the following dimensions : number of slots,
 *      number of receive ports, number of transmit ports, number of symbols, FFT size
 *
 * @param filename_ : export filename
 * @param signal : signal to be exported
 * @param num_symbols_ : number of symbols
 * @param num_sc_ : FFT size
 * @param num_tx_ports_ : number of transmit ports
 * @param transmitter : transmitter index
 * @param receiver : receiver index
 */
void exportGrid(string filename_,
                vector<complex<float> *> &signal,
                const int &num_symbols_,
                const int &num_sc_,
                const int &num_tx_ports_,
                const int &transmitter,
                const int &receiver) {

    ofstream s(filename_);

    for(int slot = 0; slot < signal.size(); slot++) {
        for(int symbol = 0; symbol < num_symbols_; symbol++) {
            for(int sc = 0; sc < num_sc_; sc++) {
                s << signal[slot][receiver * num_tx_ports_ * num_symbols_ * num_sc_
                     + transmitter * num_symbols_ * num_sc_ + symbol * num_sc_ + sc] << '\n';
            }
        }
    }
}

/** Export a signal that has the followwing dimensions : number of slots,
 *      number of receive ports, number of transmit ports, number of symbols per slot, FFT size
 *
 * @param filename_ : export filename
 * @param signal : signal to be exported
 * @param num_symbols_ : number of symbols per slot
 * @param num_sc_ : FFT size
 * @param num_tx_ports_ : number of transmit ports
 * @param transmitter : transmitter index
 * @param receiver : receiver index
 */
void exportGrid(string filename_,
                vector<vector<complex<float>>> &signal,
                const int &num_symbols_,
                const int &num_sc_,
                const int &num_tx_ports_,
                const int &transmitter,
                const int &receiver) {

    ofstream s(filename_);

    for(int slot = 0; slot < signal.size(); slot++) {
        for(int symbol = 0; symbol < num_symbols_; symbol++) {
            for(int sc = 0; sc < num_sc_; sc++) {
                s << signal[slot][receiver * num_tx_ports_ * num_symbols_ * num_sc_
                                  + transmitter * num_symbols_ * num_sc_ + symbol * num_sc_ + sc] << '\n';
            }
        }
    }
}

/** Modified from free5GRAN IFFT function. Compute the IFFT for the whole frame.
 *
 * @param[in] inputGrid             : Grid whose IFFT is to be computed
 * @param[out] output                    : grid containing samples after IFFT. Provide an empty grid as an argument.
 * @param{in] scaling_factor            : IFFT scaling factor
 * @param{in] fft_size                  : FFT size
 * @param[in] symbols_per_subframe      : number of symbols per subframe
 * @param[in] nb_subframes              : number of subframes to be sent
 * @param[in] scs                       : SCS
 */
void compute_ifft(vector<vector<complex<float>>> &inputGrid,
                  vector<complex<float>> &output,
                  const int * cp_lengths_,
                  const int * cum_sum_cp_lengths_,
                  const float &scaling_factor,
                  const int &fft_size,
                  const int &symbols_per_subframe,
                  const int &nb_subframes,
                  const int &scs) {

    //std::chrono::steady_clock::time_point t1{}, t2{};

    //t1 = std::chrono::steady_clock::now();

    /**
    fftw_complex * in_ifft  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_size);
    fftw_complex * out_ifft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_size);
    fftw_plan p_ifft   = fftw_plan_dft_1d(fft_size, in_ifft, out_ifft, FFTW_FORWARD, FFTW_MEASURE);
    */
    int offset = 0;

    for (int subframe = 0 ; subframe < nb_subframes ; subframe++) {
        offset = subframe*symbols_per_subframe;

        for (int symbol = 0; symbol < symbols_per_subframe; symbol++) {

            memset(in_ifft, 0, sizeof(fftwf_complex) * fft_size);   // reset fft array
            memset(out_ifft, 0, sizeof(fftwf_complex) * fft_size); // reset fft array

            // Prepare the input vector for the IFFT
            for (int j = 0; j < fft_size/2; j++) {
                in_ifft[j + fft_size/2][0] = scaling_factor*inputGrid[symbol+offset][j].real(); // real part
                in_ifft[j + fft_size/2][1] = scaling_factor*inputGrid[symbol+offset][j].imag(); // imaginary part

                in_ifft[j][0] = scaling_factor*inputGrid[symbol+offset][j + fft_size/2].real(); // real part
                in_ifft[j][1] = scaling_factor*inputGrid[symbol+offset][j + fft_size/2].imag(); // imaginary part
            }

            fftwf_execute(p_ifft); // Compute the IFFT

            //vector<complex<float>> temp(fft_size+cp_lengths_[symbol], 0);

            // Add cyclic prefix to the symbol
            addCyclicPrefix(output.data() + subframe * (cum_sum_cp_lengths_[symbols_per_subframe - 1] + cp_lengths_[symbols_per_subframe - 1] + fft_size)
            + cum_sum_cp_lengths_[symbol], out_ifft, cp_lengths_[symbol], fft_size);
        }
    }
    //t2 = std::chrono::steady_clock::now();
    //cout << "durée ifft : " << std::chrono::duration_cast<std::chrono::microseconds>(t2 - t1).count() << endl;

    /**
    fftw_destroy_plan(p_ifft);
    fftw_free(in_ifft);
    fftw_free(out_ifft);
    */
}

/** Modified from free5GRAN FFT function. Computes the FFT for the whole frame.
 *
 * @param[in] input                 : input Grid whose FFT is to be computed
 * @param[out] output               : output Grid containing I/Q samples for each subcarrier, computed after FFT
 *                                    Provide an initialize grid with the correct size as an argument.
 * @param nb_subframes              : number of subframes in the frame
 * @param fft_size                  : FFT size
 * @param symbols_per_sub_frame     : number of symbols per subframe
 * @param scs                       : scs
 */
void compute_fft(const complex<float> * input,
                vector<vector<complex<float>>> &output,
                const int &nb_subframes,
                const int &fft_size,
                const int &symbols_per_sub_frame,
                const int &scs,
                const int * cp_lengths_,
                const int * cum_sum_cp_lengths_) {

    int offset = 0;

    /**
    fftw_complex * in_fft  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_size);
    fftw_complex * out_fft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_size);
    fftw_plan p_fft   = fftw_plan_dft_1d(fft_size, in_fft, out_fft, FFTW_FORWARD, FFTW_MEASURE);
    */
    for(int subframe = 0; subframe < nb_subframes; subframe++) {

        // Number of symbols already transformed in the buffer
        offset = subframe*(cum_sum_cp_lengths_[symbols_per_sub_frame-1] + fft_size + cp_lengths_[symbols_per_sub_frame-1]);

        //memset(in_fft,  0,  sizeof(fftwf_complex) * fft_size); // reset fft array
        //memset(out_fft, 0, sizeof(fftwf_complex) * fft_size); // reset fft array

        for (int i = 0; i < symbols_per_sub_frame; i++) {

            // Prepare the input vector for the FFT
            // Remove the cyclic prefix
            //memcpy(in_fft, &input[cum_sum_cp_lengths_[i] + cp_lengths_[i] + offset], fft_size * sizeof(complex<float>));
            for (int j = 0; j < fft_size; j++) {
                in_fft[j][0] = input[j + cum_sum_cp_lengths_[i] + cp_lengths_[i] + offset].real(); // real part
                in_fft[j][1] = input[j + cum_sum_cp_lengths_[i] + cp_lengths_[i] + offset].imag(); // imaginary part
            }

            fftwf_execute(p_fft); // Compute the FFT

            // Prepare the output vector of the FFT
            //memcpy(&output[i + symbols_per_sub_frame*subframe][fft_size/2], out_fft, fft_size/2 * sizeof(complex<float>));
            //memcpy(&output[i + symbols_per_sub_frame*subframe], *(out_fft + fft_size/2), fft_size/2 * sizeof(complex<float>));
            for (int k=0; k < fft_size/2 ; k++) {
                output[i + symbols_per_sub_frame*subframe][k + fft_size/2] = complex<float>(out_fft[k][0], out_fft[k][1]);
                output[i + symbols_per_sub_frame*subframe][k] = complex<float>(out_fft[k + fft_size/2][0], out_fft[k + fft_size/2][1]);
            }
        }
    }
    /**
    fftw_destroy_plan(p_fft);
    fftw_free(in_fft);
    fftw_free(out_fft);
     */
}

/** free5GRAN function to compute FFT.
 *
 * @param[in] input : input signal
 * @param[out] output : output grid
 * @param[in] fft_size : FFT size
 * @param[in] num_symbols_ : number of symbols to be computed
 * @param[in] cp_lengths_ : CP lengths of each symbol
 * @param[in] cum_sum_cp_lengths_ : Cumulative sum of samples in each symbol.
 * @param[in] offset_ : offset from the first sample of input to be applied.
 */
void compute_fft(const vector<complex<float>> &input,
                 vector<vector<complex<float>>> &output,
                 const int &fft_size,
                 const int &num_symbols_,
                 const int * cp_lengths_,
                 const int * cum_sum_cp_lengths_,
                 const int &offset_) {

    /**
    fftw_complex * in_fft  = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_size);
    fftw_complex * out_fft = (fftw_complex*) fftw_malloc(sizeof(fftw_complex) * fft_size);
    fftw_plan p_fft   = fftw_plan_dft_1d(fft_size, in_fft, out_fft, FFTW_FORWARD, FFTW_MEASURE);
    */

    memset(in_fft,  0,  sizeof(fftwf_complex) * fft_size); // reset fft array
    memset(out_fft, 0, sizeof(fftwf_complex) * fft_size); // reset fft array

    for (int i = 0; i < num_symbols_; i++) {

        // Prepare the input vector for the FFT
        // Remove the cyclic prefix
        for (int j = 0; j < fft_size; j++) {
            in_fft[j][0] = input[j + cum_sum_cp_lengths_[i] + cp_lengths_[i] + offset_].real(); // real part
            in_fft[j][1] = input[j + cum_sum_cp_lengths_[i] + cp_lengths_[i] + offset_].imag(); // imaginary part
        }

        fftwf_execute(p_fft); // Compute the FFT

        // Prepare the output vector of the FFT
        for (int k=0; k < fft_size/2 ; k++) {
            output[i][k + fft_size/2] = complex<float>(out_fft[k][0], out_fft[k][1]);
            output[i][k] = complex<float>(out_fft[k + fft_size/2][0], out_fft[k + fft_size/2][1]);
        }
    }
    /**
    fftw_destroy_plan(p_fft);
    fftw_free(in_fft);
    fftw_free(out_fft);
     */
}

/** Adds the cyclic prefix to one time domain OFDM symbol
 * @param[out] temp : output vector
 * @param[in] input : fftw array containing the time domain samples of the signal
 * @param cp_length : cp length of the OFDM symbol
 * @param fft_size  : fft size
 */
void addCyclicPrefix(complex<float> * output_symbol_with_cp_,
                     fftwf_complex * input, const int &cp_length,
                     const int &fft_size) {

    // Get the index of the first sample of the cyclic prefix.
    int start_index_CP = fft_size - cp_length;

    // place the cyclic prefix at the beginning
    for (int i = 0; i < cp_length; i++) {
        output_symbol_with_cp_[i].real(input[start_index_CP + i][0]);
        output_symbol_with_cp_[i].imag(input[start_index_CP + i][1]);
    }

    // place the symbol in the remaining slots
    for (int i = 0; i < fft_size; i ++) {
        output_symbol_with_cp_[i+cp_length].real(input[i][0]);
        output_symbol_with_cp_[i+cp_length].imag(input[i][1]);
    }
}

/** free5GRAN function to generate DMRS sequences for PDSCH */
void generate_pdsch_dmrs_sequence(
        int n_symb_slot,
        int slot_number,
        int symbol_number,
        int n_scid,
        int n_id_scid,
        complex<float>* output_sequence,
        int size) {
/**
 * \fn generate_pdsch_dmrs_sequence
 * \brief Generic PDSCH DMRS sequence
 * \standard TS 38.211 V15.2.0 Section 7.4.1.1.1
 *
 * \param[in] n_symb_slot: Number of symbols per slot
 * \param[in] slot_number: Slot number within a frame
 * \param[in] symbol_number: Symbol number within a slot
 * \param[in] n_scid: DMRS sequence initialization field (0 by default)
 * \param[in] n_id_scid: Sambling ID (cell PCI by default)
 * \param[out] output_sequence: output sequence
 * \param[in] size: Sequence size
 */
    long c_init =
            (long)(pow(2, 17) * (n_symb_slot * slot_number + symbol_number + 1) *
                   (2 * n_id_scid + 1) +
                   2 * n_id_scid + n_scid) %
            (long)pow(2, 31);
    int seq[2 * size];
    generate_c_sequence(c_init, 2 * size, seq, 0);
    for (int m = 0; m < size; m++) {
        output_sequence[m] =
                (float)(1 / sqrt(2)) *
                complex<float>(1 - 2 * seq[2 * m], 1 - 2 * seq[2 * m + 1]);
    }
}

/** From free5GRAN. Used to generate the c sequence. */
void generate_c_sequence(
        long c_init,
        int length,
        int* output_sequence,
        int demod_type) {
        /**
         * \fn generate_c_sequence
         * \brief Generic pseudo-random sequence generator
         * \standard TS 38.211 V15.2.0 Section 5.2
         *
         * \param[in] c_init: Sequence initializer
         * \param[in] length: Sequence length
         * \param[out] output_sequence: output sequence
         * \param[in] demod_type: Demodulation type (0 -> Hard demodulation (to be
         * used by default) / 1 -> Soft demodulation)
         */
    int x1[1600 + length], x2[1600 + length], base_x2[32];

    for (int j = 0; j < 31; j++) {
        base_x2[j] = ((int)c_init / (int)pow(2, j)) % 2;
    }
    for (int n = 0; n < 1600 + length; n++) {
        if (n < 31) {
            x1[n] = DMRS_BASE_X1_SEQUENCE[n];
            x2[n] = base_x2[n];
        } else {
            x1[n] = (x1[n - 28] + x1[n - 31]) % 2;
            x2[n] = (x2[n - 28] + x2[n - 29] + x2[n - 30] + x2[n - 31]) % 2;
        }
    }
    for (int j = 0; j < length; j++) {
        output_sequence[j] = (x1[j + 1600] + x2[j + 1600]) % 2;
        if (demod_type == 1) {
            output_sequence[j] = (output_sequence[j] == 0) ? 1 : -1;
        }
    }
}

/** free5GRAN function to generate PSS sequences */
void generate_pss_sequence(int n_id_2, int* output_sequence) {
/**
 * \fn generate_pss_sequence
 * \brief Generate PSS sequence
 * \standard TS 38.211 V15.2.0 Section 7.4.2.2.1
 *
 * \param[in] n_id_2: N_ID_2 for which PSS is generated
 * \param[out] output_sequence: output sequence
 */
    int x_seq[SIZE_PSS_SSS_SIGNAL];
    for (int i = 0; i < SIZE_PSS_SSS_SIGNAL; i++) {
        if (i < 7) {
            x_seq[i] = PSS_BASE_SEQUENCE[i];
        } else {
            x_seq[i] = (x_seq[i - 3] + x_seq[i - 7]) % 2;
        }
    }
    for (int n = 0; n < SIZE_PSS_SSS_SIGNAL; n++) {
        int m = (n + 43 * n_id_2) % SIZE_PSS_SSS_SIGNAL;
        output_sequence[n] = 1 - 2 * x_seq[m];
    }
}

/** Computes the time domain PSS sequence to be fed to the search_pss function
 * @param time_signals_pss_ : the time domain PSS sequence
 * @param fft_size : the FFT size of the frequency domain signal
 * @param cp_length : CP length of the PSS symbol
 * @param is_downsampled_ : True if signal is downsampled by the downsampling factor, False otherwise
 */
void compute_time_domain_pss(vector<vector<complex<float>>> &time_signals_pss_,
                             const int& fft_size,
                             const int& cp_length,
                             bool is_downsampled_) {

    fftwf_complex * pss_in_local;
    fftwf_complex * pss_out_local;

    if(is_downsampled_) {
        pss_in_local = pss_in_0_downsampled;
        pss_out_local = pss_out_0_downsampled;
    } else {
        pss_in_local = pss_in_0_static;
        pss_out_local = pss_out_0_static;
    }

    vector<int> pss_seq(SIZE_PSS_SSS_SIGNAL);
    generate_pss_sequence(0, pss_seq.data());

    /*
     * Generate frequency domain signal (PSS is BPSK modulated, real part is the
     * pss sequence value and imaginary part is 0)
     */
    for (int i = 0; i < fft_size / 2; i++) {
        if (i < 63) {
            pss_in_local[i][0] = pss_seq[i + 64];
            pss_in_local[i][1] = 0;
        }
        if (i < 64) {
            pss_in_local[fft_size - i - 1][0] = pss_seq[64 - i - 1];
            pss_in_local[fft_size - i - 1][1] = 0;
        }
    }

    /*
     * Execute the IFFT
     */
    if(is_downsampled_) {
        fftwf_execute(ifft_plan_0_downsampled);
    } else {
        fftwf_execute(ifft_plan_0_static);
    }

    /*
     * Transform fftw complex signals into vectors of complex values and add
     * cyclic prefix
     */
    for (int i = 0; i < cp_length + fft_size; i++) {
        if (i < cp_length) {
            time_signals_pss_[0][i] =
                    complex<float>(pss_out_local[fft_size - cp_length + i][0],
                                   pss_out_local[fft_size - cp_length + i][1]);
        } else {
            time_signals_pss_[0][i] = complex<float>(pss_out_local[i - cp_length][0],
                                                     pss_out_local[i - cp_length][1]);
        }
    }

    /// Repeat for n_id_2 = 1
    generate_pss_sequence(1, pss_seq.data());

    /*
     * Generate frequency domain signal (PSS is BPSK modulated, real part is the
     * pss sequence value and imaginary part is 0)
     */
    for (int i = 0; i < fft_size / 2; i++) {
        if (i < 63) {
            pss_in_local[i][0] = pss_seq[i + 64];
            pss_in_local[i][1] = 0;
        }
        if (i < 64) {
            pss_in_local[fft_size - i - 1][0] = pss_seq[64 - i - 1];
            pss_in_local[fft_size - i - 1][1] = 0;
        }
    }

    /*
     * Execute the IFFT
     */
    if(is_downsampled_) {
        fftwf_execute(ifft_plan_0_downsampled);
    } else {
        fftwf_execute(ifft_plan_0_static);
    }

    /*
     * Transform fftw complex signals into vectors of complex values and add
     * cyclic prefix
     */
    for (int i = 0; i < cp_length + fft_size; i++) {
        if (i < cp_length) {
            time_signals_pss_[1][i] =
                    complex<float>(pss_out_local[fft_size - cp_length + i][0],
                                   pss_out_local[fft_size - cp_length + i][1]);
        } else {
            time_signals_pss_[1][i] = complex<float>(pss_out_local[i - cp_length][0],
                                                     pss_out_local[i - cp_length][1]);
        }
    }

    /// Repeat for n_id_2 = 2
    generate_pss_sequence(2, pss_seq.data());

    /*
     * Generate frequency domain signal (PSS is BPSK modulated, real part is the
     * pss sequence value and imaginary part is 0)
     */
    for (int i = 0; i < fft_size / 2; i++) {
        if (i < 63) {
            pss_in_local[i][0] = pss_seq[i + 64];
            pss_in_local[i][1] = 0;
        }
        if (i < 64) {
            pss_in_local[fft_size - i - 1][0] = pss_seq[64 - i - 1];
            pss_in_local[fft_size - i - 1][1] = 0;
        }
    }

    /*
     * Execute the IFFT
     */
    if(is_downsampled_) {
        fftwf_execute(ifft_plan_0_downsampled);
    } else {
        fftwf_execute(ifft_plan_0_static);
    }

    /*
     * Transform fftw complex signals into vectors of complex values and add
     * cyclic prefix
     */
    for (int i = 0; i < cp_length + fft_size; i++) {
        if (i < cp_length) {
            time_signals_pss_[2][i] =
                    complex<float>(pss_out_local[fft_size - cp_length + i][0],
                                   pss_out_local[fft_size - cp_length + i][1]);
        } else {
            time_signals_pss_[2][i] = complex<float>(pss_out_local[i - cp_length][0],
                                                     pss_out_local[i - cp_length][1]);
        }
    }
}

/** free5GRAN function to search the PSS sequence */
void search_pss(int& n_id_2,
                int& synchronisation_index,
                float& peak_value,
                const int &cp_length,
                vector<complex<float>>::const_iterator buff,
                size_t num_samples_,
                const int &fft_size,
                bool known_n_id_2_,
                vector<vector<complex<float>>> &time_signals_pss_) {
                /**
                 * \fn search_pss
                 * \brief Search for PSS correlation peak inside a signal.
                 * \details
                 * Details:
                 * - Generating the three possible PSS sequences (for N_ID_2 in [0,1,2])
                 * - Performing iFFT to retreive Time domain PSS sequence
                 * - Cross-correlation between received signal and three time domain PSS
                 * signals
                 * - Return PSS sequence with highest correlation peak.
                 *
                 * \param[out] n_id_2: Returned N_ID_2
                 * \param[out] synchronisation_index: Synchronization index corresponding to
                 * correlation peak \param[out] peak_value: Peak height \param[in] cp_length:
                 * Cyclic prefix length \param[in] buff: Input IQ signal \param[in] fft_size:
                 * FFT size and symbol size
                 */

    size_t num_samples = num_samples_;

    if(known_n_id_2_) {
        vector<complex<float>> corr(num_samples + fft_size + cp_length - 1);
#if defined(__AVX2__) and defined(AVX2_PROCESSING)
        /* Correlate with the signal corresponding to the predetermined N_ID_2 */
        cross_correlation_avx2(buff, time_signals_pss_[n_id_2].begin(), corr.begin(), num_samples,
                          fft_size + cp_length);
#else
        /* Correlate with the signal corresponding to the predetermined N_ID_2 */
        cross_correlation(buff, time_signals_pss_[n_id_2].begin(), corr.begin(), num_samples,
                          fft_size + cp_length);
#endif
        float max_value = -1;
        synchronisation_index = -1;

        /// Compute square norms instead of abs() to avoid calculating a square root
        /*
        vector<float> squared_magnitudes(corr.size());
        for(int i = 0; i < squared_magnitudes.size(); i++) {
            squared_magnitudes[i] = corr[i].real() * corr[i].real() + corr[i].imag() * corr[i].imag();
        }

        auto peak_val_it = std::max_element(squared_magnitudes.begin(), squared_magnitudes.end());
        peak_value = *(peak_val_it);
        synchronisation_index = std::distance(squared_magnitudes.begin(), peak_val_it); */

        // Search for the max value and index over the cross correlation
        float abs0 = 0;
        for (int i = 0; i < num_samples + fft_size + cp_length - 1; i++) {
            abs0 = abs(corr[i]); ////// Alsmost no penalty in time for using abs instead of squared norm
            if (abs0 > max_value) {
                max_value = abs0;
                synchronisation_index = i;
            }
        }
        peak_value = max_value;
        //cout << "synchronisation index sequential : " << synchronisation_index << endl;


        /// Compute square norms instead of abs() to avoid calculating a square root
        /*
        vector<float> squared_magnitudes(corr.size());
        for(int i = 0; i < squared_magnitudes.size(); i++) {
            squared_magnitudes[i] = corr[i].real() * corr[i].real() + corr[i].imag() * corr[i].imag();
        }

        auto peak_val_it = std::max_element(squared_magnitudes.begin(), squared_magnitudes.end());
        peak_value = *(peak_val_it);
        synchronisation_index = std::distance(squared_magnitudes.begin(), peak_val_it); */

        /*
        cout << "cross correlation values : " << endl;
        for(int i = 0; i < corr.size(); i++) {
            cout << "corr : " << corr2[i] << endl;
            cout << "corr AVX2 : " << corr[i] << endl;
        } */

        //cout << "num samples : " << num_samples << endl;
        //cout << "fft_size + cp_length : " << fft_size + cp_length << endl;

    } else { /// Correlate with all the possible n_id_2

        vector<vector<complex<float>>> corr(NUM_N_ID_2, vector<complex<float>>(num_samples + fft_size + cp_length - 1));

#if defined(__AVX2__) and defined(AVX2_PROCESSING)
        for(int i = 0; i < NUM_N_ID_2; i++) {
            cross_correlation_avx2(buff, time_signals_pss_[n_id_2].begin(), corr[i].begin(), num_samples,
                              fft_size + cp_length);
        }
#else
        for(int i = 0; i < NUM_N_ID_2; i++) {
            cross_correlation(buff, time_signals_pss_[n_id_2].begin(), corr[i].begin(), num_samples,
                              fft_size + cp_length);
        }
#endif

        float abs0, abs1, abs2, max_value;

        // Search for the max value and index over the cross correlation
        for (int i = 0; i < num_samples + fft_size + cp_length - 1; i++) {
            abs0 = corr[0][i].real() * corr[0][i].real() + corr[0][i].imag() * corr[0][i].imag();
            abs1 = corr[1][i].real() * corr[1][i].real() + corr[1][i].imag() * corr[1][i].imag();
            abs2 = corr[2][i].real() * corr[2][i].real() + corr[2][i].imag() * corr[2][i].imag();

            if (abs0 > max_value) {
                max_value = abs0;
                n_id_2 = 0;
                synchronisation_index = i;
            } else if (abs1 > max_value) {
                max_value = abs1;
                n_id_2 = 1;
                synchronisation_index = i;
            } else if (abs2 > max_value) {
                max_value = abs2;
                n_id_2 = 2;
                synchronisation_index = i;
            }
        }
        peak_value = max_value;
    }
}

/** free5GRAN function for PSS sequence cross-correlation */
void cross_correlation(
        vector<complex<float>>::const_iterator in1,
        vector<complex<float>>::const_iterator in2,
        vector<complex<float>>::iterator out,
        int size1,
        int size2) {
    /**
     * \fn cross_correlation
     * \brief Perform cross correlation (i.e. moving window correlation) between
     * signal 1 and signal 2 \param[in] in1: Signal 1 \param[in] in2: Signal 2
     * \param[out] out: Correlation result
     * \param[in] size1: Signal 1 size
     * \param[in] size2: Signal 2 size
     */
    //int common = 0;
    //int base_id1, base_id2;

    int common = 0;
    int baseId1 = 0, baseId2 = 0;
    /*
    for (int m = 0; m < size1 + size2 - 1; m++)
    {
        if (m < size2)
        {
            common++;
            baseId1 = 0;
            baseId2 = size2 - common;
        }
        else if (m > size1 - 1)
        {
            common--;
            baseId1 = size1 - common;
            baseId2 = 0;
        }
        else
        {
            baseId1 = m + 1 - size2;
            baseId2 = 0;
        }
        out[m] = 0;
        for (int n = 0; n < common; n++)
        {
            out[m] += in1[baseId1 + n] * conj(in2[baseId2 + n]);
        }
    } */

    for (int m = 0; m < size2; m++) {
        common++;
        baseId2 = size2 - common;
        //base_id2 = size2 - m + 1;
        out[m] = 0;
        for (int n = 0; n < common; n ++) {
            out[m] += in1[n] * conj(in2[baseId2 + n]);
        }
    }
    baseId2 = 0;
    for(int m = size2; m < size1; m++) {
        baseId1 = m + 1 - size2;
        out[m] = 0;
        for (int n = 0; n < size2; n ++) {
            out[m] += in1[baseId1 + n] * conj(in2[n]);
        }
    }
    baseId1 = size1 - common;
    for(int m = size1; m < size1 + size2 - 1; m++) {
        common--;
        baseId1--;
        //baseId1 = size1 - common;
        out[m] = 0;
        for (int n = 0; n < common; n ++) {
            out[m] += in1[baseId1 + n] * conj(in2[n]);
        }
    }
}

/** Cross-correlation implemented with AVX2 optimizations.
 *
 * @param in1 : Input signal to be correlated with
 * @param in2 : Sequence used for correlation
 * @param out : output correlation
 * @param size1 : number of elements in in1
 * @param size2 : number of elements in in2
 */
#if defined(__AVX2__)
void cross_correlation_avx2(
        vector<complex<float>>::const_iterator in1,
        vector<complex<float>>::const_iterator in2,
        vector<complex<float>>::iterator out,
        int size1,
        int size2) {

    __m256 out_vec_real,
           out_vec_imag,
           in1_vec, in2_vec;
    __m256 vec1, vec2;

    __m128 vec1_128, vec2_128;

    __m256 conj_vec = {1, -1, 1, -1, 1, -1, 1, -1};

    __m256i masks[3] = {
            _mm256_setr_epi32(-1, -1, 0, 0, 0, 0, 0, 0),
            _mm256_setr_epi32(-1, -1, -1, -1, 0, 0, 0, 0),
            _mm256_setr_epi32(-1, -1, -1, -1, -1, -1, 0, 0),
    };
    /*
    __m256 masks[3] = {
            {1, 1, 0, 0, 0, 0, 0, 0},
            {1, 1, 1, 1, 0, 0, 0, 0},
            {1, 1, 1, 1, 1, 1, 0, 0},
    }; */

    int common = 0;
    int baseId1 = 0, baseId2 = 0;

    for(int m = 0; m < 4; m++) {
        common ++;
        baseId2 = size2 - common;
        out[m] = 0;
        for (int n = 0; n < common; n++) {
            out[m] += in1[n] * conj(in2[baseId2 + n]);
        }
    }

    int modulo = 0;
    for(int m = 4; m < size2; m++) {
        common ++;
        baseId2 = size2 - common;
        out[m] = 0;
        out_vec_real = _mm256_set1_ps(0);
        out_vec_imag = _mm256_set1_ps(0);
        modulo = common % 4;
        for(int n = 0; n < common - modulo; n+=4) {
            in1_vec = _mm256_loadu_ps((float *) &in1[n]);
            in2_vec = _mm256_loadu_ps((float *) &in2[baseId2 + n]);
            vec1 = _mm256_mul_ps(in1_vec, in2_vec);
            vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(in2_vec, conj_vec), 0b10110001), in1_vec);
            out_vec_real = _mm256_add_ps(out_vec_real, vec1);
            out_vec_imag = _mm256_add_ps(out_vec_imag, vec2);
        }
        if(modulo > 0) {
            //cout << "modulo common : " << modulo << endl;

            in1_vec = _mm256_maskload_ps((float *) &in1[common - modulo], masks[modulo - 1]);
            in2_vec = _mm256_maskload_ps((float *) &in2[baseId2 + common - modulo], masks[modulo - 1]);
            vec1 = _mm256_mul_ps(in1_vec, in2_vec);
            vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(in2_vec, conj_vec), 0b10110001), in1_vec);

            //vec1 = _mm256_mul_ps(vec1, masks[modulo - 1]);
            //vec2 = _mm256_mul_ps(vec2, masks[modulo - 1]);

            /*
            cout << "vec1 masked : ";
            for(int i = 0; i < 8; i+=2) {
                cout << "(" << vec1[i] << "," << vec1[i + 1] << ")";
            } cout << endl;

            cout << "vec2 masked : ";
            for(int i = 0; i < 8; i+=2) {
                cout << "(" << vec2[i] << "," << vec2[i + 1] << ")";
            } cout << endl; */

            out_vec_real = _mm256_add_ps(out_vec_real, vec1);
            out_vec_imag = _mm256_add_ps(out_vec_imag, vec2);
        }
        vec1 = _mm256_hadd_ps(out_vec_real, out_vec_real);
        vec1_128 = _mm256_castps256_ps128(vec1);
        vec2_128 = _mm256_extractf128_ps(vec1, 1);
        out[m].real(_mm_cvtss_f32(_mm_add_ps(_mm_add_ps(vec1_128, _mm_permute_ps(vec1_128, 0b10110001)),
                                             _mm_add_ps(vec2_128, _mm_permute_ps(vec2_128, 0b10110001)))));

        vec1 = _mm256_hadd_ps(out_vec_imag, out_vec_imag);
        vec1_128 = _mm256_castps256_ps128(vec1);
        vec2_128 = _mm256_extractf128_ps(vec1, 1);
        out[m].imag(_mm_cvtss_f32(_mm_add_ps(_mm_add_ps(vec1_128, _mm_permute_ps(vec1_128, 0b10110001)),
                                             _mm_add_ps(vec2_128, _mm_permute_ps(vec2_128, 0b10110001)))));
    }

    baseId2 = 0;
    modulo = size2 % 4;
    if(modulo > 0) {
        for(int m = size2; m < size1; m++) {
            baseId1 = m + 1 - size2;
            out[m] = 0;
            out_vec_real = _mm256_set1_ps(0);
            out_vec_imag = _mm256_set1_ps(0);
            for (int n = 0; n < size2 - modulo; n+=4) {
                in1_vec = _mm256_loadu_ps((float *) &in1[baseId1 + n]);
                in2_vec = _mm256_loadu_ps((float *) &in2[n]);
                vec1 = _mm256_mul_ps(in1_vec, in2_vec);
                vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(in2_vec, conj_vec), 0b10110001), in1_vec);
                out_vec_real = _mm256_add_ps(out_vec_real, vec1);
                out_vec_imag = _mm256_add_ps(out_vec_imag, vec2);
            }
            in1_vec = _mm256_maskload_ps((float *) &in1[baseId1 + size2 - modulo], masks[modulo - 1]);
            in2_vec = _mm256_maskload_ps((float *) &in2[size2 - modulo], masks[modulo - 1]);
            vec1 = _mm256_mul_ps(in1_vec, in2_vec);
            vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(in2_vec, conj_vec), 0b10110001), in1_vec);
            out_vec_real = _mm256_add_ps(out_vec_real, vec1);//_mm256_mul_ps(vec1, masks[modulo - 1]));
            out_vec_imag = _mm256_add_ps(out_vec_imag, vec2);//_mm256_mul_ps(vec2, masks[modulo - 1]));

            vec1 = _mm256_hadd_ps(out_vec_real, out_vec_real);
            vec1_128 = _mm256_castps256_ps128(vec1);
            vec2_128 = _mm256_extractf128_ps(vec1, 1);
            out[m].real(_mm_cvtss_f32(_mm_add_ps(_mm_add_ps(vec1_128, _mm_permute_ps(vec1_128, 0b10110001)),
                                                 _mm_add_ps(vec2_128, _mm_permute_ps(vec2_128, 0b10110001)))));

            vec1 = _mm256_hadd_ps(out_vec_imag, out_vec_imag);
            vec1_128 = _mm256_castps256_ps128(vec1);
            vec2_128 = _mm256_extractf128_ps(vec1, 1);
            out[m].imag(_mm_cvtss_f32(_mm_add_ps(_mm_add_ps(vec1_128, _mm_permute_ps(vec1_128, 0b10110001)),
                                                 _mm_add_ps(vec2_128, _mm_permute_ps(vec2_128, 0b10110001)))));
        }
    } else {
        for(int m = size2; m < size1; m++) {
            baseId1 = m + 1 - size2;
            out[m] = 0;
            out_vec_real = _mm256_set1_ps(0);
            out_vec_imag = _mm256_set1_ps(0);
            for (int n = 0; n < size2; n+=4) {
                in1_vec = _mm256_loadu_ps((float *) &in1[baseId1 + n]);
                in2_vec = _mm256_loadu_ps((float *) &in2[n]);
                vec1 = _mm256_mul_ps(in1_vec, in2_vec);
                vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(in2_vec, conj_vec), 0b10110001), in1_vec);
                out_vec_real = _mm256_add_ps(out_vec_real, vec1);
                out_vec_imag = _mm256_add_ps(out_vec_imag, vec2);
            }
            vec1 = _mm256_permute_ps(out_vec_real, 0b11011000);
            vec1 = _mm256_hadd_ps(vec1, vec1);
            vec1_128 = _mm256_castps256_ps128(vec1);
            vec2_128 = _mm256_extractf128_ps(vec1, 1);
            out[m].real(_mm_cvtss_f32(_mm_add_ps(_mm_add_ps(vec1_128, _mm_permute_ps(vec1_128, 0b10110001)),
                                                 _mm_add_ps(vec2_128, _mm_permute_ps(vec2_128, 0b10110001)))));

            vec1 = _mm256_permute_ps(out_vec_imag, 0b11011000);
            vec1 = _mm256_hadd_ps(vec1, vec1);
            vec1_128 = _mm256_castps256_ps128(vec1);
            vec2_128 = _mm256_extractf128_ps(vec1, 1);
            out[m].imag(_mm_cvtss_f32(_mm_add_ps(_mm_add_ps(vec1_128, _mm_permute_ps(vec1_128, 0b10110001)),
                                                 _mm_add_ps(vec2_128, _mm_permute_ps(vec2_128, 0b10110001)))));
        }
    }

    //baseId1 = size1 - common;
    for(int m = size1; m < size1 + size2 - 1 - 4; m++) {
        common--;
        //baseId1 = size1 - common;
        baseId1--;
        out[m] = 0;
        out_vec_real = _mm256_set1_ps(0);
        out_vec_imag = _mm256_set1_ps(0);
        modulo = common % 4;
        for (int n = 0; n < common - modulo; n+=4) {
            in1_vec = _mm256_loadu_ps((float *) &in1[baseId1 + n]);
            in2_vec = _mm256_loadu_ps((float *) &in2[n]);
            vec1 = _mm256_mul_ps(in1_vec, in2_vec);
            vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(in2_vec, conj_vec), 0b10110001), in1_vec);
            out_vec_real = _mm256_add_ps(out_vec_real, vec1);
            out_vec_imag = _mm256_add_ps(out_vec_imag, vec2);
        }
        if(modulo > 0) {
            in1_vec = _mm256_maskload_ps((float *) &in1[baseId1 + common - modulo], masks[modulo - 1]);
            in2_vec = _mm256_maskload_ps((float *) &in2[common - modulo], masks[modulo - 1]);
            vec1 = _mm256_mul_ps(in1_vec, in2_vec);
            vec2 = _mm256_mul_ps(_mm256_permute_ps(_mm256_mul_ps(in2_vec, conj_vec), 0b10110001), in1_vec);
            out_vec_real = _mm256_add_ps(out_vec_real, vec1);//_mm256_mul_ps(vec1, masks[modulo - 1]));
            out_vec_imag = _mm256_add_ps(out_vec_imag, vec2);//_mm256_mul_ps(vec2, masks[modulo - 1]));
        }
        vec1 = _mm256_permute_ps(out_vec_real, 0b11011000);
        vec1 = _mm256_hadd_ps(vec1, vec1);
        vec1_128 = _mm256_castps256_ps128(vec1);
        vec2_128 = _mm256_extractf128_ps(vec1, 1);
        out[m].real(_mm_cvtss_f32(_mm_add_ps(_mm_add_ps(vec1_128, _mm_permute_ps(vec1_128, 0b10110001)),
                                             _mm_add_ps(vec2_128, _mm_permute_ps(vec2_128, 0b10110001)))));

        vec1 = _mm256_permute_ps(out_vec_imag, 0b11011000);
        vec1 = _mm256_hadd_ps(vec1, vec1);
        vec1_128 = _mm256_castps256_ps128(vec1);
        vec2_128 = _mm256_extractf128_ps(vec1, 1);
        out[m].imag(_mm_cvtss_f32(_mm_add_ps(_mm_add_ps(vec1_128, _mm_permute_ps(vec1_128, 0b10110001)),
                                             _mm_add_ps(vec2_128, _mm_permute_ps(vec2_128, 0b10110001)))));
    }

    for(int m = size1 + size2 - 1 - 4; m < size1 + size2 - 1; m++) {
        common--;
        //baseId1 = size1 - common;
        baseId1--;
        out[m] = 0;
        for (int n = 0; n < common; n++) {
            out[m] += in1[baseId1 + n] * conj(in2[n]);
        }
    }
}
#endif

/** free5GRAN function to generate SSS sequence
 *
 * @param[in] nId1 : N_ID_1 identifier
 * @param[in] nId2 : N_ID_2 identifier
 * @param[out] outputSequence : output sequence
 */
void generateSssSequence(int nId1, int nId2, vector<int> &outputSequence)
{
    int m0, m1;
    int x0Seq[SIZE_PSS_SSS_SIGNAL];
    int x1Seq[SIZE_PSS_SSS_SIGNAL];

    for (int i = 0; i < SIZE_PSS_SSS_SIGNAL; i++)
    {
        if (i < 7)
        {
            x0Seq[i] = SSS_BASE_X0_SEQUENCE[i];
            x1Seq[i] = SSS_BASE_X1_SEQUENCE[i];
        }
        else
        {
            x0Seq[i] = (x0Seq[i - 3] + x0Seq[i - 7]) % 2;
            x1Seq[i] = (x1Seq[i - 6] + x1Seq[i - 7]) % 2;
        }
    }
    m0 = 15 * (nId1 / 112) + 5 * nId2;
    m1 = nId1 % 112;
    for (int n = 0; n < SIZE_PSS_SSS_SIGNAL; n++)
    {
        outputSequence[n] = (1 - 2 * x0Seq[(n + m0) % SIZE_PSS_SSS_SIGNAL]) *
                (1 - 2 * x1Seq[(n + m1) % SIZE_PSS_SSS_SIGNAL]);
    }
}


/** free5GRAN function to compute the CP lengths of each symbol
 *
 * @param scs                     : scs
 * @param nfft                    : fft size
 * @param is_extended_cp          : set to "True" to use extended CP, to "False" otherwise
 * @param num_symb_per_subframes  : number of symbols per subframe (to compute the IFFT on the entire frame
 * @param cp_lengths              : computed CP lengths within a subframe
 * @param cum_sum_cp_lengths      : Number of samples for each symbol within a subframe
 */
void compute_cp_lengths(
        int scs,
        int nfft,
        int is_extended_cp,
        int num_symb_per_subframes,
        int * cp_lengths,
        int * cum_sum_cp_lengths)
{

    int nom_cp = ( ( is_extended_cp ) ? 512 : 144 );
    int base_cp = nom_cp * nfft / 2048;
    cum_sum_cp_lengths[ 0 ] = 0;
    for ( int i = 0; i < num_symb_per_subframes; i++ )
    {
        if ( i % ( num_symb_per_subframes / 2 ) == 0 )
        {
            cp_lengths[i] = ( scs * nfft - num_symb_per_subframes * nfft - ( num_symb_per_subframes - 2 ) * base_cp ) / 2;
        }
        else
        {
            cp_lengths[ i ] = base_cp;
        }
        if ( i < num_symb_per_subframes - 1 )
        {
            cum_sum_cp_lengths[ i + 1 ] = cum_sum_cp_lengths[ i ] + cp_lengths[ i ] + nfft;
        }
    }
}

/** free5GRAN function to correct the frequency offset
 *
 * @param inputSignal : intput signal to be corrected
 * @param freqOffset : frequency offset value determined previously
 * @param sampleRate : sample rate
 */
void transposeSignal( vector<complex<float>>* inputSignal, float freqOffset, double sampleRate )
{
    complex<float> j(0, 1);
    for (int i = 0; i < inputSignal->size(); i++)
    {
        (*inputSignal)[i] = (*inputSignal)[i] * exp(complex<float>(-2, 0) * j * complex<float>(M_PI, 0) * freqOffset *
                                                    complex<float>((float)i, 0) / complex<float>(sampleRate, 0));
    }
}

/** free5GRAN function to compute the frequency offset.
 *
 * @param[in] inputSignal : input signal
 * @param[in] symbolDuration : symbol duration in samples excluding cp length
 * @param[in] fftSize : FFT size
 * @param[in] cpLength : CP length of the symbol
 * @param[in] scs : Subcarrier spacing
 * @param[out] output : output frequency offset value
 * @param[in] numSymbols : number of symbols to be computed.
 */
void computeFineFrequencyOffset( complex<float> * inputSignal,
                                 int symbolDuration, int fftSize, int cpLength,
                                 int scs, float& output, int numSymbols )
{
    complex<float> out;
    float phaseOffset = 0;

    /** Looping over all the symbols */
    for (int symbol = 0; symbol < numSymbols; symbol++)
    {
        /** Initialize symbol phase offset to 0 */
        out = 0;
        /** Loop over all the samples of the cyclic prefGridDemapperix */
        for (int i = 0; i < cpLength; i++)
        {
            /** Increment symbol phase offset by the result of the correlation of the studied sample of
             * the cyclic prefGridDemapperix with the corresponding sample of the input signal
             */
            out += conj(inputSignal[i + symbol * symbolDuration]) *
                   inputSignal[i + symbol * symbolDuration + fftSize];
        }
        phaseOffset += arg(out);
    }
    /** Average phase offset over all the symbols */
    phaseOffset /= numSymbols;
    /** Computing frequency offset (output) corresponding
     * to the computed phase offset
     */
    output = scs * phaseOffset / (2 * M_PI);
}
