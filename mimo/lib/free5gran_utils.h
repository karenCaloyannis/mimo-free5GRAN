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

   This file contains all functions reused or modified from the free5GRAN library.
*/

#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <vector>
#include <complex>
#include <fftw3.h>
#include <fstream>
#include <string.h>
#include <type_traits>
#include <chrono>

#if defined(__AVX2__)
#include <x86intrin.h>
#include <immintrin.h>
#endif

#include "variables/variables.h"

/**
 * This file contains the free5GRAN functions, modified or not, used in the code
 *
 */

/** Initialize the static FFT plans
 *
 * @param fft_size : FFT size
 */
void init_fft_plans(const int &fft_size);

/** Initialize the static FFT plans used for PSS synchronization
 *
 * @param fft_size_ : FFT size
 * @param downsampling_factor_ : the downsampling factor typically set to fft_size_/PSS_SEQUENCE_SIZE
 */
void init_sync_pss_plans(int fft_size_, int downsampling_factor_);

/** Free the static FFT plans used for PSS synchronization */
void free_sync_pss_plans();

/** Initialize the static IFFT plans
 *
 * @param fft_size : FFT size
 */
void init_ifft_plans(const int &fft_size);

/** Free the static FFT plans */
void destroy_fft_plans();

/** Free the static ifft plans */
void destroy_ifft_plans();

/** free5GRAN function to search the PSS sequence */
void search_pss(int& n_id_2,
                int& synchronisation_index,
                float& peak_value,
                const int &cp_length,
                std::vector<std::complex<float>>::const_iterator buff,
                size_t num_samples_,
                const int &fft_size,
                bool known_n_id_2_,
                std::vector<std::vector<std::complex<float>>> &time_signals_pss_);
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
void exportGrid(std::string filename_,
                std::vector<std::complex<float> *> &signal,
                const int &num_symbols_,
                const int &num_sc_,
                const int &num_tx_ports_,
                const int &transmitter,
                const int &receiver);

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
void exportGrid(std::string filename_,
                std::vector<std::vector<std::complex<float>>> &signal,
                const int &num_symbols_,
                const int &num_sc_,
                const int &num_tx_ports_,
                const int &transmitter,
                const int &receiver);

/** Exports a one dimension vector
 *
 * @param filename_      : output stream
 * @param signal : signal to be exported
 */
void export1dVector(std::string filename, std::vector<std::complex<float>> &signal);

/** Exports a one dimension vector of ints
 *
 * @param filename_      : output stream
 * @param signal : vector to be exported
 */
void export1dVector(std::string filename_, std::vector<int> &signal);

/** Exports a one dimension array of complex<float>
 *
 * @param filename_      : output stream
 * @param signal : array to be exported
 * @param num_elements : number of elements in the array
 */
void export1dVector(std::string filename_, std::complex<float> * signal, int &num_elements);

/** Exports a one dimension array of ints
 *
 * @param filename_      : output stream
 * @param signal : array to be exported
 * @param num_elements : number of elements in the  array
 */
void export1dVector(std::string filename_, int * signal, int &num_elements);

/** Used to export the frequency domain grids
 *
 * @param filename : output stream
 * @param signal : grid to be exported
 */
void exportGrid(std::string filename, std::vector<std::vector<std::complex<float>>> &signal);

/** Computes the time domain PSS sequence to be fed to the search_pss function
 * @param time_signals_pss_ : the time domain PSS sequence
 * @param fft_size : the FFT size of the frequency domain signal
 * @param cp_length : CP length of the PSS symbol
 * @param is_downsampled_ : True if signal is downsampled by the downsampling factor, False otherwise
 */
void compute_time_domain_pss(std::vector<std::vector<std::complex<float>>> &time_signals_pss_,
                             const int& fft_size,
                             const int& cp_length,
                             bool is_downsampled_);

/** Adds the cyclic prefix to one time domain OFDM symbol
 * @param[out] temp : output vector
 * @param[in] input : fftw array containing the time domain samples of the signal
 * @param cp_length : cp length of the OFDM symbol
 * @param fft_size  : fft size
 */
void addCyclicPrefix(std::complex<float> * output_symbol_with_cp_,
                     fftwf_complex * input,
                     const int &cp_length,
                     const int &fft_size);

/** From free5GRAN. Used to generate the c sequence. */
void generate_c_sequence(long c_init,
                         int length,
                         int* output_sequence,
                         int demod_type);
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

/** free5GRAN function to generate DMRS sequences for PDSCH */
void generate_pdsch_dmrs_sequence(int n_symb_slot,
                                  int slot_number,
                                  int symbol_number,
                                  int n_scid,
                                  int n_id_scid,
                                  std::complex<float>* output_sequence,
                                  int size);
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

/** free5GRAN function to generate PSS sequences */
void generate_pss_sequence(int n_id_2, int* output_sequence);
/**
 * \fn generate_pss_sequence
 * \brief Generate PSS sequence
 * \standard TS 38.211 V15.2.0 Section 7.4.2.2.1
 *
 * \param[in] n_id_2: N_ID_2 for which PSS is generated
 * \param[out] output_sequence: output sequence
 */

/** free5GRAN function to generate SSS sequence
 *
 * @param[in] nId1 : N_ID_1 identifier
 * @param[in] nId2 : N_ID_2 identifier
 * @param[out] outputSequence : output sequence
 */
void generateSssSequence(int nId1, int nId2, std::vector<int> &outputSequence);

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
        int * cum_sum_cp_lengths );


/** free5GRAN function for PSS sequence cross-correlation */
void cross_correlation(
        std::vector<std::complex<float>>::const_iterator in1,
        std::vector<std::complex<float>>::const_iterator in2,
        std::vector<std::complex<float>>::iterator out,
        int size1,
        int size2);
/**
 * \fn cross_correlation
 * \brief Perform cross correlation (i.e. moving window correlation) between
 * signal 1 and signal 2 \param[in] in1: Signal 1 \param[in] in2: Signal 2
 * \param[out] out: Correlation result
 * \param[in] size1: Signal 1 size
 * \param[in] size2: Signal 2 size
 */

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
        std::vector<std::complex<float>>::const_iterator in1,
        std::vector<std::complex<float>>::const_iterator in2,
        std::vector<std::complex<float>>::iterator out,
        int size1,
        int size2);
#endif

/** Modified from free5GRAN IFFT function. Used to compute the IFFT for the whole frame.
 *
 * @param[in] inputGrid             : Grid whose IFFT is to be computed
 * @param[out] output                    : grid containing samples after IFFT. Provide an empty grid as an argument.
 * @param{in] scaling_factor            : IFFT scaling factor
 * @param{in] fft_size                  : FFT size
 * @param[in] symbols_per_subframe      : number of symbols per subframe
 * @param[in] nb_subframes              : number of subframes to be sent
 * @param[in] scs                       : SCS
 */
void compute_ifft(std::vector<std::vector<std::complex<float>>> &inputGrid,
                  std::vector<std::complex<float>> &output,
                  const int * cp_lengths_,
                  const int * cum_sum_cp_lengths_,
                  const float &scaling_factor,
                  const int &fft_size,
                  const int &symbols_per_subframe,
                  const int &nb_subframes,
                  const int &scs);

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
void compute_fft(const std::complex<float> * input,
                 std::vector<std::vector<std::complex<float>>> &output,
                 const int &nb_subframes,
                 const int &fft_size,
                 const int &symbols_per_sub_frame,
                 const int &scs,
                 const int * cp_lengths_,
                 const int * cum_sum_cp_lengths_) ; // Number of slots per RX buffer

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
void compute_fft(const std::vector<std::complex<float>> &input,
                 std::vector<std::vector<std::complex<float>>> &output,
                 const int &fft_size,
                 const int &num_symbols_,
                 const int * cp_lengths_,
                 const int * cum_sum_cp_lengths_,
                 const int &offset_);

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
void computeFineFrequencyOffset( std::complex<float> * inputSignal,
                                 int symbolDuration, int fftSize, int cpLength,
                                 int scs, float& output, int numSymbols );

/** free5GRAN function to correct the frequency offset
 *
 * @param inputSignal : intput signal to be corrected
 * @param freqOffset : frequency offset value determined previously
 * @param sampleRate : sample rate
 */
void transposeSignal( std::vector<std::complex<float>>* inputSignal, float freqOffset, double sampleRate );

#endif //STARTERPROJECT_ALAMOUTI_UTILS_H
