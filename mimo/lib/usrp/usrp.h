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

    This is a modified version of the rf class from free5GRAN to
    interface with USRP using libuhd driver.

 */

#ifndef USRP_H
#define USRP_H

#include <uhd/utils/safe_main.hpp>
#include <uhd/types/tune_request.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <uhd/usrp_clock/multi_usrp_clock.hpp>


#include <condition_variable>
#include <complex>
#include <vector>
#include <thread>
#include <mutex>
#include <iostream>
#include <chrono>
#include "../free5gran_utils.h"

/// Base class of the USRP objects, only instantiated by child classes (protected constructor).

class Usrp {

protected :
    uhd::usrp::multi_usrp::sptr usrp = NULL;                                          // pointer to the multi_usrp object
    std::string device_args{},                                                          // serial number of the USRP device to be used
    tx_antenna{}, rx_antenna{};                                                       // Name of the TX and RX paths to be used on the device
    uhd::usrp::subdev_spec_t tx_subdevice, rx_subdevice;                              // RF front-ends to be used
    uhd::tune_request_t center_frequency{};                                           // center frequency
    float bandwidth{}, sample_rate{}, tx_gain{}, rx_gain{};                                        // TX and RX gains
    std::vector<uhd::tx_streamer::sptr> tx_streamers{};                               // vector containing the tx_streamers to be used (one per thread)
    std::vector<uhd::rx_streamer::sptr> rx_streamers{};                               // vector containing the rx_streamers to be used (on per thread)

    int stop_signal_called_tx = 0; // stop the tx threads
    int stop_signal_called_rx = 0; // stop the rx threads

    std::condition_variable cv;
    std::mutex m;
    bool ready_tx = false;
    bool ready_rx = false;

    bool nothing_received = false; // Set to true in receive() function if nothing was received

    // Vector containing the ports used for transmission and reception
    std::vector<std::vector<size_t>> nb_ports_tx;
    std::vector<std::vector<size_t>> nb_ports_rx;

    // Buffers for each port group
    std::vector<std::vector<std::vector<std::complex<float>>>> data_tx;    // contains the data to be sent on each TX path for each user
    std::vector<std::vector<std::vector<std::complex<float>>>> data_rx;    // contains the data received on each RX path for each user
    std::vector<std::vector<std::complex<float>*>> buffers_tx;
    std::vector<std::vector<std::complex<float>*>> buffers_rx;

    // Number of samples in each buffer for each user
    int n_samples_tx; // Number of samples in the sending buffers
    int n_samples_rx; // Number of samples in the receiving buffers

    std::vector<std::thread> sendingThreads;                                           // vector containing references to all the sending threads
    std::vector<std::thread> receivingThreads;                                         // vector containing references to all the receiving threads
    bool separate_tx_threads;                                                          // true in case separate TX threads or 1 TX thread are used
    bool separate_rx_threads;                                                          // ture in case separate RX threads or 1 RX thread are used

    /** Constructor. Initializes the USRP device.
     *
     * @param serial_no        : serial nuber of the USRP device
     * @param tx_subdevice     : name of the TX front-end(s) to be used
     * @param rx_subdevice     : name of the RX front-end(s) to be used
     * @param sample_rate      : sample rate to be used (bandwidth to be used)
     * @param center_freq      : center frequency
     * @param tx_gain          : TX gain [dB]
     * @param rx_gain          : RX gain [dB]
     * @param tx_antenna       : name of the TX antenna to be used
     * @param rx_antenna       : name of the RX antenna to be used
     * @param num_channels_tx_ : number of TX channels to be used
     * @param num_channels_rx_ : number of RX channels to be used
     */
    Usrp(std::string device_args_,
         std::string tx_subdevice_,
         std::string rx_subdevice_,
         std::string clock_ref_,
         float bandwidth_,
         float sample_rate_,
         float center_freq_,
         float tx_gain_,
         float rx_gain_,
         std::string tx_antenna_,
         std::string rx_antenna_,
         std::vector<std::vector<size_t>> ports_per_users_tx_,
         std::vector<std::vector<size_t>> ports_per_users_rx_);

    /** Send the data
     *
     * @param buffer            : buffer containing the data to send
     * @param n_samples_to_send : number of samples in the buffer
     * @param streamer          : TX streamer to be used */
    void send(std::vector<std::complex<float> *> &buffer_,
              const int &n_samples_to_send_,
              uhd::tx_streamer::sptr &streamer_);

    /** Receives the data
     *
     * @param buffer               : buffer to contain the received samples
     * @param n_samples_to_receive : capacity of the buffer
     * @param streamer             : RX streamer to be used */
    void receive(std::vector<std::complex<float> *> &buffer_,
                 const int &n_samples_to_receive_,
                 uhd::rx_streamer::sptr &streamer_);


public :

    /** Initializes the TX buffers with the data to be transmitted
     *
     * @param data_tx1 : data to be transmitted on TX1
     * @param data_tx2 : data to be transmitted on TX2
     */
    void init_tx_buffers(const std::vector<std::vector<std::vector<std::complex<float>>>> &data_tx_);

    /** Initializes the RX buffers
     *
     * @param samples_per_buff : number of samples to be received in on buffer, on one RX path (RX1 or RX2)
     */
    void init_rx_buffers(int samples_per_buff);

    /** Configures the TX streamers for each TX thread */
    virtual void config_tx_streams() ;

    /** Configure the RX streams for each RX thread */
    virtual void config_rx_streams();

    /** Starts the sending threads
     */
    void start_sending();

    /** Starts the receiving threads
     */
    void start_receiving();

    /** Stops the sending and receiving threads */
    void stop_tx_rx_threads();

    /** Stops the receiving threads */
    void stop_rx_threads();

    /** Returns the receiving buffer
     *
     * @param[out] output : output to contain the buffer(s). Provide an empty vector as an argument, as it is resized
     *                      within the function.
     * @param channel     : in case of separate RX transmissions, provide the channel index (0 or 1) for which you want to get the data
     */
    void get_receiving_buffer(std::vector<std::vector<std::vector<std::complex<float>>>> &output);

    bool get_nothing_received();

    float get_tx_bandwith(int chan) {
        return usrp->get_rx_bandwidth(chan);
    }
    float get_rx_bandwidth(int chan) {
        return usrp->get_tx_bandwidth(chan);
    }
    float get_tx_rate(int chan) {
        return usrp->get_tx_rate(chan);
    }
    float get_rx_rate(int chan) {
        return usrp->get_rx_rate(chan);
    }
    float get_tx_gain(int chan) {
        return usrp->get_tx_gain(chan);
    }
    float get_rx_gain(int chan) {
        return usrp->get_rx_gain(chan);
    }
    float get_tx_center_frequency(int chan) {
        return usrp->get_tx_freq(chan);
    }
    float get_rx_center_frequency(int chan) {
        return usrp->get_rx_freq(chan);
    }
    int get_num_tx_channels(int chan) {
        return usrp->get_tx_num_channels();
    }
    int get_num_rx_channels(int chan) {
        return usrp->get_rx_num_channels();
    }
    float get_master_clock_rate() {
        return usrp->get_master_clock_rate();
    }

    /*
    ~USRPB210(){
        no need to free memory allocated to pointers in vectors buffer_tx1/tx2/tx1_tx2/rx1/rx2/rx1_rx2 arrays
        ==> pointers point to vectors data_tx1/tx2/rx1/rx2, whose memory is freed when the object instance is destroyed
        ==> pointers in buffer vectors are not pending because they do not point to anything afterwards.
    }
     */
};

#endif //USRP_H
