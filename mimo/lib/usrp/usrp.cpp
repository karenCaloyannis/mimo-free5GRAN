/*  Copyright (C) 2023  Telecom Paris

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

    This is a modified version of the rf class from free5GRAN to
    interface with USRP using libuhd driver.
 */

#include "usrp.h"

using namespace std;

/** Constructor. Initializes the USRP device.
     *
     * @param serial_no               : serial nuber of the USRP device
     * @param tx_subdevice            : name of the TX front-end(s) to be used
     * @param rx_subdevice            : name of the RX front-end(s) to be used
     * @param sample_rate             : sample rate to be used (bandwidth to be used)
     * @param center_freq             : center frequency
     * @param tx_gain                 : TX gain [dB]
     * @param rx_gain                 : RX gain [dB]
     * @param tx_antenna              : name of the TX antenna to be used
     * @param rx_antenna              : name of the RX antenna to be used
     * @param nb_layers_and_users_tx_ : Vector of TX layers used by each user. nb_layers_users_tx_[i] corresponds to the user i, and returns a vector
     *                                  containing the antenna ports used by user i.
     * @param nb_layers_and_users_tx_ : Vector of RX layers used by each user. nb_layers_users_rx_[i] corresponds to the user i, and returns a vector
     *                                  containing the antenna ports used by user i.
     */
Usrp::Usrp(std::string device_args_,
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
           vector<vector<size_t>> nb_ports_tx_,
           vector<vector<size_t>> nb_ports_rx_) {

    // Initialize class members
    // Usrp is configured in the child class
    device_args      = device_args_;
    tx_subdevice     = {tx_subdevice_}, rx_subdevice = {rx_subdevice_};
    tx_antenna       = tx_antenna_, rx_antenna = rx_antenna_;
    bandwidth        = bandwidth_;
    sample_rate      = sample_rate_;
    center_frequency = center_freq_;
    tx_gain          = tx_gain_,     rx_gain = rx_gain_;
    nb_ports_tx = nb_ports_tx_;
    nb_ports_rx = nb_ports_rx_;
}

/** Send the data
     *
     * @param buffer            : buffer containing the data to send
     * @param n_samples_to_send : number of samples in the buffer
     * @param streamer          : TX streamer to be used
     * @param separate_threads  : set to "True" when using separate threads, to "False" when using only one thread for 2 transmissions
     */
void Usrp::send(vector<complex<float> *> &buffer,
                const int &n_samples_to_send,
                uhd::tx_streamer::sptr &streamer) {

    //std::chrono::high_resolution::time_point t1{}, t2{};

    //uhd::set_thread_priority_safe(1.0, true);

    // Configure the metadata
    uhd::tx_metadata_t md_tx;
    md_tx.start_of_burst = true;
    md_tx.end_of_burst   = false;
    //md_tx.has_time_spec  = not (streamer->get_num_channels() == 1); // Stream now if only one channel is used for transmission
    md_tx.has_time_spec = true; /// Leave time spec to avoid phase problems
    md_tx.time_spec = usrp->get_time_now().get_real_secs() + 1.0;//+ uhd::time_spec_t::from_ticks(500, sample_rate);

    //t1 = std::chrono::steady_clock::now();
    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [this]{return ready_tx; });
    lk.unlock();

    //struct timeval tv;
    //gettimeofday(&tv, NULL);

    //printf("\nTransmitting (timeval : %d) ... \n", (int) tv.tv_sec);

    int num_samples_sent = 0;

    //t2 = std::chrono::steady_clock::now();

    while(not stop_signal_called_tx) {
        //while(num_samples_sent < n_samples_to_send) {
            //num_samples_sent += streamer->send(buffer, n_samples_to_send, md_tx);
            streamer->send(buffer, n_samples_to_send, md_tx);
            md_tx.start_of_burst = false;
        //}
        //num_samples_sent = 0;
        //if(stop_signal_called_usrp) {
        //    break;
        //}
    }

    //cout << "\ndelay send command : " << std::chrono::duration_cast<std::chrono::seconds>(t2 - t1).count() << endl;

    //cout << "\nNumber of samples sent : " << num_samples_sent << endl;
}


/** Receives the data
 *
 * @param buffer               : buffer to contain the received samples
 * @param n_samples_to_receive : capacity of the buffer
 * @param streamer             : RX streamer to be used
 * @param separate_threads     : set to "True" when using separate threads, to "False" when using only one thread for reception on th 2 RX paths
 */
void Usrp::receive(vector<complex<float> *> &buffer,
                   const int &n_samples_to_receive,
                   uhd::rx_streamer::sptr &streamer) {

    //std::chrono::steady_clock::time_point t1{}, t2{}, t1_{}, t2_{};

    // Receive the data on the RX streamer (thread 2)
    uhd::rx_metadata_t md_rx;
    md_rx.start_of_burst = true;
    md_rx.end_of_burst = false;

    std::unique_lock<std::mutex> lk(m);
    cv.wait(lk, [this]{return ready_rx; });
    lk.unlock();

    //t1_ = std::chrono::steady_clock::now();

    uhd::stream_cmd_t stream_cmd(uhd::stream_cmd_t::STREAM_MODE_NUM_SAMPS_AND_DONE);
    stream_cmd.time_spec = uhd::time_spec_t(usrp->get_time_now().get_real_secs() + 1.0); // delay to synchronize the two receiving channels before receiving packets
    //stream_cmd.stream_now = streamer->get_num_channels() == 1; // true or false
    stream_cmd.stream_now = false; /// Leave stream delay to avoid phase problems
    stream_cmd.num_samps = n_samples_to_receive;

    streamer->issue_stream_cmd(stream_cmd);

    size_t num_samples_received = 0;
    size_t num_acc_samples = 0;

    double timeout = usrp->get_time_now().get_real_secs() + 4.0;

    while(not stop_signal_called_rx) {
        while (num_acc_samples < n_samples_to_receive) {

            num_samples_received = streamer->recv(buffer, n_samples_to_receive - num_samples_received, md_rx, timeout);

            num_acc_samples += num_samples_received;
            md_rx.start_of_burst  = false;

            if ((md_rx.error_code == uhd::rx_metadata_t::ERROR_CODE_TIMEOUT) or
                (md_rx.error_code == uhd::rx_metadata_t::ERROR_CODE_LATE_COMMAND) or
                (md_rx.error_code == uhd::rx_metadata_t::ERROR_CODE_BROKEN_CHAIN) or
                //(md_rx.error_code == uhd::rx_metadata_t::ERROR_CODE_OVERFLOW) or
                (md_rx.error_code == uhd::rx_metadata_t::ERROR_CODE_ALIGNMENT) or
                (md_rx.error_code == uhd::rx_metadata_t::ERROR_CODE_BAD_PACKET))    {

                cout << "\nRX error code : " << md_rx.strerror() << endl;
                nothing_received = true;
                break;
            }
        }

        //t2 = std::chrono::steady_clock::now();
    }

    //cout << "\ndelay recv command : " << std::chrono::duration_cast<std::chrono::microseconds>(t2_ - t1_).count() << endl;

    md_rx.end_of_burst = true;

    //auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1);
    cout << "\nReceived "   << num_samples_received << " samples" << endl;
    //cout << "\nReceived in " << duration.count() << " nanoseconds" << endl;
}


/** Configures the TX streamers */
void Usrp::config_tx_streams() {

    uhd::stream_args_t stream_args_tx = {"fc32", "sc16"};

    // Configure the tx_streamers for all the groups of ports
    for(int i = 0; i < nb_ports_tx.size(); i++) {
        stream_args_tx.channels = nb_ports_tx[i]; // Get the antenna ports used by this user
        tx_streamers.push_back(usrp->get_tx_stream(stream_args_tx)); // Create a TX streamer for this user
    }
}


/** Configure the RX streamers
 *
 * @param separate_rx_streams : set to "True when using separate threads for reception,
 *                              to "False" when using one thread for simultaneous transmissions on RX1 and RX2
 *                              When receiving on only one antenna, this has no effect as it depends on the attribute
 *                             num_channels_rx initialized in the constructor.
 * @param channel              : index of the cannel to be used when receiving on only one antenna (0 or 1)
 */
void Usrp::config_rx_streams() {

    uhd::stream_args_t stream_args_rx = {"fc32", "sc16"};

    // Configure the rx_streamer for all the groups of ports
    for(int i = 0; i < nb_ports_rx.size(); i++) {
        stream_args_rx.channels = nb_ports_rx[i]; // Get the antenna ports used by this user
        rx_streamers.push_back(usrp->get_rx_stream(stream_args_rx)); // Create a RX streamer for this user
    }
}

/** Initializes the TX buffers with the data to be transmitted
 *
 * @param data_tx_ : vector of size [number of users][number of TX paths used by this user][number of samples on the TX path]
 *                   the vectors of data must be in the same order as the one provided in nb_layers_and_users_tx.
 */
void Usrp::init_tx_buffers(const vector<vector<vector<complex<float>>>> &data_tx_){

    data_tx.resize(nb_ports_tx.size());
    buffers_tx.resize(nb_ports_tx.size());

    for(int port_group = 0; port_group < nb_ports_tx.size(); port_group++) {
        data_tx[port_group].resize(nb_ports_tx[port_group].size()); // Initialize a buffer for each antenna port used by the user
        buffers_tx[port_group].resize(nb_ports_tx[port_group].size());

        for(int tx_antenna = 0; tx_antenna < data_tx[port_group].size(); tx_antenna++) {
            data_tx[port_group][tx_antenna]    = data_tx_[port_group][tx_antenna];
            buffers_tx[port_group][tx_antenna] = &data_tx[port_group][tx_antenna].front();
        }
    }

    // Number of samples must be the same on each TX paths
    n_samples_tx = data_tx[0][0].size();
    cout << "n_samples _tx: " << n_samples_tx << endl;
}


/** Initializes the RX buffers
 *
 * @param samples_per_buff : number of samples to be received in on buffer, on one RX path (RX1 or RX2)
 */
void Usrp::init_rx_buffers(int samples_per_buff) {

    data_rx.clear();
    buffers_rx.clear();

    data_rx.resize(nb_ports_rx.size());
    buffers_rx.resize(nb_ports_rx.size());

    for(int port_group = 0; port_group < nb_ports_tx.size(); port_group++) {
        data_rx[port_group].resize(nb_ports_rx[port_group].size()); // Initialize a buffer for each antenna port used by the user
        buffers_rx[port_group].resize(nb_ports_rx[port_group].size());

        for(int rx_antenna = 0; rx_antenna < data_rx[port_group].size(); rx_antenna++) {
            data_rx[port_group][rx_antenna].resize(samples_per_buff);
            buffers_rx[port_group][rx_antenna] = &data_rx[port_group][rx_antenna].front();
        }
    }

    // Number of samples must be the same on each RX paths
    n_samples_rx = data_rx[0][0].size();
}

/** Stops the receiving threads */
void Usrp::stop_rx_threads() {

    stop_signal_called_rx = 1;

    /// Clear receiving buffer
    // stop the receiving threads
    for(auto &thread : receivingThreads) {
        thread.join();
    }

    cout << "\nall RX threads stopped\n" << endl;

    /// Clear the vectors
    receivingThreads.clear();

    cout << "\nall RX threads cleared\n" << endl;

    /// Clear RX streamers
    rx_streamers.clear();

    cout << "\nRX streamers\n" << endl;

    /// reset for further sending/receiving
    stop_signal_called_rx = 0;

    {
        std::lock_guard<std::mutex> lk(m);
        ready_rx = false;
    }
}

/** Stops the sending and receiving threads */
void Usrp::stop_tx_rx_threads() {

    cout << "\nstopping the USRP tx and rx threads...\n" << endl;

    // send the last buffer then stop sending and receiving
    stop_signal_called_tx = 1;
    stop_signal_called_rx = 1;

    // stop the sending threads
    for(auto &thread : sendingThreads) {
        thread.join();
    }

    cout << "\nall TX threads stopped\n" << endl;

    // stop the receiving threads
    for(auto &thread : receivingThreads) {
        thread.join();
    }

    cout << "\nall RX threads stopped\n" << endl;

    /// Clear the vectors
    sendingThreads.clear();
    receivingThreads.clear();

    cout << "\nall threads cleared\n" << endl;

    /// Clear TX and RX streamers
    rx_streamers.clear();
    tx_streamers.clear();

    cout << "\nCleared TX and RX streamers\n" << endl;

    /// reset for further sending/receiving
    stop_signal_called_tx = 0;
    stop_signal_called_rx = 0;

    {
        std::lock_guard<std::mutex> lk(m);
        ready_rx = false;
    }

    {
        std::lock_guard<std::mutex> lk(m);
        ready_tx = false;
    }
}

/** Starts the sending threads */
void Usrp::start_sending() {

    // Start the sending threads for each port group
    for(int user = 0; user < nb_ports_tx.size(); user++) {
        sendingThreads.emplace_back(thread(&Usrp::send, this,
                                           ref(buffers_tx[user]),
                                           n_samples_tx,
                                           std::ref(tx_streamers[user])));
    }
    cout << "Started the sending threads" << endl;

    // Unlock the mutex and start sending
    {
        std::lock_guard<std::mutex> lk(m);
        ready_tx = true;
    }
    cv.notify_all();
}

/** Starts the receiving threads */
void Usrp::start_receiving() {

    // Start the receiving threads for each user
    for(int user = 0; user < nb_ports_rx.size(); user++) {
        receivingThreads.emplace_back(thread(&Usrp::receive, this,
                                             ref(buffers_rx[user]),
                                             n_samples_rx,
                                             std::ref(rx_streamers[user])));
    }
    cout << "Started the receiving threads" << endl;

    // Unlock the mutex and start receiving
    {
        std::lock_guard<std::mutex> lk(m);
        ready_rx = true;
    }
    cv.notify_all();
}


/** Returns the receiving buffer
 *
 * @param[out] output : output to contain the buffer(s). Provide an empty vector as an argument, as it is resized
 *                      within the function.
 * @param channel     : in case of separate RX transmissions, provide the channel index (0 or 1) for which you want to get the data
 */
void Usrp::get_receiving_buffer(vector<vector<vector<complex<float>>>> &output) {

    output.resize(nb_ports_rx.size());

    for(int port_group = 0 ; port_group < nb_ports_rx.size(); port_group++) {
        output[port_group].resize(nb_ports_rx[port_group].size());
        output[port_group] = data_rx[port_group];
    }
}

bool Usrp::get_nothing_received() {
    return nothing_received;
}