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

    This is a modified version of the usrp_x300 class from free5GRAN.
*/

#include "usrp_x300.h"

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
UsrpX300::UsrpX300(std::string device_args_,
                   std::string tx_subdevice_,
                   std::string rx_subdevice_,
                   std::string clock_ref_,
                   float bandwidth_,
                   float sample_rate_,
                   float center_freq_,
                   float tx_gain_,
                   float rx_gain_,
                   float master_clock_rate_,
                   std::string tx_antenna_,
                   std::string rx_antenna_,
                   vector<vector<size_t>> nb_ports_tx_,
                   vector<vector<size_t>> nb_ports_rx_)

                   : Usrp (device_args_,
                           tx_subdevice_,
                           rx_subdevice_,
                           clock_ref_,
                           bandwidth_,
                           sample_rate_,
                           center_freq_,
                           tx_gain_,
                           rx_gain_,
                           tx_antenna_,
                           rx_antenna_,
                           nb_ports_tx_,
                           nb_ports_rx_) {

    /// initialize local attributes
    master_clock_rate = master_clock_rate_;

    // Create USRP device
    cout << "devide args : " << device_args << endl;

    uhd::device_addr_t device_args(device_args_);
    usrp = uhd::usrp::multi_usrp::make(device_args);

    // Use 10 MHz clock on "ref in" port and pps signal on "pps in"
    usrp->set_time_source(clock_ref_);
    usrp->set_clock_source(clock_ref_);
    for(int i = 0; i < usrp->get_num_mboards(); i++) {
        cout << "mboard " << i << " clock source = " << usrp->get_clock_source(i) << endl;
    }

    // Tell the USRP to reset their sense of time to 0.000s on the next PPS edge
    usrp->set_time_next_pps(uhd::time_spec_t(0.0));
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // Select the subdevices for TX and RX
    usrp->set_tx_subdev_spec(tx_subdevice, uhd::usrp::multi_usrp::ALL_MBOARDS);
    usrp->set_rx_subdev_spec(rx_subdevice, uhd::usrp::multi_usrp::ALL_MBOARDS);

    cout << "Number of available TX channels : " << usrp->get_tx_num_channels() << endl;
    cout << "Number of available RX channels : " << usrp->get_rx_num_channels() << endl;

    /// Synchronously tune the two usrp devices
    usrp->clear_command_time();

    usrp->set_command_time(usrp->get_time_now() + uhd::time_spec_t(0.1));

    /// Specify master clock rate in device args. This command doesn't set the proper master clock rate
    usrp->set_master_clock_rate(master_clock_rate, uhd::usrp::multi_usrp::ALL_CHANS);

    usrp->clear_command_time();

    vector<string> lo_sources = usrp->get_tx_lo_sources(uhd::usrp::multi_usrp::ALL_LOS,
                                                        0);

    cout << "Available TX LO sources : " << endl;
    for (int i = 0; i < lo_sources.size(); i++) {
        cout << lo_sources[i] << endl;
    }

    /// Check that all channels have been configured
    for(int tx_chan = 0; tx_chan < usrp->get_tx_num_channels(); tx_chan++) {

        // Set the center frequency
        usrp->set_command_time(usrp->get_time_now() + uhd::time_spec_t(0.1));
        usrp->set_tx_freq(center_frequency, tx_chan);

        usrp->clear_command_time();

        // Set the sample rate
        usrp->set_command_time(usrp->get_time_now() + uhd::time_spec_t(0.1));
        usrp->set_tx_rate( sample_rate, tx_chan);

        usrp->clear_command_time();

        // Set TX gain
        usrp->set_command_time(usrp->get_time_now() + uhd::time_spec_t(1.0));
        usrp->set_tx_gain(tx_gain, tx_chan);

        usrp->clear_command_time();

        // Set the IF filter bandwidth
        usrp->set_command_time(usrp->get_time_now() + uhd::time_spec_t(1.0));
        usrp->set_tx_bandwidth(bandwidth, tx_chan);

        usrp->clear_command_time();

        // Set the TX antenna to be used
        usrp->set_command_time(usrp->get_time_now() + uhd::time_spec_t(1.0));
        usrp->set_tx_antenna(tx_antenna, tx_chan);

        usrp->clear_command_time();

        // Wait until LO locked
        /*
        cout << "TX LO source : "
             << usrp->get_rx_lo_source(uhd::usrp::multi_usrp::ALL_LOS, tx_chan)
             << endl;
        auto lo_locked = usrp->get_tx_sensor("lo_locked", tx_chan);
        cout << "lo_locked : " << lo_locked.to_bool() << endl;
        while (not (usrp->get_tx_sensor("lo_locked", tx_chan).to_bool())){
            //sleep for a short time in milliseconds
            cout << "wait for TX lo locked" << endl;
            sleep(1);
        } */

        std::cout << "TX Sampling rate (from host PC to USRP FPGA) : " << usrp->get_tx_rate(tx_chan) << std::endl;
        std::cout << "TX frequency : " << usrp->get_tx_freq(tx_chan) << std::endl;
        std::cout << "TX gain [dB] : " << usrp->get_tx_gain(tx_chan) << std::endl;
        std::cout << "normalized TX gain : " << usrp->get_normalized_tx_gain(tx_chan) << std::endl;
        std::cout << "TX bandwidth (IF filter bandwidth) : " << usrp->get_tx_bandwidth(tx_chan) << std::endl;
    }

    /// Wait for TX LOs to be locked
    sleep(2);

    // Configure all the available RX channels
    for(int rx_chan = 0; rx_chan < usrp->get_rx_num_channels(); rx_chan++) {

        // Set the center frequency
        usrp->set_command_time(usrp->get_time_now() + uhd::time_spec_t(1.0));
        usrp->set_rx_freq(center_frequency, rx_chan);

        usrp->clear_command_time();

        // Set the sample rate
        usrp->set_command_time(usrp->get_time_now() + uhd::time_spec_t(1.0));
        usrp->set_rx_rate(sample_rate, rx_chan);

        usrp->clear_command_time();

        // Set the RX gain
        usrp->set_command_time(usrp->get_time_now() + uhd::time_spec_t(1.0));
        usrp->set_rx_gain(rx_gain, rx_chan);

        usrp->clear_command_time();

        // Set the IF filter bandwidth
        usrp->set_command_time(usrp->get_time_now() + uhd::time_spec_t(1.0));
        usrp->set_rx_bandwidth(bandwidth, rx_chan);

        usrp->clear_command_time();

        // Set the RX antenna to be used
        usrp->set_command_time(usrp->get_time_now() + uhd::time_spec_t(1.0));
        usrp->set_rx_antenna(rx_antenna, rx_chan);

        usrp->clear_command_time();

        // Wait until LO locked
        /*
        cout << "RX LO source : "
             << usrp->get_rx_lo_source(uhd::usrp::multi_usrp::ALL_LOS, rx_chan)
             << endl;
        while (not (usrp->get_rx_sensor("lo_locked", rx_chan).to_bool())) {
            //sleep for a short time in milliseconds

        } */

        std::cout << "RX Rate (from USRP FPGA to Host PC) : " << usrp->get_rx_rate(rx_chan) << std::endl;
        std::cout << "RX frequency : " << usrp->get_rx_freq(rx_chan) << std::endl;
        std::cout << "RX gain [dB] : " << usrp->get_rx_gain(rx_chan) << std::endl;
        std::cout << "normalized RX gain : " << usrp->get_normalized_rx_gain(rx_chan) << std::endl;
        std::cout << "RX bandwidth (IF Filter Bandwidth): " << usrp->get_rx_bandwidth(rx_chan) << std::endl;
        std::cout << "RX antenna : " << usrp->get_rx_antenna(rx_chan) << std::endl;
        std::cout << "Decimation factor [Master Clock Rate / RX sampling rate] : " << usrp->get_master_clock_rate() / usrp->get_rx_rate(rx_chan) << endl;
    }

    /// Wait for RX LOs to be locked
    sleep(2);
}

/** Configures the TX streamers */
void UsrpX300::config_tx_streams() {

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
void UsrpX300::config_rx_streams() {

    uhd::stream_args_t stream_args_rx = {"fc32", "sc16"};

    // Configure the rx_streamer for all the groups of ports
    for(int i = 0; i < nb_ports_rx.size(); i++) {
        stream_args_rx.channels = nb_ports_rx[i]; // Get the antenna ports used by this user
        rx_streamers.push_back(usrp->get_rx_stream(stream_args_rx)); // Create a RX streamer for this user
    }
}