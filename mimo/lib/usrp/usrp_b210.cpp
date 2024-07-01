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

    This is a modified version of the usrp_b210 class from free5GRAN.

 */
#include "usrp_b210.h"

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
UsrpB210::UsrpB210(std::string device_args_,
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
                   vector<vector<size_t>> nb_ports_rx_)
                   : Usrp(device_args_,
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
                          nb_ports_rx_){

    // Create USRP device

    uhd::device_addr_t device_args(device_args_);
    usrp = uhd::usrp::multi_usrp::make(device_args);

    // Use 10 MHz clock on "ref in" port and pps signal on "pps in"
    usrp->set_sync_source(clock_ref_, clock_ref_);

    // Tell the USRP to reset their sense of time to 0.000s on the next PPS edge
    usrp->set_time_next_pps(uhd::time_spec_t(0.0));
    std::this_thread::sleep_for(std::chrono::milliseconds(1000));

    // Select the subdevices for TX and RX
    usrp->set_tx_subdev_spec(tx_subdevice, uhd::usrp::multi_usrp::ALL_MBOARDS);
    usrp->set_rx_subdev_spec(rx_subdevice, uhd::usrp::multi_usrp::ALL_MBOARDS);

    cout << "Number of available TX channels : " << usrp->get_tx_num_channels() << endl;
    cout << "Number of available RX channels : " << usrp->get_rx_num_channels() << endl;

    // Configure all the available TX channels
    for(int tx_chan = 0; tx_chan < usrp->get_tx_num_channels(); tx_chan++) {
        // Set the sample rate
        usrp->set_tx_rate(7.68e6, tx_chan); //10.24e6, tx_chan)
        std::cout << "TX Rate : " << usrp->get_tx_rate(tx_chan) << std::endl;

        // Set the center frequency
        usrp->set_tx_freq(center_frequency, tx_chan);
        std::cout << "TX frequency : " << usrp->get_tx_freq(tx_chan) << std::endl;

        // Wait until LO locked
        while (not (usrp->get_tx_sensor("lo_locked").to_bool())){
            //sleep for a short time in milliseconds
            sleep(1);
        }

        // Set TX gain
        usrp->set_tx_gain(tx_gain, tx_chan);
        std::cout << "TX gain [dB] : " << usrp->get_tx_gain(tx_chan) << std::endl;
        std::cout << "normalized TX gain : " << usrp->get_normalized_tx_gain(tx_chan) << std::endl;

        // Set the IF filter bandwidth
        usrp->set_tx_bandwidth(bandwidth, tx_chan);
        std::cout << "TX bandwidth : " << usrp->get_tx_bandwidth(tx_chan) << std::endl;

        // Set the TX antenna to be used
        usrp->set_tx_antenna(tx_antenna, tx_chan);
    }

    // Configure all the available RX channels
    for(int rx_chan = 0; rx_chan < usrp->get_rx_num_channels(); rx_chan++) {

        // Set the sample rate
        usrp->set_rx_rate(7.68e6, rx_chan); //10.24e6, rx_chan)
        std::cout << "RX Rate : " << usrp->get_rx_rate(rx_chan) << std::endl;

        // Set the center frequency
        usrp->set_rx_freq(center_frequency, rx_chan);
        std::cout << "RX frequency : " << usrp->get_rx_freq(rx_chan) << std::endl;

        // Wait until LO locked
        while (not (usrp->get_rx_sensor("lo_locked").to_bool())) {
            //sleep for a short time in milliseconds
            sleep(1);
        }

        // Set the RX gain
        usrp->set_rx_gain(rx_gain, rx_chan);
        std::cout << "RX gain [dB] : " << usrp->get_rx_gain(rx_chan) << std::endl;
        std::cout << "normalized RX gain : " << usrp->get_normalized_rx_gain(rx_chan) << std::endl;

        // Set the IF filter bandwidth
        usrp->set_rx_bandwidth(bandwidth, rx_chan);
        std::cout << "RX bandwidth : " << usrp->get_rx_bandwidth(rx_chan) << std::endl;

        // Set the RX antenna to be used
        usrp->set_rx_antenna(rx_antenna, rx_chan);
        std::cout << "RX antenna : " << usrp->get_rx_antenna(rx_chan) << std::endl;
    }


}

/** Configures the TX streamers */
void UsrpB210::config_tx_streams() {

    uhd::stream_args_t stream_args_tx = {"fc32", "sc8"};

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
void UsrpB210::config_rx_streams() {

    uhd::stream_args_t stream_args_rx = {"fc32", "sc8"};

    // Configure the rx_streamer for all the groups of ports
    for(int i = 0; i < nb_ports_rx.size(); i++) {
        stream_args_rx.channels = nb_ports_rx[i]; // Get the antenna ports used by this user
        rx_streamers.push_back(usrp->get_rx_stream(stream_args_rx)); // Create a RX streamer for this user
    }
}