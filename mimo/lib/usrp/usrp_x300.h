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

    This is a modified version of the usrp_x300 class from free5GRAN.
 */

#ifndef USRP_X300_H
#define USRP_X300_H

#include <uhd/utils/safe_main.hpp>
#include <uhd/types/tune_request.hpp>
#include <uhd/usrp/multi_usrp.hpp>
#include <condition_variable>
#include <complex>
#include <vector>
#include <thread>
#include <mutex>
#include <iostream>

#include "usrp.h"

/// Child Class to instantiate USRP X300 objects

class UsrpX300 : public Usrp {

private :
    float master_clock_rate = 0;

public :
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
    UsrpX300(std::string device_args_,
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
             std::vector<std::vector<size_t>> nb_layers_and_users_tx_,
             std::vector<std::vector<size_t>> nb_layers_and_users_rx_);

    /** Configures the TX streamers for each TX thread */
    void config_tx_streams() override;

    /** Configure the RX streams for each RX thread */
    void config_rx_streams() override;

};

#endif // USRP_X300_H
