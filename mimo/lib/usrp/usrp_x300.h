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
