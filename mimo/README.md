## 5G MIMO transceiver platform based on free5GRAN

Source code for the paper :

Karen Caloyannis, Anaïs Vergne, Philippe Martins. "Software Defined 
Radio platform to evaluate processing latency of 5G NR MIMO 
functions". *3rd ACM Workshop on 5G and Beyond Network 
Measurements, Modeling, and Use Cases (5G-MeMU)*, Sep 2023, New York, United States. ⟨10.1145/3609382.3610512⟩. ⟨hal-04192756⟩
Available online : [https://hal.science/hal-04192756]



The code implements a MIMO transceiver platform to benchmark MIMO decoders to be integrated to *free5GRAN*.
The transmission part transmits radio frames containing simulated PDSCH data to
be decoded at the receiver.

The code is still under development and may not be fully stable.

## Licensing 

The source code in the following directories is licensed under *GPLv3* terms : 
- ```lib/mimo```
- ```tx_rx/x300```
- ```tx_rx/x300/multisim```
- ```tx_rx/avx512```
- ```lib/avx```
- ```lib/usrp```
- ```lib/utils```
- ```plots_exports/```

License terms for *GPLv3* license are in the current directory. 

The source code from the following directories or files is licensed under *Apache 2.0* license 
terms and reuse code from the *free5GRAN* library : 
- ```lib/variables```
- ```free5gran_utils.cpp/.h```

License terms for *Apache 2.0 license* are in the root directory 
of the project. 

## Directories

The code is organized as follows :

```
├── CMakeLists.txt
├── README.md
├── lib
│   ├── avx
│   │   └── avx_ops.h
│   ├── free5gran_utils.cpp
│   ├── free5gran_utils.h
│   ├── usrp
│   │   ├── usrp.cpp
│   │   ├── usrp.h
│   │   ├── usrp_b210.cpp
│   │   ├── usrp_b210.h
│   │   ├── usrp_x300.cpp
│   │   └── usrp_x300.h
│   ├── utils
│   │   ├── channel_estimation
│   │   │   ├── channel_estimation.cpp
│   │   │   └── channel_estimation.h
│   │   ├── channel_mapping
│   │   │   ├── channel_mapping.cpp
│   │   │   └── channel_mapping.h
│   │   └── mimo
│   │       ├── mimo.cpp
│   │       ├── mimo.h
│   │       ├── transmit_diversity.cpp
│   │       ├── transmit_diversity.h
│   │       ├── vblast.cpp
│   │       └── vblast.h
│   └── variables
│       ├── variables.cpp
│       └── variables.h
└── tx_rx
    ├── avx512
    │   └── main.cpp
    └── x300
        ├── main.cpp
        └── multisim
            └── main.cpp
```


The simulation part is in the `tx_rx/x300` directory :
- `main.cpp` performs transmission and reception of one radio frame
  on the same USRP and uses one decoder among the available ones. It outputs
  The execution times for each slot and the symbol error rate.
- `multisim/main.cpp` performs transmission and reception of multiple radio
  frames on the same USRP and benchmarks all the decoders available. The mean, max.
  and variance of each functions involved are computed.
- `avx512/main.cpp` performs transmission and reception of one radio frame
  on the same USRP and uses the AVX512 SQRD decoder if the code is compiled with AVX512.

`free5gran_utils.h/.cpp` contain the functions reused from free5gran and modified versions. 
Other free5gran algorithms may be used in other files, which is indicated.

The `lib` directory contains additional functions specially implemented for this platform
and for MIMO transmission & reception, and may contain functions reused from free5GRAN.

The `variables/variables.h/.cpp` files contains global variables.

## Usage

### Compilation

Compile options are defined in the CMake file. Some CMake file variables can be set to one to use
different options :
- `USE_AXV2` to use explicit AVX2 optimizations
- `USE_AVX512` to use explicit AVX512 optimizations
- `DEBUG_MODE` to compile in debug mode.

### Macros

Set the following macros in the `tx_rx/x300/main.cpp` file :
- `VBLAST_XXX` to define the type of MIMO decoder :
    - `VBLAST_ZF` for Zero-forcing
    - `VBLAST_QRQ_COL_NORM` for QRD with column norm reordering
    - `VBLAST_SQRD` for SQRD
    - `VBLAST_QRD` for QRD without reordering

By default, all export files (channel coefficients on DMRS, interpolated coefficients,
equalized symbols, detected symbol indexes, sent symbol indexes) are the created in the same directory
as the binary file. `tx_rx/multisim/main.cpp` uses the `RES_PATH` to export them to a specific directory. Set
the macro to the wanted directory.

In the `lib/variables/variables.h` file, `TSC_FREQ` defines the TSC frequency to measure
execution time with `RDTSC/RDTSCP` commands. Verify the TSC frequency with the `sudo dmesg | grep tsc` command and
set the appropriate value. Additionally, execution time can be measured using
standard functions from `std::chrono` or `clock_gettime` by setting the `CLOCK_TYPE_XXX` macro :
- `CLOCK_TYPE_CHRONO` for`std::chrono`
- `CLOCK_TYPE_GETTIME` for `clock_gettime(CLOCK_THREAD_CPUTIME_ID,)`
- `CLOCK_TYPE_ASM` for `RDTSC/RDTSCP`. TSC frequency must be manually set.
- `CLOCK_TYPE_CLOCK` for `clock()`
- `CLOCK_TYPE_GETTIME_MONOTONIC` for `clock_gettime(CLOCK_MONOTONIC,)`

### Export files

Export files include :
- `tx*.txt` : time domain transmitted frames on each TX port
- `non_encoded.txt` : the whole transmit buffer of constellation symbols
- `sending_buffer_symbol_indexes.txt` : the indexes of symbols in the sending buffer
- `rx*.txt` : time domain received signals on each RX port
- `tx*_rx*_coefs.txt` : channel coefficients for all the TX-RX paths
- `tx*_rx*_pilots.txt` : channel coefficients for all the TX-RX paths on DMRS
- `decoded_grid.txt` : detected symbol indexes at the receiver
- `equalized_grid.txt` : equalized symbols at the receiver
- `sync_rx*.txt` : time domain PSS synchronized signals at the receiver.

Execution times for each slot are exported in `logfile.log`. 

When using the `tx_rx/x300/multisim/main.cpp` file, data is exported for each frame and uses the following format :
- `rx*_frame{FRAME_NUMBER}.txt` : time domain received signals on each RX port
- `tx*_rx*_frame{FRAME_NUMBER}_coefs.txt` : channel coefficients for all the TX-RX paths
- `tx*_rx*_frame{FRAME_NUMBER}_pilots.txt` : channel coefficients for all the TX-RX paths on DMRS
- `decoded_grid_{DECODER_TYPE}_frame{FRAME_NUMBER}.txt` : detected symbol indexes at the receiver
- `equalized_grid_{DECODER_TYPE}_frame{FRAME_NUMBER}.txt` : equalized symbols at the receiver
- `sync_rx*_frame{FRAME_NUMBER}.txt` : time domain PSS synchronized signals at the receiver.

They can be plotted using the `analyse.py` file, where parameters must be set manually.
`analyse.py` plot :
- the received frames
- the synchronized frames
- the channel coefficients (magnitude and phase) on DMRS for a given symbol and subcarrier
- the channel coefficients (magnitude and phase) on the whole grid for a given symbol and subcarrier
- the equalized constellation and the computed symbol error rate.

Examples are available in the `plot_exports` directory when retrieving export files from a distant server : 
- `scp_logfile_example.sh` : scp the logfile 
- `scp_results_example.sh` : scp the export files to be plotted 
- `vblast` directory : export files 
- plots in PDFs 
