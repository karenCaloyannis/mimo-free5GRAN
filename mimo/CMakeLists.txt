cmake_minimum_required(VERSION 3.16.3)
project(usrp_mimo)

set(CMAKE_VERBOSE_MAKEFILE True)
set(CMAKE_CXX_STANDARD 17)

set(USE_AVX512 0) # Set to 1 to use AVX512
set(DEBUG_MODE 0) # Set to 1 for debug
set(USE_AVX2 1) # Set to 1 to use AVX2

if(USE_AVX2)
    add_compile_definitions(AVX2_PROCESSING)
endif()

message("DEBUG_MODE = ${DEBUG_MODE}")
message("USE_AVX2 = ${USE_AVX2}")
message("USE_AVX512 = ${USE_AVX512}")

if(DEBUG_MODE)
    message("debug mode")
    set(CMAKE_BUILD_TYPE Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")
    add_link_options(-lpthread)
    add_compile_options(-g
                        -I/usr/include/
                        -fopenmp
                        -DBOOST_ALL_DYN_LINK -DPARALLEL
                        -march=native)
else()
    message("Release mode")
    set(CMAKE_BUILD_TYPE Release)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Ofast -ffast-math -std=c++17")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Ofast -ffast-math")
    add_link_options(-lpthread)
    add_compile_options(
            -I/usr/include/
            -fopenmp
            -DBOOST_ALL_DYN_LINK -DPARALLEL
            -march=native)
    find_package( Boost )
    find_package( Threads )
    find_package( OpenMP )

endif()

if(USE_AVX512)
    add_compile_options(-msse -mavx -msse3 -mavx2 -mavx512f)
elseif(USE_AVX2)
    add_compile_options(
            -msse -mavx -msse3 -mavx2)
endif()

unset(USE_AVX512 CACHE)
unset(DEBUG_MODE CACHE)
unset(USE_AVX2 CACHE)

add_executable(transmit_x300 tx_rx/x300/main.cpp
        lib/usrp/usrp_x300.h
        lib/usrp/usrp_x300.cpp
        lib/usrp/usrp.h
        lib/usrp/usrp.cpp
        lib/usrp/usrp_b210.h
        lib/usrp/usrp_b210.cpp
        lib/free5gran_utils.h
        lib/free5gran_utils.cpp
        lib/variables/variables.h
        lib/variables/variables.cpp
        lib/utils/channel_estimation/channel_estimation.h
        lib/utils/channel_estimation/channel_estimation.cpp
        lib/utils/channel_mapping/channel_mapping.h
        lib/utils/channel_mapping/channel_mapping.cpp
        lib/utils/mimo/mimo.h
        lib/utils/mimo/mimo.cpp
        lib/utils/mimo/transmit_diversity.h
        lib/utils/mimo/transmit_diversity.cpp
        lib/utils/mimo/vblast.h
        lib/utils/mimo/vblast.cpp
        lib/avx/avx_ops.h)

add_executable(multisim tx_rx/x300/multisim/main.cpp
        lib/usrp/usrp_b210.h
        lib/usrp/usrp_b210.cpp
        lib/usrp/usrp_x300.h
        lib/usrp/usrp_x300.cpp
        lib/usrp/usrp.h
        lib/usrp/usrp.cpp
        lib/free5gran_utils.h
        lib/free5gran_utils.cpp
        lib/variables/variables.h
        lib/variables/variables.cpp
        lib/utils/channel_estimation/channel_estimation.h
        lib/utils/channel_estimation/channel_estimation.cpp
        lib/utils/channel_mapping/channel_mapping.h
        lib/utils/channel_mapping/channel_mapping.cpp
        lib/utils/mimo/mimo.h
        lib/utils/mimo/mimo.cpp
        lib/utils/mimo/transmit_diversity.h
        lib/utils/mimo/transmit_diversity.cpp
        lib/utils/mimo/vblast.h
        lib/utils/mimo/vblast.cpp
        lib/avx/avx_ops.h)

# To test SQRD using AVX512
if(USE_AVX512)
add_executable(avx512_sqrd tx_rx/x300/avx512/main.cpp
                lib/usrp/usrp_b210.h
                lib/usrp/usrp_b210.cpp
                lib/usrp/usrp_x300.h
                lib/usrp/usrp_x300.cpp
                lib/usrp/usrp.h
                lib/usrp/usrp.cpp
                lib/free5gran_utils.h
                lib/free5gran_utils.cpp
                lib/variables/variables.h
                lib/variables/variables.cpp
                lib/utils/channel_estimation/channel_estimation.h
                lib/utils/channel_estimation/channel_estimation.cpp
                lib/utils/channel_mapping/channel_mapping.h
                lib/utils/channel_mapping/channel_mapping.cpp
                lib/utils/mimo/mimo.h
                lib/utils/mimo/mimo.cpp
                lib/utils/mimo/transmit_diversity.h
                lib/utils/mimo/transmit_diversity.cpp
                lib/utils/mimo/vblast.h
                lib/utils/mimo/vblast.cpp
                lib/avx/avx_ops.h)

target_link_libraries(avx512_sqrd fftw3f uhd ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${OpenMP_C_FLAGS} boost_system boost_log boost_thread boost_log_setup)

endif()

target_link_libraries(transmit_x300 fftw3f uhd ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${OpenMP_C_FLAGS} boost_system boost_log boost_thread boost_log_setup)

target_link_libraries(multisim fftw3f uhd ${Boost_LIBRARIES} ${CMAKE_THREAD_LIBS_INIT} ${OpenMP_C_FLAGS} boost_system boost_log boost_thread boost_log_setup)


