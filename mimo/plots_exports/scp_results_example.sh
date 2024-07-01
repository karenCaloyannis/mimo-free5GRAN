#!/bin/bash

# Chose the distant result directory 
DIR=" " # set the distant directory 
LOCAL_DIR="./vblast" # Set the local directory 
i=1 # Chose the frame number to be plotted

scp user@distant_ip:${DIR}tx1.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx2.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx3.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx4.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx1_grid.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx2_grid.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx3_grid.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx4_grid.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}non_encoded.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}sending_buffer_symbol_indexes.txt ${LOCAL_DIR}

scp user@distant_ip:${DIR}rx1_frame${i}.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}rx2_frame${i}.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}rx3_frame${i}.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}rx4_frame${i}.txt ${LOCAL_DIR}

scp user@distant_ip:${DIR}tx1_rx1_frame${i}_coefs.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx1_rx2_frame${i}_coefs.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx1_rx3_frame${i}_coefs.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx1_rx4_frame${i}_coefs.txt ${LOCAL_DIR}

scp user@distant_ip:${DIR}tx2_rx1_frame${i}_coefs.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx2_rx2_frame${i}_coefs.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx2_rx3_frame${i}_coefs.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx2_rx4_frame${i}_coefs.txt ${LOCAL_DIR}

scp user@distant_ip:${DIR}tx3_rx1_frame${i}_coefs.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx3_rx2_frame${i}_coefs.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx3_rx3_frame${i}_coefs.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx3_rx4_frame${i}_coefs.txt ${LOCAL_DIR}

scp user@distant_ip:${DIR}tx4_rx1_frame${i}_coefs.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx4_rx2_frame${i}_coefs.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx4_rx3_frame${i}_coefs.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx4_rx4_frame${i}_coefs.txt ${LOCAL_DIR}

scp user@distant_ip:${DIR}tx1_rx1_frame${i}_pilots.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx1_rx2_frame${i}_pilots.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx1_rx3_frame${i}_pilots.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx1_rx4_frame${i}_pilots.txt ${LOCAL_DIR}

scp user@distant_ip:${DIR}tx2_rx1_frame${i}_pilots.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx2_rx2_frame${i}_pilots.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx2_rx3_frame${i}_pilots.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx2_rx4_frame${i}_pilots.txt ${LOCAL_DIR}

scp user@distant_ip:${DIR}tx3_rx1_frame${i}_pilots.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx3_rx2_frame${i}_pilots.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx3_rx3_frame${i}_pilots.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx3_rx4_frame${i}_pilots.txt ${LOCAL_DIR}

scp user@distant_ip:${DIR}tx4_rx1_frame${i}_pilots.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx4_rx2_frame${i}_pilots.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx4_rx3_frame${i}_pilots.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}tx4_rx4_frame${i}_pilots.txt ${LOCAL_DIR}

scp user@distant_ip:${DIR}sync_rx1_frame${i}.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}sync_rx2_frame${i}.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}sync_rx3_frame${i}.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}sync_rx4_frame${i}.txt ${LOCAL_DIR}

scp user@distant_ip:${DIR}decoded_grid_sqrd_frame${i}.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}equalized_grid_sqrd_frame${i}.txt ${LOCAL_DIR}

scp user@distant_ip:${DIR}decoded_grid_qrd_col_norm_frame${i}.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}equalized_grid_qrd_col_norm_frame${i}.txt ${LOCAL_DIR}

scp user@distant_ip:${DIR}decoded_grid_qrd_no_reordering_frame${i}.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}equalized_grid_qrd_no_reordering_frame${i}.txt ${LOCAL_DIR}

scp user@distant_ip:${DIR}decoded_grid_zf_frame${i}.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}equalized_grid_zf_frame${i}.txt ${LOCAL_DIR}

scp user@distant_ip:${DIR}decoded_grid_alamouti_frame${i}.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}equalized_grid_alamouti_frame${i}.txt ${LOCAL_DIR}

scp user@distant_ip:${DIR}decoded_grid_alamouti_frame${i}.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}equalized_grid_alamouti_frame${i}.txt ${LOCAL_DIR}

scp user@distant_ip:${DIR}decoded_grid_siso_frame${i}.txt ${LOCAL_DIR}
scp user@distant_ip:${DIR}equalized_grid_siso_frame${i}.txt ${LOCAL_DIR}

./analyse.py
