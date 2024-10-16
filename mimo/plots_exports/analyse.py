"""
   Copyright 2023-2024 Telecom Paris

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

    This file is a modified version of the analyse.py file.
"""

#!/usr/bin/env python3

import numpy as np
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import traceback
from pathlib import Path 
import cmath
from math import isnan

Fs                      = 15.36e6
scs                     = 30e3 
fft_size                = int(Fs/scs)
print(fft_size)
mu                      = int(scs/15e3 - 1) #numerology index
nb_tx_subframes         = 10
nb_rx_subframes_sync    = nb_tx_subframes 
nb_rx_subframes_unsync  = 2*nb_tx_subframes
nb_slots                = nb_rx_subframes_sync * pow(2, mu) - 1 
nb_dmrs_rx              = 4*nb_slots # number of dmrs per frame in the synchronized signal

dmrs_freq_size = int(fft_size/2) # size of the dmrs sequence 

#specgram_noverlap = 128
#print("noverlap :"+str(specgram_noverlap))

nb_tx_ports = 4
nb_rx_ports = 4

tx_port_indexes = [0, 1, 2, 3]
rx_port_indexes = [0, 1, 2, 3]

res_path = "./vblast/"

# 0 : vblast
# 1 : alamouti 
# 2 : none 
encoding = 0 

#PDSCH length in each slot 
pdsch_length = 14
nb_symbols = int(pdsch_length* (pow(2, scs/15e3 - 1)*nb_rx_subframes_sync - 1)) # number of received PDSCH OFDM symbols

num_pdsch_re_per_slot = pdsch_length * fft_size - dmrs_freq_size * 4 * 2

frame_number = 7 # frame number to be plotted starting from 0 

""" Plot the received Signal on RX1 and RX2 on the same figure """
try:

    rx1 = []
    rx2 = []
    rx3 = []
    rx4 = [] 

    check_rx1_file = Path(res_path+"rx1_frame"+str(frame_number)+".txt")
    check_rx2_file = Path(res_path+"rx2_frame"+str(frame_number)+".txt")
    check_rx3_file = Path(res_path+"rx3_frame"+str(frame_number)+".txt")
    check_rx4_file = Path(res_path+"rx4_frame"+str(frame_number)+".txt")
    
    if(check_rx1_file.is_file()) : 
        rx1_file = open(res_path+"rx1_frame"+str(frame_number)+".txt", "r")
        for x in rx1_file:
            x = x.split("(")[1]
            x = x.split(")")[0]
            re = float(x.split(',')[0])
            im = float(x.split(',')[1])
            rx1.append(complex(re,im))
    
    if(check_rx2_file.is_file()) :
        rx2_file = open(res_path+"rx2_frame"+str(frame_number)+".txt", "r")
        for x in rx2_file:
            x = x.split("(")[1]
            x = x.split(")")[0]
            re = float(x.split(',')[0])
            im = float(x.split(',')[1])
            rx2.append(complex(re,im))
    
    if(check_rx3_file.is_file()) :
        rx3_file = open(res_path+"rx3_frame"+str(frame_number)+".txt", "r")
        for x in rx3_file:
            x = x.split("(")[1]
            x = x.split(")")[0]
            re = float(x.split(',')[0])
            im = float(x.split(',')[1])
            rx3.append(complex(re,im))
    
    if(check_rx4_file.is_file()) :
        rx4_file = open(res_path+"rx4_frame"+str(frame_number)+".txt", "r")
        for x in rx4_file:
            x = x.split("(")[1]
            x = x.split(")")[0]
            re = float(x.split(',')[0])
            im = float(x.split(',')[1])
            rx4.append(complex(re,im))
              
    
    rx_fig, ax_rx = plt.subplots(nb_rx_ports,1, figsize=(15,15))
    
    rx = [rx1, rx2, rx3, rx4]
    
    for i in range(nb_rx_ports) : 
        ax_rx[rx_port_indexes[i]].specgram(rx[rx_port_indexes[i]], NFFT=fft_size, Fs=Fs) #, noverlap = specgram_noverlap)
        ax_rx[rx_port_indexes[i]].set_xticks(np.arange(0, 0.001*nb_rx_subframes_unsync, 0.001))
        ax_rx[rx_port_indexes[i]].tick_params(axis='x', labelsize=8)
        ax_rx[rx_port_indexes[i]].set_title("Received Frame on rx"+str(i)+ "(frame " + str(frame_number) +")")
        ax_rx[rx_port_indexes[i]].set_ylim(-Fs/2, Fs/2)
    

    spec = ax_rx[0].specgram(rx2, NFFT=fft_size, Fs=Fs) #, noverlap = specgram_noverlap)
    
    cbar_ax = rx_fig.add_axes([1, 0.2, 0.05, 0.7])
    
    plt.tight_layout()
    
    plt.colorbar(spec[3], cax=cbar_ax)
    
    plt.savefig("rx_specgrams_frame"+str(frame_number)+".pdf", bbox_inches='tight', pad_inches=0.5)
    plt.close(rx_fig)

except Exception:
    traceback.print_exc()

""" Plot the synchronized Signal on RX1 and RX2 """
try :
    
    sync_rx1 = []
    sync_rx2 = []
    sync_rx3 = []
    sync_rx4 = []

    check_rx1_file = Path(res_path+"sync_rx1_frame"+str(frame_number)+".txt")
    check_rx2_file = Path(res_path+"sync_rx2_frame"+str(frame_number)+".txt")
    check_rx3_file = Path(res_path+"sync_rx3_frame"+str(frame_number)+".txt")
    check_rx4_file = Path(res_path+"sync_rx4_frame"+str(frame_number)+".txt")

    if(check_rx1_file.is_file()) :
        sync_rx1_file = open(res_path+"sync_rx1_frame"+str(frame_number)+".txt", "r")
        for x in sync_rx1_file:
            x = x.split("(")[1]
            x = x.split(")")[0]
            re = float(x.split(',')[0])
            im = float(x.split(',')[1])
            sync_rx1.append(complex(re,im))
    
    if(check_rx2_file.is_file()) :
        sync_rx2_file = open(res_path+"sync_rx2_frame"+str(frame_number)+".txt", "r")
        for x in sync_rx2_file:
            x = x.split("(")[1]
            x = x.split(")")[0]
            re = float(x.split(',')[0])
            im = float(x.split(',')[1])
            sync_rx2.append(complex(re,im))
    
    if(check_rx3_file.is_file()) :
        sync_rx3_file = open(res_path+"sync_rx3_frame"+str(frame_number)+".txt", "r")
        for x in sync_rx3_file:
            x = x.split("(")[1]
            x = x.split(")")[0]
            re = float(x.split(',')[0])
            im = float(x.split(',')[1])
            sync_rx3.append(complex(re,im))
    
    if(check_rx4_file.is_file()) :
        sync_rx4_file = open(res_path+"sync_rx4_frame"+str(frame_number)+".txt", "r")
        for x in sync_rx4_file:
            x = x.split("(")[1]
            x = x.split(")")[0]
            re = float(x.split(',')[0])
            im = float(x.split(',')[1])
            sync_rx4.append(complex(re,im))

    sync_rx = [sync_rx1, 
               sync_rx2,
               sync_rx3, 
               sync_rx4]

    sync_rx_fig, ax_sync_rx = plt.subplots(nb_rx_ports, 1, figsize = (15,15))
    
    for i in range(nb_rx_ports) : 
        ax_sync_rx[rx_port_indexes[i]].specgram(sync_rx[rx_port_indexes[i]], NFFT=fft_size, Fs=Fs)#, noverlap = specgram_noverlap)
        ax_sync_rx[rx_port_indexes[i]].set_xticks(np.arange(0, 0.001*nb_rx_subframes_sync, 0.001))
        ax_sync_rx[rx_port_indexes[i]].set_title("synchronized grid on RX"+str(i))
        ax_sync_rx[rx_port_indexes[i]].set_ylim(-Fs/2, Fs/2)
    
    
    spec = ax_sync_rx[0].specgram(sync_rx2, NFFT=fft_size, Fs=Fs) #, noverlap = specgram_noverlap)
    
    cbar_ax = sync_rx_fig.add_axes([1, 0.2, 0.05, 0.7])
    
    plt.tight_layout()
    
    plt.colorbar(spec[3], cax=cbar_ax)
    
    plt.savefig("sync_rx_frame"+str(frame_number)+".pdf", bbox_inches='tight', pad_inches=0.5)
    plt.close(sync_rx_fig)

except Exception:
    traceback.print_exc()

""" Plot the transmitted signal on each transmit antenna """
try :
    tx1 = []
    tx2 = []
    tx3 = []
    tx4 = []
	
    check_tx1_file = Path(res_path+"tx1.txt")
    check_tx2_file = Path(res_path+"tx2.txt")
    check_tx3_file = Path(res_path+"tx3.txt")
    check_tx4_file = Path(res_path+"tx4.txt")
    
    if(check_tx1_file.is_file()) :
        tx1_file = open(res_path+"tx1.txt", "r")
        for x in tx1_file:
            x = x.split("(")[1]
            x = x.split(")")[0]
            re = float(x.split(',')[0])
            im = float(x.split(',')[1])
            tx1.append(complex(re,im))
    
    if(check_tx2_file.is_file()) :
        tx2_file = open(res_path+"tx2.txt", "r")
        for x in tx2_file:
            x = x.split("(")[1]
            x = x.split(")")[0]
            re = float(x.split(',')[0])
            im = float(x.split(',')[1])
            tx2.append(complex(re,im))
    
    if(check_tx3_file.is_file()) :
        tx3_file = open(res_path+"tx3.txt", "r")
        for x in tx3_file:
            x = x.split("(")[1]
            x = x.split(")")[0]
            re = float(x.split(',')[0])
            im = float(x.split(',')[1])
            tx3.append(complex(re,im))
    
    if(check_tx4_file.is_file()) :
        tx4_file = open(res_path+"tx4.txt", "r")
        for x in tx4_file:
            x = x.split("(")[1]
            x = x.split(")")[0]
            re = float(x.split(',')[0])
            im = float(x.split(',')[1])
            tx4.append(complex(re,im))

    tx = [tx1, tx2, tx3, tx4]

    tx_fig, ax_tx = plt.subplots(nb_tx_ports,1, figsize=(15, 15))
    
    for i in range(nb_tx_ports) : 
        ax_tx[tx_port_indexes[i]].specgram(tx[tx_port_indexes[i]], NFFT=256, Fs=Fs) #, noverlap = specgram_noverlap)
        ax_tx[tx_port_indexes[i]].set_xticks(np.arange(0, 0.001*nb_tx_subframes, 0.001))
        ax_tx[tx_port_indexes[i]].set_title("Baseband signal on TX"+str(i))
        ax_tx[tx_port_indexes[i]].set_ylim(-Fs/2, Fs/2)
    
    spec = ax_tx[0].specgram(tx1, NFFT=256, Fs=Fs) #, noverlap = specgram_noverlap)
    
    cbar_ax = tx_fig.add_axes([1, 0.2, 0.05, 0.7])
    
    plt.tight_layout()
    
    plt.colorbar(spec[3], cax=cbar_ax)
    
    plt.xlabel("time [s]")
    plt.ylabel("frequency [Hz]")
    
    plt.savefig("tx.pdf", bbox_inches='tight', pad_inches=0.5)
    plt.close(tx_fig)

except Exception:
    traceback.print_exc()

""" Plot DMRS coefficients on each path """
try :
    
    print("Plotting DMRS coefs ... ")
    
    interp = "freq" # freq to show frequency domain  values on 1 DMRS symbol, otherwise show time domain values
    sc_dmrs = 50# chosen subcarrier to show time domain values 
    dmrs = 6 # index of the DMRS chosen to show frequency domain values
    nb_paths = nb_tx_ports * nb_rx_ports
    
    lines = list()
    
    for i in range(nb_tx_ports): 
        lines.append(list())
        for j in range(nb_rx_ports):
            checkfile = Path(res_path+"tx"+str(i+1)+"_rx"+str(j+1)+"_frame"+str(frame_number)+"_pilots.txt")
            if(checkfile.is_file()):
                lines[i].append(open(res_path+"tx"+str(i+1)+"_rx"+str(j+1)+"_frame"+str(frame_number)+"_pilots.txt").readlines())
                
    
    """
    tx1_rx1_file  = open(res_path+"tx1_rx1_frame"+str(frame_number)+"_pilots.txt")
    tx1_rx2_file  = open(res_path+"tx1_rx2_frame"+str(frame_number)+"_pilots.txt")
    tx1_rx3_file  = open(res_path+"tx1_rx3_frame"+str(frame_number)+"_pilots.txt")
    tx1_rx4_file  = open(res_path+"tx1_rx4_frame"+str(frame_number)+"_pilots.txt")
    
    tx2_rx1_file  = open(res_path+"tx2_rx1_frame"+str(frame_number)+"_pilots.txt")
    tx2_rx2_file  = open(res_path+"tx2_rx2_frame"+str(frame_number)+"_pilots.txt")
    tx2_rx3_file  = open(res_path+"tx2_rx3_frame"+str(frame_number)+"_pilots.txt")
    tx2_rx4_file  = open(res_path+"tx2_rx4_frame"+str(frame_number)+"_pilots.txt")
    
    tx3_rx1_file  = open(res_path+"tx3_rx1_frame"+str(frame_number)+"_pilots.txt")
    tx3_rx2_file  = open(res_path+"tx3_rx2_frame"+str(frame_number)+"_pilots.txt")
    tx3_rx3_file  = open(res_path+"tx3_rx3_frame"+str(frame_number)+"_pilots.txt")
    tx3_rx4_file  = open(res_path+"tx3_rx4_frame"+str(frame_number)+"_pilots.txt")
    
    tx4_rx1_file  = open(res_path+"tx4_rx1_frame"+str(frame_number)+"_pilots.txt")
    tx4_rx2_file  = open(res_path+"tx4_rx2_frame"+str(frame_number)+"_pilots.txt")
    tx4_rx3_file  = open(res_path+"tx4_rx3_frame"+str(frame_number)+"_pilots.txt")
    tx4_rx4_file  = open(res_path+"tx4_rx4_frame"+str(frame_number)+"_pilots.txt")
    
    
    
    lines_tx1_rx1 = tx1_rx1_file.readlines()
    lines_tx1_rx2 = tx1_rx2_file.readlines()
    lines_tx1_rx3 = tx1_rx3_file.readlines()
    lines_tx1_rx4 = tx1_rx4_file.readlines()
    
    lines_tx2_rx1 = tx2_rx1_file.readlines()
    lines_tx2_rx2 = tx2_rx2_file.readlines()
    lines_tx2_rx3 = tx2_rx3_file.readlines()
    lines_tx2_rx4 = tx2_rx4_file.readlines()
    
    lines_tx3_rx1 = tx3_rx1_file.readlines()
    lines_tx3_rx2 = tx3_rx2_file.readlines()
    lines_tx3_rx3 = tx3_rx3_file.readlines()
    lines_tx3_rx4 = tx3_rx4_file.readlines()
    
    lines_tx4_rx1 = tx4_rx1_file.readlines()
    lines_tx4_rx2 = tx4_rx2_file.readlines()
    lines_tx4_rx3 = tx4_rx3_file.readlines()
    lines_tx4_rx4 = tx4_rx4_file.readlines()
    
    lines = [[lines_tx1_rx1,
              lines_tx1_rx2,
              lines_tx1_rx3, 
              lines_tx1_rx4], 
             
             [lines_tx2_rx1,
              lines_tx2_rx2,
              lines_tx2_rx3, 
              lines_tx2_rx4], 
             
             [lines_tx3_rx1,
              lines_tx3_rx2,
              lines_tx3_rx3, 
              lines_tx2_rx4], 
             
             [lines_tx4_rx1,
              lines_tx4_rx2,
              lines_tx4_rx3, 
              lines_tx4_rx4]]
    """
    
    paths_fig, ax_paths = plt.subplots(nb_paths, 2, figsize=(60,60))
    
    titles = [[["| h00 |", "h00 phase"],
               ["| h01 |", "h01 phase"],
               ["| h02 |", "h02 phase"],
               ["| h03 |", "h03 phase"]], 
              [["| h10 |", "h10 phase"],
               ["| h11 |", "h11 phase"],
               ["| h12 |", "h12 phase"],
               ["| h13 |", "h13 phase"]],        
              [["| h20 |", "h20 phase"],
               ["| h21 |", "h21 phase"],
               ["| h22 |", "h22 phase"],
               ["| h23 |", "h23 phase"]],
              [["| h30 |", "h30 phase"],
               ["| h31 |", "h31 phase"],
               ["| h32 |", "h32 phase"],
               ["| h33 |", "h33 phase"]]]
    

    tx_rx_paths = np.zeros((nb_tx_ports, nb_rx_ports, 2, dmrs_freq_size))
    phases = np.zeros((nb_tx_ports, nb_rx_ports, dmrs_freq_size))
    
    for i in range(dmrs_freq_size):
        for j in range(nb_tx_ports):
            for k in range(nb_rx_ports):
                x = lines[tx_port_indexes[j]][rx_port_indexes[k]][dmrs*dmrs_freq_size + i]
                x = x.split("(")[1]
                x = x.split(")")[0]
                re = float(x.split(',')[0])
                im = float(x.split(',')[1])
                tx_rx_paths[tx_port_indexes[j], rx_port_indexes[k], 0, i] = re 
                tx_rx_paths[tx_port_indexes[j], rx_port_indexes[k], 1, i] = im
                #Normalize before computing the arccos 
                norm = np.sqrt(re**2+im**2)
                re = re/norm
                im = im/norm
                #Compute the phase of each complex value 
                #angle = np.arccos(re) 
                #angle = np.arctan2(im, re)
                angle = cmath.phase(complex(re, im))
                #if(im < 0):
                #    angle = 2*np.pi + angle
                phases[tx_port_indexes[j], rx_port_indexes[k], i] = angle
    
    x_values = np.arange(0, dmrs_freq_size)
    
    for i in range(nb_rx_ports): 
        for j in range(nb_tx_ports):
            # plot the squared modulus
            ax_paths[i * nb_tx_ports + j][0].plot(x_values, np.sqrt(
                tx_rx_paths[tx_port_indexes[j], rx_port_indexes[i], 0]**2 + 
                tx_rx_paths[tx_port_indexes[j], rx_port_indexes[i], 1]**2))
            
            # plot the phase 
            #ax_paths[i][1].plot(x_values, np.arctan(tx_rx_paths[i, 1, :]/tx_rx_paths[i, 0, :]))
            ax_paths[i * nb_rx_ports + j][1].plot(x_values, phases[tx_port_indexes[j], rx_port_indexes[i], :])
            
            ax_paths[i * nb_rx_ports + j][0].set_title(titles[rx_port_indexes[i]][tx_port_indexes[j]][0])
            
            ax_paths[i * nb_rx_ports + j][1].set_title(titles[rx_port_indexes[i]][tx_port_indexes[j]][1])
            
            ax_paths[i * nb_rx_ports + j][0].set_ylim(0, 1.5*max(np.sqrt(
                tx_rx_paths[tx_port_indexes[j], rx_port_indexes[i], 0]**2 + 
                tx_rx_paths[tx_port_indexes[j], rx_port_indexes[i], 1]**2)))

            #ax_paths[i][0].set_ylim(-10, 10)
            
            #ax_paths[i][1].set_ylim(-np.pi/2, np.pi/2)
            ax_paths[j * nb_tx_ports + i][1].set_ylim(-np.pi, np.pi)
            #ax_paths[i][1].set_ylim(-10, 10)
    
    plt.tight_layout()
    plt.savefig("dmrs_freq_domain_frame"+str(frame_number)+".pdf", bbox_inches='tight', pad_inches=0.5)
    plt.close(paths_fig)
    
    paths_fig, ax_paths = plt.subplots(nb_paths, 2, figsize=(60,60))

    tx_rx_paths = np.zeros((nb_tx_ports, nb_rx_ports, 2, nb_dmrs_rx))
    phases = np.zeros((nb_tx_ports, nb_rx_ports, nb_dmrs_rx))

    for i in range(nb_dmrs_rx):
        for j in range(nb_tx_ports):
            for k in range(nb_rx_ports):
                x = lines[tx_port_indexes[j]][rx_port_indexes[k]][i*dmrs_freq_size + sc_dmrs]
                x = x.split("(")[1]
                x = x.split(")")[0]
                re = float(x.split(',')[0])
                im = float(x.split(',')[1])
                tx_rx_paths[tx_port_indexes[j], rx_port_indexes[k], 0, i] = re 
                tx_rx_paths[tx_port_indexes[j], rx_port_indexes[k], 1, i] = im
                #Normalize before computing the arccos 
                norm = np.sqrt(re**2+im**2)
                re = re/norm
                im = im/norm
                #Compute the phase of each complex value 
                #angle = np.arccos(re) 
                #angle = np.arctan2(im,re)
                angle = cmath.phase(complex(re, im))
                #if(im < 0):
                #    angle = 2*np.pi + angle
                phases[tx_port_indexes[j], rx_port_indexes[k], i] = angle

    x_values = np.arange(0, nb_dmrs_rx)
    
    for i in range(nb_rx_ports): 
        for j in range(nb_tx_ports):
            # plot the squared modulus
            ax_paths[i * nb_tx_ports + j][0].plot(x_values, np.sqrt(
                tx_rx_paths[tx_port_indexes[j], rx_port_indexes[i], 0]**2 + 
                tx_rx_paths[tx_port_indexes[j], rx_port_indexes[i], 1]**2))
            
            # plot the phase 
            #ax_paths[i][1].plot(x_values, np.arctan(tx_rx_paths[i, 1, :]/tx_rx_paths[i, 0, :]))
            ax_paths[i * nb_rx_ports + j][1].plot(x_values, phases[tx_port_indexes[j], rx_port_indexes[i], :])
            
            ax_paths[i * nb_rx_ports + j][0].set_title(titles[rx_port_indexes[i]][tx_port_indexes[j]][0])
            
            ax_paths[i * nb_rx_ports + j][1].set_title(titles[rx_port_indexes[i]][tx_port_indexes[j]][1])
            
            ax_paths[i * nb_rx_ports + j][0].set_ylim(0, 1.5*max(np.sqrt(
                tx_rx_paths[tx_port_indexes[j], rx_port_indexes[i], 0]**2 + 
                tx_rx_paths[tx_port_indexes[j], rx_port_indexes[i], 1]**2)))

            #ax_paths[i][0].set_ylim(-10, 10)
            
            #ax_paths[i][1].set_ylim(-np.pi/2, np.pi/2)
            ax_paths[j * nb_tx_ports + i][1].set_ylim(-np.pi, np.pi)
            #ax_paths[i][1].set_ylim(-10, 10)
    
    plt.tight_layout()
    plt.savefig("dmrs_time_domain_frame"+str(frame_number)+".pdf", bbox_inches='tight', pad_inches=0.5)
    plt.close(paths_fig)


except Exception:
    traceback.print_exc()

""" Plot interpolated coefficients on each path """
try :
    
    print("Plotting interp coefs")
    
    interp = "freq"
    symbol = 24 # chosen symbol to plot the frequency domain interpolation 
    #sc = 2*sc_dmrs + 1 # chosen subcarrier to plot time domain interpolation 
    sc = 57
    
    lines_interp = list() 
        
    for i in range(nb_tx_ports): 
        lines_interp.append(list())
        for j in range(nb_rx_ports):
            checkfile = Path(res_path+"tx"+str(i+1)+"_rx"+str(j+1)+"_frame"+str(frame_number)+"_coefs.txt")
            if(checkfile.is_file()):
                lines_interp[i].append(open(res_path+"tx"+str(i+1)+"_rx"+str(j+1)+"_frame"+str(frame_number)+"_coefs.txt").readlines())
                
    """
    tx1_rx1_interp_file = open(res_path+"tx1_rx1_frame"+str(frame_number)+"_coefs.txt")
    tx1_rx2_interp_file = open(res_path+"tx1_rx2_frame"+str(frame_number)+"_coefs.txt")
    tx1_rx3_interp_file = open(res_path+"tx1_rx3_frame"+str(frame_number)+"_coefs.txt")
    tx1_rx4_interp_file = open(res_path+"tx1_rx4_frame"+str(frame_number)+"_coefs.txt")
    
    tx2_rx1_interp_file = open(res_path+"tx2_rx1_frame"+str(frame_number)+"_coefs.txt")
    tx2_rx2_interp_file = open(res_path+"tx2_rx2_frame"+str(frame_number)+"_coefs.txt")
    tx2_rx3_interp_file = open(res_path+"tx2_rx3_frame"+str(frame_number)+"_coefs.txt")
    tx2_rx4_interp_file = open(res_path+"tx2_rx4_frame"+str(frame_number)+"_coefs.txt")
    
    tx3_rx1_interp_file = open(res_path+"tx3_rx1_frame"+str(frame_number)+"_coefs.txt")
    tx3_rx2_interp_file = open(res_path+"tx3_rx2_frame"+str(frame_number)+"_coefs.txt")
    tx3_rx3_interp_file = open(res_path+"tx3_rx3_frame"+str(frame_number)+"_coefs.txt")
    tx3_rx4_interp_file = open(res_path+"tx3_rx4_frame"+str(frame_number)+"_coefs.txt")
    
    tx4_rx1_interp_file = open(res_path+"tx4_rx1_frame"+str(frame_number)+"_coefs.txt")
    tx4_rx2_interp_file = open(res_path+"tx4_rx2_frame"+str(frame_number)+"_coefs.txt")
    tx4_rx3_interp_file = open(res_path+"tx4_rx3_frame"+str(frame_number)+"_coefs.txt")
    tx4_rx4_interp_file = open(res_path+"tx4_rx4_frame"+str(frame_number)+"_coefs.txt")
    
    lines_tx1_rx1_interp = tx1_rx1_interp_file.readlines()
    lines_tx1_rx2_interp = tx1_rx2_interp_file.readlines()
    lines_tx1_rx3_interp = tx1_rx3_interp_file.readlines()
    lines_tx1_rx4_interp = tx1_rx4_interp_file.readlines()
    
    lines_tx2_rx1_interp = tx2_rx1_interp_file.readlines()
    lines_tx2_rx2_interp = tx2_rx2_interp_file.readlines()
    lines_tx2_rx3_interp = tx2_rx3_interp_file.readlines()
    lines_tx2_rx4_interp = tx2_rx4_interp_file.readlines()
    
    lines_tx3_rx1_interp = tx3_rx1_interp_file.readlines()
    lines_tx3_rx2_interp = tx3_rx2_interp_file.readlines()
    lines_tx3_rx3_interp = tx3_rx3_interp_file.readlines()
    lines_tx3_rx4_interp = tx3_rx4_interp_file.readlines()
    
    lines_tx4_rx1_interp = tx4_rx1_interp_file.readlines()
    lines_tx4_rx2_interp = tx4_rx2_interp_file.readlines()
    lines_tx4_rx3_interp = tx4_rx3_interp_file.readlines()
    lines_tx4_rx4_interp = tx4_rx4_interp_file.readlines()
    
    lines_interp = [[lines_tx1_rx1_interp,
                     lines_tx1_rx2_interp,
                     lines_tx1_rx3_interp, 
                     lines_tx1_rx4_interp], 
                     
                     [lines_tx2_rx1_interp,
                      lines_tx2_rx2_interp,
                      lines_tx2_rx3_interp, 
                      lines_tx2_rx4_interp], 
                     
                     [lines_tx3_rx1_interp,
                      lines_tx3_rx2_interp,
                      lines_tx3_rx3_interp, 
                      lines_tx2_rx4_interp], 
                     
                     [lines_tx4_rx1_interp,
                      lines_tx4_rx2_interp,
                      lines_tx4_rx3_interp, 
                      lines_tx4_rx4_interp]]
    """
  
    interp_fig, ax_interp = plt.subplots(nb_paths, 2, figsize=(60,60))
    
    interp_fig_time, ax_interp_time = plt.subplots(nb_paths, 2, figsize=(60,60))
    
    titles = [[["| h00 |", "h00 phase"],
               ["| h01 |", "h01 phase"],
               ["| h02 |", "h02 phase"],
               ["| h03 |", "h03 phase"]], 
              [["| h10 |", "h10 phase"],
               ["| h11 |", "h11 phase"],
               ["| h12 |", "h12 phase"],
               ["| h13 |", "h13 phase"]],        
              [["| h20 |", "h20 phase"],
               ["| h21 |", "h21 phase"],
               ["| h22 |", "h22 phase"],
               ["| h23 |", "h23 phase"]],
              [["| h30 |", "h30 phase"],
               ["| h31 |", "h31 phase"],
               ["| h32 |", "h32 phase"],
               ["| h33 |", "h33 phase"]]]
    
    """ Frequency """

    tx_rx_interp = np.zeros((nb_tx_ports, nb_rx_ports, 2, fft_size))
    phases = np.zeros((nb_tx_ports, nb_rx_ports, fft_size))

    for i in range(fft_size):
        for j in range(nb_tx_ports):
            for k in range(nb_rx_ports):
                x = lines_interp[tx_port_indexes[j]][rx_port_indexes[k]][symbol*fft_size + i]
                x = x.split("(")[1]
                x = x.split(")")[0]
                re = float(x.split(',')[0])
                im = float(x.split(',')[1])
                tx_rx_interp[tx_port_indexes[j], rx_port_indexes[k], 0, i] = re 
                tx_rx_interp[tx_port_indexes[j], rx_port_indexes[k], 1, i] = im 
                #Normalize before computing the arccos 
                norm = np.sqrt(re**2+im**2)
                re = re/norm
                im = im/norm
                #Compute the phase of each complex value 
                #angle = np.arccos(re) 
                angle = cmath.phase(complex(re, im))
                #angle = np.arctan2(im, re)
                #if(im < 0):
                #    angle = 2*np.pi - angle
                phases[tx_port_indexes[j], rx_port_indexes[k], i] = angle

    x_values = np.arange(0, fft_size)
    x_ticks  = np.arange(0, fft_size, 2)
    
    for i in range(nb_rx_ports):
        for j in range(nb_tx_ports):
            # plot the squared modulus
            ax_interp[i*nb_tx_ports + j][0].plot(x_values, np.sqrt(
                (tx_rx_interp[tx_port_indexes[j], rx_port_indexes[i], 0]**2 + 
                 tx_rx_interp[tx_port_indexes[j], rx_port_indexes[i], 1]**2)))
            
            # plot the phase 
            #ax_interp[i][1].plot(x_values, np.arctan(tx_rx_interp[i, 1]/tx_rx_interp[i, 0]))
            ax_interp[i * nb_rx_ports + j][1].plot(x_values, phases[tx_port_indexes[j], rx_port_indexes[i], :])
            
            ax_interp[i * nb_rx_ports + j][0].set_xticks(x_ticks)
            ax_interp[i * nb_rx_ports + j][0].tick_params(axis='x', labelsize=8)
            ax_interp[i * nb_rx_ports + j][0].grid()
            
            ax_interp[i * nb_rx_ports + j][1].set_xticks(x_ticks)
            ax_interp[i * nb_rx_ports + j][1].tick_params(axis='x', labelsize=8)
            ax_interp[i * nb_rx_ports + j][1].grid()
            
            ax_interp[i * nb_rx_ports + j][0].set_title(titles[rx_port_indexes[i]][tx_port_indexes[j]][0])
            ax_interp[i * nb_rx_ports + j][1].set_title(titles[rx_port_indexes[i]][tx_port_indexes[j]][1])
            
            ax_interp[i * nb_rx_ports + j][0].set_ylim(0, 1.5*max(np.sqrt(
                (tx_rx_interp[tx_port_indexes[j], rx_port_indexes[i], 0]**2 + 
                 tx_rx_interp[tx_port_indexes[j], rx_port_indexes[i], 1]**2))))
            
            #ax_interp[i][1].set_ylim(-np.pi/2, np.pi/2)
            ax_interp[i * nb_rx_ports + j][1].set_ylim(-np.pi, np.pi)
    
    plt.tight_layout()
    
    interp_fig.savefig("interp_freq_domain_frame"+str(frame_number)+".pdf", bbox_inches='tight', pad_inches=0.5)
    plt.close(interp_fig)
    
    """ Time """
    tx_rx_interp = np.zeros((nb_tx_ports, nb_rx_ports, 2, nb_symbols))
    phases = np.zeros((nb_tx_ports, nb_rx_ports, nb_symbols))
    
    for i in range(nb_symbols):
        for j in range(nb_tx_ports):
            for k in range(nb_rx_ports):
                 x = lines_interp[tx_port_indexes[j]][rx_port_indexes[k]][i*fft_size + sc]
                 x = x.split("(")[1]
                 x = x.split(")")[0]
                 re = float(x.split(',')[0])
                 im = float(x.split(',')[1])
                 tx_rx_interp[tx_port_indexes[j], rx_port_indexes[k], 0, i] = re 
                 tx_rx_interp[tx_port_indexes[j], rx_port_indexes[k], 1, i] = im
                 #Normalize before computing the arccos 
                 norm = np.sqrt(re**2+im**2)
                 re = re/norm
                 im = im/norm
                 #Compute the phase of each complex value 
                 #angle = np.arccos(re) 
                 angle = cmath.phase(complex(re, im))
                 #if(im < 0):
                 #    angle = 2*np.pi - angle
                 phases[tx_port_indexes[j], rx_port_indexes[k], i] = angle
                    

    x_values = np.arange(0, nb_symbols)
    x_ticks  = np.arange(0, nb_symbols, 1)
        
    for i in range(nb_rx_ports):
        for j in range(nb_tx_ports):
            # plot the squared modulus
            ax_interp_time[i*nb_tx_ports + j][0].plot(x_values, np.sqrt(
                (tx_rx_interp[tx_port_indexes[j], rx_port_indexes[i], 0]**2 + 
                 tx_rx_interp[tx_port_indexes[j], rx_port_indexes[i], 1]**2)))
            
            # plot the phase 
            #ax_interp[i][1].plot(x_values, np.arctan(tx_rx_interp[i, 1]/tx_rx_interp[i, 0]))
            ax_interp_time[i * nb_rx_ports + j][1].plot(x_values, phases[tx_port_indexes[j], rx_port_indexes[i], :])
            
            ax_interp_time[i * nb_rx_ports + j][0].set_xticks(x_ticks)
            ax_interp_time[i * nb_rx_ports + j][0].tick_params(axis='x', labelsize=8)
            ax_interp_time[i * nb_rx_ports + j][0].grid()
            
            ax_interp_time[i * nb_rx_ports + j][1].set_xticks(x_ticks)
            ax_interp_time[i * nb_rx_ports + j][1].tick_params(axis='x', labelsize=8)
            ax_interp_time[i * nb_rx_ports + j][1].grid()
            
            ax_interp_time[i * nb_rx_ports + j][0].set_title(titles[rx_port_indexes[i]][tx_port_indexes[j]][0])
            ax_interp_time[i * nb_rx_ports + j][1].set_title(titles[rx_port_indexes[i]][tx_port_indexes[j]][1])
            
            ax_interp_time[i * nb_rx_ports + j][0].set_ylim(0, 1.5*max(np.sqrt(
                (tx_rx_interp[tx_port_indexes[j], rx_port_indexes[i], 0]**2 + 
                 tx_rx_interp[tx_port_indexes[j], rx_port_indexes[i], 1]**2))))
            
            #ax_interp[i][1].set_ylim(-np.pi/2, np.pi/2)
            ax_interp_time[i * nb_rx_ports + j][1].set_ylim(-np.pi, np.pi)
    
    interp_fig_time.tight_layout()
    
    interp_fig_time.savefig("interp_time_domain_frame"+str(frame_number)+".pdf", bbox_inches='tight', pad_inches=0.5)
    plt.close(interp_fig_time)

except Exception:
    traceback.print_exc()

""" Plot constellation of equalized symbols  for each detector and compute the symbol error rate """

try :
    
    string_detectors = []
    
    if(encoding == 0) : # vblast 
        string_detectors =["sqrd", "qrd_col_norm", "zf", "qrd_no_reordering", "alamouti", "siso"]
    elif (encoding == 1): # alamouti 
        string_detectors =["alamouti"]
    elif (encoding == 0): # none 
        string_detectors = ["siso"]
        
    rx_constellation = np.zeros((2, 0))
    tx_constellation = np.zeros((2, 0))
    
    tx_constellation_file = open(res_path+"non_encoded.txt")
    decoded_grid_file     = open(res_path+"sending_buffer_symbol_indexes.txt")
    decoded_grid_lines = decoded_grid_file.readlines();

    sent_symbols    = np.zeros(len(decoded_grid_lines))
    
    for x in tx_constellation_file:
        x = x.split("(")[1]
        x = x.split(")")[0]
        re = float(x.split(',')[0])
        im = float(x.split(',')[1])
        if (not isnan(re)) and (not isnan(im)):
           tx_constellation = np.c_[tx_constellation, [[re],[im]]]

    for detector in string_detectors:
        
        print("Plotting equalized constellation for "+detector)
        rx_constellation_file = open(res_path+"equalized_grid_"+detector+"_frame"+str(frame_number)+".txt")
    
        
        #line = 1
        for x in rx_constellation_file:
            x = x.split("(")[1]
            x = x.split(")")[0]
            re = float(x.split(',')[0])
            im = float(x.split(',')[1])
            rx_constellation = np.c_[rx_constellation, [[re], [im]]]
            #line += 1
            #if(re > 5) or (im >5) : 
                #print(re)
                #print(im)
                #print(line)
                #print("--")
    
        constellation_fig = plt.figure()
        
        plt.xlim(min(np.concatenate([rx_constellation[0], tx_constellation[0]]))*1.5, 
                 max(np.concatenate([rx_constellation[0], tx_constellation[0]]))*1.5)
        plt.ylim(min(np.concatenate([rx_constellation[1], tx_constellation[1]]))*1.5, 
                 max(np.concatenate([rx_constellation[1], tx_constellation[1]]))*1.5)
        """
        plt.xlim(-2, 2)
        plt.ylim(-2, 2)
        """
        
        plt.grid()
        colors = []
        
        for i in range(len(decoded_grid_lines)):
            sent_symbols[i] = int(decoded_grid_lines[i])
        
        for i, label in enumerate(decoded_grid_lines):
            if(sent_symbols[i]==0): 
                colors.append("blue")
            elif(sent_symbols[i]==1) :
                colors.append("green")
            elif (sent_symbols[i]==2):
                colors.append("orange")
            else:
                colors.append("black")
    
        
        plt.scatter(rx_constellation[0, :], rx_constellation[1, :], c=colors, marker='+')
        
        detected_symbols_file = open(res_path+"decoded_grid_"+detector+"_frame"+str(frame_number)+".txt")
        detected_symbols_lines = detected_symbols_file.readlines()
        
        print("Number of sent symbols : " + str(len(sent_symbols)))
        print("Number of received symbols : " + str(len(detected_symbols_lines)))
        
        """ compute the symbol error rate """
        symbol_error_rate = 0
        detected_symbols = np.zeros(len(detected_symbols_lines))
        for i in range(len(detected_symbols_lines)):
            detected_symbols[i] = int(detected_symbols_lines[i])
        symbol_error_rate = np.sum(detected_symbols != sent_symbols)
        symbol_error_rate /= len(detected_symbols_lines)
        print("SYMBOL ERROR RATE : "+str(symbol_error_rate))
        
        plt.scatter(tx_constellation[0], tx_constellation[1], c='red', label = "transmitted constellation")
        
        #add legend manually 
        handles, labels = plt.gca().get_legend_handles_labels()
        scatter_0 = Line2D([], [], marker = '+', color = "blue", linestyle='None', label="true symbol index : symbol #0")
        scatter_1 = Line2D([], [], marker = '+', color = "green", linestyle='None', label="true symbol index : symbol #1")
        scatter_2 = Line2D([], [], marker = '+', color = "orange", linestyle='None', label="true symbol index : symbol #2")
        scatter_3 = Line2D([], [], marker = '+', color = "black", linestyle='None', label="true symbol index : symbol #3")
        
        handles.extend([scatter_0, scatter_1, scatter_2, scatter_3])
        
        #plt.xlim([-1, 1])
        #plt.ylim([-1, 1])
        
        plt.title("Received vs. transmitted constellation ("+str(detector)+")")
        plt.legend(loc='best', bbox_to_anchor=(1, 0., 0., 1), handles=handles)
        plt.savefig("equalized_constellation_"+detector+"_frame"+str(frame_number)+".pdf", bbox_inches='tight', pad_inches=0.5)
        plt.close(constellation_fig)
        
        # Reset rx constellation 
        rx_constellation = np.zeros((2, 0))
   
except Exception:
    traceback.print_exc()



""" Compute rank of the channel matrix and compute the cross correlation matrix 
    Do it only on DMRS coefficients within one OFDM symbol  
"""
try : 
    
    print("Computing empirical correlation matrix")
    
    # RX correlation 
    channel_matrix  = np.zeros((nb_symbols, nb_rx_ports, nb_tx_ports), dtype = complex)
    conjugate_transpose = np.zeros((nb_symbols, nb_tx_ports, nb_rx_ports), dtype = complex)
    hermitian_matrix = np.zeros((nb_symbols, nb_rx_ports, nb_rx_ports), dtype = complex)
    correlation_matrix = np.zeros((nb_symbols, nb_rx_ports, nb_rx_ports), dtype = complex)
    correlation_matrix_norms = np.zeros((nb_symbols, nb_rx_ports, nb_rx_ports))
    correlation_matrix_mean = np.zeros((nb_rx_ports, nb_rx_ports), dtype = complex)
    correlation_matrix_norm = np.zeros((nb_rx_ports, nb_rx_ports))
    
    #TX correlation 
    
    
    #load channel matrices
    for i in range(nb_symbols):
        for j in range(nb_tx_ports):
            for k in range(nb_rx_ports):
                x = lines_interp[tx_port_indexes[j]][rx_port_indexes[k]][i*fft_size + sc]
                x = x.split("(")[1]
                x = x.split(")")[0]
                re = float(x.split(',')[0])
                im = float(x.split(',')[1])
                channel_matrix[i, k , j] = complex(re, im)
                conjugate_transpose[i, j, k] = complex(re, -im)
    
    #compute HH^h 
    for i in range(nb_symbols):
        hermitian_matrix[i] = np.matmul(channel_matrix[i], conjugate_transpose[i])
           
    # renormalize biag diagonal coefficients to get the correlation 
    for i in range (nb_symbols): 
         for j in range(nb_rx_ports):
             for k in range(nb_rx_ports):
                 correlation_matrix[i, j, k] = hermitian_matrix[i, j, k] / np.sqrt(hermitian_matrix[i, j, j] * hermitian_matrix[i, k, k])
                 correlation_matrix_norms[i, j, k] = np.sqrt(correlation_matrix[i, j, k].real ** 2 + 
                                                             correlation_matrix[i, j, k].imag ** 2) 
    #take the mean 
    for j in range(nb_rx_ports):
         for k in range(nb_rx_ports):
             correlation_matrix_mean[j, k] = np.mean(correlation_matrix[:, j, k])
             correlation_matrix_norm[j, k] = np.sqrt(correlation_matrix_mean[j, k].real ** 2 + 
                                                     correlation_matrix_mean[j, k].imag ** 2)
             

    print("Correlation matrix : ")
    print(correlation_matrix_norm)

except Exception:
    traceback.print_exc()
    
    
