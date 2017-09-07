'''
Time-delay calculation
'''

# Import Python modules
import time
import numpy as np  
import matplotlib.pyplot as plt  
from scipy import signal 
import os,sys,h5py 

# Local modules/subfunctions
from dt_computation_sub import *

# Get input parameters
from Arguments import *

# Get the list of hdf5 filenames
ifiles=os.listdir(idir)
H = []
for ifile in ifiles:
    if ifile.endswith('.hdf5'): # Make sure it is an hdf5 file
        H.append(ifile)
nevent=len(H) # Number of events

# Waveform tapering window
ts = signal.tukey(ns,taper_alpha)

# Butterworth bandpass filter
Wn  = np.array([fmin, fmax]) * 2. * dt 
sos = signal.butter(4, Wn, 'bandpass', output='sos') # Filter second-order sections

# Set the name of the output file that contains the delays
fid  = open('dt_cluster_ef_cc.txt', 'wt')
fid2 = open('id_map_cc.txt','wt')

# Get the IDs of the events we want to compute the delays (i.e., the event index)
if os.path.exists('list_in.txt'):
    eventid=np.loadtxt('list_in.txt')
    eventid=eventid.astype(int)
else:
    eventid = np.arange(nevent)

# Main loop on seismic events
for idx, ievent1 in enumerate(eventid):    
    
    # Clock
    tstart = time.time()    

    # Filename of the 1st event in the pair
    hdf5filename= os.path.join(idir,H[ievent1])
    fid2.write('%-5d %s\n'%(ievent1,hdf5filename))
    
    # Open the 1st event hdf5 file
    h1 = h5py.File(hdf5filename, 'r')
    print('Event %d : %s'%(ievent1,hdf5filename))
    
    # Loop on all possible event pairs
    for ievent2 in eventid[idx+1:]:
        
        # Filename of the 2nd event in the pair
        hdf5filename= os.path.join(idir,H[ievent2])
        
        # Open the 2nd event hdf5 file
        h2 = h5py.File(hdf5filename, "r")

        # Loop on stations for the 1st event
        for sta in h1['STATIONS'].keys():
            
            # Define the path to the current station            
            path0 = os.path.join('/STATIONS/',sta)
            
            # If we have a record for the 2nd event
            if path0 in h2:
                
                # Set the path to the current station trace
                path = os.path.join(path0,'Trace')
                
                # Extract the waveforms
                y1 = np.array(h1[path])
                y2 = np.array(h2[path])
                
                # Filter the waveforms
                y1_f = signal.sosfilt(sos,ts*(y1-np.mean(y1)))*ts
                y2_f = signal.sosfilt(sos,ts*(y2-np.mean(y2)))*ts

                # Get the component
                cmpnt=h1[path0].attrs['CMPNT']

                # Vertical component: compute P-wave differential time
                if(cmpnt=="Z"):
                    # Compute a first delay in the time domain
                    (Rmax,Rmin,Lp,Lm) = dt_time_correl(y1_f,y2_f,40,nw,taper_alpha,interp_factor)                    

                else: # Else: compute the S-wave differential time

                    # P-wave travel time
                    tp1= float(h1[path0].attrs['TP'])
                    tp2= float(h2[path0].attrs['TP'])
                    
                    # Time interval between P and S wave
                    # (Should be improved such that each event has his S time defined not necessarily the same for both events)
                    dtps =  np.round(((tp1+tp2)/2. * (gamma - 1))*1./dt)
                    
                    # S-wave arrival time (in samples)s
                    dtps = dtps + 100 # (since P-wave arrival is 1 sec after the beginning of the record)
                    
                    # Get some points in the window before the S wave arrival
                    dtps = dtps - 50
                    
                    # Compute a first delay in the time domain
                    (Rmax,Rmin,Lp,Lm) = dt_time_correl(y1_f,y2_f,dtps.astype(int),nw,taper_alpha,interp_factor)
                    
                # Check if we can compute a delay                
                if Rmin < R_th and Rmax < R_th:
                    continue

                # Write outputs
                tp1=h1[path0].attrs['TP'] # P travel time for the 1st event
                tp2=h2[path0].attrs['TP'] # P travel time for the 2nd event
                Lp = -Lp*dt + tp2-tp1
                Lm = -Lm*dt + tp2-tp1
                #s_out = '%4d %4d %6s %6.1f %6.1f %6.1f %6.1f %6.1f %6.1f %9.2f %9.2f\n'%(ievent1,ievent2,sta,Rmax*100.,Rmin*100.,SNRp,SNRm,Cp*100.,Cm*100.,Lp*1000.,Lm*1000.)
                s_out = '%4d %4d %6s %6.1f %6.1f %9.2f %9.2f\n'%(ievent1,ievent2,sta,Rmax*100.,Rmin*100.,Lp*1000.,Lm*1000.)
                fid.write(s_out)
        h2.close()
    h1.close()
    print('-- Time elapsed: %.2f'%(time.time()-tstart))    
fid.close()
fid2.close()

