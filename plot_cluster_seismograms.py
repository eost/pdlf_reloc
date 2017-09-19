'''
Plot seismograms in a cluster at a given station
'''

import os,h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal
from glob import glob

# Get input params
from Arguments import *

for sta in ['HDLZ','HDLN','HDLE']:#'CRAZ','CRAN','GPNZ','GPNN','FLRZ','FLRN','FORZ','FORE','GBSZ','GBSE','GPSZ','GPSE','RVLZ','RVLE','VILZ','VILE','VILN']:
  for cfile in glob('clusters/c*.txt'):    

    # Butterworth bandpass filter sos
    Wn  = np.array([fmin, fmax]) * 2. * dt 
    sos = signal.butter(4, Wn, 'bandpass', output='sos') # Filter second-order sections

    # Waveform tapering window
    ts = signal.tukey(ns,taper_alpha)      

    # Get HDF5 filenames from the cluster file (cfile)
    fid = open(cfile,'rt')
    H = []
    for l in fid:
        items = l.strip().split()
        assert os.path.exists(items[1]), 'Cannot find file %s'%(items[1])
        H.append(items[1])
    fid.close()
    nevent = len(H)

    if nevent < 7:
        continue

    # Read, taper and filter waveforms
    waveforms = np.zeros((len(H),ns))
    for i,hfile in enumerate(H):
        # Read waveform
        h = h5py.File(hfile, 'r')
        if sta not in h['STATIONS']:
            print('%s not in %s'%(sta,hfile))
            h.close()
            continue
        y = np.array(h['/STATIONS/'+sta+'/Trace'])
        h.close()
        assert y.size==ns ,'Incorrect seismogram length in %s for %s'%(h,sta)
        # Taper and filter waveforms
        y_f = signal.sosfilt(sos,(y-np.mean(y)))*ts
        # Append waveform to the matrix
        waveforms[i,:] = y_f/(y_f.max()-y_f.min())

    # Display waveforms
    plt.pcolor(waveforms)
    cname = os.path.basename(cfile).strip('.txt')
    plt.title('%s - %s (%d events)'%(sta,cname,nevent))
    plt.savefig('clusters/'+cname+'_'+sta+'.png')
    plt.close()
    #plt.show()
    
