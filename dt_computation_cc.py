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
    hdf5filename1 = os.path.join(idir,H[ievent1])
    fid2.write('%-5d %s\n'%(ievent1,hdf5filename1))
    
    # Open the 1st event hdf5 file
    print('Event %d : %s'%(ievent1,hdf5filename1))
    
    # Loop on all possible event pairs
    for ievent2 in eventid[idx+1:]:
        
        # Filename of the 2nd event in the pair
        hdf5filename2 = os.path.join(idir,H[ievent2])

        out = compute_delay(ievent1,ievent2,hdf5filename1,hdf5filename2,ns,dt,fmin,fmax,nw,gamma,taper_alpha,interp_factor,R_th)
        
        for o in out:
            s_out = '%4d %4d %20s %20s %6s %6.1f %6.1f %9.2f %9.2f\n'%(o[0],o[1],o[2],o[3],o[4],o[5]*100.,o[6]*100.,o[7]*1000.,o[8]*1000.)
            fid.write(s_out)
    print('-- Time elapsed: %.2f'%(time.time()-tstart))    
fid.close()
fid2.close()

