'''
Time-delay calculation
'''

# External modukes
import time
import numpy as np  
from scipy import signal 
import os,sys,h5py
from mpi4py import MPI

# Internal modules/functions
from dt_computation_sub import *

# Get input parameters
from Arguments import *

class LogFile(object):
   """
   A simple object that deals with log files
   """
   def __init__(self, filename):
       self.filename = filename
       fid = open(self.filename,'w')
       fid.write('Log starts at %s\n'%(time.ctime()))
       fid.close()

   def write(self,textline):
      fid = open(self.filename,'a')
      fid.write('%s : %s\n'%(time.ctime(),textline))
      fid.flush()
      fid.close()


def worker(comm, TagIn, TagOut):
    '''
    This function computes delays for a given event pair
    (Rank of the Boss should be 0)

    Parameters
    ----------
    comm : mpi4py.MPI.Intracomm
          MPI communicator
    Tagin : int
          MPI_TAG for receiving data from the Boss
    Tagout : int
          MPI_TAG for sending data to the Boss

    '''

    # My rank
    rank = comm.Get_rank()

    # Notify the Boss that I am ready to work
    comm.send({'rank':rank}, dest=0, tag=TagOut)

    # Receive an event pair to process from the Boss
    data = comm.recv(source=0, tag=TagIn)

    # Work until we are done
    while data['Active']:
        # Compute delays
        ievent1,ievent2,hdf5filename1,hdf5filename2 = data['ev_pair']        
        delays = compute_delay(ievent1,ievent2,hdf5filename1,hdf5filename2,ns,dt,fmin,fmax,nw,gamma,taper_alpha,interp_factor,R_th)
        # Send results to the Boss
        ToSend = {'rank':rank, 'delays':delays}
        comm.send(ToSend, dest=0, tag=TagOut)
        # Wait for further instructions
        data = comm.recv(source=0,tag=TagIn)
        
    # All done
    return data['Active']
    

def boss(comm,TagIn,TagOut):
    '''
    This function distributes the event pairs to the workers

    Parameters
    ----------
    comm : mpi4py.MPI.Intracomm
          MPI communicator
    Tagin : int
          MPI_TAG for receiving data from workers
    Tagout : int
          MPI_TAG for sending data to workers
    '''

    # Get rank and number of processes
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Check rank
    assert rank == 0, 'Incorrect rank for master in boss function'

    # Initialize worker ranks and status
    ranks = range(1,size)
    flags = [True for e in ranks]
    ActiveWorkers = dict(zip(ranks,flags))

    # Get the list of hdf5 filenames
    ifiles=os.listdir(idir)
    H = []
    for ifile in ifiles:
        if ifile.endswith('.hdf5'): # Make sure it is an hdf5 file
            H.append(ifile)
    nevent=len(H) # Number of events
    
    # Get the IDs of the events we want to compute the delays (i.e., the event index)
    if os.path.exists('list_in.txt'):
        eventid=np.loadtxt('list_in.txt')
        eventid=eventid.astype(int)
    else:
        eventid = np.arange(nevent)

    # Index mapping
    fid = open('id_map_test.txt','wt')
    for idx, ievent1 in enumerate(eventid):
        hdf5filename1 = os.path.join(idir,H[ievent1])
        fid.write('%-5d %s\n'%(ievent1,hdf5filename1))
    fid.close()

    # Event pairs that we have to process 
    ev_pairs = []      
    for idx, ievent1 in enumerate(eventid):
        hdf5filename1 = os.path.join(idir,H[ievent1])        
        for ievent2 in eventid[idx+1:]:
            hdf5filename2 = os.path.join(idir,H[ievent2])            
            ev_pairs.append([ievent1,ievent2,hdf5filename1,hdf5filename2])

    # Compute delays until we are done
    fid  = open('dt_cluster_ef_test.txt', 'wt') # Output file including delays
    log  = LogFile('status.log')
    while list(ActiveWorkers.values()).count(True) > 0:
        
        # Listen workers
        dataRECV = comm.recv(source = MPI.ANY_SOURCE, tag = TagIn)
        wRank = dataRECV['rank']

        # Parse the data sent by the worker
        if 'delays' in dataRECV: # Receiving results
            log.write('received results from Worker %i'%(wRank))
            for o in dataRECV['delays']:
                s_out = '%4d %4d %6s %6.1f %6.1f %9.2f %9.2f\n'%(o[0],o[1],o[2],o[3]*100.,o[4]*100.,o[5]*1000.,o[6]*1000.)
                fid.write(s_out)
                fid.flush()
        else: # Worker is initialized
            log.write('Worker %i initialized'%(wRank))
            
        # Prepare message to worker
        if len(ev_pairs)>0:
            ev_pair = ev_pairs.pop(0)
            ToSend = {'Active':True,'ev_pair':ev_pair}
            log.write('Boss sending %s %s for worker %i (files %s and %s)'%(ev_pair[0],ev_pair[1],wRank,ev_pair[2],ev_pair[3]))
            log.write('  %i tasks waiting for workers.'%(len(ev_pairs)))
        else:
            ToSend = {'Active': False} # Kill the worker
            ActiveWorkers[wRank] = False
            log.write('Worker %d is terminated'%(wRank))

        # Send message to worker
        comm.send(ToSend, dest=wRank, tag = TagOut)
        log.write('-- End --')

    # Close output files
    fid.close()
    
    # All done
    return

        
    
    # Log file
    Logger = LogFile('status.log')
    Logger.clearFile()

    

    
    




def dt_computation_main(TagBoss2Worker = 1979, TagWorker2Boss = 28):
    '''
    Main function
    Args:
        * TagBoss2Worker: MPI_TAG for sending data from Boss to Worker(s)
        * TagWorker2Boss: MPI_TAG for sending data from Worker(s) to Boss
    '''
    
    comm = MPI.COMM_WORLD  # MPI communicator
    rank = comm.Get_rank() # Rank of current process

    if comm.Get_size()>1:
        if rank == 0: # Boss process
            boss(comm,TagWorker2Boss, TagBoss2Worker)
        else:         # Worker processes
            worker(comm, TagBoss2Worker, TagWorker2Boss)
    else:
        msg = "number of processors must be > 1"
        raise ValueError(msg)                 

if __name__ == '__main__':
    dt_computation_main()
