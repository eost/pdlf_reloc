'''
Simple functions to compute time-delays between seismic waveforms
'''

# Load modules
import numpy as np
from scipy import signal
import os,h5py 

def compute_delay(ievent1,ievent2,hdf5filename1,hdf5filename2,npts,dt,fmin,fmax,nw,gamma=1.73,taper_alpha=0.2,interp_factor=1,R_th=0.5):
    '''
    Compute time-delays from hdf5 files

    Parameters
    ----------
    ievent1 : int
            ID of the first event
    ievent2 : int
            ID of the second event
    hdf5filename1 : str
            HDF5 filename for the first event
    hdf5filename2 : str
            HDF5 filename for the second event
    npts : int
            number of samples in the seismograms
    dt : float
            sampling step
    fmin, fmax : float, float
            Filter corner frequencies
    nw : int
            Window size in samples
    gamma : float, default: 1.73
            Vp/Vs ratio
    taper_alpha : float, default: 0.2
            Taper width (fraction of the window size, also used to taper the overall seismogram)
    interp_factor : int
            Data interpolation factor to compute delays. i.e., the ratio of output rate / input rate
    R_th : float
            Correlation threashold

    '''

    # Waveform tapering window
    ts = signal.tukey(npts,taper_alpha)

    # Butterworth bandpass filter sos
    Wn  = np.array([fmin, fmax]) * 2. * dt 
    sos = signal.butter(4, Wn, 'bandpass', output='sos') # Filter second-order sections    

    # Open input hdf5 files
    h1 = h5py.File(hdf5filename1, 'r')
    h2 = h5py.File(hdf5filename2, 'r')

    # Output table
    out = []
    
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

            # Check seismogram length
            assert y1.size==npts ,'Incorrect seismogram length in %s for %s'%(hdf5filename1,sta)
            assert y2.size==npts ,'Incorrect seismogram length in %s for %s'%(hdf5filename2,sta)
                
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
            
            # Total delay
            tp1=h1[path0].attrs['TP'] # P travel time for the 1st event
            tp2=h2[path0].attrs['TP'] # P travel time for the 2nd event
            Lp = -Lp*dt + tp2-tp1
            Lm = -Lm*dt + tp2-tp1

            # Append output
            out.append([ievent1,ievent2,sta,Rmax,Rmin,Lp,Lm])
    h1.close()
    h2.close()

    # All done
    return out


def dt_time_correl(y1,y2,Istart,nw,taper_alpha=None,interp_factor=1,interp_Rtol=0.75):
    '''
    Compute time-delay in the time domain.

    Parameters
    ----------
    y1, y2 : ndarray
            input signals. We assume they are synchronized, filtered accordingly and have the same sampling step
    Istart : int
            sample index at the beginning of the time-window 
    nw : int
            length of the time-window in which the delay is computed
    taper_alpha : float,o optional
            Shape parameter of the tukey taper window used before fourrier transform, 
            representing the fraction of the window inside the cosine tapered region.
    interp_factor : int, default=1 (no interpolation)
            Interpolation factor. i.e., the ratio of output rate / input rate
            (must be >= 1)
    interp_Rtol : float, default=0.75
            Will perform interpolation only for points with correlation coefficient
            - larger than Rmax*interp_Rtol
            - smaller than Rmin*interp_Rtol
    Returns
    -------
    Rmax,Rmin,Lmax,Lmin : float, float, float, float
            Correlation coefficients (Rmax,Rmin) and delays in number of samples (Lmax,Lmin) 
            respectively at the minimum or maximum of the correlation function (assuming the 
            original sampling rate). Lmax and Lmin are not necessary integers if interp_factor 
            is larger than 1.
    '''
    
    # Check if interp_factor is an integer
    assert type(interp_factor)==int, 'interp_factor must be an integer'
    assert interp_factor>=1, 'interp_factor must be >=1'

    # Build the taper function
    if taper_alpha is not None:
        tw = signal.tukey(nw,taper_alpha)
    else:
        tw = np.ones((nw,))
    
    # Time-windowing
    y1 = y1[Istart:Istart+nw]*tw
    y2 = y2[Istart:Istart+nw]*tw

    # Normalization
    y1_sigma = np.std(y1)
    y2_sigma = np.std(y2)
    if y1_sigma > 0.:
        f1 = 1./y1_sigma
    else:
        f1 = 0.
    if y2_sigma > 0.:
        f2 = 1./y2_sigma
    else:
        f2 = 0.
    y1 = (y1 - np.mean(y1))*f1/float(len(y1))
    y2 = (y2 - np.mean(y2))*f2
    
    # Compute the normazlized correlation
    R=np.correlate(y1,y2,'full')    

    # Get the maximum and minimum of the cross correlation function
    Rmax = np.amax(R)
    Rmin = np.amin(R)

    # Get the delays associated with max and min (in samples)
    Lmax = np.argmax(R)-(nw-1)
    Lmin = np.argmin(R)-(nw-1)
        
    # Interpolation (optional)
    if interp_factor > 1: 
        # Original time
        ti = (np.arange(R.size) - (nw-1))
        # Find points smaller than of Rmin*interp_Rtol and larger than Rmax*interp_Rtol
        nt = 1./float(interp_factor)
        imax = ti[np.where( (R>=Rmax*interp_Rtol) | (R<=Rmin*interp_Rtol) )[0]]
        to = np.array([])
        for i in imax:            
            to = np.append(to,np.arange(i-1,i+1+nt,nt))
        # Get interpolated lag and R values
        for i in range(to.size):
            r = np.dot(R,np.sinc((to[i]-ti)))                        
            if r > Rmax:
                Rmax = r
                Lmax = to[i]
            elif r < Rmin:
                Rmin = r
                Lmin = to[i]
            
    # All done
    return(Rmax,Rmin,Lmax,Lmin)



def calcSNR(y1,y2,Istart,nw,fmin,fmax,delta,taper_alpha=0.2,smooth_length=8):
    '''
    Compute the signal to noise ratio

    Parameters
    ----------
    y1, y2 : ndarray
            Input signals. We assume they are synchronized, filtered accordingly and have the same sampling step.
    Istart : int
            The sample index at the beginning of the time-window.
    nw : int
            The length of the time-window in which the delay is computed.
    fmin,fmax : float, float
            Bounds of the frequency range use for fitting the phase.
    delta : float
            Sampling step of the input signals
    taper_alpha : float
            Shape parameter of the tukey taper window used before fourrier transform, 
            representing the fraction of the window inside the cosine tapered region.
    smooth_length : int
            Number of samples over which fourrier spectra are smoothed
    '''

    # Find the minimum and maximum frequency bounds
    freq = np.fft.fftfreq(nw, d=delta)
    ifreq1=np.argmin(np.absolute(freq-fmin))
    ifreq2=np.argmin(np.absolute(freq-fmax))

    # Build the taper function
    tw = signal.tukey(nw,taper_alpha)
    
    # Build the smoothing function of the spectra
    th = signal.hann(smooth_length);
    th = th/np.sum(th)
    
    # Taper the current seismograms around the P-wave arrival
    y1=y1[Istart:Istart+nw]*tw
    y2=y2[Istart:Istart+nw]*tw

    # Compute the Fourier transform of the first event
    y1_ft=np.fft.fft(y1)
    # Compute the Fourier transform of the second event
    y2_ft=np.fft.fft(y2)
    # Compute the conjugate complex of the second event Fourier transform
    y2_ft=np.conj(y2_ft)

    # Compute the cross-spectra between event 1 and 2
    Cxy = y1_ft*y2_ft
    # Compute the Auto-spectra of event 1 (take the real part just in case it remains a small imaginary component.
    Cxx = np.real(y1_ft*np.conj(y1_ft))
    # Compute the Auto-spectra of event 2
    Cyy = np.real(y2_ft*np.conj(y2_ft))
    
    # Smooth the spectra
    Cxx = signal.filtfilt(th,1,Cxx)
    Cxy = signal.filtfilt(th,1,Cxy)
    Cyy = signal.filtfilt(th,1,Cyy)
   
    # Compute the coherence
    Ixx=np.where(Cxx<0.0001) # Coherence diverges when Cxx=0 or Cyy=0, replace them with a small value
    Iyy=np.where(Cyy<0.0001)
    Cxx[Ixx]=0.0001
    Cyy[Iyy]=0.0001
    C=np.absolute(Cxy)/(np.sqrt(Cxx*Cyy)) # Coherency
    C_mean = np.mean(C[ifreq1:ifreq2])

    # Signal to noise ratio
    I=np.where(C>0.99)
    C[I]=0.99    
    SNR = C**2 / (1. - C**2)
    SNR_mean = np.mean(SNR[ifreq1:ifreq2])

    # All done
    return SNR_mean,C_mean


def delay_cross_spectrum(y1,y2,Istart,nw,fmin,fmax,delta,taper_alpha=0.2,smooth_length=8,C_th=0.4):
    '''
    Compute the time delay between signals y1 and y2 from sample Istart until sample Istart+nw.

    Parameters
    ----------
    y1, y2 : ndarray
            Input signals. We assume they are synchronized, filtered accordingly and have the same sampling step.
    Istart : int
            The sample index at the beginning of the time-window.
    nw : int
            The length of the time-window in which the delay is computed.
    fmin,fmax : float, float
            Bounds of the frequency range use for fitting the phase.
    delta : float
            Sampling step of the input signals
    taper_alpha : float
            Shape parameter of the tukey taper window used before fourrier transform, 
            representing the fraction of the window inside the cosine tapered region.
    smooth_length : int
            Number of samples over which fourrier spectra are smoothed
    C_th : float
            Minimum Coherence threshold to compute the delay
    '''

    # Find the minimum and maximum frequency bounds
    freq = np.fft.fftfreq(nw, d=delta)
    ifreq1=np.argmin(np.absolute(freq-fmin))
    ifreq2=np.argmin(np.absolute(freq-fmax))

    # Build the taper function
    tw = signal.tukey(nw,taper_alpha)
    
    # Build the smoothing function of the spectra
    th = signal.hann(smooth_length);
    th = th/np.sum(th)
    
    # Taper the current seismograms around the P-wave arrival
    y1=y1[Istart:Istart+nw]*tw
    y2=y2[Istart:Istart+nw]*tw

    # Compute the Fourier transform of the first event
    y1_ft=np.fft.fft(y1)
    # Compute the Fourier transform of the second event
    y2_ft=np.fft.fft(y2)
    # Compute the conjugate complex of the second event Fourier transform
    y2_ft=np.conj(y2_ft)

    # Compute the cross-spectra between event 1 and 2
    Cxy = y1_ft*y2_ft
    # Compute the Auto-spectra of event 1 (take the real part just in case it remains a small imaginary component.
    Cxx = np.real(y1_ft*np.conj(y1_ft))
    # Compute the Auto-spectra of event 2
    Cyy = np.real(y2_ft*np.conj(y2_ft))
    
    # Smooth the spectra
    Cxx = signal.filtfilt(th,1,Cxx)
    Cxy = signal.filtfilt(th,1,Cxy)
    Cyy = signal.filtfilt(th,1,Cyy)
   
    # Get the phase of the cross-spectra and unwrap it
    phi = np.unwrap(np.angle(Cxy));

    # Compute the coherence
    Ixx=np.where(Cxx<0.0001) # Coherence diverges when Cxx=0 or Cyy=0, replace them with a small value
    Iyy=np.where(Cyy<0.0001)
    Cxx[Ixx]=0.0001
    Cyy[Iyy]=0.0001
    C=np.absolute(Cxy)/(np.sqrt(Cxx*Cyy)) # Coherency
    
    # Compute the weigth for each value of the phase based on the coherence
    # The weigth is given by C^2/(1-C^2), these weight actually represent 1/\sigma^2
    # First because this weight actullay diverges towards C=1 we cannot assigned a weigth
    # higher than the one corresponding to a coherence of 0.99
    I=np.where(C>0.99)
    C[I]=0.99
    
    # Compute the weight
    w = np.power(C,2)/(1-np.power(C,2))
    
    # Set the weigth to 0 if the coherence is lower than 0.4 (this last value should be a parameter)
    I1=np.where(C<0.4)
    w[I1] = 0

    # Compute time-delay
    C_mean = np.mean(C[ifreq1:ifreq2])
    if C_mean > C_th:
        (tau,s_tau,b) = solve_w_lin_lsq(w,freq,phi,ifreq1,ifreq2)
        tau   /= (2*np.pi)
        s_tau /= (2*np.pi)
    else:
        tau   = None
        s_tau = None
        
    # All done
    return (tau,s_tau,C_mean)


def solve_w_lin_lsq(w,x,y,i1,i2):
    # Solve the weighted linear least square problem with no priori information
    # Linear least square regression following Tarantola's book (SIAM, 2005) p 271

    A = np.sum(w[i1:i2])
    B = np.sum(np.power(x[i1:i2],2)*w[i1:i2])
    C = np.sum(x[i1:i2]*w[i1:i2])
    P = np.sum(x[i1:i2]*w[i1:i2]*y[i1:i2])
    Q = np.sum(w[i1:i2]*y[i1:i2])

    # Compute the parameters
    a = (A*P-C*Q)/(A*B-C*C)
    b = (B*Q-C*P)/(A*B-C*C)
    #print(a,b)
    
    # Compute the uncertainty associated with parameter a
    s_a = 1/np.sqrt((B-(C*C)/A))

    # All done
    return(a,s_a,b)
