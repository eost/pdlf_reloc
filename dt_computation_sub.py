'''
Simple functions to compute time-delays between seismic waveforms
'''

# Load modules
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt

# 
def dt_time_correl(y1,y2,Istart,nw):
    '''
    Compute time-delay in the time domain.

    Parameters
    ----------
    y1, y2 : ndarray
            input signals. We assume they are synchronized, filtered accordingly and have the same sampling step
    nw : int
            length of the time-window in which the delay is computed
    Istart : int
            sample index at the beginning of the time-window 

    Returns
    -------
    Rmax,Rmin,Lmax,Lmin : float, float, int, int
            Correlation coefficients (Rmax,Rmin) and delays in number of samples (Lmax,Lmin) 
            respectively at the minimum or maximum of the correlation function
    '''
    
    # Windowing and normalizing signals
    y1 = y1[Istart:Istart+nw]
    y2 = y2[Istart:Istart+nw]
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
    Rmin = np.amax(-R)
    
    # Get the delays associated with max and min (in samples)
    Lmax = np.argmax(R)-(nw-1)
    Lmin = np.argmax(-R)-(nw-1)

    # All done
    return(Rmax,Rmin,Lmax,Lmin,R)

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
