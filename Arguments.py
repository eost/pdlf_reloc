
# Set the directory with the hdf5 files
idir="./east_flank/hdf5"

# Set seismogram parameters
ns = 512   # Number of samples
dt = 0.01  # Sampling step

# Data windowing
nw    = 128  # Window size
gamma = 1.73 # Vp/Vs ratio
taper_alpha = 0.2 # Taper width (fraction of the window size, also used to taper the overall seismogram)

# Spectral smoothing
smooth_length = 8 # Width of the moving window to smooth the spectra

# Build the frequency vector and the filter
fmin = 8.
fmax = 32.

# Minimum correlation and coherency threshold
R_th = 0.5 # Correlation threshold
C_th = 0.4 # Coherency threshold
