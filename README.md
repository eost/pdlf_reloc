# pdlf_reloc
A set of functions to compute time-delays and perform dd relocation.
It's a work in progress (meaning, it is not yet working).

## Dependencies
- python3
- numpy
- scipy.signal
- h5py
- matplotlib.pyplot

## Download the package
To get everything, use the following git command:
```
git clone https://github.com/eost/pdlf_reloc.git
```
or download the module directly from https://github.com/eost/pdlf_reloc.git

## Compute time-delays
Input parameters are indicated in Arguments.py

There are two possibilities to compute delays:

1/ Use an MPI version of the script to compute time-delays on multiple CPUs:
```
mpiexec -n NPROC python dt_computation_mpi.py
```
where NPROC is the number of CPUs that will be used to compute delays (should be at least 2: 1 process for the boss and at least 1 worker).

2/ Use a sequential version of the script to compute time-delays:
```
python dt_computation_cc.py
```



