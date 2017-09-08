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
Main script is dt_computation_mpi.py (delay calculation on multiple CPUs)
```
mpiexec -n NPROC python dt_computation_mpi.py
```
where NPROC is the number of CPUs that will be used to compute delays (should be at least 2)
A sequential version is also available:
```
mpiexec -n NPROC python dt_computation_cc.py
```

Input parameters are indicated in Arguments.py



