# pdlf_reloc
A set of functions to compute time-delays and perform dd relocation.
It's a work in progress (meaning, it is not yet working).

## Dependencies
- python3 (not tested on python2 but should work)
- A functional MPI 1.x/2.x/3.x implementation like OpenMPI or MPICH built with shared/dynamic libraries

Python modules:
- numpy
- scipy.signal
- h5py
- matplotlib.pyplot
- pyproj

## Download the package
To get everything, use the following git command:
```
git clone https://github.com/eost/pdlf_reloc.git
```
or download the module directly from https://github.com/eost/pdlf_reloc.git

## Define clusters
Use
```
python define_clusters.py
```

## Compute time-delays
Input parameters are indicated in Arguments.py

There are two possibilities to compute time-delays:

1/ Use an MPI version of the script to compute time-delays on multiple CPUs:
```
mpiexec -n NPROC python dt_computation_mpi.py
```
where NPROC is the number of CPUs that will be used to compute delays (should be at least 2: 1 process for the boss and at least 1 worker).

2/ Use a sequential version of the script to compute time-delays:
```
python dt_computation_cc.py
```

## DD reloc
Use:
```
python ddreloc.py
```



