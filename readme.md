![Logo](plots/clue_logo.png)

# Standalone CLUE Algorithm on GPU and CPU

Z.Chen[1], A. Di Pilato[2,3], F. Pantaleo[4], M. Rovere[4], C. Seez[5]

*[1] Northwestern University, [2]University of Bari, [3]INFN, [4] CERN, [5]Imperial College London*

## 1. Setup

The pre-requisite dependencies are C++20 compiler. Fork this repo if developers.

If CUDA/nvcc are found on the machine, the compilation is performed automatically also for the GPU case.
The path to the nvcc compiler will be automatically taken from the machine. In this case, `>=cuda12` is required.

* **On a CERN machine with GPUs:** Source the LCG View containing GCC, Boost
and CUDA:
```bash
source /cvmfs/sft.cern.ch/lcg/views/LCG_102cuda/x86_64-centos7-gcc8-opt/setup.sh

# then setup this project
git clone --recurse-submodules https://gitlab.cern.ch/kalos/clue.git
cd clue
cmake -S . -B build
cmake --build build

# if installation is needed
mkdir install
cd build/ ; cmake .. -DCMAKE_INSTALL_PREFIX=../install; make install
```

* **On an Ubuntu machine with GPUs:** 
```bash
# then setup this project
git clone --recurse-submodules https://gitlab.cern.ch/kalos/clue.git
cd clue
make
```

### 2. Run CLUE
CLUE needs three parameters:
  * `dc`
  * `rhoc`
  * `outlierDeltaFactor`

1. _dc_ is the critical distance used to compute the local density.
1. _rhoc_ is the minimum local density for a point to be promoted as a Seed.
1. _outlierDeltaFactor_ is  a multiplicative constant to be applied to `dc`.

The test program accept the following parameter from the command line:

* `-i input_filename`: input file with the points to be clustered
* `-d critical_distance`: the critical distance to be used
* `-r critical_density`: the critical density to be used
* `-o outlier_factor`: the multiplicative constant to be applied as outlier
  rejection factor
* `-e sessions`: number of times the clustering algorithm has to run on the
  same input dataset. That's useful to have a more reliable measure of the
  timing performance.
* `-u accelerator`: run with the alpaka executor. You can find valid options with `-U` Every
  single executable, in fact, has both the CPU and the GPU version embedded.
* `-U`: list enabled alpaka executors.
* `-v verbose`: activate verbose output. Among other things, this will also
  enable the saving of the results of the clustering steps in local text files.

If the projects compiles without errors, you can go run the CLUE algorithm by
```bash
# alpaka cpu serial
./build/src/clue_alpaka/mainAlpaka -i data/input/aniso_1000.csv -d 7.0 -r 10.0 -o 2 -e 10 -v -u CpuSerial

# alpaka gpu CUDA
./build/src/clue_alpaka/mainAlpaka -i data/input/aniso_1000.csv -d 7.0 -r 10.0 -o 2 -e 10 -v -u GpuCuda

# alpaka gpu OpenMP
./build/src/clue_alpaka/mainAlpaka -i data/input/aniso_1000.csv -d 7.0 -r 10.0 -o 2 -e 10 -v -u CpuOmpBlocks

# in case of original CPU without alpaka
./build/src/clue/main -i data/input/aniso_1000.csv -d 7.0 -r 10.0 -o 2 -e 10 -v

# in case of original CUDA without alpaka
./build/src/clue/main -i data/input/aniso_1000.csv -d 7.0 -r 10.0 -o 2 -e 10 -v -u cuda
```

The input files are `data/input/*.csv` with columns 
* x, y, layer, weight

The output files are `data/output/*.csv` with columns
* x, y, layer, weight, rho, delta, nh, isSeed, clusterId

If you encounter any error when compiling or running this project, please
contact us.

## 3. Examples
The clustering result of a few synthetic dataset is shown below
![Datasets](Figure3.png)

## 4. Performance on Toy Events
We generate toy events on toy detector consist of 100 layers.
The average execution time of toy events on CPU and GPU are shown below
![Execution Time](Figure5_1.png)
