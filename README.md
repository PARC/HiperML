# HiperML
HiperML for High-performance Machine Learning

Installation (tested on Ubuntu 14.04)
------------

1. Install CUDA

2. Compile the source code of HiperML. In the root directory of the project:
 ``` bash
$ mkdir build
$ cd build
$ cmake ..
$ make
$ sudo make install
 ```
All executables produced by the 'make' command above are located in folders under build/apps, including gemm/, gemv/, kmeans/, pagerank/, and triangle_count/. To get the usage of an executable in these folders, invoke the executable with the -h option.
