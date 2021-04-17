.. _build:

Building ktorch
===============

The first step is to `download the relevant zip file from PyTorch <https://pytorch.org/get-started/locally/>`_.

Once the zip file is downloaded and unzipped, the next step is to download the ktorch source code.

Get the code via git clone:

::

   > cd ~
   > git clone https://github.com/ktorch/ktorch.git
   Cloning into 'ktorch'...

Or download the zip file:

::

   > wget https://github.com/ktorch/ktorch/archive/refs/heads/master.zip
   ..
   Saving to: ‘master.zip’

   > unzip master.zip
   Archive:  master.zip
      creating: ktorch-master/
     inflating: ktorch-master/LICENSE   
     inflating: ktorch-master/Makefile  
     ..

Makefile
********

The `makefile <https://github.com/ktorch/ktorch/blob/master/Makefile>`_ can be changed to suit preferences.
Most likely, there are 3 main settings that may need to be changed in the file or specified on the command line.

CXX
^^^

The c++ compiler defaults to ``clang++``. To run with GCC:

::

make CXX=g++

TORCH
^^^^^

TORCH has the location of the libraries for PyTorch. Default is set to ~/libtorch.

::

   make TORCH=/customdir/libtorch

It may also be possible to point the make to the libraries installed for the python installation of PyTorch.


::

   # find the dir for pytorch 1.8.1 in mini conda
   find ~/miniconda3  -name libtorch.so
   /home/t/miniconda3/pkgs/pytorch-1.8.0-py3.8_cuda11.1_cudnn8.0.5_0/lib/python3.8/site-packages/torch/lib/libtorch.so
   /home/t/miniconda3/pkgs/pytorch-1.8.1-py3.8_cuda11.1_cudnn8.0.5_0/lib/python3.8/site-packages/torch/lib/libtorch.so
   /home/t/miniconda3/lib/python3.8/site-packages/torch/lib/libtorch.so

   make TORCH=/home/t/miniconda3/lib/python3.8/site-packages/torch
   clang++ -std=c++14 -std=gnu++14 -pedantic -Wall -Wfatal-errors -fPIC -O3 -D_GLIBCXX_USE_CXX11_ABI=0 -isystem /home/t/miniconda3/lib/python3.8/site-packages/torch/include -isystem /home/t/miniconda3/lib/python3.8/site-packages/torch/include/torch/csrc/api/include   -c -o ktorch.o ktorch.cpp
   ..
   clang++ -o ktorch.so ktorch.o ktensor.o kmath.o knn.o kloss.o kopt.o kmodel.o ktest.o -shared -L/home/t/miniconda3/lib/python3.8/site-packages/torch/lib -l torch -Wl,-rpath /home/t/miniconda3/lib/python3.8/site-packages/torch/lib

ABI
^^^

..
   time make

   clang++ -std=c++14 -std=gnu++14 -pedantic -Wall -Wfatal-errors -fPIC -O3 -D_GLIBCXX_USE_CXX11_ABI=0 -isystem /home/t/libtorch/include -isystem /home/t/libtorch/include/torch/csrc/api/include   -c -o ktorch.o ktorch.cpp
   clang++ -std=c++14 -std=gnu++14 -pedantic -Wall -Wfatal-errors -fPIC -O3 -D_GLIBCXX_USE_CXX11_ABI=0 -isystem /home/t/libtorch/include -isystem /home/t/libtorch/include/torch/csrc/api/include   -c -o ktensor.o ktensor.cpp
   clang++ -std=c++14 -std=gnu++14 -pedantic -Wall -Wfatal-errors -fPIC -O3 -D_GLIBCXX_USE_CXX11_ABI=0 -isystem /home/t/libtorch/include -isystem /home/t/libtorch/include/torch/csrc/api/include   -c -o kmath.o kmath.cpp
   clang++ -std=c++14 -std=gnu++14 -pedantic -Wall -Wfatal-errors -fPIC -O3 -D_GLIBCXX_USE_CXX11_ABI=0 -isystem /home/t/libtorch/include -isystem /home/t/libtorch/include/torch/csrc/api/include   -c -o knn.o knn.cpp
   clang++ -std=c++14 -std=gnu++14 -pedantic -Wall -Wfatal-errors -fPIC -O3 -D_GLIBCXX_USE_CXX11_ABI=0 -isystem /home/t/libtorch/include -isystem /home/t/libtorch/include/torch/csrc/api/include   -c -o kloss.o kloss.cpp
   clang++ -std=c++14 -std=gnu++14 -pedantic -Wall -Wfatal-errors -fPIC -O3 -D_GLIBCXX_USE_CXX11_ABI=0 -isystem /home/t/libtorch/include -isystem /home/t/libtorch/include/torch/csrc/api/include   -c -o kopt.o kopt.cpp
   clang++ -std=c++14 -std=gnu++14 -pedantic -Wall -Wfatal-errors -fPIC -O3 -D_GLIBCXX_USE_CXX11_ABI=0 -isystem /home/t/libtorch/include -isystem /home/t/libtorch/include/torch/csrc/api/include   -c -o kmodel.o kmodel.cpp
   clang++ -std=c++14 -std=gnu++14 -pedantic -Wall -Wfatal-errors -fPIC -O3 -D_GLIBCXX_USE_CXX11_ABI=0 -isystem /home/t/libtorch/include -isystem /home/t/libtorch/include/torch/csrc/api/include   -c -o ktest.o ktest.cpp
   clang++ -o ktorch.so ktorch.o ktensor.o kmath.o knn.o kloss.o kopt.o kmodel.o ktest.o -shared -L/home/t/libtorch/lib -l torch -Wl,-rpath /home/t/libtorch/lib

   real	1m36.740s
   user	1m34.677s
   sys	0m1.898s

Source files
************

- `LICENSE <https://github.com/ktorch/ktorch/blob/master/LICENSE>`_ - MIT license
- `Makefile <https://github.com/ktorch/ktorch/blob/master/Makefile>`_
- `README.md <https://github.com/ktorch/ktorch/blob/master/README.md>`_
- `docs/ <https://github.com/ktorch/ktorch/tree/master/docs>`_ - reStructuredText files for documentation at `ktorch.readthedocs.io <https://ktorch.readthedocs.io/>`_.
- `k.h <https://github.com/ktorch/ktorch/blob/master/k.h>`_ - from Kx Systems `here <https://github.com/KxSystems/kdb/blob/master/c/c/k.h>`_.
- `kloss.cpp <https://github.com/ktorch/ktorch/blob/master/kloss.cpp>`_ - code relating to loss functions
- `kloss.h <https://github.com/ktorch/ktorch/blob/master/kloss.h>`_ - redefine binary cross entropy loss functions
- `kmath.cpp <https://github.com/ktorch/ktorch/blob/master/kmath.cpp>`_ - PyTorch math routines
- `kmodel.cpp <https://github.com/ktorch/ktorch/blob/master/kmodel.cpp>`_ - code for building models (module + optimizer + loss function)
- `knn.cpp <https://github.com/ktorch/ktorch/blob/master/knn.cpp>`_ - code for building modules
- `knn.h <https://github.com/ktorch/ktorch/blob/master/knn.h>`_ - custom modules defined here
- `kopt.cpp <https://github.com/ktorch/ktorch/blob/master/kopt.cpp>`_ - optimizer code
- `ktensor.cpp <https://github.com/ktorch/ktorch/blob/master/ktensor.cpp>`_ - code for operating on tensors
- `ktest.cpp <https://github.com/ktorch/ktorch/blob/master/ktest.cpp>`_ - contains temporary tests, samples, etc. -- nothing essential to the interface library
- `ktorch.cpp <https://github.com/ktorch/ktorch/blob/master/ktorch.cpp>`_ - contains the code used by the rest of system dealing with tensors, modules, optimizers, etc.
- `ktorch.h <https://github.com/ktorch/ktorch/blob/master/ktorch.h>`_ - main header file, which, in turn includes headers from PyTorch.
- `private.h <https://github.com/ktorch/ktorch/blob/master/private.h>`_ - macros to gain access to private class elements, from `here <https://github.com/martong/access_private/blob/master/include/access_private.hpp>`_.
- `stb_image_write.h <https://github.com/ktorch/ktorch/blob/master/stb_image_write.h>`_ - minimal code to write .png files, from `here <https://github.com/nothings/stb/blob/master/stb_image_write.h>`_.

