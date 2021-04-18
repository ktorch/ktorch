.. _build:

Building ktorch
===============

The ktorch library has been built on Linux and MacOS; it has not been tested on Windows.
(Windows is just beginning to get more support with `Microsoft becoming the maintainer of the Windows version in July 2020 <https://pytorch.org/blog/microsoft-becomes-maintainer-of-the-windows-version-of-pytorch/>`_.)

The first step is to `download the relevant zip file from PyTorch here <https://pytorch.org/get-started/locally/>`_.
The zip file contains all the necessary libraries and include files; there is no need to install CUDA or Intel MKL as these components are included.
The zip file is large, around 2 gigabytes for versions which include libraries for working with GPU's and around 150 megabytes for CPU-only.
Each platform (Linux, MacOS, Windows) has additional choices for CPU-only/GPU version.

.. figure:: linux-cuda11.1.png
   :scale: 40 %
   :alt: libtorch.zip files for linux and CUDA 11.1

   libtorch.zip files for linux, version 1.8.1 and CUDA 11.1


Once the zip file is downloaded and unzipped, the next step is to download the ktorch source code.

Get the code via git clone:

::

   > cd ~
   > git clone https://github.com/ktorch/ktorch.git
   Cloning into 'ktorch'...

Or download as a zip file:

::

   > wget --quiet https://github.com/ktorch/ktorch/archive/refs/heads/master.zip

   > unzip master.zip
   Archive:  master.zip
      creating: ktorch-master/
     inflating: ktorch-master/LICENSE   
     inflating: ktorch-master/Makefile  
     ..

.. index:: Makefile

Makefile
********

The `makefile <https://github.com/ktorch/ktorch/blob/master/Makefile>`_ can be changed to suit preferences.
There are 3 main variables, CXX, TORCH and ABI, that may need to be changed in the file itself or specified on the command line.

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

It may also be possible to point the make to the libraries already installed the python version of PyTorch.


::

   # find the dir for pytorch 1.8.1 libraries in mini conda
   find ~/miniconda3/lib  -name libtorch.so 
   /home/t/miniconda3/lib/python3.8/site-packages/torch/lib/libtorch.so

   cd ~/ktorch

   make TORCH=/home/t/miniconda3/lib/python3.8/site-packages/torch
   clang++ -std=c++14 -std=gnu++14 -pedantic -Wall -Wfatal-errors -fPIC -O3 -D_GLIBCXX_USE_CXX11_ABI=0 \
           -isystem /home/t/miniconda3/lib/python3.8/site-packages/torch/include \
           -isystem /home/t/miniconda3/lib/python3.8/site-packages/torch/include/torch/csrc/api/include \
           -c -o ktorch.o ktorch.cpp
   ..
   clang++ -o ktorch.so ktorch.o ktensor.o kmath.o knn.o kloss.o kopt.o kmodel.o ktest.o \
           -shared -L/home/t/miniconda3/lib/python3.8/site-packages/torch/lib -l torch   \
            -Wl,-rpath /home/t/miniconda3/lib/python3.8/site-packages/torch/lib

ABI
^^^

In Linux, there's a choice of ABI (application binary interface). Changes in the C++11 standard created
`a newer ABI <https://developers.redhat.com/blog/2015/02/05/gcc5-and-the-c11-abi/>`_.  The supplied libtorch zip files from PyTorch come in two versions,
one for the ABI prior to the changes for the C++11 standard, and one with the new ABI.

For example, for Linux, version 1.8.1, with support for CUDA 11.1, the zip files are listed as:

::

   Download here (Pre-cxx11 ABI):
   https://download.pytorch.org/libtorch/cu111/libtorch-shared-with-deps-1.8.1%2Bcu111.zip

   Download here (cxx11 ABI):
   https://download.pytorch.org/libtorch/cu111/libtorch-cxx11-abi-shared-with-deps-1.8.1%2Bcu111.zip

In the earliest versions, PyTorch only offered the older ABI version of the zip file so users could maintain compatibility with older third-party libraries compiled under the old ABI, but now offer the choice of old or new versions.
By default, the Makefile builds code with ``-D_GLIBCXX_USE_CXX11_ABI=0`` for the older API.
The Makefile variable ``ABI`` is set to 0, but can be overwritten with the command-line call ``ABI=1`` if the newer ABI zip file is used.

::

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

Sample builds
*************

macOS, CPU only
^^^^^^^^^^^^^^^

Linux, CUDA 11.1
^^^^^^^^^^^^^^^^

Linked libraries
****************


::

   > ldd ktorch.so
	linux-vdso.so.1 (0x00007ffd8952d000)
	libtorch.so => /home/t/libtorch/lib/libtorch.so (0x00007efdbd422000)
	libstdc++.so.6 => /usr/lib/x86_64-linux-gnu/libstdc++.so.6 (0x00007efdbd099000)
	libm.so.6 => /lib/x86_64-linux-gnu/libm.so.6 (0x00007efdbccfb000)
	libgcc_s.so.1 => /lib/x86_64-linux-gnu/libgcc_s.so.1 (0x00007efdbcae3000)
	libc.so.6 => /lib/x86_64-linux-gnu/libc.so.6 (0x00007efdbc6f2000)
	/lib64/ld-linux-x86-64.so.2 (0x00007efdbdca7000)
	libtorch_cuda.so => /home/t/libtorch/lib/libtorch_cuda.so (0x00007efdae8db000)
	libtorch_cuda_cu.so => /home/t/libtorch/lib/libtorch_cuda_cu.so (0x00007efd5bcb6000)
	libtorch_cpu.so => /home/t/libtorch/lib/libtorch_cpu.so (0x00007efd49843000)
	libtorch_cuda_cpp.so => /home/t/libtorch/lib/libtorch_cuda_cpp.so (0x00007efcd49ea000)
	librt.so.1 => /lib/x86_64-linux-gnu/librt.so.1 (0x00007efcd47e2000)
	libdl.so.2 => /lib/x86_64-linux-gnu/libdl.so.2 (0x00007efcd45de000)
	libpthread.so.0 => /lib/x86_64-linux-gnu/libpthread.so.0 (0x00007efcd43bf000)
	libcudart-6d56b25a.so.11.0 => /home/t/libtorch/lib/libcudart-6d56b25a.so.11.0 (0x00007efcd4136000)
	libc10_cuda.so => /home/t/libtorch/lib/libc10_cuda.so (0x00007efcd3f06000)
	libnvToolsExt-24de1d56.so.1 => /home/t/libtorch/lib/libnvToolsExt-24de1d56.so.1 (0x00007efcd3cfc000)
	libc10.so => /home/t/libtorch/lib/libc10.so (0x00007efcd3a65000)
	libgomp-7c85b1e2.so.1 => /home/t/libtorch/lib/libgomp-7c85b1e2.so.1 (0x00007efcd383b000)


Defining api functions in k
***************************

::

   q)(`ktorch 2:`fns,1)[]
   dv         | code
   tree       | code
   addref     | code
   free       | code
   ..
