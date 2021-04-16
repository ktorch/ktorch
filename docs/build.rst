.. _build:

Building ktorch
===============

The first step is to `download the relevant zip file from PyTorch <https://pytorch.org/get-started/locally/>`_.

Once the zip file is downloaded and unzipped, the next step is to download the ktorch source code.

::

   > cd ~
   > git clone https://github.com/ktorch/ktorch.git
   Cloning into 'ktorch'...

   > cd ktorch

   > ls
   LICENSE			k.h			kmodel.cpp		ktensor.cpp		private.h
   Makefile		kloss.cpp		knn.cpp			ktest.cpp		stb_image_write.h
   README.md		kloss.h			knn.h			ktorch.cpp
   docs/			kmath.cpp		kopt.cpp		ktorch.h



- LICENSE
- Makefile
- README.md
- docs/
- k.h
- kloss.cpp - code relating to loss functions
- kloss.h - redefine binary cross entropy loss functions
- kmath.cpp - PyTorch math routines
- kmodel.cpp - code for building models (module + optimizer + loss function)
- knn.cpp - code for building modules
- knn.h - custom modules defined here
- kopt.cpp - optimizer code
- ktensor.cpp - code for operating on tensors
- ktest.cpp - contains temporary tests, samples, etc. -- nothing essential to the interface library
- ktorch.cpp - contains the code used by the rest of system dealing with tensors, modules, optimizers, etc.
- ktorch.h - main header file, which includes PyTorch headers
- private.h
- stb_image_write.h
