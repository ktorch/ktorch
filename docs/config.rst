Configuration
=============

Given a successful build of the ``ktorch.so`` interface library, it's possible to view the build configuration of the associated libraries from Pytorch's libtorch.zip and verify that various expected components are in place.

.. function:: config() -> strings
.. function:: config(::) -> (null)

	| Returns a list of strings containing the configuration output, or with null argument (as opposed to an empty list), prints configuration to stderr


On a linux machine with dual Nvidia GTX 1080 gpu's, ``config`` output looks as follows vor PyTorch version 1.8.1:

::

   q)config()
   "PyTorch built with:"
   "  - GCC 7.3"
   "  - C++ Version: 201402"
   "  - Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191122 for..
   "  - Intel(R) MKL-DNN v1.7.0 (Git Hash 7aed236906b1f7a05c0917e5257a1af05e9ff6..
   "  - OpenMP 201511 (a.k.a. OpenMP 4.5)"
   "  - NNPACK is enabled"
   ..

   q)config[]
   PyTorch built with:
     - GCC 7.3
     - C++ Version: 201402
     - Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191122 for Intel(R) 64 architecture applications
     - Intel(R) MKL-DNN v1.7.0 (Git Hash 7aed236906b1f7a05c0917e5257a1af05e9ff683)
     - OpenMP 201511 (a.k.a. OpenMP 4.5)
     - NNPACK is enabled
     - CPU capability usage: AVX2
     - CUDA Runtime 11.1
     - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86
     - CuDNN 8.0.5
     - Magma 2.5.2
     - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.1, CUDNN_VERSION=8.0.5, CXX_COMPILER=/opt/rh/devtoolset-7/root/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -fopenmp -DNDEBUG -DUSE_KINETO -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -O2 -fPIC -Wno-narrowing -Wall -Wextra -Werror=return-type -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wno-sign-compare -Wno-unused-parameter -Wno-unused-variable -Wno-unused-function -Wno-unused-result -Wno-unused-local-typedefs -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_VERSION=1.8.1, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, 
   
   ATen/Parallel:
	   at::get_num_threads() : 12
	   at::get_num_interop_threads() : 6
   OpenMP 201511 (a.k.a. OpenMP 4.5)
	   omp_get_max_threads() : 12
   Intel(R) Math Kernel Library Version 2020.0.0 Product Build 20191122 for Intel(R) 64 architecture applications
	   mkl_get_max_threads() : 6
   Intel(R) MKL-DNN v1.7.0 (Git Hash 7aed236906b1f7a05c0917e5257a1af05e9ff683)
   std::thread::hardware_concurrency() : 12
   Environment variables:
	   OMP_NUM_THREADS : [not set]
	   MKL_NUM_THREADS : [not set]
   ATen parallel backend: OpenMP

Settings
********

After reviewing the basic configuration that went into the build of ``libtorch``, it is also possible to query and set various flags that enable/disable certain features in the k interface.  See PyTorch `backends <https://pytorch.org/docs/stable/backends.html>`_  and `threads <https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html#build-options>`_ for more information.

.. function:: setting() -> dictionary
.. function:: setting(sym) -> value
.. function:: setting(sym;bool) -> null
.. function:: setting(sym;long) -> null


Stack frame
^^^^^^^^^^^
Setting ```stackframe`` true will cause the k interface, on error, to issue a message that contains information on the stack frames that can sometimes help locate where in the source code the error originated. 

::

   q)setting`stackframe   / by default, stackframe is turned off
   0b

   q)m:module enlist(`linear;1;2)
   q)forward(m;1 2)
   'mat1 and mat2 shapes cannot be multiplied (1x2 and 1x2)
     [0]  forward(m;1 2)
          ^

   q)setting`stackframe,1b   / turn stackframe on

   q)forward(m;1 2)
   'mat1 and mat2 shapes cannot be multiplied (1x2 and 1x2)
   Exception raised from addmm_impl_cpu_ at /pytorch/aten/src/ATen/native/LinearAlgebra.cpp:468 (most recent call first):
   frame #0: c10::Error::Error(c10::SourceLocation, std::string) + 0x42 (0x7fedffb7b2f2 in /home/t/libtorch/lib/libc10.so)
   frame #1: c10::detail::torchCheckFail(char const*, char const*, unsigned int, std::string const&) + 0x5b (0x7fedffb7867b in /home/t/libtorch/lib/libc10.so)
   frame #2: at::native::addmm_cpu_out(at::Tensor&, at::Tensor const&, at::Tensor const&, at::Tensor const&, c10::Scalar, c10::Scalar) + 0x75e (0x7fee7602eb1e in /home/t/libtorch/lib/libtorch_cpu.so)
   frame #3: at::native::mm_cpu(at::Tensor const&, at::Tensor const&) + 0xf1 (0x7fee760342b1 in /home/t/libtorch/lib/libtorch_cpu.so)
   ..
   frame #30: /home/t/q/l64/q() [0x4044d8]
   frame #31: __libc_start_main + 0xe7 (0x7feef714fbf7 in /lib/x86_64-linux-gnu/libc.so.6)
   frame #32: /home/t/q/l64/q() [0x4045b1]

     [0]  forward(m;1 2)
          ^

Show all options
^^^^^^^^^^^^^^^^
By default, setting ```alloptions`` is turned on to return all options for a particular module.  Turning this setting off means, by default, when a module configuration is queried, only the non-default options will be returned, which can make for a simpler module definition.

::


   q)setting `alloptions
   1b

   q)help`conv2d   / give some sample values for all possible options
   in     | 16
   out    | 32
   size   | 3 5
   stride | 1
   pad    | 0
   dilate | 1
   groups | 1
   bias   | 1b
   padmode| `zeros

   q)m:module enlist(`conv2d;8;16;4)

   q)exec options from module m
   in     | 8
   out    | 16
   size   | 4
   stride | 1
   pad    | 0
   dilate | 1
   groups | 1
   bias   | 1b
   padmode| `zeros

   q)setting `alloptions,0b  / show only non-defaults

   q)exec options from module m
   in  | 8
   out | 16
   size| 4

   q)exec options from module(m;1b)   / overide session setting by explicitly requesting all options
   in     | 8
   out    | 16
   size   | 4
   stride | 1
   pad    | 0
   dilate | 1
   groups | 1
   bias   | 1b
   padmode| `zeros

Complex dimension
^^^^^^^^^^^^^^^^^
