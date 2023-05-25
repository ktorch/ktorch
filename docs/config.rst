Configuration
=============

Given a successful build of the ``ktorch.so`` interface library, it's possible to view the build configuration of the associated libraries from Pytorch's libtorch.zip and verify that various expected components are in place.

.. function:: config() -> strings
.. function:: config(::) -> (null)
   :noindex:

	| Returns a list of strings containing the configuration output, or with null argument (as opposed to an empty list), prints the configuration to stderr


On a linux machine with dual Nvidia GTX 1080 gpu's, ``config`` output looks as follows for PyTorch version 2.0.1:

::

   q)config()                  / return strings
   "PyTorch built with:"
   "  - GCC 9.3"
   "  - C++ Version: 201703"
   "  - Intel(R) oneAPI Math Kernel Library Version 2022.2-Product Build 2022080..
   "  - Intel(R) MKL-DNN v2.7.3 (Git Hash 6dbeffbae1f23cbbeae17adb7b5b13f1f37c08..
   "  - OpenMP 201511 (a.k.a. OpenMP 4.5)"
   "  - LAPACK is enabled (usually provided by MKL)"
   "  - NNPACK is enabled"
   "  - CPU capability usage: AVX2"
   "  - CUDA Runtime 11.7"
   "  - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;ar..
   "  - CuDNN 8.5"
   "  - Magma 2.6.1"
   "  - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.7, CU..
   ..

::

   q)config[]                 / print to stderr
   PyTorch built with:
     - GCC 9.3
     - C++ Version: 201703
     - Intel(R) oneAPI Math Kernel Library Version 2022.2-Product Build 20220804 for Intel(R) 64 architecture applications
     - Intel(R) MKL-DNN v2.7.3 (Git Hash 6dbeffbae1f23cbbeae17adb7b5b13f1f37c080e)
     - OpenMP 201511 (a.k.a. OpenMP 4.5)
     - LAPACK is enabled (usually provided by MKL)
     - NNPACK is enabled
     - CPU capability usage: AVX2
     - CUDA Runtime 11.7
     - NVCC architecture flags: -gencode;arch=compute_37,code=sm_37;-gencode;arch=compute_50,code=sm_50;-gencode;arch=compute_60,code=sm_60;-gencode;arch=compute_70,code=sm_70;-gencode;arch=compute_75,code=sm_75;-gencode;arch=compute_80,code=sm_80;-gencode;arch=compute_86,code=sm_86
     - CuDNN 8.5
     - Magma 2.6.1
     - Build settings: BLAS_INFO=mkl, BUILD_TYPE=Release, CUDA_VERSION=11.7, CUDNN_VERSION=8.5.0, CXX_COMPILER=/opt/rh/devtoolset-9/root/usr/bin/c++, CXX_FLAGS=-Wno-deprecated-declarations -D_GLIBCXX_USE_CXX11_ABI=0 -fabi-version=11 -Wno-deprecated -fvisibility-inlines-hidden -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOROCTRACER -DUSE_FBGEMM -DUSE_QNNPACK -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=bool-operation -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wunused-local-typedefs -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wno-stringop-overflow -Wno-psabi -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -fdiagnostics-color=always -faligned-new -Wno-unused-but-set-variable -Wno-maybe-uninitialized -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -Wno-stringop-overflow, LAPACK_INFO=mkl, PERF_WITH_AVX=1, PERF_WITH_AVX2=1, PERF_WITH_AVX512=1, TORCH_DISABLE_GPU_ASSERTS=ON, TORCH_VERSION=2.0.1, USE_CUDA=ON, USE_CUDNN=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=ON, USE_MKLDNN=ON, USE_MPI=OFF, USE_NCCL=ON, USE_NNPACK=ON, USE_OPENMP=ON, USE_ROCM=OFF, 

   ATen/Parallel:
   	at::get_num_threads() : 6
   	at::get_num_interop_threads() : 6
   OpenMP 201511 (a.k.a. OpenMP 4.5)
   	omp_get_max_threads() : 6
   Intel(R) oneAPI Math Kernel Library Version 2022.2-Product Build 20220804 for Intel(R) 64 architecture applications
   	mkl_get_max_threads() : 6
   Intel(R) MKL-DNN v2.7.3 (Git Hash 6dbeffbae1f23cbbeae17adb7b5b13f1f37c080e)
   std::thread::hardware_concurrency() : 12
   Environment variables:
   	OMP_NUM_THREADS : [not set]
   	MKL_NUM_THREADS : [not set]
   ATen parallel backend: OpenMP

The configuration on a macbook with the M2 chip:

::

   q)config[]
   PyTorch built with:
     - GCC 4.2
     - C++ Version: 201703
     - clang 13.1.6
     - LAPACK is enabled (usually provided by MKL)
     - NNPACK is enabled
     - CPU capability usage: NO AVX
     - Build settings: BLAS_INFO=accelerate, BUILD_TYPE=Release, CXX_COMPILER=/Applications/Xcode_13.3.1.app/Contents/Developer/Toolchains/XcodeDefault.xctoolchain/usr/bin/c++, CXX_FLAGS= -Wno-deprecated -fvisibility-inlines-hidden -Wno-deprecated-declarations -DUSE_PTHREADPOOL -DNDEBUG -DUSE_KINETO -DLIBKINETO_NOCUPTI -DLIBKINETO_NOROCTRACER -DUSE_PYTORCH_QNNPACK -DUSE_XNNPACK -DUSE_PYTORCH_METAL_EXPORT -DSYMBOLICATE_MOBILE_DEBUG_HANDLE -DUSE_COREML_DELEGATE -O2 -fPIC -Wall -Wextra -Werror=return-type -Werror=non-virtual-dtor -Werror=braced-scalar-init -Werror=range-loop-construct -Werror=bool-operation -Winconsistent-missing-override -Wnarrowing -Wno-missing-field-initializers -Wno-type-limits -Wno-array-bounds -Wno-unknown-pragmas -Wunused-local-typedefs -Wno-unused-parameter -Wno-unused-function -Wno-unused-result -Wno-strict-overflow -Wno-strict-aliasing -Wno-error=deprecated-declarations -Wvla-extension -Wno-range-loop-analysis -Wno-pass-failed -Wsuggest-override -Wno-error=pedantic -Wno-error=redundant-decls -Wno-error=old-style-cast -Wconstant-conversion -Wno-invalid-partial-specialization -Wno-typedef-redefinition -Wno-unused-private-field -Wno-inconsistent-missing-override -Wno-constexpr-not-const -Wno-missing-braces -Wunused-lambda-capture -Wunused-local-typedef -Qunused-arguments -fcolor-diagnostics -fdiagnostics-color=always -fno-math-errno -fno-trapping-math -Werror=format -Werror=cast-function-type -DUSE_MPS -fno-objc-arc -Wno-unguarded-availability-new -Wno-unused-private-field -Wno-missing-braces -Wno-constexpr-not-const, LAPACK_INFO=accelerate, TORCH_DISABLE_GPU_ASSERTS=OFF, TORCH_VERSION=2.0.1, USE_CUDA=OFF, USE_CUDNN=OFF, USE_EIGEN_FOR_BLAS=ON, USE_EXCEPTION_PTR=1, USE_GFLAGS=OFF, USE_GLOG=OFF, USE_MKL=OFF, USE_MKLDNN=OFF, USE_MPI=OFF, USE_NCCL=OFF, USE_NNPACK=ON, USE_OPENMP=OFF, USE_ROCM=OFF, 

   ATen/Parallel:
   	at::get_num_threads() : 12
   	at::get_num_interop_threads() : 12
   OpenMP not found
   MKL not found
   MKLDNN not found
   std::thread::hardware_concurrency() : 12
   Environment variables:
   	OMP_NUM_THREADS : [not set]
   	MKL_NUM_THREADS : [not set]
   ATen parallel backend: native thread pool

.. _settings:

.. index::  settings; k session settings

Settings
********

After reviewing the basic configuration that went into the build of ``libtorch``, it is also possible to query and set various flags that enable/disable certain features in the k interface.  See PyTorch `backends <https://pytorch.org/docs/stable/backends.html>`_  and `threads <https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html#build-options>`_ for more information.

.. function:: setting() -> dictionary
.. function:: setting(sym) -> value
   :noindex:
.. function:: setting(sym;bool) -> null
   :noindex:
.. function:: setting(sym;long) -> null
   :noindex:

	| Calling the function with null or an empty list returns a dictionary of setting names and values. Specifying a single symbol returns the current setting. Specifying a symbol and boolean or long scalar will reset the session setting if changes are possible for that setting.

::

   q)setting()
   mkl               | 1b
   openmp            | 1b
   threads           | 6
   interopthreads    | 6
   mps               | 0b
   cuda              | 1b
   magma             | 1b
   cudnn             | 1b
   cudnnversion      | 8500
   cudadevices       | 2
   benchmark         | 0b
   deterministic     | 0
   cudnndeterministic| 0b
   stackframe        | 0b
   alloptions        | 1b
   complexfirst      | 1b

   q)setting `threads
   6

   q)setting `threads,12

   q)setting `threads
   12

   q)setting `cuda,0b
   'setting: cannot set flag for cuda
     [0]  setting `cuda,0b
          ^

.. index::  settings; MKL

MKL
^^^

The read-only setting ```mkl`` indicates if the PyTorch libraries for were built with support from Intel's Math Kernel Library.
See PyTorch `build options <https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html?highlight=threads#build-options>`_ for more detail.

.. index::  settings; OpenMP

OpenMP
^^^^^^

The read-only setting ```openmp`` indicates if the Pytorch libraries were built with OpenMP support, which handles cpu threading and shared memory.
See PyTorch `build options <https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html?highlight=threads#build-options>`_ for more detail.

.. index::  settings; CPU threads

Threads
^^^^^^^

The ```threads`` setting is used to get and set the number of threads used for parallelizing CPU operations and ```interopthreads`` controls the number of threads used across operations.
PyTorch has `more detail on threads <https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html>`_
and `tuning the number of threads <https://pytorch.org/docs/stable/notes/cpu_threading_torchscript_inference.html?highlight=threads#tuning-the-number-of-threads>`_.

::

   / l64 12(16)core 64037MB 

   q)x:tensor(`randn;1024 1024)  / random test matrices
   q)y:tensor(`randn;1024 1024)
   q)z:tensor()                  / empty output tensor

   q)mm(x;y;z)                   /  x * y -> z
   q)size z
   1024 1024

   q)setting`threads,1
   q)\ts:100 mm(x;y;z)
   1603 1120

   q)setting`threads,2   / 2 threads nearly cuts the time in half
   q)\ts:100 mm(x;y;z)
   815 1120

   q)setting`threads,4   / 4 threads still cuts the time proportionally
   q)\ts:100 mm(x;y;z)
   437 1120

   q)setting`threads,6   / 6 threads, improvement, but not quite proportional..
   q)\ts:100 mm(x;y;z)
   318 1120

   q)setting`threads,8   / 8 threads begins to slow things down
   q)\ts:100 mm(x;y;z)
   437 1120

.. index::  settings; MPS

MPS
^^^

The read-only setting ```mps`` indicates if `Apple's Metal Performance Shaders <https://pytorch.org/blog/introducing-accelerated-pytorch-training-on-mac/>`_ are available.

On a macbook pro with the M2 max chip:

::

   q)setting[]
   mkl               | 0b
   openmp            | 0b
   threads           | 12
   interopthreads    | 12
   mps               | 1b 
   cuda              | 0b
   magma             | 0b
   cudnn             | 0b
   cudnnversion      | 0N
   cudadevices       | 0
   ..

.. index::  settings; CUDA

CUDA
^^^^

The read-only setting ```cuda`` indicates if CUDA is avalable to the k session. The PyTorch libraries in ``libtorch`` that were used to build the ``ktorch.so`` library must have included CUDA support and the current machine needs working CUDA drivers and devices.  The ```cudadevices`` setting returns the number of GPU's that are available to the session.

.. index::  settings; MAGMA

MAGMA
^^^^^

`MAGMA <https://developer.nvidia.com/magma>`_ is a set of linear algebra routines for Nvidia GPUs that is included the PyTorch libraries for most recent builds -- the setting ```magma`` indicates if the k interface has magma capabilities.


.. index::  settings; CuDNN

CuDNN
^^^^^
`CuDNN <https://developer.nvidia.com/cudnn>`_ is a GPU library of routines for neural networks that should be included in the PyTorch libraries that were built with CUDA support.  The flag ```cudnn`` indicates that the routines are available and ```cudnnversion`` returns the version as a long integer, e.g. 8005 for version ``8.0.5``, 8200 for version ``8.200``.

.. _benchmark:

.. index::  settings; benchmark mode

Benchmark mode
^^^^^^^^^^^^^^
The ```benchmark`` setting indicates if CuDNN will benchmark multiple convolution algorithms and select the fastest for the available GPU hardware and problem size.  Benchmark mode is off by default, but turning it on often leads to faster training times.  If the model  being trained has variable problem sizes, variable inputs or layers that are not always activated, this may trigger too much benchmarking and slower training times.

::

   q)setting`benchmark
   0b

   q)device[]   / returns default CUDA device if any available, else `cpu
   `cuda

   q)cudadevices()  / list of available CUDA devices
   `cuda`cuda:0`cuda:1

   q)setting`benchmark
   1b


.. index::  settings; deterministic mode

Deterministic mode
^^^^^^^^^^^^^^^^^^
Setting the random seed can help in creating reproducible results, but some algorithms have random elements that are difficult to reproduce exactly.
See PyTorch notes on `reproducibility <https://pytorch.org/docs/stable/notes/randomness.html>`_.

There are two settings, ```deterministic`` and ```cudnndeterministic``, both turned off by default, that indicate whether PyTorch operations must use “deterministic” algorithms. That is, algorithms which, given the same input, and when run on the same software and hardware, always produce the same output.

When ```deterministic`` is set to ``2``, operations will use deterministic algorithms when available, and if only non-deterministic algorithms are available they will throw an error. If set to ``1``, `no error, only warnings <https://pytorch.org/docs/stable/generated/torch.set_deterministic_debug_mode.html>`_. See PyTorch for the `list of algorithms <https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html>`_ that will throw errors if invoked with deterministic flag(s) turned on.

With CUDA :ref:`Benchmark mode <benchmark>` turned off, CUDA routines will select the same algorithm at each run rather than testing a set and picking the one with the best benchmark.  But this chosen algorithm may not be deterministic unless either ```deterministic`` or ```cudnndeterministic`` is set true.
Either setting turned on causes CUDA to select a deterministic algorithm if possible.
If only ```cudnndeterministic`` is set true, then only the CUDA algorithm selection is affected.

::

   q)`deterministic`cudnndeterministic # setting()
   deterministic     | 0
   cudnndeterministic| 0b

   / bincount is example of CUDA algorithm with no deterministic implementation
   q)t:tensor(0 1 2 3 3 1 1 2;`cuda)
   q)distinct[tensor t]!tensor n:bincount t
   0| 1
   1| 3
   2| 2
   3| 2

   q)setting`deterministic,1  /warn only
   q)distinct[tensor t]!tensor n:bincount t
   [W Context.cpp:79] Warning: _bincount_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True, warn_only=True)'. You can file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation. (function alertNotDeterministic)
   0| 1
   1| 3
   2| 2
   3| 2

   q)setting`deterministic,2  /error if non-deterministic
   q)distinct[tensor t]!tensor n:bincount t
   '_bincount_cuda does not have a deterministic implementation, but you set 'torch.use_deterministic_algorithms(True)'. You can turn off determinism just for this operation, or you can use the 'warn_only=True' option, if that's acceptable for your application. You can also file an issue at https://github.com/pytorch/pytorch/issues to help us prioritize adding deterministic support for this operation.
     [0]  distinct[tensor t]!tensor n:bincount t
                                      ^

.. index::  settings; stack frame

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

.. _alloptions:
.. index::  settings; all options

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

   / overide session setting by explicitly requesting all options
   q)exec options from module(m;1b)
   in     | 8
   out    | 16
   size   | 4
   stride | 1
   pad    | 0
   dilate | 1
   groups | 1
   bias   | 1b
   padmode| `zeros

.. _complex-first:

.. index::  settings; complexfirst

Complex dimension
^^^^^^^^^^^^^^^^^

When complex tensors are returned as k values, the real and imaginary parts can be separated along the first or the last dimension.
The flag for using the first dimension can be specified explicitly when creating or retrieving a complex tensor,
but when the flag is omitted, the default setting is specified with the symbol ```complexfirst``.

::

   q)setting `complexfirst
   1b

   q)t:tensor(`complex;1 2 3;-1 0 2)
   q)tensor t
   1  2 3
   -1 0 2

   q)tensor(t;0b)
   1 -1
   2 0 
   3 2 

   q)setting `complexfirst,0b
 
   q)tensor t
   1 -1
   2 0 
   3 2 


Version
*******

Returns the version of the libtorch libraries from PyTorch. Return numeric version if null argument and string version if empty list given.

.. function:: version() -> string
.. function:: version(::) -> double
   :noindex:

::

   q)version()
   "1.10.1"

   q)version[]  / return as double, e.g. 1.0801 for version 1.8.1
   1.1001

