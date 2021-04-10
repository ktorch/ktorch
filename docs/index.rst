.. ktorch documentation master file, created by
   sphinx-quickstart on Wed Jul  8 11:59:45 2020.

:github_url: https://github.com/ktorch/ktorch

k api to pytorch
==================================

PyTorch, a deep learning framework written in python,  also has a c++ library, libtorch, which is delivered as a single zip file
containing all the necessary Nvidia libraries and routines to build and train neural networks, along with basic linear algebra routines.

This interface links k to the c++ routines, but attempts to follow the line of the python interface, which also drives the design of the c++ library.

.. toctree::
   :glob:
   :maxdepth: 1
   :caption: Overview

   overview
   build
   start
   types
   pointers
   devices


.. toctree::
   :maxdepth: 1
   :caption: Tensors

   tensors
   complex
   sparse
   vectors
   dictionaries


.. toctree::
   :maxdepth: 1
   :caption: Modules

   modules

Index
=====

* :ref:`genindex`
* :ref:`search`
