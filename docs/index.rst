.. ktorch documentation master file, created by
   sphinx-quickstart on Wed Jul  8 11:59:45 2020.

:github_url: https://github.com/ktorch/ktorch

k api to pytorch
==================================

`PyTorch <https://pytorch.org/>`_, a deep learning framework written in python,  also has a c++ library, `libtorch <https://pytorch.org/cppdocs/>`_,
which is delivered as a single zip file containing all the necessary Nvidia libraries and routines to build and train neural networks,
along with basic linear algebra routines.

This interface links k to the c++ routines, but attempts to follow the line of the python interface, which also drives the design of the c++ library.

.. toctree::
   :glob:
   :maxdepth: 1

   overview
   build
   config
   start
   types
   pointers
   devices

Tensors
^^^^^^^

.. toctree::
   :maxdepth: 1

   tensors
   complex
   sparse
   vectors
   dictionaries
   info
   tensorop
   math

Modules
^^^^^^^

.. toctree::
   :maxdepth: 1

   modules
   kmodules
   init
   loss

Training
^^^^^^^^

.. toctree::
   :maxdepth: 1

   opt 
   model
   train
   swa
   freeze
   dist
 
Index
=====

* :ref:`genindex`
* :ref:`search`
