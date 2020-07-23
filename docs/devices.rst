.. index: devices

Devices
=======

PyTorch has the capability to create or move tensors, modules and optimizers onto CPU or GPU devices.
Internally, PyTorch has a more varied set of devices than are allowed in the python or c++ interface;
the main device choices are CPU or Nvidia GPU's with compute capability >= 3.0.

| From a q/k session, there are a few functions that deal with CUDA devices. 
| On a host with 2 GPU's:

.. index: cudadevices, CUDA
::

   q)cudadevices[]
   2

   q)cudadevices()
   `cuda`cuda:0`cuda:1

On a host without any GPU's:

::

   q)cudadevices[]
   0

   q)cudadevices()
   `symbol$()

