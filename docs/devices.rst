.. index:: devices

Devices
=======

PyTorch has the capability to create or move tensors, modules and optimizers onto CPU or GPU devices.
Internally, PyTorch has a more varied set of devices than are allowed in the python or c++ interface;
the main device choices are CPU or Nvidia GPU's with compute capability >= 3.7 as of version 1.8.1.

From a q/k session, the following functions deal with CPU and CUDA devices: 

- device - query the device for the session or allocated object, e.g. tensor, vector, module, etc.
- cudadevice: query or set the default CUDA device if any available.
- cudadevices: query the count or names of the available CUDA devices.
- to: move previously allocated object to a different device.


Device
******

.. function:: device() -> sym
.. function:: device(ptr) -> sym


Default CUDA device
*******************
.. function:: cudadevice() -> sym
.. function:: cudadevice(sym) -> (null)

Available CUDA devices
**********************

.. function:: cudadevices() -> syms
.. function:: cudadevice(::) -> long

Moving to device
****************
.. function:: to(ptr;options) -> syms


| On a host with 2 GPU's:

.. index:: cudadevices, CUDA

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

