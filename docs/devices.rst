.. index:: devices

Devices
=======

PyTorch has the capability to create or move tensors, modules and optimizers onto CPU or GPU devices.
Internally, PyTorch has a more varied set of devices than are allowed in the python or c++ interface;
the main device choices are CPU or Nvidia GPU's with compute capability >= 3.7 as of version 1.8.1.

From a q/k session, the following functions deal with CPU and CUDA devices: 

- ``device`` - query the device for the session or allocated object, e.g. tensor, vector, module, etc.
- ``cudadevice`` - query or set the default CUDA device if any available.
- ``cudadevices`` - query the count or names of the available CUDA devices.
- ``to`` - move previously allocated object to a different device.


Device
******

.. function:: device() -> sym
.. function:: device(ptr) -> sym

   | For any empty or null argument, returns ```cuda`` if any CUDA devices available, else ```cpu``.

   :param ptr obj: a previously allocated :ref:`api-pointer <pointers>` to a PyTorch object, e.g. a tensor, module, etc.
   :return: sym indicating the specific device the object's memory resides on.


Default CUDA device
*******************
.. function:: cudadevice() -> sym
.. function:: cudadevice(sym) -> (null)

   | For an empty or null argument, returns the specific CUDA device that is used when the generic symbol ```cuda`` is specified.

Available CUDA devices
**********************

.. function:: cudadevices() -> syms
.. function:: cudadevice(::) -> long

   | For any empty list, the function returns a list of symbols of available CUDA devices, both specific and generic. For null argument, returns the number of CUDA devices.

   ::

   / on host with 2 GPU's
   q)cudadevices[]
   2

   q)cudadevices()
   `cuda`cuda:0`cuda:1

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

