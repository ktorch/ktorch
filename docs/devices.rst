.. index:: devices

Devices
=======

PyTorch has the capability to create or move tensors, modules and optimizers onto CPU or GPU devices.
Internally, PyTorch has a more varied set of devices than are allowed in the python or c++ interface;
the main device choices are CPU or Nvidia GPU's with compute capability >= 3.7 as of version 1.8.1.

Specifiying ```cuda`` without a device index implies the default CUDA device -- typically ```cuda:0``,
but would mean ```cuda:1`` if the default CUDA device were switched to the second GPU.

From a k session, the following functions deal with CPU and CUDA devices: 

- ``device`` - query the device for the session or allocated object, e.g. tensor, vector, module, etc.
- ``cudadevice`` - query or set the default CUDA device if any available.
- ``cudadevices`` - query the count or names of the available CUDA devices.
- ``to`` - move previously allocated object to a different device.


.. index:: cudadevice

Device
^^^^^^

.. function:: device() -> sym

   | For any empty or null argument, returns ```cuda`` if any CUDA devices available, else ```cpu``.

.. function:: device(ptr) -> sym

   :param ptr obj: a previously allocated :ref:`api-pointer <pointers>` to a PyTorch object, e.g. a tensor, module, etc.
   :return: sym indicating the specific device where object's memory resides.

::

   q)device() / on machine with 2 GPUs
   `cuda

   q)t:tensor 1 2 3
   q)device t
   `cpu

   q)to(t;`cuda:1)
   q)device t
   `cuda:1

   q)a:tensor(1 2;`cuda)
   q)b:tensor(3 4;`cuda:1)
   q)c:tensor(5 6 7;`cpu)
   q)d:dict `a`b`c!(a;b;c)

   q)device d
   a| cuda:0
   b| cuda:1
   c| cpu


.. index:: cudadevice

Default CUDA device
^^^^^^^^^^^^^^^^^^^
.. function:: cudadevice() -> sym

   | For an empty or null argument, returns the specific CUDA device that is used when the generic symbol ```cuda`` is specified.

.. function:: cudadevice(device) -> (null)
   :param sym device: a specific cuda device with index specified, e.g. ```cuda:0``
   :return: (null)

::

   q)cudadevice()
   `cuda:0

   q)cudadevice`cuda
   'unrecognized CUDA device, expecting cuda with valid device number, e.g. `cuda:0
     [0]  cudadevice`cuda
          ^

   q)cudadevice `cuda:1

   q)t:tensor(1 2 3;`cuda)
   q)device t
   `cuda:1


.. index:: cudadevices, CUDA

Available CUDA devices
^^^^^^^^^^^^^^^^^^^^^^

.. function:: cudadevices() -> syms
.. function:: cudadevice(::) -> long

   | For any empty list, the function returns a list of symbols of available CUDA devices, both specific and generic. For null argument, returns the number of CUDA devices.

::

   q)cudadevices[]     / on host with 2 GPU's
   2

   q)cudadevices()
   `cuda`cuda:0`cuda:1

Moving to device
^^^^^^^^^^^^^^^^

Once a PyTorch object is established on a device, it can be moved with the :func:`to`.
The typical case is to create a tensor or module on a device, then move to a CUDA device via ```to``.


.. function:: to(ptr;options) -> syms

