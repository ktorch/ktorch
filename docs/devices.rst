.. index:: devices
.. _devices:

Devices
=======

PyTorch has the capability to create or move tensors, modules and optimizers onto CPU or GPU devices.
Internally, PyTorch has a more varied set of devices than are allowed in the python or c++ interface;
the main device choices are CPU or Nvidia GPU's with compute capability >= 3.7 as of version 1.9.0.

Specifiying ```cuda`` without a device index implies the default CUDA device -- typically ```cuda:0``,
but would mean ```cuda:1`` if the default CUDA device were switched to the second GPU.

From a k session, the following functions deal with CPU and CUDA devices: 

- ``device`` - query the device for the session or allocated object, e.g. tensor, vector, module, etc.
- ``cudadevice`` - query or set the default CUDA device if any available.
- ``cudadevices`` - query the count or names of the available CUDA devices.
- ``to`` - move previously allocated object to a different device.


.. _dtype:

.. index:: device

Device
^^^^^^

.. function:: device() -> sym

   | For any empty or null argument, returns ```cuda`` if any CUDA devices available, else ```cpu``.

.. function:: device(ptr) -> sym

   :param pointer ptr: a previously allocated :doc:`api-pointer <pointers>` to a PyTorch object, e.g. a tensor, module, etc.
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

   :return: Given an empty or null argument, returns the specific CUDA device that is used when the generic symbol ```cuda`` is specified.

.. function:: cudadevice(device) -> null

   :param sym device: a specific cuda device with index specified, e.g. ```cuda:0``
   :return: null

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

.. index:: to; tensor and module

Moving to device
^^^^^^^^^^^^^^^^

Once a PyTorch object is established on a device, it can be moved with :func:`to`.
The typical case is to create a tensor or module on the host, then move to a CUDA device.
This k interface function is designed to behave somewhat like 
`PyTorch's tensor.to() <https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to>`_ and
`module.to() <https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=module#torch.nn.Module.to>`_  methods.

.. function:: to(ptr;options) -> null
.. function:: to(ptr;options;async) -> null

   :param ptr ptr: a previously allocated :doc:`api-pointer <pointers>` to a tensor, vector, dictionary or module.
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <tensor-attributes>`.
   :param bool async: asynchronous flag, default is false. If true, will attempt to perform host to CUDA device transfer without blocking.
   :return: Null return, supplied pointer now has specified data type, memory, device, etc.

An alternate form uses an example tensor instead of specified options to define the target device and data type.

.. function:: to(ptr;example) -> null

.. function:: to(ptr;example;async) -> null

   :param tensor ptr: a previously allocated :doc:`api-pointer <pointers>` to a tensor.
   :param tensor example: an :doc:`api-pointer <pointers>` to a previously allocated tensor whose device and datatype will be used.
   :param bool async: asynchronous flag, default is false. If true, will attempt to perform host to CUDA device transfer without blocking.
   :return: null, supplied pointer now has same device and data type as given example tensor.

::

   q)a:options t:tensor 1 2 3    / create tensor of longs on cpu
   q)ptr t                       / get internal PyTorch shared pointer to tensor
   60520816

   q)to(t;`cuda`double`grad)     / convert to CUDA tensor on default GPU, type double
   q)ptr t                       / new interal pointer, k interface ptr is unchanged
   1814122272

   q)(a;options t)               / compare options to verify the change
   device dtype  layout  gradient pin      memory    
   --------------------------------------------------
   cpu    long   strided nograd   unpinned contiguous
   cuda:0 double strided grad     unpinned contiguous

   q)to(t;`cuda`double`grad)     / call to() again
   q)ptr t                       / same internal ptr -- tensor attributes unchanged
   1814122272

   q)e:tensor()  / empty tensor
   q)to(e;t)     / use t as an example tensor

   q)options e
   device  | cuda:0     / device changed
   dtype   | double     / data type changed
   layout  | strided
   gradient| nograd     / gradient unset (only device & dtype from example tensor)
   pin     | unpinned
   memory  | contiguous


.. index:: copyto; tensor

Copy to device
^^^^^^^^^^^^^^

For tensors only, :func:`copyto` will make a copy of the current tensor with new datatype and/or new device and other given charasteristics
(this is somewhat equivalent to `PyTorch's tensor.to method <https://pytorch.org/docs/stable/tensors.html#torch.Tensor.to>`_ with ``copy=True``).

.. function:: copyto(ptr;options) -> new tensor pointer
.. function:: copyto(ptr;options;async) -> new tensor pointer

   :param ptr ptr: a previously allocated :doc:`api-pointer <pointers>` to a tensor.
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <tensor-attributes>`.
   :param bool async: asynchronous flag, default is false. If true, will attempt to perform host to CUDA device transfer without blocking.
   :return: An :doc:`api-pointer <pointers>` to the new tensor.

An alternate form uses an example tensor instead of specified options to define the target device and data type.

.. function:: copyto(ptr;example) -> ptr
.. function:: copyto(ptr;example;async) -> ptr

   :param pointer ptr: a previously allocated :doc:`api-pointer <pointers>` to a tensor.
   :param pointer example: an :doc:`api-pointer <pointers>` to a previously allocated tensor whose device and datatype will be used to create the new copy of the input tensor.
   :param bool async: asynchronous flag, default is false. If true, will attempt to perform host to CUDA device transfer without blocking.
   :return: An :doc:`api-pointer <pointers>` to the new tensor.

::

   q)t:tensor 1 2 3 4#til 24

   q)r:copyto(t; `cuda`float`channel2d`grad)

   q)options each(t;r)
   device dtype layout  gradient pin      memory    
   -------------------------------------------------
   cpu    long  strided nograd   unpinned contiguous
   cuda:0 float strided grad     unpinned channel2d 

Synchronize
^^^^^^^^^^^

PyTorch provides a `synchronize <https://pytorch.org/docs/stable/generated/torch.cuda.synchronize.html>`_ call to wait for all kernels in all streams on a CUDA device to complete. 

The k api function :func:`sync` provides a similar capabilty:

.. function:: sync() -> null
.. function:: sync(device) -> null
.. function:: sync(device index) -> null

   :param symbol device: A valid CUDA device, e.g. ```cuda`` or ```cuda:1``
   :param long index: A valid CUDA device index, e.g. ``1`` for ```cuda:1``
   :return: Waits for all streams to complete on given device/device index. If null or empty argument, uses default CUDA device if available. If a valid but non-CUDA device supplied, no action is taken. Returns null.

In addition to using default device, device name/index, it is also possible to specify a tensor or collection of tensors, a module or a model as an argument to :func:`sync`.  For a collection of tensors, the first CUDA device found for the tensors is used for the synchronization.  For a module or module, the first paramater stored on a CUDA device is used to provide the device for the synchronize step.

.. function:: sync(tensor) -> null
.. function:: sync(vector) -> null
.. function:: sync(dictionary) -> null
.. function:: sync(module) -> null
.. function:: sync(model) -> null

::

   q)sync()
   q)sync`cuda
   q)sync`cuda:1
   q)sync 1

   q)t:tensor(1 2 3e;`cuda:1)
   q)sync t

   q)m:module enlist(`linear;64;10)
   q)to(m;`cuda)
   q)sync m



