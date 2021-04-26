Tensor information
==================

Tensor size
***********

size
^^^^

stride
^^^^^^

dim
^^^

(see also sparsedim and densedim)

itemsize
^^^^^^^^

bytes
^^^^^

(see also objbytes in pointers)


Pointer information
*******************

ptr
^^^

ref
^^^

weakref
^^^^^^^

storage
^^^^^^^


Tensor options
**************

Tensor options are defined by symbols in the k interface.
The functions below take a tensor, vector or dictionary pointer and return symbol(s).
Some of the functions also accept a null or empty arg, and return a system default data type, CUDA device, etc.

options
^^^^^^^

.. function:: options() -> dict
.. function:: options(ptr) -> dict

    | For empty or null arg, returns a dictionary of default attributes for tensor creation. Given an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors, returns a dictionary or list of dictionaries of the attribute values for the tensor(s).

::

   q)options()             /show default options
   device  | cpu
   dtype   | float
   layout  | strided
   gradient| nograd
   pin     | unpinned
   memory  | contiguous

   q)options t:tensor()    /verify empty tensor arg uses defaults
   device  | cpu
   dtype   | float
   layout  | strided
   gradient| nograd
   pin     | unpinned
   memory  | contiguous

   q)options d:dict `a`b`c!("char"; tensor(010b;`cuda`sparse); tensor(2 3 4h;`cpu`pinned))
    | device dtype layout  gradient pin      memory    
   -| -------------------------------------------------
   a| cpu    char  strided nograd   unpinned contiguous
   b| cuda:0 bool  sparse  nograd   unpinned contiguous
   c| cpu    short strided nograd   pinned   contiguous

device
^^^^^^

.. function:: device() -> sym
.. function:: device(ptr) -> sym

   | For a null or empty arg, returns default CUDA device if any GPU's available, else ```cpu``.  Given an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors, returns sym(s) for the devices(s). See also the :ref:`section on devices types <device>` for more on querying for a CUDA device.

::

   q)device() /on machine with CUDA device(s)
   `cuda

   q)device t:tensor()  /use cpu if no device specified
   `cpu

   q)device d:dict`a`b`c!( tensor(1 2 3;`cuda); tensor(4 5;`cuda:1); 6 7 8.0)
   a| cuda:0
   b| cuda:1
   c| cpu


dtype
^^^^^

.. function:: dtype() -> sym
.. function:: dtype(ptr) -> sym

   | For a null or empty arg, returns the default data type, e.g. ```float``. Given an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors, returns sym(s) for the data type(s). See also the :ref:`section on data types <dtype>` for more on setting default data type.

::

   q)dtype[]
   `float

   q)dtype e:tensor()
   `float

   q)dtype f:tensor 1 2 3.0
   `double

   q)dtype v:vector(e;f)
   `float`double


layout
^^^^^^

.. function:: layout(ptr) -> sym

::

   q)layout each (s:sparse t; t:tensor 0 3 0 0 9.0)
   `sparse`strided


gradient
^^^^^^^^

.. function:: gradient(ptr) -> sym

::

   q)gradient each (d:detach s; s:sparse t; t:tensor(0 3 0 0.0;`grad))
   `nograd`grad`grad

   q)gradflag each(d;s;t)
   011b


memory
^^^^^^

.. function:: memory(ptr) -> sym

gradfn
^^^^^^

The :func:`gradfn` is not an option that is set directly, but it is the result of a chain of calculations performed on a set of tensors where any input requires gradients. The result is a symbol of the function used for back propagation, with a version number, the count of any in-place operations.

.. function:: gradfn(ptr) -> sym

::

   q)z:mean y:mul(x;x:tensor(1 2 3.0; `grad))

   q)`x`y`z!gradfn each (x;y;z)
   x| 
   y| MulBackward0
   z| MeanBackward0


Tensor flags
************

contiguous
^^^^^^^^^^

.. function:: contiguous(ptr) -> bool

coalesced
^^^^^^^^^

.. function:: coalesced(ptr) -> bool

gradflag
^^^^^^^^

.. function:: gradflag(ptr) -> bool

leaf
^^^^

.. function:: leaf(ptr) -> bool

pinned
^^^^^^

.. function:: pinned(ptr) -> bool

   | Given an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors, returns boolean(s) set true for tensor(s) with `page-locked memory <https://pytorch.org/docs/stable/notes/cuda.html?highlight=pinned%20memory>`_. Allows for quicker cpu-to-gpu transfers.

::

   q)t:tensor(1 2 3e;`pinned)
   q)pinned t
   1b

   q)to(t;1b;`cuda) /set async flag true when copying to gpu

sparseflag
^^^^^^^^^^

.. function:: sparseflag(ptr) -> bool

   | Given an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors, returns boolean(s) set true for sparse tensor(s). See also the :ref:`section on sparse tensors <sparse>` for more detail.
   

Utilities
*********

info
^^^^


detail
^^^^^^
