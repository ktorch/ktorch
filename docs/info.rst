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

options
^^^^^^^

device
^^^^^^

.. function:: device(ptr) -> sym

dtype
^^^^^

.. function:: dtype(ptr) -> sym

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

The :func:`gradfn` is not strictly an option that is set, but it is the result of a chain of calculations performed on a set of tensors where any input requires gradients.

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

coalesced
^^^^^^^^^

gradflag
^^^^^^^^

leaf
^^^^

pinned
^^^^^^

sparseflag
^^^^^^^^^^


Utilities
*********

info
^^^^


detail
^^^^^^
