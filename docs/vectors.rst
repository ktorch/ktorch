.. index::  vector
.. _vectors:

Vectors
=======

A list of tensors can be created from k values or previously allocated tensors or a combination of both.
Vectors are created, retrieved and modified using the same ``vector()`` interface function.

::

   q)v:vector(1 2;3 4 5.0)

   q)vector v
   1 2
   3 4 5f

   q)free v

Creating a vector
*****************

.. function:: vector(input1;input2;..) -> vector pointer

   | Given k arrays and/or :doc:`tensor pointers <pointers>`, creates and returns a pointer to a newly created vector of tensors.

.. note::

   When tensor pointers are included in the arguments to :func:`vector`, the tensor's memory is managed by the vector and the previous handle to the tensor is no longer valid without a reference increment (see :ref:`addref <addref>`).

::

   q)a:tensor 1 2 3
   q)b:tensor 4 5
   q)v:vector(a; addref b; 6 7 8.0)

   q)vector v
   1 2 3
   4 5
   6 7 8f

   q)tensor a      /new vector manages the tensor memory
   'stale pointer
     [0]  tensor a
          ^
   q)tensor b      /both b & v have access to the same tensor
   4 5

   q)ref b   /the tensor is referenced by b & v
   2

   q)free v

   q)ref b  /reference count decremented
   1

Retrieving vector values
************************

.. function:: vector(ptr) -> val

   :param vector ptr: an :doc:`api-pointer <pointers>` to a previously created vector of tensors.
   :return: arrays for each tensor in the given vector.

.. function:: vector(ptr;ind) -> val

   :param vector ptr: an :doc:`api-pointer <pointers>` to a previously created vector of tensors.
   :param long ind: long index or set of indices.
   :return: array(s) for each index given.

::

   q)v:vector(1 2;3 4 5i;6 7e)

   q)dtype v
   `long`int`float

   q)vector(v;1)
   3 4 5i

   q)vector(v;1 0)
   3 4 5i
   1 2


Setting vector values
*********************

.. function:: vector(ptr;ind;val) -> null

   :param vector ptr: an :doc:`api-pointer <pointers>` to a previously created vector of tensors.
   :param long ind: a long index or set of indices into the vector of tensors.
   :param array val: a corresponding value or set of values/tensors to assign to the vector replacing existing values.
   :param ptr val: an :doc:`api-pointer <pointers>` to previously created tensor(s).
   :return: null

::

   q)v:vector(1 2;3 4 5i;6 7e)

   q)vector v
   1 2
   3 4 5i
   6 7e

   q)vector(v; 1; "new tensor")
   q)vector(v; 2 0; (011b; 1 2 3h))

   q)vector v
   1 2 3h
   "new tensor"
   011b

   q)t:tensor 95 96 97.0
   q)vector(v;1;t) /v[1] replaced with tensor t

   q)vector(v;1)
   95 96 97f

   q)tensor t  /tensor memory managed by vector
   'stale pointer
     [0]  tensor t
          ^

Retrieving tensor pointers
**************************

Use the ``tensor()`` function to extract pointers from a given vector and optional indices.

.. function:: tensor(vec) -> tensors

.. function:: tensor(vec;ind) -> tensors

   :param vector-pointer vec: an :doc:`api-pointer <pointers>` to a previously created vector of tensors.
   :param long ind: an optional long index or list of indices into the vector
   :return: return tensor pointer(s) for each tensor in the vector or corresponding to supplied index or list of indices

::

   q)v:vector(1 2 3.0; 4 5i; 6 7 8e)
   q)t:tensor(v;1)

   q)tensor t
   4 5i

   q)r:tensor(v;1 2)

   q)r       / list of pointers
   56554656
   56554448

   q)tensor each r
   4 5i
   6 7 8e

   q)dtype each r
   `int`float

