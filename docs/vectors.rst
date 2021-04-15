.. _vectors:

Vectors
=======

A list of tensors can be created from k values or previously allocated tensors or a combination of both.
Vectors are created, retrieved and modified using the same :func:`vector` interface function.

::

   q)v:vector(1 2;3 4 5.0)

   q)vector v
   1 2
   3 4 5f

   q)free v

Creating a vector
*****************

.. function:: vector(input1;input2;..) -> ptr

Retrieving vector values
************************

.. function:: vector(ptr) -> val
.. function:: vector(ptr;ind) -> val
.. function:: vector(ptr;ind;val) -> (null)

Retrieving tensor pointer(s)
****************************

.. function:: tensor(vec) -> ptrs
.. function:: tensor(vec;ind) -> ptr(s)
