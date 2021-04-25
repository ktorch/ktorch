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

   | Given k arrays and/or :doc:`tensor pointers <pointers>`, creates and returns a pointer to a newly created vector of tensors.

.. note:

   When tensor pointers are included in the arguments to :func:`vector`, the tensors are freed and the newly created vector manages their memory from that point on.

::

   q)a:tensor 1 2 3
   q)b:tensor 4 5
   q)v:vector(a; addref b; 6 7 8.0)

   q)vector v
   1 2 3
   4 5
   6 7 8f

   q)tensor a      /new vector manages the tensor's memory
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
.. function:: vector(ptr;ind) -> val
.. function:: vector(ptr;ind;val) -> (null)

   :param ptr:
   :param long ind:
   :param ptr/array val:
   :return: (null)

Retrieving tensor pointers
**************************

.. function:: tensor(vec) -> ptrs
.. function:: tensor(vec;ind) -> ptr(s)
