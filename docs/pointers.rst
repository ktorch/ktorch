.. index::  pointer

Pointers
========

The k interface returns a pointer to allocated values (tensor, module, optimizer, loss function or model) that can then be used in subsequent function calls. Pointers are 1-element general lists with a scalar long value to distinguish these values from long scalars and lists created normally in a k or q session.

::

   q)t:tensor 1 2 3e

   q)type t
   0h

   q)0N!t;
   ,49017184


The api maintains a map of pointers that can be viewed via ``obj`` and released via ``free``.

.. function:: free []
.. function:: free ptr

   | Release allocated object stored in given pointer, or all allocated objects if empty arg ``[]``.

.. function:: table:obj[]

   | Return a table of allocated objects with brief descriptions.

.. code-block:: k

   q)t:tensor 1 2 3e

   q)obj[]
   ptr      obj    device dtype size elements bytes
   ------------------------------------------------
   49017184 tensor cpu    float 3    3        12   

   q)free t

   q)tensor t
   'stale pointer
   [0]  tensor t
       ^
