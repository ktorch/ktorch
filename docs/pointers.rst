.. index::  pointer

Pointers
========

The k interface returns a pointer to allocated values (:doc:`tensor<tensors>`, :doc:`module<modules>`, :doc:`optimizer <opt>` or `:docL`model`) that can then be used in subsequent function calls. Pointers are 1-element general lists with a scalar long value to distinguish these values from long scalars and lists created normally in a k session.

::

   q)t:tensor 1 2 3e

   q)type t
   0h

   q)0N!t;
   ,49017184


The api maintains a map of pointers that can be viewed via :func:`obj` and released via :func:`free`.

obj
^^^

.. function:: obj() -> table of allocated objects

   | Return a table of allocated objects with basic information, class, device, bytes allocated, etc.

free
^^^^

.. function:: free() -> null
.. function:: free(ptr) -> null

   | Release allocated object stored in given pointer, or all allocated objects if empty or null argument.


::

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

ref
^^^

.. addref_:

addref
^^^^^^

class
^^^^^

ptr
^^^

size
^^^^

bytes
^^^^^

elements
^^^^^^^^

use
^^^
