.. index::  pointer

Pointers
========

The k interface returns a pointer to allocated values (:doc:`tensor<tensors>`, :doc:`module<modules>`, :doc:`optimizer <opt>` or `:doc``model`) that can then be used in subsequent function calls. Pointers are 1-element general lists with a scalar long value to distinguish these values from long scalars and lists created normally in a k session.

::

   q)t:tensor 1 2 3e

   q)type t
   0h

   q)0N!t;
   ,49017184

.. note::
   Pointers maintain this property -- a scalar embedded in a general list -- if used in a list via ``(;)``, but otherwise lose this distinguishing characteristic  after most applying most operators.

For example:

::

   q)a:tensor 1 2 3
   q)v:vector(4 5;6 7 8)

   q)class each (a;v) / making a list preserves pointers
   `tensor`vector

   q)class each a,v   / joining makes a different type of list
   'class: need allocated torch object, e.g. tensor, module, given long scalar
     [0]  class each a,v
                ^
   q)0N!(a;v);
   (,48128768;,48129232)

   q)0N!a,v;
   (48128768;48129232)

   q)0N!0+a,v;
   48128768 48129232

The api maintains a map of pointers that can be viewed via :func:`obj` and released via :func:`free`.

obj
^^^

.. function:: obj() -> table of allocated objects

   | Return a table of allocated objects with basic information, class, device, bytes allocated, etc.

::

   q)t:tensor 1 2 3.0
   q)v:vector(1 2;3 4 5e)
   q)d:dict `a`b!(0101b; 3 4#til 12)

   q)q:module ((0;`sequential); (1; (`linear;64;10)))
   q)o:opt(`adam; q)

   q)obj[]
   ptr      class     device dtype  size elements bytes
   ----------------------------------------------------
   70619344 optimizer cpu           2    0        0    
   70616336 module    cpu           2    650      2600 
   70620704 dict      cpu           2    16       100  
   70628272 vector    cpu           2    5        28   
   70627552 tensor    cpu    double ,3   3        24   

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

ptr
^^^

class
^^^^^

size
^^^^

bytes
^^^^^

elements
^^^^^^^^

use
^^^
