.. index::  pointer

Pointers
========

The k interface returns a pointer to allocated values (:doc:`tensor<tensors>`, :doc:`module<modules>`, :doc:`optimizer <opt>` or `:doc:`model`) that can then be used in subsequent function calls. Pointers are 1-element general lists with a scalar long value to distinguish these values from long scalars and lists created normally in a k session.

::

   q)t:tensor 1 2 3

   q)type t
   0h

   q)type each t
   ,-7h

   q)0N!t;
   ,68610912

.. note::
   Pointers maintain this property -- a scalar embedded in a general list -- if used in a list via ``(;)``, but otherwise lose this distinguishing characteristic after most applying most operators.

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

Managing pointers
*****************

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
   q)to(q;`cuda)  /move module to default gpu
   q)o:opt(`adam; q)

   q)obj[]
   q)ptr        class     device dtype  size elements bytes
   ------------------------------------------------------
   1796447984 optimizer cuda:0        2    0        0    
   43618592   module    cuda:0        2    650      2600 
   43611488   dict      cpu           2    16       100  
   43346784   tensor    cpu    double ,3   3        24   
   43611232   vector    cpu           2    5        28   


free
^^^^

Pointers to PyTorch objects that are created via the k interface must be explicitly free'd. If a pointer is assigned in a function as a local variable, it must be returned or free'd within the function to avoid memory leaks.

.. function:: free() -> null
.. function:: free(ptr) -> null

   | Release allocated object stored in given pointer, or all allocated objects if empty or null argument.


::

   q)a:tensor 1 2 3
   q)v:vector(4 5;6 7 8)

   q)obj[]
   ptr      class  device dtype size elements bytes
   ------------------------------------------------
   71083104 vector cpu          2    5        40   
   70818656 tensor cpu    long  ,3   3        24   

   q)free v
   q)obj[]
   ptr      class  device dtype size elements bytes
   ------------------------------------------------
   70818656 tensor cpu    long  3    3        24   

   q)class v
   'stale pointer
     [0]  class v
          ^

   q)free[]
   q)obj[]
   ptr class device dtype size elements bytes
   ------------------------------------------

In this example, a 4-byte float with 100,000,000 elements is created repeatedly without freeing memory on the gpu:

::

   q)\ts:100 r:{t:tensor(`randn; x;`cuda); r:mean tensor t; r} 100000000
   'CUDA out of memory. Tried to allocate 382.00 MiB (GPU 0; 10.91 GiB total capacity; 9.70 GiB already allocated; 308.50 MiB free; 9.70 GiB reserved in total by PyTorch)
     [1]  {t:tensor(`randn; x;`cuda); r:mean tensor t; r}
             ^

   / 26 tensors created (9.7g) before the GPU runs out of memory
   q)select n:count i,sum[bytes]%2 xexp 30 from obj[]
   n  bytes   
   -----------
   26 9.685755

   q)free()  / free all PyTorch objects

   / add free[t] from within the function
   q)\ts:100 r:{t:tensor(`randn; x;`cuda); r:mean tensor t; free t; r} 100000000
   32436 536872016

   q)r
   -0.000203889e


Pointer information
*******************

ptr
^^^

class
^^^^^

device
^^^^^^

dtype
^^^^^

size
^^^^

elements
^^^^^^^^

bytes
^^^^^

Pointer utilities
*****************

ref
^^^

.. addref_:

addref
^^^^^^

use
^^^

str
^^^
