.. index::  pointer

Pointers
========

The k interface returns a pointer to allocated values (:doc:`tensor<tensors>`, :doc:`module<modules>`, :doc:`optimizer <opt>` or :doc:`model <model>`) that can then be used in subsequent function calls. Pointers are 1-element general lists with a scalar long value to distinguish these values from long scalars and lists created normally in a k session.

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

The api maintains a map of pointers that can be viewed via :func:`obj` and released via :ref:`free() <free>`.

.. _obj:

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


.. _free:

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

:ref:`free() <free>` also allows a list of pointers or a k dictionary of pointers to be free'd directly, without using an each call:

.. function:: free(list) -> null
.. function:: free(dictionary) -> null

   | Release allocated objects stored in all pointers in the list or dictionary of pointers

::

   q)a:tensor 1 2 3; b:tensor 4 5
   q)free each(a;b)
   ::
   ::

   q)a:tensor 1 2 3; b:tensor 4 5
   q)free(a;b)  / free uses list of pointers directly

   q)m:module`transformer

   q)parmnames each c:children(m;`decoder.layers.0)
   self_attn     | `in_proj_weight`in_proj_bias`out_proj.weight`out_proj.bias
   multihead_attn| `in_proj_weight`in_proj_bias`out_proj.weight`out_proj.bias
   linear1       | `weight`bias
   dropout       | `symbol$()
   linear2       | `weight`bias
   norm1         | `weight`bias
   norm2         | `weight`bias
   norm3         | `weight`bias
   dropout1      | `symbol$()
   dropout2      | `symbol$()
   dropout3      | `symbol$()

   q)free c  / free uses dictionary where all the values are pointers

return
^^^^^^

:func:`return` works similarly to :ref:`free() <free>` except the function returns the k value(s) associated with the allocated object in addition to free'ing the allocated memory. It is implemented only for tensors, vectors of tensors and tensor dictionaries.

.. function:: return(ptr) -> k array(s)

   | First retrieve tensor(s) into k values, then release allocated object stored in given pointer, returning k value(s).


::

   q)l:loss`mse
   q)x:tensor(1.2 1.9 3.1; `grad)
   q)y:tensor 1.0 2.0 3.0
   q)z:loss(l;x;y)
   q)backward z; r:tensor z; free z   // 3 steps: backward prop, retrieve loss, free
   q)r
   0.02

   q)backward z:loss(l;x;y); return z  // return(z) free's and returns without intermediate
   0.02


Pointer information
*******************

.. _mapped:

mapped
^^^^^^

The :func:`mapped` function returns true if given argument is an actively mapped pointer to an allocated PyTorch object.

.. function:: mapped(ptr) -> bool

   | Return ``true`` if argument is a mapped pointer toa an allocated PyTorch object.

The :func:`mapped` can be used to check if a k value is intrepreted as a pointer object and is actively mapped to a previously allocated PyTorch object:

::

   q)t:tensor 1 2 3
   q)mapped t
   1b

   q)free t
   q)mapped t
   0b

   q)mapped "string"
   0b

ptr
^^^

.. function:: ptr(ptr) -> long

   | Return the raw pointer of the underlying PyTorch object, :ref:`more detail for tensors here <tensor-ptr>`. If the k interface has multiple api-pointers refencing the same object, their raw pointers will match.

::

   q)t:tensor 1 2 3
   q)m:module enlist(`linear;128;10)
   q)o:opt(`adam; m)

   q)ptr each(t;m;o)
   60715504 61043456 61053536

class
^^^^^

.. function:: class(ptr) -> symbol

   | Given pointer, returns symbol indicating class of the object, e.g. ```tensor``, ```module``, etc. 

::

   q)t:tensor 1 2 3
   q)m:module`relu
   q)o:opt`sgd
   q)l:loss`mse

   q)class each(t;m;o;l)
   `tensor`module`optimizer`loss


device
^^^^^^

.. function:: device(ptr) -> symbol

   | Return device for the underlying PyTorch object, :ref:`more detail for tensors <tensor-device>` and more on :doc:`devices here<devices>`. Currently, objects like modules and optimizers have multiple parameters and buffers, the device for the first one is returned; no indication is given for cases where parameters or buffers may be stored across multiple GPU's.

::

   q)t:tensor 1 2 3
   q)q:module (`sequential; enlist(`linear;128;10); `relu)
   q)to(q;`cuda)

   q)device each (t;q)
   `cpu`cuda:0


dtype
^^^^^

.. function:: dtype(ptr) -> symbol

   | Return data type for tensors, dictionaries and vectors, :ref:`more detail <tensor-dtype>` and more on :doc:`data types here<types>`. Objects like modules and optimizers have multiple parameters and buffers, with the possibility of different data types for each: currently, :func:`dtype` is not implemented for pointers to these objects.

::

   q)t:tensor 1 2 3.0
   q)dtype t
   `double

   q)q:module (`sequential; enlist(`linear;128;10); `relu)
   q)dtype q
   'dtype: not implemented for modules
     [0]  dtype q
          ^

size
^^^^

.. function:: size(ptr) -> long/long list

   | Returns size, a list with the size at each dimension for :ref:`tensors, dictionaries and vectors <tensor-size>` and a count of parameters for objects like modules, loss functions, optimizers and overall models.

.. note::
   :func:`size` for some objects is different from the overall count of tensors or bytes allocated. The size of an optimizer is given as the number of parameters it is optimizing, but the number of tensor buffers can be larger (e.g. for the ``Adam`` optimizer, there are upt to three tensor buffers for each parameter). Also the bytes allocated changes after the first optimizer step -- some buffers are not initialized until the first step when gradients are applied.

::

   q)t:tensor(`randn; 64 128)
   q)q:module (`sequential; enlist(`linear;128;10); `relu)
   q)l:loss`ce
   q)o:opt(`adam; q)
   q)m:model addref each(q;l;o)

   q)`tensor`module`loss`optimizer`model!size each(t;q;l;o;m)
   tensor   | 64 128
   module   | 2
   loss     | 0
   optimizer| 2
   model    | 2


elements
^^^^^^^^

.. function:: elements(ptr) -> number of elements

   | Returns the number of elements for :ref:`tensors, dictionaries and vectors <tensor-elements>` and a count of parameter and buffer elements for objects like modules, loss functions, optimizers and overall models.

::

   q)t:tensor(`randn; 64 128)
   q)q:module (`sequential; enlist(`linear;128;10); `relu)
   q)l:loss`ce
   q)o:opt(`adamw; q; `amsgrad,1b)

   q)`tensor`module`optimizer!elements each (t;q;o) /no tensors yet for optimizer
   tensor   | 8192
   module   | 1290
   optimizer| 0

   q)backward z:loss(l; x:forward(q;t); y:tensor(`randint;0;10;64))
   q)step o

   q)`tensor`module`optimizer!elements each (t;q;o)
   tensor   | 8192
   module   | 1290
   optimizer| 3872


bytes
^^^^^
.. function:: bytes(ptr) -> number of bytes allocated

   | Returns the number of bytes for :ref:`tensors, dictionaries and vectors <tensor-bytes>` and a count of bytes allocated for parameters and buffers for objects like modules, loss functions, optimizers and overall models.

::

   q)t:tensor(`randn; 64 128)
   q)q:module (`sequential; enlist(`linear;128;10); `relu)
   q)l:loss`ce
   q)o:opt(`adamw; q; `amsgrad,1b)

   q)backward z:loss(l; x:forward(q;t); y:tensor(`randint;0;10;64))
   q)step o

   q)`tensor`module`optimizer!(tensorcount;elements;bytes)@\:/:(t;q;o)
   tensor   | 1 8192 32768
   module   | 2 1290 5160 
   optimizer| 6 3872 15496

tensorcount
^^^^^^^^^^^

.. function:: tensorcount(ptr) -> count of tensors managed by the object.

   | Return the number of tensors in :ref:`vector and dictionaries of tensors <tensor-count>`. Returns the number of parameters and buffers in modules, optimizers and models.

::

   q)t:tensor(`randn; 64 128)
   q)q:module (`sequential; enlist(`linear;128;10); `relu)
   q)l:loss`ce
   q)o:opt(`adamw; q; `amsgrad,1b)

   q)backward z:loss(l; x:forward(q;t); y:tensor(`randint;0;10;64))
   q)step o

   q)`tensor`module`optimizer!(tensorcount;elements;bytes)@\:/:(t;q;o)
   tensor   | 1 8192 32768
   module   | 2 1290 5160 
   optimizer| 6 3872 15496


Pointer utilities
*****************

ref
^^^

.. function:: ref(ptr) -> count of references

   | Given an api-pointer, returns the number of references to the underlying PyTorch object

On the example below, a linear module is created and then an optimizer is initialized to optimize that module's parameters.
The reference count is 2 until the optimizer is free'd, bring the reference count back down to 1.

::

   q)m:module enlist(`linear;512;10)
   q)o:opt(`adam; m)

   q)ref m
   2

   q)free o
   q)ref m
   1


.. _addref:

addref
^^^^^^

.. function:: addref(ptr) -> new ptr pointing to same object

   | Adds a new handle to the k interface pointing to the same PyTorch object (tensor, module, etc.)

Below is an example of adding a reference to tensor ``b`` so that when the vector is created,
a separate pointer to the tensor is maintained.  (tensor ``a`` is no longer valid, the vector is managing the tensor's memory)

::

   q)a:tensor 0101b
   q)b:tensor til 9
   q)v:vector(a; addref b)

   q)ref a
   'stale pointer
     [0]  ref a
          ^
   q)ref b
   2

   q)free v

   q)ref b
   1

q)

use
^^^

.. function:: use[ptr;tensor expression]

   | reuse api-pointer to point to a different underlying PyTorch object without explicitly freeing the original object.

   :param ptr: a pointer to a previously allocated tensor, vector or dictionary of tensors.
   :param tensor-expression: a new expression that returns a new tensor, vector or dictionary of tensors.
   :return: the previous api container frees the first tensor/vector/dictionary pointer and now contains a pointer to the new object. Returns null.

::

   q)free() / free all objects
   q)a:tensor 1 2 3
   q)b:tensor 010101b
   q)use[a]b

   q)tensor a
   010101b

   q)tensor b
   'stale pointer
     [0]  tensor b
          ^
   q)seed 123
   q)use[a]tensor(`rand;2 3)

   q)tensor a
   0.2961119 0.5165623  0.2516707
   0.6885568 0.07397246 0.866522 

   q)obj[]  / verify only one tensor allocated
   ptr      class  device dtype size elements bytes
   ------------------------------------------------
   61701984 tensor cpu    float 2 3  6        24   


str
^^^

.. function:: str(ptr) -> string with embedded newlines

   | Only implemented for tensors and modules, returns the PyTorch C++ string representation of the object allocated.

::

   q)t:tensor(`randn;2 3;`cuda)

   q)-2 str t;
   -0.8294  1.0210  0.4638
   -0.6257 -0.8949 -0.0671
   [ CUDAFloatType{2,3} ]


   q)show q:((0;`sequential); (1; (`linear;64;10)); (1;`relu))
   0 `sequential    
   1 (`linear;64;10)
   1 `relu          

   q)q:module q

   q)-2 str q;
   torch::nn::Sequential(
     (0): torch::nn::Linear(in_features=64, out_features=10, bias=true)
     (1): torch::nn::ReLU()
   )

