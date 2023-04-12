.. index::  tensor

Tensors
=======

PyTorch describes a tensor as a multi-dimensional matrix containing elements of a single data type.
The simplest way to create a tensor is to use a k value, e.g.

::

   q)t:tensor 0 1 2 3f

   q)tensor t
   0 1 2 3f

.. index::  tensor; attributes
.. _tensor-attributes:

Setting properties
******************

PyTorch defines some `properties of a tensor <https://pytorch.org/docs/stable/tensor_attributes.html>`_ as construction axes or attributes.
The two main attributes are device, e.g. cpu or gpu, and data type.
Other attributes or options determine `layout <https://pytorch.org/docs/stable/tensor_attributes.html?highlight=layout#torch.torch.layout>`_ and whether gradients are recorded for operations on the tensor.
There are additional settings to determine if memory is `pinned <https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers>`_ and `optional memory formats <https://pytorch.org/docs/stable/tensor_attributes.html?highlight=channels_last#torch.torch.memory_format>`_ where channels in 2-dimensional and 3-dimensional images (4-d & 5-d tensors) are stored with channels as the last dimension.

In the k interface, these attributes are represented as symbols:

- **device:** ```cpu`` or ```cuda``, which accepts an optional device index, e.g. ```cuda:0``  (:ref:`see: devices <devices>`)
- **dtype:** one of ```bool`byte`char`short`int`long`half`float`double`cfloat`cdouble`` (:ref:`see: types <types>`)
- **layout:** ```strided`` or ```sparse`` (`see: PyTorch layout <https://pytorch.org/docs/stable/tensor_attributes.html#torch-layout>`_)
- **grad:** either ```grad``, if gradients need to be computed for the tensor, or ```nograd``, the default setting.
- **pin:** either ```pinned`` or ```unpinned`` (`see: page-locked memory <https://pytorch.org/docs/stable/notes/cuda.html#use-pinned-memory-buffers>`_)
- **memory:** either ```preserve``, ```contiguous``, ```channel2d`` or ```channel3d`` (`see: channels last format <https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html>`_)

These symbols can be specified in any order to set the properties of a tensor, e.g. ```int`` or ```cuda`float`grad``.
The :func:`options` function will display the defaults usually in effect if no argument given.  Early versions of PyTorch allowed default attributes to be reset, but current versions only allow the default data type to be changed.

.. function:: options() -> k dictionary
.. function:: options(ptr) -> k dictionary
   :noindex:

   | Dictionary of default attributes for tensor creation (empty arg) or values of the attributes for given tensor

The :func:`dtype` function will get/set the default data type or return the data type of a previously allocated tensor.

.. function:: dtype() -> sym
.. function:: dtype(sym) -> null
   :noindex:
.. function:: dtype(ptr) -> sym
   :noindex:

   | With an empty argument, :func:`dtype` returns the default data type, with a sym data type, it sets the default data type and with a tensor :doc:`pointer <pointers>`, the function returns the tensor's datatype.
   
.. note::
   Sparse tensors, complex tensors, pinned memory and the newer memory formats are less widely used and still a work in progress in PyTorch.
   Most options settings involve data type, device and gradient.

::

   q)options()
   device  | cpu
   dtype   | float
   layout  | strided
   gradient| nograd
   pin     | unpinned
   memory  | contiguous

   q)t:tensor 1 2 3
   q)options t
   device  | cpu
   dtype   | long
   layout  | strided
   gradient| nograd
   pin     | unpinned
   memory  | contiguous

   q)dtype t       / query data type of t
   `long

   q)dtype`double  / reset default data type from float -> double

   q)free t
   q)t:tensor()    / create an empty tensor with default data type
   q)dtype t
   `double

   q)options()
   device  | cpu
   dtype   | double
   ..

   q)free t
   q)options t:tensor(1 2 3;`half`cuda`grad)   / create a tensor on gpu with half-precision
   device  | cuda:0
   dtype   | half
   layout  | strided
   gradient| grad
   pin     | unpinned
   memory  | contiguous

.. index:: tensor; creating from a k value

Creating from a k value
***********************

The api function ``tensor`` is used to create tensors from k values and retrieve the values back into a k session. The k value can be a scalar, simple list or higher dimension array.  The k value must have the same data type throughout and the same size at each dimension.

.. function:: tensor(ptr) -> value

   | Return a k value from an :doc:`api-pointer <pointers>` to a previously allocated tensor

.. function:: tensor(value) -> tensor pointer
   :noindex:
.. function:: tensor(value;options) -> tensor pointer
   :noindex:

   | Create a tensor from k value.

   :param scalar,list,array value: the k value to populate the tensor.  If no options given, the :ref:`data type <types>` for the tensor will be mapped from the data type of the k value.
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <tensor-attributes>`.
   :return: An :doc:`api-pointer <pointers>` to the allocated tensor

Examples
^^^^^^^^

::

   q)t:tensor 2 3 4#til 24

   q)size t
   2 3 4

   q)dtype t
   `long

   q)device t
   `cpu

   q)free t
   q)t:tensor(2 3 4#til 24;`cuda`double)

   q)device t
   `cuda:0

   q)dtype t
   `double

   q)last tensor t
   12 13 14 15
   16 17 18 19
   20 21 22 23

.. index::  tensor; creation using output tensor

Using an output tensor
^^^^^^^^^^^^^^^^^^^^^^

Instead of specifying creation options as the final argument in the ``tensor`` call, a previously allocated tensor can be used.
The tensor's existing attributes will be used but its values will be replaced.

.. function:: tensor(value;out-tensor) -> null

   | Read k value and store in previously created tensor

   :param scalar,list,array value: the k value to populate the tensor.
   :param ptr out-tensor: a previously allocated :doc:`api-pointer <pointers>` to a tensor which will contain the new values.
   :return: null

::

   q)options r:tensor()   / initialize empty tensor, retrieve attributes
   device  | cpu
   dtype   | float
   layout  | strided
   gradient| nograd
   pin     | unpinned
   memory  | contiguous


   q)tensor(1 2 3;r)

   q)tensor r
   1 2 3e

   q)free r                  / free tensor r, redefine on gpu as 4-byte int
   q)r:tensor((); `cuda`int)
   q)options r
   device  | cuda:0
   dtype   | int
   layout  | strided
   gradient| nograd
   pin     | unpinned
   memory  | contiguous

   q)tensor(1 2 3 4;r)

   q)tensor r
   1 2 3 4i

   q)device r
   `cuda:0

.. index:: tensor; conversion errors

Conversion errors
^^^^^^^^^^^^^^^^^
The k value given must be the same data type throughout and have the same size at each depth.
There also needs to be a defined mapping between the k type and the PyTorch type (see :ref:`data types <types>` ).
Some examples where these conditions are not met:

::

   q)t:tensor(1 2;3 4.0)
   'type mismatch at depth 1, long list vs double list
     [0]  t:tensor(1 2;3 4.0)
         ^

   q)t:tensor(1 2;3 4 5)
   'dimension mismatch at depth 1, 2 vs 3
     [0]  t:tensor(1 2;3 4 5)
         ^

   q)t:tensor `a`b`c
   'no torch type found for k: symbol list
     [0]  t:tensor `a`b`c
         ^

   q)t:tensor ([]1 2)
   'no torch type found for k: table
     [0]  t:tensor ([]1 2)
            ^

.. _tensor-undefined:
.. index:: tensor; undefined tensors

Undefined tensors
^^^^^^^^^^^^^^^^^
A tensor pointer can be created which does not point to any underlying memory. It has no device or data type,
but may be used as a placeholder or return value. In the k api, the value used to create an undefined tensor is generic null, ``(::)``.
The information function :func:`defined` will return ``false`` if a given tensor pointer has no data defined for it.

::

   q)t:tensor[]   /create undefined tensor
   q)defined t
   0b
   q)free t       /the tensor must still be free'd

   q)t:tensor(::)
   q)defined t
   0b

   q)(::)~tensor t  /unary null is returned
   1b

   q)options t  / all sym options return null
   device  | 
   dtype   | 
   layout  | 
   gradient| 
   pin     | 
   memory  | 

An undefined tensor is different from an empty tensor, which is considered defined with a device and data type.

::

   q)e:tensor()

   q)defined e
   1b

   q)tensor e
   `real$()     /4-byte float is the default data type if not defined at creation

   q)options e
   device  | cpu
   dtype   | float
   layout  | strided
   gradient| nograd
   pin     | unpinned
   memory  | contiguous


Retrieving tensor values
^^^^^^^^^^^^^^^^^^^^^^^^
The ``tensor`` function can also be used to retrieve values from a previously created tensor into a k array.

.. function:: tensor(ptr) -> value
.. function:: tensor(ptr;ind) -> value
   :noindex:
.. function:: tensor(ptr;dim;ind) -> value
   :noindex:
.. function:: tensor(ptr;flag;dim;ind) -> value
   :noindex:

   | Return a k value from an :doc:`api-pointer <pointers>` to a previously allocated tensor

   :param ptr: a previously allocated :doc:`api-pointer <pointers>` to a tensor.
   :param  bool flag: an optional flag for :ref:`complex tensors <complex>` only, true to return real & imaginary parts along first dimension, false along last dimension.
   :param long dim: an optional dimension for the subsequent index.
   :param long ind: an optional index/indices to retrieve tensor[ind] if no preceding dimension, else tensor[;;ind] if dim=2, etc..

::

   q)t:tensor 2 3 4#til 24

   q)tensor(t;1)
   12 13 14 15
   16 17 18 19
   20 21 22 23

   q)tensor(t;-1;3)   / pytorch uses -1 for last dimension, -2 for second to last, ..
   3  7  11
   15 19 23

.. note::
   The dimension used for retrieving a particular index may be specfied with negative integers, i.e. -1 for final dimension, -2 for next to final dimension.
   A single index may also be specified as a negative number, -1 for last, -2 for second to last.
   However, a list of indices can only use 0 - n-1 where n is the size of the default or specified dimension.

::

   q)t:tensor 3 4#til 12

   q)tensor(t;-1)   /default dimension is zero
   8 9 10 11

   q)tensor(t;1;2 3)
   2  3 
   6  7 
   10 11

   q)tensor(t;-1;2 3)
   2  3 
   6  7 
   10 11
        
   q)tensor(t;-2;1 2)
   4 5 6  7 
   8 9 10 11

   q)tensor(t;-1;0 -1)
   'INDICES element is out of DATA bounds, id=-1 axis_dim=4
     [0]  tensor(t;-1;0 -1)
          ^

.. _tensor-modes:
.. index:: tensor; using creation mode

Tensor creation modes
*********************

In addition to supplying k values to initialise tensors, the following methods create tensors following a particular distribution, sequence, etc. The k interface function accepts arguments somewhat similar to the PyTorch function/methods listed here.

- `arange <https://pytorch.org/docs/stable/torch.html#torch.arange>`_: returns a tensor with a sequence of integers (replaces deprecated function: `range <https://pytorch.org/docs/stable/generated/torch.range.html?highlight=range#torch.range>`_)
- `empty <https://pytorch.org/docs/stable/torch.html#torch.empty>`_: returns a tensor of given size with uninitialized values
- `eye <https://pytorch.org/docs/stable/torch.html#torch.eye>`_: returns an identity matrix
- `full <https://pytorch.org/docs/stable/torch.html#torch.full>`_: returns a tensor filled with a single value
- `linspace <https://pytorch.org/docs/stable/torch.html#torch.linspace>`_: returns a tensor with values linearly spaced in some interval
- `logspace <https://pytorch.org/docs/stable/torch.html#torch.logspace>`_: returns a tensor with values logarithmically spaced in some interval
- `ones <https://pytorch.org/docs/stable/torch.html#torch.ones>`_: returns a tensor filled with ones
- `rand <https://pytorch.org/docs/stable/torch.html#torch.rand>`_: returns a tensor with values drawn from a uniform distribution on ``[0, 1)``
- `randint <https://pytorch.org/docs/stable/torch.html#torch.randint>`_: returns a tensor with integers randomly drawn from an interval
- `randn <https://pytorch.org/docs/stable/torch.html#torch.randn>`_: returns a tensor with values drawn from a unit normal distribution
- `randperm <https://pytorch.org/docs/stable/torch.html#torch.randperm>`_: returns a tensor with a random permutation of integers in some interval
- `zeros <https://pytorch.org/docs/stable/torch.html#torch.zeros>`_: returns a tensor filled with zeros
- `complex <https://pytorch.org/docs/stable/generated/torch.complex.html>`_: creates a complex tensor from real and imaginary values
- `sparse <https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html>`_: creates a sparse tensor from indices and values


Tensors are created in the k interface using the above methods by supplying a mode symbol as the first argument to the same ``tensor`` api function.

::

   q)t:tensor(`zeros; 2 3; `int)
   q)tensor t
   0 0 0
   0 0 0

.. index:: zeros, ones, empty

.. _tensor-by-size:

Creating by size: zeros, ones, empty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Return tensor filled with `zeros <https://pytorch.org/docs/stable/torch.html#torch.zeros>`_,
`ones <https://pytorch.org/docs/stable/torch.html#torch.ones>`_,
and uninitialized (`empty <https://pytorch.org/docs/stable/torch.html#torch.empty>`_).

.. function:: tensor(mode;size) -> tensor pointer
.. function:: tensor(mode;size;options) -> tensor pointer
   :noindex:

   | Create a tensor given mode, size  and optional attribute(s).

   :param sym mode: one of ```zeros``, ```ones``, ```empty``
   :param long size: scalar/list specifiying size of array
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <tensor-attributes>`.
   :return: An :doc:`api-pointer <pointers>` to the allocated tensor

Alternate form using an input tensor to supply size, i.e. size will be derived from the input tensor,
similar to PyTorch creation function `torch.ones_like <https://pytorch.org/docs/stable/torch.html#torch.ones_like>`_.

.. function:: tensor(mode;in-tensor) -> tensor pointer
.. function:: tensor(mode;in-tensor;options) -> tensor pointer
   :noindex:

   | Create a tensor given mode and input tensor whose size will be used to create new tensor, along with optional tensor attribute(s). 

   :param sym mode: one of ```zeros``, ```ones``, ```empty``
   :param ptr in-tensor: an :doc:`api-pointer <pointers>` to a previously allocated tensor -- its size will determine size of newly created tensor. Device, data type and layout also default to those of the input tensor but can be overwritten by explicit options given in last argument.
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <tensor-attributes>`.
   :return: An :doc:`api-pointer <pointers>` to the allocated tensor.

Alternate form using an output tensor instead of options that control data type, device, etc.

.. function:: tensor(mode;size;out-tensor) -> null
   :noindex:

   :param sym mode: one of ```zeros``, ```ones``, ```empty``.
   :param long size: scalar/list specifiying size of array.
   :param ptr out-tensor: an :doc:`api-pointer <pointers>` to a previously allocated output tensor.
   :return: null return, resets values according to size given and attributes of the output tensor.

::

   q)tensor t:tensor(`zeros;3 2)
   0 0
   0 0
   0 0

   q)tensor(`ones;5;t)  / use t as an output tensor
   q)tensor t
   1 1 1 1 1e

   q)tensor(`empty;100;t) / t is filled with unitialized values
   q)tensor t
   1 1 1 1 1 0 4.332332e-37 0 2.791531e+20 1.693048e+22 7.501883e+28 2.733884e+2..

.. index:: full

.. _tensor-full:

Tensor with single value
^^^^^^^^^^^^^^^^^^^^^^^^

Creating tensor with single value: `full <https://pytorch.org/docs/stable/torch.html#torch.full>`_.

.. function:: tensor(mode;size;value) -> tensor pointer
.. function:: tensor(mode;size;value;options) -> tensor pointer
   :noindex:

   | Create a tensor given mode = ```full``, size, fill value  and optional attribute(s).

   :param sym mode: set to ```full`` 
   :param long size: scalar/list specifiying size of array
   :param scalar value: scalar fill value, real or double k type. Also possible to specify non floating point scalar, but options must include required tensor data type.
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <tensor-attributes>`.
   :return: An :doc:`api-pointer <pointers>` to the allocated tensor

Alternate form using an input tensor for size:

.. function:: tensor(mode;in-tensor;value) -> tensor pointer
.. function:: tensor(mode;in-tensor;value;options) -> tensor pointer

   | Create a tensor given mode of ```full`` and input tensor whose size will be used to create new tensor, along with fill value and optional tensor attribute(s). Similar to PyTorch creation function `torch.full_like <https://pytorch.org/docs/stable/torch.html#torch.full_like>`_.

Alternate form using an output tensor instead of options that control data type, device, etc.

.. function:: tensor(mode;size;value;out-tensor) -> null

::

   q)t:tensor(`full; 2 5; 3.0)

   q)tensor t
   3 3 3 3 3
   3 3 3 3 3

   q)b:tensor(`full;t;1b)  / create boolean tensor, use t's size

   q)tensor b
   11111b
   11111b

   q)tensor(`full;7;4.5;b)  / use b's properties, fill with 4.5 -> boolean

   q)tensor b
   1111111b

.. index:: rand, randn

.. _tensor-random:

Random tensors
^^^^^^^^^^^^^^

Return a tensor filled with random numbers from a uniform distribution on ``[0, 1)`` (`rand <https://pytorch.org/docs/stable/torch.html#torch.rand>`_) or unit normal (`randn <https://pytorch.org/docs/stable/torch.html#torch.randn>`_).

Parameters and function calls are as above for mode of ```zeros``, ```ones`` and ```empty``.

.. function:: tensor(mode;size) -> tensor pointer
.. function:: tensor(mode;size;options) -> tensor pointer
   :noindex:

   | Create a tensor given mode, size  and optional attribute(s).

   :param sym mode: one of ```rand`` or ```randn``.
   :param long size: scalar/list specifiying size of array.
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <tensor-attributes>`.
   :return: An :doc:`api-pointer <pointers>` to the allocated tensor.

Alternate form using an input tensor to supply size, i.e. size will be derived from the input tensor,

.. function:: tensor(mode;in-tensor) -> tensor pointer
   :noindex:
.. function:: tensor(mode;in-tensor;options) -> tensor pointer
   :noindex:

   | Create a tensor given mode and input tensor whose size will be used to create new tensor, along with optional tensor attribute(s). 

   :param sym mode: ```rand`` or ```randn``.
   :param ptr in-tensor: an :doc:`api-pointer <pointers>` to a previously allocated tensor -- its size will determine size of newly created tensor. Device, data type and layout also default to those of the input tensor but can be overwritten by explicit options given in last argument.
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <tensor-attributes>`.
   :return: An :doc:`api-pointer <pointers>` to the allocated tensor.

Alternate form using an output tensor instead of options that control data type, device, etc.

.. function:: tensor(mode;size;out-tensor) -> null
   :noindex:

   :param sym mode: one of ```rand`` or ```randn``.
   :param long size: scalar/list specifiying size of array.
   :param ptr out-tensor: an :doc:`api-pointer <pointers>` to a previously allocated output tensor.
   :return: null return, resets values according to size given and attributes of the output tensor.

::

   q)tensor t:tensor(`rand;10)
   0.05592483 0.7734587 0.1025799 0.6335379 0.3350263 0.5218872 0.8726696 0.9215..

   q)free t
   q)(avg;dev)@\:tensor t:tensor(`randn;10000000;`double)
   -0.0002174295 0.9999617

.. index::  randint

.. _tensor-randint:

Random integers
^^^^^^^^^^^^^^^
Create a tensor filled with random integers between given range: `randint <https://pytorch.org/docs/stable/torch.html#torch.randint>`_.
Called by specifying low, high and size, or high and size (low defaults to zero), as well as other combinations with input and output tensors.

.. function:: tensor(mode;high;size) -> tensor pointer
.. function:: tensor(mode;low;high;size) -> tensor pointer
   :noindex:
.. function:: tensor(mode;low;high;size;options) -> tensor pointer
   :noindex:

   | Create a tensor given mode, range and size, along with optional tensor attributes.

   :param sym mode: ```randint``.
   :param long low: lowest intger to be drawn from the distribution, set to zero if not given.
   :param long high: one above the highest intger to be drawn from the distribution.
   :param long size: scalar/list specifiying size of array.
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <tensor-attributes>`.
   :return: An :doc:`api-pointer <pointers>` to the allocated tensor.

An alternate form where an input tensor is supplied to provide the size of the created tensor. Tensor creation options will default to those of the input tensor unless explicitly supplied in the final argument:

.. function:: tensor(mode;in-tensor;high) -> tensor pointer
   :noindex:
.. function:: tensor(mode;in-tensor;low;high) -> tensor pointer
   :noindex:
.. function:: tensor(mode;in-tensor;low;high;options) -> tensor pointer
   :noindex:

   :param ptr in-tensor: an :doc:`api-pointer <pointers>` to a previously allocated tensor -- its size will determine size of newly created tensor. Device, data type and layout also default to those of the input tensor but can be overwritten by explicit options given in last argument.
   :return: An :doc:`api-pointer <pointers>` to the allocated tensor.

The function call can also use a final argument of a previously allocated tensor as an output tensor:

.. function:: tensor(mode;high;size;out-tensor) -> tensor pointer
   :noindex:
.. function:: tensor(mode;low;high;size;out-tensor) -> tensor pointer
   :noindex:

   :param ptr out-tensor: an :doc:`api-pointer <pointers>` to a previously allocated output tensor.
   :return: null return, resets values according to size given and attributes of the output tensor.

::

   q)free t
   q)t:tensor(`randint; -5; 6; 2 5; `float`cuda)
   q)tensor t
   4 0  -2 2 0
   4 -5 2  3 3

   q)tensor(`randint; 100; 3 9; t)
   q)tensor t
   85 55 87 0  1  81 36 97 22
   98 20 66 12 0  95 39 66 12
   21 82 59 39 64 91 54 59 91

   q)dtype t
   `float
   q)device t
   `cuda:0
   q)size t
   3 9

   q)tensor(`randint; 100; 1000000; t)
   q)avg tensor t
   49.48276
   q)size t
   ,1000000

.. index:: randperm

.. _tensor-randperm:

Random permutations
^^^^^^^^^^^^^^^^^^^
Returns `random permutations <https://pytorch.org/docs/stable/generated/torch.randperm.html#torch.randperm>`_ of integers from 0 to n-1 given n.

.. function:: tensor(mode;n) -> tensor pointer
.. function:: tensor(mode;n;options) -> tensor pointer
   :noindex:

   :param sym mode: ```randperm``.
   :param long n: return random permutation of integers from 0-n-1 given n.
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <tensor-attributes>`.
   :return: An :doc:`api-pointer <pointers>` to the allocated tensor.

The function call can also use a final argument of a previously allocated tensor as an output tensor:

.. function:: tensor(mode;n;out-tensor) -> tensor pointer
   :noindex:

   :param sym mode: ```randperm``.
   :param long n: return random permutation of integers from 0-n-1 given n.
   :param ptr out-tensor: an :doc:`api-pointer <pointers>` to a previously allocated output tensor.
   :return: null return, resets values according to size given and attributes of the output tensor.

::

   q)t:tensor(`randperm;10)
   q)tensor t
   1 2 5 8 7 9 4 3 6 0

   q)free t
   q)t:tensor(`randperm;10;`pinned`double)   / pinned memory, double data type
   q)tensor t
   6 0 9 4 1 3 5 2 8 7f

   q)tensor(`randperm;5;t)                   / use t as output tensor
   q)tensor t
   2 3 1 4 0f

.. index:: arange, range

.. _tensor-range:

Evenly spaced tensors
^^^^^^^^^^^^^^^^^^^^^
Creation modes `arange <https://pytorch.org/docs/stable/generated/torch.arange.html>`__
and the deprecated
`range <https://pytorch.org/docs/stable/generated/torch.range.html>`__)
return a 1-dimensional tensor of size (end-start)/step size, with start defaulting to zero and step size to 1.

.. function:: tensor(mode;end) -> tensor pointer
.. function:: tensor(mode;start;end) -> tensor pointer
   :noindex:
.. function:: tensor(mode;start;end;step) -> tensor pointer
   :noindex:
.. function:: tensor(mode;start;end;step;options) -> tensor pointer
   :noindex:

   :param sym mode: ```arange`` or ```range``.
   :param long start: starting value for the set of points, default is 0 for mode of ```arange``, must be given for ```range``.
   :param long end: ending value for the set of points, mode=```arange`` returns points up to but not including ``end``, mode of ```range`` returns points including end.
   :param long step: step size or gap between each pair of adjacent points, default is 1.
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <tensor-attributes>`.
   :return: An :doc:`api-pointer <pointers>` to the allocated tensor.

The function call can also use a final argument of a previously allocated tensor as an output tensor:

.. function:: tensor(mode;start;end;step;out-tensor) -> tensor pointer
   :noindex:

   :param ptr out-tensor: an :doc:`api-pointer <pointers>` to a previously allocated output tensor.
   :return: null return, resets values according to size given and attributes of the output tensor.

::

   q)tensor a:tensor(`arange;5)
   0 1 2 3 4

   q)tensor r:tensor(`range;0;5)
   0 1 2 3 4 5e

   q)t:tensor(`arange;0;10;2)
   q)tensor t
   0 2 4 6 8

   q)free t
   q)tensor t:tensor(`arange;.1;.8;.1)
   0.1 0.2 0.3 0.4 0.5 0.6 0.7e

.. index:: linspace, logspace

.. _tensor-even-spaced:

Creation modes 
`PyTorch function linspace <https://pytorch.org/docs/stable/generated/torch.linspace.html>`_ and
`logspace <https://pytorch.org/docs/stable/generated/torch.logspace.html>`__
create 1-dimensional tensors evenly spaced from ``start`` to ``end``, inclusive with linear step size or log scale of (end - start)/(steps-1).

.. function:: tensor(mode;start;end;steps) -> tensor pointer
.. function:: tensor(mode;start;end;steps;base) -> tensor pointer
   :noindex:
.. function:: tensor(mode;start;end;steps;base;options) -> tensor pointer
   :noindex:

   :param sym mode: ```linspace`` or ```logspace``.
   :param long start: starting value for the set of points.
   :param long end: ending value for the set of points.
   :param long steps: size of the created tensor running from ``start`` to ``end``.
   :param double base: optional base of the log function, default=``10.0``, only for mode=```logspace``
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <tensor-attributes>`.
   :return: An :doc:`api-pointer <pointers>` to the allocated tensor.

The function call can also use a final argument of a previously allocated tensor as an output tensor:

.. function:: tensor(mode;start;end;steps;base;out-tensor) -> tensor pointer
   :noindex:

   :param ptr out-tensor: an :doc:`api-pointer <pointers>` to a previously allocated output tensor.
   :return: null return, resets values according to size given and attributes of the output tensor.

::

   q)t:tensor(`linspace;0;9;10)
   q)tensor t
   0 1 2 3 4 5 6 7 8 9e

   q)free t
   q)t:tensor(`logspace;1;2;10)
   q)tensor t
   10 12.9155 16.68101 21.54435 27.82559 35.93814 46.41589 59.94843 77.42637 100e

   q)tensor(`logspace;1;2;10;2.0;t)
   q)tensor t
   2 2.16012 2.333058 2.519842 2.72158 2.939469 3.174802 3.428976 3.703499 4e
   q)2 xlog tensor t
   1 1.111111 1.222222 1.333333 1.444444 1.555556 1.666667 1.777778 1.888889 2

.. index:: eye

.. _tensor-identity:

Identity matrix
^^^^^^^^^^^^^^^
Function `eye <https://pytorch.org/docs/stable/generated/torch.eye.html?highlight=eye#torch.eye>`__
in PyTorch returns a 2-dimensional tensor with ones on the diagonal and zeros elsewhere.

.. function:: tensor(mode;n) -> tensor pointer
.. function:: tensor(mode;n;m) -> tensor pointer
   :noindex:
.. function:: tensor(mode;n;m;options) -> tensor pointer
   :noindex:

   :param sym mode: ```eye``.
   :param long n: number of rows in the matrix.
   :param long m: optional number of columns in the matrix, default is number of rows equal to columns.
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <tensor-attributes>`.
   :return: An :doc:`api-pointer <pointers>` to the allocated matrix.

The function call can also use a final argument of a previously allocated tensor as an output tensor:

.. function:: tensor(mode;n;out-tensor) -> tensor pointer
   :noindex:
.. function:: tensor(mode;n;m;out-tensor) -> tensor pointer
   :noindex:

   :param ptr out-tensor: an :doc:`api-pointer <pointers>` to a previously allocated output tensor.
   :return: null return, resets values according to rows or rows and columns given and attributes of the output tensor.

::

   q)t:tensor(`eye;3)
   q)tensor t
   1 0 0
   0 1 0
   0 0 1

   q)free t
   q)t:tensor(`eye;3;5;`bool`cuda)
   q)tensor t
   10000b
   01000b
   00100b

.. index:: complex

.. _tensor-complex:

Complex tensor
^^^^^^^^^^^^^^

A tensor of complex numbers can be created by supplying the real and imaginary parts, along with optional attributes. See :ref:`section on complex tensors <complex>` for other methods and details on complex tensors.  The tensor creation method below is meant to match PyTorch's `torch.complex <https://pytorch.org/docs/stable/generated/torch.complex.html>`_ function.

.. function:: tensor(mode;real;imag) -> tensor pointer
.. function:: tensor(mode;real;imag;options) -> tensor pointer
   :noindex:

   :param sym mode: ```complex``.
   :param numeric real: real part of the complex tensor as a k value.
   :param numeric imag: imaginary part of the complex tensor as a k value (same size as the real part).
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <tensor-attributes>`.
   :return: An :doc:`api-pointer <pointers>` to the allocated complex tensor.

::

   q)t:tensor(`complex;1 2 3;-1 2 0;`cdouble`cuda)
   q)options t
   device  | cpu
   dtype   | cdouble
   layout  | strided
   gradient| nograd
   pin     | unpinned
   memory  | contiguous

   q)tensor(t;1b)  / retrieve complex tensor as (real;imag)
   1  2 3
   -1 2 0

   q)tensor[t]~tensor(t;1b) /default behaviour is flag set true
   1b

   q)tensor(t;0b)  / retrieve as real,'imag
   1 -1
   2 2 
   3 0 

An alternate form of the above function call uses a single k value to create the complex tensor, with real and imaginary values across the first or last dimension of the given array.

.. function:: tensor(mode;value) -> tensor pointer
   :noindex:
.. function:: tensor(mode;value;flag) -> tensor pointer
   :noindex:
.. function:: tensor(mode;value;options) -> tensor pointer
   :noindex:
.. function:: tensor(mode;value;flag;options) -> tensor pointer
   :noindex:

   :param sym mode: ```complex``.
   :param numeric value: real & imaginary part of the complex tensor as a k value.
   :param bool flag: a flag set true to indicate real and imaginary values are across the first dimension, else last dimension.  If no flag given, the overall session flag for :ref:`complex first dimension <complex-first>` will be used.
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <tensor-attributes>`.
   :return: An :doc:`api-pointer <pointers>` to the allocated complex tensor.

::

   q)setting`complexfirst
   1b

   q)x:(1 2 3; -1 0 2)

   q)t:tensor(`complex; x)   / default setting: real and imaginary across 1st dim

   q)tensor t                / retrieval also uses default setting
   1  2 3
   -1 0 2

   q)tensor(t;0b)           / retrieve and arrange across last dimension
   1 -1
   2 0 
   3 2 

   q)use[t]tensor(`complex; x; 0b)
   'complex: single input array must have a last dimension of size 2 (real,'imaginary), given size of [2, 3]
     [0]  use[t]tensor(`complex; x; 0b)
                ^

   q)use[t]tensor(`complex;flip x; 0b)


.. index:: sparse

.. _tensor-sparse:

Sparse tensor
^^^^^^^^^^^^^

A sparse tensor can be created by supplying indices and values, along with optional attributes. See :ref:`section on sparse tensors <sparse>` for other methods and details on sparse tensors.  The tensor creation method below is meant to match PyTorch's `torch.sparse_coo_tensor <https://pytorch.org/docs/stable/generated/torch.sparse_coo_tensor.html>`_ function.

.. function:: tensor(mode;ind;val) -> tensor pointer
.. function:: tensor(mode;ind;val;options) -> tensor pointer
   :noindex:
.. function:: tensor(mode;ind;val;size) -> tensor pointer
   :noindex:
.. function:: tensor(mode;ind;val;size;options) -> tensor pointer
   :noindex:

   :param sym mode: ```sparse``.
   :param long ind: 2-d array of indices, each row corresponds to sparse dimension, each column for the non-zero values.
   :param numeric val: scalar of list of k values corresponding to the given indices.
   :param long size: scalar or list indicating the full size of the tensor; will be inferred as minimum size to hold all non-zero indices.
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <tensor-attributes>`.
   :return: An :doc:`api-pointer <pointers>` to the allocated sparse tensor.

::

   q)show 0 -1 1 rotate'(v:1 2 3),\:0 0  / sparse matrix with values along diagonal
   1 0 0
   0 2 0
   0 0 3

   q)show i:flip 3 3 vs/:0 4 8
   0 1 2
   0 1 2

   q)s:tensor(`sparse; i; v; `double)

   q)tensor s
   1 0 0
   0 2 0
   0 0 3

   q)size s
   3 3

   q)indices s
   0 1 2
   0 1 2

   q)values s
   1 2 3f

   q)use[s]tensor(`sparse; i; v; 3 5; `double)  / size beyond given indices
   q)size s
   3 5

   q)tensor s
   1 0 0 0 0
   0 2 0 0 0
   0 0 3 0 0

