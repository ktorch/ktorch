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

Setting properties of a tensor
******************************

PyTorch defines some `properties of a tensor <https://pytorch.org/docs/stable/tensor_attributes.html>`_ as construction axes or attributes.
The main two are :ref:`device <devices>` and :ref:`data type <types>`,
along with layout and whether gradients are recorded for operations on the tensor. The recognized values for these axes are represented as symbols in the k interface:

- **device:** ```cpu`` or ```cuda``, which accepts an optional device index, e.g. ```cuda:0``
- **dtype:** ```bool``, ```byte``, ```char``, ```short``, ```int``, ```long``, ```half``, ```float``, ```double``
- **layout:** ```strided`` or ```sparse``
- **grad:** either ```grad`` or ```nograd``

The ``default`` function will display the defaults usually in effect if no options are given.  Early versions of PyTorch allowed default attributes to be reset, but current versions only allow the default data type to be changed.

.. function:: default[] -> dict

.. function:: default(type) -> null

   | Dictionary of default attributes for tensor creation (empty arg) or reset default data type (sym arg representing data type, null return)

::

   q)default[]
   device  | cpu
   dtype   | float
   layout  | strided
   gradient| nograd

   q)default`double

   q)(1#`dtype)#default[]
   dtype| double


.. index:: tensor; creating from a k value

Creating a tensor from a k value
********************************

The api function ``tensor`` is used to create tensors from k values and retrieve the values back into a k session. The k value can be a scalar, simple list or higher dimension array.  The k value must have the same data type throughout and the same size at each dimension.

.. function:: tensor ptr -> value

   | Return a k value from an :ref:`api-pointer <pointers>` to a previously allocated tensor

.. function:: tensor(value) -> ptr
.. function:: tensor(value;options) -> ptr

   | Create a tensor from k value.

   :param scalar,list,array value: the k value to populate the tensor.  If no options given, the :ref:`data type <types>` for the tensor will be mapped from the data type of the k value.
   :param sym options: one or more symbols for device, data type, layout, gradients, e.g. ```cuda`` or ```cuda:0`````long`````grad``
   :return: An :ref:`api-pointer <pointers>` to the allocated tensor

Examples:
^^^^^^^^^

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

Using an output tensor:
^^^^^^^^^^^^^^^^^^^^^^^

Instead of specifying creation options as the final argument in the ``tensor`` call, a previously allocated tensor can be used.
The tensor's existing attributes will be used but its values will be replaced.

.. function:: tensor(value; out-tensor) -> null

   | Read k value and store in previously created tensor

   :param scalar,list,array value: the k value to populate the tensor.
   :param api-pointer out-tensor: previously allocated tensor which will contain the new values.
   :return: (null)

::

   q)4#info r:tensor()  / initialize empty tensor, retrieve attributes
   device  | cpu
   dtype   | float
   layout  | strided
   gradient| nograd

   q)tensor(1 2 3;r)

   q)tensor r
   1 2 3e

   q)free r  / free tensor r, redefine on gpu as 4-byte int
   q)4#info r:tensor((); `cuda`int)
   device  | cuda:0
   dtype   | int
   layout  | strided
   gradient| nograd

   q)tensor(1 2 3 4;r)

   q)tensor r
   1 2 3 4i
   q)device r
   `cuda:0

.. index:: tensor; conversion errors

Conversion errors:
^^^^^^^^^^^^^^^^^^
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

.. index:: tensor; using creation mode

Tensor creation modes
*********************

In addition to supplying k values to initialise tensors, the following methods create tensors following a particular distribution, sequence, etc. The k interface function accepts arguments somewhat similar to the pytorch function/method.

- `arange <https://pytorch.org/docs/stable/torch.html#torch.arange>`_: returns a tensor with a sequence of integers
- `empty <https://pytorch.org/docs/stable/torch.html#torch.empty>`_: returns a tensor with uninitialized values
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


Tensors are created using the above methods by supplying a mode symbol as the first argument to the same ``tensor`` api function.

::

   q)t:tensor(`zeros; 2 3; `int)
   q)tensor t
   0 0 0
   0 0 0

.. index:: zeros, ones, empty

Creating by size: zeros, ones, empty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Return tensor filled with `zeros <https://pytorch.org/docs/stable/torch.html#torch.zeros>`_,
`ones <https://pytorch.org/docs/stable/torch.html#torch.ones>`_,
and uninitialized (`empty <https://pytorch.org/docs/stable/torch.html#torch.empty>`_).

.. function:: tensor(mode;size) -> ptr

.. function:: tensor(mode;size;options) -> ptr

   | Create a tensor given mode, size  and optional attribute(s).

   :param sym mode: one of ```zeros``, ```ones``, ```empty``
   :param long size: scalar/list specifiying size of array
   :param sym options: one or more symbols for device, data type, layout, gradients, e.g. ```cuda`` or ```cuda:0`` ```long`` ```grad``
   :return: An :ref:`api-pointer <pointers>` to the allocated tensor

Alternate form using an input tensor to supply size, i.e. size will be derived from the input tensor,
similar to PyTorch creation function `torch.ones_like <https://pytorch.org/docs/stable/torch.html#torch.ones_like>`_.

.. function:: tensor(mode;in-tensor) -> ptr

.. function:: tensor(mode;in-tensor;options) -> ptr

   | Create a tensor given mode and input tensor whose size will be used to create new tensor, along with optional tensor attribute(s). 

   :param sym mode: one of ```zeros``, ```ones``, ```empty``
   :param ptr in-tensor: an :ref:`api-pointer <pointers>` to a previously allocated tensor -- its size will determine size of newly created tensor. Device, data type and layout also default to those of the input tensor but can be overwritten by explicit options given in last argument.
   :param sym options: one or more symbols for device, data type, layout, gradients, e.g. ```cuda`` or ```cuda:0`` ```long`` ```grad``.
   :return: An :ref:`api-pointer <pointers>` to the allocated tensor.

Alternate form using an output tensor instead of options that control data type, device, etc.

.. function:: tensor(mode;size;out-tensor) -> null

   :param sym mode: one of ```zeros``, ```ones``, ```empty``.
   :param long size: scalar/list specifiying size of array.
   :param ptr out-tensor: an :ref:`api-pointer <pointers>` to a previously allocated output tensor.
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

Creating tensor with single value: `full <https://pytorch.org/docs/stable/torch.html#torch.full>`_
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. function:: tensor(mode;size;value) -> ptr
.. function:: tensor(mode;size;value;options) -> ptr

   | Create a tensor given mode = ```full``, size, fill value  and optional attribute(s).

   :param sym mode: set to ```full`` 
   :param long size: scalar/list specifiying size of array
   :param scalar value: scalar fill value, real or double k type. Also possible to specify non floating point scalar, but options must include required tensor data type.
   :param sym options: one or more symbols for device, data type, layout, gradients, e.g. ```cuda`` or ```cuda:0`` ```long`` ```grad``
   :return: An :ref:`api-pointer <pointers>` to the allocated tensor

Alternate form using an input tensor for size:

.. function:: tensor(mode;in-tensor;value) -> ptr
.. function:: tensor(mode;in-tensor;value;options) -> ptr

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

Random tensors by size: rand, randn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Return a tensor filled with random numbers from a uniform distribution on ``[0, 1)`` (`rand <https://pytorch.org/docs/stable/torch.html#torch.rand>`_) or unit normal (`randn <https://pytorch.org/docs/stable/torch.html#torch.randn>`_).

Parameters and function calls are as above for mode of ```zeros``, ```ones`` and ```empty``.

.. function:: tensor(mode;size) -> ptr

.. function:: tensor(mode;size;options) -> ptr

   | Create a tensor given mode, size  and optional attribute(s).

   :param sym mode: one of ```rand`` or ```randn``.
   :param long size: scalar/list specifiying size of array.
   :param sym options: one or more symbols for device, data type, layout, gradients, e.g. ```cuda`` or ```cuda:0`` ```long`` ```grad``.
   :return: An :ref:`api-pointer <pointers>` to the allocated tensor.

Alternate form using an input tensor to supply size, i.e. size will be derived from the input tensor,

.. function:: tensor(mode;in-tensor) -> ptr

.. function:: tensor(mode;in-tensor;options) -> ptr

   | Create a tensor given mode and input tensor whose size will be used to create new tensor, along with optional tensor attribute(s). 

   :param sym mode: ```rand`` or ```randn``.
   :param ptr in-tensor: an :ref:`api-pointer <pointers>` to a previously allocated tensor -- its size will determine size of newly created tensor. Device, data type and layout also default to those of the input tensor but can be overwritten by explicit options given in last argument.
   :param sym options: one or more symbols for device, data type, layout, gradients, e.g. ```cuda`` or ```cuda:0`` ```long`` ```grad``.
   :return: An :ref:`api-pointer <pointers>` to the allocated tensor.

Alternate form using an output tensor instead of options that control data type, device, etc.

.. function:: tensor(mode;size;out-tensor) -> null

   :param sym mode: one of ```rand`` or ```randn``.
   :param long size: scalar/list specifiying size of array.
   :param ptr out-tensor: an :ref:`api-pointer <pointers>` to a previously allocated output tensor.
   :return: null return, resets values according to size given and attributes of the output tensor.

::

   q)tensor t:tensor(`rand;10)
   0.05592483 0.7734587 0.1025799 0.6335379 0.3350263 0.5218872 0.8726696 0.9215..

   q)free t
   q)(avg;dev)@\:tensor t:tensor(`randn;10000000;`double)
   -0.0002174295 0.9999617

Random integers by size: randint
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Create a tensor filled with random integers between given range: `randint <https://pytorch.org/docs/stable/torch.html#torch.randint>`_
Called by specifying low, high and size, or high and size (low defaults to zero), as well as other combinations with input and output tensors.

.. function:: tensor(mode;high;size) -> ptr

.. function:: tensor(mode;low;high;size) -> ptr

.. function:: tensor(mode;low;high;size;options) -> ptr

   | Create a tensor given mode, range and size, along with optional tensor attributes.

   :param sym mode: ```randint``.
   :param long low: lowest intger to be drawn from the distribution, set to zero if not given.
   :param long high: one above the highest intger to be drawn from the distribution.
   :param long size: scalar/list specifiying size of array.
   :param sym options: one or more symbols for device, data type, layout, gradients, e.g. ```cuda`` or ```cuda:0`` ```long`` ```grad``.
   :return: An :ref:`api-pointer <pointers>` to the allocated tensor.

An alternate form where an input tensor is supplied to provide the size of the created tensor. Tensor creation options will default to those of the input tensor unless explicitly supplied in the final argument:

.. function:: tensor(mode;in-tensor;high) -> ptr

.. function:: tensor(mode;in-tensor;low;high) -> ptr

.. function:: tensor(mode;in-tensor;low;high;options) -> ptr

   :param ptr in-tensor: an :ref:`api-pointer <pointers>` to a previously allocated tensor -- its size will determine size of newly created tensor. Device, data type and layout also default to those of the input tensor but can be overwritten by explicit options given in last argument.
   :return: An :ref:`api-pointer <pointers>` to the allocated tensor.

Creation can also use a final argument of a previously allocated tensor as an output tensor:

.. function:: tensor(mode;high;size;out-tensor) -> ptr

.. function:: tensor(mode;low;high;size;out-tensor) -> ptr

   :param ptr out-tensor: an :ref:`api-pointer <pointers>` to a previously allocated output tensor.
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

Random permutations: randperm
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Evenly spaced tensors: linspace, logspace
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

