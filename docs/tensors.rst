.. _tensors:

Tensors
=======

PyTorch describes a tensor as a multi-dimensional matrix containing elements of a single data type.
The simplest way to create a tensor is to use a k value, e.g.

.. code-block:: k

   q)t:tensor 0 1 2 3f

   q)tensor t
   0 1 2 3f

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

.. function:: dict:default[]
.. function:: default type
   Display existing default attributes for tensor creation (empty arg) or reset default data type (sym arg representing data type)

.. code-block:: k
   :emphasize-lines: 9

   q)default[]
   device  | cpu
   dtype   | float
   layout  | strided
   gradient| nograd

   q)default`double

   q)default[]
   device  | cpu
   dtype   | double
   layout  | strided
   gradient| nograd


Creating a tensor from a k value
********************************

The api function ``tensor`` is used to create tensors from k values and retrieve the values back into a k session. The k value can be a scalar, simple list or higher dimension array.  The k value must have the same data type throughout and the same size at each dimension.

.. function:: value:tensor ptr

   | Return a k value from an :ref:`api-pointer <pointers>` to a previously allocated tensor

.. function:: ptr:tensor value
.. function:: ptr:tensor(value;options)

   | Create a tensor from k value.

   :param scalar,list,array value: the k value to populate the tensor.  If no options given, the d:ref:`data type <types> will be mapped from the data type of the k value.
   :param sym options: one or more symbols for device, data type, layout, gradients, e.g. ```cuda`` or ```cuda:0`` ```long`` ```grad``
   :return: An :ref:`api-pointer <pointers>` to the allocated tensor

Examples:
^^^^^^^^^

.. code-block:: k

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

Using an output tensor:
^^^^^^^^^^^^^^^^^^^^^^^

Instead of specifying creation options as the final argument in the ``tensor`` call, a previously allocated tensor can be used.
The tensor's attributes, data type, device, etc., will be used, but its values will be replaced.

.. code-block:: k

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

Possible conversion errors:
^^^^^^^^^^^^^^^^^^^^^^^^^^^
The k value given must be the same data type throughout and have the same size at each depth.
There also needs to be a defined mapping between the k type and the PyTorch type (see :ref:`data types <types>` ).
Some examples where these conditions are not met:

.. code-block:: k

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
The

.. code-block:: k

   q)t:tensor(`zeros; 2 3; `int)
   q)tensor t
   0 0 0
   0 0 0

Creating tensors by specifying size: zeros, ones, empty
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Return tensor filled with `zeros <https://pytorch.org/docs/stable/torch.html#torch.zeros>`_,
`ones <https://pytorch.org/docs/stable/torch.html#torch.ones>`_,
and unitialized (`ones <https://pytorch.org/docs/stable/torch.html#torch.ones>`_).

.. function:: ptr:tensor(mode;size)
.. function:: ptr:tensor(mode;size;options)

   | Create a tensor given mode, size  and optional parameter attribute(s).

   :param sym mode: one of ```zeros``, ```ones``, ```empty``
   :param longs size: scalar/list specifiying size of array
   :param sym options: one or more symbols for device, data type, layout, gradients, e.g. ```cuda`` or ```cuda:0`` ```long`` ```grad``
   :return: An :ref:`api-pointer <pointers>` to the allocated tensor

.. function:: ptr:tensor(mode;in-tensor)
.. function:: ptr:tensor(mode;in-tensor;options)

   | Create a tensor given mode and input tensor whose size will be used to create new tensor, along with optional tensor attribute(s).

   :param sym mode: one of ```zeros``, ```ones``, ```empty``
   :param longs size: scalar/list specifiying size of array
   :param sym options: one or more symbols for device, data type, layout, gradients, e.g. ```cuda`` or ```cuda:0`` ```long`` ```grad``
   :return: An :ref:`api-pointer <pointers>` to the allocated tensor

.. function:: tensor(mode;size;out-tensor)
   :param sym mode: one of ```zeros``, ```ones``, ```empty``
   :param longs size: scalar/list specifiying size of array
   :param :ref:`api-pointer <pointers>` out-tensor: output tensor
   :return: null return, resets values according to size given and attributes of the output tensor

.. code-block:: k

   q)tensor t:tensor(`zeros;3 2)
   0 0
   0 0
   0 0

   q)tensor(`ones;5;t)
   q)tensor t
   1 1 1 1 1e

   q)tensor(`empty;100;t)
   q)tensor t
   1 1 1 1 1 0 4.332332e-37 0 2.791531e+20 1.693048e+22 7.501883e+28 2.733884e+2..

Creating random tensors by specifying size: rand, randn
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Return a tensor filled with random numbers from a uniform distribution on ``[0, 1)`` (`rand <https://pytorch.org/docs/stable/torch.html#torch.rand>`_) or unit normal (`randn <https://pytorch.org/docs/stable/torch.html#torch.randn>`_).

Parameters and function calls are as above for mode of ```zeros``, ```ones`` and ```empty``.

.. code-block:: k

   q)tensor t:tensor(`rand;10)
   0.05592483 0.7734587 0.1025799 0.6335379 0.3350263 0.5218872 0.8726696 0.9215..

   q)free t
   q)(avg;dev)@\:tensor t:tensor(`randn;10000000;`double)
   -0.0002174295 0.9999617

