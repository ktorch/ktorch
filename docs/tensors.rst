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

PyTorch defines some `properties of a tensor <https://pytorch.org/docs/stable/tensor_attributes.html>`_ as construction axes or attributes. The main two are :ref:`device <devices>` and ::ref`types <data type>`, along with layout and whether gradients are recorded for operations on the tensor. The recognized values for these axes are represented as symbols in the k interface:

- **device:** ```cpu`` or ```cuda``, which accepts an optional device index, e.g. ```cuda:0``
- **dtype:** ```bool``, ```byte``, ```char``, ```short``, ```int``, ```long``, ```half``, ```float``, ```double``
- **layout:** ```strided`` or ```sparse``
- **grad:** either ```grad`` or ```nograd``

Creating a tensor from a k value
********************************

The api function ``tensor`` is used to create tensors from k values and retrieve the values back into a k session. The k value can be a scalar, simple list or higher dimension array.  The k value must have the same data type and the same size at each dimension.

.. function:: value:tensor ptr

   | Return a k value from previously allocated tensor
   :param ::ref`api-pointer <pointers>` ptr: pointer to a previously allocated tensor

.. function:: ptr:tensor value
.. function:: ptr:tensor(value;options)

   | Create a tensor from k value.

   :param scalar,list,array value: the k value to populate the tensor. 
   :param sym options: one or more symbols for device, data type, layout, gradients, e.g. ```cuda`` or ```cuda:0`` ```long`` ```grad``
   :return: pointer to the allocated tensor

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

