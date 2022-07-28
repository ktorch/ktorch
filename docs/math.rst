Tensor math
===========

Most of the PyTorch `math operations <https://pytorch.org/docs/stable/torch.html#math-operations>`_ are implemented in the k api.

.. toctree::
   :maxdepth: 1

   math-pointwise
   math-reduce
   math-compare
   math-spectral
   math-linalg
   math-random
   math-other

.. _math-output:


Calling k api functions
***********************

Any PyTorch function names which are reserved in k are changed to an upper case first letter; underscores in PyTorch names are removed and most functions with an ``is`` prefix are used without the prefix, e.g. ``isnan`` becomes ``nan`` in k.

The k-api functions are primarily designed to take tensor arguments but should also work with k arrays. All the functions use a single argument,
a list, with some occasional ambiguity interpreting a single k array as a single argument or a multi-arg list. If the default behaviour using k arrays is not useful, it should also be possible to create tensors or other PyTorch objects from the k arrays first before using them as arguments to the library function and converting any PyTorch object returned back to k values after the function call.

Some of the operations support an additional argument of an :ref:`output <math-output>` tensor -- instead of returning a result, the values of the output tensor are overwritten. Some :ref:`in-place <math-inplace>` operations are supported using k unary null as an argument for output tensor(s).

Output tensors
^^^^^^^^^^^^^^

For math operations that support output tensor(s), the output tensor/vector is always the final argument in the list.
The supplied output tensor or vector must have the correct data type and can be empty or of the same size as the output.
Output tensors of the wrong data type will cause an error, whereas the wrong shape or size will generate a warning.

::

   q)x:tensor .2 .1 1.1 3.4 -4.5
   q)y:tensor()

   q)ceil(x;y)
   'Found dtype Float but expected Double
     [0]  ceil(x;y)
          ^

   q)y:tensor 0#0f
   q)dtype y
   `double

   q)ceil(x;y)
   q)tensor y
   1 1 2 4 -4f

   q)ceil(-1.1 2.3;y)
   [W Resize.cpp:24] Warning: An output with one or more elements was resized since it had shape [5], which does not match the required output shape [2].This behavior is deprecated, and in a future PyTorch release outputs will not be resized unless they have zero elements. You can explicitly reuse an out tensor t by resizing it, inplace, to zero elements with t.resize_(0). (function resize_output_check)

   q)tensor y
   -1 3f

An example of an output vector, the minimum and maximum values of given input:

::

   q)v:vector()
   q)aminmax(2 4#til 8;1;v)
   q)vector v
   0 4
   3 7


.. _math-inplace:

In-place operations
^^^^^^^^^^^^^^^^^^^

Some of the operations also support an in-place version: in PyTorch the in-place version is usually denoted with a trailing underscore, e.g. `torch.ceil_ <https://pytorch.org/docs/stable/generated/torch.Tensor.ceil_.html>`_. In the k-api, the inplace version is called using `k unary null <https://code.kx.com/q/ref/identity/>`_ in place of the output tensor:

::

   q)x:tensor -1.2 -.2 1.1 4.5
   q)ceil(x;[])

   q)tensor x
   -1 -0 2 5f


.. note::

   Both output tensors and in-place operations have been largely excluded from newer PyTorch development due to the complications created for automatic differentiation. 

