.. _overview:

Overview
========

`PyTorch <https://pytorch.org/>`_, a deep learning framework written in python,  also has a c++ library, `libtorch <https://pytorch.org/cppdocs/>`_,
which is delivered as a single zip file containing all the necessary Nvidia libraries and routines to build and train neural networks,
along with basic linear algebra routines.

This interface links k to the c++ routines, but attempts to follow the line of the python interface, which also drives the design of the c++ library.
All the code for the interface is in a single shared library, named ``ktorch.so`` for this documentation.
All the k-api routines are defined in a q namespace via the ``fns`` function:

::

   q)f:key{key[x]set'x}(`ktorch 2:`fns,1)[]
   q)f
   `dv`tree`addref`free`return`use`obj`tensorcount`to`copyto`info`detail`state`s..

Most of the functions accept a single argument which can be expanded with a list to approximate the python syntax.

::

   q)x:tensor 1 2 3.0
   q)y:tensor(1 2 3.0; `cuda`float)


Here is the python sequence to define a tensor ``x`` as 8-byte floating point with gradient calculations:

::

   >>> x=torch.tensor([1,2,3.0], dtype=torch.double, requires_grad=True)
   >>> y=x * x
   >>> z=torch.mean(y)

   >>> z.backward()
   >>> x.grad
   tensor([0.6667, 1.3333, 2.0000], dtype=torch.float64)


Using the k interface:

::

   q)x:tensor(1 2 3.0; `grad)
   q)y:mul(x;x)
   q)z:mean(y)

   q)backward z
   q)grad x
   0.6666667 1.333333 2
   q)free(x;y;z)

Extra steps
^^^^^^^^^^^
Pytorch objects (:doc:`tensors <tensors>`, :doc:`modules <modules>`, etc.) that are created in the k interface must be free'd explicitly,
which is an extra step that is handled automatically via PyTorch's python environment.

And all operations on tensors must be accomplished via functions rather than direct operations.
In the above example, multiplying tensors in python is ``x * x``, but is carried out via ``mul(x;x)`` in k.

Using GPU's
^^^^^^^^^^^
Moving tensors and calculations to GPU :doc:`devices <devices>` is accomplished in about the same way as in the Python interface:

::

   >>> x=torch.tensor([1,2,3.0], device="cuda")
   >>> x
   tensor([1., 2., 3.], device='cuda:0')

   >>> y=torch.tensor([4,5])
   >>> y=y.to('cuda')
   >>> y
   tensor([4, 5], device='cuda:0')

In k:

::

   q)x:tensor(1 2 3e; `cuda)
   q)device x
   `cuda:0

   q)y:tensor 4 5
   q)to(y;`cuda)
   q)device y
   `cuda:0

Defining modules
^^^^^^^^^^^^^^^^

Modules in the k api are mostly structured to resemble PyTorch's `sequential <https://pytorch.org/docs/1.11/generated/torch.nn.Sequential.html?highlight=sequentiall>`_ container.
The forward calculation is defined implicitly via the sequence of contained child modules, chaining the output of one child module to the input of the next child module.

There are addtional :ref:`k-api convenience modules <kmodules>` defined for the interface that expand the domain of sequential-like designs.
There is also a :ref:`callback <module-callback>` module that allows a PyTorch module to call back into a k function that can perform a set of calculations that aren't covered by the predefined set of modules.
