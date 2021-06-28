.. _kmodules:

Modules for k api
=================

The following modules are not directly available in PyTorch;
these modules were defined for the k api either for convenience when only the functional form is defined in PyTorch,
or to allow modules to be defined with a sequential or sequential-like container which defines the forward calculation implicitly.

Convenience modules
*******************

.. index:: pad

.. _module-pad:


Pad
^^^
The k api adds a module equivalent of the PyTorch `pad <https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html>`_ function.
Given the padding size, along with optional padding mode of  ```constant``, ```reflect``, ```replicate`` or ```circular``,
and padding value (mode=```constant`` only), returns a tensor padded to given size.

::

   q)help`pad
   pad  | 1 2 2 1 1 2
   mode | `constant
   value| 0f

   q)m:module enlist(`pad;1 1 2 2)  /1-col padding on left,right, 2-col top & bottom

   q)tensor r:forward(m; 3 4#1)
   0 0 0 0 0 0
   0 0 0 0 0 0
   0 1 1 1 1 0
   0 1 1 1 1 0
   0 1 1 1 1 0
   0 0 0 0 0 0
   0 0 0 0 0 0

   q)free m
   q)m:module enlist(`pad;1 1 2 2;`replicate)

   q)use[r]forward(m;1 1 3 4#1e+til 12)

   q)tensor[r]. 0 0
   1 1 2  3  4  4 
   1 1 2  3  4  4 
   1 1 2  3  4  4 
   5 5 6  7  8  8 
   9 9 10 11 12 12
   9 9 10 11 12 12
   9 9 10 11 12 12


.. index:: squeeze
.. index:: unsqueeze

.. _module-squeeze:
.. _module-unsqueeze:

Squeeze & Unsqueeze
^^^^^^^^^^^^^^^^^^^
https://pytorch.org/docs/stable/generated/torch.squeeze.html
https://pytorch.org/docs/stable/generated/torch.unsqueeze.html

.. index:: expand

.. _module-expand:

Expand
^^^^^^
https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html

.. index:: reshape

.. _module-reshape:

Reshape
^^^^^^^
This module implements the PyTorch `reshape https://pytorch.org/docs/stable/generated/torch.reshape.html>`_ function, returning a tensor with the given size. One dimension may be given as -1 and will be recalculated to accomdate the tensor's overall number of elements. ``reshape`` attempts to use the same underlying storage as the input tensor, but if the input is not contiguous or has incompatible strides, ``reshape`` may create a copy.

::

   q)help`reshape
   size| -1 1 28 28

   q)m:module enlist(`reshape;-1 3)

   q)tensor r:forward(m; til 6)
   0 1 2
   3 4 5


.. index:: cat

.. _module-cat:

Cat
^^^
This module implements one form of the PyTorch `cat <https://pytorch.org/docs/stable/generated/torch.cat.html>`_ function: the form where 2 tensors are catenated along the given dimension (0 if no dimension given).

::

   q)help`cat
   dim| 0

   q)m:module`cat

   q)exec options from module m
   dim| 0

   q)tensor r:forward(m;1 2 3;4 5 6)
   1 2 3 4 5 6

   q)free m
   q)m:module enlist(`cat;1)

   q)use[r]forward(m;2 3#1 2 3;2 1#4)
   q)tensor r
   1 2 3 4
   1 2 3 4


.. index:: mul

.. _module-mul:

Mul
^^^
This module is similar to one form of the PyTorch function `mul <https://pytorch.org/docs/stable/generated/torch.mul.html>`_, where two tensors are multiplied element-wise. If the shapes of the two inputs are not identical, they must be `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_.

::

   q)m:module`mul

   q)tensor r:forward(m;1 2 3;10)
   10 20 30

   q)use[r]forward(m;4 3#til 12; 1 3#1 10 100)
   q)tensor r
   0 10  200 
   3 40  500 
   6 70  800 
   9 100 1100

   q)use[r]forward(m;4 1#til 4;1 4#til 4)
   q)tensor r
   0 0 0 0
   0 1 2 3
   0 2 4 6
   0 3 6 9


.. index:: onehot

.. _module-onehot:

One hot
^^^^^^^
PyTorch defines a `one_hot <https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html>`_ function to convert indices to 0,1 tensors with 1 at the index location and 0's everywhere.  An optional number of classes is supplied to set the number of 0,1 values per index; if not supplied, the number of classes is inferred from the input. The result of the forward calculation is a tensor with an additional dimension added at the end corresponding to the number of classes.

::

   q)help `onehot
   classes| 10

   q)m:module enlist(`onehot;5)

   q)tensor r:forward(m; 4 1 2 0)
   0 0 0 0 1
   0 1 0 0 0
   0 0 1 0 0
   1 0 0 0 0

   q)free r

   q)first tensor r:forward(m; 2 3#4 1 2 0)
   0 0 0 0 1
   0 1 0 0 0
   0 0 1 0 0

   q)size r
   2 3 5

.. index:: select

.. _module-select:

Select
^^^^^^
PyTorch defines a `select method <https://pytorch.org/docs/stable/generated/torch.Tensor.select.html>`_ on a tensor to select or "slice" along a given dimension and index.  For a k array, this is similar to ``x i`` or ``x[;;i]``.
This convenience module allows for the select operation to be added to a sequence of modules as part of the forward calculation, e.g. select the final column of an output from the previous model.

::

   q)help`select
   dim| 1
   ind| -1

   q)m:module enlist(`select;1;-1)  // dim 1, final column

   q)show x:3 4#til 12
   0 1 2  3 
   4 5 6  7 
   8 9 10 11

   q)tensor r:forward(m;x)
   3 7 11

.. index:: indexselect

.. _module-indexselect:

Index select
^^^^^^^^^^^^
PyTorch defines an `index_select function <https://pytorch.org/docs/stable/generated/torch.index_select.html>`_ and 
`tensor method <https://pytorch.org/docs/stable/generated/torch.Tensor.index_select.html>`_ which indexes a tensor along a given dimension using supplied indices, similar to the the above ``select``, but with a list of indices.

::

   q)help`indexselect
   dim| 1
   ind| 0 1 2

   q)m:module enlist(`indexselect; 0; 0 2)

   q)show x:3 4#til 12
   0 1 2  3 
   4 5 6  7 
   8 9 10 11

   q)tensor f:forward(m;x)
   0 1 2  3 
   8 9 10 11



Container modules
*****************


.. index:: fork

.. _module-fork:

Fork
^^^^


.. index:: recur

.. _module-recur:

Recur
^^^^^


.. index:: resid

.. _module-resid:

Residual
^^^^^^^^


.. index:: nbeats

.. _module-nbeats:

NBeats
^^^^^^


.. index:: seqjoin

.. _module-seqjoin:

Sequential join
^^^^^^^^^^^^^^^


.. index:: seqnest

.. _module-seqnest:

Sequential nest
^^^^^^^^^^^^^^^


Transformation modules
**********************


.. index:: transform

.. _module-transform:

Transform container
^^^^^^^^^^^^^^^^^^^

.. index:: randomcrop

.. _module-randomcrop:

Random cropping
^^^^^^^^^^^^^^^

.. index:: randomflip

.. _module-randomflip:

Random flipping
^^^^^^^^^^^^^^^

.. index:: zscore

.. _module-zscore:

Zscore
^^^^^^
