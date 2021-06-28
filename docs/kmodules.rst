.. _kmodules:

Modules for k api
=================

The following modules are not directly available in PyTorch;
these modules were defined for the k api either for convenience when only the functional form is defined in PyTorch,
or to allow modules to be defined with a sequential or sequential-like container which defines the forward calculation implicitly.

Convenience modules
*******************


Pad
^^^
https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html


Squeeze & Unsqueeze
^^^^^^^^^^^^^^^^^^^
https://pytorch.org/docs/stable/generated/torch.squeeze.html
https://pytorch.org/docs/stable/generated/torch.Tensor.squeeze.html
https://pytorch.org/docs/stable/generated/torch.unsqueeze.html
https://pytorch.org/docs/stable/generated/torch.Tensor.unsqueeze.html

Expand & Reshape
^^^^^^^^^^^^^^^^
https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html

https://pytorch.org/docs/stable/generated/torch.reshape.html
https://pytorch.org/docs/stable/generated/torch.Tensor.reshape.html

Cat
^^^
https://pytorch.org/docs/stable/generated/torch.cat.html

Mul
^^^

One hot
^^^^^^^
`one_hot <https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html>`_


Select
^^^^^^
PyTorch defines a `select method <https://pytorch.org/docs/stable/generated/torch.Tensor.select.html>`_ on a tensor to select or "slice" along a given dimension and index.  For a k array, this is similar to ``x i`` or ``x[;;i]``.
This convenience module allows for the select operation to be added to a sequence of modules as part of the forward calculation, e.g. select the final column of an output from the previous model.

::

   q)help`select
   dim| 1
   ind| -1

   q)m:module enlist(`select;1;-1)  // dim 1, final index

   q)show x:3 4#til 12
   0 1 2  3 
   4 5 6  7 
   8 9 10 11

   q)tensor r:forward(m;x)
   3 7 11


Index select
^^^^^^^^^^^^
PyTorch defines an `index_select function <https://pytorch.org/docs/stable/generated/torch.index_select.html>`_ and 
`tensor method <https://pytorch.org/docs/stable/generated/torch.Tensor.index_select.html>`_ which indexes a tensor along a given dimension using supplied ineices, similar to the the above ``select``, but with a list of indices.

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


Fork
^^^^


Recur
^^^^^


Residual
^^^^^^^^


NBeats
^^^^^^


Sequential join
^^^^^^^^^^^^^^^


Sequential nest
^^^^^^^^^^^^^^^


Transformation modules
**********************


Transform container
^^^^^^^^^^^^^^^^^^^

Random cropping
^^^^^^^^^^^^^^^

Random flipping
^^^^^^^^^^^^^^^

Zscore
^^^^^^
