.. _kmodules:

k api modules
=============

The following modules are not directly available in PyTorch;
these modules were defined for the k api either for convenience when only the functional form is defined in PyTorch,
or to allow modules to be defined with a sequential or sequential-like container which defines the forward calculation implicitly.

The k api modules are defined in `knn.h <https://github.com/ktorch/ktorch/blob/master/knn.h>`_ and other included headers.

- :ref:`callback <module-callback>` - allows for PyTorch c++ modules to call back into q functions for the forward calculation.
- :ref:`cat <module-cat>` - catenate two tensors together along specified dimension.
- :ref:`droppath <module-droppath>` - deactivate some layers during training (see `Stochastic Depth <https://arxiv.org/abs/1603.09382>`_ paper).
- :ref:`embedpos <module-embedpos>` - create learnable embedding of the position of a tensor.
- :ref:`embedseq <module-embedseq>` - combine token and position embedding.
- :ref:`expand <module-expand>` - expand tensor by repeating values across given dimension.
- :ref:`fork <module-fork>` - create two branches to seperately process input tensor.
- :ref:`indexselect <module-select>` - select indices from a tensor at given dimension.
- :ref:`matmul <module-matmul>` - multiply two matrices or sets of matrices..
- :ref:`mul <module-mul>` - multiply two tensors.
- :ref:`nbeats <module-nbeats>` - implementation of NBeats block'
- :ref:`onehot <module-onehot>` - map integer index to a boolean list with 1 at the index position and 0 elsewhere.
- :ref:`pad <module-pad>` - generic padding module.
- :ref:`randomcrop <module-randomcrop>` - random crop of an image given size, padding mode and fill value.
- :ref:`randomflip <module-randomflip>` - irandom horizontal or vertical flip of an image.
- :ref:`recur <module-recur>` - defines recurrent modules accepting & returning vector inputs.
- :ref:`reshape <module-reshape>` - reshape tensor to given size.
- :ref:`residual <module-residual>` - allows input to be added directly to an additional sequence of layers operating on the input.
- :ref:`select <module-select>` - select a single index from a tensor at a given dimension.
- :ref:`selfattention <module-selfattention>` - self-attention, with input, optional mask(s) and initial layer norm.
- :ref:`seqjoin <module-seqjoin>` - allows separate sequences for processing two inputs and joining via given operation.
- :ref:`seqdict <module-seqdict>` - allows a module dictionary of separate sequential blocks, maintaining a single input tensor and auxiliary inputs.
- :ref:`seqlist <module-seqlist>` - allows a module list of separate sequential blocks, maintaining a single input tensor and auxiliary inputs.
- :ref:`seqnest <module-seqnest>` - a derivied class of PyTorch's sequential module which allows nesting.
- :ref:`squeeze <module-squeeze>` -  remove one or more dimensions of tensor with size=1.
- :ref:`transform <module-transform>` -  container module defining a set of transformations to perform on data for both training and evaluation.
- :ref:`transpose <module-transpose>` -  transpose two dimensions of a tensor
- :ref:`unsqueeze <module-unsqueeze>` - create a new dimension with size=1.
- :ref:`zscore <module-zscore>` -  subtract given mean(s) and divide by given standard deviation(s).


Convenience modules
*******************

The following module form of defined PyTorch functions collects the function options and stores them in a module, invoking the function with the options as the module's ``forward`` method.

.. index:: cat
.. _module-cat:

cat
^^^
This module implements one form of the PyTorch `cat <https://pytorch.org/docs/stable/generated/torch.cat.html>`_ function: the form where 2 tensors are catenated along the given dimension (0 if no dimension given).
The c++ module is defined in `knn/fns.h <https://github.com/ktorch/ktorch/blob/master/knn/fns.h>`_, `fns.cpp <https://github.com/ktorch/ktorch/blob/master/knn/fns.cpp>`_.

::

   q)help`cat
   module | `cat
   pytorch| "torch.cat"
   forward| 1b
   result | `tensor
   n      | 2
   args   | `tensor`tensor
   options| (,`dim)!,0

   q)m:module`cat

   q)options m
   dim| 0

   q)evaluate(m; 2 3#til 6; 1 3#6 7 8)
   0 1 2
   3 4 5
   6 7 8

   q)tensor r:forward(m;1 2 3;4 5 6)
   1 2 3 4 5 6

.. index:: expand
.. _module-expand:

expand
^^^^^^
PyTorch method `expand <https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html>`_ is implemented as a module with an option of expanded sizes for each singleton dimension, with ``-1`` used to define no change for the dimension.
The c++ module is defined in `knn/reshape.h <https://github.com/ktorch/ktorch/blob/master/knn/reshape.h>`_, `reshape.cpp <https://github.com/ktorch/ktorch/blob/master/knn/reshape.cpp>`_.

::

   q)help`expand
   size| -1 -1 28 28

   q)m:module enlist(`expand;-1 3)

   q)tensor r:forward(m;4 1#til 4)
   0 0 0
   1 1 1
   2 2 2
   3 3 3

.. index:: matmul
.. _module-matmul:

matmul
^^^^^^
PyTorch function `matmul <https://pytorch.org/docs/stable/generated/torch.matmul.html>`_ is implemented as a k api module ``matmul``.
The c++ module is defined in `knn/fns.h <https://github.com/ktorch/ktorch/blob/master/knn/fns.h>`_, `fns.cpp <https://github.com/ktorch/ktorch/blob/master/knn/fns.cpp>`_.
There are no options to define the module: the forward calculation expects two tensors, typically matrices or sets of matrices, but support for 1-dimensional tensors, along with 4-dimensional tensors is also detailed in the PyTorch function `description <https://pytorch.org/docs/stable/generated/torch.matmul.html>`_.

::

   q)x:(1 2 3.0;4 5 6.0)
   q)y:flip x
   q)x mmu y
   14 32
   32 77

   q)m:module`matmul
   q)evaluate(m;x;y)
   14 32
   32 77

   q)count z:evaluate(m;3#enlist x;y)
   3

   q)z 0
   14 32
   32 77

   q)z 2
   14 32
   32 77

.. index:: mul
.. _module-mul:

mul
^^^
This module is similar to one form of the PyTorch function `mul <https://pytorch.org/docs/stable/generated/torch.mul.html>`_, where two tensors are multiplied element-wise. If the shapes of the two inputs are not identical, they must be `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_.
The c++ module is defined in `knn/fns.h <https://github.com/ktorch/ktorch/blob/master/knn/fns.h>`_, `fns.cpp <https://github.com/ktorch/ktorch/blob/master/knn/fns.cpp>`_.

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

onehot
^^^^^^^
PyTorch defines a `one_hot <https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html>`_ function to convert indices to 0,1 tensors with 1 at the index location and 0's everywhere.  An optional number of classes is supplied to set the number of 0,1 values per index; if not supplied, the number of classes is inferred from the input. The result of the forward calculation is a tensor with an additional dimension added at the end corresponding to the number of classes.
The c++ module is defined in `knn/onehot.h <https://github.com/ktorch/ktorch/blob/master/knn/onehot.h>`_, `onehot.cpp <https://github.com/ktorch/ktorch/blob/master/knn/onehot.cpp>`_.

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


.. index:: pad
.. _module-pad:

pad
^^^
The k api adds a module equivalent of the PyTorch `pad <https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html>`_ function.
Given the padding size, along with optional padding mode of  ```constant``, ```reflect``, ```replicate`` or ```circular``,
and padding value (mode = ```constant`` only), returns a tensor padded to given size.
The c++ module is defined in `knn/pad.h <https://github.com/ktorch/ktorch/blob/master/knn/pad.h>`_, `pad.cpp <https://github.com/ktorch/ktorch/blob/master/knn/pad.cpp>`_.

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


.. index:: reshape
.. _module-reshape:

reshape
^^^^^^^
This module implements the PyTorch `reshape <https://pytorch.org/docs/stable/generated/torch.reshape.html>`_ function, returning a tensor with the given size. One dimension may be given as -1 and will be recalculated to accomdate the tensor's overall number of elements. ``reshape`` attempts to use the same underlying storage as the input tensor, but if the input is not contiguous or has incompatible strides, ``reshape`` may create a copy.
The c++ module is defined in `knn/reshape.h <https://github.com/ktorch/ktorch/blob/master/knn/reshape.h>`_, `reshape.cpp <https://github.com/ktorch/ktorch/blob/master/knn/reshape.cpp>`_.

::

   q)help`reshape
   size| -1 1 28 28

   q)m:module enlist(`reshape;-1 3)

   q)tensor r:forward(m; til 6)
   0 1 2
   3 4 5


.. index:: select
.. _module-select:

select
^^^^^^
PyTorch defines a `select method <https://pytorch.org/docs/stable/generated/torch.Tensor.select.html>`_ on a tensor to select or "slice" along a given dimension and index.  For a k array, this is similar to ``x[i]`` or ``x[;;i]``.
This convenience module allows for the select operation to be added to a sequence of operations as part of the forward calculation, e.g. select the final column of an output from the previous model.
The c++ module is defined in `knn/select.h <https://github.com/ktorch/ktorch/blob/master/knn/select.h>`_, `select.cpp <https://github.com/ktorch/ktorch/blob/master/knn/select.cpp>`_.

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

indexselect
^^^^^^^^^^^^
PyTorch defines a `index_select <https://pytorch.org/docs/stable/generated/torch.index_select.html>`_ function 
which indexes a tensor along a given dimension using supplied indices, similar to the the above ``select``, but with a list of indices rather than a single scalar.
The c++ module is also defined in `knn/select.h <https://github.com/ktorch/ktorch/blob/master/knn/select.h>`_, `select.cpp <https://github.com/ktorch/ktorch/blob/master/knn/select.cpp>`_.

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

.. index:: squeeze
.. _module-squeeze:

squeeze
^^^^^^^
PyTorch defines an `squeeze <https://pytorch.org/docs/stable/generated/torch.squeeze.html>`_ function 
which collapses all size=1 dimensions or only one along a given dimension.
The c++ module is defined in `knn/squeeze.h <https://github.com/ktorch/ktorch/blob/master/knn/squeeze.h>`_, `squeeze.cpp <https://github.com/ktorch/ktorch/blob/master/knn/squeeze.cpp>`_.

::

   q)t:tensor 1 3 1#1 2 3
   q)size t
   1 3 1

   q)m:module`squeeze
   q)size s:forward(m;t)
   ,3
   q)tensor s
   1 2 3

   q)free(m;s)
   q)m:module enlist(`squeeze;0)

   q)size s:forward(m;t)
   3 1
   q)tensor s
   1
   2
   3

.. index:: unsqueeze
.. _module-unsqueeze:

unsqueeze
^^^^^^^^^
PyTorch also defines an an inverse to the :func:`squeeze` function, `unsqueeze <https://pytorch.org/docs/stable/generated/torch.unsqueeze.html>`_,
which adds a size=1 dimension to a given tensor.
The c++ module is also defined in `knn/squeeze.h <https://github.com/ktorch/ktorch/blob/master/knn/squeeze.h>`_, `squeeze.cpp <https://github.com/ktorch/ktorch/blob/master/knn/squeeze.cpp>`_.

::

   q)m:module enlist(`unsqueeze;0)
   q)t:tensor 1 2 3

   q)size u:forward(m;t)
   1 3

   q)free(m;u)
   q)m:module enlist(`unsqueeze;1)
   q)size u:forward(m;t)
   3 1
   q)tensor u
   1
   2
   3

.. index:: transpose
.. _module-transpose:

transpose
^^^^^^^^^
PyTorch defines a `transpose <https://pytorch.org/docs/stable/generated/torch.transpose.html>`_ function which swaps two dimensions.
The k api module, 
defined in `knn/reshape.h <https://github.com/ktorch/ktorch/blob/master/knn/reshape.h>`_, `reshape.cpp <https://github.com/ktorch/ktorch/blob/master/knn/reshape.cpp>`_, is defined with two dimensions which default to -2 and -1 for the 2nd to last and the last dimension.

::

   q)m:module`transpose  /default dimensions
   q)options m
   dim0| -2
   dim1| -1

   q)evaluate(m;(1 2 3;4 5 6))  /flip
   1 4
   2 5
   3 6

   q)t:module enlist(`transpose;0;2) /specify both dimensions
   q)options t
   dim0| 0
   dim1| 2

   q)r:forward(t;1 2 3#0)
   q)size r
   3 2 1

Transformations
***************

.. index:: randomcrop
.. _module-randomcrop:

randomcrop
^^^^^^^^^^

The ``randomcrop`` module is similar to PyTorch implementation
`RandomCrop <https://pytorch.org/vision/stable/transforms.html?highlight=randomcrop#torchvision.transforms.RandomCrop>`_  transform, taking arguments of desired output size, amount of padding (a single number or all 4 numbers for left,right,top,bottom), padding mode and fill value if padding mode = ```constant``.
See :ref:`pad <module-pad>` for more on the padding that is applied prior to the cropping.
The c++ module is defined in `knn/transform.h <https://github.com/ktorch/ktorch/blob/master/knn/transform.h>`_, `transform.cpp <https://github.com/ktorch/ktorch/blob/master/knn/transform.cpp>`_.

::

   q)help`randomcrop
   size   | 32
   pad    | 4
   padmode| `reflect
   value  | 0f

   q)m:module enlist(`randomcrop;5;1)

   q)tensor r:forward(m;5 5#1)
   0 0 0 0 0
   0 1 1 1 1
   0 1 1 1 1
   0 1 1 1 1
   0 1 1 1 1

   q)use[r]forward(m;5 5#1)
   q)tensor r
   1 1 1 1 0
   1 1 1 1 0
   1 1 1 1 0
   1 1 1 1 0
   0 0 0 0 0

.. index:: randomflip
.. _module-randomflip:

randomflip
^^^^^^^^^^

PyTorch has transforms for
`horizontal <https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomHorizontalFlip>`_ and
`vertical <https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomVerticalFlip>`_ flips where the image is randomly flipped with the given probability. The ``randomflip`` module in the k api takes a second argument of dimension to flip, e.g. -1 for last or horizontal flip, -2 for 2nd to last or vertical flip. The input image can have several leading dimensions, e.g. n x c x h x w where n is number of images, c is number of channels, h & w are height and width.
The c++ module is defined in `knn/transform.h <https://github.com/ktorch/ktorch/blob/master/knn/transform.h>`_, `transform.cpp <https://github.com/ktorch/ktorch/blob/master/knn/transform.cpp>`_.

::

   q)help `randomflip
   p  | 0.5
   dim| -1

   q)m:module enlist(`randomflip; .8; -1)  / high prob of flip along last dim

   q)tensor r:forward(m; 4 4#til 16)
   3  2  1  0 
   7  6  5  4 
   11 10 9  8 
   15 14 13 12

   q)v:module enlist(`randomflip; .8; -2)  / vertical flip

   q)use[r]forward(v; 64 3 4 4#til 3072)  / 64 3-channel images, 4x4
   q)tensor[(r;0)][0]
   12 13 14 15
   8  9  10 11
   4  5  6  7 
   0  1  2  3 

.. index:: zscore
.. _module-zscore:

zscore
^^^^^^

The ``zscore`` module is similar to PyTorch's `normalize <https://pytorch.org/vision/stable/transforms.html?highlight=normalize#torchvision.transforms.Normalize>`_ transform.  The argument are the mean(s) and standard deviation(s), along with a flag for transforming the tensor in place. Means and standard deviations can be supplied for each RGB channel or in some other form where the sample statistics can be broadcast across the input tensor.
The c++ module is defined in `knn/transform.h <https://github.com/ktorch/ktorch/blob/master/knn/transform.h>`_, `transform.cpp <https://github.com/ktorch/ktorch/blob/master/knn/transform.cpp>`_.

::

   q)help `zscore
   mean   | 0.51 0.49 0.47
   std    | 0.25 0.25 0.21
   inplace| 0b

   q)x:1000?100.0
   q)m:module enlist(`zscore; avg x; sdev x)

   q)tensor r:forward(m;x)
   1.559878 -1.312209 1.343763 0.9798773 -0.8676595 0.5637196 -1.044945 -0.63768..

   q)(avg;sdev)@\:tensor r
   3.730349e-17 1

.. index:: transform
.. _module-transform:

transform container
^^^^^^^^^^^^^^^^^^^

The ``transform`` module is somewhat similar to PyTorch's `vision transformer <https://pytorch.org/vision/stable/transforms.html>`_ in that it defines a set of transformations to perform on image data. The k api module can contain up to two sequential modules, the first for defining transforms on inputs when the module is in training mode, the second sequential contains transforms to perform on inputs when the module is in evaluation mode. Either sequential child module may be empty, the evaluation module may be omitted.
The c++ container module is defined with the other transforms in `knn/transform.h <https://github.com/ktorch/ktorch/blob/master/knn/transform.h>`_, `transform.cpp <https://github.com/ktorch/ktorch/blob/master/knn/transform.cpp>`_.

::

   q)m:module`transform

   q)module(m; 1; `sequential)           // define random flip & crop, zscore for training inputs
   q)module(m; 2; (`randomflip;.5;-1))
   q)module(m; 2; (`randomcrop; 32; 4))
   q)module(m; 2; (`zscore; .5; .25))

   q)module(m; 1; `sequential)           // define zscore only for evaluation mode
   q)module(m; 2; (`zscore; .5; .25))

   q)-2 str m;  // PyTorch string representation of the transform
   Transform((
     (train): torch::nn::Sequential(
       (0): RandomFlip(p=0.5, dim=-1)
       (1): RandomCrop(size=[32, 32], pad=[4, 4, 4, 4])
       (2): Zscore(mean=0.5, stddev=0.25, inplace=false)
     )
     (eval): torch::nn::Sequential(
       (0): Zscore(mean=0.5, stddev=0.25, inplace=false)
     )
   )

Embeddings
**********

The k api has two modules related to learned positional embeddings:

.. index:: embedpos
.. _module-embedpos:

embedpos
^^^^^^^^
The learnable position weights reqire two dimensions, the rows for each index in the sequence, and the cols or hidden dimension of the embedding.
The c++ module is defined in `knn/transform.h <https://github.com/ktorch/ktorch/blob/master/knn/transform.h>`_, `transform.cpp <https://github.com/ktorch/ktorch/blob/master/knn/transform.cpp>`_.

Create position embedding for a sequence of maximum length 8, with 12 columns for the embedding dimension.

::

   q)m:module enlist(`embedpos;8;12)
   q)parmnames m
   ,`pos

   q)p:parm(m;`pos)
   q)size p
   1 8 12

   q)distinct raze over t:tensor p
   ,0e

Reinitialize the position embedding for this example:

::

   q)parm(m; `pos; "e"$1 8 12#100 sv'til[8]cross til 12)
   q)first tensor p
   0   1   2   3   4   5   6   7   8   9   10  11 
   100 101 102 103 104 105 106 107 108 109 110 111
   200 201 202 203 204 205 206 207 208 209 210 211
   300 301 302 303 304 305 306 307 308 309 310 311
   400 401 402 403 404 405 406 407 408 409 410 411
   500 501 502 503 504 505 506 507 508 509 510 511
   600 601 602 603 604 605 606 607 608 609 610 611
   700 701 702 703 704 705 706 707 708 709 710 711

The learnable embeddings are 1 x rows x cols with the extra leading dimension creating a broadcastable tensor to add to a token embedding of batchsize x rows x cols.  

::

   q)show x:3 7#21?8  / batch size 3, sequence length 7
   4 0 2 1 2 1 2
   3 2 5 4 7 5 6
   6 1 0 5 2 4 5

   q)r:forward(m;x)
   q)size r
   1 7 12

   q)squeeze tensor r
   0   1   2   3   4   5   6   7   8   9   10  11 
   100 101 102 103 104 105 106 107 108 109 110 111
   200 201 202 203 204 205 206 207 208 209 210 211
   300 301 302 303 304 305 306 307 308 309 310 311
   400 401 402 403 404 405 406 407 408 409 410 411
   500 501 502 503 504 505 506 507 508 509 510 511
   600 601 602 603 604 605 606 607 608 609 610 611

   q)tensor[r]~evaluate(m; 100 7#0)
   1b

.. note:

   The batch dimension and the input values do not change the result values or size; the only relevant input is the sequence length, i.e. the number of columns of the input.

.. index:: embedseq
.. _module-embedseq:

embedseq
^^^^^^^^

This module adds the result of a token embedding with the position embedding.  There are 3 relevant dimensions, the number of possible tokens, the embedding dimension and the maximum sequence length.  The token embedding has one row per token and columns for the embedding dimension.
The learned position embedding has one row per sequence length and columns for the embedding dimension.
The c++ module is defined in `knn/transform.h <https://github.com/ktorch/ktorch/blob/master/knn/transform.h>`_, `transform.cpp <https://github.com/ktorch/ktorch/blob/master/knn/transform.cpp>`_.

::

   q)m:module enlist(`embedseq;10;12;8)
   q)options m
   rows  | 10
   cols  | 12
   length| 8

   q)show x:3 8#24?10
   9 2 7 0 1 9 2 1
   8 8 1 7 2 4 5 4
   2 7 8 5 6 4 1 3

   q)size r:forward(m;x)
   3 8 12

In the above example, batch size is 3, i.e. 3 different sequences of length 8 are given as inputs; output has 1 plane per batch, 1 row per sequence and 1 column per embedding dimension.

We can verify that the sequence embedding is the result of adding token embedding and position embedding:

::

   q)mpos:module enlist(`embedpos;8;12)
   q)mtok:module enlist(`embed;10;12)

   q)parmnames m
   `tok.weight`pos.pos

   q)e:parm(m;`tok.weight)  / get token embedding
   q)parm(mtok;`weight;e)   / set separate embedding module to same weights

   q)r1:evaluate(mpos;x)    / separate position embedding
   q)r2:evaluate(mtok;x)    / token embedding

   q)tensor[r]~add(r1;r2)   / forward result of separate embeddings added (w'broadcasting)
   1b

Container modules
*****************


.. index:: fork
.. _module-fork:

fork
^^^^
The ``fork`` container module splits the input tensor into two parts and returns the combined result, i.e. ``(f(x); g(x))``
The module is defined to contain sequential modules for each part to allow for a set of module operations or a single module or a mix of both.
The c++ module is defined in `knn/fork.h <https://github.com/ktorch/ktorch/blob/master/knn/fork.h>`_, `fork.cpp <https://github.com/ktorch/ktorch/blob/master/knn/fork.cpp>`_.

::

   q)m:module (`fork; seq(`sequential;`sigmoid); seq(`sequential;`relu))

   q)-2 str m;
   knn::Fork(
     (qa): torch::nn::Sequential(
       (0): torch::nn::Sigmoid()
     )
     (qb): torch::nn::Sequential(
       (0): torch::nn::ReLU()
     )
   )

   q)x: -3 -2 -1 0 1 2 3e

   q)evaluate(m;x)
   0.04743 0.1192 0.2689 0.5 0.7311 0.8808 0.9526
   0       0      0      0   1      2      3     

   q)(sigmoid x; relu x)
   0.04743 0.1192 0.2689 0.5 0.7311 0.8808 0.9526
   0       0      0      0   1      2      3     

In this example the ``fork`` includes a single ``sigmoid`` module and a sequence:

::

   q)m:module (`fork; `sigmoid; seq(`sequential;`relu;(`drop;.5)))
   q)-2 str m;
   knn::Fork(
     (a): torch::nn::Sigmoid()
     (qb): torch::nn::Sequential(
       (0): torch::nn::ReLU()
       (1): torch::nn::Dropout(p=0.5, inplace=false)
     )
   )

   q)x: -3 -2 -1 0 1 2 3e
   q)vector v:forward(m;x)
   0.04743 0.1192 0.2689 0.5 0.7311 0.8808 0.9526
   0       0      0      0   2      4      0     


.. index:: recur
.. _module-recur:

recur
^^^^^
The recurrent modules in PyTorch,
`RNN <https://pytorch.org/docs/stable/generated/torch.nn.RNN.html>`_,
`GRU <https://pytorch.org/docs/stable/generated/torch.nn.GRU.html>`_ and
`LSTM <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html>`_,
have different signatures for their forward calculation with ``RNN`` and ``GRU`` having 2 tensor args and a tuple of tensors returned,
whereas the ``LSTM`` module uses a tensor and optional tuple argument and returns a nested tuple of output and hidden state.

::

   q)select module,result,args from help`module where module in `rnn`gru`lstm
   module result args         
   ---------------------------
   gru    tuple  tensor tensor
   lstm   nested tensor tuple 
   rnn    tuple  tensor tensor

The ``recur`` module handles the 3 recurrent layers with the same input/output: a vector of tensors is given and a vector of tensors is returned.
In the case of an ``LSTM`` module, up to 3 tensors may be given: one for input and two others to describe the hidden state, whereas for ``RNN`` and
``GRU`` layers one input tensor and one hidden state tensor is expected. Output is a vector of tensors, 2 in the case of ``RNN`` and ``GRU`` and 3 tensors for ``LSTM``. The ``recur`` module is designed to allow a simpler, more uniform interface for all three recurrent models.
The c++ module is defined in `knn/recur.h <https://github.com/ktorch/ktorch/blob/master/knn/recur.h>`_, `recur.cpp <https://github.com/ktorch/ktorch/blob/master/knn/recur.cpp>`_.

::

   q)q:module(`recur; enlist(`lstm;100;512))

   q)-2 str q;
   knn::Recur(
     (lstm): torch::nn::LSTM(input_size=100, hidden_size=512, num_layers=1, bias=true, batch_first=false, dropout=0, bidirectional=false)
   )

   q)x:tensor(`randn;3 5 100)

   q)size v:forward(q;x)  / returns one output tensor and two describing hidden state
   3 5 512
   1 5 512
   1 5 512

   q)vector(v; 0; tensor(`randn; 3 5 100))  /add new input to current hidden state
   q)use[v]forward(q;v)

.. index:: residual
.. _module-residual:

residual
^^^^^^^^
The ``residual`` module implements one of the following forms of adding the input to the result of a sequence of operations on that input:

- f(x) + x
- a(f(x) + x)
- f(x) + g(x)
- a(f(x) + g(x))

where ``f`` and ``g`` represent sequences of operations and ``a`` is a function applied to the sum of operations.
Adding the original input helps training by keeping gradients from exploding/vanishing during training of deep networks.
The c++ module is defined in `knn/residual.h <https://github.com/ktorch/ktorch/blob/master/knn/residual.h>`_, `residual.cpp <https://github.com/ktorch/ktorch/blob/master/knn/residual.cpp>`_.

The most common residual layer is addition of input after running the input through a sequential module,
``f(x) + x``:

::

   q)m:module(`residual; seq(`sequential; `mul; `relu))

   q)-2 str m;
   Residual(
     (q1): torch::nn::Sequential(
       (0): knn::Mul()
       (1): torch::nn::ReLU()
     )
   )

   q)x:-3 -2 -1 0 1 2 3e; y:10000e

   q)evaluate(m;x;y)
   -3 -2 -1 0 10001 20002 30003e

   q)mul(x;y)
   -30000 -20000 -10000 0 10000 20000 30000e

   q)relu(mul(x;y))
   0 0 0 0 10000 20000 30000e

   q)x+relu(mul(x;y))
   -3 -2 -1 0 10001 20002 30003e

   q)f:{relu mul(x;y)}
   q)x + f[x;y]
   -3 -2 -1 0 10001 20002 30003e

   q)(x+f[x]y) ~ evaluate(m;x;y)
   1b

In this example, the residual module applies two different embeddings to the input before adding the results together:

::

   q)q:module(`residual; seq(`sequential; (`embed; 8; 12)); seq(`sequential; (`embedpos;5;12)))

   q)-2 str q;
   Residual(
     (q1): torch::nn::Sequential(
       (0): torch::nn::Embedding(num_embeddings=8, embedding_dim=12)
     )
     (q2): torch::nn::Sequential(
       (0): knn::EmbedPosition(rows=5, cols=12)
     )
   )

   q)x:3 5#15?8  /3 batches of 5-integer sequences

   q)y:evaluate(q;x)

   q)y1:evaluate(q;`q1;x)
   q)y2:evaluate(q;`q2;x)

   q)y~add(y1;y2)
   1b

.. index:: seqjoin
.. _module-seqjoin:

seqjoin
^^^^^^^
This module is made up of 2 separate sequential modules and a join function: ``join(f(x), g(y))``
Both sequential layers are defined first, followed by a single join function which expects two inputs.

It is also possible to define only one sequential layer, in which case the join becomes: ``join(f(x),y)``
If the sequential child modules are not named, their names default to ``qx`` and ``qy``.

The forward calculation expects two inputs, ``x`` and ``y``; if the 2nd input is not supplied, ``y`` is set to ``x``.

The c++ module is defined in `knn/seq.h <https://github.com/ktorch/ktorch/blob/master/knn/seq.h>`_, `seq.cpp <https://github.com/ktorch/ktorch/blob/master/knn/seq.cpp>`_.

In the example below ``f(x) is relu(x)`` and ``g(y) is sigmoid(x)`` with ``cat`` as the join function.

::

   q)q:module(`seqjoin; seq(`sequential`f; `relu); seq(`sequential`g; `sigmoid); `cat)

   q)-2 str q;
   SeqJoin(
     (f): torch::nn::Sequential(
       (0): torch::nn::ReLU()
     )
     (g): torch::nn::Sequential(
       (0): torch::nn::Sigmoid()
     )
     (join): knn::Cat(dim=0)

   q)x:-1 2 1e
   q)y:1 2 3e

   q)evaluate(q;x;y)
   0 2 1 0.7311 0.8808 0.9526e

   q)relu[x],sigmoid y
   0 2 1 0.7311 0.8808 0.9526e

In the next example, only one sequential is defined and only one input is used.

::

   q)q:module`seqjoin
   q)module(q;1;`sequential)
   q)module(q;2;`relu)
   q)module(q;1;`matmul)

   q)-2 str q;
   SeqJoin(
     (qx): torch::nn::Sequential(
       (0): torch::nn::ReLU()
     )
     (join): knn::Matmul()
   )

   q)x:-3 -2 -1 0 1 2 3e
   q)evaluate(q;x)
   14e

   q)matmul(relu x;x)
   14e

.. index:: seqdict
.. _module-seqdict:

seqdict
^^^^^^^
The ``seqdict`` module is derived from PyTorch's `ModuleDict <https://pytorch.org/docs/stable/generated/torch.nn.ModuleDict.html>`_, but is designed to be a dictionary of named sequential modules which accept up to 8 tensor inputs and return a single tensor output.
The tensor output from the first sequential in the module dictionary is used as the first input into the next sequential and so on, until the end of the dictionary.

If the first sequential module in the list does not require all the input arguments, they will be used for any subsequent modules that require multiple inputs. 

For example, if the ``seqlist`` module has 2 sequences, :math:`F` and :math:`G`, where :math:`F` requires a single input and :math:`G` requires three inputs:

.. math::

      \begin{align*}
      inputs & = a,b,c \\
      x & = F(a) \\
      x & = G(x,b,c) \\
      & ..
      \end{align*}

The c++ module is defined in `knn/seqdict.h <https://github.com/ktorch/ktorch/blob/master/knn/seqdict.h>`_.

::

   q)q:module`seqdict
   q)module(q; 1; `sequential`a)
   q)module(q; 2; `sigmoid`f)
   q)module(q; 1; `sequential`b)
   q)module(q; 2; `cat`c)

   q)moduletypes q
      | seqdict
   a  | sequential
   a.f| sigmoid
   b  | sequential
   b.c| cat

   q)-2 str q;
   knn::SeqDict(
     (a): torch::nn::Sequential(
       (f): torch::nn::Sigmoid()
     )
     (b): torch::nn::Sequential(
       (c): knn::Cat(dim=0)
     )
   )

   q)x:-2 -1 0 1 2 3.0
   q)y:99 100.0

   q)evaluate(q;x;y)
   0.1192 0.2689 0.5 0.7311 0.8808 0.9526 99 100

   q)cat(sigmoid x;y)
   0.1192 0.2689 0.5 0.7311 0.8808 0.9526 99 100

A similar example using a nested tree definition of the ``seqdict`` module:

::

   q)q:module(`seqdict; (`sequential`a;`sigmoid`f); (`sequential`b;`cat`c;enlist(`reshape;`mat;2 4)))

   q)moduletypes q
        | seqdict
   a    | sequential
   a.f  | sigmoid
   b    | sequential
   b.c  | cat
   b.mat| reshape

   q)x:-2 -1 0 1 2 3.0
   q)y:99 100.0

   q)evaluate(q;x;y)
   0.1192 0.2689 0.5 0.7311
   0.8808 0.9526 99  100   


.. index:: seqlist
.. _module-seqlist:

seqlist
^^^^^^^
The ``seqlist`` module is similar to the :ref:`seqdict <module-seqdict>` module, but stores a list of sequential modules rather than a set of names mapped to modules.
It is derived from PyTorch's `ModuleList <https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html>`_, but is intended to be a list of sequential modules which accept up to 8 tensor inputs and return a single tensor output.
The tensor output from the first sequential in the list is used as the first input into the next sequential and so on, until the end of the list.

If the first sequential module in the list does not require all the input arguments, they will be used for any subsequent modules that require multiple inputs. 

For example, if the ``seqlist`` module has 2 sequences, :math:`F` and :math:`G`, where :math:`F` requires a single input and :math:`G` requires three inputs:

.. math::

      \begin{align*}
      inputs & = a,b,c \\
      x & = F(a) \\
      x & = G(x,b,c) \\
      & ..
      \end{align*}

The c++ module is defined in `knn/seqlist.h <https://github.com/ktorch/ktorch/blob/master/knn/seqlist.h>`_.

::

   q)m:module(`seqlist; seq(`sequential; `sigmoid); seq(`sequential;`relu); seq(`sequential;`mul))

   q)-2 str m;
   knn::SeqList(
     (0): torch::nn::Sequential(
       (0): torch::nn::Sigmoid()
     )
     (1): torch::nn::Sequential(
       (0): torch::nn::ReLU()
     )
     (2): torch::nn::Sequential(
       (0): knn::Mul()
     )
   )

   q)x:-3 -2 -1 0 1 2 3e
   q)y:1000e

   q)evaluate(m;x;y)
   47.43 119.2 268.9 500 731.1 880.8 952.6e

   q)mul(relu sigmoid x;y)
   47.43 119.2 268.9 500 731.1 880.8 952.6e

.. index:: seqnest
.. _module-seqnest:

seqnest
^^^^^^^
The `Pytorch c++ implementation of the Sequential module <https://pytorch.org/cppdocs/api/classtorch_1_1nn_1_1_sequential_impl.html>`_
uses a templated forward calculation: the result type defaults to a tensor, but can be defined as anything, along with the input arguments.
This prevents sequential modules from being nested, the outer sequential module cannot store the child sequential.
The ``seqnest`` module is derived from the main sequential class, but redefines the forward call as having a fixed result and number of arguments:
the forward call returns a tensor and accepts 1-3 tensor inputs.
The c++ module is defined in `knn/seq.h <https://github.com/ktorch/ktorch/blob/master/knn/seq.h>`_, `seq.cpp <https://github.com/ktorch/ktorch/blob/master/knn/seq.cpp>`_.

::

   q)q:module`sequential
   q)module(q; 1; `sequential)
   'sequential: unable to create type-erased module, forward method uses template
     [0]  module(q; 1; `sequential)
          ^

   q)module(q; 1; `seqnest)
   q)module(q; 2; (`linear; 5; 2))
   q)module(q; 2; `relu)

   q)-2 str q;
   torch::nn::Sequential(
     (0): knn::SeqNest(
       (0): torch::nn::Linear(in_features=5, out_features=2, bias=true)
       (1): torch::nn::ReLU()
     )
   )

   q)evaluate(q; 3 5#0 1 2 3 4e)
   0 2.25932
   0 2.25932
   0 2.25932

.. index:: callback
.. _module-callback:

callback
********
The ``callback`` module is a container designed to allow the forward calculation to call back to a specified function in the k session: this allows more flexible calculations than those supported by the c++ modules at the expense of some overhead and memory management.
The c++ module is defined in `knn/callback.h <https://github.com/ktorch/ktorch/blob/master/knn/callback.h>`_, `callback.cpp <https://github.com/ktorch/ktorch/blob/master/knn/callback.cpp>`_.

Available callbacks
^^^^^^^^^^^^^^^^^^^
The callback module is defined by the name of the k function to be called, along with the signature, expressed in terms of the input(s) and output.
the :func:`callbacks` function displays a table of callbacks defined in the interface:

::

   q)callbacks()
   fn            in                    out   
   ------------------------------------------
   "{[m;x]}"     `tensor               tensor
   "{[m;x;y]}"   `tensor`tensor        tensor
   "{[m;x;y;z]}" `tensor`tensor`tensor tensor

Defining a callback
^^^^^^^^^^^^^^^^^^^
To initialize a callback, specify a name, function name (or inline definition), and optional input/output types:

::

   q)cb1:module enlist(`callback; `cb; `f)

   q)-2 str cb1;
   knn::Callback(fn=f, in=tensor, out=tensor)

   q)f:{[m;x]relu x}

   q)evaluate(cb1; -2 -1 0 1 2e)
   0 0 0 1 2e

Simple callback functions can be defined as an input string:

::

   q)cb2:module enlist(`callback; `cb; "{[m;x;y] cat(x;y)}"; `tensor`tensor)

   q)-2 str cb2;
   knn::Callback(fn={[m;x;y] cat(x;y)}, in=tensor,tensor, out=tensor)

   q)evaluate(cb2; 1 2 3;4 5)
   1 2 3 4 5

.. note:

   The ``callback`` module expects up to 3 symbols, the type, module name and callback function name.  If only 2 symbols given, these will be parsed as module type and name.

::

   q)m:module(`callback;`cb)
   'callback: no k function defined, 2nd symbol, `cb, defines callback module name
     [0]  m:module(`callback;`cb)
            ^

   q)m:module(`callback;`cb;`f)     / 3rd symbol is given for function name

   q)-2 str m;
   knn::Callback(fn=f, in=tensor, out=tensor)

   q)m:module(`callback;`f;`tensor)  / 3rd symbol is recognizable result type

   q)-2 str m;
   knn::Callback(fn=tensor, in=tensor, out=tensor)

Child modules
^^^^^^^^^^^^^

The ``callback`` module is a generic container designed to hold any type of child modules: it is useful to define names with the child modules to reference them directly in the callback function. If the child modules are not given explicit names, the names will default to their sequence number, ```0``, ```1``, etc.

Here the callback child modules are defined using depth & value updates:

::

   q)m:module(`callback;`cb;`f)
   q)module(m; 1; (`linear;`a;5;2))  / add fully connected layer
   q)module(m; 1; (`sigmoid;`b))     / add activation function
   q)module(m; 1; (`drop;`c;.5))     / add dropout

   q)-2 str m;
   knn::Callback(fn=f, in=tensor, out=tensor)(
     (a): torch::nn::Linear(in_features=5, out_features=2, bias=true)
     (b): torch::nn::Sigmoid()
     (c): torch::nn::Dropout(p=0.5, inplace=false)
   )

   
In the next example the callback module is defined as a nested tree:

::

   q)m:module seq((`callback;`cb;`f); (`linear;`a;5;2); (`sigmoid;`b); (`drop;`c;.5))

   q)-2 str m;
   knn::Callback(fn=f, in=tensor, out=tensor)(
     (a): torch::nn::Linear(in_features=5, out_features=2, bias=true)
     (b): torch::nn::Sigmoid()
     (c): torch::nn::Dropout(p=0.5, inplace=false)
   )

The callback function in k is given the top-level module and tensor argument(s). Here, the function runs the forward call on each child module in turn, re-using the tensor ``y``, then returning it back to the module defined and allocated in c++:

::

   q)m:module seq((`callback;`cb;`f); (`linear;`a;5;2); (`sigmoid;`b); (`drop;`c;.5))
   q)f:{[m;x] y:kforward(m;`a;x); use[y]kforward(m;`b;y); use[y]kforward(m;`c;y); y}
   q)x:tensor(`randn; 3 5)

   q)seed 123          / set seed (for dropout layer)
   q)y:forward(m;x)    / forward call -> c++ -> k -> c++ -> k

Another form of the callback function:

::

   q)f:{{use[y]kforward(x;z;y);y}[x]/[y;`a`b`c]}

   q)seed 123
   q)tensor[y]~tensor y1:forward(m;x)
   1b

The forward calculation can be verified by defining a sequential module, using the same weight & bias and random seed:

::

   q)q:module seq(`sequential`q; (`linear;`a;5;2); (`sigmoid;`b); (`drop;`c))

   q)parmnames m
   `a.weight`a.bias

   q){t:parm(m;x); parm(q;x;t); free t}'[parmnames m];  /match wt & bias in callback

   q)seed 123
   q)yq:forward(q;x)
   q)tensor[y]~tensor yq
   1b

kforward
^^^^^^^^

The :func:`kforward` function is for use within k callback functions: it uses the overall parent module's train/eval setting to run the forward calculation with or without gradients.  If the callback module is nested within a larger parent module, the ``kforward`` call will follow the gradient on/off setting of its parent.

The top-level :ref:`forward calculation <forward>` will set the parent :ref:`training mode <module-training>` which will also turn on/off gradient calculation and determine whether to return tensors or k arrays.

::

   q)m:((`callback;`cb;`f); (`linear;`a;5;2); (`sigmoid;`b); (`drop;`c;.5))
   q)m:module seq m
   q)f:{{use[y]kforward(x;z;y);y}[x]/[y;`a`b`c]}
   q)x:tensor(`randn; 3 5)

   q)seed 123
   q)y:forward(m;x)   /return tensor from training mode with gradient calc
   q)k:evaluate(m;x)  /return k array using evaluation mode & no gradients

   q)tensor[y]~k      /differing results - training uses dropout layer
   0b

   q)k
   0.5788 0.4996
   0.8176 0.3275
   0.7367 0.4316

Other
*****

.. index:: droppath
.. _module-droppath:

droppath
^^^^^^^^
``droppath`` is used to drop layers according to the probability given to define the module. The input tensor to the module is set to zero along the first dimension according to the given probability during training only. The magnitude of the remaining input values is scaled up by the reciprocal of the zeroing probabilty to preserve the overall magnitude of the signal.

The c++ module is defined in `knn/drop.h <https://github.com/ktorch/ktorch/blob/master/knn/drop.h>`_, `drop.cpp <https://github.com/ktorch/ktorch/blob/master/knn/drop.cpp>`_.

A popular PyTorch implementation is found `here <https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/layers/drop.py>`_, part of
`Pytorch Image Models (timm) <https://timm.fast.ai/>`_.

::

   q)seed 123
   q)p:.1
   q)x:abs return tensor(`randn;10 1)
   q)y:x % 1-p
   q)d:module enlist(`droppath; p)

   q)([]x;y;r:return forward(d;x))
   x      y      r     
   --------------------
   0.1115 0.1239 0.1239
   0.1204 0.1337 0.1337
   0.3696 0.4107 0.4107
   0.2404 0.2671 0.2671
   1.197  1.33   1.33  
   0.2093 0.2325 0.2325
   0.9724 1.08   1.08  
   0.755  0.8389 0.8389
   0.3239 0.3599 0.3599
   0.1085 0.1206 0          /drop

   q)([]x;y;r:return forward(d;x))
   x      y      r     
   --------------------
   0.1115 0.1239 0.1239
   0.1204 0.1337 0.1337
   0.3696 0.4107 0.4107
   0.2404 0.2671 0.2671
   1.197  1.33   1.33  
   0.2093 0.2325 0.2325
   0.9724 1.08   0           /drop
   0.755  0.8389 0.8389
   0.3239 0.3599 0.3599
   0.1085 0.1206 0.1206


.. index:: nbeats
.. _module-nbeats:

nbeats
^^^^^^
The ``nbeats`` module is derived from PyTorch's `ModuleList <https://pytorch.org/docs/stable/generated/torch.nn.ModuleList.html>`_, accepting a series of module lists or sequential modules, but designed to perform a particular forward calculation specific to `NBEATS - a neural network architecture for time-series forecasting <https://arxiv.org/abs/1905.10437>`_.

- given ``x`` inputs & ``y`` (initially empty)
- for each sequential block:
   - ``b,f = forward(x)``
   - ``x=x-b``, ``y=y+f``
- return ``y``

The c++ module is defined in `knn/nbeats.h <https://github.com/ktorch/ktorch/blob/master/knn/nbeats.h>`_, `nbeats.cpp <https://github.com/ktorch/ktorch/blob/master/knn/nbeats.cpp>`_.

A basic example of the ``nbeats`` forward calculation with k functions:

::

   q)fork:{x-/:0 1}
   q)block:{[x]y:x 1; r:fork x@:0; (x-r 0;y+r 1)}

   q)fork 1 2 3  /apply the identity function and also increment
   1 2 3
   0 1 2

   q)3 block\(1 2 3;0) /fork and accumulate subtraction & addition on each result
   1 2 3   0      
   0 0 0   0 1 2  
   0  0 0  -1 0 1 
   0  0  0 -2 -1 0

Implementing the above with modules:

::

   q)c:enlist(`callback; "{[m;x]add(x;-1)}")
   q)q:(`sequential; (`fork; (`sequential; 1#`identity); (`sequential; c)))
   q)f:module q  / fork input -> (x; x+1)

   q)-2 str f;
   torch::nn::Sequential(
     (0): knn::Fork(
       (qa): torch::nn::Sequential(
         (0): torch::nn::Identity()
       )
       (qb): torch::nn::Sequential(
         (0): knn::Callback(fn={[m;x]add(x;-1)}, in=tensor, out=tensor)
       )
     )
   )

   q)evaluate(f; 1 2 3)
   1 2 3
   0 1 2

Creating a ``nbeats`` module with three blocks:

::

   q)m:module(`nbeats; q; q; q)

   q)evaluate(m;1 2 3)
   -2 -1 0

In an actual timeseries module, the sequential block ``q`` in the above example would be a sequence 
of ``linear`` layers followed by ``relu`` activations.


.. index:: selfattention
.. _module-selfattention:

selfattention
^^^^^^^^^^^^^

Pytorch implemented their `Multi-head attention <https://pytorch.org/docs/stable/generated/torch.nn.MultiheadAttention.html>`_ module to handle both self and cross attention; the module involves a lot of initialization options and up to 7 inputs for the forward calculation.
The k api adds a simpler self attention module for use when weighting different positions of a single sequence, i.e. when the key, query and value representations come from the same input.
The c++ module is defined in `knn/attention.h <https://github.com/ktorch/ktorch/blob/master/knn/attention.h>`_, `attention.cpp <https://github.com/ktorch/ktorch/blob/master/knn/attention.cpp>`_.

There are four options specific to the self-attention module:

.. function:: module enlist(`selfattention; dim; heads; dropout; norm) -> module

   :param long dim: the dimension of the embedding, no default. 
   :param long heads: the number of parallel attention heads (must be a multiple of given ``dim``, no default)
   :param double dropout: the probability of setting some outputs to zero (default is ``0.0``).
   :param bool norm: set ``true`` to first pass the input through a `layernorm <https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html>`_ layer, default is ``false``. When the normalization flag is set ``true``, the ``bias`` of the first linear layer after the normalization is turned off.

::

   q)help`selfattention
   module | `selfattention
   pytorch| "knn.SelfAttention"
   forward| 1b
   result | `tensor
   n      | 1
   args   | `tensor`tensor`tensor
   options| `dim`heads`dropout`norm!(512;8;0.1;0b)

   q)a:module enlist(`selfattention; 256; 4; .2; 1b)

   q)options a
   dim    | 256
   heads  | 4
   dropout| 0.2
   norm   | 1b

   q)-2 str a;
   knn::SelfAttention(dim=256, heads=4, dropout=0.2, norm=true)(
     (norm): torch::nn::LayerNorm([256], eps=1e-05, elementwise_affine=true)
     (in): torch::nn::Linear(in_features=256, out_features=768, bias=false)
     (drop): torch::nn::Dropout(p=0.2, inplace=false)
     (out): torch::nn::Linear(in_features=256, out_features=256, bias=true)
   )

The forward calculation accepts 1-3 tensors:

.. function:: forward(module; input) -> attention scores
.. function:: forward(module; input; mask) -> attention scores
   :noindex:
.. function:: forward(module; input; mask; padmask) -> attention scores
   :noindex:

   :param module ptr: An :doc:`api-pointer <pointers>` to the allocated module.
   :param tensor input: Tensor or k array, ``batch size x sequence length x embedding dimension``.
   :param tensor mask: An optional square matrix with ``-inf`` where attention is to be masked, ``sequence length x sequence length``.
   :param tensor padmask: An optional tensor or array indicating padding in the batch inputs, ``batch size x sequence length``.
   :return: :func:`forward` and :func:`eforward` returns a tensor, :func:`evaluate` returns a k array, output of same size as input.

::

   q)b:64; d:512; h:8; n:128  / batch size 64, dim 512, heads 8, seq length 128
   q)a:module enlist(`selfattention; d; h; .1; 1b)

   q)x:tensor(`randn; b,n,d)
   q)u:triu((2#n)#-0we;1)   /upper triangular attention mask

   q)u
   0 -0w -0w -0w -0w -0w -0w -..
   0 0   -0w -0w -0w -0w -0w -..
   0 0   0   -0w -0w -0w -0w -..
   0 0   0   0   -0w -0w -0w -..
   0 0   0   0   0   -0w -0w -..
   0 0   0   0   0   0   -0w -..
   0 0   0   0   0   0   0   -..
   ..

   q)y:forward(a; x; u)
   q)size y
   64 128 512

