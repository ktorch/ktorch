.. _kmodules:

Modules for k api
=================

The following modules are not directly available in PyTorch;
these modules were defined for the k api either for convenience when only the functional form is defined in PyTorch,
or to allow modules to be defined with a sequential or sequential-like container which defines the forward calculation implicitly.

The k api modules are defined in `knn.h <https://github.com/ktorch/ktorch/blob/master/knn.h>`_.

For example, the ``SeqNest`` module is defined to create a nestable Sequential module by fixing the forward method to return a tensor (rather than the standard, templated return which cannot be nested within another Sequential module. See..)

::

   // ----------------------------------------------------------------------------------
   // SeqNest - derived from Sequential to allow nested sequentials 
   //         - no templatized forward result means can be stored as an AnyModule
   //         - forward method accepts up to three tensors x,y,z w'y & z optional
   //           forward result is tensor only
   // ---------------------------------------------------------------------------------
   class TORCH_API SeqNestImpl : public torch::nn::SequentialImpl {
     public:
     using SequentialImpl::SequentialImpl;
   
     void pretty_print(std::ostream& stream) const override {
       stream << "SeqNest";
     }
   
     torch::Tensor forward(const torch::Tensor& x,const torch::Tensor& y={},const torch::Tensor& z={}) {
      if(y.defined())
       return z.defined() ? SequentialImpl::forward(x,y,z) : SequentialImpl::forward(x,y);
      else
       return SequentialImpl::forward(x);
     }

    protected:
     FORWARD_HAS_DEFAULT_ARGS({1, torch::nn::AnyValue(torch::Tensor())}, {2, torch::nn::AnyValue(torch::Tensor())})
   };

   TORCH_MODULE(SeqNest);


Convenience modules
*******************

The following module form of defined PyTorch functions collects the function options and stores them in a module, invoking the function with the options as the module's ``forward`` method.

.. index:: pad

.. _module-pad:


Pad
^^^
The k api adds a module equivalent of the PyTorch `pad <https://pytorch.org/docs/stable/generated/torch.nn.functional.pad.html>`_ function.
Given the padding size, along with optional padding mode of  ```constant``, ```reflect``, ```replicate`` or ```circular``,
and padding value (mode = ```constant`` only), returns a tensor padded to given size.

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

Squeeze & unsqueeze
^^^^^^^^^^^^^^^^^^^
Pytorch functions `squeeze <https://pytorch.org/docs/stable/generated/torch.squeeze.html>`_ and
` unsqueeze <https://pytorch.org/docs/stable/generated/torch.unsqueeze.html> are implemented in module form to allow them to be included in the sequence of operations of a container module's forward calculation. ```squeeze`` removes dimensions of size 1 throught the tensor if no dimension given, or only at the given dimension and ```unsqueeze`` adds a dimension of size 1 inserted at the given position (no default allowed).

::

   q)help`unsqueeze
   dim    | 0
   inplace| 0b

   q)m:module enlist(`unsqueeze;1)

   q)tensor r:forward(m; 1 2 3)
   1
   2
   3

   q)s:module `squeeze
   q)use[r]forward(s;r)
   q)tensor r
   1 2 3

   q)q:module seq(`sequential; (`unsqueeze;1); (`unsqueeze;1); (`squeeze;0))

   q)use[r]forward(q;1 2 3)
   q)tensor r
   1
   2
   3


.. index:: expand

.. _module-expand:

Expand
^^^^^^
PyTorch method `expand <https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html>`_ is implemented as a module with an option of expanded sizes for each singleton dimension, with ``-1`` used to define no change for the dimension.

::

   q)help`expand
   size| -1 -1 28 28

   q)m:module enlist(`expand;-1 3)

   q)tensor r:forward(m;5 1#til 5)
   0 0 0
   1 1 1
   2 2 2
   3 3 3
   4 4 4

.. index:: reshape

.. _module-reshape:

Reshape
^^^^^^^
This module implements the PyTorch `reshape <https://pytorch.org/docs/stable/generated/torch.reshape.html>`_ function, returning a tensor with the given size. One dimension may be given as -1 and will be recalculated to accomdate the tensor's overall number of elements. ``reshape`` attempts to use the same underlying storage as the input tensor, but if the input is not contiguous or has incompatible strides, ``reshape`` may create a copy.

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
PyTorch defines a `select method <https://pytorch.org/docs/stable/generated/torch.Tensor.select.html>`_ on a tensor to select or "slice" along a given dimension and index.  For a k array, this is similar to ``x[i]`` or ``x[;;i]``.
This convenience module allows for the select operation to be added to a sequence of operations as part of the forward calculation, e.g. select the final column of an output from the previous model.

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
PyTorch defines an `index_select <https://pytorch.org/docs/stable/generated/torch.index_select.html>`_ function 
which indexes a tensor along a given dimension using supplied indices, similar to the the above ``select``, but with a list of indices rather than a single scalar.

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

The ``transform`` module is somewhat similar to PyTorch's `vision transformer <https://pytorch.org/vision/stable/transforms.html>`_ in that it defines a set of transformations to perform on image data. The k api module can contain up to two sequential modules, the first for defining transforms on inputs when the module is in training mode, the second sequential contains transforms to perform on inputs when the module is in evaluation mode. Either sequential child module may be empty, the evaluation module may be omitted.

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

.. index:: randomcrop

.. _module-randomcrop:

Random cropping
^^^^^^^^^^^^^^^

The ``randomcrop`` module is similar to PyTorch's
`RandomCrop <https://pytorch.org/vision/stable/transforms.html?highlight=randomcrop#torchvision.transforms.RandomCrop>`_  transform, taking arguments of desired output size, amount of padding (a single number or all 4 numbers for left,right,top,bottom), padding mode and fill value if padding mode = ```constant``.
See :ref:`pad <module_pad>` for more on the padding that is applied prior to the cropping.

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

Random flipping
^^^^^^^^^^^^^^^

PyTorch has transforms for
`horizontal <https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomHorizontalFlip>`_ and
`vertical <https://pytorch.org/vision/stable/transforms.html#torchvision.transforms.RandomVerticalFlip>`_ flips where the image is randomly flipped with the given probability. The ``randomflip`` module in the k api takes a second argument of dimension to flip, e.g. -1 for last or horizontal flip, -2 for 2nd to last or vertical flip. The input image can have several leading dimensions, e.g. n x c x h x w where n is number of images, c is number of channels, h & w are height and width.

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

Zscore
^^^^^^

The ``zscore`` module is similar to PyTorch's `normalize <https://pytorch.org/vision/stable/transforms.html?highlight=normalize#torchvision.transforms.Normalize>`_ transform.  The argument are the mean(s) and standard deviation(s), along with a flag for transforming the tensor in place. Means and standard deviations can be supplied for each RGB channel or in some other form where the sample statistics can be broadcast across the input tensor.

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

