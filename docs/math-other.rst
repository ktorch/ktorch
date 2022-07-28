Other
=====

Some of the PyTorch functions listed under `other math operations <https://pytorch.org/docs/stable/torch.html#other-operations>`_ are implemented in the k api and documented here.

bincount
^^^^^^^^

`torch.bincount <https://pytorch.org/docs/stable/generated/torch.bincount.html>`_ is implemented as :func:`bincount``.

.. function:: bincount(input;weight;bins) -> counts

   | Allowable argument combinations:

    - ``bincount(input)``
    - ``bincount(input;weight)``
    - ``bincount(input;weight;bins)``
    - ``bincount(input;bins)``

   :param list,tensor input: input array or tensor :doc:`pointer <pointers>` of non-negative integers (short,int,long)
   :param list,tensor weight: optional weights, of same length as input
   :param long bins: the optional number of bins, default is ``1+max(input)``
   :return: The counts of each element in input, or if weights supplied, the sum of weights corresponding to each element. If wither ``input`` or ``weight`` supplied as tensor, result is a tensor, else k list.

::

   q)x:4 3 6 3 4
   q)w:0 0.25 0.5 0.75 1

   q)(til 1+max x;bincount x)
   0 1 2 3 4 5 6
   0 0 0 2 2 0 1

   q)(til 10;bincount(x;10))
   0 1 2 3 4 5 6 7 8 9
   0 0 0 2 2 0 1 0 0 0

   q)(til 1+max x;bincount(x;w))
   0 1 2 3 4 5 6  
   0 0 0 1 1 0 0.5

   q)@[(1+max x)#0.0;key g;:;sum each w get g:group x]
   0 0 0 1 1 0 0.5

   q)bincount(x;w;8)
   0 0 0 1 1 0 0.5 0

   q)w:tensor w
   q)y:bincount(x;w)
   q)tensor y
   0 0 0 1 1 0 0.5

cumprod
^^^^^^^

`torch.cumprod <https://pytorch.org/docs/stable/generated/torch.cumprod.html>`_ is implemented as :func:`cumprod`.

.. function:: cumprod(x;dim;dtype) -> cumulative products
.. function:: cumprod(x;dim;dtype;output) -> null

   | Allowable argument combinations:

    - ``cumprod(x)``
    - ``cumprod(x;dim)``
    - ``cumprod(x;dim;dtype)``
    - ``cumprod(x;dtype)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param long dim: the optional dimension along which to evaluate, uses final dimension if none given
   :param symbol dtype: optional data type, e.g. ``double``, to use to convert input before calculations
   :param tensor output: an optional tensor :doc:`pointer <pointers>` to use for function output
   :return: Returns cumulative product across last dimension unless other dimension given.  If ``output`` tensor supplied, result is written to given tensor, null return.

::

   q)cumprod 1 2 3
   1 2 6

   q)cumprod(1 2 3;4 5 6)
   1 2  6  
   4 20 120

   q)cumprod((1 2 3;4 5 6);0)
   1 2  3 
   4 10 18

   q)y:cumsum(x:tensor (1 2 3;4 5 6);0)  / tensor input
   q)tensor y
   1 2 3
   5 7 9

   q)cumsum(x;y) / tensor output, last dim
   q)tensor y
   1 3 6 
   4 9 15

cumsum
^^^^^^

`torch.cumsum <https://pytorch.org/docs/stable/generated/torch.cumsum.html>`_ is implemented as :func:`cumsum`.

Function :func:`cumsum` takes the same arguments as :func:`cumprod` and returns cumulative sums.

.. function:: cumsum(x;dim;dtype) -> cumulative sums
.. function:: cumsum(x;dim;dtype;output) -> null

   | Allowable argument combinations:

    - ``cumsum(x)``
    - ``cumsum(x;dim)``
    - ``cumsum(x;dim;dtype)``
    - ``cumsum(x;dtype)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param long dim: the optional dimension along which to evaluate, uses final dimension if none given
   :param symbol dtype: optional data type, e.g. ``double``, to use to convert input before calculations
   :param tensor output: an optional tensor :doc:`pointer <pointers>` to use for function output
   :return: Returns cumulative sums across last dimension unless other dimension given.  If ``output`` tensor supplied, result is written to given tensor, null return.

diag
^^^^

`torch.diag <https://pytorch.org/docs/stable/generated/torch.diag.html>`_ is implemented as :func:`diag`, which either extracts a diagonal from 2-d input, or creates a 2-d square tensor from a 1-dimensional list.

.. function:: diag(input) -> extracted diagonal or created square matrix
.. function:: diag(input;offset) -> extracted diagonal or created square matrix
.. function:: diag(input;offset;output) -> null

   :param array,tensor input: input list/array or tensor :doc:`pointer <pointers>`
   :param long offset: the optional offset from which to extract or place the diagonal, default=0
   :param tensor output: an optional tensor :doc:`pointer <pointers>` to use for function output
   :return: If given a 1-d tensor or list, returns a square matrix tensor or k array. If given a 2-d array/tensor, returns the extracted diagonal as a k array/tensor. If output tensor supplied, writes output to tensor and returns null.

::

   q)diag 1 2 3
   1 0 0
   0 2 0
   0 0 3

   q)diag diag 1 2 3
   1 2 3

   q)diag(1 2 3 4;1)
   0 1 0 0 0
   0 0 2 0 0
   0 0 0 3 0
   0 0 0 0 4
   0 0 0 0 0

   q)diag(;1)diag(1 2 3 4;1)
   1 2 3 4

   q)x:tensor 1 2 3
   q)y:diag x
   q)tensor y
   1 0 0
   0 2 0
   0 0 3


diagflat
^^^^^^^^

`torch.diagflat <https://pytorch.org/docs/stable/generated/torch.diagflat.html>`_ is implemented as :func:`diagflat`.

.. function:: diagflat(input) -> square matrix
.. function:: diagflat(input;offset) -> square matrix

   :param array,tensor input: input list/array or tensor :doc:`pointer <pointers>`
   :param long offset: the optional offset for placing the diagonal, default=0
   :return: Given a 1-d tensor or list, returns a square matrix tensor or k array.

::

   q)diagflat 1 2 3
   1 0 0
   0 2 0
   0 0 3

   q)diagflat(1 2 3; -3)
   0 0 0 0 0 0
   0 0 0 0 0 0
   0 0 0 0 0 0
   1 0 0 0 0 0
   0 2 0 0 0 0
   0 0 3 0 0 0

   q)diag(;-3) diagflat(1 2 3; -3)
   1 2 3

diagonal
^^^^^^^^

`torch.diagonal <https://pytorch.org/docs/stable/generated/torch.diagonal.html>`_ is implemented as :func:`diagonal`.

.. function:: diagonal(input;offset;dim1;dim2) -> diagonal(s)

   | Allowable argument combinations:

    - ``diagonal(input)``
    - ``diagonal(input;offset)``
    - ``diagonal(input;offset;dim1)``
    - ``diagonal(input;offset;dim1;dim2)``

   :param array,tensor input: input array or tensor :doc:`pointer <pointers>`
   :param long offset: the optional offset from which to extract the diagonal, default=0
   :param long dim1: the 1st dimension from which to take the diagonal, default=0
   :param long dim2: the 2nd dimension from which to take the diagonal, default=1
   :return: Given a tensor or list, returns the extracted diagonal(s) as a tensor or k array.

::

   q)show x:4 4#til 16
   0  1  2  3 
   4  5  6  7 
   8  9  10 11
   12 13 14 15

   q)diagonal x
   0 5 10 15

   q)diagonal(x; -1)
   4 9 14


   q)x:tensor(x;x)+0 .1
   q)y:diagonal x
   q)tensor y
   0 4.1
   1 5.1
   2 6.1
   3 7.1

   q)use[y]diagonal(x; 0; 1; 2)
   q)tensor y
   0   5   10   15  
   0.1 5.1 10.1 15.1

flatten
^^^^^^^

`torch.flatten <https://pytorch.org/docs/stable/generated/torch.flatten.html>`_ is implemented as :func:`flatten`.

.. function:: flatten(input;start;end) -> flatten(s)

   | Allowable argument combinations:

    - ``flatten(input)``
    - ``flatten(input;start)``
    - ``flatten(input;start;end)``

   :param array,tensor input: input array or tensor :doc:`pointer <pointers>`
   :param long start: the first dimension to flatten, default=0, the first dimension of the input
   :param long end: the final dimension to flatten, default=-1, the final dimension of the input
   :return: Given a tensor or array, returns a reshaped tensor/array, removing the dimensions given or implied.

::

   q)x:2 3 4#til 24

   q)flatten x
   0 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23

   q)flatten(x;1;2)
   0  1  2  3  4  5  6  7  8  9  10 11
   12 13 14 15 16 17 18 19 20 21 22 23

Because :func:`flatten` is also implemented as a :ref:`module <module-args>`, named arguments are also accepted:

::

   q)flatten(x;`start`end!1 2)
   0  1  2  3  4  5  6  7  8  9  10 11
   12 13 14 15 16 17 18 19 20 21 22 23

   q)flatten(x;`end,1)
   0  1  2  3 
   4  5  6  7 
   8  9  10 11
   12 13 14 15
   16 17 18 19
   20 21 22 23

Flip
^^^^

`torch.flip <https://pytorch.org/docs/stable/generated/torch.flip.html>`_ is implemented as :func:`Flip`, which reverses the order of an n-dimensional tensor along axis given in a list of dimensions.

.. function:: Flip(input;dims) -> reversed along dims

   :param array,tensor input: input array or tensor :doc:`pointer <pointers>`
   :param long dim: a list of dimensions defining axis to flip
   :return: Returns an array if k array input, else tensor with the given axis reversed.

::

   q)x:tensor 2 3 4#til 24
   q)y:Flip(x;1 2)
   q)-2 str y;
   (1,.,.) = 
     11  10   9   8
      7   6   5   4
      3   2   1   0

   (2,.,.) = 
     23  22  21  20
     19  18  17  16
     15  14  13  12
   [ CPULongType{2,3,4} ]

   q)use[y]Flip(x;0 2)
   q)-2 str y;
   (1,.,.) = 
     15  14  13  12
     19  18  17  16
     23  22  21  20

   (2,.,.) = 
      3   2   1   0
      7   6   5   4
     11  10   9   8
   [ CPULongType{2,3,4} ]


histc
^^^^^

`torch.histc <https://pytorch.org/docs/stable/generated/torch.histc.html>`_ is implemented as :func:`histc`.

Computes the histogram of a tensor.
The elements are sorted into equal width bins between low and high limits. If low and high are both zero, the minimum and maximum values of the input are used.

Elements lower than min and higher than max are ignored.

.. function:: histc(input;bins;low;high) -> counts
.. function:: histc(input;bins;low;high;output) -> null

   | Allowable argument combinations:

    - ``histc(input)``
    - ``histc(input;bins)``
    - ``histc(input;bins;low)``
    - ``histc(input;bins;low;high)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor input: input array or tensor :doc:`pointer <pointers>`
   :param long bins: the optional number of bins, default is ``1+max(input)``
   :param double low:  lower limit, input values smaller than ``low`` are ignored, default=0
   :param double high: high limit, input values larger than ``high`` are ignored, default=0
   :return: Returns counts of values in equal width bins between high and low limits. Returns a tensor if tensor input, else a karray. If output tensor given, function output is written to the supplied tensor, null return.

::

   q)x:tensor(`randn; 1000)
   q)y:histc x
   q)size y
   ,100
   q)tensor y
   1 0 0 1 1 1 2 4 2 0 2 1 4 3 2 3 4 2 6 9 4 6 3 10 4 15 12 11 11 18 16 13 15 16..

   q)use[y]histc(x;11)
   q)tensor y
   12 22 73 141 190 225 182 102 38 14 1e

   q)use[y]histc(x;7;-3;3)
   q)-3 -2 1 0 1 2 3!tensor y
   -3| 21
   -2| 89
   1 | 231
   0 | 335
   1 | 236
   2 | 75
   3 | 12


renorm
^^^^^^

`torch.renorm <https://pytorch.org/docs/stable/generated/torch.renorm.html>`_ is implemented as :func:`renorm`.

.. function:: renorm(input;p;dim;maxnorm) -> renormalized input
.. function:: renorm(input;p;dim;maxnorm;output) -> null

   :param array,tensor input: input array or tensor :doc:`pointer <pointers>` of at least 2 dimensions
   :param double p:  the power for the norm computation
   :param long dim: the dimension of the input for calculating the renormalization
   :param double maxnorm: the maximum norm to use as the upper limit for each sub-tensor
   :return: Returns an array or tensor where each sub-tensor of ``input`` along dimension ``dim`` is normalized such that the p-norm of the sub-tensor is lower than ``maxnorm``. If an output tensor supplied, output is written to the supplied tensor, null return.

::

   q)x:3#'1 2 3 4.0
   q)renorm(x;2;1;5.0)
   0.9129 0.9129 0.9129
   1.826  1.826  1.826 
   2.739  2.739  2.739 
   3.651  3.651  3.651 

   q)r:tensor 0#0.0
   q)renorm(x;2;1;4;r)
   q)tensor r
   0.7303 0.7303 0.7303
   1.461  1.461  1.461 
   2.191  2.191  2.191 
   2.921  2.921  2.921 

roll
^^^^

`torch.roll <https://pytorch.org/docs/stable/generated/torch.roll.html>`_ is implemented as :func:`roll`,
to roll the input along the given dimension(s).
If dimensions are not specified, the input will be flattened before rolling and then restored to the original shape.

.. function:: renorm(input;shifts;dims) -> rolled input

   :param array,tensor input: input array or tensor :doc:`pointer <pointers>`
   :param long shift: the number of places by which the elements of the input are shifted. For multiple shifts, ``dim`` must specifiy corresponding dimensions.
   :param long dim: the dimension(s) or axis along which to roll
   :return: Returns an array or tensor where each sub-tensor of ``input`` along dimension ``dim`` is normalized such that the p-norm of the sub-tensor is lower than ``maxnorm``. If an output tensor supplied, output is written to the supplied tensor, null return.

::

   q)roll(1 2 3 4;2)
   3 4 1 2

   q)roll(1 2 3 4;-1)
   2 3 4 1

   q)x:4 2#1+til 8

   q)roll(x;1)
   8 1
   2 3
   4 5
   6 7

   q)roll(x;1;0)
   7 8
   1 2
   3 4
   5 6

   q)roll(x;2 1;0 1)
   6 5
   8 7
   2 1
   4 3

tensordot
^^^^^^^^^

`torch.tensordot <https://pytorch.org/docs/stable/generated/torch.tensordot>`_ is implemented as :func:`tensordot`, which calculates a generalized matrix product.

.. function:: tensordot(x;y;dim) -> product
.. function:: tensordot(x;y;dim1;dim2) -> product

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param array,tensor y: input array or tensor :doc:`pointer <pointers>`
   :param long dim: number of dimensions to contract
   :param longs dim1: list of dimensions for first input
   :param longs dim2: list of dimensions for second input
   :return: See `torch.tensordot <https://pytorch.org/docs/stable/generated/torch.tensordot>`_ for calculation. If any input is a tensor, result is returned as a tensor, else k value.

::

   q)x:y:3 3#0 1 2 3 4 5 6 7 8.0

   q)tensordot(x;y;1)
   15 18 21 
   42 54 66 
   69 90 111

   q)x$y
   15 18 21 
   42 54 66 
   69 90 111

   q)tensordot(x;y;2)
   204f

   q)sum raze x*y
   204f

   q)x:tensor 3 4 5#til 60
   q)y:tensor 4 3 2#til 24

   q)z:tensordot(x;y;1 0;0 1)
   q)tensor z
   4400 4730
   4532 4874
   4664 5018
   4796 5162
   4928 5306


trace
^^^^^

`torch.trace <https://pytorch.org/docs/stable/generated/torch.trace>`_ is implemented as :func:`trace`, which returns the sum of the elements of the diagonal of the 2-dim input.

.. function:: trace(input) -> sum of diagonal

   :parm matrix,tensor input: 2-d array or :doc:`tensor <pointers>`
   :return: Diagonal elements as k list if k input else tensor.

::

   q)x:3 3#til 9
   q)diagonal x
   0 4 8
   q)trace x
   12

tril
^^^^

`torch.tril <https://pytorch.org/docs/stable/generated/torch.tril.html>`_ is implemented as :func:`tril`,
which resets all but the lower triangular part of the input matrix (2-D tensor) or batch of matrices to zero.

.. function:: tril(input) -> input with zeros for values not part of lower triangle
.. function:: tril(input;offset) -> input with zeros for values not part of lower triangle as of offset
.. function:: tril(input;offset;output) -> null

   :param array,tensor input: input or tensor :doc:`pointer <pointers>` of 2-d matrix or batches of 2-d matrices
   :param long offset: the optional offset from which to determine the lower triangle, default=0
   :param tensor output: an optional tensor :doc:`pointer <pointers>` to use for function output, k unary null implies an in-place operation
   :return: Returns input matrix or set of matrices with non-zero values only for lower triangle as of given offset. If given output tensor, lower triangle is written to tensor with null return.

::

   q)tril 4 4#1
   1 0 0 0
   1 1 0 0
   1 1 1 0
   1 1 1 1

   q)tril 4 5#1
   1 0 0 0 0
   1 1 0 0 0
   1 1 1 0 0
   1 1 1 1 0

   q)last tril(2 4 4#1; -1)
   0 0 0 0
   1 0 0 0
   1 1 0 0
   1 1 1 0

Examples using output tensor:

::

   q)t:tensor 3 3#1+til 9
   q)r:tensor 0#0
   q)tril(t;r)
   q)tensor r
   1 0 0
   4 5 0
   7 8 9

   q)tril(t;[])  /null output tensor for in-place operation
   q)tensor t
   1 0 0
   4 5 0
   7 8 9

triu
^^^^

`torch.triu <https://pytorch.org/docs/stable/generated/torch.triu.html>`_ is implemented as :func:`triu`,
which resets all but the upper triangular part of the input matrix (2-D tensor) or batch of matrices to zero.

.. function:: triu(input) -> input with zeros for values not part of upper triangle
.. function:: triu(input;offset) -> input with zeros for values not part of upper triangle as of offset
.. function:: triu(input;offset;output) -> null

   :param array,tensor input: input or tensor :doc:`pointer <pointers>` of 2-d matrix or batches of 2-d matrices
   :param long offset: the optional offset from which to determine the upper triangle, default=0
   :param tensor output: an optional tensor :doc:`pointer <pointers>` to use for function output, k unary null implies an in-place operation
   :return: Returns input matrix or set of matrices with non-zero values only for upper triangle as of given offset. If given output tensor, upper triangle is written to tensor with null return.

::

   q)triu 3 3#1b
   111b
   011b
   001b

   q)first triu(2 3 3#1b; 1)
   011b
   001b
   000b

   q)t:tensor 3 3#1+til 9
   q)triu(t;[])  /output=null becomes in-place operation
   q)tensor t
   1 2 3
   0 5 6
   0 0 9

