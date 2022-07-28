Reduction operators
===================

PyTorch `reduction <https://pytorch.org/docs/stable/torch.html#reduction-ops>`_ operators:

 - `torch.argmax <https://pytorch.org/docs/stable/generated/torch.argmax.html>`_ implemented as :func:`argmax`
 - `torch.argmin <https://pytorch.org/docs/stable/generated/torch.argmin.html>`_ implemented as :func:`argmin`
 - `torch.amax <https://pytorch.org/docs/stable/generated/torch.amax.html>`_ implemented as :func:`amax`
 - `torch.amin <https://pytorch.org/docs/stable/generated/torch.amin.html>`_ implemented as :func:`amin`
 - `torch.aminmax <https://pytorch.org/docs/stable/generated/torch.aminmax.html>`_ implemented as :func:`aminmax`
 - `torch.all <https://pytorch.org/docs/stable/generated/torch.all.html>`_ implemented as :func:`All`
 - `torch.any <https://pytorch.org/docs/stable/generated/torch.any.html>`_ implemented as :func:`Any`
 - `torch.max <https://pytorch.org/docs/stable/generated/torch.max.html>`_ implemented as :func:`Max`
 - `torch.min <https://pytorch.org/docs/stable/generated/torch.min.html>`_ implemented as :func:`Min`
 - `torch.dist <https://pytorch.org/docs/stable/generated/torch.dist.html>`_ implemented as :func:`dist`
 - `torch.logsumexp <https://pytorch.org/docs/stable/generated/torch.logsumexp.html>`_ implemented as :func:`logsumexp`
 - `torch.mean <https://pytorch.org/docs/stable/generated/torch.mean.html>`_ implemented as :func:`mean`
 - `torch.nanmean <https://pytorch.org/docs/stable/generated/torch.nanmean.html>`_ implemented as :func:`nanmean`
 - `torch.median <https://pytorch.org/docs/stable/generated/torch.median.html>`_ implemented as :func:`median`
 - `torch.nanmedian <https://pytorch.org/docs/stable/generated/torch.nanmedian.html>`_ implemented as :func:`nanmedian`
 - `torch.mode <https://pytorch.org/docs/stable/generated/torch.mode.html>`_ implemented as :func:`mode`
 - `torch.norm <https://pytorch.org/docs/stable/generated/torch.norm.html>`_ implemented as separate :ref:`norm calculations <normfns>`: :func:`fnorm`, :func:`nnorm` and :func:`pnorm`.
 - `torch.nansum <https://pytorch.org/docs/stable/generated/torch.nansum.html>`_ implemented as :func:`nansum`
 - `torch.prod <https://pytorch.org/docs/stable/generated/torch.prod.html>`_ implemented as :func:`prod`
 - `torch.std <https://pytorch.org/docs/stable/generated/torch.std.html>`_ implemented as :func:`std`
 - `torch.std_mean <https://pytorch.org/docs/stable/generated/torch.std_mean.html>`_ implemented as :func:`meanstd`
 - `torch.sum <https://pytorch.org/docs/stable/generated/torch.sum.html>`_ implemented as :func:`sum`
 - `torch.unique <https://pytorch.org/docs/stable/generated/torch.unique.html>`_ implemented as :func:`unique`
 - `torch.unique_consecutive <https://pytorch.org/docs/stable/generated/torch.unique_consecutive.html>`_ implemented as :func:`uniquec`
 - `torch.var <https://pytorch.org/docs/stable/generated/torch.var.html>`_ implemented as :func:`variance`
 - `torch.var_mean <https://pytorch.org/docs/stable/generated/torch.var_mean.html>`_ implemented as :func:`meanvar`

Any / All
^^^^^^^^^

- `Any <https://pytorch.org/docs/stable/generated/torch.any.html>`_ - returns ``true`` if any ``true``, with optional dimension
- `All <https://pytorch.org/docs/stable/generated/torch.all.html>`_ - returns ``ture`` if all ``true``, with optional dimension

.. function:: Any(x;dim;keepdim) -> any ``true`` across optional dimension
.. function:: Any(x;dim;keepdim;output) -> null

   | Allowable argument combinations:

    - ``Any(x)``
    - ``Any(x;dim)``
    - ``Any(x;dim;keepdim)``
    - ``Any(x;keepdim)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param long dim: the optional dimension along which to evaluate.
   :param bool keepdim: default ``false``, set ``true`` to preserve the dimension of the input
   :param tensor output: an optional tensor :doc:`pointer <pointers>` to use for function output
   :return: Returns a single boolean if input given without additional dimension, else a list or higher dimension array/tensor depending on setting of the ``keepdim`` flag.  If ``output`` tensor supplied, result is written to given tensor, null return.

Function :func:`All` has the same syntax:

.. function:: All(x;dim;keepdim) -> any ``true`` across optional dimension
.. function:: All(x;dim;keepdim;output) -> null

   :param: Function :func:`All` uses the same parameters as :func:`Any`
   :return: Returns a single boolean if input given without additional dimension, else a list or higher dimension array/tensor depending on setting of the ``keepdim`` flag.  If ``output`` tensor supplied, result is written to given tensor, null return.

::

   q)show x:2 5#011b
   01101b
   10110b

   q)All x
   0b
   q)Any x
   1b

   q)Any(x;0)
   11111b

   q)Any(x;1;1b)
   ,1b
   ,1b

   q)x:tensor x     /use tensor in place of array
   q)y:Any(x;1;1b)
   q)tensor y
   ,1b
   ,1b

   q)All(x;1;1b;y)  /use output tensor
   q)tensor y
   ,0b
   ,0b


amax / amin
^^^^^^^^^^^

- `amax <https://pytorch.org/docs/stable/generated/torch.amax.html>`_ - returns maximum values across specified dimension
- `amin <https://pytorch.org/docs/stable/generated/torch.amin.html>`_ - returns minimum values across specified dimension

Function :func:`amax` returns maximum value(s), with the option of specifying a dimension along which to consider:

.. function:: amax(x;dim;keepdim) -> maximum values across specified dimension
.. function:: amax(x;dim;keepdim;output) -> null

   | Allowable argument combinations:

    - ``amax(x)``
    - ``amax(x;dim)``
    - ``amax(x;dim;keepdim)``
    - ``amax(x;keepdim)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param long dim: the optional dimension along which to calculate the maximum values and indices
   :param bool keepdim: default ``false``, set ``true`` to preserve the dimension of the input for the maximum values
   :param tensor output: an optional tensor :doc:`pointer <pointers>` to use for function output
   :return: Returns a single maximum if an array or tensor given without additional dimension, else a list or higher dimension array/tensor depending on setting of the ``keepdim`` flag.  If ``output`` tensor supplied, result is written to given tensor, null return.

Function :func:`amin` has the same syntax:

.. function:: amin(x;dim;keepdim) -> minimum values across specified dimension
.. function:: amin(x;dim;keepdim;output) -> null

   :param: Function :func:`amin` uses the same parameters as :func:`amax`
   :return: Returns a single minimum if an array or tensor given without additional dimension, else a list or higher dimension array/tensor depending on setting of the ``keepdim`` flag.  If ``output`` tensor supplied, result is written to given tensor, null return.

::

   q)amax 1.1 2 3.5
   3.5

   q)y:amax x:tensor 1.1 2 3.5
   q)dim y
   0
   q)tensor y
   3.5

   q)amin(x;0;y)
   q)tensor y
   1.1

   q)amax 2 3#til 6
   5

   q)amax(2 3#til 6;1)
   2 5


aminmax
^^^^^^^

PyTorch `aminmax <https://pytorch.org/docs/stable/generated/torch.aminmax.html>`_ returns minimum and maximum values across specified dimension.

.. function:: aminmax(x;dim;keepdim) -> minimum and maximum values
.. function:: aminmax(x;dim;keepdim;output) -> null

   | Allowable argument combinations:

    - ``aminmax(x)``
    - ``aminmax(x;dim)``
    - ``aminmax(x;dim;keepdim)``
    - ``aminmax(x;keepdim)``
    - any of the above combinations followed by a trailing output vector

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param long dim: the optional dimension along which to calculate the minimum and maximum values.
   :param bool keepdim: default ``false``, set ``true`` to preserve the dimension of the input for the minimum and maximum values.
   :param vector output: a vector `pointer <vectors>`, either an empty vector or a 2-element vector of data type matching input.
   :return: Returns minimum and maximum values of given input, or minimum and maximum values along the supplied dimension. Returns a single tensor if a single tensor supplied, else a vector of minimum and maximum values.  If a vector supplied as final argument, writes minimum and maximum values to the vector and returns null.

::

   q)x:tensor(0.1 1.1 2.1; 3.2 4.2 5.2)
   q)tensor x
   0.1 1.1 2.1
   3.2 4.2 5.2

   q)size v:aminmax(x;0;0b)
   3
   3
   q)vector v
   0.1 1.1 2.1
   3.2 4.2 5.2

   q)use[v]aminmax(x;1;1b)
   q)size v
   2 1
   2 1

   q)vector(v;0)
   0.1
   3.2


argmax / argmin
^^^^^^^^^^^^^^^

- `argmax <https://pytorch.org/docs/stable/generated/torch.argmax.html>`_ - returns maximum indices across specified dimension
- `argmin <https://pytorch.org/docs/stable/generated/torch.argmin.html>`_ - returns minimum indices across specified dimension

Function :func:`argmax` returns indices of maximum value(s), with the option of specifying a dimension along which to consider:

.. function:: argmax(x;dim;keepdim) -> indices of maximum values across specified dimension
.. function:: argmax(x;dim;keepdim;output) -> null

   | Allowable argument combinations:

    - ``argmax(x)``
    - ``argmax(x;dim)``
    - ``argmax(x;dim;keepdim)``
    - ``argmax(x;keepdim)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param long dim: the optional dimension along which to determine the indices of maximum values
   :param bool keepdim: default ``false``, set ``true`` to preserve the dimension of the input for the maximum indices
   :param tensor output: an optional tensor :doc:`pointer <pointers>` to use for function output
   :return: Returns a single index of maximum if an array or tensor given without additional dimension; the index is into a flattened 1-d list made from the array. If a dimension is given, returns a list or higher dimension array/tensor depending on setting of the ``keepdim`` flag.  If ``output`` tensor supplied, result is written to given tensor, null return.

Function :func:`argmin` returns indices of minimum value(s), with the same syntax:

.. function:: argmin(x;dim;keepdim) -> indices of minimum values across specified dimension
.. function:: argmin(x;dim;keepdim;output) -> null

   :param: Function :func:`argmin` uses the same parameters as :func:`argmax`
   :return: Returns a single index of minimum if an array or tensor given without additional dimension; the index is into a flattened 1-d list made from the array. If a dimension is given, returns a list or higher dimension array/tensor depending on setting of the ``keepdim`` flag.  If ``output`` tensor supplied, result is written to given tensor, null return.

::

   q)argmax 2 3#til 6
   5

   q)argmax(2 3#til 6;1)
   2 2

   q)argmax((1 3 2; 6 5 4); 1)
   1 0

   q)y:argmin(x:tensor(1 3 2; 6 5 4); 1; 1b)

   q)size y
   2 1

   q)tensor y
   0
   2

Max / Min
^^^^^^^^^

- `Max <https://pytorch.org/docs/stable/generated/torch.maximum.html>`_ - maximum values and indices across specified dimension or overall maximum
- `Min <https://pytorch.org/docs/stable/generated/torch.minimum.html>`_ - minimum values and indices across specified dimension or overall minimum

:func:`Max` returns maximum values, and if a dimension is given, the indices where the values occur in the given dimension.

.. function:: Max(x) -> k array or tensor of maximum value
.. function:: Max(x;dim;keepdim) -> maximum values and indices
.. function:: Max(x;dim;keepdim;output) -> null

   | Allowable argument combinations:

    - ``Max(x)``
    - ``Max(x;dim)``
    - ``Max(x;dim;keepdim)``
    - ``Max(x;keepdim)``
    - any of the above combinations followed by a trailing output vector

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param long dim: the optional dimension along which to calculate the maximum values and indices
   :param bool keepdim: default ``false``, set ``true`` to preserve the dimension of the input for the values and indices
   :param vector output: a vector `pointer <vectors>`
   :return: Returns maximum values of given input along with the indices where the values occur along the supplied dimension. Returns a single maximum if a single tensor supplied without additional arguments.  If a vector supplied as final argument, writes maximum values and indices to the vector and returns null.

:func:`Min` has the same syntax:

.. function:: Min(x) -> k array or tensor of minimum value
.. function:: Min(x;dim;keepdim) -> minimum values and indices
.. function:: Min(x;dim;keepdim;output) -> null

   :param: Function :func:`Min` uses the same parameters as :func:`Max`
   :return: Returns minimum values of given input along with the indices where the values occur along the supplied dimension. Returns a single minimum if a single tensor supplied without additional arguments.  If a vector supplied as final argument, writes minimum values and indices to the vector and returns null.

::

   q)Max 1 2 3
   3

   q)y:Max x:tensor 1 2 3
   q)tensor y
   3

   q)use[x].1+2 3#til 6
   q)v:Max(x;0)
   q)vector v
   3.1 4.1 5.1
   1   1   1  


   q)use[x].1+2 3#til 6
   q)tensor x
   0.1 1.1 2.1
   3.1 4.1 5.1
   q)vector v:Max(x;0)
   3.1 4.1 5.1
   1   1   1  

   q)use[v]0#'vector v
   q)Min(x;1;v)
   q)vector v
   0.1 3.1
   0   0  

.. note::

   Specifying a output vector with incorrect data types will cause an error; a warning displays if the shape of each element does not match the result.

::

   q)v:vector()
   q)x:2 3#til 6
   q)Max(x;0;v)

   q)vector v
   3 4 5
   1 1 1

   q)Max(x;1;v)
   [W Resize.cpp:24] Warning: An output with one or more elements was resized since it had shape [3], which does not match the required output shape [2].This behavior is deprecated, and in a future PyTorch release outputs will not be resized unless they have zero elements. You can explicitly reuse an out tensor t by resizing it, inplace, to zero elements with t.resize_(0). (function resize_output_check)

   q)vector v
   2 5
   2 2

   q)to(v;`short)
   q)Max(x;1;v)
   'Expected out tensor to have dtype long int, but got short int instead
     [0]  Max(x;1;v)
          ^

mean / nanmean
^^^^^^^^^^^^^^

.. function:: mean(x;dim;keepdim;dtype) -> mean, overall or over given dimensions
.. function:: mean(x;dim;keepdim;dtype;output) -> null

   | Allowable argument combinations:

    - ``mean(x)``
    - ``mean(x;dim)``
    - ``mean(x;dim;keepdim)``
    - ``mean(x;dim;keepdim;dtype)``
    - ``mean(x;dtype)``
    - ``mean(x;keepdim)``
    - ``mean(x;keepdim;dtype)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param longs dim: the optional dimension(s) along which to calculate the mean
   :param bool keepdim: default ``false``, set ``true`` to preserve the dimension of the input for the mean
   :param tensor output: a tensor `pointer <pointers>` to contain the means.
   :return: Returns overall mean or means along specified dimension(s).  If an output tensor supplied as final argument, writes mean(s) to the tensor and returns null.

:func:`nanmean` has the same syntax, but calculates the mean based on all ``non-Nan`` values, whereas :func:`mean` will return ``NaN`` if any ``NaN`` in the input.

.. function:: nanmean(x;dim;keepdim;dtype) -> mean, overall or over given dimensions
.. function:: nanmean(x;dim;keepdim;dtype;output) -> null

::

   q)mean 1 2 3.4
   2.133333

   q)mean(1 2 3.4;`float)
   2.133333e

   q)mean(1 2 3.4 0n;`float)
   0Ne

   q)nanmean(1 2 3.4 0n;`float)
   2.133333e

   q)show x:(1 2 3 0n;5 0n 6 7)
   1 2 3  
   5   6 7

   q)mean(x;0 1)
   0n
   q)nanmean(x;0 1)
   4f

   q)y:mean(x;0;0b;`float)  /mean down the rows
   q)tensor y
   3 0N 4.5 0Ne

   q)nanmean(x;0;`float;y)  /output tensor, omit nulls
   q)tensor y
   3 2 4.5 7e

   q)use[y]nanmean(x;1;1b;`float)  /keep original dimensionality
   q)tensor y
   2
   6


median / nanmedian
^^^^^^^^^^^^^^^^^^

.. function:: median(x) -> overall median
.. function:: median(x;dim;keepdim) -> median values and indices over final or given dimensions
.. function:: median(x;dim;keepdim;output) -> null

   | Allowable argument combinations:

    - ``median(x)``
    - ``median(x;dim)``
    - ``median(x;dim;keepdim)``
    - ``median(x;dim;keepdim)``
    - ``median(x;keepdim)``
    - any of the above combinations followed by a trailing output vector

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param long dim: the optional dimension along which to calculate the median
   :param bool keepdim: default ``false``, set ``true`` to preserve the dimension of the input for the median
   :param vector output: a vector `pointer <vectors>`
   :return: Returns overall median or medians & indices along specified dimension(s).  If an output vector supplied as final argument, writes medians and indices to the vector and returns null.

:func:`nanmedian` has the same syntax, but calculates the median based on all ``non-Nan`` values, whereas :func:`median` will return ``NaN`` if any ``NaN`` in the input.

.. function:: nanmedian(x) -> overall median
.. function:: nanmedian(x;dim;keepdim) -> median values and indices over final or given dimensions
.. function:: nanmedian(x;dim;keepdim;output) -> null

::

   q)median(1 2 3 4 5.0)
   3f

   q)median(1 2 3 4 5 0n)
   0n

   q)nanmedian(1 2 3 4 5 0n)
   3f

   q)x:tensor(`randn;3 5)
   q)tensor x
   1.89   -1.47 0.373  0.11   -0.864
   -0.698 0.718 -0.881 0.0457 0.0117
   -0.416 1.57  0.765  -1.01  0.912 

   q)v:median(x;1)
   q)vector v
   0.11 0.0117 0.765
   3    4      2    

   q)use[v]median(x;0)  /median across rows
   q)vector v
   -0.416 0.718 0.373 0.0457 0.0117
   2      1     0     1      1     

   q)median(x;0;v)  /using output vector
   q)vector v
   -0.416 0.718 0.373 0.0457 0.0117
   2      1     0     1      1     


mode
^^^^

.. function:: mode(x;dim;keepdim) -> mode values and indices over final or given dimensions
.. function:: mode(x;dim;keepdim;output) -> null

   | Allowable argument combinations:

    - ``mode(x)``
    - ``mode(x;dim)``
    - ``mode(x;dim;keepdim)``
    - ``mode(x;dim;keepdim)``
    - ``mode(x;keepdim)``
    - any of the above combinations followed by a trailing output vector

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param long dim: the optional dimension along which to calculate the mode, defaults to last dim
   :param bool keepdim: default ``false``, set ``true`` to preserve the dimension of the input for the mode values and indices
   :param vector output: a vector `pointer <vectors>`
   :return: Returns arrays or tensor vector of mode values & indices along specified dimension(s).  If an output vector supplied as final argument, writes medians and indices to the vector and returns null.

::

   q)x:tensor(`randint;3;5 7)
   q)tensor x
   0 0 1 1 1 0 2
   0 2 2 2 0 0 0
   2 0 1 0 0 0 2
   0 0 2 0 2 0 2
   2 1 1 1 2 1 2

   q)v:mode x
   q)vector v
   0 0 0 0 1
   5 6 5 5 5

   q)use[v]mode(x;0)
   q)vector v
   0 0 1 0 0 0 2
   3 3 4 3 2 3 4

std
^^^

The PyTorch `std <https://pytorch.org/docs/stable/generated/torch.std.html>`_ function is reimplemented for the k api as :func:`std` to calculate the standard deviation:

.. function:: std(x;dim;unbiased;keepdim) -> standard deviation
.. function:: std(x;dim;unbiased;keepdim;output) -> null

   | Allowable argument combinations:

    - ``std(x)``
    - ``std(x;dim)``
    - ``std(x;dim;unbiased)``
    - ``std(x;dim;unbiased;keepdim)``
    - ``std(x;unbiased)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`, will be converted to 4-byte float if integral type before calculations
   :param longs dim: the optional dimension(s) along which to calculate the standard deviation
   :param bool unbiased: default ``true``, set ``false`` to calculate sample standard deviation without Bessel's correction.
   :param bool keepdim: default ``false``, set ``true`` to preserve the dimension of the input for the standard deviation calculations.
   :param tensor output: an output tensor :doc:`pointer <pointers>`
   :return: Returns array or tensor of standard deviation overall or along specified dimension(s).  If an output tensor supplied as final argument, writes standard deviation(s) to the tensor and returns null.

var
^^^

PyTorch's `var <https://pytorch.org/docs/stable/generated/torch.var.html>`_ function is implemented as :func:`variance` for the k-api.

.. function:: variance(x;dim;unbiased;keepdim) -> standard deviation
.. function:: variance(x;dim;unbiased;keepdim;output) -> null

The arguments and calling syntax are the same as for the :func:`std` k api function.

::

   q)x:"e"$3 5#til 15

   q)std x
   4.47e

   q)sdev raze x  / k equivalent
   4.47

   q)std(x; -1; 1b; 1b)  / calculate along columns, keep dim
   1.58
   1.58
   1.58

   q)sdev each x
   1.58 1.58 1.58

   q)x:tensor x
   q)y:variance(x; -1; 1b; 1b)  / calculate along columns, keep dim
   q)tensor y
   2.5
   2.5
   2.5

   q)svar each tensor x
   2.5 2.5 2.5

   q)variance(x; 1; 0b; 1b; y)
   q)tensor y
   2
   2
   2

   q)var each tensor x
   2 2 2f

meanstd
^^^^^^^

PyTorch's `std_mean <https://pytorch.org/docs/stable/generated/torch.std_mean.html>`_ is implemented as k api function :func:`meanstd`:

.. function:: meanstd(x;dim;unbiased;keepdim) -> mean and standard deviation

   | Allowable argument combinations:

    - ``meanstd(x)``
    - ``meanstd(x;dim)``
    - ``meanstd(x;dim;unbiased)``
    - ``meanstd(x;dim;unbiased;keepdim)``
    - ``meanstd(x;unbiased)``

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`, will be converted to 4-byte float if integral type before calculations
   :param longs dim: the optional dimension(s) along which to calculate the standard deviation
   :param bool unbiased: default ``true``, set ``false`` to calculate sample standard deviation without Bessel's correction.
   :param bool keepdim: default ``false``, set ``true`` to preserve the dimension of the input for the standard deviation calculations.
   :param tensor output: an output tensor :doc:`pointer <pointers>`
   :return: Returns mean and standard deviation as an array or tensor along the first dimension.

meanvar
^^^^^^^

PyTorch's `var_mean <https://pytorch.org/docs/stable/generated/torch.var_mean.html>`_ is implemented as k api function :func:`meanvar`:

.. function:: meanvar(x;dim;unbiased;keepdim) -> mean and variance.

The arguments and calling syntax are the same as for the :func:`meanstd` k api function.

::

   q)x:"e"$3 5#til 15
   q)meanstd x
   7 4.47e

   q)meanstd(x;0)  / across rows
   5 6 7 8 9
   5 5 5 5 5

   q)meanstd(x;1)  / across cols
   2    7    12  
   1.58 1.58 1.58

   q)(avg each x;sdev each x)  / k equivalent calculation
   2    7    12  
   1.58 1.58 1.58

   q)meanvar(x;1)
   2   7   12 
   2.5 2.5 2.5

   q)x:tensor x
   q)y:meanvar(x;1)
   q)tensor y
   2   7   12 
   2.5 2.5 2.5

prod
^^^^

.. function:: prod(x;dim;keepdim;dtype) -> overall product or product over given dimension
.. function:: prod(x;dim;keepdim;dtype;output) -> null

   | Allowable argument combinations:

    - ``prod(x)``
    - ``prod(x;dim)``
    - ``prod(x;dim;keepdim)``
    - ``prod(x;dim;keepdim;dtype)``
    - ``prod(x;dtype)``
    - ``prod(x;keepdim)``
    - ``prod(x;keepdim;dtype)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param long dim: the optional dimension along which to calculate the product
   :param bool keepdim: default ``false``, set ``true`` to preserve the dimension of the input for the product
   :param tensor output: a tensor `pointer <pointers>` to contain the product
   :return: Returns overall product or product along specified dimension.  If an output tensor supplied as final argument, writes the product to the tensor and returns null.

::

   q)show x:(1 2 3e;4 5 6e)
   1 2 3
   4 5 6

   q)prod x
   720e

   q)prod(x;`double)
   720f

   q)prod(x;1;1b;`double)
   6  
   120

   q)y:tensor 0#0n
   q)prod(x;1;1b;`double;y)
   q)tensor y
   6  
   120


sum / nansum
^^^^^^^^^^^^

.. function:: sum(x;dim;keepdim;dtype) -> sum, overall or over given dimensions
.. function:: sum(x;dim;keepdim;dtype;output) -> null

   | Allowable argument combinations:

    - ``sum(x)``
    - ``sum(x;dim)``
    - ``sum(x;dim;keepdim)``
    - ``sum(x;dim;keepdim;dtype)``
    - ``sum(x;dtype)``
    - ``sum(x;keepdim)``
    - ``sum(x;keepdim;dtype)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param longs dim: the optional dimension(s) along which to sum
   :param bool keepdim: default ``false``, set ``true`` to preserve the dimension of the input for the sum(s)
   :param tensor output: a tensor `pointer <pointers>` to contain the sum(s).
   :return: Returns overall sum or sums along specified dimension(s).  If an output tensor supplied as final argument, writes sum(s) to the tensor and returns null.

:func:`nansum` has the same syntax, but sums over all ``non-Nan`` values, whereas :func:`sum` will return ``NaN`` if any ``NaN`` in the input.

.. function:: nansum(x;dim;keepdim;dtype) -> sum, overall or over given dimensions
.. function:: nansum(x;dim;keepdim;dtype;output) -> null

::

   q)x:tensor(1 2 3 4.0; 5 6 0n 8)
   q)tensor x
   1 2 3 4
   5 6   8

   q)y:Sum(x)
   q)tensor y
   0n

   q)use[y]nansum(x)
   q)tensor y
   29f

   q)use[y]nansum(x;0;`float)
   q)tensor y
   6 8 3 12e

   q)Sum(x;0;`float;y)
   q)tensor y
   6 8 0N 12e

unique
^^^^^^

.. function:: unique(x;dim;sort;indices;counts) -> unique elements with optional indices and counts

   | Allowable argument combinations:

    - ``unique(x)``
    - ``unique(x;dim)``
    - ``unique(x;dim;sort)``
    - ``unique(x;dim;sort;indices)``
    - ``unique(x;dim;sort;indices;counts)``
    - ``unique(x;sort)``
    - ``unique(x;sort;indices)``
    - ``unique(x;sort;indices;counts)``

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param long dim: the optional dimension along which to find unique elements
   :param bool sort: default ``true`` to return unique elements in sorted order
   :param bool indices: default ``false``, set ``true`` to return indices as well
   :param bool counts: default ``false``, set ``true`` to return counts as well
   :return: Return unique elements and indices and or counts depending on flags, as array(s) if k array input, else a tensor or vector of tensors if tensor input.

::

   q)x:1 1 1 2 2 0 0 0 1 1 1 3

   q)unique x
   0 1 2 3

   q)unique(x;0b)  / no sorting of result
   3 0 2 1

   q)unique(x;1b;0b;1b) /also return counts
   0 1 2 3
   3 6 2 1

   q)unique((x;x))
   0 1 2 3

   q)unique((x;x);1)  /unique across columns
   0 1 2 3
   0 1 2 3

   q)unique((x;x);0)  /across rows
   1 1 1 2 2 0 0 0 1 1 1 3

   q)x:tensor x
   q)v:unique(x;1b;1b;1b)
   q)vector v
   0 1 2 3
   1 1 1 2 2 0 0 0 1 1 1 3
   3 6 2 1

uniquec
^^^^^^^

.. function:: uniquec(x;dim;indices;counts) -> first element from consectutive groups with optional indices and counts

   | Allowable argument combinations:

    - ``uniquec(x)``
    - ``uniquec(x;dim)``
    - ``uniquec(x;dim;indices)``
    - ``uniquec(x;dim;indices;counts)``
    - ``uniquec(x;indices)``
    - ``uniquec(x;indices;counts)``

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param long dim: the optional dimension along which to find first elements in consecutive groups
   :param bool indices: default ``false``, set ``true`` to return indices as well
   :param bool counts: default ``false``, set ``true`` to return counts as well
   :return: Return first element from consecutive groups with optional indices and counts. Returns array or arrays if input array, else tensor of vector of tensors if input tensor.

::

   q)x:1 1 1 2 2 0 0 0 1 1 1 3

   q)uniquec x
   1 2 0 1 3

   q)uniquec(x;0b;1b)
   1 2 0 1 3
   3 2 3 3 1

   q)show x:(x;asc x)
   1 1 1 2 2 0 0 0 1 1 1 3
   0 0 0 1 1 1 1 1 1 2 2 3

   q)uniquec(x;1)
   1 2 0 1 1 3
   0 1 1 1 2 3

   q)x:tensor x
   q)v:uniquec(x;1;1b;1b)
   q)size v
   2 6
   ,12
   ,6

   q)vector v
   (1 2 0 1 1 3;0 1 1 1 2 3)
   0 0 0 1 1 2 2 2 3 4 4 5
   3 2 3 1 2 1


dist
^^^^

The PyTorch `dist <https://pytorch.org/docs/stable/generated/torch.dist.html>`_ function calculates the p-norm of the difference of two inputs:

.. function:: dist(x;y) -> norm of x minus y
.. function:: dist(x;y;p) -> p-norm of x minus y

   :param array,tensor x: first input array or tensor :doc:`pointer <pointers>`
   :param array,tensor y: second input array or tensor :doc:`pointer <pointers>`
   :param double p: the optional type of norm, default is 2
   :return: returns the norm of x minus y as a k scalar if k inputs, else tensor if any input is a tensor.

::

   q)x:1 2 3e
   q)y:0 3 9e

   q)dist(x;y)
   6.164414e

   q)dist(x;y;3)
   6.018462e

   q){xexp[sum abs[x]xexp y]1%y}[x-y]'[2 3]
   6.164414 6.018462

   q)x:tensor x
   q)y:tensor y
   q)z:dist(x;y)
   q)tensor z
   6.164414e

   
logsumexp
^^^^^^^^^

.. function:: logsumexp(x;dim;keepdim) -> log of summed exponentials
.. function:: logsumexp(x;dim;keepdim;output) -> null

   | Allowable argument combinations:

    - ``logsumexp(x;dim)``
    - ``logsumexp(x;dim;keepdim)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param long dim: the dimension(s) along which to calculate the log of sums of exponentials
   :param bool keepdim: default ``false``, set ``true`` to preserve the dimension of the input for the log of sums of exponentials
   :param tensor output: an optional tensor :doc:`pointer <pointers>` to use for function output
   :return: Returns log of the overall sum(s) of exponentials as a k array if array input else returns tensor if tensor input.  If ``output`` tensor supplied, result is written to given tensor, null return.

::


   q)show x:2 3#-1 0 1 2 .5
   -1 0   1 
   2  0.5 -1

   q)logsumexp x
   'logsumexp: needs explicit dimension(s) for log of summed exponentials
     [0]  logsumexp x
          ^
   q)logsumexp(x;0 1)
   2.6

   q)log sum exp raze x
   2.6

   q)logsumexp(x;1)
   1.41 2.24

   q)log sum each exp x
   1.41 2.24

   q)logsumexp(x;0)
   2.05 0.974 1.13

   q)log sum exp x
   2.05 0.974 1.13

   q)x:tensor x
   q)y:logsumexp(x;0)
   q)tensor y
   2.05 0.974 1.13

.. _normfns:

Norm calculations
^^^^^^^^^^^^^^^^^

The PyTorch `norm <https://pytorch.org/docs/stable/generated/torch.norm.html>`_ function is implemented as separate c++ functions :func:`fnorm`, :func:`nnorm` and :func:`pnorm`.

fnorm
*****

Function :func:`fnorm` calculates the Frobenius norm of a matrix or set of matrices, defined as the square root of the sum of squared elements of the matrix.

.. function:: fnorm(x;dim;keepdim) -> Frobenius norm
.. function:: fnorm(x;dim;keepdim;output) -> null

   | Allowable argument combinations:

    - ``fnorm(x)``
    - ``fnorm(x;dim)``
    - ``fnorm(x;dim;keepdim)``
    - ``fnorm(x;keepdim)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param longs dim: the optional dimension(s) along which to calculate the norm
   :param bool keepdim: default ``false``, set ``true`` to preserve the dimension of the input
   :param tensor output: an optional tensor :doc:`pointer <pointers>` to use for function output

::

   q)x:2 2#1 2 3 4.0
   q)fnorm x
   5.477226

   q)sqrt sum raze x*x
   5.477226

   q)x:3 2 2#1 2 3 4.0
   q)fnorm(x;1 2)
   5.477226 5.477226 5.477226

   q)r:tensor 0#0n
   q)fnorm(x;1 2;r)
   q)tensor r
   5.477226 5.477226 5.477226

   q)x:tensor x
   q)y:fnorm(x;1 2;1b)
   q)tensor y
   5.477226
   5.477226
   5.477226

nnorm
*****

Function :func:`nnorm` calculates the nuclear norm of a matrix, i.e. the trace norm, the sum of singular values of a matrix.

.. function:: nnorm(x;keepdim) -> nuclear norm
.. function:: nnorm(x;keepdim;output) -> null


   | Allowable argument combinations:

    - ``nnorm(x)``
    - ``nnorm(x;keepdim)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: 2-d array or tensor :doc:`pointer <pointers>` input
   :param bool keepdim: default ``false``, set ``true`` to preserve the dimension of the input
   :param tensor output: an optional tensor :doc:`pointer <pointers>` to use for function output

::

   q)x:2 2#1 2 3 4.0

   q)nnorm x
   5.830952

   q)sums svd[x]1     /singular values from SVD 
   5.464986 5.830952

   q)r:tensor 0#0n  /output tensor, keepdim
   q)nnorm(x;1b;r)
   q)tensor r
   5.830952
   q)size r
   1 1

pnorm
*****

.. function:: pnorm(x;p;dim;keepdim) -> p-norm
.. function:: pnorm(x;p;dim;keepdim;output) -> null

   | Allowable argument combinations:

    - ``pnorm(x)``
    - ``pnorm(x;p)``
    - ``pnorm(x;p;dim)``
    - ``pnorm(x;p;dim;keepdim)``
    - ``pnorm(x;dim)``
    - ``pnorm(x;dim;keepdim)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param double p: the optional order of the norm, default is 2.0. ``p=0w`` returns max absolute value of the elements, ``p=-0w`` calculates the min absolute value of the elements.
   :param longs dim: the optional dimension(s) along which to calculate the norm
   :param bool keepdim: default ``false``, set ``true`` to preserve the dimension of the input
   :param tensor output: an optional tensor :doc:`pointer <pointers>` to use for function output
   :return: Calculates the norm over the full input or the dimension(s) indicated, returns tensor if tensor input, else array returned. If output tensor given as final argument, result is written to this tensor and null return.

.. note::

   Since both ``p`` and a single dimension are possible arguments, ``p`` must be given as a double to distinguish from a single dimension argument.

::

   q)x:2 2#1 2 3 4.0

   q)pnorm x
   5.477226

   q)sqrt sum raze x*x
   5.477226

   q)pnorm(x;-0w)
   1f
   q)pnorm(x;0w)
   4f
   q)(min;max)@\:abs raze x
   1 4f

   q)pnorm(x;1.0) /taxicab norm
   10f
   q)sum raze abs x
   10f

   q)pnorm(x;1) /2nd arg interpreted as dimension
   2.236068 5
   q)sqrt sum each x*x
   2.236068 5

   q)pnorm((x;10*x;100*x);1 2)
   5.477226 54.77226 547.7226


