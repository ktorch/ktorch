Comparison operators
====================

Boolean results
***************

These PyTorch `comparison <https://pytorch.org/docs/stable/torch.html#comparison-ops>`_ operators return boolean results from comparing values in a pair of inputs:

- `allclose <https://pytorch.org/docs/stable/generated/torch.allclose.html>`_ - checks if the values in two inputs are all within a comparison tolerance
- `close <https://pytorch.org/docs/stable/generated/torch.isclose.html>`_ - ``true`` for each element in the two inputs that are within tolerance
- `equal <https://pytorch.org/docs/stable/generated/torch.equal.html>`_ - checks if both supplied inputs have the same size and identical elements

- `eq <https://pytorch.org/docs/stable/generated/torch.eq.html>`_ - equal
- `ge <https://pytorch.org/docs/stable/generated/torch.ge.html>`_ - greater than or equal
- `gt <https://pytorch.org/docs/stable/generated/torch.gt.html>`_ - greater than
- `le <https://pytorch.org/docs/stable/generated/torch.le.html>`_ - less than or equal
- `lt <https://pytorch.org/docs/stable/generated/torch.lt.html>`_ - less than
- `ne <https://pytorch.org/docs/stable/generated/torch.ne.html>`_ - not equal

close & allclose
^^^^^^^^^^^^^^^^

The `close <https://pytorch.org/docs/stable/generated/torch.isclose.html>`_ function checks if the values in two inputs are within a comparison tolerance and returns boolean ``true`` for each element, else ``false`` whereas 
`allclose <https://pytorch.org/docs/stable/generated/torch.allclose.html>`_ returns
a single boolean ``true`` if all inputs are within a comparison tolerance, ``false`` else.
The functions allow relative and absolute comparison tolerances to be given as additional arguments, along with a flag for comparing ``NaN`` values.

.. function:: close(x;y;rtol;atol;nanflag) -> booolean scalar

   | Allowable argument combinations:

    - ``close(x;y)``
    - ``close(x;y;nanflag)``
    - ``close(x;y;rtol)``
    - ``close(x;y;rtol;atol)``
    - ``close(x;y;rtol;atol;nanflag)``

   :param array,tensor x: 1st required input array or tensor :doc:`pointer <pointers>`
   :param array,tensor y: 2nd required input array or tensor :doc:`pointer <pointers>`
   :param double rtol: relative tolerance, default is ``1e-05``
   :param double atol: absolute tolerance, default is ``1e-08``
   :param boolean nanflag: set ``true`` to consider ``NaN`` values as equal, default is ``false``
   :return: The function returns a k boolean scalar if :math:`∣x-y∣≤atol+rtol×∣y∣`.

::

   q)close(1 2 3e;1 2.00001 3.0001e)
   110b

   q)close(1 2 3e;1 2.00001 3.0001e;1e-4)
   111b

   q)allclose(1 2 3e;1.000001 2.000001 2.99999e)
   1b

   q)allclose(1 1 1f;.999999999)
   1b

equal
^^^^^

The `equal <https://pytorch.org/docs/stable/generated/torch.equal.html>`_ function returns ``true`` if both supplied inputs have the same size and identical elements, else ``false``.

.. function:: equal(x;y) -> boolean scalar

   :param array,tensor x: 1st required input array or tensor :doc:`pointer <pointers>`
   :param array,tensor y: 2nd required input array or tensor :doc:`pointer <pointers>`
   :return: returns a k boolean scalar, ``true`` if ``x`` and ``y`` are both the same size and have identical elements, else ``false``.

::

   q)equal(2 3#til 6;til 6)
   0b
   q)equal(2 3#til 6;2 3#til 6)
   1b
   q)equal(2 3#til 6;1 2 3#til 6)
   0b

   q)x:tensor 1 2 3e
   q)y:tensor 1 2 3.0000001e
   q)equal(x;y)
   1b

   q)use[y]tensor 1 2 3.000001e
   q)equal(x;y)
   0b

ge, gt, le, lt, ne
^^^^^^^^^^^^^^^^^^

These `comparison <https://pytorch.org/docs/stable/torch.html#comparison-ops>`_ operators accept two inputs and an optional output tensor:

- `eq <https://pytorch.org/docs/stable/generated/torch.eq.html>`_ - equal
- `ge <https://pytorch.org/docs/stable/generated/torch.ge.html>`_ - greater than or equal
- `gt <https://pytorch.org/docs/stable/generated/torch.gt.html>`_ - greater than
- `le <https://pytorch.org/docs/stable/generated/torch.le.html>`_ - less than or equal
- `lt <https://pytorch.org/docs/stable/generated/torch.lt.html>`_ - less than
- `ne <https://pytorch.org/docs/stable/generated/torch.ne.html>`_ - not equal

.. function:: comparison(x;y) -> k array or tensor
.. function:: comparison(x;y;output) -> null

   :param array,tensor x: 1st required input array or tensor :doc:`pointer <pointers>`
   :param array,tensor y: 2nd required input array or tensor :doc:`pointer <pointers>`
   :param pointer output: an :doc:`api-pointer <pointers>` to a previously allocated tensor to be used for output
   :return: The function returns a k array if both inputs given as k arrays and otherwise returns a tensor.  If an output tensor is supplied, this tensor is filled with the output values and null is returned.  A special case of a 3rd argument of unary null is interpreted as an in-place operation, with the output values overwriting the previous values in the first input tensor.

::

   q)eq(1 2 3;4 3 3)
   001b

   q)x:tensor 1 2 3
   q)y:tensor 1 5 1

   q)z:ne(x;y)
   q)tensor z
   011b

   q)ne(x;y;z)
   q)tensor z
   011b

   q)ne(x;y;[])
   q)tensor x
   0 1 1


Special values
**************

These `comparison <https://pytorch.org/docs/stable/torch.html#comparison-ops>`_ functions return a tensor/array of booleans, one per input element.
The k api function name removes the ``is`` prefix, e.g. ``torch.isnan`` becomes ``nan``.

- `finite <https://pytorch.org/docs/stable/generated/torch.isfinite.html>`_ - returns ``true`` for each element that is finite
- `inf <https://pytorch.org/docs/stable/generated/torch.isinf.html>`_ - returns ``true`` if element is positive or negative infinity
- `nan <https://pytorch.org/docs/stable/generated/torch.isnan.html>`_ - returns ``true`` for each element that is ``NaN``
- `neginf <https://pytorch.org/docs/stable/generated/torch.isneginf.html>`_ - returns ``true`` for each element that is negative infinity
- `posinf <https://pytorch.org/docs/stable/generated/torch.isposinf.html>`_ - returns ``true`` for each element that is positive infinity

.. function:: special(input) -> boolean array or boolean tensor

   :param array,tensor input: a k array or tensor :doc:`pointer <pointers>`
   :return: Returns a boolean array for each element in given input array, or boolean tensor if input is also a tensor, with ``true`` if element matches a special value.

::

   q)x!(x:`finite`inf`neginf`posinf`nan){x y}\:0 0n -0w 0w
   finite| 1000b
   inf   | 0011b
   neginf| 0010b
   posinf| 0001b
   nan   | 0100b

   q)x:tensor 0 0n -0w 0we

   q)dtype x
   `float

   q)y:inf x
   q)tensor y
   0011b


Min / Max compare
*****************

These comparison functions find the maximum or minimum values comparing the elements of two inputs.
Function :func:`fmax` and :func:`fmin` are similar to :func:`maximum` and :func:`minimum` except in the handling of ``NaN``:
:func:`fmax`/:func:`fmin` will pick the ``non-NaN`` value if comparing a mix of ``NaN`` and ``non-NaN`` elements
while :func:`maximum`/:func:`minimum` will use ``NaN``.

- `fmax <https://pytorch.org/docs/stable/generated/torch.fmax.html>`_ - compares 2 inputs and returns the maximum for each element
- `fmin <https://pytorch.org/docs/stable/generated/torch.fmin.html>`_ - compares 2 inputs and returns the minimum for each element
- `maximum <https://pytorch.org/docs/stable/generated/torch.maximum.html>`_ - compares 2 inputs and returns the maximum for each element
- `minimum <https://pytorch.org/docs/stable/generated/torch.minimum.html>`_ - compares 2 inputs and returns the minimum for each element

fmax / fmin
^^^^^^^^^^^

.. function:: fmax(x;y) -> tensor or k array with maximum values
.. function:: fmax(x;y;output) -> null

   :param array,tensor x: 1st required input array or tensor :doc:`pointer <pointers>`
   :param array,tensor y: 2nd required input array or tensor :doc:`pointer <pointers>`
   :param tensor output: an optional tensor :doc:`pointer <pointers>` to use for function output
   :return: Returns tensor with the maximum element across the the two inputs, returns a k array if both inputs are k arrays. Null return if output tensor specified, maximum elements are written to output tensor.

.. function:: fmin(x;y) -> tensor or k array with minimum values
.. function:: fmin(x;y;output) -> null

   :param: Function :func:`fmin` uses the same parameters as :func:`fmax`
   :return: Returns tensor with the minimum element across the the two inputs, returns a k array if both inputs are k arrays. Null return if output tensor specified, minimum elements are written to output tensor.

::

   q)fmin(1 2 0n;1 2 3)
   1 2 3f

   q)maximum(1 2 0n;1 2 3)
   1 2 0n

   q)r:tensor 0#0n
   q)maximum(1 2 0n;1 2 3;r)
   q)tensor r
   1 2 0n

   q)x:tensor 1 2 3e
   q)y:tensor 1 1.999999 3.000001
   q)z:fmin(x;y)
   q)tensor z
   1 1.999999 3

maximum / minimum
^^^^^^^^^^^^^^^^^

.. function:: maximum(x;y) -> tensor or k array with maximum values
.. function:: maximum(x;y;output) -> null

   :param array,tensor x: 1st required input array or tensor :doc:`pointer <pointers>`
   :param array,tensor y: 2nd required input array or tensor :doc:`pointer <pointers>`
   :param tensor output: an optional tensor :doc:`pointer <pointers>` to use for function output
   :return: Returns tensor with the maximum element across the the two inputs, returns a k array if both inputs are k arrays. Null return if output tensor specified, maximum elements are written to output tensor. Any ``NaN`` encountered will be returned in the comparison.

.. function:: minimum(x;y) -> tensor or k array with minimum values
.. function:: minimum(x;y;output) -> null

   :param: Function :func:`minimum` uses the same parameters as :func:`maximum`
   :return: Returns tensor with the minimum element across the the two inputs, returns a k array if both inputs are k arrays. Null return if output tensor specified, minimum elements are written to output tensor. Any ``NaN`` encountered will be returned in the comparison.

::

   q)x:tensor 1 2 3e
   q)y:tensor 1 1.999999 3.000001e
   q)z:minimum(x;y)

   q)tensor z
   1 1.999999 3e

Sorting
*******

- `sort <https://pytorch.org/docs/stable/generated/torch.sort.html>`_  - returns a sorted array or tensor and the indices used to sort.
- `msort <https://pytorch.org/docs/stable/generated/torch.msort.html>`_ - sorts the elements of the input along the first dimension
- `argsort <https://pytorch.org/docs/stable/generated/torch.argsort.html>`_  - returns the indices to sort the input
- `topk <https://pytorch.org/docs/stable/generated/torch.topk.html>`_ - returns ``k`` largest/smallest elements and their indices
- `kthvalue <https://pytorch.org/docs/stable/generated/torch.kthvalue.html>`_ - return ``k-th`` smallest elements and their indices
- `In <https://pytorch.org/docs/stable/generated/torch.isin.html>`_ - test if input elements in set

sort
^^^^

.. function:: sort(x;dim;descend;stable) -> values and indices
.. function:: sort(x;dim;descend;stable;output) -> null

   | Allowable argument combinations:

    - ``sort(x)``
    - ``sort(x;descend)``
    - ``sort(x;descend;stable)``
    - ``sort(x;dim)``
    - ``sort(x;dim;descend)``
    - ``sort(x;dim;descend;stable)``
    - any of the above combinations followed by a trailing output vector

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param long dim: the optional dimension on which to sort, default is last dimension
   :param bool descend: default ``false``, set ``true`` to return indices in descending order
   :param bool stable: default ``false``, set ``true`` to guarentee the order of equivalent elements
   :param vector output: a vector `pointer <vectors>`
   :return: Returns sorted values and indices as k arrays or tensors depending on form of input. If output vector supplied, values and indices are written to the vector and null returned.

::

   q)sort 5 2 9
   2 5 9
   1 0 2

   q)sort(5 2 9;1b) / descending
   9 5 2
   2 0 1

Using :func:`sort` with a 2-d tensor:

::

   q)x:tensor(`randn;5 2)
   q)tensor x
   -0.198 0.882
   0.858  0.628
   -0.399 0.72
   -0.329 -0.382
   2.14   -1.49

   q)v:sort(x;0)
   q)size v
   5 2
   5 2

   q)vector(v;0)
   -0.399 -1.49
   -0.329 -0.382
   -0.198 0.628
   0.858  0.72
   2.14   0.882

   q)vector(v;1)
   2 4
   3 3
   0 1
   1 2
   4 0

   q)sort(x;0;1b;v)  /descending, re-using vector
   q)vector(v;0)
   2.14   0.882
   0.858  0.72
   -0.198 0.628
   -0.329 -0.382
   -0.399 -1.49


msort
^^^^^

Function  :func:`msort` sorts the elements of the input along the first dimension, e.g. sorts the rows of a matrix in ascending order.

.. function:: msort(x) -> sorted array or tensor
.. function:: msort(x;output) -> null

   :param array,tensor x: required input array or tensor :doc:`pointer <pointers>`.
   :param pointer output: an optional :doc:`pointer <pointers>` to a previously allocated tensor to be used for output.
   :return: The function returns a k array if given a k array as input and returns a tensor if a tensor is given as the input argument.  The output is sorted on the first dimension. If an output tensor is supplied, this tensor is filled with the output values and null is returned.

::

   q)x:tensor(`randn;3 2)
   q)tensor x
   1.8    -0.534
   -0.177 -0.0571
   -0.692 -1.36

   q)msort tensor x
   -0.692 -1.36
   -0.177 -0.534
   1.8    -0.0571

   q)y:msort x
   q)tensor y
   -0.692 -1.36
   -0.177 -0.534
   1.8    -0.0571

   q)msort(neg tensor x;y)
   q)tensor y
   -1.8  0.0571
   0.177 0.534
   0.692 1.36


argsort
^^^^^^^

The :func:`argsort` function returns the indices needed to sort along a given dimension (default is last dimension if non specified).
By default the indices specify an ascending order, a flag can be set ``true`` to return indices in descending order.

.. function:: argsort(x;dim;descend) -> indices

   | Allowable argument combinations:

    - ``argsort(x;descend)``
    - ``argsort(x;dim)``
    - ``argsort(x;dim;descend)``

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param long dim: the optional dimension on which to sort, default is last dimension
   :param bool descend: default ``false``, set ``true`` to return indices in descending order
   :return: If a k array is given, an array of indices is returned, else a tensor :doc:`pointer <pointers>`

::

   q)x:tensor(`randn;2 3)
   q)tensor x
   -0.628 -0.162 1.19
    1.15   0.677 0.626

   q)argsort(tensor x)
   0 1 2
   2 1 0

   q)y:argsort(x;0;1b)
   q)tensor y
   1 1 0
   0 0 1

topk
^^^^

Function :func:`topk` returns the largest/smallest values of given input along with their indices.
The sorting is done along the last dimension unless a different dimension is specified.

.. function:: topk(x;k;dim;largest;sorted) -> values and indices
.. function:: topk(x;k;dim;largest;sorted;output) -> null

   | Allowable argument combinations:

    - ``topk(x;k)``
    - ``topk(x;k;dim)``
    - ``topk(x;k;dim;largest)``
    - ``topk(x;k;dim;largest;sorted)``
    - ``topk(x;k;largest)``
    - ``topk(x;k;largest;sorted)``
    - any of the above argument combinations with a trailing output vector

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param long k: the number of values to find, required (cannot exceed input size in operating dimension)
   :param long dim: the optional dimension on which to sort, default is last dimension
   :param bool largest: default ``true``, set ``false`` to find smallest values and indices
   :param bool sorted: default ``true``, return elements in sorted order, set ``false`` for no defined order
   :param vector output: a vector `pointer <vectors>`
   :return: If a k array is given, an array of indices is returned, else a tensor :doc:`pointer <pointers>`

::

   q)topk(100.1+til 7; 3)
   106.1 105.1 104.1
   6     5     4

   q)x:tensor 100.1+til 7
   q)v:topk(x; 3; 0b)
   q)vector v
   100.1 101.1 102.1
   0     1     2

An example using an output vector with a 2-d input and operating over the first dimension:

::

   q)x:tensor 4 2#100.1+til 8
   q)v:vector()
   q)topk(x;2;0;v)

   q)vector(v;0)
   106.1 107.1
   104.1 105.1

   q)vector(v;1)
   3 3
   2 2

kthvalue
^^^^^^^^

Function :func:`kthvalue` returns the k-th smallest value and index over the last or specified dimension of input.

.. function:: kthvalue(x;k;dim;keepdim) -> values and indices
.. function:: kthvalue(x;k;dim;keepdim;output) -> null

   | Allowable argument combinations:

    - ``kthvalue(x;k)``
    - ``kthvalue(x;k;dim)``
    - ``kthvalue(x;k;dim;keepdim)``
    - ``kthvalue(x;k;keepdim)``
    - any of the above argument combinations with a trailing output vector

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param long k: k for the k-th smallest element
   :param long dim: the optional dimension on which to sort, default is last dimension
   :param bool keepdim: default ``false``, set ``true`` to preserve the dimension of the input for the minimum and maximum values.
   :param vector output: a vector `pointer <vectors>`
   :return: Returns k-th smallest values and indices as k arrays or tensors depending on form of input. If output vector supplied, values and indices are written to the vector and null returned.

::

   q)kthvalue(1 2 3 4 5;2)
   2 1

   q)kthvalue(1 2 3 4 5;3)
   3 2

   q)show x:5 3#1.1+til 15
   1.1  2.1  3.1
   4.1  5.1  6.1
   7.1  8.1  9.1
   10.1 11.1 12.1
   13.1 14.1 15.1

   q)kthvalue(x;3;1)  / 3rd smallest across columns
   3.1 6.1 9.1 12.1 15.1
   2   2   2   2    2

   q)kthvalue(x;3;0)  / 3rd smallest across rows
   7.1 8.1 9.1
   2   2   2

   q)x:tensor x
   q)v:vector()
   q)kthvalue(x;3;0;v)
   q)vector v
   7.1 8.1 9.1
   2   2   2

In
^^

Function :func:`In`, renamed from PyTorch's `isin <https://pytorch.org/docs/stable/generated/torch.isin.html>`_ tests if first input is in the second input -- either input can be a scalar, but not both inputs.

.. function:: In(x;y;unique;invert) -> boolean true for each element of x in y
.. function:: In(x;y;unique;invert;output) -> null

   | Allowable argument combinations:

    - ``In(x;y)``
    - ``In(x;y;unique)``
    - ``In(x;y;unique;invert)``
    - any of the above argument combinations with a trailing output tensor

   :param scalar,array,tensor x: input scalar, array or tensor :doc:`pointer <pointers>`.
   :param scalar,array,tensor y: input scalar, array or tensor :doc:`pointer <pointers>`.
   :param bool unique: optional, default ``false``, set ``true`` if both inputs have unique elements.
   :param bool invert: optional, default ``false``, set ``true`` to return the opposite of ``x`` in ``y``.
   :param tensor output: output tensor :doc:`pointer <pointers>`, must have boolean datatype.
   :return: If either ``x`` or ``y`` is a tensor, returns a boolean tensor, else returns k scalar/array indicating if ``x`` in ``y``. If output tensor supplied, results are written to the supplied tensor, null return.

::

   q)In(1 2 3;3 4 5 6)
   001b

   q)In(1 2 3;3 4 5 6;1b;1b)  /turn on unique & invert result
   110b

   q)x:tensor(`arange;5)
   q)r:In(5 9 1 0;x)
   q)tensor r
   0011b

   q)In(5 9 1 0;x;1b;1b;r)  /unique,invert w'output tensor
   q)tensor r
   1100b

