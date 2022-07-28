Pointwise operations
====================

PyTorch `pointwise <https://pytorch.org/docs/stable/torch.html#pointwise-ops>`_ operations apply to each element of the input(s). 

Single argument
***************

The following pointwise operations take a single input, along with an optional output tensor for the result.
If the function name conficts with a k/q function, the first letter is capitalized, e.g. ``torch.abs`` becomes ``Abs`` in the k api.

- `Abs <https://pytorch.org/docs/stable/generated/torch.abs.html>`_ - absolute value
- `Acos <https://pytorch.org/docs/stable/generated/torch.acos.html>`_ - arccosine
- `angle <https://pytorch.org/docs/stable/generated/torch.angle.html>`_ - angle
- `Asin <https://pytorch.org/docs/stable/generated/torch.asin.html>`_ - arcsine
- `Atan <https://pytorch.org/docs/stable/generated/torch.atan.html>`_ - arctangent
- `bitwisenot <https://pytorch.org/docs/stable/generated/torch.bitwise_not.html>`_ - bitwise not
- `ceil <https://pytorch.org/docs/stable/generated/torch.ceil.html>`_ - ceiling
- `cosh <https://pytorch.org/docs/stable/generated/torch.cosh.html>`_ - hyperbolic cosine
- `Cos <https://pytorch.org/docs/stable/generated/torch.cos.html>`_ - cosine
- `digamma <https://pytorch.org/docs/stable/generated/torch.digamma.html>`_ - log derivative of gamma
- `erf <https://pytorch.org/docs/stable/generated/torch.erf.html>`_ - error function
- `erfc <https://pytorch.org/docs/stable/generated/torch.erfc.html>`_ - complimentary error function
- `erfinv <https://pytorch.org/docs/stable/generated/torch.erfinv.html>`_ - inverse error function
- `Exp <https://pytorch.org/docs/stable/generated/torch.exp.html>`_ - exponential
- `expm1 <https://pytorch.org/docs/stable/generated/torch.expm1.html>`_ - exponential minus 1
- `Floor <https://pytorch.org/docs/stable/generated/torch.floor.html>`_ - floor
- `frac <https://pytorch.org/docs/stable/generated/torch.frac.html>`_ - fractional
- `lgamma <https://pytorch.org/docs/stable/generated/torch.lgamma.html>`_ - natural log of the absolute value of the gamma function
- `Log <https://pytorch.org/docs/stable/generated/torch.log.html>`_ - log
- `log10 <https://pytorch.org/docs/stable/generated/torch.log10.html>`_ - log to the base 10
- `log1p <https://pytorch.org/docs/stable/generated/torch.log1p.html>`_ - natural log of ``1 + input``
- `log2 <https://pytorch.org/docs/stable/generated/torch.log2.html>`_ - log to the base 2
- `Neg <https://pytorch.org/docs/stable/generated/torch.neg.html>`_ - negative
- `Not <https://pytorch.org/docs/stable/generated/torch.logical_not.html>`_ - logical not
- `Reciprocal <https://pytorch.org/docs/stable/generated/torch.reciprocal.html>`_ - reciprocal
- `round <https://pytorch.org/docs/stable/generated/torch.round.html>`_ - round
- `rsqrt <https://pytorch.org/docs/stable/generated/torch.rsqrt.html>`_ - reciprocal square root
- `sgn <https://pytorch.org/docs/stable/generated/torch.sgn.html>`_ - signs of input elements for complex tensors
- `sigmoid <https://pytorch.org/docs/stable/generated/torch.sigmoid.html>`_ - sigmoid
- `sign <https://pytorch.org/docs/stable/generated/torch.sign.html>`_ - sign
- `Sin <https://pytorch.org/docs/stable/generated/torch.sin.html>`_ - sine
- `sinh <https://pytorch.org/docs/stable/generated/torch.sinh.html>`_ - hyperbolic sine
- `Sqrt <https://pytorch.org/docs/stable/generated/torch.sqrt.html>`_ - square root
- `Tan <https://pytorch.org/docs/stable/generated/torch.tan.html>`_ - tangent
- `tanh <https://pytorch.org/docs/stable/generated/torch.tanh.html>`_ - hyperbolic tangent
- `trunc <https://pytorch.org/docs/stable/generated/torch.trunc.html>`_ - truncated whole numbers

The functions all have the same calling syntax:

.. function:: pointwise(x) -> k array
.. function:: pointwise(x;output) -> null
.. function:: pointwise(x;null) -> null

   :param array,tensor x: required input array or tensor :doc:`pointer <pointers>`.
   :param pointer output: an optional :doc:`pointer <pointers>` to a previously allocated tensor to be used for output. A k unary null is interpreted as an in-place operation.
   :return: The function returns a k array if given a k array as input and returns a tensor if a tensor is given as the input argument.  If an output tensor is supplied, this tensor is filled with the output values and null is returned.  A special case of a 2nd argument of unary null is interpreted as an in-place operation, with the output values overwriting the previous values in the input tensor.

::

   q)Abs -2 1 0
   2 1 0

   q)x:tensor -2 1 0
   q)y:Abs x
   q)tensor y
   2 1 0

Using an output tensor:

::

   q)Abs(-3 3 2 -1; y)
   [W Resize.cpp:24] Warning: An output with one or more elements was resized since it had shape [3], which does not match the required output shape [4].This behavior is deprecated, and in a future PyTorch release outputs will not be resized unless they have zero elements. You can explicitly reuse an out tensor t by resizing it, inplace, to zero elements with t.resize_(0). (function resize_output_check)

   q)tensor y
   3 3 2 1

   q)use[y]0#0 / make empty long list
   q)Abs(-3 3 2 -1; y)
   q)tensor y
   3 3 2 1

Using null in place of an output tensor runs the function as an in-place operation:

::

   q)Neg(y;[])
   q)tensor y
   -3 -3 -2 -1

Two arguments
*************

The following pointwise functions accept two separate arguments and an optional output tensor:

- `atan2 <https://pytorch.org/docs/stable/generated/torch.atan2.html>`_ - arctangent 2
- `Div <https://pytorch.org/docs/stable/generated/torch.div.html>`_ - divide
- `fmod <https://pytorch.org/docs/stable/generated/torch.fmod.html>`_ - floating point remainder
- `fpow <https://pytorch.org/docs/stable/generated/torch.float_power.html>`_ = double precision power function
- `mul <https://pytorch.org/docs/stable/generated/torch.mul.html>`_ - multiply
- `pow <https://pytorch.org/docs/stable/generated/torch.pow.html>`_ = power function
- `remainder <https://pytorch.org/docs/stable/generated/torch.remainder.html>`_ - remainder (modulus)
- `xor <https://pytorch.org/docs/stable/generated/torch.logical_xor.html>`_ - logical xor

.. function:: pointwise(x;y) -> k array or tensor
.. function:: pointwise(x;y;output) -> null

   :param array,tensor x: 1st required input array or tensor :doc:`pointer <pointers>`
   :param array,tensor y: 2nd required input array or tensor :doc:`pointer <pointers>`
   :param pointer output: a :doc:`pointer <pointers>` to a previously allocated tensor to be used for output
   :return: The function returns a k array if both inputs given as k arrays and otherwise returns a tensor.  If an output tensor is supplied, this tensor is filled with the output values and null is returned.  A special case of a 3rd argument of unary null is interpreted as an in-place operation, with the output values overwriting the previous values in the first input tensor.

::

   q)mul(1 4 9;1 2 3)
   1 8 27

   q)x:tensor 1 4 9.0
   q)y:tensor 1 2 3.0

   q)z:mul(x;y)
   q)tensor z
   1 8 27f

   q)Div(x;y;z)
   q)tensor z
   1 2 3f

   q)Div(x;y;[])
   q)tensor x
   1 2 3f

   q)pow(1 2 3;2)
   1 4 9

   q)pow(1 2 3;2.5)
   1 5.656854 15.58846e

   q)fpow(1 2 3;2.5)
   1 5.656854 15.58846

Other addition
**************

add
^^^
`torch.add <https://pytorch.org/docs/stable/generated/torch.add.html>`_  is implemented as k api function :func:`add`.

.. math::
   \text{{result}}_i = \text{{x}}_i + \text{{multiplier}} \times \text{{y}}_i

Supports `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ tensors, `type promotion <https://pytorch.org/docs/stable/tensor_attributes.html#type-promotion-doc>`_ and integer, floating-point and complex inputs.

.. function:: add(x;y;multiplier) -> x + multiplier * y
.. function:: add(x;y;multiplier;output) -> null

   :param array,tensor x: 1st required input array or tensor :doc:`pointer <pointers>`
   :param array,tensor y: 2nd required input array or tensor :doc:`pointer <pointers>`
   :param numeric multiplier: optional numeric scalar, default=1, used to multiply ``y`` before adding to ``x``
   :param pointer output: a :doc:`pointer <pointers>` to a previously allocated tensor to be used for output
   :return: Returns ``x + multiplier * y` as tensor if ``x`` or ``y`` given as tensor, else k array. If output tensor supplied, result is written to tensor with null return.

::

   q)add(1 2 3;4 5 6)
   5 7 9

   q)add(1 2 3;4 5 6;100)
   401 502 603

Using `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ tensors:

::

   q)seed 123
   q)x:tensor(`randn;1 3)
   q)y:tensor(`randn;3 1)
   q)r:add(x;y)

   q)tensor r
   -0.352 -0.12 -0.61
   -1.31  -1.08 -1.57
   0.0978 0.33  -0.16

   q)raze[tensor x]+\:/:raze tensor y
   -0.352 -0.12 -0.61
   -1.31  -1.08 -1.57
   0.0978 0.33  -0.16

   q)add(x;y;1000;r)   /output tensor w'multiplier
   q)tensor r
   -240.5 -240.3 -240.8
   -1197  -1197  -1197 
   209.2  209.4  208.9 

addcdiv
^^^^^^^
`torch.addcdiv <https://pytorch.org/docs/stable/generated/torch.addcdiv.html>`_ is implemented by k api function :func:`addcdiv`.

.. math::
    \text{result}_i = \text{x}_i + \text{multiplier} \times \frac{\text{y}_i}{\text{z}_i}

The shapes of ``x``, ``y`` and ``z`` must be `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_.

.. function:: addcdiv(x;y;z;multiplier) -> result of addition and division
.. function:: addcdiv(x;y;z;multiplier;output) -> null

   :param array,tensor x: 1st required input array or tensor :doc:`pointer <pointers>`
   :param array,tensor y: 2nd required input array or tensor :doc:`pointer <pointers>`
   :param array,tensor z: 3rd required input array or tensor :doc:`pointer <pointers>`
   :param numeric multiplier: optional numeric scalar, default=1, used to multiply ``y / z`` before adding to ``x``
   :param pointer output: a :doc:`pointer <pointers>` to a previously allocated tensor to be used for output
   :return: Returns ``x + multiplier * y / z`` as tensor if any of ``x``, ``y`` or ``z`` given as tensor, else k array. If output tensor supplied, result is written to tensor with null return.
   
::

   q)x:1 2 3.0
   q)y:9 16 25.0
   q)z:3  4  5.0

   q)addcdiv(x;y;z)
   4 6 8f

   q)x+y%z
   4 6 8f

Using `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ tensors:

::

   q)seed 123
   q)`x`y`z set'{tensor(`randn;x;`double)}'[(1 3;3 1;1 3)];
   q)r:addcdiv(x;y;z)

   q)tensor r
   0.136  0.439  -1.11
   1.12   1.71   -4.06
   -0.327 -0.157 0.276

   q)raze[tensor x]+/:tensor[y]mmu reciprocal tensor z
   0.136  0.439  -1.11
   1.12   1.71   -4.06
   -0.327 -0.157 0.276

   q)addcdiv(x;y;z;100;r)
   q)tensor r
   24.6  32    -74.6
   123   159   -370 
   -21.6 -27.6 64.2 

addcmul
^^^^^^^
`torch.addcmul <https://pytorch.org/docs/stable/generated/torch.addcmul.html>`_ is implemented by k api function :func:`addcmul`.

.. math::
    \text{result}_i = \text{x}_i + \text{multiplier} \times \text{y}_i \times \text{z}_i

The shapes of ``x``, ``y`` and ``z`` must be `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_.

.. function:: addcmul(x;y;z;multiplier) -> result of addition and division
.. function:: addcmul(x;y;z;multiplier;output) -> null

   :param array,tensor x: 1st required input array or tensor :doc:`pointer <pointers>`
   :param array,tensor y: 2nd required input array or tensor :doc:`pointer <pointers>`
   :param array,tensor z: 3rd required input array or tensor :doc:`pointer <pointers>`
   :param numeric multiplier: optional numeric scalar, default=1,  used to multiply ``y * z`` before adding to ``x``
   :param pointer output: a :doc:`pointer <pointers>` to a previously allocated tensor to be used for output
   :return: Returns ``x + multiplier * (y * z)`` as tensor if any of ``x``, ``y`` or ``z`` given as tensor, else k array. If output tensor supplied, result is written to tensor with null return.
   
::

   q)x:1 2 3
   q)y:3 4 5
   q)z:2 1 2

   q)x+y*z
   7 6 13

   q)addcmul(x;y;z)
   7 6 13

Using `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ tensors:

::

   q)seed 123
   q)`x`y`z set'{tensor(`randn;x;`double)}'[(1 3;3 1;1 3)];
   q)r:addcmul(x;y;z)
   q)tensor r
   0.122  0.302   -0.448
   1.05   1.02    -0.757
   -0.315 -0.0376 -0.302

   q)raze[tensor x]+/:tensor[y]mmu tensor z  /k equivalent calc
   0.122  0.302   -0.448
   1.05   1.02    -0.757
   -0.315 -0.0376 -0.302

   q)addcmul(x;y;z;100;r)  /use multiplier
   q)tensor r
   23.3  18.3  -8.16
   116   90.5  -39.1
   -20.5 -15.7 6.41 

clamp
*****
`torch.clamp <https://pytorch.org/docs/stable/generated/torch.clamp.html?highlight=clamp>`_ is implemented by function :func:`clamp`.

Clamps all inputs into the range `[` :attr:`lo`, :attr:`hi` `]`.

.. math::
   y_i = \min(\max(x_i, \text{lo}_i), \text{hi}_i)

If `lo` is null, there is no lower bound, and if `hi` is null there is no upper bound.

.. function:: clamp(x;lo;hi) -> input limited to min of lo and max of hi
.. function:: clamp(x;lo;hi;output) -> null

   :param array,tensor x:
   :param numeric lo: the minimum limit to be returned, can be set to a null scalar value, e.g. ``0N`` or ``0n`` to have no effect
   :param numeric hi: the maximum limit to be returned, can be set to a null scalar value, e.g. ``0N`` or ``0n`` to have no effect
   :param tensor output: optional output tensor
   :return: Returns clamped input as a k array if array input, else tensor. If output tensor supplied, writes clamped input to supplied tensor and returns null.

::

   q)x:-5 -1 0 1 9

   q)(x; clamp(x;-2;7))
   -5 -1 0 1 9
   -2 -1 0 1 7

   q)(x; clamp(x;0N;1))
   -5 -1 0 1 9
   -5 -1 0 1 1

   q)(x; clamp(x;-2;0N))
   -5 -1 0 1 9
   -2 -1 0 1 9

lerp
****
`torch.lerp <https://pytorch.org/docs/stable/generated/torch.lerp.html>`_ is implemented by function :func:`lerp`,
which returns a linear interpolation of two inputs based on a scalar or array/tensor :attr:`wt`.

.. math::
    \text{x}_i + \text{wt}_i \times (\text{y}_i - \text{x}_i)

The shapes of :attr:`x` and :attr:`y` must be
`broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics>`_.
If :attr:`wt` is not a scalar, then
the shapes of :attr:`wt`, :attr:`x`, and :attr:`y` must be `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html#broadcasting-semantics>`_.

.. function:: lerp(x;y;wt) -> interpolated output
.. function:: lerp(x;y;wt;output) -> null

   :param array,tensor x: 1st input, as k array or tensor
   :param array,tensor y: 2nd input, as k array or tensor
   :param scalar,array,tensor wt: 3rd input, may be scalar, k array or tensor
   :param tensor output: optional tensor to use for output
   :return: Returns the interpolation between ``x`` and ``y`` using weight(s) ``wt``.  Returns a tensor if any input is a tensor else an array. If an output tensor is supplied, the interpolation is written to the supplied tensor and null returned.

::

   q)x:1 2 3 4 5.0
   q)y:10*x
   q)w:.1

   q)lerp(x;y;w)
   1.9 3.8 5.7 7.6 9.5

   q)x+w*y-x
   1.9 3.8 5.7 7.6 9.5

   q)w:1%x
   q)lerp(x;y;w)
   10 11 12 13 14f

   q)x+w*y-x
   10 11 12 13 14f

   q)`x`y`w set'2 2#/:(x;y;w);
   q)lerp(x;y;w)
   10 11
   12 13


mvlgamma
********
`torch.special.imultigammaln <https://pytorch.org/docs/stable/special.html#torch.special.multigammaln>`_ is impemented by function :func:`mvlgamma`.

Computes the multivariate log-gamma function with dimension :math:`p` element-wise, given by

.. math::
    \log(\Gamma_{p}(x)) = C + \displaystyle \sum_{i=1}^{p} \log\left(\Gamma\left(x - \frac{i - 1}{2}\right)\right)

where :math:`C = \log(\pi) \times \frac{p (p - 1)}{4}` and :math:`\Gamma(\cdot)` is the Gamma function.

All elements must be greater than :math:`\frac{p - 1}{2}`, otherwise an error would be thrown.

.. function:: mvlgamma(x;p) -> multivariate log-gamma
.. function:: mvlgamma(x;p;output) -> null

   :param array,tensor x: input k array or tensor
   :param long p: the number of dimensions
   :param tensor output: optional output tensor
   :return: Returns the element-wise multivariate log-gamma function as an array if array input else tensor. If output tensor supplied, the function output iw written to the supplied tensor and null is returned.

::

   q)seed 123
   q)show x:uniform(2 3#0.0;1;2)
   1.369 1.013 1.592
   1.093 1.472 1.522

   q)mvlgamma(x;2)
   0.546  1.111  0.4124
   0.9353 0.4675 0.4403

