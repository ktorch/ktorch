Random sampling
===============
`Random sampling <https://pytorch.org/docs/stable/torch.html#random-sampling>`_ in PyTorch is implemented by the functions here and in
the section on :ref:`parameter initialization <init>`.


bernoulli
^^^^^^^^^
`torch.bernoulli <https://pytorch.org/docs/stable/generated/torch.bernoulli.html>`_ is implemented by function :func:`bernoulli`.

Draws binary random numbers (0 or 1) from a Bernoulli distribution.

The input should contain probabilities to be used for drawing the binary random number,
with all values in the range: :math:`0 \leq \text{x}_i \leq 1`.

The :math:`\text{i}^{th}` element of the output will draw a
value :math:`1` according to the :math:`\text{i}^{th}` probability value given in the input.

.. math::
    \text{out}_{i} \sim \mathrm{Bernoulli}(p = \text{x}_{i})

.. function:: bernoulli(x) -> Bernoulli values
.. function:: bernoulli(x;output) -> null

   :param array,tensor x: a k array or tensor of floating point numbers between 0 and 1 indicating the probability of drawing a 1
   :param tensor output: an optional output tensor, may be of integral type
   :return: Returns an array if k array input, else tensor, of same shape as the input ``x`` with 0`s and 1`s drawn according to the porbaility in the input. If output tensor supplied, output is written to the tensor and null is returned.

::

   q)seed 0
   q)bernoulli 5 3# .1 .5 .99e
   0 0 1
   0 1 1
   0 0 1
   0 1 1
   1 1 1

   q)r:tensor 0#0b
   q)bernoulli(5 3# .1 .5 .99e;r)
   q)tensor r
   001b
   011b
   011b
   001b
   111b

   q)seed 0
   q)x:tensor 1 2 3 4%4
   q)y:bernoulli x
   q)tensor y
   0 0 1 1f


multinomial
^^^^^^^^^^^
`torch.multinomial <https://pytorch.org/docs/stable/generated/torch.multinomial.html>`_ is implemented by :func:`multinomial`,
which returns an array or tensor where each row contains indices sampled from the multinomial probability distribution located in the corresponding row of the input.


.. function:: multinomial(x;n;replace) -> indices
.. function:: multinomial(x;n;replace;output) -> null

   | Allowable argument combinations:

    - ``multinomial(x)``
    - ``multinomial(x;n)``
    - ``multinomial(x;n;replace)``
    - ``multinomial(x;replace)``
    - any of the above combinations followed by a trailing output tensor

   :param: array,tensor x: a 1-dim or 2-dim array or tensor with probabilities or weights
   :param: long n: optional, set to 1 if not specified, the number of samples (per row if 2-dim input)
   :param: bool replace: flag set ``false`` by default, set ``true`` to allow samples to be drawn with replacement
   :param: tensor output: an optional output tensor
   :return: Returns an list or array if input is a k list or array, else a tensor of indices sampled. If an output tensor supplied, the indices are written to the supplied tensor and null returned.

::

   q)seed 123

   q)multinomial .2 .6 .4
   1

   q)multinomial(.2 .6 .4;10;1b)
   0 1 1 1 1 2 0 2 0 1

   q)x:(.2 .6 .4; .5 .0 .5)

   q)multinomial x
   1 2

   q)multinomial(x;4;1b)
   1 1 1 2
   0 2 0 2

