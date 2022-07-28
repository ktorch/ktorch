.. _init:

Initializing parameters
=======================

When modules are created, any parameters are given initial values: the default initializations can be overwritten
using other probability distributions or heuristics that make for more stable training or quicker convergence.
The k api implements the `PyTorch initialization routines <https://pytorch.org/docs/stable/nn.init.html>`_ and some of
the `probability distributions <https://pytorch.org/docs/stable/distributions.html>`_ used to reset initial parameter values.

- `zeros <https://pytorch.org/docs/stable/nn.init.html?highlight=torch%20nn%20init#torch.nn.init.zeros_>`_: reset tensor to zeros.
- `ones <https://pytorch.org/docs/stable/nn.init.html?highlight=torch%20nn%20init#torch.nn.init.ones_>`_: reset tensor to ones.
- `fill <https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.constant_>`_: fill tensor with a single value.
- `eye <https://pytorch.org/docs/stable/nn.init.html?highlight=torch%20nn%20init#torch.nn.init.eye_>`_: set 2d tensor to the identity matrix.
- `dirac <https://pytorch.org/docs/stable/nn.init.html?highlight=torch%20nn%20init#torch.nn.init.dirac_>`_: fill 3,4,5d tensor with the Dirac delta function.
- `orthogonal <https://pytorch.org/docs/stable/nn.init.html?highlight=torch%20nn%20init#torch.nn.init.orthogonal_>`_: fill tensor with a semi-orthogonal matrix.
- `knormal <https://pytorch.org/docs/stable/nn.init.html?highlight=torch%20nn%20init#torch.nn.init.kaiming_normal_>`_: Kaiming initialization using a normal distribution.
- `kuniform <https://pytorch.org/docs/stable/nn.init.html?highlight=torch%20nn%20init#torch.nn.init.kaiming_uniform_>`_: Kaiming initialization using a uniform distribution.
- `snormal <https://pytorch.org/docs/stable/nn.init.html?highlight=torch%20nn%20init#torch.nn.init.sparse_>`_: fill 2d matrix as sparse, with non-sparse elements form a normal distribution.
- `xnormal <https://pytorch.org/docs/stable/nn.init.html?highlight=torch%20nn%20init#torch.nn.init.xavier_normal_>`_: Xavier initialization using a normal distribution.
- `xuniform <https://pytorch.org/docs/stable/nn.init.html?highlight=torch%20nn%20init#torch.nn.init.xavier_uniform_>`_: Xavier initialization using a uniform distribution.

The normal & uniform probability distributions are included in the `module initialization group <https://pytorch.org/docs/stable/nn.init.html>`_ as well as part of `a broader group of distributions <https://pytorch.org/docs/stable/distributions.html>`_:

- `normal <https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.normal_>`_: fills tensor with values from the `normal distribution <https://pytorch.org/docs/stable/distributions.html#normal>`_, with optional mean & standard deviation.
- `uniform <https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.uniform_>`_: fills tensor with values drawn from the `uniform distribution <https://pytorch.org/docs/stable/distributions.html#uniform>`_ with optional lower & upper bounds.

Additional distributions implemented in the k-api:

- `cauchy <https://pytorch.org/docs/stable/distributions.html#cauchy>`_: samples from a Cauchy (Lorentz) distribution given median and half-width.
- `exponential <https://pytorch.org/docs/stable/distributions.html#exponential>`_: creates an exponential distribution parameterized by rate.
- `geometric <https://pytorch.org/docs/stable/distributions.html#geometric>`_: creates a geometric distribution given probability of success of Bernoulli trials.
- `lognormal <https://pytorch.org/docs/stable/distributions.html#lognormal>`_: creates a log-normal distribution given mean & standard deviation of log of the distribution.
- `random <https://pytorch.org/docs/stable/generated/torch.Tensor.random_.html>`_: fills tensor with numbers sampled from the discrete uniform distribution with optional low & high limits.

Utility to calculate the recommended :func:`gain` value for given nonlinearity:

- `gain <https://pytorch.org/docs/stable/nn.init.html#torch.nn.init.calculate_gain>`_: given non-linearity name, e.g. ``relu`` and optional parameter, returns factor used to scale standard deviation.


.. note::

   The above initialization routines reset existing tensors with gradient calculations disabled.

Syntax
^^^^^^

The initialization functions have a common syntax and accept a tensor or a collection of tensors, along with other arguments that typically override mean, standard deviation or some other property of the underlying distribution:

.. function:: fn(tensors) -> null

.. function:: fn(tensors; options..) -> null

   :param pointer tensors: an :doc:`api-pointer <pointers>` to an existing tensor or vector/dictionary of tensors.
   :param scalar options: typically numeric scalars, with symbol scalars used for Kaiming initialization.

   :return: The input tensor or collection of tensors are modified in place, null return.

::

   q)t:tensor 3 4#0e
   q)normal(t;2;.5)
   q)tensor t
   1.990077 1.99109  2.886662 1.551937
   2.027538 1.058909 1.429988 1.673985
   2.082229 3.062976 2.641823 2.190108

   q)d:dict `a`b!(10#0e;4 2#0)
   q)random(d;0;10)
   q)dict d
   a| 3 5 9 6 9 9 3 9 2 1e
   b| (7 8;3 4;0 1;6 5)

   q)v:vector(1 2 3e; 4 5#0e)
   q)uniform(v)
   q)vector v
   0.02381 0.2233 0.7237e
   (0.3632 0.4588 0.7514 0.4138 0.4747e;0.5428 0.1709 0.5703 0.04745 0.5007e;0.6..

Tensor/vector indices
^^^^^^^^^^^^^^^^^^^^^

Numeric indices (longs) can be used with a non-scalar tensor or a vector of tensors:

.. function:: fn(tensors; indices) -> null

.. function:: fn(tensors; indices; options..) -> null

   :param pointer tensors: an :doc:`api-pointer <pointers>` to an existing tensor or vector of tensors.
   :param long indices: a single index or set of indices into the first dimension of a tensor or a vector of tensors.
   :param scalar options: typically numeric scalars, with symbol scalars used for Kaiming initialization.

   :return: The input tensor or vector of tensors are modified in place using given indices, null return.

::

   q)t:tensor 3 4#0e
   q)normal(t;0 2)  / normal(0,1) distribution for first & final rows
   q)tensor t
   0.6594 -0.5249 1.596   -0.19  
   0      0       0       0      
   0.5576 0.6255  -0.2015 -0.6794

Using indices with a vector of tensors:

::

   q)v:vector(98 99 100;10#0)
   q)random(v;1;0;10)  / random integers over interval 0,10)
   q)vector v
   98 99 100
   3 5 5 0 7 5 2 7 5 6

.. note::

   There is some ambiguity in an argument list with a single index or an index and partially specified distribution options: the initial scalar(s) are interpreted as distribution options unless given as a 1-element list.

::

   q)v:vector(98 99 100.0; 10#.0; 5#.0)
   q)random(v;2;5)  / scalars 2 & 5 are used as the range for the random sample
   q)vector v
   4 2 4f
   3 4 4 2 2 3 2 2 3 4f
   3 2 2 2 2f

   q)v:vector(98 99 100.0; 10#.0; 5#.0)
   q)random(v; 1#2; 5) /index is enlisted to distinguish from option
   q)vector v
   98 99 100f
   0 0 0 0 0 0 0 0 0 0f
   2 4 0 2 1f

No ambiguity with a single index if all the distribution options are also specified:

::

   q)v:vector(98 99 100.0; 10#.0; 5#.0)
   q)random(v; 2; 0; 5)
   q)vector v
   98 99 100f
   0 0 0 0 0 0 0 0 0 0f
   3 3 4 0 3f


Tensor names
^^^^^^^^^^^^

Tensor names can be used to index a subset of a dictionary of tensors. Parameter or buffer names must be supplied if a module or model is given as the leading argument:

.. function:: fn(tensors; names) -> null

.. function:: fn(tensors; names; options..) -> null

   :param pointer tensors: an :doc:`api-pointer <pointers>` to an existing dictionary, module, model or optimizer.
   :param symbol names: keys into the given dictionary or names of parameters/buffers in the supplied module.
   :param scalar options: typically numeric scalars, with symbol scalars used for Kaiming initialization.

   :return: The named tensors, parameters or buffers are modified in place, null return.

::

   q)p:parms m:module enlist(`linear;2;2)
   q)dict p
   weight| 0.5732 -0.2588 0.4686 0.398  
   bias  | 0.4718         -0.6752       

   q)normal(p;`weight;0;.01)
   q)dict p
   weight| -0.01368 0.007652   -0.01319 -0.0006103
   bias  | 0.4718              -0.6752            

   q)zeros m  / modules require parameter or buffer names
   'zeros: not implemented for single module argument
     [0]  zeros m
          ^

   q)zeros(m;`bias)
   q)dict p
   weight| -0.01368 0.007652   -0.01319 -0.0006103
   bias  | 0                   0                  

.. note::

   If a module has both a parameter and a buffer with the same name, only the parameter will be reset. Access to the buffer in this case will have to be via functions :func:`buffer` or :func:`buffers`, which search only the buffer namespaces.

Kaiming initialization
^^^^^^^^^^^^^^^^^^^^^^
The Kaiming initialization functions, :func:`knormal` and :func:`kuniform` accept up to three options: the name of the non-linearity, the fan mode & slope of the rectifier (typically ```leakyrelu``).

.. function:: knormal(tensors) -> null
.. function:: kuniform(tensors) -> null

.. function:: knormal(tensors; nonlinearity; fanmode; slope) -> null
.. function:: kuniform(tensors; nonlinearity; fanmode; slope) -> null

   :param pointer tensors: an :doc:`api-pointer <pointers>` to an existing tensor, vector or dictionary of tensors.
   :param symbol nonlinearity: name of the non-linear function, e.g. ```relu`` or ```leakyrelu``, used to calculate standard deviation (normal distribution) or bounds (uniform distribution).
   :param symbol fanmode: one of ```fanin`` or ```fanout`` to preserve the magnitude of the variance of the weights in the forward (in) or backwards (out) pass.
   :param double slope: the negative slope of the rectifier used after this layer, e.g. for ```leakyrelu``.

   :return: The tensors are modified in place, null return.

.. note::

   The symbol and double scalar options may be given in any order following the initial tensor specification.

::

   q)t:tensor 3 4#0e
   q)kuniform(t)
   q)tensor t
   0.7619  -0.2598 -0.8482 -1.068
   0.4993  -0.8645 -0.2984 0.9116
   -0.8443 0.6993  0.4329  1.043 

   q)kuniform(t;`leakyrelu;`fanout)
   q)tensor t
   -0.2935 0.7956 -1.237  -0.6511
   -0.8029 1.043  -1.293  -0.9753
   0.3985  0.8391 -0.6392 -0.0994

The Kaiming initialization functions may also be used with indices as the 2nd argument:

.. function:: knormal(tensors; indices) -> null
.. function:: knormal(tensors; indices; options..) -> null

   :param pointer tensors: an :doc:`api-pointer <pointers>` to an existing tensor or vector of tensors; if tensor, indices select on 1st dimension.
   :param long indices: the index or indices into the vector or 1st dimension of a given tensor, enlist scalar index to avoid confusion with other numeric argument.

::

   q)t:tensor 2 3 4#0e
   q)kuniform(t;1#1)

   q)tensor(t;0)
   0 0 0 0
   0 0 0 0
   0 0 0 0

   q)tensor(t;1)
   -0.8719 -1.073  -1.144  -0.6039
   -0.3201 0.1402  -0.8489 -0.6861
   0.151   -0.9593 0.02821 -0.191 

The Kaiming initialization functions are also used with parameter/buffer names as the 2nd argument:

.. function:: knormal(tensors; names) -> null
.. function:: knormal(tensors; names; options..) -> null

   :param pointer tensors: an :doc:`api-pointer <pointers>` to an existing dictionary or module.
   :param symbol names: the name or names of dictionary tensors or module parameters/buffers, scalar names can be enlisted to avoid confusion with other scalar symbol arguments.

::

   q)m:module(`sequential; enlist(`linear;`fc;2;2); enlist(`leakyrelu;`fn;.01))
   q)p:parms m

   q)dict p
   fc.weight| 0.211  -0.5037 0.2513 0.03965
   fc.bias  | 0.08189        -0.04078      

   q)knormal(m;`fc.weight;`fanout;`relu;.01)

   q)dict p
   fc.weight| 1.063   0.9703 -0.1206 1.102 
   fc.bias  | 0.08189        -0.04078      


.. note:

   If there is a tensor name that is the same as a non-linearity or fan mode, the name will be used as option rather than a parameter key unless the name is given as a 1-element list.

::
   
   q)d:dict`fanout`relu!(2 3#0e;1 4#0e)
   q)kuniform(d;`fanout)
   q)dict d  / both tensors reset, name interpreted as option
   fanout| (0.8979 0.1735 1.717e;1.669 -0.468 1.427e)
   relu  | ,-1.817 1.026 2.206 0.0301e

   q)d:dict`fanout`relu!(2 3#0e;1 4#0e)
   q)kuniform(d;1#`fanout) / enlist to treat as key
   q)dict d
   fanout| (-0.2362 1.208 -0.4897e;-0.3338 -0.6536 -0.01446e)
   relu  | ,0 0 0 0e

Using k arrays
^^^^^^^^^^^^^^

The initialization routines also accept k arrays as input, returning k arrays after the initialization is applied:

.. function:: fn(input) -> output

.. function:: fn(input; options..) -> output

   :param k-array input: a scalar, list or n-dim array 
   :param scalar options: typically numeric scalars, with symbol scalars used for Kaiming initialization.

   :return: An output array of the same shape and type as input, with initialization applied.

::

   q)normal(3 4#0e)
   1.144   0.03057 0.9454  -0.3712
   -0.8005 0.4368  -0.2662 0.03962
   2.15    -0.503  1.133   -0.2594

   q)normal(3 4#0e;0;.01)
   -0.002944 -0.01436  0.002313 -0.005135
   -0.009178 -0.002622 0.01533  -0.01474 
   -0.002925 0.002028  0.01697  -0.002394


Scalar inputs may require enlisting to distinguish from scalar options:

::

   q)random(0;0;9)  /arg is read as a single 3-element list
   6317635140054588591 5831672079708576995 3983176133206258450

   q)random((1#0);0;9)  /enlist value to interpret other orgs as lower & upper bounds
   ,3
   q)random((1#0);0;9)
   ,6

   q)random(0e;5)  / scalar type of real distinguishes input from upper bound
   4e

Calculating gain
^^^^^^^^^^^^^^^^

Return the recommended gain value for the given nonlinearity function; this is the factor used to scale standard deviation.
 

.. function:: gain(nonlinearity) -> value
.. function:: gain(nonlinearity; factor) -> value

   :param symbol nonlinearity: name of the non-linear function, e.g. ```relu``, ```leakyrelu``, ```linear``, etc.
   :param double factor: optional parameter or factor, e.g.  negative slope for ```leakyrelu``.

   :return: The recommended gain value (scalar double) for the given nonlinearity function.

::

   q)s:`conv1d`conv2d`conv3d`convtranspose1d`convtranspose2d`convtranspose3d
   q)s,:`linear`sigmoid`tanh`relu`leakyrelu

   q)s!gain each s
   conv1d         | 1
   conv2d         | 1
   conv3d         | 1
   convtranspose1d| 1
   convtranspose2d| 1
   convtranspose3d| 1
   linear         | 1
   sigmoid        | 1
   tanh           | 1.666667
   relu           | 1.414214
   leakyrelu      | 1.414143

   q)gain(`leakyrelu;.5)
   1.264911

