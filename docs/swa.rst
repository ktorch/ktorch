
Weight averaging
================

`Stochastic weight averaging <https://pytorch.org/blog/pytorch-1.6-now-includes-stochastic-weight-averaging>`_ in PyTorch is done by creating a copy of the module whose weights will be averaged and training that module with a learning rate schedule and an epoch when averaging will begin.

The k api uses three different functions to accomplish the weight averaging.

- :func:`copyparms` - copies from module -> dictionary of parameters at the start of averaging and copies from dictionary -> module at the end of averaging.
- :func:`avgparms`- maintains a running average of model parameters at each call.
- :func:`batchnorm` - updates any batchnorm layers after averaging and also restores momentum settings


Copy parameters
***************

.. function:: copyparms(module) -> dictionary

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module, model or optimizer.
   :return: An :doc:`api-pointer <pointers>` to a dictionary of parameter tensors.

::

   q)m:module enlist(`linear;64;10)
   q)p:copyparms m

   q)size p
   weight| 10 64
   bias  | ,10

The same :func:`copyparms` functions can be used to copy parameters back into the module:

.. function:: copyparms(module; dictionary) -> k boolean dictionary 

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module, model or optimizer.
   :param pointer dictionary: An :doc:`api-pointer <pointers>` to a dictionary of parameter tensors.
   :return: A k dictionary with parameter names mapped to true if copy was successful, else false.

::

   q)m:module enlist(`linear;64;10)
   q)parmnames m
   `weight`bias

   q)p:copyparms m
   q)dict(p;`bias)
   0.07392 -0.1152 0.1046 0.07062 -0.0002719 -0.07141 0.01454 0.1125 0.1195 0.00..

   q)dict(p;`bias;0)  / zero-out bias

   q)copyparms(m;p)   / replace parameters
   weight| 1
   bias  | 1

   q)b:parm(m;`bias)  / verify module bias is zero after copyparms call
   q)tensor b
   0 0 0 0 0 0 0 0 0 0e

.. note::

   Copying parameters occurs without any gradient implications. Also, copying parameters back to the module requires that the parameters have matching names and conformable shapes, i.e. a scalar may be used to reset an array, but different sized arrays will generate an error.

::

   q)m:module enlist(`linear;4;2)

   q)p:dict`weight`other!1 2
   q)copyparms(m;p)  /scalar copy succeeds for `weight
   weight| 1
   bias  | 0

   q)dict(p;`weight;2 3#1)
   q)copyparms(m;p)
   'The size of tensor a (4) must match the size of tensor b (3) at non-singleton dimension 1
     [0]  copyparms(m;p)
          ^

   q)dict(p;`weight;1 4#1)
   q)copyparms(m;p)
   weight| 1
   bias  | 0


Average parameters
******************

The :func:`avgparms` function given module and dictionary uses the module's current parameter values to maintain a running average in the supplied dictionary.
The function adds or increments a scalar tensor with the key ``.n``, which maintains the count of the average.
Module parameters are not permitted to contain a ``.`` because this is used as a depth indicator, so ``.n`` should not overwrite any parameter name.

.. function:: avgparms(module; dictionary) -> count of average

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module, model or optimizer.
   :param pointer dictionary: An :doc:`api-pointer <pointers>` to a dictionary of parameter tensors.
   :return: Adds or increments a key ``.n`` to maintain count of the averaging and also returns as long scalar.

::

   q)m:module enlist(`linear;2;1)
   q)p:copyparms m

   q)dict p
   weight| -0.4876026 -0.3645594
   bias  | -0.1412306           

   q)parm(m;`weight;10)  /update module weight to 10
   q)avgparms(m;p)
   2

   q)dict p
   weight| ,4.756198 4.81772e
   bias  | ,-0.1412306e
   .n    | 2f

   q)\ts:10 avgparms(m;p)
   0 960

   q)dict p
   weight| ,9.126033 9.136288e
   bias  | ,-0.1412306e
   .n    | 12f

Example
*******

In the example below, a single linear module is trained on MNIST data. After the first 20 epochs, the model accuracy on test data is 92.26%.
A copy of the module's parameters is then used to initiate a 10-epoch run with weight averaging, increasing accuracy to 92.40%.
One more 10-epoch run with weight averaging brings the model accuracy up to 92.51%:

::

   q){key[x]set'get x}(`ktorch 2:`fns,1)[];
   q)\l examples/mnist/mnist.q

   q)d:mnist`:examples/mnist/data
   q)d:@[;`y`Y;"j"$]@[d;`x`X;{resize("e"$-1+x%127.5;-1 784)}]

   q)q:module enlist(`linear;784;10)
   q)m:model(q; loss`ce; opt(`sgd; q; .04))

   q)train(m; `batchsize`shuffle; 100,1b)
   q)train(m; d`x; d`y);

   q)\ts:20 run m
   1655 528

   q)avg d.Y={x?max x}each evaluate(m;d`X)
   0.9226

   q)p:copyparms m
   q)\ts:10 {run x; avgparms(x;y);}[m;p]
   810 2080

   q)copyparms(m;p)
   weight| 1
   bias  | 1

   q)avg d.Y={x?max x}each evaluate(m;d`X)
   0.924

   q)\ts:10 {run x; avgparms(x;y);}[m;p]
   807 2080

   q)copyparms(m;p)
   weight| 1
   bias  | 1

   q)avg d.Y={x?max x}each evaluate(m;d`X)
   0.9251

Update batchnorm layers
***********************

Once the averaged parameters are copied back to the trained module, any `batchnorm <https://pytorch.org/docs/stable/nn.html#normalization-layers>`_ layers will have incorrect running mean and variance statistics. The :func:`batchnorm` function is designed to accept a model with previously defined data and recalculate the mean and standard deviation.

Recalculating mean, var
^^^^^^^^^^^^^^^^^^^^^^^

.. function:: batchnorm(module) -> dictionary

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module, model or optimizer.
   :return: A k dictionary with names of batchnorm modules mapped to their original momentum settings.


If a :ref:`model <model>` is given as the argument and the model has a defined set of training data, the :func:`batchnorm` function will perform all the steps below:

1. For each `batchnorm <https://pytorch.org/docs/stable/nn.html#normalization-layers>`_ layer

  * the current momentum setting is saved, then reset to ``None``
  * the running mean is reset to 0
  * the running variance reset to 1

2. The forward calculation for the module contained by the model is called on the model's defined data

  * the lack of a defined momentum will cause the batchnorm layers to calculate a simple running average and variance through the model's data using the defined batch size
  * no gradient calculations are performed during the forward calls
  * the previous momentum settings are then restored for each batchnorm layer

3. The function returns a k dictionary with the names of the batchnorm modules along with their original momentum setting.

.. note::

   If a model is given without any defined data, or if the :func:`batchnorm` function is called with an allocated module or optimizer, then only steps 1 and 3 are performed and the running mean and variance of the batchnorm layers will have to be explicitly calculated over the training data.

Restoring momentum
^^^^^^^^^^^^^^^^^^

If the :func:`batchnorm` function invocation does not recalculate the running mean and variance directly,
the result from the first call (without a second argument) can be used to restore the original momentum setting following a manual recalculation of mean and variance.

.. function:: batchnorm(module; dictionary) -> null

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module, model or optimizer.
   :param kdictionary dictionary: A k dictionary with names of batchnorm modules mapped to their original momentum settings.
   :return: Restores the momentum setting in each batchnorm layer named in the k dictionary with its corresponding value. Null return.


Momentum example
^^^^^^^^^^^^^^^^

In the example below, a linear model to classify MNIST digits has a single batchnorm layer. After training, the batchnorm mean and variance are recalculated using the :func:`batchnorm` function called with a model and defined data, then again, only to reset momentum, with the mean and variance recalculated with explicit forward calls.

First, read in `MNIST <https://ktorch-examples.readthedocs.io/en/latest/mnist.html>`_ data:

::

   q){key[x]set'get x}(`ktorch 2:`fns,1)[];  /define interface in root namespace

   q)\l examples/mnist/mnist.q
   q)d:mnist`:examples/mnist/data
   q)d:@[;`y`Y;"j"$]@[d;`x`X;{resize("e"$-1+x%127.5;-1 784)}]

Define module with a ``batchnorm`` layer:

::

   q)q:(`sequential; (`linear;`a;784;800); (`batchnorm1d;`b;800;1e-7;.1); `relu`c; (`linear;`d;800;10))
   q)q:seq q

   q)q
   `sequential
   ,(`linear;`a;784;800)
   ,(`batchnorm1d;`b;800;1e-07;0.1)
   ,`relu`c
   ,(`linear;`d;800;10)

   q)q:module q; p:parms(q;`b); b:buffers(q;`b)

   q)dict p
   weight| 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1..
   bias  | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0..

   q)dict b
   running_mean       | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ..
   running_var        | 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ..
   num_batches_tracked| 0

Create a :ref:`model <model>` with the module (increment reference count to maintain the original module handle) and define data for training and testing:

::

   q)m:model(addref q;loss`ce; opt(`sgd; q; .05; .9; `nesterov,1b))
   q)train(m; `batchsize`shuffle; 100,1b); train(m; d`x; d`y);
   q)test(m; `batchsize`metrics; (1000;`accuracy)); test(m; d`X; d`Y);

During initial training, the running mean and variance statistics are updated using the default ``momentum`` setting of ``.10``, 
e.g. :math:`avg = .9 * prev + .1 * new`.

::

   q)\ts:20 run m  /train 20 epochs on cpu in about 15 seconds
   15045 528

   q)testrun m     /accuracy of 98.5%
   98.5

   q)show b1:dict b    /value of running mean & variance calculations
   running_mean       | -1.131 -0.08618 1.274 0.3693 -1.509 0.2754 0.2031 0.2767..
   running_var        | 1.148 1.131 0.5452 0.5967 1.486 0.8175 0.5692 0.8316 0.5..
   num_batches_tracked| 12000

   q)exec first options from module q where module=`batchnorm1d
   in      | 800
   eps     | 1e-07
   momentum| 0.1
   affine  | 1b
   track   | 1b


Recalculate the mean and variance by calling the :func:`batchnorm` function with the model and its associated training data:

::

   q)show r:batchnorm m  /return name of batchnorm layers w'momentum setting
   b| 0.1

   q)show b2:dict b   /different mean & variance (without exponential averaging)
   running_mean       | -1.125 -0.05887 1.271 0.3727 -1.54 0.269 0.2114 0.2175 -..
   running_var        | 1.214 1.097 0.5356 0.6177 1.423 0.837 0.5294 0.7965 0.56..
   num_batches_tracked| 600

   q)avg each b1%b2
   running_mean       | 1.024
   running_var        | 1.012
   num_batches_tracked| 20

The recalculated mean and variance (simple average) differ somewhat from the result of the 20 training runs using the exponentially weighted average.

In the steps below, the :func:`batchnorm` function will only be used to retrieve and reset the momentum and running statistics.
The mean and variance are recalculated by running the forward calculation in batches (using :ref:`nforward <forward>` without any gradient calculation). Finally, the :func:`batchnorm` function is used to restore the batchnorm layer's original momentum setting:

::

   q)show r:batchnorm q
   b| 0.1

   q)exec first options from module q where module=`batchnorm1d
   in      | 800
   eps     | 1e-05
   momentum| 0n       / momentum set to 'none'
   affine  | 1b
   track   | 1b

   q)dict b
   b.running_mean       | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ..
   b.running_var        | 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ..
   b.num_batches_tracked| 0


Calling :func:`batchnorm` on a module resets the ``momentum`` setting and resets the running mean to 0 and the variance to 1, as well as resetting the number of batches tracked to 0.
Next, use a copy of the model data to create a separate tensor that will be batched and used to do the forward calculation to reset the running mean and variance statistics of the ``batchnorm1d`` layer:

::

   q)restore m  /restore full size of model inputs after batching
   60000

   q)x:tensor input m  / set x to inputs associated with the model
   q)size x
   60000 784

   q)while[batch(x;100); free nforward(q;x)]

   q)show b3:dict b
   running_mean       | -1.125 -0.05887 1.271 0.3727 -1.54 0.269 0.2114 0.2175 -..
   running_var        | 1.214 1.097 0.5356 0.6177 1.423 0.837 0.5294 0.7965 0.56..
   num_batches_tracked| 600

   q)avg each b2%b3
   running_mean       | 1
   running_var        | 1
   num_batches_tracked| 1

   q)batchnorm(q;r)  / restore momentum setting

   q)exec first options from module q where module=`batchnorm1d
   in      | 800
   eps     | 1e-07
   momentum| 0.1
   affine  | 1b
   track   | 1b

