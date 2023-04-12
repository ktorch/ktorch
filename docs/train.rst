Training steps
==============

Given a set of inputs and targets, the main training steps are:

- reset any previously calculated gradients to zero
- call the forward method of the module describing the neural network to get model outputs
- call the loss function with the model outputs and targets
- compute the gradients on all the parameters involved in the loss calculation
- use an optimizer to reduce the parameters by some fraction of the gradients

From the starting example using generated `spirals <https://github.com/ktorch/examples/blob/master/start/spirals.q>`_,
begin with module ``q``, loss ``l`` and optimizer ``o``:

::

   q)n:1000; k:3;              /n:sample size, k:classes

   q)q:module seq(`sequential; (`linear; 2; 100); `relu; (`linear; 100; k))
   q)l:loss`ce                 /cross-entropy loss
   q)o:opt(`sgd; q; .2; .99)   /gradient descent: learning rate .2, momentum .99

Create inputs ``x`` and targets ``y``:

::

   q)spiral:{x:til[x]%x-1; flip x*/:(sin;cos)@\:z+sum 4*(x;y)}
   q)z:tensor(`randn; k,n; `double)                   /unit normal
   q)x:tensor "e"$raze spiral[n]'[til k;.2*tensor z]  /spirals w'noise
   q)y:tensor til[k]where k#n                         /class 0,1,2
 
With these  PyTorch objects, the k api training steps become:

::

   q)zerograd o          /set gradients to zero
   q)rx:forward(q;x)     /calculate the output of the module
   q)ry:loss(l;rx;y)     /calculate loss from output & target
   q)backward ry         /backward calc computes parameter gradients
   q)step o              /optimizer updates parameters from gradients

Putting all the steps into a function ``f`` which frees intermediate tensors and returns loss:

::

   q)f:{[q;l;o;x;y]zerograd o; x:forward(q;x); r:tensor y:loss(l;x;y); backward y; free each(x;y); step o; r}

   q)\ts:3 show f[q;l;o;x;y]
   1.031206e
   0.9710665e
   0.9085126e
   5 4194736

Higher-level routines are available to perform the above steps using a :ref:`model <model>`, which mangages a
:ref:`module <modules>` and
:ref:`loss <loss>` function together with an
:ref:`optimizer <optimizer>`.

::

   q)m:model(q;l;o)
   q)g:{[m;x;y]nograd m; r:backward(m;x;y); step m; r}

   q)\ts:3 show g[m;x;y]
   0.8101332e
   0.7579743e
   0.7397364e
   6 4194704

There are also routines to manage training :ref:`options <model-options>` for a model, defining :ref:`data <model-data>` and
processing :ref:`batches <model-batches>`.
The full training and testing :ref:`run <model-run>` through each batch of data can sometimes be accomplished with a 
:ref:`single call <model-run>` which handles :ref:`batching <model-batches>`, :ref:`resetting <model-resetgrad>` gradients,
running :ref:`forward <model-forward>` and :ref:`loss <model-loss>` calculations, :ref:`backward <model-backward>` propagation of the gradients and
:ref:`updating <model-step>` the parameters, along with accumulating model :ref:`metrics <model-metrics>`.

.. _model-resetgrad:

Reset gradients
***************

PyTorch added a flag to its `zero_grad() <https://pytorch.org/docs/1.11/generated/torch.optim.Optimizer.zero_grad.html>`_ method for modules and optimizers allowing the method to release the tensor gradient memory rather than set all its values to zero.  The k-api implements the call with the ``set_to_none`` flag set true as a separate function, :func:`nograd`.

.. function:: nograd(object) -> null
.. function:: zerograd(object) -> null

   :param pointer object: an :doc:`api-pointer <pointers>` to a previously allocated tensor, vector or dictionary of tensors, or a module, optimizer or model.
   :return: :func:`nograd` resets the gradient as undefined, :func:`zerograd` sets the gradient values to zero.

Using :func:`nograd` is a little quicker and uses less memory with some possible side effects detailed `here <https://pytorch.org/docs/1.11/generated/torch.optim.Optimizer.zero_grad.html>`_.

Tensor gradient
^^^^^^^^^^^^^^^

An individual tensor gradient can be reset, along with a vector or dictionary of tensors:

::

   q)backward z:mean y:mul(x; x:tensor(1 2 3.0; `grad))
   q)grad x
   0.6666667 1.333333 2

   q)zerograd x
   q)grad x
   0 0 0f

   q)nograd x
   q)grad x
   
   q)(::)~grad x
   1b

In this example, a dictionary of tensors is used to reset gradients:

::

   q)x2:mul(x; x:tensor(1 2 3.0;`grad))
   q)y2:mul(y; y:tensor(2 4 6.0;`grad))
   q)backward(z:add(x2;y2); 1 1 1)

   q)grad each(x;y)
   2 4 6 
   4 8 12

   q)d:dict`x`y!addref each(x;y)
   q)zerograd d

   q)grad each(x;y)
   0 0 0
   0 0 0

   q)nograd d
   q)grad each(x;y)
   ::
   ::

Module gradient
^^^^^^^^^^^^^^^

In training, gradients are usually reset using a 
:ref:`module <modules>`,
:ref:`optimizer <optimizer>` or
:ref:`model <model>`.

If an optimizer is managing the parameters from a single module, using any of the 3 types of objects resets the same set of gradients.
However, it is possible to define an optimizer which manages the parameters of multiple modules.
In this case, resetting the gradients using the individual module objects handles each module's gradients separately, whereas using the optimizer object (or its parent model) as the argument will reset all gradients at once.

::

   q)a:module seq(`sequential; (`linear;`a;64;20); `relu)
   q)b:module seq(`sequential; (`linear;`b;20;10); `drop)

   q)parmnames a
   `a.weight`a.bias
   q)aw:parm(a;`a.weight)
   q)bw:parm(b;`b.weight)

Define an optimizer with 2 groups of parameters and run a sample forward and backward calculation:

::

   q)o:opt`sgd; opt(o;0;a); opt(o;1;b)

   q)x:tensor(`randn; 3 64)
   q)backward z:ce(yb:forward(b; ya:forward(a; x)); 0 2 9)

Resetting gradients via module or optimizer objects:

::

   q)zerograd a

   q)grad each(aw;bw)
   (0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ..
   (0 -0.5661 -0.4733 0 -0.1193 0 0 0 0 -0.3972 -0.262 0 0 0 0 -0.00413 -0.2554 ..

   q)zerograd o
   q)grad each(aw;bw)
   (0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ..
   (0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0e;0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0..

   q)nograd b
   q)grad each(aw;bw)
   (0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ..
   ::

   q)nograd o
   q)grad each(aw;bw)
   ::
   ::

.. _model-forward:

Forward calculation
*******************

Forward calculation requires a module and input(s).  These arguments are supplied directly to the 
different api functions for :ref:`forward <forward>` calculation, or defined implicitly with a model and its training/testing data
and batching.

Here, an `identity <https://pytorch.org/docs/stable/generated/torch.nn.Identity.html>`_ module returns inputs, either using the whole tensor, or in batches:

::

   q)x:tensor til 7
   q)q:module`identity

   q)tensor y:forward(q;x)
   0 1 2 3 4 5 6

   q)while[batch(x;3); use[y]forward(q;x); show tensor y]
   0 1 2
   3 4 5
   ,6

A different form of the :ref:`forward <forward>` calculation runs in evaluation mode without gradients and returns one or more k array(s):


::

   q)evaluate(q;x)
   0 1 2 3 4 5 6

Another form of the forward calculation uses a single model argument and retrieves the input(s) from model's defined training/testing data and any batching in effect.

.. function:: forward(model) -> tensor
.. function:: nforward(model) -> tensor

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :return: performs the forward calculation of the module contained by the model in training mode, with gradients for the :func:`forward` call, and with no gradient calculation for the :func:`nforward` call. Both functions return tensors.

A similar syntax is used to run the forward calculations in
`evaluation <https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=eval#torch.nn.Module.eval>`_ 
mode without gradient calculations and returning either tensor(s) or k array(s):

.. function:: eforward(model) -> tensor
.. function:: evaluate(model) -> k array

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :return: performs the forward calculation of the module contained by the model in `evaluation <https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=eval#torch.nn.Module.eval>`_ mode and without gradients, returning tensor(s) or k array(s).

::

   q)m:model(module`identity; loss`mse; opt`sgd)
   q)train(m;`batchsize;3)
   q)train(m; til 7)
   3

   q)data m
   0 1 2 3 4 5 6
   ::

   q)tensor y:forward m
   0 1 2 3 4 5 6

   q)while[batch m; use[y]forward m; show tensor y]
   0 1 2
   3 4 5
   ,6


.. note:

   When using a single model argument, the forward calculations in training mode are designed to use the model's training data and the evaluation mode calls use the testing data.


Continuing the above example with an evaluation call fails because no testing data is defined:

::

   q)eforward m
   'identity: forward(tensor) not implemented given empty
     [0]  eforward m
          ^


After defining some test data:

::

   q)test(m;`batchsize;5)
   q)test(m;til 10)
   2

   q)while[testbatch m; show evaluate m]
   0 1 2 3 4
   5 6 7 8 9

.. _model-loss:

Loss calculation
****************

:ref:`Loss <losscalc>` calculation uses the output of a module along with the targets.
The api :func:`loss` function uses the same syntax whether an explicit :ref:`loss <loss>` module is used, 
or the loss module is derived from the model used as the first argument:

::

   q)m:model(module`identity; loss`mse; opt`sgd)

   q)train(m;`batchsize;3)
   q)train(m; .1 2.7 3 4 5 6 7e; 1 2 3 4 5 6 7e)
   3

   q)tensor l:loss(m;input m;target m)
   0.1857143e

   q)while[batch m; use[l]loss(m;input m;target m); show tensor l]
   0.4333333e
   0e
   0e

   q)0.4333333e*3%7
   0.1857143


.. _model-backward:

Backward calculation
********************

PyTorch implements backward calculations as a 
`tensor <https://pytorch.org/docs/stable/generated/torch.Tensor.backward.html>`_ method and an
autograd `function <https://pytorch.org/docs/stable/generated/torch.autograd.backward.html?highlight=backward#torch.autograd.backward>`_.

The k-api implements the backward calculation as a function that can accept either tensor(s) or a :ref:`model <model>`.
The most common direct use is to calculate model outputs, then the loss from model outputs and known targets.
This loss tensor (usually a scalar tensor) is used to as the argument to the backward call to get the gradients of tensors involved in the chain of forward calculations.

An optional mode, a symbol, can be specified to control whether the graph is retained upon completion of the backward call (default is ``false``),
and whether a graph is to be created to calculate higher order derivatives (default is ``false``).

.. csv-table::
   :header: "mode", "retain", "create", "description"
   :widths: 12, 8, 8, 40

   free,false,false,"free graph, no higher order derivatives"
   retain,true,false,"retain graph, no higher order derivatives"
   create,true,true,"retain graph, create graph for higher order derivatives"
   createfree,false,true,"free graph & create graph for higher order derivatives"

This table is available in a q session using the :func:`help`:

::

   q)help`backward
   mode       retain create description                                         ..
   -----------------------------------------------------------------------------..
   free       0      0      "free graph, no higher order derivatives"           ..
   retain     1      0      "retain graph, no higher order derivatives"         ..
   ..


Tensor
^^^^^^

.. function:: backward(tensor) -> null
.. function:: backward(tensor;mode) -> null
   :noindex:
.. function:: backward(tensor;gradtensor) -> null
   :noindex:
.. function:: backward(tensor;gradtensor;mode) -> null
   :noindex:

   :param pointer tensor: an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors.
   :param pointer gradtensor: an :doc:`api-pointer <pointers>` to tensor(s) or k arrays for each non-scalar tensor given in 1st arg.
   :param symbol mode: optional, controls whether graph is to be retained or second order derivatives are required, default is ```free``.

   :return: Runs the backward calculation on the chain of calculations attached to the tensor(s), populating the gradients..

::

   q)z:mean y:mul(x;x:tensor(1 2 3e;`grad))

   q)select name:`x`y`z,device,dtype,gradient,gradfn,leaf from info'[(x;y;z)]
   name device dtype gradient gradfn        leaf
   ---------------------------------------------
   x    cpu    float grad                   1   
   y    cpu    float grad     MulBackward0  0   
   z    cpu    float grad     MeanBackward0 0   

   q)backward(y;1e;`retain)
   q)grad x
   2 4 6e

   q)zerograd x
   q)grad x
   0 0 0e

   q)backward z

   q)grad x
   0.6666667 1.333333 2e

In this example, the calculation graph needs to be retained in order to calculate gradients with respect to both mean and standard deviation:

::

   q)m:mean y:mul(x;x:tensor(1 2 3e;`grad)); z:std y

   q)backward m
   q)grad x
   0.6666667 1.333333 2e

   q)backward z
   'Trying to backward through the graph a second time (or directly access saved tensors after they have already been freed). Saved intermediate values of the graph are freed when you call .backward() or autograd.grad(). Specify retain_graph=True if you need to backward through the graph a second time or if you need to access saved tensors after calling backward.
     [0]  backward z
          ^

   q)free(x;y;m;z)
   q)m:mean y:mul(x;x:tensor(1 2 3e;`grad)); z:std y

   q)backward(m;`retain)
   q)grad x
   0.6666667 1.333333 2e

   q)nograd x
   q)backward z
   q)grad x
   -0.9072646 -0.3299144 3.216666e

Model
^^^^^

If the :func:`backward` function is used with a model as the first argument,
it is possible to perform all the following calculations, either separately, or in a single step:

- run forward calculation with input(s) to get model output(s)
- run loss function with model output(s) and defined target(s) to get loss tensor
- run backward calculations on the loss tensor to determine gradients on all learnable parameters in the chain of loss calculations

.. function:: backward(model) -> loss scalar
.. function:: backward(model;inputs;targets) -> loss scalar
   :noindex:

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :param tensor inputs: tensor(s) or k array(s) with model inputs, e.g. images for vision models, sequences for transformers, etc.
   :param tensor targets: tensor(s) or array(s) with targets, e.g. classes for images or next token in sequences.
   :return: Runs the forward calculation on supplied inputs to get model outputs, calculates loss with outputs and supplied targets, then runs the backward calculations on the loss to get gradients. The model's defined training data is used if no inputs or targets supplied. Returns the loss as a k scalar.


In the example below, the forward, loss and backward steps are calculated separately using a model and random inputs and targets.
The :func:`backward` call is on the tensor returned from the :func:`loss` call with model outputs and targets:

::

   q)q:module seq(`sequential; (`linear;`a;64;10); (`relu;`b))

   q)parmnames q
   `a.weight`a.bias

   q)b:parm(q;`a.bias)

   q)seed 7
   q)x:tensor(`randn; 10 64)
   q)y:tensor(`randint;10;10)
   q)m:model(q; loss`ce; opt(`sgd;q))

   q)yhat:forward(m;x)
   q)z:loss(m;yhat;y)
   q)backward z

   q)tensor z
   2.358e
   q)grad b
   -0.1293 -0.06398 0.07666 0.06759 0.02692 0.07495 -0.03646 0.06521 -0.01563 0...

After defining batchsize, defining training inputs and targets for the model,
calling the :func:`backward` with a single model argument runs the forward, loss and backward step together:

::

   q)train(m;`batchsize;3)
   q)input(m;x); target(m;y)
   4

   q)nograd m
   q)backward m
   2.358e

   q)grad b
   -0.1293 -0.06398 0.07666 0.06759 0.02692 0.07495 -0.03646 0.06521 -0.01563 0...

This part of the example runs the same calculations in explicit batches:

::

   q)nograd m
   q)g:l:n:(); while[batch m; nograd m; n,:datasize m; l,:backward m; g,:enlist grad b]

   q)n
   3 3 3 1

   q)l
   2.146 2.302 2.602 2.43e
  
   q)n wavg/:(l;g)
   2.358
   -0.1293 -0.06398 0.07666 0.06759 0.02692 0.07495 -0.03646 0.06521 -0.01563 0...


(The higher level :func:`run` and :func:`testrun` functions can be used with conforming models to handle the bach loop and calculation of loss, subsequent gradients, updating of parameters and output of model metrics.)

.. _model-clip:

Gradient clipping
*****************

PyTorch provides utilities to clip gradients by 
`value <https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_value_.html>`_ and by
`norm <https://pytorch.org/docs/stable/generated/torch.nn.utils.clip_grad_norm_.html>`_.

The k api provides two separate functions, :func:`clipv` and :func:`clip`, for clipping by value and by norm, along with model
:ref:`training options <model-train>` for incorporating the gradient clipping into the model's overall training :ref:`run <model-run>`.

Clip value
^^^^^^^^^^

.. function:: clipv(tensors;value) -> null

   :param pointer tensors: an :doc:`api-pointer <pointers>` to a previously created tensor, vector or dictionary of tensors, or any of: module, optimizer or model.
   :param double value: maximum magnitude of the gradients of the supplied tensor(s); the gradients are clipped in the range ``[-value, value]``.
   :return: Null value for double scalar, ``0n``.

::

   q)z:mean y:mul(x;x:tensor(1 2 3e;`grad))

   q)backward z
   q)grad x
   0.6666667 1.333333 2e

   q)clipv(x;1.2)
   0n

   q)grad x
   0.6666667 1.2 1.2e

Clipping the gradients of a module by value:

::

   q)m:module enlist(`linear;10;3)
   q)parmnames m
   `weight`bias
   q)p:parm(m;`weight)

   q)x:tensor(`randn;7 10)
   q)y:tensor(`randint;3;7)
   q)backward z:ce(yhat:forward(m;x); y)

   q)grad p
   -0.288 0.0644 -0.382 -0.0119 0.386    0.109  -0.154 -0.487 0.08    -0.0597
   0.134  0.108  0.495  0.0396  -0.378   0.037  0.401  0.37   -0.0302 0.0436 
   0.154  -0.173 -0.114 -0.0277 -0.00841 -0.146 -0.247 0.117  -0.0498 0.016  

   q)(min;max)@\:raze grad p
   -0.487 0.495e

   q)clipv(m;.4)
   0n

   q)(min;max)@\:raze grad p
   -0.4 0.4e

Clip norm
^^^^^^^^^

.. function:: clip(tensors;maxnorm) -> previous norm
.. function:: clip(tensors;maxnorm;normtype) -> previous norm
   :noindex:
.. function:: clip(tensors;maxnorm;groupflag) -> previous norm
   :noindex:
.. function:: clip(tensors;maxnorm;normtype;groupflag) -> previous norm
   :noindex:

   :param pointer tensors: an :doc:`api-pointer <pointers>` to a previously created tensor, vector or dictionary of tensors, or any of: module, optimizer or model.
   :param double maxnorm: maximum norm of the gradients of the supplied tensors.
   :param double maxnorm: norm exponent, default=``2`` if none supplied.
   :param bool groupflag: if ``true`` and model or optimizer given in first argument, maximum norms will apply to each parameter group defined for the optimizer. Default is ``false``: the maximum norm is clipped across the gradients of all parameter tensors together.
   :return: The previous norm for the gradients, scalar unless group flag is ``true``, then list of previous norms.


In this example, the parameters from two linear modules are used to populate two optimizer groups:

::

   q)a:module enlist(`linear;`a;64;32)
   q)b:module enlist(`linear;`b;32;3)
   q)o:opt`sgd
   q)opt(o;0;a)
   q)opt(o;1;b)

The group flag is turned on for clipping the gradient to a norm of ``1.00`` for each group:

::

   q)x:tensor(`randn;7 64)
   q)y:tensor(`randint;3;7)
   q)backward z:ce(yb:forward(b;ya:forward(a;x)); y)

   q)clip(o;1.0;1b)
   1.767239 1.109446

Verify that the previous group norms were clipped:

::

   q)clip(o;.95;1b)
   0.999999 0.9999992

   q)clip(o;.9;1b)
   0.9499991 0.949999

When the group flag is ``false`` by default, verify that the norm across all the parameters is changed and also clipped:

::

   q)clip(o; .75)
   1.272791

   q)clip(o; .5)
   0.7499995

.. _model-step:

Optimizer step
**************

The final step in the training loop after 
:ref:`calculating model outputs <model-forward>`,
getting the :ref:`loss <model-loss>` from comparing outputs to targets,
:ref:`calculating <model-backward>` and optionally
:ref:`clipping gradients <model-clip>`
is using the optimizer to subtract some fraction of the gradients from the current value of the parameters.

step
^^^^

.. function:: step(optimizer) -> null
.. function:: step(model) -> null
   :noindex:

   :param pointer optimizer: an :doc:`pointer <pointers>` to a previously created :ref:`optimizer <optimizer>`.
   :param pointer model: a :doc:`pointer <pointers>` to a previously created :ref:`model <model>`.
   :return: Given optimizer or deriving optimizer from given model, updates parameters using the gradients from previous :ref:`backward <model-backward>` calculations. Null return.

.. note::

   The `LBFGS <https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html>`_ optimizer cannot be used with :func:`step` because the optimizer requires the full model along with inputs & targets in order to reevaluate the model repeatedly before applying the parameter updates. See function :func:`backstep`.

In the example below a tensor is updated via repeated :func:`step` calls to converge on the given target using `mean squared error <https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html>`_:

::

   q)x:tensor(0.5 2.5 3e; `grad)
   q)y:tensor(1.0 2.0 3e)
   q)z:tensor()
   q)o:opt(`sgd;x)

   q)\ts:20 {[x;y;z;o]use[z]mse(x;y); backward z; step o}[x;y;z;o]
   3 3472

   q)([]tensor x;tensor y)
   x     y
   -------
   1.052 1
   1.948 2
   3     3


backstep
^^^^^^^^

The :func:`backstep` function is similar to :func:`backward`, but also includes the optimizer step that updates the model's learnable parameters.
If the model has defined gradient clipping, this is also performed after the backward calculation and just before the optimizer step that updates the parameters.

.. function:: backstep(model) -> loss scalar
.. function:: backstep(model;inputs;targets) -> loss scalar
   :noindex:

   :param pointer model: a :doc:`pointer <pointers>` to a previously created :ref:`model <model>`.
   :param tensor inputs: tensor(s) or k array(s) with model inputs, e.g. images for vision models, sequences for transformers, etc.
   :param tensor targets: tensor(s) or array(s) with targets, e.g. classes for images or next token in sequences.
   :return: If only model argument supplied, predefined model input and target data will be used, else supplied inputs & targets. The function first calculates model outputs, then the loss using these outputs and defined/supplied targets.  The backward calculations from the loss establish gradients and the optimizer updates its parameters with these gradients. The loss is returned as a k double scalar.

In the example below, a :ref:`callback <module-callback>` module with a single trainable parameter ``x`` is defined and incorporated in a model with mean squared error for the loss function and a `LBFGS <https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html>`_ optimizer.
After a single :func:`backstep` call in which the optimizer evaluates the model multiple times, the trainable parameter approaches the target:

::

   q)m:module enlist (`callback;`cb;"{[m;x]parm(m;0N!`x)}"; (`parms;(`x;.5 2.5 3)))
   q)m:model(m;loss`mse;opt(`lbfgs;m))
   q)tensor x:parm(m;`x)
   0.5 2.5 3

   q)backstep(m;0;1 2 3.0)
   `x
   `x
   `x
   0.1666667

   q)tensor x
   1 2 3f


Train/Test
**********

The functions :func:`train` and :func:`test` can be used to set up training and testing using a :ref:`model <model>`.

.. _model-options:

Options
^^^^^^^

The following options are used for both training and testing modes:

- ``batchsize`` - batch size for train/test data, long integer, default is ``32`` for training, ``100`` for testing.
- ``droplast`` - flag indicating whether to drop last batch if not full size, default is ``false``.
- ``hidden`` - flag indicating model has a hidden state as part of input/output, default is ``false``.
- ``tensor`` - flag set ``true`` if k arrays are to be returned, ``false`` for tensors. Default is ``false``.
- ``dictionary`` - flag set ``true`` if dictionary to be returned, default is ``false``. If both ``tensor`` and ``dictionary`` options are ``true``, a tensor dictionary is returned, else a k dictionary.

.. _model-metrics:

- ``metrics`` - symbol(s) indicating what outputs to be calculated and returned by training run.
   - ``loss`` - return average loss across all batches
   - ``batchloss`` - returns a list/tensor with individual batch losses
   - ``accuracy`` - returns the percentage of the model predictions that are correct.
   - ``predict`` - returns model predictions.
   - ``output`` - returns model output.
   - ``hidden`` - returns model hidden state (see recurrent models, `RNN <https://pytorch.org/docs/stable/generated/torch.nn.RNN.html>`_ and `GRU <https://pytorch.org/docs/stable/generated/torch.nn.quantized.dynamic.GRU.html>`_).
   - ``hiddencell`` - returns 2nd part of hidden state (see `LSTM <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html>`_).

.. _model-train:

These options are only used for training:

- ``clipgroup`` - flag set ``true`` to indicate that gradients should be clipped according to optimizer group, defalut is ``false`` indicating that the gradient norm should be calculated across all gradients of the model.
- ``clipnorm`` -  maximum norm of the gradients, default is k unary null. If not null, can be a single number for the maximum norm, or a pair of numbers, maximum norm and the order of the norm (default is ``2``). Specifying infinity, ``0w``, for the order implies ``infinity norm`` or maximum of the absolute values.
- ``clipvalue`` - maximum value used for gradient clipping, default is k unary null.
- ``shuffle`` - flag indicating that traing data is to be reshuffled at end of epoch, default is ``false``.
- ``sync`` - do an explicit sync between GPU and CPU, defalut is ``false``. No sync is required, and execution in asynchronous mode should be faster, but there are exceptions, see PyTorch `issue <https://github.com/pytorch/pytorch/issues/63618>`_.

These training options are only relevant for :ref:`distributed training <dist>`:

- ``task`` - indicates task index, 0-n, for n+1 tasks (akin to "rank" in PyTorch's `init_process_group <https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group>`_).
- ``tasks`` - total number of tasks, default is 1, can be set larger for distributed training (sometimes called "world size" in distributed training, e.g. `init_process_group <https://pytorch.org/docs/stable/distributed.html#torch.distributed.init_process_group>`_).
- ``shufflecuda`` - set boolean flag ``true`` if all tasks are to be run on CUDA devices allowing for faster generation of the permutation index. Default is ``false`` which creates a CPU generator and random permutation on the CPU, then transfers the permutation index to the relevant device. With multiple tasks using the same dataset, a generator is created and seeded identically for all tasks and their devices; this must be on the CPU if any of the tasks are set to run on a non-CUDA device.
- ``shuffleseed`` - initializes the `generator <https://pytorch.org/docs/stable/generated/torch.Generator.html>`_ with the same seed given for all tasks, so all tasks can use the same permutation index and select their own random, non-overlapping subsets.

Get options
^^^^^^^^^^^

.. function:: train(model) -> k dictionary of all option names and values
.. function:: train(model;names) -> k value or dictionary of given option names and thier corresponding values
   :noindex:

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :param symbol names: a single symbol or list of symbols, e.g. ```shuffle`` or ```batchsize`shuffle``.
   :return: if a single name is given, returns a single k value, else a k dictionary of names mapped to values.

::

   q)m:model(module`sequential; loss`ce; opt`sgd)

   q)train m
   batchsize | 32
   droplast  | 0b
   hidden    | 0b
   shuffle   | 0b
   ..

   q)train(m;`clipnorm`batchsize`metrics)
   clipnorm | ::
   batchsize| 32
   metrics  | ,`loss

   q)train(m; `shuffle)
   0b


Set options
^^^^^^^^^^^

.. function:: train(model;dictionary) -> null

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model`.
   :param k-dictionary dictionary: a k dictionary mapping option names to corresponding k values.
   :return: options matching dictionary keys are reset to mapped values, null return.

.. function:: train(model;names;values) -> null
   :noindex:

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :param symbol names: a single symbol or list of symbols, e.g. ```shuffle`` or ```batchsize`shuffle``.
   :param k-values values: a single k scalar or enlisted value or a list of values corresponding to supplied names.
   :return: named options are reset to supplied values, null return.

::
 
   q)k:`batchsize`shuffle
   q)m:model(module`sequential; loss`ce; opt`sgd)

   q)train(m;k)
   batchsize| 32
   shuffle  | 0b

   q)train(m;k!100,1b)
   q)train(m;k)
   batchsize| 100
   shuffle  | 1b

   q)train(m;`batchsize`shuffle; (256;0b))
   q)train(m;k)
   batchsize| 256
   shuffle  | 0b

Setting non-scalar options:

::

   q)train(m;`metrics)
   ,`loss

   q)train(m;`metrics;`output`accuracy`loss)

   q)train(m;`metrics)
   `output`accuracy`loss


Setting ```clipnorm`` with defined and null value:

::

   q)train(m; `clipgroup`clipnorm)
   clipgroup| 0b
   clipnorm | ::

   q)train(m; `clipgroup`clipnorm; (0b;2))
   q)train(m; `clipgroup`clipnorm)
   clipgroup| 0b
   clipnorm | 2 2f

   q)train(m; `clipgroup`clipnorm; (1b;2))
   q)train(m; `clipgroup`clipnorm)
   clipgroup| 1b
   clipnorm | 2 2f

   q)train(m; `clipgroup`clipnorm; (0b;()))
   q)train(m; `clipgroup`clipnorm)
   clipgroup| 0b
   clipnorm | ::

.. _model-data:

Defining data
^^^^^^^^^^^^^

The :func:`train` and :func:`test` functions are also used to specify inputs and targets for training and testing.

.. function:: train(model;inputs;targets) -> number of batches
.. function:: test(model;inputs;targets) -> number of batches
   :noindex:

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :param tensor inputs: input(s) can be a single tensor or a set of tensor(s) or k arrays.
   :param tensor targets:  targets are typically a single tensor or k array, but may also be given as a set of tensors or arrays.
   :return: Returns the number of batches that will be processed given size of supplied data and the ``batchsize`` setting.

Many models use a single tensor for input and another for the target, but it is possible to specify multiple inputs/targets:

::

   q)m:model(module`identity; loss`mse; opt`sgd)
   q)train(m;`batchsize;100)

   q)x1:tensor(`randn; 500 32)
   q)x2:tensor(`randint;10;500)
   q)y:tensor(500#1.0e)

   q)train(m; (x1;x2); y)
   5

Data can be specified using a vector of tensors:

::

   q)v:vector(x1;x2)
   q)train(m;v)     /no 2nd arg, i.e. empty target
   5

   q)count each input m
   500 500

   q)(::)~target m
   1b

   q)train(m;v;y)  /2-tensor input, single target tensor
   5

   q)(count each input m;count target m)
   500 500
   500

   q)train(m; (v;0 1); y)  /multiple indices
   5

   q)train(m; ((v;0);(v;1)); y)  /separate vector-index args
   5


Converting the vector to a dictionary:

::

   q)d:dict `a`b!vector v

   q)train(m;d;y)
   5

   q)train(m; (d;`a`b); y)
   5

   q)train(m; ((d;`a);(d;`b)); y)
   5

   q)(count each input m;count target m)
   500 500
   500



In the sample below, a simple linear model of 784 inputs (e.g. the flattened 28 x 28 MNIST image) and 10-column output of the 0-9 digits
is used, along with random data for images and labels:

::

   q)q:module enlist(`linear;784;10)
   q)m:model(q; loss`ce; opt(`sgd;q))

   q)x:tensor(`randn; 60000 784)
   q)y:tensor(`randint; 10; 60000)

   q)train(m;`batchsize`metrics;1000,`batchloss)
   q)train(m; x; y)
   60

   q)X:tensor(`randn; 10000 784)
   q)Y:tensor(`randint; 10; 10000)

   q)test(m;`batchsize;5000)
   q)test(m;X;Y)
   2

.. note::

   :func:`train` and :func:`test` expect the input(s) and target(s) sizes to match along the first dimension.

::

   q)m:model(module`identity; loss`mse; opt`sgd)

   q)train(m; 100#0; 10#1)
   'tensor size mismatch, 100 vs 10
     [0]  train(m; 100#0; 10#1)
          ^

The training and testing routines are currently not implemented to handle a mix of differently shaped inputs:
e.g. for transformers requiring a batch of input data, together with an attention mask -- the attention mask is a constant
whose shape would generate the above ``mismatch`` error. This type of model requires a lower level of input batching,
explicitly selecting the subset of inputs together with the attention mask.


Model data
**********

The :func:`train` and :func:`test` functions can be used to define training & testing options and data.
Separate functions :func:`data` and :func:`testdata` can also be used to retrieve and define model data,
along with functions that deal with training/testing inputs separately from targets,
:func:`input` and :func:`testinput`,
along with :func:`target` and :func:`testtarget`.

data, testdata
^^^^^^^^^^^^^^

These functions retrieve previously defined data as k array(s), for each tensor stored by the model.

.. function:: data(model) -> arrays
.. function:: testdata(model) -> arrays

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :return: Returns the training or test data defined for training/testing input as k array(s).

::

   q)m:model(module`identity; loss`mse; opt`sgd)
   q)data(m; 1 2 3; 10 11 12);

   q)data m
   1  2  3 
   10 11 12

   q)data(m; (.1 .2 .3; 3 5#1 2 3 where 3#5); 10 11 12);

   q)first data m
   0.1       0.2       0.3      
   1 1 1 1 1 2 2 2 2 2 3 3 3 3 3

   q)last data m
   10 11 12

The functions can also be used to define model inputs and targets, for both training and testing modes:

.. function:: data(model;inputs;targets) -> number of batches
.. function:: testdata(model;inputs;targets) -> number of batches

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :param tensor inputs: input(s) can be a single tensor or a set of tensor(s) or k arrays.
   :param tensor targets:  targets are typically a single tensor or k array, but may also be given as a set of tensors or arrays.
   :return: Returns the number of batches that will be processed given size of supplied data and the ``batchsize`` setting.

::

   q)m:model(module`identity; loss`mse; opt`sgd)
   q)test(m;`batchsize;5)
   q)testdata(m; til 100; 100#0 1)
   20

datasize, testsize
^^^^^^^^^^^^^^^^^^

These functions retrieve the number of training or test inputs and are sensitive to the particular batch if called
in a batch-by-batch context

.. function:: datasize(model) -> count of training samples
.. function:: testsize(model) -> count of test samples

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :return: Returns the overall count of training/test inputs and targets, or, if used in a batch-by-batch context, the count in the particular batch.

::

   q)m:model(module`relu; loss`ce;opt`sgd)    /dummy model
   q)train(m; `batchsize; 3)
   q)data(m; 1+til 10; 100+til 10)
   4

   q)datasize m
   10

   q)while[batch m; 0N!(datasize m; data m)]
   (3;(1 2 3;100 101 102))
   (3;(4 5 6;103 104 105))
   (3;(7 8 9;106 107 108))
   (1;(,10;,109))

input, testinput
^^^^^^^^^^^^^^^^

The functions used to retrieve model inputs take a single argument of a previously defined model:

.. function:: input(model) -> tensor or vector of tensors
.. function:: testinput(model) -> tensor or vector of tensors

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :return: Returns the data defined for training/testing input as tensor(s)

::

   q)m:model(module`identity; loss`mse; opt`sgd)
   q)train(m;`batchsize;16)
   q)input(m; x:101+til 64)
   4

   q)x~return input m
   1b

   q)while[batch m; show return input m]
   101 102 103 104 105 106 107 108 109 110 111 112 113 114 115 116
   117 118 119 120 121 122 123 124 125 126 127 128 129 130 131 132
   133 134 135 136 137 138 139 140 141 142 143 144 145 146 147 148
   149 150 151 152 153 154 155 156 157 158 159 160 161 162 163 164

The same retrieval functions can also be used to define inputs by supplying tensor(s) or array(s) along with the model:

.. function:: input(model;inputs) -> number of batches
.. function:: testinput(model;inputs) -> number of batches

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :return: Defines the inputs for training/testing and returns the number of batches given input size(s) and ``batchsize`` setting.

::

   q)m:model(module`identity; loss`mse; opt`sgd)
   q)train(m;`batchsize)
   32

   q)input(m; til 100)
   4

   q)til[100]~return input m
   1b

   q)x1:tensor(`randn;64 5)
   q)x2:tensor(`randn;64 1)

   q)input(m;x1;x2)
   2

   q)v:input m

   q)class v
   `vector

   q)size v
   64 5
   64 1

   q)(tensor x1;tensor x2)~return v
   1b


target, testtarget
^^^^^^^^^^^^^^^^^^

The :func:`target` function accepts arguments in the same way as :func:`input` but defines the target side of the model,
the known labels or classes of the inputs used to compare model outputs to compute training loss and testing accuracy.

.. function:: target(model) -> arrays
.. function:: testtarget(model) -> arrays

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :return: Returns the data defined for training/testing targets as k array(s), one per tensor.

.. function:: target(model;targets) -> number of batches
   :noindex:
.. function:: testtarget(model;targets) -> number of batches
   :noindex:

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :return: Defines the targets for training/testing and returns the number of batches given target size(s) and ``batchsize`` setting.

Shuffling data
**************

shuffle
^^^^^^^

.. function:: shuffle(tensors) -> null
.. function:: shuffle(tensors;dim) -> null
   :noindex:

   :param pointer tensors: an :doc:`api-pointer <pointers>` to a previously created tensor, vector or dictionary of tensors, or a model
   :param long dim: the dimension where the reordering will be applied, default is ``0`` (models currently only implemented to shuffle across 1st dimension).
   :return: Removes any batching in place for the tensor(s), shuffles all tensors across the same dimension using the same permutation, null return.

::

   q)x:tensor til 7
   q)tensor x
   0 1 2 3 4 5 6

   q)shuffle x
   q)tensor x
   2 0 4 6 3 5 1

   q)use[x]tensor(3 4#til 12)
   q)shuffle x
   q)tensor x
   0 1 2  3 
   8 9 10 11
   4 5 6  7 

   q)shuffle(x;1)
   q)tensor x
   2  3  0 1
   10 11 8 9
   6  7  4 5

Collections of tensors are shuffled together across the same dimension:

::

   q)d:dict`a`b!(3 4#til 12;1 2 3)
   q)dict d
   a| 0 1 2  3  4 5 6  7  8 9 10 11
   b| 1         2         3        

   q)shuffle d
   q)dict d
   a| 8 9 10 11 4 5 6  7  0 1 2  3 
   b| 3         2         1        


.. note::

   Shuffling a tensor or set of tensors also has the side effect of removing any batching in place and restoring the tensor(s) to their full size in order to shuffle all elements.

::

   q)x:tensor til 7
   q)tensor x
   0 1 2 3 4 5 6

   q)batch(x;3;0)
   q)tensor x
   0 1 2

   q)shuffle x
   q)tensor x
   5 4 6 1 2 3 0


unshuffle
^^^^^^^^^

When the training data defined for a model is shuffled, 
an internal permutation vector of indices is maintained which allows the original order to be restored via the :func:`unshuffle` function.

.. function:: unshuffle(model) -> null

   :param pointer model: an :doc:`api-pointer <pointers>` to a model.
   :return: Restores shuffled training data to its orignal order, null return.

::

   q)m:model(module`identity; loss`mse; opt`sgd)
   q)train(m;til 7;100+til 7)
   1

   q)data m
   0   1   2   3   4   5   6  
   100 101 102 103 104 105 106

   q)shuffle m
   q)data m
   2   3   0   1   5   6   4  
   102 103 100 101 105 106 104

   q)shuffle m
   q)data m
   1   4   3   0   6   2   5  
   101 104 103 100 106 102 105

   q)unshuffle m

   q)data m
   0   1   2   3   4   5   6  
   100 101 102 103 104 105 106


.. _model-batches:

Batching data
*************

The higher level routines for `running training and testing calculations <run>` over batches of data handle the batching internally,
but it may also be useful to process each batch explicity using the model structures.
(see :ref:`tensor batches <tensor-batch>` for more batching routines that operate on tensors directly without a model and their associated structures.)

batch, testbatch
^^^^^^^^^^^^^^^^

.. function:: batch(model) -> flag
.. function:: testbatch(model) -> flag

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :return: Returns ``true`` if there remains a batch of data to process, ``false`` if all batches have been processed. When ``false`` is returned, the underlying tensors are reset to their original size.

::

   q)m:model(module`identity; loss`mse; opt`sgd)
   q)train(m;`batchsize;3)
   q)train(m; til 10; 10+til 10)
   4

   q)while[batch m; show input m]
   0 1 2
   3 4 5
   6 7 8
   ,9

restore, testrestore
^^^^^^^^^^^^^^^^^^^^

The model batching will restore the full size of the underlying tensors once the final batch has been encountered;
i.e. when the call to the batching function returns ``false``, the tensors are reset to their full size.
But if the batching is stopped before the final batch, it may be necessary to restore the tensors to their full size explicitly.

.. function:: restore(model) -> restore full size of training tensors along first dimension
.. function:: testrestore(model) -> restore full size of test tensors along first dimension

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :return: Resets any subset of training/testing data and restores all data tensors to full size, returning size of first dimension.

::

   q)m:model(module`identity; loss`mse; opt`sgd)
   q)train(m;`batchsize;3)
   q)train(m; til 10; 10+til 10)
   4

   q)batch m
   1b

   q)input m
   0 1 2

   q)restore m
   10

   q)input m
   0 1 2 3 4 5 6 7 8 9

batchinit, testinit
^^^^^^^^^^^^^^^^^^^

Batches are [re]initialized for models in three steps:

- restore tensors to their full size after any batching.
- shuffle training data if the training option for ``shuffle`` is set ``true``.
- reset internal structures to store the :ref:`metrics <model-metrics>` defined for the model.

The :func:`batchinit` and :func:`testinit` functions perform these steps as part of the normal :func:`run`/:func:`testrun` cycle,
but are also available as stand alone api functions to restore the model state after an interuption or error during model training/testing:

.. function:: batchinit(model) -> number of training batches
.. function:: testinit(model) -> number of test batches

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :return: Restores any batched data tensors to full size, shuffle if required, reset metrics and return number of batches.

::

   q)m:model(module`identity; loss`mse; opt`sgd)

   q)batchinit m  / no training data defined
   -1

   q)train(m;`batchsize;3)
   q)train(m;til 7;100+til 7)
   3

   q)batch m
   1b

   q)data m
   0   1   2  
   100 101 102

   q)batchinit m
   3

   q)data m
   0   1   2   3   4   5   6  
   100 101 102 103 104 105 106



.. _model-run:

Running calculations
********************

Functions :func:`run` and :func:`testrun` manage the full training and testing run through a dataset:
First the :ref:`model options <model-options>` determine the batch size and whether the training data is to be shuffled.
Then for each batch, the :ref:`forward <model-forward>` calculation takes an input batch and calculates model outputs.
The model outputs, together with the target batch are used to calculate the :ref:`loss <model-loss>`.
From the calculated loss tensor, the :ref:`backward <model-backward>` call provides the gradients.
Model :ref:`training <model-train>` options indicate any gradient clipping required before the call to the optimizer to 
:ref:`update <model-step>` the trainable parameters. Finally, :ref:`metrics <model-metrics>` specified in the options
are calculated and stored for the batch.

After the calculations and metrics for each batch are completed, the results are returned as k arrays or tensors,
as a list or dictionary, depending on the :ref:`options <model-options>` for output.

run
^^^

.. function:: run(model) -> metrics

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :return: Returns required metric(s) as k array(s) or tensor(s) with the option to return as a list or dictionary with metric names as keys.

An alternate form allows for specification of inputs and targets:

.. function:: run(model;inputs;targets) -> metrics
   :noindex:

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :param tensor inputs: tensor(s) or k array(s) with model inputs, e.g. images for vision models, sequences for transformers, etc.
   :param tensor targets: tensor(s) or array(s) with targets, e.g. classes for images or next token in sequences.
   :return: Returns required metric(s) as k array(s) or tensor(s) with the option to return as a list or dictionary with metric names as keys.

::

   q)q:module enlist(`linear;784;10)
   q)m:model(q; loss`ce; opt(`sgd;q))

   q)x:tensor(`randn; 60000 784)
   q)y:tensor(`randint; 10; 60000)

   q)train(m;`batchsize`dictionary`metrics;(1000;1b;`loss`batchloss))
   q)train(m; x; y)
   60

   q)r:run m

   q)r
   loss     | 2.44
   batchloss| 2.45 2.43 2.43 2.45 2.48 2.46 2.44 2.43 2.44 2.43 2.44 2.43 2.46 2..

   q)r:run(m;x;y)

   q)r
   loss     | 2.42
   batchloss| 2.42 2.41 2.41 2.43 2.45 2.44 2.42 2.41 2.42 2.41 2.42 2.41 2.43 2..

testrun
^^^^^^^

:func:`testrun` uses the same syntax as :func:`run` to calculate metrics for the model using the test data.
There are no test options to shuffle the data or clip gradients because :func:`testrun` turns on
evaluation mode for the underlying module and turns off the calculation of gradients.
Batch size is typically larger than the training batch size since the lack of gradients means more
test data will fit in available memory for each batch.

.. function:: testrun(model) -> metrics

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :return: Returns required metric(s) as k array(s) or tensor(s) with the option to return as a list or dictionary with metric names as keys.

An alternate form allows for specification of inputs and targets:

.. function:: testrun(model;inputs;targets) -> metrics
   :noindex:

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously created :ref:`model <model>`.
   :param tensor inputs: tensor(s) or k array(s) with model inputs, e.g. images for vision models, sequences for transformers, etc.
   :param tensor targets: tensor(s) or array(s) with targets, e.g. classes for images or next token in sequences.
   :return: Returns required metric(s) as k array(s) or tensor(s) with the option to return as a list or dictionary with metric names as keys.

