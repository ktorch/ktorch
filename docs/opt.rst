.. _optimizer:

Optimizers
==========

The k-api implements the optimizers in the PyTorch C++ libraries, a subset of all the optimizers in the python interface.
The most commonly used optimizers are stochastic gradient descent(SGD) & Adam/AdamW;
the limited memory Broyden–Fletcher–Goldfarb–Shanno algorithm (L-BFGS) is implemented but rarely used and more likely to have edge conditions that have not been widely tested.
The k-api also implements a newer algorithm, the Layer-wise Adaptive Moments optimizer for Batch training(LAMB) which is not part of PyTorch as of version 1.10 but has been used to speed up training of language and vision models by using much larger batch sizes.

- `adagrad <https://pytorch.org/docs/stable/generated/torch.optim.Adagrad.html>`_: adaptive gradient descent (different learning rates for each neuron).
- `adam <https://pytorch.org/docs/stable/generated/torch.optim.Adam.html>`_: adaptive moment estimation, using a ratio of decaying average of past gradients divided by the square root of the squared average(variance) of the gradients.
- `adamw <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html>`_: adam with improved handling of weight decay.
- `lbfgs <https://pytorch.org/docs/stable/generated/torch.optim.LBFGS.html>`_: limited-memory Broyden–Fletcher–Goldfarb–Shanno algorithm.
- `rmsprop <https://pytorch.org/docs/stable/generated/torch.optim.RMSprop.html>`_: a version of adagrad where learning rate is the exponential average of the gradients instead of the cumulative sum of squared gradients.
- `sgd <https://pytorch.org/docs/stable/generated/torch.optim.SGD.html>`_: stochastic gradient descent, with optional momentum.

- `lamb <https://developer.nvidia.com/blog/pretraining-bert-with-layer-wise-adaptive-learning-rates>`__: layer-wise adaptive moments for batch training, used with larger batch sizes, augments adam-like approach with a trust ratio of weight norm divided by gradient norm. (see more detail on `options <lamb>`).

An optimizer is defined in the k api using the :ref:`opt() <optinit>` function, which expects the name of the optimizer,
along with the parameters to be optimized, then a set of optimizer-specific settings, e.g. learning rate, momentum, etc.

.. _optinit:

.. function:: opt(name; parameters; options..) -> optimizer
.. function:: opt(name; group; parameters; options..) -> optimizer

   :param symbol name: e.g. ```adamw`` or ```sgd``
   :param long group: optional :ref:`parameter group <optgroups>` number.
   :param module parameters: the most common way to define the parameters to be optimized is to supply an :doc:`api-pointer <pointers>` to an allocated module whose parameters will be used, but it is also possible to supply an individual tensor, a vector or dictionary of tensors and a full model. (see section on :ref:`specifying parameters <optparms>` for more).
   :param long/double/symbol options: options may be specified positionally, or in name-value pairs or via a k dictionary, or a combination of both.

   :return: An :doc:`api-pointer <pointers>` to a new optimizer.

At minimum, the optimizer definition requires the name of the optimizer type to be created; other parts of the definition can be supplied later.


::

   q)o:opt`sgd

   q)opt o
   optimizer| `sgd
   options  | ,`lr`momentum`dampening`decay`nesterov!(0.01;0f;0f;0f;0b)
   parms    | +`parmgroup`pointer`module`name`size!(`long$();`long$();`symbol$()..

   q)free o


   q)m:module enlist(`linear;128;10)
   q)to(m; `cuda)

   q)o:opt(`sgd; m; .1; .99) / learning rate of .1, momentum .99

   q)opt o
   optimizer| `sgd
   options  | ,`lr`momentum`dampening`decay`nesterov!(0.1;0.99;0f;0f;0b)
   parms    | +`parmgroup`pointer`module`name`size!(0 0;63102096 63425808;`linea..

.. note::

   Parameters or module(s) that are intended to run on an available CUDA device should be moved to the gpu before assigning them to an optimizer -- the optimizer will create its buffers on the device where the parameters assigned to it reside.

.. _opthelp:

Help
^^^^

A table of optimizers and their options is available via the :ref:`help() <opthelp>` function:

::

   q)help`optimizer
   optimizer pytorch               options                                      ..
   -----------------------------------------------------------------------------..
   adagrad   "torch.optim.Adagrad" `lr`lrdecay`decay`init`eps!0.01 0 0 0 1e-10  ..
   adam      "torch.optim.Adam"    `lr`beta1`beta2`eps`decay`amsgrad!(0.001;0.9;..
   adamw     "torch.optim.AdamW"   `lr`beta1`beta2`eps`decay`amsgrad!(0.001;0.9;..
   lamb      ""                    `lr`beta1`beta2`eps`decay`unbiased`globalnorm..
   lbfgs     "torch.optim.LBFGS"   `lr`iter`eval`gradtol`changetol`history`searc..
   rmsprop   "torch.optim.RMSprop" `lr`alpha`eps`decay`momentum`centered!(0.01;0..
   sgd       "torch.optim.SGD"     `lr`momentum`dampening`decay`nesterov!(0.01;0..

If the help function is given the name of an individual optimizer, it returns the dictionary of available options with default values:

::

   q)help`adamw
   lr     | 0.001
   beta1  | 0.9
   beta2  | 0.999
   eps    | 1e-08
   decay  | 0.01
   amsgrad| 0b

Options
^^^^^^^

Specifying positional options:

::

   q)m:module enlist(`linear;128;10)

   q)o:opt(`adam; m; .0002; .85; .99; 1e-8; .02)  /specify the first 5 positional options

   q)exec first options from opt o
   lr     | 0.0002
   beta1  | 0.85
   beta2  | 0.99
   eps    | 1e-08
   decay  | 0.02
   amsgrad| 0b

Positional options and name-value pair(s) can be mixed if the positional options are specified first,
followed by name-value pair(s) or a dictionary:

::

   q)o:opt(`adam; m; .0002; `decay,.02) /learning rate py position, weight decay by name

   q)exec first options from opt o
   lr     | 0.0002
   beta1  | 0.9
   beta2  | 0.999
   eps    | 1e-08
   decay  | 0.02
   amsgrad| 0b

Options can be supplied only via name-value pairs or a dictionary:

::

   q)o:opt(`adam; m; `lr`decay!.01 .05)

   q)exec first options from opt o
   lr     | 0.01
   beta1  | 0.9
   beta2  | 0.999
   eps    | 1e-08
   decay  | 0.05
   amsgrad| 0b

   q)o:opt(`sgd; m; ((`lr;.01);(`momentum;.9))) /list of name-value pairs

   q)o:opt(`sgd; m; (`lr,.01),(`momentum,.9))   /alternate name-value form

   q)exec first options from opt o
   lr       | 0.01
   momentum | 0.9
   dampening| 0f
   decay    | 0f
   nesterov | 0b

.. _optparms:

Specifying parameters
^^^^^^^^^^^^^^^^^^^^^

The second argument of the :ref:`opt() <optinit>` function is typically a module (which contains all the submodules of a model).
But the parameters may be specified with other collections of tensors:

Single tensor:

::

   q)t:tensor(1 2 3e; `grad)
   q)o:opt(`sgd; t)

Vector of tensors:

::

   q)v:vector(1 2 3e; 1 1.2 9e; 77 78e)
   q)gradflag(v;1b)

   q)o:opt(`sgd;v)

Vector with a single index or a list of indices:

::

   q)o:opt(`sgd;(v;1))

   q)o:opt(`sgd; (v;2 0))

Tensor dictionary:

::

   q)d:dict `a`b!(1 2 3e;4 5e)
   q)gradflag(d;1b)

   q)o:opt(`sgd; d)           /dictionary

   q)o:opt(`sgd; (d;`b))      /dictionary with single key

   q)o:opt(`sgd; (d;`b`a))    /with list of keys

   q)opt[o]`parms
   parmgroup pointer  module   name size
   -------------------------------------
   0         83526432 parmdict b    2   
   0         83398432 parmdict a    3   

Module & child modules:

::

  q)q:module seq(`sequential; (`linear;`a;128;64); `relu`relu1; (`linear;`b;64;10); `relu`relu2)

  q)names q
  `a`relu1`b`relu2

  q)o:opt(`sgd;q)  /typical case, specifying all parameters in a module

  q)o:opt(`sgd;(q;0))   /specifying by index

   q)opt[o]`parms
   parmgroup pointer  module name     size  
   -----------------------------------------
   0         83362896 linear a.weight 64 128
   0         83423968 linear a.bias   ,64   


   q)o:opt(`sgd; (q;`b`a))   /specifying by submodule name(s)

   q)opt[o]`parms
   parmgroup pointer  module name     size  
   -----------------------------------------
   0         83376160 linear b.weight 10 64 
   0         83460704 linear b.bias   ,10   
   0         83362896 linear a.weight 64 128
   0         83423968 linear a.bias   ,64   



Get optimizer definition
^^^^^^^^^^^^^^^^^^^^^^^^

The same :ref:`opt()<optinit>` function that is used to define an optimizer can be used to retrieve the definition.
In this kind of call the created optimizer is used as the 1st argument rather than an optimizer name.

.. _optdef:

.. function:: opt(optimizer) -> dictionary
.. function:: opt(optimizer;all) -> dictionary

   :param pointer optimizer: an :doc:`api-pointer <pointers>` to an allocated optimizer.
   :param boolean all: an optional flag set true to return all options and set false to return only non-default options. If not specified, the flag uses the :ref:`global setting <settings>` for :ref:`show all options <alloptions>`.

   :return: A k dictionary of optimizer type, options (a list of dictionaries, one per parameter group) and a table describing the parameters managed by the optimizer.

::

   q)q:module seq(`sequential; (`linear;`a;128;10); `relu`b)

   q)o:opt(`adamw; q; .0002)

   q)show d:opt o
   optimizer| `adamw
   options  | ,`lr`beta1`beta2`eps`decay`amsgrad!(0.0002;0.9;0.999;1e-08;0.01;0b)
   parms    | +`parmgroup`pointer`module`name`size!(0 0;80428176 80752400;`linea..

   q)first d`options
   lr     | 0.0002
   beta1  | 0.9
   beta2  | 0.999
   eps    | 1e-08
   decay  | 0.01
   amsgrad| 0b

   q)d`parms
   parmgroup pointer  module name     size  
   -----------------------------------------
   0         80428176 linear a.weight 10 128
   0         80752400 linear a.bias   ,10   

The optimizer definition retrieved via :ref:`opt() <optdef>` cannot be used directly to create a new optimizer, but the options can be reused.

In the example below, two linear modules are used in the creation of an ``adamw`` optimizer with two :ref:`parameter groups <optgroups>`:


::

   q)m0:module enlist(`linear;128;64)
   q)m1:module enlist(`linear; 64;10)

   q)o:opt`adamw
   q)opt(o; 0; m0; .01; `decay,.01)
   q)opt(o; 1; m1; .02; `decay,.04)

   q)d:opt o
   q)d`options
   lr   beta1 beta2 eps   decay amsgrad
   ------------------------------------
   0.01 0.9   0.999 1e-08 0.01  0      
   0.02 0.9   0.999 1e-08 0.04  0      

Now create a new optimizer, copying the previous options:

::

   q)n:opt d`optimizer
   q){opt(x; y; (); z)}[n]'[til count d`options; d`options];

The new optimizer has matching options for each parameter group, but no parameters defined:

::

   q)opt[n]~'d
   optimizer| 1
   options  | 1
   parms    | 0

The same modules used in the first optimizer can be added to the newer instance so that the two definitions match:

::

   q)opt(n;0;m0)
   q)opt(n;1;m1)

   q)opt[n]~'d
   optimizer| 1
   options  | 1
   parms    | 1

The two optimizers now have the same definition. If the first optimizer had undergone one or more update steps, then more state information would be required to recreate the optimizer, see :ref:`optimizer state <optstate>`.

Storage and other information
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These generic k-api functions return basic attributes of the optimizer and the size and storage of tensors associated with it:

.. function:: class(optimizer) -> optimizer symbol
.. function:: objtype(optimizer) -> optimizer type
.. function:: size(optimizer) -> number of parameters assigned to the optimizer
.. function:: tensorcount(optimizer) -> number of tensor buffers currently in the optimizer state
.. function:: elements(optimizer) -> count of elements in the optimizer buffers
.. function:: bytes(optimizer) -> total bytes of the optimizer buffers

When an optimizer is initialized, it may have no tensors stored or associated with it:

::

   q)o:opt(`adamw; ())

   q){x!x@\:y}[`class`objtype`size`tensorcount`elements`bytes;o]
   class      | `optimizer
   objtype    | `adamw
   size       | 0
   tensorcount| 0
   elements   | 0
   bytes      | 0

After module parameters are defined, there are still no buffers initialized until the first step is run:

::

   q)m:module enlist(`linear;64;10)

   q)opt(o;m)

   q){x!x@\:y}[`class`objtype`size`tensorcount`elements`bytes;o]
   class      | `optimizer
   objtype    | `adamw
   size       | 2
   tensorcount| 0
   elements   | 0
   bytes      | 0

After an optimization step, for ``adamw``, buffers used for the average of the gradient and the squared gradient are created:

::

   q)backward z:ce(y:forward(m; 20 64#1e); 20?10); step o

   q){x!x@\:y}[`class`objtype`size`tensorcount`elements`bytes;o]
   class      | `optimizer
   objtype    | `adamw
   size       | 2
   tensorcount| 4
   elements   | 1302
   bytes      | 5216

   q)2*(1*1 8)+65*10*1 4  /buffers for 10 x 54 wt and 10-element bias + step counter
   1302 5216

.. _optgroups:

Parameter groups
^^^^^^^^^^^^^^^^

An optimizer's parameters can be divided into groups with different settings for each group.
If no groups are specified during an optimizer definition, all options and parameters are defined in the first group.

The optimizer can be initialized without any parameters to start:

::

   q)o:opt(`sgd; (); .1; .9)  / learning rate of .1, momentum of .9 as defaults

Parameters can be added incrementally to separate groups with different settings for each:

::

   q)m0:module enlist(`linear;128;64)
   q)m1:module enlist(`linear;64;10)

   q)opt(o; 0; m0)
   q)opt(o; 1; m1; .05; .95) / different learning rate & momentum for 2nd group

   q)d:opt o

   q)d`options
   lr   momentum dampening decay nesterov
   --------------------------------------
   0.1  0.9      0         0     0       
   0.05 0.95     0         0     0       

   q)d`parms
   parmgroup pointer  module name     size  
   -----------------------------------------
   0         83836176 linear 0.weight 64 128
   0         83512464 linear 0.bias   ,64   
   1         83843488 linear 1.weight 10 64 
   1         83842736 linear 1.bias   ,10   

Parameter groups must be defined consecutively.

::

   q)o:opt(`sgd;();.1)  /implicit group 0
   q)opt(o;1;();.01)    /group 1's learning rate
   q)opt(o;2;();.001)   /group 2..

   q)exec options from opt o
   lr    momentum dampening decay nesterov
   ---------------------------------------
   0.1   0        0         0     0       
   0.01  0        0         0     0       
   0.001 0        0         0     0       

   q)opt(o;4;();.0001)
   'opt: group 4 invalid, cannot be greater than number of groups(3)
     [0]  opt(o;4;();.0001)
          ^

.. _optstate:

Parameter state
^^^^^^^^^^^^^^^

The optimizer definition retrieved via :ref:`opt() <optdef>` can be used to create a new optimizer with the same options,
but cannot be used directly to recreate the set of parameters managed by the optimizers or the state of the buffers after one or more update steps.

An optimizer is typically associated with a module (or set of modules), but the PyTorch optimizer design deliberately makes no direct association
to the module(s), only their underlying parameters. An optimizer manages a set of parameter tensors and stores no other information
about them.  This allows for very general use of a PyTorch optimizer, but complicates recreating the saved state of an optimizer
(see `this tutorial <https://pytorch.org/tutorials/beginner/saving_loading_models.html>`_ for how the state is handled in python).

The k-api :ref:`state() <optstate>` function attempts to link any available module information with the optimizer when the state is retrieved to allow for easier restoration.


.. function:: state(optimizer) -> dictionary
.. function:: state(optimizer;all) -> dictionary

   :param pointer optimizer: an :doc:`api-pointer <pointers>` to an allocated optimizer.
   :param boolean all: an optional flag set true to return all options and set false to return only non-default options. If not specified, the flag uses the :ref:`global setting <settings>` for :ref:`show all options <alloptions>`.

   :return: A k dictionary of optimizer type, options (a list of dictionaries, one per parameter group) and a table describing the parameters managed by the optimizer. The output of state is similar to the output of :ref:`opt() <optdef>` except the parameter table includes a final column of all the buffers updated by the optimizer at each step.

In the example below, an ``adamw`` optimizer is created to manage the parameters of a sequential module:

::

   q)q:module seq(`sequential; (`linear;`a;128;64); `relu; (`linear;`b;64;10); `relu)

   q)o:opt(`adamw; q; .0002)

   q)s:state o

   q)s`parms
   parmgroup pointer  module name     size   buffers        
   ---------------------------------------------------------
   0         86044448 linear a.weight 64 128 (`symbol$())!()
   0         86045456 linear a.bias   ,64    (`symbol$())!()
   0         86050448 linear b.weight 10 64  (`symbol$())!()
   0         86051344 linear b.bias   ,10    (`symbol$())!()

The buffers are not initialized until the optimizer performs a step.
Then each parameter is initialized with an optimzer state, a set of buffers used to update the parameter once the gradients have been calculated:

::

   q)y:forward(q; 20 128#1e)
   q)z:ce(y; 20?10)
   q)backward z

   q)step o

   q)s:state o
   q)s`parms
   parmgroup pointer  module name     size   buffers                            ..
   -----------------------------------------------------------------------------..
   0         86044448 linear a.weight 64 128 `step`exp_avg`exp_avg_sq`max_exp_av..
   0         86045456 linear a.bias   ,64    `step`exp_avg`exp_avg_sq`max_exp_av..
   0         86050448 linear b.weight 10 64  `step`exp_avg`exp_avg_sq`max_exp_av..
   0         86051344 linear b.bias   ,10    `step`exp_avg`exp_avg_sq`max_exp_av..

   q)last s .`parms`buffers
   step          | 1
   exp_avg       | 0 0 0 0.01468916 0 0.0003890909 0 0 0 0e
   exp_avg_sq    | 0 0 0 2.157714e-05 0 1.513918e-08 0 0 0 0e
   max_exp_avg_sq| ::

.. _optrestor:

Restoring state
^^^^^^^^^^^^^^^

An optimizer can be restored from a previously saved state, along with the module(s) used to supply the optimizer with parameters.
The same :ref:`opt() <optinit>` function is used, but is supplied with two different arguments: a state dictionary and a module.

.. function:: opt(state; module) -> optimizer

   :param dictionary state: a k dictionary saved from the ouput of the :ref:`state() <optstate>` call.
   :param pointer module: a re-created module whose parameters are to be managed by the optimizer

   :return: An :doc:`api-pointer <pointers>` to a new optimizer.

::

   q)q:module seq(`sequential; (`linear;`a;128;64); `relu; (`linear;`b;64;10); `relu)

   q)to(q;`cuda)              / move to gpu

   q)o:opt(`adamw; q; .0002)  / then define optimizer from module q

Run at least one optimization step to initialize buffers used to track steps, gradient averages, etc.

::

   q)y:forward(q; 20 128#1e)  /forward calc on dummy input
   q)z:ce(y; 20?10)           /calculate cross-entropy loss with random targets
   q)backward z               /calculate gradients
   q)step o                   /run an optimization step to initialize buffers

Save state to file and erase current instances of the module and optimizer:

::

   q)`:/tmp/q set state q    /save module state to file
   `:/tmp/q

   q)`:/tmp/o set s:state o  /save optimizer state
   `:/tmp/o

   q)free[]                  /free all pytorch objects

Restore objects from file and check if state matches:

::

   q)q:module get`:/tmp/q  /re-create module 
   q)to(q;`cuda)           /move to gpu

   q)o:opt(get`:/tmp/o;q)  /re-create optimizer

   q)s~'state o  /compare current state
   optimizer| 1
   options  | 1
   parms    | 0

   q)s[`parms]~''state[o]`parms                /state matches except for memory pointers
   parmgroup pointer module name size buffers
   ------------------------------------------
   1         0       1      1    1    1      
   1         0       1      1    1    1      
   1         0       1      1    1    1      
   1         0       1      1    1    1      


Managing multiple modules
^^^^^^^^^^^^^^^^^^^^^^^^^

The most common case is for an optimizer to be created with a single module whose parameters will be updated by the optimizer whenever the :ref:`step() <optstep>` call occurs.  But there are cases where an optimizer may manage the parameters from more than one module or other collections of tensor parameters.

In the case of multiple modules/tensors, the k-api adds a container module as part of the optimizer interface that maintains all the objects that were used to add parameters to the optimizer.  This container module can be used to save the full state of the optimizer and recreate all contributing  objects.

For an example, start with two sequential modules:

::

   q)m0:module seq(`sequential`A; (`linear;`fc;128;64); `relu`actfn)
   q)m1:module seq(`sequential`B; (`linear;`fc; 64;10); `relu`actfn)

Add both to an ``adamw`` optimizer:

::

   q)o:opt`adamw
   q)opt(o; 0; m0; .001)
   q)opt(o; 1; m1; .0001)

   q)opt[o]`parms
   parmgroup pointer  module name        size  
   --------------------------------------------
   0         84442256 linear A.fc.weight 64 128
   0         84794560 linear A.fc.bias   ,64   
   1         84805488 linear B.fc.weight 10 64 
   1         84806336 linear B.fc.bias   ,10   

Run one step with 1's as input so that the optimizer state will include buffers maintaining gradient and squared gradient averages:

::

   q)y:forward(m0;20 128#1e)
   q)use[y]forward(m1; y)
   q)backward z:ce(y; 20?10)
   q)step o

   q)s:state o

   q)s`parms
   parmgroup pointer  module name        size   buffers                         ..
   -----------------------------------------------------------------------------..
   0         84442256 linear A.fc.weight 64 128 `step`exp_avg`exp_avg_sq`max_exp..
   0         84794560 linear A.fc.bias   ,64    `step`exp_avg`exp_avg_sq`max_exp..
   1         84805488 linear B.fc.weight 10 64  `step`exp_avg`exp_avg_sq`max_exp..
   1         84806336 linear B.fc.bias   ,10    `step`exp_avg`exp_avg_sq`max_exp..

To save the full state of the optimizer will also require saving the modules (and any other tensor parameters) that went into the optimizer's parameter group(s).  
Using the :func:`module` with an optimizer pointer will return this container module maintained by the optimizer.

::

   q)m:module o      /get container module

   q)childnames m    /check names of direct children
   `A`B

   q)names m         /check full set of child modules at all depths
   `A`A.fc`A.actfn`B`B.fc`B.actfn

Save the container module and optimizer state to file:

::

   q)`:/tmp/m set state m    /save container module
   `:/tmp/m

   q)`:/tmp/o set s:state o  /save optimizer state
   `:/tmp/o

   q)free[]    /free all allocated pytorch objects

Recreate from saved files:

:: 

   q)m:module get`:/tmp/m     /recreate container module
   q)o:opt(get`:/tmp/o; m)    /recreate optimizer w'most recent state

   q)s~'state o
   optimizer| 1
   options  | 1
   parms    | 0

   q)s[`parms]~''state[o]`parms
   parmgroup pointer module name size buffers
   ------------------------------------------
   1         0       1      1    1    1      
   1         0       1      1    1    1      
   1         0       1      1    1    1      
   1         0       1      1    1    1      

All but the active tensor pointers are the same (new parameter tensors will have different pointers after being recreated)


.. _lr:

Learning rate
^^^^^^^^^^^^^

Optimizer options can be set & reset via the main :ref:`opt() <optinit>` function by supplying the optimizer,
optional group and any options, omitting the parameter specification:

::

  q)m:module enlist(`linear;128;64)
  q)o:opt(`sgd;m;.1;.8)

  q)first opt[o]`options
  lr       | 0.1
  momentum | 0.8
  dampening| 0f
  decay    | 0f
  nesterov | 0b

Reset learning rate & momentum:

::

  q)opt(o; (); .002; .9)
  q)first opt[o]`options
  lr       | 0.002
  momentum | 0.9
  dampening| 0f
  decay    | 0f
  nesterov | 0b

The :ref:`lr() <lr>` function is a simpler way to query and set only the learning rate:

.. function:: lr(optimizer) -> learning rates
.. function:: lr(optimizer;rates) -> null

   :param pointer optimizer: an :doc:`api-pointer <pointers>` to an allocated optimizer.
   :param long rates: an optional scalar or vector of rates (one per parameter group).

   :return: If no rates supplied, returns the learning rate(s) currently defined, else sets the rate(s) defined and returns null.

In the example below, an optimizer is defined with two parameter groups:

::

   q)m0:module enlist(`linear;128;64)
   q)m1:module enlist(`linear; 64;10)

   q)o:opt`sgd
   q)opt(o; 0; m0;  .01; .9)   /learning rate .01,  momentum .9
   q)opt(o; 1; m1; .001; .99)  /learning rate .001, momentum .99

The learning rate function will return the two defined learning rates:

::

  q)lr o
  0.01 0.001

The :ref:`lr() <lr>` function can also be used to reset the rates:

::

   q)lr(o; .9*lr(o))

   q)opt[o]`options
   lr     momentum dampening decay nesterov
   ----------------------------------------
   0.009  0.9      0         0     0       
   0.0009 0.99     0         0     0       


Resetting gradients
^^^^^^^^^^^^^^^^^^^

Each time gradients are calculated (via a ``backward`` call), the gradients are accumulated for all parameters involved in the chain of calculations.
There are occasions where it is useful to accumluate several gradients before applying an update based on their total (or average, etc.),
but the typical sequence is to zero out any accumulated gradients, run the backward calculations, then apply the update based on the gradient via a :ref:`step() <optstep>` call.

.. function:: zerograd(optimizer) -> null

   :param pointer optimizer: an :doc:`api-pointer <pointers>` to an allocated optimizer (may also be callad with a :doc:`module <modules>` or :doc:`model <model>`, as well as a :doc:`vector <vectors>`, :doc:`dictionary <dictionaries>` or an individual tensor)

In the example below, the optimizer manages a single tensor:

::

   q)x:tensor(.5 1 4.0; `grad)
   q)y:tensor  1 2 3.0
   q)o:opt(`sgd; x; 1.0) /learning rate=1, i.e. use gradient w'out scaling

The mean-squared loss is calculated and the backward call calculates the gradients:

::

   q)backward z:mse(x;y)

   q)grad x
   -0.3333333 -0.6666667 0.6666667

   q)step o
   q)tensor x
   0.8333333 1.666667 3.333333

The gradients must be reset to zero so that the next increment will reflect only the most recent loss calculation:

::

   q)zerograd o
   q)grad x
   0 0 0f

   q)use[z]mse(x;y); backward z
   q)grad x
   -0.1111111 -0.2222222 0.2222222

   q)step o
   q)tensor x
   0.9444444 1.888889 3.111111

.. _optnograd:

A newer method of resetting gradients was added to version ``1.11.0`` via a flag for the `zero_grad <https://pytorch.org/docs/stable/generated/torch.optim.Optimizer.zero_grad.html>`_ method for modules and optimizers but has not been added for optimizers in the c++ interface.
The k api implements this option as a separate function, :func:`nograd`, rather than a flag.

.. function:: nograd(optimizer) -> null

   :param pointer optimizer: an :doc:`api-pointer <pointers>` to an allocated optimizer (may also be callad with a :doc:`module <modules>` or :doc:`model <model>`, as well as a :doc:`vector <vectors>`, :doc:`dictionary <dictionaries>` or an individual tensor)

   :result: sets the gradients to an undefined tensor, null return.

.. _optstep:

Step
^^^^

.. function:: step(optimizer) -> null

   :param pointer optimizer: an :doc:`api-pointer <pointers>` to an allocated optimizer (may also be called with a :doc:`module <modules>` or :doc:`model <model>`)

   :return: The optimizer calculates and applies an update to its set of parameters, returns null.

For this example, create a single tensor and have it managed by a ``sgd`` optimizer:

::

   q)t:tensor(.5 2 4e; `grad)
   q)o:opt(`sgd; t; 1.0)

The typical process when calling the update step of an optimizer:

- zero out any previous gradients
- calculate the loss
- run the backward calculation to get parameter gradients
- call the optimizer step function

Here the mean-squared loss is calculated comparing the tensor to a target of ``1 2 3`` until the tensor is brought close to the target:

::

   q)f:{zerograd o; backward l:mse(t; 1 2 3e); step o; (return l;tensor t)}

   q)`loss`tensor!/:1_(.00001<first@) f\0w  /run update steps until error below .00001
   loss       tensor          
   ---------------------------
   0.41667    0.83333 2 3.3333
   0.046296   0.94444 2 3.1111
   0.005144   0.98148 2 3.037 
   0.00057156 0.99383 2 3.0123
   6.3507e-05 0.99794 2 3.0041
   7.0566e-06 0.99931 2 3.0014

Some of the above steps are incorporated into higher level routines that perform the whole sequence, see :doc:`models <model>` and :doc:`training steps <train>`.

.. _lamb:

LAMB
^^^^

The main feature of the different variations of the LAMB optimizer is that it multiplies the update from Adam-style optimizers, an update using the ratio of the running mean of the gradient divided by the square root of the squared gradient, by a ``trust ratio`` that divides the norm of the parameter weights by the norm of the gradient or adjusted gradient that is used for the update.
The `original paper <https://arxiv.org/abs/1904.00962>`_
that introduced the optimizer found that its update steps allowed for much larger batch sizes and quicker training.

The version of the LAMB optimizer implemented for the k api is based on ``NVLAMB``, a version proposed by NVIDIA and `detailed here <https://developer.nvidia.com/blog/pretraining-bert-with-layer-wise-adaptive-learning-rates>`_.

The first five options are the same as those used by the `adamw <https://pytorch.org/docs/stable/generated/torch.optim.AdamW.html>`_ optimizer,
with the remaining specific to the LAMB update:

- **lr**: learning rate ``(default=.001)``
- **beta1**: coefficient for calculating running average of the gradient (default=.9)``
- **beta2**: coefficient for calculating running average of the squared gradient ``(default=.999)``
- **eps**: minimum denominator used when squared gradient term approaches zero ``(default=1e-8)``
- **decay**: weight decay coefficient  ``(default=0.0)``
- **unbiased**: if flag is true, adjusts running averages by dividing by (1 - beta ** steps) to prevent biased averages in the early steps of the optimizer ``(default=true)``
- **globalnorm**: if true, optimizer will calculate a global gradient norm across all model parameters and divide individual gradients by this norm ``(default=true)``
- **trustclip**: flag is set true to clip the trust ratio to within the supplied min & max limits ``(default=true)``
- **trustmin**: minimum trust ratio ``(default=0.0)``
- **trustmax**: maximum trust ratio ``(default=1.0)``

::

   q)help`lamb
   lr        | 0.001
   beta1     | 0.9
   beta2     | 0.999
   eps       | 1e-08
   decay     | 0f
   unbiased  | 1b
   globalnorm| 1b
   trustclip | 1b
   trustmin  | 0f
   trustmax  | 1f

