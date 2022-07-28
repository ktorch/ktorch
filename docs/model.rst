
.. _model:

Models
======

A model groups a module and loss function together with an optimizer: the three components allow the parameters in the module to be trained or fitted by minimizing the loss.

Defining a model
****************

The main k-api function is :func:`model`. It is used to create a model given the parts: a
:ref:`module <modules>`,
:ref:`loss <loss>` function and an
:ref:`optimizer <optimizer>`.

.. function:: model(module; loss; optimizer) -> model pointer

   :param pointer module: an :doc:`api-pointer <pointers>` to an allocated :ref:`module <modules>`.
   :param pointer loss: an :doc:`api-pointer <pointers>` to a :ref:`loss <loss>` module.
   :param pointer optimizer: an :doc:`api-pointer <pointers>` to an :ref:`optimizer <optimizer>`.

   :return: An :doc:`api-pointer <pointers>` to a new model.

::

   q)q:module((0;`sequential); (1;(`linear;64;10)); (1;`relu))
   q)l:loss`ce
   q)o:opt(`sgd;q)

   q)m:model(q;l;o)

   q)class m
   `model

.. note::

   The :func:`model` function assumes ownership of the memory allocated by the module, loss and optimizer given as arguments.  Use :ref:`addref <addref>` to increment the reference of the arguments to :func:`model` to maintain their use.

::

   q)q:module((0;`sequential); (1;(`linear;64;10)); (1;`relu))
   q)l:loss`ce
   q)o:opt(`sgd;q)

   q)m:model(q;l;o)

   q)class q
   'stale pointer
     [0]  class q
          ^

   q)free m
   q)q:module((0;`sequential); (1;(`linear;64;10)); (1;`relu))
   q)l:loss`ce
   q)o:opt(`sgd;q)

   q)m:model(addref q;l;o)

   q)class q
   `module


Redefining a model
******************

.. function:: model(model; object; ..) -> null

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously allocated model.
   :param pointer object: one or more :doc:`api-pointer <pointers>` to a module, loss or optimizer.

   :return: The model's module, loss and/or optimizer is replaced by the supplied object(s). Null result.

::

   q)m:model(module`relu; loss`ce; opt`sgd)

   q)model[m]`loss
   module | `ce
   options| `weight`ignore`smoothing`reduce!(::;-100;0f;`mean)

   q)model(m;loss`mse)

   q)model[m]`loss
   module | `mse
   options| (,`reduce)!,`mean

Retrieving model definition
***************************

The same :func:`model` function used to define/update a model can also be used to retrieve its definition.

.. function:: model(model) -> k dictionary of settings
.. function:: model(model; alloptions) -> k dictionary of settings

   :param pointer model: an :doc:`api-pointer <pointers>` to a previously allocated model.
   :param bool alloptions: an optional flag set true to retrieve all options, false to retrieve only non-default options. Uses :ref:`global setting <alloptions>` if not specified.

   :return: Returns a k dictionary with the options for defining the model's module, loss and optimizer as well as train and test options.

.. note::

   The returned k dictionary contains all the settings that define the components of the model, but not the parameter values/buffers in the modules or the optimizer. The parameters and buffers are initialized as different tensors upon object creation, see model :ref:`state <modelstate>` for more on creating an exact copy of an existing or saved model.

::

   q)q:module seq(`sequential; (`linear;64;10); `relu)
   q)m:model(q; loss`ce; opt(`sgd;q))

   q)d:model m
   q)key d
   `module`loss`optimizer`train`test

The values for ```module``, ```loss`` and ```optimizer`` contain the definitions for the model's module, loss and optimizer.
See :ref:`training options <model-options>` for more on the values of the dictionary's ```train`` and ```test`` keys.

::

   q)q:module enlist(`linear;2;1)
   q)m:model(q; loss`ce; opt(`sgd;q))

   q)d:model m
   q)key d
   `module`loss`optimizer`train`test

The ```module`` key maps to a dictionary or table used to define the model's underlying module, typically a container module with several layers of child modules:

::

   q)d`module
   depth module     name options                
   ---------------------------------------------
   0     sequential      (`symbol$())!()        
   1     linear     0    `in`out`bias!(64;10;1b)
   1     relu       1    (,`inplace)!,0b        

The ```loss`` key is a dictionary that defines the model's loss criterion:

::

   q)d`loss
   module | `ce
   options| `weight`ignore`smoothing`reduce!(::;-100;0f;`mean)

The ```optimizer`` key has the model's optimizer type, options (one dictionary for each :ref:`parameter group <optgroups>`) and a table of parameter descriptions including parameter group:

::

   q)d`optimizer
   optimizer| `sgd
   options  | ,`lr`momentum`dampening`decay`nesterov!(0.01;0f;0f;0f;0b)
   parms    | +`parmgroup`pointer`module`name`size!(0 0;71225792 71226480;`linea..

   q)d .`optimizer`options
   lr   momentum dampening decay nesterov
   --------------------------------------
   0.01 0        0         0     0       

   q)d .`optimizer`parms
   parmgroup pointer  module name     size 
   ----------------------------------------
   0         71225792 linear 0.weight 10 64
   0         71226480 linear 0.bias   ,10  


Recreating a model
^^^^^^^^^^^^^^^^^^

Given the dictionary result from a previous :func:`model` call, it is possible to recreate the parts of the model, then the model itself:

::

   q)d:model m

   q)q:module d`module
   q)l:loss d`loss
   q)o:opt(d`optimizer;q)

   q)m2:model(q; l; o)

   q)model[m]~'model m2
   module   | 1
   loss     | 1
   optimizer| 0
   train    | 1
   test     | 1

The new, recreated model differs in its optimizer settings only in the parameter pointers in memory:

::

   q)model[m][`optimizer]~'model[m2]`optimizer
   optimizer| 1
   options  | 1
   parms    | 0

   q)(model[m] .`optimizer`parms)~''model[m2].`optimizer`parms
   parmgroup pointer module name size
   ----------------------------------
   1         0       1      1    1   
   1         0       1      1    1   


.. note::

   The output of the :func:`model` function when supplied with a previously allocated model pointer can be used to create a new model with the same settings, but not the same parameter and buffer values (these are initialized with some random processes).  To recreate the model with the exact parameter and buffer values, used the output from the :func:`state` call.

.. _modelstate:

Model state
***********

The :func:`state` function can be called with a module, loss, optimizer of model argument.
It will return a k dictionary of settings and parameter and buffer values to allow the object to be completely recreated.

The k dictionary result is much larger than the result from :func:`model` call because it includes k arrays with all the parameter and buffer values needed to completely recreate the model's underlying module and optimizer:

::

   q)q:module seq(`sequential; (`linear;64;10); `relu)
   q)m:model(q; loss`ce; opt(`sgd;q))

   q)d:model m
   q)s:state m
   q)key s
   `module`loss`optimizer`train`test

   q)q:module s`module

   q)s[`module]~''state q
   depth module name options parms buffers
   ---------------------------------------
   1     1      1    1       1     1      
   1     1      1    1       1     1      
   1     1      1    1       1     1      


The k dictionary returned by :func:`state` has additional columns:

::

   q){cols x`module}'[(d;s)]
   `depth`module`name`options
   `depth`module`name`options`parms`buffers

   q){cols x .`optimizer`parms}'[(d;s)]
   `parmgroup`pointer`module`name`size
   `parmgroup`pointer`module`name`size`buffers

See the section in :ref:`optimizers <optimizer>` on :ref:`restoring <optstate>` from state for more details on recreating an optimizer from a saved state.
If a model is to run on a CUDA device, first create the module on the cpu, then move to an available :ref:`CUDA <devices>` device. Then create the optimizer from the newly created module -- in this way, the optimizer buffers will match the device of the module parameters.

Retrieving model objects
************************

When the model is created, the pointers for the underlying module, loss and optimizer are freed. New pointers to these objects can be created by their individual creation functions: :func:`module`, :func:`loss` and :func:`opt`:

::

   q)q:module seq(`sequential; (`linear;64;10); `relu)
   q)l:loss`mse
   q)o:opt(`sgd;q)

   q)mapped each(q;l;o)
   111b

   q)m:model(q;l;o)

   q)mapped each(q;l;o)
   000b

   q)q:module m
   q)l:loss m
   q)o:opt m

   q)mapped each(q;l;o)
   111b

