.. _modules:

Modules
=======

A `PyTorch module <https://pytorch.org/docs/stable/nn.html>`_ is a container that typically accepts tensor input(s) and produces tensor output(s),
often using learnable weights or parameters.

As an example, in PyTorch, `the linear module <https://pytorch.org/docs/stable/generated/torch.nn.Linear.html#torch.nn.Linear>`_ is defined by, at minimum, by supplying the number of input and output features:

::

   >>> import torch
   >>> m=torch.nn.Linear(784,10)

   >>> print(m)
   Linear(in_features=784, out_features=10, bias=True)

The k interface uses a single function, :func:`module` to both define modules and retrieve their definition.

::

   q)m:module enlist(`linear;784;10)

   q)module m
   depth  | 0
   module | `linear
   name   | `
   options| `in`out`bias!(784;10;1b)

.. _module-args:

Defining a module
^^^^^^^^^^^^^^^^^

.. function:: module(arg) -> module pointer

   | Create a module given type, an optional name and one or more options.

   :param list arg: a symbol scalar, symbol list or enclosed general list defining the type of module, an optional name and settings.
                    At its simplest, the argument is a single scalar indicating the type of module, e.g. ```relu``,
                    or it may be an enclosed general list, e.g. ``(`linear;784;10)`` or ``(`linear;`in`out!784 10)``
                    or some mix of positional and named options.
                    A single general list is enclosed to allow for a tree structure defining multiple modules.
   :return: An :doc:`api-pointer <pointers>` to the allocated module.

Module definitions must include the type of module, e.g. ```linear``.
For activation functions this is often all that is needed, e.g. ```relu``,
but most other module types  require some additional options be specified before the module can be created.
Additional specifications require an enclosed general list to allow for multiple definitions in a single call to :func:`module`.

For example, `the PyTorch 2D convolution <https://pytorch.org/docs/stable/generated/torch.nn.Conv2d.html>`_ module has three required arguments,
input channels, output channels and size, along with several other optional settings.

::

   q)help`conv2d
   in     | 16
   out    | 32
   size   | 3 5
   stride | 1
   pad    | 0
   dilate | 1
   groups | 1
   bias   | 1b
   padmode| `zeros

The three required arguments for a 2D convolution can be given via position:

::

   q)m:module enlist(`conv2d; 16; 32; 3)

   q)module m
   depth  | 0
   module | `conv2d
   name   | `
   options| `in`out`size`stride`pad`dilate`groups`bias`padmode!(16;32;3;1;0;1;1;..

Additional arguments can be given out of position by using their symbol name:

:: 

   q)m:module enlist(`conv2d; 16; 32; 3; `bias,0b)


Some or all arguments can be specified by using name-value pairs -- the name-value pairs must be given after all positional arguments:

::

   q)m:module enlist(`conv2d; `C; ((`in;16);(`out;32);(`size;3)))

The module settings can also be given as a dictionary:

::

   q)m:module enlist(`conv2d; `C; `in`out`size`bias!(16;32;3;0b))


Retrieving a module definition
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The same :func:`module` function that is used to define a module can be used to retrieve the module's definition -- the module type, name and settings.

.. function:: module(ptr) -> dictionary/table
.. function:: module(ptr;flag) -> dictionary/table

   | Return a dictionary/table with module depth,type,name and options. An optional boolean flag indicates whether all options or only non-default options are to be returned.

   :param module ptr: An :doc:`api-pointer <pointers>` to the allocated module.
   :param boolean flag: An optional flag, set true to return all options, false to only return non-default options. If not specified, the flag uses the :ref:`global setting <settings>` for :ref:`show all options <alloptions>`.
   :return: A dictionary for a single module, a table for a container module.


In the examples below, a module and a sequence of modules are created, then their definitions retrieved via :func:`module`.

::

   q)m:module enlist(`conv2d; 16; 32; 3; `bias,0b)

   q)module m
   depth  | 0
   module | `conv2d
   name   | `
   options| `in`out`size`stride`pad`dilate`groups`bias`padmode!(16;32;3;1;0;1;1;..

   q)exec options from module(m;0b)  /retrieve only non-default settings
   in  | 16
   out | 32
   size| 3
   bias| 0b

   q)q:module (`sequential; enlist(`linear;784;10); `relu)

   q)module q
   depth module     name options                 
   ----------------------------------------------
   0     sequential      (`symbol$())!()         
   1     linear     0    `in`out`bias!(784;10;1b)
   1     relu       1    (,`inplace)!,0b         

.. note::

   Retrieving a module's definition is different from retrieving its entire state which returns a much larger result including learned parameters and any internal buffers.

The result returned by :func:`module` can be used to create a new module of identically defined layers but different parameters and buffers which are randomly initialized upon module creation.

::

   q)m:module enlist(`linear;2;1)

   q)show d:module m
   depth  | 0
   module | `linear
   name   | `
   options| `in`out`bias!(2;1;1b)

   q)m2:module d

   q)d~module m2
   1b  / identical definitions

Examining the full state reveals that the randomly initialized parameters are different:

::

   q)state m
   depth  | 0
   module | `linear
   name   | `
   options| `in`out`bias!(2;1;1b)
   parms  | `weight`bias!(,0.1977904 -0.4527635e;,-0.3943974e)
   buffers| (`symbol$())!()

   q)state m2
   depth  | 0
   module | `linear
   name   | `
   options| `in`out`bias!(2;1;1b)
   parms  | `weight`bias!(,0.6101567 0.02841701e;,-0.5737828e)
   buffers| (`symbol$())!()

   q)state[m]~'state m2
   depth  | 1
   module | 1
   name   | 1
   options| 1
   parms  | 0
   buffers| 1


Defining a network
^^^^^^^^^^^^^^^^^^

Most neural networks are made up of a container module which in turn contains individual modules or blocks of further containers with individual models.

The PyTorch `Sequential <https://pytorch.org/docs/stable/generated/torch.nn.Sequential.html>`_ module is a widely used container that chains the input through all the child modules to return the container module's output.

The k api uses sequential-like modules wherever possible to allow creation of networks without having to explicitly define the module's ``forward()`` function.
 
::

   /apply linear then relu layer to input
   q)q:module (`sequential; enlist(`linear;5;2); `relu)

   q)module q
   depth module     name options              
   -------------------------------------------
   0     sequential      (`symbol$())!()      
   1     linear     0    `in`out`bias!(5;2;1b)
   1     relu       1    (,`inplace)!,0b      

   q)x:tensor(`randn; 2 5)         /random inputs

   q)show tensor y:forward(q;x)    /apply the sequence of modules (linear,relu)
   0.5273134 0.8875197
   0.8908029 0       

   q)p:"f"$exec first parms from state q where module=`linear  /extract weight & bias

   q)p.bias+/:("f"$tensor x)$flip p.weight  /apply linear layer
   0.5273133 0.8875197 
   0.890803  -0.5097261

   q)show k:0|p.bias+/:("f"$tensor x)$flip p.weight  /apply linear then relu
   0.5273133 0.8875197
   0.890803  0        

   q)-2 raze each .Q.fmt[12;8]''[tensor[y]-k];  /k computations in 8-byte float vs 4-byte for pytorch
     0.00000003 -0.00000005
    -0.00000003  0.00000000

Trees & depth-value pairs
*************************

A series of modules can be defined with lists of varying levels of enclosure (tree) or depth-value pairs.
Two utilities, :func:`tree` and :func:`dv`, convert one representation to the other.

.. function:: tree(d) -> tree
.. function:: dv(t) -> depth-value pairs

::

   q)show t:(`a;(`b; enlist(`c;1); enlist(`d;2)))
   `a
   (`b;,(`c;1);,(`d;2))

   q)show d:dv t
   0 `a    
   1 `b    
   2 (`c;1)
   2 (`d;2)

   q)t~tree d
   1b

Using the tree representation, the k api can create a ```sequential`` parent with two nested sequential module blocks.  (Pytorch C++ interface uses templated forward calls that prevents nesting of the Sequential module as supplied, but a subclass, tagged as ```seqnest`` avoids the template and can be nested.)

::

   q)q1:(`seqnest`block1; enlist(`linear;`linear1;10;5); enlist(`relu;`relu1))
   q)q2:(`seqnest`block2; enlist(`linear;`linear2; 5;2); enlist(`relu;`relu2))
   q)t:(`sequential`parent; q1; q2)

   q)t  /tree representation
   `sequential`parent
   (`seqnest`block1;,(`linear;`linear1;10;5);,`relu`relu1)
   (`seqnest`block2;,(`linear;`linear2;5;2);,`relu`relu2)

   q)q:module t

   q)module q
   depth module     name    options               
   -----------------------------------------------
   0     sequential parent  (`symbol$())!()       
   1     seqnest    block1  (`symbol$())!()       
   2     linear     linear1 `in`out`bias!(10;5;1b)
   2     relu       relu1   (,`inplace)!,0b       
   1     seqnest    block2  (`symbol$())!()       
   2     linear     linear2 `in`out`bias!(5;2;1b) 
   2     relu       relu2   (,`inplace)!,0b       


This same network can be defined using depth-value pairs:

::

   q)d:()
   q)d,:(0; `sequential`parent)
   q)d,:(1; `seqnest`block1)
   q)d,:(2;(`linear;`linear1;10;5))
   q)d,:(2; `relu`relu1)
   q)d,:(1; `seqnest`block2)
   q)d,:(2;(`linear;`linear2;5;2))
   q)d,:(2; `relu`relu2)
   q)d:0N 2#d

   q)t~tree d
   1b

   q)d:module d
   q)module[q]~module d
   1b

.. note::

   A layer defined by a single symbol, e.g. ```relu`` or a 2-element list of module type and name need not be enlisted in the tree representation. The symbol(s) will resolve to the correct definition without the ``enlist()``.

For example:

::

   q)tree dv(`sequential`a; enlist(`linear;5;2); `relu`r)
   `sequential`a
   ,(`linear;5;2)
   ,`relu`r

   q)q:module (`sequential`a; enlist(`linear;5;2); `relu`r)

   q)state q
   depth module     name options               parms                            ..
   -----------------------------------------------------------------------------..
   0     sequential a    (`symbol$())!()       (`symbol$())!()                  ..
   1     linear     0    `in`out`bias!(5;2;1b) `weight`bias!((-0.2608652 -0.1890..
   1     relu       r    (,`inplace)!,0b       (`symbol$())!()                  ..


seq utility
***********

The :func:`seq` utility function enlists all but the first element in the supplied list.
If specifying a list in ``k)``, the enlist operator is simpler,
e.g. ``(`a; ,`b; ,(`c;1))`` vs ``(`a; enlist`b; `enlist(`c;1))`` in ``q)``. The :func:`seq` can be used to simplify the tree representation of a model:

.. function:: seq(layers) -> 2-level tree

::

   q)seq`a`b`c
   `a
   ,`b
   ,`c

Tree depth using :func:`dv`:

::

   q)dv seq`a`b`c
   0 `a
   1 `b
   1 `c

:func:`seq` is limited to 2 levels. Here the inner sequence is not translated in its full depth:

::

   q)seq(`a;(`b;(`c;1));`d)
   `a
   ,(`b;(`c;1))
   ,`d

   q)dv seq(`a;(`b;(`c;1));`d)
   0 `a         
   1 (`b;(`c;1))                 / << should be depth 2
   1 `d         

The :func:`seq` call can be embedded at different depths:

::

   q)dv (`a;seq(`b;(`c;1));`d)
   0 `a    
   1 `b    
   2 (`c;1)
   1 `d    

   q)dv (`a;seq(`b;(`c;1));`d;seq`e`f)
   0 `a    
   1 `b    
   2 (`c;1)
   1 `d    
   1 `e    
   2 `f    

In the following example, two blocks are created with enlisted ```linear`` and ```relu`` layers:

::

   q)q1:(`seqnest`block1; enlist(`linear;`linear1;10;5); enlist(`relu;`relu1))
   q)q2:(`seqnest`block2; enlist(`linear;`linear2; 5;2); enlist(`relu;`relu2))
   q)t:(`sequential`parent; q1; q2)

The :func:`seq` utility function can be used to in place of repeating the ``enlist()`` calls:

::

   q)q1:seq(`seqnest`block1; (`linear;`linear1;10;5); `relu`relu1)
   q)q2:seq(`seqnest`block2; (`linear;`linear2; 5;2); `relu`relu2)
   q)t:(`sequential`parent; q1; q2)

.. _module-names:

Naming modules
^^^^^^^^^^^^^^

Modules can be named: this is optional for the parent module, whereas child modules are given the name of their sequence in the parent if no name supplied.

This first example defines a sequential model with names for the parent and each child layer:

::

   q)q1:seq(`seqnest`block1; (`linear;`linear1;10;5); `relu`relu1)
   q)q2:seq(`seqnest`block2; (`linear;`linear2; 5;2); `relu`relu2)
   q)q:module(`sequential`seq; q1; q2)

   q)module q
   depth module     name    options               
   -----------------------------------------------
   0     sequential seq     (`symbol$())!()       
   1     seqnest    block1  (`symbol$())!()       
   2     linear     linear1 `in`out`bias!(10;5;1b)
   2     relu       relu1   (,`inplace)!,0b       
   1     seqnest    block2  (`symbol$())!()       
   2     linear     linear2 `in`out`bias!(5;2;1b) 
   2     relu       relu2   (,`inplace)!,0b       

   q)0N 1#names q               /module names
   seq               
   seq.block1        
   seq.block1.linear1
   seq.block1.relu1  
   seq.block2        
   seq.block2.linear2
   seq.block2.relu2  

   q)0N 1#parmnames q           /parameter names
   seq.block1.linear1.weight
   seq.block1.linear1.bias  
   seq.block2.linear2.weight
   seq.block2.linear2.bias  

Defining the same network without supplying any names: 

::

   q)q1:seq(`seqnest; (`linear;10;5); `relu)
   q)q2:seq(`seqnest; (`linear; 5;2); `relu)
   q)m:module(`sequential; q1; q2)

   q)module m
   depth module     name options               
   --------------------------------------------
   0     sequential      (`symbol$())!()       
   1     seqnest    0    (`symbol$())!()       
   2     linear     0    `in`out`bias!(10;5;1b)
   2     relu       1    (,`inplace)!,0b       
   1     seqnest    1    (`symbol$())!()       
   2     linear     0    `in`out`bias!(5;2;1b) 
   2     relu       1    (,`inplace)!,0b       

   q)0N 1#names m  /module names
      
   0  
   0.0
   0.1
   1  
   1.0
   1.1

   q)0N 1#parmnames m  /parameter names
   0.0.weight
   0.0.bias  
   1.0.weight
   1.0.bias  

Module state
^^^^^^^^^^^^

The full definition and state of a module or set of modules is returned by the :func:`state` function.  It returns a dictionary for a single module or a table for a container of modules (even if the container is empty).


.. function:: state(ptr) -> dictionary/table
.. function:: state(ptr;flag) -> dictionary/table

   | Return a dictionary/table with module depth, type, name, options, parameters and buffers. An optional boolean flag indicates whether all options or only non-default options are to be returned.

   :param module ptr: An :doc:`api-pointer <pointers>` to the allocated module.
   :param boolean flag: An optional flag, set true to return all options, false to only return non-default options. If not specified, the flag uses the :ref:`global setting <settings>` for :ref:`show all options <alloptions>`.
   :return: Returns a dictionary for a single module, returns a table for container modules, one row per parent and all child modules.

This first example shows the state dictionary of a single linear module:

::

   q)a:module enlist(`linear;`example;2;1)

   q)show s:state a
   depth  | 0
   module | `linear
   name   | `example
   options| `in`out`bias!(2;1;1b)
   parms  | `weight`bias!(,-0.4117883 -0.7008631e;,0.03681418e)
   buffers| (`symbol$())!()

   q)b:module s        /create an identical module, equivalent to pytorch module.clone() method

   q)state[b]~state a  /identical state
   1b

   q)ptr each(a;b)     /different internal pointers
   53174592 53176528

   q)ref each(a;b)     /no other references
   1 1

This second example shows the state table for a sequential container module:

::

   q)q:module((0;`sequential); (1; (`linear;64;10)); (1;`relu))

   q)show s:state q
   depth module     name options                 parms                          ..
   -----------------------------------------------------------------------------..
   0     sequential      (`symbol$())!()         (`symbol$())!()                ..
   1     linear     0    `in`out`bias!(64;10;1b) `weight`bias!((0.002459586 0.08..
   1     relu       1    (,`inplace)!,0b         (`symbol$())!()                ..

The state can be used to define a clone of the network, or saved to file and retrieved later to re-create the module and its parameters:

::

   q)q2:module s

   q)save`:/tmp/s
   `:/tmp/s

   q)q3:module get`:/tmp/s

   q)count distinct state each(q;q2;q3)
   1

   q)count distinct ptr each(q;q2;q3)
   3

Module help
^^^^^^^^^^^^

help
****

The :func:`help` function with the argument ```module`` will return a table of all available modules along with their equivalent name in Pytorch and a dictionary of example options.

::

   q)5?help`module
   module     pytorch                        options                            ..
   -----------------------------------------------------------------------------..
   reflect2d  "torch.nn.ReflectionPad2d"     (,`pad)!,1 1 2 0                   ..
   maxpool1d  "torch.nn.MaxPool1d"           `size`stride`pad`dilate`ceiling!(3;..
   rrelu      "torch.nn.RReLU"               `lower`upper`inplace!(0.125;0.33333..
   layernorm  "torch.nn.LayerNorm"           `shape`eps`affine!(32 10;1e-05;1b) ..
   fmaxpool2d "torch.nn.FractionalMaxPool2d" `size`outsize`ratio!(2 4;16 32;()) ..

If :func:`help` is called with an individual module, it will return the example options dictionary:

::

   q)help`linear
   in  | 784
   out | 10
   bias| 1b

   q)help`conv2d
   in     | 16
   out    | 32
   size   | 3 5
   stride | 1
   pad    | 0
   dilate | 1
   groups | 1
   bias   | 1b
   padmode| `zeros

str
***

The :func:`str` function outputs the string representation of a module in PyTorch terms: from the api, :func:`str` allows for an approximate visual comparison of the C++ modules with the modules originating in python.

.. function:: str(module) -> string

   | Returns the PyTorch string representation of the module with embedded newlines.

In python:

::

   >>> m=torch.nn.Sequential(torch.nn.Linear(784,10),torch.nn.ReLU())

   >>> print(m)
   Sequential(
     (0): Linear(in_features=784, out_features=10, bias=True)
     (1): ReLU()
   )

In k:

::

   q)m:module seq(`sequential; (`linear;784;10); `relu)

   / output is a string with embedded newlines
   q)str m 
   "torch::nn::Sequential(\n  (0): torch::nn::Linear(in_features=784, out_featur..

   q)-2 str m;
   torch::nn::Sequential(
     (0): torch::nn::Linear(in_features=784, out_features=10, bias=true)
     (1): torch::nn::ReLU()
   )

.. index:: forward
.. _forward:

Forward calculation
^^^^^^^^^^^^^^^^^^^

The forward calculation of single module or a sequence of modules takes input tensor(s) and feeds the tensor(s) through the set of modules, passing the output of one module as the input of the next module through to the final layer. The output of the final layer is returned as the output of the network.

Most modules accept a single tensor as input and return a single output tensor, multiple inputs can occur for recurrent networks (inputs and previous hidden state), along with generative networks that take some information on the object to be generated along with the random inputs.

.. function:: forward(module;input;..) -> tensor result
.. function:: nforward(module;input;..) -> tensor result
.. function:: eforward(module;input;..) -> tensor result
.. function:: evaluate(module;input;..) -> k array result

   :param pointer module: An :doc:`api-pointer <pointers>` to the allocated module.
   :param tensor input: A single tensor/k array or a list of tensors/arrays, depending on the module requirement.
   :return: :func:`forward`, :func:`eforward` and :func:`nforward` return tensor result(s), :func:`evaluate` returns k array(s).

To train the model (training mode and automatic gradient calculation on), use :func:`forward`.

To use training mode but no gradient calculation, use :func:`nforward`.
This is an unusual combination of settings -- one example is during recalculation of `batchnorm statistics <https://ktorch.readthedocs.io/en/latest/swa.html#update-batchnorm-layers>`_ after weight averaging.

Functions :func:`eforward` and :func:`evaluate` turn off both training mode and gradient calculations and are both intended for inference, i.e. running the previously trained model to make predictions or some other classification or result on data not used as part of the training.
The :func:`eforward` function returns tensor(s) and :func:`evaluate` returns k array(s).

Module inputs
^^^^^^^^^^^^^

Module input is often a single tensor or might include a sequence and hidden state(s) for recurrent networks, or a class and random noise for a generative model.

For the simplest case, one input tensor and one output tensor:

::

   q)r:forward(m;3 3#1e)              /use k array as input
   q)tensor r
   -0.02664497 0.3843636 -0.01880136
   -0.02664497 0.3843636 -0.01880136
   -0.02664497 0.3843636 -0.01880136

   q)x:tensor 3 3#1e
   q)use[r]forward(m; x)               /use tensor as input

   q)tensor r
   -0.02664497 0.3843636 -0.01880136
   -0.02664497 0.3843636 -0.01880136
   -0.02664497 0.3843636 -0.01880136
   
   q)to(m; `cuda)   /move module to default CUDA device

   /k input is converted to tensor on same device as 1st parameter of module
   q)use[r]forward(m; 3 3#1e)

   q)tensor r
   -0.02664497 0.3843636 -0.01880136
   -0.02664497 0.3843636 -0.01880136
   -0.02664497 0.3843636 -0.01880136

   q)device r
   `cuda:0

   /if tensor supplied, it should be on the same device and of the same data type as the module parameters
   q)evaluate(m;x)
   'Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu! (when checking arugment for argument mat1 in method wrapper_addmm)
     [0]  evaluate(m;x)
          ^

   q)evaluate(m;3 3#1.0)
   'expected scalar type Float but found Double
     [0]  evaluate(m;3 3#1.0)
          ^

   q)evaluate(m;3 3#1e)
   -0.02664497 0.3843636 -0.01880136
   -0.02664497 0.3843636 -0.01880136
   -0.02664497 0.3843636 -0.01880136

An example of multiple inputs and outputs is the `LSTM module <https://pytorch.org/docs/stable/generated/torch.nn.LSTM.html>`_, which accepts either a single input tensor, or three input tensors, one input and two containing the module's hidden state. Recurrent modules like ``LSTM`` are more complicated as they return both an output tensor and one or more tensors describing the hidden state.  The hidden state is usually added to the next input during training.

::

   q)m:module enlist(`lstm;64;512;2;1b;1b)  /input:64, hidden:512, layers:2, bias:true, batchfirst:true

   q)x:tensor(`randn; 30 50 64)   /batch size 30, sequence of 50, input size of 64

   q)v:forward(m;x)               /input single tensor, return tensor, two tensors defining hidden state

   q)size v
   30 50 512
   2  30 512
   2  30 512

   q)yhat:tensor(v;0)                 /extract model output for use in calculating loss, backprop..

   q)use[x]tensor(`randn; 30 50 64)   /simulate new inputs
   q)vector(v;0;x)                    /replace output in vector w'hidden state
   q)use[v]forward(m;v)               /repeat forward call supplying 3 tensors

   q)x:tensor(`randn; 30 50 64)       /simulate new inputs
   q)h1:tensor(v;1)                   /extract hidden state as tensors
   q)h2:tensor(v;2)
   q)use[v]forward(m;x;h1;h2)          /alternate ways to supply 3 tensor inputs
   q)use[v]forward(m; (x;h1;h2))
   
Other operations
^^^^^^^^^^^^^^^^

to
**

The function :func:`to` is similar to the `Pytorch method module.to() <https://pytorch.org/docs/stable/generated/torch.nn.Module.html?highlight=#torch.nn.Module.to>`_.

.. function:: to(ptr;options) -> null
.. function:: to(ptr;options;async-flag) -> null

   :param module ptr: An :doc:`api-pointer <pointers>` to the allocated module.
   :param symbol options: A symbol for device and/or data type, e.g. ```cuda`` or ```double`` or ```cuda`double``.
   :param boolean flag: async-flag, true or false, ``1b`` or ``0b``, default is false. If true, and if CPU tensors are in pinned memory, the transfer to GPU will be performed asyncronously.

The most common use is to define a module on the CPU, then move to a CUDA device if available.

::

   q)m:module enlist(`linear;784;10)

   q)to(m;`cuda)    /move to default CUDA device

   q)to(m;`cuda:1)  /move to a specific CUDA device

   q)p:parms m

   q)device p
   weight| cuda:1
   bias  | cuda:1


clone
*****

Function :func:`clone` makes a deep copy of the supplied module, copying all options, tensors and buffers.

.. function:: clone(module) -> copy of module

::

   q)a:module enlist(`linear;784;10)
   q)b:clone(a)

   q)p1:parms a
   q)p2:parms b

   q)dict[p1]~dict p2  /all parameter values match
   1b

   q)ptr[p1]=ptr p2    /but different pointers, i.e. tensors
   weight| 0
   bias  | 0

.. index:: training

.. _module-training:

training
********

Function :func:`training` reports or resets the training mode of the given module:

.. function:: training(module) -> boolean
.. function:: training(module;flag) -> null

   :param module ptr: An :doc:`api-pointer <pointers>` to an allocated module.
   :param boolean flag: optional training flag, ``1b`` or ``0b``.  If supplied sets the training mode on/off.
   :return: if flag supplied, then null returned, else the current state of the module's training mode.

::

   q)training m
   1b

   q)training(m;0b)

   q)training m
   0b

Module namelists
^^^^^^^^^^^^^^^^

Modules contain submodules, parameters and buffers, with the following functions defined to return the names and module types as symbols.

names
*****

The :func:`names` function returns the names of all child modules contained in the top-level module or in a specified child(specified by name or position) in the top-level module.


.. function:: names(module) -> symbols
.. function:: names(module;childname) -> symbols
.. function:: names(module;childindex) -> symbols

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module.
   :param symbol childname: the name of a child module (optional)
   :param long   childindex: the index of a child module (optional)
   :return: if no child name or index given, returns the names of all modules contained by the given module, else all modules contained in the given child.

::

   q)m:module`transformer

   q)0N 1#names m
   encoder                            
   encoder.layers                     
   encoder.layers.0                   
   encoder.layers.0.self_attn         
   encoder.layers.0.self_attn.out_proj
   encoder.layers.0.linear1           
   encoder.layers.0.dropout           
   encoder.layers.0.linear2           
   encoder.layers.0.norm1             
   encoder.layers.0.norm2             
   encoder.layers.0.dropout1          
   encoder.layers.0.dropout2          
   encoder.layers.1                   
   encoder.layers.1.self_attn         
   ..

   q)0N 1#names(m;`encoder.layers.0)
   self_attn         
   self_attn.out_proj
   linear1           
   dropout           
   linear2           
   norm1             
   norm2             
   dropout1          
   dropout2          

childnames
**********

The :func:`childnames` function works similarly to the :func:`names` function except that it only returns the names of the immediate children of the module or the module's child named in the optional 2nd argument.

.. function:: childnames(module) -> symbols
.. function:: childnames(module;childname) -> symbols
.. function:: childnames(module;childindex) -> symbols

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module.
   :param symbol childname: the name of a child module (optional)
   :param long   childindex: the index of a child module (optional)
   :return: if no child name or index supplied, returns the names of the immediate child modules contained by the given module, else the child modules contained in the given child.

::

   q)m:module`transformer

   q)childnames m
   `encoder`decoder

   q)names(m;`encoder.layers)
   `0`0.self_attn`0.self_attn.out_proj`0.linear1`0.dropout`0.linear2`0.norm1`0.n..

   q)childnames(m;`encoder.layers)
   `0`1`2`3`4`5

parmnames
*********

The :func:`parmnames` function returns all parameter names for a module or the subset of parameter names contained in a specified child (and all subsequent child modules contained by that child).

.. function:: parmnames(module) -> symbols
.. function:: parmnames(module;childname) -> symbols
.. function:: parmnames(module;childindex) -> symbols

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module.
   :param symbol childname: the name of a child module (optional)
   :param long   childindex: the index of a child module (optional)
   :return: if no child name or index supplied, returns all the parameter names in the module, else only those in indicated child and below.

::

   q)m:module seq(`sequential; (`linear;`fc;256;64;0b); (`batchnorm2d;`bnorm;64); `relu`relu)

   q)parmnames m
   `fc.weight`bnorm.weight`bnorm.bias

   q)names m
   `fc`bnorm`relu

   q)parmnames(m;`bnorm)
   `weight`bias

   q)parmnames(m;1)
   `weight`bias

buffernames
***********

The :func:`buffernames` function returns all buffer names for a module or the subset of buffer names contained in a specified child (and all subsequent child modules contained by that child).

.. function:: buffernames(module) -> symbols
.. function:: buffernames(module;childname) -> symbols
.. function:: buffernames(module;childindex) -> symbols

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module.
   :param symbol childname: the name of a child module (optional)
   :param long   childindex: the index of a child module (optional)
   :return: if no child name or index supplied, returns all the buffer names in the module, else only those in indicated child and below.

::

   q)m:module seq(`sequential; (`linear;`fc;256;64); (`batchnorm2d;`bnorm;64); `relu)

   q)buffernames m
   `bnorm.running_mean`bnorm.running_var`bnorm.num_batches_tracked

   q)buffernames(m;`bnorm)
   `running_mean`running_var`num_batches_tracked

   q)buffernames(m;1)
   `running_mean`running_var`num_batches_tracked

objtype
*******

.. function:: objtype(module) -> type

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module.
   :return: A symbol indicating the module type.

::

   q)m:module seq(`sequential; (`linear;`fc;256;64;0b); (`batchnorm2d;`bnorm;64); `relu`relu)

   q)objtype m
   `sequential

   q)c:children m

   q)objtype each c
   fc   | linear
   bnorm| batchnorm2d
   relu | relu

   q)free'[c];


moduletypes
***********

The :func:`moduletypes` function maps module names to their types:

.. function:: moduletypes(module) -> dictionary 
.. function:: moduletypes(module;types) -> dictionary 

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module.
   :param symbol types: An optional scalar/list of module type(s).
   :return: If a scalar module type is supplied as a 2nd argument, result is a symbol list of module names matching the given type. If a list of types is given, or no types, a dictionary mapping names to corresponding types is returned.

::

   q)m:module seq(`sequential; (`linear;`linear1;128;64); `relu`f1; (`linear;`linear2;64;10); `relu`f2)

   q)moduletypes m
          | sequential
   linear1| linear
   f1     | relu
   linear2| linear
   f2     | relu

   q)moduletypes(m;`sequential)  / root module key is ` regardless if given name.
   ,`

   q)moduletypes(m;`linear)
   `linear1`linear2

   q)moduletypes(m;`linear`relu)
   linear1| linear
   f1     | relu
   linear2| linear
   f2     | relu


parmtypes
*********

The :func:`parmtypes` function maps parameter names to their module type:

.. function:: parmtypes(module) -> dictionary 

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module, model or optimizer.
   :return: Returns a k dictionary mapping parameter name to underlying module type.

::

   q)q:module seq(`sequential; (`conv2d;`a;3;64;3); (`batchnorm2d;`b;64); (`relu;`c))

   q)parmtypes q
   a.weight| conv2d
   a.bias  | conv2d
   b.weight| batchnorm2d
   b.bias  | batchnorm2d

The mapping can be used to distinguish parameters for :ref:`optimizer groups <optgroups>`
or for special :ref:`parameter initialization<init>`:

::

   q)w:{x where x like "*.weight"}where parmtypes[q]=`conv2d

   q){xnormal(x;y;gain`relu)}[q]'[w]; /xavier initialization for conv layers
   

Module objects
^^^^^^^^^^^^^^

Given a module (or :doc:`model <model>` or :doc:`optimizer <opt>`), the functions :ref:`modules() <modulesfn>` and :ref:`children() <children>` return k dictionaries with keys of module names and values of module pointers whereas :ref:`child() <childfn>` returns a single module pointer.

.. _modulesfn:

module tree
***********

The :ref:`modules() <modules>` function returns the full sub-tree of modules of the entire module or starting from the indicated child name/index.

.. function:: modules(module) -> dictionary of modules
.. function:: modules(module;childname) -> dictionary of modules in named child module
.. function:: modules(module;childindex) -> dictionary of modules in indexed child module

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module.
   :param symbol childname: the name of a child module (optional)
   :param long   childindex: the index of a child module (optional)
   :return: if no child name or index supplied, returns a map of module name to module pointer for all submodules, else only those in the indicated child and below.

::

   q)m:module`transformer

   q)show d:modules m
   encoder                            | 146865120
   encoder.layers                     | 146865216
   encoder.layers.0                   | 146865328
   encoder.layers.0.self_attn         | 146865440
   encoder.layers.0.self_attn.out_proj| 146865568
   encoder.layers.0.linear1           | 146865680
   encoder.layers.0.dropout           | 146865792
   encoder.layers.0.linear2           | 146865904
   encoder.layers.0.norm1             | 146866016
   encoder.layers.0.norm2             | 146866128
   encoder.layers.0.dropout1          | 146866240
   encoder.layers.0.dropout2          | 146866352
   encoder.layers.1                   | 146866464
   encoder.layers.1.self_attn         | 146801664
   ..

.. _children:

children
********

The :ref:`children() <children>` function returns only the direct children of the module or the indicated main module child named or indexed.

.. function:: children(module) -> dictionary of direct child modules
.. function:: children(module;childname) -> dictionary of direct children in named child module
.. function:: children(module;childindex) -> dictionary of direct children in indexed child module

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module.
   :param symbol childname: the name of a child module (optional)
   :param long   childindex: the index of a child module (optional)
   :return: if no child name or index supplied, returns a map of module name to module pointer for direct child modules for the given module or those in the indicated child (but not below as in the :ref:`modules() <modules>` function).

::

   q)m:module`transformer

   q)show c:children m
   encoder| 146881552
   decoder| 146864528

   q)free c
   q)show c:children(m; `encoder.layers.0)
   self_attn| 146865024
   linear1  | 146739072
   dropout  | 62198416 
   linear2  | 146840784
   norm1    | 146815792
   norm2    | 146739552
   dropout1 | 146746080
   dropout2 | 146857536

.. _childfn:

child
*****

The :func:`child` function returns a single module pointer.

.. function:: child(module;childname) -> module of named child
.. function:: child(module;childindex) -> module of indexed child

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module.
   :param symbol childname: the name of a child module
   :param long   childindex: the index of a child module
   :return: Returns an :doc:`api-pointer <pointers>` to the named/indexed child.

::

   q)m:module`transformer

   q)c:child(m; `decoder.layers.0.multihead_attn)

   q)-2 str c;
   torch::nn::MultiheadAttention(
     (out_proj): torch::nn::Linear(in_features=512, out_features=512, bias=true)
   )

parm
****
Retrieve single parameter from given module:

.. function:: parm(module;name) -> tensor

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module.
   :param symbol name: The name of a module parameter, at any level in the module

   :return: An :doc:`api-pointer <pointers>` to the parameter tensor.

::

   q)m:module enlist(`batchnorm2d;64)
   q)parmnames m
   `weight`bias

   q)b:parm(m;`bias)
   q)tensor b
   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0..

Parameters of child modules are indicated with ``.``, e.g. ``parent.child.weight``.

::

   q)m:module`transformer
   q)show s:rand parmnames m
   `decoder.layers.1.norm1.weight

   q)t:parm(m;s)
   q)size t
   ,512


:func:`parm` may also be called with a 3rd argument of values to assign to the parameter:

.. function:: parm(module;name;value) -> null

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module.
   :param symbol name: The name of a module parameter, at any level in the module
   :param tensor value: An :doc:`api-pointer <pointers>` to an allocated tensor; the module parameter will be set to this tensor's values; the value(s) may also be given as a k scalar, list or multi-dimensional array.

   :return: Resets the parameter named with the supplied values, returns null.

::

   q)m:module enlist(`batchnorm2d;64)
   q)b:parm(m;`bias)

   q)tensor b
   0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0..

   q)parm(m;`bias; .0001)
   q)tensor b
   0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 0.0001 ..

buffer
******
Retrieve a single buffer from a given module or reset its values. Arguments and syntax are the same as for :func:`parm`.

::

   q)m:module enlist(`batchnorm2d;64)

   q)buffernames m
   `running_mean`running_var`num_batches_tracked

   q)tensor t:buffer(m;`running_var)
   1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1..

parms
*****
.. function:: parms(module) -> tensor dictionary
.. function:: parms(module;child) -> tensor dictionary

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module.
   :return: A tensor dictionary of parameters in the given module.

::

   q)m:module`sequential
   q)module(m; enlist(`linear;`linear1;128;64))
   q)module(m; enlist(`batchnorm2d;`bnorm;128))
   q)module(m; `relu`f1)
   q)module(m; enlist(`linear;`linear2;64;10))
   q)module(m; `relu`f2)

   q)p:parms m

   q)class p
   `dictionary

   q)objtype p
   `parameter

   q)0N 1#names p
   linear1.weight
   linear1.bias  
   bnorm.weight  
   bnorm.bias    
   linear2.weight
   linear2.bias  

buffers
*******
.. function:: buffers(module) -> tensor dictionary
.. function:: buffers(module;child) -> tensor dictionary

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module.
   :return: A tensor dictionary of buffers in the given module.

::

   q)m:module`sequential
   q)module(m; enlist(`linear;`linear1;128;64))
   q)module(m; enlist(`batchnorm2d;`bnorm;128))

   q)b:buffers m

   q)objtype b
   `buffer

   q)0N 1#names b
   bnorm.running_mean       
   bnorm.running_var        
   bnorm.num_batches_tracked

   q)bn:buffers(m;`bnorm)
   q)names bn
   `running_mean`running_var`num_batches_tracked

   q)dict bn
   running_mean       | 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 ..
   running_var        | 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 ..
   num_batches_tracked| 0


Other information
^^^^^^^^^^^^^^^^^

inputmodule
***********

Given a sequence of modules, :func:`inputmodule` returns the type of module that accepts the first input through the network.

.. function:: inputmodule(module) -> symbol

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module.
   :return: The type of module that accepts the first input given to the network.

::

   q)q:module(`sequential; enlist(`linear;50;10); `relu; `drop)

   q)-2 str q;
   torch::nn::Sequential(
     (0): torch::nn::Linear(in_features=50, out_features=10, bias=true)
     (1): torch::nn::ReLU()
     (2): torch::nn::Dropout(p=0.5, inplace=false)
   )

   q)inputmodule q
   `linear

   q)outputmodule q
   `drop

outputmodule
************

Given a sequence of modules, :func:`outputmodule` returns the type of module that produces the final output.

.. function:: outputmodule(module) -> symbol

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module.
   :return: The type of module that handles the final output of the network.

::

   q)q:module(`sequential; enlist(`linear;50;10); `relu; `drop)

   q)-2 str q;
   torch::nn::Sequential(
     (0): torch::nn::Linear(in_features=50, out_features=10, bias=true)
     (1): torch::nn::ReLU()
     (2): torch::nn::Dropout(p=0.5, inplace=false)
   )

   q)inputmodule q
   `linear

   q)outputmodule q
   `drop
