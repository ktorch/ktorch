.. _loss:

Loss Modules
============

The k api implements the subset of the PyTorch loss functions that are implemented in the C++ libraries:

- `bce <https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html>`_: measures the binary cross entropy between input and target probabilities.
- `bcelogits <https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html>`_: combines a sigmoid layer and binary cross entropy loss in one operation.
- `ce <https://pytorch.org/docs/stable/generated/torch.nn.CrossEntropyLoss.html>`_: cross entropy loss between input and target.
- `cosineloss <https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html>`_: cosine embedding loss.
- `ctc <https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html>`_: connectionist temporal classification loss.
- `hinge <https://pytorch.org/docs/stable/generated/torch.nn.HingeEmbeddingLoss.html>`_: hinge embedding loss.
- `huber <https://pytorch.org/docs/stable/generated/torch.nn.HuberLoss.html>`_: Huber loss (combines advantages of mean-squared and level 1 loss)
- `kl <https://pytorch.org/docs/stable/generated/torch.nn.KLDivLoss.html>`_: Kullback-Leibler divergence.
- `l1 <https://pytorch.org/docs/stable/generated/torch.nn.L1Loss.html>`_: level 1 loss (mean absolute error).
- `margin <https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html>`_: margin ranking loss.
- `mse <https://pytorch.org/docs/stable/generated/torch.nn.MSELoss.html>`_: mean squared errer.
- `multilabel <https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelMarginLoss.html>`_: multi-label margin loss.
- `multimargin <https://pytorch.org/docs/stable/generated/torch.nn.MultiMarginLoss.html>`_: multi margin loss.
- `multisoft <https://pytorch.org/docs/stable/generated/torch.nn.MultiLabelSoftMarginLoss.html>`_: multi-label soft margin loss.
- `nll <https://pytorch.org/docs/stable/generated/torch.nn.NLLLoss.html>`_: negative log-likelihood loss.
- `pairwise <https://pytorch.org/docs/stable/generated/torch.nn.PairwiseDistance.html>`_: pair-wise distance.
- `poissonloss <https://pytorch.org/docs/stable/generated/torch.nn.PoissonNLLLoss.html>`_: negative log likelihood loss with Poisson distribution of target.
- `similar <https://pytorch.org/docs/stable/generated/torch.nn.CosineSimilarity.html>`_: cosine similarity
- `smoothl1 <https://pytorch.org/docs/stable/generated/torch.nn.SmoothL1Loss.html>`_: smooth L1 loss using squared loss when below a threshold.
- `softmargin <https://pytorch.org/docs/stable/generated/torch.nn.SoftMarginLoss.html>`_: soft margin loss.
- `triplet <https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html>`_: triplet margin loss.

.. _losshelp:

Help
^^^^

A table of loss modules and their options is available via the :ref:`help() <losshelp>` function: 

::

   q)help`loss

   module     pytorch                        forward result n args              ..
   -----------------------------------------------------------------------------..
   bce        "torch.nn.BCELoss"             1       tensor 2 `tensor`tensor`ten..
   bcelogits  "torch.nn.BCEWithLogitsLoss"   1       tensor 2 `tensor`tensor`ten..
   ce         "torch.nn.CrossEntropyLoss"    1       tensor 2 `tensor`tensor    ..
   cosineloss "torch.nn.CosineEmbeddingLoss" 1       tensor 3 `tensor`tensor`ten..
   ctc        "torch.nn.CTCLoss"             1       tensor 4 `tensor`tensor`ten..
   ..

Supplying the help function with an individual loss module returns a a single row of the above table, a dictionary of attributes defining the loss type, a PyTorch description, a flag indicating that it has a non-templatized forward call, the result type, minimum number of arguments, argument types and initialization options:

::

   q)help`ce
   module | `ce
   pytorch| "torch.nn.CrossEntropyLoss"
   forward| 1b
   result | `tensor
   n      | 2
   args   | `tensor`tensor
   options| `weight`ignore`smoothing`reduce!(::;-100;0f;`mean)

Using the :func:`options` function with the name of a loss module returns the initialization options:

::

   q)options`ce
   weight   | ::
   ignore   | -100
   smoothing| 0f
   reduce   | `mean

.. _lossinit:

loss
^^^^

The main k api function is :ref:`loss <lossinit>` which is used to create a loss module as well as :ref:`retrieve <lossget>` its definition and perform the loss :ref:`calculation <losscalc>`:

.. function:: loss(name) -> loss module
.. function:: loss(name; options..) -> loss module

   :param symbol name: e.g. ```ce`` or ```mse``
   :param list options: scalar or list of settings, either positional, named or both.

   :return: An :doc:`api-pointer <pointers>` to a new loss module.


Options
^^^^^^^

Loss components are implemented as a type of :doc:`module <modules>`, and follow the same rules for specifying options, retrieving definition and state.  But loss modules typically don't have trainable parameters and are more likely to be initialized with default settings.

Reduce
******

Most of the loss modules have an option for reducing the output: ```mean``, ```sum`` or ```none``.
This is usually given in the last option position (sometimes it is the only option that can be specified).

::

   q)l:loss(`mse;`none)  /no reduction
   q)show tensor t:loss(l; 1 2 3e; 0 2 4e); free(l;t)
   1 0 1e

   q)l:loss(`mse;`sum)  /sum losses
   q)show tensor t:loss(l; 1 2 3e; 0 2 4e); free(l;t)
   2e

   q)l:loss(`mse;`mean)
   q)show tensor t:loss(l; 1 2 3e; 0 2 4e); free(l;t)
   0.6666667e

Verify that all but the distance functions have ```reduce`` as their final option:

::

   q)select module, options from help`loss where not (last key@)'[options]=`reduce
   module   options                     
   -------------------------------------
   pairwise `p`eps`keepdim!(2f;1e-06;0b)
   similar  `dim`eps!(1;1e-08)          

Positional options
******************

Loss options can be specified by position after the 1st argument of loss type:

::

   q)options`ce
   weight   | ::
   ignore   | -100
   smoothing| 0f
   reduce   | `mean

Specifying class weights, with no class to be ignored, smoothing factor of .1 and mean reduction:

::

   q)l:loss(`ce; .25 .25 .12 .38; -100; .1; `mean)

   q)options l
   weight   | 0.25 0.25 0.12 0.38
   ignore   | -100
   smoothing| 0.1
   reduce   | `mean

Named options
*************

After 1st argument of loss type, other arguments can be specified by name:

::

   q)l:loss(`ce; `smoothing,.1)

   q)options l
   weight   | ::
   ignore   | -100
   smoothing| 0.1
   reduce   | `mean

Multiple named arguments can be supplied via a dictionary or via lists:

::

   q)l1:loss(`ce; `smoothing`reduce!(.1;`none))
   q)l2:loss(`ce; (`smoothing;.1;`reduce;`none))
   q)l3:loss(`ce; ((`smoothing;.1); (`reduce;`none)))

   q)options each (l1;l2;l3)
   weight ignore smoothing reduce
   ------------------------------
   ::     -100   0.1       none  
   ::     -100   0.1       none  
   ::     -100   0.1       none  


Mixed options
*************

Positional and named arguments can be mixed if the positional arguments are supplied first, then named arguments:

::

   q)l:loss(`ce; 0.0,4#5%4; `reduce`none)

   q)options l
   weight   | 0 1.25 1.25 1.25 1.25
   ignore   | -100
   smoothing| 0f
   reduce   | `none

.. _lossget:

Retrieve options
****************

The same :func:`loss` function that is used to create a loss module can also be used to retrieve the options previously defined:

.. function:: loss(module) -> k dictionary
.. function:: loss(module;flag) ->  k dictionary

   :param pointer module: An :doc:`api-pointer <pointers>` to the created loss module.
   :param boolean flag: An optional flag, set true to return all options, false to only return non-default options. If not specified, the flag uses the :ref:`global setting <settings>` for :ref:`show all options <alloptions>`.
   :return: A dictionary with module type and options used. If ``flag`` is true, all options are returned, else if ``false``, only non-default options are given.

::

   q)l:loss(`ce; (); -100; .1; `none)

   q)loss l
   module | `ce
   options| `weight`ignore`smoothing`reduce!(::;-100;0.1;`none)

   q)loss(l;0b)
   module | `ce
   options| `smoothing`reduce!(0.1;`none)


.. _losscalc:

Loss calculation
^^^^^^^^^^^^^^^^

The same :func:`loss` function that is used to define losses and retrieve their definition is also used to calculate loss by supplying the defined module together with tensors for outputs and targets.

.. function:: loss(module; output; target) -> tensor

   :param pointer module: an :doc:`api-pointer <pointers>` to an already created loss module.
   :param pointer output: an :doc:`api-pointer <pointers>` to a tensor, usually the output of a model.
   :param pointer target: an :doc:`api-pointer <pointers>` to a tensor of desired targets.

   :return: A tensor with the calculated loss.

Loss is often calculated using these steps:

- build model
- define loss module
- define optimizer to apply gradients to model parameters
- given inputs, calculate model outputs
- use outputs & targets to calculate loss
- run backwards calculations to set gradients
- run an optimizer step to apply gradients to update parameters


In the example below, define ``x`` as both input and parameter to update, ``y`` as target, ``l`` as a mean-squared loss module and ``o`` as a stochastic gradient descent optimizer:

::

   q)x:tensor(.5 2 4e; `grad)
   q)y:tensor  1 2 3e
   q)l:loss`mse
   q)o:opt(`sgd;x;.1)

Then define the steps in a function ``f`` which sets to zero any previous gradient, calculates loss, calculates & applies gradients and returns loss and updated parameters:

::

   q)f:{[l;o;x;y;z]zerograd o; backward z:loss(l;x;y); step o; `loss`x!(return z;tensor x)}

Then, running the steps repeatedly until the loss drops below ``.1``:

::

   q){.1<first x} f[l;o;x;y]\`loss`x!(0we;tensor x)
   loss       x                   
   -------------------------------
   0w         0.5       2 4       
   0.4166667  0.5333334 2 3.933333
   0.362963   0.5644445 2 3.871111
   0.3161811  0.5934815 2 3.813037
   ..
   0.1048393  0.7659145 2 3.468171
   0.09132666 0.7815202 2 3.43696 

Running until loss below ``1e-06``:

::

   q)-3#{1e-06<first x} f[l;o;x;y]\`loss`x!(0we;tensor x)
   loss         x                   
   ---------------------------------
   1.278361e-06 0.9991827 2 3.001635
   1.113556e-06 0.9992372 2 3.001526
   9.70067e-07  0.999288  2 3.001424   /x approaches value of y, 1 2 3

Functional form
^^^^^^^^^^^^^^^

The loss modules are also implemented as functions which can be called directly with outputs, targets and options:

.. function:: fn(output; target) -> tensor

.. function:: fn(output; target; options..) -> tensor

   :param pointer input: an :doc:`api-pointer <pointers>` to a tensor, usually the output of a model.
   :param pointer output: an :doc:`api-pointer <pointers>` to a tensor of desired targets.
   :param list options: scalar or list of settings, either positional, named or both.

   :return: A tensor with the calculated loss.

::

   q)x:tensor .5 2 5e
   q)y:tensor  1 2 3e

   q)tensor z:mse(x;y)
   1.416667e

   q)use[z]mse(x;y;`none)
   q)tensor z
   0.25 0 4e

The loss functions also accept and return k arrays, allowing options after the input & target tensors are supplied:

::

   q)mse(1 3 6 9.0; 1 2 4 8.0)
   1.5

   q)mse(1 3 6 9.0; 1 2 4 8.0; `none)
   0 1 4 1f

Losses with 3-4 tensors
^^^^^^^^^^^^^^^^^^^^^^^

Some loss mudules/functions require more than the ouput/target pair:

::

   q)select module,pytorch,args from help`loss where not args~\:2#`tensor
   module     pytorch                        args                        
   ----------------------------------------------------------------------
   bce        "torch.nn.BCELoss"             `tensor`tensor`tensor       
   bcelogits  "torch.nn.BCEWithLogitsLoss"   `tensor`tensor`tensor       
   cosineloss "torch.nn.CosineEmbeddingLoss" `tensor`tensor`tensor       
   ctc        "torch.nn.CTCLoss"             `tensor`tensor`tensor`tensor
   margin     "torch.nn.MarginRankingLoss"   `tensor`tensor`tensor       
   triplet    "torch.nn.TripletMarginLoss"   `tensor`tensor`tensor       



`cosineloss <https://pytorch.org/docs/stable/generated/torch.nn.CosineEmbeddingLoss.html>`_,
`margin <https://pytorch.org/docs/stable/generated/torch.nn.MarginRankingLoss.html>`_ and 
`triplet <https://pytorch.org/docs/stable/generated/torch.nn.TripletMarginLoss.html>`_ losses require three inputs:

.. function:: loss(module; output1; output2; target) -> tensor

   :param pointer module: an :doc:`api-pointer <pointers>` to an allocated loss module.
   :param pointer output1: an :doc:`api-pointer <pointers>` to a tensor model output.
   :param pointer output2: an :doc:`api-pointer <pointers>` to an additional tensor output.
   :param pointer target: an :doc:`api-pointer <pointers>` to a tensor of desired targets.

   :return: A tensor with the calculated loss.



`ctc <https://pytorch.org/docs/stable/generated/torch.nn.CTCLoss.html>`_ loss takes 4 tensor args,
output & target along with tensors with output & target lengths.

.. function:: loss(module; output; target; output lengths; target lengths) -> tensor

   :param pointer module: an :doc:`api-pointer <pointers>` to an allocated loss module.
   :param pointer output: an :doc:`api-pointer <pointers>` to a tensor model output.
   :param pointer target: an :doc:`api-pointer <pointers>` to a tensor of desired targets.
   :param pointer output lengths: an :doc:`api-pointer <pointers>` to a tensor with each output length.
   :param pointer target lengths: an :doc:`api-pointer <pointers>` to a tensor with each target length.

   :return: A tensor with the calculated loss.

Binary cross entropy
^^^^^^^^^^^^^^^^^^^^
`bce <https://pytorch.org/docs/stable/generated/torch.nn.BCELoss.html>`_ and 
`bcelogits <https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html>`_ both include an option for batch weights when initializing the loss module. 
This option does not work well as part of the loss module's definition: each batch requires redefining the module.  The k-api redefines these loss modules to allow the weights to be supplied as part of the forward calculation instead. An optional 3rd tensor in the forward calculation supplies the weights for each input in a batch. PyTorch requires that the batch weights be the same shape as the tensors of outputs and targets.

.. function:: loss(module; output; weight) -> tensor

   :param pointer module: an :doc:`api-pointer <pointers>` to an allocated loss module.
   :param pointer output: an :doc:`api-pointer <pointers>` to a tensor of model output.
   :param pointer target: an :doc:`api-pointer <pointers>` to a tensor of desired targets.
   :param pointer weight: an :doc:`api-pointer <pointers>` to an optional tensor of weights of same shape as output & target.

   :return: A tensor with the calculated loss.


Define the loss module without any reduction:

::

   q)l:loss(`bce;`none)
   q)a:tensor 0.1 .9 .5e
   q)b:tensor 0 1 2e

   q)tensor r:loss(l;a;b)
   0.1053605 0.1053605 0.6931472e

Recalculate the loss with weights:

::

   q)w:tensor 1 1.1 .5
   q)use[r]loss(l;a;b;w)
   q)tensor r
   0.1053605 0.1158966 0.3465736e

A similar invocation using the functional form and k values instead of allocated tensors:

::

   q)bce(0.1 .9 .5e; 0 1 2e; 1 1.1 .5; `none)
   0.1053605 0.1158966 0.3465736e

`bcelogits <https://pytorch.org/docs/stable/generated/torch.nn.BCEWithLogitsLoss.html>`_, binary cross entropy with logits, also includes weights for each classification, along with batch-specific weights. For the module, only the class weights are part of the module definition:

::

   q)options`bcelogits
   weight| ::
   reduce| `mean

   q)l:loss(`bcelogits; 5#1.0; `none)   /same wt for each of 5 classes
   q)options l
   weight| 1 1 1 1 1f
   reduce| `none

   q)tensor r:loss(l; 3 5#1.5e; 3 5#1e)
   0.2014133 0.2014133 0.2014133 0.2014133 0.2014133
   0.2014133 0.2014133 0.2014133 0.2014133 0.2014133
   0.2014133 0.2014133 0.2014133 0.2014133 0.2014133

Redefine the module with different weights for each class:

::

   q)l:loss(`bcelogits; 1 .1 1.5 2 .25; `none)
   q)tensor r:loss(l; 3 5#1.5e; 3 5#1e)
   0.2014133 0.02014133 0.30212 0.4028267 0.05035333
   0.2014133 0.02014133 0.30212 0.4028267 0.05035333
   0.2014133 0.02014133 0.30212 0.4028267 0.05035333

In addition to the class-level weights, the loss for the module can be calculated with batch-level weights:

::

   q)l:loss(`bcelogits; 1 .1 1.5 2 .25; `none)
   q)tensor r:loss(l; 3 5#1.5e; 3 5#1e)
   0.2014133 0.02014133 0.30212 0.4028267 0.05035333
   0.2014133 0.02014133 0.30212 0.4028267 0.05035333
   0.2014133 0.02014133 0.30212 0.4028267 0.05035333

   q)w:tensor(`randn; 3 5)  /random batch-level weights

   q)tensor r:loss(l; 3 5#1.5e; 3 5#1e; w)
   -0.2860931 0.03823455    -0.3896965  -0.02535867 -0.08954446
   0.05641879 -0.0001597845 -0.3369139  -0.7810489  -0.05370528
   -0.1737809 -0.0126433    -0.07412164 0.03684831  0.0189163  


To distinguish the two cases of invoking the functional equivalent of binary cross entropy with logits, the case with class-level weights only and the case with both class and batch-level weights, the k api interface defines two functions:

- bcelogit1: expects 2 tensors for output & target, along with the options for class-level weight and reduction method.
- bcelogit2: allows up to 3 batch-level tensors, output, target and batch weight, along with the options for class-level weight and reduction.

::

   q)\P 4

   q)bcelogit1(3 5#1.5e; 3 5#1e;`none)
   0.2014 0.2014 0.2014 0.2014 0.2014
   0.2014 0.2014 0.2014 0.2014 0.2014
   0.2014 0.2014 0.2014 0.2014 0.2014

With class weights:

::

   q)bcelogit1(3 5#1.5e; 3 5#1e; .25 .5 1 2 4e; `none)
   0.05035 0.1007 0.2014 0.4028 0.8057
   0.05035 0.1007 0.2014 0.4028 0.8057
   0.05035 0.1007 0.2014 0.4028 0.8057

Adding batch-level weights:

::

   q)show w:expand(3 1#.1 1 2e;-1 5)
   0.1 0.1 0.1 0.1 0.1
   1   1   1   1   1  
   2   2   2   2   2  

   q)bcelogit2(3 5#1.5e; 3 5#1e; w; .25 .5 1 2 4e; `none)
   0.005035 0.01007 0.02014 0.04028 0.08057
   0.05035  0.1007  0.2014  0.4028  0.8057 
   0.1007   0.2014  0.4028  0.8057  1.611  
