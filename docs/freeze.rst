Freeze layers
=============

To "freeze" or turn off training for all or selected parts of a network it's necessary to turn off the gradient flag on the learnable parameters.

::

   q)q:module(`sequential; enlist(`linear;`a;64;10); `relu`b)

   q)parmtypes q
   a.weight| linear
   a.bias  | linear

   q)p:parm(q;`a.weight)
   q)gradient p
   `grad

   q)to(p;`nograd)
   q)gradient p
   `nograd

   q)gradflag(p;0b)  /use flag instead of tensor option symbol
   q)gradflag p
   0b

   q)free p

The k api uses the functions :func:`freeze` and :func:`unfreeze` to make this process simpler.

freeze all parameters
*********************

.. function:: freeze(module) -> null
.. function:: unfreeze(module) -> null

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module, model or parameter dictionary
   :return: turns gradient calculation off/on (freeze/unfreeze) for all parameters in the module. Null return.

::

   q)q:module(`sequential; enlist(`linear;`a;64;10); `relu`b)
   q)p:parms q

   q)gradient p
   a.weight| grad
   a.bias  | grad

   q)freeze q
   q)gradient p
   a.weight| nograd
   a.bias  | nograd

   q)unfreeze q
   q)gradient p
   a.weight| grad
   a.bias  | grad

Freeze selected parameters
**************************

.. function:: freeze(module;names) -> null
.. function:: unfreeze(module;names) -> null

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module, model or parameter dictionary
   :param symbol names: a single parameter name or list of names to freeze/unfreeze
   :return: turns gradient calculation off/on (freeze/unfreeze) for parameter(s) named. Null return.

::

   q)q:module seq(`sequential; (`conv2d;`a;3;64;3); (`batchnorm2d;`b;64); `relu)
   q)gradflag q
   a.weight| 1
   a.bias  | 1
   b.weight| 1
   b.bias  | 1

   q)freeze(q;`b.weight`b.bias)
   q)gradflag q
   a.weight| 1
   a.bias  | 1
   b.weight| 0
   b.bias  | 0


Freeze and set value(s)
***********************

.. function:: freeze(module;names;values) -> null
.. function:: unfreeze(module;names;values) -> null

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module or containing model
   :param symbol names: a single parameter name or list of names to freeze/unfreeze
   :param array,tensor values: corresponding tensor/vector or k array values
   :return: turns gradient calculation off/on (freeze/unfreeze) for parameter(s) named and sets their values to those supplied. Null return.

::

   q)q:module seq(`sequential; (`linear;`a;4;4); `relu)
   q)p:parms q
   q)first dict p
   0.389  -0.367  0.24   -0.015
   0.0758 -0.391  0.205  -0.41 
   -0.143 -0.221  -0.231 -0.218
   0.13   -0.0675 0.0174 -0.16 

   q)i:tensor(`eye;4)
   q)freeze(q;`a.weight`a.bias;(i;0.0))

   q)dict(p;`a.weight)
   1 0 0 0
   0 1 0 0
   0 0 1 0
   0 0 0 1

   q)gradflag p
   a.weight| 0
   a.bias  | 0


Freeze with dictionary
**********************

Instead of supplyint a PyTorch module object, along with names and values, a single k dictionary or PyTorch dictionary can also be used as a 2nd argument, supplying parameter names and new values with a single argument:

.. function:: freeze(module;dictionary) -> null
.. function:: unfreeze(module;dictionary) -> null

   :param pointer module: An :doc:`api-pointer <pointers>` to an allocated module or containing model
   :param kdict,tensordict dictionary: a k-dictionary of name-value pairs or a tensor dictionary of name-tensor pairs
   :return: turns gradient calculation off/on (freeze/unfreeze) for parameter keys, setting their values to those in the dictionary,. Null return.

::

   q)q:module seq(`sequential; (`linear;`a;4;4); `relu)
   q)d:dict(`a.weight`a.bias; (tensor(`eye;4); 0))
   q)freeze(q;d)

   q)gradflag q
   a.weight| 0
   a.bias  | 0

   q)w:parm(q;`a.weight)
   q)tensor w
   1 0 0 0
   0 1 0 0
   0 0 1 0
   0 0 0 1

