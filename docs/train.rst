
Training steps
==============

Given a set of inputs and targets, the main training steps are:

- reset any previously calculated gradients to zero
- call the forward method of the module describing the neural network to get model outputs
- call the loss function with the model outputs and targets
- compute the gradients on all the parameters involved in the loss calculation
- clip the gradients (optional)
- use an optimizer to reduce the parameters by some fraction of the gradients


From the starting examples using generated `spirals <https://github.com/ktorch/examples/blob/master/start/spirals.q>`_,
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
   q)x:tensor "e"$raze spiral[n]'[til k;.2*tensor z]  /generate spirals w'noise
   q)y:tensor til[k]where k#n                         /class 0,1,2
 
With these  PyTorch objects, the k api training steps become:

::

   q)zerograd o          /set gradients to zero
   q)rx:forward(q;x)     /calculate the output of the module
   q)ry:loss(l;rx;y)     /calculate loss from output & target
   q)backward ry         /backward calc computes parameter gradients
   q)step o              /use optimizer to update parameters from gradients

Putting all the steps into a function ``f`` which frees intermediate tensors and returns loss:

::

   q)f:{[q;l;o;x;y]zerograd o; rx:forward(q;x); r:tensor ry:loss(l;rx;y); backward ry; free each(rx;ry); step o; r}

   q)\ts:3 show f[q;l;o;x;y]
   1.031206e
   0.9710665e
   0.9085126e
   5 4194736
