.. _start:

Quick start
===========

Once the interface library is :doc:`built <build>`, define the functions in the root namespace:

::

   q){key[x]set'x;}(`ktorch 2:`fns,1)[];
   
Create a tensor, retrieve its values:

::

   q)t:tensor 1 2 3.0

   q)tensor t
   1 2 3f

If CUDA devices are available, move the tensor to the default gpu:

::

   q)to(t;`cuda)

Check the global table of created PyTorch objects:

::

   q)obj[]
   ptr      class  device dtype  size elements bytes
   -------------------------------------------------
   81096864 tensor cuda:0 double 3    3        24   

   q)free t


   q)tensor t:tensor 1; show obj[]; free t
   ptr      obj    device dtype size elements bytes
   ------------------------------------------------
   37532000 tensor cpu    long       1        8    
  
Quick regression using a PyTorch module and a gradient descent optimizer:

::

   q)y:2*x:0N 1#1 2 3e
   q)o:opt(`sgd; m:module enlist(`linear;1;1); .1)

   q)dict p:parms m  /parameters of the module, randomly initialized
   weight| 0.0002031326
   bias  | -0.5911677  

Having created the input ``x`` and target ``y``, along with a linear module ``m`` and an optimizer ``o`` to perform gradient descent,
create a function to zero out any previous gradients, calculate the output of the module given inputs and the mean-squared error compared to the targets:

::

   q)f:{[m;o;x;y]zerograd o; backward z:mse(yh:forward(m;x);y); step o; free z; return yh}

The ``backward`` call will calculate the gradients and the optimizer ``step`` will apply an update to the parameters using the learning rate multiplied by the calculated gradient (10% of the gradient in this case).
Each call to ``f`` will calculate ``yhat = weight*x + bias``, calculate the mean-squared loss vs ``y``, then adjust ``weight`` and ``bias`` to have ``y = yhat``

::

   q)\ts:100 yhat:f[m;o;x;y]
   8 1440

   q)dict p
   weight| 1.990668  
   bias  | 0.02121399

   q)([]x;y;yhat)
   x y yhat    
   ------------
   1 2 2.012174
   2 4 4.002613
   3 6 5.993051

   q)mse(y;yhat)
   6.777422e-05


The PyTorch objects used in this example:

::

   q)obj[]
   ptr      class      device dtype size elements bytes
   ----------------------------------------------------
   57491376 optimizer  cpu          2    0        0    
   57492432 dictionary cpu          2    2        8    
   57485984 module     cpu          2    2        8    


   q)free(m;o;p)

   q)obj[]
   ptr class device dtype size elements bytes
   ------------------------------------------


A more detailed model using two linear layers and an activation function is available in the `examples <https://ktorch-examples.readthedocs.io/en/latest/start.html>`_.
