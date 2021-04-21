.. _overview:

Overview
========

::

   >>> x=torch.tensor([1,2,3.0], dtype=torch.double, requires_grad=True)
   >>> y=x * x
   >>> z=torch.mean(y)

   >>> z.backward()
   >>> x.grad
   tensor([0.6667, 1.3333, 2.0000], dtype=torch.float64)


::

   >>> x=torch.tensor([1,2,3.0], dtype=torch.double, requires_grad=True)
   >>> torch.mean(x*x).backward()
   >>> x.grad
   tensor([0.6667, 1.3333, 2.0000], dtype=torch.float64)

::

   q)x:tensor(1 2 3.0; `grad)
   q)y:mul(x;x)
   q)z:mean(y)

   q)backward z
   q)grad x
   0.6666667 1.333333 2
   q)free'[(x;y;z)];

