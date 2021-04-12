.. _complex:

Complex tensors
===============

`Complex numbers <https://pytorch.org/docs/stable/complex_numbers.html>`_ are described as a work in progress in Pytorch, and are represented using the notation of complex numbers in python:

::

   >>> complex(3.0, -2.0)
   (3-2j)

   >>> torch.complex(torch.tensor([3,4.0]), torch.tensor([-2,7.0]))
   tensor([3.-2.j, 4.+7.j])


Creating from a k value
***********************

In a k session, complex tensors are created and retrieved using the same :func:`tensor` function used to create real-valued tensors.
Creating a tensor directly from a k value requires adding the complex data type, ```cfloat`` or ```cdouble``, as part of the tensor options:

.. function:: tensor(value;options) -> ptr

   | Create a tensor from k value.

   :param scalar,list,array value: the k value to populate the real part of the complex tensor.
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <Setting properties>`. Must include data type of ```cfloat`` or ```cdouble`` amongst the supplied options.
   :return: An :ref:`api-pointer <pointers>` to the allocated complex tensor

::

   q)t:tensor(1 2 3;`cfloat)
   q)tensor t      / return (real;imag)
   1 2 3
   0 0 0

   q)tensor(t;0b)  / return real,'imag
   1 0
   2 0
   3 0

   q)real t        / return real values only
   1 2 3e

   q)imag t        / return imaginary values only
   0 0 0e

Tensor creation modes
*********************

Most of the :ref:`creation modes <tensor--modes>`  will also create complex tensors if data type is set to ```cfloat`` or ```cdouble`` as part of the tensor options. 
Usually only the real part of the tensor is defined, with the imaginary part set to zero.
This is true for :ref:`zeros <tensor-by-size>` and :ref:`ones <tensor-by-size>`
Exceptions are creation modes 

::

   / create uninitialized tensor, real & imaginary parts may be any value
   q)tensor t:tensor(`empty;5;`cdouble)
   1.736005e-310 2.48021e-321  4.056773e-320 1.743432e-310 2.48021e-321 
   4.044421e-320 1.740885e-310 2.48021e-321  4.070607e-320 1.751071e-310

   q)use[t]tensor(`rand;5;`cdouble); tensor t   / uniform random 
   0.1500104 0.3352091 0.2414377  0.4360392 0.291383
   0.5904075 0.1125289 0.01854667 0.2212064 0.355647

   q)use[t]tensor(`randn;5;`cdouble); tensor t  / standard normal
   0.2019709 -0.6007159 -0.1383445 0.3822946 -0.3757848
   -0.465213 -0.335503  1.170153   -1.166904 0.6392463 

Complex information
*******************

After a complex tensor is created, there are some PyTorch information functions that allow retrieval of the real and imaginary parts:

- `real <https://pytorch.org/docs/stable/generated/torch.real.html>`_: given an :ref:`api-pointer <pointers>` to a complex tensor, returns the real part.
- `imag <https://pytorch.org/docs/stable/generated/torch.imag.html>`_: given an :ref:`api-pointer <pointers>` to a complex tensor, returns the imaginary part.
- `isreal <https://pytorch.org/docs/stable/generated/torch.isreal.html>`_: given an :ref:`api-pointer <pointers>` to a complex tensor, returns true where the imaginary part is zero.

The k interface implements these functions to return either k values or new tensors:

.. function:: real(ptr) -> value
.. function:: imag(ptr) -> value
.. function:: real(enlisted-ptr) -> ptr
.. function:: imag(enlisted-ptr) -> ptr

   :param ptr ptr: a previously allocated :ref:`api-pointer <pointers>` to a complex tensor
   :return: Given a ptr, returns a k array containing the real or imaginary parts of the allocated tensor. If the ptr is enlisted, returns a new :ref:`ptr <pointers>` to a tensor with the real or imaginary values.

::

   q)t:tensor(`complex;1 2 3;-1 0 2)

   q)real t
   1 2 3e

   q)i:imag enlist t

   q)dtype i
   `float

   q)tensor i
   -1 0 2e

.. function:: isreal(ptr) -> value
.. function:: isreal(enlisted-ptr) -> ptr

   :param ptr ptr: a previously allocated :ref:`api-pointer <pointers>` to a complex tensor
   :return: Given a ptr, returns a k boolean array with 1's where the imaginary part of the complex tensor is zero. If the ptr is enlisted, returns a new :ref:`ptr <pointers>` to a tensor with the boolean values.

::

   q)t:tensor(`complex;1 2 3;-1 0 2)
   q)isreal t
   010b

   q)b:isreal enlist t

   q)dtype b
   `bool

   q)tensor b
   010b


- `abs <https://pytorch.org/docs/stable/generated/torch.abs.html>`_:
- `angle <https://pytorch.org/docs/stable/generated/torch.angle.html>`_:
- `sgn <https://pytorch.org/docs/stable/generated/torch.sgn.html>`_:
