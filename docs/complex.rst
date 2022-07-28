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

.. function:: tensor(value;options) -> tensor pointer

   | Create a tensor from k value.

   :param scalar,list,array value: the k value to populate the real part of the complex tensor.
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <tensor-attributes>`. Must include data type of ```cfloat`` or ```cdouble`` amongst the supplied options.
   :return: An :doc:`api-pointer <pointers>` to the allocated complex tensor

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
The :ref:`complex <tensor-complex>` creaton mode allows the user to create a complex tensor from the k session by specifying the real and imaginary parts.

::

   q)t:tensor(`complex; 1 2 3; -1 0 2)
   q)dtype t
   `cfloat

   q)tensor t
   1  2 3
   -1 0 2

Most of the other :ref:`creation modes <tensor-modes>`  will also create complex tensors if data type is set to ```cfloat`` or ```cdouble`` as part of the tensor options. 
Usually only the real part of the tensor is defined, with the imaginary part set to zero.
This is true for :ref:`zeros <tensor-by-size>` and :ref:`ones <tensor-by-size>`, along with :ref:`full <tensor-full>`, 
:ref:`linspace <tensor-even-spaced>`,
:ref:`logspace <tensor-even-spaced>` and
:ref:`eye <tensor-identity>`.

::

   q)t:tensor(`full;7;2.5;`cfloat)
   q)tensor t
   2.5 2.5 2.5 2.5 2.5 2.5 2.5
   0   0   0   0   0   0   0  

   q)use[t]tensor(`ones;2 3;`cfloat)

   q)tensor(t;0)
   1 1 1
   0 0 0

   q)real t
   1 1 1
   1 1 1

   q)imag t
   0 0 0
   0 0 0

   q)use[t]tensor(`linspace;0;9;10;`cfloat)
   q)tensor t
   0 1 2 3 4 5 6 7 8 9
   0 0 0 0 0 0 0 0 0 0

Exceptions are creation modes 
:ref:`empty <tensor-by-size>`,
:ref:`rand <tensor-random>` and
:ref:`randn <tensor-random>`, which can define non-zero imaginary parts in the created complex tensor.

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

Creation modes :ref:`arange <tensor-range>`, :ref:`randint <tensor-randint>` and :ref:`randperm <tensor-randperm>` don't allow complex types.

::

   q)tensor(`arange; 10; `cfloat)
   '"arange_cpu" not implemented for 'ComplexFloat'
     [0]  tensor(`arange; 10; `cfloat)
          ^

Real & imaginary parts
**********************

After a complex tensor is created, there are some PyTorch information functions that allow retrieval of the real and imaginary parts:

- `real <https://pytorch.org/docs/stable/generated/torch.real.html>`_: given an :doc:`api-pointer <pointers>` to a complex tensor, returns the real part.
- `imag <https://pytorch.org/docs/stable/generated/torch.imag.html>`_: given an :doc:`api-pointer <pointers>` to a complex tensor, returns the imaginary part.
- `isreal <https://pytorch.org/docs/stable/generated/torch.isreal.html>`_: given an :doc:`api-pointer <pointers>` to a complex tensor, returns true where the imaginary part is zero.

The k interface implements these functions to return either k values or new tensors:

.. function:: real(ptr) -> value
.. function:: real(enlisted-ptr) -> tensor pointer
.. function:: imag(ptr) -> value
.. function:: imag(enlisted-ptr) -> tensor pointer

   :param ptr: a previously allocated :doc:`api-pointer <pointers>` to a complex tensor
   :return: Given a ptr, returns a k array containing the real or imaginary parts of the allocated tensor. If the ptr is enlisted, returns a new :doc:`api-pointer <pointers>` to a tensor with the real or imaginary values.

::

   q)t:tensor(`complex;1 2 3;-1 0 2)

   q)real t
   1 2 3e

   q)i:imag enlist t

   q)dtype i
   `float

   q)tensor i
   -1 0 2e

The ``isreal`` function returns a boolean k value or a pointer to an allocated boolean tensor:

.. function:: isreal(ptr) -> value
.. function:: isreal(enlisted-ptr) -> tensor pointer

   :param ptr: a previously allocated :doc:`api-pointer <pointers>` to a complex tensor
   :return: Given a ptr, returns a k boolean array with 1's where the imaginary part of the complex tensor is zero. If the ptr is enlisted, returns a new :doc:`api-pointer <pointers>` to a tensor with the boolean values.

::

   q)t:tensor(`complex;1 2 3;-1 0 2)
   q)isreal t
   010b

   q)b:isreal enlist t

   q)dtype b
   `bool

   q)tensor b
   010b

..
   - `abs <https://pytorch.org/docs/stable/generated/torch.abs.html>`_:
   - `angle <https://pytorch.org/docs/stable/generated/torch.angle.html>`_:
   - `sgn <https://pytorch.org/docs/stable/generated/torch.sgn.html>`_:
