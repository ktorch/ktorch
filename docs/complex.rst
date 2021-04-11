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

Most of the creation modes


Complex information
*******************

After a complex tensor is created, there are some information functions that allow retrieval of parts of the complex tensor.

- `real <https://pytorch.org/docs/stable/generated/torch.real.html>`_: given an :ref:`api-pointer <pointers>` to a complex tensor, returns the real part.
- `imag <https://pytorch.org/docs/stable/generated/torch.imag.html>`_: given an :ref:`api-pointer <pointers>` to a complex tensor, returns the imaginary part.
- `isreal <https://pytorch.org/docs/stable/generated/torch.isreal.html>`_: given an :ref:`api-pointer <pointers>` to a complex tensor, returns true where the imaginary part is zero.

.. function:: real(ptr) -> value
.. function:: real(enlisted-ptr) -> ptr
.. function:: imag(ptr) -> value
.. function:: imag(enlisted-ptr) -> ptr
.. function:: isreal(ptr) -> value
.. function:: isreal(enlisted-ptr) -> ptr


- `abs <https://pytorch.org/docs/stable/generated/torch.abs.html>`_:
- `angle <https://pytorch.org/docs/stable/generated/torch.angle.html>`_:
- `sgn <https://pytorch.org/docs/stable/generated/torch.sgn.html>`_:
