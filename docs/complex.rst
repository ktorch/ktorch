.. _complex:

Complex tensors
===============

`Complex numbers <https://pytorch.org/docs/stable/complex_numbers.html>`_ are described as a work in progress in Pytorch, and are represented using the notation of complex numbers in python:

::

   >>> complex(3.0, -2.0)
   (3-2j)

   >>> torch.complex(torch.tensor([3,4.0]), torch.tensor([-2,7.0]))
   tensor([3.-2.j, 4.+7.j])


In a k session, complex tensors are created and retrieved using the same :doc:`tensor <tensors>` function.
Creating a tensor directly from a k value requires adding the complex data type, ```cfloat`` or ```cdouble``, as part of the tensor options:

.. function:: tensor(value;options) -> ptr

   | Create a tensor from k value.

   :param scalar,list,array value: the k value to populate the real part of the complextensor.
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


- `real <https://pytorch.org/docs/stable/generated/torch.real.html>`_:
- `imag <https://pytorch.org/docs/stable/generated/torch.imag.html>`_:
- `isreal <https://pytorch.org/docs/stable/generated/torch.isreal.html>`_:

- `abs <https://pytorch.org/docs/stable/generated/torch.abs.html>`_:
- `angle <https://pytorch.org/docs/stable/generated/torch.angle.html>`_:
- `sgn <https://pytorch.org/docs/stable/generated/torch.sgn.html>`_:
