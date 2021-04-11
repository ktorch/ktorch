.. _complex:

Complex tensors
===============

`Complex numbers <https://pytorch.org/docs/stable/complex_numbers.html>`_ are described as a work in progress in Pytorch, and are represented using the notation of complex numbers in python:

::

   >>> complex(3.0, -2.0)
   (3-2j)

   >>> torch.complex(torch.tensor([3,4.0]), torch.tensor([-2,7.0]))
   tensor([3.-2.j, 4.+7.j])


In a k session, complex tensors are created and retrieved using the same ``tensor`` interface function, along with a few other routines designed to match the ones in PyTorch.


- `real <https://pytorch.org/docs/stable/generated/torch.real.html>`_:
- `imag <https://pytorch.org/docs/stable/generated/torch.imag.html>`_:
- `isreal <https://pytorch.org/docs/stable/generated/torch.isreal.html>`_:

- `abs <https://pytorch.org/docs/stable/generated/torch.abs.html>`_:
- `angle <https://pytorch.org/docs/stable/generated/torch.angle.html>`_:
- `sgn <https://pytorch.org/docs/stable/generated/torch.sgn.html>`_:
