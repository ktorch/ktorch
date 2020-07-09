Tensors
=======


Tensor creation modes
*********************

In addition to supplying k/q values to create tensors, the following creation methods create tensors following a particulare distribution, sequence, etc.

- `arange <https://pytorch.org/docs/stable/torch.html#torch.arange>`_: returns a tensor with a sequence of integers
- `empty <https://pytorch.org/docs/stable/torch.html#torch.empty>`_: returns a tensor with uninitialized values
- `eye <https://pytorch.org/docs/stable/torch.html#torch.eye>`_: returns an identity matrix,
- `full <https://pytorch.org/docs/stable/torch.html#torch.full>`_: returns a tensor filled with a single value
- `linspace <https://pytorch.org/docs/stable/torch.html#torch.linspace>`_: returns a tensor with values linearly spaced in some interval
- `logspace <https://pytorch.org/docs/stable/torch.html#torch.logspace>`_: returns a tensor with values logarithmically spaced in some interval
- `ones <https://pytorch.org/docs/stable/torch.html#torch.ones>`_: returns a tensor filled with all ones,
- `rand <https://pytorch.org/docs/stable/torch.html#torch.rand>`_: returns a tensor filled with values drawn from a uniform distribution on ``[0, 1)``
- `randint <https://pytorch.org/docs/stable/torch.html#torch.randint>`_: returns a tensor with integers randomly drawn from an interval
- `randn <https://pytorch.org/docs/stable/torch.html#torch.randn>`_: returns a tensor filled with values drawn from a unit normal distribution
- `randperm <https://pytorch.org/docs/stable/torch.html#torch.randperm>`_: returns a tensor filled with a random permutation of integers in some interval
- `zeros <https://pytorch.org/docs/stable/torch.html#torch.zeros>`_: returns a tensor filled with all zeros

