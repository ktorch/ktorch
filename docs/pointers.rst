.. _pointers:

Pointers
========

The k interface returns a pointer to allocated values (tensor, module, optimizer, loss function or model) that can then be used in subsequent function calls. Pointers are 1-element general lists with a scalar long value to distinguish these values from long scalars and lists.

The api maintains a map of pointers that can be viewed via ``obj`` and released via ``free``.
