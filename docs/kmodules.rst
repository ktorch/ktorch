.. _kmodules:

Modules for k api
=================

The following modules are not directly available in PyTorch;
these modules were defined for the k api either for convenience when only the functional form is defined in PyTorch,
or to allow modules to be defined with a sequential or sequential-like container which defines the forward calculation implicitly.

Convenience modules
*******************


Pad
^^^


Squeeze & Unsqueeze
^^^^^^^^^^^^^^^^^^^

Expand & Reshape
^^^^^^^^^^^^^^^^

Cat
^^^

Mul
^^^

One hot
^^^^^^^


Select
^^^^^^


IndexSelect
^^^^^^^^^^^


Container modules
*****************


Fork
^^^^


Recur
^^^^^


Residual
^^^^^^^^


NBeats
^^^^^^


Sequential join
^^^^^^^^^^^^^^^


Sequential nest
^^^^^^^^^^^^^^^


Transformation modules
**********************


Transform container
^^^^^^^^^^^^^^^^^^^

Random cropping
^^^^^^^^^^^^^^^

Random flipping
^^^^^^^^^^^^^^^

Zscore
^^^^^^
