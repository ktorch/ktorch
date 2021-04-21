.. _sparse:

Sparse tensors
==============

PyTorch `documentation on sparse tensors <<https://pytorch.org/docs/stable/sparse.html>`_ describes the implementation as "in beta" and subject to change.  

Creating from a k value
***********************

In a k session, sparse tensors are created and retrieved using the same :func:`tensor` function used to create dense tensors.
Creating a tensor directly from a k value requires adding ```sparse`` option for layout.

.. function:: tensor(value;options) -> ptr

   | Create a tensor from k value.

   :param scalar,list,array value: the k values for the sparse tensor; only non-zero values will be included.
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <Setting properties>`. Must include layout of ```sparse`` among the supplied options.
   :return: An :ref:`api-pointer <pointers>` to the allocated sparse tensor

::

   q)show x:./[3 4#0;(0 1;2 3;1 2);:;12 -99 4]
   0 12 0 0  
   0 0  4 0  
   0 0  0 -99

   q)t:tensor(x;`sparse)
   q)tensor t
   0 12 0 0  
   0 0  4 0  
   0 0  0 -99

   q)indices t
   0 1 2
   1 2 3

   q)values t
   12 4 -99

Tensor creation modes
*********************
The :ref:`sparse <tensor-sparse>` creaton mode allows the user to create a sparse tensor from the k session by specifying indices and values.

::

   q)t:tensor(`sparse; flip(0 1;2 3;1 2); 12 -99 4; `double)

   q)indices t
   0 2 1
   1 3 2

   q)nnz t  / number of non-zero values
   3

   q)size t
   3 4

   q)values t
   12 -99 4f


A few of the other :ref:`creation modes <tensor-modes>`  will also create sparse tensors if layout is set to ```sparse`` as part of the tensor options.

::

