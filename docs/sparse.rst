.. _sparse:

Sparse tensors
==============

PyTorch `documentation on sparse tensors <https://pytorch.org/docs/stable/sparse.html>`_ describes the implementation as "in beta" and subject to change.  

Creating from a k value
***********************

In a k session, sparse tensors are created and retrieved using the same :func:`tensor` function used to create dense tensors.
Creating a tensor directly from a k value requires adding the ```sparse`` option for layout.

.. function:: tensor(value;options) -> tensor pointer

   | Create a tensor from k value.

   :param scalar,list,array value: the k values for the sparse tensor; only non-zero values will be included.
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <tensor-attributes>`. Must include layout of ```sparse`` among the supplied options.
   :return: An :doc:`api-pointer <pointers>` to the allocated sparse tensor

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

Sparse creation mode
********************

The :ref:`sparse <tensor-sparse>` creaton mode of the same :func:`tensor` function allows the user to create a sparse tensor from the k session by specifying indices and values.

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


Sparse functions
****************

sparse
^^^^^^
There's also a separate :func:`sparse` function to create a sparse tensor from k arrays or existing dense tensors, with an option of specifying the number of sparse dimensions. This is somewhat the same as PyTorch's `tensor.to_sparse() method <https://pytorch.org/docs/stable/sparse.html#torch.Tensor.to_sparse>`_.

.. function:: sparse(input) -> tensor
.. function:: sparse(input;sparsedim) -> tensor

   :param input: k value or an :doc:`api-pointer <pointers>` to an existing dense tensor.
   :param long sparsedim: an optional number of sparse dimensions, must be less than or equal to total numer of dimensions.
   :return: An :doc:`api-pointer <pointers>` to a new sparse tensor

::

   q)show x:./[5 3#0.0;(1 1;3 0);:;9 3.0]
   0 0 0
   0 9 0
   0 0 0
   3 0 0
   0 0 0

   q)a:sparse x

   q)indices a
   1 3
   1 0
   q)values a
   9 3f

   q)b:sparse(x;1)   / 1 sparse dimension and 1 dense dimension

   q)indices b       / rows with non-zero values
   1 3

   q)values b
   0 9 0
   3 0 0

There's an additional form of the :func:`sparse` which is more in line with PyTorch's `tensor.sparse_mask(mask) method <https://pytorch.org/docs/stable/sparse.html#torch.Tensor.sparse_mask>`_.


.. function:: sparse(input;sparse-tensor) -> tensor

   :param input: k value or an :doc:`api-pointer <pointers>` to an existing dense tensor.
   :param ptr sparse-tensor: an :doc:`api-pointer <pointers>` to an existing sparse tensor
   :return: an :doc:`api-pointer <pointers>` to a new sparse tensor created by using values in given input at indices supplied by the sparse tensor argument.

::

   q)s:sparse(0 9.0 0; 0 0 -1.0; 5.5 0 0)
   q)values s
   9 -1 5.5
   q)indices s       /indices will be used to create new tensor
   0 1 2
   1 2 0

   q)show x:3 3#til 9
   0 1 2
   3 4 5
   6 7 8

   q)t:sparse(x;s)

   q)indices t       /same indices as given in sparse tensor
   0 1 2
   1 2 0

   q)values t        /but values from dense input
   1 5 6

   q)values[t]~.[x;]'[flip indices s]
   1b

sparseindex
^^^^^^^^^^^
This function derives the indices of a the non-zero values in the input array/tensor, with one row per sparse dimension and one column per non-zero value.

.. function:: sparseindex(input) -> tensor
.. function:: sparseindex(input;sparsedim) -> tensor

   :param input: k value or an :doc:`api-pointer <pointers>` to an existing dense tensor.
   :param long sparsedim: an optional number of sparse dimensions, must be less than or equal to total numer of dimensions.
   :return: if input is a k value, returns a k matrix of indices with as many rows as sparse dimensions, one column per non-zero value; if input is a tensor,  returns an :doc:`api-pointer <pointers>` to a tensor of indices.

::

   q)show x:./[5 3#0.0;(1 1;3 0);:;9 3.0]
   0 0 0
   0 9 0
   0 0 0
   3 0 0
   0 0 0

   q)sparseindex x
   1 3
   1 0

   q)sparseindex(x;1)
   1 3

   q)t:tensor x
   q)i:sparseindex(t;1) /input is tensor, so tensor is output
   q)tensor i
   1 3

indices
^^^^^^^
The indices of a sparse tensor are returned as a matrix or pointer to a 2-dimensional matrix: one row per sparse dimension and one column for each sparse value.

.. function:: indices(ptr) -> value
.. function:: indices(enlisted-ptr) -> tensor

   :param ptr: a previously allocated :doc:`api-pointer <pointers>` to a sparse tensor
   :return: given a ptr, returns a k matrix containing indices of the non-zero values. If the ptr is enlisted, returns a new :doc:`api-pointer <pointers>` to a tensor with the indices.

::

   q)t:tensor(2 4#0 0 0 1 0 0 3 0;`sparse`double)

   q)tensor t
   0 0 0 1
   0 0 3 0

   q)indices t
   0 1
   3 2

   q)values t
   1 3f

   q)nnz t
   2

values
^^^^^^

.. function:: values(ptr) -> list
.. function:: values(enlisted-ptr) -> tensor

   :param ptr: a previously allocated :doc:`api-pointer <pointers>` to a sparse tensor
   :return: given a ptr, returns a k array containing non-zero values of the tensor. If the ptr is enlisted, returns a new :doc:`api-pointer <pointers>` to a tensor with the values.

::

   q)show i:1 2#0 2  / indices of rows w'non-zero values
   0 2

   q)show v:0 10+\:til 5  / rows with non-zero values
   0  1  2  3  4 
   10 11 12 13 14

   q)t:tensor(`sparse; i; v; 4 5)

   q)tensor t
   0  1  2  3  4 
   0  0  0  0  0 
   10 11 12 13 14
   0  0  0  0  0 

   q)sparsedim t   / only 1st dimension is sparse
   1

   q)densedim t   /2nd dimension is dense
   1

   q)values t
   0  1  2  3  4 
   10 11 12 13 14

   q)nnz t
   2            / 2 rows have non-zero values


dense
^^^^^

The k api function :func:`dense` is similar to the Pytorch `tensor.to_dense() method <https://pytorch.org/docs/stable/sparse.html?highlight=dense#torch.Tensor.to_dense>`_.

.. function:: dense(ptr) -> sparse-tensor pointer

   :param ptr: a previously allocated :doc:`api-pointer <pointers>` to a sparse tensor
   :return: an `api-pointer <pointers>` to a new dense tensor constructed from the sparse input.

nnz
^^^

.. function:: nnz(ptr) -> n

   :param ptr: a previously allocated :doc:`api-pointer <pointers>` to a sparse tensor
   :return: returns long integer scalar with the number of non-zero values; this count is over the sparse dimensions.

::

   q)show x:./[4 3#0.0;(1 1;1 2;2 2);:;9 3 2.0]
   0 0 0
   0 9 3
   0 0 2
   0 0 0

   q)nnz t:sparse x
   3

   q)values t
   9 3 2f

   q)use[t]sparse(x;1)  / 1 sparse dim, 1 dense dim
   q)nnz t              / count across sparse dim
   2                    / 2 rows

   q)values t
   0 9 3
   0 0 2


.. _sparsedim:

sparsedim
^^^^^^^^^

.. function:: sparsedim(ptr) -> dim

   :param ptr: a previously allocated :doc:`api-pointer <pointers>` to a tensor, sparse or dense.
   :return: returns long integer scalar with the number of sparse dimensions (zero for dense tensors).

::

   q)show x:4 3#(7#0),8
   0 0 0
   0 0 0
   0 8 0
   0 0 0

   q)s:sparse x
   q)sparsedim s
   2

   q)use[s]sparse(x; 1) /1 sparse, 1 dense dimension
   q)sparsedim s
   1

   q)densedim s
   1

.. _densedim:

densedim
^^^^^^^^

.. function:: densedim(ptr) -> dim

   :param ptr: a previously allocated :doc:`api-pointer <pointers>` to a tensor, sparse or dense.
   :return: returns long integer scalar with the number of dense dimensions.

::

   q)s:tensor(`sparse; 1 2#3 1; 2 3#til 6; 5 3) /indices & values of hybrid sparse

   q)tensor s
   0 0 0
   3 4 5
   0 0 0
   0 1 2
   0 0 0

   q)values s
   0 1 2
   3 4 5

   q)sparsedim s
   1

   q)densedim s
   1

.. _coalesce:

coalesce
^^^^^^^^

PyTorch sparse tensor format permits uncoalesced sparse tensors, where there may be duplicate indices; in this case, the interpretation is that the value at that index is the sum of all values with the same index.
The tensor can maintain the duplicates -- most operations work identically on coalesced or uncoalesced sparse tensors. But if the duplicate indices need to be removed, the :func:`coalesce` function will perform the operation, as well as sort the indices.

.. function:: coalesce(ptr) -> null

   :param ptr: a previously allocated :doc:`api-pointer <pointers>` to a sparse tensor.
   :returns: coalesces the sparse tensor in place, creating a new tensor where the indices are unique and the values for duplicate indices are summed. Returns null, upon completion, the tensor pointer refers to a coalesced sparse tensor.

::

   q)t:tensor(`sparse; 1 3#2 1 1; 9 5 -11; 10)

   q)indices t
   2 1 1

   q)values t
   9 5 -11

   q)tensor t
   0 -6 9 0 0 0 0 0 0 0

   q)coalesce t

   q)tensor t
   0 -6 9 0 0 0 0 0 0 0

   q)indices t
   1 2

   q)values t
   -6 9

coalesced
^^^^^^^^^

.. function:: coalesced(ptr) -> bool

   :param ptr: a previously allocated :doc:`api-pointer <pointers>` to a tensor, sparse or dense.
   :return: returns true if dense tensor or if the sparse tensor was created from dense input, else false for those sparse tensors created with indices and values and for which :func:`coalesce` has not been run.


