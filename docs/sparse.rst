.. _sparse:

Sparse tensors
==============

PyTorch `documentation on sparse tensors <https://pytorch.org/docs/stable/sparse.html>`_ describes the implementation as "in beta" and subject to change.  

Creating from a k value
***********************

In a k session, sparse tensors are created and retrieved using the same :func:`tensor` function used to create dense tensors.
Creating a tensor directly from a k value requires adding the ```sparse`` option for layout.

.. function:: tensor(value;options) -> ptr

   | Create a tensor from k value.

   :param scalar,list,array value: the k values for the sparse tensor; only non-zero values will be included.
   :param sym options: one or more symbols for device, data type and other :ref:`tensor attributes <Setting properties>`. Must include layout of ```sparse`` among the supplied options.
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


sparse
******

There's also a separate :func:`sparse` function to create a sparse tensor from k arrays or existing dense tensors, with an option of specifying the number of sparse dimensions.

.. function:: sparse(input) -> ptr
.. function:: sparse(input;sparsedim) -> ptr

   :param input: k value or an :doc:`api-pointer <pointers>` to an existing dense tensor.
   :param long sparsedim: an optional number of sparse dimensions, must be less than or equal to total numer of dimensions.
   :return: An :doc:`api-pointer <pointers>` to the allocated sparse tensor

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

   q)indices b       / indices of rows w'non-zero values
   1 3

   q)values b
   0 9 0
   3 0 0


indices
^^^^^^^
The indices of a sparse tensor are returned as a matrix or pointer to a 2-dimensional matrix: one row per sparse dimension and one column for each sparse value.

.. function:: indices(ptr) -> value
.. function:: indices(enlisted-ptr) -> ptr

   :param ptr: a previously allocated :doc:`api-pointer <pointers>` to a sparse tensor
   :return: Given a ptr, returns a k matrix containing indices of the non-zero values. If the ptr is enlisted, returns a new :doc:`api-pointer <pointers>` to a tensor with the indices.

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
.. function:: values(enlisted-ptr) -> ptr

   :param ptr: a previously allocated :doc:`api-pointer <pointers>` to a sparse tensor
   :return: Given a ptr, returns a k array containing non-zero values of the tensor. If the ptr is enlisted, returns a new :doc:`api-pointer <pointers>` to a tensor with the values.

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

nnz
^^^

.. function:: nnz(ptr) -> n

   :param ptr: a previously allocated :doc:`api-pointer <pointers>` to a sparse tensor
   :return: Given a ptr, returns long integer scalar with the count of non-zero values.

sparsedim and densedim
^^^^^^^^^^^^^^^^^^^^^^

coalesce and coalesced
^^^^^^^^^^^^^^^^^^^^^^




