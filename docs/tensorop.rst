Tensor operations
=================


.. _tensor-batch:

Indexing
********

select
^^^^^^

The PyTorch `select <https://pytorch.org/docs/stable/generated/torch.select.html#torch.select>`_ function is implemented as :func:`index`.

.. function:: index(input;dim;index) -> slice

   :param array,tensor input: a k array or tensor :doc:`pointer <pointers>`
   :param long dim: the dimension to slice
   :param long index: the index to select
   :return: a slice of the input array/tensor, returned as a k array/tensor.

::

   q)x:2 3 4#.1+til 24

   q)index(x;1;2)
   8.1  9.1  10.1 11.1
   20.1 21.1 22.1 23.1

   q)x:tensor x
   q)t:index(x;-1;-2)  /last dimension, 2nd to last index
   q)tensor t
   2.1  6.1  10.1
   14.1 18.1 22.1

   q)tensor[x][;;2]
   2.1  6.1  10.1
   14.1 18.1 22.1

index_select
^^^^^^^^^^^^

The PyTorch `index_select <https://pytorch.org/docs/stable/generated/torch.index_select.html>`_ function is also implemented as :func:`index`.

.. function:: index(input;dim;indices) -> slice

   :param array,tensor input: a k array or tensor :doc:`pointer <pointers>`
   :param long dim: the dimension to slice
   :param long,tensor indices: the list or tensor of indices to select
   :return: a slice of the input array/tensor, returned as a k array/tensor.

::

   q)x:2 3 4#.1+til 24
   q)index(x;-1;0 1)
   0.1 1.1   4.1 5.1   8.1 9.1  
   12.1 13.1 16.1 17.1 20.1 21.1

   q)i:tensor 0 1
   q)x:tensor x
   q)t:index(x;-1;i)
   q)tensor t
   0.1 1.1   4.1 5.1   8.1 9.1  
   12.1 13.1 16.1 17.1 20.1 21.1

   q)tensor[x][;;0 1]
   0.1 1.1   4.1 5.1   8.1 9.1  
   12.1 13.1 16.1 17.1 20.1 21.1


Batching tensors
****************

To train a network, a subset of data in memory is fed through a network to calculate its output and gradients,
which in turn allows an optimizer to update trainable parameters. This batching can be done via :func:`batch`,
which operates in one mode by dividing the tensor(s) in place into separate batches, returning a flag to indicate a batch is available to process:


batch
^^^^^

.. function:: batch(pointer;batchsize) -> flag

   :param pointer pointer: An :doc:`api-pointer <pointers>` to an allocated tensor, vector of tensors or a tensor dictionary. For vector or dictionary, the tensors must all have the same size across the batching dimension.
   :param long scalar/list batchsize: The size of the batch, expected as some fraction of total tensor size, typically in the first dimension. If batching across a different dimension is required, batchsize can be entered as a 2-element list, ``(batchsize;dimension)``.
   :return: A boolean is returned, ``true`` to indicate more batches remain, ``false`` to indicate that all batches have been processed and the tensor(s) have been restored to full size.

::

   q)t:tensor til 10

   q)while[batch(t;3); show tensor t]; show tensor t
   0 1 2
   3 4 5
   6 7 8
   ,9
   0 1 2 3 4 5 6 7 8 9

Batching is usually done across the first dimension, but batch size can also be specified together with a dimension:

::

   q)use[t]1 11+\:til 10

   q)tensor t
   1  2  3  4  5  6  7  8  9  10
   11 12 13 14 15 16 17 18 19 20

   q)while[batch(t;5 1); show tensor t; -2"";]; tensor t
   1  2  3  4  5 
   11 12 13 14 15

   6  7  8  9  10
   16 17 18 19 20

   1  2  3  4  5  6  7  8  9  10
   11 12 13 14 15 16 17 18 19 20

.. note::

   Batching in place across the first dimension usually results in contiguous tensors; many algorithms are optimized for contiguous tensors, so batching across another dimension may cause slower throughput if a particular algorithm needs to copy data to make it contiguous in memory or use a slower implementation.

::

   q)t:tensor til 10

   q)batch(t;3)
   1b

   q)tensor t
   0 1 2

   q)contiguous t
   1b
   
   q)use[t]1 11+\:til 10

   q)tensor t
   1  2  3  4  5  6  7  8  9  10
   11 12 13 14 15 16 17 18 19 20


   q)batch(t;3 1)
   1b

   q)tensor t
   1  2  3 
   11 12 13

   q)contiguous t
   0b

The :func:`batch` function can also be used with a 3rd argument indicating which batch is required:

.. function:: batch(pointer;batchsize;index) -> null

   :param pointer pointer: An :doc:`api-pointer <pointers>` to an allocated tensor, vector of tensors or a tensor dictionary. For vector or dictionary, the tensors must all have the same size across the batching dimension.
   :param long scalar/list batchsize: The size of the batch, expected as some fraction of total tensor size, typically in the first dimension. If batching across a different dimension is required, batchsize can be entered as a 2-element list, ``(batchsize;dimension)``.
   :param long index: the batch, 0 to n-1 where n is the ceiling of total size at batching dimension divided by batchsize.
   :return: null

::

   q)t:tensor til 10

   q)batch(t;3;0)

   q)tensor t
   0 1 2

   q)batch(t;3;2)

   q)tensor t
   6 7 8

   q)batch(t;3;4)
   'batch[4] invalid, valid range is from 0-2
     [0]  batch(t;3;4)
          ^

   q)w:3; n:ceiling restore[t]%w; n{batch(x;y;z);show tensor t; z+1}[t;w]/0;
   0 1 2
   3 4 5
   6 7 8
   ,9

   q)restore t
   10

If :func:`batch` is called using a vector or dictionary of pointers, each tensor must have the same size across the batching dimension:

::

   q)d:dict`x`y!(.1+til 10; til 9)

   q)while[batch(d;4); show dict d]
   'dictionary[y] size=9, but previous tensor(s) have size=10 for dim 0
     [0]  while[batch(d;4); show dict d]
                ^
   q)dict(d;`y;til 10)

   q)while[batch(d;4); show dict d]
   x| 0.1 1.1 2.1 3.1
   y| 0   1   2   3  
   x| 4.1 5.1 6.1 7.1
   y| 4   5   6   7  
   x| 8.1 9.1
   y| 8   9  

restore
^^^^^^^

If :func:`batch` is used without specifying an index, i.e. if the function is called until a false return, then the batched tensor(s) will be restored to their full size(s). But if there is some error prior to processing all the batches, or if the :func:`batch` function is used with an index for the batch required, then the tensor(s) may need to be restored to their full size via the function :func:`restore`.

.. function:: restore(ptr) -> size

.. function:: restore(ptr;dim) -> size

   :param pointer ptr: An :doc:`api-pointer <pointers>` to an allocated tensor, vector of tensors or a tensor dictionary. For vector or dictionary, the tensors must all have the same size across the batching dimension.
   :param long dim: Optional batching dimension, default is zero.
   :return: total restored size in the batching dimension.

::

   q)t:tensor til 10
   q)batch(t;4)
   1b

   q)tensor t
   0 1 2 3

   q)restore t
   10

   q)tensor t
   0 1 2 3 4 5 6 7 8 9

.. warning::
   :func:`restore` must be called with the same dimension that was used for batching. Typically batching is done over the first dimension(dimension=0), but tensor(s) can be corrupted if :func:`restore` is called without matching the same dimension given or implied in the :func:`batch` call. For example, if batching with dimension=1 and restoring without a matching dimension call may resize the tensor incorrectly with uninitialized values where the tensor size/stride has been changed.

::

   q)tensor t:tensor 2 10#til 20
   0  1  2  3  4  5  6  7  8  9 
   10 11 12 13 14 15 16 17 18 19

   q)batch(t;5 1)
   1b

   q)tensor t
   0  1  2  3  4 
   10 11 12 13 14

   q)restore t     /batching dimension=1 omitted as 2nd argument
   4

   q)tensor t      /20-element storage with uninitialized values
   0  1  2  3  4 
   10 11 12 13 14
   0  0  0  0  0 
   0  0  0  0  0 

   q)size t
   4 5


Views
*****

PyTorch `views <https://pytorch.org/docs/stable/tensor_view.html>`_ share the same underlying data as the base tensor: the data is not copied to a separate block of memory (the underlying "storage"), but a new tensor points to the same memory but with different offset, size or stride.
When the size of a tensor is changed, it may be created via a "view" or a new storage allocation depending on whether the size needs to be increased or non-contiguous storage requires a copy to contiguous memory.

Three similar k api functions, :func:`reshape`, :func:`resize` and :func:`View` deal with reshaping/resizing a tensor.

- :func:`View` will create and return a new tensor which uses the same storage and has the same number of elements; it will signal an error if unable to match the number of elements, or, if the base tensor is not contiguous.
- :func:`reshape` will attempt to use the same data as the base tensor but will make a copy of non-contiguous storage if necessary; it will also signal an error if the number of elements specified by the new size does not match the number of elements in the new tensor.
- :func:`resize` can shrink or expand the number of elements: the operation may extend the existing storage or allocate a new block of memory.

To test if one tensor is a view of another, use :func:`sptr` to retrieve the storage pointer for each tensor or the :func:`alias` which will return true if both tensors share the same storage (see :ref:`sptr <sptr>` and :ref:`alias <alias>` in :doc:`tensor information <info>`).

::

   q)tensor a:tensor 3 3#til 9
   0 1 2
   3 4 5
   6 7 8

   q)tensor b:transpose a
   0 3 6
   1 4 7
   2 5 8

   q)sptr each(a;b)
   50422976 50422976

   q)alias(a;b)
   1b


reshape
^^^^^^^
The k api function behaves somewhat the same as the PyTorch `reshape <https://pytorch.org/docs/stable/generated/torch.reshape.html>`_ returning a new tensor pointer that may reference the same storage as the input tensor or a new storage, as required for the reshape.


.. function:: reshape(tensor;size) -> tensor

   :param pointer tensor: An :doc:`api-pointer <pointers>` to an allocated tensor.
   :param longs size: The shape of the new tensor, with any single dimension set to -1, to allow that size to be inferred from the remaining dimensions and the number of elements in the input tensor.
   :return: An :doc:`api-pointer <pointers>` to a new tensor with the given size. This tensor will share the underlying storage if possible, else will use a newly allocated underlying storage.

.. function:: reshape(tensor;example) -> tensor 

   :param pointer tensor: An :doc:`api-pointer <pointers>` to an allocated tensor.
   :param pointer example: A tensor can be supplied as the 2nd argument; the example tensor's size will be used for the reshape.
   :return: An :doc:`api-pointer <pointers>` to a new tensor with the given size. This tensor will share the underlying storage if possible, else will use a newly allocated underlying storage.

::

   q)a:tensor til 10
   q)sptr a    /underlying storage pointer
   63902912

   q)b:reshape(a;-1 4)
   'shape '[-1, 4]' is invalid for input of size 10
     [0]  b:reshape(a;-1 4)
            ^
   q)b:reshape(a;-1 5)
   q)size b
   2 5

   q)same(a;b)  /not the same tensor
   0b

   q)sptr b
   63902912

   q)sptr[a]~sptr b  /same storage pointer
   1b

   q)alias(a;b)   /i.e. same storage
   1b

In this example, the transpose of the original tensor is not contiguous, so reshape creates a new, contiguous block of memory for the returned tensor:

::

   q)a:tensor 2 4#til 8
   q)b:transpose a

   q)tensor b
   0 4
   1 5
   2 6
   3 7

   q)alias(a;b)    /a & b use the same underlying storage
   1b

   q)contiguous b  /but the transpose is not contiguous
   0b

   q)c:reshape(b;2 4)  /so a reshape based on b creates a new contiguous block

   q)alias(a;c)
   0b


The :func:`reshape` funciton will also accept and output k arrays, creating and freeing intermediate input/output tensors:

.. function:: reshape(input;size) -> k array
.. function:: reshape(input;example tensor) -> k array

::

   q)reshape(2 3 4#til 24; 3 8)
   0  1  2  3  4  5  6  7 
   8  9  10 11 12 13 14 15
   16 17 18 19 20 21 22 23


resize
^^^^^^

The :func:`resize` function accepts the same arguments as :func:`reshape` or :func:`View` but creates the newly shaped tensor inplace
(see PyTorch `resize <https://pytorch.org/docs/stable/generated/torch.Tensor.resize_.html>`_ for more allocation details).

.. function:: resize(tensor;size) -> null

   :param pointer tensor: An :doc:`api-pointer <pointers>` to an allocated tensor.
   :param longs size: The new shape for the tensor, with any single dimension set to -1, to allow that size to be inferred from the remaining dimensions and the number of elements in the input tensor.
   :return: Null. If the tensor's overall number of elements has been increased, the new elements will be unitialized.

.. function:: resize(tensor;example) -> null

   :param pointer tensor: An :doc:`api-pointer <pointers>` to an allocated tensor.
   :param tensor example: A tensor can be supplied as the 2nd argument; the example tensor's size will be used for the resize.
   :return: null. If the tensor's overall number of elements has been increased, the new elements will be unitialized.


::

   q)t:tensor 2 3#til 6

   q)sptr t
   53138624

   q)resize(t; 2 2)  /shrink size
   q)sptr t          /same storage pointer
   53138624

   q)resize(t;2 3)  /restore original size

   q)tensor t
   0 1 2
   3 4 5

   q)sptr t         /original storage still in use
   53138624

   q)resize(t;2 4)  /increase the size

   q)sptr t
   53115968

   q)tensor t  /new elements are uninitialized
   0 1 2 3
   4 5 0 0

The :func:`resize` will also accept and output k arrays, creating and freeing intermediate input/output tensors:

.. function:: resize(input;size) -> k array
.. function:: resize(input;example tensor) -> k array

   q)resize(1 2 3;4 4)  /new elements are uninitialized 
   1 2 3 53117040
   0 0 0 0       
   0 0 0 0       
   0 0 0 0       

   q)resize(1 2 3e;4 4)
   1           2 3           0
   5.01071e-37 0 5.01071e-37 0
   0           0 0           0
   0           0 0           0


View
^^^^

The :func:`View` function accepts the same arguments as :func:`reshape` and :func:`resize` but will never allocate new storage: the function will signal an error if the new view cannot be created (see PyTorch `view <https://pytorch.org/docs/stable/generated/torch.Tensor.view.html>`_).

.. function:: View(tensor;size) -> tensor

   :param pointer tensor: An :doc:`api-pointer <pointers>` to an allocated tensor.
   :param longs size: The shape of the new view of the tensor, with any single dimension set to -1, to allow that size to be inferred from the remaining dimensions and the number of elements in the input tensor.
   :return: An :doc:`api-pointer <pointers>` to a new tensor with the given size. This view will always share the underlying storage.

.. function:: View(tensor;example) -> tensor

   :param pointer tensor: An :doc:`api-pointer <pointers>` to an allocated tensor.
   :param example tensor: A tensor can be supplied as the 2nd argument; the example tensor's size will be used for the new view.
   :return: An :doc:`api-pointer <pointers>` to a new tensor with the given size. This view will always share the underlying storage.

::

   q)a:tensor 4 4#til 16
   q)b:View(a;2 8)

   q)alias(a;b)
   1b

   q)tensor b
   0 1 2  3  4  5  6  7 
   8 9 10 11 12 13 14 15

   q)tensor c:transpose b
   0 8 
   1 9 
   2 10
   3 11
   4 12
   5 13
   6 14
   7 15

   q)contiguous c
   0b

   q)d:View(c;4 4)
   'view size is not compatible with input tensor's size and stride (at least one dimension spans across two contiguous subspaces). Use .reshape(...) instead.
     [0]  d:View(c;4 4)
            ^

The :func:`View` will also accept and output k arrays, creating and freeing intermediate input/output tensors:

.. function:: View(input;size) -> k array
.. function:: View(input;example tensor) -> k array

::

   q)View(1 2 3 4;2 2)
   1 2
   3 4


narrow
^^^^^^

The :func:`narrow` function returns a new tensor that is a narrowed version of the input tensor; the narrowing occurs over the given dimension with an indicated offset and size (see PyTorch `narrow <https://pytorch.org/docs/stable/generated/torch.narrow.html?highlight=narrow#torch.narrow>`_).


.. function:: narrow(tensor;dim;offset;size) -> tensor pointer

   :param tensor pointer: An :doc:`api-pointer <pointers>` to an allocated tensor.
   :param long dim: The dimension over which the tensor will be narrowed.
   :param long offset: The offset in the given dimension where the narrowed tensor will begin.
   :param long size: The size of the narrowed tensor in the given dimension.
   :return: An :doc:`api-pointer <pointers>` to a new tensor with narrowed size, sharing the original tensor's underlying storage.

::

   q)tensor a:tensor 5 4#til 20
   0  1  2  3 
   4  5  6  7 
   8  9  10 11
   12 13 14 15
   16 17 18 19

   q)tensor b:narrow(a; 0; 1; 3)
   4  5  6  7 
   8  9  10 11
   12 13 14 15

   q)tensor c:narrow(a; 1; 2; 1)
   2 
   6 
   10
   14
   18

   q)alias(a;b)
   1b

   q)alias(b;c)
   1b

   q)ref a
   3

The :func:`narrow` function also accepts k arrays as input, returning the narrowed k array as a result, creating and freeing the intermediate tensors used in the narrowing operation.

.. function:: narrow(k array;dim;offset;size) -> k array

::

   q)narrow(til 10;0;3;3)
   3 4 5

   q)narrow(til 10;0;9;2)
   'start (9) + length (2) exceeds dimension size (10).
     [0]  narrow(til 10;0;9;2)
          ^

transpose
^^^^^^^^^

Returns a new tensor that is a view of the original tensor with two dimensions swapped
(see PyTorch `transpose <https://pytorch.org/docs/stable/generated/torch.transpose.html?highlight=transpose#torch.transpose>`_).

If no dimensions are given, the input tensor must be a 2-D array; the two dimensions will be swapped.

.. function:: transpose(tensor) -> tensor pointer

.. function:: transpose(tensor;inplace) -> tensor pointer or null

   :param tensor pointer: An :doc:`api-pointer <pointers>` to an allocated 2-D tensor.
   :param bool inplace: Flag set ``true`` if the transpose is to be done in place. Default is ``false``.
   :return: An :doc:`api-pointer <pointers>` to a new 2-D tensor transposing dimensions 0 and 1. If inplace is ``true``, the tensor is transposed and null is returned.

::

   q)tensor t:tensor 3 4#til 12
   0 1 2  3 
   4 5 6  7 
   8 9 10 11

   q)v:transpose t
   q)tensor v
   0 4 8 
   1 5 9 
   2 6 10
   3 7 11

   q)transpose(v;1b)
   q)tensor[v]~tensor t
   1b

.. function:: transpose(tensor;dim1;dim2) -> tensor pointer

.. function:: transpose(tensor;dim1;dim2;inplace) -> tensor pointer or null

   :param tensor pointer: An :doc:`api-pointer <pointers>` to an allocated n-dimensional tensor.
   :param long dim1: The first dimension to be transposed.
   :param long dim2: The 2nd dimension to be transposed.
   :param bool inplace: Flag set ``true`` if the transpose is to be done in place. Default is ``false``.
   :return: An :doc:`api-pointer <pointers>` to a new tensor transposing the two given dimensions. If inplace is ``true``, the input tensor is transposed and null is returned.

::

   q)t:tensor 2 3 4#til 24
   q)v:transpose(t;0;2)

   q)tensor(v;0)
   0 12
   4 16
   8 20

   q)size v
   4 3 2

   q)-2 str t;
   (1,.,.) = 
      0   1   2   3
      4   5   6   7
      8   9  10  11

   (2,.,.) = 
     12  13  14  15
     16  17  18  19
     20  21  22  23
   [ CPULongType{2,3,4} ]

   q)-2 str v;
   (1,.,.) = 
      0  12
      4  16
      8  20

   (2,.,.) = 
      1  13
      5  17
      9  21

   (3,.,.) = 
      2  14
      6  18
     10  22

   (4,.,.) = 
      3  15
      7  19
     11  23
   [ CPULongType{4,3,2} ]

   q)transpose(v;0;2;1b)

   q)tensor[t]~tensor v
   1b

The :func:`transpose` function also accepts a multidimensional k array for transposing:

.. function:: transpose(input) -> k array 

.. function:: transpose(input;dim1;dim2) -> k array

::

   q)transpose 3 3#til 9
   0 3 6
   1 4 7
   2 5 8

   q)0N!x:2 1 3#til 6;
   (,0 1 2;,3 4 5)

   q)0N!transpose(x;1;2);
   ((,0;,1;,2);(,3;,4;,5))

permute
^^^^^^^
Returns a view of the given tensor with its dimensions permuted (see PyTorch `permute <https://pytorch.org/docs/stable/generated/torch.permute.html>`_).

.. function:: permute(tensor;dims) -> tensor pointer

   :param tensor pointer: An :doc:`api-pointer <pointers>` to an allocated n-dimensional tensor.
   :param longs dims: The desired reordering of the dimensions.
   :return: An :doc:`api-pointer <pointers>` to a new tensor reordering the dimensions.
   
::

   q)t:tensor 2 3 4#til 24

   q)v:permute(t; 2 0 1)

   q)-2 str v;
   (1,.,.) = 
      0   4   8
     12  16  20
   
   (2,.,.) = 
      1   5   9
     13  17  21
   
   (3,.,.) = 
      2   6  10
     14  18  22
   
   (4,.,.) = 
      3   7  11
     15  19  23
   [ CPULongType{4,2,3} ]

   q)alias(t;v)
   1b


The :func:`permute` function also accepts a multidimensional k array:

.. function:: permute(input;dims) -> k array 

::

   q)0N!permute(2 3 4#til 24;2 0 1)
   ((0 4 8;12 16 20);(1 5 9;13 17 21);(2 6 10;14 18 22);(3 7 11;15 19 23))
   0  4  8  12 16 20
   1  5  9  13 17 21
   2  6  10 14 18 22
   3  7  11 15 19 23


detach   
^^^^^^
Returns new tensor(s), detached from the current gradient calculation graph (see PyTorch `detach <https://pytorch.org/docs/stable/generated/torch.Tensor.detach.html>`_). The detached tensor will never require a gradient. An optional flag can be set ``true`` to detach the tensor in place.

.. function:: detach(tensor) -> tensor pointer
.. function:: detach(tensor;inplace) -> tensor pointer or null if inplace flag true

::

   q)z:mean y:mul(x;x:tensor(1 2 3.0;`grad))
   q)`x`y`z!{`gradflag`gradfn@\:x}each(x;y;z)
   x| 1b `             
   y| 1b `MulBackward0 
   z| 1b `MeanBackward0


   q)d:detach z
   q)gradfn d
   `

   q)gradflag z
   1b

   q)alias(d;z)
   1b

:func:`detach` can also operate on a vector or dictionary of tensors, with optional indices/keys. The detach is done inplace.

.. function:: detach(vector) -> null
.. function:: detach(vector;ind) -> null

::

   q)z:mean y:mul(x;x:tensor(1 2 3.0;`grad))
   q)v:vector(x;y;z)

   q)flip`gradflag`gradfn@\:v
   1b `             
   1b `MeanBackward0
   1b `MeanBackward0

   q)detach(v;0 2)

   q)flip`gradflag`gradfn@\:v
   0b `            
   1b `MulBackward0
   0b `            

.. function:: detach(dictionary) -> null
.. function:: detach(dictionary;keys) -> null

::

   q)z:mean y:mul(x;x:tensor(1 2 3.0;`grad))
   q)d:dict`x`y`z!(x;y;z)

   q)`gradflag`gradfn@\:d
   x  y              z             
   --------------------------------
   1b 1b             1b            
   `  `MeanBackward0 `MeanBackward0

   q)detach(d;`y)

   q)`gradflag`gradfn@\:d
   x  y  z             
   --------------------
   1b 0b 1b            
   `  `  `MeanBackward0

   q)detach d

   q)`gradflag`gradfn@\:d
   x  y  z 
   --------
   0b 0b 0b
   `  `  ` 

expand   
^^^^^^

Returns a new view of the input tensor with singleton dimensions expanded to a larger size
(see PyTorch `expand <https://pytorch.org/docs/stable/generated/torch.Tensor.expand.html>`_).

.. function:: expand(tensor;sizes) -> new tensor

   :param pointer tensor: An :doc:`api-pointer <pointers>` to an allocated tensor.
   :param longs dims: Passing -1 as the size for a dimension to leave it unchanged, else a larger size for singleton dimension(s).
   :return: An :doc:`api-pointer <pointers>` to a new expanded tensor.

An example tensor can also be given as the second argument, in which case the size of the example tensor will be used for the expansion.

.. function:: expand(tensor;example) -> new tensor

   :param pointer tensor: An :doc:`api-pointer <pointers>` to an allocated tensor.
   :param pointer example: An :doc:`api-pointer <pointers>` to an allocated tensor whose size will be used for the expand argument.
   :return: An :doc:`api-pointer <pointers>` to a new expanded tensor.

::

   q)a:tensor 3 1#1 2 3

   q)b:expand(a;-1 4)

   q)tensor b
   1 1 1 1
   2 2 2 2
   3 3 3 3

The :func:`expand` function also accepts a k array as input:

.. function:: expand(input;dims) -> k array 

::

   q)expand(1 3#1 2 3;2 -1)
   1 2 3
   1 2 3

