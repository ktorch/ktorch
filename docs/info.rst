Tensor information
==================

Tensor size
***********

.. _tensor-size:

size
^^^^

.. function:: size(ptr) -> long list / lists / dictionary

   | Given an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors, returns the sizes at each dimension of the tensor(s).

::

   q)a:tensor 3 4#5
   q)size a
   3 4

   q)d:dict `a`b!(a;3 4 5#0.0)
   q)size d
   a| 3 4
   b| 3 4 5

stride
^^^^^^

Stride is the jump necessary to go from one element to the next one across each dimension.

.. function:: stride(ptr) -> long list / lists / dictionary

   | Given an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors, returns the strides at each dimension of the tensor(s).

::

   q)a:tensor 3 4#5
   q)size a
   3 4
   q)stride a
   4 1

   q)b:transpose a
   q)size b
   4 3
   q)stride b
   1 4

   q)d:dict `a`b`c!(a;b;3 4 5#0.0)
   q)stride d
   a| 4 1
   b| 1 4
   c| 20 5 1

dim
^^^

.. function:: dim(ptr) -> long scalar / list / dictionary

   | Given an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors, returns the dimension(s) of the tensor(s).  See also :ref:`sparsedim` and :ref:`densedim` for dimensions of sparse tensors.

::

   q)t:tensor()
   q)dim t
   1

   q)use[t]tensor 1b
   q)dim t
   0

   q)use[t]tensor 3 4 5#til 60
   q)dim t
   3


itemsize
^^^^^^^^

.. function:: itemsize(ptr) -> long scalar / list / dictionary

   | Given an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors, returns the element size(s) of the tensor(s).

::

   q)d:dict `a`b`c!(2 3 4.0; 1e; "string")
   q)itemsize d
   a| 8
   b| 4
   c| 1

numel
^^^^^

.. function:: numel(ptr) -> long scalar / list / dictionary

   | Given an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors, returns the number of elements in the tensor(s). This count may be less than the full number of elements of the tensor's underlying storage for certain tensor views.

::

   q)a:tensor til 10
   q)b:narrow(a;0;5;5)   / new tensor, narrowed to last 5 elements

   q)alias(a;b)
   1b                    / same underlying storage

   q)d:dict `a`b!(a;b)

   q)numel d             / elements in the tensor
   a| 10
   b| 5

   q)elements d          / full size in storage
   a| 10
   b| 10

.. _tensor-elements:

elements
^^^^^^^^

.. function:: elements(ptr) -> long scalar / list / dictionary

   | Given an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors, returns the full number of elements in the tensor's underlying storage.

::

   q)a:tensor til 10
   q)b:narrow(a;0;5;5)   / new tensor, narrowed to last 5 elements

   q)alias(a;b)
   1b                    / same underlying storage

   q)d:dict `a`b!(a;b)

   q)numel d             / elements in the tensor
   a| 10
   b| 5

   q)elements d          / full size in storage
   a| 10
   b| 10

.. _tensor-bytes:

bytes
^^^^^

.. function:: bytes(ptr) -> long scalar / list / dictionary

   | Given an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors, returns the bytes allocated in the tensor's underlying storage.

::

   q)a:tensor til 10
   q)b:narrow(a;0;5;5)   / new tensor, narrowed to last 5 elements

   q)alias(a;b)
   1b                    / same underlying storage

   q)d:dict `a`b!(a;b)

   q)numel d             / elements in the tensor
   a| 10
   b| 5

   q)numel[d]*itemsize d  / elements * bytes per element
   a| 80
   b| 40

   q)bytes d              / size of storage
   a| 80
   b| 80


.. _tensor-count:

tensorcount
^^^^^^^^^^^

.. function:: tensorcount(ptr) -> long

   | Given an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors, returns the number of tensors contained.

::

   q)t:tensor 1 2 3.0
   q)tensorcount t
   1

   q)v:vector(t; 001b; 4 5 6)
   q)tensorcount v
   3


Pointer information
*******************

.. _tensor-ptr:

ptr
^^^

.. function:: ptr(api-pointer) -> long (s)

   | Given an :doc:`api-pointer<pointers>` to a tensor, vector or dictionary of tensors, returns long representation(s) of the raw ptr(s) to tensor(s), managed by PyTorch via a smart pointer which maintains reference count and manages their memory.

::

   q)a:tensor 1 2 3
   q)b:addref a

   q)class each (a;b)
   `tensor`tensor

   q)(a;b)           / different api handles -> same tensor
   40292432
   40292096

   q)ptr each(a;b)   / underlying tensor matches
   39965072 39965072

   q)same(a;b)
   1b

.. _sptr:

sptr
^^^^

.. function:: sptr(api-pointer) -> long (s)

   | Given an :doc:`api-pointer<pointers>` to a tensor, vector or dictionary of tensors, returns long representation(s) of the raw ptr(s) to tensor's underlying storage. Different tensors can share the same underlying memory.

::

   q)a:tensor til 12
   q)b:reshape(a;3 4)  / new tensor, same underlying storage

   q)ptr each(a;b)
   40292928 39965072   / different raw tensor pointers

   q)sptr each(a;b)
   40296576 40296576   / same raw storage pointers

   q)same(a;b)    / not the same tensor
   0b

   q)alias(a;b)   / shares the same underlying storage
   1b

ref
^^^

.. function:: ref(api-pointer) -> long scalar / list / dictionary

   | Given an :doc:`api-pointer<pointers>` to a tensor, vector or dictionary of tensors, returns the count of references to the tensor(s).

::

   q)a:tensor 1 2 3
   q)b:addref a

   q)ref a
   2

   q)same(a;b)
   1b

   q)free b
   q)ref a
   1

sref
^^^^

.. function:: sref(api-pointer) -> long scalar / list / dictionary

   | Given an :doc:`api-pointer<pointers>` to a tensor, vector or dictionary of tensors, returns the count of references to the underlying storage of the tensor(s).

::

   q)a:tensor til 12
   q)ref a  
   1         /single reference to tensor
   q)sref a
   1         /and its underlying storage

   q)b:reshape(a;3 4)  / reshape into new matrix
   q)ref b
   1
   q)ref a
   2

   q)sref a  / two references to same underlying storage
   2

   q)sptr[a]~sptr b
   1b


weakref
^^^^^^^

.. function:: weakref(api-pointer) -> count of non-owning refrences to same tensor

   | Given an :doc:`api-pointer<pointers>` to a tensor, vector or dictionary of tensors, returns the count of weak references (non-owning references) to the tensors. Less likely to be used in the k interface which maintains owning references in the k session.


same
^^^^

.. function:: same(tensor1;tensor2) -> bool

   | Given two :doc:`api-pointers<pointers>` to tensors, returns true if both api handles point to the same tensor.

::

   q)a:tensor 1 2 3
   q)b:addref a

   q)ptr'[(a;b)]
   40292928 40292928

   q)same(a;b)
   1b

.. _alias:

alias
^^^^^

.. function:: alias(tensor1;tensor2) -> bool

   | Given two :doc:`api-pointers<pointers>` to tensors, returns true if both api handles point to the same underlying storage.

::

   q)a:tensor(1 2 3.0; `grad)
   q)b:detach a                / copy of a without gradient tracking

   q)gradient each(a;b)
   `grad`nograd

   q)same(a;b)
   0b              /different tensors

   q)alias(a;b)
   1b              /same underlying storage

offset
^^^^^^

.. function:: offset(api-pointer) -> long scalar / list / dictionary

   | Given an :doc:`api-pointer<pointers>` to a tensor, vector or dictionary of tensors, returns the offset into the underlying storage of the tensor(s).

::

   q)a:tensor(3 4#til 12)
   q)b:narrow(a;0;1;2)    /narrow tensor to two final rows

   q)alias(a;b)
   1b

   q)offset each(a;b)
   0 4  / narrowed tensor starts 4 elements in

Tensor options
**************

Tensor options are defined by symbols in the k interface.
The functions below take a tensor, vector or dictionary pointer and return symbol(s).
Some of the functions also accept a null or empty arg, and return a system default data type, CUDA device, etc.

options
^^^^^^^

.. function:: options() -> dictionary of symbols

.. function:: options(ptr) -> dictionary of symbols

    | For empty or null arg, returns a dictionary of default attributes for tensor creation. Given an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors, returns a dictionary or list of dictionaries of the attribute values for the tensor(s).

::

   q)options()             /show default options
   device  | cpu
   dtype   | float
   layout  | strided
   gradient| nograd
   pin     | unpinned
   memory  | contiguous

   q)options t:tensor()    /verify empty tensor arg uses defaults
   device  | cpu
   dtype   | float
   layout  | strided
   gradient| nograd
   pin     | unpinned
   memory  | contiguous

   q)options d:dict `a`b`c!("char"; tensor(010b;`cuda`sparse); tensor(2 3 4h;`cpu`pinned))
    | device dtype layout  gradient pin      memory    
   -| -------------------------------------------------
   a| cpu    char  strided nograd   unpinned contiguous
   b| cuda:0 bool  sparse  nograd   unpinned contiguous
   c| cpu    short strided nograd   pinned   contiguous

.. _tensor-device:

device
^^^^^^

.. function:: device() -> symbol scalar

.. function:: device(ptr) -> symbol scalar / list / dictionary

   | For a null or empty arg, returns default CUDA device if any GPU's available, else ```cpu``.  Given an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors, returns sym(s) for the devices(s). See also the :doc:`section on devices types <devices>` for more on querying for a CUDA device.

::

   q)device() /on machine with CUDA device(s)
   `cuda

   q)device t:tensor()  /use cpu if no device specified
   `cpu

   q)device d:dict`a`b`c!( tensor(1 2 3;`cuda); tensor(4 5;`cuda:1); 6 7 8.0)
   a| cuda:0
   b| cuda:1
   c| cpu


.. _tensor-dtype:


dtype
^^^^^

.. function:: dtype() -> symbol

.. function:: dtype(ptr) -> symbol scalar / list / dictionary

   | For a null or empty arg, returns the default data type, e.g. ```float``. Given an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors, returns sym(s) for the data type(s). See also the :ref:`section on data types <dtype>` for more on setting default data type.

::

   q)dtype[]
   `float

   q)dtype e:tensor()
   `float

   q)dtype f:tensor 1 2 3.0
   `double

   q)dtype v:vector(e;f)
   `float`double


layout
^^^^^^

.. function:: layout(ptr) -> symbol scalar / list / dictionary

::

   q)layout each (s:sparse t; t:tensor 0 3 0 0 9.0)
   `sparse`strided


gradient
^^^^^^^^

.. function:: gradient(ptr) -> symbol scalar / list / dictionary

::

   q)gradient each (d:detach s; s:sparse t; t:tensor(0 3 0 0.0;`grad))
   `nograd`grad`grad

   q)gradflag each(d;s;t)
   011b


memory
^^^^^^

.. function:: memory(ptr) -> symbol scalar / list / dictionary

gradfn
^^^^^^

The :func:`gradfn` is not an option that is set directly, but it is the result of a chain of calculations performed on a set of tensors where any input requires gradients. The result is a symbol of the function used for back propagation, with a version number, the count of any in-place operations.

.. function:: gradfn(ptr) -> symbol scalar / list / dictionary

::

   q)z:mean y:mul(x;x:tensor(1 2 3.0; `grad))

   q)`x`y`z!gradfn each (x;y;z)
   x| 
   y| MulBackward0
   z| MeanBackward0


Tensor flags
************

contiguous
^^^^^^^^^^
Up until version PyTorch version 1.5, contiguous meant the tensor is contiguous in memory in C order. Then, with the introduction of the new `memory format <https://pytorch.org/docs/stable/tensor_attributes.html?highlight=memory%20format#torch-memory-format>`_ attribute, the definition of contiguous became more complicated and is now defined as *contiguous in memory in the order specified by memory format*.
More notes on the new memory format `here <https://pytorch.org/tutorials/intermediate/memory_format_tutorial.html>`_.

.. function:: contiguous(ptr) -> boolean scalar / list / dictionary

   | Returns true if the tensor is contiguous in memory in C order.

.. function:: contiguous(ptr;memory-format) -> boolean scalar / list / dictionary

   :param api-pointer ptr: an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors.
   :param sym memory-format: optional symbol indicating memory format, e.g. ```contiguous`` or ```channel2d``.
   :return: true if tensor(s) contiguous in memory in the order specified by the supplied memory format.

::

   q)a:tensor 2 3#til 6
   q)contiguous a
   1b

   q)b:transpose a
   q)tensor b
   0 3
   1 4
   2 5

   q)contiguous b
   0b

   q)t:tensor(2 3 4 5#til 120;`channel2d)

   q)contiguous t
   0b

   q)contiguous(t;`channel2d)
   1b


coalesced
^^^^^^^^^

.. function:: coalesced(ptr) -> boolean scalar / list / dictionary

   | Returns true if sparse tensor is known to have no duplicate entries. Dense tensors have coalesced set true by definition. See :ref:`sparse tensors <coalesce>` for more detail.

::

   q)s:tensor(`sparse; 1 3#4 0 0; 9 10 -20; 10)
   q)tensor s
   -10 0 0 0 9 0 0 0 0 0

   q)indices s
   4 0 0

   q)values s
   9 10 -20

   q)coalesced s
   0b

   q)coalesce s
   q)indices s   /indices are now sorted and unique
   0 4

   q)values s    /values summed for same index
   -10 9

   q)coalesced s
   1b

defined
^^^^^^^

.. function:: defined(ptr) -> boolean scalar / list / dictionary

   | Returns true if tensor has a defined data type and device. See :ref:`undefined tensors <tensor-undefined>` for more detail.

gradflag
^^^^^^^^

.. function:: gradflag(ptr) -> boolean scalar / list / dictionary

   | Returns true/false if the tensor's requires gradient property was turned on/off via symbol: ```grad``/```nograd``.

::

   q)d:dict `a`b!(tensor(1 2 3.0;`grad); 4 5 6.0)

   q)gradflag d
   a| 1
   b| 0

leaf
^^^^

.. function:: leaf(ptr) -> boolean scalar / list / dictionary

   | All tensors that don't require gradients are leaf tensors by convention.  For tensors requiring gradients, they will be leaf tensors if they were created by the user instead of as the result of an operation.  Only leaf tensors will have their gradients populated during a call to :func:`backward`.

::

   q)z:mean y:mul(x;x:tensor(1 2 3.0; `grad))
   q)`x`y`z!leaf each(x;y;z)
   x| 1
   y| 0
   z| 0

   q)`x`y`z!gradfn each(x;y;z)
   x| 
   y| MulBackward0
   z| MeanBackward0

   q)backward z
   q)grad x
   0.6666667 1.333333 2


pinned
^^^^^^

.. function:: pinned(ptr) -> boolean scalar / list /dictionary

   | Given an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors, returns boolean(s) set true for tensor(s) with `page-locked memory <https://pytorch.org/docs/stable/notes/cuda.html?highlight=pinned%20memory>`_. Allows for quicker cpu-to-gpu transfers.

::

   q)t:tensor(1 2 3e;`pinned)
   q)pinned t
   1b

   q)to(t`cuda;1b) /set async flag true when copying to gpu

   q)device t
   `cuda:0

   q)pinned t /only cpu memory can be pinned
   0b


sparseflag
^^^^^^^^^^

.. function:: sparseflag(ptr) -> boolean scalar / list / dictionary

   | Given an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors, returns boolean(s) set true for sparse tensor(s). See also the :ref:`section on sparse tensors <sparse>` for more detail.
   

::

   q)d:dict `a`b`c!(1 2 3; 4 5; sparse(0 0 0 0 2 0))

   q)sparseflag d
   a| 0
   b| 0
   c| 1

Utilities
*********

info
^^^^

.. function:: info(ptr) -> dictionary

   | Given a tensor pointer, returns a dictionary of attributes of the tensor.

::

   q)t:tensor(`randn;2 3;`cfloat)
   q)info t
   device    | `cpu
   dtype     | `cfloat
   layout    | `strided
   gradient  | `nograd
   pin       | `unpinned
   memory    | `contiguous
   leaf      | 1b
   gradfn    | `
   dim       | 2
   sparsedim | 0
   size      | 2 3
   stride    | 3 1
   numel     | 6
   itemsize  | 8
   contiguous| 1b
   coalesced | 1b
   offset    | 0
   ptr       | 53851424
   ref       | 1


detail
^^^^^^

.. function:: detail(ptr) -> dictionary

   | Given a tensor pointer, returns a dictionary of attributes of the tensor as well as a separate dictionary describing the underlying storage, a contiguous one-dimensional array of bytes containing the tensor data. If the tensor is sparse, :func:`detail` returns a list with the detail for both the indices and values.

::

   q)s:tensor(0 0 1 0 2 0 0;`sparse)

   q)detail s
          | device dtype layout  gradient pin      memory     leaf gradfn dim sp..
   -------| --------------------------------------------------------------------..
   indices| cpu    long  strided nograd   unpinned contiguous 1           2   0 ..
   values | cpu    long  strided nograd   unpinned contiguous 1           1   0 ..

   q)first detail s
   device    | `cpu
   dtype     | `long
   layout    | `strided
   gradient  | `nograd
   pin       | `unpinned
   memory    | `contiguous
   leaf      | 1b
   gradfn    | `
   dim       | 2
   sparsedim | 0
   size      | 1 2
   stride    | 1 1
   numel     | 2
   itemsize  | 8
   contiguous| 1b
   coalesced | 1b
   offset    | 0
   ptr       | 53851424
   ref       | 1
   storage   | `size`itemsize`ref`ptr`data!(2;8;2;53855680;2 4)

str
^^^

.. function:: str(ptr) -> string

   | Returns the PyTorch C++ string representation of the object with embedded newlines.

::

   q)t:tensor(`randn;2 3)

   q)-2 str t;
    0.1526  0.5672  0.0854
    1.4224  1.6250 -1.2955
   [ CPUFloatType{2,3} ]

   q)to(t;`cuda`double)

   q)-2 str t;
    0.1526  0.5672  0.0854
    1.4224  1.6250 -1.2955
   [ CUDADoubleType{2,3} ]


