.. _types:

Data types
==========

.. csv-table::
   :header: "PyTorch", "k", "Description"
   :widths: 10, 10, 40

   byte, byte, "byte or 8-bit integer (unsigned)"
   char, char, "char or 8-bit integer (signed)"
   bool, boolean, "boolean"
   short, short, "16-bit integer (signed)"
   int, int, "32-bit integer (signed)"
   long, long, "64-bit integer (signed)"
   half, real, "16-bit floating point, only available w'CUDA, converts to k real"
   float, real, "32-bit floating point, default data type"
   double, float, "64-bit floating point"
   cfloat, real, "complex number with 32-bit real and imaginary components"
   cdouble, double, "complex number with 64-bit real and imaginary components"

PyTorch defines 9 regular and 2 complex data types for tensors.
All but `half` have a direct mapping to a k data type: `half` values are mapped to k `real` values.

::

   q)d:`byte`char`bool`short`int`long`half`float`double

   q){d:dtype t:tensor(x;y); v:tensor t; free t; (d;v)}[0 1]'[d]
   `byte   0x0001    
   `char   "\000\001"
   `bool   01b       
   `short  0 1h      
   `int    0 1i      
   `long   0 1       
   `half   0 1e      
   `float  0 1e      
   `double 0 1f      

.. _dtype:

dtype function
**************

The q interface function: ``dtype`` can be used to query or set the default data type, as well as query the data type of a given tensor, vector or dictionary of tensors.

.. function:: dtype() -> sym

   | Given an empty or null arg, returns the default data type as a symbol.

.. function:: dtype(sym) -> null

   | Given a symbol, sets default data type for the k session, defined initially as ```float``. This is usually a floating point data type. Returns null.

.. function:: dtype(ptr) -> sym

   | Given an :doc:`api-pointer <pointers>` to a tensor, vector or dictionary of tensors, returns sym(s) representing the data type(s) of the allocated tensor, vector or dictionary of tensors.

::

   q)dtype[]          / query default data type
   `float

   q)dtype`double     / set default data type

   q)dtype()          / verify change
   `double

   q)dtype t:tensor 01b
   `bool

   q)dtype v:vector(01b;1 2.3;4 5h)
   `bool`double`short

   q)dtype d:dict `a`b`c!(0;"abc";9e)
   a| long
   b| char
   c| float


help
****

.. function:: help(topic) -> k dictionary

   :param symbol topic: for help with data types, either ```ktype`` or ```dtype``.
   :return: k dictionary mapping between PyTorch and k data types.


The ```ktype`` map shows the implicit conversion from a k array to a PyTorch data type (the conversion can be made explicitly by specifing a data type in addition to the k array):

::

   q)help`ktype
   e| float
   f| double
   e| half
   b| bool
   x| byte
   c| char
   j| long
   i| int

   q)dtype f:tensor 1 2 3.0
   `double

   q)dtype h:tensor(1 2 3.0; `short)
   `short


The ```dtype`` map shows the k type that results when PyTorch tensors are converted back into k arrays:

::

   q)help`dtype
   float  | e
   double | f
   half   | e
   bool   | b
   byte   | x
   char   | c
   long   | j
   int    | i
   short  | h
   chalf  | e
   cfloat | e
   cdouble| f

(```chalf``, ```cfloat`` and ```cdouble`` are :doc:`complex <complex>` data types)
