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
   half, real, "16-bit floating point, only available w'CUDA, converts to k ``real``"
   float, real, "32-bit floating point, default data type"
   double, float, "64-bit floating point"

PyTorch defines 9 data types for tensors.
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
