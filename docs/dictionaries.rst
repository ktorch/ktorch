.. _dictionaries:

Dictionaries
============

A dictionary of tensors can be created from k values or previously allocated tensors or a combination of both.
Dicionaries are created, retrieved and modified using the same :func:`dict` interface function.

::

   q)t:tensor 1 2 3e
   q)d:dict `a`t!(4 5 6 7.0; t)

   q)dict d
   a| 4 5 6 7f
   t| 1 2 3e


Creating a dictionary
*********************

.. function:: dict(k dictionary) -> dictionary pointer

   | Given a k dictionary of symbol keys and corresponding values or previously created tensor pointers, returns a pointer to a dictionary of tensors.

An alternate form accepts symbol keys and values as separate arguments:

.. function:: dict(keys;values) -> dictionary pointer
   :noindex:

   :param sym keys: symbol keys to name the input values/tensors.
   :param array/tensor values: k arrays/tensor pointers laigned with the keys.
   :return: an :doc:`api-pointer <pointers>` to a dictionary of tensors.

.. note::

   When tensor pointers are included in the arguments to :func:`dict`, the tensor's memory is managed by the dictionary and the previous handle to the tensor is no longer valid without a reference increment (see :ref:`addref <addref>`).

::

   q)a:tensor 1 2
   q)b:tensor 3 4 5.0
   q)d:dict `a`b!(a;addref b)

   q)dict d
   a| 1 2
   b| 3 4 5f

   q)tensor b
   3 4 5f

   q)ref b
   2       /2 references to the same tensor

   q)free d
   q)ref b
   1 

   q)d:dict(`x`y;(1 2;3 4 5))   / use (keys;values) form

   q)dict d
   x| 1 2
   y| 3 4 5


Retrieving dictionary values
****************************

.. function:: dict(ptr) -> val
.. function:: dict(ptr;key) -> val
   :noindex:

   :param dictionary ptr: an :doc:`api-pointer <pointers>` to a previously created dictionary of tensors.
   :param sym key: optional symbol(s) to retrieve,
   :return: k dictionary if no key specified, else array(s) for each key given.

::

   q)d:dict(`x`y; (1 2;3 4 5))

   q)dict d
   x| 1 2
   y| 3 4 5

   q)dict(d;`y)
   3 4 5

   q)dict(d;`y`y`x)
   3 4 5
   3 4 5
   1 2


Setting dictionary values
*************************

.. function:: dict(ptr;key;val) -> null

   :param dictionary ptr: an :doc:`api-pointer <pointers>` to a previously created dictionary of tensors.
   :param sym key: a symbol or set of symbol keys to the dictionary of tensors.
   :param ptr/array val: a corresponding value/tensor or set of values/tensors to assign to the dictionary replacing existing values. If the keys do not exist , the keys and values are appended to the dictionary.
   :return: null

::

   q)d:dict(`x`y; (1 2;3 4 5))

   q)dict(d;`x;01010b)
   q)dict(d;`z`zz; ("z";"zz"))

   q)dict d
   x | 01010b
   y | 3 4 5
   z | "z"
   zz| "zz"

   q)t:tensor til 9
   q)dict(d;`x;t)

   q)dict(d;`x)
   0 1 2 3 4 5 6 7 8

   q)ref t          /t no longer manages the tensor memory
   'stale pointer
     [0]  ref t
          ^


Retrieving tensor pointers
**************************

Use the :func:`tensor` function to extract tensor pointers from a given dictionary and optional key(s).

.. function:: tensor(dict) -> tensor pointers
.. function:: tensor(dict;key) -> tensor pointer(s)
   :noindex:

   :param dict-pointer vec: an :doc:`api-pointer <pointers>` to a previously created dictionary of tensors.
   :param sym key: a symbol or set of symbol keys to the dictionary of tensors.
   :return: return tensor pointer(s) for each tensor in the dictionary, or each tensor corresponding to supplied key(s)

::


   q)d:dict(`x`y; (1 2;3 4 5))

   q)t:tensor d
   q)tensor each t
   x| 1 2
   y| 3 4 5

   q)size each t
   x| 2
   y| 3

   q)free each t
   x| ::
   y| ::

   q)t:tensor(d;`y)
   q)tensor t
   3 4 5

