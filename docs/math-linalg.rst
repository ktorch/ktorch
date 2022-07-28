Linear Algebra
==============

PyTorch routines in the older section for 
`BLAS & LAPACK <https://pytorch.org/docs/stable/torch.html#blas-and-lapack-operations>`_
and the newer `linalg <https://pytorch.org/docs/stable/linalg.html>`_ namespace are documented here.

Matrix properties
*****************

det
^^^

`torch.det <https://pytorch.org/docs/stable/generated/torch.det.html>`_ is implemented with :func:`det`.

.. function:: det(matrix) -> determinant

   :param array,tensor matrix: square input matrix or tensor :doc:`pointer <pointers>` to a matrix or batch of matrices
   :return: If k array given as input returns determinant as k scalar else tensor scalar.

::

   q)seed 123
   q)x:tensor(`randn;3 3)
   q)y:det x
   q)tensor y
   0.3735779e

   q)x:return x
   q)det(x;x)
   0.3735779 0.3735779e

logdet
^^^^^^

`torch.logdet <https://pytorch.org/docs/stable/generated/torch.logdet.html>`_ is implemented with :func:`logdet`.

.. function:: logdet(matrix) -> log determinant

   :param array,tensor matrix: square input matrix or tensor :doc:`pointer <pointers>` to a matrix or batch of matrices
   :return: If k array given as input returns log determinant as k scalar else tensor scalar.

::

   q)seed 123
   q)x:tensor(`randn;3 3)
   q)y:logdet x
   q)tensor y
   -0.9846289e

   q)x:return x
   q)logdet(x;x)
   -0.9846289 -0.9846289e

slogdet
^^^^^^^

`torch.slogdet <https://pytorch.org/docs/stable/generated/torch.slogdet.html>`_ is implemented with :func:`slogdet`,
which computes the sign and natural logarithm of the absolute value of the determinant of a square matrix or a set of matrices.

.. function:: slogdet(matrix) -> log determinant

   :param array,tensor matrix: square input matrix or tensor :doc:`pointer <pointers>` to a matrix or batch of matrices
   :return: If k array given as input returns sign and log determinant as k list(s) else tensor vector.

::

   q)seed 101
   q)x:tensor(`randn;3 3)
   q)y:logdet x
   q)tensor y
   0Ne

   q)v:slogdet x
   q)vector v
   -1 0.5532773e

   q)x:return x
   q)slogdet(x;x;x;x)
   -1        -1        -1        -1       
   0.5532773 0.5532773 0.5532773 0.5532773

mrank
^^^^^

`torch.linalg.matrix_rank <https://pytorch.org/docs/stable/generated/torch.linalg.matrix_rank.html>`_ is implemented as :func:`mrank`.

.. function:: mrank(matrix;atol;rtol;hermitian) -> rank
.. function:: mrank(matrix;atol;rtol;hermitian;output) -> null

   | Allowable argument combinations:

    - ``mrank(matrix)``
    - ``mrank(matrix;atol)``
    - ``mrank(matrix;atol;rtol)``
    - ``mrank(matrix;atol;rtol;hermitian)``
    - ``mrank(matrix;hermitian)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor matrix: input matrix or tensor :doc:`pointer <pointers>` to a matrix or batch of matrices
   :param double atol: absolute tolerance, default=0 if left unspecified.
   :param double rtol: relative tolerance, default is derived from input, see `torch.linalg.matrix_rank <https://pytorch.org/docs/stable/generated/torch.linalg.matrix_rank.html>`_
   :param bool hermitian: optional flag, set ``false`` by default, set ``true`` to indicate Hermitian if complex input else symmetric for real input.
   :param tensor output: an optional `complex <complex>` tensor to use for function output
   :return: Returns the numerical rank of the input matrix or matrices, as an array if input is an array, else as tensor. If output tensor supplied, writes output to tensor and returns null.

::

   q)x:return tensor(`eye;10)
   q)mrank x
   10

   q)mrank .[x;0 0;:;0e]
   9

   q)x:tensor(`randn; 2 4 3 3; `cdouble)
   q)y:mrank x
   q)tensor y
   3 3 3 3
   3 3 3 3

   q)use[y]mrank(x;1b) /hermitian=true
   q)tensor y
   3 3 3 3
   3 3 3 3

   q)use[y]mrank(x;1.0;0.0;1b) /atol=1, rtol=0, Hermitian=true
   q)tensor y
   2 2 1 2
   2 2 3 2

   q)use[y]mrank(x;1.0;0.0;0b) /atol=1, rtol=0, Hermitian=false
   q)tensor y
   1 1 2 2
   2 2 2 2


Decompositions
**************

chol
^^^^
`torch.linalg.cholesky <https://pytorch.org/docs/stable/generated/torch.linalg.cholesky.html>`_ is implemented as function :func:`chol`, which returns the Cholesky factorization  :math:`L` for each matrix input.

.. math::

    A = LL^{\text{H}}\mathrlap{\qquad L \in \mathbb{K}^{n \times n}}

where :math:`L` is a lower triangular matrix and
:math:`L^{\text{H}}` is the conjugate transpose when :math:`L` is complex, and the transpose when :math:`L` is real-valued.

.. function:: chol(x;upper) -> Cholesky decomposition
.. function:: chol(x;upper;output) -> null

   :param array,tensor x: a k array or tensor :doc:`pointer <pointers>` of shape :math:`(*, n, n)` where * is zero or more batch dimensions consisting of symmetric or Hermitian positive-definite matrices
   :param bool upper: default=``false`` to return lower triangular output, set ``true`` for upper triangular output
   :param tensor output: an optional :doc:`tensor <pointers>` to use for function output
   :return: Returns lower/upper triangular Cholesky factors for each of the input matrices, as a tensor if tensor input, else k array.  If an output tensor is given, the factors are written to this tensor and null is returned.

::

   q)seed 123
   q)a:tensor(`randn;3 3;`double)
   q)t:transpose a
   q)use[a]mm(a;t)

   q)L:chol a
   q){x mmu flip x}tensor L
   0.1635  -0.1946 -0.1022
   -0.1946 1.534   1.205  
   -0.1022 1.205   1.62   

   q)tensor a
   0.1635  -0.1946 -0.1022
   -0.1946 1.534   1.205  
   -0.1022 1.205   1.62   


cholx
^^^^^
`torch.linalg.cholesky_ex <https://pytorch.org/docs/stable/generated/torch.linalg.cholesky_ex.html>`_ is implemented as function :func:`cholx`, which computes the same Cholesky decomposition as :func:`chol`, but with additional error codes and options.

.. function:: cholx(x;upper;check) -> Cholesky decomposition and error codes
.. function:: cholx(x;upper;check;output) -> null

   :param array,tensor x: a k array or tensor :doc:`pointer <pointers>` of shape :math:`(*, n, n)` where * is zero or more batch dimensions consisting of symmetric or Hermitian positive-definite matrices
   :param bool upper: default = ``false`` to return lower triangular output, set ``true`` for upper triangular output
   :param bool check: default = ``false`` for no checking, to check error codes before returning, set ``true``
   :param vector output: an optional :doc:`vector <vectors>` to use for function output of decomposition and errors
   :return: Returns lower/upper triangular Cholesky factors for each of the input matrices along with error code(s). If tensor input, returns vector of tensors, else k array.  If an output vector is given, the factors and errors are written to this vector and null is returned.

::

   q)a:tensor(`randn;3 3;`double)
   q)t:transpose a
   q)use[a]mm(a;t)

   q)v:cholx a
   q){x mmu flip x}vector(v;0)
   0.1635  -0.1946 -0.1022
   -0.1946 1.534   1.205  
   -0.1022 1.205   1.62   

   q)tensor a
   0.1635  -0.1946 -0.1022
   -0.1946 1.534   1.205  
   -0.1022 1.205   1.62   

   q)vector(v;1)
   0i

Supply incorrect input, i.e. not symmetric or positive-definite matrix:

::

   q)cholx(t;1b;0b;v)  / upper triangular, no checks
   q)vector(v;1)
   1i

   q)cholx(t;1b;1b;v)  / turn on error checking
   'torch.linalg.cholesky_ex: The factorization could not be completed because the input is not positive-definite (the leading minor of order 1 is not positive-definite).
     [0]  cholx(t;1b;1b;v)  / turn on error checking
          ^

.. _eig:

eig
^^^
`pytorch.linalg.eig <https://pytorch.org/docs/stable/generated/torch.linalg.eig.html>`_ is implemented by function :func:`eig`, which calculates eigenvalues and eigenvectors of a square matrix or set of square matrices.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **eigenvalue decomposition** of a square matrix
:math:`A \in \mathbb{K}^{n \times n}` (if it exists) is defined as

.. math::

    A = V \operatorname{diag}(\Lambda) V^{-1}\mathrlap{\qquad V \in \mathbb{C}^{n \times n}, \Lambda \in \mathbb{C}^n}


.. function:: eig(x) -> eigenvalues and eigenvectors
.. function:: eig(x;output) -> null

   :param array,tensor x: k array or tensor :doc:`pointer <pointers>` to a square matrix or batches of square matrices
   :param vector output: an optional :doc:`vector <vectors>` to use for function output of eigenvalues and eigenvectors
   :return: Returns a vector of tensors if ``x`` is a tensor, else a 2-element k list with eigenvalues and eigenvectors, corresponding to :math:`\Lambda` and :math:`V` above. The eigenvalues and vectors will be complex even when input ``x`` is real. If an output vector is supplied, function output is written to the vector and null returned.

.. note:: By default, complex tensors are converted to k arrays with the real and imaginary parts along the 1st dimension, see :ref:`settings <complex-first>` for more detail.

::

   q)show x:3 3#2 -3 0.0, 2 -5 0.0, 0 0 3.0
   2 -3 0
   2 -5 0
   0 0  3

   q)v:first each eig x  / take real part only

   q)mmu/[(v 1;diag v 0;inverse v 1)]
   2 -3 0
   2 -5 0
   0 0  3

eigvals
^^^^^^^
`pytorch.linalg.eigvals <https://pytorch.org/docs/stable/generated/torch.linalg.eigvals.html>`_ is implemented by function :func:`eigvals`, which calculates eigenvalues only (see :ref:`eig <eig>` for more detail on the full eigenvalue decomposition).

.. function:: eigvals(x) -> eigenvalues
.. function:: eigvals(x;output) -> null

   :param array,tensor x: k array or tensor :doc:`pointer <pointers>` to a square matrix or batches of square matrices
   :param vector output: an optional tensor :doc:`pointer <pointers>` to use for function output of eigenvalues
   :return: Returns a complex valued tensor if ``x`` is a tensor, else a 2-element k list with the real and imaginary part of the eigenvalues. If an output tensor is supplied, function output is written to the tensor and null returned.

.. note:: By default, complex tensors are converted to k arrays with the real and imaginary parts along the 1st dimension, see :ref:`settings <complex-first>` for more detail.

::

   q)show x:(7 1 1.0; 3 1 2.0; 1 3 2.0)
   7 1 1
   3 1 2
   1 3 2

   q)v:eigvals x

   q)v 0          /real part
   8 3 -1f

   q)v 1          /imaginary part
   0 0 0f

   q)/check determinant zero for all eigenvalues:
   q){det x-diag count[x]#y}[x]'[v 0]
   8.349e-14 1.679e-14 3.07e-14

.. _eigh:

eigh
^^^^
`pytorch.linalg.eigh <https://pytorch.org/docs/stable/generated/torch.linalg.eigh.html>`_ is implemented by function :func:`eigh`, which calculates eigenvalues and eigenvectors for a symmetric or complex Hermitian matrix or a set of matrices.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **eigenvalue decomposition** of a complex Hermitian or real symmetric matrix
:math:`A \in \mathbb{K}^{n \times n}` is defined as

.. math::

    A = Q \operatorname{diag}(\Lambda) Q^{\text{H}}\mathrlap{\qquad Q \in \mathbb{K}^{n \times n}, \Lambda \in \mathbb{R}^n}

where :math:`Q^{\text{H}}` is the conjugate transpose when :math:`Q` is complex, and the transpose when :math:`Q` is real-valued.
:math:`Q` is orthogonal in the real case and unitary in the complex case.

.. function:: eigh(x;upper) -> eigenvalues and eigenvectors
.. function:: eigh(x;upper;output) -> null

   :param array,tensor x: k array or tensor :doc:`pointer <pointers>` to a square matrix or batches of square matrices
   :param bool upper: an optional flag, set ``false`` by default to indicate only the lower tirangular part of the matrix is used, set ``true`` to use only the pper triangle
   :param vector output: an optional :doc:`vector <vectors>` to use for function output of eigenvalues and eigenvectors
   :return: Returns a vector of tensors if ``x`` is a tensor, else a 2-element k list with eigenvalues and eigenvectors, corresponding to :math:`\Lambda` and :math:`Q` above. The eigenvalues and vectors will be complex even when input ``x`` is real. If an output vector is supplied, function output is written to the vector and null returned.

::

   q)seed 123
   q)x:return tensor(`randn;3 3;`double)
   q)x:x mmu flip x  / make symmetric

   q)x
   0.1635  -0.1946 -0.1022
   -0.1946 1.534   1.205  
   -0.1022 1.205   1.62   

   q)v:eigh x
   q)v 0                     /eigen values
   0.1276 0.3907 2.8

   q)v 1
   0.9594  -0.2709 -0.07897  /eigen vectors
   0.248   0.6761  0.6938  
   -0.1346 -0.6852 0.7158  

   q)mmu/[(v 1;diag v 0;flip v 1)]
   0.1635  -0.1946 -0.1022
   -0.1946 1.534   1.205  
   -0.1022 1.205   1.62   

   q)allclose(x;mmu/[(v 1;diag v 0;flip v 1)])
   1b

eigvalsh
^^^^^^^^
`pytorch.linalg.eigvalsh <https://pytorch.org/docs/stable/generated/torch.linalg.eigvalsh.html>`_ is implemented by function :func:`eigvalsh`, which returns only the eiganvalues from the decomposition of a symmetric or complex Hermitian matrix (see :ref:`eigh <eigh>` for more detail).

.. function:: eigvalsh(x) -> eigenvalues
.. function:: eigvalsh(x;output) -> null

   :param array,tensor x: k array or tensor :doc:`pointer <pointers>` to a square symmetric or complex Hermitian matrix or batches of matrices
   :param bool upper: an optional flag, set ``false`` by default to indicate only the lower tirangular part of the matrix is used, set ``true`` to use only the pper triangle
   :param vector output: an optional tensor :doc:`pointer <pointers>` to use for function output of eigenvalues
   :return: Returns a tensor if ``x`` is a tensor, else a k list. If an output tensor is supplied, function output is written to the tensor and null returned.

::

   q)seed 123
   q)x:tensor(`randn;3 3;`cdouble)
   q)t:transpose x
   q)use[x]add(x;t)

   q)real x
   1.287  0.1874 -0.622
   0.1874 -1.52  -1.13 
   -0.622 -1.13  1.006 

   q)imag x
   0.2174  -0.3182 1.072 
   -0.3182 1.347   0.7224
   1.072   0.7224  0.6038

   q)return eigvalsh x
   -2.124 0.181 2.716


qr
^^
`torch.linalg.qr <https://pytorch.org/docs/stable/generated/torch.linalg.qr.html>`_ is implemented as function :func:`qr`, which computes the QR decomposition of a matrix of batches of matrices.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **full QR decomposition** of a matrix
:math:`A \in \mathbb{K}^{m \times n}` is defined as

.. math::

    A = QR\mathrlap{\qquad Q \in \mathbb{K}^{m \times m}, R \in \mathbb{K}^{m \times n}}

where :math:`Q` is orthogonal in the real case and unitary in the complex case, and :math:`R` is upper triangular.

When `m > n` (tall matrix), as `R` is upper triangular, its last `m - n` rows are zero.
In this case, we can drop the last `m - n` columns of `Q` to form the
**reduced QR decomposition**:

.. math::

    A = QR\mathrlap{\qquad Q \in \mathbb{K}^{m \times n}, R \in \mathbb{K}^{n \times n}}

The reduced QR decomposition agrees with the full QR decomposition when `n >= m` (wide matrix).

.. function:: qr(x;mode) -> QR decomposition
.. function:: qr(x;mode;output) -> null

   :param array,tensor x: k array or tensor of shape :math:`(*, m, n)` where :math:`*` is zero or more batch dimensions
   :param symbol mode: optional, if given, must be one of the following:
    - ```reduced`` - default, return Q of shape :math:`(*,m,k)` and R of shape :math:`(*,k,n)`
    - ```complete`` - return Q of shape :math:`(*,m,m)` and R of shape :math:`(*,m,n)`
    - ```r`` - return empty Q and R of shape :math:`(*,k,n)`
   :param vector output: an optional :doc:`vector <vectors>` to use for function output of Q and R matrices
   :return: Returns the :math:`Q` and :math:`R` matrices as a 2-element tensor vector if tensor input, else a k list. If an output vector given, the matrices are written to the vector supplied and null is returned.

::

   q)show x:(12 -51 4.0; 6 167 -68.0; -4 24 -41.0)
   12 -51 4  
   6  167 -68
   -4 24  -41

   q)y[0] mmu last y:qr x
   12 -51 4  
   6  167 -68
   -4 24  -41

.. _lu:

lu
^^
`torch.linalg.lu_factor <https://pytorch.org/docs/stable/generated/torch.linalg.lu_factor.html>`_ is implemented as function :func:`lu`.

This function computes a compact representation of the LU decomposition given a matrix or set of matrices.
If the matrix is square, this representation may be used in :func:`lusolve`
to solve system of linear equations that use the same input matrix.

The returned decomposition has 2 parts:
The ``LU`` matrix has the same shape as the input matrix or matrices. Its upper and lower triangular
parts encode the non-constant elements of ``L`` and ``U`` of the LU decomposition.

The returned permutation matrix is represented by a 1-indexed vector. `pivots[i] == j` represents
that in the `i`-th step of the algorithm, the `i`-th row was permuted with the `j-1`-th row.

On CUDA, pivot can be set ``false`` to function returns the LU decomposition without pivoting if the decomposition exists.

.. function:: lu(x;pivot) -> compact factorization and pivots
.. function:: lu(x;pivot;output) -> null

   :param array,tensor x: k array or tensor of shape :math:`(*, m, n)` where :math:`*` is zero or more batch dimensions
   :param bool pivot: set ``true`` by default but can be set ``false`` with ``CUDA`` tensors to attempt the LU decomposition without pivoting.
   :param vector output: an optional :doc:`vector <vectors>` to use for function output of the LU compact decomposition and the pivots
   :return: Returns a 2-element tensor vector if tensor input, else a 2-element k array, with the ``LU`` matrix and pivots. If an output vector supplied, these elements are written to the vector and null returned.

::

   q)x:tensor(x;`cuda) /use CUDA to turn off pivoting
   q)v:lu(x;0b)
   q)show m:vector(v;0)
   3  -7 -2 2 
   -1 -2 -1 2 
   2  -5 -1 1 
   -3 8  3  -1

   q)vector(v;1)
   1 2 3 4i

   q)show U:triu m
   3 -7 -2 2 
   0 -2 -1 2 
   0 0  -1 1 
   0 0  0  -1

   q)show L:tril[(m;-1)]+diag count[m]#1.0
   1  0  0 0
   -1 1  0 0
   2  -5 1 0
   -3 8  3 1

   q)L mmu U
   3  -7 -2 2 
   -3 5  1  0 
   6  -4 0  -5
   -9 5  -5 12

   q)tensor[x]~L mmu U
   1b

lux
^^^
`torch.linalg.lu_factor_ex <https://pytorch.org/docs/stable/generated/torch.linalg.lu_factor_ex.html>`_ is implemented as function :func:`lux`, which computes the compact LU factorization of a matrix or a set of matrices, but includes additional flag for error checking and returns additional error codes. See `lu <lu>` for more information on the compact LU factorization.

.. function:: lux(x;pivot;check) -> compact factorization with pivots and error codes
.. function:: lux(x;pivot;check;output) -> null

   :param array,tensor x: k array or tensor of shape :math:`(*, m, n)` where :math:`*` is zero or more batch dimensions
   :param bool pivot: set ``true`` by default but can be set ``false`` with ``CUDA`` tensors to attempt the LU decomposition without pivoting.
   :param bool check: set ``false`` by default, set ``true`` for the function to signal an error if any LU decomposition fails
   :param vector output: an optional :doc:`vector <vectors>` to use for function output of the LU compact decomposition and the pivots
   :return: Returns a 3-element tensor vector if tensor input, else a 3-element k array, with the ``LU`` matrix, pivots and error codes. If an output vector supplied, these elements are written to the vector and null returned.

::

   q)show x:(3 -7 -2 2.0; -3 5 1 0.0; 6 -4 0 -5.0; -9 5 -5 12.0)
   3  -7 -2 2 
   -3 5  1  0 
   6  -4 0  -5
   -9 5  -5 12

   q)v:lux x

   q)v 0 / compact LU matrix
   -9      5      -5      12     
   -0.3333 -5.333 -3.667  6      
   -0.6667 0.125  -2.875  2.25   
   0.3333  -0.625 -0.1304 0.04348

   q)v 1 /pivots
   4 4 3 4i

   q)v 2 /error codes
   0i

   q)x-mmu/[luunpack 2#v]
   0 0 0 0         
   0 0 0 -2.082e-17
   0 0 0 0         
   0 0 0 0         

lun
^^^
`torch.lu_unpack <https://pytorch.org/docs/stable/generated/torch.lu_unpack.html>`_ is implemented by function :func:`lun`, 
which unpacks the data and pivots from the LU factorization, see :ref:`lu <lu>`.
The result of :func:`lun` is a permutation matrix P and the lower triangular matrix L and upper triangular matrix M such that
the original input matrix can be recreated by the matrix product of P x L x U.

.. function:: lun(lu;dataflag;pivotflag) -> P, L, U matrix or matrices
.. function:: lun(lu;dataflag;pivotflag;output) -> null

   :param array,vector lu: output from :func:`lu`, either a 2-element k list or tensor vector of the compact LU matrix and pivots 
   :param bool dataflag: set ``true`` by default, optional flag that can be set ``false`` to skip unpack of L & U matrices
   :param bool pivotflag: set ``true`` by default, optional flag that can be set ``false`` to skip processing of pivots
   :param vector output: an optional :doc:`vector <vectors>` to use for function output of unpacked pivot, lower triangular and upper triangular matrices.
   :result: Returns 3 matrices or sets of matrices, as a vector of tensors if any tensor input, else a k list with the unpacked pivot information and the lower and upper triangular matrices. If a trailing output vector supplied, output is written to the vector and null retuned.

An alternate form of the function takes two inputs: the compact LU matrix and the pivot information:

.. function:: lun(matrix;pivot;dataflag;pivotflag) -> P, L, U matrix or matrices
.. function:: lun(matrix;pivot;dataflag;pivotflag;output) -> null

   :param array,tensor matrix: the compact LU matrix or set of matrices from a previous :func:`lu` call
   :param array,tensor pivot: the pivot information from a previous :func:`lu` call

	| Remaining parameters and results are the same as the prior :func:`lun` call which uses the vector/list output of :func:`lu` directly.

::

   q)show x:(3 -7 -2 2.0; -3 5 1 0.0; 6 -4 0 -5.0; -9 5 -5 12.0)
   3  -7 -2 2 
   -3 5  1  0 
   6  -4 0  -5
   -9 5  -5 12

   q)v:lu x   /compact LU factorization
   q)u:lun v  /unpacked

   q)u 0      /unpacked from pivots
   0 1 0 0
   0 0 0 1
   0 0 1 0
   1 0 0 0

   q)u 1     /L - lower triangular matrix
   1       0      0       0
   -0.3333 1      0       0
   -0.6667 0.125  1       0
   0.3333  -0.625 -0.1304 1

   q)u 2     /U - upper triangular matrix
   -9 5      -5     12     
   0  -5.333 -3.667 6      
   0  0      -2.875 2.25   
   0  0      0      0.04348

   q)mmu/[u]  /product of unpacked LU factorization
   3  -7 -2 2        
   -3 5  1  2.082e-17
   6  -4 0  -5       
   -9 5  -5 12       

   q)allclose(x; mmu/[u])
   1b

Same as above example, but using tensors and tensor vectors instead of k arrays:

::

   q)x:tensor x
   q)v:lu x
   q)u:lun v
   
   q)size u
   4 4
   4 4
   4 4

   q)allclose(x;mmu/[vector u])
   1b

   q)m:tensor(v;0)  /extract compact LU matrix
   q)p:tensor(v;1)  /exctact pivot information
   q)use[u]lun(m;p) /inputs are individual tensors
   q)allclose(x;mmu/[vector u])
   1b

.. _svd:

svd
^^^
`torch.linalg.svd <https://pytorch.org/docs/stable/generated/torch.linalg.svd.html>`_ is implemented as function :func:`svd`, which computes the singular value decomposition (SVD) of a matrix or a set of matrices.

Letting :math:`\mathbb{K}` be :math:`\mathbb{R}` or :math:`\mathbb{C}`,
the **full SVD** of a matrix
:math:`A \in \mathbb{K}^{m \times n}`, if `k = min(m,n)`, is defined as

.. math::

    A = U \operatorname{diag}(S) V^{\text{H}}
    \mathrlap{\qquad U \in \mathbb{K}^{m \times m}, S \in \mathbb{R}^k, V \in \mathbb{K}^{n \times n}}

where :math:`\operatorname{diag}(S) \in \mathbb{K}^{m \times n}`,
:math:`V^{\text{H}}` is the conjugate transpose when :math:`V` is complex, and the transpose when :math:`V` is real-valued.
The matrices  :math:`U`, :math:`V` (and thus :math:`V^{\text{H}}`) are orthogonal in the real case, and unitary in the complex case.

When `m > n` (resp. `m < n`) we can drop the last `m - n` (resp. `n - m`) columns of `U` (resp. `V`) to form the **reduced SVD**:

.. math::

    A = U \operatorname{diag}(S) V^{\text{H}}
    \mathrlap{\qquad U \in \mathbb{K}^{m \times k}, S \in \mathbb{R}^k, V \in \mathbb{K}^{k \times n}}

where :math:`\operatorname{diag}(S) \in \mathbb{K}^{k \times k}`.
In this case, :math:`U` and :math:`V` also have orthonormal columns.

.. function:: svd(x;full) -> QR decomposition
.. function:: svd(x;full;output) -> null

   :param array,tensor x: k array or tensor of shape :math:`(*, m, n)` where :math:`*` is zero or more batch dimensions
   :param bool full: set ``true`` by default to compute the full SVD, set ``false`` to return the reduced SVD
   :param vector output: an optional :doc:`vector <vectors>` to use for function output of :math:`U`, :math:`S` and :math:`V^{\text{H}}` matrices
   :return: Returns :math:`U`, :math:`S` and :math:`V^{\text{H}}` matrices as a 3-element tensor vector if tensor input given, else a k list. If an output vector given, these matrices are written to the supplied vector and null is returned.

::

   q)show x:(3 2 2.0; 2 3 -2.0)
   3 2 2 
   2 3 -2

   q){mmu/[(x;diag y;z)]} . svd(x;0b)
   3 2 2 
   2 3 -2


svdvals
^^^^^^^
`torch.linalg.svdvals <https://pytorch.org/docs/stable/generated/torch.linalg.svdvals.html>`_ is implemented as function :func:`svdvals`, which computes the singular values of a matrix or a set of matrices (see :ref:`svd <svd>` for more detail on the full SVD decomposition).

.. function:: svdvals(x) -> singular values of QR decomposition
.. function:: svd(x;output) -> null

   :param array,tensor x: k array or tensor of shape :math:`(*, m, n)` where :math:`*` is zero or more batch dimensions
   :param tensor output: an optional tensor :doc:`pointer <pointers>` to use for function output
   :return: Returns the singular values as a tensor if tensor input, else a k list. If an output tensr given, these values are written to the supplied tensor and null is returned.

::

   q)x:(3 2 2.0; 2 3 -2.0)

   q)svdvals x
   5 3f

   q)sqrt first eigvals x mmu flip x
   5 3f

Solvers
*******

solve
^^^^^
`torch.linalg.solve <https://pytorch.org/docs/stable/generated/torch.linalg.solve.html>`_ is implemented as :func:`solve`,
which calculates the solution of a square system of linear equations, ``ax = b``

.. function:: solve(a;b) -> x
.. function:: solve(a;b;output) -> null

   :param array,tensor a: input array or tensor :doc:`pointer <pointers>` of shape ``(*, n, n)``, where * is zero or more batch dimensions.
   :param array,tensor b: input array or tensor :doc:`pointer <pointers>` of shape right-hand side values of shape ``(*, n)``, ``(*, n, k)``, ``(n)`` or ``(n, k)`` according to the rules described `here <https://pytorch.org/docs/stable/generated/torch.linalg.solve.html>`_.
   :param tensor output: an optional :doc:`tensor <pointers>` to use for function output
   :return: The solution ``x`` of a square system of linear equations, ``ax = b``,  as a tensor if any tensor input, else as an array.  If an output tensor given, solution is written to the tensor with null return.

::

   q)a:tensor(`randn; 3 3; `double)
   q)b:tensor(`randn; 3 4; `double)
   q)x:solve(a;b)  /solve ax=b

   q)tensor[a] mmu tensor x
   2.02  0.603 0.0223 -0.964
   0.478 -1.32 0.386  0.42  
   0.32  0.311 0.0215 1.08  

   q)tensor b
   2.02  0.603 0.0223 -0.964
   0.478 -1.32 0.386  0.42  
   0.32  0.311 0.0215 1.08  

   
trisolve
^^^^^^^^
`torch.linalg.solve_triangular <https://pytorch.org/docs/stable/generated/torch.linalg.solve_triangular.html>`_ is implemented as :func:`trisolve`, which calculates the solution of a triangular system of linear equations.

.. function:: trisolve(a;b;upper;left;unitriangular) -> x
.. function:: trisolve(a;b;upper;left;unitriangular;output) -> null

   :param array,tensor a: input array or tensor :doc:`pointer <pointers>` of shape ``(*, n, n)`` or ``(*, k, k)``  if ``left = true``
   :param array,tensor b: input array or tensor :doc:`pointer <pointers>` of shape right-hand side values of shape ``(*, n, k)``
   :param bool upper: required flag, set ``true`` if ``a`` is an upper triangular matrix or matrices, ``false`` for lower triangular
   :param bool left: default is ```true`` to solve for ``x`` in ``ax = b``, ``false`` to solve ``xa = b``
   :param bool unitriangular: default is ``false``, set ``true`` if the diagonal elements of ``a`` are all equal to 1
   :param tensor output: an optional :doc:`tensor <pointers>` to use for function output (``b`` may be passed as an output tensor with the result overwriting values in ``b``)
   :return: The solution ``x`` of a triangular system of linear equations, ``ax = b`` or ``xa = b`` as a tensor if any tensor input, else as an array.  If an output tensor given, solution is written to the tensor with null return.


::

   q)seed 123
   q)a:tensor(`randn; 3 3; `double)
   q)b:tensor(`randn; 3 4; `double)

   q)triu(a;[])
   q)tensor a
   -0.111 0.12 -0.37
   0      -1.2 0.209
   0      0    0.324

   q)x:trisolve(a;b;1b)
   q)B:mm(a;x)

   q)allclose(b;B)
   1b


cholsolve
^^^^^^^^^
`torch.cholesky_solve <https://pytorch.org/docs/stable/generated/torch.cholesky_solve.html>`_ is implemented by function
:func:`cholsolve`, which solves a linear system of equations with a positive semidefinite matrix to be inverted given its Cholesky factor matrix :math:`u`.

By default, :math:`u` is lower triangular and the result returned is:

.. math::
    (u u^T)^{{-1}} b

If :math:`u` is passed with the ``upper=true``, the result becomes:

.. math::
    (u^T u)^{{-1}} b


.. function:: cholsolve(b;u;upper) -> solution matrix or batch of matrices
.. function:: cholsolve(b;u;upper;output) -> null

   :param array,tensor b: input array or tensor :doc:`pointer <pointers>` of size :math:`(*, m, k)`, where :math:`*` is zero or more batch dimensions
   :param array,tensor u: input array or tensor :doc:`pointer <pointers>` of Cholesky factors, size :math:`(*, m, m)`, where :math:`*` is zero or more batch dimensions
   :param bool upper: optional flag, ``false`` by default, set ``true`` to indicate that ``u`` is upper triangular
   :param tensor output: an optional :doc:`tensor <pointers>` to use for function output
   :return: solution matrix or set of matrices as tensor if any input supplied as tensor else k array. If output tensor supplied, results are written to the supplied tensor and null returned.

::

   q)seed 123
   q)a:tensor(`randn;3 3;`double)
   q)t:transpose a
   q)use[a]mm(a;t) /make symmetric positive definite

   q)u:chol a
   q)b:tensor(`randn;3 2;`double)
   q)r:cholsolve(b;u)

   q)i:inverse a
   q)s:mm(i;b)      /compare alternate solution
   q)allclose(r;s)
   1b

   q)tensor r
   -1.712 1.685 
   -1.707 0.2702
   1.572  0.123 

   q)inv[tensor a]mmu tensor b
   -1.712 1.685 
   -1.707 0.2702
   1.572  0.123 

lstsq
^^^^^

`torch.linalg.lstsq <https://pytorch.org/docs/stable/generated/torch.linalg.lstsq.html>`_  is implemented as :func:`lstsq`,
which calculates a solution to the least squares problem of a system of linear equations.

.. function:: lstsq(a;b;rcond;method) -> vector of x,residuals,rank,singular values
.. function:: lstsq(a;b;rcond;method;output) -> null

   | Allowable argument combinations:

    - ``lstsq(a;b)``
    - ``lstsq(a;b;rcond)``
    - ``lstsq(a;b;rcond;method)``
    - ``lstsq(a;b;method)``
    - any of the above combinations followed by a trailing output vector

   :param array,tensor a: input array or tensor :doc:`pointer <pointers>` of shape ``(*, m, n)`` where ``*`` is zero or more batch dimensions
   :param array,tensor b: input array or tensor :doc:`pointer <pointers>` of shape ``(*, m, k)`` where ``*`` is zero or more batch dimensions
   :param double rcond: optional effective rank of ``a``, if not specified or set to ``0n``, the machine precision of the data type of ``a`` multiplied by the maximum value of dimensions ``(m, n)`` is used
   :param symbol method: optional name of the LAPACK/MAGMA method, one of ```gels``, ```gelsd``, ```gelss`` or ```gelsy``
   :param vector output: a vector `pointer <vectors>` to contain function output
   :return: The least squares solution ``x`` for ``ax = b``, along with residuals, rank and any singular values. Returns a vector of tensors if either ``a`` or ``b`` is given as a tensor, else as a k list. If output vector supplied, writes function output to given vector and returns null.

::

   q)a:1 3 3#10 2 3 3 10 5 5 6 12e
   q)b:2 3 3#2 5 1 3 2 1 5 1 9 4 2 9 2 0 3 2 5 3e

   q)`x`resid`rank`singular!lstsq(a;b)
   x       | ((0.0793 0.535 -0.123e;0.113 0.146 -0.352e;0.327 -0.212 0.977e);(0...
   resid   | `real$()
   rank    | ,3
   singular| `real$()

   q)`x`resid`rank`singular!lstsq(a;b;`gelsd)  /singular value decomposition
   x       | ((0.0793 0.535 -0.123e;0.113 0.146 -0.352e;0.327 -0.212 0.977e);(0...
   resid   | `real$()
   rank    | ,3
   singular| ,19.1 7.75 5.29e

   q)v:vector()
   q)lstsq(a;b;`gelsd;v)
   q)vector v
   ((0.0793 0.535 -0.123e;0.113 0.146 -0.352e;0.327 -0.212 0.977e);(0.394 0.102 ..
   `real$()
   ,3
   ,19.1 7.75 5.29e

lusolve
^^^^^^^
`torch.lu_solve <https://pytorch.org/docs/stable/generated/torch.lu_solve.html>`_ is implemented by function :func:`lusolve`.

Returns the LU solve of the linear system :math:`Ax = b` using the partially pivoted
LU factorization of A from :func:`lu`.

.. function:: lusolve(b;lu) -> solution x of Ax=b
.. function:: lusolve(b;lu;output) -> null

   :param array,tensor b: the right hand side of :math:`Ax = b`, a an array or tensor of size :math:`(*, m, k)` where :math:`*` is zero or more batch dimensions
   :param array,vector lu: output from :func:`lu`, either a 2-element k list or tensor vector of the compact LU matrix and pivots 
   :param tensor output: an optional tensor :doc:`pointer <pointers>` to use for function output
   :result: Returns solution :math:`x` of :math:`Ax = b` as a matrix or set of matrices, result is a tensor if any inputs are tensors or a vector of tensors, else a k array. If optional trailing argument is an output tensor, solution is written to the supplied tensor and null returned.

An alternate form of the call:

.. function:: lusolve(b;matrix;pivot) -> solution x of Ax=b
.. function:: lusolve(b;matrix;pivot;output) -> null

   :param array,tensor b: the right hand side of :math:`Ax = b`, a an array or tensor of size `:math:`(*, m, k)` where `:math:`*` is zero or more batch dimensions
   :param array,tensor matrix: the compact LU matrix or set of matrices from a previous :func:`lu` call
   :param array,tensor pivot: the pivot information from a previous :func:`lu` call

	| Remaining parameters and results are the same as the prior :func:`lusolve` call which uses the vector/list output of :func:`lu` directly.

::

   q)a:tensor(`randn;2 3 3)
   q)b:tensor(`randn;2 3 1)
   q)v:lu a  /LU factorization

   q)x:lusolve(b;v)

   q)x:lusolve(b;v)   / solve for x in Ax=b
   q)r:bmm(a;x)       / recalc b from Ax
   q)allclose(b;r)
   1b

   q)m:tensor(v;0)    / extract compact LU matrix from factorization
   q)pv:tensor(v;1)   / extract tensor of pivot information

   q)x1:lusolve(b;m;pv)  / alternate call with individual tensors
   q)equal(x;x1)
   1b



Inverses
********

inverse
^^^^^^^
`torch.linalg.inv <https://pytorch.org/docs/stable/generated/torch.inv.html>`_ is implemented by :func:`inverse`.

.. function:: inverse(matrix) -> k array
.. function:: inverse(matrix;output) -> null

   :param array,tensor matrix: square input matrix or tensor :doc:`pointer <pointers>` to a matrix or batch of matrices
   :param tensor output: a :doc:`pointer <pointers>` to a previously allocated tensor to be used for output
   :return: The function returns a k array if given a k array as input and returns a tensor if a tensor is given as input.  If an output tensor is supplied, this tensor is filled with the output values and null is returned.

::

   q)x:tensor(`randn;3 3)
   q)tensor x
   0.336 1.12   0.17  
   0.214 -0.497 0.519 
   0.489 -0.23  -0.491

   q)inverse tensor x
   0.654 0.916  1.2   
   0.646 -0.447 -0.249
   0.348 1.12   -0.73 

   q)y:inverse x
   q)tensor y
   0.654 0.916  1.2   
   0.646 -0.447 -0.249
   0.348 1.12   -0.73 

   q)z:tensor 0#0e
   q)inverse(x;z)
   q)tensor z
   0.654 0.916  1.2   
   0.646 -0.447 -0.249
   0.348 1.12   -0.73 

pinverse
^^^^^^^^
`torch.linalg.pinv <https://pytorch.org/docs/stable/generated/torch.linalg.pinv.html>`_ is implemented by :func:`pinverse`, which calculates the pseudo-inverse with optional absolute and relative tolerance.

.. function:: pinverse(input;atol;rtol;hermitian) -> rank
.. function:: pinverse(input;atol;rtol;hermitian;output) -> null

   | Allowable argument combinations:

    - ``pinverse(input)``
    - ``pinverse(input;atol)``
    - ``pinverse(input;atol;rtol)``
    - ``pinverse(input;atol;rtol;hermitian)``
    - ``pinverse(input;hermitian)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor input: input ``* x N x M`` array or tensor :doc:`pointer <pointers>`  where ``*`` can be other batch dimensions
   :param double atol: absolute tolerance, default=0 if left unspecified.
   :param double rtol: relative tolerance, default is derived from input, see `torch.linalg.pinv <https://pytorch.org/docs/stable/generated/torch.linalg.pinv.html>`_
   :param bool hermitian: optional flag, set ``false`` by default, set ``true`` to indicate Hermitian if complex input else symmetric for real input.
   :param tensor output: an optional `complex <complex>` tensor to use for function output
   :return: Returns the pseudo-inverse(s) of the input, as an array if input is an array, else as tensor. If output tensor supplied, writes output to tensor and returns null.

::

   q)seed 123
   q)show x:return tensor(`randn;2 3;`double)
   -0.1115 0.1204 -0.3696
   -0.2404 -1.197 0.2093 

   q)pinverse x
   -1.022  -0.2864
   -0.2266 -0.8089
   -2.471  -0.177 

   q)x mmu pinverse x
   1         -8.327e-17
   5.551e-16 1         

   q)x:tensor((x;x;x);(x;x;x))
   q)size x
   2 3 2 3

   q)y:pinverse x
   q)size y
   2 3 3 2

   q)tensor[y][0][0]
   -1.022  -0.2864
   -0.2266 -0.8089
   -2.471  -0.177 

cholinverse
^^^^^^^^^^^
`torch.cholesky_inverse <https://pytorch.org/docs/stable/generated/torch.cholesky_inverse.html>`_ is implemented as function :func:`cholinverse`,
which computes the inverse of a symmetric positive-definite matrix using its Cholesky factorizations.

If ``upper`` is false, :math:`u` is lower triangular
such that the returned tensor is

.. math::
    (uu^{{T}})^{{-1}}

If ``upper`` is true, :math:`u` is upper
triangular such that the returned tensor is

.. math::
    (u^T u)^{{-1}}

.. function:: cholinverse(x;upper) -> inverse
.. function:: cholinverse(x;upper;output) -> null

   :param array,tensor x: the input factorizations as k array or :doc:`tensor <pointers>` pointer of size :math:`(*, n,n)`, where :math:`*` is zero or more batch dimensions
   :param bool upper: default=``false`` for lower triangular input, set ``true`` for upper triangular input
   :param tensor output: an optional output tensor for function output
   :return: Returns the inverse(s) derived from the input factorizations, returned as a tensor if tensor input, else k array. If cout tensor supplied, output is written to the given tensor and null returned.

::

   q)seed 123
   q)a:tensor(`randn;3 3;`double)
   q)t:transpose a
   q)use[a]mm(a;t)

   q)u:chol a
   q)i:cholinverse u

   q)tensor[a]mmu tensor i
   1         -2.776e-17 2.776e-17
   2.22e-16  1          2.22e-16 
   3.331e-16 0          1        


Matrix functions
****************

power
^^^^^

`torch.linalg.matrix_power <https://pytorch.org/docs/stable/generated/torch.linalg.matrix_power.html>`_ is implemented as :func:`power`.

.. function:: power(matrix;n) -> nth power of square matrix
.. function:: power(matrix;n;output) -> null

   :param matrix,tensor matrix: a k array or :doc:`pointer <pointers>` to a tensor of a square matrix or batches of matrices.
   :param long n: integer power
   :param tensor output: a :doc:`pointer <pointers>` to a previously allocated tensor to be used for output
   :return: The function returns the input matrix/tensor raised to the given integer power. If an output tensor supplied, the raised matrix iw written to the supplied tensor and null is returned.

::

   q)x:3 3#0 1 2 3 4 5 6 7 8.0
   q)power(x;2)
   15 18 21 
   42 54 66 
   69 90 111

   q)power(x;3)
   180 234  288 
   558 720  882 
   936 1206 1476

   q)x$x$x
   180 234  288 
   558 720  882 
   936 1206 1476

   q)x:tensor(x;x;x)  /batches of matrices
   q)y:power(x;2)
   q)size y
   3 3 3

   q)tensor(y;2)
   15 18 21 
   42 54 66 
   69 90 111

Matrix products
***************


bmm
^^^
`torch.bmm <https://pytorch.org/docs/stable/generated/torch.bmm.html>`_ is implemented as :func:`bmm`, computing the batch matrix-matrix product.

Both inputs must be 3-dimensional arrays or tensors each containing the same number of matrices.

If ``x`` is a 
:math:`(b \times n \times m)` tensor, ``y`` is a
:math:`(b \times m \times p)` tensor and the result will be a
:math:`(b \times n \times p)` tensor.

.. math::
    \text{result}_i = \text{x}_i \mathbin{@} \text{y}_i


.. function:: bmm(x;y) -> matrix products
.. function:: bmm(x;y;output) -> null

   :param array,tensor x: a k array or tensor :doc:`pointer <pointers>` to batches of matrices
   :param array,tensor y: a k array or tensor :doc:`pointer <pointers>` to the same number of matrices as in ``x``
   :param tensor output: an optional :doc:`pointer <pointers>` to a previously allocated tensor to be used for output
   :return: Returns the matrix products as an array if both inputs given as k arrays, else tensor. If output tensor given, result is written to the given tensor and null return.

.. note:: This function does not support `broadcasting <https://pytorch.org/docs/stable/notes/broadcasting.html>`_.
          For broadcasting matrix products, see :func:`matmul`.

::

   q)x:tensor(`randn; 100 3 4; `double)
   q)y:tensor(`randn; 100 4 5; `double)
   q)z:bmm(x;y)

   q)size z
   100 3 5

   q)allclose(z;tensor[x]mmu tensor y)
   1b

cross
^^^^^
`torch.linalg.cross <https://pytorch.org/docs/stable/generated/torch.linalg.cross.html>`_ is implemented as function :func:`Cross`, which computes the cross product of two vectors or batches of vectors.

.. function:: Cross(x;y;dim) -> cross product
.. function:: Cross(x;y;dim;output) -> null

   :param array,tensor x: a k array or tensor :doc:`pointer <pointers>` 
   :param array,tensor y: a k array or tensor :doc:`pointer <pointers>` with the same number of elements as ``x``
   :param long dim: the dimension along which to calculate the cross product, default=-1, the last dimension
   :param tensor output: an optional :doc:`pointer <pointers>` to a previously allocated tensor to be used for output
   :return: Returns a single cross product if given single vectors, else batches of cross products along given ``dim``, with output having the same batch dimensions of the inputs `broadcast <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ to a common shape. If all inputs are k arrays, returns k array, else tensor. If output tensor given, results are written there and null return.

::

   q)seed 123
   q)x:return tensor(`randn; 3;`double)
   q)y:return tensor(`randn; 3;`double)

   q)show z:Cross(x;y)
   -0.4172 0.1122 0.1624

   q)(x[i]*y j) - x[j:2 0 1]*y i:1 2 0
   -0.4172 0.1122 0.1624

   q)x:return tensor(`randn;10 3;`double)
   q)y:return tensor(`randn;10 3;`double)
   q)z:Cross(x;y)

   q)allclose(z; (x[;i]*y[;j]) - x[;j:2 0 1]*y[;i:1 2 0])
   1b
   
dot
^^^
`torch.dot <https://pytorch.org/docs/stable/generated/torch.dot.html>`_ is implemented as function :func:`dot`, which computes the dot product of two 1-dimensional inputs with the same number of elements.

.. function:: dot(x;y) -> matrix products
.. function:: dot(x;y;output) -> null

   :param list,tensor x: a k list or 1-dimensional tensor :doc:`pointer <pointers>` 
   :param list,tensor y: a k list or 1-dimensional tensor :doc:`pointer <pointers>` with the same number of elements as ``x``
   :param tensor output: an optional :doc:`pointer <pointers>` to a previously allocated tensor to be used for output
   :return: Returns the dot products as a k scalar if both inputs given as k lists, else tensor. If output tensor given, result is written to the given tensor and null return.

::

   q)dot(2 3;1 10)
   32

   q)x:tensor 2 3
   q)z:dot(x;1 10)
   q)tensor z
   32

householder
^^^^^^^^^^^
`torch.linalg.householder_product <https://pytorch.org/docs/stable/generated/torch.linalg.householder_product.html>`_ is implemented as function :func:`householder`, which computes the first n columns of a product of Householder matrices.

.. function:: householder(x;y) -> Householder product
.. function:: householder(x;y;output) -> null

   :param array,tensor x: a k array or tensor :doc:`pointer <pointers>` of shape :math:`(*, m, n)` where :math:`*` is zero or more batch dimensions
   :param array,tensor y: a k array or tensor :doc:`pointer <pointers>` of shape :math:`(*, k)` where :math:`*` is zero or more batch dimensions
   :param tensor output: an optional :doc:`pointer <pointers>` to a previously allocated tensor to be used for output
   :return: Returns the Householder products as a k array if both inputs given as k arrays, else tensor. If output tensor given, result is written to the given tensor and null return.

::

   q)h:tensor(`randn;3 2 2;`cdouble)
   q)tau:tensor(`randn;3 1;`cdouble)
   q)q:householder(h;tau)
   q)size q
   3 2 2


matmul
^^^^^^
`torch.matmul <https://pytorch.org/docs/stable/generated/torch.matmul.html>`_ is implemented as function :func:`matmul` which calculates the matrix product of two inputs. The function behaves differently depending on the dimensions of the inputs, e.g. if both inputs are 1-dimensional, the dot product is returned.
If both inputs are matrices, a matrix-matrix product is calculated. See the PyTorch `documentation <https://pytorch.org/docs/stable/generated/torch.matmul.html>`_ for all the possible cases.


.. function:: matmul(x;y) -> matrix product
.. function:: matmul(x;y;output) -> null

   :param array,tensor x: a k array or tensor :doc:`pointer <pointers>`
   :param array,tensor y: a k array or tensor :doc:`pointer <pointers>`
   :param tensor output: an optional :doc:`pointer <pointers>` to a previously allocated tensor to be used for output
   :return: Returns the matrix products as an array if both inputs given as k arrays, else tensor. If output tensor given, result is written to the given tensor and null return.

::

   q)matmul(1 2 3;4 5 6)
   32

   q)matmul(2 3#1 2 3;4 5 6)
   32 32

   q)matmul(5 2 3#1 2 3;4 5 6)
   32 32
   32 32
   32 32
   32 32
   32 32

   q)x:tensor(`randn; 100 3 4)
   q)y:tensor(`randn; 100 4 5)

   q)z:matmul(x;y)
   q)size z
   100 3 5


mm
^^
`torch.mm <https://pytorch.org/docs/stable/generated/torch.mm.html>`_ is implemented as function :func:`mm`, which calculates a matrix multiplication of inputs ``x`` and ``y``.

If ``x`` is a :math:`(n \times m)` matrix and ``y`` is a :math:`(m \times p)` matrix, the result will be a :math:`(n \times p)` matrix.

.. function:: mm(x;y) -> matrix product
.. function:: mm(x;y;output) -> null

   :param array,tensor x: a k matrix or tensor :doc:`pointer <pointers>`
   :param array,tensor y: a k matrix or tensor :doc:`pointer <pointers>`
   :param tensor output: an optional :doc:`pointer <pointers>` to a previously allocated tensor to be used for output
   :return: Returns the matrix product as a matrix if both inputs given as k arrays, else tensor. If output tensor given, result is written to the given tensor and null return.

.. note:: This function does not support `broadcasting <https://pytorch.org/docs/stable/notes/broadcasting.html>`_.
          For broadcasting matrix products, see :func:`matmul`.

::

   q)seed 123
   q)x:tensor(`randn; 2 3; `double)
   q)y:tensor(`randn; 3 1; `double)
   q)z:mm(x;y)

   q)tensor z
   -0.102
   1.21  

   q)tensor[x]mmu tensor y
   -0.102
   1.21  

mmt
^^^
Function :func:`mmt` is a k api function that implements :func:`mm`, but transposing the second input, i.e. :math:`x y^T`. Syntax and parameters are the same as for :func:`mm`.

.. function:: mmt(x;y) -> matrix product
.. function:: mmt(x;y;output) -> null

::

   q)x:3 2#0 1 2 3 4 5.0
   q)y:3 2#6 7 8 9 8 9.0

   q)x mmu flip y
   7  9  9 
   33 43 43
   59 77 77

   q)mmt(x;y)
   7  9  9 
   33 43 43
   59 77 77

mtm
^^^
Function :func:`mtm` is a k api function that implements :func:`mm`, but with the first input transposed, i.e. :math:`x^T y`. Syntax and parameters are the same as for :func:`mm`.

.. function:: mtm(x;y) -> matrix product
.. function:: mtm(x;y;output) -> null

::

   q)x:3 2#0 1 2 3 4 5.0
   q)y:3 2#6 7 8 9 8 9.0

   q)flip[x]mmu y
   48 54
   70 79

   q)mtm(x;y)
   48 54
   70 79

multidot
^^^^^^^^
`torch.linalg.multi_dot <https://pytorch.org/docs/stable/generated/torch.linalg.multi_dot.html>`_ is implemented as function :func:`multidot`, which efficiently multiplies multiple matrices by reordering the multiplications so that the fewest arithmetic operations are performed.

.. function:: multidot(x) -> product

   :param: array,tensors,vector x: a list of k arrays and/or tensors or a single vector of tensors.
   :return: Returns the product of the given matrices as a k array if all k arrays given, else tensor.

::

   q)n:1000000
   q)x:tensor(`randn; n,5; `double)
   q)y:tensor(`randn; 5,n; `double)
   q)z:tensor(`randn; n,1; `double)

   q)r:mm(x;y)
   '[enforce fail at alloc_cpu.cpp:73] . DefaultCPUAllocator: can't allocate memory: you tried to allocate 8000000000000 bytes. Error code 12 (Cannot allocate memory)
     [0]  r:mm(x;y)
            ^

   q)r:multidot(x;y;z)
   q)size r
   1000000 1


mv
^^
`torch.mv <https://pytorch.org/docs/stable/generated/torch.mv.html>`_ is implemented as function :func:`mv`, which calculates a matrix-vector product.

If ``x`` is a :math:`(n \times m)` matrix/tensor, ``y`` is a 1-dimensional  list/tensor of size :math:`m` and result  will be 1-dimensional list of size :math:`n`.

.. function:: mv(x;y) -> matrix-vector product
.. function:: mv(x;y;output) -> null

   :param array,tensor x: a k matrix or 2-dimensional tensor :doc:`pointer <pointers>`
   :param array,tensor y: a k list or 1-dimensional tensor :doc:`pointer <pointers>`
   :param tensor output: an optional :doc:`pointer <pointers>` to a previously allocated tensor to be used for output
   :return: Returns the matrix-vector product as a matrix if both inputs given as k arrays, else tensor. If output tensor given, result is written to the given tensor and null return.

.. note:: This function does not support `broadcasting <https://pytorch.org/docs/stable/notes/broadcasting.html>`_.

::

   q)x:3 2#0 1 2 3 4 5.0
   q)y:10 100.0

   q)x mmu y
   100 320 540f

   q)mv(x;y)
   100 320 540f

outer
^^^^^
`torch.outer <https://pytorch.org/docs/stable/generated/torch.outer.html>`_ is implemented as function :func:`outer` which calculates the outer product of two 1-dimensional inputs.

If ``x`` is a list of size :math:`n` and ``y`` is a list of size :math:`m`, then the result is a matrix of size :math:`(n \times m)`.

.. function:: outer(x;y) -> outer product
.. function:: outer(x;y;output) -> null

   :param array,tensor x: a k list or 1-dimensional tensor :doc:`pointer <pointers>`
   :param array,tensor y: a k list or 1-dimensional tensor :doc:`pointer <pointers>`
   :param tensor output: an optional :doc:`pointer <pointers>` to a previously allocated tensor to be used for output
   :return: Returns the outer product as a matrix if both inputs given as k list, else tensor. If output tensor given, result is written to the given tensor and null return.

.. note:: This function does not support `broadcasting <https://pytorch.org/docs/stable/notes/broadcasting.html>`_.

::

   q)outer(.1 1;2 3 4.0)
   0.2 0.3 0.4
   2   3   4  

   q)x:tensor .1 1
   q)z:outer(x;2 3 4.0)
   q)tensor z
   0.2 0.3 0.4
   2   3   4  

   q)outer(x;1 2 3.0;z)
   q)tensor z
   0.1 0.2 0.3
   1   2   3  

Matrix products with addition
*****************************

PyTorch has a series of functions of the form: :math:`\beta\ \text{x} + \alpha\ \text{f}(\text{y}, \text{z})`,
where optional multipliers, :math:`\beta` and :math:`\alpha`, are set to ``1`` if not supplied.
Required inputs :math:`\text{x}, \text{y}, \text{z}` may be supplied as k arrays or tensors, with 
the result returned as an allocated tensor if any of the inputs was also a tensor.

If the last argument, aside from inputs and multipliers, is also a tensor, this is taken to be an output tensor:
function results will be written to the tensor and null returned.

addbmm
^^^^^^
`torch.addbmm <https://pytorch.org/docs/stable/generated/torch.addbmm.html>`_ is implemented by function :func:`addbmm`, which adds input ``x`` to the sum of  matrix-matrix products of the 3-d batches of matrices given in ``y`` and ``z``.


If :math:`y` is :math:`(b \times n \times m)` and :math:`z` is :math:`(b \times m \times p)`,
:math:`x` must be `broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_
with a :math:`(n \times p)` array/tensor
and the result will be a :math:`(n \times p)` array or tensor.

.. math::
    \beta\ \text{x} + \alpha\ (\sum_{i=0}^{b-1} \text{y}_i \mathbin{@} \text{z}_i)


.. function:: addbmm(x;y;z;beta;alpha) -> sum of batch of matrix-matrix multiplications
.. function:: addbmm(x;y;z;beta;alpha;output) -> null

   | Allowable argument combinations:

    - ``addbmm(x;y;z)``
    - ``addbmm(x;y;z;beta)``
    - ``addbmm(x;y;z;beta;alpha)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: a k array or tensor :doc:`pointer <pointers>`
   :param array,tensor y: a 3-dimensional array or tensor :doc:`pointer <pointers>` with batches of matrices
   :param array,tensor z: a 3-dimensional array or tensor :doc:`pointer <pointers>` with the same number of matrices as ``y``
   :param number beta: a numeric scalar, must be integer type if inputs are integral, else double, used as multiplier for ``x``, default=1
   :param number alpha: a numeric scalar, must be integer type if inputs are integral, else double, used as multiple for sum of matrix products, default=1
   :param tensor output: an optional :doc:`pointer <pointers>` to a previously allocated tensor to be used for output
   :return: Returns the sum of matrix products as a matrix if all inputs given as k arrays, else tensor. If output tensor given, result is written to the given tensor and null return.

::

   q)y:"f"$5 2 3#til 30
   q)z:"f"$5 3 4#til 60

   q)show r:addbmm(10.0;y;z)
   7670 7865 8060 8255
   8930 9170 9410 9650

   q)10+sum y mmu z
   7670 7865 8060 8255
   8930 9170 9410 9650

   q)r~/:{addbmm(x;y;z)}[;y;z] each (1; 1 4; 2 4)#\:10.0
   111b

addmm
^^^^^
`torch.addmm <https://pytorch.org/docs/stable/generated/torch.addmm.html>`_ is implemented by function :func:`addmm`, which adds input ``x`` to the matrix product of ``y`` and ``z``.

If ``y`` is :math:`(n \times m)` and ``z``` is :math:`(m \times p)`, then ``x`` must be
`broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_
with a :math:`(n \times p)` matrix
and the result will be a :math:`(n \times p)` matrix.

.. math::
    \beta\ \text{x} + \alpha\ (\text{y}_i \mathbin{@} \text{y}_i)

.. function:: addmm(x;y;z;beta;alpha) ->  sum of matrix-matrix multiplication with additional input
.. function:: addmm(x;y;z;beta;alpha;output) -> null

   | Allowable argument combinations:

    - ``addmm(x;y;z)``
    - ``addmm(x;y;z;beta)``
    - ``addmm(x;y;z;beta;alpha)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: a k array or tensor :doc:`pointer <pointers>`
   :param array,tensor y: a k matrix or tensor :doc:`pointer <pointers>`
   :param array,tensor z: a k matrix or tensor :doc:`pointer <pointers>`
   :param number beta: a numeric scalar, must be integer type if inputs are integral, else double, used as multiplier for ``x``, default=1
   :param number alpha: a numeric scalar, must be integer type if inputs are integral, else double, used as multiple for matrix product of ``y`` and ``z``, default=1
   :param tensor output: an optional :doc:`pointer <pointers>` to a previously allocated tensor to be used for output
   :return: Returns matrix if all inputs given as k arrays, else tensor. If output tensor given, result is written to the given tensor and null return.

::

   q)x:2 5#til 10
   q)y:2 3#til 6
   q)z:3 5#til 15

   q)addmm(x;y;z)
   25 29 33  37  41 
   75 88 101 114 127

   q){x set"f"$get x}'[`x`y`z];
   q)x+y mmu z
   25 29 33  37  41 
   75 88 101 114 127

Using ``beta`` and ``alpha`` multipliers:

::

   q)addmm(x;y;z;100)
   25  128 231 334 437 
   570 682 794 906 1018

   q)(100*x)+y mmu z
   25  128 231 334 437 
   570 682 794 906 1018

   q)addmm(x;y;z;100;.1)
   2.5 102.8 203.1 303.4 403.7
   507 608.2 709.4 810.6 911.8

   q)(100*x)+.1*y mmu z
   2.5 102.8 203.1 303.4 403.7
   507 608.2 709.4 810.6 911.8

addmv
^^^^^
`torch.addmv <https://pytorch.org/docs/stable/generated/torch.addmv.html>`_ is implemented by function :func:`addmv`, which adds input ``x`` to the matrix-vector product of ``y`` and ``z``.

.. math::
    \beta\ \text{x} + \alpha\ (\text{y} \mathbin{@} \text{z})

If :math:`y` is a matrix of size :math:`n \times m` and :math:`z` is a vector of size :math:`m`, then :math:`x` must be
`broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with a vector of size 
:math:`n` and the result will be a 1-dimensional tensor or list of size :math:`n`.

.. function:: addmv(x;y;z;beta;alpha) ->  sum of matrix-vector multiplication with additional input
.. function:: addmv(x;y;z;beta;alpha;output) -> null

   | Allowable argument combinations:

    - ``addmv(x;y;z)``
    - ``addmv(x;y;z;beta)``
    - ``addmv(x;y;z;beta;alpha)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: a k list or 1-dimensional tensor :doc:`pointer <pointers>`
   :param array,tensor y: a k matrix or 2-dimensional tensor :doc:`pointer <pointers>`
   :param array,tensor z: a k list or 1-dimensional tensor :doc:`pointer <pointers>`
   :param number beta: a numeric scalar, must be integer type if inputs are integral, else double, used as multiplier for ``x``, default=1
   :param number alpha: a numeric scalar, must be integer type if inputs are integral, else double, used as multiple for product of ``y`` and ``z``, default=1
   :param tensor output: an optional :doc:`pointer <pointers>` to a previously allocated tensor to be used for output
   :return: Returns matrix if all inputs given as k arrays, else tensor. If output tensor given, result is written to the given tensor and null return.

::

   q)x:1 2
   q)y:2 3#0 1 2 3 4 5
   q)z:10 100 1000

   q)mv(y;z)
   2100 5430
   q)x+mv(y;z)
   2101 5432

   q)addmv(x;y;z)
   2101 5432

   q)x+("f"$y)mmu "f"$z
   2101 5432f


addr
^^^^
`torch.addr <https://pytorch.org/docs/stable/generated/torch.addr>`_ is implemented by function :func:`addr`, which adds input ``x`` to the outer product of inputs ``y`` and ``z``.

.. math::
    \text{out} = \beta\ \text{input} + \alpha\ (\text{vec1} \otimes \text{vec2})

If :math:`y` is a vector of size :math:`n` and :math:`z` is a vector of size :math:`m`, then :math:`x` must be
`broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ with a 
:math:`(n \times m)` matrix and the result will be a matrix of size :math:`(n \times m)`.

.. function:: addr(x;y;z;beta;alpha) ->  sum of matrix-matrix multiplication with additional input
.. function:: addr(x;y;z;beta;alpha;output) -> null

   | Allowable argument combinations:

    - ``addr(x;y;z)``
    - ``addr(x;y;z;beta)``
    - ``addr(x;y;z;beta;alpha)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: a k array or tensor :doc:`pointer <pointers>`
   :param array,tensor y: a k list or 1-dimensional tensor :doc:`pointer <pointers>`
   :param array,tensor z: a k list or 1-dimensional tensor :doc:`pointer <pointers>`
   :param number beta: a numeric scalar, must be integer type if inputs are integral, else double, used as multiplier for ``x``, default=1
   :param number alpha: a numeric scalar, must be integer type if inputs are integral, else double, used as multiple for outer product of ``y`` and ``z``, default=1
   :param tensor output: an optional :doc:`pointer <pointers>` to a previously allocated tensor to be used for output
   :return: Returns matrix if all inputs given as k arrays, else tensor. If output tensor given, result is written to the given tensor and null return.

::

   q)x:2 3#til 6
   q)y:1 2
   q)z:3 4 5

   q)outer(y;z)
   3 4 5 
   6 8 10

   q)x+outer(y;z)
   3 5  7 
   9 12 15

   q)addr(x;y;z)
   3 5  7 
   9 12 15

   q)x+y*\:z
   3 5  7 
   9 12 15

   q)addr(x;y;z;100)
   3   104 205
   306 408 510


baddbmm
^^^^^^^
`torch.baddbmm <https://pytorch.org/docs/stable/generated/torch.baddbmm.html>`_ is implemented by function :func:`baddbmm`, which calculates a set of matrix-matrix products between 3-dimensional inputs ``y`` and ``z`` and adds the result to input ``x``.

.. math::
    \beta\ \text{x}_i + \alpha\ (\text{y}_i \mathbin{@} \text{z}_i)

If :math:`y` is :math:`(b \times n \times m)` and :math:`z` is
:math:`(b \times m \times p)`, then :math:`x` must be
`broadcastable <https://pytorch.org/docs/stable/notes/broadcasting.html>`_ 
with a :math:`(b \times n \times p)` array/tensor  and the result will be a :math:`(b \times n \times p)` array/tensor.

.. function:: baddbmm(x;y;z;beta;alpha) -> sum of batch of matrix-matrix multiplications
.. function:: baddbmm(x;y;z;beta;alpha;output) -> null

   | Allowable argument combinations:

    - ``baddbmm(x;y;z)``
    - ``baddbmm(x;y;z;beta)``
    - ``baddbmm(x;y;z;beta;alpha)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: a k array or tensor :doc:`pointer <pointers>`
   :param array,tensor y: a 3-dimensional array or tensor :doc:`pointer <pointers>` with batches of matrices
   :param array,tensor z: a 3-dimensional array or tensor :doc:`pointer <pointers>` with the same number of matrices as ``y``
   :param number beta: a numeric scalar, must be integer type if inputs are integral, else double, used as multiplier for ``x``, default=1
   :param number alpha: a numeric scalar, must be integer type if inputs are integral, else double, used as multiple for the matrix products, default=1
   :param tensor output: an optional :doc:`pointer <pointers>` to a previously allocated tensor to be used for output
   :return: If all inputs given as k arrays, returns a 3-dimensional k array else a tensor. If output tensor given, result is written to the given tensor and null return.

::

   q)seed 123
   q)x:tensor(`randn; 7 3 5; `double)
   q)y:tensor(`randn; 7 3 2; `double)
   q)z:tensor(`randn; 7 2 5; `double)

   q)r:baddbmm(x;y;z)
   q)size r
   7 3 5

   q)allclose(r; tensor[x]+tensor[y] mmu tensor z)
   1b

   q)baddbmm(x;y;z;0;r) /beta=0, output to r
   q)allclose(r; tensor[y]mmu tensor z)
   1b
