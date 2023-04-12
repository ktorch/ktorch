Spectral operations
===================

1-d Fourier
***********

 - `torch.fft.fft <https://pytorch.org/docs/stable/generated/torch.fft.fft.html>`_ -  discrete transform, :func:`fft`
 - `torch.fft.ifft <https://pytorch.org/docs/stable/generated/torch.fft.ifft.html>`_ -  inverse discrete transform, :func:`ifft`
 - `torch.fft.rfft <https://pytorch.org/docs/stable/generated/torch.fft.rfft.html>`_ -  transform of real-valued input, :func:`rfft`
 - `torch.fft.irfft <https://pytorch.org/docs/stable/generated/torch.fft.irfft.html>`_ -  inverse of transform of real input, :func:`irfft`
 - `torch.fft.hfft <https://pytorch.org/docs/stable/generated/torch.fft.hfft.html>`_ -  discrete transform of a Hermitian signal, :func:`hfft`
 - `torch.fft.ihfft <https://pytorch.org/docs/stable/generated/torch.fft.ihfft.html>`_ -  inverse of :func:`hfft`, implemented as :func:`ihfft`

fft
^^^

`torch.fft.fft <https://pytorch.org/docs/stable/generated/torch.fft.fft.html>`_,  the discrete  Fourier transform, is implemented as :func:`fft`.

.. function:: fft(x;size;dim;norm) -> 1-dimensional Fast Fourier transform
.. function:: fft(x;size;dim;norm;output) -> null
   :noindex:


   | Allowable argument combinations:

    - ``fft(x)``
    - ``fft(x;size)``
    - ``fft(x;size;dim)``
    - ``fft(x;size;dim;norm)``
    - ``fft(x;norm)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param long size: the optional signal length
   :param long dim: the optional dimension along which to evaluate, defaults to final dimension.
   :param symbol norm: default ```backward`` to normalize by 1/n, ```forward`` for no normalization, ```ortho`` to normalize by ``1/sqrt(n)``
   :param tensor output: an optional `complex <complex>` tensor to use for function output
   :return: Returns an array if a k array used as input, else a tensor, with real and imaginary parts along 1st dimension. If output tensor supplied, writes output to tensor and returns null.

::

   q)x:0 1 2 3e
   q)fft x
   6 -2 -2 -2
   0 2  0  -2

   q)fft(x;3)
   3 -1.5      -1.5      
   0 0.8660254 -0.8660254

   q)fft(x;4;-1;`forward)
   1.5 -0.5 -0.5 -0.5
   0   0.5  0    -0.5

   q)x:tensor x
   q)y:fft(x;4;-1;`forward)
   q)tensor y
   1.5 -0.5 -0.5 -0.5
   0   0.5  0    -0.5

   q)fft(x;`ortho;y)  /use output tensor
   q)tensor y
   3 -1 -1 -1
   0 1  0  -1

ifft
^^^^

`torch.fft.ifft <https://pytorch.org/docs/stable/generated/torch.fft.ifft.html>`_, the inverse discrete transform, is implemented with :func:`ifft`.

.. function:: ifft(x;size;dim;norm) -> inverse discrete transform
.. function:: ifft(x;size;dim;norm;output) -> null
   :noindex:

   | Same allowable argument combinations as :func:`fft`. If using k array(s) as input, a complex tensor must first be constructed from the k arrays of real and imaginary parts.

::

   q)fft 0 1 2 3 4e
   10 -2.5  -2.5   -2.5    -2.5  
   0  3.441 0.8123 -0.8123 -3.441

   q)x:tensor(`complex;fft 0 1 2 3 4e)
   q)tensor x
   10 -2.5  -2.5   -2.5    -2.5  
   0  3.441 0.8123 -0.8123 -3.441

   q)y:ifft x

   q)tensor y
   0 1 2 3 4
   0 0 0 0 0

.. note::

   Complex inputs built from k arrays are sensitive to the global `complexfirst <complex-first>` setting: by default real and imaginary parts are along the first dimension.

::

   q)y:ifft x:tensor(`complex; (6 -2 -2 -2e; 0 2  0  -2e))
   q)tensor y
   0 1 2 3
   0 0 0 0

   q)setting`complexfirst
   1b

   q)setting`complexfirst,0b

   q)use[y]ifft x
   q)tensor y
   0 0
   1 0
   2 0
   3 0


rfft
^^^^

`torch.fft.rfft <https://pytorch.org/docs/stable/generated/torch.fft.rfft.html>`_, the transform of real-valued input, is implemented as :func:`rfft`.

.. function:: rfft(x;size;dim;norm) -> real transform
.. function:: rfft(x;size;dim;norm;output) -> null
   :noindex:

   | Same allowable argument combinations as :func:`fft`

::

   q)rfft 0 1 2 3 4e
   10 -2.5     -2.5     
   0  3.440955 0.8122992

   q)rfft(0 1 2 3 4e;4)
   6 -2 -2
   0 2  0 

irfft
^^^^^

`torch.fft.irfft <https://pytorch.org/docs/stable/generated/torch.fft.irfft.html>`_, the inverse of the transform of real input, is implemented by function :func:`irfft`

.. function:: irfft(x;size;dim;norm) -> inverse of real transform
.. function:: irfft(x;size;dim;norm;output) -> null
   :noindex:

   | Same allowable argument combinations as :func:`fft`


::

   q)rfft 0 1 2 3e
   6 -2 -2
   0 2  0 

   q)x:tensor(`complex; rfft 0 1 2 3e)
   q)y:irfft x
   q)tensor y
   0 1 2 3e

   q)n:5  /need signal length for odd sizes
   q)rfft(0 1 2 3 4e; n)
   10 -2.5  -2.5  
   0  3.441 0.8123

   q)use[x]tensor(`complex; rfft(0 1 2 3 4e; n))
   q)use[y]irfft(x; n)
   q)tensor y
   0 1 2 3 4e

hfft
^^^^

`torch.fft.hfft <https://pytorch.org/docs/stable/generated/torch.fft.hfft.html>`_, the discrete transform of a Hermitian signal, is implemented as :func:`hfft`.

.. function:: hfft(x;size;dim;norm) -> discrete transform of Hermitian signal
.. function:: hfft(x;size;dim;norm;output) -> null
   :noindex:

   | Same allowable argument combinations as :func:`fft`

::

   q)x:tensor(`linspace;0;1;5)
   q)tensor x
   0 0.25 0.5 0.75 1e

   q)y:ifft x
   q)tensor y
   0.5 -0.125 -0.125   -0.125  -0.125
   -0  -0.172 -0.04061 0.04061 0.172 

   q)z:hfft(y;5)
   q)tensor z
   0 0.25 0.5 0.75 1e

ihfft
^^^^^

`torch.fft.ihfft <https://pytorch.org/docs/stable/generated/torch.fft.ihfft.html>`_ -  inverse of :func:`hfft`, implemented as :func:`ihfft`

.. function:: ihfft(x;size;dim;norm) -> inverse of transform of Hermitian
.. function:: ihfft(x;size;dim;norm;output) -> null
   :noindex:

   | Same allowable argument combinations as :func:`fft`

::

   q)ihfft til 5
   2  -0.5    -0.5   
   -0 -0.6882 -0.1625

   q)ifft til 5
   2  -0.5    -0.5    -0.5   -0.5  
   -0 -0.6882 -0.1625 0.1625 0.6882

2-d Fourier
***********

The 2-dimensional Fourier transforms are similar to the N-dimensional variants, except the default dimensions are set to the final two dimensions of the given input. The 2-d routines are designed to match NumPy's 2-d implementations (see `pull request <https://github.com/pytorch/pytorch/pull/45164>`_).

 - `torch.fft.fft2 <https://pytorch.org/docs/stable/generated/torch.fft.fft2.html>`_ -  2-d discrete transform, :func:`fft2`
 - `torch.fft.ifft2 <https://pytorch.org/docs/stable/generated/torch.fft.ifft2.html>`_ -  2-d inverse discrete transform, :func:`ifft2`
 - `torch.fft.rfft2 <https://pytorch.org/docs/stable/generated/torch.fft.rfft2.html>`_ -  2-d discrete transform of real input, :func:`rfft2`
 - `torch.fft.irfft2 <https://pytorch.org/docs/stable/generated/torch.fft.irfft2.html>`_ -  2-d inverse of transform of real input, :func:`irfft2`
 - `torch.fft.hfft2 <https://pytorch.org/docs/stable/generated/torch.fft.hfft2.html>`_ -  2-d discrete transform of a Hermitian signal, :func:`hfft2`
 - `torch.fft.ihfft2 <https://pytorch.org/docs/stable/generated/torch.fft.ihfft2.html>`_ -  2-d inverse of :func:`hfft2`, implemented as :func:`ihfft2`


fft2
^^^^

`torch.fft.fft2 <https://pytorch.org/docs/stable/generated/torch.fft.fft2.html>`_,  the 2-d discrete Fourier transform, is implemented as :func:`fft2`.

.. function:: fft2(x;size;dim;norm) -> 1-dimensional Fast Fourier transform
.. function:: fft2(x;size;dim;norm;output) -> null
   :noindex:


   | Allowable argument combinations:

    - ``fft2(x)``
    - ``fft2(x;size)``
    - ``fft2(x;size;dim)``
    - ``fft2(x;size;dim;norm)``
    - ``fft2(x;norm)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param longs size: the optional signal length in the transformed dimensions, ``dim[i]`` will be zero-padded or trimmed to given length before computing the transform. A length of -1 indicates no padding for that dimension. Default sizes set to input sizes.
   :param longs dim: the optional dimension(s) to be transformed, default is final 2 dimensions.
   :param symbol norm: default ```backward`` to normalize by 1/n, ```forward`` for no normalization, ```ortho`` to normalize by ``1/sqrt(n)``
   :param tensor output: an optional `complex <complex>` tensor to use for function output
   :return: Returns an array if a k array used as input, else a tensor, with real and imaginary parts along 1st dimension. If output tensor supplied, writes output to tensor and returns null.

::

   q)x:0 1 2 3e
   q)fft x
   6 -2 -2 -2
   0 2  0  -2

   q)first fft2((x;x);4;1)
   6 -2 -2 -2
   6 -2 -2 -2

   q)last fft2((x;x);4;1)
   0 2 0 -2
   0 2 0 -2
 
ifft2
^^^^^

`torch.fft.ifft2 <https://pytorch.org/docs/stable/generated/torch.fft.ifft2.html>`_, the 2-d inverse discrete transform, is implemented with :func:`ifft2`.

.. function:: ifft2(x;size;dim;norm) -> inverse discrete transform
.. function:: ifft2(x;size;dim;norm;output) -> null
   :noindex:

   | Same allowable argument combinations as :func:`fft2`. If using k array(s) as input, a complex tensor must first be constructed from the k arrays of real and imaginary parts.

::

   q)x:tensor(`randn;5 5;`cdouble)
   q)y:ifft2 x

   q)y0:ifft(x;5;0)   / two equivalent 1-dimensional calls
   q)y1:ifft(y0;5;1)

   q)allclose(y;y1)
   1b

rfft2
^^^^^

`torch.fft.rfft2 <https://pytorch.org/docs/stable/generated/torch.fft.rfft2.html>`_, the 2-d transform of real-valued input, is implemented as :func:`rfft2`.

.. function:: rfft2(x;size;dim;norm) -> real transform
.. function:: rfft2(x;size;dim;norm;output) -> null
   :noindex:

   | Same allowable argument combinations as :func:`fft2`

::

   q)x:tensor(`randn;5 5)
   q)y:rfft2 x

   q)y0:rfft(x;5;1)  / combination of 1-d calls to rfft & fft
   q)y1:fft(y0;5;0)

   q)allclose(y;y1)
   1b

irfft2
^^^^^^

`torch.fft.irfft2 <https://pytorch.org/docs/stable/generated/torch.fft.irfft2.html>`_, the 2-d inverse of the transform of real input, is implemented by function :func:`irfft2`

.. function:: irfft2(x;size;dim;norm) -> inverse of real transform
.. function:: irfft2(x;size;dim;norm;output) -> null
   :noindex:

   | Same allowable argument combinations as :func:`fft2`


::

   q)x:tensor(`randn;10 9)
   q)y:rfft2 x

   q)r:irfft2(y;10 9)  / size needed if original dim(s) odd

   q)allclose(x;r)
   1b


hfft2
^^^^^

`torch.fft.hfft2 <https://pytorch.org/docs/stable/generated/torch.fft.hfft2.html>`_, the discrete transform of a Hermitian signal, is implemented as :func:`hfft2`.

.. function:: hfft2(x;size;dim;norm) -> discrete transform of Hermitian signal
.. function:: hfft2(x;size;dim;norm;output) -> null
   :noindex:

   | Same allowable argument combinations as :func:`fft2`

::

   q)x:tensor(`randn;10 9)  /real, frequency-space signal
   q)y:ihfft2 x             /Hermitian-symmetric time-domain signal
   q)z:hfft2(y;size x)      /roundtrip back to original signal

   q)allclose(x;z)
   1b


ihfft2
^^^^^^

`torch.fft.ihfft2 <https://pytorch.org/docs/stable/generated/torch.fft.ihfft2.html>`_ -  inverse of :func:`hfft2`, implemented as :func:`ihfft2`

.. function:: ihfft2(x;size;dim;norm) -> inverse of transform of Hermitian
.. function:: ihfft2(x;size;dim;norm;output) -> null
   :noindex:

   | Same allowable argument combinations as :func:`fft2`

::

   q)x:tensor(`randn;10 9)  /real, frequency-space signal
   q)y:ihfft2 x             /Hermitian-symmetric time-domain signal
   q)z:hfft2(y;size x)      /roundtrip back to original signal

   q)allclose(x;z)
   1b

N-dimensional Fourier
*********************

 - `torch.fft.fftn <https://pytorch.org/docs/stable/generated/torch.fft.fftn.html>`_ -  N-dim discrete transform, :func:`fftn`
 - `torch.fft.ifftn <https://pytorch.org/docs/stable/generated/torch.fft.ifftn.html>`_ -  N-dim inverse discrete transform, :func:`ifftn`
 - `torch.fft.rfftn <https://pytorch.org/docs/stable/generated/torch.fft.rfftn.html>`_ -  N-dim discrete transform of real input, :func:`rfftn`
 - `torch.fft.irfftn <https://pytorch.org/docs/stable/generated/torch.fft.irfftn.html>`_ -  N-dim inverse of transform of real input, :func:`irfftn`
 - `torch.fft.hfftn <https://pytorch.org/docs/stable/generated/torch.fft.hfftn.html>`_ -  N-dim discrete transform of a Hermitian signal, :func:`hfftn`
 - `torch.fft.ihfftn <https://pytorch.org/docs/stable/generated/torch.fft.ihfftn.html>`_ -  N-d inverse of :func:`hfftn`, implemented as :func:`ihfftn`

fftn
^^^^

`torch.fft.fftn <https://pytorch.org/docs/stable/generated/torch.fft.fftn.html>`_,  the N-dim discrete Fourier transform, is implemented as :func:`fftn`.

.. function:: fftn(x;size;dim;norm) -> 1-dimensional Fast Fourier transform
.. function:: fftn(x;size;dim;norm;output) -> null
   :noindex:


   | Allowable argument combinations:

    - ``fftn(x)``
    - ``fftn(x;size)``
    - ``fftn(x;size;dim)``
    - ``fftn(x;size;dim;norm)``
    - ``fftn(x;norm)``
    - any of the above combinations followed by a trailing output tensor

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param longs size: the optional signal length in the transformed dimensions, ``dim[i]`` will be zero-padded or trimmed to given length before computing the transform. A length of -1 indicates no padding for that dimension. By default, size is set to input sizes.
   :param longs dim: the optional dimension(s) to be transformed, default is all dimensions or the last dimensions corresponding to the sizes given.
   :param symbol norm: default ```backward`` to normalize by 1/n, ```forward`` for no normalization, ```ortho`` to normalize by ``1/sqrt(n)``
   :param tensor output: an optional `complex <complex>` tensor to use for function output
   :return: Returns an array if a k array used as input, else a tensor, with real and imaginary parts along 1st dimension. If output tensor supplied, writes output to tensor and returns null.

::

   q)x:tensor(`randn;10 10;`cdouble)
   q)y:fftn x
       
   q)y0:fft(x;10;0)  /compare to two 1-dim calls
   q)y1:fft(y0;10;1)

   q)allclose(y;y1)
   1b

 
ifftn
^^^^^

`torch.fft.ifftn <https://pytorch.org/docs/stable/generated/torch.fft.ifftn.html>`_, the N-dim inverse discrete transform, is implemented with :func:`ifftn`.

.. function:: ifftn(x;size;dim;norm) -> inverse discrete transform
.. function:: ifftn(x;size;dim;norm;output) -> null
   :noindex:

   | Same allowable argument combinations as :func:`fftn`. If using k array(s) as input, a complex tensor must first be constructed from the k arrays of real and imaginary parts.

::

   q)x:tensor(`randn;10 10;`cdouble)
   q)y:ifftn x

   q)y0:ifft(x;10;0)  /compare to two 1-dim calls
   q)y1:ifft(y0;10;1)

   q)allclose(y;y1)
   1b


rfftn
^^^^^

`torch.fft.rfftn <https://pytorch.org/docs/stable/generated/torch.fft.rfftn.html>`_, the N-dim transform of real-valued input, is implemented as :func:`rfftn`.

.. function:: rfftn(x;size;dim;norm) -> real transform
.. function:: rfftn(x;size;dim;norm;output) -> null
   :noindex:

   | Same allowable argument combinations as :func:`fftn`

::

   q)x:tensor(`rand;10 10)
   q)y:rfftn x
   q)size y
   10 6

   q)f:fftn x  /full output from fftn()
   q)size f
   10 10

   q)use[f]index(f;1;til 6)
   q)allclose(y;f)
   1b

irfftn
^^^^^^

`torch.fft.irfftn <https://pytorch.org/docs/stable/generated/torch.fft.irfftn.html>`_, the N-dim inverse of the transform of real input, is implemented by function :func:`irfftn`

.. function:: irfftn(x;size;dim;norm) -> inverse of real transform
.. function:: irfftn(x;size;dim;norm;output) -> null
   :noindex:

   | Same allowable argument combinations as :func:`fftn`

::

   q)x:tensor(`rand;10 9;`double)
   q)y:rfftn x
   q)z:irfftn y

   q)size z  /can't match size of original x with old dim(s)
   10 8

   q)use[z]irfftn(y;size x)  /specify size explicitly
   q)size z
   10 9

   q)allclose(x;z)
   1b


hfftn
^^^^^

`torch.fft.hfftn <https://pytorch.org/docs/stable/generated/torch.fft.hfftn.html>`_, the discrete transform of a Hermitian signal, is implemented as :func:`hfftn`.

.. function:: hfftn(x;size;dim;norm) -> discrete transform of Hermitian signal
.. function:: hfftn(x;size;dim;norm;output) -> null
   :noindex:

   | Same allowable argument combinations as :func:`fftn`

::

   q)x:tensor(`rand;10 9)
   q)y:ihfftn(x;size x)    /inverse
   q)z:hfftn(y;size x)     /get back x

   q)allclose(x;z)
   1b


ihfftn
^^^^^^

`torch.fft.ihfftn <https://pytorch.org/docs/stable/generated/torch.fft.ihfftn.html>`_ -  inverse of :func:`hfftn`, implemented as :func:`ihfftn`

.. function:: ihfftn(x;size;dim;norm) -> inverse of transform of Hermitian
.. function:: ihfftn(x;size;dim;norm;output) -> null
   :noindex:

   | Same allowable argument combinations as :func:`fftn`

::

   q)x:tensor(`rand;10 10;`double)
   q)y:ihfftn x
   q)size y
   10 6

   q)z:ifftn x /full output
   q)size z
   10 10

   q)use[z]index(z;-1;til 6)
   q)allclose(y;z)
   1b


Helper functions
****************

- `torch.fft.fftfreq <https://pytorch.org/docs/stable/generated/torch.fft.fftfreq.html>`_ - discrete sample frequency for signal of given size, :func:`fftfreq`
- `torch.fft.rfftfreq <https://pytorch.org/docs/stable/generated/torch.fft.rfftfreq.html>`_ - sample frequencies for :func:`rfft`, implemented as :func:`rfftfreq`
- `torch.fft.fftshift <https://pytorch.org/docs/stable/generated/torch.fft.fftshift.html>`_ - reorders N-dim FFT data to have negative frequence terms first, :func:`fftshift`
- `torch.fft.ifftshift <https://pytorch.org/docs/stable/generated/torch.fft.ifftshift.html>`_ - inverse of :func:`fftshift` implemented as :func:`ifftshift`

fftfreq
^^^^^^^

.. function:: fftfreq(length;scale;options) -> sample frequencies of given length
.. function:: fftfreq(length;scale;output) -> null
   :noindex:

   | Allowable argument combinations:

    - ``fftfreq(length)``
    - ``fftfreq(length;scale)``
    - ``fftfreq(length;scale;options)``
    - ``fftfreq(length;options)``
    - any of the above combinations with a trailing output tensor in place of tensor options

   :param long length: the Fourier sample frequency length
   :param double scale: the sampling length scale (spacing between samples), default=w for unit spacing.
   :param symbols options: optional tensor :ref:`attributes <tensor-attributes>`, e.g. ```cuda`double`grad``, ```float``
   :param tensor output: in place of ``options``, an output tensor :doc:`pointer <pointers>` to contain the frequencies
   :return: The discrete Fourier Transform sample frequencies for the given length, as a tensor, or, if an output tensor supplied, written to the given tensor, null return.

::

   q)r:fftfreq 5
   q)tensor r
   0 0.2 0.4 -0.4 -0.2e

   q)use[r]fftfreq(5;2;`double)
   q)tensor r
   0 0.1 0.2 -0.2 -0.1

   q)fftfreq(5;r)
   q)tensor r
   0 0.2 0.4 -0.4 -0.2

rfftfreq
^^^^^^^^

.. function:: rfftfreq(length;scale;options) -> sample frequencies of given length
.. function:: rfftfreq(length;scale;output) -> null
   :noindex:

   | Allowable argument combinations are the same as for :func:`fftfreq`

   :return: Returns Hermitian 1-sided output, so only positive frequency terms are returned.

::

   q)r:rfftfreq 5
   q)tensor r
   0 0.2 0.4e

   q)rfftfreq(5;2;r)
   q)tensor r
   0 0.1 0.2e

fftshift
^^^^^^^^

.. function:: fftshift(x;dim) -> reordered N-dim data to have negative frequency terms first

   :param array,tensor x: input array or tensor :doc:`pointer <pointers>`
   :param long dim: the optional dimension(s) along which to reorder, defaults to all dimensions.
   :return: Return shifted array if array input else return tensor for given tensor input.

::

   q)x:fftfreq 4
   q)tensor x
   0 0.25 -0.5 -0.25e

   q)fftshift tensor x
   -0.5 -0.25 0 0.25e

   q)a:fftfreq(5;1%5)
   q)tensor a
   0 1 2 -2 -1e

   q)b:add(a;0N 1#.1*tensor a)
   q)tensor b
   0    1   2   -2   -1  
   0.1  1.1 2.1 -1.9 -0.9
   0.2  1.2 2.2 -1.8 -0.8
   -0.2 0.8 1.8 -2.2 -1.2
   -0.1 0.9 1.9 -2.1 -1.1

   q)fftshift tensor b
   -2.2 -1.2 -0.2 0.8 1.8
   -2.1 -1.1 -0.1 0.9 1.9
   -2   -1   0    1   2  
   -1.9 -0.9 0.1  1.1 2.1
   -1.8 -0.8 0.2  1.2 2.2


ifftshift
^^^^^^^^^

.. function:: ifftshift(x;dim) -> reordered N-dim data to inverse ordering of :func:`fftshift`

   | Uses same parameters and syntax as :func:`fftshift`

::

   q)a:fftfreq(5;1%5)
   q)tensor a
   0 1 2 -2 -1e

   q)fftshift tensor a
   -2 -1 0 1 2e

   q)ifftshift fftshift tensor a
   0 1 2 -2 -1e

Window functions
****************

 - `torch.bartlett_window <https://pytorch.org/docs/stable/generated/torch.bartlett_window.html>`_ - implemented as :func:`bartlett`
 - `torch.blackman <https://pytorch.org/docs/stable/generated/torch.blackman_window.html>`_ - implemented as :func:`blackman`
 - `torch.hann_window <https://pytorch.org/docs/stable/generated/torch.hann_window.html>`_ - implemented as :func:`hann`
 - `torch.hamming_window <https://pytorch.org/docs/stable/generated/torch.hamming_window.html>`_ - implemented as :func:`hamming`
 - `torch.kaiser_window <https://pytorch.org/docs/stable/generated/torch.kaiser_window.html>`_ - implemented as :func:`kaiser`


bartlett
^^^^^^^^

Bartlett window function.

.. math::
    w[n] = 1 - \left| \frac{2n}{N-1} - 1 \right| = \begin{cases}
        \frac{2n}{N - 1} & \text{if } 0 \leq n \leq \frac{N - 1}{2} \\
        2 - \frac{2n}{N - 1} & \text{if } \frac{N - 1}{2} < n < N \\
    \end{cases},

where :math:`N` is the full window length.

.. function:: bartlett(length;periodic;options) -> 1-d tensor containing the window

   | Allowable argument combinations:

    - ``bartlett(length)``
    - ``bartlett(length;periodic)``
    - ``bartlett(length;periodic;options)``
    - ``bartlett(length;options)``

   :param long length: the size of the returned window
   :param bool periodic: default ``true`` to return a window to be used as a periodic function, ``false`` for a symmetric window
   :param symbols options: optional tensor :ref:`attributes <tensor-attributes>`, e.g. ```cuda`double`grad``, ```float``
   :return: A 1-d tensor of given length containing the window.

::

   q)x:bartlett 11
   q)tensor x
   0 0.1818 0.3636 0.5455 0.7273 0.9091 0.9091 0.7273 0.5455 0.3636 0.1818e

   q)x:bartlett(11;0b;`double)
   q)tensor x
   0 0.2 0.4 0.6 0.8 1 0.8 0.6 0.4 0.2 0

   q)x:bartlett 21
   q)-2("j"$20*tensor x)#'"*";

   **
   ****
   ******
   ********
   **********
   ***********
   *************
   ***************
   *****************
   *******************
   *******************
   *****************
   ***************
   *************
   ***********
   **********
   ********
   ******
   ****
   **

blackman
^^^^^^^^

Blackman window function.

.. math::
    w[n] = 0.42 - 0.5 \cos \left( \frac{2 \pi n}{N - 1} \right) + 0.08 \cos \left( \frac{4 \pi n}{N - 1} \right)

where :math:`N` is the full window length.

.. function:: blackman(length;periodic;options) -> 1-d tensor containing the window

   | Allowable argument combinations:

    - ``blackman(length)``
    - ``blackman(length;periodic)``
    - ``blackman(length;periodic;options)``
    - ``blackman(length;options)``

   :param long length: the size of the returned window
   :param bool periodic: default ``true`` to return a window to be used as a periodic function, ``false`` for a symmetric window
   :param symbols options: optional tensor :ref:`attributes <tensor-attributes>`, e.g. ```cuda`double`grad``, ```float``
   :return: A 1-d tensor of given length containing the window.

::

   q)x:blackman 21
   q)tensor x
   -2.98e-08 0.00831 0.0361 0.0905 0.179 0.304 0.459 0.63 0.793 0.92 0.991 0.991..

   q)-2("j"$20*tensor x)#'"*";
   
   
   *
   **
   ****
   ******
   *********
   *************
   ****************
   ******************
   ********************
   ********************
   ******************
   ****************
   *************
   *********
   ******
   ****
   **
   *

hann
^^^^

Hann window function.

.. math::
    w[n] = \frac{1}{2}\ \left[1 - \cos \left( \frac{2 \pi n}{N - 1} \right)\right] =
            \sin^2 \left( \frac{\pi n}{N - 1} \right),

where :math:`N` is the full window length.

.. function:: hann(length;periodic;options) -> 1-d tensor containing the window

   | Allowable argument combinations:

    - ``hann(length)``
    - ``hann(length;periodic)``
    - ``hann(length;periodic;options)``
    - ``hann(length;options)``

   :param long length: the size of the returned window
   :param bool periodic: default ``true`` to return a window to be used as a periodic function, ``false`` for a symmetric window
   :param symbols options: optional tensor :ref:`attributes <tensor-attributes>`, e.g. ```cuda`double`grad``, ```float``
   :return: A 1-d tensor of given length containing the window.

::

   q)x:hann(21;1b;`double)
   q)tensor x
   0 0.0222 0.0869 0.188 0.317 0.463 0.611 0.75 0.867 0.95 0.994 0.994 0.95 0.86..

   q)-2("j"$20*tensor x)#'"*";
   
   
   **
   ****
   ******
   *********
   ************
   ***************
   *****************
   *******************
   ********************
   ********************
   *******************
   *****************
   ***************
   ************
   *********
   ******
   ****
   **

hamming
^^^^^^^

Hamming window function.

.. math::
    w[n] = \alpha - \beta\ \cos \left( \frac{2 \pi n}{N - 1} \right),

where :math:`N` is the full window length.

.. function:: hamming(length;periodic;alpha;beta;options) -> 1-d tensor containing the window

   | Allowable argument combinations:

    - ``hamming(length)``
    - ``hamming(length;periodic)``
    - ``hamming(length;periodic;alpha)``
    - ``hamming(length;periodic;alpha;beta)``
    - any of the above with a final argument of tensor option(s)

   :param long length: the size of the returned window
   :param bool periodic: default ``true`` to return a window to be used as a periodic function, ``false`` for a symmetric window
   :param double alpha: the :math:`\alpha` in the above equation, default = 0.54
   :param double beta: the :math:`\beta` in the above equation, default = 0.46
   :param symbols options: optional tensor :ref:`attributes <tensor-attributes>`, e.g. ```cuda`double`grad``, ```float``
   :return: A 1-d tensor of given length containing the window.

::

   q)x:hamming(21;`double)
   q)y:hamming(21;1b;.54;.46;`double)
   q)equal(x;y)
   1b

   q)-2("j"$20*tensor x)#'"*";
   **
   **
   ***
   *****
   *******
   **********
   *************
   ***************
   ******************
   *******************
   ********************
   ********************
   *******************
   ******************
   ***************
   *************
   **********
   *******
   *****
   ***
   **

kaiser
^^^^^^

Computes the Kaiser window with given length and shape parameter ``beta``.

Let :math:`I_0` be the zera-oth order modified Bessel function of the first kind and
``N = L - 1`` if ``periodic`` is ``false`` and ``L`` if :attr:`periodic` is ``true``,
where ``L`` is the ``length`` parameter. This function computes:

.. math::
    out_i = I_0 \left( \beta \sqrt{1 - \left( {\frac{i - N/2}{N/2}} \right) ^2 } \right) / I_0( \beta )

Calling ``torch.kaiser_window(L, B, periodic=True)`` is equivalent to calling
``torch.kaiser_window(L + 1, B, periodic=False)[:-1])``.

.. function:: kaiser(length;periodic;alpha;beta;options) -> 1-d tensor containing the window

   | Allowable argument combinations:

    - ``kaiser(length)``
    - ``kaiser(length;periodic)``
    - ``kaiser(length;periodic;beta)``
    - any of the above with a final argument of tensor option(s)

   :param long length: the size of the returned window
   :param bool periodic: default ``true`` to return a window to be used as a periodic function, ``false`` for a symmetric window
   :param double beta: the :math:`\beta` in the above equation, the shape parameter for the window, default = 12.0
   :param symbols options: optional tensor :ref:`attributes <tensor-attributes>`, e.g. ```cuda`double`grad``, ```float``
   :return: A 1-d tensor of given length containing the window.

::

   q)x:kaiser 21

   q){r:equal(x;y); free y; r}[x]kaiser(21;`float)
   1b
   q){r:equal(x;y); free y; r}[x]kaiser(21;1b;12;`float)
   1b

   q)-2("j"$20*tensor x)#'"*";
   
   
   
   *
   **
   ****
   *******
   **********
   **************
   ******************
   ********************
   ********************
   ******************
   **************
   **********
   *******
   ****
   **
   *
   
