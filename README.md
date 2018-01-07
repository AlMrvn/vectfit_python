Duplication of the [Fast Relaxed Vector-Fitting algorithm](http://www.sintef.no/Projectweb/VECTFIT/) in python.

This version comes from the work of Phil Reinhold, and Pedro H. N. Vieira.
The changes are just notation and the extension to vectors of function
in order to fit several function with the same set of poles. This possibility
was already discussed in Bjorn Gustavsen papers:

 [1] B. Gustavsen and A. Semlyen, "Rational approximation of frequency
     domain responses by Vector Fitting", IEEE Trans. Power Delivery,
     vol. 14, no. 3, pp. 1052-1061, July 1999.

 [2] B. Gustavsen, "Improving the pole relocating properties of vector
     fitting", IEEE Trans. Power Delivery, vol. 21, no. 3, pp. 1587-1592,
     July 2006.

 [3] D. Deschrijver, M. Mrozowski, T. Dhaene, and D. De Zutter,
     "Macromodeling of Multiport Systems Using a Fast Implementation of
     the Vector Fitting Method", IEEE Microwave and Wireless Components
     Letters, vol. 18, no. 6, pp. 383-385, June 2008.

Example of use:
```python
import vectfit
import numpy as np
import matplotlib.pyplot as plt

# Create some test data using known poles and residues
# Substitute your source of data as needed

# Note our independent variable lies along the imaginary axis
test_s = 1j*np.linspace(1, 1e5, 800)

# the poles come as complex conjugate. There is need for
test_poles = [
    -4500,
    -41000,
    -100+5000j, -100-5000j,
    -120+15000j, -120-15000j,
    -3000+35000j, -3000-35000j,
    -200+45000j, -200-45000j,
    -1500+45000j, -1500-45000j,
    -500+70000j, -500-70000j,
    -1000+73000j, -1000-73000j,
    -2000+90000j, -2000-90000j,
]
# there is (#poles)x(#function) residue:
test_residues = [
    -3000,
    -83000,
    -5+7000j, -5-7000j,
    -20+18000j, -20-18000j,
    6000+45000j, 6000-45000j,
    40+60000j, 40-60000j,
    90+10000j, 90-10000j,
    50000+80000j, 50000-80000j,
    1000+45000j, 1000-45000j,
    0, 0
]
test_residues_2 = [
    0,
    -83000,
    0, 0,
    -20+18000j, -20-18000j,
    0, 0,
    0, 0,
    0, 0,
    50000+80000j, 50000-80000j,
    0, 0,
    -5000+92000j, -5000-92000j
]

test_residues = np.vstack([np.array(test_residues),
                           np.array(test_residues_2)])
# d == offset, h == slope
test_d = [0., 0.]
test_h = [0., 0.]

# creating the atual data
test_f = vectfit.model(test_s, test_poles, test_residues, test_d, test_h)

# Run algorithm, results hopefully match the known model parameters
poles, residues, d, h = vecfit.vector_fitting(test_f, test_s, asymptote=None)
fitted = vectfit.rational_model(test_s, poles, residues, d, h)
plt.figure()
plt.plot(test_s.imag/1e3, test_f.real)
plt.plot(test_s.imag/1e3, test_f.imag)
plt.plot(test_s.imag/1e3, fitted.real, 'r--')
plt.plot(test_s.imag/1e3, fitted.imag, 'b--')
plt.show()

```
