# -*- coding: utf-8 -*-
"""
@author: Phil Reinhold

Duplication of the vector fitting algorithm in python 
(http://www.sintef.no/Projectweb/VECTFIT/)

All credit goes to Bjorn Gustavsen for his MATLAB implementation, 
and the following papers:

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
     
Version 2 is a modification mainly of naming, code organization
and documentation by Pedro H. N. Vieira.

A warning about Ill conditioning of the problem may arise. To ignore
it in your code use

```
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fitted = vector_fitting(f,s)
```
"""
import numpy as np
import matplotlib.pyplot as plt
import warnings

def rational_model(s, poles, residues, d, h):
    """
    Complex rational function.
    
    Parameters
    ----------
    s : array of complex frequencies.
    poles : array of the pn
    residues : array of the rn
    d : real, offset
    h : real, slope
    
    Returns
    -------
     N
    ----
    \       rn
     \   ------- + d + s*h
     /    s - pn
    /
    ----
    n=1
    """
    f = lambda s: (residues/(s - poles)).sum() + d + s*h
    y = np.vectorize(f)
    return y(s)

def flag_poles(poles, Ns):
    """
    Identifies a given pole:
        0 : real
        1 : complex
        2 : complex.conjugate()
        
    Parameters
    ----------
    poles : initial poles guess
        note: All complex poles must come in sequential complex
        conjugate pairs
    Ns : number of samples being used (s.size)
    
    Returns
    -------
    cindex : identifying array
    """
    N = len(poles)
    cindex = np.zeros(N)
    for i, p in enumerate(poles):
        if p.imag != 0:
            if i == 0 or cindex[i-1] != 1:
                assert poles[i].conjugate() == poles[i+1], (
                        "Complex poles"" must come in conjugate "
                        +"pairs: %s, %s" % (poles[i], poles[i+1]))
                cindex[i] = 1
            else:
                cindex[i] = 2
                
    return cindex

def residues_equation(f, s, poles, cindex, sigma_residues=True):
    """
    Builds the first linear equation to solve. See Appendix A.
    
    Parameters
    ----------
    f : array of the complex data to fit
    s : complex sampling points of f
    poles : initial poles guess
        note: All complex poles must come in sequential complex
        conjugate pairs
    cindex : identifying array of the poles (real or complex)
    f_residues : bool, default=True
        signals if the residues of sigma (True) or f (False) are being
        calculated. The equation is a bit different in each case.
    Returns
    -------
    A, b : of the equation Ax = b
    """
    Ns = len(s)
    N = len(poles)
    A = np.zeros((Ns, 2*N+2), dtype=np.complex64)
    for i, p in enumerate(poles):
        if cindex[i] == 0:
            A[:, i] = 1/(s - p)
        elif cindex[i] == 1:
            A[:, i] = 1/(s - p) + 1/(s - p.conjugate())
        elif cindex[i] == 2:
            A[:, i] = 1j/(s - p) - 1j/(s - p.conjugate())
        else:
            raise RuntimeError("cindex[%s] = %s" % (i, cindex[i]))
        
        if sigma_residues:
            A[:, N+2+i] = -A[:, i]*f

    A[:, N] = 1
    A[:, N+1] = s

    b = f
    A = np.vstack((A.real, A.imag))
    b = np.concatenate((b.real, b.imag))
    cA = np.linalg.cond(A)
    if cA > 1e13:
        message = ('Ill Conditioned Matrix. Cond(A) = ' + str(cA) 
                    + ' . Consider scaling the problem down.')
        warnings.warn(message, UserWarning)
    return A, b

def fitting_poles(f, s, poles):
    """
    Calculates the poles of the fitting function.
    
    Parameters
    ----------
    f : array of the complex data to fit
    s : complex sampling points of f
    poles : initial poles guess
        note: All complex poles must come in sequential complex
        conjugate pairs
    
    Returns
    -------
    new_poles : adjusted poles
    """
    N = len(poles)
    Ns = len(s)
    cindex = flag_poles(poles, Ns)

    # calculates the residues of sigma
    A, b = residues_equation(f, s, poles, cindex)
    # Solve Ax == b using pseudo-inverse
    x, residuals, rnk, s = np.linalg.lstsq(A, b, rcond=-1)

    # We only want the "tilde" part in (A.4)
    x = x[-N:]

    # Calculation of zeros of sigma, which are equal to the poles
    # of the fitting function: Appendix B
    A = np.diag(poles)
    b = np.ones(N)
    c = x
    for i, (ci, p) in enumerate(zip(cindex, poles)):
        if ci == 1:
            x, y = p.real, p.imag
            A[i, i] = A[i+1, i+1] = x
            A[i, i+1] = -y
            A[i+1, i] = y
            b[i] = 2
            b[i+1] = 0
            #cv = c[i]
            #c[i,i+1] = real(cv), imag(cv)

    H = A - np.outer(b, c)
    H = H.real
    eig = np.linalg.eigvals(H)
    new_poles = np.sort(eig)
    unstable = new_poles.real > 0
    new_poles[unstable] -= 2*new_poles.real[unstable]
    return new_poles

def fitting_residues(f, s, poles):
    """
    Calculates the poles of the fitting function.
    
    Parameters
    ----------
    f : array of the complex data to fit
    s : complex sampling points of f
    poles : calculated poles (by fitting _poles)
    
    Returns
    -------
    residues : adjusted residues
    d : adjusted offset
    h : adjusted slope
    """
    N = len(poles)
    Ns = len(s)
    cindex = flag_poles(poles, Ns)

    # calculates the residues of sigma
    A, b = residues_equation(f, s, poles, cindex, False)
    # Solve Ax == b using pseudo-inverse
    x, residuals, rnk, s = np.linalg.lstsq(A, b, rcond=-1)

    # Recover complex values
    x = np.complex64(x)
    for i, ci in enumerate(cindex):
       if ci == 1:
           r1, r2 = x[i:i+2]
           x[i] = r1 - 1j*r2
           x[i+1] = r1 + 1j*r2

    residues = x[:N]
    d = x[N].real
    h = x[N+1].real
    return residues, d, h

def vector_fitting(f, s, poles_pairs=10, loss_ratio=0.01, n_iter=3,
                   initial_poles=None):
    """
    Makes the vector fitting of a complex function.
    
    Parameters
    ----------
    f : array of the complex data to fit
    s : complex sampling points of f
    poles_pairs : optional int, default=10
        number of conjugate pairs of the fitting function.
        Only used if initial_poles=None
    loss_ratio : optional float, default=0.01
        The initial poles guess, if not given, are estimated as
        w*(-loss_ratio + 1j)
    n_iter : optional int, default=3
        number of iterations to do when calculating the poles, i.e.,
        consecutive pole fitting
    initial_poles : optional array, default=None
        The initial pole guess
    
    Returns
    -------
    poles : adjusted poles
    residues : adjusted residues
    d : adjusted offset
    h : adjusted slope
    """
    w = s.imag
    if initial_poles == None:
        beta = np.linspace(w[0], w[-1], poles_pairs+2)[1:-1]
        initial_poles = np.array([])
        p = np.array([[-loss_ratio + 1j], [-loss_ratio - 1j]])
        for b in beta:
            initial_poles = np.append(initial_poles, p*b)
        
    poles = initial_poles
    for _ in range(n_iter):
        poles = fitting_poles(f, s, poles)
        
    residues, d, h = fitting_residues(f, s, poles)
    return poles,residues, d, h
    
def print_params(poles, residues, d, h):
    cfmt = "{0.real:g} + {0.imag:g}j"
    print("poles: " + ", ".join(cfmt.format(p) for p in poles))
    print("residues: " + ", ".join(cfmt.format(r) for r in residues))
    print("offset: {:g}".format(d))
    print("slope: {:g}".format(h))

def vectfit_auto_rescale(f, s, **kwargs):
    s_scale = abs(s[-1])
    f_scale = abs(f[-1])
    print('SCALED')
    poles_s, residues_s, d_s, h_s = vector_fitting(f / f_scale, s / s_scale, **kwargs)
    #rescaling :
    poles    = poles_s * s_scale
    residues = residues_s * f_scale * s_scale
    d = d_s * f_scale
    h = h_s * f_scale / s_scale
    print('UNSCALED')
    print_params(poles, residues, d, h)
    return poles, residues, d, h

if __name__ == '__main__':
    test_s = 1j*np.linspace(1, 1e5, 800)
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
        -5000+92000j, -5000-92000j
    ]
    test_d = .2
    test_h = 2e-5

    test_f = sum(c/(test_s - a) for c, a in zip(test_residues, test_poles))
    test_f +=  test_h*test_s #+ test_d
    #vectfit_auto(test_f, test_s)

    poles, residues, d, h = vectfit_auto_rescale(test_f, test_s)
    fitted = rational_model(test_s, poles, residues, d, h)
    plt.figure()
    plt.plot(test_s.imag, test_f.real)
    plt.plot(test_s.imag, test_f.imag)
    plt.plot(test_s.imag, fitted.real, 'r--')
    plt.plot(test_s.imag, fitted.imag, 'b--')
    plt.show()
