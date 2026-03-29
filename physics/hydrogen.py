import numpy as np
import scipy.special as sp
import math

def radial_function(r, n, l, Z=1):
    """
    Computes the radial part of the hydrogen-like wavefunction R_nl(r).
    
    Args:
        r:  Radial distance (numpy array or float).
        n:  Principal quantum number (n >= 1).
        l:  Azimuthal quantum number (0 <= l < n).
        Z:  Atomic number.
        
    Returns:
        Radial wavefunction values R_nl(r).
    """
    # Bohr radius a0 can be considered 1 for atomic units
    a0 = 1.0 
    rho = 2 * Z * r / (n * a0)
    
    # Normalization constant
    prefactor = np.sqrt((2 * Z / (n * a0))**3 *
                        math.factorial(n - l - 1) / 
                        (2 * n * math.factorial(n + l)))
    
    # Associated Laguerre polynomial L_{n-l-1}^{2l+1}(rho)
    laguerre = sp.genlaguerre(n - l - 1, 2 * l + 1)
    
    return prefactor * np.exp(-rho / 2) * (rho**l) * laguerre(rho)


def angular_function(theta, phi, l, m):
    """
    Computes the angular part of the wavefunction (real spherical harmonics).
    Fully compatible replacement for scipy sph_harm.
    """
    from scipy.special import lpmv
    
    m_abs = abs(m)
    
    # Associated Legendre polynomial
    P_lm = lpmv(m_abs, l, np.cos(theta))

    # Normalization constant
    K = np.sqrt(
        ((2*l + 1)/(4*np.pi)) *
        math.factorial(l - m_abs) /
        math.factorial(l + m_abs)
    )

    # Real spherical harmonics (chemistry standard)
    if m > 0:
        return np.sqrt(2) * K * P_lm * np.cos(m * phi)
    elif m < 0:
        return np.sqrt(2) * K * P_lm * np.sin(m_abs * phi)
    else:
        return K * P_lm


def hydrogen_wavefunction(r, theta, phi, n, l, m, Z=1):
    """
    Computes the full hydrogen-like wavefunction psi(r, theta, phi).
    """
    if not (n >= 1):
        raise ValueError("n must be >= 1")
    if not (0 <= l < n):
        raise ValueError("l must be between 0 and n-1")
    if not (-l <= m <= l):
        raise ValueError("m must be between -l and l")
        
    R = radial_function(r, n, l, Z)
    Y = angular_function(theta, phi, l, m)
    
    return R * Y


def radial_probability(r, n, l, Z=1):
    """
    Computes the radial probability density P(r) = r^2 |R_nl(r)|^2.
    """
    R = radial_function(r, n, l, Z)
    return (r**2) * (np.abs(R)**2)

def orbital_name(n, l):
    """Returns the spectroscopic name of the orbital (e.g. '1s', '2p')."""
    orbitals = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k']
    if l < len(orbitals):
        return f"{n}{orbitals[l]}"
    return f"{n}(l={l})"
