import numpy as np

def generate_cartesian_grid(size=10.0, resolution=80):
    """
    Generate a 3D Cartesian grid.
    
    Args:
        size (float): The physical size of the grid bounding box from -size to size.
        resolution (int): The number of points along each dimension.
        
    Returns:
        tuple: (X, Y, Z) meshgrid arrays, and a linspace 1D array.
    """
    lin = np.linspace(-size, size, resolution)
    # Using 'ij' indexing so that axes correspond exactly to x, y, z arrays natively
    X, Y, Z = np.meshgrid(lin, lin, lin, indexing='ij')
    return X, Y, Z, lin

def cartesian_to_spherical(X, Y, Z):
    """
    Convert 3D Cartesian coordinates to spherical coordinates.
    
    Args:
        X, Y, Z: 3D numpy arrays of x, y, z coordinates.
        
    Returns:
        tuple: (r, theta, phi) 3D numpy arrays.
        r: Radial distance [0, inf)
        theta: Polar angle [0, pi]
        phi: Azimuthal angle [-pi, pi]
    """
    # Radial distance
    # Add a small epsilon to avoid division by zero later on
    epsilon = 1e-12
    r = np.sqrt(X**2 + Y**2 + Z**2)
    r_safe = np.where(r == 0, epsilon, r)
    
    # Polar angle theta (angle with z-axis)
    theta = np.arccos(Z / r_safe)
    
    # Azimuthal angle phi (angle in xy-plane)
    phi = np.arctan2(Y, X)
    
    return r, theta, phi
