import numpy as np
from skimage.measure import marching_cubes
import plotly.graph_objects as go

def compute_isosurface(vol, grid_lin, isovalue, is_density=True):
    """
    Computes isosurface from 3D scalar field.
    
    Args:
        vol: 3D numpy array of scalar values (probability density or wavefunction).
        grid_lin: 1D array of coordinates along one dimension (assume uniform cube).
        isovalue: The threshold value.
        is_density: True if vol is |psi|^2.
        
    Returns:
        tuple: (verts, faces, normals, values) for mesh generation.
    """
    try:
        spacing = (grid_lin[1] - grid_lin[0],) * 3
        verts, faces, normals, values = marching_cubes(vol, level=isovalue, spacing=spacing)
            
        # Offset vertices to fall within correct bounds
        verts += grid_lin[0]
        return verts, faces, normals, values
        
    except ValueError as e:
        # Happens when isovalue is outside domain
        return None, None, None, None
