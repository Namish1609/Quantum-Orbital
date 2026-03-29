import plotly.graph_objects as go
import numpy as np
from visualization.isosurface import compute_isosurface    

def render_orbital(vol, grid_lin, isovalue, title="Orbital Visualization", show_phase=False, original_wavefunction=None, opacity=0.8):
    """
    Renders the orbital using Plotly 3D mesh.
    """
    fig = go.Figure()

    if not show_phase or original_wavefunction is None:    
        # Standard rendering (single color based on distance or uniform)
        verts, faces, _, _ = compute_isosurface(vol, grid_lin, isovalue, is_density=True)
        if verts is not None:
            x, y, z = verts[:, 0], verts[:, 1], verts[:, 2]
            i, j, k = faces[:, 0], faces[:, 1], faces[:, 2]
            fig.add_trace(go.Mesh3d(
                x=x, y=y, z=z, i=i, j=j, k=k,
                color='lightblue', opacity=opacity,
                lighting=dict(ambient=0.4, diffuse=0.8, roughness=0.1, specular=1.0)
            ))
    else:
        # Render positive and negative phases
        # Positive Phase
        try:
            verts_pos, faces_pos, _, _ = compute_isosurface(original_wavefunction, grid_lin, isovalue, is_density=False)
            if verts_pos is not None:
                x_p, y_p, z_p = verts_pos[:, 0], verts_pos[:, 1], verts_pos[:, 2]
                i_p, j_p, k_p = faces_pos[:, 0], faces_pos[:, 1], faces_pos[:, 2]

                fig.add_trace(go.Mesh3d(
                    x=x_p, y=y_p, z=z_p, i=i_p, j=j_p, k=k_p,
                    color='red', opacity=opacity, name='+ Phase'
                ))
        except Exception as e:
            pass

        # Negative Phase
        try:
            verts_neg, faces_neg, _, _ = compute_isosurface(-original_wavefunction, grid_lin, isovalue, is_density=False)
            if verts_neg is not None:
                x_n, y_n, z_n = verts_neg[:, 0], verts_neg[:, 1], verts_neg[:, 2]
                i_n, j_n, k_n = faces_neg[:, 0], faces_neg[:, 1], faces_neg[:, 2]

                fig.add_trace(go.Mesh3d(
                     x=x_n, y=y_n, z=z_n, i=i_n, j=j_n, k=k_n,
                    color='blue', opacity=opacity, name='- Phase'
                ))
        except Exception as e:
            pass

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X', showbackground=False, visible=False),
            yaxis=dict(title='Y', showbackground=False, visible=False),
            zaxis=dict(title='Z', showbackground=False, visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def render_scatter(X, Y, Z, density, psi_real, num_points=10000, point_size=2, density_scale=1.0, show_phase=False, opacity=0.8, title="Electron Cloud"):
    """
    Renders the electron cloud using probabilistic Monte Carlo sampling.
    """
    x_flat = X.ravel()
    y_flat = Y.ravel()
    z_flat = Z.ravel()
    dens_flat = density.ravel()
    psi_flat = psi_real.ravel()
    
    # Probability scaling
    probs = dens_flat ** density_scale
    prob_sum = np.sum(probs)
    if prob_sum == 0 or np.isnan(prob_sum):
        probs = np.ones_like(probs) / len(probs)
    else:
        probs /= prob_sum
        
    num_points = min(num_points, len(x_flat))
    if num_points <= 0:
       return go.Figure()
       
    # Sample points based on probability density
    sampled_indices = np.random.choice(len(x_flat), size=num_points, p=probs, replace=True)
    
    # Calculate step size to add random continuous jitter
    step_x = np.abs(X[0, 1, 0] - X[0, 0, 0]) if X.shape[1] > 1 else 0
    if step_x == 0 and len(np.unique(x_flat)) > 1:
        step_x = np.abs(np.unique(x_flat)[1] - np.unique(x_flat)[0])
    jitter = step_x / 2.0
    
    x_samp = x_flat[sampled_indices] + np.random.uniform(-jitter, jitter, size=num_points)
    y_samp = y_flat[sampled_indices] + np.random.uniform(-jitter, jitter, size=num_points)
    z_samp = z_flat[sampled_indices] + np.random.uniform(-jitter, jitter, size=num_points)
    psi_samp = psi_flat[sampled_indices]
    
    fig = go.Figure()
    
    if show_phase:
        pos_mask = psi_samp > 0
        neg_mask = psi_samp <= 0
        
        if np.any(pos_mask):
            fig.add_trace(go.Scatter3d(
                x=x_samp[pos_mask], y=y_samp[pos_mask], z=z_samp[pos_mask],
                mode='markers',
                marker=dict(size=point_size, color='red', opacity=opacity),
                name='+ Phase'
            ))
            
        if np.any(neg_mask):
            fig.add_trace(go.Scatter3d(
                x=x_samp[neg_mask], y=y_samp[neg_mask], z=z_samp[neg_mask],
                mode='markers',
                marker=dict(size=point_size, color='blue', opacity=opacity),
                name='- Phase'
            ))
    else:
        fig.add_trace(go.Scatter3d(
            x=x_samp, y=y_samp, z=z_samp,
            mode='markers',
            marker=dict(
                size=point_size,
                color=dens_flat[sampled_indices],
                colorscale='Viridis',
                opacity=opacity,
                colorbar=dict(title="Density")
            )
        ))
        
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis=dict(title='X', showbackground=False, visible=False),
            yaxis=dict(title='Y', showbackground=False, visible=False),
            zaxis=dict(title='Z', showbackground=False, visible=False),
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )
    return fig

def render_density(vol, grid_lin, isovalue, title="Electron Density"):
    """
    Renders electron density specifically.
    """
    return render_orbital(vol, grid_lin, isovalue, title=title, show_phase=False)
