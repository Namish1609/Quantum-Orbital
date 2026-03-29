from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import os
from functools import lru_cache

from grid.grid import generate_cartesian_grid, cartesian_to_spherical
from physics.hydrogen import hydrogen_wavefunction, radial_probability
from visualization.isosurface import compute_isosurface

app = FastAPI(title="Quantum Orbital API")

# Update this for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

frontend_build_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../frontend/build'))
if os.path.exists(os.path.join(frontend_build_dir, "static")):
    app.mount("/static", StaticFiles(directory=os.path.join(frontend_build_dir, "static")), name="static")
def compute_volume(n: int, l: int, m: int, Z: float, resolution: int, size: float):
    # Cap resolution to prevent memory exhaustion, but allow enough for high-fidelity isosurfaces
    res = min(resolution, 200)
    
    X, Y, Z_grid, lin = generate_cartesian_grid(size=size, resolution=res)
    r, theta, phi = cartesian_to_spherical(X, Y, Z_grid)
    
    psi = hydrogen_wavefunction(r, theta, phi, n, l, m, Z)
    return X, Y, Z_grid, lin, psi

@app.get("/scatter")
def get_scatter(
    n: int = Query(..., ge=1, le=10),
    l: int = Query(..., ge=0),
    m: int = Query(...),
    Z: float = Query(1.0, gt=0),
    resolution: int = Query(40, le=200),
    size: float = Query(30.0, gt=0),
    num_points: int = Query(200000, le=500000),
    density_scale: float = Query(1.0, gt=0)
):
    if l >= n: raise HTTPException(400, "l must be strict less than n.")
    if abs(m) > l: raise HTTPException(400, "m must be between -l and l.")

    resolution = min(resolution, 120)
    num_points = min(num_points, 200000)

    try:
        X, Y, Z_grid, lin, psi = compute_volume(n, l, m, Z, resolution, size)
        density = np.abs(psi)**2
        psi_real = np.real(psi)
        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z_grid.flatten()
        density_flat = density.flatten()
        phase_flat = np.sign(psi_real.flatten())

        # Scale density and normalize as probabilities
        prob_dist = np.power(density_flat, density_scale)
        dist_sum = np.sum(prob_dist)
        
        if dist_sum == 0:
            return {"points": []}
            
        prob_dist /= dist_sum
        
        # Monte Carlo sampling
        num_samples = min(num_points, 500000)
        chosen_indices = np.random.choice(len(X_flat), size=num_samples, p=prob_dist, replace=True)
        
        # Jittering to prevent hard grid snapping
        delta = lin[1] - lin[0] if len(lin) > 1 else size/resolution
        jitter_x = np.random.uniform(-delta/2, delta/2, num_samples)
        jitter_y = np.random.uniform(-delta/2, delta/2, num_samples)
        jitter_z = np.random.uniform(-delta/2, delta/2, num_samples)
        
        X_sampled = X_flat[chosen_indices] + jitter_x
        Y_sampled = Y_flat[chosen_indices] + jitter_y
        Z_sampled = Z_flat[chosen_indices] + jitter_z
        density_sampled = density_flat[chosen_indices]
        phase_sampled = phase_flat[chosen_indices]

        # Normalize density for visualization
        d_max = np.max(density_sampled) if len(density_sampled) > 0 else 1
        if d_max > 0:
            density_sampled = density_sampled / d_max
            
        points = np.column_stack((X_sampled, Y_sampled, Z_sampled, density_sampled, phase_sampled))
        return {"points": points.tolist()}
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/isosurface")
def get_isosurface(
    n: int = Query(..., ge=1, le=10),
    l: int = Query(..., ge=0),
    m: int = Query(...),
    Z: float = Query(1.0, gt=0),
    resolution: int = Query(60, le=200),
    size: float = Query(30.0, gt=0),
    isovalue: float = Query(0.01, gt=0),
    show_phase: bool = Query(True)
):
    if l >= n: raise HTTPException(400, "l must be strict less than n.")        
    if abs(m) > l: raise HTTPException(400, "m must be between -l and l.")      

    resolution = min(resolution, 120)

    try:
        X, Y, Z_grid, lin, psi = compute_volume(n, l, m, Z, resolution, size)

        density = np.abs(psi)**2

        d_max = np.max(density)
        if d_max > 0:
            density = density / d_max
            
        psi_real = np.real(psi)
        if d_max > 0:
            psi_real = psi_real / np.sqrt(d_max)

        mask = density > isovalue
        density = density * mask
            
        surfaces = []
        try:
            if show_phase:
                verts_pos, faces_pos, _, _ = compute_isosurface(psi_real, lin, np.sqrt(isovalue))
                if verts_pos is not None and len(verts_pos) > 0:
                    colors_pos = np.zeros((len(verts_pos), 3), dtype=np.float32)
                    colors_pos[:] = [1.0, 0.2, 0.2]
                    surfaces.append({
                        "vertices": verts_pos.tolist(),
                        "faces": faces_pos.flatten().tolist(),
                        "vertex_colors": colors_pos.flatten().tolist()
                    })

                verts_neg, faces_neg, _, _ = compute_isosurface(-psi_real, lin, np.sqrt(isovalue))
                if verts_neg is not None and len(verts_neg) > 0:
                    colors_neg = np.zeros((len(verts_neg), 3), dtype=np.float32)
                    colors_neg[:] = [0.2, 0.2, 1.0]
                    surfaces.append({
                        "vertices": verts_neg.tolist(),
                        "faces": faces_neg.flatten().tolist(),
                        "vertex_colors": colors_neg.flatten().tolist()
                    })
            else:
                verts, faces, _, _ = compute_isosurface(density, lin, isovalue)
                if verts is not None and len(verts) > 0:
                    surfaces.append({
                        "vertices": verts.tolist(),
                        "faces": faces.flatten().tolist(),
                        "color": "#00ffff"
                    })
        except Exception as e:
            pass

        return {"surfaces": surfaces}

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/radial")
def get_radial(
    n: int = Query(...),
    l: int = Query(...),
    Z: float = Query(1.0),
    size: float = Query(30.0)
):
    r_1d = np.linspace(0.01, size, 200)
    P_r = radial_probability(r_1d, n, l, Z)
    
    # Return array of {r, P} for recharts
    data = [{"r": float(r), "P": float(p)} for r, p in zip(r_1d, P_r)]
    max_idx = int(np.argmax(P_r))
    return {
        "data": data,
        "max_r": float(r_1d[max_idx])
    }

@app.get("/")
def serve_root():
    return FileResponse(os.path.join(frontend_build_dir, "index.html"))

@app.get("/{full_path:path}")
def serve_react_app(full_path: str):
    return FileResponse(os.path.join(frontend_build_dir, "index.html"))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)