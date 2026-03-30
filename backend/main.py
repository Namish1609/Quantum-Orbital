from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np

import sys

import os

from functools import lru_cache



# Add parent directory to path to reach local physics packages

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from grid.grid import generate_cartesian_grid, cartesian_to_spherical

from physics.hydrogen import hydrogen_wavefunction, radial_probability

from visualization.isosurface import compute_isosurface



app = FastAPI(title="Quantum Orbital API")



# Ensure static folder exists to prevent fastAPI from crashing
FRONTEND_BUILD_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'frontend', 'build'))

if not os.path.exists(FRONTEND_BUILD_DIR):
    try:
        os.makedirs(os.path.join(FRONTEND_BUILD_DIR, "static", "js"), exist_ok=True)
        os.makedirs(os.path.join(FRONTEND_BUILD_DIR, "static", "css"), exist_ok=True)
        # Touch minimal index so endpoints load
        if not os.path.exists(os.path.join(FRONTEND_BUILD_DIR, "index.html")):
            with open(os.path.join(FRONTEND_BUILD_DIR, "index.html"), "w") as f:
                f.write("<html><body>Building...</body></html>")
    except:
        pass



# Update this for production
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools():
    return {}


app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_BUILD_DIR, "static")), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_BUILD_DIR, "index.html"))

@lru_cache(maxsize=32)

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



    try:

        # PURE SPHERICAL REJECTION SAMPLING
        # Restrict the sampling radius to an effective domain to vastly improve acceptance rates
        # for highly localized orbitals (like 1s) within a large bounding box
        eff_r_max = min(size, (3.5 * n**2 + 5.0) / Z)
        
        # Calculate max density over a grid of parameters
        r_c = np.linspace(0.01, eff_r_max, 40)
        theta_c = np.linspace(0, np.pi, 20)
        phi_c = np.linspace(0, 2*np.pi, 20)
        R_grid, Theta_grid, Phi_grid = np.meshgrid(r_c, theta_c, phi_c, indexing='ij')
        psi_c = hydrogen_wavefunction(R_grid.flatten(), Theta_grid.flatten(), Phi_grid.flatten(), n, l, m, Z)
        
        # We must apply the same density scaling map
        d_c = np.abs(psi_c)**2
        p_c = np.power(d_c, density_scale)
        max_p = np.max(p_c) * 1.2
        if max_p == 0: max_p = 1.0

        accepted_x, accepted_y, accepted_z, accepted_d, accepted_phase = [], [], [], [], []
        batch_size = 200000
        max_iterations = 80
        num_samples = min(num_points, 500000)

        for _ in range(max_iterations):
            if len(accepted_x) >= num_samples: break
            
            u_r = np.random.rand(batch_size)
            r_b = eff_r_max * np.cbrt(u_r)

            

            u_theta = np.random.rand(batch_size)

            theta_b = np.arccos(2 * u_theta - 1)

            

            u_phi = np.random.rand(batch_size)

            phi_b = 2 * np.pi * u_phi



            psi_b = hydrogen_wavefunction(r_b, theta_b, phi_b, n, l, m, Z)

            d_b = np.abs(psi_b)**2

            p_b = np.power(d_b, density_scale)

            

            u = np.random.uniform(0, max_p, batch_size)

            mask = u < p_b

            

            x_b = r_b[mask] * np.sin(theta_b[mask]) * np.cos(phi_b[mask])

            y_b = r_b[mask] * np.sin(theta_b[mask]) * np.sin(phi_b[mask])

            z_b = r_b[mask] * np.cos(theta_b[mask])

            

            accepted_x.extend(x_b)

            accepted_y.extend(y_b)

            accepted_z.extend(z_b)

            accepted_d.extend(d_b[mask])

            

            phase = np.sign(np.real(psi_b[mask]))

            accepted_phase.extend(phase)



        X_sampled = np.array(accepted_x[:num_samples])

        Y_sampled = np.array(accepted_y[:num_samples])

        Z_sampled = np.array(accepted_z[:num_samples])

        density_sampled = np.array(accepted_d[:num_samples])

        phase_sampled = np.array(accepted_phase[:num_samples])

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

    isovalue: float = Query(0.0025, gt=0),

    show_phase: bool = Query(True)

):

    if l >= n: raise HTTPException(400, "l must be strict less than n.")

    if abs(m) > l: raise HTTPException(400, "m must be between -l and l.")



    try:

        X, Y, Z_grid, lin, psi = compute_volume(n, l, m, Z, resolution, size)

        

        density = np.abs(psi)**2

        

        d_max = np.max(density)

        if d_max > 0:

            density = density / d_max

            

        psi_real = np.real(psi)

        

        surfaces = []

        try:

            # We compute the isosurface on the NORMALIZED density [0, 1]

            # The isovalue slider corresponds to a percentage of the peak probability density.

            verts, faces, _, _ = compute_isosurface(density, lin, isovalue)

            if verts is not None:

                if show_phase:

                    # Evaluate wavefunction phase at precise vertex locations

                    r_v, theta_v, phi_v = cartesian_to_spherical(verts[:, 0], verts[:, 1], verts[:, 2])

                    psi_v = hydrogen_wavefunction(r_v, theta_v, phi_v, n, l, m, Z)

                    phases = np.sign(np.real(psi_v))

                    

                    colors = np.zeros((len(verts), 3), dtype=np.float32)

                    colors[phases > 0] = [1.0, 0.2, 0.2]  # Red for positive

                    colors[phases <= 0] = [0.2, 0.2, 1.0] # Blue for negative

                    

                    surfaces.append({

                        "vertices": verts.tolist(),

                        "faces": faces.flatten().tolist(),

                        "vertex_colors": colors.flatten().tolist()

                    })

                else:

                    surfaces.append({

                        "vertices": verts.tolist(),

                        "faces": faces.flatten().tolist(),

                        "color": "#00ffff"

                    })

        except Exception as e:

            # marching_cubes fails if isovalue is outside data range, this is fine just return empty

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



if __name__ == "__main__":

    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

