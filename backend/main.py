from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import numpy as np
import hashlib
import json

import sys

import os
import threading
from pathlib import Path



# Add parent directory to path to reach local physics packages

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))



from grid.grid import cartesian_to_spherical

from physics.hydrogen import hydrogen_wavefunction, radial_probability

from visualization.isosurface import compute_isosurface



app = FastAPI(title="Quantum Orbital API")

MAX_ISOSURFACE_RESOLUTION = 160
MAX_SCATTER_POINTS = 50000000
CACHE_VERSION = 1
CACHE_ROOT = Path(os.path.abspath(os.path.join(os.path.dirname(__file__), 'cache', 'orbitals')))
CACHE_MAX_SIZE_BYTES = int(os.getenv("ORBITAL_CACHE_MAX_BYTES", "8589934592"))
CACHE_MAX_ENTRY_BYTES = int(os.getenv("ORBITAL_CACHE_MAX_ENTRY_BYTES", "2147483648"))
CACHE_LOCK = threading.Lock()



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
app.add_middleware(GZipMiddleware, minimum_size=1024)

@app.get("/.well-known/appspecific/com.chrome.devtools.json")
async def chrome_devtools():
    return {}


app.mount("/static", StaticFiles(directory=os.path.join(FRONTEND_BUILD_DIR, "static")), name="static")

@app.get("/")
async def root():
    return FileResponse(os.path.join(FRONTEND_BUILD_DIR, "index.html"))

def compute_volume(n: int, l: int, m: int, Z: float, resolution: int, size: float):

    # Cap resolution to prevent memory exhaustion, but allow enough for high-fidelity isosurfaces

    res = min(resolution, MAX_ISOSURFACE_RESOLUTION)

    lin = np.linspace(-size, size, res, dtype=np.float32)
    x = lin[:, None, None]
    y = lin[None, :, None]
    z = lin[None, None, :]

    # Build spherical coordinates directly from broadcasted axes to reduce peak memory pressure.
    epsilon = np.float32(1e-12)
    r = np.sqrt(x * x + y * y + z * z)
    r_safe = np.where(r == 0, epsilon, r)
    theta = np.arccos(np.clip(z / r_safe, -1.0, 1.0))
    phi = np.arctan2(y, x)

    psi = hydrogen_wavefunction(r, theta, phi, n, l, m, Z)
    return lin, psi


def get_point_range_for_n(n: int):
    if n <= 4:
        return 1000000, MAX_SCATTER_POINTS
    if n <= 7:
        return 10000000, MAX_SCATTER_POINTS
    return 30000000, MAX_SCATTER_POINTS


def get_default_point_count_for_n(n: int):
    if n <= 4:
        return 3000000
    if n <= 7:
        return 15000000
    return 40000000


def _normalize_cache_value(value):
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return round(float(value), 8)
    if isinstance(value, (list, tuple)):
        return [_normalize_cache_value(v) for v in value]
    return value


def _build_cache_key(namespace: str, params: dict) -> str:
    normalized = {k: _normalize_cache_value(v) for k, v in sorted(params.items())}
    key_payload = {
        "version": CACHE_VERSION,
        "namespace": namespace,
        "params": normalized,
    }
    encoded = json.dumps(key_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_path(namespace: str, params: dict) -> Path:
    cache_dir = CACHE_ROOT / namespace
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{_build_cache_key(namespace, params)}.npz"


def _safe_unlink(path: Path):
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _prune_cache():
    if not CACHE_ROOT.exists():
        return

    entries = []
    total_size = 0
    for file_path in CACHE_ROOT.rglob("*.npz"):
        try:
            stat = file_path.stat()
        except OSError:
            continue
        entries.append((stat.st_mtime, stat.st_size, file_path))
        total_size += stat.st_size

    if total_size <= CACHE_MAX_SIZE_BYTES:
        return

    entries.sort(key=lambda item: item[0])
    for _, size, file_path in entries:
        _safe_unlink(file_path)
        total_size -= size
        if total_size <= CACHE_MAX_SIZE_BYTES:
            break


def _estimate_payload_bytes(payload: dict[str, np.ndarray]) -> int:
    total = 0
    for value in payload.values():
        total += int(value.nbytes)
    return total


def _save_npz_cache(path: Path, payload: dict[str, np.ndarray]):
    if _estimate_payload_bytes(payload) > CACHE_MAX_ENTRY_BYTES:
        return

    tmp_path = path.with_suffix(".tmp")
    with CACHE_LOCK:
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            with open(tmp_path, "wb") as handle:
                np.savez_compressed(handle, **payload)
            os.replace(tmp_path, path)
        finally:
            _safe_unlink(tmp_path)
        _prune_cache()


def _load_npz_cache(path: Path):
    if not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=False) as payload:
            data = {name: payload[name] for name in payload.files}
        os.utime(path, None)
        return data
    except Exception:
        _safe_unlink(path)
        return None


def _save_json_cache(path: Path, payload: dict):
    blob = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    _save_npz_cache(path, {"blob": np.frombuffer(blob, dtype=np.uint8)})


def _load_json_cache(path: Path):
    cached = _load_npz_cache(path)
    if cached is None or "blob" not in cached:
        return None

    try:
        blob = cached["blob"].tobytes()
        return json.loads(blob.decode("utf-8"))
    except Exception:
        _safe_unlink(path)
        return None


def _scatter_binary_response(points: np.ndarray) -> Response:
    points_f32 = points.astype(np.float32, copy=False)
    return Response(
        content=points_f32.tobytes(),
        media_type="application/octet-stream",
        headers={
            "X-Point-Count": str(points_f32.shape[0]),
            "X-Point-Stride": "5",
        },
    )



@app.get("/scatter")

def get_scatter(

    n: int = Query(..., ge=1, le=10),

    l: int = Query(..., ge=0),

    m: int = Query(...),

    Z: float = Query(1.0, gt=0),

    resolution: int = Query(40, le=200),

    size: float = Query(30.0, gt=0),

    num_points: int | None = Query(None, ge=1000, le=MAX_SCATTER_POINTS),

    density_scale: float = Query(1.0, gt=0),

    binary: bool = Query(False)

):

    if l >= n: raise HTTPException(400, "l must be strict less than n.")

    if abs(m) > l: raise HTTPException(400, "m must be between -l and l.")

    min_points, max_points = get_point_range_for_n(n)
    requested_points = get_default_point_count_for_n(n) if num_points is None else num_points
    if requested_points < min_points or requested_points > max_points:
        raise HTTPException(
            400,
            f"num_points for n={n} must be between {min_points} and {max_points}."
        )

    scatter_cache_params = {
        "n": n,
        "l": l,
        "m": m,
        "Z": Z,
        "resolution": resolution,
        "size": size,
        "num_points": requested_points,
        "density_scale": density_scale,
    }
    scatter_cache_file = _cache_path("scatter", scatter_cache_params)
    cached_scatter = _load_npz_cache(scatter_cache_file)
    if cached_scatter is not None and "points" in cached_scatter:
        cached_points = cached_scatter["points"]
        if binary:
            return _scatter_binary_response(cached_points)
        return {"points": cached_points.tolist()}



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

        accepted_chunks = []
        accepted_total = 0
        num_samples = min(requested_points, MAX_SCATTER_POINTS)
        batch_size = min(1000000, max(200000, num_samples // 80))
        max_iterations = max(80, int(np.ceil(num_samples / batch_size) * 8))

        for _ in range(max_iterations):
            if accepted_total >= num_samples: break
            
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

            accepted_count = int(np.count_nonzero(mask))
            if accepted_count == 0:
                continue

            take_count = min(accepted_count, num_samples - accepted_total)

            

            x_b = r_b[mask] * np.sin(theta_b[mask]) * np.cos(phi_b[mask])

            y_b = r_b[mask] * np.sin(theta_b[mask]) * np.sin(phi_b[mask])

            z_b = r_b[mask] * np.cos(theta_b[mask])

            d_selected = d_b[mask]
            phase = np.sign(np.real(psi_b[mask]))

            chunk = np.column_stack((
                x_b[:take_count],
                y_b[:take_count],
                z_b[:take_count],
                d_selected[:take_count],
                phase[:take_count],
            )).astype(np.float32, copy=False)
            accepted_chunks.append(chunk)
            accepted_total += take_count

        if not accepted_chunks:
            empty_points = np.empty((0, 5), dtype=np.float32)
            _save_npz_cache(scatter_cache_file, {"points": empty_points})
            if binary:
                return _scatter_binary_response(empty_points)
            return {"points": []}

        points = np.concatenate(accepted_chunks, axis=0)

        # Normalize density for visualization
        d_max = np.max(points[:, 3]) if points.shape[0] > 0 else 1
        if d_max > 0:
            points[:, 3] = points[:, 3] / d_max

        _save_npz_cache(scatter_cache_file, {"points": points.astype(np.float32, copy=False)})

        if binary:
            return _scatter_binary_response(points)

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

    isosurface_cache_params = {
        "n": n,
        "l": l,
        "m": m,
        "Z": Z,
        "resolution": resolution,
        "size": size,
        "isovalue": isovalue,
        "show_phase": show_phase,
    }
    isosurface_cache_file = _cache_path("isosurface", isosurface_cache_params)
    cached_isosurface = _load_json_cache(isosurface_cache_file)
    if cached_isosurface is not None and "surfaces" in cached_isosurface:
        return cached_isosurface



    try:

        lin, psi = compute_volume(n, l, m, Z, resolution, size)

        

        density = np.abs(psi)**2

        

        d_max = np.max(density)

        if d_max > 0:

            density = density / d_max

            

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

        response_payload = {"surfaces": surfaces}
        _save_json_cache(isosurface_cache_file, response_payload)
        return response_payload



    except MemoryError:

        raise HTTPException(
            status_code=503,
            detail="Insufficient memory for this isosurface request. Try reducing resolution or grid size."
        )

    except Exception as e:

        import traceback

        traceback.print_exc()

        raise HTTPException(status_code=500, detail=str(e))



@app.get("/radial")

def get_radial(

    n: int = Query(..., ge=1, le=10),

    l: int = Query(..., ge=0),

    Z: float = Query(1.0, gt=0),

    size: float = Query(30.0, gt=0)

):

    if l >= n:
        raise HTTPException(400, "l must be strict less than n.")

    radial_cache_params = {
        "n": n,
        "l": l,
        "Z": Z,
        "size": size,
    }
    radial_cache_file = _cache_path("radial", radial_cache_params)
    cached_radial = _load_npz_cache(radial_cache_file)
    if cached_radial is not None and {"r", "p", "max_r"}.issubset(cached_radial.keys()):
        cached_r = cached_radial["r"]
        cached_p = cached_radial["p"]
        cached_max_r = float(np.asarray(cached_radial["max_r"]).item())
        cached_data = [{"r": float(r), "P": float(p)} for r, p in zip(cached_r, cached_p)]
        return {
            "data": cached_data,
            "max_r": cached_max_r,
        }

    try:
        r_1d = np.linspace(0.01, size, 200)

        P_r = radial_probability(r_1d, n, l, Z)

        # Return array of {r, P} for recharts
        data = [{"r": float(r), "P": float(p)} for r, p in zip(r_1d, P_r)]

        max_idx = int(np.argmax(P_r))

        response_payload = {

            "data": data,

            "max_r": float(r_1d[max_idx])

        }

        _save_npz_cache(
            radial_cache_file,
            {
                "r": r_1d.astype(np.float32),
                "p": P_r.astype(np.float32),
                "max_r": np.array(float(r_1d[max_idx]), dtype=np.float32),
            },
        )

        return response_payload

    except Exception as e:

        import traceback

        traceback.print_exc()

        raise HTTPException(status_code=500, detail=str(e))



if __name__ == "__main__":

    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)

