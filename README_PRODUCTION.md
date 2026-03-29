# Quantum Orbital Simulator - Production Version

This project has been updated from a Streamlit prototype to a high-performance production web system using FastAPI for the backend and React + Three.js for the frontend.

## 🏗️ Architecture
- **Backend**: FastAPI (`/backend`) serves computed points using optimized NumPy vectorization and caching. Streamlit has been completely removed. Plotly usage is removed. The mathematical calculations have been preserved but heavily optimized to minimize overhead and bandwidth.
- **Frontend**: React + Three.js + React Three Fiber (`/frontend`) efficiently renders the points on the GPU using `Float32Array` within `BufferGeometry`.

## 🚀 Running Locally

### 1. Start the Backend
```bash
cd backend
python -m pip install -r requirements.txt
python main.py
```
This will start the compute server on `http://localhost:8000`.

### 2. Start the Frontend
```bash
cd frontend
npm install
npm install @react-three/fiber @react-three/drei three react-scripts
npm start
```
This will start the development server on `http://localhost:3000`.

## ⚙️ Optimizations & Performance Details
- **Math/Physics Vectors**: Removed `numpy.nditer` or nested python loops if any were present; all operations are done on flattened `ndarray` maps using boolean masking to find thresholds.
- **Payload Minimizer**: Responses only include data points that meet `> 50%` of the volume density or a defined relative threshold (`threshold = max_density * 0.05`). 
- **Downsampling**: Max point limits set to exactly 50,000 using `np.random.choice()` safely reducing bandwidth. Cap imposed on spatial resolution limit at `50x50x50`.
- **Three.js Float32 Array BufferGeometry**: Sent data is immediately parsed entirely inside optimal typed arrays instead of React virtual DOM components. It leverages hardware WebGL Points primitive and AdditiveBlending. 
- **LRU Cache Caching**: Identical query parameters are instantaneous after first load (`@lru_cache`).
- **Phasing Support**: Uses `Math.sign` mapped to vertex colors securely dynamically adjusted via slider events.

## 🌍 Deployment
- **Backend**: Deploy `backend/` utilizing Uvicorn and requirements.txt (`uvicorn main:app --host 0.0.0.0 --port $PORT`). Applicable for DigitalOcean Apps, Render, or Heroku.
- **Frontend**: Run `npm run build` inside `frontend/` and host static files via Netlify, Vercel, or AWS S3.