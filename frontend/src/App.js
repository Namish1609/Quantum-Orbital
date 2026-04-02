import React, { useState, useEffect, useMemo } from 'react';
import { Canvas, useThree } from '@react-three/fiber';
import { OrbitControls, Text, Billboard } from '@react-three/drei';
import * as THREE from 'three';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';
import './App.css';

// Create a circular sprite texture for classic hard-edged point rendering.
const createCircleTexture = () => {
  const canvas = document.createElement('canvas');
  canvas.width = 32;
  canvas.height = 32;
  const ctx = canvas.getContext('2d');
  ctx.beginPath();
  ctx.arc(16, 16, 15, 0, 2 * Math.PI);
  ctx.fillStyle = 'white';
  ctx.fill();
  const texture = new THREE.CanvasTexture(canvas);
  texture.generateMipmaps = false;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  return texture;
};
const circleTexture = createCircleTexture();

const BASELINE_PIXELS = 1920 * 1080;
const PHASE_NEG_COLOR = new THREE.Color('#3333ff');
const DENSITY_C1 = new THREE.Color('#30005c');
const DENSITY_C2 = new THREE.Color('#c51b7d');
const DENSITY_C3 = new THREE.Color('#ff8c00');
const MAX_TOTAL_POINTS = 50000000;
const BASE_POINT_SIZE = 0.1;
const POINT_SIZE_DISTANCE_STEP = 20;
const POINT_SIZE_STEP_INCREMENT = 0.05;
const MAX_AUTO_POINT_SIZE = 0.25;
const PHASE_INTENSITY_HARDLIMIT = 0.2;

const getAutoPointSizeForDistance = (distance) => {
  const safeDistance = Number.isFinite(distance) ? Math.max(0, distance) : 0;
  const stepCount = Math.floor(safeDistance / POINT_SIZE_DISTANCE_STEP);
  return Math.min(MAX_AUTO_POINT_SIZE, BASE_POINT_SIZE + stepCount * POINT_SIZE_STEP_INCREMENT);
};

const getPointRangeForN = (nValue) => {
  if (nValue <= 4) return { min: 1000000, max: MAX_TOTAL_POINTS, defaultValue: 3000000, step: 100000 };
  if (nValue <= 7) return { min: 10000000, max: MAX_TOTAL_POINTS, defaultValue: 15000000, step: 250000 };
  return { min: 30000000, max: MAX_TOTAL_POINTS, defaultValue: 40000000, step: 500000 };
};

const getPointCountForN = (nValue) => {
  return getPointRangeForN(nValue).defaultValue;
};

const clampPointCountForN = (nValue, count) => {
  const range = getPointRangeForN(nValue);
  return Math.max(range.min, Math.min(range.max, count));
};

// --- Reusable Three.js Components ---

const ScatterPlot = ({ data, pointSize, opacity, showPhase, enableSimpleGlow }) => {
  const { size, viewport } = useThree();
  const dpr = viewport.dpr || 1;
  const intensityScale = Math.max(0, Math.min(1, opacity));

  const adaptivePointSize = useMemo(() => {
    const cssPixels = Math.max(1, size.width * size.height);
    const areaScale = Math.sqrt(cssPixels / BASELINE_PIXELS);
    const dprScale = Math.max(1, Math.min(2.5, dpr));
    const visibilityScale = Math.max(1, areaScale * 0.85) * (1 + (dprScale - 1) * 0.25);
    return Math.min(MAX_AUTO_POINT_SIZE, pointSize * visibilityScale);
  }, [pointSize, size.width, size.height, dpr]);

  const { positions, colors } = useMemo(() => {
    const hasFlatPoints = !!data && data.pointsFlat instanceof Float32Array && data.pointsFlat.length > 0;
    const hasNestedPoints = !!data && Array.isArray(data.points) && data.points.length > 0;
    if (!hasFlatPoints && !hasNestedPoints) return { positions: null, colors: null };

    const pointStride = hasFlatPoints ? Math.max(5, Number(data.pointStride) || 5) : 5;
    const count = hasFlatPoints ? Math.floor(data.pointsFlat.length / pointStride) : data.points.length;
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);

    const phaseNegR = PHASE_NEG_COLOR.r;
    const phaseNegG = PHASE_NEG_COLOR.g;
    const phaseNegB = PHASE_NEG_COLOR.b;
    const c1r = DENSITY_C1.r;
    const c1g = DENSITY_C1.g;
    const c1b = DENSITY_C1.b;
    const c2r = DENSITY_C2.r;
    const c2g = DENSITY_C2.g;
    const c2b = DENSITY_C2.b;
    const c3r = DENSITY_C3.r;
    const c3g = DENSITY_C3.g;
    const c3b = DENSITY_C3.b;

    for (let i = 0; i < count; i++) {
        let x;
        let y;
        let z;
        let density;
        let phase;
        if (hasFlatPoints) {
          const base = i * pointStride;
          x = data.pointsFlat[base + 0];
          y = data.pointsFlat[base + 1];
          z = data.pointsFlat[base + 2];
          density = data.pointsFlat[base + 3];
          phase = data.pointsFlat[base + 4];
        } else {
          const point = data.points[i];
          x = point[0];
          y = point[1];
          z = point[2];
          density = point[3];
          phase = point[4];
        }
        const idx = i * 3;
        const r = Math.sqrt(x*x + y*y + z*z);

        pos[idx + 0] = x;
        pos[idx + 1] = y;
        pos[idx + 2] = z;

        if (showPhase) {
            const distanceFade = Math.max(0.1, 30.0 / (r + 30.0));
          const intensity = Math.max(PHASE_INTENSITY_HARDLIMIT, density * 5) * distanceFade * intensityScale;
            const baseR = phase > 0 ? 1 : phaseNegR;
            const baseG = phase > 0 ? 0 : phaseNegG;
            const baseB = phase > 0 ? 0 : phaseNegB;

            col[idx + 0] = baseR * intensity;
            col[idx + 1] = baseG * intensity;
            col[idx + 2] = baseB * intensity;
        } else {
            if (density < 0.5) {
                const t = density * 2.0;
                col[idx + 0] = (c1r + (c2r - c1r) * t) * intensityScale;
                col[idx + 1] = (c1g + (c2g - c1g) * t) * intensityScale;
                col[idx + 2] = (c1b + (c2b - c1b) * t) * intensityScale;
            } else {
                const t = (density - 0.5) * 2.0;
                col[idx + 0] = (c2r + (c3r - c2r) * t) * intensityScale;
                col[idx + 1] = (c2g + (c3g - c2g) * t) * intensityScale;
                col[idx + 2] = (c2b + (c3b - c2b) * t) * intensityScale;
            }
        }
    }
    return { positions: pos, colors: col };
  }, [data, showPhase, intensityScale]);

  if (!positions) return null;

  return (
    <points>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={positions.length / 3}
          array={positions}
          itemSize={3}
        />
        <bufferAttribute
          attach="attributes-color"
          count={colors.length / 3}
          array={colors}
          itemSize={3}
        />
      </bufferGeometry>
      {enableSimpleGlow ? (
        <pointsMaterial
          size={adaptivePointSize}
          vertexColors={true}
          transparent={true}
          opacity={Math.min(1, 0.35 + opacity * 0.55)}
          sizeAttenuation={true}
          map={circleTexture}
          alphaTest={0.35}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
          depthTest={true}
          toneMapped={false}
        />
      ) : (
        <pointsMaterial
          size={adaptivePointSize}
          vertexColors={true}
          transparent={false}
          opacity={1}
          sizeAttenuation={true}
          map={circleTexture}
          alphaTest={0.5}
          depthWrite={true}
          depthTest={true}
          toneMapped={false}
        />
      )}
    </points>
  );
};

const IsosurfaceMesh = ({ data, opacity, showPhase, isovalue }) => {
    const geometries = useMemo(() => {
      if (!data || !data.surfaces || data.surfaces.length === 0) return null;
      
      return data.surfaces.map(surf => {
          const geo = new THREE.BufferGeometry();
          const vertices = new Float32Array(surf.vertices.flat());
          
          // CRITICAL: WebGL indices must be Uint32Array or Uint16Array for setIndex!
          const indicesArray = surf.faces.flat();
          const indices = new Uint32Array(indicesArray);
          
          geo.setAttribute('position', new THREE.BufferAttribute(vertices, 3));
          geo.setIndex(new THREE.BufferAttribute(indices, 1));
          
          if (surf.vertex_colors) {
              const colors = new Float32Array(surf.vertex_colors);
              geo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
          }
          
          geo.computeVertexNormals();
          
          // Calculate density color manually for isosurface when showPhase is OFF
          let surfaceColor = surf.color;
          if (!showPhase) {
              const c1 = new THREE.Color("#30005c"); // Dark Purple 
              const c2 = new THREE.Color("#c51b7d"); // Magenta
              const c3 = new THREE.Color("#ff8c00"); // Orange
              const baseC = new THREE.Color();
              if (isovalue < 0.5) {
                  baseC.copy(c1).lerp(c2, isovalue * 2.0);
              } else {
                  baseC.copy(c2).lerp(c3, (isovalue - 0.5) * 2.0);
              }
              surfaceColor = baseC.getHex();
          }
          
          return { geo, color: surfaceColor, hasColors: !!surf.vertex_colors };
      });
    }, [data, showPhase, isovalue]);
  
    if (!geometries) return null;
  
    return (
      <group>
        {geometries.map((item, i) => (
          <mesh key={i} geometry={item.geo}>
              <meshStandardMaterial
                color={item.hasColors ? 0xffffff : item.color}
                vertexColors={item.hasColors}
                transparent={true}
                opacity={opacity}
                side={THREE.DoubleSide}
                roughness={0.4}
                metalness={0.1}
              />
          </mesh>
        ))}
      </group>
    );
};

// --- Custom Axes Helper extending both directions ---
const FullAxes = ({ size }) => {
  const lineScale = size * 1.5; // Make the axes longer than the atom grid
  const xAxis = useMemo(() => new Float32Array([-lineScale, 0, 0, lineScale, 0, 0]), [lineScale]);
  const yAxis = useMemo(() => new Float32Array([0, -lineScale, 0, 0, lineScale, 0]), [lineScale]);
  const zAxis = useMemo(() => new Float32Array([0, 0, -lineScale, 0, 0, lineScale]), [lineScale]);
  
  return (
    <group>
      <line>
        <bufferGeometry>
            <bufferAttribute attach="attributes-position" count={2} array={xAxis} itemSize={3} />
        </bufferGeometry>
        <lineBasicMaterial color="#ff4444" opacity={0.6} transparent={true} />
      </line>
      <line>
        <bufferGeometry>
            <bufferAttribute attach="attributes-position" count={2} array={yAxis} itemSize={3} />
        </bufferGeometry>
        <lineBasicMaterial color="#44ff44" opacity={0.6} transparent={true} />
      </line>
      <line>
        <bufferGeometry>
            <bufferAttribute attach="attributes-position" count={2} array={zAxis} itemSize={3} />
        </bufferGeometry>
        <lineBasicMaterial color="#4444ff" opacity={0.6} transparent={true} />
      </line>

      {/* Axis Labels - Using Billboard so they always face the camera */}
      <Billboard position={[lineScale + 1, 0, 0]}>
        <Text color="#ff4444" fontSize={size * 0.15}>X</Text>
      </Billboard>
      <Billboard position={[-lineScale - 1, 0, 0]}>
        <Text color="#ff4444" fontSize={size * 0.1}>-X</Text>
      </Billboard>
      
      <Billboard position={[0, lineScale + 1, 0]}>
        <Text color="#44ff44" fontSize={size * 0.15}>Y</Text>
      </Billboard>
      <Billboard position={[0, -lineScale - 1, 0]}>
        <Text color="#44ff44" fontSize={size * 0.1}>-Y</Text>
      </Billboard>

      <Billboard position={[0, 0, lineScale + 1]}>
        <Text color="#4444ff" fontSize={size * 0.15}>Z</Text>
      </Billboard>
      <Billboard position={[0, 0, -lineScale - 1]}>
        <Text color="#4444ff" fontSize={size * 0.1}>-Z</Text>
      </Billboard>
    </group>
  );
};

// --- Main App Component ---

const App = () => {
  const [currentPage, setCurrentPage] = useState('welcome'); // welcome, chemistry, howto, simulator

  // Global Quantum Controls
  const [Z, setZ] = useState(1);
  const [n, setN] = useState(3);
  const [l, setL] = useState(2);
  const [m, setM] = useState(0);
  
  // App Mode & Advanced
  const [mode, setMode] = useState('scatter'); 
  const [showPhase, setShowPhase] = useState(true);
  const [enableSimpleGlow, setEnableSimpleGlow] = useState(false);
  const [gridSize, setGridSize] = useState(100);
  
  // Scatter Controls
  const [scatterGridRes, setScatterGridRes] = useState(120);
  const [numPoints, setNumPoints] = useState(getPointCountForN(3));
  const [radialPeakDistance, setRadialPeakDistance] = useState(0);
  const [densityScale, setDensityScale] = useState(1.0);
  const [scatterOpacity, setScatterOpacity] = useState(0.8);
  
  // Isosurface Controls
  const [isoGridRes, setIsoGridRes] = useState(100);
  const [isovalue, setIsovalue] = useState(0.03); // Percentage mapped scale
  const [isoOpacity, setIsoOpacity] = useState(0.7);

  // Data States
  const [plotData, setPlotData] = useState(null);
  const [radialData, setRadialData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [renderKey, setRenderKey] = useState(0); // Force Three.js rerenders

  // Graph resizing state
  const [graphHeight, setGraphHeight] = useState(220);
  const pointRange = useMemo(() => getPointRangeForN(n), [n]);
  const pointSize = useMemo(() => getAutoPointSizeForDistance(radialPeakDistance), [radialPeakDistance]);

  const canvasDpr = useMemo(() => {
    const deviceDpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    const scatterDprCap = numPoints >= 30000000 ? 1.0 : numPoints >= 10000000 ? 1.15 : numPoints >= 5000000 ? 1.25 : 1.35;
    const maxDpr = mode === 'scatter' ? scatterDprCap : 2;
    return [1, Math.min(deviceDpr, maxDpr)];
  }, [mode, numPoints]);

  const API_BASE_URL = process.env.NODE_ENV === 'development' ? 'http://localhost:8000' : '';

  const parseApiResponse = async (response, endpointName) => {
    const raw = await response.text();

    if (!response.ok) {
      let detail = '';
      if (raw) {
        try {
          const parsed = JSON.parse(raw);
          detail = typeof parsed?.detail === 'string' ? parsed.detail : JSON.stringify(parsed);
        } catch {
          detail = raw;
        }
      }

      const statusSuffix = response.statusText ? ` ${response.statusText}` : '';
      throw new Error(detail || `${endpointName} request failed (${response.status}${statusSuffix}).`);
    }

    if (!raw) {
      throw new Error(`${endpointName} returned an empty response.`);
    }

    try {
      return JSON.parse(raw);
    } catch {
      throw new Error(`${endpointName} returned invalid JSON.`);
    }
  };

  const fetchData = async () => {
    setLoading(true);
    setErrorMsg("");
    
    try {
      // Fetch Radial Data
      const radialRes = await fetch(`${API_BASE_URL}/radial?Z=${Z}&n=${n}&l=${l}&size=${gridSize}`);
      const radialJson = await parseApiResponse(radialRes, 'Radial API');
      
      if (Array.isArray(radialJson.data)) {
        setRadialData(radialJson.data);
      }
      if (typeof radialJson.max_r === 'number' && Number.isFinite(radialJson.max_r)) {
        setRadialPeakDistance(radialJson.max_r);
      }

      // Fetch 3D Data
      const params = new URLSearchParams({
        Z, n, l, m,
        show_phase: showPhase,
        size: gridSize
      });

      let endpoint = '';
      if (mode === 'scatter') {
        params.append('resolution', scatterGridRes);
        params.append('num_points', numPoints);
        params.append('density_scale', densityScale);
        params.append('binary', 'true');
        endpoint = '/scatter';
      } else {
        params.append('resolution', isoGridRes);
        params.append('isovalue', isovalue);
        endpoint = '/isosurface';
      }

      const res3d = await fetch(`${API_BASE_URL}${endpoint}?${params.toString()}`);
      let data3d;
      if (mode === 'scatter') {
        if (!res3d.ok) {
          await parseApiResponse(res3d, '3D API');
        }

        const stride = Math.max(5, Number(res3d.headers.get('x-point-stride')) || 5);
        const binaryPayload = await res3d.arrayBuffer();
        const flatPoints = new Float32Array(binaryPayload);
        if (flatPoints.length % stride !== 0) {
          throw new Error('Scatter API returned malformed binary point data.');
        }
        data3d = { pointsFlat: flatPoints, pointStride: stride };
      } else {
        data3d = await parseApiResponse(res3d, '3D API');
      }
      
      // Warn if arrays are empty 
      if (mode === 'isosurface' && (!data3d.surfaces || data3d.surfaces.length === 0)) {
          setErrorMsg("No Isosurface found at this Threshold! Try lowering the Isovalue.");
      }
      
      setPlotData(data3d);
      setRenderKey(Date.now()); // Update the unique key to force remount the geometries

    } catch (err) {
      console.error(err);
      setErrorMsg(err instanceof Error ? err.message : 'Failed to fetch simulation data.');
    } finally {
      setLoading(false);
    }
  };

  const handleUpdate = (e) => {
    e.preventDefault();
    fetchData();
  };

  const startResize = (e) => {
    e.preventDefault();
    const startY = e.clientY;
    const startHeight = graphHeight;

    const onMouseMove = (moveEvent) => {
      const delta = startY - moveEvent.clientY;
      const newHeight = Math.max(100, Math.min(startHeight + delta, window.innerHeight * 0.8));
      setGraphHeight(newHeight);
    };

    const onMouseUp = () => {
      window.removeEventListener('mousemove', onMouseMove);
      window.removeEventListener('mouseup', onMouseUp);
    };

    window.addEventListener('mousemove', onMouseMove);
    window.addEventListener('mouseup', onMouseUp);
  };

  useEffect(() => {
    if (l >= n) setL(n - 1);
    // eslint-disable-next-line
  }, [n]);

  useEffect(() => {
    setNumPoints((prev) => clampPointCountForN(n, prev));
  }, [n]);

  useEffect(() => {
    if (Math.abs(m) > l) setM(0);
    // eslint-disable-next-line
  }, [l]);

  useEffect(() => {
    fetchData();
    // eslint-disable-next-line
  }, []); // Initial load

  return (
    <>
      {/* Top Navigation Bar */}
      <div className="top-nav">
        <h3>Quantum Orbital Explorer</h3>
        <div className="top-nav-buttons">
          <button onClick={() => setCurrentPage('welcome')} style={navButtonStyle(currentPage === 'welcome')}>Welcome & Math</button>
          <button onClick={() => setCurrentPage('chemistry')} style={navButtonStyle(currentPage === 'chemistry')}>Chemistry Concepts</button>
          <button onClick={() => setCurrentPage('howto')} style={navButtonStyle(currentPage === 'howto')}>How To Use</button>
          <button onClick={() => setCurrentPage('simulator')} style={navButtonStyle(currentPage === 'simulator')}><strong>Simulator</strong></button>
        </div>
      </div>

      {currentPage === 'welcome' && (
        <div style={pageStyle} className="page-style">
          <h1>Welcome to the Quantum Orbital Explorer</h1>
          <p>Welcome to an interactive journey into the quantum world of atoms.</p>
          <p>This platform is designed to help students, educators, chemistry enthusiasts, and researchers visualize one of the most important concepts in modern physics and chemistry: the spatial probability distribution of electrons around an atomic nucleus.</p>
          <p>Unlike classical physics, electrons do not move in fixed circular paths around the nucleus like planets orbiting the sun. Instead, quantum mechanics describes electrons as wave-like entities whose locations can only be predicted probabilistically.</p>
          <p>This simulator transforms those mathematical probability distributions into beautiful three-dimensional visual models so that abstract equations become visually intuitive.</p>
          
          <h2>The Mathematical Foundation</h2>
          <p>The entire simulation is based on the exact analytical solutions of the time-independent Schrödinger equation for hydrogen-like atoms.</p>
          <p>The equation is:</p>
          <div style={mathStyles}>
            <p><strong>Ĥψ = Eψ</strong></p>
          </div>
          <p>Here:</p>
          <ul>
            <li><strong>Ĥ</strong> is the Hamiltonian operator</li>
            <li><strong>ψ</strong> is the wavefunction</li>
            <li><strong>E</strong> is the energy eigenvalue</li>
          </ul>
          
          <p>For a single-electron atom, such as hydrogen, this equation can be solved exactly. Because atoms possess spherical symmetry, spherical coordinates are the natural coordinate system: <strong>(r, θ, φ)</strong></p>
          
          <p>The total wavefunction is written as:</p>
          <div style={mathStyles}>
            <p><strong>ψ<sub>n,l,m</sub>(r, θ, φ) = R<sub>n,l</sub>(r) &middot; Y<sub>l,m</sub>(θ, φ)</strong></p>
          </div>
          <p>This means the solution separates into two physically meaningful parts.</p>

          <h3>Radial Part: R<sub>n,l</sub>(r)</h3>
          <p>This determines how the electron probability changes as we move farther from the nucleus. It controls:</p>
          <ul>
            <li>orbital size</li>
            <li>radial nodes</li>
            <li>shell expansion</li>
          </ul>

          <h3>Angular Part: Y<sub>l,m</sub>(θ, φ)</h3>
          <p>These are spherical harmonics. They determine:</p>
          <ul>
            <li>shape of lobes</li>
            <li>orientation in 3D space</li>
            <li>nodal planes</li>
          </ul>

          <h2>Quantum Numbers</h2>
          <p>The wavefunction depends on three quantum numbers.</p>

          <h3>Principal Quantum Number (n)</h3>
          <p>This determines the main energy level. Higher <strong>n</strong> means:</p>
          <ul>
            <li>larger orbitals</li>
            <li>more nodes</li>
            <li>higher energy</li>
          </ul>
          <p>Examples: <strong>n=1</strong> &rarr; K shell | <strong>n=2</strong> &rarr; L shell | <strong>n=3</strong> &rarr; M shell</p>

          <h3>Azimuthal Quantum Number (l)</h3>
          <p>This defines the shape.</p>
          <ul>
            <li><strong>l=0</strong> &rarr; s orbital</li>
            <li><strong>l=1</strong> &rarr; p orbital</li>
            <li><strong>l=2</strong> &rarr; d orbital</li>
            <li><strong>l=3</strong> &rarr; f orbital</li>
          </ul>

          <h3>Magnetic Quantum Number (m)</h3>
          <p>This determines the orientation of the orbital in space. It changes how the orbital rotates and aligns along different axes.</p>

          <h2>Probability Density</h2>
          <p>The measurable quantity is not the wavefunction itself but its magnitude squared.</p>
          <div style={mathStyles}>
            <p><strong>P(r, θ, φ) = |ψ|<sup>2</sup></strong></p>
          </div>
          <p>This gives the probability density. Dense regions correspond to places where the electron is more likely to be found.</p>

          <h2>Why This Matters</h2>
          <p>This mathematical model forms the foundation of atomic structure, chemical bonding, spectroscopy, quantum chemistry, molecular orbital theory, and periodic table trends.</p>
          <p>Every modern chemistry concept ultimately originates from these equations.</p>
          
          <button onClick={() => setCurrentPage('chemistry')} style={nextButton}>Continue to Chemistry Concepts</button>
        </div>
      )}

      {currentPage === 'chemistry' && (
        <div style={pageStyle} className="page-style">
          <h1>Chemistry Concepts</h1>
          <p>Atomic orbitals are one of the most important concepts in chemistry. They explain how atoms bond, why molecules take specific shapes, and why different elements behave differently.</p>
          <p>Orbitals are NOT physical paths. They are probability regions derived from the quantum mechanical wavefunction. These regions indicate where electrons are most likely to be found.</p>

          <h2>s Orbitals</h2>
          <p>s orbitals correspond to: <strong>l=0</strong></p>
          <p>These are spherical. Because they are perfectly symmetric, they play a major role in the hydrogen atom, alkali metals, and sigma bonding. Examples: 1s, 2s, 3s. Higher s orbitals contain radial nodes.</p>

          <h2>p Orbitals</h2>
          <p>p orbitals correspond to: <strong>l=1</strong></p>
          <p>These are dumbbell shaped. They contain one angular nodal plane. There are three orientations: <strong>p<sub>x</sub></strong>, <strong>p<sub>y</sub></strong>, <strong>p<sub>z</sub></strong>.</p>
          <p>These orbitals are fundamental for covalent bonding, pi bonds, and molecular geometry.</p>

          <h2>d Orbitals</h2>
          <p>d orbitals correspond to: <strong>l=2</strong></p>
          <p>These are more complex. Common shapes include four-lobed clover structures and dumbbell plus torus ring structures.</p>
          <p>These are extremely important in transition metal chemistry, crystal field splitting, catalysts, and magnetic materials.</p>

          <h2>f Orbitals</h2>
          <p>f orbitals correspond to: <strong>l=3</strong></p>
          <p>These are highly complex multi-lobed structures. They are essential in lanthanides, actinides, and rare earth chemistry. These explain the unique electronic and magnetic properties of heavy elements.</p>

          <h2>Nodes</h2>
          <p>Nodes are regions where the probability is zero. At these points: <strong>ψ = 0</strong>. No electron can exist there. There are two major types: radial nodes and angular nodes.</p>

          <h2>Wavefunction Phase</h2>
          <p>The wavefunction has sign. Positive and negative signs represent phase. In your simulator:</p>
          <ul>
            <li><strong>Red</strong> = positive phase</li>
            <li><strong>Blue</strong> = negative phase</li>
          </ul>
          <p>This is extremely important in chemistry because orbital overlap depends on phase. Constructive overlap leads to bonding. Destructive overlap leads to antibonding orbitals.</p>

          <h2>Applications in Chemistry</h2>
          <p>These concepts explain sigma bonds, pi bonds, hybridization, molecular orbital theory, spectroscopy, and reaction mechanisms.</p>
          <p><strong>Without orbitals, modern chemistry cannot be understood.</strong></p>
          
          <button onClick={() => setCurrentPage('howto')} style={nextButton}>Continue to How To Use</button>
        </div>
      )}

      {currentPage === 'howto' && (
        <div style={pageStyle} className="page-style">
          <h1>How To Use This App</h1>
          <h2>1. Select Quantum Numbers</h2>
          <p>Use the sliders on the left panel to pick a valid hydrogenic state. Remember that <strong>l</strong> must be strictly less than <strong>n</strong>, and <strong>m</strong> is bounded between <strong>-l</strong> and <strong>l</strong>. You can also adjust the atomic number <strong>Z</strong> to see the orbital shrinkage of heavier atomic nuclei like Helium (<strong>Z = 2</strong>) or Carbon (<strong>Z = 6</strong>).</p>
          
          <h2>2. Visualization Modes</h2>
          <ul>
            <li><strong>Probabilistic Scatter:</strong> Uses a Monte Carlo simulation to randomly place dots proportional to the mathematical probability density. High density means more dots.</li>
            <li><strong>Isosurface Mode:</strong> Draws a solid 3D mesh boundary containing all regions where the probability is greater than your selected threshold boundary. This looks like a solid balloon.</li>
          </ul>

          <h2>3. Advanced Controls</h2>
          <p>Toggle phase mapping to see positive and negative regions of the wavefunction.</p>
          <p>You can adjust grid sizes for larger orbitals (like <strong>n = 7</strong>), or increase resolution up to 200 for incredibly smooth 3D structures.</p>
          
          <h2>4. Changing Views</h2>
          <p>Use your mouse to interact with the 3D model:</p>
          <ul>
            <li><strong>Left-Click & Drag:</strong> Rotate the orbital</li>
            <li><strong>Right-Click & Drag:</strong> Pan / move the target</li>
            <li><strong>Scroll:</strong> Zoom in and out</li>
          </ul>
          
          <button onClick={() => setCurrentPage('simulator')} style={{...nextButton, backgroundColor: '#28a745'}}>Launch Simulator 🚀</button>
        </div>
      )}

      {currentPage === 'simulator' && (
        <div className="App">
          
          {/* Sidebar Controls */}
      <div className="controls-sidebar">
        <h3 style={{marginTop: 0}}>Quantum Simulator</h3>
        
        <form onSubmit={handleUpdate} style={{display: 'flex', flexDirection: 'column', gap: '10px'}}>
          <div className="control-group">
            <label>Principal (n): {n} <input type="range" value={n} min={1} max={10} onChange={e => setN(parseInt(e.target.value))} /></label>
            <label>Azimuthal (l): {l} <input type="range" value={l} min={0} max={n-1} onChange={e => setL(parseInt(e.target.value))} /></label>
            <label>Magnetic (m): {m} <input type="range" value={m} min={-l} max={l} onChange={e => setM(parseInt(e.target.value))} /></label>
            <label>Target (Z): {Z} <input type="range" value={Z} min={1} max={20} onChange={e => setZ(parseInt(e.target.value))} /></label>
          </div>

          <hr style={{ borderColor: '#444' }}/>

          <div className="control-group">
            <label>
              <input type="radio" value="scatter" checked={mode === 'scatter'} onChange={() => setMode('scatter')} />
              Probabilistic Scatter
            </label><br/>
            <label>
              <input type="radio" value="isosurface" checked={mode === 'isosurface'} onChange={() => setMode('isosurface')} />
              Isosurface Mode
            </label>
          </div>

          <hr style={{ borderColor: '#444' }}/>

          <details style={{ marginBottom: '10px' }}>
            <summary style={{ cursor: 'pointer', fontWeight: 'bold', color: '#00aaff', marginBottom: '10px' }}>Advanced Controls</summary>
            <div style={{ marginLeft: '10px', display: 'flex', flexDirection: 'column', gap: '10px' }}>
              <div className="control-group">
                <label>
                  <input type="checkbox" checked={showPhase} onChange={e => setShowPhase(e.target.checked)} /> Show Wavefunction Phase
                </label>
                <label>
                  <input type="checkbox" checked={enableSimpleGlow} onChange={e => setEnableSimpleGlow(e.target.checked)} /> Enable Simple Glow
                </label>
                <label>Grid Size: {gridSize} <input type="range" value={gridSize} min={5} max={100} onChange={e => setGridSize(parseInt(e.target.value))} /></label>
              </div>

              <hr style={{ borderColor: '#444', margin: '5px 0' }}/>

              {mode === 'scatter' ? (
                <div className="control-group">
                  <span style={{color:'#aaa', fontSize:'0.9rem'}}>Scatter Params</span>
                  <label>Resolution: {scatterGridRes} <input type="range" value={scatterGridRes} min={30} max={150} onChange={e => setScatterGridRes(parseInt(e.target.value))} /></label>
                  <label>Points: {numPoints.toLocaleString()} <input type="range" value={numPoints} min={pointRange.min} max={pointRange.max} step={pointRange.step} onChange={e => setNumPoints(parseInt(e.target.value, 10))} /></label>
                  <span style={{color:'#777', fontSize:'0.8rem'}}>Allowed for n={n}: {pointRange.min.toLocaleString()} - {pointRange.max.toLocaleString()}</span>
                  <span style={{color:'#777', fontSize:'0.8rem'}}>Auto Point Size: {pointSize.toFixed(2)} (+{POINT_SIZE_STEP_INCREMENT.toFixed(2)} every {POINT_SIZE_DISTANCE_STEP} units, max {MAX_AUTO_POINT_SIZE.toFixed(2)})</span>
                  <label>Density Pow: {densityScale.toFixed(1)} <input type="range" value={densityScale} min={0.1} max={2.0} step={0.1} onChange={e => setDensityScale(parseFloat(e.target.value))} /></label>
                  <label>Opacity: {scatterOpacity.toFixed(1)} <input type="range" value={scatterOpacity} min={0.1} max={1.0} step={0.1} onChange={e => setScatterOpacity(parseFloat(e.target.value))} /></label>
                </div>
              ) : (
                <div className="control-group">
                  <span style={{color:'#aaa', fontSize:'0.9rem'}}>Isosurface Params</span>
                  <label>Resolution: {isoGridRes} <input type="range" value={isoGridRes} min={40} max={200} onChange={e => setIsoGridRes(parseInt(e.target.value))} /></label>
                  <label>Isovalue Threshold (%): {(isovalue * 100).toFixed(1)}% <input type="range" value={isovalue} min={0.01} max={0.99} step={0.01} onChange={e => setIsovalue(parseFloat(e.target.value))} /></label>
                  <label>Opacity: {isoOpacity.toFixed(1)} <input type="range" value={isoOpacity} min={0.1} max={1.0} step={0.1} onChange={e => setIsoOpacity(parseFloat(e.target.value))} /></label>
                </div>
              )}
            </div>
          </details>

          <button type="submit" style={{ padding: '10px', backgroundColor: '#007bff', color: 'white', border: 'none', borderRadius: '4px', cursor: 'pointer', fontWeight: 'bold' }}>
            {loading ? 'Computing...' : 'Apply Changes'}
          </button>

          {/* Ad Placeholder for Sidebar */}
          <div className="google-ad-placeholder" style={{ marginTop: '20px', width: '100%', minHeight: '350px', backgroundColor: '#333', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px dashed #555' }}>
            <span style={{ color: '#aaa', fontSize: '0.9rem' }}>Google Ad Slot</span>
          </div>

          <div style={{minHeight: '40px', marginTop: '10px'}}>
             {errorMsg && <p style={{color: '#ff6b6b', fontSize: '0.85rem', margin: 0}}>{errorMsg}</p>}
          </div>
        </form>
      </div>

      {/* Main Render Area */}
      <div className="main-content" style={{ display: 'flex', flexDirection: 'column', height: '100vh', overflow: 'hidden' }}>
          <div className="canvas-container" style={{ flex: 1, minHeight: 0 }}>
            <Canvas dpr={canvasDpr} gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }} performance={{ min: 0.6 }} camera={{ position: [0, 0, gridSize * 1.5], fov: 60 }}>
                <color attach="background" args={['#111111']} />
                <ambientLight intensity={0.5} />
                <pointLight position={[100, 100, 100]} intensity={1} />
                <pointLight position={[-100, -100, -100]} intensity={0.5} />
                
                {plotData && (mode === 'scatter' ? (
                  <ScatterPlot key={renderKey} data={plotData} pointSize={pointSize} opacity={scatterOpacity} showPhase={showPhase} enableSimpleGlow={enableSimpleGlow} />
                ) : (
                  <IsosurfaceMesh key={renderKey} data={plotData} opacity={isoOpacity} showPhase={showPhase} isovalue={isovalue} />
                ))}
                
                <FullAxes size={gridSize} />
                <OrbitControls 
                  makeDefault
                  enablePan={true} 
                  enableZoom={true} 
                  enableRotate={true} 
                  autoRotate={true} 
                  autoRotateSpeed={0.5} 
                  panSpeed={1.5}
                />
            </Canvas>
          </div>

          <div style={{ height: '30px', flexShrink: 0, backgroundColor: '#111', color: '#888', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem' }}>
            <span><strong style={{color:'#ccc'}}>Controls:</strong> Left-Click to Rotate &bull; Right-Click to Pan/Move Target &bull; Scroll to Zoom</span>
          </div>

          <div
            className="resize-handle"
            onMouseDown={startResize}
            style={{
              height: '8px',
              backgroundColor: '#444',
              cursor: 'row-resize',
              display: 'flex',
              justifyContent: 'center',
              alignItems: 'center',
              borderTop: '1px solid #222',
              borderBottom: '1px solid #222',
              zIndex: 10
            }}
          >
            <div style={{ width: '40px', height: '2px', backgroundColor: '#888', borderRadius: '1px' }}></div>
          </div>

          <div className="radial-graph-container" style={{ height: `${graphHeight}px`, flexShrink: 0, overflow: 'hidden', backgroundColor: '#1a1a1a', padding: '10px', display: 'flex', flexDirection: 'row' }}>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column', minHeight: 0 }}>
              <h4 style={{ margin: '0 0 10px 0', textAlign: 'center', color: '#ccc', flexShrink: 0 }}>Radial Probability Distribution</h4>
              <ResponsiveContainer width="100%" height="80%" style={{ flex: 1, minHeight: 0 }}>
                  <LineChart data={radialData} margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis dataKey="r" stroke="#ccc" tick={{fill: '#ccc'}} tickFormatter={(val) => Number(val).toFixed(1)} tickCount={5} minTickGap={40} />
                  <YAxis stroke="#ccc" tick={{fill: '#ccc'}} tickFormatter={(val) => Number(val).toFixed(2)} tickCount={5} />
                  <Tooltip contentStyle={{ backgroundColor: '#222', border: '1px solid #444', color: '#fff' }} formatter={(val) => Number(val).toFixed(3)} />
                  <Legend />
                  <Line type="monotone" dataKey="P" stroke="#00aaff" dot={false} strokeWidth={2} name="Probability P(r)" />
                  </LineChart>
              </ResponsiveContainer>
            </div>
            
            {!showPhase && (
              <div className="density-container" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', width: '80px', marginLeft: '10px' }}>
                <div style={{ fontSize: '0.8rem', color: '#ccc', marginBottom: '10px', flexShrink: 0 }}>Density</div>
                <div className="density-bar" style={{ display: 'flex', flexDirection: 'row', height: '80%', minHeight: 0 }}>
                  <div className="density-labels" style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', marginRight: '8px', fontSize: '0.75rem', color: '#aaa' }}>
                    <span>100</span>
                    <span>80</span>
                    <span>60</span>
                    <span>40</span>
                    <span>20</span>
                    <span>0</span>
                  </div>
                  <div className="density-gradient" style={{ width: '15px', background: 'linear-gradient(to bottom, #ff8c00, #c51b7d, #30005c)', borderRadius: '3px', height: '100%' }}></div>
                </div>
              </div>
            )}
          </div>
      </div>
    </div>
      )}
    </>
  );
};

// Simple Styles for Info Pages
const pageStyle = {
  padding: '40px',
  color: '#fff',
  maxWidth: '800px',
  margin: '0 auto',
  fontFamily: 'sans-serif',
  lineHeight: '1.6'
};

const mathStyles = {
  backgroundColor: '#222',
  padding: '15px',
  borderRadius: '5px',
  fontFamily: 'monospace',
  fontSize: '1.2rem',
  color: '#00aaff',
  textAlign: 'center',
  margin: '20px 0'
};

const nextButton = {
  marginTop: '30px',
  padding: '12px 24px',
  backgroundColor: '#007bff',
  color: '#fff',
  border: 'none',
  borderRadius: '4px',
  fontSize: '1rem',
  cursor: 'pointer',
  fontWeight: 'bold'
};

// Reusable styling for the navigation buttons
const navButtonStyle = (isActive) => ({
  background: 'none',
  border: 'none',
  color: isActive ? '#00aaff' : '#ccc',
  fontSize: '1rem',
  cursor: 'pointer',
  padding: '5px 10px',
  borderBottom: isActive ? '2px solid #00aaff' : '2px solid transparent'
});

export default App;
