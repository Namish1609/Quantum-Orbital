import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Billboard } from '@react-three/drei';
import * as THREE from 'three';
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer
} from 'recharts';
import { InlineMath, BlockMath } from 'react-katex';
import 'katex/dist/katex.min.css';
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

const PHASE_POS_COLOR = new THREE.Color('#ff1f1f');
const PHASE_NEG_COLOR = new THREE.Color('#1a2bff');
const DENSITY_C1 = new THREE.Color('#30005c');
const DENSITY_C2 = new THREE.Color('#c51b7d');
const DENSITY_C3 = new THREE.Color('#ff8c00');
const MAX_TOTAL_POINTS = 50000000;
const PHASE_INTENSITY_HARDLIMIT = 0.2;
const SCATTER_OPACITY = 1.0;
const FIXED_GRID_SIZE = 200;
const POINTS_PER_N_STEP = 5000000;
const POINT_SIZE_MIN_PX = 1.5;
const POINT_SIZE_MAX_PX = 8.0;
const POINT_SIZE_SCREEN_RADIUS_FACTOR = 0.002;
const POINT_SIZE_HIGH_N_MIN_PX = 1.75;
const POINT_SIZE_HIGH_N_BOOST_CAP_PX = 0.25;
const MAX_RENDER_POINTS_DEFAULT = 1800000;
const MAX_RENDER_POINTS_HIGH_N = 1200000;
const MAX_SIMULTANEOUS_SLICE_AXES = 2;
const SLICE_AXIS_ORDER = ['x', 'y', 'z'];
const GRAPH_RETRACE_DURATION_MS = 320;
const ORBITAL_CENTER = new THREE.Vector3(0, 0, 0);

const getSliceClippingPlane = (axis, centerOffset = 0) => {
  const center = Number.isFinite(centerOffset) ? centerOffset : 0;
  if (axis === 'x') {
    return new THREE.Plane(new THREE.Vector3(-1, 0, 0), center);
  }
  if (axis === 'y') {
    return new THREE.Plane(new THREE.Vector3(0, -1, 0), center);
  }
  return new THREE.Plane(new THREE.Vector3(0, 0, -1), center);
};

const SlicePlaneVisual = ({ axis, offset, extent }) => {
  const safeOffset = Number.isFinite(offset) ? offset : 0;
  const safeExtent = Math.max(4, extent);
  const transform = useMemo(() => {
    if (axis === 'x') {
      return {
        position: [safeOffset, 0, 0],
        rotation: [0, Math.PI / 2, 0],
      };
    }
    if (axis === 'y') {
      return {
        position: [0, safeOffset, 0],
        rotation: [Math.PI / 2, 0, 0],
      };
    }
    return {
      position: [0, 0, safeOffset],
      rotation: [0, 0, 0],
    };
  }, [axis, safeOffset]);

  return (
    <mesh position={transform.position} rotation={transform.rotation}>
      <planeGeometry args={[safeExtent, safeExtent]} />
      <meshBasicMaterial color="#3ec5ff" transparent={true} opacity={0.08} side={THREE.DoubleSide} depthWrite={false} toneMapped={false} />
    </mesh>
  );
};

const getPointCountForN = (nValue) => {
  const scaled = Math.max(1, nValue) * POINTS_PER_N_STEP;
  return Math.min(MAX_TOTAL_POINTS, scaled);
};

const getRenderPointBudget = (nValue) => {
  const cores = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 8 : 8;
  const baseBudget = nValue >= 8 ? MAX_RENDER_POINTS_HIGH_N : MAX_RENDER_POINTS_DEFAULT;
  if (cores <= 4) return Math.floor(baseBudget * 0.7);
  return baseBudget;
};

const getProjectedOrbitalRadiusPx = (camera, viewportHeight, orbitalRadius) => {
  if (!Number.isFinite(orbitalRadius) || orbitalRadius <= 0 || viewportHeight <= 0) {
    return 0;
  }

  if (camera?.isPerspectiveCamera) {
    const distance = Math.max(0.01, camera.position.distanceTo(ORBITAL_CENTER));
    const fovRad = THREE.MathUtils.degToRad(camera.fov || 60);
    const worldToPixels = viewportHeight / (2 * Math.tan(fovRad / 2) * distance);
    return orbitalRadius * worldToPixels;
  }

  if (camera?.isOrthographicCamera) {
    const worldHeight = Math.max(0.01, camera.top - camera.bottom);
    const worldToPixels = viewportHeight / worldHeight;
    return orbitalRadius * worldToPixels;
  }

  return 0;
};

// --- Reusable Three.js Components ---

const ScatterPlot = ({ data, opacity, showPhase, enableSimpleGlow, nValue, sliceEnabled, selectedSliceAxes, sliceOffsets }) => {
  const intensityScale = Math.max(0, Math.min(1, opacity));
  const materialRef = useRef(null);
  const clippingPlanes = useMemo(() => {
    if (!sliceEnabled || !Array.isArray(selectedSliceAxes) || selectedSliceAxes.length === 0) return [];
    return selectedSliceAxes.map((axis) => getSliceClippingPlane(axis, sliceOffsets?.[axis] ?? 0));
  }, [sliceEnabled, selectedSliceAxes, sliceOffsets]);

  const { positions, colors, orbitalRadius } = useMemo(() => {
    const hasFlatPoints = !!data && data.pointsFlat instanceof Float32Array && data.pointsFlat.length > 0;
    const hasNestedPoints = !!data && Array.isArray(data.points) && data.points.length > 0;
    if (!hasFlatPoints && !hasNestedPoints) return { positions: null, colors: null, orbitalRadius: 0 };

    const pointStride = hasFlatPoints ? Math.max(5, Number(data.pointStride) || 5) : 5;
    const sourceCount = hasFlatPoints ? Math.floor(data.pointsFlat.length / pointStride) : data.points.length;
    const renderBudget = Math.max(250000, getRenderPointBudget(nValue));
    const sampleStep = Math.max(1, Math.ceil(sourceCount / renderBudget));
    const renderCount = Math.ceil(sourceCount / sampleStep);
    const pos = new Float32Array(renderCount * 3);
    const col = new Float32Array(renderCount * 3);

    const phasePosR = PHASE_POS_COLOR.r;
    const phasePosG = PHASE_POS_COLOR.g;
    const phasePosB = PHASE_POS_COLOR.b;
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
    let maxRadius = 0;

    for (let srcIndex = 0, renderIndex = 0; srcIndex < sourceCount; srcIndex += sampleStep, renderIndex++) {
        let x;
        let y;
        let z;
        let density;
        let phase;
        if (hasFlatPoints) {
          const base = srcIndex * pointStride;
          x = data.pointsFlat[base + 0];
          y = data.pointsFlat[base + 1];
          z = data.pointsFlat[base + 2];
          density = data.pointsFlat[base + 3];
          phase = data.pointsFlat[base + 4];
        } else {
          const point = data.points[srcIndex];
          x = point[0];
          y = point[1];
          z = point[2];
          density = point[3];
          phase = point[4];
        }
        const idx = renderIndex * 3;
        const r = Math.sqrt(x*x + y*y + z*z);
        if (r > maxRadius) maxRadius = r;

        pos[idx + 0] = x;
        pos[idx + 1] = y;
        pos[idx + 2] = z;

        if (showPhase) {
          const distanceFade = Math.max(0.1, 30.0 / (r + 30.0));
          const intensity = Math.max(PHASE_INTENSITY_HARDLIMIT, density * 5) * distanceFade * intensityScale;
          const baseR = phase > 0 ? phasePosR : phaseNegR;
          const baseG = phase > 0 ? phasePosG : phaseNegG;
          const baseB = phase > 0 ? phasePosB : phaseNegB;

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
    return { positions: pos, colors: col, orbitalRadius: maxRadius };
  }, [data, showPhase, intensityScale, nValue]);

  useFrame((state) => {
    if (!materialRef.current) return;
    const projectedRadiusPx = getProjectedOrbitalRadiusPx(state.camera, state.size.height, orbitalRadius);
    const minSizePx = nValue > 5 ? POINT_SIZE_HIGH_N_MIN_PX : POINT_SIZE_MIN_PX;
    const highNBoostPx = nValue > 5
      ? Math.min(POINT_SIZE_HIGH_N_BOOST_CAP_PX, (nValue - 5) * 0.05)
      : 0;
    const adaptiveSize = THREE.MathUtils.clamp(
      projectedRadiusPx * POINT_SIZE_SCREEN_RADIUS_FACTOR + highNBoostPx,
      minSizePx,
      POINT_SIZE_MAX_PX
    );
    materialRef.current.size = adaptiveSize;
  });

  if (!positions) return null;

  return (
    <points frustumCulled={false}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={positions.length / 3}
          array={positions}
          itemSize={3}
          usage={THREE.StaticDrawUsage}
        />
        <bufferAttribute
          attach="attributes-color"
          count={colors.length / 3}
          array={colors}
          itemSize={3}
          usage={THREE.StaticDrawUsage}
        />
      </bufferGeometry>
      {enableSimpleGlow ? (
        <pointsMaterial
          ref={materialRef}
          size={nValue > 5 ? POINT_SIZE_HIGH_N_MIN_PX : POINT_SIZE_MIN_PX}
          vertexColors={true}
          transparent={true}
          opacity={Math.min(1, 0.35 + opacity * 0.55)}
          sizeAttenuation={false}
          clippingPlanes={clippingPlanes}
          map={circleTexture}
          alphaTest={0.2}
          blending={THREE.AdditiveBlending}
          depthWrite={false}
          depthTest={true}
          toneMapped={false}
        />
      ) : (
        <pointsMaterial
          ref={materialRef}
          size={nValue > 5 ? POINT_SIZE_HIGH_N_MIN_PX : POINT_SIZE_MIN_PX}
          vertexColors={true}
          transparent={false}
          opacity={1}
          sizeAttenuation={false}
          clippingPlanes={clippingPlanes}
          map={circleTexture}
          alphaTest={0.2}
          depthWrite={true}
          depthTest={true}
          toneMapped={false}
        />
      )}
    </points>
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
  
  // Visual Controls
  const [showPhase, setShowPhase] = useState(true);
  const [enableSimpleGlow, setEnableSimpleGlow] = useState(false);
  const [showAxes, setShowAxes] = useState(true);
  const [showSliceView, setShowSliceView] = useState(false);
  const [selectedSliceAxes, setSelectedSliceAxes] = useState(['x']);
  const [sliceOffsets, setSliceOffsets] = useState({ x: 0, y: 0, z: 0 });
  const [isSliceSliderDragging, setIsSliceSliderDragging] = useState(false);
  const [activeSliceDragAxis, setActiveSliceDragAxis] = useState(null);
  const gridSize = FIXED_GRID_SIZE;
  
  // Scatter Controls
  const numPoints = useMemo(() => getPointCountForN(n), [n]);

  // Data States
  const [plotData, setPlotData] = useState(null);
  const [animatedRadialData, setAnimatedRadialData] = useState([]);
  const [loading, setLoading] = useState(false);
  const [errorMsg, setErrorMsg] = useState("");
  const [renderKey, setRenderKey] = useState(0); // Force Three.js rerenders
  const graphRetreatTimerRef = useRef(null);

  // Graph resizing state
  const [graphHeight, setGraphHeight] = useState(220);
  const sliceOffsetLimit = useMemo(() => Math.max(2, gridSize * 0.95), [gridSize]);
  const slicePlaneExtent = useMemo(() => Math.max(10, gridSize * 2.4), [gridSize]);

  const updateSliceOffsetForAxis = (axis, value) => {
    const numeric = Number(value);
    if (!Number.isFinite(numeric)) return;
    setSliceOffsets((prev) => ({
      ...prev,
      [axis]: THREE.MathUtils.clamp(numeric, -sliceOffsetLimit, sliceOffsetLimit),
    }));
  };

  const toggleSliceAxis = (axis) => {
    setSelectedSliceAxes((prev) => {
      if (prev.includes(axis)) {
        if (prev.length === 1) return prev;
        return prev.filter((item) => item !== axis);
      }
      if (prev.length >= MAX_SIMULTANEOUS_SLICE_AXES) return prev;
      const next = [...prev, axis];
      next.sort((a, b) => SLICE_AXIS_ORDER.indexOf(a) - SLICE_AXIS_ORDER.indexOf(b));
      return next;
    });
  };

  const animateGraphTransition = (nextData) => {
    if (!Array.isArray(nextData)) return;
    if (graphRetreatTimerRef.current) {
      clearTimeout(graphRetreatTimerRef.current);
      graphRetreatTimerRef.current = null;
    }

    setAnimatedRadialData((prev) => {
      if (prev.length > 1) {
        const retreatData = [...prev].reverse();
        graphRetreatTimerRef.current = setTimeout(() => {
          setAnimatedRadialData(nextData);
          graphRetreatTimerRef.current = null;
        }, GRAPH_RETRACE_DURATION_MS);
        return retreatData;
      }
      return nextData;
    });
  };

  const canvasDpr = useMemo(() => {
    const deviceDpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
    const scatterDprCap = numPoints >= 30000000 ? 1.0 : numPoints >= 10000000 ? 1.15 : numPoints >= 5000000 ? 1.25 : 1.35;
    const dprMin = numPoints >= 30000000 ? 0.75 : 1;
    return [dprMin, Math.min(deviceDpr, scatterDprCap)];
  }, [numPoints]);

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
        animateGraphTransition(radialJson.data);
      }

      // Fetch 3D Data
      const params = new URLSearchParams({
        Z, n, l, m,
        show_phase: showPhase,
        size: gridSize
      });

      params.append('binary', 'true');

      const res3d = await fetch(`${API_BASE_URL}/scatter?${params.toString()}`);
      if (!res3d.ok) {
        await parseApiResponse(res3d, 'Scatter API');
      }

      const stride = Math.max(5, Number(res3d.headers.get('x-point-stride')) || 5);
      const binaryPayload = await res3d.arrayBuffer();
      const flatPoints = new Float32Array(binaryPayload);
      if (flatPoints.length % stride !== 0) {
        throw new Error('Scatter API returned malformed binary point data.');
      }
      const data3d = { pointsFlat: flatPoints, pointStride: stride };
      
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
    if (Math.abs(m) > l) setM(0);
    // eslint-disable-next-line
  }, [l]);

  useEffect(() => {
    if (activeSliceDragAxis && !selectedSliceAxes.includes(activeSliceDragAxis)) {
      setActiveSliceDragAxis(null);
      setIsSliceSliderDragging(false);
    }
  }, [selectedSliceAxes, activeSliceDragAxis]);

  useEffect(() => {
    return () => {
      if (graphRetreatTimerRef.current) {
        clearTimeout(graphRetreatTimerRef.current);
      }
    };
  }, []);

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

      {(currentPage === 'welcome' || currentPage === 'chemistry' || currentPage === 'howto') && (
        <>
          <div className="info-side-ad info-side-ad-left" aria-label="Left ad panel">
            <div className="info-side-ad-title">Ad Space</div>
            <div className="info-side-ad-body">Sponsored Panel</div>
          </div>
          <div className="info-side-ad info-side-ad-right" aria-label="Right ad panel">
            <div className="info-side-ad-title">Ad Space</div>
            <div className="info-side-ad-body">Sponsored Panel</div>
          </div>
        </>
      )}

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
            <BlockMath math={'\\hat{H}\\psi = E\\psi'} />
          </div>
          <p>Here:</p>
          <ul>
            <li><InlineMath math={'\\hat{H}'} /> is the Hamiltonian operator</li>
            <li><InlineMath math={'\\psi'} /> is the wavefunction</li>
            <li><InlineMath math={'E'} /> is the energy eigenvalue</li>
          </ul>
          
          <p>
            For a single-electron atom, such as hydrogen, this equation can be solved exactly. Because atoms possess spherical symmetry,
            spherical coordinates are the natural coordinate system: <InlineMath math={'(r,\\theta,\\phi)'} />
          </p>
          
          <p>The total wavefunction is written as:</p>
          <div style={mathStyles}>
            <BlockMath math={'\\psi_{n,\\ell,m}(r,\\theta,\\phi)=R_{n,\\ell}(r)\\,Y_{\\ell,m}(\\theta,\\phi)'} />
          </div>
          <p>This means the solution separates into two physically meaningful parts.</p>

          <h3>Radial Part: <InlineMath math={'R_{n,\\ell}(r)'} /></h3>
          <p>This determines how the electron probability changes as we move farther from the nucleus. It controls:</p>
          <ul>
            <li>orbital size</li>
            <li>radial nodes</li>
            <li>shell expansion</li>
          </ul>

          <h3>Angular Part: <InlineMath math={'Y_{\\ell,m}(\\theta,\\phi)'} /></h3>
          <p>These are spherical harmonics. They determine:</p>
          <ul>
            <li>shape of lobes</li>
            <li>orientation in 3D space</li>
            <li>nodal planes</li>
          </ul>

          <h2>Quantum Numbers</h2>
          <p>The wavefunction depends on three quantum numbers.</p>

          <h3>Principal Quantum Number (<InlineMath math={'n'} />)</h3>
          <p>This determines the main energy level. Higher <strong>n</strong> means:</p>
          <ul>
            <li>larger orbitals</li>
            <li>more nodes</li>
            <li>higher energy</li>
          </ul>
          <p><InlineMath math={'n=1\\to\\text{K shell},\\;n=2\\to\\text{L shell},\\;n=3\\to\\text{M shell}'} /></p>

          <h3>Azimuthal Quantum Number (<InlineMath math={'\\ell'} />)</h3>
          <p>This defines the shape.</p>
          <ul>
            <li><InlineMath math={'\\ell=0'} /> &rarr; s orbital</li>
            <li><InlineMath math={'\\ell=1'} /> &rarr; p orbital</li>
            <li><InlineMath math={'\\ell=2'} /> &rarr; d orbital</li>
            <li><InlineMath math={'\\ell=3'} /> &rarr; f orbital</li>
          </ul>

          <h3>Magnetic Quantum Number (<InlineMath math={'m'} />)</h3>
          <p>This determines the orientation of the orbital in space. It changes how the orbital rotates and aligns along different axes.</p>

          <h2>Probability Density</h2>
          <p>The measurable quantity is not the wavefunction itself but its magnitude squared.</p>
          <div style={mathStyles}>
            <BlockMath math={'P(r,\\theta,\\phi)=|\\psi|^2'} />
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
          <p>s orbitals correspond to: <InlineMath math={'\\ell=0'} /></p>
          <p>These are spherical. Because they are perfectly symmetric, they play a major role in the hydrogen atom, alkali metals, and sigma bonding. Examples: 1s, 2s, 3s. Higher s orbitals contain radial nodes.</p>

          <h2>p Orbitals</h2>
          <p>p orbitals correspond to: <InlineMath math={'\\ell=1'} /></p>
          <p>These are dumbbell shaped. They contain one angular nodal plane. There are three orientations: <InlineMath math={'p_x,\\;p_y,\\;p_z'} />.</p>
          <p>These orbitals are fundamental for covalent bonding, pi bonds, and molecular geometry.</p>

          <h2>d Orbitals</h2>
          <p>d orbitals correspond to: <InlineMath math={'\\ell=2'} /></p>
          <p>These are more complex. Common shapes include four-lobed clover structures and dumbbell plus torus ring structures.</p>
          <p>These are extremely important in transition metal chemistry, crystal field splitting, catalysts, and magnetic materials.</p>

          <h2>f Orbitals</h2>
          <p>f orbitals correspond to: <InlineMath math={'\\ell=3'} /></p>
          <p>These are highly complex multi-lobed structures. They are essential in lanthanides, actinides, and rare earth chemistry. These explain the unique electronic and magnetic properties of heavy elements.</p>

          <h2>Nodes</h2>
          <p>Nodes are regions where the probability is zero. At these points: <InlineMath math={'\\psi=0'} />. No electron can exist there. There are two major types: radial nodes and angular nodes.</p>

          <h2>Wavefunction Phase</h2>
          <p>The wavefunction <InlineMath math={'\\psi'} /> has sign. Positive and negative signs represent phase. In your simulator:</p>
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
          <p>
            Use the sliders on the left panel to pick a valid hydrogenic state. Remember that <InlineMath math={'\\ell<n'} />, and
            <InlineMath math={'-\\ell\\le m\\le \\ell'} />. You can also adjust the atomic number <InlineMath math={'Z'} /> to see
            the orbital shrinkage of heavier atomic nuclei like Helium (<InlineMath math={'Z=2'} />) or Carbon (<InlineMath math={'Z=6'} />).
          </p>
          
          <h2>2. Visualization Modes</h2>
          <ul>
            <li><strong>Probabilistic Scatter:</strong> Uses a Monte Carlo simulation to randomly place dots proportional to the mathematical probability density. High density means more dots.</li>
          </ul>

          <h2>3. Controls</h2>
          <p>Toggle phase mapping to see positive and negative regions of the wavefunction.</p>
          <p>The simulator uses a fixed high grid range for stable detail, and you can still enable glow for stronger visual contrast.</p>
          
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
            <h4 className="settings-title">Settings</h4>
            <label>
              <input type="checkbox" checked={showPhase} onChange={e => setShowPhase(e.target.checked)} /> Show Wavefunction Phase
            </label>
            <label>
              <input type="checkbox" checked={enableSimpleGlow} onChange={e => setEnableSimpleGlow(e.target.checked)} /> Enable Simple Glow
            </label>
            <label>
              <input type="checkbox" checked={showAxes} onChange={e => setShowAxes(e.target.checked)} /> Show Axes
            </label>
            <label>
              <input type="checkbox" checked={showSliceView} onChange={e => setShowSliceView(e.target.checked)} /> Slice View
            </label>
            {showSliceView && (
              <>
                <div className="slice-axis-row" role="radiogroup" aria-label="Slice axis">
                  <label className="slice-axis-option">
                    <input type="checkbox" checked={selectedSliceAxes.includes('x')} onChange={() => toggleSliceAxis('x')} /> X Axis
                  </label>
                  <label className="slice-axis-option">
                    <input type="checkbox" checked={selectedSliceAxes.includes('y')} onChange={() => toggleSliceAxis('y')} /> Y Axis
                  </label>
                  <label className="slice-axis-option">
                    <input type="checkbox" checked={selectedSliceAxes.includes('z')} onChange={() => toggleSliceAxis('z')} /> Z Axis
                  </label>
                </div>
                {selectedSliceAxes.map((axis) => (
                  <label className="slice-offset-label" key={axis}>
                    Slice Offset ({axis.toUpperCase()}): {(sliceOffsets[axis] ?? 0).toFixed(2)}
                    <input
                      type="range"
                      min={-sliceOffsetLimit}
                      max={sliceOffsetLimit}
                      step={0.1}
                      value={sliceOffsets[axis] ?? 0}
                      onChange={(e) => updateSliceOffsetForAxis(axis, parseFloat(e.target.value))}
                      onMouseDown={() => { setIsSliceSliderDragging(true); setActiveSliceDragAxis(axis); }}
                      onMouseUp={() => { setIsSliceSliderDragging(false); setActiveSliceDragAxis(null); }}
                      onTouchStart={() => { setIsSliceSliderDragging(true); setActiveSliceDragAxis(axis); }}
                      onTouchEnd={() => { setIsSliceSliderDragging(false); setActiveSliceDragAxis(null); }}
                      onPointerDown={() => { setIsSliceSliderDragging(true); setActiveSliceDragAxis(axis); }}
                      onPointerUp={() => { setIsSliceSliderDragging(false); setActiveSliceDragAxis(null); }}
                      onBlur={() => { setIsSliceSliderDragging(false); setActiveSliceDragAxis(null); }}
                    />
                  </label>
                ))}
              </>
            )}
          </div>

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
            <Canvas dpr={canvasDpr} onCreated={({ gl }) => { gl.localClippingEnabled = true; }} gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }} performance={{ min: 0.6 }} camera={{ position: [0, 0, gridSize * 1.5], fov: 60 }}>
                <color attach="background" args={['#111111']} />
                <ambientLight intensity={0.5} />
                <pointLight position={[100, 100, 100]} intensity={1} />
                <pointLight position={[-100, -100, -100]} intensity={0.5} />
                
                {plotData && (
                  <ScatterPlot
                    key={renderKey}
                    data={plotData}
                    opacity={SCATTER_OPACITY}
                    showPhase={showPhase}
                    enableSimpleGlow={enableSimpleGlow}
                    nValue={n}
                    sliceEnabled={showSliceView}
                    selectedSliceAxes={selectedSliceAxes}
                    sliceOffsets={sliceOffsets}
                  />
                )}

                {showSliceView && isSliceSliderDragging && activeSliceDragAxis && (
                  <SlicePlaneVisual axis={activeSliceDragAxis} offset={sliceOffsets[activeSliceDragAxis] ?? 0} extent={slicePlaneExtent} />
                )}
                
                {showAxes && <FullAxes size={gridSize} />}
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
                  <LineChart data={animatedRadialData} margin={{ top: 5, right: 30, left: 10, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#444" />
                  <XAxis
                    dataKey="r"
                    stroke="#ccc"
                    tick={{fill: '#ccc'}}
                    tickFormatter={(val) => Number(val).toFixed(1)}
                    tickCount={5}
                    minTickGap={40}
                  />
                  <YAxis
                    stroke="#ccc"
                    tick={{fill: '#ccc'}}
                    tickFormatter={(val) => Number(val).toFixed(2)}
                    tickCount={5}
                  />
                  <Tooltip contentStyle={{ backgroundColor: '#222', border: '1px solid #444', color: '#fff' }} formatter={(val) => Number(val).toFixed(3)} />
                  <Line
                    type="monotone"
                    dataKey="P"
                    stroke="#00aaff"
                    dot={false}
                    strokeWidth={2}
                    isAnimationActive={true}
                    animationDuration={GRAPH_RETRACE_DURATION_MS}
                    animationEasing="ease-in-out"
                  />
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
  border: '1px solid #3a3a3a',
  padding: '15px',
  borderRadius: '5px',
  fontSize: '1.2rem',
  color: '#eaf6ff',
  textAlign: 'center',
  margin: '20px 0',
  overflowX: 'auto'
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
