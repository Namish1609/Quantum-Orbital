import React, { useState, useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame } from '@react-three/fiber';
import { OrbitControls, Text, Billboard } from '@react-three/drei';
import * as THREE from 'three';
import {
  LineChart,
  Line,
  AreaChart,
  Area,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  Legend
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

const quantumSymbolCards = [
  {
    symbol: 'n',
    title: 'Principal Quantum Number',
    range: 'n = 1, 2, 3, ...',
    insight: 'Sets shell size and energy level depth.',
  },
  {
    symbol: 'l',
    title: 'Azimuthal Quantum Number',
    range: '0 <= l <= n - 1',
    insight: 'Controls orbital family and angular nodes.',
  },
  {
    symbol: 'm',
    title: 'Magnetic Quantum Number',
    range: '-l <= m <= l',
    insight: 'Defines orientation relative to chosen axis.',
  },
  {
    symbol: 'Z',
    title: 'Nuclear Charge',
    range: 'Z >= 1',
    insight: 'Higher Z contracts orbitals and raises binding energy.',
  },
];

const classicalVsQuantumRows = [
  {
    topic: 'Electron Behavior',
    classical: 'Point particle on deterministic orbit',
    quantum: 'Wavefunction with probabilistic localization',
  },
  {
    topic: 'Allowed States',
    classical: 'Any radius and energy in principle',
    quantum: 'Discrete energy eigenstates from boundary conditions',
  },
  {
    topic: 'Visual Output',
    classical: 'Single line trajectory',
    quantum: '3D cloud with nodes, phase, and symmetry',
  },
  {
    topic: 'Chemical Meaning',
    classical: 'Poor predictor of bonding',
    quantum: 'Explains bonding, geometry, spectra, and trends',
  },
];

const orbitalFamilyRows = [
  {
    family: 's',
    l: '0',
    geometry: 'Spherical shell',
    keyUse: 'Core shielding and sigma bonding',
  },
  {
    family: 'p',
    l: '1',
    geometry: 'Two-lobed dumbbell',
    keyUse: 'Directional covalent and pi interactions',
  },
  {
    family: 'd',
    l: '2',
    geometry: 'Clover / torus hybrid forms',
    keyUse: 'Transition metals, splitting, catalysis',
  },
  {
    family: 'f',
    l: '3',
    geometry: 'Multi-lobed high-order surfaces',
    keyUse: 'Lanthanides, actinides, magnetic behavior',
  },
];

const simulatorControlRows = [
  {
    control: 'n slider',
    effect: 'Expands shell and adds radial complexity',
    expectedVisual: 'Larger orbital envelope with extra nodal rings',
  },
  {
    control: 'l slider',
    effect: 'Changes angular momentum family',
    expectedVisual: 'Spherical to directional lobe transitions',
  },
  {
    control: 'm slider',
    effect: 'Rotates orientation basis',
    expectedVisual: 'Different axis alignment for the same family',
  },
  {
    control: 'Show Phase',
    effect: 'Reveals sign of wavefunction',
    expectedVisual: 'Red and blue regions split by nodal boundaries',
  },
  {
    control: 'Slice View',
    effect: 'Cuts through volume along selected axis',
    expectedVisual: 'Interior nodal surfaces become immediately visible',
  },
];

const energyProfileData = [
  { shell: 'n=1', radius: 14, density: 92 },
  { shell: 'n=2', radius: 29, density: 74 },
  { shell: 'n=3', radius: 42, density: 58 },
  { shell: 'n=4', radius: 56, density: 44 },
  { shell: 'n=5', radius: 71, density: 35 },
  { shell: 'n=6', radius: 88, density: 27 },
];

const orbitalComplexityData = [
  { orbital: 's', nodes: 1, directionality: 2, chemistry: 7 },
  { orbital: 'p', nodes: 3, directionality: 6, chemistry: 8 },
  { orbital: 'd', nodes: 6, directionality: 8, chemistry: 9 },
  { orbital: 'f', nodes: 9, directionality: 9, chemistry: 10 },
];

const phaseInsightData = [
  { metric: 'Bonding Overlap', positive: 92, negative: 40 },
  { metric: 'Interference Risk', positive: 35, negative: 88 },
  { metric: 'Spectral Signature', positive: 76, negative: 70 },
  { metric: 'Spatial Contrast', positive: 81, negative: 84 },
  { metric: 'Learning Clarity', positive: 86, negative: 71 },
];

const workflowChartData = [
  { step: 'Choose State', confidence: 18 },
  { step: 'Apply Changes', confidence: 35 },
  { step: 'Inspect Shape', confidence: 58 },
  { step: 'Toggle Phase', confidence: 76 },
  { step: 'Slice + Compare', confidence: 91 },
];

const quickLinks = [
  { id: 'welcome', label: 'Welcome & Math' },
  { id: 'chemistry', label: 'Chemistry Concepts' },
  { id: 'howto', label: 'How To Use' },
  { id: 'simulator', label: 'Simulator' },
];

const chartTooltipStyle = {
  backgroundColor: '#0f1722',
  border: '1px solid #2f4c6a',
  borderRadius: '10px',
  color: '#eaf4ff',
};

const AnimatedSection = ({ children, className = '' }) => {
  const [isVisible, setIsVisible] = useState(false);
  const domRef = useRef();

  useEffect(() => {
    const observer = new IntersectionObserver((entries) => {
      setIsVisible(entries[0].isIntersecting);
    }, { threshold: 0.1 });

    if (domRef.current) observer.observe(domRef.current);
    return () => observer.disconnect();
  }, []);

  return (
    <div ref={domRef} className={`fade-in-section ${isVisible ? 'is-visible' : ''} ${className}`}>
      {children}
    </div>
  );
};

const InfoSection = ({ title, subtitle, children }) => (
  <AnimatedSection className="info-section">
    <div className="info-section-heading">
      <h2>{title}</h2>
      {subtitle && <p>{subtitle}</p>}
    </div>
    {children}
  </AnimatedSection>
);

const ExpandableFormula = ({ math, className = '' }) => {
  const [isOpen, setIsOpen] = useState(false);
  const [isClosing, setIsClosing] = useState(false);
  const closeTimerRef = useRef(null);

  useEffect(() => () => {
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current);
    }
  }, []);

  const openFormula = () => {
    if (closeTimerRef.current) {
      clearTimeout(closeTimerRef.current);
      closeTimerRef.current = null;
    }
    setIsClosing(false);
    setIsOpen(true);
  };

  const closeFormula = () => {
    if (isClosing) return;
    setIsClosing(true);
    closeTimerRef.current = setTimeout(() => {
      setIsOpen(false);
      setIsClosing(false);
      closeTimerRef.current = null;
    }, 220);
  };

  return (
    <>
      <div className={`math-card ${className}`.trim()}>
        <button className="formula-expand-btn" type="button" onClick={openFormula} aria-label="Expand formula">
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M8 3H3v5"></path>
            <line x1="3" y1="3" x2="10" y2="10"></line>
            <path d="M16 21h5v-5"></path>
            <line x1="14" y1="14" x2="21" y2="21"></line>
          </svg>
        </button>
        <BlockMath math={math} />
      </div>

      {isOpen && (
        <div className={`formula-modal-backdrop ${isClosing ? 'is-closing' : 'is-open'}`} onClick={closeFormula} role="dialog" aria-modal="true">
          <div className={`formula-modal ${isClosing ? 'is-closing' : 'is-open'}`} onClick={(event) => event.stopPropagation()}>
            <button className="formula-close-btn" type="button" onClick={closeFormula} aria-label="Close expanded formula">
              x
            </button>
            <BlockMath math={math} />
          </div>
        </div>
      )}
    </>
  );
};

const SymbolCardVertical = () => (
  <div className="symbol-vertical-list">
    <article className="glass-card flex-row-card">
      <div className="orbital-symbol symbol-n">
        <div className="circle-ring r1"></div>
        <div className="circle-ring r2"></div>
        <div className="circle-ring r3"></div>
      </div>
      <div className="orbital-info">
        <h3>Principal Quantum Number (<InlineMath math={'n'} /> = 1, 2, 3...)</h3>
        <p><strong>Energy & Distance.</strong> Determines the overall size and energy of the orbital. As <InlineMath math={'n'} /> increases, the electron spends more time further from the nucleus, meaning higher energy and less stability. Also defines the total number of nodes (<InlineMath math={'n - 1'} />).</p>
      </div>
    </article>

    <article className="glass-card flex-row-card">
      <div className="orbital-symbol symbol-l">
        <svg width="40" height="40" viewBox="0 0 120 70" style={{ transform: 'scale(1.2)' }}>
          <path d="M 10 60 A 50 50 0 0 1 110 60 Z" fill="none" stroke="currentColor" strokeWidth="3"/>
          <line x1="5" y1="60" x2="115" y2="60" stroke="currentColor" strokeWidth="3"/>
          <line x1="60" y1="60" x2="60" y2="5" stroke="currentColor" strokeWidth="3" strokeDasharray="4 4"/>
          <line x1="60" y1="60" x2="10" y2="10" stroke="#ff3b6a" strokeWidth="3"/>
          <path d="M 40 40 Q 50 30 60 40" fill="none" stroke="#fff" strokeWidth="2"/>
          <circle cx="60" cy="60" r="4" fill="#fff"/>
        </svg>
      </div>
      <div className="orbital-info">
        <h3>Angular Momentum Number (<InlineMath math={'l'} /> = 0, ..., <InlineMath math={'n-1'} />)</h3>
        <p><strong>Shape & Subshell.</strong> Dictates the geometry of the probability cloud (s, p, d, f) and tells us the exact number of angular nodes. <InlineMath math={'l=0'} /> is spherical, <InlineMath math={'l=1'} /> has lobes, <InlineMath math={'l=2'} /> has clover shapes, etc. Fundamental to chemical bonding directionality.</p>
      </div>
    </article>

    <article className="glass-card flex-row-card">
      <div className="orbital-symbol symbol-m">
        <div className="compass">
          <div className="needle top"></div>
          <div className="needle bottom"></div>
        </div>
      </div>
      <div className="orbital-info">
        <h3>Magnetic Quantum Number (<InlineMath math={'m_l'} /> = <InlineMath math={'-l'} /> to <InlineMath math={'+l'} />)</h3>
        <p><strong>Spatial Orientation.</strong> Determines exactly how the orbital aligns in 3D space along the x, y, and z axes. Gives rise to the degenerate orbital sets, e.g. the 3 distinct <InlineMath math={'p'} /> orbitals or 5 distinct <InlineMath math={'d'} /> orbitals interacting with a magnetic field.</p>
      </div>
    </article>

    <article className="glass-card flex-row-card">
      <div className="orbital-symbol symbol-z">
        <div className="nucleus-dot"></div>
        <div className="electron-orbit o1"></div>
        <div className="electron-orbit o2"></div>
        <div className="electron-orbit o3"></div>
      </div>
      <div className="orbital-info">
        <h3>Atomic Number (<InlineMath math={'Z'} />)</h3>
        <p><strong>Nuclear Charge.</strong> Represents the number of protons in the nucleus. A higher <InlineMath math={'Z'} /> increases electrostatic pull, dramatically contracting the orbital size and lowering its energy. This parameter proves why heavier elements have incredibly compact deeper inner shells.</p>
      </div>
    </article>
  </div>
);

const SymbolCardGrid = () => (
  <div className="symbol-grid">
    {quantumSymbolCards.map((card) => (
      <article className="symbol-card" key={card.symbol}>
        <div className="symbol-mark">{card.symbol}</div>
        <h3>{card.title}</h3>
        <p className="symbol-range">{card.range}</p>
        <p>{card.insight}</p>
      </article>
    ))}
  </div>
);

const DataTable = ({ columns, rows, caption }) => (
  <div className="table-wrap">
    <table className="quantum-table">
      {caption && <caption>{caption}</caption>}
      <thead>
        <tr>
          {columns.map((column) => (
            <th key={column}>{column}</th>
          ))}
        </tr>
      </thead>
      <tbody>
        {rows.map((row) => (
          <tr key={Object.values(row).join('-')}>
            {Object.values(row).map((value) => (
              <td key={String(value)}>{value}</td>
            ))}
          </tr>
        ))}
      </tbody>
    </table>
  </div>
);

const InfoFooter = ({ onNavigate }) => (
  <footer className="site-footer">
    <div className="site-footer-grid">
      <section className="footer-about">
        <h3>About</h3>
        <p>
          Quantum Orbital Explorer is an educational visualization studio for understanding Schrodinger solutions,
          orbital topology, and chemistry-ready intuition.
        </p>
      </section>

      <section className="footer-links">
        <h3>Quick Links</h3>
        <div className="footer-link-list">
          {quickLinks.map((link) => (
            <button
              className="footer-link-button"
              key={link.id}
              onClick={() => onNavigate(link.id)}
              type="button"
            >
              {link.label}
            </button>
          ))}
        </div>
      </section>

      <section className="footer-contact">
        <h3>Contact</h3>
        <div className="contact-inline">
          <span className="contact-chip">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M22 16.92v3a2 2 0 0 1-2.18 2A19.8 19.8 0 0 1 3.1 5.18 2 2 0 0 1 5.08 3h3a2 2 0 0 1 2 1.72c.12.9.33 1.78.62 2.62a2 2 0 0 1-.45 2.11L9.02 10.68a16 16 0 0 0 4.3 4.3l1.23-1.23a2 2 0 0 1 2.11-.45c.84.29 1.72.5 2.62.62A2 2 0 0 1 22 16.92z"></path>
            </svg>
            +91 7411515850
          </span>
          <span className="contact-chip">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <rect x="3" y="5" width="18" height="14" rx="2" ry="2"></rect>
              <polyline points="3 7 12 13 21 7"></polyline>
            </svg>
            navion.team.official@gmail.com
          </span>
          <span className="contact-chip">
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <rect x="3" y="5" width="18" height="14" rx="2" ry="2"></rect>
              <polyline points="3 7 12 13 21 7"></polyline>
            </svg>
            labs@quantum-orbital-explorer.local
          </span>
        </div>
        <p className="contact-hours">Response Window: Monday to Friday, 09:00-18:00 UTC</p>
      </section>

      <section className="footer-privacy">
        <h3>Privacy Policy</h3>
        <p>No account is required to run simulations.</p>
        <p>Input parameters are processed to generate orbital visuals and are not sold to third parties.</p>
        <p>Operational logs are retained briefly to maintain reliability and performance.</p>
      </section>
    </div>
    <div className="site-footer-bottom">
      <span>2026 Quantum Orbital Explorer. Built for education, labs, and modern chemistry classrooms.</span>
    </div>
  </footer>
);

const InfoPageLayout = ({ eyebrow, title, message, children, onNavigate, ctaLabel, ctaTarget, heroClassName = '' }) => (
  <div className="info-page-shell">
    {/* Left Fixed Ad Sidebar */}
    <aside className="ad-sidebar ad-sidebar-left">
      <div className="ad-content">
        <span>Ad Space</span>
      </div>
    </aside>

    <main className="page-style">
      <header className={`hero-panel ${heroClassName}`.trim()}>
        <p className="hero-eyebrow">{eyebrow}</p>
        <h1>{title}</h1>
        <p className="hero-message">{message}</p>
        {ctaLabel && ctaTarget && (
          <button className="hero-cta" onClick={() => onNavigate(ctaTarget)} type="button">
            {ctaLabel}
          </button>
        )}
      </header>

      {children}
    </main>

    {/* Right Fixed Ad Sidebar */}
    <aside className="ad-sidebar ad-sidebar-right">
      <div className="ad-content">
        <span>Ad Space</span>
      </div>
    </aside>

    <InfoFooter onNavigate={onNavigate} />
  </div>
);

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
      <div className={`top-nav ${currentPage === 'simulator' ? 'top-nav-simulator' : ''}`}>
        <h3>Quantum Orbital Explorer</h3>
        <div className="top-nav-buttons">
          <button onClick={() => setCurrentPage('welcome')} style={navButtonStyle(currentPage === 'welcome')}>Welcome & Math</button>
          <button onClick={() => setCurrentPage('chemistry')} style={navButtonStyle(currentPage === 'chemistry')}>Chemistry Concepts</button>
          <button onClick={() => setCurrentPage('howto')} style={navButtonStyle(currentPage === 'howto')}>How To Use</button>
          <button onClick={() => setCurrentPage('simulator')} style={navButtonStyle(currentPage === 'simulator')}><strong>Simulator</strong></button>
        </div>
      </div>

      {currentPage === 'welcome' && (
        <InfoPageLayout
          eyebrow="Interactive Quantum Learning Platform"
          title="See Orbitals Like a Professional Science Product"
          message="Turn equations into rich 3D intuition with visual storytelling, symbol boxes, and decision-ready comparison views inspired by modern product design."
          heroClassName="hero-welcome"
          onNavigate={setCurrentPage}
          ctaLabel="Continue to Chemistry Concepts"
          ctaTarget="chemistry"
        >
          <InfoSection
            title="The Equation Driving The Simulator"
            subtitle="The model is built from exact hydrogenic solutions in spherical coordinates."
          >
            <div className="info-grid two-column">
              <article className="glass-card">
                <p>
                  The time-independent Schrodinger equation gives the stationary states for one-electron atoms:
                </p>
                <ExpandableFormula math={'\\hat{H}\\psi = E\\psi'} />
                <p>
                  The total wavefunction separates naturally into radial and angular components:
                </p>
                <ExpandableFormula math={'\\psi_{n,\\ell,m}(r,\\theta,\\phi)=R_{n,\\ell}(r)\\,Y_{\\ell,m}(\\theta,\\phi)'} />
                <p>
                  Probability density is measured through <InlineMath math={'|\\psi|^2'} />, which is what your scatter cloud approximates.
                </p>
              </article>

              <article className="glass-card chart-card">
                <h3>Shell Growth vs Relative Density</h3>
                <ResponsiveContainer width="100%" height={260}>
                  <AreaChart data={energyProfileData} margin={{ top: 10, right: 10, left: 0, bottom: 0 }}>
                    <defs>
                      <linearGradient id="radiusGradient" x1="0" y1="0" x2="0" y2="1">
                        <stop offset="0%" stopColor="#30c6ff" stopOpacity={0.8} />
                        <stop offset="100%" stopColor="#30c6ff" stopOpacity={0.05} />
                      </linearGradient>
                    </defs>
                    <CartesianGrid stroke="#2f4660" strokeDasharray="4 4" />
                    <XAxis dataKey="shell" stroke="#a7c9e8" />
                    <YAxis stroke="#a7c9e8" />
                    <Tooltip contentStyle={chartTooltipStyle} />
                    <Legend />
                    <Area type="monotone" dataKey="radius" name="Relative Radius" stroke="#30c6ff" fill="url(#radiusGradient)" strokeWidth={2.2} />
                    <Line type="monotone" dataKey="density" name="Relative Density" stroke="#ffd265" strokeWidth={2} dot={false} />
                  </AreaChart>
                </ResponsiveContainer>
              </article>
            </div>
          </InfoSection>

          <InfoSection
            title="Deciphering The Quantum Numbers"
            subtitle="The four fundamental parameters of the quantum model."
          >
            <SymbolCardVertical />
          </InfoSection>

          <InfoSection
            title="Radial & Angular Separability"
            subtitle="How we construct volumetric regions mathematically."
          >
            <div className="info-grid two-column">
              <article className="glass-card">
                <h3>The Radial Function <InlineMath math={'R_{n, l}(r)'} /></h3>
                <p>Governs the amplitude of the electron's distance <InlineMath math={'r'} /> from the nucleus. Built from Laguerre polynomials.</p>
                <ExpandableFormula className="compact" math={'R_{n,l}(r) = \\sqrt{\\left(\\frac{2Z}{n a_0}\\right)^3 \\dots} e^{-\\frac{Zr}{na_0}} L_{n-l-1}^{2l+1}\\left(\\frac{2Zr}{na_0}\\right)'} />
                <p>Features include exponential decay towards infinity and oscillating zeroes (radial nodes).</p>
              </article>

              <article className="glass-card">
                <h3>Spherical Harmonics <InlineMath math={'Y_{l, m}(\\theta, \\phi)'} /></h3>
                <p>Controls the angular distribution around the nucleus using Legendre polynomials. This generates the familiar orbital shapes.</p>
                <ExpandableFormula className="compact harmonic-compact" math={'Y_{l,m}(\\theta,\\phi) = \\sqrt{\\frac{(2l+1)}{4\\pi}\\frac{(l-m)!}{(l+m)!}} P_l^m(\\cos\\theta) e^{im\\phi}'} />
                <p>Phase rules dictate structural geometry and directivity, key components in chemistry.</p>
              </article>
            </div>
          </InfoSection>
        </InfoPageLayout>
      )}

      {currentPage === 'chemistry' && (
        <InfoPageLayout
          eyebrow="Quantum Foundations"
          title="Orbital Structures and Wavefunction Phase"
          message="Understand the building blocks of matter. By mapping atomic orbitals and phase, we can predict bonding, geometry, and chemical reactivity."
          heroClassName="hero-chemistry"
          onNavigate={setCurrentPage}
          ctaLabel="Continue to How To Use"
          ctaTarget="howto"
        >
          <InfoSection
            title="The Orbital Families"
            subtitle="The shape and energy of probability clouds are defined by quantum numbers."
          >
            <div className="info-grid">
              <article className="glass-card flex-row-card">
                <div className="orbital-symbol s-orbital">
                  <div className="shape-sphere"></div>
                </div>
                <div className="orbital-info">
                  <h3>s Orbitals (<InlineMath math={'l = 0'} />)</h3>
                  <p><strong>Spherical shell.</strong> The simplest orbital shape with radial symmetry. They form the core layers of atoms and participate in direct sigma (<InlineMath math={'\\sigma'} />) bonds.</p>
                </div>
              </article>
              <article className="glass-card flex-row-card">
                <div className="orbital-symbol p-orbital">
                  <div className="shape-lobe top-lobe"></div>
                  <div className="shape-lobe bottom-lobe"></div>
                </div>
                <div className="orbital-info">
                  <h3>p Orbitals (<InlineMath math={'l = 1'} />)</h3>
                  <p><strong>Two-lobed dumbbell.</strong> They feature one angular node separating two phases. Essential for directional covalent bonding and pi (<InlineMath math={'\\pi'} />) interactions.</p>
                </div>
              </article>
              <article className="glass-card flex-row-card">
                <div className="orbital-symbol d-orbital">
                  <div className="shape-clover"></div>
                </div>
                <div className="orbital-info">
                  <h3>d Orbitals (<InlineMath math={'l = 2'} />)</h3>
                  <p><strong>Clover / torus hybrid forms.</strong> Encountering two angular nodes, they drive transition metal chemistry, ligand field splitting, and catalysis.</p>
                </div>
              </article>
              <article className="glass-card flex-row-card">
                <div className="orbital-symbol f-orbital">
                  <div className="shape-complex"></div>
                </div>
                <div className="orbital-info">
                  <h3>f Orbitals (<InlineMath math={'l = 3'} />)</h3>
                  <p><strong>Multi-lobed high-order surfaces.</strong> Featuring three angular nodes, these buried valence layers are responsible for lanthanides, actinides, and complex magnetic behavior.</p>
                </div>
              </article>
            </div>
          </InfoSection>

          <InfoSection
            title="Understanding Wavefunction Phase"
            subtitle="Regions of probability across complex geometries."
          >
            <div className="info-grid two-column">
              <article className="glass-card">
                <h3>Radial Nodes</h3>
                <p>
                  Radial nodes are spherical shells where the radial wavefunction becomes zero.
                  They are controlled by <InlineMath math={'n-l-1'} /> and appear as concentric zero-probability boundaries moving outward from the nucleus.
                </p>
              </article>

              <article className="glass-card">
                <h3>Angular Nodes</h3>
                <p>
                  Angular nodes are planar or conical surfaces determined by the spherical harmonics.
                  Their count equals <InlineMath math={'l'} />, and they split orbitals into directional lobes with alternating phase.
                </p>
              </article>
            </div>
          </InfoSection>

          <InfoSection title="Phase Interference Comparison">
            <DataTable
              columns={["Wavefunction Interaction", "Phase Match", "Quantum Interference", "Chemical Outcome"]}
              rows={[
                ["Positive / Positive (+ / +)", "Matching signs", "Amplitudes add together", "Forms a stable bonding orbital"],
                ["Negative / Negative (- / -)", "Matching signs", "Amplitudes add together", "Forms a stable bonding orbital"],
                ["Positive / Negative (+ / -)", "Opposite signs", "Amplitudes cancel out (\u03c8 = 0)", "Forms a repulsive anti-bonding orbital"],
                ["Node Existence", "Zero crossing points", "Probability is exactly zero", "Divides regions of opposite phase"]
              ]}
            />
          </InfoSection>

          <InfoSection title="Positive vs Negative Phase">
            <div className="info-grid two-column">
              <article className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
                <div style={{ margin: '20px 0' }}>
                  <svg width="80" height="80" viewBox="0 0 100 100">
                    <circle cx="50" cy="50" r="40" fill="rgba(255, 107, 107, 0.2)" stroke="#ff6b6b" strokeWidth="3"/>
                    <line x1="30" y1="50" x2="70" y2="50" stroke="#ff6b6b" strokeWidth="4"/>
                    <line x1="50" y1="30" x2="50" y2="70" stroke="#ff6b6b" strokeWidth="4"/>
                  </svg>
                </div>
                <h3>Positive Phase (+)</h3>
                <p>Constructive interference regions. Mathematically positive wavefunction amplitude.</p>
              </article>
              <article className="glass-card" style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', textAlign: 'center' }}>
                <div style={{ margin: '20px 0' }}>
                  <svg width="80" height="80" viewBox="0 0 100 100">
                    <circle cx="50" cy="50" r="40" fill="rgba(107, 234, 255, 0.2)" stroke="#6beaff" strokeWidth="3"/>
                    <line x1="30" y1="50" x2="70" y2="50" stroke="#6beaff" strokeWidth="4"/>
                  </svg>
                </div>
                <h3>Negative Phase (-)</h3>
                <p>Destructive interference regions. Mathematically negative wavefunction amplitude.</p>
              </article>
            </div>
            <div className="glass-card" style={{ marginTop: '16px' }}>
              <p>
                The wavefunction <InlineMath math={'\\psi'} /> conceptually operates like a wave mechanics equation, meaning it holds mathematical signs (+ or -). Since calculating real-world probability requires squaring it (<InlineMath math={'|\\psi|^2'} />), the negative signs technically disappear in pure space-finding probabilities.
              </p>
              <p>
                However, <strong>phase dictates bonding rules</strong>. When you bring two atoms together, their orbital clouds mathematically overlap. 
                If they overlap with the <strong>same phase (e.g., both positive)</strong>, they constructively interfere, forming a covalent bond. 
                If they overlap with <strong>opposite phases (+ and -)</strong>, they destructively interfere, actively repelling from one another to create an anti-bonding condition.
              </p>
            </div>
          </InfoSection>
        </InfoPageLayout>
      )}

      {currentPage === 'howto' && (
        <InfoPageLayout
          eyebrow="Professional Workflow"
          title="From Parameter Selection To Interpretation"
          message="Use this page as an execution checklist so every simulation run gives actionable chemistry insight, not just a beautiful picture."
          heroClassName="hero-howto"
          onNavigate={setCurrentPage}
          ctaLabel="Launch Simulator"
          ctaTarget="simulator"
        >
          <InfoSection
            title="Control-to-Result Table"
            subtitle="A practical control map for faster classroom demos and more consistent research exploration."
          >
            <DataTable
              columns={["Control", "What It Changes"]}
              rows={simulatorControlRows.map((row) => [row.control, row.effect])}
            />
          </InfoSection>

          <InfoSection
            title="Interaction Guide"
            subtitle="Follow these commands to inspect internal nodes and external symmetries."
          >
            <div className="info-grid">
              <article className="glass-card flex-row-card">
                <div className="orbital-symbol">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#6beaff" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <path d="M4 3l7 17 2.8-6.1L20 11 4 3z"></path>
                    <path d="M13.3 13.9L17 21"></path>
                  </svg>
                </div>
                <div className="orbital-info">
                  <h3>Left-Click + Drag (Rotate)</h3>
                  <p><strong>Inspect 3D Geometry.</strong> Freely rotate the visualization sphere to find the optimal viewpoint, exposing hidden angular nodal planes and orbital lobes.</p>
                </div>
              </article>
              
              <article className="glass-card flex-row-card">
                <div className="orbital-symbol">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#ff6b6b" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="12" y1="3" x2="12" y2="21"></line>
                    <line x1="3" y1="12" x2="21" y2="12"></line>
                    <polyline points="12 3 9 6 15 6 12 3"></polyline>
                    <polyline points="12 21 9 18 15 18 12 21"></polyline>
                    <polyline points="3 12 6 9 6 15 3 12"></polyline>
                    <polyline points="21 12 18 9 18 15 21 12"></polyline>
                  </svg>
                </div>
                <div className="orbital-info">
                  <h3>Right-Click + Drag (Pan)</h3>
                  <p><strong>Move the Origin.</strong> Shift the entire coordinate system sideways or up and down to center off-axis structures exactly where you need them.</p>
                </div>
              </article>

              <article className="glass-card flex-row-card">
                <div className="orbital-symbol">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#39dcb1" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <circle cx="11" cy="11" r="6"></circle>
                    <line x1="16" y1="16" x2="21" y2="21"></line>
                    <line x1="11" y1="8" x2="11" y2="14"></line>
                    <line x1="8" y1="11" x2="14" y2="11"></line>
                  </svg>
                </div>
                <div className="orbital-info">
                  <h3>Mouse Wheel (Zoom)</h3>
                  <p><strong>Dive Through Shells.</strong> Zoom deep into the nucleus or pull back to see the sprawling outer lobes of high-energy <InlineMath math={'n > 6'} /> states.</p>
                </div>
              </article>

              <article className="glass-card flex-row-card">
                <div className="orbital-symbol">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="#ff8a5b" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                    <line x1="4" y1="6" x2="20" y2="6"></line>
                    <line x1="4" y1="12" x2="20" y2="12"></line>
                    <line x1="4" y1="18" x2="20" y2="18"></line>
                    <circle cx="9" cy="6" r="2" fill="#ff8a5b"></circle>
                    <circle cx="15" cy="12" r="2" fill="#ffd265"></circle>
                    <circle cx="11" cy="18" r="2" fill="#ff8a5b"></circle>
                  </svg>
                </div>
                <div className="orbital-info">
                  <h3>Z Slider (Nuclear Charge)</h3>
                  <p><strong>Electrostatic Compression.</strong> Increasing <InlineMath math={'Z'} /> pulls probability density inward, tightening shells and shifting energy behavior in real time.</p>
                </div>
              </article>
            </div>
          </InfoSection>
        </InfoPageLayout>
      )}

      {currentPage === 'simulator' && (
        <div className="App simulator-active">
          
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

          <button type="submit" className="sim-apply-btn">
            {loading ? 'Computing...' : 'Apply Changes'}
          </button>

          {/* Ad Placeholder for Sidebar */}
          <div className="google-ad-placeholder simulator-ad-slot">
            <span className="simulator-ad-slot-label">Google Ad Slot</span>
          </div>

          <div style={{minHeight: '40px', marginTop: '10px'}}>
             {errorMsg && <p style={{color: '#ff6b6b', fontSize: '0.85rem', margin: 0}}>{errorMsg}</p>}
          </div>
        </form>
      </div>

      {/* Main Render Area */}
      <div className="main-content simulator-main-content">
          <div className="canvas-container" style={{ flex: 1, minHeight: 0 }}>
            <Canvas dpr={canvasDpr} onCreated={({ gl }) => { gl.localClippingEnabled = true; }} gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }} performance={{ min: 0.6 }} camera={{ position: [0, 0, gridSize * 1.5], fov: 60 }}>
                <color attach="background" args={['#0f1726']} />
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

          <div className="simulator-instruction-bar">
            <span><strong className="simulator-instruction-label">Controls:</strong> Left-Click to Rotate &bull; Right-Click to Pan/Move Target &bull; Scroll to Zoom</span>
          </div>

          <div
            className="resize-handle simulator-resize-handle"
            onMouseDown={startResize}
          >
            <div className="simulator-resize-grip"></div>
          </div>

          <div className="radial-graph-container simulator-graph-panel" style={{ height: `${graphHeight}px` }}>
            <div className="simulator-graph-main">
              <h4 className="simulator-graph-title">Radial Probability Distribution</h4>
              <div className="simulator-graph-chart">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={animatedRadialData} margin={{ top: 5, right: 24, left: 10, bottom: 20 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#3f526d" />
                  <XAxis
                    dataKey="r"
                    stroke="#d5e5f6"
                    tick={{ fill: '#d5e5f6' }}
                    tickFormatter={(val) => Number(val).toFixed(1)}
                    tickCount={5}
                    minTickGap={40}
                  />
                  <YAxis
                    stroke="#d5e5f6"
                    tick={{ fill: '#d5e5f6' }}
                    tickFormatter={(val) => Number(val).toFixed(2)}
                    tickCount={5}
                  />
                  <Tooltip contentStyle={{ backgroundColor: '#0f1722', border: '1px solid #3d5677', color: '#eaf4ff' }} formatter={(val) => Number(val).toFixed(3)} />
                  <Line
                    type="monotone"
                    dataKey="P"
                    stroke="#57d3ff"
                    dot={false}
                    strokeWidth={2.1}
                    isAnimationActive={true}
                    animationDuration={GRAPH_RETRACE_DURATION_MS}
                    animationEasing="ease-in-out"
                  />
                  </LineChart>
                </ResponsiveContainer>
              </div>
              <p className="graph-axis-caption">Probability(r)</p>
            </div>
            
            {!showPhase && (
              <div className="density-container">
                <div className="density-title">Density</div>
                <div className="density-bar">
                  <div className="density-labels">
                    <span>100</span>
                    <span>80</span>
                    <span>60</span>
                    <span>40</span>
                    <span>20</span>
                    <span>0</span>
                  </div>
                  <div className="density-gradient"></div>
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

// Reusable styling for the navigation buttons
const navButtonStyle = (isActive) => ({
  background: isActive ? 'linear-gradient(135deg, rgba(48, 198, 255, 0.23), rgba(122, 247, 176, 0.2))' : 'rgba(8, 19, 32, 0.35)',
  border: isActive ? '1px solid rgba(96, 219, 255, 0.5)' : '1px solid rgba(112, 157, 196, 0.22)',
  color: isActive ? '#dfffff' : '#d3e6fa',
  fontSize: '1rem',
  cursor: 'pointer',
  padding: '8px 14px',
  borderRadius: '999px',
  transition: 'all 0.2s ease',
  letterSpacing: '0.01em'
});

export default App;
