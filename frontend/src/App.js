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
  { id: 'welcome', label: 'Welcome' },
  { id: 'chemistry', label: 'Chemistry Concepts' },
  { id: 'howto', label: 'How To Use' },
  { id: 'faqs', label: 'FAQs' },
  { id: 'learn-quantum-theory', label: 'Learn' },
  { id: 'simulator', label: 'Simulator' },
];

const siteFaqs = [
  {
    question: 'Is Quantum Orbital Explorer free to use?',
    answer: 'Yes. The current version is free for learning, classroom demos, and self-study. If advanced paid plans are introduced later, they will be shown clearly in the app before any billing step.',
  },
  {
    question: 'Do I need to create an account before using the simulator?',
    answer: 'No account is required for normal simulation use. You can open the site, choose quantum numbers, and run visualizations directly.',
  },
  {
    question: 'What exactly does the simulator calculate?',
    answer: 'It computes hydrogenic orbital wavefunction-derived probability distributions and renders them as dense 3D scatter clouds, with a radial probability graph for distance-based interpretation.',
  },
  {
    question: 'Which quantum values can I control?',
    answer: 'You can control principal n, azimuthal l, magnetic m, and nuclear charge Z. These controls change shell size, symmetry, orientation, and contraction behavior.',
  },
  {
    question: 'What do the red and blue regions represent?',
    answer: 'They represent opposite wavefunction phase signs, not positive/negative electric charge. The phase view helps interpret constructive versus destructive overlap behavior.',
  },
  {
    question: 'What is Slice View used for?',
    answer: 'Slice View clips the cloud along selected axes so you can inspect interior nodal shells and angular boundaries that are usually hidden by outer density.',
  },
  {
    question: 'Are these visuals physically meaningful or just artistic?',
    answer: 'They are physically motivated visualizations based on analytical orbital models. They are educationally accurate for hydrogenic intuition, while advanced many-electron systems still require higher-level methods.',
  },
  {
    question: 'Can I use this in a classroom, lecture, or presentation?',
    answer: 'Yes. The interface is designed for teaching and communication. You can use it for educational demos, module explanations, and concept walkthroughs.',
  },
  {
    question: 'Why can high-n states feel heavy on some devices?',
    answer: 'Higher n increases point complexity and rendering load. The app applies performance guards, but older hardware may still show slower interaction during very dense scenes.',
  },
  {
    question: 'Does the website store my private simulation content?',
    answer: 'Simulation parameter requests are used to generate results and improve reliability. The app does not require private profile data to run core visualization features.',
  },
];

const learnTopics = [
  {
    id: 'learn-quantum-theory',
    label: '1. Quantum Theory, Evolution, and Dual Nature',
    shortLabel: 'Quantum Theory Foundations',
    eyebrow: 'Learn Module 1',
    title: 'What Is Quantum Theory? Evolution and Dual Nature',
    message: 'Follow the transition from classical certainty to quantum probability through experiments, equations, and interpretation frameworks.',
    theoryCards: [
      {
        title: 'Historical Evolution',
        text: 'Quantum theory emerged when blackbody radiation, photoelectric effect, and atomic spectra resisted classical explanations. Planck introduced energy quanta, Einstein formalized light quanta, Bohr quantized atomic states, and modern wave mechanics unified these insights into predictive models.',
      },
      {
        title: 'Wave-Particle Duality',
        text: 'Matter and radiation behave with both wave-like and particle-like signatures. Electrons diffract like waves in a crystal lattice yet register as localized detection events, proving that the wavefunction is an amplitude map while measurement outcomes are discrete.',
      },
      {
        title: 'Measurement and Uncertainty',
        text: 'Quantum measurement changes the state description from distributed amplitudes to observed outcomes. The uncertainty relation is not an instrument flaw; it is a structural rule linking conjugate observables through Fourier-limited representations.',
      },
      {
        title: 'Modern Interpretation Stack',
        text: 'Current workflows combine state vectors, operators, expectation values, and numerical simulation. In this simulator, that abstract stack is translated into visible orbital density, phase coloration, and node boundaries.',
      },
    ],
    equations: [
      {
        title: 'Planck-Einstein Relation',
        math: 'E = h\\nu',
        description: 'Energy is quantized by frequency. This relation is foundational for spectroscopy and motivates discrete atomic transitions.',
      },
      {
        title: 'de Broglie Wavelength',
        math: '\\lambda = \\frac{h}{p}',
        description: 'Momentum determines wavelength. This equation explains electron diffraction and the wave basis of bound-state orbitals.',
      },
    ],
    chartTitle: 'Concept Maturity Across Eras',
    chartSubtitle: 'Trend view of theory precision and experiment confirmation over key milestones.',
    chartData: [
      { stage: '1900', precision: 18, validation: 14 },
      { stage: '1913', precision: 31, validation: 26 },
      { stage: '1926', precision: 63, validation: 57 },
      { stage: '1950', precision: 78, validation: 74 },
      { stage: 'Today', precision: 94, validation: 92 },
    ],
    chartLines: [
      { dataKey: 'precision', name: 'Theory Precision', color: '#30c6ff' },
      { dataKey: 'validation', name: 'Experimental Validation', color: '#ffd265' },
    ],
    table: {
      columns: ['Concept', 'Classical Picture', 'Quantum Picture', 'Why It Matters'],
      rows: [
        ['Light', 'Continuous wave only', 'Photon and wave dual behavior', 'Explains photoelectric thresholds'],
        ['Electron', 'Point mass trajectory', 'State amplitude with probabilistic detection', 'Enables orbital and bonding models'],
        ['Energy', 'Continuous', 'Discrete eigenvalues in bound states', 'Explains line spectra'],
        ['Prediction', 'Deterministic path', 'Probability amplitudes and operators', 'Matches observed measurement statistics'],
      ],
    },
    imageSlots: [
      { title: 'Werner Heisenberg', description: 'Place a portrait and short timeline annotation.' },
      { title: 'Erwin Schrodinger', description: 'Place an image with notes on wave mechanics.' },
      { title: 'Double-Slit Experiment Diagram', description: 'Insert interference pattern image and interpretation labels.' },
    ],
    faqs: [
      {
        question: 'Why was wave-particle duality needed in the first place?',
        answer: 'Classical models alone could not explain photoelectric and diffraction experiments. Duality was needed to match both discrete detections and interference patterns.',
      },
      {
        question: 'Does uncertainty come from bad instruments?',
        answer: 'No. Quantum uncertainty is a fundamental property of conjugate observables and is built into the state description itself.',
      },
      {
        question: 'Which equation should I remember first in this module?',
        answer: 'Start with E = h nu and lambda = h / p. Together they capture quantized energy and wave-like matter behavior.',
      },
    ],
  },
  {
    id: 'learn-quantum-numbers-detail',
    label: '2. Quantum Numbers in Super Detail',
    shortLabel: 'Quantum Numbers Deep Dive',
    eyebrow: 'Learn Module 2',
    title: 'Explaining Quantum Numbers in Super Detail',
    message: 'Map each quantum number to measurable consequences in shell size, symmetry, degeneracy, and chemistry.',
    theoryCards: [
      {
        title: 'Principal Quantum Number (n)',
        text: 'n controls orbital scale, energy ladder position, and radial node budget. For hydrogenic systems, larger n expands average radius and compresses adjacent energy gaps as states approach ionization limit.',
      },
      {
        title: 'Azimuthal Quantum Number (l)',
        text: 'l determines orbital family (s, p, d, f), angular momentum magnitude, and the count of angular nodes. It governs shape complexity and directional chemistry behavior in molecular overlap.',
      },
      {
        title: 'Magnetic Quantum Number (m)',
        text: 'm indexes orientation states for fixed n and l. In external fields, m-degenerate states split (Zeeman effect), which directly connects orbital orientation to spectroscopy.',
      },
      {
        title: 'Nuclear Charge (Z)',
        text: 'Z controls Coulomb attraction. Larger Z contracts wavefunctions, raises binding strength, and modifies radial probability concentration for identical quantum numbers.',
      },
    ],
    equations: [
      {
        title: 'Hydrogenic Energy Level',
        math: 'E_n = -\\frac{13.6 Z^2}{n^2}\\,\\text{eV}',
        description: 'Energy depends strongly on Z and n; this relation is the baseline for shell ordering in one-electron atoms.',
      },
      {
        title: 'Angular Momentum Magnitude',
        math: '|\\mathbf{L}| = \\hbar\\sqrt{l(l+1)}',
        description: 'Quantized angular momentum emerges directly from boundary conditions of spherical harmonics.',
      },
    ],
    chartTitle: 'Parameter Sensitivity by Quantum Number',
    chartSubtitle: 'Relative impact of each parameter on size, directionality, and energy spacing.',
    chartData: [
      { parameter: 'n', size: 96, direction: 48, energy: 92 },
      { parameter: 'l', size: 42, direction: 95, energy: 54 },
      { parameter: 'm', size: 18, direction: 82, energy: 28 },
      { parameter: 'Z', size: 88, direction: 36, energy: 97 },
    ],
    chartLines: [
      { dataKey: 'size', name: 'Size Impact', color: '#39dcb1' },
      { dataKey: 'direction', name: 'Directionality Impact', color: '#30c6ff' },
      { dataKey: 'energy', name: 'Energy Impact', color: '#ffd265' },
    ],
    table: {
      columns: ['Parameter', 'Allowed Values', 'Primary Effect', 'Visualization Cue'],
      rows: [
        ['n', '1, 2, 3, ...', 'Shell size and baseline energy', 'Cloud radius and radial layering'],
        ['l', '0 to n-1', 'Shape family and angular nodes', 's/p/d/f morphology changes'],
        ['m', '-l to +l', 'Orientation state', 'Rotation and lobe alignment'],
        ['Z', '>= 1', 'Coulomb contraction and binding', 'Density shifts toward nucleus'],
      ],
    },
    imageSlots: [
      { title: 'Quantum Number Pyramid Graphic', description: 'Insert hierarchy image showing n, l, m dependencies.' },
      { title: 'Orbital Orientation Set', description: 'Add image panel for px, py, pz orientation examples.' },
      { title: 'Spectral Splitting Visual', description: 'Insert Zeeman/Stark splitting reference graphic.' },
    ],
    faqs: [
      {
        question: 'Why does m change orientation but not orbital family?',
        answer: 'Orbital family is set by l. The m value only selects orientation states within that same l family.',
      },
      {
        question: 'Can l ever be bigger than n - 1?',
        answer: 'No. The allowed values are restricted by boundary conditions, so l must satisfy 0 <= l <= n - 1.',
      },
      {
        question: 'Does changing Z change the allowed quantum numbers?',
        answer: 'No. Z changes energy and spatial contraction, but the allowed quantum-number structure remains the same.',
      },
    ],
  },
  {
    id: 'learn-orbital-geometry',
    label: '3. Geometry of Orbitals in Detail',
    shortLabel: 'Orbital Geometry',
    eyebrow: 'Learn Module 3',
    title: 'Geometry of Orbitals in Detail',
    message: 'Understand how spherical harmonics generate directional lobes, nodal planes, and family-specific 3D geometry.',
    theoryCards: [
      {
        title: 'Symmetry Families',
        text: 's orbitals preserve full rotational symmetry, while p, d, and f families break symmetry into directional lobes. Angular nodal structures partition space into alternating phase regions.',
      },
      {
        title: 'Orientation and Degeneracy',
        text: 'For each l value, multiple m states correspond to differently oriented but energetically degenerate solutions in isotropic Coulomb fields. Distortions and fields lift this degeneracy.',
      },
      {
        title: 'Real vs Complex Orbitals',
        text: 'Chemistry often uses real combinations of spherical harmonics to visualize lobes aligned with Cartesian axes. Physics workflows may retain complex forms for angular momentum operators.',
      },
      {
        title: 'Geometry to Bonding',
        text: 'Orbital geometry predicts overlap direction and bond strength. Sigma overlap is head-on, while pi overlap is lateral and strongly sensitive to lobe orientation and phase.',
      },
    ],
    equations: [
      {
        title: 'Wavefunction Separation',
        math: '\\psi_{n,l,m}(r,\\theta,\\phi)=R_{n,l}(r)Y_{l,m}(\\theta,\\phi)',
        description: 'All orbital geometry is encoded in angular and radial factors multiplied together.',
      },
      {
        title: 'Spherical Harmonic Core Form',
        math: 'Y_{l,m}(\\theta,\\phi)=N_{l,m}P_l^m(\\cos\\theta)e^{im\\phi}',
        description: 'Associated Legendre structure defines nodal planes/cones and azimuthal behavior.',
      },
    ],
    chartTitle: 'Shape Complexity vs Orbital Family',
    chartSubtitle: 'Comparing angular node growth and directional anisotropy.',
    chartData: orbitalComplexityData,
    chartLines: [
      { dataKey: 'nodes', name: 'Node Count Index', color: '#30c6ff' },
      { dataKey: 'directionality', name: 'Directionality Index', color: '#ff8a5b' },
      { dataKey: 'chemistry', name: 'Chemistry Relevance', color: '#ffd265' },
    ],
    table: {
      columns: ['Family', 'l Value', 'Typical Node Character', 'Chemistry Implication'],
      rows: [
        ['s', '0', 'No angular node', 'Isotropic overlap and shielding'],
        ['p', '1', 'Single nodal plane', 'Directional sigma and pi bonding'],
        ['d', '2', 'Two angular nodes', 'Transition-metal field splitting'],
        ['f', '3', 'Three angular nodes', 'Strong anisotropy and complex magnetism'],
      ],
    },
    imageSlots: [
      { title: 's, p, d, f Shape Atlas', description: 'Insert multi-panel orbital geometry image.' },
      { title: 'Cartesian Orientation Labels', description: 'Place annotated px/py/pz and d-orbital orientation visuals.' },
      { title: 'Crystal Field Splitting Diagram', description: 'Insert octahedral/tetrahedral splitting graphics.' },
    ],
    faqs: [
      {
        question: 'Why do p orbitals have two lobes?',
        answer: 'A single angular node splits space into two opposite-phase regions, creating the familiar two-lobe geometry.',
      },
      {
        question: 'Are real-orbital drawings and complex orbitals both valid?',
        answer: 'Yes. Real combinations are often used for chemistry visualization, while complex forms are common in angular-momentum operator work.',
      },
      {
        question: 'Do these orbital shapes represent electron paths?',
        answer: 'No. They represent probability amplitude structure, not classical trajectories of a particle orbiting like a planet.',
      },
    ],
  },
  {
    id: 'learn-nodes-antinodes',
    label: '4. Nodes and Antinodes',
    shortLabel: 'Nodes and Antinodes',
    eyebrow: 'Learn Module 4',
    title: 'Explaining Nodes and Antinodes',
    message: 'Translate interference mathematics into orbital zero-surfaces (nodes) and high-amplitude regions (antinodes).',
    theoryCards: [
      {
        title: 'Node Definition',
        text: 'A node is a locus where the wavefunction amplitude is exactly zero. It appears as a surface in 3D orbitals and separates regions of opposite phase.',
      },
      {
        title: 'Antinode Definition',
        text: 'Antinodes are regions of high amplitude magnitude and therefore high probability density after squaring. Orbital lobes correspond to antinode-rich regions.',
      },
      {
        title: 'Interference Logic',
        text: 'When amplitudes overlap with opposite sign, destructive interference creates nodal boundaries. Matching sign creates constructive reinforcement and antinode growth.',
      },
      {
        title: 'Practical Interpretation',
        text: 'In chemistry, nodal surfaces reduce overlap and can weaken bonding channels, while antinode alignment increases electron sharing in bonding orbitals.',
      },
    ],
    equations: [
      {
        title: 'Node Condition',
        math: '\\psi(r,\\theta,\\phi)=0',
        description: 'A node is strictly an amplitude zero condition before squaring.',
      },
      {
        title: 'Probability Density',
        math: '\\rho(r,\\theta,\\phi)=|\\psi(r,\\theta,\\phi)|^2',
        description: 'Antinodes become high-density volumes in probability maps.',
      },
    ],
    chartTitle: 'Interference Balance',
    chartSubtitle: 'A conceptual chart showing where constructive and destructive regimes dominate.',
    chartData: phaseInsightData,
    chartLines: [
      { dataKey: 'positive', name: 'Constructive Regime', color: '#ff6b6b' },
      { dataKey: 'negative', name: 'Destructive Regime', color: '#6beaff' },
    ],
    table: {
      columns: ['Region Type', 'Wavefunction Sign Behavior', 'Density Signature', 'Chemical Consequence'],
      rows: [
        ['Node', 'Sign changes across boundary', 'Near-zero density', 'Weak overlap pathway'],
        ['Antinode', 'Stable sign region', 'High density lobe', 'Strong overlap pathway'],
        ['Constructive overlap', 'Same sign combination', 'Density increase', 'Bond stabilization tendency'],
        ['Destructive overlap', 'Opposite sign combination', 'Cancellation zone', 'Antibonding tendency'],
      ],
    },
    imageSlots: [
      { title: 'Nodal Plane Cutaway', description: 'Insert a sliced orbital showing zero-density boundaries.' },
      { title: 'Phase Color Overlay', description: 'Add red/blue lobe map illustrating sign inversion.' },
      { title: 'Bonding vs Antibonding Pair', description: 'Insert two-orbital overlap comparison image.' },
    ],
    faqs: [
      {
        question: 'Is a node the same as "no electron exists there" forever?',
        answer: 'A node is where amplitude is exactly zero for that state, so probability there is zero for that stationary state description.',
      },
      {
        question: 'Do opposite phase colors mean opposite electric charge?',
        answer: 'No. They indicate opposite signs of the wavefunction phase, which matters for interference and overlap behavior.',
      },
      {
        question: 'Can antibonding orbitals still be occupied?',
        answer: 'Yes. They can be occupied depending on electron configuration, usually reducing net bond order when populated.',
      },
    ],
  },
  {
    id: 'learn-radial-angular-nodes',
    label: '5. Radial and Angular Nodes',
    shortLabel: 'Radial and Angular Nodes',
    eyebrow: 'Learn Module 5',
    title: 'Radial and Angular Nodes: Full Breakdown',
    message: 'Separate where nodal shells come from radial terms and where nodal planes/cones come from angular terms.',
    theoryCards: [
      {
        title: 'Radial Nodes',
        text: 'Radial nodes are spherical shells where the radial component crosses zero. Their count is n-l-1, and they control ring-like layering in radial distributions.',
      },
      {
        title: 'Angular Nodes',
        text: 'Angular nodes arise from spherical harmonic structure and equal l in count. They appear as planes or cones slicing through orbital volume.',
      },
      {
        title: 'Combined Node Budget',
        text: 'Total nodes always equal n-1 for hydrogenic states. The partition between radial and angular nodes determines whether orbitals look shell-heavy or direction-heavy.',
      },
      {
        title: 'Visualization Strategy',
        text: 'Use slice views to inspect radial shell boundaries, then rotate to identify angular nodal planes. Both views are needed for complete state interpretation.',
      },
    ],
    equations: [
      {
        title: 'Radial Node Count',
        math: 'N_r = n-l-1',
        description: 'The radial component contributes this many spherical node shells.',
      },
      {
        title: 'Angular Node Count',
        math: 'N_a = l',
        description: 'Angular structure contributes planes/cones defined by spherical harmonics.',
      },
    ],
    chartTitle: 'Node Partition by Orbital Family',
    chartSubtitle: 'Comparison of radial and angular node contributions for selected states.',
    chartData: [
      { state: '2s', radial: 1, angular: 0 },
      { state: '2p', radial: 0, angular: 1 },
      { state: '3p', radial: 1, angular: 1 },
      { state: '3d', radial: 0, angular: 2 },
      { state: '4f', radial: 0, angular: 3 },
    ],
    chartLines: [
      { dataKey: 'radial', name: 'Radial Nodes', color: '#57d3ff' },
      { dataKey: 'angular', name: 'Angular Nodes', color: '#ff8a5b' },
    ],
    table: {
      columns: ['State', 'n', 'l', 'Radial Nodes', 'Angular Nodes'],
      rows: [
        ['2s', '2', '0', '1', '0'],
        ['2p', '2', '1', '0', '1'],
        ['3p', '3', '1', '1', '1'],
        ['3d', '3', '2', '0', '2'],
      ],
    },
    imageSlots: [
      { title: 'Radial Shell Cross-Section', description: 'Insert shell cutaway with node ring annotations.' },
      { title: 'Angular Node Planes', description: 'Insert p/d/f nodal plane and cone overlays.' },
      { title: 'Combined Node Diagram', description: 'Insert hybrid view showing both radial and angular nodes.' },
    ],
    faqs: [
      {
        question: 'Is the total node count always n - 1?',
        answer: 'For hydrogenic orbitals, yes. The total comes from radial nodes plus angular nodes and equals n - 1.',
      },
      {
        question: 'How do I separate radial and angular nodes in the simulator?',
        answer: 'Use Slice View and camera rotation. Radial nodes appear as shell-like boundaries; angular nodes appear as planes or cones.',
      },
      {
        question: 'Why can a d orbital have zero radial nodes?',
        answer: 'Because radial node count is n - l - 1. For 3d, that is 3 - 2 - 1 = 0.',
      },
    ],
  },
  {
    id: 'learn-probability-functions',
    label: '6. Probability Distribution Functions',
    shortLabel: 'Probability Functions',
    eyebrow: 'Learn Module 6',
    title: 'Probability Distribution Functions: All Core Types',
    message: 'Understand point density, radial distribution, angular weighting, cumulative probability, and expectation-value interpretations.',
    theoryCards: [
      {
        title: 'Point Probability Density',
        text: 'The local probability density is |psi|^2. In 3D visualization, denser point clouds correspond to higher local measurement likelihood.',
      },
      {
        title: 'Radial Distribution Function',
        text: 'Radial probability includes Jacobian weighting: P(r)=4pi r^2|R(r)|^2. This creates peaks away from the origin even when local density is high near small r.',
      },
      {
        title: 'Angular Distribution',
        text: 'Angular probability comes from |Y_lm(theta,phi)|^2 and encodes direction preference. It determines where lobes and nodal planes appear.',
      },
      {
        title: 'Cumulative Probability',
        text: 'Integrated probability up to radius r indicates how likely the electron is within a finite sphere, useful for comparing contracted vs diffuse states.',
      },
    ],
    equations: [
      {
        title: 'Local Density',
        math: '\\rho(r,\\theta,\\phi)=|\\psi_{n,l,m}(r,\\theta,\\phi)|^2',
        description: 'Core quantity behind all scatter rendering intensity decisions.',
      },
      {
        title: 'Radial Distribution',
        math: 'P(r)=4\\pi r^2|R_{n,l}(r)|^2',
        description: 'Used to interpret where probability mass is concentrated over radius.',
      },
    ],
    chartTitle: 'Distribution Function Comparison',
    chartSubtitle: 'Conceptual magnitude comparison of local, radial, angular, and cumulative measures.',
    chartData: [
      { radius: 'r1', local: 84, radial: 36, angular: 66, cumulative: 14 },
      { radius: 'r2', local: 70, radial: 62, angular: 74, cumulative: 32 },
      { radius: 'r3', local: 49, radial: 81, angular: 68, cumulative: 56 },
      { radius: 'r4', local: 31, radial: 59, angular: 48, cumulative: 78 },
      { radius: 'r5', local: 17, radial: 28, angular: 27, cumulative: 92 },
    ],
    chartLines: [
      { dataKey: 'local', name: 'Local Density', color: '#6beaff' },
      { dataKey: 'radial', name: 'Radial Distribution', color: '#ffd265' },
      { dataKey: 'angular', name: 'Angular Weighting', color: '#39dcb1' },
      { dataKey: 'cumulative', name: 'Cumulative Probability', color: '#ff8a5b' },
    ],
    table: {
      columns: ['Function Type', 'Expression Core', 'Physical Meaning', 'Best Use Case'],
      rows: [
        ['Local density', '|psi|^2', 'Pointwise likelihood', '3D cloud coloring'],
        ['Radial distribution', '4pi r^2|R|^2', 'Probability by shell radius', 'Peak radius interpretation'],
        ['Angular distribution', '|Ylm|^2', 'Directional preference', 'Lobe orientation studies'],
        ['Cumulative distribution', 'Integral from 0 to r', 'Contained probability mass', 'Contraction comparisons'],
      ],
    },
    imageSlots: [
      { title: 'Radial Curve Panel', description: 'Insert radial distribution graph image with labeled peaks.' },
      { title: 'Angular Heatmap', description: 'Add spherical angular probability map image.' },
      { title: 'Cumulative Probability Plot', description: 'Insert cumulative curve with threshold markers.' },
    ],
    faqs: [
      {
        question: 'What is the difference between |psi|^2 and P(r)?',
        answer: '|psi|^2 is local point density, while P(r) includes the 4pi r^2 geometric factor and describes probability by radius shell.',
      },
      {
        question: 'Why does radial probability often peak away from the nucleus?',
        answer: 'The r^2 factor expands shell volume with radius, so shell probability can peak away from the center even when local density is high near small r.',
      },
      {
        question: 'Can cumulative probability exceed 1?',
        answer: 'No. Properly normalized cumulative probability approaches 1 as radius approaches infinity.',
      },
    ],
  },
  {
    id: 'learn-solved-examples',
    label: '7. Solved Examples',
    shortLabel: 'Solved Examples',
    eyebrow: 'Learn Module 7',
    title: 'Solved Examples: From Inputs to Interpretation',
    message: 'Walk through complete solved workflows that connect chosen quantum numbers to geometry, nodes, and probability outcomes.',
    theoryCards: [
      {
        title: 'Example 1: 2p (n=2, l=1, m=0, Z=1)',
        text: 'Predict one angular node, zero radial nodes, and dumbbell geometry oriented by m basis. Validate by rotating and slicing in the simulator.',
      },
      {
        title: 'Example 2: 3s (n=3, l=0, m=0, Z=1)',
        text: 'Predict spherical symmetry and two radial nodes. Confirm by radial graph peaks and shell boundary transitions in slice mode.',
      },
      {
        title: 'Example 3: 4d (n=4, l=2, m=1, Z=2)',
        text: 'Predict two angular nodes, one radial node, and contracted density versus hydrogen due to larger Z.',
      },
      {
        title: 'Interpretation Checklist',
        text: 'Always compute node counts first, then predict shape family, then verify with phase display and radial chart. This sequence avoids interpretation errors.',
      },
    ],
    equations: [
      {
        title: 'Total Node Rule',
        math: 'N_{\\text{total}} = n-1 = (n-l-1)+l',
        description: 'Use this to cross-check consistency in every solved example.',
      },
      {
        title: 'Expectation Radius (Hydrogenic Trend)',
        math: '\\langle r \\rangle \\propto \\frac{n^2}{Z}',
        description: 'Gives quick intuition for cloud size scaling before simulation.',
      },
    ],
    chartTitle: 'Solved Workflow Confidence Curve',
    chartSubtitle: 'Conceptual confidence gain while progressing through a worked problem.',
    chartData: workflowChartData,
    chartLines: [
      { dataKey: 'confidence', name: 'Interpretation Confidence', color: '#30c6ff' },
    ],
    table: {
      columns: ['Example', 'Input Set', 'Predicted Nodes', 'Predicted Geometry', 'Validation Step'],
      rows: [
        ['2p', 'n=2, l=1, m=0, Z=1', '0 radial, 1 angular', 'Two-lobed', 'Rotate + phase toggle'],
        ['3s', 'n=3, l=0, m=0, Z=1', '2 radial, 0 angular', 'Spherical shells', 'Slice + radial chart'],
        ['4d', 'n=4, l=2, m=1, Z=2', '1 radial, 2 angular', 'Clover-like', 'Compare to Z=1 contraction'],
        ['5f', 'n=5, l=3, m=2, Z=3', '1 radial, 3 angular', 'Complex multi-lobe', 'Node checklist + orientation'],
      ],
    },
    imageSlots: [
      { title: 'Solved Example Worksheet', description: 'Insert worksheet screenshot with parameter-to-result notes.' },
      { title: 'Orbital Snapshot Gallery', description: 'Add side-by-side images for solved examples.' },
      { title: 'Instructor Annotation Layer', description: 'Insert marked-up image showing node callouts.' },
    ],
    faqs: [
      {
        question: 'What is the best order for solving a new orbital example?',
        answer: 'Use this order: compute node counts, predict geometry, then verify with phase and slice tools.',
      },
      {
        question: 'Why should I compare the same state at different Z values?',
        answer: 'It isolates nuclear-charge effects, making contraction and density redistribution easier to observe and explain.',
      },
      {
        question: 'If my prediction and visualization do not match, what should I check first?',
        answer: 'First check n, l, m limits and whether phase/slice settings changed your interpretation of internal structure.',
      },
    ],
  },
];

const learnMenuLinks = [
  ...learnTopics.map((topic) => ({ id: topic.id, label: topic.label, shortLabel: topic.shortLabel })),
];

const infoPageLinks = [
  { id: 'about', label: 'About' },
  { id: 'terms', label: 'Terms and Conditions' },
  { id: 'privacy', label: 'Privacy Policy' },
  { id: 'contact', label: 'Contact' },
];

const PAGE_ROUTE_MAP = {
  welcome: '#/welcome',
  chemistry: '#/chemistry',
  howto: '#/how-to-use',
  faqs: '#/faqs',
  'learn-quantum-theory': '#/learn/quantum-theory',
  'learn-quantum-numbers-detail': '#/learn/quantum-numbers-detail',
  'learn-orbital-geometry': '#/learn/orbital-geometry',
  'learn-nodes-antinodes': '#/learn/nodes-and-antinodes',
  'learn-radial-angular-nodes': '#/learn/radial-angular-nodes',
  'learn-probability-functions': '#/learn/probability-functions',
  'learn-solved-examples': '#/learn/solved-examples',
  simulator: '#/simulator',
  about: '#/about',
  terms: '#/terms-and-conditions',
  privacy: '#/privacy-policy',
  contact: '#/contact',
};

const HASH_PAGE_MAP = Object.entries(PAGE_ROUTE_MAP).reduce((acc, [page, route]) => {
  acc[route] = page;
  return acc;
}, {});

const learnTopicMap = learnTopics.reduce((acc, topic) => {
  acc[topic.id] = topic;
  return acc;
}, {});

const normalizeHashRoute = (rawHash) => {
  if (!rawHash) return '';
  const hash = rawHash.startsWith('#') ? rawHash : `#${rawHash}`;
  return hash.endsWith('/') ? hash.slice(0, -1) : hash;
};

const getPageFromHash = (rawHash) => {
  const normalized = normalizeHashRoute(rawHash);
  if (normalized === '#/learn') return 'learn-quantum-theory';
  return HASH_PAGE_MAP[normalized] || null;
};

const isLearnPage = (pageId) => typeof pageId === 'string' && pageId.startsWith('learn-');

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

const FAQAccordion = ({ question, answer }) => {
  const [isOpen, setIsOpen] = useState(false);
  const answerRef = useRef(null);
  const [maxHeight, setMaxHeight] = useState(0);

  useEffect(() => {
    if (!isOpen) {
      setMaxHeight(0);
      return;
    }

    if (answerRef.current) {
      setMaxHeight(answerRef.current.scrollHeight);
    }
  }, [isOpen, answer]);

  useEffect(() => {
    if (!isOpen || typeof window === 'undefined') return undefined;

    const syncHeight = () => {
      if (answerRef.current) {
        setMaxHeight(answerRef.current.scrollHeight);
      }
    };

    window.addEventListener('resize', syncHeight);
    return () => window.removeEventListener('resize', syncHeight);
  }, [isOpen]);

  return (
    <article className={`faq-item ${isOpen ? 'is-open' : ''}`.trim()}>
      <button
        type="button"
        className="faq-trigger"
        onClick={() => setIsOpen((prev) => !prev)}
        aria-expanded={isOpen}
      >
        <span className="faq-question">{question}</span>
        <span className={`faq-triangle ${isOpen ? 'is-open' : ''}`.trim()} aria-hidden="true"></span>
      </button>
      <div className="faq-answer-wrap" style={{ maxHeight: `${maxHeight}px` }} aria-hidden={!isOpen}>
        <div className="faq-answer" ref={answerRef}>
          <p>{answer}</p>
        </div>
      </div>
    </article>
  );
};

const QuantumSymbolMark = ({ symbol }) => {
  const normalizedSymbol = String(symbol).toLowerCase();

  if (normalizedSymbol === 'n') {
    return (
      <div className="symbol-mark symbol-mark-n" aria-hidden="true">
        <span className="circle-ring r1"></span>
        <span className="circle-ring r2"></span>
        <span className="circle-ring r3"></span>
        <span className="nucleus-core"></span>
        <span className="symbol-letter">n</span>
      </div>
    );
  }

  if (normalizedSymbol === 'l') {
    return (
      <div className="symbol-mark symbol-mark-l" aria-hidden="true">
        <span className="theta-mark">{"\u03B8"}</span>
        <span className="symbol-letter">l</span>
      </div>
    );
  }

  if (normalizedSymbol === 'm') {
    return (
      <div className="symbol-mark symbol-mark-m" aria-hidden="true">
        <span className="compass">
          <span className="needle top"></span>
          <span className="needle bottom"></span>
        </span>
        <span className="symbol-letter">m</span>
      </div>
    );
  }

  if (normalizedSymbol === 'z') {
    return (
      <div className="symbol-mark symbol-mark-z" aria-hidden="true">
        <span className="charge-ring cr1"></span>
        <span className="charge-ring cr2"></span>
        <span className="charge-ring cr3"></span>
        <span className="charge-core"></span>
        <span className="charge-sign">+</span>
        <span className="symbol-letter">z</span>
      </div>
    );
  }

  return <div className="symbol-mark">{symbol}</div>;
};

const SymbolCardGrid = () => (
  <div className="symbol-grid">
    {quantumSymbolCards.map((card) => (
      <article className="symbol-card" key={card.symbol}>
        <QuantumSymbolMark symbol={card.symbol} />
        <h3>{card.title}</h3>
        <p className="symbol-range">{card.range}</p>
        <p>{card.insight}</p>
      </article>
    ))}
  </div>
);

const DataTable = ({ columns, rows, caption, compact = false }) => (
  <div className={`table-wrap ${compact ? 'is-compact' : ''}`.trim()}>
    <table className={`quantum-table ${compact ? 'is-compact' : ''}`.trim()}>
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

const ContactChips = () => (
  <>
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
  </>
);

const LearnTopicPage = ({ topic, onNavigate }) => {
  const xAxisKey = topic.chartData?.length > 0 ? Object.keys(topic.chartData[0])[0] : 'x';

  return (
    <InfoPageLayout
      eyebrow={topic.eyebrow}
      title={topic.title}
      message={topic.message}
      heroClassName="hero-howto"
      onNavigate={onNavigate}
      showLearnSidebar={true}
      activeLearnId={topic.id}
      ctaLabel="Open Simulator"
      ctaTarget="simulator"
    >
      <InfoSection
        title="Core Theory"
        subtitle="Detailed conceptual notes for this module."
      >
        <div className="info-grid two-column">
          {topic.theoryCards.map((card) => (
            <article className="glass-card" key={card.title}>
              <h3>{card.title}</h3>
              <p>{card.text}</p>
            </article>
          ))}
        </div>
      </InfoSection>

      <InfoSection
        title="Equations Driving This Topic"
        subtitle="Use these equations as the formal bridge between theory and what you see in 3D."
      >
        <div className="info-grid two-column">
          {topic.equations.map((equation) => (
            <article className="glass-card" key={equation.title}>
              <h3>{equation.title}</h3>
              <ExpandableFormula className="compact" math={equation.math} />
              <p>{equation.description}</p>
            </article>
          ))}
        </div>
      </InfoSection>

      <InfoSection title={topic.chartTitle} subtitle={topic.chartSubtitle}>
        <article className="glass-card chart-card">
          <ResponsiveContainer width="100%" height={280}>
            <LineChart data={topic.chartData} margin={{ top: 10, right: 12, left: 2, bottom: 0 }}>
              <CartesianGrid stroke="#2f4660" strokeDasharray="4 4" />
              <XAxis dataKey={xAxisKey} stroke="#a7c9e8" />
              <YAxis stroke="#a7c9e8" />
              <Tooltip contentStyle={chartTooltipStyle} />
              <Legend />
              {topic.chartLines.map((line) => (
                <Line key={line.dataKey} type="monotone" dataKey={line.dataKey} name={line.name} stroke={line.color} strokeWidth={2.2} dot={false} />
              ))}
            </LineChart>
          </ResponsiveContainer>
        </article>
      </InfoSection>

      <InfoSection title="Structured Reference Table" subtitle="Fast lookup summary for teaching, revision, and solved workflows.">
        <DataTable columns={topic.table.columns} rows={topic.table.rows} compact />
      </InfoSection>

      {Array.isArray(topic.faqs) && topic.faqs.length > 0 && (
        <InfoSection title="Module FAQs" subtitle="Quick answers for common questions in this module.">
          <div className="faq-list">
            {topic.faqs.map((faq) => (
              <FAQAccordion key={faq.question} question={faq.question} answer={faq.answer} />
            ))}
          </div>
        </InfoSection>
      )}

      <InfoSection title="Image Slots" subtitle="Use these placeholders for portraits, orbital renders, and annotated diagrams.">
        <div className="learn-image-grid">
          {topic.imageSlots.map((slot) => (
            <article className="glass-card learn-image-card" key={slot.title}>
              <h3>{slot.title}</h3>
              <div className="learn-image-placeholder">Image Placeholder</div>
              <p>{slot.description}</p>
            </article>
          ))}
        </div>
      </InfoSection>
    </InfoPageLayout>
  );
};

const InfoFooter = ({ onNavigate }) => (
  <footer className="site-footer">
    <div className="site-footer-grid">
      <section className="footer-about">
        <h3>Company</h3>
        <p style={{ marginBottom: '12px', color: '#c0d9ef', fontSize: '0.85rem', lineHeight: '1.4' }}>
          Quantum Orbital Explorer is a specialized studio built to deliver professional, physics-accurate visualizations for modern chemistry research and classrooms.
        </p>
        <div className="footer-link-list footer-company-links">
          <button className="footer-link-button" type="button" onClick={() => onNavigate('about')}>about</button>
          <button className="footer-link-button" type="button" onClick={() => onNavigate('terms')}>terms</button>
          <button className="footer-link-button" type="button" onClick={() => onNavigate('contact')}>contact</button>
        </div>
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
        <ContactChips />
      </section>

      <section className="footer-privacy">
          <h3>Privacy Policy</h3>
          <ul style={{ margin: "0 0 12px 0", paddingLeft: "16px", color: "#c0d9ef", fontSize: "0.85rem", lineHeight: "1.4" }}>
            <li style={{ marginBottom: "6px" }}>Use inputs strictly for visual rendering without tracking profiles.</li>
            <li style={{ marginBottom: "6px" }}>Run operational logs under tight retention to debug latency.</li>
            <li>Protect workflows without injected ad-sale pipelines.</li>
          </ul>
          <div className="footer-link-list">
            <button className="footer-link-button" type="button" onClick={() => onNavigate('privacy')}>read full privacy-policy</button>
          </div>
        </section>
    </div>
    <div className="site-footer-bottom">
      <span>2026 Quantum Orbital Explorer. Built for education, labs, and modern chemistry classrooms.</span>
    </div>
  </footer>
);

const InfoPageLayout = ({
  eyebrow,
  title,
  message,
  children,
  onNavigate,
  ctaLabel,
  ctaTarget,
  heroClassName = '',
  showLearnSidebar = false,
  activeLearnId = null,
}) => (
  <div className={`info-page-shell ${showLearnSidebar ? 'has-learn-sidebar' : ''}`.trim()}>
    <div className={`info-page-main-layout ${showLearnSidebar ? 'has-learn-sidebar' : ''}`.trim()}>
      {showLearnSidebar && (
        <aside className="learn-sidebar" aria-label="Learn modules">
          <div className="learn-sidebar-inner">
            <h4>Learn Modules</h4>
            <div className="learn-sidebar-links">
              {learnMenuLinks.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={`learn-sidebar-link ${activeLearnId === item.id ? 'is-active' : ''}`.trim()}
                  onClick={() => onNavigate(item.id)}
                >
                  {item.label}
                </button>
              ))}
            </div>
          </div>
        </aside>
      )}

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
    </div>

    <InfoFooter onNavigate={onNavigate} />
  </div>
);

// --- Main App Component ---

const App = () => {
  const [currentPage, setCurrentPage] = useState(() => {
    if (typeof window === 'undefined') return 'welcome';
    return getPageFromHash(window.location.hash) || 'welcome';
  }); // welcome, chemistry, howto, faqs, learn-*, simulator, about, terms, privacy, contact

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

  const navigateToPage = (pageId) => {
    if (!PAGE_ROUTE_MAP[pageId]) return;

    setCurrentPage(pageId);

    if (typeof window !== 'undefined') {
      const nextHash = PAGE_ROUTE_MAP[pageId];
      if (window.location.hash !== nextHash) {
        window.location.hash = nextHash;
      }
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

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
    if (typeof window === 'undefined') return undefined;

    const syncPageWithHash = () => {
      const hashPage = getPageFromHash(window.location.hash);
      if (hashPage) {
        setCurrentPage(hashPage);
      }
    };

    if (!getPageFromHash(window.location.hash)) {
      window.location.hash = PAGE_ROUTE_MAP.welcome;
    } else {
      syncPageWithHash();
    }

    window.addEventListener('hashchange', syncPageWithHash);
    return () => window.removeEventListener('hashchange', syncPageWithHash);
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
          <button type="button" onClick={() => navigateToPage('welcome')} style={navButtonStyle(currentPage === 'welcome')}>Welcome</button>
          <button type="button" onClick={() => navigateToPage('chemistry')} style={navButtonStyle(currentPage === 'chemistry')}>Chemistry Concepts</button>
          <button type="button" onClick={() => navigateToPage('howto')} style={navButtonStyle(currentPage === 'howto')}>How To Use</button>
          <button type="button" onClick={() => navigateToPage('faqs')} style={navButtonStyle(currentPage === 'faqs')}>FAQs</button>

          <div className="learn-nav-item">
            <button
              type="button"
              className="learn-nav-trigger"
              onClick={() => navigateToPage('learn-quantum-theory')}
              style={navButtonStyle(isLearnPage(currentPage))}
            >
              Learn
            </button>
            <div
              className="learn-dropdown"
              role="menu"
              aria-label="Learn pages"
              onWheelCapture={(event) => event.preventDefault()}
              onWheel={(event) => event.preventDefault()}
              onTouchMove={(event) => event.preventDefault()}
            >
              {learnMenuLinks.map((item) => (
                <button
                  key={item.id}
                  type="button"
                  className={`learn-dropdown-item ${currentPage === item.id ? 'is-active' : ''}`.trim()}
                  onClick={() => navigateToPage(item.id)}
                >
                  {item.shortLabel || item.label}
                </button>
              ))}
            </div>
          </div>

          <button type="button" onClick={() => navigateToPage('simulator')} style={navButtonStyle(currentPage === 'simulator')}><strong>Simulator</strong></button>
        </div>
      </div>

      {currentPage === 'welcome' && (
        <InfoPageLayout
          eyebrow="Interactive Quantum Learning Platform"
          title="Quantum Orbital Explorer"
          message="A visual learning and simulation workspace for understanding orbital behavior, chemistry intuition, and quantum model interpretation."
          heroClassName="hero-welcome"
          onNavigate={navigateToPage}
          ctaLabel="Continue to Chemistry Concepts"
          ctaTarget="chemistry"
        >
          <InfoSection
            title="Welcome Overview"
            subtitle="Core context after the main header."
          >
            <div className="info-grid two-column">
              <article className="glass-card">
                <h3>What is this tool?</h3>
                <p>
                  This tool is an interactive orbital visualization platform. It combines a 3D simulator, radial probability graph, and guided theory pages in one environment.
                </p>
              </article>

              <article className="glass-card">
                <h3>Why is it useful?</h3>
                <p>
                  It turns abstract concepts into inspectable geometry. Students and researchers can immediately test predictions about shape, nodes, phase, and shell behavior.
                </p>
              </article>

              <article className="glass-card">
                <h3>Who is it for?</h3>
                <p>
                  It is for high-school and college learners, chemistry and physics instructors, and anyone who wants intuition for wave mechanics through visual exploration.
                </p>
              </article>

              <article className="glass-card">
                <h3>What can users visualize?</h3>
                <p>
                  Users can visualize orbital clouds, phase polarity, nodal surfaces, orientation changes with quantum numbers, and radial probability trends in real time.
                </p>
              </article>
            </div>
          </InfoSection>

          <InfoSection
            title="Equation Driving The Simulator"
            subtitle="Mathematical structure behind the rendered orbital cloud."
          >
            <div className="info-grid two-column">
              <article className="glass-card">
                <h3>Formula Components</h3>
                <p>
                  The simulator uses the stationary one-electron quantum model where the total state is built from a radial part and an angular part. The backend computes density samples from these components and the frontend renders them as a 3D point cloud.
                </p>
                <p>
                  <strong>Formula form:</strong> <InlineMath math={'\\psi_{n,l,m}(r,\\theta,\\phi)=R_{n,l}(r)Y_l^m(\\theta,\\phi)'} />
                </p>
              </article>

              <article className="glass-card">
                <h3>Wavefunction Equation</h3>
                <ExpandableFormula className="compact" math={'\\psi_{n,l,m}(r,\\theta,\\phi)=R_{n,l}(r)Y_l^m(\\theta,\\phi)'} />
                <p>
                  The rendered probability cloud is sampled from <InlineMath math={'|\\psi_{n,l,m}(r,\\theta,\\phi)|^2'} />, which maps where the electron is most likely to be observed.
                </p>
              </article>
            </div>

            <article className="glass-card equation-cta-card">
              <p>For more detailed understanding, please check out the Learn modules.</p>
              <button className="hero-cta equation-learn-button" type="button" onClick={() => navigateToPage('learn-quantum-theory')}>
                Checkout Learn Modules
              </button>
            </article>
          </InfoSection>
        </InfoPageLayout>
      )}

      {currentPage === 'chemistry' && (
        <InfoPageLayout
          eyebrow="Quantum Foundations"
          title="Orbital Structures and Wavefunction Phase"
          message="Understand the building blocks of matter. By mapping atomic orbitals and phase, we can predict bonding, geometry, and chemical reactivity."
          heroClassName="hero-chemistry"
          onNavigate={navigateToPage}
          ctaLabel="Continue to How To Use"
          ctaTarget="howto"
        >
          <InfoSection
            title="Deciphering The Quantum Numbers"
            subtitle="Foundational parameter definitions in the same page format before orbital-family exploration."
          >
            <SymbolCardGrid />
          </InfoSection>

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
          onNavigate={navigateToPage}
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

      {currentPage === 'faqs' && (
        <InfoPageLayout
          eyebrow="Support and Guidance"
          title="FAQs"
          message="Common questions about the website, simulator workflow, accuracy scope, and usage policy."
          heroClassName="hero-howto"
          onNavigate={navigateToPage}
          ctaLabel="Open Simulator"
          ctaTarget="simulator"
        >
          <InfoSection
            title="Frequently Asked Questions"
            subtitle="Tap a question to expand the answer."
          >
            <div className="faq-list">
              {siteFaqs.map((faq) => (
                <FAQAccordion key={faq.question} question={faq.question} answer={faq.answer} />
              ))}
            </div>
          </InfoSection>

          <InfoSection
            title="Need More Help?"
            subtitle="Open support channels or continue into guided learning."
          >
            <div className="info-grid two-column">
              <article className="glass-card">
                <h3>Still Have a Question?</h3>
                <p>
                  If your case is not covered in the list above, use the contact page and share your exact quantum settings so support can reproduce your result quickly.
                </p>
                <button className="hero-cta" type="button" onClick={() => navigateToPage('contact')}>
                  Go to Contact
                </button>
              </article>

              <article className="glass-card">
                <h3>Build Stronger Fundamentals</h3>
                <p>
                  For concept-first clarity, use the Learn modules where each topic includes theory, equations, visuals, and mini FAQs.
                </p>
                <button className="hero-cta" type="button" onClick={() => navigateToPage('learn-quantum-theory')}>
                  Open Learn Modules
                </button>
              </article>
            </div>
          </InfoSection>
        </InfoPageLayout>
      )}

      {learnTopicMap[currentPage] && (
        <LearnTopicPage topic={learnTopicMap[currentPage]} onNavigate={navigateToPage} />
      )}

      {currentPage === 'about' && (
        <InfoPageLayout
          eyebrow="Information Page"
          title="About Quantum Orbital Explorer"
          message="This dedicated URL page describes product purpose, architecture, and intended educational outcomes."
          heroClassName="hero-howto"
          onNavigate={navigateToPage}
          ctaLabel="Open Learn Modules"
          ctaTarget="learn-quantum-theory"
        >
          <InfoSection title="Platform Purpose" subtitle="Why this tool exists and what it delivers.">
            <div className="info-grid two-column">
              <article className="glass-card">
                <h3>Mission</h3>
                <p>
                  Quantum Orbital Explorer is designed to bridge theory and intuition through high-fidelity visualization, structured learning pages, and guided controls.
                </p>
              </article>
              <article className="glass-card">
                <h3>Outcome</h3>
                <p>
                  Users move from formulas and abstract rules to concrete orbital interpretations by directly observing geometry, nodes, and probability behavior.
                </p>
              </article>
            </div>
          </InfoSection>

          <InfoSection title="System Overview" subtitle="Core components and responsibilities.">
            <DataTable
              columns={['Layer', 'Role', 'Key Output']}
              rows={[
                ['Frontend', 'Interactive controls and visual layout', '3D orbital cloud and learning pages'],
                ['Backend', 'Radial/scatter calculations and caching', 'Binary point payloads and radial arrays'],
                ['Learning Hub', 'Theory/equations/charts/tables', 'Curriculum-style explainers and references'],
              ]}
            />
          </InfoSection>
        </InfoPageLayout>
      )}

      {currentPage === 'terms' && (
        <InfoPageLayout
          eyebrow="Information Page"
          title="Terms and Conditions"
          message="This dedicated URL page outlines acceptable use, content scope, and operational limitations."
          heroClassName="hero-howto"
          onNavigate={navigateToPage}
          ctaLabel="Back to Welcome"
          ctaTarget="welcome"
        >
          <InfoSection title="Usage Terms" subtitle="Summary terms for learning and simulation usage.">
            <DataTable
              columns={['Term Area', 'Condition', 'Practical Meaning']}
              rows={[
                ['Educational Use', 'Content is educational and exploratory', 'Use outputs as learning support'],
                ['Scientific Review', 'Interpretations should be reviewed in context', 'Validate for formal assessments'],
                ['Availability', 'Service may change as modules evolve', 'Layouts and content can be updated'],
                ['External Media', 'Image placeholders may link to third-party assets', 'Ensure rights for uploaded media'],
              ]}
            />
          </InfoSection>

          <InfoSection title="Policy Links" subtitle="Dedicated routes for legal and contact pages.">
            <div className="footer-link-list">
              {infoPageLinks.map((link) => (
                <button key={link.id} className="footer-link-button" type="button" onClick={() => navigateToPage(link.id)}>
                  {link.label}: {PAGE_ROUTE_MAP[link.id]}
                </button>
              ))}
            </div>
          </InfoSection>
        </InfoPageLayout>
      )}

      {currentPage === 'privacy' && (
        <InfoPageLayout
          eyebrow="Information Page"
          title="Privacy Policy"
          message="This dedicated URL page explains what data is used for simulation and how operational logs are treated."
          heroClassName="hero-howto"
          onNavigate={navigateToPage}
          ctaLabel="Go to Contact"
          ctaTarget="contact"
        >
          <InfoSection title="Data Handling" subtitle="Summary of data categories and usage.">
            <DataTable
              columns={['Data Type', 'Purpose', 'Retention']}
              rows={[
                ['Simulation parameters', 'Generate scatter and radial outputs', 'Short-lived request scope'],
                ['Operational logs', 'Performance and reliability debugging', 'Limited retention window'],
                ['Contact details', 'Respond to user inquiries', 'Retained only for communication workflow'],
              ]}
            />
          </InfoSection>

          <InfoSection title="User Control" subtitle="Navigation links to supporting policy pages.">
            <div className="footer-link-list">
              <button className="footer-link-button" type="button" onClick={() => navigateToPage('about')}>About URL: {PAGE_ROUTE_MAP.about}</button>
              <button className="footer-link-button" type="button" onClick={() => navigateToPage('terms')}>Terms URL: {PAGE_ROUTE_MAP.terms}</button>
              <button className="footer-link-button" type="button" onClick={() => navigateToPage('contact')}>Contact URL: {PAGE_ROUTE_MAP.contact}</button>
            </div>
          </InfoSection>
        </InfoPageLayout>
      )}

      {currentPage === 'contact' && (
        <InfoPageLayout
          eyebrow="Information Page"
          title="Contact"
          message="This dedicated URL page centralizes support channels, response windows, and communication workflow."
          heroClassName="hero-howto"
          onNavigate={navigateToPage}
          ctaLabel="Open Simulator"
          ctaTarget="simulator"
        >
          <InfoSection title="Primary Contact Channels">
            <article className="glass-card">
              <ContactChips />
            </article>
          </InfoSection>

          <InfoSection title="Support Workflow" subtitle="How requests are processed.">
            <DataTable
              columns={['Stage', 'What Happens', 'Expected Result']}
              rows={[
                ['Message received', 'Contact details and issue category are captured', 'Ticket context is prepared'],
                ['Technical review', 'Simulation or content issue is investigated', 'Actionable response prepared'],
                ['Follow-up', 'Resolution steps are shared with user', 'Issue closed or escalated'],
              ]}
            />
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



