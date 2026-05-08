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

const moduleThreeOrientationData = [
  { family: 's', orientations: 1, anisotropy: 1 },
  { family: 'p', orientations: 3, anisotropy: 5 },
  { family: 'd', orientations: 5, anisotropy: 8 },
  { family: 'f', orientations: 7, anisotropy: 10 },
];

const moduleFourWaveInterferenceData = [
  { point: 'A', amplitude: 0.9, density: 0.81 },
  { point: 'B', amplitude: 0.45, density: 0.2 },
  { point: 'C (Node)', amplitude: 0.0, density: 0.0 },
  { point: 'D', amplitude: -0.45, density: 0.2 },
  { point: 'E', amplitude: -0.9, density: 0.81 },
];

const moduleFiveNodeGrowthData = [
  { shell: 'n=1', total: 0, radial: 0, angularCap: 0 },
  { shell: 'n=2', total: 1, radial: 1, angularCap: 1 },
  { shell: 'n=3', total: 2, radial: 2, angularCap: 2 },
  { shell: 'n=4', total: 3, radial: 3, angularCap: 3 },
  { shell: 'n=5', total: 4, radial: 4, angularCap: 4 },
];

const moduleSixCumulativeShellData = [
  { radiusA0: 0.5, oneS: 0.09, twoS: 0.02, twoP: 0.01 },
  { radiusA0: 1.0, oneS: 0.32, twoS: 0.08, twoP: 0.03 },
  { radiusA0: 1.5, oneS: 0.58, twoS: 0.18, twoP: 0.09 },
  { radiusA0: 2.0, oneS: 0.76, twoS: 0.31, twoP: 0.18 },
  { radiusA0: 3.0, oneS: 0.93, twoS: 0.57, twoP: 0.41 },
  { radiusA0: 4.0, oneS: 0.98, twoS: 0.74, twoP: 0.62 },
  { radiusA0: 5.0, oneS: 0.995, twoS: 0.85, twoP: 0.76 },
];

const moduleSevenValidationData = [
  { stage: 'Initial Guess', predicted: 42, verified: 24 },
  { stage: 'Node Check', predicted: 61, verified: 52 },
  { stage: 'Shape Check', predicted: 74, verified: 68 },
  { stage: 'Phase Check', predicted: 86, verified: 82 },
  { stage: 'Final Interpretation', predicted: 94, verified: 93 },
];

const PLANCK_CONSTANT = 6.62607015e-34;
const REDUCED_PLANCK = 1.054571817e-34;
const BOLTZMANN_CONSTANT = 1.380649e-23;
const LIGHT_SPEED = 2.99792458e8;
const ELEMENTARY_CHARGE = 1.602176634e-19;
const ELECTRON_MASS = 9.1093837015e-31;
const PLANCK_EV_SECONDS = 4.135667696e-15;
const PHOTOELECTRIC_WORK_FUNCTION_EV = 2.3;

const planckSpectralRadiance = (wavelengthMeters, temperatureKelvin) => {
  const exponent = (PLANCK_CONSTANT * LIGHT_SPEED) / (wavelengthMeters * BOLTZMANN_CONSTANT * temperatureKelvin);
  const numerator = 2 * PLANCK_CONSTANT * LIGHT_SPEED * LIGHT_SPEED;
  const denominator = Math.pow(wavelengthMeters, 5) * (Math.exp(exponent) - 1);
  return numerator / denominator;
};

const rayleighJeansSpectralRadiance = (wavelengthMeters, temperatureKelvin) => {
  const numerator = 2 * LIGHT_SPEED * BOLTZMANN_CONSTANT * temperatureKelvin;
  const denominator = Math.pow(wavelengthMeters, 4);
  return numerator / denominator;
};

const moduleOneBlackbodyRaw = Array.from({ length: 23 }, (_, index) => 300 + index * 120).map((wavelengthNm) => {
  const wavelengthMeters = wavelengthNm * 1e-9;
  return {
    wavelengthNm,
    planck3500: planckSpectralRadiance(wavelengthMeters, 3500),
    planck5000: planckSpectralRadiance(wavelengthMeters, 5000),
    planck6500: planckSpectralRadiance(wavelengthMeters, 6500),
    rayleigh6500: rayleighJeansSpectralRadiance(wavelengthMeters, 6500),
  };
});

const moduleOneBlackbodyReference = Math.max(...moduleOneBlackbodyRaw.map((point) => point.planck6500));

const moduleOneBlackbodyData = moduleOneBlackbodyRaw.map((point) => ({
  wavelengthNm: point.wavelengthNm,
  planck3500: Number((point.planck3500 / moduleOneBlackbodyReference).toFixed(4)),
  planck5000: Number((point.planck5000 / moduleOneBlackbodyReference).toFixed(4)),
  planck6500: Number((point.planck6500 / moduleOneBlackbodyReference).toFixed(4)),
  rayleigh6500: Number((point.rayleigh6500 / moduleOneBlackbodyReference).toFixed(4)),
}));

const moduleOnePhotoelectricData = [0.35, 0.45, 0.55, 0.65, 0.8, 1.0, 1.2].map((frequencyPHz) => {
  const frequencyHz = frequencyPHz * 1e15;
  const photonEnergyEV = PLANCK_EV_SECONDS * frequencyHz;
  const kineticEnergyEV = Math.max(0, photonEnergyEV - PHOTOELECTRIC_WORK_FUNCTION_EV);
  return {
    frequencyPHz: Number(frequencyPHz.toFixed(2)),
    photonEnergyEV: Number(photonEnergyEV.toFixed(3)),
    kineticEnergyEV: Number(kineticEnergyEV.toFixed(3)),
  };
});

const moduleOneDeBroglieData = [0.5, 1, 2, 5, 10, 20, 50, 100].map((kineticEV) => {
  const kineticJoules = kineticEV * ELEMENTARY_CHARGE;
  const momentum = Math.sqrt(2 * ELECTRON_MASS * kineticJoules);
  const wavelengthPm = (PLANCK_CONSTANT / momentum) * 1e12;
  return {
    kineticEV,
    wavelengthPm: Number(wavelengthPm.toFixed(2)),
  };
});

const moduleOneUncertaintyData = [20, 30, 50, 80, 120, 180].map((deltaXPm) => {
  const deltaXMeters = deltaXPm * 1e-12;
  const minDeltaP = REDUCED_PLANCK / (2 * deltaXMeters);
  return {
    deltaXPm,
    minDeltaPScaled: Number((minDeltaP * 1e24).toFixed(3)),
  };
});

const moduleOneRadialDistributionData = Array.from({ length: 49 }, (_, index) => Number((index * 0.25).toFixed(2))).map((radiusA0) => {
  const oneS = 4 * radiusA0 * radiusA0 * Math.exp(-2 * radiusA0);
  const twoS = 0.125 * radiusA0 * radiusA0 * Math.pow(2 - radiusA0, 2) * Math.exp(-radiusA0);
  const twoP = (1 / 24) * Math.pow(radiusA0, 4) * Math.exp(-radiusA0);
  return {
    radiusA0,
    oneS: Number(oneS.toFixed(4)),
    twoS: Number(twoS.toFixed(4)),
    twoP: Number(twoP.toFixed(4)),
  };
});

const moduleTwoRadialNodeRaw = Array.from({ length: 81 }, (_, index) => Number((index * 0.2).toFixed(2))).map((radiusA0) => {
  const oneS = 4 * radiusA0 * radiusA0 * Math.exp(-2 * radiusA0);
  const twoS = radiusA0 * radiusA0 * Math.pow(2 - radiusA0, 2) * Math.exp(-radiusA0);
  const threeS = radiusA0 * radiusA0 * Math.pow(27 - 18 * radiusA0 + 2 * radiusA0 * radiusA0, 2) * Math.exp((-2 * radiusA0) / 3);
  return {
    radiusA0,
    oneS,
    twoS,
    threeS,
  };
});

const moduleTwoNodeReference = Math.max(...moduleTwoRadialNodeRaw.map((point) => Math.max(point.oneS, point.twoS, point.threeS)));

const moduleTwoRadialNodeData = moduleTwoRadialNodeRaw.map((point) => ({
  radiusA0: point.radiusA0,
  oneS: Number((point.oneS / moduleTwoNodeReference).toFixed(4)),
  twoS: Number((point.twoS / moduleTwoNodeReference).toFixed(4)),
  threeS: Number((point.threeS / moduleTwoNodeReference).toFixed(4)),
}));

const quickLinks = [
  { id: 'welcome', label: 'Welcome' },
  { id: 'chemistry', label: 'Chemistry Concepts' },
  { id: 'howto', label: 'How To Use' },
  { id: 'learn-quantum-theory', label: 'Learn' },
  { id: 'faqs', label: 'FAQs' },
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
    shortLabel: 'Quantum Foundations',
    eyebrow: 'Learn Module 1',
    title: 'Core Theory: Quantum Foundations for Orbital Visualization',
    message: 'The ideas that shape what you see inside the simulator.',
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
    label: '2. Quantum Numbers, Eigenstates, and Spherical Harmonics',
    shortLabel: 'Quantum Numbers',
    eyebrow: 'Learn Module 2',
    title: 'Quantum Numbers, Eigenstates, Angular Momentum, and Spherical Harmonics in Full Detail',
    message: 'The complete mathematical framework behind atomic orbital states.',
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
        question: 'Why do quantum numbers emerge from mathematics instead of memorization rules?',
        answer: 'Each quantum number is a separation constant or operator eigenvalue forced by boundary conditions and symmetry in the hydrogenic Schrodinger problem.',
      },
      {
        question: 'How do spherical harmonics translate into the orbital shapes we see?',
        answer: 'The angular factor Y_l^m controls lobe count, nodal planes or cones, and orientation; the simulator shows these angular signatures in 3D density and phase views.',
      },
      {
        question: 'How are nodes counted for a specific orbital such as 3p?',
        answer: 'Use N_total = n - 1, N_angular = l, and N_radial = n - l - 1. For 3p, total nodes are 2, with one angular and one radial node.',
      },
    ],
  },
  {
    id: 'learn-orbital-geometry',
    label: '3. Geometry of Orbitals in Detail',
    shortLabel: 'Orbital Geometry',
    eyebrow: 'Learn Module 3',
    title: 'Orbital Geometry: Symmetry, Orientation, and Bonding Directionality',
    message: 'A long-form geometric treatment of orbital families, degeneracy, orientation states, and directional overlap.',
    theoryCards: [
      {
        title: 'Symmetry Classes and Families',
        text: 's orbitals are isotropic and preserve full rotational symmetry, while p, d, and f states are anisotropic. Each increase in l adds angular structure and creates richer directional chemistry behavior.',
      },
      {
        title: 'Orientation Manifold from m',
        text: 'For fixed n and l, the magnetic quantum number m produces 2l+1 orientation states. In the pure Coulomb field these are degenerate, but external fields or ligand environments split and reorder them.',
      },
      {
        title: 'Real and Complex Orbital Bases',
        text: 'Chemistry uses real linear combinations to align orbital lobes with Cartesian directions, while quantum operator algebra often keeps complex Y_l^m forms. Both are mathematically equivalent basis choices.',
      },
      {
        title: 'Geometry Controls Overlap Integrals',
        text: 'Sigma overlap is strongest along internuclear axes, while pi overlap depends on side-on alignment and phase continuity. Geometry therefore predicts bond direction, strength trends, and orbital mixing pathways.',
      },
      {
        title: 'Nodal Topology',
        text: 'Angular nodes are not decorative boundaries. They are exact zero-amplitude surfaces that divide phase regions and determine where constructive overlap can or cannot occur.',
      },
      {
        title: 'Degeneracy in Field-Free Atoms',
        text: 'The hydrogenic Hamiltonian is spherically symmetric, so all m states at fixed n and l share energy. This degeneracy encodes symmetry and is lifted by perturbations such as Zeeman or crystal fields.',
      },
      {
        title: 'Chemical Consequences of Shape',
        text: 'Directional reactivity, ligand preference, and magnetic anisotropy all inherit geometry from the orbital angular factor. Orbital shape is therefore a predictive variable, not a passive visualization artifact.',
      },
      {
        title: 'Simulator Interpretation',
        text: 'Use density view to identify occupied spatial zones, phase view to detect sign boundaries, and slice mode to inspect interior nodal surfaces that are hidden in surface-level rendering.',
      },
    ],
    deepDiveSections: [
      {
        title: 'Separation Framework',
        text: 'Orbital geometry is encoded in the angular factor after separating the Schrodinger equation in spherical coordinates. This guarantees that l and m are geometric quantum numbers, not empirical labels.',
        equation: '\\psi_{n,l,m}(r,\\theta,\\phi)=R_{n,l}(r)Y_{l,m}(\\theta,\\phi)',
      },
      {
        title: 'Associated Legendre Structure',
        text: 'The theta dependence uses associated Legendre functions. Their zeros generate nodal planes or nodal cones, producing the recognizable lobe architectures of p, d, and f states.',
        equation: 'Y_{l,m}(\\theta,\\phi)=N_{l,m}P_l^m(\\cos\\theta)e^{im\\phi}',
      },
      {
        title: 'Orientation Count',
        text: 'Each l value has a finite orientation set. Higher l gives more orientation channels and therefore richer anisotropy in external fields and bonding environments.',
        equation: 'N_{\\text{orientations}}=2l+1',
      },
      {
        title: 'Real Orbital Construction',
        text: 'Cartesian-labeled orbitals are real combinations of complex m states. This change of basis preserves physics but makes directional interpretation easier for chemistry problems.',
        equation: 'p_x\\propto\\frac{1}{\\sqrt{2}}\\left(Y_1^{-1}-Y_1^{1}\\right)',
      },
      {
        title: 'Angular Node Budget',
        text: 'Angular node count equals l, so shape complexity climbs in a controlled and quantized sequence from s to f families.',
        equation: 'N_{\\text{angular}}=l',
      },
      {
        title: 'Overlap Integral Logic',
        text: 'Constructive overlap requires same-sign amplitude in shared space. Opposite signs cancel overlap and reduce bonding gain.',
        equation: 'S=\\int \\psi_A^*(\\mathbf{r})\\psi_B(\\mathbf{r})\\,d\\tau',
      },
      {
        title: 'Crystal Field Distortion',
        text: 'When spherical symmetry breaks in ligand fields, d-state degeneracy splits into energy subsets. Geometry directly controls splitting magnitude and occupancy order.',
        equation: '\\Delta_{CF}=E_{\\text{upper}}-E_{\\text{lower}}',
      },
      {
        title: 'Visualization Strategy',
        text: 'Do not identify family from one camera angle. Rotate to verify lobe multiplicity, then use phase coloring to track sign domains and nodal boundaries.',
      },
      {
        title: 'Geometry to Spectroscopy',
        text: 'Angular selection rules and dipole matrix elements depend on geometry. Spectral intensity patterns are therefore geometric fingerprints of orbital states.',
        equation: '\\Delta l=\\pm1',
      },
      {
        title: 'Geometry to Reactivity',
        text: 'In molecular systems, frontier orbital geometry predicts where electron donation and back-donation are strongest. Directionality governs reaction pathways and activation barriers.',
      },
    ],
    supplementarySections: [
      {
        title: 'Orthogonality of Angular States',
        text: 'Different angular basis states are orthogonal over solid angle, which guarantees clean state separation and prevents double counting of orientation information.',
        equation: '\\int Y_{l,m}^*(\\theta,\\phi)Y_{l^{\\prime},m^{\\prime}}(\\theta,\\phi)\\,d\\Omega=\\delta_{l,l^{\\prime}}\\delta_{m,m^{\\prime}}',
      },
      {
        title: 'Parity Signature of Orbital Families',
        text: 'Each l family has a definite parity. This symmetry controls selection-rule behavior and explains why inversion properties matter in spectroscopy and bonding models.',
        equation: 'Y_{l,m}(\\pi-\\theta,\\phi+\\pi)=(-1)^lY_{l,m}(\\theta,\\phi)',
      },
      {
        title: 'Ladder-Operator Connectivity',
        text: 'm states inside one l manifold are connected algebraically through angular momentum ladder operators, revealing orientation structure as an operator-generated sequence.',
        equation: '\\hat{L}_{\\pm}Y_l^m=\\hbar\\sqrt{l(l+1)-m(m\\pm1)}\\,Y_l^{m\\pm1}',
      },
    ],
    equations: [
      {
        title: 'Wavefunction Separation',
        math: '\\psi_{n,l,m}(r,\\theta,\\phi)=R_{n,l}(r)Y_{l,m}(\\theta,\\phi)',
        description: 'Orbital geometry and radial extent are factored into independent mathematical parts.',
      },
      {
        title: 'Spherical Harmonic Core Form',
        math: 'Y_{l,m}(\\theta,\\phi)=N_{l,m}P_l^m(\\cos\\theta)e^{im\\phi}',
        description: 'Associated Legendre functions and azimuthal phase terms set lobe topology and orientation behavior.',
      },
      {
        title: 'Orientation Multiplicity',
        math: 'g_l = 2l + 1',
        description: 'Counts how many m-resolved orientations exist for a chosen l family.',
      },
      {
        title: 'Orbital Overlap Integral',
        math: 'S=\\int \\psi_A^*(\\mathbf{r})\\psi_B(\\mathbf{r})\\,d\\tau',
        description: 'Provides a formal directional measure of bonding compatibility between two orbitals.',
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
    secondaryCharts: [
      {
        title: 'Orientation Multiplicity and Anisotropy',
        subtitle: 'Growth of m-state count and directional anisotropy from s to f families.',
        data: moduleThreeOrientationData,
        lines: [
          { dataKey: 'orientations', name: 'Orientation Count', color: '#63d6ff' },
          { dataKey: 'anisotropy', name: 'Anisotropy Index', color: '#ff9f6e' },
        ],
      },
    ],
    table: {
      columns: ['Family', 'l Value', 'Typical Node Character', 'Chemistry Implication'],
      rows: [
        ['s', '0', 'No angular node', 'Isotropic overlap and shielding'],
        ['p', '1', 'Single nodal plane', 'Directional sigma and pi bonding'],
        ['d', '2', 'Two angular nodes', 'Transition-metal field splitting'],
        ['f', '3', 'Three angular nodes', 'Strong anisotropy and complex magnetism'],
        ['g', '4', 'Four angular nodes', 'Advanced high-l modeling and shape control'],
      ],
    },
    imageSlots: [
      {
        title: 'Spherical Harmonics Atlas',
        src: 'https://upload.wikimedia.org/wikipedia/commons/6/62/Spherical_Harmonics.png',
        alt: 'Table of spherical harmonics showing angular patterns',
        credit: 'Source: Wikimedia Commons',
        description: 'Angular basis functions that generate orbital lobe topology and node geometry.',
      },
      {
        title: 'Hydrogenic Orbital Family Grid',
        src: 'https://upload.wikimedia.org/wikipedia/commons/b/b0/Atomic_orbitals_n1234_m-eigenstates.png',
        alt: 'Hydrogenic s p d f orbital family comparison',
        credit: 'Source: Wikimedia Commons',
        description: 'Direct visual comparison of n, l, and m-resolved orbital states.',
      },
      {
        title: 'Real Spherical Harmonics 2D Table',
        src: 'https://upload.wikimedia.org/wikipedia/commons/3/36/Real_Spherical_Harmonics_Figure_Table_Complex_2D.png',
        alt: 'Real spherical harmonics figure table',
        credit: 'Source: Wikimedia Commons',
        description: 'Real-basis harmonic forms used for axis-aligned orbital interpretation in chemistry.',
      },
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
    title: 'Nodes and Antinodes: Interference Topology in Atomic Orbitals',
    message: 'A full explanation of how destructive and constructive interference create node surfaces, phase domains, and bonding consequences.',
    theoryCards: [
      {
        title: 'Node as an Exact Zero Condition',
        text: 'A node is the set of points where wavefunction amplitude is identically zero. It is a strict eigenfunction property and not a low-density approximation.',
      },
      {
        title: 'Antinode as Maximum Amplitude Zone',
        text: 'Antinodes are regions where amplitude magnitude is large, so probability density is high after squaring. Orbital lobes are antinode-dominant volumes.',
      },
      {
        title: 'Interference and Sign Structure',
        text: 'Opposite-sign amplitude overlap drives destructive cancellation and node creation. Same-sign overlap reinforces density and expands antinode zones.',
      },
      {
        title: 'Bonding Interpretation',
        text: 'Nodal surfaces suppress overlap pathways and often correlate with antibonding character, while antinode alignment supports constructive overlap and bond stabilization.',
      },
      {
        title: 'Phase Color Is Not Charge',
        text: 'Red and blue lobes represent opposite signs of wavefunction phase, not positive and negative electric charge. This sign is essential for overlap calculations.',
      },
      {
        title: 'Node Geometry in 3D',
        text: 'Nodes appear as planes, cones, or shells depending on whether the zero comes from angular or radial factors. Their dimensionality is a geometric fingerprint of the state.',
      },
      {
        title: 'Experimental Consequences',
        text: 'Interference topology influences transition strengths, orbital mixing patterns, and spectroscopic signal intensities in real atomic and molecular systems.',
      },
      {
        title: 'Simulator Reading Workflow',
        text: 'Find sign domains in phase mode, then cut with slice mode to expose hidden nodes. Only after that interpret bonding or antibonding tendencies.',
      },
    ],
    deepDiveSections: [
      {
        title: 'Standing-Wave Origin',
        text: 'Nodes are expected whenever boundary conditions force standing-wave solutions. Quantum nodes are the 3D analog of fixed points in string standing waves.',
        equation: '\\psi(x)=A\\sin(kx)',
      },
      {
        title: 'Node Condition',
        text: 'Node positions occur where the total amplitude vanishes. The condition is exact and can be solved analytically for hydrogenic states.',
        equation: '\\psi(r,\\theta,\\phi)=0',
      },
      {
        title: 'Density Interpretation',
        text: 'Probability density is phase-insensitive after squaring, so node surfaces remain zero while opposite-sign antinodes can have the same density magnitude.',
        equation: '\\rho=|\\psi|^2',
      },
      {
        title: 'Constructive and Destructive Branches',
        text: 'Superposed states produce branch-dependent outcomes: addition increases amplitude where phases align, subtraction can cancel amplitude where they oppose.',
        equation: '\\psi_{\\text{tot}}=\\psi_A\\pm\\psi_B',
      },
      {
        title: 'Nodal Surfaces and Chemistry',
        text: 'When nodal surfaces lie between atomic centers, overlap integrals fall and antibonding character rises. This directly affects molecular stability trends.',
      },
      {
        title: 'Angular-Node Planes',
        text: 'For p states, one angular node appears as a plane. For d and f states, multiple angular node surfaces partition space into alternating sign domains.',
        equation: 'N_{\\text{angular}}=l',
      },
      {
        title: 'Radial-Node Shells',
        text: 'Radial zeros produce spherical node shells that separate inner and outer antinode zones. These shells are critical in ns and np radial interpretation.',
        equation: 'N_{\\text{radial}}=n-l-1',
      },
      {
        title: 'Orbital Mixing Constraint',
        text: 'Only orbitals with compatible symmetry and phase arrangement mix efficiently. Node mismatch suppresses mixing channels and affects reactivity pathways.',
      },
      {
        title: 'Visual Parsing Rule',
        text: 'Never infer nodes from color alone. Confirm with slice planes and rotational checks so hidden boundaries are not mistaken for low-density tails.',
      },
      {
        title: 'From Geometry to Spectra',
        text: 'Selection-rule allowed transitions depend on overlap of initial and final states. Node architecture controls matrix-element magnitude and transition intensity.',
      },
    ],
    supplementarySections: [
      {
        title: 'Phase Inversion Across Nodes',
        text: 'Crossing a true node flips wavefunction sign. The associated phase jump is the reason opposite-color lobes are separated by zero-amplitude boundaries.',
        equation: '\\Delta\\phi=\\pi',
      },
      {
        title: 'Interference Intensity Rule',
        text: 'Observed constructive or destructive behavior depends on squared total amplitude, not on amplitudes viewed in isolation.',
        equation: 'I\\propto|\\psi_A+\\psi_B|^2',
      },
      {
        title: 'Dipole-Moment Cancellation by Symmetry',
        text: 'Nodal symmetry can suppress transition strength when positive and negative contributions cancel in the dipole integral.',
        equation: '\\mu_{if}=\\int \\psi_f^*(\\mathbf{r})\\,\\mathbf{r}\\,\\psi_i(\\mathbf{r})\\,d\\tau',
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
      {
        title: 'Superposition Interference',
        math: '\\psi_{\\text{total}}=\\psi_A+\\psi_B',
        description: 'Relative sign of component amplitudes determines whether overlap reinforces or cancels.',
      },
      {
        title: 'Bonding-Antibonding Pair',
        math: '\\psi_{\\pm}=\\frac{1}{\\sqrt{2}}\\left(\\psi_A\\pm\\psi_B\\right)',
        description: 'Plus gives bonding tendency; minus introduces central node and antibonding character.',
      },
    ],
    chartTitle: 'Interference Balance',
    chartSubtitle: 'A conceptual chart showing where constructive and destructive regimes dominate.',
    chartData: phaseInsightData,
    chartLines: [
      { dataKey: 'positive', name: 'Constructive Regime', color: '#ff6b6b' },
      { dataKey: 'negative', name: 'Destructive Regime', color: '#6beaff' },
    ],
    secondaryCharts: [
      {
        title: 'Amplitude-to-Density Mapping',
        subtitle: 'How signed amplitude converts into non-negative probability density across a node crossing.',
        data: moduleFourWaveInterferenceData,
        lines: [
          { dataKey: 'amplitude', name: 'Amplitude', color: '#8ed7ff' },
          { dataKey: 'density', name: 'Density', color: '#ffd265' },
        ],
      },
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
      {
        title: 'Hydrogen Density Plot Grid',
        src: 'https://upload.wikimedia.org/wikipedia/commons/e/e7/Hydrogen_Density_Plots.png',
        alt: 'Hydrogen orbital density plot panel',
        credit: 'Source: Wikimedia Commons',
        description: 'Node and antinode regions across multiple hydrogenic states.',
      },
      {
        title: 'Molecular Orbital Diagram (H2)',
        src: 'https://upload.wikimedia.org/wikipedia/commons/a/a8/Dihydrogen-MO-Diagram.svg',
        alt: 'Dihydrogen molecular orbital diagram',
        credit: 'Source: Wikimedia Commons',
        description: 'Bonding and antibonding construction from two atomic basis orbitals.',
      },
      {
        title: 'He2 Orbital Energy Diagram',
        src: 'https://upload.wikimedia.org/wikipedia/commons/b/bc/Molecular_orbital_diagram_of_He2.png',
        alt: 'Molecular orbital diagram of helium dimer',
        credit: 'Source: Wikimedia Commons',
        description: 'Illustrates node-driven antibonding occupancy effects in a weakly bound system.',
      },
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
    shortLabel: 'Radial vs Angular',
    eyebrow: 'Learn Module 5',
    title: 'Radial and Angular Nodes: Full Breakdown',
    message: 'A complete node-partition framework: where radial zeros come from, where angular zeros come from, and how to classify any orbital state quickly.',
    theoryCards: [
      {
        title: 'Radial Nodes',
        text: 'Radial nodes are spherical shell surfaces where the radial function changes sign. They create concentric probability regions and are counted by n-l-1.',
      },
      {
        title: 'Angular Nodes',
        text: 'Angular nodes come from spherical harmonics and equal l in count. They appear as planes or cones and split space into sign-opposed angular sectors.',
      },
      {
        title: 'Combined Node Budget',
        text: 'Total nodes are always n-1 in hydrogenic states. Partitioning between radial and angular contributions predicts whether a state looks shell-dominated or direction-dominated.',
      },
      {
        title: 'Visualization Strategy',
        text: 'Identify shells first in radial plots and slice cuts, then rotate in 3D to classify angular planes or cones. Both checks are required for correct state diagnosis.',
      },
      {
        title: 'Radial Polynomial Roots',
        text: 'Radial nodes are roots of associated Laguerre polynomial factors inside R_{n,l}(r). The polynomial order controls how many zero crossings occur.',
      },
      {
        title: 'Node Topology by Family',
        text: 's states have no angular nodes, p states have one plane, d states have two angular surfaces, and f states have three. Radial shells may still coexist.',
      },
      {
        title: 'Why 3d Has Zero Radial Nodes',
        text: 'For 3d, n-l-1 = 0. So all node budget is angular, which yields strong directional structure without concentric radial sign shells.',
      },
      {
        title: 'Impact on Chemistry and Spectra',
        text: 'Node placement changes overlap and transition amplitudes. Radial node mismatch reduces radial overlap, while angular mismatch suppresses directional coupling.',
      },
    ],
    deepDiveSections: [
      {
        title: 'Node Accounting Master Formula',
        text: 'Every hydrogenic orbital obeys a strict node partition. Start with the total, then split into radial and angular contributions.',
        equation: 'N_{\\text{total}}=N_r+N_a=n-1',
      },
      {
        title: 'Radial Node Rule',
        text: 'Radial node count falls as l increases for fixed n, because more of the node budget is allocated to angular structure.',
        equation: 'N_r=n-l-1',
      },
      {
        title: 'Angular Node Rule',
        text: 'Angular nodes grow directly with l. This growth drives lobe complexity and orientation dependence.',
        equation: 'N_a=l',
      },
      {
        title: 'Radial Probability Interpretation',
        text: 'Radial plots expose shell-localized probability and reveal radial node crossings as exact zeros between peaks.',
        equation: 'P(r)=4\\pi r^2|R_{n,l}(r)|^2',
      },
      {
        title: 'Case Study 2s vs 2p',
        text: '2s has one radial node and no angular nodes, while 2p has no radial nodes and one angular node. Same n, different partition, very different geometry.',
      },
      {
        title: 'Case Study 3p',
        text: '3p has one radial and one angular node, so it combines shell layering with directional splitting. This makes it a canonical mixed-topology example.',
      },
      {
        title: 'Case Study 4f',
        text: '4f allocates most complexity to angular structure, producing high anisotropy with multiple angular zero surfaces and rich lobe partitioning.',
      },
      {
        title: 'Node Validation Sequence',
        text: 'Compute node counts from quantum numbers, predict geometry, then verify with phase and slicing tools. This keeps interpretation reproducible.',
      },
      {
        title: 'Common Interpretation Error',
        text: 'Low-density tails are often mistaken for nodes. A true node requires strict zero amplitude, not simply low point intensity in sampled rendering.',
      },
      {
        title: 'From Nodes to Reactivity',
        text: 'Node placement modifies overlap pathways and therefore bond strength trends, ligand interactions, and orbital energy ordering in molecular fields.',
      },
    ],
    supplementarySections: [
      {
        title: 'Laguerre-Polynomial Order and Radial Zeros',
        text: 'Radial zeros are encoded in associated Laguerre structure. Increasing polynomial order increases radial sign changes and therefore shell-level node count.',
        equation: 'p=n-l-1',
      },
      {
        title: 'Radial Length-Scale Estimate',
        text: 'For quick intuition, hydrogenic orbital extent scales approximately with n^2/Z. This helps forecast contraction trends before plotting.',
        equation: 'r_{\\text{scale}}\\sim\\frac{n^2a_0}{Z}',
      },
      {
        title: 'Fast Node Classification Protocol',
        text: 'Use this order: compute N_total, split into N_r and N_a, then confirm radial shells with P(r) and angular boundaries with phase-slice inspection.',
        equation: 'N_{\\text{total}}=n-1',
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
      {
        title: 'Total Node Count',
        math: 'N_{\\text{total}} = n-1',
        description: 'The full node budget that must equal radial plus angular contributions.',
      },
      {
        title: 'Radial Distribution',
        math: 'P(r)=4\\pi r^2|R_{n,l}(r)|^2',
        description: 'Best graph for locating radial nodes as shell-level zero crossings.',
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
    secondaryCharts: [
      {
        title: 'Maximum Node Growth with n',
        subtitle: 'Upper bounds for radial and angular contributions as shell index increases.',
        data: moduleFiveNodeGrowthData,
        lines: [
          { dataKey: 'total', name: 'Total Node Cap', color: '#63d6ff' },
          { dataKey: 'radial', name: 'Radial Cap', color: '#ffd265' },
          { dataKey: 'angularCap', name: 'Angular Cap', color: '#ff8fb8' },
        ],
      },
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
      {
        title: 'Hydrogen 1s Radial Function',
        src: 'https://upload.wikimedia.org/wikipedia/commons/1/1f/Hydrogen_1s_Radial.svg',
        alt: 'Hydrogen 1s radial function graphic',
        credit: 'Source: Wikimedia Commons',
        description: 'Reference radial behavior with no radial node crossing.',
      },
      {
        title: 'Hydrogen 2s Radial Function',
        src: 'https://upload.wikimedia.org/wikipedia/commons/7/73/Hydrogen_2s_Radial.svg',
        alt: 'Hydrogen 2s radial function graphic',
        credit: 'Source: Wikimedia Commons',
        description: 'Shows one radial node through a sign-changing radial profile.',
      },
      {
        title: 'Hydrogen 2p Radial Function',
        src: 'https://upload.wikimedia.org/wikipedia/commons/c/c9/Hydrogen_2p_Radial.svg',
        alt: 'Hydrogen 2p radial function graphic',
        credit: 'Source: Wikimedia Commons',
        description: 'Contrasts with 2s by shifting node budget from radial to angular structure.',
      },
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
    shortLabel: 'Probability Maps',
    eyebrow: 'Learn Module 6',
    title: 'Probability Distribution Functions: All Core Types',
    message: 'A complete probability toolkit: local density, radial and angular distributions, cumulative probability, and expectation-value interpretation.',
    theoryCards: [
      {
        title: 'Point Probability Density',
        text: 'Local measurement likelihood is set by |psi|^2 at each point in space. In scatter rendering, denser regions correspond to higher local detection probability.',
      },
      {
        title: 'Radial Distribution Function',
        text: 'Radial shell probability includes geometric Jacobian weighting, so maxima can occur away from the origin even if local density is highest near small r.',
      },
      {
        title: 'Angular Distribution',
        text: 'Angular weighting from |Y_l^m|^2 sets direction preference, lobe placement, and angular node geometry.',
      },
      {
        title: 'Cumulative Probability',
        text: 'Cumulative probability integrates shell contributions up to radius r, giving a direct way to compare contraction and spread across states or charge values Z.',
      },
      {
        title: 'Normalization as Physical Consistency',
        text: 'Any valid stationary state must integrate to unit total probability. Without normalization, comparisons across states lose physical meaning.',
      },
      {
        title: 'Expectation Values',
        text: 'Expectation values summarize distributions with weighted averages, such as mean radius and radial spread. They are not guaranteed to match the most probable radius.',
      },
      {
        title: 'Most Probable vs Mean Radius',
        text: 'The modal radius maximizes P(r), while mean radius uses integral weighting. Distinguishing these avoids common interpretation errors in radial plots.',
      },
      {
        title: 'Probability in Visualization Workflows',
        text: '3D point clouds, radial curves, and cumulative curves are complementary views of one normalized probability model. Read them together for robust interpretation.',
      },
    ],
    deepDiveSections: [
      {
        title: 'Local Density Definition',
        text: 'Pointwise probability density is the squared magnitude of the full wavefunction and is the core quantity behind volumetric orbital rendering.',
        equation: '\\rho(r,\\theta,\\phi)=|\\psi_{n,l,m}(r,\\theta,\\phi)|^2',
      },
      {
        title: 'Radial Jacobian Effect',
        text: 'Spherical shells grow with r^2, so shell-level probability can rise away from the nucleus even while local density decays.',
        equation: 'P(r)=4\\pi r^2|R_{n,l}(r)|^2',
      },
      {
        title: 'Angular Weighting',
        text: 'Directionality follows angular probability and encodes anisotropy. This is the origin of lobe-dominant directions in p, d, and f states.',
        equation: 'W(\\theta,\\phi)=|Y_l^m(\\theta,\\phi)|^2',
      },
      {
        title: 'Cumulative Distribution',
        text: 'Integrating radial probability from 0 to r gives containment probability. It is ideal for comparing compact vs diffuse states.',
        equation: 'C(r)=\\int_0^r P(r^{\\prime})\\,dr^{\\prime}',
      },
      {
        title: 'Normalization Constraint',
        text: 'All comparisons assume unit-normalized states. This constraint keeps probability interpretation consistent across modules.',
        equation: '\\int |\\psi|^2 d\\tau = 1',
      },
      {
        title: 'Expectation Radius',
        text: 'Mean radius weights each shell by r and therefore emphasizes tail contributions more than a modal estimate does.',
        equation: '\\langle r \\rangle = \\int_0^\\infty rP(r)\\,dr',
      },
      {
        title: 'Variance and Spread',
        text: 'Second-moment structure quantifies orbital spread and gives uncertainty context for radial extent differences.',
        equation: '\\sigma_r^2=\\langle r^2\\rangle-\\langle r\\rangle^2',
      },
      {
        title: 'Z-Scaling Intuition',
        text: 'As Z increases for fixed n,l,m, distributions contract inward and cumulative probability reaches high values at smaller radii.',
      },
      {
        title: 'Sampling Caution in 3D Clouds',
        text: 'Finite Monte Carlo or grid sampling can blur low-density tails. Use radial and cumulative plots to validate point-cloud intuition.',
      },
      {
        title: 'Interpretation Workflow',
        text: 'Use local density for shape, radial distribution for shell peaks and nodes, and cumulative curves for containment thresholds.',
      },
    ],
    supplementarySections: [
      {
        title: 'Most Probable Radius for 1s',
        text: 'The highest radial probability location for a hydrogenic 1s state is not at the nucleus; it occurs at one Bohr-radius-scaled distance.',
        equation: 'r_{\\mathrm{mp}}^{1s}=\\frac{a_0}{Z}',
      },
      {
        title: 'Mean Radius for 1s',
        text: 'The expectation radius exceeds the most-probable radius because distribution tails contribute to the integral-weighted average.',
        equation: '\\langle r\\rangle_{1s}=\\frac{3a_0}{2Z}',
      },
      {
        title: 'Peak-Condition Logic',
        text: 'Maximum shell probability locations are found by differentiating radial probability and setting the derivative to zero.',
        equation: '\\frac{dP(r)}{dr}=0',
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
      {
        title: 'Angular Probability Weighting',
        math: 'W(\\theta,\\phi)=|Y_l^m(\\theta,\\phi)|^2',
        description: 'Determines directional preference and lobe weighting in angular space.',
      },
      {
        title: 'Normalization',
        math: '\\int |\\psi|^2 d\\tau = 1',
        description: 'Ensures all plotted probabilities are physically meaningful and comparable.',
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
    secondaryCharts: [
      {
        title: 'Cumulative Probability by Radius',
        subtitle: 'Illustrative cumulative containment trends for 1s, 2s, and 2p states.',
        data: moduleSixCumulativeShellData,
        lines: [
          { dataKey: 'oneS', name: '1s', color: '#63d6ff' },
          { dataKey: 'twoS', name: '2s', color: '#ffd265' },
          { dataKey: 'twoP', name: '2p', color: '#ff8fb8' },
        ],
      },
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
      {
        title: 'Radial Probability Reference Plot',
        src: 'https://upload.wikimedia.org/wikipedia/commons/d/d7/Radial_Wave_function_Probability_for_Hydrogen_Atom.png',
        alt: 'Hydrogen radial wave function probability graph',
        credit: 'Source: Wikimedia Commons',
        description: 'Canonical radial probability reference used to identify shell peaks and node zeros.',
      },
      {
        title: 'Probability Density of Hydrogen',
        src: 'https://upload.wikimedia.org/wikipedia/commons/e/e2/Probability_density_of_hydrogen.svg',
        alt: 'Probability density distribution of hydrogen orbitals',
        credit: 'Source: Wikimedia Commons',
        description: 'Spatial density mapping that complements radial-only views.',
      },
      {
        title: 'Electron Probability Density Image',
        src: 'https://upload.wikimedia.org/wikipedia/commons/9/98/Electron_in_hydrogen%2C_density_of_probability.jpg',
        alt: 'Electron in hydrogen probability density illustration',
        credit: 'Source: Wikimedia Commons',
        description: 'Visual intuition for probability concentration and diffuse tails.',
      },
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
    shortLabel: 'Solved Workflows',
    eyebrow: 'Learn Module 7',
    title: 'Solved Examples: From Inputs to Interpretation',
    message: 'A step-by-step problem-solving module that converts quantum numbers into predictions, checks them against visualization, and explains discrepancies.',
    theoryCards: [
      {
        title: 'Example 1: 2p (n=2, l=1, m=0, Z=1)',
        text: 'Predict one angular node and zero radial nodes, then verify dumbbell geometry and nodal plane orientation using rotate, phase, and slice controls.',
      },
      {
        title: 'Example 2: 3s (n=3, l=0, m=0, Z=1)',
        text: 'Predict spherical symmetry with two radial nodes. Confirm with radial plot zero crossings and concentric shell transitions in slice mode.',
      },
      {
        title: 'Example 3: 4d (n=4, l=2, m=1, Z=2)',
        text: 'Predict two angular nodes and one radial node, then compare Z=2 contraction against Z=1 to validate charge-scaling intuition.',
      },
      {
        title: 'Interpretation Checklist',
        text: 'Use a fixed sequence: compute node budget, predict geometry and orientation, verify with phase and radial plots, then finalize interpretation.',
      },
      {
        title: 'Error-Driven Refinement',
        text: 'When prediction and render disagree, isolate whether the mismatch came from node counting, orientation assumptions, or state-input mistakes.',
      },
      {
        title: 'Parameter Sensitivity Checks',
        text: 'Vary one parameter at a time to separate effects cleanly: n for size and total nodes, l for family and angular structure, m for orientation, and Z for contraction.',
      },
      {
        title: 'Cross-Validation Habit',
        text: 'No solved example is complete until equation-based prediction and visualization-based evidence agree under multiple camera and slice settings.',
      },
      {
        title: 'Communicating the Solution',
        text: 'A strong solution report states inputs, computed node counts, expected geometry, observed evidence, and final physical interpretation in a traceable format.',
      },
    ],
    deepDiveSections: [
      {
        title: 'Step 1: Parse Inputs',
        text: 'Write n, l, m, and Z explicitly before any interpretation. Early input ambiguity is the most common source of incorrect conclusions.',
      },
      {
        title: 'Step 2: Compute Node Budget',
        text: 'Compute total, radial, and angular nodes and verify partition consistency before predicting geometry.',
        equation: 'N_{\\text{total}}=n-1=(n-l-1)+l',
      },
      {
        title: 'Step 3: Predict Family and Orientation',
        text: 'Use l for family and m for orientation state. Record this prediction before looking at the rendered result to avoid visual bias.',
      },
      {
        title: 'Step 4: Predict Radius Trend',
        text: 'Estimate relative cloud size using hydrogenic scaling so you have an a priori expectation for contraction or expansion.',
        equation: '\\langle r \\rangle \\propto \\frac{n^2}{Z}',
      },
      {
        title: 'Step 5: Verify in Density View',
        text: 'Confirm gross shape class and radial extent first. Density mode is best for structural silhouette checks.',
      },
      {
        title: 'Step 6: Verify in Phase View',
        text: 'Confirm sign regions and nodal boundaries. Phase mode catches hidden interpretation errors that density alone can miss.',
      },
      {
        title: 'Step 7: Slice for Internal Nodes',
        text: 'Use one or two slice planes to reveal internal shells and angular planes that may be occluded in full-volume rendering.',
      },
      {
        title: 'Step 8: Compare Against Formulae',
        text: 'If mismatch remains, return to equations and recompute node counts and selection constraints. Avoid patching conclusions ad hoc.',
      },
      {
        title: 'Step 9: Document Confidence',
        text: 'State confidence level and why. Confidence should rise only when independent checks align across equations, geometry, and plot evidence.',
      },
      {
        title: 'Step 10: Generalize Pattern',
        text: 'Extract reusable rules from the solved case so future states can be solved faster with fewer errors.',
      },
    ],
    supplementarySections: [
      {
        title: 'Spectroscopic Back-Check',
        text: 'After solving geometry and nodes, validate energetic plausibility by comparing predicted transition energy with observed wavelength data.',
        equation: '\\Delta E=\\frac{hc}{\\lambda}',
      },
      {
        title: 'Quantifying Prediction Error',
        text: 'A simple relative-error metric helps compare model predictions against measured or reference values in a reproducible way.',
        equation: '\\varepsilon_{\\mathrm{rel}}=\\frac{|x_{\\mathrm{pred}}-x_{\\mathrm{ref}}|}{|x_{\\mathrm{ref}}|}',
      },
      {
        title: 'Capacity Consistency Check',
        text: 'For shell-level reasoning, keep state counting consistent with degeneracy rules. This avoids occupancy mistakes in multi-step solved examples.',
        equation: 'g_n=2n^2\\;\\text{(including spin)}',
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
      {
        title: 'Angular Momentum Magnitude',
        math: '|\\mathbf{L}|=\\hbar\\sqrt{l(l+1)}',
        description: 'Adds a quantitative check on orbital angular character in solved states.',
      },
      {
        title: 'Orientation Multiplicity',
        math: '2l+1',
        description: 'Counts how many m orientations are available at fixed n and l.',
      },
    ],
    chartTitle: 'Solved Workflow Confidence Curve',
    chartSubtitle: 'Conceptual confidence gain while progressing through a worked problem.',
    chartData: workflowChartData,
    chartLines: [
      { dataKey: 'confidence', name: 'Interpretation Confidence', color: '#30c6ff' },
    ],
    secondaryCharts: [
      {
        title: 'Predicted vs Verified Accuracy',
        subtitle: 'How a disciplined solve sequence improves agreement between forecast and observed visualization outcomes.',
        data: moduleSevenValidationData,
        lines: [
          { dataKey: 'predicted', name: 'Prediction Confidence', color: '#8ed7ff' },
          { dataKey: 'verified', name: 'Verified Confidence', color: '#66f0c7' },
        ],
      },
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
      {
        title: 'Hydrogen Orbitals Overview',
        src: 'https://upload.wikimedia.org/wikipedia/commons/0/08/Hydrogen_Orbitals.png',
        alt: 'Hydrogen orbital shapes overview',
        credit: 'Source: Wikimedia Commons',
        description: 'Reference panel for solved-family identification during worked examples.',
      },
      {
        title: 'Hydrogen 3p Orbital Example',
        src: 'https://upload.wikimedia.org/wikipedia/commons/8/8f/Hydrogen_orbital_3p.png',
        alt: 'Hydrogen 3p orbital image',
        credit: 'Source: Wikimedia Commons',
        description: 'Useful benchmark for mixed radial/angular node interpretation.',
      },
      {
        title: 'Hydrogen Density Plot Atlas',
        src: 'https://upload.wikimedia.org/wikipedia/commons/e/e7/Hydrogen_Density_Plots.png',
        alt: 'Hydrogen density plot atlas for multiple orbitals',
        credit: 'Source: Wikimedia Commons',
        description: 'Multi-case visual sheet for rapid validation in worked solutions.',
      },
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
  ...learnTopics.map((topic, index) => ({
    id: topic.id,
    moduleNumber: index + 1,
    label: topic.label,
    shortLabel: topic.shortLabel,
    numberedShortLabel: `${index + 1}. ${topic.shortLabel}`,
  })),
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

const ModuleOneSectionHeading = ({ title }) => (
  <div className="info-section-heading module1-heading">
    <h2>{title}</h2>
    <span className="module1-heading-rule" aria-hidden="true"></span>
  </div>
);

const moduleOneSectionExpansionNote = 'Read each subsection through a five-step validation loop: define the physical claim, identify the equation that governs it, check the dimensional meaning of every symbol, compare the prediction with at least one plotted trend, and finally verify the visual pattern in the 3D view. This discipline prevents common errors such as interpreting amplitude as probability density, misreading phase sign as electric charge, or treating sparse tails as real nodes. When all five checks agree, your interpretation is typically robust enough for classroom teaching, report writing, and exam-level reasoning. If one check fails, pause and reconcile the mismatch before proceeding to the next concept. Over time, this method builds intuition without sacrificing mathematical accuracy.';

const moduleTwoSectionExpansionNote = 'Use each subsection as an operator-to-observable translation drill. First state which quantity is quantized, then identify the eigenvalue equation that constrains it, then map that constraint to orbital size, orientation, node count, or transition behavior. Next, test whether the simulator output obeys that prediction under camera rotation, phase toggling, and slice inspection. Finally, connect the same conclusion to a spectroscopic or chemical consequence so the concept is not isolated to one representation. This procedure converts quantum numbers from memorized symbols into a reproducible analysis workflow that remains accurate across solved problems and real data interpretation.';

const ModuleOneCoreTheoryContent = () => (
  <>
    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="Core Quantum Theory" />
      <p className="module1-detail-extension">{moduleOneSectionExpansionNote}</p>
      <p className="module1-intro-note">A 10-part foundation for understanding every visual in this simulator.</p>
      <p>
        Module 1 explains why quantum mechanics was needed, how core equations were built,
        and how those equations map to orbitals, nodes, and probability clouds.
      </p>
      <p>
        The flow below starts with the failure of classical physics and ends with direct interpretation
        of what you control in the simulator.
      </p>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="1) Why Classical Physics Failed" />
      <p className="module1-detail-extension">{moduleOneSectionExpansionNote}</p>
      <div className="module1-split">
        <div className="module1-text-flow">
          <p>
            Classical theory predicted that blackbody radiation should keep rising at shorter wavelengths.
            That prediction diverges in the ultraviolet range and is known as the ultraviolet catastrophe.
          </p>
          <p>The two equations below summarize the mismatch between classical and quantum descriptions:</p>
          <div className="module1-equation-stack">
            <ExpandableFormula className="compact module1-formula-card" math={'B_\\lambda(T)=\\frac{2ck_B T}{\\lambda^4}'} />
            <ExpandableFormula className="compact module1-formula-card" math={'B_\\lambda(\\lambda,T)=\\frac{2hc^2}{\\lambda^5}\\frac{1}{e^{hc/(\\lambda k_B T)}-1}'} />
          </div>
          <p>
            The first is Rayleigh-Jeans (classical). The second is Planck&apos;s law (quantum), which matches experiment.
          </p>
        </div>
        <figure className="module1-figure is-wide">
          <img
            src="https://upload.wikimedia.org/wikipedia/commons/1/19/Black_body.svg"
            alt="Blackbody radiation comparison showing ultraviolet catastrophe"
            loading="lazy"
          />
          <figcaption>Ultraviolet catastrophe reference figure. Source: Wikimedia Commons.</figcaption>
        </figure>
      </div>

      <div className="module1-graph-panel">
        <h3>Ultraviolet Catastrophe Graph</h3>
        <p>Normalized spectral radiance versus wavelength for Planck and Rayleigh-Jeans models.</p>
        <div className="module1-graph-wrap">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={moduleOneBlackbodyData} margin={{ top: 6, right: 16, left: 8, bottom: 10 }}>
              <CartesianGrid stroke="#29425c" strokeDasharray="4 4" />
              <XAxis dataKey="wavelengthNm" stroke="#a7c9e8" tick={{ fill: '#cfe5f8' }} tickFormatter={(value) => `${value}`} />
              <YAxis stroke="#a7c9e8" tick={{ fill: '#cfe5f8' }} tickFormatter={(value) => Number(value).toFixed(1)} />
              <Tooltip
                contentStyle={chartTooltipStyle}
                labelFormatter={(value) => `Wavelength: ${value} nm`}
                formatter={(value, name) => [`${Number(value).toFixed(3)} (normalized)`, name]}
              />
              <Legend />
              <Line type="monotone" dataKey="planck3500" name="Planck 3500 K" stroke="#ff9f6e" strokeWidth={2.15} dot={false} />
              <Line type="monotone" dataKey="planck5000" name="Planck 5000 K" stroke="#63d6ff" strokeWidth={2.15} dot={false} />
              <Line type="monotone" dataKey="planck6500" name="Planck 6500 K" stroke="#ffe07a" strokeWidth={2.15} dot={false} />
              <Line type="monotone" dataKey="rayleigh6500" name="Rayleigh-Jeans 6500 K" stroke="#f04f8f" strokeWidth={2.1} strokeDasharray="7 5" dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="2) Planck and Energy Quanta" />
      <p className="module1-detail-extension">{moduleOneSectionExpansionNote}</p>
      <div className="module1-split">
        <div className="module1-text-flow">
          <p>
            Max Planck resolved the blackbody problem by proposing that energy exchange is quantized.
            Radiation is emitted and absorbed in discrete packets.
          </p>
          <ExpandableFormula className="compact module1-formula-card" math={'E=h\\nu'} />
          <p>
            This single relation established the quantum energy scale and became the basis for spectroscopy,
            transitions, and later quantum wave mechanics.
          </p>
        </div>
        <figure className="module1-figure">
          <img
            src="https://upload.wikimedia.org/wikipedia/commons/c/c7/Max_Planck_1933.jpg"
            alt="Max Planck portrait"
            loading="lazy"
          />
          <figcaption>Max Planck. Source: Wikimedia Commons.</figcaption>
        </figure>
      </div>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="3) Photoelectric Effect and Einstein" />
      <p className="module1-detail-extension">{moduleOneSectionExpansionNote}</p>
      <div className="module1-split module1-split-reverse">
        <figure className="module1-figure is-wide">
          <img
            src="https://upload.wikimedia.org/wikipedia/commons/5/52/Photoelectric_effect_-_stopping_voltage_diagram_for_zinc_-_English.svg"
            alt="Photoelectric effect stopping voltage diagram"
            loading="lazy"
          />
          <figcaption>Photoelectric-effect stopping-voltage diagram. Source: Wikimedia Commons.</figcaption>
        </figure>
        <div className="module1-text-flow">
          <p>
            Einstein extended Planck&apos;s idea and explained why electrons are emitted only above a threshold frequency.
            Intensity changes the number of emitted electrons, but not their maximum kinetic energy at fixed frequency.
          </p>
          <ExpandableFormula className="compact module1-formula-card" math={'K_{max}=h\\nu-\\phi'} />
          <p>
            Here phi is the work function of the material. Below threshold, emission does not occur.
          </p>
        </div>
      </div>

      <div className="module1-graph-panel">
        <h3>Photoelectric Effect: Maximum Kinetic Energy vs Frequency</h3>
        <p>For a representative metal with work function phi = 2.3 eV.</p>
        <div className="module1-graph-wrap">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={moduleOnePhotoelectricData} margin={{ top: 6, right: 16, left: 8, bottom: 10 }}>
              <CartesianGrid stroke="#29425c" strokeDasharray="4 4" />
              <XAxis dataKey="frequencyPHz" stroke="#a7c9e8" tick={{ fill: '#cfe5f8' }} tickFormatter={(value) => `${value}`} />
              <YAxis stroke="#a7c9e8" tick={{ fill: '#cfe5f8' }} tickFormatter={(value) => Number(value).toFixed(1)} />
              <Tooltip
                contentStyle={chartTooltipStyle}
                labelFormatter={(value) => `Frequency: ${value} PHz`}
                formatter={(value, name) => [`${Number(value).toFixed(3)} eV`, name]}
              />
              <Legend />
              <Line type="monotone" dataKey="kineticEnergyEV" name="Kmax" stroke="#66f0c7" strokeWidth={2.3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="4) Matter Waves (de Broglie)" />
      <p className="module1-detail-extension">{moduleOneSectionExpansionNote}</p>
      <p>
        Electrons show wave behavior. Their wavelength is linked directly to momentum.
      </p>
      <ExpandableFormula className="compact module1-formula-card" math={'\\lambda=\\frac{h}{p}'} />
      <p>
        This relation explains electron diffraction and why bound electronic states around nuclei form standing-wave-like structures.
      </p>
      <div className="module1-graph-panel">
        <h3>de Broglie Wavelength vs Kinetic Energy</h3>
        <p>Wavelength decreases as electron momentum rises.</p>
        <div className="module1-graph-wrap">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={moduleOneDeBroglieData} margin={{ top: 6, right: 16, left: 8, bottom: 10 }}>
              <CartesianGrid stroke="#29425c" strokeDasharray="4 4" />
              <XAxis dataKey="kineticEV" stroke="#a7c9e8" tick={{ fill: '#cfe5f8' }} tickFormatter={(value) => `${value}`} />
              <YAxis stroke="#a7c9e8" tick={{ fill: '#cfe5f8' }} tickFormatter={(value) => Number(value).toFixed(0)} />
              <Tooltip
                contentStyle={chartTooltipStyle}
                labelFormatter={(value) => `Kinetic energy: ${value} eV`}
                formatter={(value) => [`${Number(value).toFixed(2)} pm`, 'Wavelength']}
              />
              <Line type="monotone" dataKey="wavelengthPm" stroke="#ffd265" strokeWidth={2.3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="5) Born Interpretation" />
      <p className="module1-detail-extension">{moduleOneSectionExpansionNote}</p>
      <p>
        The wavefunction is not a trajectory. It is a probability amplitude.
      </p>
      <ExpandableFormula className="compact module1-formula-card" math={'P(r,\\theta,\\phi)=|\\psi(r,\\theta,\\phi)|^2'} />
      <p>
        Orbital plots in this simulator visualize this probability density, so bright regions indicate higher detection probability.
      </p>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="6) Heisenberg Uncertainty Principle" />
      <p className="module1-detail-extension">{moduleOneSectionExpansionNote}</p>
      <div className="module1-split">
        <div className="module1-text-flow">
          <p>
            Position and momentum are constrained by a fundamental lower bound.
            This is a structural feature of quantum states, not an instrument defect.
          </p>
          <ExpandableFormula className="compact module1-formula-card" math={'\\Delta x\\,\\Delta p\\geq\\frac{\\hbar}{2}'} />
          <p>
            As position uncertainty decreases, minimum momentum uncertainty increases.
          </p>
        </div>
        <figure className="module1-figure">
          <img
            src="https://upload.wikimedia.org/wikipedia/commons/e/ee/Werner_Heisenberg_Portrait_%283x4_cropped%29.jpg"
            alt="Werner Heisenberg portrait"
            loading="lazy"
          />
          <figcaption>Werner Heisenberg. Source: Wikimedia Commons.</figcaption>
        </figure>
      </div>

      <div className="module1-graph-panel">
        <h3>Minimum Momentum Uncertainty vs Position Uncertainty</h3>
        <p>Scaled in 10^-24 kg m/s for readability.</p>
        <div className="module1-graph-wrap">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={moduleOneUncertaintyData} margin={{ top: 6, right: 16, left: 8, bottom: 10 }}>
              <CartesianGrid stroke="#29425c" strokeDasharray="4 4" />
              <XAxis dataKey="deltaXPm" stroke="#a7c9e8" tick={{ fill: '#cfe5f8' }} tickFormatter={(value) => `${value}`} />
              <YAxis stroke="#a7c9e8" tick={{ fill: '#cfe5f8' }} tickFormatter={(value) => Number(value).toFixed(2)} />
              <Tooltip
                contentStyle={chartTooltipStyle}
                labelFormatter={(value) => `Delta x: ${value} pm`}
                formatter={(value) => [`${Number(value).toFixed(3)} x10^-24 kg m/s`, 'Delta p min']}
              />
              <Line type="monotone" dataKey="minDeltaPScaled" stroke="#67f0c7" strokeWidth={2.3} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="7) Schrödinger Equation for Hydrogenic Atoms" />
      <p className="module1-detail-extension">{moduleOneSectionExpansionNote}</p>
      <p>
        Bound states are obtained by solving the time-independent Schrödinger equation with the Coulomb potential.
      </p>
      <ExpandableFormula className="compact module1-formula-card" math={'-\\frac{\\hbar^2}{2\\mu}\\nabla^2\\psi-\\frac{Ze^2}{4\\pi\\epsilon_0 r}\\psi=E\\psi'} />
      <p>
        This yields quantized energies and wavefunctions that form the orbital families shown in visualization.
      </p>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="8) Separation, Quantum Numbers, and Orbital Structure" />
      <p className="module1-detail-extension">{moduleOneSectionExpansionNote}</p>
      <p>
        The solution separates into radial and angular components, each tied to quantum numbers.
      </p>
      <ExpandableFormula className="compact module1-formula-card" math={'\\psi_{n,l,m}(r,\\theta,\\phi)=R_{n,l}(r)Y_l^m(\\theta,\\phi)'} />
      <ul className="module1-bullet-list">
        <li>n sets the shell scale and principal energy level.</li>
        <li>l sets orbital shape family (s, p, d, f).</li>
        <li>m sets orientation for a chosen quantization axis.</li>
      </ul>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="9) Radial Probability Distribution (1s, 2s, 2p)" />
      <p className="module1-detail-extension">{moduleOneSectionExpansionNote}</p>
      <p>
        Radial probability identifies where the electron is most likely to be found at a distance r from the nucleus.
      </p>
      <div className="module1-graph-panel">
        <h3>Hydrogen Radial Probability Curves</h3>
        <p>Comparison of normalized radial trends for 1s, 2s, and 2p states.</p>
        <div className="module1-graph-wrap">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={moduleOneRadialDistributionData} margin={{ top: 6, right: 16, left: 8, bottom: 10 }}>
              <CartesianGrid stroke="#29425c" strokeDasharray="4 4" />
              <XAxis dataKey="radiusA0" stroke="#a7c9e8" tick={{ fill: '#cfe5f8' }} tickFormatter={(value) => `${value}`} />
              <YAxis stroke="#a7c9e8" tick={{ fill: '#cfe5f8' }} tickFormatter={(value) => Number(value).toFixed(2)} />
              <Tooltip
                contentStyle={chartTooltipStyle}
                labelFormatter={(value) => `r/a0: ${value}`}
                formatter={(value, name) => [`${Number(value).toFixed(4)}`, name]}
              />
              <Legend />
              <Line type="monotone" dataKey="oneS" name="1s" stroke="#63d6ff" strokeWidth={2.15} dot={false} />
              <Line type="monotone" dataKey="twoS" name="2s" stroke="#ffe07a" strokeWidth={2.15} dot={false} />
              <Line type="monotone" dataKey="twoP" name="2p" stroke="#ff8fb8" strokeWidth={2.15} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="10) Direct Mapping to This Simulator" />
      <p className="module1-detail-extension">{moduleOneSectionExpansionNote}</p>
      <p>
        The interface is a visual expression of core quantum theory, not a decorative model.
      </p>
      <ul className="module1-bullet-list">
        <li>n changes shell scale and radial node count.</li>
        <li>l changes angular structure and orbital family.</li>
        <li>m rotates orientation of the same family.</li>
        <li>Phase and slice views expose sign structure and interior nodes.</li>
      </ul>
      <p>
        Reading these visuals through the equations above is the key learning goal of Module 1.
      </p>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="11) Quantitative Validation and Common Misconceptions" />
      <p className="module1-detail-extension">{moduleOneSectionExpansionNote}</p>
      <p>
        A physically correct interpretation should pass three checks: normalization, node consistency, and spectral consistency.
        If one fails, the interpretation is incomplete even if the rendered image looks plausible.
      </p>
      <div className="module1-equation-stack">
        <ExpandableFormula className="compact module1-formula-card" math={'\\int |\\psi|^2\\,d\\tau=1'} />
        <ExpandableFormula className="compact module1-formula-card" math={'\\Delta E = h\\nu'} />
      </div>
      <ul className="module1-bullet-list">
        <li>Phase colors represent wavefunction sign, not positive or negative electric charge.</li>
        <li>Orbital surfaces are probability structures, not classical electron trajectories.</li>
        <li>Radial and angular node counts must agree with n and l for any valid state interpretation.</li>
      </ul>
      <p>
        Treat this section as a reliability filter: if your explanation satisfies these checks, your model-to-visual mapping is usually sound.
      </p>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="12) Correspondence Principle and the High-n Limit" />
      <p className="module1-detail-extension">{moduleOneSectionExpansionNote}</p>
      <p>
        Quantum and classical descriptions are not disconnected theories. In the large-n regime, level spacing compresses and behavior approaches classical expectations.
      </p>
      <div className="module1-equation-stack">
        <ExpandableFormula className="compact module1-formula-card" math={'E_n=-\\frac{13.6}{n^2}\\,\\text{eV}'} />
        <ExpandableFormula className="compact module1-formula-card" math={'\\Delta E_{n\\to n+1}\\approx\\frac{27.2}{n^3}\\,\\text{eV}\\quad(n\\gg1)'} />
      </div>
      <p>
        This trend explains why spectral lines crowd at high principal quantum number and why quantum predictions recover smooth classical-like behavior in coarse resolution limits.
      </p>
    </AnimatedSection>
  </>
);

const ModuleTwoQuantumNumbersContent = () => (
  <>
    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="Quantum Numbers, Eigenstates, Angular Momentum, and Spherical Harmonics in Full Detail" />
      <p className="module1-detail-extension">{moduleTwoSectionExpansionNote}</p>
      <p className="module1-intro-note">The complete mathematical framework behind atomic orbital states.</p>
      <p>
        Every orbital displayed in this simulator is a mathematically allowed stationary state of an electron in the Coulomb field.
        These states are exact solutions of the Schrodinger equation, not arbitrary geometric sketches.
      </p>
      <p>
        To understand quantum numbers deeply, one must track their mathematical origin, the operator whose value each number represents,
        and how the full set maps directly to orbital size, shape, orientation, and nodal structure.
      </p>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="1) From the Schrodinger Equation to Allowed States" />
      <p className="module1-detail-extension">{moduleTwoSectionExpansionNote}</p>
      <p>
        The starting point is the time-independent Schrodinger equation for a hydrogen-like species.
      </p>
      <div className="module1-equation-stack">
        <ExpandableFormula className="compact module1-formula-card" math={'-\\frac{\\hbar^2}{2\\mu}\\nabla^2\\psi(r,\\theta,\\phi)-\\frac{Ze^2}{4\\pi\\epsilon_0 r}\\psi(r,\\theta,\\phi)=E\\psi(r,\\theta,\\phi)'} />
        <ExpandableFormula className="compact module1-formula-card" math={'V(r)=-\\frac{Ze^2}{4\\pi\\epsilon_0 r}'} />
      </div>
      <p>
        The negative sign in the potential is physically essential: it encodes attraction between nucleus and electron.
        Because the potential depends only on radius r, the problem is spherically symmetric and is naturally expressed in (r, theta, phi).
      </p>
      <p>
        That symmetry is the structural reason angular momentum operators appear, and it is also the reason orbital states are organized by quantum numbers.
      </p>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="2) Separation of Variables and the Birth of Quantum Numbers" />
      <p className="module1-detail-extension">{moduleTwoSectionExpansionNote}</p>
      <p>
        The wavefunction is separated into radial and angular factors so each physical dependency can be solved under its own boundary conditions.
      </p>
      <ExpandableFormula className="compact module1-formula-card" math={'\\psi(r,\\theta,\\phi)=R(r)\\Theta(\\theta)\\Phi(\\phi)'} />
      <p>
        This is more than a formal trick. It isolates distance-dependent behavior from directional behavior,
        and each resulting differential equation contributes an allowed set of constants.
      </p>
      <p>
        Those constants are exactly the quantum numbers. Their allowed values are forced by regularity, single-valuedness,
        and normalizability of the full solution.
      </p>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="3) Eigenstates and Eigenvalues in Physical Detail" />
      <p className="module1-detail-extension">{moduleTwoSectionExpansionNote}</p>
      <p>
        In quantum mechanics, observables are represented by operators. A state is an eigenstate when operator action returns
        the same state times a constant eigenvalue.
      </p>
      <div className="module1-equation-stack">
        <ExpandableFormula className="compact module1-formula-card" math={'\\hat{A}\\psi=a\\psi'} />
        <ExpandableFormula className="compact module1-formula-card" math={'\\hat{H}\\psi=E\\psi'} />
      </div>
      <p>
        Atomic orbitals are energy eigenstates of the Hamiltonian, so each stationary orbital carries a definite allowed energy.
        This is why atomic spectra are discrete: transitions occur between quantized eigenvalues, not a continuous continuum.
      </p>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="4) Angular Momentum Operators and Quantization" />
      <p className="module1-detail-extension">{moduleTwoSectionExpansionNote}</p>
      <p>
        In a spherically symmetric Coulomb field, orbital angular momentum is conserved and the angular wavefunction must satisfy
        simultaneous eigenvalue equations for total angular momentum and its z-projection.
      </p>
      <div className="module1-equation-stack">
        <ExpandableFormula className="compact module1-formula-card" math={'\\hat{L}^2Y_\\ell^m=\\ell(\\ell+1)\\hbar^2Y_\\ell^m'} />
        <ExpandableFormula className="compact module1-formula-card" math={'L=\\sqrt{\\ell(\\ell+1)}\\hbar'} />
        <ExpandableFormula className="compact module1-formula-card" math={'\\hat{L}_zY_\\ell^m=m\\hbar Y_\\ell^m'} />
        <ExpandableFormula className="compact module1-formula-card" math={'L_z=m\\hbar'} />
      </div>
      <p>
        The first pair quantizes orbital angular momentum magnitude, while the second pair sets orientation projection along the chosen axis.
        These are measurable quantities, not symbolic labels.
      </p>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="5) Spherical Harmonics and Orbital Geometry" />
      <p className="module1-detail-extension">{moduleTwoSectionExpansionNote}</p>
      <div className="module1-split">
        <div className="module1-text-flow">
          <p>
            The angular solutions Y_l^m(theta, phi) are the spherical harmonics. They define lobe count, nodal planes and cones,
            orientation, and phase sign regions. Orbital shape is therefore a direct mathematical consequence of angular eigenfunctions.
          </p>
          <div className="module1-equation-stack">
            <ExpandableFormula className="compact module1-formula-card" math={'Y_\\ell^m(\\theta,\\phi)'} />
            <ExpandableFormula className="compact module1-formula-card" math={'Y_0^0=\\text{constant}'} />
            <ExpandableFormula className="compact module1-formula-card" math={'Y_1^0\\propto\\cos\\theta'} />
            <ExpandableFormula className="compact module1-formula-card" math={'\\theta=\\frac{\\pi}{2}\\Rightarrow\\cos\\theta=0'} />
          </div>
          <p>
            The s case has no angular dependence and is spherically symmetric. The p case has a nodal plane at theta = pi/2,
            producing two opposite lobes. For l = 2, spherical harmonics generate d-family multi-lobed structures.
          </p>
        </div>
        <figure className="module1-figure is-wide">
          <img
            src="https://upload.wikimedia.org/wikipedia/commons/thumb/6/62/Spherical_Harmonics.png/1280px-Spherical_Harmonics.png"
            alt="Spherical harmonics gallery"
            loading="lazy"
          />
          <figcaption>Spherical harmonics gallery. Source: Wikimedia Commons.</figcaption>
        </figure>
      </div>
      <figure className="module1-figure is-wide">
        <img
          src="https://upload.wikimedia.org/wikipedia/commons/thumb/b/b0/Atomic_orbitals_n1234_m-eigenstates.png/1280px-Atomic_orbitals_n1234_m-eigenstates.png"
          alt="s p d f orbital comparison"
          loading="lazy"
        />
        <figcaption>s/p/d/f orbital comparison with m-state structure. Source: Wikimedia Commons.</figcaption>
      </figure>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="6) Principal Quantum Number n" />
      <p className="module1-detail-extension">{moduleTwoSectionExpansionNote}</p>
      <p>
        The principal quantum number arises from the radial equation and sets shell index, baseline energy, radial extent, and total node count.
      </p>
      <div className="module1-equation-stack">
        <ExpandableFormula className="compact module1-formula-card" math={'n=1,2,3,\\dots'} />
        <ExpandableFormula className="compact module1-formula-card" math={'E_n=-\\frac{13.6Z^2}{n^2}\\text{ eV}'} />
      </div>
      <p>
        Larger n gives larger average radius, weaker binding, and richer radial oscillation structure.
        In visualization, this appears as larger clouds with increased radial layering.
      </p>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="7) Azimuthal Quantum Number l" />
      <p className="module1-detail-extension">{moduleTwoSectionExpansionNote}</p>
      <p>
        The azimuthal quantum number determines orbital family and angular complexity, with allowed values tied to n.
      </p>
      <div className="module1-equation-stack">
        <ExpandableFormula className="compact module1-formula-card" math={'\\ell=0,1,2,\\dots,n-1'} />
        <ExpandableFormula className="compact module1-formula-card" math={'N_{\\text{angular}}=\\ell'} />
      </div>
      <p>
        It sets angular momentum magnitude and the number of angular nodes, so l directly controls orbital shape family
        and nodal surface count.
      </p>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="8) Magnetic Quantum Number m" />
      <p className="module1-detail-extension">{moduleTwoSectionExpansionNote}</p>
      <p>
        For fixed n and l, m selects the orientation state. This is the directional degree of freedom that appears as rotated
        but family-equivalent orbitals in isotropic fields.
      </p>
      <div className="module1-equation-stack">
        <ExpandableFormula className="compact module1-formula-card" math={'m=-\\ell,\\dots,+\\ell'} />
        <ExpandableFormula className="compact module1-formula-card" math={'\\text{Number of orientations}=2\\ell+1'} />
      </div>
      <p>
        The simulator makes this visible when m changes lobe alignment while preserving the parent orbital family fixed by l.
      </p>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="9) Spin Quantum Number" />
      <p className="module1-detail-extension">{moduleTwoSectionExpansionNote}</p>
      <p>
        Electron spin is intrinsic angular momentum and is independent of orbital motion around the nucleus.
      </p>
      <ExpandableFormula className="compact module1-formula-card" math={'m_s=\\pm\\frac12'} />
      <p>
        Although spatial orbital shape is set by n, l, and m, complete electronic state specification also requires spin.
        This becomes essential when constructing many-electron configurations and selection rules.
      </p>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="10) Nodes in Full Detail and Direct Orbital Mapping" />
      <p className="module1-detail-extension">{moduleTwoSectionExpansionNote}</p>
      <p>
        Node accounting unifies radial and angular structure and provides the cleanest bridge from equations to 3D orbital interpretation.
      </p>
      <div className="module1-equation-stack">
        <ExpandableFormula className="compact module1-formula-card" math={'N_{\\text{total}}=n-1'} />
        <ExpandableFormula className="compact module1-formula-card" math={'N_{\\text{angular}}=\\ell'} />
        <ExpandableFormula className="compact module1-formula-card" math={'N_{\\text{radial}}=n-\\ell-1'} />
        <ExpandableFormula className="compact module1-formula-card" math={'3p: N_{\\text{total}}=2; N_{\\text{angular}}=1; N_{\\text{radial}}=1'} />
      </div>
      <p>
        The radial probability plot below shows how node structure emerges as radius-dependent zero crossings and redistributions in density.
      </p>
      <div className="module1-graph-panel">
        <h3>Radial Node Probability Plot</h3>
        <p>Normalized radial probability trends for 1s, 2s, and 3s hydrogenic states.</p>
        <div className="module1-graph-wrap">
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={moduleTwoRadialNodeData} margin={{ top: 6, right: 16, left: 8, bottom: 10 }}>
              <CartesianGrid stroke="#29425c" strokeDasharray="4 4" />
              <XAxis dataKey="radiusA0" stroke="#a7c9e8" tick={{ fill: '#cfe5f8' }} tickFormatter={(value) => `${value}`} />
              <YAxis stroke="#a7c9e8" tick={{ fill: '#cfe5f8' }} tickFormatter={(value) => Number(value).toFixed(2)} />
              <Tooltip
                contentStyle={chartTooltipStyle}
                labelFormatter={(value) => `r/a0: ${value}`}
                formatter={(value, name) => [`${Number(value).toFixed(4)}`, name]}
              />
              <Legend />
              <Line type="monotone" dataKey="oneS" name="1s" stroke="#63d6ff" strokeWidth={2.15} dot={false} />
              <Line type="monotone" dataKey="twoS" name="2s" stroke="#ffe07a" strokeWidth={2.15} dot={false} />
              <Line type="monotone" dataKey="threeS" name="3s" stroke="#ff8fb8" strokeWidth={2.15} dot={false} />
            </LineChart>
          </ResponsiveContainer>
        </div>
      </div>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="11) Selection Rules and Spectroscopic Consequences" />
      <p className="module1-detail-extension">{moduleTwoSectionExpansionNote}</p>
      <p>
        Quantum numbers are not only geometric labels; they govern which transitions are allowed in electromagnetic spectroscopy.
        These rules explain why some spectral lines are strong while others are forbidden.
      </p>
      <div className="module1-equation-stack">
        <ExpandableFormula className="compact module1-formula-card" math={'\\Delta l=\\pm1'} />
        <ExpandableFormula className="compact module1-formula-card" math={'\\Delta m=0,\\pm1'} />
      </div>
      <p>
        In practical terms, state geometry, phase structure, and operator selection rules work together: not every mathematically imaginable jump is physically observable.
      </p>
      <p>
        This is why quantum number literacy is essential for interpreting both orbital renderings and real spectroscopic measurements.
      </p>
    </AnimatedSection>

    <AnimatedSection className="info-section module1-section">
      <ModuleOneSectionHeading title="12) Degeneracy Counting and State Capacity" />
      <p className="module1-detail-extension">{moduleTwoSectionExpansionNote}</p>
      <p>
        Beyond individual orbital interpretation, quantum numbers also define how many states are available in each shell and subshell.
        This counting is essential for connecting atomic orbitals to electronic configuration logic.
      </p>
      <div className="module1-equation-stack">
        <ExpandableFormula className="compact module1-formula-card" math={'g_l=2l+1'} />
        <ExpandableFormula className="compact module1-formula-card" math={'g_n=n^2\\quad(\\text{orbital states}),\\qquad G_n=2n^2\\quad(\\text{including spin})'} />
      </div>
      <p>
        These degeneracy relations provide a compact consistency check when moving from one-electron hydrogenic intuition to multi-electron shell-capacity reasoning.
      </p>
    </AnimatedSection>
  </>
);

const ModuleInlineFigure = ({ slot }) => {
  if (!slot?.src) return null;

  const captionParts = [slot.title, slot.description, slot.credit].filter(Boolean);

  return (
    <figure className="module1-figure is-wide">
      <img src={slot.src} alt={slot.alt || slot.title} loading="lazy" />
      <figcaption>{captionParts.join(' ')}</figcaption>
    </figure>
  );
};

const ModuleLongFormChartPanel = ({ title, subtitle, data, lines }) => {
  const xAxisKey = Array.isArray(data) && data.length > 0 ? Object.keys(data[0])[0] : 'x';

  return (
    <div className="module1-graph-panel">
      <h3>{title}</h3>
      {subtitle && <p>{subtitle}</p>}
      <div className="module1-graph-wrap">
        <ResponsiveContainer width="100%" height="100%">
          <LineChart data={data} margin={{ top: 6, right: 16, left: 8, bottom: 10 }}>
            <CartesianGrid stroke="#29425c" strokeDasharray="4 4" />
            <XAxis dataKey={xAxisKey} stroke="#a7c9e8" tick={{ fill: '#cfe5f8' }} tickFormatter={(value) => `${value}`} />
            <YAxis stroke="#a7c9e8" tick={{ fill: '#cfe5f8' }} tickFormatter={(value) => (typeof value === 'number' ? Number(value).toFixed(2) : value)} />
            <Tooltip
              contentStyle={chartTooltipStyle}
              formatter={(value, name) => [typeof value === 'number' ? Number(value).toFixed(4) : value, name]}
            />
            <Legend />
            {Array.isArray(lines) && lines.map((line) => (
              <Line key={line.dataKey} type="monotone" dataKey={line.dataKey} name={line.name} stroke={line.color} strokeWidth={2.15} dot={false} />
            ))}
          </LineChart>
        </ResponsiveContainer>
      </div>
    </div>
  );
};

const ModuleThreeToSevenLongFormContent = ({ topic }) => {
  const theorySections = Array.isArray(topic.theoryCards)
    ? topic.theoryCards.map((item) => ({ ...item, kind: 'theory' }))
    : [];
  const deepSections = Array.isArray(topic.deepDiveSections)
    ? topic.deepDiveSections.map((item) => ({ ...item, kind: 'deep' }))
    : [];
  const supplementalSections = Array.isArray(topic.supplementarySections)
    ? topic.supplementarySections.map((item) => ({ ...item, kind: 'supplement' }))
    : [];
  const sections = [...theorySections, ...deepSections, ...supplementalSections];

  const inlineImages = Array.isArray(topic.imageSlots)
    ? topic.imageSlots.filter((slot) => slot?.src)
    : [];
  const imageAnchorIndices = [0, 3, 7].filter((index) => index < sections.length);
  const sectionImageMap = new Map();
  imageAnchorIndices.forEach((sectionIndex, imageIndex) => {
    const slot = inlineImages[imageIndex];
    if (slot) sectionImageMap.set(sectionIndex, slot);
  });

  const equationSectionNumber = sections.length + 1;
  const primaryChartSectionNumber = equationSectionNumber + 1;
  const secondaryChartStart = primaryChartSectionNumber + 1;
  const secondaryChartCount = Array.isArray(topic.secondaryCharts) ? topic.secondaryCharts.length : 0;
  const tableSectionNumber = secondaryChartStart + secondaryChartCount;
  const closingSectionNumber = tableSectionNumber + 1;

  return (
    <>
      <AnimatedSection className="info-section module1-section">
        <ModuleOneSectionHeading title={topic.title} />
        <p className="module1-intro-note">{topic.message}</p>
        <p>
          This module is written as a rigorous long-form walkthrough. Each heading is intended to connect formal equations,
          physical interpretation, and the exact visual patterns you can verify in the simulator.
        </p>
        <p>
          Use this sequence as a scientific reading path: define the concept, test with equations, and validate with phase,
          slicing, and radial plots before making conclusions.
        </p>
        <p>
          To deepen retention, pause after every section and restate the claim in your own words, then cross-check one equation,
          one chart trend, and one image cue before moving on. This deliberate pass turns each heading from passive reading into
          a reproducible scientific reasoning step that can be reused in classroom explanation, exam derivation, and simulator practice.
        </p>
      </AnimatedSection>

      {sections.map((section, index) => {
        const number = index + 1;
        const imageSlot = sectionImageMap.get(index);
        const splitClassName = `module1-split ${number % 2 === 0 ? 'module1-split-reverse' : ''}`.trim();

        return (
          <AnimatedSection className="info-section module1-section" key={`${section.title}-${number}`}>
            <ModuleOneSectionHeading title={`${number}) ${section.title}`} />
            {imageSlot ? (
              <div className={splitClassName}>
                <div className="module1-text-flow">
                  <p>{section.text}</p>
                  <p>
                    {section.kind === 'theory'
                      ? 'Interpret this heading with both geometry and probability in mind: shape alone is incomplete unless node structure and phase sign are also consistent.'
                      : section.kind === 'deep'
                        ? 'Use this as a quantitative checkpoint. If rendered behavior disagrees with this relation, verify quantum inputs and node accounting before revising physical interpretation.'
                        : 'Treat this as an advanced extension layer: connect it to at least one equation or chart before accepting a final interpretation.'}
                  </p>
                  <p>
                    A strong analysis should also identify what would falsify the current claim. For example, if the predicted node count,
                    orientation pattern, or radial trend is not observed under rotation and slice inspection, return to the quantum-number
                    constraints before accepting the conclusion. This habit keeps interpretation accurate and prevents confirmation bias.
                  </p>
                  {section.equation && <ExpandableFormula className="compact module1-formula-card" math={section.equation} />}
                </div>
                <ModuleInlineFigure slot={imageSlot} />
              </div>
            ) : (
              <>
                <p>{section.text}</p>
                <p>
                  {section.kind === 'theory'
                    ? 'A reliable interpretation should remain valid under camera rotation, slice inspection, and phase-view toggles.'
                    : section.kind === 'deep'
                      ? 'When using this relation, treat it as a formal constraint that must match both plotted trends and observed node topology.'
                      : 'Use this advanced extension to strengthen cross-check quality between formal equations and rendered structure.'}
                </p>
                <p>
                  For deeper mastery, convert the section claim into a short prediction sentence before looking at the graph or figure,
                  then verify whether observed behavior supports that prediction. Repeating this predict-and-verify cycle across sections
                  builds durable intuition and keeps your reasoning aligned with quantitative constraints.
                </p>
                {section.equation && <ExpandableFormula className="compact module1-formula-card" math={section.equation} />}
              </>
            )}
          </AnimatedSection>
        );
      })}

      {Array.isArray(topic.equations) && topic.equations.length > 0 && (
        <AnimatedSection className="info-section module1-section">
          <ModuleOneSectionHeading title={`${equationSectionNumber}) Equation Toolbox`} />
          <p>
            These equations are the core reference set for this module. Revisit them whenever a visual intuition and a formal prediction appear to disagree.
          </p>
          <div className="module1-equation-stack">
            {topic.equations.map((equation) => (
              <div className="module1-formula-card" key={equation.title}>
                <h3>{equation.title}</h3>
                <ExpandableFormula className="compact" math={equation.math} />
                <p>{equation.description}</p>
              </div>
            ))}
          </div>
        </AnimatedSection>
      )}

      {Array.isArray(topic.chartData) && topic.chartData.length > 0 && (
        <AnimatedSection className="info-section module1-section">
          <ModuleOneSectionHeading title={`${primaryChartSectionNumber}) ${topic.chartTitle}`} />
          <ModuleLongFormChartPanel title={topic.chartTitle} subtitle={topic.chartSubtitle} data={topic.chartData} lines={topic.chartLines} />
        </AnimatedSection>
      )}

      {Array.isArray(topic.secondaryCharts) && topic.secondaryCharts.map((chart, index) => (
        <AnimatedSection className="info-section module1-section" key={chart.title}>
          <ModuleOneSectionHeading title={`${secondaryChartStart + index}) ${chart.title}`} />
          <ModuleLongFormChartPanel title={chart.title} subtitle={chart.subtitle} data={chart.data} lines={chart.lines} />
        </AnimatedSection>
      ))}

      {topic.table && (
        <AnimatedSection className="info-section module1-section">
          <ModuleOneSectionHeading title={`${tableSectionNumber}) Structured Reference Table`} />
          <p>
            Use this table as a compact verification grid for solved problems, tutorials, and classroom explanations.
          </p>
          <DataTable columns={topic.table.columns} rows={topic.table.rows} compact />
        </AnimatedSection>
      )}

      <AnimatedSection className="info-section module1-section">
        <ModuleOneSectionHeading title={`${closingSectionNumber}) Accuracy Checklist`} />
        <ul className="module1-bullet-list">
          <li>Confirm the node budget matches n and l before interpreting shape.</li>
          <li>Use phase and slice views to verify hidden boundaries, not only surface density.</li>
          <li>Cross-check at least one plotted trend against one equation from the toolbox.</li>
          <li>Treat image references as contextual aids; always privilege the equations for final interpretation.</li>
        </ul>
      </AnimatedSection>
    </>
  );
};

const LearnTopicPage = ({ topic, onNavigate }) => {
  const topicIndex = learnTopics.findIndex((item) => item.id === topic.id);
  const previousTopic = topicIndex > 0 ? learnMenuLinks[topicIndex - 1] : null;
  const nextTopic = topicIndex >= 0 && topicIndex < learnMenuLinks.length - 1 ? learnMenuLinks[topicIndex + 1] : null;

  const moduleNavigationSection = (
    <InfoSection title="Module Navigation" subtitle="Go to the previous or next learning module.">
      <div className="learn-module-footer-nav">
        {previousTopic && (
          <button type="button" className="learn-module-nav-btn is-prev" onClick={() => onNavigate(previousTopic.id)}>
            <span className="learn-module-nav-arrow" aria-hidden="true">&larr;</span>
            <span className="learn-module-nav-copy">
              <span className="learn-module-nav-kicker">Previous Module</span>
              <span className="learn-module-nav-title">{previousTopic.numberedShortLabel}</span>
            </span>
          </button>
        )}
        {nextTopic && (
          <button type="button" className="learn-module-nav-btn is-next" onClick={() => onNavigate(nextTopic.id)}>
            <span className="learn-module-nav-copy">
              <span className="learn-module-nav-kicker">Next Module</span>
              <span className="learn-module-nav-title">{nextTopic.numberedShortLabel}</span>
            </span>
            <span className="learn-module-nav-arrow" aria-hidden="true">&rarr;</span>
          </button>
        )}
      </div>
    </InfoSection>
  );

  if (topic.id === 'learn-quantum-theory') {
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
        <ModuleOneCoreTheoryContent />

        {Array.isArray(topic.faqs) && topic.faqs.length > 0 && (
          <InfoSection title="Module FAQs" subtitle="Quick answers for common questions in this module.">
            <div className="faq-list">
              {topic.faqs.map((faq) => (
                <FAQAccordion key={faq.question} question={faq.question} answer={faq.answer} />
              ))}
            </div>
          </InfoSection>
        )}

        {moduleNavigationSection}
      </InfoPageLayout>
    );
  }

  if (topic.id === 'learn-quantum-numbers-detail') {
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
        <ModuleTwoQuantumNumbersContent />

        {Array.isArray(topic.faqs) && topic.faqs.length > 0 && (
          <InfoSection title="Module FAQs" subtitle="Quick answers for common questions in this module.">
            <div className="faq-list">
              {topic.faqs.map((faq) => (
                <FAQAccordion key={faq.question} question={faq.question} answer={faq.answer} />
              ))}
            </div>
          </InfoSection>
        )}

        {moduleNavigationSection}
      </InfoPageLayout>
    );
  }

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
      <ModuleThreeToSevenLongFormContent topic={topic} />

      {Array.isArray(topic.faqs) && topic.faqs.length > 0 && (
        <InfoSection title="Module FAQs" subtitle="Quick answers for common questions in this module.">
          <div className="faq-list">
            {topic.faqs.map((faq) => (
              <FAQAccordion key={faq.question} question={faq.question} answer={faq.answer} />
            ))}
          </div>
        </InfoSection>
      )}

      {moduleNavigationSection}
    </InfoPageLayout>
  );
};

const InfoFooter = ({ onNavigate }) => (
  <footer className="site-footer">
    <div className="site-footer-grid">
      <section className="footer-about">
        <h3>Company</h3>
        <p className="footer-summary">
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
          <ul className="footer-privacy-list">
            <li>Use inputs strictly for visual rendering without tracking profiles.</li>
            <li>Run operational logs under tight retention to debug latency.</li>
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
}) => {
  const contentBlocks = React.Children.toArray(children);
  const insertIndex = Math.ceil(contentBlocks.length / 2);
  const firstHalfBlocks = contentBlocks.slice(0, insertIndex);
  const secondHalfBlocks = contentBlocks.slice(insertIndex);

  return (
    <div className={`info-page-shell ${showLearnSidebar ? 'has-learn-sidebar' : ''}`.trim()}>
      <div className={`info-page-main-layout ${showLearnSidebar ? 'has-learn-sidebar' : ''}`.trim()}>
        {showLearnSidebar && (
          <aside className="learn-sidebar" aria-label="Learn modules">
            <div className="learn-sidebar-inner">
              <h4>Learning Modules</h4>
              <div className="learn-sidebar-links">
                {learnMenuLinks.map((item) => (
                  <button
                    key={item.id}
                    type="button"
                    className={`learn-sidebar-link ${activeLearnId === item.id ? 'is-active' : ''}`.trim()}
                    onClick={() => onNavigate(item.id)}
                  >
                    {item.numberedShortLabel || item.shortLabel || item.label}
                  </button>
                ))}
              </div>
            </div>
          </aside>
        )}

        <main className="page-style">
          {showLearnSidebar && (
            <>
              <h4 className="mobile-learn-module-heading">Learning Modules</h4>
              <nav className="mobile-learn-module-nav" aria-label="Learning modules">
                {learnMenuLinks.map((item) => (
                  <button
                    key={`mobile-${item.id}`}
                    type="button"
                    className={`mobile-learn-module-link ${activeLearnId === item.id ? 'is-active' : ''}`.trim()}
                    onClick={() => onNavigate(item.id)}
                  >
                    {item.numberedShortLabel || item.label}
                  </button>
                ))}
              </nav>
            </>
          )}

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

          {firstHalfBlocks}

          <div className="mobile-inline-ad-slot" aria-label="Advertisement">
            <div className="mobile-inline-ad-content">
              <span>Ad Space</span>
            </div>
          </div>

          {secondHalfBlocks}
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
};

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
  const [isMobileNavOpen, setIsMobileNavOpen] = useState(false);
  const [mobileSimulatorPanel, setMobileSimulatorPanel] = useState(null);
  const sliceOffsetLimit = useMemo(() => Math.max(2, gridSize * 0.95), [gridSize]);
  const slicePlaneExtent = useMemo(() => Math.max(10, gridSize * 2.4), [gridSize]);

  const navigateToPage = (pageId) => {
    if (!PAGE_ROUTE_MAP[pageId]) return;

    setIsMobileNavOpen(false);
    setMobileSimulatorPanel(null);
    setCurrentPage(pageId);

    if (typeof window !== 'undefined') {
      const nextHash = PAGE_ROUTE_MAP[pageId];
      if (window.location.hash !== nextHash) {
        window.location.hash = nextHash;
      }
      window.scrollTo({ top: 0, behavior: 'smooth' });
    }
  };

  const toggleMobileSimulatorPanel = (panelKey) => {
    setMobileSimulatorPanel((prev) => (prev === panelKey ? null : panelKey));
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
    if (typeof window === 'undefined') return undefined;
    const handleResize = () => {
      if (window.innerWidth > 768) {
        setIsMobileNavOpen(false);
        setMobileSimulatorPanel(null);
      }
    };

    window.addEventListener('resize', handleResize);
    return () => window.removeEventListener('resize', handleResize);
  }, []);

  useEffect(() => {
    if (currentPage !== 'simulator') {
      setMobileSimulatorPanel(null);
    }
    setIsMobileNavOpen(false);
  }, [currentPage]);

  useEffect(() => {
    fetchData();
    // eslint-disable-next-line
  }, []); // Initial load

  return (
    <>
      {/* Top Navigation Bar */}
      <div className={`top-nav ${currentPage === 'simulator' ? 'top-nav-simulator' : ''} ${isMobileNavOpen ? 'mobile-nav-open' : ''}`.trim()}>
        <h3>Quantum Orbital Explorer</h3>
        <button
          type="button"
          className="mobile-nav-toggle"
          aria-expanded={isMobileNavOpen}
          aria-controls="top-nav-buttons"
          aria-label={isMobileNavOpen ? 'Close navigation menu' : 'Open navigation menu'}
          onClick={() => setIsMobileNavOpen((prev) => !prev)}
        >
          {isMobileNavOpen ? 'Close' : 'Menu'}
        </button>

        <div id="top-nav-buttons" className="top-nav-buttons">
          <button type="button" onClick={() => navigateToPage('welcome')} style={navButtonStyle(currentPage === 'welcome')}>Welcome</button>
          <button type="button" onClick={() => navigateToPage('chemistry')} style={navButtonStyle(currentPage === 'chemistry')}>Chemistry Concepts</button>
          <button type="button" onClick={() => navigateToPage('howto')} style={navButtonStyle(currentPage === 'howto')}>How To Use</button>

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

          <button type="button" onClick={() => navigateToPage('faqs')} style={navButtonStyle(currentPage === 'faqs')}>FAQs</button>

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

      {learnTopicMap[currentPage] && (
        <LearnTopicPage topic={learnTopicMap[currentPage]} onNavigate={navigateToPage} />
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

      {currentPage === 'about' && (
        <InfoPageLayout
          eyebrow="Information Page"
          title="About Quantum Orbital Explorer"
          message="This page explains the platform mission, scientific design approach, and educational outcomes in a clear professional format."
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
                  Quantum Orbital Explorer is built to close the gap between abstract quantum mechanics and practical interpretation.
                  The platform combines mathematically grounded models, visual simulation, and structured learning content so users can move from equations to confident physical reasoning.
                </p>
              </article>
              <article className="glass-card">
                <h3>Outcome</h3>
                <p>
                  Users progress from symbolic formulas to concrete orbital interpretation by verifying geometry, nodes, phase regions, and radial trends in a single workflow.
                  The intended result is not just visualization literacy, but transferable scientific judgment.
                </p>
              </article>
              <article className="glass-card">
                <h3>Scientific Standard</h3>
                <p>
                  Content is written to remain consistent with hydrogenic quantum mechanics, operator methods, and standard spectroscopy conventions.
                  Visual explanations are intentionally tied back to equations to reduce conceptual drift.
                </p>
              </article>
              <article className="glass-card">
                <h3>Professional Use Cases</h3>
                <p>
                  The platform supports classroom delivery, independent study, concept revision, and pre-lab conceptual preparation.
                  It is designed for educational clarity while preserving technical rigor.
                </p>
              </article>
            </div>
          </InfoSection>

          <InfoSection title="System Overview" subtitle="Core components and responsibilities.">
            <div className="info-grid two-column">
              <article className="glass-card">
                <h3>Interactive Frontend</h3>
                <p>
                  The frontend coordinates parameter control, 3D visualization, graph rendering, and learning navigation.
                  Its primary role is to make high-dimensional quantum behavior inspectable without losing physical context.
                </p>
              </article>
              <article className="glass-card">
                <h3>Computational Backend</h3>
                <p>
                  The backend performs radial and scatter computations, serves cached orbital payloads, and returns numerically stable outputs for rendering.
                  This architecture prioritizes both fidelity and responsive user interaction.
                </p>
              </article>
              <article className="glass-card">
                <h3>Learning Layer</h3>
                <p>
                  The learning layer integrates theory, equations, plotted behavior, and contextual images in one sequence.
                  Its purpose is to turn simulator interaction into disciplined scientific interpretation.
                </p>
              </article>
              <article className="glass-card">
                <h3>Quality Workflow</h3>
                <p>
                  Content and visuals are organized so each claim can be checked against equations, graph trends, and node/phase behavior.
                  This reduces ambiguity and supports reliable explanation quality.
                </p>
              </article>
            </div>
          </InfoSection>

          <InfoSection title="Editorial Positioning" subtitle="How to interpret and use this platform professionally.">
            <article className="glass-card">
              <p>
                Quantum Orbital Explorer is an educational and analytical tool, not a substitute for domain-specific computational chemistry packages used in research-grade many-electron modeling.
                For hydrogenic intuition, conceptual training, and structured interpretation practice, it is intentionally designed to be explicit, transparent, and pedagogically rigorous.
              </p>
              <p>
                The recommended usage pattern is simple: formulate a prediction, test it in the simulator, verify with equations and radial/phase evidence, then document conclusions in clear scientific language.
              </p>
            </article>
          </InfoSection>
        </InfoPageLayout>
      )}

      {currentPage === 'terms' && (
        <InfoPageLayout
          eyebrow="Information Page"
          title="Terms and Conditions"
          message="This page describes acceptable use, scientific scope, responsibility boundaries, and policy governance in plain professional language."
          heroClassName="hero-howto"
          onNavigate={navigateToPage}
          ctaLabel="Back to Welcome"
          ctaTarget="welcome"
        >
          <InfoSection title="Usage Terms" subtitle="Core terms for educational and simulation use.">
            <div className="info-grid two-column">
              <article className="glass-card">
                <h3>Educational and Professional Scope</h3>
                <p>
                  This platform is intended for education, concept development, and guided scientific interpretation.
                  It may support professional preparation workflows, but it is not a legal substitute for specialized research pipelines where full many-body methods are required.
                </p>
              </article>
              <article className="glass-card">
                <h3>User Responsibility</h3>
                <p>
                  Users are responsible for validating critical conclusions before using them in graded, legal, or publication-level contexts.
                  Outputs should be interpreted with domain judgment and, when needed, independent verification.
                </p>
              </article>
              <article className="glass-card">
                <h3>Service Evolution</h3>
                <p>
                  Features, layout, and educational modules may be updated as the platform evolves.
                  Continued use after updates implies acceptance of the current published terms.
                </p>
              </article>
              <article className="glass-card">
                <h3>Media and External Assets</h3>
                <p>
                  Some visuals may reference externally hosted media.
                  Where users upload or publish derivative content, responsibility for rights compliance remains with the user.
                </p>
              </article>
            </div>
          </InfoSection>

          <InfoSection title="Professional Conduct" subtitle="Behavioral and operational expectations.">
            <article className="glass-card">
              <p>
                Use of the service must remain lawful, non-abusive, and consistent with educational intent.
                Attempts to disrupt platform stability, reverse-engineer protected infrastructure, or misuse contact channels are outside acceptable use and may result in restricted access.
              </p>
              <p>
                The platform is provided in good faith for learning and interpretation support.
                While significant care is taken in content quality, no absolute warranty is made for uninterrupted availability or fitness for any specific external workflow.
              </p>
            </article>
          </InfoSection>

          <InfoSection title="Policy Links" subtitle="Dedicated routes for legal and contact pages.">
            <div className="footer-link-list">
              {infoPageLinks.map((link) => (
                <button key={link.id} className="footer-link-button" type="button" onClick={() => navigateToPage(link.id)}>
                  {link.label}
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
          message="This page explains what information is processed, why it is processed, and how retention is managed in clear operational terms."
          heroClassName="hero-howto"
          onNavigate={navigateToPage}
          ctaLabel="Go to Contact"
          ctaTarget="contact"
        >
          <InfoSection title="Data Handling" subtitle="What information is processed and why.">
            <div className="info-grid two-column">
              <article className="glass-card">
                <h3>Simulation Parameters</h3>
                <p>
                  Parameter inputs such as n, l, m, and Z are processed to generate requested orbital results.
                  These inputs are used for computation and response delivery, not for behavioral profiling.
                </p>
              </article>
              <article className="glass-card">
                <h3>Operational Diagnostics</h3>
                <p>
                  Limited operational logs may be used to monitor latency, reliability, and error conditions.
                  Log retention is managed with a minimal-window approach aligned with maintenance requirements.
                </p>
              </article>
              <article className="glass-card">
                <h3>Contact Information</h3>
                <p>
                  If you submit inquiries, provided contact details are used strictly for communication and support follow-up.
                  They are not used for unrelated outreach workflows.
                </p>
              </article>
              <article className="glass-card">
                <h3>Purpose Limitation</h3>
                <p>
                  Processed information is handled for simulation service delivery, platform stability, and user-requested support.
                  Use outside these purposes is not part of normal policy operation.
                </p>
              </article>
            </div>
          </InfoSection>

          <InfoSection title="Retention and User Controls" subtitle="How long data is kept and what options users have.">
            <article className="glass-card">
              <p>
                Simulation request context is generally short-lived and operationally scoped. Diagnostic information is retained only as needed for stability monitoring and service improvement.
                Contact-request context is preserved only for active communication workflows.
              </p>
              <p>
                Users can request clarification about policy interpretation through the contact route.
                Where applicable, updates to policy wording are reflected on this page so current expectations remain explicit and auditable.
              </p>
            </article>
          </InfoSection>

          <InfoSection title="User Control" subtitle="Navigation links to supporting policy pages.">
            <div className="footer-link-list">
              <button className="footer-link-button" type="button" onClick={() => navigateToPage('about')}>About</button>
              <button className="footer-link-button" type="button" onClick={() => navigateToPage('terms')}>Terms and Conditions</button>
              <button className="footer-link-button" type="button" onClick={() => navigateToPage('contact')}>Contact</button>
            </div>
          </InfoSection>
        </InfoPageLayout>
      )}

      {currentPage === 'contact' && (
        <InfoPageLayout
          eyebrow="Information Page"
          title="Contact"
          message="Connect with the Quantum Orbital Explorer team for technical support, educational guidance, and platform policy clarification."
          heroClassName="hero-howto"
          onNavigate={navigateToPage}
          ctaLabel="Open Simulator"
          ctaTarget="simulator"
        >
          <InfoSection title="Primary Contact Channels" subtitle="Use the channel that best matches your request type.">
            <article className="glass-card">
              <ContactChips />
              <p>
                For the fastest response, include your objective (bug report, content clarification, classroom support, or policy question)
                and, where relevant, the exact quantum settings you used in the simulator.
              </p>
            </article>
          </InfoSection>

          <InfoSection title="Response Standards" subtitle="What to expect after you reach out.">
            <div className="info-grid two-column">
              <article className="glass-card">
                <h3>Initial Acknowledgement</h3>
                <p>
                  Messages are acknowledged in the order received. Most routine educational and product questions are acknowledged
                  within one business day, with faster responses when request context is complete.
                </p>
              </article>
              <article className="glass-card">
                <h3>Technical Review</h3>
                <p>
                  If the request involves simulator behavior, the team reproduces the issue using provided parameters,
                  validates expected physics behavior, and prepares clear resolution steps.
                </p>
              </article>
              <article className="glass-card">
                <h3>Resolution and Follow-up</h3>
                <p>
                  You receive a direct explanation, next actions, and escalation guidance if needed.
                  Complex requests may be resolved in staged updates so progress remains transparent.
                </p>
              </article>
              <article className="glass-card">
                <h3>Teaching and Workshop Support</h3>
                <p>
                  Educators can request structured guidance for classroom delivery, including recommended module sequences,
                  interpretation checkpoints, and simulator demonstration flow.
                </p>
              </article>
            </div>
          </InfoSection>

          <InfoSection title="How To Get Fast, Accurate Support" subtitle="Include these details in your first message.">
            <article className="glass-card">
              <ul className="module1-bullet-list">
                <li>Request type: technical issue, concept clarification, policy question, or classroom planning.</li>
                <li>If technical: include n, l, m, Z values, and what behavior you expected versus observed.</li>
                <li>If content-related: include module name, heading title, and the exact line of confusion.</li>
                <li>If policy-related: reference the relevant About, Terms, or Privacy section for precision.</li>
              </ul>
            </article>
          </InfoSection>
        </InfoPageLayout>
      )}

      {currentPage === 'simulator' && (
        <div className={`App simulator-active ${mobileSimulatorPanel ? `mobile-panel-${mobileSimulatorPanel}` : ''}`.trim()}>
          <div className="mobile-sim-toolbar" role="toolbar" aria-label="Simulator mobile panels">
            <button
              type="button"
              className={`mobile-sim-toolbar-btn ${mobileSimulatorPanel === 'controls' ? 'is-active' : ''}`.trim()}
              onClick={() => toggleMobileSimulatorPanel('controls')}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <circle cx="12" cy="12" r="3"></circle>
                <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09a1.65 1.65 0 0 0-1-1.51 1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06a1.65 1.65 0 0 0 .33-1.82 1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09a1.65 1.65 0 0 0 1.51-1 1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06a1.65 1.65 0 0 0 1.82.33h.1a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82v.1a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z"></path>
              </svg>
              Settings
            </button>
            <button
              type="button"
              className={`mobile-sim-toolbar-btn ${mobileSimulatorPanel === 'graph' ? 'is-active' : ''}`.trim()}
              onClick={() => toggleMobileSimulatorPanel('graph')}
            >
              Graph
            </button>
          </div>
          
          {/* Sidebar Controls */}
      <div className="controls-sidebar">
        <button
          type="button"
          className="mobile-panel-close mobile-panel-close-left"
          onClick={() => setMobileSimulatorPanel(null)}
          aria-label="Hide settings panel"
        >
          &lt;
        </button>
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
            <button
              type="button"
              className="mobile-panel-close mobile-panel-close-down"
              onClick={() => setMobileSimulatorPanel(null)}
              aria-label="Hide graph panel"
            >
              v
            </button>
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



