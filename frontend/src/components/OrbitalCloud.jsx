import React, { useMemo, useRef } from 'react';
import { Billboard, Text } from '@react-three/drei';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { getRenderPointBudget } from '../utils/science.js';

const CENTER = new THREE.Vector3(0, 0, 0);
const DENSITY_LOW = new THREE.Color('#23313a');
const DENSITY_MID = new THREE.Color('#0b8f87');
const DENSITY_HIGH = new THREE.Color('#ff6a3d');
const PHASE_POSITIVE = new THREE.Color('#e6422e');
const PHASE_NEGATIVE = new THREE.Color('#5b3cc4');

function makeCircleTexture() {
  const canvas = document.createElement('canvas');
  canvas.width = 48;
  canvas.height = 48;
  const context = canvas.getContext('2d');
  const gradient = context.createRadialGradient(24, 24, 1, 24, 24, 23);
  gradient.addColorStop(0, 'rgba(255,255,255,1)');
  gradient.addColorStop(0.62, 'rgba(255,255,255,0.95)');
  gradient.addColorStop(1, 'rgba(255,255,255,0)');
  context.fillStyle = gradient;
  context.beginPath();
  context.arc(24, 24, 23, 0, Math.PI * 2);
  context.fill();
  const texture = new THREE.CanvasTexture(canvas);
  texture.generateMipmaps = false;
  texture.minFilter = THREE.LinearFilter;
  texture.magFilter = THREE.LinearFilter;
  return texture;
}

function getSlicePlane(axis, offset = 0) {
  const safeOffset = Number.isFinite(offset) ? offset : 0;
  if (axis === 'x') return new THREE.Plane(new THREE.Vector3(-1, 0, 0), safeOffset);
  if (axis === 'y') return new THREE.Plane(new THREE.Vector3(0, -1, 0), safeOffset);
  return new THREE.Plane(new THREE.Vector3(0, 0, -1), safeOffset);
}

function projectedRadiusPx(camera, viewportHeight, orbitalRadius) {
  if (!Number.isFinite(orbitalRadius) || orbitalRadius <= 0 || viewportHeight <= 0) return 0;

  if (camera?.isPerspectiveCamera) {
    const distance = Math.max(0.01, camera.position.distanceTo(CENTER));
    const fovRad = THREE.MathUtils.degToRad(camera.fov || 60);
    const worldToPixels = viewportHeight / (2 * Math.tan(fovRad / 2) * distance);
    return orbitalRadius * worldToPixels;
  }

  if (camera?.isOrthographicCamera) {
    const worldHeight = Math.max(0.01, camera.top - camera.bottom);
    return orbitalRadius * (viewportHeight / worldHeight);
  }

  return 0;
}

export function OrbitalPoints({
  data,
  n,
  showPhase,
  luminous,
  sliceEnabled,
  selectedSliceAxes,
  sliceOffsets,
}) {
  const materialRef = useRef(null);
  const sprite = useMemo(() => makeCircleTexture(), []);
  const clippingPlanes = useMemo(() => {
    if (!sliceEnabled || selectedSliceAxes.length === 0) return [];
    return selectedSliceAxes.map((axis) => getSlicePlane(axis, sliceOffsets[axis] ?? 0));
  }, [selectedSliceAxes, sliceEnabled, sliceOffsets]);

  const prepared = useMemo(() => {
    const hasPoints = data?.pointsFlat instanceof Float32Array && data.pointsFlat.length > 0;
    if (!hasPoints) return null;

    const pointStride = Math.max(5, Number(data.pointStride) || 5);
    const sourceCount = Math.floor(data.pointsFlat.length / pointStride);
    const sampleStep = Math.max(1, Math.ceil(sourceCount / Math.max(250_000, getRenderPointBudget(n))));
    const renderCount = Math.ceil(sourceCount / sampleStep);
    const positions = new Float32Array(renderCount * 3);
    const colors = new Float32Array(renderCount * 3);
    let maxRadius = 0;

    for (let sourceIndex = 0, renderIndex = 0; sourceIndex < sourceCount; sourceIndex += sampleStep, renderIndex += 1) {
      const base = sourceIndex * pointStride;
      const x = data.pointsFlat[base];
      const y = data.pointsFlat[base + 1];
      const z = data.pointsFlat[base + 2];
      const density = THREE.MathUtils.clamp(data.pointsFlat[base + 3] || 0, 0, 1);
      const phase = data.pointsFlat[base + 4] || 1;
      const write = renderIndex * 3;
      const radius = Math.sqrt(x * x + y * y + z * z);
      if (radius > maxRadius) maxRadius = radius;

      positions[write] = x;
      positions[write + 1] = y;
      positions[write + 2] = z;

      const color = new THREE.Color();
      if (showPhase) {
        color.copy(phase >= 0 ? PHASE_POSITIVE : PHASE_NEGATIVE);
        const intensity = Math.max(0.28, density * 1.35) * Math.max(0.42, 40 / (radius + 40));
        color.multiplyScalar(intensity);
      } else if (density < 0.5) {
        color.copy(DENSITY_LOW).lerp(DENSITY_MID, density * 2);
      } else {
        color.copy(DENSITY_MID).lerp(DENSITY_HIGH, (density - 0.5) * 2);
      }

      colors[write] = color.r;
      colors[write + 1] = color.g;
      colors[write + 2] = color.b;
    }

    return { positions, colors, maxRadius, renderCount };
  }, [data, n, showPhase]);

  useFrame((state) => {
    if (!materialRef.current || !prepared) return;
    const radiusPixels = projectedRadiusPx(state.camera, state.size.height, prepared.maxRadius);
    const minSize = n > 5 ? 1.75 : 1.4;
    materialRef.current.size = THREE.MathUtils.clamp(radiusPixels * 0.002 + (n > 5 ? 0.2 : 0), minSize, 7.5);
  });

  if (!prepared) return null;

  return (
    <points frustumCulled={false}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          array={prepared.positions}
          count={prepared.positions.length / 3}
          itemSize={3}
          usage={THREE.StaticDrawUsage}
        />
        <bufferAttribute
          attach="attributes-color"
          array={prepared.colors}
          count={prepared.colors.length / 3}
          itemSize={3}
          usage={THREE.StaticDrawUsage}
        />
      </bufferGeometry>
      <pointsMaterial
        ref={materialRef}
        size={n > 5 ? 1.75 : 1.4}
        map={sprite}
        alphaTest={0.1}
        vertexColors
        transparent={luminous}
        opacity={luminous ? 0.88 : 1}
        sizeAttenuation={false}
        clippingPlanes={clippingPlanes}
        blending={luminous ? THREE.AdditiveBlending : THREE.NormalBlending}
        depthWrite={!luminous}
        depthTest
        toneMapped={false}
      />
    </points>
  );
}

export function SlicePlane({ axis, offset, extent }) {
  const safeExtent = Math.max(10, extent);
  const transform = useMemo(() => {
    if (axis === 'x') return { position: [offset, 0, 0], rotation: [0, Math.PI / 2, 0] };
    if (axis === 'y') return { position: [0, offset, 0], rotation: [Math.PI / 2, 0, 0] };
    return { position: [0, 0, offset], rotation: [0, 0, 0] };
  }, [axis, offset]);

  return (
    <mesh position={transform.position} rotation={transform.rotation}>
      <planeGeometry args={[safeExtent, safeExtent]} />
      <meshBasicMaterial color="#e6422e" transparent opacity={0.12} side={THREE.DoubleSide} depthWrite={false} toneMapped={false} />
    </mesh>
  );
}

export function ScientificAxes({ size }) {
  const lineScale = size * 1.35;
  const labelSize = THREE.MathUtils.clamp(size * 0.018, 1.1, 2.2);
  const axes = useMemo(
    () => ({
      x: new Float32Array([-lineScale, 0, 0, lineScale, 0, 0]),
      y: new Float32Array([0, -lineScale, 0, 0, lineScale, 0]),
      z: new Float32Array([0, 0, -lineScale, 0, 0, lineScale]),
    }),
    [lineScale],
  );

  return (
    <group>
      <line>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" count={2} array={axes.x} itemSize={3} />
        </bufferGeometry>
        <lineBasicMaterial color="#e6422e" transparent opacity={0.58} />
      </line>
      <line>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" count={2} array={axes.y} itemSize={3} />
        </bufferGeometry>
        <lineBasicMaterial color="#7aa600" transparent opacity={0.58} />
      </line>
      <line>
        <bufferGeometry>
          <bufferAttribute attach="attributes-position" count={2} array={axes.z} itemSize={3} />
        </bufferGeometry>
        <lineBasicMaterial color="#5b3cc4" transparent opacity={0.58} />
      </line>
      {[
        ['X', [lineScale + 2, 0, 0], '#e6422e'],
        ['Y', [0, lineScale + 2, 0], '#7aa600'],
        ['Z', [0, 0, lineScale + 2], '#5b3cc4'],
      ].map(([label, position, color]) => (
        <Billboard key={label} position={position}>
          <Text color={color} fontSize={labelSize} anchorX="center" anchorY="middle">
            {label}
          </Text>
        </Billboard>
      ))}
    </group>
  );
}

export function Nucleus() {
  return (
    <mesh>
      <sphereGeometry args={[1.3, 32, 32]} />
      <meshStandardMaterial color="#f6c84c" emissive="#f6c84c" emissiveIntensity={0.75} roughness={0.38} />
    </mesh>
  );
}
