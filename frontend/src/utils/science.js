export const FIXED_Z = 1;
export const FIXED_GRID_SIZE = 200;
export const POINTS_PER_N_STEP = 5_000_000;
export const MAX_TOTAL_POINTS = 50_000_000;
export const MAX_RENDER_POINTS_DEFAULT = 1_800_000;
export const MAX_RENDER_POINTS_HIGH_N = 1_200_000;
export const MAX_SLICE_AXES = 2;
export const SLICE_AXIS_ORDER = ['x', 'y', 'z'];

export const familyNames = ['s', 'p', 'd', 'f', 'g', 'h', 'i', 'k', 'l', 'm'];

export function familyForL(lValue) {
  return familyNames[lValue] || `l=${lValue}`;
}

export function getPointCountForN(nValue) {
  const scaled = Math.max(1, Number(nValue) || 1) * POINTS_PER_N_STEP;
  return Math.min(MAX_TOTAL_POINTS, scaled);
}

export function getRenderPointBudget(nValue) {
  const cores = typeof navigator !== 'undefined' ? navigator.hardwareConcurrency || 8 : 8;
  const baseBudget = nValue >= 8 ? MAX_RENDER_POINTS_HIGH_N : MAX_RENDER_POINTS_DEFAULT;
  return cores <= 4 ? Math.floor(baseBudget * 0.7) : baseBudget;
}

export function clampNumber(value, min, max) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return min;
  return Math.max(min, Math.min(max, numeric));
}

export function normalizeQuantumState({ n, l, m }) {
  const safeN = clampNumber(Math.round(n), 1, 10);
  const safeL = clampNumber(Math.round(l), 0, safeN - 1);
  const safeM = clampNumber(Math.round(m), -safeL, safeL);
  return { n: safeN, l: safeL, m: safeM };
}

export function orbitalName({ n, l, m }) {
  return `${n}${familyForL(l)} (m=${m})`;
}

export function nodeSummary({ n, l }) {
  const total = Math.max(0, n - 1);
  const angular = Math.max(0, l);
  const radial = Math.max(0, n - l - 1);
  return { total, angular, radial };
}

export function energyEv(n, z = FIXED_Z) {
  if (!Number.isFinite(n) || n <= 0) return 0;
  return -13.6 * z * z / (n * n);
}

export function formatCompactNumber(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) return '0';
  if (numeric >= 1_000_000) return `${(numeric / 1_000_000).toFixed(1)}M`;
  if (numeric >= 1_000) return `${(numeric / 1_000).toFixed(1)}K`;
  return `${numeric}`;
}

export function getCanvasDpr(numPoints) {
  const deviceDpr = typeof window !== 'undefined' ? window.devicePixelRatio || 1 : 1;
  const cap = numPoints >= 30_000_000 ? 1.0 : numPoints >= 10_000_000 ? 1.15 : numPoints >= 5_000_000 ? 1.25 : 1.35;
  const min = numPoints >= 30_000_000 ? 0.75 : 1;
  return [min, Math.min(deviceDpr, cap)];
}

export function radialPeak(radialData) {
  if (!Array.isArray(radialData) || radialData.length === 0) return null;
  return radialData.reduce((best, point) => {
    if (!best || Number(point.P) > Number(best.P)) return point;
    return best;
  }, null);
}
