import { FIXED_Z, FIXED_GRID_SIZE } from './science.js';

const API_BASE_URL = import.meta.env.DEV ? 'http://127.0.0.1:8000' : '';

async function parseJsonResponse(response, endpointName) {
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

    const suffix = response.statusText ? ` ${response.statusText}` : '';
    throw new Error(detail || `${endpointName} request failed (${response.status}${suffix}).`);
  }

  if (!raw) throw new Error(`${endpointName} returned an empty response.`);

  try {
    return JSON.parse(raw);
  } catch {
    throw new Error(`${endpointName} returned invalid JSON.`);
  }
}

export async function fetchRadialData({ n, l, size = FIXED_GRID_SIZE }) {
  const url = `${API_BASE_URL}/radial?Z=${FIXED_Z}&n=${n}&l=${l}&size=${size}`;
  const response = await fetch(url);
  const payload = await parseJsonResponse(response, 'Radial API');

  if (!Array.isArray(payload.data)) {
    throw new Error('Radial API response did not include a data array.');
  }

  return payload;
}

export async function fetchScatterData({ n, l, m, size = FIXED_GRID_SIZE }) {
  const params = new URLSearchParams({
    Z: String(FIXED_Z),
    n: String(n),
    l: String(l),
    m: String(m),
    size: String(size),
    binary: 'true',
  });
  const response = await fetch(`${API_BASE_URL}/scatter?${params.toString()}`);

  if (!response.ok) {
    await parseJsonResponse(response, 'Scatter API');
  }

  const stride = Math.max(5, Number(response.headers.get('x-point-stride')) || 5);
  const source = response.headers.get('x-orbital-source') || 'computed';
  const arrayBuffer = await response.arrayBuffer();
  const pointsFlat = new Float32Array(arrayBuffer);

  if (pointsFlat.length % stride !== 0) {
    throw new Error('Scatter API returned malformed binary point data.');
  }

  return {
    pointsFlat,
    pointStride: stride,
    source,
    count: pointsFlat.length / stride,
  };
}
