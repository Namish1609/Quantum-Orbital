import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Canvas } from '@react-three/fiber';
import { OrbitControls } from '@react-three/drei';
import { InlineMath } from 'react-katex';
import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import {
  Activity,
  Box,
  Braces,
  Eye,
  Gauge,
  Loader2,
  Maximize2,
  Microscope,
  Minimize2,
  RefreshCw,
  RotateCcw,
  Scissors,
  Sparkles,
  Zap,
} from 'lucide-react';
import { OrbitalPoints, ScientificAxes } from './OrbitalCloud.jsx';
import { fetchRadialData, fetchScatterData } from '../utils/api.js';
import {
  FIXED_GRID_SIZE,
  FIXED_Z,
  MAX_SLICE_AXES,
  SLICE_AXIS_ORDER,
  clampNumber,
  energyEv,
  familyForL,
  formatCompactNumber,
  getCanvasDpr,
  getPointCountForN,
  nodeSummary,
  normalizeQuantumState,
  orbitalName,
  radialPeak,
} from '../utils/science.js';

const chartTooltip = {
  background: 'var(--tooltip-bg)',
  border: '1px solid var(--line)',
  borderRadius: 8,
  color: 'var(--ink)',
  boxShadow: 'var(--shadow-soft)',
};

function InlineFormula({ math }) {
  return (
    <span className="math-symbol">
      <InlineMath math={math} />
    </span>
  );
}

function ControlSection({ icon: Icon, label, children }) {
  return (
    <section className="control-section">
      <div className="control-section-title">
        <Icon size={16} aria-hidden="true" />
        <span>{label}</span>
      </div>
      {children}
    </section>
  );
};

function Metric({ label, value, tone = 'neutral' }) {
  return (
    <div className={`metric-cell ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function RangeControl({ label, value, min, max, step = 1, onChange, descriptor }) {
  return (
    <label className="range-row">
      <span>
        {label}
        <strong>{value}</strong>
      </span>
      <input
        type="range"
        min={min}
        max={max}
        step={step}
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
      />
      {descriptor && <em>{descriptor}</em>}
    </label>
  );
}

function ToggleButton({ active, icon: Icon, label, onClick }) {
  return (
    <button className={`toggle-chip ${active ? 'active' : ''}`} type="button" onClick={onClick} aria-pressed={active}>
      <Icon size={15} aria-hidden="true" />
      {label}
    </button>
  );
}

function AxisSelector({ selectedAxes, onToggle }) {
  return (
    <div className="axis-selector" role="group" aria-label="Slice axes">
      {SLICE_AXIS_ORDER.map((axis) => (
        <button
          key={axis}
          type="button"
          className={selectedAxes.includes(axis) ? 'active' : ''}
          onClick={() => onToggle(axis)}
        >
          {axis.toUpperCase()}
        </button>
      ))}
    </div>
  );
}

function Legend({ showPhase }) {
  if (showPhase) {
    return (
      <div className="legend-strip">
        <span><i className="phase-dot positive" /> positive phase</span>
        <span><i className="phase-dot negative" /> negative phase</span>
      </div>
    );
  }

  return (
    <div className="density-legend" aria-label="Density legend">
      <span>low</span>
      <div />
      <span>high</span>
    </div>
  );
}

function RadialChart({ data, large = false }) {
  if (!Array.isArray(data) || data.length === 0) {
    return (
      <div className={`graph-empty-state ${large ? 'large' : ''}`.trim()}>
        Waiting for radial dataset
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%" minWidth={1} minHeight={1}>
      <LineChart data={data} margin={large ? { top: 18, right: 28, bottom: 20, left: 14 } : { top: 14, right: 18, bottom: 14, left: 6 }}>
        <CartesianGrid stroke="var(--chart-grid)" strokeDasharray="4 6" />
        <XAxis dataKey="r" tick={{ fill: 'var(--ink-muted)', fontSize: large ? 12 : 11 }} tickMargin={8} tickFormatter={(value) => Number(value).toFixed(0)} />
        <YAxis width={large ? 54 : 44} tick={{ fill: 'var(--ink-muted)', fontSize: large ? 12 : 11 }} tickFormatter={(value) => Number(value).toFixed(2)} />
        <Tooltip contentStyle={chartTooltip} formatter={(value) => Number(value).toFixed(4)} labelFormatter={(value) => `r = ${Number(value).toFixed(2)}`} />
        <Line type="monotone" dataKey="P" stroke="var(--teal-bright)" strokeWidth={large ? 3 : 2.5} dot={false} isAnimationActive animationDuration={360} />
      </LineChart>
    </ResponsiveContainer>
  );
}

export default function Simulator() {
  const [n, setN] = useState(3);
  const [l, setL] = useState(2);
  const [m, setM] = useState(0);
  const [showPhase, setShowPhase] = useState(true);
  const [luminous, setLuminous] = useState(false);
  const [showAxes, setShowAxes] = useState(true);
  const [sliceEnabled, setSliceEnabled] = useState(false);
  const [selectedSliceAxes, setSelectedSliceAxes] = useState(['x']);
  const [sliceOffsets, setSliceOffsets] = useState({ x: 0, y: 0, z: 0 });
  const [activeSliceAxis, setActiveSliceAxis] = useState(null);
  const [plotData, setPlotData] = useState(null);
  const [renderKey, setRenderKey] = useState('');
  const [radialData, setRadialData] = useState([]);
  const [autoApply, setAutoApply] = useState(false);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [graphExpanded, setGraphExpanded] = useState(false);
  const [mobilePanel, setMobilePanel] = useState(null);
  const requestIdRef = useRef(0);

  const state = useMemo(() => normalizeQuantumState({ n, l, m }), [n, l, m]);
  const pointsRequested = useMemo(() => getPointCountForN(state.n), [state.n]);
  const nodes = useMemo(() => nodeSummary(state), [state]);
  const peak = useMemo(() => radialPeak(radialData), [radialData]);
  const canvasDpr = useMemo(() => getCanvasDpr(pointsRequested), [pointsRequested]);
  const sliceLimit = useMemo(() => Math.max(2, FIXED_GRID_SIZE * 0.95), []);

  useEffect(() => {
    if (l >= n) setL(n - 1);
  }, [l, n]);

  useEffect(() => {
    if (Math.abs(m) > l) setM(0);
  }, [l, m]);

  useEffect(() => {
    if (activeSliceAxis && !selectedSliceAxes.includes(activeSliceAxis)) {
      setActiveSliceAxis(null);
    }
  }, [activeSliceAxis, selectedSliceAxes]);

  const runSimulation = useCallback(async () => {
    const safeState = normalizeQuantumState({ n, l, m });
    setN(safeState.n);
    setL(safeState.l);
    setM(safeState.m);
    setLoading(true);
    setError('');
    const requestId = requestIdRef.current + 1;
    requestIdRef.current = requestId;

    try {
      const [radial, scatter] = await Promise.all([
        fetchRadialData({ n: safeState.n, l: safeState.l, size: FIXED_GRID_SIZE }),
        fetchScatterData({ n: safeState.n, l: safeState.l, m: safeState.m, size: FIXED_GRID_SIZE }),
      ]);
      if (requestId !== requestIdRef.current) return;
      setRadialData(radial.data);
      setPlotData(scatter);
      setRenderKey(`${safeState.n}:${safeState.l}:${safeState.m}:${scatter.count}:${Date.now()}`);
    } catch (caught) {
      if (requestId !== requestIdRef.current) return;
      setError(caught instanceof Error ? caught.message : 'Simulation request failed.');
    } finally {
      if (requestId === requestIdRef.current) {
        setLoading(false);
      }
    }
  }, [l, m, n]);

  useEffect(() => {
    runSimulation();
  }, []);

  useEffect(() => {
    if (!autoApply) return undefined;
    const timer = window.setTimeout(() => {
      runSimulation();
    }, 620);
    return () => window.clearTimeout(timer);
  }, [autoApply, runSimulation]);

  const resetState = () => {
    setN(3);
    setL(2);
    setM(0);
    setShowPhase(true);
    setLuminous(false);
    setShowAxes(true);
    setSliceEnabled(false);
    setSelectedSliceAxes(['x']);
    setSliceOffsets({ x: 0, y: 0, z: 0 });
  };

  const toggleSliceAxis = (axis) => {
    setSelectedSliceAxes((previous) => {
      if (previous.includes(axis)) {
        if (previous.length === 1) return previous;
        return previous.filter((item) => item !== axis);
      }
      if (previous.length >= MAX_SLICE_AXES) return previous;
      return [...previous, axis].sort((a, b) => SLICE_AXIS_ORDER.indexOf(a) - SLICE_AXIS_ORDER.indexOf(b));
    });
  };

  const updateSliceOffset = (axis, value) => {
    setSliceOffsets((previous) => ({
      ...previous,
      [axis]: clampNumber(value, -sliceLimit, sliceLimit),
    }));
  };

  const toggleGraphExpanded = () => {
    setGraphExpanded((value) => {
      const nextValue = !value;
      if (nextValue) setMobilePanel(null);
      return nextValue;
    });
  };

  return (
    <main className={`lab-route ${mobilePanel ? `mobile-${mobilePanel}` : ''}`}>
      <div className="mobile-lab-switcher" role="toolbar" aria-label="Simulator panels">
        <button type="button" className={mobilePanel === 'controls' ? 'active' : ''} onClick={() => setMobilePanel((value) => (value === 'controls' ? null : 'controls'))}>
          <Microscope size={15} />
          Controls
        </button>
        <button type="button" className={graphExpanded ? 'active' : ''} onClick={toggleGraphExpanded} aria-pressed={graphExpanded}>
          <Activity size={15} />
          Graph
        </button>
      </div>

      <aside className="lab-sidebar">
        <div className="sidebar-heading">
          <Microscope size={22} />
          <div>
            <p>orbital state</p>
            <h1>{orbitalName(state)}</h1>
          </div>
        </div>

        <div className="metric-grid">
          <Metric label="family" value={familyForL(state.l)} tone="teal" />
          <Metric label="nodes" value={nodes.total} tone="coral" />
          <Metric label="points" value={formatCompactNumber(pointsRequested)} tone="green" />
          <Metric label="energy" value={`${energyEv(state.n).toFixed(2)} eV`} />
        </div>

        <form className="control-stack" onSubmit={(event) => { event.preventDefault(); runSimulation(); }}>
          <ControlSection icon={Gauge} label="quantum controls">
            <RangeControl label={<>Principal <InlineFormula math="n" /></>} value={state.n} min={1} max={10} onChange={setN} descriptor="shell and total nodes" />
            <RangeControl label={<>Azimuthal <InlineFormula math="l" /></>} value={state.l} min={0} max={Math.max(0, state.n - 1)} onChange={setL} descriptor={`${familyForL(state.l)} family`} />
            <RangeControl label={<>Magnetic <InlineFormula math="m" /></>} value={state.m} min={-state.l} max={state.l} onChange={setM} descriptor="orientation state" />
            <div className="fixed-row">
              <span>Nuclear charge <InlineFormula math="Z" /></span>
              <strong>{FIXED_Z}</strong>
            </div>
          </ControlSection>

          <button
            className={`live-recompute ${autoApply ? 'active' : ''}`}
            type="button"
            onClick={() => setAutoApply((value) => !value)}
            aria-pressed={autoApply}
          >
            <Zap size={16} aria-hidden="true" />
            <span>
              <strong>Live recompute</strong>
              <em>{autoApply ? 'Slider changes refresh automatically' : 'Manual apply keeps heavy states calm'}</em>
            </span>
            <i aria-hidden="true" />
          </button>

          <div className="toggle-grid">
            <ToggleButton active={showPhase} icon={Eye} label="Phase" onClick={() => setShowPhase((value) => !value)} />
            <ToggleButton active={luminous} icon={Sparkles} label="Glow" onClick={() => setLuminous((value) => !value)} />
            <ToggleButton active={showAxes} icon={Box} label="Axes" onClick={() => setShowAxes((value) => !value)} />
            <ToggleButton active={sliceEnabled} icon={Scissors} label="Slice" onClick={() => setSliceEnabled((value) => !value)} />
          </div>

          {sliceEnabled && (
            <div className="slice-controls">
              <AxisSelector selectedAxes={selectedSliceAxes} onToggle={toggleSliceAxis} />
              {selectedSliceAxes.map((axis) => (
                <RangeControl
                  key={axis}
                  label={`${axis.toUpperCase()} offset`}
                  value={Number(sliceOffsets[axis] ?? 0).toFixed(1)}
                  min={-sliceLimit}
                  max={sliceLimit}
                  step={0.1}
                  onChange={(value) => updateSliceOffset(axis, value)}
                  descriptor="clip plane position"
                />
              ))}
            </div>
          )}

          <div className="simulation-actions">
            <button className="reset-button" type="button" onClick={resetState}>
              <RotateCcw size={16} />
              Reset
            </button>
            <button className="run-button" type="submit" disabled={loading}>
              {loading ? <Loader2 className="spin" size={17} /> : <RefreshCw size={17} />}
              {loading ? 'Computing' : 'Apply state'}
            </button>
          </div>
        </form>

        {error && (
          <div className="error-panel" role="alert">
            {error}
          </div>
        )}

      </aside>

      <section className={`lab-stage ${graphExpanded ? 'graph-expanded' : ''}`} aria-label={graphExpanded ? 'Expanded radial probability graph' : '3D orbital renderer'}>
        <div className="stage-toolbar">
          <div>
            <p>{graphExpanded ? 'radial probability expanded' : 'hydrogenic wavefunction density'}</p>
            <h2>{graphExpanded ? <><InlineFormula math="P(r)" /> distribution</> : `${state.n}${familyForL(state.l)} orbital field`}</h2>
          </div>
          <div className="stage-tools">
            {graphExpanded ? (
              <button className="graph-expand-button active" type="button" onClick={() => setGraphExpanded(false)}>
                <Minimize2 size={15} aria-hidden="true" />
                Show orbital
              </button>
            ) : (
              <>
                <Legend showPhase={showPhase} />
                <button className="stage-graph-button" type="button" onClick={toggleGraphExpanded}>
                  <Activity size={15} aria-hidden="true" />
                  Graph analysis
                  <Maximize2 size={14} aria-hidden="true" />
                </button>
              </>
            )}
          </div>
        </div>

        {graphExpanded ? (
          <div className="expanded-graph-stage">
            <div className="expanded-graph-canvas">
              <RadialChart data={radialData} large />
            </div>
            <div className="expanded-graph-footer">
              <Metric label="radial peak" value={peak ? Number(peak.r).toFixed(2) : '-'} tone="teal" />
              <Metric label="radial nodes" value={nodes.radial} tone="green" />
              <Metric label="angular nodes" value={nodes.angular} tone="coral" />
            </div>
          </div>
        ) : (
          <>
            <div className="canvas-stage">
              <Canvas
                dpr={canvasDpr}
                camera={{ position: [0, 0, FIXED_GRID_SIZE * 1.45], fov: 58 }}
                gl={{ antialias: true, alpha: false, powerPreference: 'high-performance' }}
                onCreated={({ gl }) => {
                  gl.localClippingEnabled = true;
                }}
              >
                <color attach="background" args={['#101316']} />
                <fog attach="fog" args={['#101316', 360, 760]} />
                <ambientLight intensity={0.68} />
                <pointLight position={[120, 120, 80]} intensity={1.1} />
                <pointLight position={[-120, -80, -120]} intensity={0.5} color="#8fd5ce" />
                {plotData && (
                  <OrbitalPoints
                    key={renderKey}
                    data={plotData}
                    n={state.n}
                    showPhase={showPhase}
                    luminous={luminous}
                    sliceEnabled={sliceEnabled}
                    selectedSliceAxes={selectedSliceAxes}
                    sliceOffsets={sliceOffsets}
                  />
                )}
                {showAxes && <ScientificAxes size={FIXED_GRID_SIZE} />}
                <OrbitControls makeDefault enablePan enableZoom enableRotate autoRotate autoRotateSpeed={0.45} panSpeed={1.25} />
              </Canvas>
              <div className="stage-hud" aria-hidden="true">
                <span><InlineFormula math="n" /> {state.n}</span>
                <span><InlineFormula math="l" /> {state.l}</span>
                <span><InlineFormula math="m" /> {state.m}</span>
                <span>{nodes.total} nodes</span>
              </div>
              {!plotData && !loading && (
                <div className="stage-empty">
                  <Gauge size={22} />
                  <p>Start the backend, then apply a state to load orbital samples.</p>
                </div>
              )}
            </div>

            <div className="stage-footer">
              <span><Braces size={15} /> Rotate, pan, and zoom directly in the field.</span>
              <span>{plotData ? `${formatCompactNumber(plotData.count)} loaded samples (${plotData.source})` : 'waiting for orbital payload'}</span>
            </div>
          </>
        )}
      </section>
    </main>
  );
}
