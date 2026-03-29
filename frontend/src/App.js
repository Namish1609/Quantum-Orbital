import React, { useState, useEffect, useMemo } from 'react';
import { Canvas } from '@react-three/fiber';
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

// Create a synchronous circular texture for the points to match Plotly's markers
const createCircleTexture = () => {
    const canvas = document.createElement('canvas');
    canvas.width = 32;
    canvas.height = 32;
    const ctx = canvas.getContext('2d');
    ctx.beginPath();
    ctx.arc(16, 16, 15, 0, 2 * Math.PI);
    ctx.fillStyle = 'white';
    ctx.fill();
    return new THREE.CanvasTexture(canvas);
};
const circleTexture = createCircleTexture();

// --- Reusable Three.js Components ---

const ScatterPlot = ({ data, pointSize, opacity, showPhase }) => {
  const { positions, colors } = useMemo(() => {
    if (!data || !data.points || data.points.length === 0) return { positions: null, colors: null };
    
    const count = data.points.length;
    const pos = new Float32Array(count * 3);
    const col = new Float32Array(count * 3);

    const phasePosColor = new THREE.Color("red");
    const phaseNegColor = new THREE.Color("#3333ff");

    for (let i = 0; i < count; i++) {
        const [x, y, z, density, phase] = data.points[i];
        
        pos[i * 3 + 0] = x;
        pos[i * 3 + 1] = y;
        pos[i * 3 + 2] = z;

        let color = new THREE.Color();
        if (showPhase) {
            color.copy(phase > 0 ? phasePosColor : phaseNegColor);
            color.multiplyScalar(Math.max(0.2, density * 3)); // enhance brightness locally without going full black
        } else {
            const c1 = new THREE.Color("#30005c"); // Dark Purple (Low Density)
            const c2 = new THREE.Color("#c51b7d"); // Magenta/Purple (Mid Density)
            const c3 = new THREE.Color("#ff8c00"); // Orange (High Density)
            
            if (density < 0.5) {
                color.copy(c1).lerp(c2, density * 2.0);
            } else {
                color.copy(c2).lerp(c3, (density - 0.5) * 2.0);
            }
        }

        col[i * 3 + 0] = color.r;
        col[i * 3 + 1] = color.g;
        col[i * 3 + 2] = color.b;
    }
    return { positions: pos, colors: col };
  }, [data, showPhase]);

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
      <pointsMaterial
        size={pointSize}
        vertexColors={true}
        transparent={true}
        opacity={opacity}
        sizeAttenuation={true}
        map={circleTexture}
        alphaTest={0.05}
        depthWrite={false}
      />
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
  const [gridSize, setGridSize] = useState(100);
  
  // Scatter Controls
  const [scatterGridRes, setScatterGridRes] = useState(120);
  const [numPoints, setNumPoints] = useState(500000);
  const [pointSize, setPointSize] = useState(0.10);
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

  const API_BASE_URL = process.env.NODE_ENV === 'development' ? 'http://localhost:8000' : '';

  const fetchData = async () => {
    setLoading(true);
    setErrorMsg("");
    
    try {
      // Fetch Radial Data
      const radialRes = await fetch(`${API_BASE_URL}/radial?Z=${Z}&n=${n}&l=${l}&size=${gridSize}`);
      const radialJson = await radialRes.json();
      
      if (radialJson.data) {
        setRadialData(radialJson.data);
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
        endpoint = '/scatter';
      } else {
        params.append('resolution', isoGridRes);
        params.append('isovalue', isovalue);
        endpoint = '/isosurface';
      }

      const res3d = await fetch(`${API_BASE_URL}${endpoint}?${params.toString()}`);
      if (!res3d.ok) {
         const d = await res3d.json();
         const errorDetail = typeof d.detail === 'string' ? d.detail : JSON.stringify(d.detail);
         throw new Error(errorDetail || "API failed");
      }
      const data3d = await res3d.json();
      
      // Warn if arrays are empty 
      if (mode === 'isosurface' && (!data3d.surfaces || data3d.surfaces.length === 0)) {
          setErrorMsg("No Isosurface found at this Threshold! Try lowering the Isovalue.");
      }
      
      setPlotData(data3d);
      setRenderKey(Date.now()); // Update the unique key to force remount the geometries

    } catch (err) {
      console.error(err);
      setErrorMsg(err.message);
    } finally {
      setLoading(false);
    }
  };

  const handleUpdate = (e) => {
    e.preventDefault();
    fetchData();
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
    fetchData();
    // eslint-disable-next-line
  }, []); // Initial load

  return (
    <>
      {/* Top Navigation Bar */}
      <div style={{ display: 'flex', backgroundColor: '#333', padding: '10px 20px', gap: '20px', borderBottom: '2px solid #555' }}>
        <h3 style={{ margin: 0, color: '#00aaff', marginRight: 'auto' }}>Quantum Orbital Explorer</h3>
        <button onClick={() => setCurrentPage('welcome')} style={navButtonStyle(currentPage === 'welcome')}>Welcome & Math</button>
        <button onClick={() => setCurrentPage('chemistry')} style={navButtonStyle(currentPage === 'chemistry')}>Chemistry Concepts</button>
        <button onClick={() => setCurrentPage('howto')} style={navButtonStyle(currentPage === 'howto')}>How To Use</button>
        <button onClick={() => setCurrentPage('simulator')} style={navButtonStyle(currentPage === 'simulator')}><strong>Simulator</strong></button>
      </div>

      {currentPage === 'welcome' && (
        <div style={pageStyle}>
          <h1>Welcome to the Quantum Orbital Explorer</h1>
          
          <div className="google-ad-placeholder" style={{ margin: '20px 0', width: '100%', minHeight: '90px', backgroundColor: '#333', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px dashed #555' }}>
            <span style={{ color: '#aaa', fontSize: '0.9rem' }}>Google Ad Slot</span>
          </div>

          <p>This application renders 3D atomic orbitals using the analytical solutions to the Schrödinger equation for hydrogen-like atoms.</p>
          
          <h2>The Mathematics</h2>
          <p>The time-independent Schrödinger equation for a single-electron atom is exactly solvable. The wavefunction $\psi(r, \theta, \phi)$ in spherical coordinates is given by:</p>
          
          <div style={mathStyles}>
            <p>{"$\\psi_{n,l,m}(r, \\theta, \\phi) = R_{n,l}(r) \\cdot Y_{l,m}(\\theta, \\phi)$"}</p>
          </div>
          
          <p>Where:</p>
          <ul>
            <li><strong>$n$ (Principal Quantum Number):</strong> Determines the energy shell and size of the orbital.</li>
            <li><strong>$l$ (Azimuthal/Angular Momentum):</strong> Determines the shape of the subshell ($s, p, d, f$).</li>
            <li><strong>$m$ (Magnetic Quantum Number):</strong> Determines the orientation of the orbital in 3D space.</li>
            <li><strong>{"$R_{n,l}$"}</strong> is the Radial wave function based on generalized Laguerre polynomials.</li>
            <li><strong>{"$Y_{l,m}$"}</strong> are the Spherical Harmonics.</li>
          </ul>

          <p>The <b>probability density</b> of finding an electron at a specific point in space is the magnitude squared of the wavefunction:</p>
          <div style={mathStyles}>
            <p>{"$P(r, \\theta, \\phi) = |\\psi_{n,l,m}|^2$"}</p>
          </div>
          
          <button onClick={() => setCurrentPage('simulator')} style={nextButton}>Go to Simulator</button>
        </div>
      )}

      {currentPage === 'chemistry' && (
        <div style={pageStyle}>
          <h1>Chemistry Concepts</h1>
          <div className="google-ad-placeholder" style={{ margin: '20px 0', width: '100%', minHeight: '90px', backgroundColor: '#333', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px dashed #555' }}>
            <span style={{ color: '#aaa', fontSize: '0.9rem' }}>Google Ad Slot</span>
          </div>
          <h2>Atomic Orbitals ($s, p, d, f$)</h2>
          <p>Orbitals are regions in an atom where there is a high probability of finding electrons. They are the chemical foundation for how atoms bond to form molecules.</p>
          <ul>
            <li><strong>$l=0$ ($s$-orbitals):</strong> Spherical. They have anti-nodes at the center.</li>
            <li><strong>$l=1$ ($p$-orbitals):</strong> Dumbbell-shaped with a planar node at the nucleus. There are 3 orientations ($m = -1, 0, 1$).</li>
            <li><strong>$l=2$ ($d$-orbitals):</strong> Clover-shaped or dumbbell with a donut ring. Important for transition metals.</li>
            <li><strong>$l=3$ ($f$-orbitals):</strong> Complex multi-lobed structures. Important for lanthanides and actinides.</li>
          </ul>
          <h2>Phases & Wavefunctions</h2>
          <p>A wavefunction has a sign (positive or negative). When plotting the <b>Wavefunction Phase</b>, you see Red (Positive) and Blue (Negative) lobes. The interface between these colors represents an <i>angular node</i> where the probability of finding an electron is exactly zero.</p>
          
          <button onClick={() => setCurrentPage('howto')} style={nextButton}>Continue to How To Use</button>
        </div>
      )}

      {currentPage === 'howto' && (
        <div style={pageStyle}>
          <h1>How To Use This App</h1>
          <div className="google-ad-placeholder" style={{ margin: '20px 0', width: '100%', minHeight: '90px', backgroundColor: '#333', display: 'flex', alignItems: 'center', justifyContent: 'center', border: '1px dashed #555' }}>
            <span style={{ color: '#aaa', fontSize: '0.9rem' }}>Google Ad Slot</span>
          </div>
          <h2>1. Select Quantum Numbers</h2>
          <p>Use the sliders on the left panel to pick a valid hydrogenic state. Remember that $l$ must be strictly less than $n$, and $m$ is bounded between $-l$ and $l$. You can also adjust $Z$ to see the orbital shrinkage of heavier atomic nuclei like Helium ($Z=2$) or Carbon ($Z=6$).</p>
          
          <h2>2. Visualization Modes</h2>
          <ul>
            <li><strong>Probabilistic Scatter:</strong> Uses a Monte Carlo simulation to randomly place dots proportional to the mathematical probability density. High density means more dots.</li>
            <li><strong>Isosurface Mode:</strong> Draws a solid 3D mesh boundary containing all regions where the probability is greater than your selected threshold boundary. Like a solid balloon.</li>
          </ul>

          <h2>3. Advanced Controls</h2>
          <p>Toggle phase mapping, adjust grid sizes for larger orbitals (like $n=7$), or increase resolution up to 200 for incredibly smooth 3D structures.</p>
          
          <button onClick={() => setCurrentPage('simulator')} style={{...nextButton, backgroundColor: '#28a745'}}>Launch Simulator 🚀</button>
        </div>
      )}

      {currentPage === 'simulator' && (
        <div className="App" style={{ display: 'flex', height: 'calc(100vh - 54px)', width: '100vw', backgroundColor: '#111', color: '#fff', fontFamily: 'sans-serif', overflow: 'hidden' }}>
          
          {/* Sidebar Controls */}
      <div className="controls-sidebar" style={{ width: '300px', padding: '15px', overflowY: 'auto', backgroundColor: '#222', borderRight: '1px solid #444', flexShrink: 0 }}>
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
                <label>Grid Size: {gridSize} <input type="range" value={gridSize} min={5} max={100} onChange={e => setGridSize(parseInt(e.target.value))} /></label>
              </div>

              <hr style={{ borderColor: '#444', margin: '5px 0' }}/>

              {mode === 'scatter' ? (
                <div className="control-group">
                  <span style={{color:'#aaa', fontSize:'0.9rem'}}>Scatter Params</span>
                  <label>Resolution: {scatterGridRes} <input type="range" value={scatterGridRes} min={30} max={150} onChange={e => setScatterGridRes(parseInt(e.target.value))} /></label>
                  <label>Points: {numPoints} <input type="range" value={numPoints} min={1000} max={500000} step={1000} onChange={e => setNumPoints(parseInt(e.target.value))} /></label>
                  <label>Point Size: {pointSize.toFixed(2)} <input type="range" value={pointSize} min={0.1} max={10.0} step={0.1} onChange={e => setPointSize(parseFloat(e.target.value))} /></label>
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
      <div className="main-content" style={{ flex: 1, display: 'flex', flexDirection: 'column', height: 'calc(100vh - 54px)', position: 'relative' }}>
          <div style={{ flex: 2, position: 'relative', background: '#000' }}>
            <Canvas dpr={[1, 2]} gl={{ antialias: true, alpha: false }} camera={{ position: [0, 0, gridSize * 1.5], fov: 60 }}>
                <color attach="background" args={['#111111']} />
                <ambientLight intensity={0.5} />
                <pointLight position={[100, 100, 100]} intensity={1} />
                <pointLight position={[-100, -100, -100]} intensity={0.5} />
                
                {plotData && (mode === 'scatter' ? (
                  <ScatterPlot key={renderKey} data={plotData} pointSize={pointSize} opacity={scatterOpacity} showPhase={showPhase} />
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

          <div style={{ height: '30px', backgroundColor: '#111', color: '#888', display: 'flex', alignItems: 'center', justifyContent: 'center', fontSize: '0.8rem', borderTop: '1px solid #333' }}>
            <span><strong style={{color:'#ccc'}}>Controls:</strong> Left-Click to Rotate &bull; Right-Click to Pan/Move Target &bull; Scroll to Zoom</span>
          </div>

          <div style={{ height: '220px', backgroundColor: '#1a1a1a', padding: '10px', borderTop: '2px solid #333', display: 'flex', flexDirection: 'row' }}>
            <div style={{ flex: 1, display: 'flex', flexDirection: 'column' }}>
              <h4 style={{ margin: '0 0 10px 0', textAlign: 'center', color: '#ccc' }}>Radial Probability Distribution</h4>
              <ResponsiveContainer width="100%" height="90%">
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
              <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', width: '80px', marginLeft: '10px' }}>
                <div style={{ fontSize: '0.8rem', color: '#ccc', marginBottom: '10px' }}>Density</div>
                <div style={{ display: 'flex', flexDirection: 'row', height: '160px' }}>
                  <div style={{ display: 'flex', flexDirection: 'column', justifyContent: 'space-between', marginRight: '8px', fontSize: '0.75rem', color: '#aaa' }}>
                    <span>100</span>
                    <span>80</span>
                    <span>60</span>
                    <span>40</span>
                    <span>20</span>
                    <span>0</span>
                  </div>
                  <div style={{ width: '15px', background: 'linear-gradient(to bottom, #ff8c00, #c51b7d, #30005c)', borderRadius: '3px' }}></div>
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
